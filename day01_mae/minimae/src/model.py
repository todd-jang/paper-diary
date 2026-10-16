"""
model.py — FAIR MAE (He et al., CVPR 2022) 충실 재현 (mini 스케일)

논문 핵심 3요소를 모두 포함:
  (1) 비대칭 인코더-디코더 (인코더는 보이는 패치만, 마스크 토큰 없음)
  (2) 높은 마스킹 비율 (기본 0.75)
  (3) per-patch 정규화 MSE  ← 공식 구현의 핵심 디테일 (고주파 둔감화)

추가로 논문과 동일하게:
  - 학습 파라미터가 아닌 **고정 2D sin-cos 위치 임베딩** 사용
  - class token 없음
  - 디코더에 **공유 마스크 토큰** + 전체 위치 임베딩
"""
import math
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 2D sin-cos 위치 임베딩 (논문/official MAE와 동일)
# ---------------------------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)          # w first
    grid = np.stack(grid, axis=0)               # (2, G, G)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])   # (G*G, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---------------------------------------------------------------------------
# Transformer block (pre-norm, MAE와 동일한 구조: LN->Attn->residual, LN->MLP->residual)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=384):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        return x


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------
class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        img_size=224, patch_size=16, in_ch=3,
        embed_dim=384, depth=12, num_heads=6,       # encoder (기본 ViT-Small)
        dec_embed_dim=256, dec_depth=4, dec_num_heads=8,
        mlp_ratio=4.0, mask_ratio=0.75, norm_pix_loss=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # ---- encoder ----
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        N = self.patch_embed.num_patches
        self.enc_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.enc_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # ---- decoder (가볍게, 비대칭) ----
        self.dec_embed = nn.Linear(embed_dim, dec_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        self.dec_blocks = nn.ModuleList(
            [Block(dec_embed_dim, dec_num_heads, mlp_ratio) for _ in range(dec_depth)]
        )
        self.dec_norm = nn.LayerNorm(dec_embed_dim, eps=1e-6)
        self.dec_pred = nn.Linear(dec_embed_dim, patch_size * patch_size * in_ch, bias=True)

        self._init_weights()

    def _init_weights(self):
        # 고정 sin-cos 위치 임베딩 (학습 X)
        self.register_buffer(
            "enc_pos_embed",
            torch.from_numpy(get_2d_sincos_pos_embed(self.enc_pos_dim(), self.patch_embed.grid)).float().unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "dec_pos_embed",
            torch.from_numpy(get_2d_sincos_pos_embed(self.dec_pos_dim(), self.patch_embed.grid)).float().unsqueeze(0),
            persistent=False,
        )
        # 가중치 초기화: trunc normal
        self.apply(self._init_weights_module)
        nn.init.normal_(self.mask_token, std=0.02)

    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

    def enc_pos_dim(self):
        return self.enc_blocks[0].norm1.normalized_shape[0]

    def dec_pos_dim(self):
        return self.dec_blocks[0].norm1.normalized_shape[0]

    # ------------------------------------------------------------------
    def patchify(self, imgs):
        p = self.patch_size
        g = self.patch_embed.grid
        B, C, H, W = imgs.shape
        x = imgs.reshape(B, C, g, p, g, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, g * g, p * p * C)

    def unpatchify(self, x):
        p = self.patch_size
        g = self.patch_embed.grid
        B = x.shape[0]
        x = x.reshape(B, g, g, p, p, self.in_ch).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, self.in_ch, g * p, g * p)

    def random_masking(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)  # 1=masked, 0=visible
        return x_vis, mask, ids_restore

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        x = x + self.enc_pos_embed
        x, mask, ids_restore = self.random_masking(x)
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.dec_embed(x)
        B, L, D = x.shape
        N = self.patch_embed.num_patches
        mask_tokens = self.mask_token.expand(B, N - L, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = x + self.dec_pos_embed                       # full(패치 전체) 위치 임베딩
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        x = self.dec_pred(x)
        return self.unpatchify(x)

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)                     # (B, N, p*p*C)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        pred = self.patchify(pred)                       # (B, N, p*p*C)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                         # (B, N)
        loss = (loss * mask).sum() / mask.sum()          # 가려진 패치만
        return loss

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# ---------------------------------------------------------------------------
# 프리셋
# ---------------------------------------------------------------------------
def mae_vit_tiny(img_size=224, patch_size=16, **kw):
    return MaskedAutoencoderViT(img_size=img_size, patch_size=patch_size,
                                embed_dim=192, depth=6, num_heads=3,
                                dec_embed_dim=128, dec_depth=2, dec_num_heads=4, **kw)


def mae_vit_small(img_size=224, patch_size=16, **kw):
    return MaskedAutoencoderViT(img_size=img_size, patch_size=patch_size,
                                embed_dim=384, depth=12, num_heads=6,
                                dec_embed_dim=256, dec_depth=4, dec_num_heads=8, **kw)


if __name__ == "__main__":
    m = mae_vit_tiny()
    n = sum(p.numel() for p in m.parameters())
    print(f"ViT-Tiny MAE params: {n/1e6:.2f}M")
    x = torch.randn(2, 3, 224, 224)
    loss, pred, mask = m(x)
    print("loss", float(loss), "pred", tuple(pred.shape), "mask ratio", float(mask.mean()))
