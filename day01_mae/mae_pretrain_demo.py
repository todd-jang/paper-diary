"""
MAE (Masked Autoencoders) — 최소 구현 + 학습 데모
=================================================
He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
arXiv:2111.06377

이 스크립트는 MAE의 핵심 두 설계를 *작게* 재현한다:
  (1) 비대칭 인코더-디코더: 인코더는 "보이는 패치만" 처리, 디코더가 픽셀 복원
  (2) 높은 마스킹 비율 (기본 75%)

실행:
  python mae_pretrain_demo.py --data synthetic --epochs 3        # 의존성 0, 즉시 실행
  python mae_pretrain_demo.py --data cifar --epochs 20           # CIFAR-10 다운로드
    python mae_pretrain_demo.py --data imagenet1k --data-dir /path/to/train --img-size 224 --patch-size 16
  python mae_pretrain_demo.py --data cifar --mask-ratio 0.5      # 마스킹 비율 실험

관찰 포인트:
  - train loss(=마스킹된 패치의 픽셀 MSE)가 epoch에 따라 감소하는가?
  - mask_ratio를 올리면 loss가 더 천천히/높게 시작하는가? (어려운 과제가 됨)
"""
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1) Patch Embedding — 이미지를 패치 토큰으로 자른다
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Conv2d(stride=patch) 한 방으로 패치 분할 + 선형 투영을 동시에 수행."""

    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size는 patch_size로 나누어져야 함"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        # kernel=stride=patch_size → 겹치지 않는 패치별 선형 투영
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                  # (B, D, G, G)
        x = x.flatten(2)                  # (B, D, G*G)
        x = x.transpose(1, 2)             # (B, N, D)  N=패치 수
        return x


# ---------------------------------------------------------------------------
# 2) Transformer Block (pre-norm) — 인코더/디코더 공용
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        # attention (pre-norm 잔차)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        # MLP 잔차
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 3) MAE 본체
# ---------------------------------------------------------------------------
class MAE(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_ch=3,
        # --- 인코더 (무겁게) ---
        enc_dim=192,
        enc_depth=6,
        enc_heads=6,
        # --- 디코더 (가볍게, 비대칭!) ---
        dec_dim=96,
        dec_depth=2,
        dec_heads=6,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, enc_dim)
        N = self.patch_embed.num_patches

        # 인코더 위치 임베딩 (학습 가능). 실제 MAE는 sin-cos도 사용 가능.
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, N, enc_dim))
        self.enc_blocks = nn.ModuleList([Block(enc_dim, enc_heads) for _ in range(enc_depth)])
        self.enc_norm = nn.LayerNorm(enc_dim)

        # 디코더: 공유 마스크 토큰 + 위치 임베딩 + (투영)
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, N, dec_dim))
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        # 픽셀 복원 헤드: 패치 하나의 픽셀값(patch*patch*3)을 예측
        self.dec_pred = nn.Linear(dec_dim, patch_size * patch_size * in_ch)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.enc_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    # ------------------------------------------------------------------
    # 랜덤 마스킹: 각 샘플마다 패치를 섞고, 앞쪽 일부만 "보이는 패치"로 남긴다
    # ------------------------------------------------------------------
    def random_masking(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))       # 보이는 패치 수 (예: 64 * 0.25 = 16)

        noise = torch.rand(B, N, device=x.device)       # 샘플별 랜덤 순열
        ids_shuffle = torch.argsort(noise, dim=1)       # 섞인 인덱스
        ids_restore = torch.argsort(ids_shuffle, dim=1) # 원래 순서로 되돌리는 인덱스

        ids_keep = ids_shuffle[:, :len_keep]            # 앞 len_keep 개 = 보이는 패치
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # mask: 0=보임, 1=가림 (원래 순서 기준)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore

    # ------------------------------------------------------------------
    # 인코더: 보이는 패치만 처리 (마스크 토큰을 넣지 않는다!)
    # ------------------------------------------------------------------
    def forward_encoder(self, x):
        x = self.patch_embed(x)                          # (B, N, D)
        B, N, D = x.shape
        # 위치 임베딩은 마스킹과 무관하게 "전체 패치 기준"으로 더한 뒤 마스킹
        x = x + self.enc_pos_embed
        x, mask, ids_restore = self.random_masking(x)    # (B, len_keep, D)
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)
        return x, mask, ids_restore

    # ------------------------------------------------------------------
    # 디코더: 인코더 출력 + 마스크 토큰 → 전체 패치 순서 복원 → 픽셀 예측
    # ------------------------------------------------------------------
    def forward_decoder(self, x, ids_restore):
        x = self.enc_to_dec(x)                           # (B, len_keep, dec_dim)
        B, L, D = x.shape
        N = self.patch_embed.num_patches

        # 가려진 자리에 공유 마스크 토큰을 채운다
        mask_tokens = self.mask_token.expand(B, N - L, -1)
        x = torch.cat([x, mask_tokens], dim=1)           # (B, N, D) — 아직 섞인 순서

        # unshuffle: 원래 패치 순서로 되돌린다
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = x + self.dec_pos_embed                       # 디코더 위치 임베딩

        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        x = self.dec_pred(x)                             # (B, N, patch*patch*3)
        x = self.unpatchify(x)                           # (B, 3, H, W)
        return x

    def unpatchify(self, x):
        p = self.patch_size
        g = self.patch_embed.grid
        B = x.shape[0]
        x = x.reshape(B, g, g, p, p, self.in_ch)
        x = x.permute(0, 5, 1, 3, 2, 4)                  # (B, C, G, p, G, p)
        x = x.reshape(B, self.in_ch, g * p, g * p)       # (B, C, H, W)
        return x

    def patchify(self, imgs):
        """타깃 이미지를 (B, N, p*p*C)로 자른다."""
        p = self.patch_size
        g = self.patch_embed.grid
        B, C, H, W = imgs.shape
        x = imgs.reshape(B, C, g, p, g, p)
        x = x.permute(0, 2, 4, 3, 5, 1)                  # (B, G, G, p, p, C)
        x = x.reshape(B, g * g, p * p * C)
        return x

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)      # (B, 3, H, W)

        target = self.patchify(imgs)                          # (B, N, p*p*C)
        pred = self.patchify(pred)                            # (B,3,H,W) -> (B, N, p*p*C)

        # 손실: "가려진 패치"에서만 MSE (보이는 패치는 정답을 그대로 복사하면 되므로 제외)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                              # (B, N)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask


# ---------------------------------------------------------------------------
# 4) 데이터
# ---------------------------------------------------------------------------
def get_loader(kind, img_size, batch_size, root="./data"):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # [0,1]
    ])
    if kind == "cifar":
        ds = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    elif kind == "imagenet1k":
        ds = datasets.ImageFolder(root, transform=tf)
    else:
        # synthetic: 의존성 없이 즉시 실행 (무작위 이미지)
        ds = datasets.FakeData(
            size=256, image_size=(3, img_size, img_size),
            num_classes=10, transform=transforms.ToTensor(),
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


# ---------------------------------------------------------------------------
# 5) 학습 루프
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "cifar", "imagenet1k"], default="synthetic")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=32)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[device] {device}")

    loader = get_loader(args.data, args.img_size, args.batch_size, root=args.data_dir)
    model = MAE(
        img_size=args.img_size, patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] total params = {n_params/1e6:.2f}M | patches = {model.patch_embed.num_patches}"
          f" | keep = {int(model.patch_embed.num_patches*(1-args.mask_ratio))}"
          f" | mask_ratio = {args.mask_ratio}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        n = 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            loss, _, _ = model(imgs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
        print(f"[epoch {epoch}/{args.epochs}] mask-MSE loss = {running/max(n,1):.4f}")

    print("[done] 손실이 감소하면 MAE가 '가려진 픽셀 복원'을 학습한 것.")


if __name__ == "__main__":
    main()
