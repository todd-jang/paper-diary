#!/usr/bin/env python3
"""SimMIM 축소판(reduced) 데모 — Day 02 실습.

MAE 데모(day01_mae/mae_pretrain_demo.py)와의 핵심 차이 3가지를 코드로 보여준다:
  1. 마스크 토큰을 "삭제"하지 않고 인코더에 "함께" 넣는다 (MAE는 제거).
  2. 별도 디코더 없이 "단일 인코더 + 경량 선형 head"로 픽셀을 예측한다.
  3. 손실은 가린 픽셀에서만 ℓ1 (MAE는 ℓ2/MSE).

자기완결적 합성 데이터(사각형 도형)로 다운로드 없이 즉시 실행된다.

사용:
  /opt/homebrew/bin/python3.14 day02_beit_simmim/simmim_demo.py --steps 300
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 0) 데이터: 자기완결적 합성 이미지 (사각형 도형 + 저주파 배경)
# --------------------------------------------------------------------------- #
class ShapesDataset(torch.utils.data.Dataset):
    """32x32 RGB 이미지 = 배경색 + 밝은 사각형 + 약한 노이즈."""

    def __init__(self, n: int, size: int = 32, seed: int = 0) -> None:
        self.n, self.size, self.seed = n, size, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed * 100_000 + i)
        s = self.size
        img = torch.rand(3, 1, 1, generator=g).expand(3, s, s).clone() * 0.5  # 배경
        h = int(torch.randint(6, 15, (1,), generator=g))
        w = int(torch.randint(6, 15, (1,), generator=g))
        y = int(torch.randint(0, s - h, (1,), generator=g))
        x = int(torch.randint(0, s - w, (1,), generator=g))
        img[:, y:y + h, x:x + w] = torch.rand(3, 1, 1, generator=g) + 0.3  # 도형
        img = img + torch.randn(3, s, s, generator=g) * 0.02
        return img.clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# 1) SimMIM 모델: 단일 인코더 + 경량 선형 head
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class SimMIM(nn.Module):
    def __init__(self, img: int = 32, patch: int = 4, dim: int = 128,
                 depth: int = 4, heads: int = 4,
                 mask_ratio: float = 0.6, mask_patch: int = 2) -> None:
        super().__init__()
        self.patch = patch
        self.mask_ratio = mask_ratio
        self.mask_patch = mask_patch
        self.grid = img // patch
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        # ★ 학습된 "공유" 마스크 토큰 (삭제 대신 대체 → 인코더가 본다)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        # ★ 예측 head = 단일 선형층 (논문: "as light as a linear layer")
        self.head = nn.Linear(dim, patch * patch * 3)

    def _block_mask(self, bsz: int, device: torch.device) -> torch.Tensor:
        g, np_ = self.grid, self.mask_patch
        nb = g // np_
        n_masked = int(round(self.mask_ratio * nb * nb))
        m = torch.zeros(bsz, nb * nb, dtype=torch.bool)
        for b in range(bsz):
            m[b, torch.randperm(nb * nb)[:n_masked]] = True
        m = m.view(bsz, nb, nb)
        m = m.repeat_interleave(np_, 1).repeat_interleave(np_, 2)
        return m.reshape(bsz, g * g).to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(x).flatten(2).transpose(1, 2) + self.pos  # (B,N,dim)
        mask = self._block_mask(x.shape[0], x.device)           # (B,N) bool
        z = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(z), z)
        for blk in self.blocks:
            z = blk(z)
        pred = self.head(self.norm(z))                           # (B,N,3*p*p)
        return pred, mask


def pixel_patches(x: torch.Tensor, patch: int) -> torch.Tensor:
    """(B,3,H,W) -> (B,N,3*p*p) 픽셀 타깃."""
    u = F.unfold(x, kernel_size=patch, stride=patch)  # (B, 3*p*p, N)
    return u.transpose(1, 2)


# --------------------------------------------------------------------------- #
# 2) 학습 루프 (ℓ1, 가린 픽셀만)
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--mask-patch", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="day02_beit_simmim/runs")
    args = ap.parse_args()

    dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = ShapesDataset(n=4096, size=32, seed=7)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     drop_last=True, num_workers=0)
    model = SimMIM(mask_ratio=args.mask_ratio, mask_patch=args.mask_patch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    print(f"[SimMIM] device={dev} steps={args.steps} mask_ratio={args.mask_ratio} "
          f"mask_patch={args.mask_patch} patch={model.patch} grid={model.grid}x{model.grid}")

    it = iter(dl)
    losses: list[float] = []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)
        x = x.to(dev)
        pred, mask = model(x)
        target = pixel_patches(x, model.patch)
        loss = (pred[mask] - target[mask]).abs().mean()  # ★ ℓ1, 가린 픽셀만
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 10) == 0 or step == 1:
            print(f"  step {step:4d}  ℓ1(masked) = {loss.item():.4f}  "
                  f"({time.time() - t0:.1f}s)")

    # 손실 곡선 저장
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3.5))
        plt.plot(losses)
        plt.xlabel("step"); plt.ylabel("L1 loss (masked pixels)")
        plt.title("SimMIM reduced demo — masked-pixel L1")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(out / "simmim_loss_curve.png", dpi=120)
        print(f"[saved] {out / 'simmim_loss_curve.png'}")
    except Exception as e:  # noqa: BLE001 - 플롯 실패는 치명적이지 않음
        print(f"[warn] plot skipped: {e}")

    n0 = sum(losses[:10]) / min(10, len(losses))
    n1 = sum(losses[-10:]) / min(10, len(losses))
    torch.save({"model": model.state_dict(), "losses": losses},
               out / "simmim_demo.pt")
    print(f"\n[결과] 초기 ℓ1≈{n0:.4f} → 최종 ℓ1≈{n1:.4f}  (감소={n0 - n1:+.4f})")
    print(f"[saved] {out / 'simmim_demo.pt'}")


if __name__ == "__main__":
    main()
