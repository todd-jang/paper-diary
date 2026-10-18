#!/usr/bin/env python3
"""BEiT식 **2단계** MIM 축소판 — "왜 iBOT는 1단계인가"의 대조군.

Day 03 의 `ibot_demo.py`(온라인 soft, 1단계)와 **정확히 대조**되도록 만든 오프라인 2단계 파이프라인:

  Stage 1  VQ-VAE(=dVAE-lite) 토크나이저를 **먼저 학습** → **동결(freeze)**
  Stage 2  동결된 토크나이저가 만든 **hard 토큰 ID**(argmin nearest code)를
           가린 패치에서 **분류(CE)** 로 예측   ← BEiT 의 MIM

핵심 교육 포인트(코드로 확인):
  * `z = argmin‖z_e − e‖` (nearest code) 는 **미분 불가** → 토크나이저 학습과 MIM 학습을
    한 그래디언트로 묶을 수 없다 → **stage 를 쪼갤 수밖에 없다** = 2단계의 이유.
  * 정답은 **사전학습·동결된** 코드북 → iBOT 처럼 EMA 로 진화하지 않는다(오프라인).
  * 타깃은 **hard one-hot** → iBOT 의 **soft 분포**와 대비.

합성 데이터(무작위 사각형). iBOT 데모와 동일 데이터/백본 부품을 재사용한다.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ibot_demo 와 동일한 합성 데이터·블록마스킹·트랜스포머 블록 재사용(중복 제거)
from ibot_demo import Block, blockwise_masks, make_batch


# --------------------------------------------------------------------------------------
# Stage 1 : VQ-VAE (dVAE-lite) — 이산 토큰을 만드는 "오프라인 토크나이저"
# --------------------------------------------------------------------------------------
class VQVAE(nn.Module):
    def __init__(self, img: int = 32, patch: int = 4, code_dim: int = 32, n_code: int = 64,
                 beta: float = 0.25, lambda_div: float = 2.0) -> None:
        super().__init__()
        self.patch = patch
        self.grid = img // patch
        self.n_code = n_code
        self.beta = beta
        self.lambda_div = lambda_div
        # encoder: 32x32 → (conv4/4) 8x8 → 8x8
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 4), nn.ReLU(), nn.Conv2d(64, code_dim, 3, padding=1),
        )
        self.codebook = nn.Parameter(torch.randn(n_code, code_dim))         # dVAE 의 '코드북'
        # decoder: 8x8 → 8x8 → (convT4/4) 32x32
        self.dec = nn.Sequential(
            nn.Conv2d(code_dim, 64, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(64, 3, 4, 4),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)                                                  # (B, d, gh, gw)

    def quantize(self, h: torch.Tensor):
        b, d, gh, gw = h.shape
        z = F.normalize(h.permute(0, 2, 3, 1).reshape(-1, d), dim=-1)       # 단위벡터화(붕괴 완화)
        e = F.normalize(self.codebook, dim=-1)
        sim = z @ e.t()                                                     # cosine 유사도
        idx = sim.argmax(1)                                                 # hard 토큰 ID
        zq = e[idx]
        return zq, idx, z, e, sim, (b, gh, gw)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        zq, idx, z, e, sim, (b, gh, gw) = self.quantize(h)
        zq_img = zq.reshape(b, gh, gw, -1).permute(0, 3, 1, 2).contiguous()
        recon = self.dec(zq_img)
        # VQ 손실: 코드북 정렬(‖sg[z]−e‖²) + 커밋먼트(β‖z−sg[e]‖²)
        vq = F.mse_loss(z.detach(), zq) + self.beta * F.mse_loss(z, zq.detach())
        # 사용 균등화: 소프트 사용확률의 엔트로피를 최대화(Σ p·log p 최소화) → 코드북 붕괴 방지
        soft = F.softmax(sim / 0.1, dim=-1)
        avg = soft.mean(0)
        div = (avg * (avg + 1e-8).log()).sum()
        return recon, idx, vq + self.lambda_div * div, h


def codebook_stats(idx: torch.Tensor, n_code: int) -> tuple[float, int]:
    counts = torch.bincount(idx.flatten().cpu(), minlength=n_code).float()
    p = counts / counts.sum().clamp(min=1)
    perplexity = float(torch.exp(-(p * (p + 1e-10).log()).sum()))
    return perplexity, int((counts > 0).sum())


# --------------------------------------------------------------------------------------
# Stage 2 : MIM — 가린 패치의 [hard 토큰 ID]를 분류(CE)
# --------------------------------------------------------------------------------------
class MIMViT(nn.Module):
    def __init__(self, img: int = 32, patch: int = 4, dim: int = 64, depth: int = 2,
                 heads: int = 4, n_code: int = 64) -> None:
        super().__init__()
        g = img // patch
        self.proj = nn.Conv2d(3, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, g * g, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))              # 학습된 [MASK]
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_code)                                  # K-code 분류기
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        t = self.proj(x).flatten(2).transpose(1, 2) + self.pos              # (B,N,dim)
        t = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(t), t)  # 가린 자리 = [MASK]
        return self.head(self.norm(self.blocks(t)))                        # (B,N,n_code)


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-steps", type=int, default=300)
    ap.add_argument("--stage2-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--img", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--n-code", type=int, default=64)
    ap.add_argument("--mask-ratio", type=float, default=0.4)
    ap.add_argument("--mask-patch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(__file__), "runs"))
    args = ap.parse_args()

    dev = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    torch.manual_seed(0)
    os.makedirs(args.outdir, exist_ok=True)
    grid = args.img // args.patch
    n_patch = grid * grid

    t0 = time.time()
    print(f"[BEiT 2-stage] device={dev} n_code={args.n_code} grid={grid}x{grid} "
          f"mask_ratio={args.mask_ratio} batch={args.batch}")

    # ============================== Stage 1 ==============================
    print("\n== Stage 1: VQ-VAE(dVAE-lite) 토크나이저 학습 (오프라인) ==")
    vq = VQVAE(args.img, args.patch, n_code=args.n_code).to(dev)
    opt1 = torch.optim.Adam(vq.parameters(), lr=args.lr)
    s1: list[tuple[int, float, float, int]] = []
    for step in range(1, args.stage1_steps + 1):
        x = make_batch(args.batch, args.img, seed=1000 + step).to(dev)
        recon, idx, vq_loss, _ = vq(x)
        loss = F.mse_loss(recon, x) + vq_loss
        opt1.zero_grad(set_to_none=True)
        loss.backward()
        opt1.step()
        if step == 1 or step % 50 == 0 or step == args.stage1_steps:
            ppl, used = codebook_stats(idx, args.n_code)
            rec = float(F.mse_loss(recon, x).item())
            print(f"  step {step:4d}  recon={rec:.5f}  vq={float(vq_loss.item()):.5f}  "
                  f"perplexity={ppl:5.1f}  used={used}/{args.n_code}")
            s1.append((step, rec, ppl, used))
    for p in vq.parameters():
        p.requires_grad_(False)
    vq.eval()
    print("  → 토크나이저 동결(freeze) ✔  (이후 절대 업데이트되지 않음)")

    # ============================== Stage 2 ==============================
    print("\n== Stage 2: MIM — 가린 패치의 hard 토큰 ID 분류(CE) ==")
    mim = MIMViT(args.img, args.patch, n_code=args.n_code).to(dev)
    opt2 = torch.optim.AdamW(mim.parameters(), lr=args.lr, weight_decay=0.04)
    s2: list[tuple[int, float, float]] = []
    for step in range(1, args.stage2_steps + 1):
        x = make_batch(args.batch, args.img, seed=2000 + step).to(dev)
        with torch.no_grad():                                               # hard 라벨은 동결 토크나이저가 생성
            _, idx, _, _ = vq(x)
            labels = idx.reshape(args.batch, n_patch)                       # (B,N) 정수 토큰 ID
        mask = blockwise_masks(args.batch, grid, args.mask_ratio, args.mask_patch, dev)
        logits = mim(x, mask)                                               # (B,N,n_code)
        loss = F.cross_entropy(logits[mask], labels[mask])
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()
        if step == 1 or step % 50 == 0 or step == args.stage2_steps:
            acc = (logits[mask].argmax(-1) == labels[mask]).float().mean().item()
            print(f"  step {step:4d}  CE={float(loss.item()):.4f}  masked-token acc={acc*100:5.1f}%")
            s2.append((step, float(loss.item()), acc))

    # ============================== 요약 ==============================
    print("\n[결과 요약]")
    print(f"  Stage 1(토크나이저): recon {s1[0][1]:.5f} → {s1[-1][1]:.5f}  "
          f"(codebook used {s1[-1][3]}/{args.n_code}, perplexity {s1[-1][2]:.1f})")
    print(f"  Stage 2(MIM, hard CE): CE {s2[0][1]:.4f} → {s2[-1][1]:.4f}  "
          f"acc {s2[0][2]*100:.1f}% → {s2[-1][2]*100:.1f}%")
    print(f"  총 {time.time()-t0:.1f}s / 2단계 파이프라인: (① 토크나이저 학습·동결) → (② MIM)")
    print("  대조: ibot_demo.py 는 EMA teacher 만으로 **1단계**·soft 타깃(내부에서 토큰 생성).")

    torch.save({"vqvae": vq.state_dict(), "mim": mim.state_dict(), "codebook": vq.codebook.detach()},
               os.path.join(args.outdir, "beit_2stage.pt"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot([r[0] for r in s1], [r[1] for r in s1], marker="o")
        ax[0].set_title("Stage 1: VQ-VAE recon MSE"); ax[0].set_xlabel("step"); ax[0].set_ylabel("MSE")
        ax[1].plot([r[0] for r in s2], [r[1] for r in s2], marker="o", label="CE")
        ax[1].plot([r[0] for r in s2], [1 - r[2] for r in s2], marker="s", label="1 - acc")
        ax[1].set_title("Stage 2: MIM (hard token CE)"); ax[1].set_xlabel("step")
        ax[1].legend()
        fig.suptitle("BEiT-style 2-stage: tokenizer(frozen) -> MIM on hard tokens")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "beit_2stage_curve.png"), dpi=120)
        print(f"  saved {os.path.join(args.outdir, 'beit_2stage_curve.png')}")
    except Exception as e:  # pragma: no cover
        print(f"  (matplotlib skipped: {e})")


if __name__ == "__main__":
    main()
