#!/usr/bin/env python3
"""iBOT 축소판 — 온라인 토크나이저(EMA teacher) 자기증류 데모.

Day 03 (iBOT, arXiv:2111.07832) 의 핵심 아이디어를 최소 코드로 재현한다:

  * student 는 **blockwise-mask** 된 뷰를 보고, teacher(= student 의 EMA) 는 **원본(마스크 없는)** 뷰를 본다.
  * L_MIM  : student 의 '가린 패치' 출력을 teacher 의 **soft 토큰분포**(hard argmax 아님)에 CE 로 맞춘다.
  * L_CLS  : 두 뷰의 [CLS] 토큰 자기증류(전역 의미) — DINO 계열.
  * teacher 는 stop-grad + EMA 로 갱신 → **외부 토크나이저(DALL-E dVAE) 없이** "온라인 토크나이저"가 된다.

합성 데이터(무작위 사각형)로 즉시 실행 — 다운로드 불필요.
Day 02 의 SimMIM 데모와의 결정적 차이: 픽셀 복원(ℓ1)이 아니라 **soft 토큰분포 예측(CE) + EMA teacher**.
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# 모델 : 아주 작은 ViT (patch 4×4, 2층, dim 64)
# --------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    """student / teacher 공용 백본 + 두 개의 projection head([CLS]용, patch용)."""

    def __init__(self, img: int = 32, patch: int = 4, dim: int = 64, depth: int = 2,
                 heads: int = 4, n_proto: int = 64) -> None:
        super().__init__()
        self.patch = patch
        self.grid = img // patch
        n = self.grid * self.grid
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.patch_head = nn.Linear(dim, n_proto)              # online tokenizer 의 "단어장"
        self.cls_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_proto))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        t = self.proj(x).flatten(2).transpose(1, 2)            # (B, N, dim)
        t = torch.cat([self.cls_token.expand(b, -1, -1), t], dim=1)
        t = self.blocks(t + self.pos)
        t = self.norm(t)
        cls, patches = t[:, 0], t[:, 1:]                       # (B,dim), (B,N,dim)
        return self.cls_head(cls), self.patch_head(patches)    # logits


# --------------------------------------------------------------------------------------
# 합성 데이터 / 증강 / blockwise 마스킹
# --------------------------------------------------------------------------------------
def make_batch(bs: int, img: int = 32, seed: int | None = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.zeros(bs, 3, img, img)
    for b in range(bs):
        for _ in range(int(torch.randint(1, 4, (1,), generator=g))):
            ch = int(torch.randint(0, 3, (1,), generator=g))
            w = int(torch.randint(4, 12, (1,), generator=g))
            h = int(torch.randint(4, 12, (1,), generator=g))
            y = int(torch.randint(0, img - h, (1,), generator=g))
            xx = int(torch.randint(0, img - w, (1,), generator=g))
            x[b, ch, y:y + h, xx:xx + w] = 1.0
    return x


def augment(x: torch.Tensor) -> torch.Tensor:
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    sy, sx = int(torch.randint(-4, 5, (1,))), int(torch.randint(-4, 5, (1,)))
    return torch.roll(x, shifts=(sy, sx), dims=(2, 3))


def blockwise_masks(bs: int, grid: int, ratio: float, blk: int = 2, device: str = "cpu") -> torch.Tensor:
    """각 샘플마다 blk×blk 블록 단위로 ratio 만큼 가린 불리언 마스크 (B, grid*grid)."""
    masks = torch.zeros(bs, grid, grid)
    target = max(1, int(ratio * grid * grid))
    for b in range(bs):
        covered = 0
        while covered < target:
            i = int(torch.randint(0, grid - blk + 1, (1,)))
            j = int(torch.randint(0, grid - blk + 1, (1,)))
            if masks[b, i:i + blk, j:j + blk].sum() == 0:
                masks[b, i:i + blk, j:j + blk] = 1.0
            covered = int(masks[b].sum().item())
    return masks.reshape(bs, -1).bool().to(device)


def soft_target(logits: torch.Tensor, temp: float, center: bool = True) -> torch.Tensor:
    z = logits - logits.mean(dim=0, keepdim=True) if center else logits   # centering (DINO)
    return F.softmax(z / temp, dim=-1)


# --------------------------------------------------------------------------------------
# 학습 루프
# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--mask-ratio", type=float, default=0.4)
    ap.add_argument("--mask-patch", type=int, default=2)
    ap.add_argument("--n-proto", type=int, default=64)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--tau-t", type=float, default=0.04)
    ap.add_argument("--tau-s", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(__file__), "runs"))
    args = ap.parse_args()

    dev = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    torch.manual_seed(0)
    os.makedirs(args.outdir, exist_ok=True)

    grid = args.img // args.patch
    student = TinyViT(args.img, args.patch, n_proto=args.n_proto).to(dev)
    teacher = copy.deepcopy(student).to(dev)
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.04)

    print(f"[iBOT] device={dev} steps={args.steps} batch={args.batch} "
          f"mask_ratio={args.mask_ratio} mask_patch={args.mask_patch} grid={grid}x{grid} "
          f"n_proto={args.n_proto} ema={args.ema}")

    hist: list[tuple[int, float, float, float]] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x = make_batch(args.batch, args.img, seed=step).to(dev)
        u, v = augment(x), augment(x)
        mu = blockwise_masks(args.batch, grid, args.mask_ratio, args.mask_patch, dev)
        mv = blockwise_masks(args.batch, grid, args.mask_ratio, args.mask_patch, dev)

        def mask_view(view: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            # 마스크된 패치를 0 으로 (student 는 가린 뷰만 본다)
            b, c, h, w = view.shape
            pm = m.reshape(b, grid, grid).unsqueeze(1).float()          # (B,1,grid,grid)
            pm = F.interpolate(pm, size=(h, w), mode="nearest")
            return view * (1.0 - pm)

        sc_u, sp_u = student(mask_view(u, mu))
        sc_v, sp_v = student(mask_view(v, mv))
        with torch.no_grad():
            tc_u, tp_u = teacher(u)
            tc_v, tp_v = teacher(v)

        # --- L_CLS : 두 뷰 [CLS] 교차 자기증류 (대칭) ---
        t_cu = soft_target(tc_u, args.tau_t)
        t_cv = soft_target(tc_v, args.tau_t)
        s_cu = F.log_softmax(sc_u / args.tau_s, dim=-1)
        s_cv = F.log_softmax(sc_v / args.tau_s, dim=-1)
        loss_cls = -0.5 * ((t_cv * s_cu).sum(-1).mean() + (t_cu * s_cv).sum(-1).mean())

        # --- L_MIM : 가린 패치를 teacher 의 soft 토큰분포에 맞춤 (in-view) ---
        def mim(student_patch: torch.Tensor, teacher_patch: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            t = soft_target(teacher_patch, args.tau_t)                  # teacher 는 마스크 없는 뷰
            s = F.log_softmax(student_patch / args.tau_s, dim=-1)
            ce = -(t * s).sum(-1)                                       # (B,N)
            return (ce * m).sum() / m.sum().clamp(min=1)

        loss_mim = 0.5 * (mim(sp_u, tp_u, mu) + mim(sp_v, tp_v, mv))
        loss = loss_cls + loss_mim

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():                                           # EMA teacher
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.mul_(args.ema).add_(ps.detach(), alpha=1.0 - args.ema)

        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"  step {step:4d}  L_CLS={loss_cls.item():.4f}  L_MIM={loss_mim.item():.4f}  "
                  f"total={loss.item():.4f}")
        hist.append((step, float(loss_cls.item()), float(loss_mim.item()), float(loss.item())))

    first3 = hist[:3]
    last3 = hist[-3:]
    avg = lambda rows, k: sum(r[k] for r in rows) / len(rows)
    print(f"[결과] 초기 L_CLS≈{avg(first3,1):.4f} L_MIM≈{avg(first3,2):.4f} "
          f"→ 최종 L_CLS≈{avg(last3,1):.4f} L_MIM≈{avg(last3,2):.4f}   "
          f"(총 {time.time()-t0:.1f}s)")

    torch.save({"student": student.state_dict(), "teacher": teacher.state_dict()},
               os.path.join(args.outdir, "ibot_demo.pt"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [r[0] for r in hist]
        plt.figure(figsize=(7, 4))
        plt.plot(steps, [r[1] for r in hist], label="L_CLS")
        plt.plot(steps, [r[2] for r in hist], label="L_MIM")
        plt.plot(steps, [r[3] for r in hist], label="total", linestyle="--")
        plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title("iBOT toy: online-tokenizer self-distillation")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "ibot_loss_curve.png"), dpi=120)
        print(f"  saved {os.path.join(args.outdir, 'ibot_loss_curve.png')}")
    except Exception as e:  # pragma: no cover
        print(f"  (matplotlib skipped: {e})")


if __name__ == "__main__":
    main()
