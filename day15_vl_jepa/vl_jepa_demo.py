"""VL-JEPA 축소 데모 — 수렴 목적학습 제거 작은 JEPA로 본 논문 4대 검증항목을 MVC 구조로 실측.

논문 (arXiv 2512.10942 v2) 핵심 구조를 군집 축소 모델 군 3모듈로 재현:
    1. Context Encoder: 비전 입력을 잠재 공간 s_x로 인코딩 (마스킹된 후보 밴딩)
    2. Predictor: 언어 임베딩 z + s_x를 결합하여 타깃 잠재 s_y 예측
    3. Target Encoder: 타깃 패치의 표현 s_y 인코딩 (EMA 옵션 — V4 실증 요소)

V1. Latent Space Prediction — ||s_hat_y - s_y|| 초기 대비 감소 확인
V2. Language Conditioning Effect — 정답 프롬프트 vs 셔플 프롬프트 예측 오차 갭
V3. Collapse 방지 — Cov(S) 유효 랭크(entropy of singular values)가 학습 동안 유지
V4. EMA 타깃 인코더 — tau=0.99 (EMA) vs tau=0.0 (공유) 수렴/붕괴 차이
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# --------------------------------------------------------------------------- #
# 설정
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DemoConfig:
    dim: int = 64            # 잠재 공간 차원
    n_prompts: int = 8       # 언어 프롬프트 클러스터 수
    n_samples: int = 512     # 합성 샘플 수
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    tau_ema: float = 0.99    # EMA 타깃 인코더 계수 (V4-A)
    mask_ratio: float = 0.75 # context-side random patch masking
    infonce_temp: float = 0.1
    l_infonce: float = 0.3   # 균일화항 가중치 (논문 = 양방향 InfoNCE의 균일화 성분)


# --------------------------------------------------------------------------- #
# 합성 데이터 — 프롬프트 z에 결합된 비전 토큰 클러스터
# --------------------------------------------------------------------------- #

def make_synthetic_data(cfg: DemoConfig, device: str) -> tuple[Tensor, Tensor, Tensor]:
    """vision(패치토큰 NTP), prompt(정답 z), target(타깃 시점 패치토큰 NTP) 생성.

    프롬프트 z에 따라 타깃 분포가 결정되는 e(프롬프트-비전 결합 구조) -> 정답 z만이
    predictor에게 유의미한 조건이 되게 설계. (V2 갭이 이 구조에서 실측됨.)
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    n = cfg.n_samples
    n_patch = 8
    prompt_idx = torch.randint(0, cfg.n_prompts, (n,), generator=g)
    # 프롬프트 클러스터의 중심 = 타깃 분포 평행이동
    centers = torch.randn(cfg.n_prompts, cfg.dim // 2, generator=g) * 2.0
    shift = centers[prompt_idx]

    vision = torch.randn(n, n_patch, cfg.dim // 2, generator=g)          # 컨텍스트 패치
    target = torch.randn(n, n_patch, cfg.dim // 2, generator=g) + shift.unsqueeze(1) * 0.6
    prompt_embed = F.one_hot(prompt_idx, cfg.n_prompts).float()
    noise = torch.randn(n, cfg.n_prompts, generator=g) * 0.05
    prompt_embed = prompt_embed + noise                                  # z = soft 조건
    return vision.to(device), prompt_embed.to(device), target.to(device)


# --------------------------------------------------------------------------- #
# 3-모듈 — 축소 JEPA
# --------------------------------------------------------------------------- #

class ContextEncoder(nn.Module):
    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, patches: Tensor) -> Tensor:                      # [N, P, in] -> [N, P, dim]
        return self.net(patches)


class Predictor(nn.Module):
    def __init__(self, dim: int, n_prompts: int) -> None:
        super().__init__()
        self.z_proj = nn.Linear(n_prompts, dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)       # 쿼리 1토큰 풀링

    def forward(self, s_x: Tensor, z: Tensor) -> Tensor:
        # [N, P, dim] + z -> [N, dim]: 쿼리 토큰이 패치 토큰 + z 조건을 attention으로 통합
        n, p, _ = s_x.shape
        z_ctx = self.z_proj(z).unsqueeze(1).expand(-1, p, -1)
        tokens = torch.cat([self.query.expand(n, -1, -1), s_x + z_ctx], dim=1)
        attn = torch.softmax(tokens @ tokens.transpose(1, 2) / math.sqrt(tokens.size(-1)), dim=-1)
        pooled = torch.bmm(attn[:, :1], tokens).squeeze(1)             # 쿼리-어텐션 풀링 [N, dim]
        return self.net(pooled)


class TargetEncoder(nn.Module):
    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, patches: Tensor) -> Tensor:
        return self.net(patches)


def random_patch_mask(patches: Tensor, ratio: float, g: torch.Generator) -> Tensor:
    """context-side 랜덤 패치 마스킹 — 남은 패치만 인코딩 (situ I-JEPA 계열)."""
    n, p, d = patches.shape
    keep = int(round(p * (1.0 - ratio)))
    keep = max(keep, 1)
    perm = torch.rand(n, p, generator=g).argsort(dim=1)[:, :keep]
    return patches.gather(1, perm.unsqueeze(-1).expand(-1, -1, d).to(patches.device))


# --------------------------------------------------------------------------- #
# 검증 메트릭
# --------------------------------------------------------------------------- #

def effective_rank(z: Tensor) -> float:
    """공분산 특이값의 entropy 기반 유효 차원 — collapse(붕괴) 감지. 낮을수록 붕괴."""
    z = z.cpu()
    sv = torch.linalg.svdvals(z - z.mean(dim=0, keepdim=True))
    p = sv / (sv.sum() + 1e-8)
    ent = -(p[p > 1e-12] * p[p > 1e-12].log()).sum().item()
    return math.exp(ent)


def align_gap(pred: Tensor, tgt: Tensor) -> Tensor:
    return F.cosine_similarity(pred, tgt, dim=-1).mean()


def infonce_loss(anchor: Tensor, positive: Tensor, temperature: float) -> Tensor:
    """양방향 InfoNCE = 균일화항 구조 표지. 동일 배치 내 추론을 위한 원형 거리 기반."""
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    logits = a @ p.T / temperature
    labels = torch.arange(a.size(0), device=a.device)
    loss_a2p = F.cross_entropy(logits, labels)
    loss_p2a = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a2p + loss_p2a)


# --------------------------------------------------------------------------- #
# 학습/평가
# --------------------------------------------------------------------------- #

@dataclass
class RunResult:
    tau: float
    losses: list[float]
    initial_err: float
    final_err: float
    rank_final: float
    rank_init: float
    gap_true: float
    gap_shuffled: float


def run(cfg: DemoConfig, tau: float, device: str) -> RunResult:
    vision, prompt, target = make_synthetic_data(cfg, device)
    g = torch.Generator(device="cpu").manual_seed(42)

    ctx_enc = ContextEncoder(cfg.dim // 2, cfg.dim).to(device)
    tgt_enc = TargetEncoder(cfg.dim // 2, cfg.dim).to(device)
    tgt_enc.load_state_dict(ctx_enc.state_dict())          # 동일 가중치 초기화 (EMA 정합성)
    pred = Predictor(cfg.dim, cfg.n_prompts).to(device)

    if tau <= 0.0:
        # tau=0: online과 타깃 인코더 가중치 공유 — V4 대조군
        tgt_enc = ctx_enc

    params = list(ctx_enc.parameters()) + list(pred.parameters())
    if tau > 0.0:
        params += list(tgt_enc.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr)

    losses: list[float] = []
    n = vision.size(0)
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=g).to(device)
        ep_loss = 0.0
        batches = 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            v, z, t = vision[idx], prompt[idx], target[idx]

            v_masked = random_patch_mask(v, cfg.mask_ratio, g)     # context masking
            s_x = ctx_enc(v_masked)                                # [N, P', dim]
            s_y_hat = pred(s_x, z)                                 # 예측 타깃 embedding
            with torch.no_grad():
                s_y = tgt_enc(t).mean(dim=1)                       # 실측 타깃 embedding [N, dim]

            # 논문 손실 = 정렬항(L1/L2 거리) + 균일화항(InfoNCE 균일화 성분)
            l_align = F.mse_loss(s_y_hat, s_y)
            l_unif = infonce_loss(s_y_hat, s_y, cfg.infonce_temp) * cfg.l_infonce
            loss = l_align + l_unif
            opt.zero_grad()
            loss.backward()
            opt.step()

            if tau > 0.0:                                          # EMA update on target enc
                with torch.no_grad():
                    for pe, ce in zip(tgt_enc.parameters(), ctx_enc.parameters()):
                        pe.mul_(tau).add_(ce.detach(), alpha=1.0 - tau)

            ep_loss += loss.item()
            batches += 1
        losses.append(ep_loss / max(batches, 1))

    # 평가 — 마스킹 없이 풀 컨텍스트
    with torch.no_grad():
        s_x_all = ctx_enc(vision)
        s_y_hat_all = pred(s_x_all, prompt)
        s_y_all = tgt_enc(target).mean(dim=1)

        err_all = (s_y_hat_all - s_y_all).pow(2).sum(-1)               # [N]
        baseline_err = (s_y_all - s_y_all.mean(0, keepdim=True)).pow(2).sum(-1).mean().item()
        final_err = err_all.mean().item()

        # V2: 정답 프롬프트 vs 셔플 프롬프트
        perm_z = prompt[torch.randperm(n, generator=g).to(device)]
        s_y_hat_shuffled = pred(s_x_all, perm_z)
        gap_true = align_gap(s_y_hat_all, s_y_all).item()
        gap_shuffled = align_gap(s_y_hat_shuffled, s_y_all).item()

        rank_init = effective_rank(tgt_enc(target).mean(dim=1))
        rank_final = effective_rank(s_y_hat_all)

    return RunResult(
        tau=tau,
        losses=losses,
        initial_err=baseline_err,
        final_err=final_err,
        rank_init=rank_init,
        rank_final=rank_final,
        gap_true=gap_true,
        gap_shuffled=gap_shuffled,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="VL-JEPA 축소 데모 — V1~V4 검증 실측")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", type=str, default="mps")
    args = ap.parse_args()

    device = args.device if torch.backends.mps.is_available() else "cpu"
    cfg = DemoConfig(epochs=args.epochs)

    print("=" * 64)
    print("VL-JEPA 축소 데모 — V1~V4 검증 항목")
    print(f"device={device}, dim={cfg.dim}, epochs={cfg.epochs}, mask={cfg.mask_ratio}")
    print("=" * 64)

    results = [run(cfg, tau, device) for tau in (cfg.tau_ema, 0.0)]

    for r in results:
        tag = f"tau={r.tau:.2f}" + (" (EMA)" if r.tau > 0 else " (shared)")
        print(f"\n--- {tag} ---")
        print(f"loss curve: first={r.losses[0]:.4f}, last={r.losses[-1]:.4f}")
        delta = (1 - r.final_err / max(r.initial_err, 1e-8)) * 100
        word = "감소" if delta >= 0 else "증가"
        print(f"[V1] baseline_err(means)={r.initial_err:.3f} -> final_err={r.final_err:.3f} "
              f"({abs(delta):.1f}% 오차 {word})")
        print(f"[V2] align(gap) true={r.gap_true:.4f} vs shuffled={r.gap_shuffled:.4f} "
              f"(gap={r.gap_true - r.gap_shuffled:+.4f})")
        print(f"[V3] effective_rank target={r.rank_init:.2f} -> pred={r.rank_final:.2f}")

    r_ema, r_sh = results
    print("\n--- [V4] EMA vs shared ---")
    print(f"final_err  : EMA={r_ema.final_err:.3f} vs shared={r_sh.final_err:.3f}")
    print(f"rank_final : EMA={r_ema.rank_final:.2f} vs shared={r_sh.rank_final:.2f}")

    summary = {
        "epochs": cfg.epochs,
        "device": device,
        "runs": [
            {
                "tau": r.tau, "final_err": round(r.final_err, 4),
                "gap_true": round(r.gap_true, 4), "gap_shuffled": round(r.gap_shuffled, 4),
                "rank_final": round(r.rank_final, 2),
                "loss_curve": [round(l, 4) for l in r.losses],
            }
            for r in results
        ],
    }
    print("\nJSON summary:\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
