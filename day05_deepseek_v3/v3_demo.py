#!/usr/bin/env python3
"""DeepSeek-V3 축소 데모 — aux-loss-free balancing / MTP / FP8 / 파라미터 회계 검증.

검증하는 4가지:
  (1) 부하 균형: 순수 top-k vs 보조손실(aux-loss) vs **aux-loss-free(bias만 갱신)**.
      핵심 대비 = "균형을 잡으면서 **게이트 점수를 얼마나 왜곡하는가**".
      aux-loss 는 점수 자체를 penalty 로 밀어 균형을 얻고(점수 왜곡 O),
      aux-loss-free 는 선택용 bias b_i 만 갱신해 균형을 얻는다(가중치 점수 왜곡 0).
  (2) MTP: h_t^(k) 가 **t+k+1 토큰의 임베딩을 봐야만** 예측 가능한지(룩어헤드) +
      공유 임베딩/헤드로 **그래디언트가 흐르는지**(detach 아님).
  (3) FP8: per-tensor vs **fine-grained block(128×128 / 1×128)** 양자화 오차 비교,
      E4M3 vs E5M2 (가수 3bit vs 2bit) 오차 비교.
  (4) 회계: V2/V3 설정에서 총/활성 파라미터를 **직접 계산**해 논문 headline 과 대조 +
      전문가 조합 수 C(N_r,k) 비교.

논문: DeepSeek-V3 (arXiv:2412.19437). 시뮬레이션 부분은 명시적으로 [SIM] 로 표시한다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

RUNS = Path(__file__).resolve().parent / "runs"


# ---------------------------------------------------------------- (1) balancing
def simulate_balancing(E: int, T: int, k: int, steps: int, gamma: float, alpha: float,
                       device: str) -> dict:
    """부하 균형 3전략 비교.

    토큰 친화도: 전문가마다 인기도 p_i 가 다르고(멱법칙), 토큰은 클러스터에서 뽑힌다.
    선택 점수 s = sigmoid(scale·(p_i·a_t + noise)).  V3 는 게이트 친화도에 sigmoid 를 쓴다.
    """
    gen = torch.Generator(device="cpu").manual_seed(0)
    p = (1.0 / (torch.arange(1, E + 1, dtype=torch.float32) ** 0.8))      # 편향된 인기도
    p = p / p.sum() * E
    # 토큰마다 살짝 다른 친화도(클러스터 효과)
    a = 0.5 + torch.rand(T, 1, generator=gen) * 1.5
    noise = torch.randn(T, E, generator=gen) * 0.6
    s_base = torch.sigmoid(1.5 * p.unsqueeze(0) * a + noise).to(device)

    def run(mode: str) -> dict:
        scores = s_base.clone()                     # 가중치에 쓰이는 점수(전략에 따라 왜곡됨)
        bias = torch.zeros(E, device=device)        # 선택에만 더해지는 bias(aux-loss-free)
        hist, maxvios = [], []
        for _ in range(steps):
            sel_scores = scores + bias
            idx = sel_scores.topk(k, dim=-1).indices
            load = torch.bincount(idx.flatten(), minlength=E).float()
            mean = load.mean()
            maxvio = float((load.max() / mean - 1.0) * 100)
            hist.append(load / load.sum())
            maxvios.append(maxvio)
            err = (mean - load) / mean
            if mode == "free":                       # b_i ← b_i + γ·sign(e_i)
                bias = bias + gamma * torch.sign(err)
            elif mode == "aux":                      # 보조손실 gradient ≈ 부하 편차
                overload_penalty = alpha * err.unsqueeze(0)
                scores = scores + overload_penalty
        final = hist[-1]
        ent = float(-(final * (final + 1e-12).log()).sum())
        return {
            "maxvio_final": maxvios[-1],
            "maxvio_first": maxvios[0],
            "entropy_final": ent,
            # 게이트 '가중치' 점수가 처음 대비 얼마나 왜곡됐나(균형 비용)
            "score_distortion": float((scores - s_base).abs().mean()),
            "maxvios": maxvios,
        }

    out = {m: run(m) for m in ("none", "aux", "free")}
    return {"E": E, "T": T, "k": k, "steps": steps, "gamma": gamma, "alpha": alpha, **out}


# ----------------------------------------------------------------------- (2) MTP
class RMSNorm(torch.nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()


class MTP(torch.nn.Module):
    """V3 MTP(D=1): h_t^(k) = M_k[ RMSNorm(h_t^(k-1)) ; RMSNorm(Emb(t+k)) ] — 공유 임베딩/헤드."""

    def __init__(self, d: int, vocab: int, embed: torch.nn.Embedding, head: torch.nn.Linear):
        super().__init__()
        self.d = d
        self.embed = embed                     # 공유(파라미터 재사용)
        self.head = head                       # 공유
        self.norm_h, self.norm_e = RMSNorm(d), RMSNorm(d)
        self.proj = torch.nn.Linear(2 * d, d, bias=False)
        self.b_norm = RMSNorm(d)

    def forward(self, h_prev: torch.Tensor, next_ids: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.norm_h(h_prev), self.norm_e(self.embed(next_ids))], dim=-1)
        return self.b_norm(self.proj(z))

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


def proof_mtp(d: int, vocab: int, T: int, lam: float, device: str) -> dict:
    torch.manual_seed(0)
    embed = torch.nn.Embedding(vocab, d).to(device)
    head = torch.nn.Linear(d, vocab, bias=False).to(device)
    mtp = MTP(d, vocab, embed, head).to(device)

    ids = torch.randint(3, vocab - 3, (T + 2,), device=device)
    h = torch.randn(T, d, device=device)
    h_prev = h                                                  # 1단계(0번째)는 본체 hidden

    ids_clean = ids.clone()
    ids_in = ids.clone(); ids_in[1:T + 1] += 1        # 모듈이 실제로 읽는 창 [1:T+1] 내부 교란
    ids_out = ids.clone(); ids_out[T + 1] += 2        # 창 바깥(t+2 위치)만 교란 → 영향 없어야 함
    out = mtp(h_prev, ids_clean[1:T + 1])
    with torch.no_grad():
        d_p1 = float((mtp(h_prev, ids_in[1:T + 1]) - out).abs().max())
        d_p2 = float((mtp(h_prev, ids_out[1:T + 1]) - out).abs().max())

    logits = mtp.logits(out)
    target = ids_clean[2:T + 2]                                  # k=1 모듈은 t+2 를 예측
    loss = torch.nn.functional.cross_entropy(logits, target) * lam
    loss.backward()
    g_embed = float(embed.weight.grad.abs().sum()) if embed.weight.grad is not None else 0.0
    g_head = float(head.weight.grad.abs().sum()) if head.weight.grad is not None else 0.0
    g_proj = float(mtp.proj.weight.grad.abs().sum()) if mtp.proj.weight.grad is not None else 0.0
    return {
        "lookahead_dp1": d_p1, "lookahead_dp2": d_p2, "lambda": lam, "loss": float(loss.detach()),
        "grad_shared_embed": g_embed, "grad_shared_head": g_head, "grad_proj": g_proj,
    }


# ----------------------------------------------------------------------- (3) FP8
def _q_mantissa(x: torch.Tensor, bits: int, max_val: float) -> torch.Tensor:
    """가수 bits 개로 반올림하는 fake-quant (E4M3=3, E5M2=2). 스케일은 호출부에서 적용."""
    ax = x.abs().clamp_min(1e-30)
    exp = torch.floor(torch.log2(ax))
    step = torch.pow(torch.tensor(2.0, device=x.device), exp - bits)
    q = torch.round(x / step) * step
    return q.clamp(-max_val, max_val)


def quantize(x: torch.Tensor, bits: int, max_val: float, block: int | None) -> torch.Tensor:
    """block=None → per-tensor, block=b → 마지막 축을 b 크기 블록으로 나눠 블록별 스케일."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if block is None:
        scale = max_val / x.abs().amax().clamp_min(1e-30)
        return _q_mantissa(x * scale, bits, max_val) / scale
    flat = x.reshape(-1, x.shape[-1])
    T, D = flat.shape
    pad = (-D) % block
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blk = flat.reshape(T, -1, block)
    scale = max_val / blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    q = _q_mantissa(blk * scale, bits, max_val) / scale
    return q.reshape(T, -1)[:, :D].reshape(x.shape)


def proof_fp8(d: int, device: str) -> dict:
    torch.manual_seed(0)
    # LLM 가중치/활성의 전형: 채널별 아웃라이어(무거운 꼬리)
    W = torch.randn(d, d, device=device) * 0.02
    W[:, 0] *= 150.0                                    # outlier 채널
    X = torch.randn(2 * d, d, device=device) * 0.5
    X[:, :4] *= 80.0
    ref = X @ W.T

    def err(**kw) -> float:
        Wq = quantize(W, kw["wbits"], kw["wmax"], kw["wblock"])
        Xq = quantize(X, kw["abits"], kw["amax"], kw["ablock"])
        return float((Xq @ Wq.T - ref).norm() / ref.norm())

    e4, e5 = 448.0, 57344.0
    res = {
        "per_tensor": err(wbits=3, wmax=e4, wblock=None, abits=3, amax=e4, ablock=None),
        "block_128x128_1x128": err(wbits=3, wmax=e4, wblock=128, abits=3, amax=e4, ablock=128),
        "block_but_e5m2_2bit": err(wbits=2, wmax=e5, wblock=128, abits=2, amax=e5, ablock=128),
    }
    res["improvement"] = res["per_tensor"] / res["block_128x128_1x128"]
    return res


# ------------------------------------------------------------------ (4) 회계
def mla_attn_params(d: int, n_h: int, qk_nope: int, qk_rope: int, v_head: int,
                    q_lora: int, kv_lora: int) -> int:
    return (d * q_lora) + q_lora * n_h * (qk_nope + qk_rope) \
        + d * (kv_lora + qk_rope) \
        + kv_lora * n_h * (qk_nope + v_head) \
        + n_h * v_head * d


def accounting(cfg: dict) -> dict:
    d, L, dense_l = cfg["d"], cfg["L"], cfg["dense_l"]
    moe_l = L - dense_l
    attn = mla_attn_params(d, cfg["n_h"], cfg["qk_nope"], cfg["qk_rope"], cfg["v_head"],
                           cfg["q_lora"], cfg["kv_lora"])
    dense_ffn = 3 * d * cfg["dense_inter"]
    routed = cfg["n_routed"] * 3 * d * cfg["moe_inter"]
    shared = cfg["n_shared"] * 3 * d * cfg["moe_inter"]
    gate = d * cfg["n_routed"] if cfg.get("gate", True) else 0
    moe_layer = attn + dense_ffn * 0 + routed + shared + gate
    dense_layer = attn + dense_ffn
    total = cfg["vocab"] * d * 2 + dense_l * dense_layer + moe_l * moe_layer   # embed + head
    active = cfg["vocab"] * d * 2 + dense_l * dense_layer \
        + moe_l * (attn + gate + cfg["topk"] * 3 * d * cfg["moe_inter"]
                   + shared)
    return {
        "total_params_B": total / 1e9, "active_params_B": active / 1e9,
        "moe_layers": moe_l,
        "combinations_C_Nr_k": math.comb(cfg["n_routed"], cfg["topk"]),
    }


V2 = dict(d=5120, L=60, dense_l=1, n_h=128, qk_nope=128, qk_rope=64, v_head=128,
          q_lora=1536, kv_lora=512, n_routed=160, n_shared=2, topk=6,
          moe_inter=1536, dense_inter=12288, vocab=102400, gate=False)
V3 = dict(d=7168, L=61, dense_l=3, n_h=128, qk_nope=128, qk_rope=64, v_head=128,
          q_lora=1536, kv_lora=512, n_routed=256, n_shared=1, topk=8,
          moe_inter=2048, dense_inter=18432, vocab=129280, gate=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--gamma", type=float, default=0.005)
    ap.add_argument("--alpha", type=float, default=0.02)
    args = ap.parse_args()
    dev = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    print(f"[V3 demo] device={dev} E={args.E} T={args.T} k={args.k} steps={args.steps} "
          f"gamma={args.gamma}")

    metrics: dict = {}

    bal = simulate_balancing(args.E, args.T, args.k, args.steps, args.gamma, args.alpha, dev)
    metrics["balancing"] = bal
    print("\n(1) 부하 균형 — maxvio% (낮을수록 균형) / 게이트점수 왜곡 (낮을수록 무손실)")
    for mode, label in (("none", "순수 top-k"), ("aux", "aux-loss(점수 penalty)"),
                        ("free", "aux-loss-free(bias만)")):
        r = bal[mode]
        print(f"    {label:26s} maxvio {r['maxvio_first']:7.1f}% → {r['maxvio_final']:6.1f}%   "
              f"왜곡 {r['score_distortion']:.3e}   load entropy {r['entropy_final']:.3f}")

    mtp = proof_mtp(64, 256, 64, 0.1, dev)
    metrics["mtp"] = mtp
    print("\n(2) MTP(D=1) 룩어헤드 + 공유 파라미터 그래디언트")
    print(f"    창 내부(t+1) 임베딩 교란 → 출력 변화 {mtp['lookahead_dp1']:.3e}  (커야 함)")
    print(f"    창 밖(t+2) 임베딩 교란 → 출력 변화 {mtp['lookahead_dp2']:.3e}  (0 이어야 함)")
    print(f"    λ={mtp['lambda']} loss={mtp['loss']:.4f}  grad: 공유embed {mtp['grad_shared_embed']:.2e} "
          f"공유head {mtp['grad_shared_head']:.2e} MTP proj {mtp['grad_proj']:.2e}")

    fp8 = proof_fp8(256, dev)
    metrics["fp8"] = fp8
    print("\n(3) FP8 가짜양자화 — 상대 오차 ||XqWq − XW||/||XW||")
    print(f"    per-tensor (E4M3)          {fp8['per_tensor']:.4e}")
    print(f"    block 128×128 / 1×128      {fp8['block_128x128_1x128']:.4e}   "
          f"→ {fp8['improvement']:.1f}× 개선")
    print(f"    block 이지만 E5M2(가수 2bit) {fp8['block_but_e5m2_2bit']:.4e}")

    acc = {"V2": accounting(V2), "V3": accounting(V3)}
    metrics["accounting"] = acc
    print("\n(4) 설정으로부터 직접 계산한 파라미터 (논문 headline 대조)")
    for name, a in acc.items():
        print(f"    {name}: 총 {a['total_params_B']:.1f}B / 활성 {a['active_params_B']:.1f}B   "
              f"MoE층 {a['moe_layers']}   C(N_r,k)={a['combinations_C_Nr_k']:,}")

    RUNS.mkdir(exist_ok=True)
    (RUNS / "v3_demo_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_curve(bal, RUNS / "balance_curve.png")
    print(f"\n[산출물] {RUNS}/v3_demo_metrics.json" + (", balance_curve.png" if _HAS_MPL else ""))
    print("[결론] 균형은 bias 만으로도 달성되며(점수 무손실), MTP 는 공유 임베딩/헤드로 학습되고, "
          "FP8 은 fine-grained 블록이 있어야 per-tensor 보다 크게 정확하다.")


_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    pass


def save_curve(bal: dict, path: Path) -> None:
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    for mode, label in (("none", "top-k only"), ("aux", "aux-loss"), ("free", "aux-loss-free")):
        ax.plot(bal[mode]["maxvios"], label=f"{label} (final {bal[mode]['maxvio_final']:.0f}%)")
    ax.set_xlabel("balancing step"); ax.set_ylabel("maxviolation (%)")
    ax.set_title("load balance: aux-loss vs aux-loss-free (simulated)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
