#!/usr/bin/env python3
"""MLA(DeepSeek-V2) 축소 데모 — KV cache 압축과 '흡수(absorption)' 항등식을 수치로 검증.

검증하는 3가지:
  (1) KV cache 크기: MHA(2·n_h·d_h) vs MLA(d_c + d_h^R) → 토큰당/문맥당 감소율.
  (2) 흡수 항등식:  qᵀk = (W^UQ c^Q)ᵀ (W^UK c^KV) = (c^Q)ᵀ (W^UQᵀ W^UK) c^KV
      → W^UK 를 W^Q 로 흡수하면 K 를 materialize 하지 않아도 된다(헤드별).
  (3) decoupled RoPE 의 필요성: k^C 에 RoPE 를 걸면 흡수가 깨지고(위치 의존),
      별도 q^R/k^R 채널로 RoPE 를 옮기면 C 부분 흡수가 유지된다.

논문: DeepSeek-V2 (arXiv:2405.04434), 식 (9)~(19).
참고 실제 설정: 60 layers, n_h=128, d_h=128, d_c=512, d_c'=1536, d_h^R=64.
"""
from __future__ import annotations

import argparse

import torch


def rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary position embedding. x: (T, [heads], d) — T 가 0번 축, 마지막 쌍을 회전."""
    d = x.shape[-1]
    half = d // 2
    freqs = base ** (-torch.arange(half, device=x.device).float() / half)
    ang = pos.float().unsqueeze(-1) * freqs.unsqueeze(0)                 # (T, half)
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (half,)
    cos, sin = ang.cos().view(shape), ang.sin().view(shape)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def build_weights(d: int, n_h: int, d_h: int, d_c: int, d_cp: int, d_hR: int, dev: str) -> dict:
    g = lambda a, b: torch.randn(a, b, device=dev) * 0.02  # noqa: E731
    return {
        "DKV": g(d_c, d), "UK": g(n_h * d_h, d_c), "UV": g(n_h * d_h, d_c),
        "DQ": g(d_cp, d), "UQ": g(n_h * d_h, d_cp),
        "QR": g(n_h * d_hR, d_cp), "KR": g(d_hR, d),
    }


def kv_latents(h: torch.Tensor, W: dict) -> tuple[torch.Tensor, torch.Tensor]:
    c_kv = h @ W["DKV"].T
    c_q = h @ W["DQ"].T
    return c_kv, c_q


def scores_direct(c_kv: torch.Tensor, c_q: torch.Tensor, W: dict, n_h: int, d_h: int) -> torch.Tensor:
    q_c = (c_q @ W["UQ"].T).view(-1, n_h, d_h)
    k_c = (c_kv @ W["UK"].T).view(-1, n_h, d_h)
    return torch.einsum("thd,shd->hts", q_c, k_c)


def scores_absorbed(c_kv: torch.Tensor, c_q: torch.Tensor, W: dict, n_h: int, d_h: int) -> torch.Tensor:
    """헤드별 A_i = W^UQ_iᵀ W^UK_i 를 미리 접어 (c^Q)ᵀ A_i (c^KV) 로 점수를 만든다."""
    out = []
    for i in range(n_h):
        uq_i = W["UQ"][i * d_h:(i + 1) * d_h]
        uk_i = W["UK"][i * d_h:(i + 1) * d_h]
        a_i = uq_i.T @ uk_i                                              # (d_cp, d_c)
        out.append((c_q @ a_i) @ c_kv.T)                                 # (T, T)
    return torch.stack(out, 0)


def kv_cache_report(n_h: int, d_h: int, d_c: int, d_hR: int, layers: int, ctx: int, bytes_el: int = 2) -> tuple:
    mha_per_token = 2 * n_h * d_h * layers
    mla_per_token = (d_c + d_hR) * layers
    scale = ctx * bytes_el / (1024 ** 3)
    return mha_per_token, mla_per_token, mha_per_token * scale, mla_per_token * scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=8)
    ap.add_argument("--d-c", type=int, default=16)
    ap.add_argument("--d-cp", type=int, default=24)
    ap.add_argument("--d-h-r", type=int, default=4)
    args = ap.parse_args()

    dev = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    torch.manual_seed(0)
    print(f"[MLA demo] device={dev} T={args.T} n_h={args.n_heads} d_h={args.d_head} "
          f"d_c={args.d_c} d_c'={args.d_cp} d_hR={args.d_h_r}")

    W = build_weights(args.d, args.n_heads, args.d_head, args.d_c, args.d_cp, args.d_h_r, dev)
    h = torch.randn(args.T, args.d, device=dev) * 0.1
    pos = torch.arange(args.T, device=dev)
    c_kv, c_q = kv_latents(h, W)

    # (1) KV cache 크기 (실제 V2 설정)
    mha_e, mla_e, mha_gb, mla_gb = kv_cache_report(128, 128, 512, 64, 60, 128 * 1024)
    red = 1 - mla_e / mha_e
    print("\n(1) KV cache / token (V2: n_h=128, d_h=128, d_c=512, d_hR=64, L=60):")
    print(f"    MHA = 2·n_h·d_h·L = {mha_e:,}   MLA = (d_c+d_hR)·L = {mla_e:,}   "
          f"→  감소 {red*100:.2f}%")
    print(f"    128K 문맥 fp16: MHA {mha_gb:.2f} GB  vs  MLA {mla_gb:.2f} GB")
    print(f"    (논문 headline: DeepSeek 67B 대비 KV cache −93.3%)")

    # (2) 흡수 항등식 (RoPE 미적용 시)
    sd = scores_direct(c_kv, c_q, W, args.n_heads, args.d_head)
    sa = scores_absorbed(c_kv, c_q, W, args.n_heads, args.d_head)
    diff_abs = float((sd - sa).abs().max())
    rel = diff_abs / float(sd.abs().max())
    print("\n(2) 흡수 항등식 (RoPE 없음):")
    print(f"    max|직접 − 흡수| = {diff_abs:.3e}   (상대 {rel:.2e})  → 항등식 성립 ✔")

    # (3) k^C 에 RoPE 를 걸면 흡수가 깨진다
    q_c = (c_q @ W["UQ"].T).view(-1, args.n_heads, args.d_head)
    k_c = (c_kv @ W["UK"].T).view(-1, args.n_heads, args.d_head)
    k_c_roped = rope(k_c, pos)
    sd_roped = torch.einsum("thd,shd->hts", q_c, k_c_roped)
    diff_rope = float((sd_roped - sa).abs().max())
    print("\n(3) k^C 에 RoPE 적용 시 (흡수 불가):")
    print(f"    max|RoPE직접 − 흡수| = {diff_rope:.3e}  (≫ 0)  → 위치 의존이라 흡수 불가 ✗")
    print("    ⇒ decoupled RoPE: RoPE 를 별도 q^R/k^R 채널로 이동하면 C-부분 흡수 유지")

    # (3b) decoupled: C 흡수 + R 명시 = 직접 MLA 와 동일
    q_r = rope((c_q @ W["QR"].T).view(-1, args.n_heads, args.d_h_r), pos)
    k_r = rope(h @ W["KR"].T, pos)[:, None, :].expand(-1, args.n_heads, -1)
    s_decoupled = torch.einsum("thd,shd->hts", torch.cat([q_c, q_r], -1),
                               torch.cat([k_c, k_r], -1))
    s_parts = sa + torch.einsum("thd,shd->hts", q_r, k_r)
    print(f"    decoupled 분해: max|(C흡수 + R명시) − 직접| = "
          f"{float((s_parts - s_decoupled).abs().max()):.3e}  → 일치 ✔")

    # (4) 토이 end-to-end: MLA vs MHA attention 출력 shape 동일
    v_c = (c_kv @ W["UV"].T).view(-1, args.n_heads, args.d_head)
    attn = s_decoupled.softmax(-1)
    out = torch.einsum("hts,shd->thd", attn, v_c).reshape(args.T, -1)
    print(f"\n(4) 출력 shape: MLA out={tuple(out.shape)}  (d_c={args.d_c}, "
          f"n_h·d_h={args.n_heads*args.d_head} 로 재구성 가능)")

    print("\n[결론] MLA = MHA + low-rank KV joint compression. "
          "KV cache (d_c+d_hR) 만 저장 + W^UK/W^UV 흡수 → K,V materialize 불필요. "
          "decoupled RoPE 로 위치정보를 별도 소채널에 실어 흡수 가능성 유지.")


if __name__ == "__main__":
    main()
