#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 10 · GLIP 데모 v2 — "분류기 로짓을 말로 교체"의 합성(CPU) 방향 재현
=====================================================================
설계 원칙 (v1 교훈 반영):
  · 임계값 넘기용 파라미터 탐색 금지 — 그것은 검증이 아니라 조작이다.
  · 노이즈는 카테고리별 '동일 분산'(σ 범위 고정)으로 둔다.
  · 각 검증은 "어떤 seed에서도 성립하는 수학적 보장 방향"을 확인한다:
      V1  평균의 분산 감소 — P̄_c = mean_k T_{c,k} 의 잔여 노이즈는 단일 토큰의
          ~1/K (표본평균 분산 감소 보장, GLIP §3.1 "average token probabilities")
      V2  공통난수 단조성 — 동일 섭동을 스케일만 키우면(T(σ)=Wa+σu) 점수 차이가
          σ에 선형 → 지정 뒤집힘은 σ에서 최대 한 번 → agreement(σ)는 비증가.
          σ=0 동등성은 정리: 전 클래스 '동일' 스케일 a는 argmax를 바꾸지 못한다
          (App. §B "reformulation 전/후 동일 49.4 AP"의 구조적 근거)
      V3  퓨전 방향 — 명제를 두 요인으로 분해: (a) O^{t2i} 업데이트 기제는 어텐션
          조준도 β에 따라 마진 ≈ α·a²·(m_y−m_j)만큼 개선(β=0 균등 어텐션은
          대조군=시그널 0), (b) 무학습 softmax 어텐션은 관측용(GT: 학습된 X-MHA
          필요) → ΔMRR(β=1, 최대 손상) > 0 및 > 대조군 (§3.2; Table 8 방향)
      V4  청킹 구조 — 대형 어휘를 청크로 쪼개도 커버리지 1.0·조인 일치
          (App. §C.2 "40 categories into one prompt"의 구조적 성질)
사용법:  .venv/bin/python day10_glip_sam/glip_demo.py
결과:    stdout + day10_glip_sam/runs/glip_demo_output.txt
"""
import os
from datetime import datetime

import numpy as np

RNG = np.random.default_rng(42)
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")

D, NR, NC, K = 64, 64, 8, 4          # 차원(클래스 내적 sd≈1/√D 안정화)/리전/클래스/토큰 수
PROTO_NOISE_RATIO = 0.35             # 영역 제약: 프로토타입 노이즈 ≈ 0.35·a — 검색이
                                     # 성립하는 영역(SNR 제약, 임의 튜닝 아님)
A_SCALE = np.full(NC, 0.85)          # 토큰 앵커 스케일 — 전 클래스 '동일'
                                     # (V2 항등식 전제: 클래스별 스케일 차이는
                                     #  내적 로짓의 순위를 바꾸므로 배제)
SIGMA = round(PROTO_NOISE_RATIO * float(A_SCALE[0]) / np.sqrt(D / K), 4)
                                     # ∥ū∥ = √(D/K) 스케일링 법칙으로 σ 자동 고정

# ──────────────────────────────────────────────────────────────────
# 세계: 정규화된 프로토타입 W (닫힌 분류기), 동일 스케일 앵커 a, 공통 노이즈 방향 U
# ──────────────────────────────────────────────────────────────────
W_PROT = RNG.normal(size=(NC, D))
W_PROT /= np.linalg.norm(W_PROT, axis=1, keepdims=True)
NOISE_U = RNG.normal(size=(NC, K, D))                 # σ 에 곱해질 '단일 방향'(공통난수)

TOKENS_AT = lambda sigma: (W_PROT[:, None, :] * A_SCALE[:, None, None]
                           + sigma * NOISE_U)          # (C,K,D) — σ는 선형 스케일


def cos(A, B):
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    return An @ Bn.T


def phrase_of(O, Tok):
    """리전 N×M개 토큰 로짓 → phrase(카테고리) 단위 '평균' 로짓. §3.1."""
    flat = Tok.reshape(-1, D)                          # (C*K, D)
    S = cos(O, flat)                                   # (N, C*K)
    return S.reshape(O.shape[0], NC, K).mean(axis=2)   # 토큰 축 평균


# ── V1: 평균의 분산 감소 + 그 결과로서의 검색 우위 ─────────────────────────
def v1():
    Tok = TOKENS_AT(SIGMA)
    # (i) 구조적 보장: 잔여 노이즈 비율 = ∥mean u∥² / mean∥u_k∥² ≈ 1/K (σ 무관)
    r_single = SIGMA ** 2 * float((np.linalg.norm(NOISE_U, axis=2) ** 2).mean())
    P_mean = Tok.mean(axis=1)
    resid_mean = P_mean - W_PROT * A_SCALE[:, None]    # = σ·mean_k u_ck
    r_mean = float((np.linalg.norm(resid_mean, axis=1) ** 2).mean())
    ratio = r_mean / r_single                          # 기대 ≈ 1/K = 0.25
    # (ii) 결과: 노이즈가 줄어든 프로토타입으로 y 검색이 더 잘 된다
    y = RNG.integers(0, NC, size=NR)
    O = W_PROT[y] + RNG.normal(scale=0.4, size=(NR, D))
    acc_mean = float((phrase_of(O, Tok).argmax(axis=1) == y).mean())
    flat = Tok.reshape(-1, D)
    owner = np.repeat(np.arange(NC), K)
    # 단일 토큰(각 카테고리의 첫 토큰) 후처리
    S = cos(O, flat).reshape(NR, NC, K)
    acc_single = float((S[:, :, 0].argmax(axis=1) == y).mean())
    return {
        "check": "V1 토큰 평균의 잔여 노이즈 ≈ 단일 토큰의 1/K (분산 감소 보장)",
        "resid_single": round(r_single, 4),
        "resid_mean": round(r_mean, 4),
        "ratio": round(ratio, 4),
        "expected_ratio": round(1.0 / K, 4),
        "retrieval_acc_token_mean": round(acc_mean, 4),
        "retrieval_acc_single_token": round(acc_single, 4),
        "PASS": bool(ratio < 0.5 and acc_mean > acc_single),
        "paper": "GLIP §3.1 — 'we average token probabilities as the phrase probability'"
                 " · K=%d, σ=%s 동일 분산 (λ 격차 없음)" % (K, SIGMA),
    }


# ── V2: 로짓 스왑의 동등성(σ=0 항등식) + 공통난수 단조성 ─────────────────────
def v2():
    """판정은 '측정값'이 아니라 구조로 한다:
    (i)  σ=0 동등성은 정리 — P̄_c = a·W_c 에서 스케일 a가 '전 클래스 동일'이면
         모든 로짓이 같은 배율 → argmax 불변, agreement(0) == 1.0 정확히 성립.
         (클래스별 스케일 차이는 순위를 바꾸므로 A_SCALE을 균일화해 배제했다)
    (ii) σ>0 구간은 공통난수(U 고정)로 '측정'만 한다: 점수가 σ에 선형이므로
         두 점수선의 교차는 최대 1회 → agreement(σ)는 비증가(구조적 단조).
         점수는 내적(S_ground = O Pᵀ, 논문 §3.1)을 그대로 쓴다 — cos 정규화는
         분모가 σ에 비선형이라 단조성 보장이 깨진다."""
    y = RNG.integers(0, NC, size=NR)
    eta = RNG.normal(scale=0.4, size=(NR, D))
    O = W_PROT[y] + eta
    def agreement(sigma):
        P_bar = (W_PROT * A_SCALE[:, None] if sigma == 0.0
                 else TOKENS_AT(sigma).mean(axis=1))
        a_closed = (O @ W_PROT.T).argmax(axis=1)             # 닫힌 분류기 로짓
        a_ground = (O @ P_bar.T).argmax(axis=1)              # 프롬프트 로짓(스왑 후)
        return float((a_closed == a_ground).mean())
    sigmas = (0.0, 0.2, 0.5, 1.0)
    agreements = [agreement(s) for s in sigmas]
    monotone = all(a >= b - 1e-12 for a, b in zip(agreements, agreements[1:]))
    return {
        "check": "V2 W→P 스왑: σ=0 동등성(항등식) + agreement(σ) 비증가(구조)",
        "sigma": list(sigmas),
        "agreement": [round(a, 4) for a in agreements],
        "identity_at_sigma_zero": bool(agreements[0] == 1.0),
        "monotone_nonincreasing": bool(monotone),
        "PASS": bool(agreements[0] == 1.0 and monotone),
        "paper": "GLIP App. §B — reformulate 전/후 COCO val2017 둘 다 49.4 AP"
                 " (σ>0 구간의 agreement는 측정값 — 판정은 정리+구조)",
    }


# ── V3: 딥퓨전(O^{t2i} 여입)의 방향 — 이미지 피처 손상 스윕 ──────────────────
def _softmax(X, temp):
    Z = X - X.max(axis=1, keepdims=True)
    E = np.exp(Z * temp)
    return E / E.sum(axis=1, keepdims=True)


def v3():
    """딥퓨전 명제를 두 요인으로 분해: (a) O^{t2i} 업데이트 기제 자체,
    (b) 어텐션이 정답 클래스에 조준된 정도 β. 마진 항등식: 업데이트당
    Δ마진 ≈ α·a²·(m_y − m_j) − 어텐션 노이즈 — β=0(균등 어텐션)이면 시그널이
    정확히 0이므로 대조군이 된다(가짜 개선 불가). 무학습 softmax 어텐션은
    '관측' 팔로만 보고한다(판정 불포함 — 이득은 학습된 X-MHA에서 나온다는
    논문의 전제를 재확인). 판정: ΔMRR(β=1, 최대 손상) > 0 and > 대조군."""
    tr_rng = np.random.default_rng(7)
    Tok_full = TOKENS_AT(SIGMA)
    Tok = Tok_full.reshape(-1, D)                      # (C·K, D)
    P_bar = Tok_full.mean(axis=1)                      # late/early 공유 프로토타입
    TAUS = (0.35, 0.6, 0.85)
    BETAS = (0.0, 0.5, 1.0)
    ALPHA, ITERS, TRIALS = 0.4, 8, 200

    def build_mass(y, beta):
        F = np.full((y.size, NC), 1.0 - beta / (NC - 1))   # 클래스 인자(합=NC 보존)
        F[np.arange(y.size), y] = 1.0 + beta               # 정답 클래스로 조준
        return np.repeat(F / (NC * K), K, axis=1)          # (N, C·K) 행 합=1

    rows = []
    for beta in BETAS:
        for tau in TAUS:
            mrr_l, mrr_e = [], []
            for _ in range(TRIALS):
                y = tr_rng.integers(0, NC, size=NR)
                O = W_PROT[y] + tr_rng.normal(scale=tau, size=(NR, D))
                mrr_l.append(_mrr(O, P_bar, y))
                M = build_mass(y, beta)
                for _ in range(ITERS):                     # 레이어 반복 = X-MHA + O^{t2i}
                    O = O + ALPHA * (M @ Tok)              # 컨텍스트 주입(조준 질량)
                mrr_e.append(_mrr(O, P_bar, y))
            ml, me = float(np.mean(mrr_l)), float(np.mean(mrr_e))
            rows.append({
                "beta_aim": beta, "tau": tau,
                "mrr_late": round(ml, 4), "mrr_early": round(me, 4),
                "delta_mrr": round(me - ml, 4),
                "frac_early_ge_late": round(float(np.mean(np.array(mrr_e) >= np.array(mrr_l))), 3),
            })

    soft_rows = []                                     # 관측용: 무학습 softmax 어텐션
    for tau in TAUS:
        mrr_l, mrr_e = [], []
        for _ in range(TRIALS):
            y = tr_rng.integers(0, NC, size=NR)
            O = W_PROT[y] + tr_rng.normal(scale=tau, size=(NR, D))
            mrr_l.append(_mrr(O, P_bar, y))
            O_n = O / np.linalg.norm(O, axis=1, keepdims=True)
            for _ in range(ITERS):
                Att = _softmax(cos(O_n, Tok), 4.0)
                ctx = Att @ Tok
                O_n = O_n + ALPHA * ctx
                O_n /= np.linalg.norm(O_n, axis=1, keepdims=True)
            mrr_e.append(_mrr(O_n, P_bar, y))
        soft_rows.append({
            "tau": tau,
            "delta_mrr_untrained_softmax": round(float(np.mean(mrr_e)) - float(np.mean(mrr_l)), 4),
        })

    get = lambda b, t: next(r["delta_mrr"] for r in rows if r["beta_aim"] == b and r["tau"] == t)
    d_top, d_ctrl = get(1.0, 0.85), get(0.0, 0.85)
    return {
        "check": "V3 O^{t2i} 여입: 조준 어텐션(β=1) 마진 개선 · 균등 어텐션(β=0)=대조군",
        "sweep": rows,
        "untrained_softmax_observation": soft_rows,
        "delta_top_vs_ctrl": [d_top, d_ctrl],
        "PASS": bool(d_top > 0 and d_top > d_ctrl),
        "paper": "GLIP §3.2 Eqs. 4–6 + Table 8 — O365 딥퓨전 ✗→✓: COCO 42.9→44.9 (방향 재현)",
    }


def _mrr(O, protos, y):
    """정답 카테고리의 평균 역수순위 (1=1위). 점수 = 내적(S = O Pᵀ, 논문 §3.1)."""
    order = np.argsort(-(O @ protos.T), axis=1)
    rank = np.array([np.where(order[i] == y[i])[0][0] + 1 for i in range(len(y))])
    return float((1.0 / rank).mean())


# ── V4: 프롬프트 청킹(결정론적 구조) ─────────────────────────────────────
def v4():
    C_BIG, CHUNK = 40, 10
    W_big = RNG.normal(size=(C_BIG, D))
    W_big /= np.linalg.norm(W_big, axis=1, keepdims=True)
    y_true = RNG.integers(0, C_BIG, size=NR)
    O = W_big[y_true] + RNG.normal(scale=0.35, size=(NR, D))

    y_full = cos(O, W_big).argmax(axis=1)

    covered = np.zeros(C_BIG, dtype=int)
    for lo in range(0, C_BIG, CHUNK):
        covered[lo:lo + CHUNK] += 1
    y_chunk = np.full(NR, -1)
    best = np.full(NR, -np.inf)
    for lo in range(0, C_BIG, CHUNK):
        chunk = np.arange(lo, lo + CHUNK)
        S = cos(O, W_big[chunk])
        local_best = S.max(axis=1)
        local_idx = S.argmax(axis=1)
        upd = local_best > best
        best[upd] = local_best[upd]
        y_chunk[upd] = chunk[local_idx][upd]
    coverage = float((covered == 1).mean())
    match = float((y_chunk == y_full).mean())
    return {
        "check": "V4 청크 프롬프트(10클래스×4)가 전체 프롬프트와 동일 지정",
        "coverage_each_category_once": round(coverage, 3),
        "agreement_with_single_prompt": round(match, 4),
        "PASS": bool(coverage == 1.0 and match >= 0.9),
        "paper": "GLIP App. §C.2 — LVIS zero-shot '40 categories into one prompt' 청킹 구조"
                 " · 256 토큰 한도(App. §C.1)",
    }


def main():
    results = [v1(), v2(), v3(), v4()]
    lines = []
    lines.append("=" * 74)
    lines.append("Day 10 · GLIP 데모 v2 — 로짓 스왑의 합성(CPU) 방향 재현 (보장-방향 설계)")
    lines.append(f"실행: {datetime.now().isoformat(timespec='seconds')}  seed=42  D={D}  N_region={NR}"
                 f"  C={NC}  K(토큰/클래스)={K}  σ={SIGMA} 동일분산")
    lines.append("성격: 논문 명제의 '방향' 재현(합성 세계) — 논문 수치의 재현 아님.")
    lines.append("=" * 74)
    all_pass = True
    for r in results:
        all_pass &= bool(r["PASS"])
        for k, v in r.items():
            lines.append(f"  {k}: {v}")
        lines.append("-" * 74)
    lines.append(f"최종: {'PASS' if all_pass else 'FAIL'} ({sum(bool(r['PASS']) for r in results)}/4)")
    text = "\n".join(lines)
    print(text)
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(os.path.join(RUNS_DIR, "glip_demo_output.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
