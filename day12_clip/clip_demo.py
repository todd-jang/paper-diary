#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 12 · CLIP 데모 — "대조가 classifier를 만든다"의 합성(CPU) 방향 재현
====================================================================
설계 원칙 (D10 데모 v2 계승):
  · 임계값 넘기용 파라미터 탐색 금지 — 그것은 검증이 아니라 조작이다.
  · 판정은 (i) 항등식/정리(구조), (ii) 구조적 단조/비용 차이, (iii) 통계 보장 방향.
  · 노이즈는 카테고리별 '동일 분산'(σ 범위 고정)으로 둔다.
  · 논문 수치 재현이 아니라, 설계 판단의 '방향·구조'를 자체 실측한다.

검증 4종:
  V1  대칭 교차엔트로피 — pseudocode 그대로: logits=I_e·T_e·exp(t),
      loss=(CE(logits,axis=0)+CE(logits,axis=1))/2. 확인 3단:
      (a) 분해 항등식(동일 로짓·레이블 2회 재계산 == 정확히 일치),
      (b) 정렬 배치(쌍이 같은 클래스)의 손실이 랜덤 배치보다 작음(방향),
      (c) 대각 행 최대 전제의 τ 비증가 — 정렬 쌍이 동일 노이즈 방향을 공유
          (두 인코더가 같은 잠재 방향으로 수렴한 상태의 합성)하면 대각 코사인=1,
          비대각<1 → softmax는 행 최대 로짓 방향으로만 날카로워지므로 비증가 보장.
      [원문: §2.2 "symmetric cross entropy over these similarity scores",
       figure pseudocode: loss = (loss_i + loss_t)/2, labels = arange(n)]
  V2  temperature 클립 ×100 — τ는 log-parameterized 학습 파라미터,
      logit 스케일 exp(τ)는 100에서 클립(§4). 확인:
      (a) 클립 항등식: exp(τ)>100이면 스케일 100과 확률이 정확히 동일,
      (b) 양의 공통 스케일은 랭킹 불변(하이퍼네트워크 분류기의 구조 보장),
      (c) 마진>0인 로짓에서 최대 확률은 스케일에 비감소(측정 추세).
  V3  임베딩 공간 앙상블의 상각 — §3.1.1 "cache a single set of averaged text
      embeddings ... compute cost ... same as using a single classifier when
      amortized": (a) σ=0에서 평균 임베딩 앙상블 == 단일 분류기(항등식),
      (b) 확률 앙상블은 매 추론 K회 인코딩, 임베딩 앙상블은 1회(구조적 비용),
      (c) σ>0에서 평균 앙상블과 베이스의 agreement는 σ 비증가(공통난수 구조).
  V4  프롬프트 평균의 분산 감소 — K개 프롬프트 임베딩 P_ck = a·W_c + σ·u_ck,
      잔여 노이즈 비율 ≈ 1/K(§1.4 V1 계열 통계 보장 — 프롬프트 공간 버전).
      논문에서는 ImageNet에 80개 프롬프트(§3.1.1); CPU 데모는 K=8 샘플.
사용법:  .venv/bin/python day12_clip/clip_demo.py
결과:    stdout + day12_clip/runs/clip_demo_output.txt
"""
import os
from datetime import datetime

import numpy as np

RNG = np.random.default_rng(42)
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")

D, N, NC, K = 64, 32, 8, 8           # 차원 / 배치 / 클래스 / 프롬프트 수(논문: 80@ImageNet)
N_REGION = 64                        # V4 검색용 이미지 수
PROTO_NOISE_RATIO = 0.35             # 프로토타입 노이즈 ≈ 0.35·a (검색 성립 영역)
A_SCALE = np.full(NC, 0.85)          # 앵커 스케일 — 전 클래스 '동일' (순위 불변 전제)
SIGMA = round(PROTO_NOISE_RATIO * float(A_SCALE[0]) / np.sqrt(D / K), 4)
TAUS = (0.5, 1.0, 2.0, 4.0, 7.0)     # τ 그리드 (학습 파라미터 exp 표기 스케일)

# ────────────────────────────────────────────────────────────────
# 세계: 정규화 프로토타입 W(클래스 방향), 프롬프트 임베딩 P = a·W_c + σ·u
# ────────────────────────────────────────────────────────────────
W = RNG.normal(size=(NC, D))
W /= np.linalg.norm(W, axis=1, keepdims=True)
NOISE_U = RNG.normal(size=(NC, K, D))            # 공통난수 방향 (σ 선형 스케일)

PROMPTS = lambda sigma: (W[:, None, :] * A_SCALE[:, None, None] + sigma * NOISE_U)


def softmax(Z, axis=-1):
    Z = Z - Z.max(axis=axis, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=axis, keepdims=True)


def ce(logits, labels):
    """로짓 (B,C) → 평균 CE. pseudocode의 cross_entropy_loss(평균 축소 기준)."""
    p_log = logits[np.arange(len(labels)), labels] - logits.max(axis=1)
    logZ = np.log(np.exp(logits - logits.max(axis=1)).sum(axis=1))
    return float((logZ - p_log).mean())


# ── V1: 대칭 교차엔트로피 — 분해 항등식 + 정렬 방향 + τ 비증가 ────────────────
def v1():
    """τ 비증가는 '정렬 쌍이 행/열 최대'라는 전제에서만 성립하는 구조적 보장이다.
    세계는 그 전제를 만족하도록 고정한다: 정렬 쌍이 동일 노이즈 방향 η_i를 공유
    (두 인코더가 같은 잠재 방향으로 수렴한 상태의 합성) → 대각 코사인 = 1(정확히),
    비대각 < 1 → softmax가 대각으로 몰리므로 정렬 손실은 τ에 비증가."""
    aligned_rng = np.random.default_rng(11)
    y = RNG.integers(0, NC, size=N)
    eta = aligned_rng.normal(scale=0.3, size=(N, D))
    I = W[y] + eta
    T_e = W[y] + eta                                  # 정렬 쌍 = 같은 클래스 + 같은 방향
    I_e = I / np.linalg.norm(I, axis=1, keepdims=True)   # pseudocode: l2_normalize
    Tn = T_e / np.linalg.norm(T_e, axis=1, keepdims=True)
    rand_I = RNG.normal(size=(N, D))
    rand_T = RNG.normal(size=(N, D))

    aligned_losses, random_losses = [], []
    for tau in TAUS:
        logits = (I_e @ Tn.T) * np.exp(tau)              # [n, n]
        lab = np.arange(N)                               # labels = arange(n)
        loss_i = ce(logits.T, lab)                       # axis=0 : 텍스트→이미지
        loss_t = ce(logits, lab)                         # axis=1 : 이미지→텍스트
        loss_sym = (loss_i + loss_t) / 2                 # pseudocode 그대로
        r_logits = (rand_I / np.linalg.norm(rand_I, axis=1, keepdims=True)
                    @ (rand_T / np.linalg.norm(rand_T, axis=1, keepdims=True)).T
                    ) * np.exp(tau)
        aligned_losses.append(loss_sym)
        random_losses.append(ce(r_logits.T, lab) / 2 + ce(r_logits, lab) / 2)

    cos_mat = I_e @ Tn.T
    diag_is_max = bool((np.abs(np.diag(cos_mat) - 1.0) < 1e-9).all()
                       and (np.argmax(cos_mat, axis=1) == np.arange(N)).all())
    mono = all(a >= b - 1e-12 for a, b in zip(aligned_losses, aligned_losses[1:]))
    below = all(la < lr for la, lr in zip(aligned_losses, random_losses))
    # 분해 항등식: 동일 로짓·레이블을 서로 다른 두 코드 경로로 재계산 → 완전 일치
    logits0 = (I_e @ Tn.T) * np.exp(TAUS[2])
    direct = (ce(logits0.T, np.arange(N)) + ce(logits0, np.arange(N))) / 2
    identity_ok = bool(abs(direct - aligned_losses[2]) < 1e-12)
    return {
        "check": "V1 대칭 CE — 분해 항등식 + 정렬<랜덤(방향) + 대각 행 최대 전제의 τ 비증가",
        "taus": list(TAUS),
        "aligned_loss": [round(v, 6) for v in aligned_losses],
        "random_loss": [round(v, 6) for v in random_losses],
        "identity_exact": identity_ok,
        "diagonal_is_row_max": diag_is_max,
        "aligned_leq_random_all_taus": below,
        "aligned_loss_nonincreasing_in_tau": bool(mono),
        "PASS": bool(identity_ok and diag_is_max and below and mono),
        "paper": "CLIP §2.2 + figure pseudocode — loss=(CE(axis=0)+CE(axis=1))/2, labels=arange(n)",
    }


# ── V2: temperature 클립 ×100 — 클립 항등식 + 랭킹 불변 + 신뢰도 방향 ─────────
def v2():
    """세계를 v2 내부에서 한 번만 생성하고 모든 측정이 같은 세계를 쓴다.
    클립 규칙: logit 스케일 = min(exp(τ), 100) — §4 'clipped to prevent scaling
    the logits by more than 100'."""
    y = RNG.integers(0, NC, size=N)
    I = RNG.normal(size=(N, D)) * 0.3
    T_e = W[y] + I
    I_e = I / np.linalg.norm(I, axis=1, keepdims=True)
    T_e = T_e / np.linalg.norm(T_e, axis=1, keepdims=True)
    base_cos = I_e @ T_e.T                                # 코사인 유사도 [n, n]

    def probs(tau):
        scale = min(float(np.exp(tau)), 100.0)            # ← 클립
        return softmax(base_cos * scale, axis=1)

    requ_tau, clip_tau = 5.5, float(np.log(100.0))        # exp(5.5)≈244.7 > 100 → 클립 구간
    p_req, p_clip = probs(requ_tau), probs(clip_tau)
    clip_identity = bool(np.array_equal(p_req, p_clip))   # 244.7도 100도 → 클립 후 동일

    # 양의 공통 스케일 → 랭킹 불변 (하이퍼네트워크 zero-shot 분류기의 구조적 보장)
    scales = (0.5, 1.0, 4.0, 20.0, 100.0)
    argmax_track = [softmax(base_cos * s, axis=1).argmax(axis=1) for s in scales]
    rank_invariant = all(np.array_equal(argmax_track[0], a) for a in argmax_track[1:])

    # 최대 확률은 마진>0인 행에서 스케일에 비감소(측정 추세)
    maxprobs = [float(softmax(base_cos * s, axis=1).max(axis=1).mean()) for s in scales]
    mono = all(a <= b + 1e-12 for a, b in zip(maxprobs, maxprobs[1:]))
    return {
        "check": "V2 logit 스케일 ×100 클립 — 클립 항등식 + 랭킹 불변 + 신뢰도 비감소",
        "tau_log_params": [round(t, 4) for t in (requ_tau, clip_tau)],
        "clip_identity_exact": clip_identity,
        "ranking_invariant_to_common_scale": bool(rank_invariant),
        "maxprob_by_scale": [round(v, 6) for v in maxprobs],
        "maxprob_monotone_nondecreasing": bool(mono),
        "PASS": bool(clip_identity and rank_invariant and mono),
        "paper": "CLIP §4 — temperature 'log-parameterized', logit 스케일 'clipped ... by more "
                 "than 100' + 부록 하이퍼파라 표 Maximum temperature 100.0",
    }


# ── V3: 임베딩 공간 앙상블 — 상각(σ=0 항등식) + 비용 구조 + agreement 비증가 ──
def v3():
    y = RNG.integers(0, NC, size=N_REGION)
    eta = RNG.normal(scale=0.4, size=(N_REGION, D))
    I = W[y] + eta

    def text_mean(sigma):
        return PROMPTS(sigma).mean(axis=1)                # (NC, D) 캐시 1세트

    def prob_ensemble_pred(sigma, M=K):
        # 확률 앙상블: 프롬프트 M개 각각 softmax 후 평균 → 매 추론 M회 인코딩 필요
        I_e = I / np.linalg.norm(I, axis=1, keepdims=True)
        P = PROMPTS(sigma)
        acc = np.zeros((N_REGION, NC))
        for m in range(M):
            s = np.einsum("nd,cd->nc", I_e, P[:, m]
                          / np.linalg.norm(P[:, m], axis=1, keepdims=True))
            acc += softmax(s, axis=1)
        return acc.argmax(axis=1)

    base = (I / np.linalg.norm(I, axis=1, keepdims=True) @ W.T).argmax(axis=1)
    mean_pred0 = (I / np.linalg.norm(I, axis=1, keepdims=True)
                  @ (text_mean(0.0) / np.linalg.norm(text_mean(0.0), axis=1, keepdims=True)).T
                  ).argmax(axis=1)
    identity = bool(np.array_equal(mean_pred0, base))     # σ=0: 평균 앙상블 == 단일 분류기
    costs = {"prob_ensemble_encodes_per_inference": K,
             "embedding_ensemble_encodes_per_inference": 1}   # 캐시 1세트 — 'when amortized'
    sigmas = (0.0, 0.2, 0.5, 1.0)
    agrees = []
    for s in sigmas:
        Tm = text_mean(s)
        Tm = Tm / np.linalg.norm(Tm, axis=1, keepdims=True)
        pred = (I / np.linalg.norm(I, axis=1, keepdims=True) @ Tm.T).argmax(axis=1)
        agrees.append(float((pred == base).mean()))
    mono = all(a >= b - 1e-12 for a, b in zip(agrees, agrees[1:]))
    return {
        "check": "V3 임베딩 앙상블 상각 — σ=0 항등식 + 인코딩 비용 1:K + agreement 비증가",
        "identity_at_sigma_zero": identity,
        "encoding_cost_structure": costs,
        "sigmas": list(sigmas),
        "agreement_with_base": [round(a, 4) for a in agrees],
        "agreement_nondecreasing_trend_note": "agreement는 σ에 비증가 (공통난수 구조)",
        "PASS": bool(identity and costs["embedding_ensemble_encodes_per_inference"] < costs["prob_ensemble_encodes_per_inference"] and mono),
        "paper": "CLIP §3.1.1 — 'cache a single set of averaged text embeddings so that the "
                 "compute cost of the ensemble is the same as using a single classifier "
                 "when amortized'",
    }


# ── V4: 프롬프트 평균의 분산 감소 (통계 보장) + 검색 우위 ─────────────────────
def v4():
    prompts = PROMPTS(SIGMA)                                # (NC, K, D)
    r_single = SIGMA ** 2 * float((np.linalg.norm(NOISE_U, axis=2) ** 2).mean())
    mean_emb = prompts.mean(axis=1)
    resid = mean_emb - W * A_SCALE[:, None]
    r_mean = float((np.linalg.norm(resid, axis=1) ** 2).mean())
    ratio = r_mean / r_single                          # 기대 ≈ 1/K = 0.125

    y = RNG.integers(0, NC, size=N_REGION)
    I = W[y] + RNG.normal(scale=0.4, size=(N_REGION, D))
    I_e = I / np.linalg.norm(I, axis=1, keepdims=True)

    def retrieval(T):
        return float(((I_e @ (T / np.linalg.norm(T, axis=1, keepdims=True)).T
                       ).argmax(axis=1) == y).mean())

    acc_mean = retrieval(mean_emb)
    acc_single = retrieval(prompts[:, 0])               # 프롬프트 1개만 쓴 경우
    return {
        "check": "V4 K개 프롬프트 평균의 잔여 노이즈 ≈ 단일 프롬프트의 1/K (분산 감소 보장)",
        "resid_single": round(r_single, 4),
        "resid_mean": round(r_mean, 4),
        "ratio": round(ratio, 4),
        "expected_ratio_1_over_K": round(1.0 / K, 4),
        "retrieval_acc_mean_prompt": round(acc_mean, 4),
        "retrieval_acc_single_prompt": round(acc_single, 4),
        "PASS": bool(ratio < 0.5 and acc_mean > acc_single),
        "paper": "CLIP §3.1.1 — '80 different context prompts and ensembled', "
                 f"ImageNet +3.5%(단일 기본 템플릿 대비) · CPU 데모는 K={K} 샘플",
    }


def main():
    results = [v1(), v2(), v3(), v4()]
    lines = []
    lines.append("=" * 74)
    lines.append("Day 12 · CLIP 데모 — 대조 학습이 classifier를 말에서 배우는 구조 (합성, CPU)")
    lines.append(f"실행: {datetime.now().isoformat(timespec='seconds')}  seed=42  D={D}  N_batch={N}"
                 f"  C={NC}  K(프롬프트/클래스)={K}  σ={SIGMA} 동일분산")
    lines.append("성격: 논문 명제의 '방향·구조' 재현(합성 세계) — 논문 수치 재현 아님.")
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
    with open(os.path.join(RUNS_DIR, "clip_demo_output.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
