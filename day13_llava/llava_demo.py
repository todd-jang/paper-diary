#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 13 · LLaVA 데모 — "시각 토크나이저가 LLM의 벤치에 앉는다"의 합성(CPU) 방향 재현
=================================================================================
설계 원칙 (D10/D12 데모 계승):
  · 임계값 넘기용 파라미터 탐색 금지 — 그것은 검증이 아니라 조작이다.
  · 판정은 (i) 항등식/정리(구조), (ii) 구조적 단조/방향, (iii) 통계 보장 방향.
  · 노이즈는 세계 생성 시 고정된 규모(σ)로 둔다.
  · 논문 수치 재현이 아니라, 설계 판단의 '방향·구조'를 자체 실측한다.

검증 4종 (02_수식유도.md §6 대응표):
  V1  식(1) 선형 projection의 정합 — Stage 1: 비전·LLM 동결, θ={W}만 학습.
      확인:
      (a) 완전 정합 항등식 — CLIP 피처 공간이 언어 공간의 가역 선형상(σ=0)이면
          최소제곱 W는 잃는 것이 없다(σ=0 복원 resid ≈ 0),
      (b) 최소제곱 잔차 직교성(정리) — 정규방정식 잔차가 설계 공간에 직교
          Zᵀ·r ≈ 0 — Stage 1이 '정합'이라는 이름의 수학,
      (c) 모듈 동결(θ={W}) — CLIP 피처·어휘 임베딩 배열이 fitting 전후 비트 동일,
      (d) 방향(측정) — heldout 개념 정렬 코사인 > 랜덤 probe.
      [원문: §4.1 Eq.(1) H_v = W·Z_v · §4.2 Stage 1 "training a compatible
       visual tokenizer for the frozen LLM" · 8×A100 ≤4h]
  V2  Table 4 데이터 혼입 — 3종 지시 데이터(conversation 58K / detailed 23K /
      complex reasoning 77K)의 능력 방향. 확인:
      (a) 적산 항등식 — 능력 스코어를 루프 적산 경로와 폐형 행렬 경로로 각각
          계산 → 정확 일치,
      (b) All 순서 방향 — Full > Detail+Complex > Conv+5%D+10%C > Conv-only
          (Table 4 All 85.1 > 81.9 > 80.5 > 73.8의 동일 순서),
      (c) 공동 확장 — 대화 전용보다 5% detail + 10% complex 혼입이 Conv 축을
          끌어올림(80.5 > 73.8 동일 방향),
      (d) No-tuning 최저 — 모든 혼합이 무튜닝(21.5) 위.
      감쇠 곡선(포화)은 "같은 종류의 데이터를 더 넣어도 제자리"를 합성하는 장치.
  V3  식(2)~(3) loss 마스크 — assistant 답변 토큰(녹색)만 학습 신호. 확인:
      (a) 마스크 항등식 — (i) 전 위치 CE를 계산 뒤 마스크 가중합 경로,
          (ii) 답변 위치 rows만 곧바로 평균 경로 → 정확 일치,
      (b) 기울기 관문(구조) — 마스크 손실의 dW를 위치별로 분해하면 instruction
          위치 행이 정확히 0이고, 전체 dW == 답변 위치 rows만의 dW와 동일
          (마스크가 학습 신호의 목표물을 정한다),
      (c) grounding 방향 — X_v를 모든 조건부에 걸어두면(식 3) 답변 CE가 시각
          토큰을 예시 간 동일한 중립 토큰으로 바꾼 대조군보다 낮다(측정 방향),
      (d) 답변 CE 비증가 — 학습 스텝 세 구간 추세 하락.
      참고(관측): 식(2)의 첫 턴 [X_q,X_v]/[X_v,X_q] 무작위 배치 — mean-pooled
      조건부 하에서 CE 불변(합성 언어 모델의 특성, 실측 표기).
  V4  Table 7 judge 앙상블 — LLaVA 90.92 + GPT-4(judge) 92.53. 확인:
      (a) 분해 항등식 — 앙상블 정확도를 겹친 기반항 경로와 문항 분기 경로로
          각각 계산 → 정확 일치,
      (b) 심판 호출률(구조) — P(disagree) = 1 − pA·pB (가지 합의 열합 축약 항등식,
          Monte-Carlo 관측도 일치),
      (c) 방향 — judge 앙상블 > 단독 최고(90.92 > 82.69 계열 구조),
          Monte-Carlo 관측이 해석값과 통계 근사 일치.
      세계: ScienceQA test 4241 규모 가정. 문항 클래스별 정답률(이미지 불필요 /
      입력 요구)로 '텍스트 전용 심판이 어디서 고치는가'를 합성.
      파라미터(pIF/pIR/p_j)는 [합성 규모] 라벨 — 판정 탐색용 튜닝 아님.
사용법:  .venv/bin/python day13_llava/llava_demo.py
결과:    stdout + day13_llava/runs/llava_demo_output.txt
"""
import os
from datetime import datetime

import numpy as np

RNG = np.random.default_rng(42)
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")

# ────────────────────────────────────────────────────────────────
# V1 세계 — 개념 E(언어 공간 어휘 방향) → CLIP 피처 공간 가역 선형상 A
# ────────────────────────────────────────────────────────────────
C_CONCEPT, D_VEC, D_CLIP = 128, 64, 64       # 개념 128 · 학습 112 > 차원 64 — 과소결정(암기) 방지 위계
SIGMA_CLIP = 0.05                    # 세계 고정 노이즈(측정 규모, 튜닝 아님)

_BASIS = RNG.normal(size=(C_CONCEPT, D_VEC))
E_CONCEPT = _BASIS / np.linalg.norm(_BASIS, axis=1, keepdims=True)  # (C, D) 단위 개념 방향
LINEAR_MAP = RNG.normal(size=(D_VEC, D_CLIP)) / np.sqrt(D_VEC)      # 풀랭크 가역(정사각)


def clip_features(sigma):
    """Z = g(X_v) — CLIP 그리드 피처 (C, D_CLIP): E·A + σ·N."""
    return E_CONCEPT @ LINEAR_MAP + sigma * RNG.normal(size=(C_CONCEPT, D_CLIP))


def _fit_projection(Z_tr, Y_tr):
    W, *_ = np.linalg.lstsq(Z_tr, Y_tr, rcond=None)   # 최소제곱 — 단일 선형 W
    return W


def v1():
    """Stage 1: θ={W}. (a) σ=0 완전 정합 항등식 (b) 잔차 직교성 (c) 모듈 동결."""
    Z_noisy = clip_features(SIGMA_CLIP)              # 여기서만 RNG 소비(비트 고정)
    N_TR = 112                                       # train 112 / heldout 16 (과소결정 방지)
    Z_tr, Z_te = Z_noisy[:N_TR], Z_noisy[N_TR:]
    E_tr, E_te = E_CONCEPT[:N_TR], E_CONCEPT[N_TR:]

    snapshot = {"Z_tr": Z_tr.copy(), "E_tr": E_tr.copy()}   # 동결 감사 스냅샷

    W = _fit_projection(Z_tr, E_tr)                  # 유일 trainable
    r_tr = Z_tr @ W - E_tr                           # 최소제곱 잔차

    # (a) 무노이즈 세계 완전 정합: 가역 선형상 → pinv 역산이 E를 그대로 돌려준다
    Z0 = E_CONCEPT @ LINEAR_MAP                      # σ=0 세계 (독립 재생성)
    W0 = _fit_projection(Z0, E_CONCEPT)
    recon_exact = float(np.abs(Z0 @ W0 - E_CONCEPT).max())

    # (b) 잔차 직교성(정규방정식): 설계 공간과의 직교 — 열 z_i에 대해 |z_iᵀr| ≤ |r|·ε
    col_ortho = (np.abs(Z_tr.T @ r_tr).max(axis=-1)
                 / (np.linalg.norm(Z_tr, axis=0) * np.linalg.norm(r_tr) + 1e-30))
    ortho = float(col_ortho.max())

    # (c) 모듈 동결: fitting 전후 비트 동일
    immutable = bool(np.array_equal(snapshot["Z_tr"], Z_tr)
                     and np.array_equal(snapshot["E_tr"], E_tr))

    # (d) 방향(측정): heldout 개념의 정렬 코사인 vs 랜덤 probe
    H_te = Z_te @ W
    cos = [float(E_te[i] @ H_te[i] / (np.linalg.norm(E_te[i]) * np.linalg.norm(H_te[i])))
           for i in range(len(E_te))]
    H_probe = Z_te @ RNG.normal(size=(D_CLIP, D_VEC))
    cos_probe = [float(E_te[i] @ H_probe[i] / (np.linalg.norm(E_te[i]) * np.linalg.norm(H_probe[i])))
                 for i in range(len(E_te))]
    return {
        "check": "V1 식(1) Stage 1 정합 — σ=0 완전 정합 항등식 + 잔차 직교성 + 모듈 동결(θ={W})",
        "fit_operators": {"encoder": "frozen (CLIP)", "projection W": "trainable",
                          "llm_vocab": "frozen"},
        "noiseless_full_reconstruction_max_abs": round(recon_exact, 12),
        "residual_orthogonality_scaled_max": round(ortho, 12),
        "modules_bit_identical_after_fit": immutable,
        "heldout_mean_cos(fit/probe)": [round(float(np.mean(cos)), 4),
                                        round(float(np.mean(cos_probe)), 4)],
        "PASS": bool(recon_exact < 1e-8 and ortho < 1e-9 and immutable),
        "paper": "2304.08485 §4.1 Eq.(1) H_v = W·Z_v · §4.2 Stage 1 'training a compatible "
                 "visual tokenizer for the frozen LLM' (CC-595K · θ={W}만 · lr 2e-3 · ≤4h)",
    }


# ────────────────────────────────────────────────────────────────
# V2 세계 — 3종 지시 데이터의 능력 적산: 자등 감쇠(포화) + 비대칭 전이
# 사전 고정 파라미터(세계 규모, 판정 탐색 아님): 포화 τ=30, 전이 κ=0.08
# ────────────────────────────────────────────────────────────────
AXES = ("conv", "detail", "complex")
TRAN = np.array([  # TRAN[from_type, to_axis] — 비대칭 전이(κ 동일 스케일)
    [1.00, 0.08, 0.08],   # conversation 예시 1단위
    [0.08, 1.00, 0.08],   # detailed description 예시 1단위
    [0.08, 0.08, 1.00],   # complex reasoning 예시 1단위
])
TAU_SAT = 30.0     # (단위: K개) 포화 컨스탄트 — 같은 종류 데이터의 추가 기여는 감쇠
BASE = 25.0        # 합성 기저(No tuning 상태의 수준)

ROWS = {           # Table 4 행 구성 — 데이터 개수(단위 K), 팩트시트 (a)·(e)
    "Full (58/23/77)": (58.0, 23.0, 77.0),
    "Detail+Complex":  (0.0, 23.0, 77.0),
    "Conv+5%D+10%C":   (58.0, 1.15, 7.70),
    "Conv only":       (58.0, 0.0, 0.0),
    "No tuning":       (0.0, 0.0, 0.0),
}


def _axis_scores_closed(counts):
    """폐형: 능력_s = base + Σ_t sat(count_t)·TRAN[t,s], sat(c) = 1 − exp(−c/τ)."""
    sat = 1.0 - np.exp(-np.asarray(counts, dtype=np.float64) / TAU_SAT)   # (3,)
    return BASE + TRAN.T @ sat                                            # (3,)


def _axis_scores_loop(counts):
    """루프 적산: 데이터 유형별로 한 축씩 기여를 하나씩 더한다(다른 코드 경로)."""
    sat = 1.0 - np.exp(-np.asarray(counts, dtype=np.float64) / TAU_SAT)
    scores = np.full(3, BASE)
    for t in range(3):
        for s in range(3):
            scores[s] += sat[t] * TRAN[t, s]
    return scores


def v2():
    """Table 4 — 3종 혼입이 능력 방향을 만든다."""
    result_rows = {name: _axis_scores_loop(counts) for name, counts in ROWS.items()}

    # (a) 적산 항등식: 두 경로 최대 편차
    max_dev = max(float(np.abs(_axis_scores_loop(c) - _axis_scores_closed(c)).max())
                  for c in ROWS.values())

    full = result_rows["Full (58/23/77)"]
    dc = result_rows["Detail+Complex"]
    mixed = result_rows["Conv+5%D+10%C"]
    conv_only = result_rows["Conv only"]
    no_tune = result_rows["No tuning"]

    all_scores = {n: float(s.mean()) for n, s in result_rows.items()}
    all_order = sorted(all_scores, key=all_scores.get, reverse=True)
    order_ok = all_order == ["Full (58/23/77)", "Detail+Complex", "Conv+5%D+10%C",
                             "Conv only", "No tuning"]
    co_boost = bool(mixed[0] > conv_only[0] + 1e-12)          # (c) Conv 축 공동 확장
    full_best_axis = bool((full >= dc).all() and (full >= mixed).all()
                          and (full >= conv_only).all())       # (b) 보조: 전 축 Full 우위
    above_no_tune = bool(all(s.mean() > no_tune.mean() + 1e-12
                             for n, s in result_rows.items() if n != "No tuning"))
    return {
        "check": "V2 Table 4 데이터 혼입 — 적산 항등식(2경로) + All 순서 방향 + 공동 확장 + No-tuning 최저",
        "scores(conv/detail/complex)": {n: [round(x, 4) for x in s]
                                        for n, s in result_rows.items()},
        "all_col_relative_mean": {n: round(v, 4) for n, v in all_scores.items()},
        "all_col_match_paper_order": order_ok,
        "full_best_on_every_axis": full_best_axis,
        "small_mix_co_boosts_conv": co_boost,
        "all_above_no_tuning": above_no_tune,
        "loop_vs_closed_max_dev": round(max_dev, 12),
        "PASS": bool(max_dev < 1e-12 and order_ok and co_boost
                     and full_best_axis and above_no_tune),
        "paper": "Table 4 — All: Full 85.1 > Detail+Complex 81.9 > Conv+5%D+10%C 80.5 > "
                 "Conv-only 73.8 >> No-tuning 21.5 · §3 3종 = 58K/23K/77K (합계 158K)",
    }


# ────────────────────────────────────────────────────────────────
# V3 세계 — 미니 언어 모델: mean-pooled prefix + 선형 head(φ 학습 흐름의 합성).
# 어휘: 0..15 instruction 토큰 · 16..31 answer 토큰 · 32..47 visual(개념) 토큰
# 시퀀스: [VIS·c, instr×8, answer×6] — 총 길이 15
# ────────────────────────────────────────────────────────────────
D3, VOCAB = 48, 48
N_INSTR, N_ANS = 8, 6
L_SEQ = 1 + N_INSTR + N_ANS
POS_INSTR = np.arange(1, 1 + N_INSTR)
POS_ANS = np.arange(1 + N_INSTR, L_SEQ)
MASK = np.zeros(L_SEQ, dtype=np.float64)
MASK[POS_ANS] = 1.0
N_EX = 96
C3 = 16
EMB = RNG.normal(size=(VOCAB, D3))                    # 고정 워드 임베딩(데모에선 동결)
INSTR_TEMPLATE = RNG.integers(0, 16, size=N_INSTR)    # 동일 지시 템플릿이 반복 — 개념 신호는 시각 토큰에만
CONCEPT_OF = np.arange(N_EX) % C3
ANS_TOKEN = (CONCEPT_OF[:, None] * 5 + np.arange(N_ANS)[None, :] * 7) % 16 + 16


def _sequences(with_visual=True):
    seq = np.empty((N_EX, L_SEQ), dtype=np.int64)
    if with_visual:
        seq[:, 0] = 32 + CONCEPT_OF
    else:                    # 대조군: 개념 신호를 예시·개념 간 동일 '중립 토큰'으로 대체
        seq[:, 0] = 15
    seq[:, 1:1 + N_INSTR] = INSTR_TEMPLATE
    seq[:, 1 + N_INSTR:] = ANS_TOKEN
    return seq


def _prefix_pool(seq):
    """h_t = mean(임베딩[0..t-1]) — 식(3)의 조건부 인코딩(합성: mean-pool)."""
    Eseq = EMB[seq]                                                 # (n, L, D)
    cum = np.cumsum(Eseq, axis=1)
    counts = np.arange(1, L_SEQ + 1, dtype=np.float64)
    return cum / counts[None, :, None]                              # (n, L, D)


def _softmax_P(W, seq):
    H = _prefix_pool(seq)
    Plog = np.einsum("nld,dv->nlv", H, W)
    Z = Plog - Plog.max(axis=2, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=2, keepdims=True)


def _targets(seq):
    Y = np.zeros((N_EX, L_SEQ, VOCAB))
    rows = np.arange(N_EX)[:, None]
    Y[rows, POS_ANS[None, :], seq[:, POS_ANS]] = 1.0
    return Y


def _ce_per(W, seq):
    """모든 예측 위치(1..L-1)의 CE (n, L) — 답변 위치만 쓸 값."""
    P = _softmax_P(W, seq)
    idx = (np.arange(N_EX)[:, None], np.arange(L_SEQ)[None, :], seq)
    return -np.log(np.clip(P[idx], 1e-12, None))


def _loss_two_paths(W, seq):
    """마스크 항등식용: (i) 전 위치 계산 후 마스크 가중합 == (ii) 답변 rows 평균."""
    ce_per = _ce_per(W, seq)
    path_weighted = float(np.sum(ce_per * MASK) / (N_EX * MASK.sum()))   # (i)
    path_rows = float(ce_per[:, POS_ANS].mean())                        # (ii)
    return path_weighted, path_rows


def _grad_decomposed(W, seq):
    """dL/dW를 위치별로 분해: rows_t = Σ_n coef_{n,t}·h_{n,t}/N, coef = mask·(P−Y)."""
    H = _prefix_pool(seq)
    P = _softmax_P(W, seq)
    Y = _targets(seq)
    coef = MASK[None, :, None] * (P - Y)                                # (n, L, V)
    return np.einsum("nlv,nld->lvd", coef, H) / (N_EX * MASK.sum())     # (L, V, D)


def _grad_ans_only(W, seq):
    """답변 위치 rows만으로 조립한 dW(다른 코드 경로). 반환 (V, D)."""
    H = _prefix_pool(seq)
    P = _softmax_P(W, seq)
    Y = _targets(seq)
    coef = P[:, POS_ANS] - Y[:, POS_ANS]                                # (n, A, V)
    return np.einsum("nav,nad->vd", coef, H[:, POS_ANS]) / (N_EX * MASK.sum())


def _train_masked(seq, steps=600, lr=0.35):
    W = RNG.normal(size=(D3, VOCAB)) * 0.05
    hist = []
    for _ in range(steps):
        dW = _grad_decomposed(W, seq).sum(axis=0)                       # (V, D)
        W -= lr * dW.T                                                  # W: (D, V)
        hist.append(_loss_two_paths(W, seq)[1])
    return W, np.array(hist)


def _turn_order_invariance(seq):
    """식(2) 참고 관측 — mean-pooled 조건부에서 [X_q,X_v]↔[X_v,X_q] CE 불변."""
    s2 = seq.copy()
    s2[:, [0, 1]] = s2[:, [1, 0]]                     # 첫 턴 두 자리 교환
    W_fixed = RNG.normal(size=(D3, VOCAB))
    la = _loss_two_paths(W_fixed, seq)[1]
    lb = _loss_two_paths(W_fixed, s2)[1]
    return bool(abs(la - lb) < 1e-12)


def v3():
    """식(2)~(3) — loss 마스크의 토큰 예산과 X_v 조건부(grounding)."""
    seq = _sequences(with_visual=True)
    W0 = RNG.normal(size=(D3, VOCAB)) * 0.05          # 초기 세계(동일 RNG 경로)

    # (a) 마스크 항등식
    p1, p2 = _loss_two_paths(W0, seq)
    identity = bool(abs(p1 - p2) < 1e-10)

    # (b) 기울기 관문: instruction 위치 행 == 0 정확히, 전체 dW == 답변 rows만 dW
    dW_rows = _grad_decomposed(W0, seq)                                # (L, V, D)
    instr_rows_zero = bool(float(np.abs(dW_rows[POS_INSTR]).max()) == 0.0)
    grad_full = dW_rows.sum(axis=0)                                    # (V, D)
    grad_ans = _grad_ans_only(W0, seq)                                 # (V, D)
    decomposition = bool(float(np.abs(grad_full - grad_ans).max()) < 1e-10)

    # (c) grounding 방향 + (d) CE 추세 (학습 = masked GD만)
    seq_neut = _sequences(with_visual=False)
    W_vis, hist = _train_masked(seq)
    W_neut, hist_neut = _train_masked(seq_neut)
    w1 = float(hist[:200].mean())
    w2 = float(hist[300:500].mean())
    w3 = float(hist[-40:].mean())
    turn_invariance = _turn_order_invariance(seq)
    # 위치 9 = 답변 첫 토큰 — 시각 토큰-only 관문: 중립 대체 세계에서는 예측 근거가 없다
    ce9_vis = float(_ce_per(W_vis, seq)[:, POS_ANS[0]].mean())
    ce9_neut = float(_ce_per(W_neut, seq_neut)[:, POS_ANS[0]].mean())
    uniform_ce = float(np.log(16.0))
    return {
        "check": "V3 식(2)~(3) loss 마스크 — 마스크 항등식 + 기울기 관문 + grounding 방향 + CE 하락 추세",
        "mask_identity_exact": identity,
        "grad_instr_rows_exactly_zero": instr_rows_zero,
        "grad_equals_answer_rows_only": decomposition,
        "token_budget": {"loss_tokens_per_sample": int(MASK.sum()),
                         "sequence_length": int(L_SEQ),
                         "answer_tokens": int(N_ANS)},
        "turn_order_invariance_exact(observed)": turn_invariance,
        "answer_ce_windows(first/mid/last)": [round(w1, 4), round(w2, 4), round(w3, 4)],
        "answer_first_token_ce(visual/neutral/uniform)": [round(ce9_vis, 4),
                                                          round(ce9_neut, 4),
                                                          round(uniform_ce, 4)],
        "final_ce_with_visual": round(float(hist[-1]), 4),
        "final_ce_neutral_visual": round(float(hist_neut[-1]), 4),
        "grounding_direction_holds": bool(ce9_vis < ce9_neut and hist[-1] < hist_neut[-1]),
        "PASS": bool(identity and instr_rows_zero and decomposition
                     and ce9_vis < ce9_neut and w3 < w2 < w1),
        "paper": "Eq.(2) 첫 턴 [X_q,X_v]/[X_v,X_q] 무작위 · Eq.(3) pθ(Xa|Xv,Xinstruct) — "
                 "'only green sequence/tokens are used to compute the loss' (Table 2) · "
                 "X_v가 모든 조건부에 명시(grounding)",
    }


# ────────────────────────────────────────────────────────────────
# V4 세계 — ScienceQA judge 앙상블. 문항 클래스: 이미지 불필요(0.4) / 입력 요구(0.6)
# 사전 고정(합성 규모, 판정 탐색 아님):
#   A(LLaVA 흐름): pIF=0.92 · pIR=0.90  → 단독 ≈ 0.908 (원문 90.92 방향)
#   B(GPT-4 text-only 흐름): pIF=0.95 · pIR=0.745 → 단독 ≈ 0.827 (원문 82.69 방향)
#   judge strength p_j = 0.75 (관측 파라미터 — 심판 규칙의 합성)
# ────────────────────────────────────────────────────────────────
F_FREE, F_VISUAL = 0.40, 0.60
PA_FREE, PA_VIS = 0.92, 0.90
PB_FREE, PB_VIS = 0.95, 0.745
PJUDGE = 0.75


def _ensemble_formula():
    """격자 기반항 경로: ens = Σ_classes f·[pA·pB + p_j·(qA·pB + pA·qB)]."""
    out = 0.0
    per_class = []
    for f, pA, pB in ((F_FREE, PA_FREE, PB_FREE), (F_VISUAL, PA_VIS, PB_VIS)):
        term = pA * pB + PJUDGE * (1 - pA) * pB + PJUDGE * pA * (1 - pB)
        per_class.append(term)
        out += f * term
    return out, tuple(per_class)


def _ensemble_branch():
    """분기 경로: 문항당 논리 분기로 같은 값을 조립한다(다른 코드 경로)."""
    out = 0.0
    for f, pA, pB in ((F_FREE, PA_FREE, PB_FREE), (F_VISUAL, PA_VIS, PB_VIS)):
        qA, qB = 1 - pA, 1 - pB
        p_both_right = pA * pB          # 둘 다 정답
        p_a_wrong = qA * pB             # A만 틀림 → 심판이 고침(p_j)
        p_b_wrong = pA * qB             # B만 틀림 → 심판이 고침(p_j)
        p_both_wrong = qA * qB          # 둘 다 틀림 → 심판도 고칠 수 없음(공헌 0)
        per_item = (p_both_right * 1.0
                    + p_a_wrong * PJUDGE
                    + p_b_wrong * PJUDGE
                    + p_both_wrong * 0.0)
        out += f * per_item
    return out


def _mc_ensemble(n=400_000):
    """Monte-Carlo 관측 — 동일 규칙의 재현성 확인용."""
    free = RNG.random(n) < F_FREE
    pA_of = np.where(free, PA_FREE, PA_VIS)
    pB_of = np.where(free, PB_FREE, PB_VIS)
    a_ok = RNG.random(n) < pA_of
    b_ok = RNG.random(n) < pB_of
    both = a_ok & b_ok
    exactly_one = a_ok ^ b_ok
    judge_lucky = RNG.random(n) < PJUDGE
    val = (both.sum() + (judge_lucky & exactly_one).sum()) / n
    disagree_all = (exactly_one | (~a_ok & ~b_ok)).mean()               # 심판 호출률
    return float(val), float(disagree_all)


def v4():
    ens, per_class = _ensemble_formula()
    ens_branch = _ensemble_branch()
    single_a = F_FREE * PA_FREE + F_VISUAL * PA_VIS
    single_b = F_FREE * PB_FREE + F_VISUAL * PB_VIS
    mc_val, mc_disagree = _mc_ensemble()

    # (b) 심판 호출률 열합 축약: qA·pB + pA·qB + qA·qB == 1 − pA·pB (각 클래스)
    collapsed = all(abs((1 - pA) * pB + pA * (1 - pB) + (1 - pA) * (1 - pB)
                        - (1 - pA * pB)) < 1e-12
                    for pA, pB in ((PA_FREE, PB_FREE), (PA_VIS, PB_VIS)))
    disagree_closed = F_FREE * (1 - PA_FREE * PB_FREE) + F_VISUAL * (1 - PA_VIS * PB_VIS)
    direction = bool(ens > max(single_a, single_b) + 1e-9)
    return {
        "check": "V4 Table 7 judge 앙상블 — 분해 항등식(2경로) + 호출률 열합 축약 + 단독 최고 넘기는 방향",
        "param_note": "pIF/pIR/p_j 는 [합성 규모] — 방향 구성용, 판정 탐색 아님",
        "ensemble_accuracy_formula": round(ens, 4),
        "ensemble_accuracy_branch_path": round(ens_branch, 12),
        "decomposition_identity_exact": bool(abs(ens - ens_branch) < 1e-12),
        "single_A(llava-like)": round(single_a, 4),
        "single_B(gpt4-text-like)": round(single_b, 4),
        "ensemble_beats_best_single": direction,
        "ens_minus_best_single": round(ens - max(single_a, single_b), 4),
        "judge_call_rate(closed_1-pApB)": round(disagree_closed, 4),
        "judge_call_rate(mc)": round(mc_disagree, 4),
        "call_rate_collapsed_identity": collapsed,
        "mc_vs_formula": {"mc": round(mc_val, 4), "formula": round(ens, 4),
                          "abs_dev": round(abs(mc_val - ens), 4),
                          "under_5e-3": bool(abs(mc_val - ens) < 5e-3)},
        "PASS": bool(abs(ens - ens_branch) < 1e-12 and abs(mc_val - ens) < 5e-3
                     and direction and collapsed),
        "paper": "Table 7 — LLaVA 90.92 · LLaVA+GPT-4(judge) 92.53 new SoTA · "
                 "GPT-4† 82.69[저자 eval] · §5.2 'first time that GPT-4 is used for "
                 "model ensembling' · 텍스트 전용 심판은 '이미지 없이 답되는 문제'에서 개선을 만든다",
    }


def main():
    results = [v1(), v2(), v3(), v4()]
    lines = []
    lines.append("=" * 74)
    lines.append("Day 13 · LLaVA 데모 — 시각 토크나이저가 언어 LLM 벤치에 앉는 구조 (합성, CPU)")
    lines.append(f"실행: {datetime.now().isoformat(timespec='seconds')}  seed=42  "
                 f"C={C_CONCEPT}  D={D_VEC}  D_CLIP={D_CLIP}  |  N_ex={N_EX}  L_seq={L_SEQ}"
                 f"  |  V4 세계: ScienceQA test 4241 규모 가정")
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
    with open(os.path.join(RUNS_DIR, "llava_demo_output.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
