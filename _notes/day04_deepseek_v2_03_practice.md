---
layout: default
title: "Day 04 DeepSeek-V2 — 03. 실습 — MLA 축소 데모로 KV cache 압축·흡수·decoupled RoPE 검증"
paper_id: deepseek-v2
date: 2026-10-18
---

# 03. 실습 — MLA 축소 데모로 KV cache 압축·흡수·decoupled RoPE 검증

> 스크립트: `day04_deepseek_v2/mla_demo.py`
> 실행: `cd ~/paper-diary && /opt/homebrew/bin/python3.14 day04_deepseek_v2/mla_demo.py --device mps`
> 환경: Python 3.14 · torch 2.14.0 · Apple MPS

---

## 0. 무엇을 검증하나

| # | 검증 | 기대 |
|---|---|---|
| (1) | KV cache: MHA vs MLA (실제 V2 설정) | 대폭 감소 |
| (2) | 흡수 항등식 `qᵀk = (c^Q)ᵀ A_i c^KV` (RoPE×) | **일치(≈0)** |
| (3) | `k^C`에 RoPE 적용 시 흡수 | **깨짐(≠0)** |
| (3b) | decoupled RoPE: C흡수 + R명시 = 직접 MLA | **일치(≈0)** |
| (4) | 출력 shape | 헤드 concat 복원 가능 |

---

## 1. 실제 실행 결과 (Apple MPS, 그대로 붙여넣음)

```
[MLA demo] device=mps T=64 n_h=4 d_h=8 d_c=16 d_c'=24 d_hR=4

(1) KV cache / token (V2: n_h=128, d_h=128, d_c=512, d_hR=64, L=60):
    MHA = 2·n_h·d_h·L = 1,966,080   MLA = (d_c+d_hR)·L = 34,560   →  감소 98.24%
    128K 문맥 fp16: MHA 480.00 GB  vs  MLA 8.44 GB
    (논문 headline: DeepSeek 67B 대비 KV cache −93.3%)

(2) 흡수 항등식 (RoPE 없음):
    max|직접 − 흡수| = 7.276e-12   (상대 2.02e-07)  → 항등식 성립 ✔

(3) k^C 에 RoPE 적용 시 (흡수 불가):
    max|RoPE직접 − 흡수| = 4.590e-05  (≫ 0)  → 위치 의존이라 흡수 불가 ✗
    ⇒ decoupled RoPE: RoPE 를 별도 q^R/k^R 채널로 이동하면 C-부분 흡수 유지
    decoupled 분해: max|(C흡수 + R명시) − 직접| = 2.910e-11  → 일치 ✔

(4) 출력 shape: MLA out=(64, 32)  (d_c=16, n_h·d_h=32 로 재구성 가능)

[결론] MLA = MHA + low-rank KV joint compression. KV cache (d_c+d_hR) 만 저장 + W^UK/W^UV 흡수 → K,V materialize 불필요. decoupled RoPE 로 위치정보를 별도 소채널에 실어 흡수 가능성 유지.
```

---

## 2. 결과 읽기

### (1) KV cache — 병목의 실제 크기
- V2 설정에서 토큰당 KV: MHA `1,966,080` → MLA `34,560` = **−98.24%**.
  - `MLA/MHA = 576/32768 = 1.76%` (헤드 수가 같다는 가정의 순수 저랭크 이득).
  - 논문 headline **−93.3%**는 더 큰 모델(DeepSeek 67B, GQA)과의 비교라 수치가 다름 — **비교 대상이 다르면 %도 다르다**는 점 주의.
- `128K` 문맥 fp16에서 MHA 480 GB → MLA 8.44 GB. **"128K를 실사용 가능하게 만든 것"**이 MLA의 실질 기여.

### (2) 흡수 항등식 — `1e-12` 급 일치
- `A_i = W^UQ_iᵀ W^UK_i` 를 미리 접어도 점수가 **동일**(부동소수 오차 수준).
- 함의: 추론 시 **K,V를 아예 안 만든다** → 메모리·연산 동시 절감.

### (3) RoPE를 `k^C`에 걸면 `4.6e-05`로 어긋남
- 위치행렬이 `W^UK`와 얽혀 **흡수 불가**. (`(2)`의 `1e-12` ↔ `(3)`의 `4.6e-05`: 7자리 차이)
- 이게 **decoupled RoPE의 존재 이유**.

### (3b) decoupled — `2.9e-11`로 다시 일치
- `qᵀk = q^Cᵀk^C(흡수) + q^Rᵀk^R(명시)` 가 직접 MLA와 일치.
- 위치정보는 **`d_h^R=64` 소채널**에만 실려 cache에 추가 → 그래서 `(d_c + d_h^R)`.

---

## 3. 손으로 따라하기 (축소 설정)

```python
# d=64, n_h=4, d_h=8, d_c=16, d_cp=24, d_hR=4
c_kv = h @ W_DKV.T          # (T, 16)  ← cache 에 저장되는 것
c_q  = h @ W_DQ.T           # (T, 24)

# 직접: q^C = W^UQ c^Q, k^C = W^UK c^KV
q_c  = (c_q @ W_UQ.T).view(T, 4, 8)
k_c  = (c_kv @ W_UK.T).view(T, 4, 8)
s_dir  = einsum("thd,shd->hts", q_c, k_c)

# 흡수: A_i = W^UQ_i.T @ W^UK_i  (헤드별)
A_i = W_UQ[i*8:(i+1)*8].T @ W_UK[i*8:(i+1)*8]     # (24, 16)
s_abs_i = (c_q @ A_i) @ c_kv.T                     # (T, T)
# s_dir[i] ≈ s_abs_i   (오차 1e-11)
```

**포인트:** cache에는 `c_kv`(16차원)만 있고, `k,v`(4×8=32차원 ×2)는 **계산 시점에 흡수로 대체**된다.

---

## 4. 실패/삽질 기록

- **broadcast 버그**: `rope()`가 `(T,heads,d)` 텐서에서 `cos/sin`을 `(T,half)`로 만들어 `n_h` 축과 충돌 →
  `cos.view((T,) + (1,)*(ndim-2) + (half,))` 로 중간 축 singleton 삽입해 해결.
- **흡수는 "헤드별"로 접어야 함**: `W^UQᵀ W^UK` 전체를 한 번에 접으면 헤드 합만 나옴 → 헤드 슬라이스별로 `A_i` 를 만들 것.

---

## 5. 파일 위치

- `day04_deepseek_v2/mla_demo.py`
- 실행 로그: `day04_deepseek_v2/runs/mla_demo_stdout.txt` (2026-10-19 재실행 · MPS · `torch.manual_seed(0)` → 본 문서 §1과 **문자 단위 일치**)
  - 재현: `cd ~/paper-diary && /opt/homebrew/bin/python3.14 day04_deepseek_v2/mla_demo.py --device mps > day04_deepseek_v2/runs/mla_demo_stdout.txt`
  - 본 데모는 stdout만 출력하므로 `runs/` 산출물은 로그 1종(JSON/PNG 없음).
