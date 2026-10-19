#!/usr/bin/env python3
"""DeepSeek-R1 축소 데모 — GRPO 그룹 상대 이점 / 규칙 기반 보상(식 4) / 언어 일관성 보상(식 7) / 보상 해킹(B.5).

검증하는 4가지:
  (1) GRPO 그룹 상대 이점 (식 1–3).  가치망(value net) 없이 **그룹 내 정규화**만으로
      이점을 만든다. 확인할 성질 3개:
        · Σ_i A_i = 0  (그룹 평균이 내장 베이스라인이다 — 항등적으로 성립)
        · 보상 스케일 불변 (r → c·r 로 바꿔도 갱신이 동일; 정규화 없으면 갱신이 c배)
        · 난이도 게이팅: 전원 정답/전원 오답 그룹은 std=0 → A=0 → 그래디언트가 정확히 0
      추가로 그룹 베이스라인의 분산 감소 효과를 무베이스라인 REINFORCE 와 비교한다.
  (2) 규칙 기반 보상 파서 (식 4).  Reward_rule = Reward_acc + Reward_format 을 실제로 계산한다.
      형식 보상은 <think>/<answer> 태그 구조만 보고, 정확도 보상은 \\boxed{} 정답만 본다.
      두 항이 **서로 다른 실패 모드**를 잡는다는 점, box 형식이 정확도 검증의 전제조건이라는 점,
      형식 보상 **단독으로도 획득 가능**하다는 점(= 해킹 표면)을 케이스 분해로 보인다.
  (3) 언어 일관성(LC) 보상의 trade-off (식 7 · 부록 B.6).  모델링 가정: 논문이 미시 기구를 주지
      않으므로, 관측된 "정렬 시 경미한 성능 저하"의 함의를 **외국어 추론 토큰이 과제에 더 유효하다**로
      명시한다. 가정 아래에서 λ↑ → 혼용률↓·과제 성공률 소폭↓ 로 B.6 과 같은 모양이 나온다.
  (4) 보상 해킹 재현 (부록 B.5 · Figure 6).  보상 모델에 체계적 편향(길이 선호)이 있으면
      프록시 보상은 **상승**하는데 실제 지표는 **하락**한다. 규칙 기반 검증 게이트를 붙이면 사라진다.

논문: DeepSeek-R1 (arXiv:2501.12948).  시뮬레이션은 [SIM], 해석적으로 성립하는 성질은 [EXACT],
파서처럼 계산이 곧 정의인 부분도 [EXACT] 로 표시한다.  난수는 모두 CPU 생성기로 고정해
장치(CPU/MPS)와 무관하게 같은 수치가 나온다.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple

import torch

RUNS = Path(__file__).resolve().parent / "runs"


class Rollout(NamedTuple):
    idx: torch.Tensor
    reward: torch.Tensor
    group_std: torch.Tensor


# ============================================================ (1) GRPO 그룹 상대 이점
def grpo_group_advantage(N: int, K: int, G: int, steps: int, lr: float, device: str) -> dict:
    """문맥 밴딧으로 보는 GRPO.

    질문 q 마다 정답 행동 correct[q] 가 있고, 정책은 K개 행동 위의 softmax 다.
    초기 정책은 질문별 정답 확률 p_q 가 되도록 세팅한다(난이도 버킷 easy/mid/hard).
    각 스텝: 질문마다 G개 출력을 샘플 → 그룹 내 정규화 이점 → 정책 경사.
    """
    gen = torch.Generator().manual_seed(0)
    p_easy, p_mid, p_hard = 0.9, 0.5, 0.1
    n_hard = N // 3
    n_mid = N // 3
    n_easy = N - n_mid - n_hard
    p = torch.cat([torch.full((n_easy,), p_easy), torch.full((n_mid,), p_mid),
                   torch.full((n_hard,), p_hard)])
    correct = torch.randint(0, K, (N,), generator=gen)

    # 정답 확률이 p_q 가 되도록 초기 logit 을 맞춘다: P = e^z / (e^z + (K-1)) → z = log(p/(1-p)·(K-1))
    logits0 = torch.zeros(N, K)
    logits0[torch.arange(N), correct] = torch.log(p / (1.0 - p) * (K - 1))
    correct_dev = correct.to(device)

    def bucket_acc(lg: torch.Tensor) -> dict:
        prob_ok = torch.softmax(lg, dim=-1)[torch.arange(N), correct_dev]
        cut1, cut2 = n_easy, n_easy + n_mid
        return {"easy": float(prob_ok[:cut1].mean()),
                "mid": float(prob_ok[cut1:cut2].mean()),
                "hard": float(prob_ok[cut2:].mean())}

    def rollout(lg: torch.Tensor, g: torch.Generator) -> Rollout:
        prob = torch.softmax(lg.detach(), dim=-1).cpu()
        idx = torch.multinomial(prob, G, replacement=True, generator=g).to(lg.device)
        r = (idx == correct_dev.unsqueeze(1)).float()
        return Rollout(idx=idx, reward=r, group_std=r.std(dim=1, unbiased=False, keepdim=True))

    lg = logits0.to(device).clone().requires_grad_(True)
    opt = torch.optim.Adam([lg], lr=lr)
    train_gen = torch.Generator().manual_seed(0)
    acc_hist, zero_share = [], []
    for it in range(steps):
        roll = rollout(lg, train_gen)
        idx, r, std = roll.idx, roll.reward, roll.group_std
        zero_share.append(float((std.squeeze(1) < 1e-8).float().mean()))
        adv = (r - r.mean(dim=1, keepdim=True)) / std.clamp_min(1e-8)
        logp = torch.log_softmax(lg, dim=-1).gather(1, idx)           # N x G
        loss = -(adv.detach() * logp).mean(dim=1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, steps // 20) == 0 or it == steps - 1:
            acc_hist.append({"step": it, **bucket_acc(lg.detach())})

    # ---- [EXACT] Σ_i A_i = 0 : 그룹 평균을 빼는 순간 합이 항등적으로 0 (내장 베이스라인)
    roll = rollout(lg.detach(), torch.Generator().manual_seed(99))
    r, std = roll.reward, roll.group_std
    A = (r - r.mean(dim=1, keepdim=True)) / std.clamp_min(1e-8)
    sum_adv_absmax = float(A.sum(dim=1).abs().max())

    # ---- [EXACT] 보상 스케일 불변: r → c·r 로 바꿔도 갱신이 동일 / 정규화 없으면 c배
    def grads(scale: float, normalise: bool) -> torch.Tensor:
        lg_t = logits0.to(device).clone().requires_grad_(True)
        roll_t = rollout(lg_t, torch.Generator().manual_seed(1234))
        r = roll_t.reward * scale
        std = r.std(dim=1, unbiased=False, keepdim=True)
        adv = (r - r.mean(dim=1, keepdim=True)) / std.clamp_min(1e-8) if normalise else r
        logp = torch.log_softmax(lg_t, dim=-1).gather(1, roll_t.idx)
        (-(adv.detach() * logp).mean(dim=1).mean()).backward()
        return lg_t.grad.clone()

    g1, g100 = grads(1.0, True), grads(100.0, True)
    rg1, rg100 = grads(1.0, False), grads(100.0, False)

    # ---- [EXACT] 보상 평행이동 불변: r → r+c 여도 정규화 이점은 불변 (A=(r−mean)/std)
    def grads_shift(shift: float, normalise: bool) -> torch.Tensor:
        lg_t = logits0.to(device).clone().requires_grad_(True)
        roll_t = rollout(lg_t, torch.Generator().manual_seed(555))
        r = roll_t.reward + shift
        std = r.std(dim=1, unbiased=False, keepdim=True)
        adv = (r - r.mean(dim=1, keepdim=True)) / std.clamp_min(1e-8) if normalise else r
        logp = torch.log_softmax(lg_t, dim=-1).gather(1, roll_t.idx)
        (-(adv.detach() * logp).mean(dim=1).mean()).backward()
        return lg_t.grad.clone()

    s0, s5 = grads_shift(0.0, True), grads_shift(5.0, True)
    rs0, rs5 = grads_shift(0.0, False), grads_shift(5.0, False)

    return {
        "N": N, "K": K, "G": G, "steps": steps,
        "bucket_acc_start": acc_hist[0], "bucket_acc_final": acc_hist[-1],
        "acc_hist": acc_hist,
        "zero_signal_share_start": zero_share[0], "zero_signal_share_final": zero_share[-1],
        "sum_advantage_absmax": sum_adv_absmax,
        "scale_invariance_grpo_absdiff": float((g1 - g100).abs().max()),
        "scale_invariance_raw_ratio": float(rg100.norm() / rg1.norm()),
        "baseline_shift_grpo_absdiff": float((s0 - s5).abs().max()),
        "baseline_shift_raw_relchange": float((rs5 - rs0).norm() / rs0.norm()),
    }


# ==================================================== (2) 규칙 기반 보상 파서 (식 4)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
BOX_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def format_reward(text: str) -> dict:
    """형식 보상 = 0.5·(사고 블록 존재+비어있지 않음) + 0.5·(답 블록 존재+비어있지 않음).

    README 가 경고한 "<think>\\n\\n</think>" 우회(사고 회피)는 내용 있는 사고가 아니므로 사고 항을 못 받는다.
    """
    m_think, m_ans = THINK_RE.search(text), ANSWER_RE.search(text)
    think_ok = bool(m_think and m_think.group(1).strip())
    ans_ok = bool(m_ans and m_ans.group(1).strip())
    return {"think_ok": think_ok, "answer_ok": ans_ok,
            "reward": 0.5 * think_ok + 0.5 * ans_ok,
            "bypass_empty_think": bool(m_think and not m_think.group(1).strip())}


def accuracy_reward(text: str, gold: str) -> dict:
    """정확도 보상 = \\boxed{...} 안의 값을 정규화 후 gold 와 비교 (0/1)."""
    m = BOX_RE.search(text)
    pred = m.group(1).strip().replace(" ", "") if m else None
    ok = pred is not None and pred == gold.strip().replace(" ", "")
    return {"pred": pred, "gold": gold, "reward": 1.0 if ok else 0.0}


def proof_rule_reward() -> dict:
    """식 (4) 의 분해: 두 항이 각각 다른 실패 모드를 잡는다(동일 가중 합)."""
    cases = {
        "정답·형식 모두": ("<think>6*7=42</think><answer>\\boxed{42}</answer>", "42"),
        "정답·형식 무":   ("6*7=42 이므로 42", "42"),
        "오답·형식 유":   ("<think>6*7=41</think><answer>\\boxed{41}</answer>", "42"),
        "오답·우회형식":  ("<think>\n\n</think><answer>\\boxed{0}</answer>", "42"),
        "오답·형식 무":   ("모르겠습니다", "42"),
    }
    rows = []
    for name, (text, gold) in cases.items():
        fr = format_reward(text)
        ar = accuracy_reward(text, gold)
        rows.append({"case": name, "format": fr["reward"], "accuracy": ar["reward"],
                     "total_eq4": fr["reward"] + ar["reward"], "bypass": fr["bypass_empty_think"]})
    fmt_only = sum(r["format"] for r in rows) / len(rows)
    no_box = next(r for r in rows if r["case"] == "정답·형식 무")
    return {"cases": rows, "format_only_mean_reward": fmt_only,
            "accuracy_lost_without_box": no_box["accuracy"] == 0.0}


# ================================ (3) 언어 일관성 보상의 trade-off (식 7 · 부록 B.6)
def lc_tradeoff(S: int, G: int, steps: int, lr: float, lams: list[float], device: str) -> dict:
    """토큰 예산이 고정된 사고 사슬(S 스텝)에서 LC 보상 가중 λ 를 스윕한다.

    행동 3종: 0 = 목표언어 추론, 1 = 외국어 추론, 2 = 목표언어 filler.
    진행 성공률(가정): 행동1 = 1.0, 행동0 = 0.7, 행동2 = 0.0. 과제 성공 = 진행 토큰 2개 이상.
    LC 보상 = 목표언어 토큰 비율 = (#0 + #2)/S.  합성 보상 R = R_task + λ·R_lang.
    """
    progress_p = torch.tensor([0.7, 1.0, 0.0])

    def run(lam: float, seed: int) -> dict:
        sg = torch.Generator().manual_seed(seed)
        init_g = torch.Generator().manual_seed(seed)
        logits = (torch.randn(S, 3, generator=init_g) * 0.1).to(device).requires_grad_(True)
        opt = torch.optim.Adam([logits], lr=lr)
        hist = []
        for it in range(steps):
            prob = torch.softmax(logits, dim=-1).detach().cpu()
            idx = torch.multinomial(prob.reshape(-1, 3), G, replacement=True, generator=sg)
            idx = idx.reshape(S, G).to(device)
            pp = progress_p.to(device)
            prog = (torch.rand(S, G, generator=sg) .to(device) < pp[idx]).float()
            r_task = (prog.sum(dim=0) >= 2).float()
            r_lang = (((idx == 0) | (idx == 2)).float().sum(dim=0) / S)
            R = r_task + lam * r_lang
            adv = (R - R.mean()) / R.std(unbiased=False).clamp_min(1e-8)
            logp_s = torch.log_softmax(logits, dim=-1).gather(1, idx)
            loss = -(adv.detach().unsqueeze(0) * logp_s).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it % max(1, steps // 10) == 0 or it == steps - 1:
                with torch.no_grad():
                    pi = torch.softmax(logits, dim=-1)
                    hist.append({"step": it,
                                 "foreign": float(pi[:, 1].mean()),
                                 "filler": float(pi[:, 2].mean()),
                                 "target_reason": float(pi[:, 0].mean())})
        # 학습 후 정책으로 실제 롤아웃해 과제 성공률을 실측한다(해석식이 아니라 표본)
        with torch.no_grad():
            pi_cpu = torch.softmax(logits, dim=-1).cpu().reshape(-1, 3)
            n_eval = 2000
            idx_e = torch.multinomial(pi_cpu, n_eval, replacement=True,
                                      generator=torch.Generator().manual_seed(1234)).reshape(S, n_eval)
            prog_e = (torch.rand(S, n_eval, generator=torch.Generator().manual_seed(4321))
                      < progress_p[idx_e]).float()
            task_rate = float((prog_e.sum(dim=0) >= 2).float().mean())
            foreign_rate = float(((idx_e == 1).float().sum(dim=0) / S).mean())
            filler_rate = float(((idx_e == 2).float().sum(dim=0) / S).mean())
        return {"lam": lam, "final": hist[-1], "hist": hist,
                "task_success": task_rate, "foreign_share": foreign_rate, "filler_share": filler_rate}

    out = []
    for i, lam in enumerate(lams):
        runs = [run(lam, 100 + 10 * i + k) for k in range(5)]
        avg = lambda key: sum(r[key] for r in runs) / len(runs)
        out.append({"lam": lam, "hist": runs[0]["hist"], "final": runs[0]["final"],
                    "task_success": avg("task_success"), "foreign_share": avg("foreign_share"),
                    "filler_share": avg("filler_share"), "n_seeds": len(runs)})
    return {"S": S, "G": G, "steps": steps, "progress_p": progress_p.tolist(), "sweeps": out,
            "frontier": lc_frontier(S, [0.0, 0.25, 0.5, 0.75, 1.0])}


PROGRESS_FOREIGN = 1.0
PROGRESS_TARGET_LANG = 0.7


def lc_frontier(S: int, qs: list[float]) -> list[dict]:
    import math
    gen = torch.Generator().manual_seed(2024)
    rows = []
    for q in qs:
        p = q * PROGRESS_TARGET_LANG + (1.0 - q) * PROGRESS_FOREIGN
        exact = sum(math.comb(S, k) * p ** k * (1.0 - p) ** (S - k) for k in range(2, S + 1))
        n_eval = 20000
        target_token = (torch.rand(S, n_eval, generator=gen) < q).float()
        prog_p = torch.where(target_token > 0.5, PROGRESS_TARGET_LANG, PROGRESS_FOREIGN)
        prog = (torch.rand(S, n_eval, generator=gen) < prog_p).float()
        rows.append({"target_share": q, "mixing": 1.0 - q, "task_success_exact": exact,
                     "task_success_sim": float((prog.sum(dim=0) >= 2).float().mean())})
    return rows


# ============================================== (4) 보상 해킹 재현 (부록 B.5 · Figure 6)
def reward_hacking(G: int, steps: int, lr: float, use_gate: bool, device: str) -> dict:
    """보상 모델에 '길이 선호' 체계적 편향이 있을 때의 최적화 결과.

    정책은 고정 예산 아래 세 행동 중 하나를 고른다: 0 = 정답을 노리는 추론, 1 = 길이 부풀리기, 2 = 포기.
    프록시 보상은 [0.3, 1.0, 0.0] — 노력보다 길이 부풀리기를 더 짝하게 매긴다.  실제 지표는 행동 0 에서만 1.
    두 행동이 같은 예산을 두고 경쟁하므로 프록시를 최대화하는 해는 실제 지표를 떨어뜨린다.
    use_gate=True 면 규칙 기반 검증을 통과한 행동(0)에만 보상을 인정한다 → 최적해가 실제 지표와 일치한다.
    """
    proxy_table = torch.tensor([0.3, 1.0, 0.0], device=device)
    true_table = torch.tensor([1.0, 0.0, 0.0], device=device)
    g = torch.Generator().manual_seed(0)
    logits = torch.zeros(3, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    hist = []
    for it in range(steps):
        prob = torch.softmax(logits, dim=-1)
        a = torch.multinomial(prob.detach().cpu(), G, replacement=True, generator=g).to(device)
        proxy = proxy_table[a]
        reward = torch.where(a == 0, proxy, torch.zeros_like(proxy)) if use_gate else proxy
        adv = (reward - reward.mean()) / reward.std(unbiased=False).clamp_min(1e-8)
        logp = torch.log_softmax(logits, dim=-1)[a]
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, steps // 10) == 0 or it == steps - 1:
            with torch.no_grad():
                pi = torch.softmax(logits, dim=-1)
            hist.append({"step": it, "proxy": float(proxy.mean()), "true_acc": float(true_table[a].mean()),
                         "p_effort": float(pi[0]), "p_pad": float(pi[1]), "p_giveup": float(pi[2])})
    return {"use_gate": use_gate, "hist": hist, "final": hist[-1]}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lc-steps", type=int, default=600)
    args = ap.parse_args()
    dev = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    print(f"[R1 demo] device={dev} N={args.N} K={args.K} G={args.G} steps={args.steps}")

    metrics: dict = {}

    g = grpo_group_advantage(args.N, args.K, args.G, args.steps, args.lr, dev)
    metrics["grpo"] = g
    print("\n(1) GRPO 그룹 상대 이점 — 가치망 없이 그룹 정규화만으로")
    for b in ("easy", "mid", "hard"):
        print(f"    정답확률 {b:5s}: {g['bucket_acc_start'][b]:.3f} → {g['bucket_acc_final'][b]:.3f}")
    print(f"    Σ_i A_i 의 최대 |값| = {g['sum_advantage_absmax']:.2e}   [EXACT: 그룹평균 = 내장 베이스라인]")
    print(f"    난이도 게이팅: 전원정답/전원오답 그룹 비율 {g['zero_signal_share_start']:.3f} → "
          f"{g['zero_signal_share_final']:.3f} (이 그룹의 A=0 → 그래디언트 0)")
    print(f"    보상 스케일 불변: 정규화 시 |Δ(r) − Δ(100r)|max = {g['scale_invariance_grpo_absdiff']:.2e} / "
          f"정규화 없을 때 ‖Δ(100r)‖/‖Δ(r)‖ = {g['scale_invariance_raw_ratio']:.1f}배")
    print(f"    보상 평행이동 불변(베이스라인 효과): 정규화 시 |Δ(r) − Δ(r+5)|max = "
          f"{g['baseline_shift_grpo_absdiff']:.2e} / 정규화 없을 때 상대변화 {g['baseline_shift_raw_relchange']:.3f}")

    rr = proof_rule_reward()
    metrics["rule_reward"] = rr
    print("\n(2) 규칙 기반 보상 분해 (식 4: Reward_rule = Reward_acc + Reward_format)")
    for r in rr["cases"]:
        print(f"    {r['case']:12s} 정확도 {r['accuracy']:.1f} + 형식 {r['format']:.1f} = {r['total_eq4']:.1f}"
              + ("   ← 우회(<think> 빈 블록) 형식 강등" if r["bypass"] else ""))
    print(f"    해킹 표면: 정답을 못 맞히고 형식만 갖춘 정책의 평균 보상 = "
          f"{rr['format_only_mean_reward']:.2f} (정답 0인데 0 이 아님)")
    print(f"    box 형식이 없으면 정확도 보상이 0 이 된다(규칙 검증의 전제조건): "
          f"{rr['accuracy_lost_without_box']}")

    lc = lc_tradeoff(4, 16, args.lc_steps, 0.08, [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0], dev)
    metrics["lc_tradeoff"] = lc
    print("\n(3) 언어 일관성 보상 λ 스윕 (식 7 · B.6) — 외국어 추론이 과제에 더 유효하다는 가정")
    for s in lc["sweeps"]:
        print(f"    λ={s['lam']:.2f}  혼용률 {s['foreign_share']*100:5.1f}%   "
              f"filler {s['filler_share']*100:5.1f}%   과제 성공률(실측) {s['task_success']*100:5.1f}%")
    print(f"    비용 곡선(목표어 비율 q 로 직접 계산; 진행률 가정 외국어 {PROGRESS_FOREIGN:.1f} / "
          f"목표어 {PROGRESS_TARGET_LANG:.1f}, S={lc['S']}, 과제 = 진행 {2}개 이상):")
    for f in lc["frontier"]:
        print(f"      q={f['target_share']:.2f}  혼용 {f['mixing']*100:5.1f}%   "
              f"과제 성공률 해석해 {f['task_success_exact']*100:5.1f}%   표본 {f['task_success_sim']*100:5.1f}%")

    rh = reward_hacking(args.G, args.steps, 0.1, False, dev)
    rg = reward_hacking(args.G, args.steps, 0.1, True, dev)
    metrics["reward_hacking"] = {"ungated": rh, "gated": rg}
    print("\n(4) 보상 해킹 재현 (B.5 · Figure 6) — 보상↑ 인데 실제 지표↓")
    print(f"    게이트 없음: 프록시 보상 {rh['hist'][0]['proxy']:.3f} → {rh['final']['proxy']:.3f}"
          f"   실제 정답률 {rh['hist'][0]['true_acc']:.3f} → {rh['final']['true_acc']:.3f}")
    print(f"    검증 게이트: 프록시 보상 {rg['hist'][0]['proxy']:.3f} → {rg['final']['proxy']:.3f}"
          f"   실제 정답률 {rg['hist'][0]['true_acc']:.3f} → {rg['final']['true_acc']:.3f}")

    RUNS.mkdir(exist_ok=True)
    (RUNS / "r1_demo_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    saved = [str(RUNS / "r1_demo_metrics.json")]
    for name, fn in (("grpo_accuracy.png", _plot_grpo), ("lc_tradeoff.png", _plot_lc),
                     ("reward_hacking.png", _plot_hack)):
        if fn(g, lc, rh, rg, RUNS / name):
            saved.append(str(RUNS / name))
    print("\n[산출물] " + ", ".join(saved))
    print("[결론] GRPO 는 가치망 없이 그룹 평균을 베이스라인으로 쓰고(ΣA=0), 난이도가 극단인 그룹은 "
          "이점이 0이 되어 자연히 걸러진다. 규칙 보상은 정확도·형식을 분리해 계산하며 box 형식은 "
          "정확도 검증의 전제조건이다. LC 보상은 혼용률을 낮추는 대신 과제 성능을 조금 깎고, "
          "보상 모델 편향은 게이트 없이는 반드시 해킹된다.")


_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    pass


def _plot_grpo(g: dict, lc: dict, rh: dict, rg: dict, path: Path) -> bool:
    if not _HAS_MPL:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    for b in ("easy", "mid", "hard"):
        ax.plot([h["step"] for h in g["acc_hist"]], [h[b] for h in g["acc_hist"]], label=b)
    ax.set_xlabel("GRPO step"); ax.set_ylabel("P(correct)")
    ax.set_title("group-relative advantage: no value net (simulated)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return True


def _plot_lc(g: dict, lc: dict, rh: dict, rg: dict, path: Path) -> bool:
    if not _HAS_MPL:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    lams = [s["lam"] for s in lc["sweeps"]]
    ax.plot(lams, [s["foreign_share"] * 100 for s in lc["sweeps"]], marker="o", label="mixing rate")
    ax.plot(lams, [s["task_success"] * 100 for s in lc["sweeps"]], marker="s", label="task success")
    ax.set_xlabel("LC reward weight λ"); ax.set_ylabel("%")
    ax.set_title("language-consistency reward: mixing down, task slightly down (simulated)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return True


def _plot_hack(g: dict, lc: dict, rh: dict, rg: dict, path: Path) -> bool:
    if not _HAS_MPL:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    xs = [h["step"] for h in rh["hist"]]
    ax.plot(xs, [h["proxy"] for h in rh["hist"]], label="proxy reward (ungated)")
    ax.plot(xs, [h["true_acc"] for h in rh["hist"]], label="true accuracy (ungated)")
    ax.set_xlabel("step"); ax.set_ylabel("value")
    ax.set_title("reward hacking: proxy rises while truth falls (simulated)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return True


if __name__ == "__main__":
    main()
