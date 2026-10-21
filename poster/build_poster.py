#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
논문 포스터 빌더 (Day 08 산출물 기반)
=====================================
입력: day07_detr/runs/chart_*.png · day08_rtdetr/runs/chart_*.png
출력: poster/runs/poster_det_rtdetr.pdf (+ PNG 미리보기)

사용:
  .venv/bin/python poster/build_poster.py              # A1 세로
  .venv/bin/python poster/build_poster.py --size A0    # A0 세로(인쇄)
  .venv/bin/python poster/build_poster.py --slides    # 16:9 슬라이드 3장

주의: 모든 수치는 `poster/content.md`의 검증된 값과 일치한다(원문 표 인용 / 자체 실측 구분 표기).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

matplotlib.rcParams["font.family"] = ["NanumGothic", "AppleGothic"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "runs")
os.makedirs(OUT, exist_ok=True)

INK = "#16202a"
MUTED = "#5b6b7a"
ACCENT = "#1f5f8b"
ACCENT2 = "#b4543a"
PANEL = "#f7f9fb"
LINE = "#d9e0e6"

SIZES = {"A0": (33.1, 46.8), "A1": (23.4, 33.1), "A2": (16.5, 23.4)}


def load(rel: str):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print(f"  [skip] {rel} 없음")
        return None
    return plt.imread(path)


def panel(ax, x, y, w, h, title: str, title_color: str = ACCENT) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.006",
                                linewidth=1.1, edgecolor=LINE, facecolor=PANEL, transform=ax.transAxes))
    ax.text(x + 0.014, y + h - 0.028, title, transform=ax.transAxes, fontsize=15,
            color=title_color, weight="bold", va="top")
    ax.plot([x + 0.014, x + w - 0.014], [y + h - 0.045, y + h - 0.045], color=LINE, lw=1.0,
            transform=ax.transAxes)


def body(ax, x, y, lines, size=11.5, leading=1.55, color=INK):
    tx = ax.text(x, y, "", transform=ax.transAxes, fontsize=size, color=color, va="top")
    tx.set_text("\n".join(lines))
    return tx


def bullet(ax, x, y, head, rest, size=11.5, color=INK, head_color=ACCENT):
    ax.text(x, y, "▪", transform=ax.transAxes, fontsize=size, color=head_color, va="top")
    ax.text(x + 0.016, y, head, transform=ax.transAxes, fontsize=size, color=head_color,
            weight="bold", va="top")
    ax.text(x + 0.016 + 0.001, y, rest, transform=ax.transAxes, fontsize=size, color=color, va="top")


def _imshow(ax, img, extent, placed: list) -> None:
    ax.imshow(img, extent=extent, transform=ax.transAxes, aspect="auto")
    placed.append(tuple(extent))


def check_overlaps(fig, ax, placed) -> list:
    """텍스트·이미지 블록의 겹침을 기하로 검사(내일 문안 수정 후 재사용)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = []
    boxes = []
    for t in ax.texts:
        try:
            bb = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb.width == 0 or bb.height == 0:
            continue
        inv = ax.transAxes.inverted()
        corners = inv.transform(bb)
        b = (float(corners[:, 0].min()), float(corners[:, 1].min()),
             float(corners[:, 0].max()), float(corners[:, 1].max()))
        boxes.append((b, t.get_text()[:28]))
    for i, (b1, t1) in enumerate(boxes):
        for b2, t2 in boxes[i + 1:]:
            if b1[0] < b2[2] and b2[0] < b1[2] and b1[1] < b2[3] and b2[1] < b1[3]:
                area = (min(b1[2], b2[2]) - max(b1[0], b2[0])) * (min(b1[3], b2[3]) - max(b1[1], b2[1]))
                if area > 0.0015:
                    issues.append(f"TEXT×TEXT {t1!r} ↔ {t2!r} (면적 {area:.4f})")
    for ext in placed:
        for b, t in boxes:
            if b[0] < ext[1] and ext[0] < b[2] and b[1] < ext[3] and ext[2] < b[3]:
                area = (min(b[2], ext[1]) - max(b[0], ext[0])) * (min(b[3], ext[3]) - max(b[1], ext[2]))
                if area > 0.0015:
                    issues.append(f"TEXT×IMAGE {t!r} (면적 {area:.4f})")
    return issues


def build_poster(size: str = "A1") -> str:
    w_in, h_in = SIZES[size]
    placed: list = []
    fig = plt.figure(figsize=(w_in, h_in), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", transform=ax.transAxes))

    # ── 제목 블록 ──
    ax.add_patch(Rectangle((0, 0.875), 1, 0.125, facecolor="#eef3f7", transform=ax.transAxes))
    ax.text(0.035, 0.955, "NMS 없이 실시간:  DETR → RT-DETR", transform=ax.transAxes,
            fontsize=30, weight="bold", color=INK, va="center")
    ax.text(0.035, 0.905, "세트 예측이 정확도와 속도를 동시에 얻는 구조적 이유", transform=ax.transAxes,
            fontsize=15, color=MUTED, va="center")
    ax.text(0.965, 0.955, "Carion et al., ECCV 2020 (arXiv:2005.12872)\nLv et al., CVPR 2024 (arXiv:2304.08069)",
            transform=ax.transAxes, fontsize=10.5, color=MUTED, va="center", ha="right")

    colw = 0.305
    xs = [0.028, 0.348, 0.668]
    top = 0.855
    row1_h = 0.40
    row2_h = 0.235

    # ── ② 문제 ──
    panel(ax, xs[0], top - row1_h, colw, row1_h, "① 문제 — NMS의 딜레마")
    body(ax, xs[0] + 0.014, top - 0.075, [
        "두 임계값이 정확도와 시간을 반대 방향으로 움직인다",
        "(논문 Table 1 · T4 TensorRT FP16)",
        "",
        "conf 0.001 → 0.05",
        "   AP 52.9 → 51.2   NMS 2.36 → 1.06 ms",
        "IoU  0.5   → 0.8",
        "   AP 52.1 → 52.8   NMS 2.24 → 2.46 ms",
        "",
        "YOLO는 보통 NMS 시간을 제외한 모델 속도만",
        "보고한다 → RT-DETR은 NMS 플러그인까지 포함한",
        "end-to-end benchmark를 새로 세웠다(§3.2).",
    ], size=11.2)
    img = load("day08_rtdetr/runs/chart_nms_cost.png")
    if img is not None:
        _imshow(ax, img, (xs[0] + 0.012, xs[0] + colw - 0.012, top - row1_h + 0.028,
                               top - row1_h + 0.185), placed)

    # ── ③ 방법 ──
    panel(ax, xs[1], top - row1_h, colw, row1_h, "② 방법 — 세 번의 이동")
    body(ax, xs[1] + 0.014, top - 0.075, [
        "(1) 중복 억제를 학습으로 (DETR §3.1)",
        "   헝가리아리안 최적 양자택일 매칭 →",
        "   NMS 없이 중복이 구조적으로 불가",
        "   매칭 비용=확률 / 손실=로그 우도",
        "   박스 손실=스케일 불변 GIoU + L1",
        "",
        "(2) 컴퓨트를 설계로 (RT-DETR §4.2)",
        "   인코더가 GFLOPs 49%인데 AP는 11%",
        "   → AIFI(S5만, 400토큰) + CCFF(conv 융합)",
        "",
        "(3) 초기화를 품질로 (RT-DETR §4.3)",
        "   U=‖P-C‖를 손실에 → 분류·위치 일치 피처",
        "   Prop both 0.30% → 0.67%,  AP 47.9 → 48.7",
        "   ※ 랭킹은 그대로(분류 점수 기준)",
    ], size=11.0)
    img = load("day07_detr/runs/chart_pipeline_diagram.png")
    if img is not None:
        _imshow(ax, img, (xs[1] + 0.012, xs[1] + colw - 0.012, top - row1_h + 0.022,
                               top - row1_h + 0.125), placed)

    # ── ④ 결과 ──
    panel(ax, xs[2], top - row1_h, colw, row1_h, "③ 결과 — 정확도와 속도를 함께")
    tbl = [
        ("", "DETR", "FRCNN+", "RT-DETR"),
        ("AP", "42.0", "42.0", "53.1"),
        ("AP_S", "20.5", "26.6", "34.8"),
        ("AP_L", "61.1", "53.4", "70.0"),
        ("params", "41M", "42M", "42M"),
        ("FPS", "28", "26", "108"),
    ]
    ty = top - 0.082
    for r, (c0, c1, c2, c3) in enumerate(tbl):
        bold = r == 0
        col = ACCENT if r in (1, 5) else INK
        ax.text(xs[2] + 0.018, ty - r * 0.0225, c0, transform=ax.transAxes, fontsize=10.8,
                color=MUTED, weight="bold" if bold else "normal")
        for k, v in enumerate((c1, c2, c3)):
            ax.text(xs[2] + 0.105 + k * 0.065, ty - r * 0.0225, v, transform=ax.transAxes,
                    fontsize=10.8, color=col, weight="bold" if (bold or k == 2) else "normal",
                    ha="center")
    body(ax, xs[2] + 0.014, ty - 0.150, [
        "동일 파라미터(41M vs 42M)에서 AP 동급,",
        "대신 AP_S -5.5 · AP_L +7.8로 성질이 뒤집힘.",
        "RT-DETR은 DINO-Deformable 대비 +2.2 AP·21× FPS.",
        "",
        "※ 조건부 비교: 입력 640×640(72ep) vs 800×1333(500ep),",
        "   하드웨어·정밀도 상이 → 절대값 비교 아님",
    ], size=10.8)
    img = load("day07_detr/runs/chart_detr_ap_breakdown.png")
    if img is not None:
        _imshow(ax, img, (xs[2] + 0.012, xs[2] + colw - 0.012, top - row1_h + 0.016,
                               top - row1_h + 0.128), placed)

    # ── ⑤ 우리 실험 ──
    r2top = top - row1_h - 0.025
    panel(ax, xs[0], r2top - row2_h, colw, row2_h, "④ 우리의 실험 (자체 실측·합성 데이터)")
    body(ax, xs[0] + 0.014, r2top - 0.062, [
        "· 인코더 변형 A~E 지연 순서를 논문과 동일하게 재현:",
        "   A < D_S5 < E < B < D < C",
        "· attention 토큰 8배 → 시간 42.3배(이론 64배)",
        "   8400 vs 400 토큰 = 441배 토큰 쌍 차이",
        "· NMS 딜레마 두 방향 재현(conf↑/IoU↑)",
        "· uncertainty 단독 최적화 퇴화 해 정량화:",
        "   s 0.90→0.690, q 0.20→0.693(서로 만남)",
        "   분류 손실 병행 시 0.794/0.793(품질이 확신 추종)",
    ], size=10.6)
    img = load("day08_rtdetr/runs/chart_uncertainty_degeneracy.png")
    if img is not None:
        _imshow(ax, img, (xs[0] + 0.012, xs[0] + colw - 0.012, r2top - row2_h + 0.022,
                               r2top - row2_h + 0.115), placed)

    # ── ⑥ 한계 ──
    panel(ax, xs[1], r2top - row2_h, colw, row2_h, "⑤ 한계 — 배운 것", title_color=ACCENT2)
    body(ax, xs[1] + 0.014, r2top - 0.062, [
        "· 소형 객체는 여전히 약점: AP_S 34.8",
        "  (YOLOv8-L 35.3 대비 -0.5, R101도 -0.9)",
        "  멀티스케일 특징을 쓰되 전역 attention을 S5에",
        "  제한한 설계의 대가(논문은 원인을 단독 입증 안 함)",
        "· 49/11은 2차 통계(RT-DETR←Lin et al.←Deformable)",
        "· 138%·120%는 상대 개수 차이지 %p 개선이 아님",
        "· selection≠objective: uncertainty는 손실을 바꿈",
        "· 논문은 2D COCO만 평가 → 3D 확장 미검증",
    ], size=10.6)
    img = load("day08_rtdetr/runs/chart_attention_scaling.png")
    if img is not None:
        _imshow(ax, img, (xs[1] + 0.012, xs[1] + colw - 0.012, r2top - row2_h + 0.022,
                               r2top - row2_h + 0.115), placed)

    # ── ⑦ 흐름 ──
    panel(ax, xs[2], r2top - row2_h, colw, row2_h, "⑥ 학습 흐름 · 다음")
    body(ax, xs[2] + 0.014, r2top - 0.062, [
        "D1 MAE → D2 BEiT/SimMIM → D3 iBOT/DINOv2",
        "→ D4~6 DeepSeek(V2/V3/R1)",
        "→ D7 DETR + RT-DETR 심화 (96.6%)",
        "→ D8 RT-DETR 본문 (97.0%)",
        "",
        "산출물: 심화자료 16종 · 데모 4종 · 차트 22종",
        "       논문 대조 오류 0건",
        "",
        "다음: GLIP+SAM → CLIP → LLaVA(2일) → VL-JEPA",
        "병행: 3D 트랙 (DETR3D→PETR→StreamPETR→Sparse4D)",
    ], size=10.6)
    img = load("day08_rtdetr/runs/chart_encoder_variants.png")
    if img is not None:
        _imshow(ax, img, (xs[2] + 0.012, xs[2] + colw - 0.012, r2top - row2_h + 0.022,
                               r2top - row2_h + 0.115), placed)

    # ── ⑧ 한 줄 요약 ──
    ax.add_patch(Rectangle((0, 0.0), 1, 0.055, facecolor="#eef3f7", transform=ax.transAxes))
    ax.text(0.5, 0.028, "중복 제거를 학습으로 · 컴퓨트를 설계로 · 초기화를 품질로 옮기면  실시간과 정확도를 함께 얻는다",
            transform=ax.transAxes, fontsize=16, color=ACCENT, weight="bold", ha="center", va="center")

    issues = check_overlaps(fig, ax, placed)
    if issues:
        print(f"  [layout] 겹침 {len(issues)}건:")
        for msg in issues[:12]:
            print("   -", msg)
    else:
        print("  [layout] 겹침 없음")

    path = os.path.join(OUT, f"poster_det_rtdetr_{size}.pdf")
    fig.savefig(path)
    png = os.path.join(OUT, f"poster_det_rtdetr_{size}.png")
    fig.savefig(png, dpi=60)
    plt.close(fig)
    print(f"  saved {os.path.relpath(path, ROOT)}  (+ PNG 미리보기)")
    return path


def build_slides() -> str:
    """16:9 슬라이드 3장(발표용)."""
    slides = [
        ("NMS 없이 실시간: DETR → RT-DETR",
         ["중복 제거를 후처리(NMS)에서 학습(세트 예측)으로 옮기면,",
          "정확도와 속도를 함께 얻을 수 있다 — 그 구조적 이유",
          "",
          "Carion et al., ECCV 2020 (arXiv:2005.12872)",
          "Lv et al., CVPR 2024 (arXiv:2304.08069)"]),
        ("세 번의 이동",
         ["1) 중복 억제를 학습으로 — 헝가리아리안 최적 양자택일 매칭(§3.1)",
          "2) 컴퓨트를 설계로 — AIFI(S5만)+CCFF(conv 융합)(§4.2)",
          "3) 초기화를 품질로 — U=‖P-C‖를 손실에(§4.3)",
          "",
          "42.0 AP(동일 파라미터) → 53.1 AP / 108 FPS"]),
        ("한계와 다음",
         ["AP_S 34.8(-0.5) · 49/11은 2차 통계 · 138/120은 상대 개수 차",
          "선택≠최적화(uncertainty는 손실을 바꾼다) · 2D 전용 검증",
          "",
          "다음: GLIP+SAM → CLIP → LLaVA → VL-JEPA (3D 트랙 병행)"]),
    ]
    paths = []
    for i, (title, lines) in enumerate(slides, start=1):
        placed: list = []
        fig = plt.figure(figsize=(13.33, 7.5), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0.88), 1, 0.12, facecolor="#eef3f7", transform=ax.transAxes))
        ax.text(0.05, 0.94, title, transform=ax.transAxes, fontsize=26, weight="bold", color=INK, va="center")
        ax.text(0.05, 0.5, "\n".join(lines), transform=ax.transAxes, fontsize=16, color=INK, va="center",
                linespacing=1.8)
        if i == 1:
            img = load("day08_rtdetr/runs/chart_nms_cost.png")
            if img is not None:
                _imshow(ax, img, (0.56, 0.95, 0.16, 0.82), placed)
        if i == 2:
            img = load("day07_detr/runs/chart_pipeline_diagram.png")
            if img is not None:
                _imshow(ax, img, (0.50, 0.95, 0.18, 0.80), placed)
        if i == 3:
            img = load("day07_detr/runs/chart_rtdetr_encoder_tradeoff.png")
            if img is not None:
                _imshow(ax, img, (0.56, 0.95, 0.16, 0.82), placed)
        p = os.path.join(OUT, f"slide_{i}.pdf")
        fig.savefig(p)
        fig.savefig(os.path.join(OUT, f"slide_{i}.png"), dpi=60)
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="A1", choices=list(SIZES))
    ap.add_argument("--slides", action="store_true")
    args = ap.parse_args()
    if args.slides:
        for p in build_slides():
            print("  saved", os.path.relpath(p, ROOT))
    else:
        p = build_poster(args.size)
        print("  saved", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
