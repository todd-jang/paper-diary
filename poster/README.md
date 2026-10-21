# Paper Poster Kit (for the 2026-10-22 self-study session)

> **Purpose**: everything needed to finish a full-color academic poster in one session.
> **Workflow**: 1) edit `content.md` (copy) 2) `make_poster_charts.py` (regenerate figures in the poster
> palette) 3) `build_poster.py` (render PDF) 4) open `runs/poster_det_rtdetr_A0.png` to check 5) print.

> **Design**: 3-column layout, 6-color palette (navy / teal / amber / coral / violet / mint), tinted
> panels, colored metric cards, a rendered results table with mini bars, a native pipeline flow diagram,
> a 2x2 chart gallery, a Day-timeline, and a colored takeaway footer. **All poster text is English.**

## 제출용 HTML 포스터
- 한글: `poster_ko.html`
- English: `poster_en.html`
- 두 버전은 A0 세로 인쇄 규격이며 `poster.css`를 공유한다.
- 그림은 `runs/figures/`에서 의미별로 배치했다. 한글판은 원본 차트를, 영문판은 `-en` 접미사가 붙은 영어 축·범례 차트를 사용한다.
- 영어 차트를 다시 생성하려면 `./.venv/bin/python poster/make_english_figures.py`를 실행한다. 원본 실험 JSON을 읽고 별도 영문 PNG를 만들어 한글 그림을 덮어쓰지 않는다.
- 브라우저에서 HTML을 열고 인쇄 대화상자에서 A0·배경 그래픽 포함·PDF 저장을 선택한다.

## 구성
| 파일 | 역할 |
|---|---|
| `README.md` | 이 안내(사용 순서·주의) |
| `outline.md` | 포스터 구조 설계(A0 세로 3열) — 무엇을 어디에 놓는지 |
| `content.md` | **문안 본문**(검증된 수치만 사용) — 그대로 복사해 쓰면 됨 |
| `make_poster_charts.py` | 포스터 전용 차트 6종을 포스터 팔레트로 재생성(`figs/`) — NMS 비용 · attention 스케일링 · 인코더 변형 · uncertainty 퇴화 · L1/GIoU · **AP vs FPS(논문 수치)** |
| `poster_en.html` / `poster.css` | **포스터 본체**(9카드 · hero 통계 · journey 스트립) — 편집 대상 |
| `export_poster.py` | HTML → A0/A1 PDF + PNG 변환(헤드리스 Chromium) |
| `make_english_figures.py` | 영문 라벨 버전 차트 생성(`runs/figures/*-en.png`) |
| `build_poster.py` | 대체 빌더(matplotlib, 색상 시스템, 레이아웃 겹침 검사 내장) |
| `runs/` | 생성 결과(PDF·PNG 미리보기) |
| `poster_ko.html`, `poster_en.html` | 그림을 포함한 한글·영문 제출용 포스터 |
| `poster.css` | 두 HTML 포스터의 컬러 및 인쇄 레이아웃 |
| `runs/figures/` | Day 07·08 원본 실습 그림의 포스터용 복사본 |
| `make_english_figures.py` | 실험 JSON에서 영문 축·범례 차트를 재생성 |

## Build (canonical path: HTML -> PDF)

The poster source of truth is **`poster_en.html` + `poster.css`** (9 cards, hero stats, journey strip).
`export_poster.py` renders it with headless Chromium into a print-ready PDF at the exact page size.

```bash
cd /Users/robotjang/paper-diary
.venv/bin/python poster/export_poster.py            # A0 PDF (841x1189mm) + PNG preview
.venv/bin/python poster/export_poster.py --size A1  # A1 (594x841mm)
.venv/bin/python poster/export_poster.py --html poster_ko.html --tag ko
```
One-time setup (already done): `.venv/bin/python -m pip install playwright && .venv/bin/python -m playwright install chromium`

**Alternative (matplotlib) builder** - useful for a fully scripted, layout-checked variant:
```bash
.venv/bin/python poster/make_poster_charts.py     # poster-palette figures -> poster/figs/
.venv/bin/python poster/build_poster.py           # A1/A0 poster + 4 slides (16:9)
```
It prints a layout report (`[layout] no overlaps` or a collision list) - run it after every copy edit.

## Honesty rules (the diary's core rule)
1. **Every number must be traceable** to a paper table/section or to a logged measurement of ours.
2. **Label the source** in every caption: *paper Table X* / *our measurement* / *simulation*.
3. **Compare only within one domain** (DETR family vs YOLO family separately; no cross-domain wins).
4. **State the limitations** - the paper's own (AP_S -0.5) and ours (synthetic data, wall-clock latency
   varies up to 2x between runs).
5. **Mark conditional comparisons**: 640x640/72ep vs 800x1333/500ep are not the same experiment.
