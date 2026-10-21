# 🖼️ 논문 포스터 키트 (2026-10-22 자율세션용)

> **목적**: 10/22(목) 자율세션의 과제인 논문 포스터를 하루 안에 완성하기 위한 템플릿·문안·빌드 스크립트.
> **사용법**: ① `content.md`의 문안을 확인/수정 → ② `build_poster.py` 실행 → ③ `runs/poster_det_rtdetr.pdf` 확인 → ④ 인쇄소 제출(또는 HTML 편집).

## 구성
| 파일 | 역할 |
|---|---|
| `README.md` | 이 안내(사용 순서·주의) |
| `outline.md` | 포스터 구조 설계(A0 세로 3열) — 무엇을 어디에 놓는지 |
| `content.md` | **문안 본문**(검증된 수치만 사용) — 그대로 복사해 쓰면 됨 |
| `build_poster.py` | 포스터 PDF 생성기(matplotlib, 한글 폰트, 차트 자동 배치) |
| `runs/` | 생성 결과(PDF·PNG 미리보기) |

## 빌드
```bash
cd /Users/robotjang/paper-diary
.venv/bin/python poster/build_poster.py            # A1 세로 PDF + PNG 미리보기
.venv/bin/python poster/build_poster.py --size A0   # A0 세로(인쇄용)
.venv/bin/python poster/build_poster.py --slides    # 16:9 슬라이드 3장(발표용)
```

## 주의(정직성 규칙 — 이 다이어리의 핵심 원칙)
1. **포스터의 모든 수치는 원문 표/절에서 확인된 값만 쓴다.** (Day 07·08 채점 기준을 그대로 적용)
2. **자기 실측과 논문 수치를 반드시 구분 표기** — 캡션에 "논문 Table X"(인용) / "자체 실측"(우리 데모) / "시뮬레이션"을 구분.
3. **같은 도메인 안에서만 비교** — DETR 계열끼리, YOLO 계열끼리. 교차 숫자 비교 금지(README 비교 원칙).
4. **한계도 적는다** — 논문 한계(예: RT-DETR의 AP_S −0.5)와 우리 실습의 한계(합성 데이터·지연 가변)를 함께 적는 것이 오히려 신뢰를 높인다.
