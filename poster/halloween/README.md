# Halloween Poster Session (2026-10-30 자율세션)

할로윈 한정 테마 포스터 4종 — 팀원 공유 논문 4편.

- `halloween.css` — `../poster.css`와 동일한 구조 클래스(hero/journey/columns/card/…)에
  다크 퍼플 배경 + 펌킨 앰버/베놈 그린 액센트를 입힌 테마 시트.
- 소스 규칙은 기존 키트와 동일: **모든 수치는 원문 표/그림/절 라벨 + 조건 명시**.
  검증 불가 수치는 게재하지 않는다(라이브러리니안 팩트시트만 사용).
- 빌드 (`repo root` 기준, A0 = 정식 인쇄본 / A1 = compact):
  ```bash
  .venv/bin/python poster/export_poster.py --html halloween/poster_stamplus.html --tag halloween-stamplus --size A0
  .venv/bin/python poster/export_poster.py --html halloween/poster_search_r1.html --tag halloween-searchr1 --size A0
  .venv/bin/python poster/export_poster.py --html halloween/poster_pe.html --tag halloween-pe --size A0
  .venv/bin/python poster/export_poster.py --html halloween/poster_symmetry.html --tag halloween-symmetry --size A0
  ```
  (A1은 같은 명령에 `--size A1`만 교체)
- 결과물: `../runs/poster_halloween-*_{A1,A0}.{pdf,png}`

| 파일 | 논문 | arXiv |
|---|---|---|
| `poster_stamplus.html` | Better, Stronger, Faster, and Broader: Structured All-Mask Prediction (STAMPlus) | 2608.02791 |
| `poster_search_r1.html` | Search-R1: Training LLMs to Reason and Leverage Search Engines with RL | 2503.09516 |
| `poster_pe.html` | Perception Encoder | 2504.13181 |
| `poster_symmetry.html` | Left-Right Symmetry Breaking in CLIP-style VLMs | 2601.12809 |
