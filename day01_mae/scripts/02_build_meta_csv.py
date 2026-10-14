"""TL_A_sampled 프레임 → CSV (path, clip, split)."""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = Path("/Users/robotjang/spatial-avla-pipeline/exports/TL_A_sampled")
OUT  = ROOT / "exports" / "mae_smoke_meta.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted(SRC.rglob("*.jpg")):   # ← frames/ 제거
    parts = f.parts
    # .../TL_A_sampled/{split}/{clip}/xxx.jpg
    clip  = parts[-2]
    split = parts[-3]
    rows.append({"path": str(f), "clip": clip, "split": split})

with open(OUT, "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=["path", "clip", "split"])
    w.writeheader()
    w.writerows(rows)

print(f"[ok] wrote {len(rows)} rows → {OUT}")
if rows:
    print(f"[ok] splits = {dict(Counter(r['split'] for r in rows))}")
    print(f"[ok] clips  = {dict(Counter(r['clip']  for r in rows))}")