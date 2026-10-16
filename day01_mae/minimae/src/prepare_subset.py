"""
prepare_subset.py — ImageNet-1K parquet shard → ImageFolder 서브셋 구성

입력: HuggingFace `mrm8488/ImageNet1K-train` 의 parquet shard (image(bytes,path), label)
출력: <out>/train/<label>/<idx>.jpg , <out>/val/<label>/<idx>.jpg , <out>/class_map.json

사용:
  py prepare_subset.py --shards "data/train-0000*.parquet" --out data/subset \
      --per-class 1200 --val-per-class 60
"""
import argparse
import glob
import io
import json
import os
import random
from collections import defaultdict

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="parquet 파일들 (glob은 셸 확장)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=1200, help="클래스당 train 최대 장수")
    ap.add_argument("--val-per-class", type=int, default=50)
    ap.add_argument("--max-classes", type=int, default=1000000)
    ap.add_argument("--max-side", type=int, default=256, help="긴 변 최대 크기로 리사이즈 후 저장")
    ap.add_argument("--classes-file", default="data/imagenet_classes.txt",
                    help="1000개 클래스 이름(표준 ImageNet 순서). 있으면 class_map.json 생성")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    shards = []
    for s in args.shards:
        shards.extend(sorted(glob.glob(s)))
    assert shards, "shard 파일을 찾지 못함"

    buckets = defaultdict(list)  # label -> [(path, bytes)]
    for sp in shards:
        pf = pq.ParquetFile(sp)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["image", "label"]).to_pydict()
            for im, lb in zip(t["image"], t["label"]):
                buckets[lb].append((im["path"], im["bytes"]))
        print(f"[read] {sp} -> 누적 클래스 {len(buckets)}")

    labels = sorted(buckets.keys())[: args.max_classes]
    tr_dir = os.path.join(args.out, "train")
    va_dir = os.path.join(args.out, "val")
    os.makedirs(tr_dir, exist_ok=True)
    os.makedirs(va_dir, exist_ok=True)

    n_tr = n_va = 0
    for lb in tqdm(labels, desc="classes"):
        items = buckets[lb]
        rng.shuffle(items)
        nval = min(args.val_per_class, max(0, len(items) - 1)) if args.val_per_class > 0 else 0
        train_items, val_items = items[nval: nval + args.per_class], items[:nval]

        ctr = os.path.join(tr_dir, str(lb)); os.makedirs(ctr, exist_ok=True)
        cva = os.path.join(va_dir, str(lb)); os.makedirs(cva, exist_ok=True)
        for i, (_, b) in enumerate(train_items):
            _save(b, os.path.join(ctr, f"{i:05d}.jpg"), args.max_side); n_tr += 1
        for i, (_, b) in enumerate(val_items):
            _save(b, os.path.join(cva, f"{i:05d}.jpg"), args.max_side); n_va += 1

    # 클래스 인덱스 → 사람이 읽는 이름 (있으면)
    names = None
    if os.path.exists(args.classes_file):
        with open(args.classes_file) as f:
            names = [l.strip() for l in f if l.strip()]
    class_map = {str(lb): (names[lb] if names and lb < len(names) else f"class_{lb}") for lb in labels}
    with open(os.path.join(args.out, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)

    print(f"[done] classes={len(labels)}  train={n_tr}  val={n_va}")
    print(f"[map ] {len(class_map)} 클래스 이름 -> {os.path.join(args.out, 'class_map.json')}")
    print(f"[out ] {args.out}")


def _save(b: bytes, path: str, max_side: int):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    im.save(path, quality=95)


if __name__ == "__main__":
    main()
