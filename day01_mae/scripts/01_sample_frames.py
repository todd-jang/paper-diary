"""TL_A의 450프레임/클립 → 클립당 N프레임 균등 샘플링."""
import glob
import os
import shutil

SRC = "/Users/robotjang/spatial-avla-pipeline/exports/TL_A"
DST = "/Users/robotjang/spatial-avla-pipeline/exports/TL_A_sampled"
N_PER_CLIP = 20  # 클립당 샘플 수


def main():
    os.makedirs(DST, exist_ok=True)
    for clip_dir in sorted(glob.glob(f"{SRC}/*/*/frames")):
        parts = clip_dir.split("/")
        split, clip = parts[-3], parts[-2]
        frames = sorted(glob.glob(f"{clip_dir}/*.jpg"))
        if len(frames) == 0:
            print(f"[skip] no frames: {clip_dir}")
            continue

        if len(frames) <= N_PER_CLIP:
            idx = list(range(len(frames)))
        else:
            idx = [round(i * (len(frames) - 1) / (N_PER_CLIP - 1))
                   for i in range(N_PER_CLIP)]

        out_dir = f"{DST}/{split}/{clip}"
        os.makedirs(out_dir, exist_ok=True)
        for i in idx:
            shutil.copy(frames[i], out_dir)
        print(f"[ok] {split}/{clip}: {len(idx)} frames → {out_dir}")


if __name__ == "__main__":
    main()