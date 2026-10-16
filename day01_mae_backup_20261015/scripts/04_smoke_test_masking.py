import sys
from pathlib import Path

# day01_mae/ 를 sys.path에 추가
MAE_ROOT = Path(__file__).resolve().parent.parent   # .../day01_mae
sys.path.insert(0, str(MAE_ROOT))

import torch
from data.masking import patchify, random_masking


def main():
    # 더미 이미지: (8, 3, 224, 224)
    imgs = torch.rand(8, 3, 224, 224)

    # 1) patchify
    patches = patchify(imgs, patch_size=16)   # (B, 196, 768)
    print(f"[patchify] patches = {patches.shape}")

    # 2) random masking (75%)
    x_kept, ids_keep, ids_shuffle = random_masking(patches, mask_ratio=0.75)
    print(f"[masking] x_kept  = {x_kept.shape}")
    print(f"[masking] keep ratio = {x_kept.shape[1] / patches.shape[1]:.3f}")
    print(f"[masking] ids_keep[:1, :10] = {ids_keep[0, :10].tolist()}")

    # 3) 마스킹 비율 변화
    for r in [0.5, 0.75, 0.9]:
        xk, _, _ = random_masking(patches, mask_ratio=r)
        print(f"[ratio={r}] kept tokens = {xk.shape[1]}")


if __name__ == "__main__":
    main()