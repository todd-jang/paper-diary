import sys
from pathlib import Path
import pandas as pd
# 이 스크립트 파일 기준으로 mae/ 루트를 sys.path에 추가
MAE_ROOT = Path(__file__).resolve().parent.parent   # .../mae
sys.path.append(str(MAE_ROOT))

from data.dataset import FrameMAEDataset
from data.transforms import train_tf
from torch.utils.data import DataLoader

CSV = "/Users/robotjang/spatial-avla-pipeline/exports/mae_smoke_meta.csv"

def main():
    ds = FrameMAEDataset(CSV, transform=train_tf)
    print(f"[dataset] len = {len(ds)}")
    print(f"[dataset] sample shape = {ds[0].shape}")

    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    imgs = next(iter(dl))
    print(f"[dataloader] batch shape = {imgs.shape}")
    print(f"[dataloader] dtype = {imgs.dtype}")
    print(f"[dataloader] min/max = {imgs.min().item():.3f} / {imgs.max().item():.3f}")

if __name__ == "__main__":
    main()