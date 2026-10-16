import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FrameMAEDataset(Dataset):
    """MAE 사전학습용 프레임 데이터셋 (라벨 없음)."""

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["path"]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img  # 라벨 없음