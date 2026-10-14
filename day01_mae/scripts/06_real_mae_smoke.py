"""진짜 MAE (transformers ViTMAE) 1 epoch 스모크."""
import sys
from pathlib import Path

MAE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAE_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining, ViTMAEConfig

from data.dataset import FrameMAEDataset
from data.transforms import train_tf

CSV = "/Users/robotjang/paper-diary/day01_mae/exports/mae_smoke_meta.csv"
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def main():
    # 1) ViT-MAE 설정 (Base, 작게 시작)
    config = ViTMAEConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=384,       # Base는 768, 스모크는 384로 축소
        num_hidden_layers=6,   # Base는 12
        num_attention_heads=6,
        intermediate_size=1536,
        mask_ratio=0.75,
    )
    from transformers import ViTMAEForPreTraining
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    print(f"[model] params = {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 2) 데이터
    ds = FrameMAEDataset(CSV, transform=train_tf)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    print(f"[data] len={len(ds)} batches={len(dl)} device={DEVICE}")

    # 3) 학습
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for step, imgs in enumerate(dl):
        imgs = imgs.to(DEVICE)
        out = model(pixel_values=imgs)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[step {step:03d}] loss = {loss.item():.4f}")
        if step >= 5:  # 6 step만
            break

    print("[done] real MAE smoke finished.")


if __name__ == "__main__":
    main()