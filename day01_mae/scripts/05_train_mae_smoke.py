import sys
from pathlib import Path

MAE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAE_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import FrameMAEDataset
from data.transforms import train_tf
from data.masking import patchify, random_masking

CSV = "/Users/robotjang/spatial-avla-pipeline/exports/mae_smoke_meta.csv"
PATCH = 16
MASK_RATIO = 0.75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DummyMAE(nn.Module):
    """인코더/디코더 없이 shape만 맞추는 더미."""
    def __init__(self, patch_dim, dim=768):
        super().__init__()
        self.proj = nn.Linear(patch_dim, dim)
        self.out  = nn.Linear(dim, patch_dim)

    def forward(self, x_kept, ids_restore, N):
        h = self.proj(x_kept)                       # (B, N_keep, 768)
        B, N_keep, D = h.shape
        # 마스크 토큰 채워 넣기 (학습 가능한 파라미터 대신 0으로)
        mask_tokens = torch.zeros(B, N - N_keep, D, device=h.device)
        h_full = torch.cat([h, mask_tokens], dim=1)
        # 원래 순서로 복원
        h_full = torch.gather(
            h_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )
        return self.out(h_full)                     # (B, N, patch_dim)


def main():
    ds = FrameMAEDataset(CSV, transform=train_tf)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    print(f"[data] len={len(ds)} batches={len(dl)} device={DEVICE}")

    patch_dim = PATCH * PATCH * 3   # 768
    model = DummyMAE(patch_dim=patch_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    model.train()
    for step, imgs in enumerate(dl):
        imgs = imgs.to(DEVICE)
        patches = patchify(imgs, PATCH)                  # (B,196,768)
        x_kept, ids_keep, ids_shuffle = random_masking(patches, MASK_RATIO)

        # ids_restore 계산 (ids_shuffle의 역순)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        pred = model(x_kept, ids_restore, N=patches.shape[1])

        # 손실: 마스킹된 패치에만
        B, N, D = patches.shape
        len_keep = x_kept.shape[1]
        mask = torch.ones(B, N, device=DEVICE)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)        # (B, N)
        mask = mask.unsqueeze(-1).expand(-1, -1, D)      # (B, N, D)

        loss = crit(pred * mask, patches * mask)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 5 == 0:
            print(f"[step {step:03d}] loss = {loss.item():.4f}")

    print("[done] 1 epoch smoke finished.")


if __name__ == "__main__":
    main()