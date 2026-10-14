"""ViTMAE 인코더 feature 추출 + spatial-avla JSON GT 연결 스모크."""
import sys, json
from pathlib import Path

MAE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAE_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining, ViTMAEConfig

from data.dataset import FrameMAEDataset
from data.transforms import val_tf   # 평가용 (deterministic)

CSV = "/Users/robotjang/paper-diary/day01_mae/exports/mae_smoke_meta.csv"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    # 1) ViTMAE 로드 (스모크 사이즈)
    config = ViTMAEConfig(
        image_size=224, patch_size=16,
        hidden_size=384, num_hidden_layers=6,
        num_attention_heads=6, intermediate_size=1536,
        mask_ratio=0.0,   # feature 추출 시엔 마스킹 X
    )
    model = ViTMAEForPreTraining(config).to(DEVICE).eval()

    # 2) 데이터 (val_tf: deterministic)
    ds = FrameMAEDataset(CSV, transform=val_tf)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    # 3) feature 추출
    with torch.no_grad():
        imgs = next(iter(dl)).to(DEVICE)
        # ViTMAE의 인코더만 사용
        outputs = model.vit(pixel_values=imgs, interpolate_pos_encoding=True)
        # outputs.last_hidden_state: (B, 1+N, D)  ← CLS + patch tokens
        feats = outputs.last_hidden_state
        print(f"[feature] shape = {feats.shape}")   # [8, 197, 384]
        print(f"[feature] CLS = {feats[:, 0, :5]}")  # CLS token 처음 5차원
        print(f"[feature] patch mean = {feats[:, 1:, :].mean().item():.4f}")

    # 4) spatial-avla JSON GT와 매핑 예시
    # JSON 파일 하나 열어서 GT 확인
    json_path = "/Users/robotjang/spatial-avla-pipeline/exports/TL_A/train/C04/clip_meta.json"   # 예: exports/TL_A/train/C01/frames/xxx.json
    if Path(json_path).exists():
        d = json.load(open(json_path))
        print(f"[gt] siren detected = {d['audio']['siren']['detected']}")
        print(f"[gt] doa gt_dir_deg = {d['audio']['doa']['gt_dir_deg']}")
        print(f"[gt] vision_anomaly_score = {d['event_trigger']['details']['vision_anomaly_score']}")


if __name__ == "__main__":
    main()