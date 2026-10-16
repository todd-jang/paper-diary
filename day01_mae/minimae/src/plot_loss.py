"""plot_loss.py — metrics CSV에서 사전학습 loss 곡선 이미지 생성 (선택 도구)"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/metrics_tiny.csv")
    ap.add_argument("--out", default="runs/loss_curve.png")
    a = ap.parse_args()
    df = pd.read_csv(a.metrics)
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["loss"], marker="o", ms=3, color="#1f77b4")
    plt.xlabel("epoch")
    plt.ylabel("masked-patch MSE")
    plt.title("mini-MAE pretraining loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
