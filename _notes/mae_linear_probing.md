---
layout: default
title: "Day 01: MAE Linear Probing & Reconstruction Check"
paper_id: mae
date: 2026-10-16
---

# MAE Linear Probing & Reconstruction Check

- Evaluated ViT-Base encoder on ImageNet-1K subset (14 classes / 16k images)
- Pre-training reconstruction loss dropped steadily from 1.0289 to 0.5387 over 20 epochs
- Linear probing achieved 39.76% top-1 accuracy on MPS without fine-tuning encoder weights
