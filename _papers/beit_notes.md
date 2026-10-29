---
layout: default
title: "BEiT & SimMIM Deep Dive"
date: 2026-10-29
---

# BEiT & SimMIM Notes

## BEiT (Bidirectional Encoder representation from Image Transformers)

### Core Idea
- Pre-training method for Vision Transformers inspired by BERT
- Uses **masked image modeling (MIM)** with a discrete visual tokenizer (dVAE)
- Randomly masks image patches → predicts the visual tokens of the masked patches

### Architecture
1. **Image Tokenizer (dVAE)**: Converts image patches into discrete visual tokens
2. **Backbone ViT**: Standard Vision Transformer that takes masked patches as input
3. **Pre-training Objective**: Cross-entropy loss on predicted vs. ground-truth visual tokens

### Key Differences from MAE
| Aspect | BEiT | MAE |
|--------|------|-----|
| Target | Discrete visual tokens (dVAE) | Raw pixel reconstruction |
| Tokenizer | Requires separate dVAE training | No tokenizer needed |
| Masking ratio | ~40% | ~75% |
| Decoder | Linear head | Lightweight decoder |

---

## SimMIM (Simple Framework for Masked Image Modeling)

### Core Idea
- Simplifies MIM by directly regressing raw pixel values
- No need for discrete tokenizer or complex decoder
- Uses a simple L1 loss on masked patches

### Architecture
1. **Random Masking**: Large mask patches (e.g., 32×32)
2. **Encoder**: Swin Transformer (or any backbone)
3. **Prediction Head**: Single linear layer
4. **Loss**: L1 regression on raw pixels of masked region

### Key Takeaways
- Simpler is better: raw pixel targets work surprisingly well
- Large mask patch sizes are beneficial
- Works well with hierarchical architectures (Swin Transformer)
