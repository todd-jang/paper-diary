---
layout: default
title: "Day 01: Pre-training Loss Analysis"
paper_id: beit-simmim
date: 2026-10-29
---

# Experiment Notes

Ran initial mask ratio tests at 40% vs 60% on visual token prediction:

- **Target representation**: dVAE discrete tokens (vocabulary size 8192)
- **Masking strategy**: Block-wise masking
- **Observation**: 40% masking yielded faster initial convergence, but 60% forced the model to learn richer contextual semantics across distant patches.
