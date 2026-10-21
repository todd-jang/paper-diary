# Poster copy (English, verified numbers only)

> Rules: (1) every number must be traceable to a paper table/section or to our own logged measurement;
> (2) label the source (paper / our measurement / simulation) in the caption;
> (3) compare only within one domain; (4) state the limitations too.

## Title
**NMS-Free Real-Time Detection: DETR -> RT-DETR**
Why moving duplicate suppression from post-processing into the loss buys accuracy and speed at once
*Carion et al., ECCV 2020, arXiv:2005.12872 · Lv et al., CVPR 2024, arXiv:2304.08069*

## 1. Problem - the NMS dilemma
Two thresholds move accuracy and time in opposite directions (paper Table 1, T4 TensorRT FP16):
- conf 0.001 -> 0.05: **AP 52.9 -> 51.2**, NMS time **2.36 -> 1.06 ms**
- IoU 0.5 -> 0.8: **AP 52.1 -> 52.8**, NMS time **2.24 -> 2.46 ms**
YOLO papers usually report model speed, excluding NMS time -> RT-DETR builds an end-to-end benchmark (§3.2).

## 2. Method - three moves
**(1) Duplicates -> loss (DETR §3.1).** Optimal bipartite matching assigns one prediction per GT, so
duplicates become structurally impossible. Matching cost uses probabilities, the loss uses log-likelihoods.
Box loss = scale-invariant GIoU + L1 (weights 2 / 5).

**(2) Compute -> design (RT-DETR §4.2).** The encoder was 49% of GFLOPs but 11% of AP (cited stat).
AIFI keeps attention only on S5 (400 tokens); CCFF fuses scales with convolutions.
Table 3: variant D_S5 cuts latency 35% and adds 0.4 AP.

**(3) Initialization -> quality (RT-DETR §4.3).** U = ||P - C|| (localization vs classification mismatch)
enters the loss, so the encoder learns features that agree in both: AP 47.9 -> 48.7,
Prop cls 0.35% -> 0.82%, Prop both 0.30% -> 0.67%. The *ranking* is unchanged (public code still uses
class score for top-K).

Auxiliary prediction heads make every decoder layer a usable predictor, so depth can be traded for
latency without retraining (§5.4).

## 3. Results
| | DETR (R50) | Faster R-CNN-FPN+ | RT-DETR (R50) |
|---|---|---|---|
| AP | 42.0 | 42.0 | **53.1** |
| AP_S / AP_L | 20.5 / 61.1 | 26.6 / 53.4 | 34.8 / 70.0 |
| Params | 41M | 42M | 42M |
| FPS | 28 | 26 | **108** (T4, TensorRT FP16, 640x640) |

- DETR matches Faster R-CNN at equal parameter count, but with **AP_S -5.5 / AP_L +7.8**.
- RT-DETR: **+2.2 AP and 21x faster** than DINO-Deformable (5 -> 108 FPS).
- **Conditional comparison**: 640x640 / 72 ep (RT-DETR) vs 800x1333 / 500 ep (DETR); absolute numbers
  are not directly comparable.

## 4. Our own experiments (synthetic data, CPU)
1. Latency order of encoder variants A-E reproduces the paper: A < D_S5 < E < B < D < C.
2. Attention cost ~ M^2: 8x tokens -> **42.3x** time (theory 64x). 8400 vs 400 tokens = **441x** gap.
3. NMS dilemma reproduced in both directions (conf up / IoU up).
4. Uncertainty alone has a degenerate optimum: s 0.90 -> **0.690**, q 0.20 -> **0.693**; with the
   classification loss, s 0.794 / q 0.793 (quality follows confidence).

*Limits: synthetic data, no training; only directions and orderings are verified, not AP or absolute latency.*

## 5. Limitations
- Small objects remain weak: AP_S 34.8 (-0.5 vs YOLOv8-L; R101 -0.9) - the cost of restricting global
  attention to S5 while keeping multi-scale features.
- 49% GFLOPs / 11% AP is a **second-hand statistic**, not our measurement.
- 138% / 120% are **relative counts**, not percentage-point gains.
- **Selection != objective**: uncertainty changes the loss, not the ranking.
- 2D COCO only: depth, BEV and multi-view remain untested (see the 3D track).

## 6. Takeaway
**Move duplicate removal into the loss, move compute into the design, move initialization into quality.**
