#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 07 보충 — RT-DETR 파이프라인 통째 데모
===========================================

BACKBONE → EFFICIENT HYBRID ENCODER → UNCERTAINTY-MINIMAL QUERY SELECTION → DECODER(+AUX HEADS)
   └ 그 위: 헝가리아 세트 손실(매칭 + ∅ 하향 10 + L1/GIoU) → end-to-end 학습 → AP@0.5

원문 대응:
  · AIFI : S5 단일 스케일 self-attention (RT-DETR §4.2, D_S5 ablation)
  · CCFF : top-down FPN + bottom-up PAN 융합 경로 (fusion block: 1x1 conv + Rep-style block + add)
  · 쿼리 선택: top-K(분류 점수) + content/position 쿼리 (§4.3)
  · uncertainty-minimal: U = relu(score - quality) 를 손실에 추가 (식 2·3의 1차원 프록시)
  · 디코더: 6층, 각 층에 보조 예측 헤드(aux loss) → 학습된 모든 층에서 추론 가능 (RT Table 5)

합성 데이터: 128x128 RGB 이미지에 랜덤 직사각형 1~4개(3 클래스) → 정규화 cxcywh 박스.
실행: .venv/bin/python day07_detr/detr_pipeline_demo.py
"""
import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cpu")

IMG = 64
NUM_CLASSES = 3
D_MODEL = 64
NUM_QUERIES = 30
DEC_LAYERS = 6
BATCH = 8
EPOCHS = 80
LR = 1e-3
LAMBDA_U = 1.0
BOX_W, BOX_GIOU = 5.0, 2.0
EOS_COEF = 0.1
COST_W = (2.0, 5.0, 2.0)
ANCHOR_SIZE = 0.25
SEED = 20261020


# ---------------------------------------------------------------------------
# 합성 데이터
# ---------------------------------------------------------------------------
def make_dataset(n: int, seed: int) -> Tuple[torch.Tensor, List[dict]]:
    rng = random.Random(seed)
    images, targets = [], []
    for _ in range(n):
        canvas = np.zeros((IMG, IMG, 3), dtype=np.float32)
        boxes, labels = [], []
        for _ in range(rng.randint(1, 4)):
            cls = rng.randrange(NUM_CLASSES)
            w = rng.uniform(0.18, 0.38)
            h = rng.uniform(0.18, 0.38)
            cx = rng.uniform(w / 2, 1 - w / 2)
            cy = rng.uniform(h / 2, 1 - h / 2)
            color = np.array([[220, 40, 40], [40, 200, 60], [50, 90, 230]][cls], dtype=np.float32) / 255.0
            x0, y0 = int((cx - w / 2) * IMG), int((cy - h / 2) * IMG)
            x1, y1 = int((cx + w / 2) * IMG), int((cy + h / 2) * IMG)
            canvas[y0:y1, x0:x1] = color
            boxes.append([cx, cy, w, h])
            labels.append(cls)
        images.append(torch.from_numpy(canvas).permute(2, 0, 1))
        targets.append({
            "labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        })
    return torch.stack(images), targets


# ---------------------------------------------------------------------------
# 기하 유틸
# ---------------------------------------------------------------------------
def boxes_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    return torch.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2,
                        b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)


def iou_elementwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax = boxes_to_xyxy(a)
    bx = boxes_to_xyxy(b)
    ix1 = torch.maximum(ax[..., 0], bx[..., 0])
    iy1 = torch.maximum(ax[..., 1], bx[..., 1])
    ix2 = torch.minimum(ax[..., 2], bx[..., 2])
    iy2 = torch.minimum(ax[..., 3], bx[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_a = (ax[..., 2] - ax[..., 0]) * (ax[..., 3] - ax[..., 1])
    area_b = (bx[..., 2] - bx[..., 0]) * (bx[..., 3] - bx[..., 1])
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-9)


def giou_loss_elem(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """같은 길이의 두 박스 집합에 대한 원소별 GIoU 손실 (매칭된 쌍 계산용)."""
    ax = boxes_to_xyxy(b1)
    bx = boxes_to_xyxy(b2)
    ix1 = torch.maximum(ax[..., 0], bx[..., 0])
    iy1 = torch.maximum(ax[..., 1], bx[..., 1])
    ix2 = torch.minimum(ax[..., 2], bx[..., 2])
    iy2 = torch.minimum(ax[..., 3], bx[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_a = (ax[..., 2] - ax[..., 0]) * (ax[..., 3] - ax[..., 1])
    area_b = (bx[..., 2] - bx[..., 0]) * (bx[..., 3] - bx[..., 1])
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=1e-9)
    cx1 = torch.minimum(ax[..., 0], bx[..., 0])
    cy1 = torch.minimum(ax[..., 1], bx[..., 1])
    cx2 = torch.maximum(ax[..., 2], bx[..., 2])
    cy2 = torch.maximum(ax[..., 3], bx[..., 3])
    enclose = (cx2 - cx1) * (cy2 - cy1)
    return 1 - (iou - (enclose - union) / enclose.clamp(min=1e-9))


def iou_pairwise(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """쌍별 IoU: (N,4) x (M,4) → (N,M)."""
    a = boxes_to_xyxy(b1).unsqueeze(1)
    b = boxes_to_xyxy(b2).unsqueeze(0)
    ix1 = torch.maximum(a[..., 0], b[..., 0])
    iy1 = torch.maximum(a[..., 1], b[..., 1])
    ix2 = torch.minimum(a[..., 2], b[..., 2])
    iy2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-9)


def giou_pairwise(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """쌍별 GIoU 손실: (N,4) x (M,4) → (N,M). 매칭 비용 행렬용."""
    a = boxes_to_xyxy(b1).unsqueeze(1)
    b = boxes_to_xyxy(b2).unsqueeze(0)
    ix1 = torch.maximum(a[..., 0], b[..., 0])
    iy1 = torch.maximum(a[..., 1], b[..., 1])
    ix2 = torch.minimum(a[..., 2], b[..., 2])
    iy2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=1e-9)
    cx1 = torch.minimum(a[..., 0], b[..., 0])
    cy1 = torch.minimum(a[..., 1], b[..., 1])
    cx2 = torch.maximum(a[..., 2], b[..., 2])
    cy2 = torch.maximum(a[..., 3], b[..., 3])
    enclose = (cx2 - cx1) * (cy2 - cy1)
    return 1 - (iou - (enclose - union) / enclose.clamp(min=1e-9))


def iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    ax2, ay2 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    bx2, by2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / np.maximum(union, 1e-9)


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.s3 = nn.Sequential(nn.Conv2d(32, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.s4 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.s5 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        s3 = self.s3(x)
        s4 = self.s4(s3)
        s5 = self.s5(s4)
        return [s3, s4, s5]


class FusionBlock(nn.Module):
    """CCFF fusion block: 1x1 conv 2갈래 + bottleneck + element-wise add."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.SiLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.SiLU())
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, groups=out_ch), nn.BatchNorm2d(out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.SiLU(),
        )
        self.conv3 = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.bottleneck(self.conv1(x))
        b = self.conv2(x)
        return self.conv3(a + b)


def pos_embed_2d(h: int, w: int, d: int) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                          torch.arange(w, dtype=torch.float32), indexing="ij")
    out = []
    for v in (y, x):
        omega = torch.arange(d // 4, dtype=torch.float32) / (d // 4)
        omega = 1.0 / (10000 ** omega)
        ang = v.flatten()[:, None] * omega[None]
        out += [ang.sin(), ang.cos()]
    return torch.cat(out, dim=1)[None]


class HybridEncoder(nn.Module):
    """AIFI(S5만 self-attention) + CCFF(top-down FPN / bottom-up PAN)."""

    def __init__(self, in_channels: List[int], d_model: int = D_MODEL) -> None:
        super().__init__()
        self.input_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, d_model, 1), nn.BatchNorm2d(d_model)) for c in in_channels]
        )
        self.aifi = nn.TransformerEncoderLayer(d_model, 4, 256, 0.1, batch_first=True)
        self.lateral = nn.ModuleList([nn.Conv2d(d_model, d_model, 1) for _ in in_channels[:-1]])
        self.fpn_blocks = nn.ModuleList(
            [FusionBlock(d_model * 2, d_model) for _ in range(len(in_channels) - 1)]
        )
        self.down = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d_model, d_model, 3, 2, 1), nn.BatchNorm2d(d_model))
            for _ in range(len(in_channels) - 1)
        ])
        self.pan_blocks = nn.ModuleList(
            [FusionBlock(d_model * 2, d_model) for _ in range(len(in_channels) - 1)]
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        proj = [p(f) for p, f in zip(self.input_proj, feats)]

        s5 = proj[-1]
        b, _, h, w = s5.shape
        flat = s5.flatten(2).permute(0, 2, 1)
        pos = pos_embed_2d(h, w, flat.shape[-1]).to(flat.device)
        flat = self.aifi(flat + pos)
        proj[-1] = flat.permute(0, 2, 1).reshape(b, -1, h, w)

        inner = [proj[-1]]
        for idx in range(len(proj) - 1, 0, -1):
            high = self.lateral[len(proj) - 1 - idx](inner[0])
            up = F.interpolate(high, scale_factor=2.0, mode="nearest")
            inner.insert(0, self.fpn_blocks[len(proj) - 1 - idx](torch.cat([up, proj[idx - 1]], dim=1)))

        outs = [inner[0]]
        for idx in range(len(proj) - 1):
            down = self.down[idx](outs[-1])
            outs.append(self.pan_blocks[idx](torch.cat([down, inner[idx + 1]], dim=1)))
        return outs


class RTDETR(nn.Module):
    def __init__(self, num_queries: int = NUM_QUERIES, dec_layers: int = DEC_LAYERS) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.encoder = HybridEncoder([32, 64, 128])
        self.enc_cls = nn.Linear(D_MODEL, NUM_CLASSES + 1)
        self.enc_reg = nn.Linear(D_MODEL, 4)
        self.enc_iou = nn.Linear(D_MODEL, 1)
        self.num_queries = num_queries
        self.dec_layers = dec_layers
        self.query_pos_head = nn.Sequential(nn.Linear(4, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, D_MODEL))
        self.dec_layer = nn.TransformerDecoderLayer(D_MODEL, 4, 256, 0.1, batch_first=True)
        self.cls_head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, NUM_CLASSES + 1))
        self.box_head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, 4))

    def forward(self, images: torch.Tensor) -> dict:
        feats = self.backbone(images)
        levels = self.encoder(feats)

        flat_levels, sizes, anchors = [], [], []
        for lv in levels:
            _, _, h, w = lv.shape
            sizes.append((h, w))
            flat_levels.append(lv.flatten(2).permute(0, 2, 1))
            ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                                    torch.arange(w, dtype=torch.float32), indexing="ij")
            anchors.append(torch.stack([(xs.flatten() + 0.5) / w, (ys.flatten() + 0.5) / h,
                                         torch.full((h * w,), ANCHOR_SIZE),
                                         torch.full((h * w,), ANCHOR_SIZE)], -1))
        memory = torch.cat(flat_levels, dim=1)
        anchor_grid = torch.cat(anchors, dim=0).to(memory.device)
        n_cand = memory.shape[1]

        enc_logits = self.enc_cls(memory)
        enc_boxes = torch.sigmoid(self.enc_reg(memory))
        quality = torch.sigmoid(self.enc_iou(memory)).squeeze(-1)

        probs = enc_logits.softmax(-1)
        score = probs[..., :NUM_CLASSES].max(-1).values
        topk = score.topk(self.num_queries, dim=1).indices
        gather_idx = topk.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        content = memory.gather(1, gather_idx)
        ref_boxes = enc_boxes.gather(1, topk.unsqueeze(-1).expand(-1, -1, 4))

        pos_query = self.query_pos_head(ref_boxes.clamp(1e-4, 1 - 1e-4).logit())
        tgt = content + pos_query

        mem_pos = torch.cat(
            [pos_embed_2d(h, w, D_MODEL).to(memory.device).expand(memory.shape[0], -1, -1)
             for (h, w) in sizes], dim=1
        )

        all_logits, all_boxes = [], []
        for _ in range(self.dec_layers):
            tgt = self.dec_layer(tgt, memory + mem_pos)
            all_logits.append(self.cls_head(tgt))
            all_boxes.append(torch.sigmoid(self.box_head(tgt)))

        return {
            "enc_logits": enc_logits,
            "enc_boxes": enc_boxes,
            "score": score,
            "quality": quality,
            "topk": topk,
            "n_cand": n_cand,
            "all_logits": all_logits,
            "all_boxes": all_boxes,
            "content": content,
            "ref_boxes": ref_boxes,
        }


# ---------------------------------------------------------------------------
# 손실
# ---------------------------------------------------------------------------
@torch.no_grad()
def match_one(pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
              tgt_labels: torch.Tensor, tgt_boxes: torch.Tensor):
    if tgt_labels.numel() == 0:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
    probs = pred_logits.softmax(-1)
    cost_class = -probs[:, tgt_labels]
    cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)
    cost_giou = -giou_pairwise(pred_boxes, tgt_boxes)
    C = COST_W[0] * cost_class + COST_W[1] * cost_bbox + COST_W[2] * cost_giou
    rows, cols = linear_sum_assignment(C.cpu().numpy())
    return torch.as_tensor(rows, dtype=torch.int64), torch.as_tensor(cols, dtype=torch.int64)


def quality_targets(enc_boxes: torch.Tensor, targets: List[dict]) -> torch.Tensor:
    """각 후보 위치의 예측 박스가 GT 중 best와 얼마나 겹치는지(IoU 타깃, 0~1)."""
    out = torch.zeros(enc_boxes.shape[:2], device=enc_boxes.device)
    for b, t in enumerate(targets):
        if len(t["boxes"]) == 0:
            continue
        iou = iou_pairwise(enc_boxes[b], t["boxes"].to(enc_boxes.device))
        out[b] = iou.max(dim=1).values
    return out


def layer_loss(pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
               targets: List[dict], device: torch.device) -> torch.Tensor:
    num_boxes = sum(len(t["labels"]) for t in targets)
    loss = torch.zeros((), device=device)
    for b, t in enumerate(targets):
        rows, cols = match_one(pred_logits[b], pred_boxes[b], t["labels"], t["boxes"])
        if len(rows) == 0:
            continue
        tgt_cls = torch.full((pred_logits.shape[1],), NUM_CLASSES, dtype=torch.int64, device=device)
        tgt_cls[rows] = t["labels"][cols]
        w = torch.ones(NUM_CLASSES + 1, device=device)
        w[-1] = EOS_COEF
        loss = loss + F.cross_entropy(pred_logits[b], tgt_cls, w)
        pb = pred_boxes[b][rows]
        tb = t["boxes"][cols]
        loss = loss + BOX_W * F.l1_loss(pb, tb) + BOX_GIOU * giou_loss_elem(pb, tb).mean()
    return loss / max(num_boxes, 1)


def encoder_head_loss(out: dict, targets: List[dict], device: torch.device) -> torch.Tensor:
    """선택된 top-K 인코더 피처의 분류/박스 헤드를 직접 감독.

    RT-DETR의 criterion이 enc_topk_logits / enc_topk_bboxes를 손실에 포함시키는 것과 같은 역할:
    이게 없으면 인코더의 분류 헤퍼가 학습되지 않아 top-K 선택 자체가 무의미해진다.
    """
    pred_logits = out["all_logits"][-1]
    pred_boxes = out["all_boxes"][-1]
    num_boxes = sum(len(t["labels"]) for t in targets)
    loss = torch.zeros((), device=device)
    w = torch.ones(NUM_CLASSES + 1, device=device)
    for b, t in enumerate(targets):
        rows, cols = match_one(pred_logits[b], pred_boxes[b], t["labels"], t["boxes"])
        if len(rows) == 0:
            continue
        enc_idx = out["topk"][b][rows]
        e_logits = out["enc_logits"][b][enc_idx]
        e_boxes = out["enc_boxes"][b][enc_idx]
        gtb = t["boxes"][cols]
        loss = loss + F.cross_entropy(e_logits, t["labels"][cols], w)
        loss = loss + BOX_W * F.l1_loss(e_boxes, gtb) + BOX_GIOU * giou_loss_elem(e_boxes, gtb).mean()

        # 매칭되지 않은 선택 슬롯은 ∅로 감독 (DETR 슬롯 규칙과 동일) — 이게 없으면
        # 인코더 점수가 붕괴해 uncertainty 항이 '점수 낮추기'로 회피한다
        all_idx = out["topk"][b]
        unmatched = torch.ones(all_idx.numel(), dtype=torch.bool)
        unmatched[rows] = False
        if unmatched.any():
            e_logits_u = out["enc_logits"][b][all_idx[unmatched]]
            tgt_u = torch.full((e_logits_u.shape[0],), NUM_CLASSES, dtype=torch.int64, device=device)
            w_u = torch.ones(NUM_CLASSES + 1, device=device)
            w_u[-1] = EOS_COEF
            loss = loss + F.cross_entropy(e_logits_u, tgt_u, w_u)
    return loss / max(num_boxes, 1)


def set_loss(out: dict, targets: List[dict], use_uncertainty: bool) -> Tuple[torch.Tensor, dict]:
    device = out["all_logits"][-1].device
    total = layer_loss(out["all_logits"][-1], out["all_boxes"][-1], targets, device)
    for i in range(len(out["all_logits"]) - 1):
        total = total + layer_loss(out["all_logits"][i], out["all_boxes"][i], targets, device)

    # IoU(품질) 헤드 감독: 후보 박스 vs best GT의 IoU를 예측 (RT-DETR Fig 6의 'IoU score')
    total = total + encoder_head_loss(out, targets, device)

    q_tgt = quality_targets(out["enc_boxes"], targets)
    total = total + F.binary_cross_entropy(out["quality"].clamp(1e-4, 1 - 1e-4), q_tgt)

    stats = {"u_mean": 0.0}
    if use_uncertainty:
        q = out["quality"].gather(1, out["topk"])
        s = out["score"].gather(1, out["topk"])
        u = (s - q).clamp(min=0)
        stats["u_mean"] = float(u.mean())
        total = total + LAMBDA_U * u.mean()
    return total, stats


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def ap_at_50(preds: List[Tuple[float, int, np.ndarray]],
             gts: List[Tuple[int, np.ndarray]], iou_thr: float = 0.5) -> float:
    n_gt = len(gts)
    if n_gt == 0 or not preds:
        return 0.0
    preds = sorted(preds, key=lambda p: -p[0])
    matched = [False] * n_gt
    tp: List[int] = []
    fp: List[int] = []
    for _, cls, box in preds:
        best_iou, best_j = 0.0, -1
        for j, (g_cls, g_box) in enumerate(gts):
            if g_cls != cls or matched[j]:
                continue
            iou = float(iou_np(box[None], g_box[None]).ravel()[0])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            matched[best_j] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def evaluate(model: RTDETR, images: torch.Tensor, targets: List[dict], depth: int = DEC_LAYERS) -> dict:
    model.eval()
    preds: List[Tuple[float, int, np.ndarray]] = []
    gts: List[Tuple[int, np.ndarray]] = []
    prop_cls = prop_both = prop_n = 0
    for i in range(len(images)):
        out = model(images[i:i + 1])
        s_sel = out["score"].gather(1, out["topk"])[0]
        prop_cls += int((s_sel > 0.5).sum())

        logits = out["all_logits"][depth - 1][0]
        boxes = out["all_boxes"][depth - 1][0]
        if depth == DEC_LAYERS and len(targets[i]["boxes"]) > 0:
            iou_q = iou_pairwise(boxes, targets[i]["boxes"].to(boxes.device)).max(dim=1).values
            cls_q = logits.softmax(-1)[:, :NUM_CLASSES].max(dim=1).values
            prop_both += int(((cls_q > 0.5) & (iou_q > 0.5)).sum())
            prop_n += cls_q.numel()
        probs = logits.softmax(-1)
        for q in range(probs.shape[0]):
            cls_scores = probs[q, :NUM_CLASSES]
            cls = int(cls_scores.argmax())
            score = float(cls_scores.max())
            if score < 0.05:
                continue
            preds.append((score, cls, boxes[q].numpy()))
    for t in targets:
        for lab, box in zip(t["labels"].tolist(), t["boxes"].numpy()):
            gts.append((int(lab), box))
    return {
        "ap": ap_at_50(preds, gts),
        "prop_cls": prop_cls / max(prop_n, 1),
        "prop_both": prop_both / max(prop_n, 1),
    }


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------
def run(mode: str, use_uncertainty: bool, images: torch.Tensor, targets: List[dict],
        val_images: torch.Tensor, val_targets: List[dict], log: List[dict]) -> dict:
    torch.manual_seed(SEED)
    model = RTDETR().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    print(f"\n=== [{mode}] epochs={EPOCHS}, batch={BATCH}, K={NUM_QUERIES}, dec_layers={DEC_LAYERS} ===")
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(images))
        tot, nb, u_acc = 0.0, 0, 0.0
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            out = model(images[idx])
            loss, stats = set_loss(out, [targets[j] for j in idx.tolist()], use_uncertainty)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            tot += float(loss)
            u_acc += stats["u_mean"]
            nb += 1
        sched.step()
        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            m = evaluate(model, val_images, val_targets)
            print(f"  ep{ep+1:3d} loss={tot/max(nb,1):.4f}  U={u_acc/max(nb,1):.3f}  "
                  f"AP@0.5={m['ap']:.3f}  Prop both={m['prop_both']:.3f}  ({time.time()-t0:.0f}s)")
            log.append({"mode": mode, "epoch": ep + 1, "loss": tot / max(nb, 1),
                        "u": u_acc / max(nb, 1), **m})

    final = evaluate(model, val_images, val_targets)
    depth_curve = {d: evaluate(model, val_images, val_targets, depth=d)["ap"] for d in (6, 5, 4, 3, 2, 1)}
    return {"mode": mode, "final": final, "depth_curve": depth_curve,
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6}


def trace_pipeline() -> dict:
    model = RTDETR().eval()
    x = torch.randn(2, 3, IMG, IMG)
    with torch.no_grad():
        feats = model.backbone(x)
        levels = model.encoder(feats)
        out = model(x)
    return {
        "input": tuple(x.shape),
        "backbone_S3_S4_S5": [tuple(f.shape) for f in feats],
        "AIFI_input_tokens(S5)": (feats[2].shape[0], feats[2].shape[2] * feats[2].shape[3], D_MODEL),
        "CCFF_out_levels": [tuple(l.shape) for l in levels],
        "candidates_N": int(out["n_cand"]),
        "selected_K": int(out["topk"].shape[1]),
        "content_query": tuple(out["content"].shape),
        "position_query": tuple(out["ref_boxes"].shape),
        "decoder_layers": DEC_LAYERS,
        "aux_head_outputs": len(out["all_logits"]) - 1,
        "pred_logits": tuple(out["all_logits"][-1].shape),
        "pred_boxes": tuple(out["all_boxes"][-1].shape),
    }


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print("=" * 76)
    print("Day 07 보충 — RT-DETR 파이프라인 통째 데모 (합성 데이터, CPU)")
    print("=" * 76)

    print("\n[STEP 0] 파이프라인 텐서 흐름")
    trace = trace_pipeline()
    for k, v in trace.items():
        print(f"  {k:26s}: {v}")

    print("\n[STEP 1] 합성 데이터")
    train_x, train_y = make_dataset(300, SEED)
    val_x, val_y = make_dataset(100, SEED + 1)
    print(f"  train {tuple(train_x.shape)} / val {tuple(val_x.shape)} (이미지당 1~4 직사각형, {NUM_CLASSES} 클래스)")

    log: List[dict] = []
    results = {
        "vanilla": run("vanilla (분류 점수 top-K)", False, train_x, train_y, val_x, val_y, log),
        "uncertainty": run("uncertainty-minimal (U 항 추가)", True, train_x, train_y, val_x, val_y, log),
    }

    print("\n" + "=" * 76)
    print("[RESULT] 두 방식 비교 (동일 시드 · 동일 데이터 · 동일 스케줄)")
    print("=" * 76)
    print(f"{'항목':<34}{'vanilla':>14}{'uncertainty':>16}")
    rows = [
        ("AP@0.5 (val)", lambda r: f"{r['final']['ap']:.3f}"),
        ("Prop cls (>0.5)", lambda r: f"{r['final']['prop_cls']:.3f}"),
        ("Prop both (cls & IoU >0.5)", lambda r: f"{r['final']['prop_both']:.3f}"),
        ("파라미터(M)", lambda r: f"{r['params_m']:.2f}"),
    ]
    for name, fmt in rows:
        print(f"{name:<34}{fmt(results['vanilla']):>14}{fmt(results['uncertainty']):>16}")

    print("\n  재학습 없이 디코더 깊이를 줄였을 때의 AP@0.5 (RT-DETR Table 5 대응):")
    print(f"{'사용 층':<34}{'vanilla':>14}{'uncertainty':>16}")
    for d in sorted(results["vanilla"]["depth_curve"], reverse=True):
        print(f"{str(d) + '층':<34}"
              f"{results['vanilla']['depth_curve'][d]:>14.3f}"
              f"{results['uncertainty']['depth_curve'][d]:>16.3f}")

    payload = {
        "config": {"IMG": IMG, "D_MODEL": D_MODEL, "NUM_QUERIES": NUM_QUERIES, "DEC_LAYERS": DEC_LAYERS,
                   "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR, "LAMBDA_U": LAMBDA_U,
                   "EOS_COEF": EOS_COEF, "COST_W": COST_W, "ANCHOR_SIZE": ANCHOR_SIZE, "SEED": SEED},
        "trace": trace,
        "results": {k: {"final": v["final"], "depth_curve": v["depth_curve"], "params_m": v["params_m"]}
                    for k, v in results.items()},
        "log": log,
    }
    path = os.path.join(OUT_DIR, "detr_pipeline_metrics.json")
    json.dump(payload, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {path}")


if __name__ == "__main__":
    main()
