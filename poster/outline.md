# Poster layout (A0 portrait, 841 x 1189 mm)

One poster = one claim:
> "Moving duplicate removal from post-processing into the learning objective buys real-time speed
> without paying accuracy."

## Grid (3 columns)
```
+---------------------------+---------------------------+---------------------------+
| HEADER (full width)       |  NMS-Free Real-Time ...   |  metric cards 42.0/53.1/108|
+---------------------------+---------------------------+---------------------------+
| 1 The NMS dilemma        | 2 Three moves             | 3 Accuracy and speed       |
|   Fig 1 NMS cost (ours)  |   flow diagram (native)    |   results table (colored)   |
|                           |   bullets                 |   Fig 2 AP vs FPS (paper)  |
+---------------------------+---------------------------+---------------------------+
| 4 Our own experiments     | 5 Limitations             | 6 Study trail & next       |
|   2x2 figure gallery      |   bullets                 |   Day timeline chips       |
+---------------------------+---------------------------+---------------------------+
| FOOTER: one-line takeaway + arXiv links + colored chips                    |
+---------------------------------------------------------------------------+
```

## Block messages
1. **Problem** - NMS trades accuracy against time through two thresholds (Table 1).
2. **Method** - three moves: duplicates->loss, compute->design, init->quality.
3. **Results** - 42.0 AP at 41M (Faster R-CNN parity) -> 53.1 AP / 108 FPS; AP_S -5.5 vs AP_L +7.8.
4. **Our experiments** - latency order reproduced, M^2 scaling, NMS dilemma, uncertainty degeneracy.
5. **Limitations** - AP_S gap, second-hand stat, relative counts, selection!=objective, 2D only.
6. **Trail** - 8 study days, 16 docs, 4 demos, 22 charts, 0 citation errors; next GLIP+SAM.

## Print checklist
- [ ] Paper size: A0 (841x1189 mm) or A1 (594x841 mm) per the venue
- [ ] Fonts embedded in the PDF (the builder uses Helvetica/Arial families)
- [ ] Body text >= 24 pt when printed; title >= 72 pt
- [ ] Figures >= 300 dpi (poster figures are rendered at 150 dpi, A0 is 2x A1 so effective 300 dpi)
- [ ] arXiv links + code links present in the footer
