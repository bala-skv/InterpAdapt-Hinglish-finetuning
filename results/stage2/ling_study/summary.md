# Stage 2 study `ling_study`


| Arm | Seeds | Macro-F1 (mean ± sd) | Per seed | Accuracy | Head budget | Effective params | GPU |
|---|---|---|---|---|---|---|---|
| H-random112 | 3 | 0.594 ± 0.003 | 0.597, 0.595, 0.591 | 0.621 | 229376 | 868352 | NVIDIA GeForce RTX 2080 Ti |
| H-random34 | 3 | 0.583 ± 0.004 | 0.588, 0.581, 0.580 | 0.613 | 69632 | 512000 | NVIDIA GeForce RTX 2080 Ti |
| H-soft | 1 | 0.568 | 0.568 | 0.601 | 350208 | 1038336 | NVIDIA GeForce RTX 2080 Ti |
| H-topk112 | 3 | 0.592 ± 0.007 | 0.596, 0.595, 0.584 | 0.620 | 229376 | 917504 | NVIDIA GeForce RTX 2080 Ti |
| H-topk34 | 3 | 0.576 ± 0.001 | 0.576, 0.576, 0.578 | 0.607 | 69632 | 512000 | NVIDIA GeForce RTX 2080 Ti |
| H-uniform | 3 | 0.604 ± 0.005 | 0.610, 0.602, 0.600 | 0.629 | 688128 | 1376256 | NVIDIA GeForce RTX 2080 Ti |
| HM-random | 3 | 0.592 ± 0.005 | 0.598, 0.590, 0.588 | 0.619 | 229376 | 1120256 | NVIDIA GeForce RTX 2080 Ti |
| HM-random34 | 1 | 0.588 | 0.588 | 0.617 | 69632 | 763904 | NVIDIA GeForce RTX 2080 Ti |
| HM-topk | 3 | 0.590 ± 0.004 | 0.594, 0.590, 0.586 | 0.616 | 229376 | 1169408 | NVIDIA GeForce RTX 2080 Ti |
| HM-topk34 | 1 | 0.584 | 0.584 | 0.611 | 69632 | 763904 | NVIDIA GeForce RTX 2080 Ti |
| M-random3 | 3 | 0.575 ± 0.001 | 0.574, 0.576, 0.574 | 0.603 | 0 | 251904 | NVIDIA GeForce RTX 2080 Ti |
| M-soft | 1 | 0.585 | 0.585 | 0.613 | 0 | 1427456 | NVIDIA GeForce RTX 2080 Ti |
| M-topk3 | 3 | 0.573 ± 0.003 | 0.572, 0.577, 0.571 | 0.604 | 0 | 251904 | NVIDIA GeForce RTX 2080 Ti |
| M-uniform | 1 | 0.606 | 0.606 | 0.629 | 0 | 2351104 | NVIDIA GeForce RTX 2080 Ti |

Zero-shot base macro-F1 (same evaluation): 0.319

## Paired comparisons (first minus second, same seed)

| Comparison | Per-seed differences | Mean | Bootstrap 95% CI | Reading |
|---|---|---|---|---|
| heads k=112: top-k minus random | -0.000, +0.000, -0.007 | -0.002 | [-0.009, +0.004] | not distinguishable (mixed signs, or CI includes 0) |
| heads k=112: top-k minus uniform LoRA | -0.013, -0.006, -0.016 | -0.012 | [-0.020, -0.004] | consistent direction and CI excludes 0 |
| heads k=34: top-k minus random | -0.012, -0.005, -0.002 | -0.007 | [-0.014, +0.000] | not distinguishable (mixed signs, or CI includes 0) |
| MLP layers k=3: top-k minus random | -0.002, +0.001, -0.002 | -0.001 | [-0.011, +0.009] | not distinguishable (mixed signs, or CI includes 0) |
| heads+MLP: top-k minus random | -0.005, +0.000, -0.002 | -0.002 | [-0.009, +0.005] | not distinguishable (mixed signs, or CI includes 0) |
| heads+MLP top-k minus heads-only top-k | -0.003, -0.005, +0.002 | -0.002 | [-0.007, +0.003] | not distinguishable (mixed signs, or CI includes 0) |
| heads+MLP at 34 heads: top-k minus random | -0.004 | -0.004 | [-0.018, +0.010] | single seed: indicative only (the interval covers evaluation noise, not seed-to-seed variation) |
| heads+MLP minus heads-only at 34 heads (top-k) | +0.009 | +0.009 | [-0.004, +0.021] | single seed: indicative only (the interval covers evaluation noise, not seed-to-seed variation) |
