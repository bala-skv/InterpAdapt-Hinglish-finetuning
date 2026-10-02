#!/usr/bin/env python3
"""Compute correlations from the retained Stage 1 v2 trace."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "results" / "stage1" / "full_trace_v2.json"

with TRACE.open(encoding="utf-8") as fh:
    data = json.load(fh)

summary = data["summary"]
pairs = ["en_hi-latn", "hi-deva_hi-latn", "en_hi-deva"]

for i, a_name in enumerate(pairs):
    for b_name in pairs[i + 1:]:
        a = np.asarray(summary[a_name]["heatmap_mean"]).reshape(-1)
        b = np.asarray(summary[b_name]["heatmap_mean"]).reshape(-1)
        print(f"{a_name} vs {b_name}: pearson={pearsonr(a,b)[0]:.6f} spearman={spearmanr(a,b)[0]:.6f}")
