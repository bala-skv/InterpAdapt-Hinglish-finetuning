import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "stage1" / "full_trace_v2.json"
OUT = ROOT / "figures" / "stage1" / "final_full_heatmaps.png"

with RESULTS.open("r", encoding="utf-8") as f:
    data = json.load(f)

summary = data.get("summary", {})
pairs = ["en_hi-latn", "hi-deva_hi-latn", "en_hi-deva"]
titles = [
    "English <-> Hindi (Latin)",
    "Hindi (Deva) <-> Hindi (Latin)",
    "English <-> Hindi (Deva)",
]

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for idx, pair in enumerate(pairs):
    if pair not in summary:
        continue
    heatmap = np.array(summary[pair]["heatmap_mean"])
    if heatmap.ndim == 1:
        heatmap = heatmap.reshape((28, 12))

    ax = axes[idx]
    sns.heatmap(
        heatmap,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Recovery Fraction"},
    )
    ax.set_title(titles[idx])
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.invert_yaxis()

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300)
print(f"Saved to {OUT}")
