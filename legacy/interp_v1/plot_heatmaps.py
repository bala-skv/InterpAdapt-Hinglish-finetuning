import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

try:
    with open('runs/full_trace_v2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error reading JSON: {e}")
    sys.exit(1)

summary = data.get('summary', {})
pairs = ['en_hi-latn', 'hi-deva_hi-latn', 'en_hi-deva']
titles = ['English <-> Hindi (Latin)', 'Hindi (Deva) <-> Hindi (Latin)', 'English <-> Hindi (Deva)']

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for idx, pair in enumerate(pairs):
    if pair not in summary:
        continue
    # heatmap_mean is a flat list of 336 or a list of 28 lists of 12
    # In full_trace_v2.py: stack.mean(0).tolist() -> 28x12
    heatmap = np.array(summary[pair]['heatmap_mean'])
    if heatmap.ndim == 1:
        heatmap = heatmap.reshape((28, 12))
        
    ax = axes[idx]
    sns.heatmap(heatmap, cmap='viridis', ax=ax, cbar_kws={'label': 'Recovery Fraction'})
    ax.set_title(titles[idx])
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Layer')
    ax.invert_yaxis() # Layer 0 at bottom

plt.tight_layout()
out_path = 'final_full_heatmaps.png'
plt.savefig(out_path, dpi=300)
print(f"Saved to {out_path}")
