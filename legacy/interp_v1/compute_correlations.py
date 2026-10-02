import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

data = json.load(open('runs/full_trace_v2.json', encoding='utf-8'))
summary = data['summary']

heatmaps = {
    'lang': np.array(summary['en_hi-latn']['heatmap_mean']).flatten(),
    'script': np.array(summary['hi-deva_hi-latn']['heatmap_mean']).flatten(),
    'both': np.array(summary['en_hi-deva']['heatmap_mean']).flatten()
}

print("Pearson:")
print("Language vs Script:", pearsonr(heatmaps['lang'], heatmaps['script'])[0])
print("Language vs Both:", pearsonr(heatmaps['lang'], heatmaps['both'])[0])
print("Script vs Both:", pearsonr(heatmaps['script'], heatmaps['both'])[0])

print("\nSpearman:")
print("Language vs Script:", spearmanr(heatmaps['lang'], heatmaps['script'])[0])
print("Language vs Both:", spearmanr(heatmaps['lang'], heatmaps['both'])[0])
print("Script vs Both:", spearmanr(heatmaps['script'], heatmaps['both'])[0])
