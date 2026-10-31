import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

path = Path("C:/Users/LENOVO/InterpAdapt-Hinglish-finetuning/runs/full_trace_results.json")
with open(path) as f:
    data = json.load(f)

# Flatten the 28x12 heatmaps into 336-element arrays
h1 = np.array(data["heatmaps"]["en_hi-latn"]).flatten()
h2 = np.array(data["heatmaps"]["hi-deva_hi-latn"]).flatten()
h3 = np.array(data["heatmaps"]["en_hi-deva"]).flatten()

print(f"Correlation en_hi-latn (Language) vs hi-deva_hi-latn (Script): {pearsonr(h1, h2)[0]:.4f}")
print(f"Correlation en_hi-latn (Language) vs en_hi-deva (Both): {pearsonr(h1, h3)[0]:.4f}")
print(f"Correlation hi-deva_hi-latn (Script) vs en_hi-deva (Both): {pearsonr(h2, h3)[0]:.4f}")