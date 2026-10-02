import sys, torch
from pathlib import Path
repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "phase0"))
from model_loading import load_base_model, LoadConfig
cfg = LoadConfig(precision="fp16")
model, tok = load_base_model(cfg)
print(f"neela tokens: {tok.encode('neela')}")
print(f"blue tokens: {tok.encode('blue')}")