#!/usr/bin/env python3
"""
full_trace.py - Stage 1 Full Trace on 38 Concepts

Runs the full 28-layer x 12-head trace over all concepts in tracing_set_v1.json.
Averages the heatmaps across concepts for the 3 conditions:
1. en <-> hi_latn       (language contrast)
2. hi_deva <-> hi_latn  (script contrast)
3. en <-> hi_deva       (both)

Owner: Daniel
"""

from __future__ import annotations
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm

_STAGE1_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _STAGE1_DIR.parent
for _p in (_STAGE1_DIR, _REPO_ROOT, _REPO_ROOT / "phase0", _REPO_ROOT / "bala"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from patching import capture_activations, patched_logits, NUM_LAYERS, NUM_HEADS
from model_loading import load_base_model, LoadConfig
from sequence_scoring import score_sequence, contrast_metric, recovery

def get_conditions(item):
    """Returns the (source, target) kwargs for the 3 pairs."""
    return {
        "en_hi-latn": {
            "hi_prompt": item["hi_latn_prompt"], "hi_target": item["hi_latn_target"],
            "en_prompt": item["en_prompt"], "en_target": item["en_target"]
        },
        "hi-deva_hi-latn": {
            "hi_prompt": item["hi_deva_prompt"], "hi_target": item["hi_deva_target"],
            "en_prompt": item["hi_latn_prompt"], "en_target": item["hi_latn_target"]
        },
        "en_hi-deva": {
            "hi_prompt": item["hi_deva_prompt"], "hi_target": item["hi_deva_target"],
            "en_prompt": item["en_prompt"], "en_target": item["en_target"]
        }
    }

def run_trace_for_pair(model, tok, pair_kwargs) -> list[list[float]]:
    """Runs 28x12 patching for a single concept and pair."""
    # 1. Clean run (Source)
    hi_prompt = pair_kwargs["hi_prompt"]
    hi_target = pair_kwargs["hi_target"]
    en_prompt = pair_kwargs["en_prompt"]
    en_target = pair_kwargs["en_target"]

    base = contrast_metric(model, tok, hi_prompt, hi_target, en_prompt, en_target, reduction="mean")
    m_clean = base["m"]
    hi_patch_pos = base["hi_patch_position"]
    en_patch_pos = base["en_patch_position"]

    # 2. Corrupted run (Target)
    # The corrupted run scores the Target (English/Latn) prompt on both targets to get baseline contrast
    corrupted = contrast_metric(model, tok, en_prompt, hi_target, en_prompt, en_target, reduction="mean")
    m_corrupted = corrupted["m"]

    # 3. Capture activations from Clean (Source)
    src_ids = tok.encode(hi_prompt + hi_target, add_special_tokens=False, return_tensors="pt").to(model.device)
    stored = capture_activations(model, src_ids, hi_patch_pos)

    # 4. Patching Loop
    en_ids_n = tok.encode(en_prompt + hi_target, add_special_tokens=False, return_tensors="pt").to(model.device)
    en_ids_f = tok.encode(en_prompt + en_target, add_special_tokens=False, return_tensors="pt").to(model.device)
    
    # We need the prefix length for en_prompt to calculate score_sequence equivalent from logits
    # score_sequence uses n_prompt, which is the length of en_prompt tokens
    n_prompt = len(tok.encode(en_prompt, add_special_tokens=False))
    # Wait, BPE merge issues across prompt/target boundary might occur!
    # sequence_scoring.py handles this internally, but we're doing manual patched_logits.
    # We can reconstruct ScoredSequence by calling our own logprobs reduction
    
    heatmap = []
    for L in range(NUM_LAYERS):
        layer_rec = []
        for H in range(NUM_HEADS):
            lg_n = patched_logits(model, en_ids_n, en_patch_pos, L, H, stored)
            lg_f = patched_logits(model, en_ids_f, en_patch_pos, L, H, stored)
            
            # calculate logp from logits
            def calc_logp(logits, ids):
                prompt_ids = tok.encode(en_prompt, add_special_tokens=False)
                full_ids = ids[0].tolist()
                n_p = len(prompt_ids)
                if full_ids[:n_p] != prompt_ids:
                    n_p = 0
                    for x, y in zip(prompt_ids, full_ids):
                        if x != y: break
                        n_p += 1
                n_t = len(full_ids) - n_p
                lps = logits.log_softmax(-1)
                per_token = [lps[0, n_p + k - 1, full_ids[n_p + k]].item() for k in range(n_t)]
                return sum(per_token) / n_t
                
            m_p = calc_logp(lg_n, en_ids_n) - calc_logp(lg_f, en_ids_f)
            
            try:
                rec = recovery(m_p, m_clean, m_corrupted)
            except ValueError:
                rec = 0.0
            layer_rec.append(rec)
        heatmap.append(layer_rec)
    
    return heatmap

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model (fp16 trace precision) ...")
    cfg = LoadConfig(precision="fp16")
    model, tok = load_base_model(cfg)
    model.eval()
    
    data_file = _REPO_ROOT / "bala" / "data" / "tracing_set_v1.json"
    if not data_file.exists():
        print(f"Dataset not found: {data_file}")
        sys.exit(1)
        
    with open(data_file) as f:
        concepts = json.load(f)
        
    print(f"Loaded {len(concepts)} concepts.")
    
    # Aggregate heatmaps: sum them up, then divide by N
    results = {
        "en_hi-latn": [[0.0]*NUM_HEADS for _ in range(NUM_LAYERS)],
        "hi-deva_hi-latn": [[0.0]*NUM_HEADS for _ in range(NUM_LAYERS)],
        "en_hi-deva": [[0.0]*NUM_HEADS for _ in range(NUM_LAYERS)]
    }
    
    # Also save raw scores for correlation analysis
    correlations = {
        "en_hi-latn": [],
        "hi-deva_hi-latn": [],
        "en_hi-deva": []
    }
    
    valid_counts = {k: 0 for k in results}
    
    for item in tqdm(concepts, desc="Processing Concepts"):
        pairs = get_conditions(item)
        for cond_name, pair_kwargs in pairs.items():
            try:
                # check if clean/corrupted are indistinguishable
                hi_prompt = pair_kwargs["hi_prompt"]
                en_prompt = pair_kwargs["en_prompt"]
                hi_target = pair_kwargs["hi_target"]
                en_target = pair_kwargs["en_target"]
                
                c_clean = contrast_metric(model, tok, hi_prompt, hi_target, en_prompt, en_target, reduction="mean")
                c_corr = contrast_metric(model, tok, en_prompt, hi_target, en_prompt, en_target, reduction="mean")
                
                if abs(c_clean["m"] - c_corr["m"]) < 1e-8:
                    continue # Skip concept for this condition if no contrast
                
                # Trace
                hm = run_trace_for_pair(model, tok, pair_kwargs)
                
                # Accumulate
                for L in range(NUM_LAYERS):
                    for H in range(NUM_HEADS):
                        results[cond_name][L][H] += hm[L][H]
                        
                valid_counts[cond_name] += 1
                
                # Store raw contrast for correlation
                correlations[cond_name].append({
                    "concept_id": item["concept_id"],
                    "m_clean": c_clean["m"],
                    "m_corrupted": c_corr["m"]
                })
                
            except Exception as e:
                print(f"Error on {item['concept_id']} - {cond_name}: {e}")
                
    # Average the heatmaps
    for cond_name in results:
        N = valid_counts[cond_name]
        if N > 0:
            for L in range(NUM_LAYERS):
                for H in range(NUM_HEADS):
                    results[cond_name][L][H] /= N
                    
    payload = {
        "heatmaps": results,
        "correlations": correlations,
        "valid_counts": valid_counts
    }
    
    out_file = Path("runs/full_trace_results.json")
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"\nTrace complete! Results saved to {out_file}")

if __name__ == "__main__":
    main()
