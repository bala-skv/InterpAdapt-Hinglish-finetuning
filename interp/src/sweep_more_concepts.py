import json
import torch
from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "bala"))
sys.path.insert(0, str(repo_root / "phase0"))
from model_loading import load_base_model, LoadConfig
from sequence_scoring import contrast_metric, score_sequence
from stage1.patching import OProjPatchLayer, capture_activations, NUM_LAYERS
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = LoadConfig(precision="fp16")
    model, tok = load_base_model(cfg)
    model.eval()
    with open("bala/data/tracing_set_v1.json") as f:
        concepts = json.load(f)[:3]                        
    for item in concepts:
        print(f"\n--- Concept: {item['concept_id']} ---")
        hi_prompt = item["hi_deva_prompt"]
        hi_target = item["hi_deva_target"]
        en_prompt = item["en_prompt"]
        en_target = item["en_target"]
        c_clean = contrast_metric(model, tok, hi_prompt, hi_target, en_prompt, en_target, reduction="mean")
        c_corr = contrast_metric(model, tok, en_prompt, hi_target, en_prompt, en_target, reduction="mean")
        m_clean = c_clean["m"]
        m_corrupted = c_corr["m"]
        tgt_patch_pos = c_clean["en_patch_position"]
        src_patch_pos = c_clean["hi_patch_position"]
        src_ids = tok.encode(hi_prompt + hi_target, add_special_tokens=False, return_tensors="pt").to(device)
        en_ids_n = tok.encode(en_prompt + hi_target, add_special_tokens=False, return_tensors="pt").to(device)
        en_ids_f = tok.encode(en_prompt + en_target, add_special_tokens=False, return_tensors="pt").to(device)
        n_p = len(tok.encode(en_prompt, add_special_tokens=False))
        n_t_n = en_ids_n.shape[1] - n_p
        n_t_f = en_ids_f.shape[1] - n_p
        stored = capture_activations(model, src_ids, src_patch_pos)
        print(f"L=0..27 Recoveries:")
        for L in range(NUM_LAYERS):
            with OProjPatchLayer(model, tgt_patch_pos, L, stored):
                lg_n = model(input_ids=en_ids_n, use_cache=False).logits
                lg_f = model(input_ids=en_ids_f, use_cache=False).logits
            lps_n = lg_n.log_softmax(-1)
            per_tok_n = [lps_n[0, n_p + k - 1, en_ids_n[0, n_p + k]].item() for k in range(n_t_n)]
            m_n = sum(per_tok_n) / n_t_n
            m_f = score_sequence(model, tok, en_prompt, en_target, reduction="mean").logp
            m_patched = m_n - m_f
            try:
                rec = (m_patched - m_corrupted) / (m_clean - m_corrupted)
            except:
                rec = 0.0
            print(f"{rec:.4f}", end=", ")
        print()
if __name__ == "__main__":
    main()