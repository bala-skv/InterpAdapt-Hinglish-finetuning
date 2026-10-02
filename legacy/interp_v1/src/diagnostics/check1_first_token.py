import sys, torch, torch.nn.functional as F
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "stage1"))

from model_loading import load_base_model, LoadConfig
from stage1_sanity_checks import _make_ids, _SOURCE_PROMPT, _SOURCE_TARGET, _TARGET_PROMPT, _TARGET_TARGET, _contrast, _recovery
from patching import capture_activations, OProjPatchAll

def _score_first_token(model, input_ids: torch.Tensor, prompt_len: int) -> float:
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
    lp = F.log_softmax(logits[0, prompt_len - 1], dim=-1)
    tgt = input_ids[0, prompt_len]
    return lp[tgt].item()

def main():
    device = torch.device("cuda", 0)
    cfg = LoadConfig(precision="fp16")
    model, tok = load_base_model(cfg)
    model.eval()
    
    print(f"Target neela tokens: {tok.encode('neela')}")
    print(f"Target blue tokens: {tok.encode('blue')}")

    src_ids, src_plen = _make_ids(tok, _SOURCE_PROMPT, _SOURCE_TARGET, device)
    tgt_ids, tgt_plen = _make_ids(tok, _TARGET_PROMPT, _TARGET_TARGET, device)
    src_patch_pos = src_plen - 1
    tgt_patch_pos = tgt_plen - 1

    m_clean = _contrast(
        _score_first_token(model, src_ids, src_plen),
        _score_first_token(model, *_make_ids(tok, _SOURCE_PROMPT, _TARGET_TARGET, device)),
    )
    m_corrupted = _contrast(
        _score_first_token(model, *_make_ids(tok, _TARGET_PROMPT, _SOURCE_TARGET, device)),
        _score_first_token(model, tgt_ids, tgt_plen),
    )

    stored = capture_activations(model, src_ids, src_patch_pos)

    with OProjPatchAll(model, tgt_patch_pos, stored):
        en_ids_native, en_plen_n = _make_ids(tok, _TARGET_PROMPT, _SOURCE_TARGET, device)
        en_ids_foreign, en_plen_f = _make_ids(tok, _TARGET_PROMPT, _TARGET_TARGET, device)
        logits_native  = model(input_ids=en_ids_native,  use_cache=False).logits
        logits_foreign = model(input_ids=en_ids_foreign, use_cache=False).logits

    lp_n = F.log_softmax(logits_native[0, en_plen_n - 1], dim=-1)[en_ids_native[0, en_plen_n]].item()
    lp_f = F.log_softmax(logits_foreign[0, en_plen_f - 1], dim=-1)[en_ids_foreign[0, en_plen_f]].item()

    m_patched = _contrast(lp_n, lp_f)
    print(f"\nFirst token recovery: {_recovery(m_clean, m_corrupted, m_patched):.6f}")

if __name__ == "__main__":
    main()