"""
check1_diagnostic.py  -  Structural ceiling vs. bug diagnostic for Check 1

Imports _score/_score_from_logits DIRECTLY from stage1_sanity_checks.py so that
experiment A MUST reproduce the 0.8125 from the original sanity gate.
If A != 0.8125, there is a metric bug and we stop.
If A == 0.8125, then B (full residual splice) tells us whether 0.8125 is a
structural ceiling or a harness bug.

Interpretation:
  B -> ~1.0  : structural ceiling. 0.8125 is defensible.
  B ~ 0.8125 : genuine harness bug -- full residual splice should exceed o_proj.
"""
from __future__ import annotations
import sys, json, torch
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "bala"))
sys.path.insert(0, str(repo / "phase0"))

from model_loading import load_base_model, LoadConfig
from stage1.patching import OProjPatchAll, capture_activations, NUM_LAYERS

# Import scoring DIRECTLY from the sanity checks to guarantee metric consistency
sys.path.insert(0, str(repo / "stage1"))
from stage1_sanity_checks import (
    _score, _score_from_logits, _make_ids, _contrast, _recovery,
    _SOURCE_PROMPT, _SOURCE_TARGET, _TARGET_PROMPT, _TARGET_TARGET,
)


class FullResidualSplice:
    """Pre-hook every transformer layer: overwrite full hidden state at patch_pos."""
    def __init__(self, model, patch_pos, clean_states):
        self.model, self.patch_pos, self.clean_states = model, patch_pos, clean_states
        self.handles = []

    def _make_hook(self, li):
        src, pos = self.clean_states[li], self.patch_pos
        def _h(_m, args):
            h = args[0].clone()
            h[0, pos, :] = src.to(device=h.device, dtype=h.dtype)
            return (h,) + args[1:]
        return _h

    def __enter__(self):
        for i, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(self._make_hook(i)))
        return self

    def __exit__(self, *_):
        for h in self.handles: h.remove()
        self.handles.clear()
        return False


@torch.no_grad()
def capture_residual(model, ids, patch_pos):
    states = {}
    def _make_hook(li):
        def _h(_m, args):
            states[li] = args[0][0, patch_pos].detach().to(torch.float16).cpu()
        return _h
    hs = [layer.register_forward_pre_hook(_make_hook(i))
          for i, layer in enumerate(model.model.layers)]
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        for h in hs: h.remove()
    assert len(states) == NUM_LAYERS, f"Got {len(states)}, expected {NUM_LAYERS}"
    return states


def main():
    if not torch.cuda.is_available(): print("ERROR: CUDA required"); return 1
    device = torch.device("cuda", 0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Prompts: SRC={repr(_SOURCE_PROMPT[:30])}... TGT={repr(_TARGET_PROMPT[:30])}...\n")

    cfg = LoadConfig(precision="fp16")
    model, tok = load_base_model(cfg)
    model.eval()
    print("Loaded.\n")

    src_ids, src_plen = _make_ids(tok, _SOURCE_PROMPT, _SOURCE_TARGET, device)
    tgt_ids, tgt_plen = _make_ids(tok, _TARGET_PROMPT, _TARGET_TARGET, device)
    src_pos = src_plen - 1
    tgt_pos = tgt_plen - 1

    en_ids_n, en_plen_n = _make_ids(tok, _TARGET_PROMPT, _SOURCE_TARGET, device)
    en_ids_f, en_plen_f = _make_ids(tok, _TARGET_PROMPT, _TARGET_TARGET, device)

    print(f"src_pos={src_pos}  tgt_pos={tgt_pos}")

    # Baselines using EXACT same _score as sanity checks
    m_clean = _contrast(
        _score(model, src_ids, src_plen),
        _score(model, *_make_ids(tok, _SOURCE_PROMPT, _TARGET_TARGET, device)),
    )
    m_corrupted = _contrast(
        _score(model, en_ids_n, en_plen_n),
        _score(model, en_ids_f, en_plen_f),
    )
    print(f"m_clean={m_clean:.6f}  m_corrupted={m_corrupted:.6f}")
    print(f"denominator={m_clean - m_corrupted:.6f}\n")
    if abs(m_clean - m_corrupted) < 1e-6:
        print("ERROR: denominator ~0."); return 1

    def _patched_rec(lg_n, lg_f):
        m_p = _contrast(
            _score_from_logits(lg_n, en_ids_n, en_plen_n),
            _score_from_logits(lg_f, en_ids_f, en_plen_f),
        )
        return _recovery(m_clean, m_corrupted, m_p)

    # A: all-heads o_proj patch -- MUST reproduce ~0.8125
    print("[A] All-heads o_proj patch (expect ~0.8125) ...")
    stored = capture_activations(model, src_ids, src_pos)
    print(f"    Layers: {len(stored)} (expected {NUM_LAYERS})")
    with OProjPatchAll(model, tgt_pos, stored):
        la_n = model(input_ids=en_ids_n, use_cache=False).logits
        la_f = model(input_ids=en_ids_f, use_cache=False).logits
    rec_a = _patched_rec(la_n, la_f)
    print(f"    Recovery A = {rec_a:.6f}")
    if not (0.5 < rec_a < 1.1):
        print(f"    WARNING: A={rec_a:.4f} does not reproduce ~0.8125. Metric mismatch or harness bug.")

    # B: full residual splice
    print("\n[B] Full residual-stream splice ...")
    res = capture_residual(model, src_ids, src_pos)
    print(f"    Layers: {len(res)} (expected {NUM_LAYERS})")
    with FullResidualSplice(model, tgt_pos, res):
        lb_n = model(input_ids=en_ids_n, use_cache=False).logits
        lb_f = model(input_ids=en_ids_f, use_cache=False).logits
    rec_b = _patched_rec(lb_n, lb_f)
    print(f"    Recovery B = {rec_b:.6f}")

    gap = rec_b - rec_a
    print(f"\n=== RESULT ===")
    print(f"  A (o_proj only)   : {rec_a:.4f}")
    print(f"  B (full residual) : {rec_b:.4f}")
    print(f"  Gap (B-A)         : {gap:.4f}")

    if rec_b > 0.95:
        verdict = "STRUCTURAL CEILING: embedding/MLP leakage explains the ~0.19 gap. 0.8125 is defensible."
    elif gap > 0.10:
        verdict = "PARTIAL CEILING: some structural leakage, but gap not fully explained."
    else:
        verdict = "BUG: full-residual splice does not improve substantially. Not embedding leakage."
    print(f"  VERDICT: {verdict}")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/check1_diagnostic.json").write_text(
        json.dumps({"rec_A": rec_a, "rec_B": rec_b, "gap": gap,
                    "m_clean": m_clean, "m_corrupted": m_corrupted,
                    "verdict": verdict}, indent=2))
    print("  Saved -> runs/check1_diagnostic.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())