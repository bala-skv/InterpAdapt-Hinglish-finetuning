#!/usr/bin/env python3
"""
full_trace_v2.py - Stage 1 full trace, corrected.

Fixes over full_trace.py
------------------------
1. CLEAN CONTRAST. v1 computed
       m_clean = logp(hi_tgt | hi_prompt) - logp(en_tgt | en_prompt)
   which mixes two runs. The clean contrast is measured on the Hindi run:
       m_clean = logp(hi_tgt | hi_prompt) - logp(en_tgt | hi_prompt)
   v1 therefore divided by logp(hi_tgt|hi_prompt) - logp(hi_tgt|en_prompt)
   (e.g. river_flow on the script pair: 0.034 instead of ~6), inflating
   scores 3-6x on average and far more on small-denominator concepts.

2. ONE SCORING PATH. Clean, corrupted and patched contrasts all go through
   the same `_logp_from_logits`, computed in fp32. v1 scored clean/corrupted
   in fp32 (sequence_scoring) and patched in fp16 (calc_logp), so an
   unpatched head could show nonzero recovery from precision alone.
   Because every number shares one code path, patching nothing now gives
   recovery exactly 0 -- asserted per concept below.

3. PER-CONCEPT OUTPUT. Every concept's raw m_patched grid is saved, so
   recovery can be recomputed offline, histograms and bootstrap CIs are
   possible, and a bad concept can be dropped without a rerun.

4. ROBUST AGGREGATES. Mean, median and 10%-trimmed mean are all saved.
   A few concepts with small contrasts can dominate a plain mean.

Protocol (per concept, per pair)
--------------------------------
  source run  = "hi" side, target run = "en" side (the pair's second cond)
  m_clean     = logp(hi_t | hi_p) - logp(en_t | hi_p)       [source run]
  m_corrupted = logp(hi_t | en_p) - logp(en_t | en_p)       [target run]
  m_patched   = same as m_corrupted, one head patched from source run
  Recovery    = (m_patched - m_corrupted) / (m_clean - m_corrupted)

Component: Stage 1 v2 tracing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]

import sys
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from interp_adapt.tracing.patching import NUM_HEADS, NUM_LAYERS, capture_activations, patched_logits  # noqa: E402
from interp_adapt.tracing.model_loading import LoadConfig, load_base_model  # noqa: E402

# (source condition, target condition). Source = the side patched FROM.
PAIRS = {
    "en_hi-latn":      ("hi_latn", "en"),
    "hi-deva_hi-latn": ("hi_deva", "hi_latn"),
    "en_hi-deva":      ("hi_deva", "en"),
}

MIN_CONTRAST = 0.5


# ---------------------------------------------------------------------------
# Tokenization and scoring -- the ONE code path every number goes through
# ---------------------------------------------------------------------------

def _encode(tok, prompt: str, target: str):
    """Return (full_ids list, n_prompt, patch_pos).

    Tokenizes prompt+target as one string, as the model sees it. BPE may
    merge the prompt's trailing space into the target's first token, so
    n_prompt is the length of the common prefix, not len(encode(prompt)).
    """
    p_ids = tok.encode(prompt, add_special_tokens=False)
    f_ids = tok.encode(prompt + target, add_special_tokens=False)
    n = 0
    for a, b in zip(p_ids, f_ids):
        if a != b:
            break
        n += 1
    if n == 0 or n >= len(f_ids):
        raise ValueError(f"bad prompt/target split: {prompt!r} + {target!r}")
    return f_ids, n, n - 1


def _logp_from_logits(logits: torch.Tensor, full_ids: list[int], n_prompt: int) -> float:
    """Mean per-token teacher-forced log-prob of the target. fp32."""
    lp = logits[0].float().log_softmax(-1)
    n_t = len(full_ids) - n_prompt
    total = sum(lp[n_prompt + k - 1, full_ids[n_prompt + k]].item() for k in range(n_t))
    return total / n_t


@torch.no_grad()
def _run(model, full_ids, patch_pos=None, layer=None, head=None, stored=None):
    ids = torch.tensor([full_ids], device=model.device)
    return patched_logits(model, ids, patch_pos, layer, head, stored)


# ---------------------------------------------------------------------------
# One concept, one pair
# ---------------------------------------------------------------------------

def trace_pair(model, tok, item: dict, src: str, tgt: str) -> dict:
    hi_p, hi_t = item[f"{src}_prompt"], item[f"{src}_target"]
    en_p, en_t = item[f"{tgt}_prompt"], item[f"{tgt}_target"]

    # source (clean) run: both targets after the SOURCE prompt
    s_hi_ids, s_hi_n, s_patch = _encode(tok, hi_p, hi_t)
    s_en_ids, s_en_n, _ = _encode(tok, hi_p, en_t)
    m_clean = (_logp_from_logits(_run(model, s_hi_ids), s_hi_ids, s_hi_n)
               - _logp_from_logits(_run(model, s_en_ids), s_en_ids, s_en_n))

    # target (corrupted) run: both targets after the TARGET prompt
    t_hi_ids, t_hi_n, t_hi_patch = _encode(tok, en_p, hi_t)
    t_en_ids, t_en_n, t_en_patch = _encode(tok, en_p, en_t)
    m_corrupted = (_logp_from_logits(_run(model, t_hi_ids), t_hi_ids, t_hi_n)
                   - _logp_from_logits(_run(model, t_en_ids), t_en_ids, t_en_n))

    contrast = m_clean - m_corrupted
    out = {
        "concept_id": item["concept_id"],
        "category": item["category"],
        "m_clean": m_clean,
        "m_corrupted": m_corrupted,
        "contrast": contrast,
        "n_tokens_hi": len(s_hi_ids) - s_hi_n,
        "n_tokens_en": len(t_en_ids) - t_en_n,
        "skipped": False,
    }
    if contrast < MIN_CONTRAST:
        out["skipped"] = True
        return out

    # Capture from the source run at the source's last prompt token.
    stored = capture_activations(
        model, torch.tensor([s_hi_ids], device=model.device), s_patch
    )

    # In-path zero-heads check: an unpatched target run must reproduce
    # m_corrupted exactly, through the same code, or recovery is meaningless.
    m_unpatched = (
        _logp_from_logits(_run(model, t_hi_ids, t_hi_patch, None, None, None), t_hi_ids, t_hi_n)
        - _logp_from_logits(_run(model, t_en_ids, t_en_patch, None, None, None), t_en_ids, t_en_n)
    )
    if abs(m_unpatched - m_corrupted) > 1e-6:
        raise RuntimeError(
            f"{item['concept_id']} {src}->{tgt}: unpatched run gives "
            f"{m_unpatched} but m_corrupted is {m_corrupted}; scoring paths diverge"
        )

    m_patched = np.zeros((NUM_LAYERS, NUM_HEADS))
    for L in range(NUM_LAYERS):
        for H in range(NUM_HEADS):
            lp_hi = _logp_from_logits(
                _run(model, t_hi_ids, t_hi_patch, L, H, stored), t_hi_ids, t_hi_n)
            lp_en = _logp_from_logits(
                _run(model, t_en_ids, t_en_patch, L, H, stored), t_en_ids, t_en_n)
            m_patched[L, H] = lp_hi - lp_en

    out["m_patched"] = m_patched.tolist()
    out["recovery"] = ((m_patched - m_corrupted) / contrast).tolist()
    return out


def _trimmed_mean(stack: np.ndarray, frac: float = 0.1) -> np.ndarray:
    n = stack.shape[0]
    k = int(n * frac)
    s = np.sort(stack, axis=0)
    return s[k:n - k].mean(axis=0) if n - 2 * k > 0 else s.mean(axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=_REPO_ROOT / "data" / "tracing" / "tracing_set_v1.json")
    p.add_argument("--out", type=Path, default=_REPO_ROOT / "results" / "stage1" / "full_trace_v2.json")
    p.add_argument("--pairs", nargs="*", default=list(PAIRS))
    p.add_argument("--limit", type=int, default=None, help="first N concepts (smoke test)")
    args = p.parse_args()

    model, tok = load_base_model(LoadConfig(precision="fp16"))
    model.eval()
    print(f"device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")

    concepts = json.loads(args.data.read_text(encoding="utf-8"))
    if args.limit:
        concepts = concepts[: args.limit]
    print(f"{len(concepts)} concepts, pairs: {args.pairs}")

    per_concept: dict[str, list[dict]] = {k: [] for k in args.pairs}
    for item in tqdm(concepts, desc="concepts"):
        for name in args.pairs:
            src, tgt = PAIRS[name]
            per_concept[name].append(trace_pair(model, tok, item, src, tgt))

    summary = {}
    for name, rows in per_concept.items():
        used = [r for r in rows if not r["skipped"]]
        if not used:
            summary[name] = {"n_used": 0}
            continue
        stack = np.array([r["recovery"] for r in used])  # [N, 28, 12]
        summary[name] = {
            "n_used": len(used),
            "n_skipped": len(rows) - len(used),
            "skipped": [r["concept_id"] for r in rows if r["skipped"]],
            "contrast_mean": float(np.mean([r["contrast"] for r in used])),
            "heatmap_mean": stack.mean(0).tolist(),
            "heatmap_median": np.median(stack, 0).tolist(),
            "heatmap_trimmed": _trimmed_mean(stack).tolist(),
            "heatmap_std": stack.std(0).tolist(),
        }
        top = np.argsort(-stack.mean(0).flatten())[:5]
        print(f"\n{name}: n={len(used)}  contrast mean {summary[name]['contrast_mean']:.2f}")
        print("  top5 (L,H,mean,median):",
              [(int(i // NUM_HEADS), int(i % NUM_HEADS),
                round(float(stack.mean(0).flatten()[i]), 4),
                round(float(np.median(stack, 0).flatten()[i]), 4)) for i in top])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "version": 2,
        "protocol": "m_clean on source run; one fp32 scoring path; mean reduction",
        "summary": summary,
        "per_concept": per_concept,
    }, ensure_ascii=False))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
