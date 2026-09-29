"""Regenerate the Stage-1 per-head causal scores the CRA masks consume.

Source of truth is Daniel's **v2** trace (``interp/runs/full_trace_v2.json``,
``version: 2`` -- m_clean on the source run, one fp32 scoring path, mean
reduction). We take the Pure-Language, Latin-script contrast::

    summary["en_hi-latn"]["heatmap_mean"]      # shape [num_layers][num_q_heads]

one causal score per (layer, query-head), and write it to
``jawed/data/stage1_head_scores.json`` as ``{"scores": [[...], ...],
"_provenance": {...}}`` -- the exact shape/format
``scripts/train_cra_compare.py::load_layer_head_scores`` expects.

The earlier scores file was built from the v1 trace
(``interp/results/full_trace_results.json``); the head *ranking* is similar but
the magnitudes differ (v1 max ~0.105 vs v2 max ~0.025), so any CRA run against
the v1 file must be discarded. Run this to refresh, no GPU required::

    python jawed/scripts/build_stage1_scores.py

Owner: Jawed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Repo layout: this file is jawed/scripts/build_stage1_scores.py
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "interp" / "runs" / "full_trace_v2.json"
DST = REPO_ROOT / "jawed" / "data" / "stage1_head_scores.json"
CONTRAST = "en_hi-latn"          # Pure Language, Latin script (separates language from script)
FIELD = "heatmap_mean"
EXPECT_LAYERS = 28
EXPECT_HEADS = 12


def main() -> int:
    with open(SRC, encoding="utf-8") as fh:
        trace = json.load(fh)

    version = trace.get("version")
    summary = trace.get("summary", {})
    if CONTRAST not in summary:
        raise SystemExit(f"{SRC} summary has no {CONTRAST!r} (keys: {list(summary)})")
    heatmap = summary[CONTRAST][FIELD]

    n_layers = len(heatmap)
    n_heads = len(heatmap[0]) if heatmap else 0
    if (n_layers, n_heads) != (EXPECT_LAYERS, EXPECT_HEADS):
        raise SystemExit(
            f"unexpected shape [{n_layers}][{n_heads}], expected "
            f"[{EXPECT_LAYERS}][{EXPECT_HEADS}] -- refusing to write"
        )
    scores = [[float(v) for v in row] for row in heatmap]

    flat = [(v, li, hi) for li, row in enumerate(scores) for hi, v in enumerate(row)]
    flat.sort(reverse=True)
    top = [{"layer": li, "head": hi, "score": round(v, 6)} for v, li, hi in flat[:5]]

    payload = {
        "scores": scores,
        "_provenance": {
            "source": str(SRC.relative_to(REPO_ROOT)).replace("\\", "/"),
            "trace_version": version,
            "contrast": CONTRAST,
            "field": FIELD,
            "shape": [n_layers, n_heads],
            "score_min": min(v for v, _, _ in flat),
            "score_max": max(v for v, _, _ in flat),
            "top_heads": top,
            "regenerated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "Regenerated from the v2 fp32 trace; supersedes the v1 file.",
        },
    }

    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(f"wrote {DST.relative_to(REPO_ROOT)}  (v{version}, {CONTRAST}, shape [{n_layers}][{n_heads}])")
    print(f"  range [{payload['_provenance']['score_min']:.4f}, {payload['_provenance']['score_max']:.4f}]")
    print("  top heads: " + ", ".join(f"L{h['layer']}H{h['head']}={h['score']:.4f}" for h in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
