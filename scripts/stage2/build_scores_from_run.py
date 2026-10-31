#!/usr/bin/env python3
"""Build the Stage-2 score files from a finished concept-pipeline run.

    python scripts/stage2/build_scores_from_run.py \
        --run results/concepts/linguistic_v1_ada \
        --keep results/concepts/linguistic_v1_ada/orig_rules/keep.json \
        --name ling39

Writes (CPU only, no GPU):

    data/stage2_scores/<name>_heads.json   {"scores": [[28 x 12]], "_provenance": {...}}
    data/stage2_scores/<name>_mlp.json     {"scores": [28],        "_provenance": {...}}

Each score is the mean recovery over the kept concepts on one contrast (default
en_hi-latn, the contrast the pilot masks used; the SAIL task is Latin-script).
The head file has exactly the format ``train_cra_compare.py`` already reads.

Safeguards: the trace must be a v2-format trace; every kept concept must be in the
trace; with no --keep file all traced concepts are used, and the means are checked
against the trace's own summary so a stale or mismatched file cannot slip through.

Component: Stage 1 score export for Stage 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECT_LAYERS, EXPECT_HEADS = 28, 12


def mean_recovery(trace: dict, contrast: str, keep: set | None, shape: tuple) -> tuple[np.ndarray, list]:
    rows = [r for r in trace["per_concept"][contrast] if not r.get("skipped")]
    if keep is not None:
        missing = sorted(keep - {r["concept_id"] for r in rows})
        if missing:
            raise SystemExit(f"kept concepts missing from the trace: {missing}")
        rows = [r for r in rows if r["concept_id"] in keep]
    if not rows:
        raise SystemExit("no concepts selected")
    arr = np.array([np.array(r["recovery"], dtype=float).reshape(shape) for r in rows])
    return arr.mean(0), [r["concept_id"] for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="results/concepts/<run>")
    ap.add_argument("--keep", type=Path, default=None, help="JSON list of concept ids to use (default: all traced)")
    ap.add_argument("--name", required=True, help="output prefix, e.g. ling39")
    ap.add_argument("--contrast", default="en_hi-latn")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "stage2_scores")
    a = ap.parse_args()

    keep = set(json.loads(a.keep.read_text(encoding="utf-8"))) if a.keep else None
    heads_trace = json.loads((a.run / "heads_trace.json").read_text(encoding="utf-8"))
    mlp_trace = json.loads((a.run / "mlp_trace.json").read_text(encoding="utf-8"))
    for name, t in (("heads", heads_trace), ("mlp", mlp_trace)):
        if t.get("version") != 2:
            raise SystemExit(f"{name} trace is not version 2 (got {t.get('version')!r}); "
                             "scores from older traces have different magnitudes and must not be used")

    heads, ids_h = mean_recovery(heads_trace, a.contrast, keep, (EXPECT_LAYERS, EXPECT_HEADS))
    mlp, ids_m = mean_recovery(mlp_trace, a.contrast, keep, (EXPECT_LAYERS,))
    if ids_h != ids_m:
        raise SystemExit("heads and MLP traces cover different concepts")
    if keep is None:                       # cross-check against the trace's own summary
        ref = np.array(heads_trace["summary"][a.contrast]["heatmap_mean"])
        if not np.allclose(heads, ref, atol=1e-9):
            raise SystemExit("recomputed head means disagree with the trace summary")

    prov = {
        "run": str(a.run),
        "contrast": a.contrast,
        "n_concepts": len(ids_h),
        "concept_ids_sha256": hashlib.sha256(",".join(sorted(ids_h)).encode()).hexdigest()[:16],
        "keep_file": str(a.keep) if a.keep else None,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{a.name}_heads.json").write_text(
        json.dumps({"scores": heads.tolist(), "_provenance": prov}, indent=1), encoding="utf-8")
    (a.out_dir / f"{a.name}_mlp.json").write_text(
        json.dumps({"scores": mlp.tolist(), "_provenance": prov}, indent=1), encoding="utf-8")

    top_h = np.argsort(-heads.flatten())[:5]
    top_m = np.argsort(-mlp)[:5]
    print(f"{len(ids_h)} concepts | contrast {a.contrast}")
    print("top heads :", ", ".join(f"L{i // EXPECT_HEADS}H{i % EXPECT_HEADS} {heads.flatten()[i]:+.4f}" for i in top_h))
    print("top layers:", ", ".join(f"L{i} {mlp[i]:+.4f}" for i in top_m))
    print(f"heads with positive score: {(heads > 0).sum()} | rank-112 head score: {np.sort(heads.flatten())[::-1][111]:+.5f}"
          f" | rank-34 head score: {np.sort(heads.flatten())[::-1][33]:+.5f}")
    print(f"wrote {a.out_dir / (a.name + '_heads.json')} and {a.out_dir / (a.name + '_mlp.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
