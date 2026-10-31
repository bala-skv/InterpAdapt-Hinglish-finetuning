#!/usr/bin/env python3
"""Summarise a Stage-2 study: one row per arm, with seed statistics and paired tests.

    python scripts/stage2/summarize_study.py --study ling_study

Reads runs/<study>/<arm>-s<seed>/{results.json,config.json,preds.json}; only runs with
a DONE marker are used. Writes results/stage2/<study>/summary.json and summary.md
(CPU only). Nothing here trains or evaluates a model.

What it reports
  * per arm: seeds, mean +- sd of validation macro-F1 and accuracy, per-seed values,
    parameter accounting (legacy head budget and effective trainable parameters), GPU;
  * paired comparisons that matter for the hypotheses (matched budget, same seed):
      top-k vs random, top-k vs uniform  (heads at 112 and 34; MLP; heads+MLP);
    for each: per-seed macro-F1 differences, their mean, and a paired bootstrap over
    validation examples (the CI covers evaluation noise; with 3 seeds the seed-to-seed
    spread is shown next to it, not hidden);
  * warnings: more than one GPU model among the runs, missing seeds, or an arm that
    did not finish.

Reading the result honestly: with 3 seeds, call a difference real only if the
per-seed differences all point the same way AND the bootstrap interval excludes 0.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = ["negative", "neutral", "positive"]

# (title, first arm, second arm): first minus second, same seed
COMPARISONS = [
    ("heads k=112: top-k minus random", "H-topk112", "H-random112"),
    ("heads k=112: top-k minus uniform LoRA", "H-topk112", "H-uniform"),
    ("heads k=34: top-k minus random", "H-topk34", "H-random34"),
    ("MLP layers k=3: top-k minus random", "M-topk3", "M-random3"),
    ("heads+MLP: top-k minus random", "HM-topk", "HM-random"),
    ("heads+MLP top-k minus heads-only top-k", "HM-topk", "H-topk112"),
    ("heads+MLP at 34 heads: top-k minus random", "HM-topk34", "HM-random34"),
    ("heads+MLP minus heads-only at 34 heads (top-k)", "HM-topk34", "H-topk34"),
]


def macro_f1(gold, pred) -> float:
    f1 = []
    for lab in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1) / len(f1)


def paired_bootstrap(gold, pred_a, pred_b, B=2000, seed=0):
    """CI of macro-F1(a) - macro-F1(b), resampling validation examples jointly."""
    rng = np.random.default_rng(seed)
    g = np.array(gold)
    a, b = np.array(pred_a), np.array(pred_b)
    n = len(g)
    diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        diffs[i] = macro_f1(g[idx], a[idx]) - macro_f1(g[idx], b[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def load_runs(study_dir: Path):
    runs = defaultdict(dict)       # arm -> seed -> dict
    unfinished = []
    for d in sorted(study_dir.iterdir()):
        m = re.fullmatch(r"(.+)-s(\d+)", d.name)
        if not (d.is_dir() and m):
            continue
        arm, seed = m.group(1), int(m.group(2))
        if not (d / "DONE").is_file():
            unfinished.append(d.name)
            continue
        res = json.loads((d / "results.json").read_text())
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").is_file() else {}
        preds = json.loads((d / "preds.json").read_text()) if (d / "preds.json").is_file() else None
        wid = (d / "wandb_run_id").read_text().strip() if (d / "wandb_run_id").is_file() else None
        runs[arm][seed] = {"res": res, "cfg": cfg, "preds": preds, "wandb_id": wid, "dir": str(d)}
    return runs, unfinished


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default="ling_study")
    ap.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "stage2")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    study_dir = a.runs_root / a.study
    if not study_dir.is_dir():
        raise SystemExit(f"no such study folder: {study_dir}")
    runs, unfinished = load_runs(study_dir)
    if not runs:
        raise SystemExit("no finished runs (no DONE markers) yet")

    out = {"study": a.study, "arms": {}, "comparisons": [], "warnings": []}
    gpus = {r["res"].get("gpu") for arm in runs.values() for r in arm.values()}
    if len(gpus) > 1:
        out["warnings"].append(f"runs used different GPUs: {sorted(map(str, gpus))}; "
                               "fp16 numbers differ slightly between GPU models, compare with care")
    if unfinished:
        out["warnings"].append(f"unfinished runs (no DONE marker): {unfinished}")

    for arm, seeds in sorted(runs.items()):
        f1 = np.array([seeds[s]["res"]["cra_macro_f1"] for s in sorted(seeds)])
        acc = np.array([seeds[s]["res"]["cra_acc"] for s in sorted(seeds)])
        first = seeds[sorted(seeds)[0]]["res"]
        out["arms"][arm] = {
            "seeds": sorted(seeds), "n_seeds": len(seeds),
            "macro_f1_mean": float(f1.mean()), "macro_f1_sd": float(f1.std(ddof=1)) if len(f1) > 1 else None,
            "macro_f1_per_seed": f1.tolist(),
            "acc_mean": float(acc.mean()), "acc_per_seed": acc.tolist(),
            "base_macro_f1": first.get("base_macro_f1"),
            "adapted_budget": first.get("adapted_budget"), "effective_params": first.get("effective_params"),
            "active_heads": first.get("active_heads"), "mlp_wrappers": first.get("mlp_wrappers"),
            "gpu": first.get("gpu"), "n_eval": first.get("n_eval"),
            "wandb_ids": {s: seeds[s]["wandb_id"] for s in sorted(seeds)},
        }

    for title, x, y in COMPARISONS:
        if x not in runs or y not in runs:
            continue
        common = sorted(set(runs[x]) & set(runs[y]))
        if not common:
            continue
        per_seed = [runs[x][s]["res"]["cra_macro_f1"] - runs[y][s]["res"]["cra_macro_f1"] for s in common]
        ci = None
        if all(runs[x][s]["preds"] and runs[y][s]["preds"] for s in common):
            # pool the per-seed paired bootstraps: concatenate the seeds' examples (same gold each time)
            gold = sum((runs[x][s]["preds"]["gold"] for s in common), [])
            pa = sum((runs[x][s]["preds"]["cra"] for s in common), [])
            pb = sum((runs[y][s]["preds"]["cra"] for s in common), [])
            ci = paired_bootstrap(gold, pa, pb, a.boot)
        same_sign = all(d > 0 for d in per_seed) or all(d < 0 for d in per_seed)
        if len(common) < 3:
            verdict = ("single seed: indicative only (the interval covers evaluation noise, not "
                       "seed-to-seed variation)")
        elif same_sign and ci and (ci[0] > 0 or ci[1] < 0):
            verdict = "consistent direction and CI excludes 0"
        else:
            verdict = "not distinguishable (mixed signs, or CI includes 0)"
        out["comparisons"].append({
            "title": title, "a": x, "b": y, "seeds": common, "per_seed_diff": per_seed,
            "mean_diff": float(np.mean(per_seed)), "bootstrap_ci95": ci, "verdict": verdict})

    d = a.out_root / a.study
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    L = [f"# Stage 2 study `{a.study}`", ""]
    for w in out["warnings"]:
        L.append(f"> **warning:** {w}")
    L += ["", "| Arm | Seeds | Macro-F1 (mean ± sd) | Per seed | Accuracy | Head budget | Effective params | GPU |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, r in out["arms"].items():
        sd = "" if r["macro_f1_sd"] is None else f" ± {r['macro_f1_sd']:.3f}"
        L.append(f"| {arm} | {r['n_seeds']} | {r['macro_f1_mean']:.3f}{sd} | "
                 f"{', '.join(f'{v:.3f}' for v in r['macro_f1_per_seed'])} | {r['acc_mean']:.3f} | "
                 f"{r['adapted_budget']} | {r['effective_params']} | {r['gpu']} |")
    L += ["", "Zero-shot base macro-F1 (same evaluation): "
          + ", ".join(sorted({fmt(r['base_macro_f1']) for r in out['arms'].values()})), "",
          "## Paired comparisons (first minus second, same seed)", "",
          "| Comparison | Per-seed differences | Mean | Bootstrap 95% CI | Reading |", "|---|---|---|---|---|"]
    for c in out["comparisons"]:
        ci = "n/a" if c["bootstrap_ci95"] is None else f"[{c['bootstrap_ci95'][0]:+.3f}, {c['bootstrap_ci95'][1]:+.3f}]"
        L.append(f"| {c['title']} | {', '.join(f'{v:+.3f}' for v in c['per_seed_diff'])} | "
                 f"{c['mean_diff']:+.3f} | {ci} | {c['verdict']} |")
    (d / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {d / 'summary.json'} and {d / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
