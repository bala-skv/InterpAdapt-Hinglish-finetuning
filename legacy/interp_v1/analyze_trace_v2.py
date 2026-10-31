#!/usr/bin/env python3
"""
analyze_trace_v2.py - everything to do after full_trace_v2.py finishes.

No GPU needed. Reads runs/full_trace_v2.json and writes runs/analysis_v2/:

    summary.json              all numbers below, machine-readable
    heatmaps.png              mean heatmaps, colour scale fitted to the data
    histograms.png            distribution of the 336 head scores per pair
    top_heads_ci.png          top heads per pair with bootstrap 95% CIs
    scores_<pair>.npz         per-head scores for the Stage 2 mask

Steps
-----
1. Validity: n_used per pair, contrast means vs the validation run.
2. Structure: bootstrap CIs per head over concepts, leave-one-out influence
   of single concepts on the top heads, top-5 stability across resamples.
3. Correlations: Pearson, Spearman, Pearson without the top-2 heads, and a
   shuffled-within-layer null (keeps layer structure, destroys head identity).
4. Figures.
5. Mask export: raw and clipped-at-zero scores for the gating pair.

Usage:
    python stage1/analyze_trace_v2.py
    python stage1/analyze_trace_v2.py --trace runs/full_trace_v2.json --boot 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

L, H = 28, 12

# Mean contrasts from the three-condition validation run (mean reduction).
EXPECTED_CONTRAST = {"en_hi-latn": 13.27, "hi-deva_hi-latn": 6.17, "en_hi-deva": 16.20}
GATING_PAIR = "en_hi-latn"   # language contrast, script held constant (matches SAIL)


def _lh(i: int) -> str:
    return f"L{i // H}H{i % H}"


# ---------------------------------------------------------------------------
# 1. Validity
# ---------------------------------------------------------------------------

def check_validity(summary: dict, per_concept: dict) -> dict:
    out = {}
    print("\n=== 1. validity ===")
    for name, rows in per_concept.items():
        used = [r for r in rows if not r["skipped"]]
        c = float(np.mean([r["contrast"] for r in used])) if used else float("nan")
        exp = EXPECTED_CONTRAST.get(name)
        ok = exp is None or abs(c - exp) / exp < 0.15
        out[name] = {"n_used": len(used), "n_total": len(rows),
                     "contrast_mean": c, "expected": exp, "contrast_ok": ok}
        flag = "ok" if ok else "MISMATCH -- clean contrast may still be wrong"
        print(f"  {name:<18} n={len(used)}/{len(rows)}  contrast {c:.2f} "
              f"(expected ~{exp})  {flag}")
    return out


# ---------------------------------------------------------------------------
# 2. Structure
# ---------------------------------------------------------------------------

def bootstrap(stack: np.ndarray, B: int, rng) -> dict:
    """stack [N, L, H] -> per-head mean, 95% CI, top-5 frequency."""
    N = stack.shape[0]
    flat = stack.reshape(N, -1)
    means = np.empty((B, flat.shape[1]))
    top5 = np.zeros(flat.shape[1])
    for b in range(B):
        m = flat[rng.integers(0, N, N)].mean(0)
        means[b] = m
        top5[np.argsort(-m)[:5]] += 1
    return {
        "mean": flat.mean(0),
        "lo": np.percentile(means, 2.5, axis=0),
        "hi": np.percentile(means, 97.5, axis=0),
        "top5_freq": top5 / B,
    }


def signflip_max_threshold(stack: np.ndarray, n: int, rng, alpha: float = 0.05) -> float:
    """Family-wise threshold over all 336 heads.

    Null hypothesis: a head's per-concept recoveries are symmetric around zero.
    Flip each concept's whole grid by a random sign, record the MAX head mean,
    repeat. A head whose observed mean exceeds the (1-alpha) quantile of that
    max is significant with family-wise error rate alpha -- this corrects for
    testing 336 heads at once, which per-head CIs do not (about 8 heads clear
    'CI > 0' by chance alone).
    """
    N = stack.shape[0]
    flat = stack.reshape(N, -1)
    maxes = np.empty(n)
    for k in range(n):
        s = rng.choice([-1.0, 1.0], size=N)
        maxes[k] = (s[:, None] * flat).mean(0).max()
    return float(np.quantile(maxes, 1 - alpha))


def leave_one_out(stack: np.ndarray, heads: list[int], ids: list[str]) -> dict:
    """Largest single-concept influence on each head's mean."""
    N = stack.shape[0]
    flat = stack.reshape(N, -1)
    full = flat.mean(0)
    out = {}
    for h in heads:
        loo = np.array([(flat[:, h].sum() - flat[i, h]) / (N - 1) for i in range(N)])
        shift = full[h] - loo
        i = int(np.argmax(np.abs(shift)))
        out[_lh(h)] = {"mean": float(full[h]),
                       "max_shift": float(shift[i]),
                       "max_shift_concept": ids[i],
                       "max_shift_frac": float(abs(shift[i]) / abs(full[h]))
                       if full[h] else float("nan")}
    return out


def analyze_structure(per_concept: dict, B: int, rng) -> tuple[dict, dict]:
    print("\n=== 2. structure ===")
    results, stacks = {}, {}
    for name, rows in per_concept.items():
        used = [r for r in rows if not r["skipped"]]
        stack = np.array([r["recovery"] for r in used])       # [N, L, H]
        stacks[name] = stack
        ids = [r["concept_id"] for r in used]
        bs = bootstrap(stack, B, rng)
        thr = signflip_max_threshold(stack, B, rng)
        order = np.argsort(-bs["mean"])
        top10 = [int(i) for i in order[:10]]
        sig = [int(i) for i in np.where(bs["lo"] > 0)[0]]
        sig_sorted = sorted(sig, key=lambda i: -bs["mean"][i])
        fwer = [int(i) for i in np.where(bs["mean"] > thr)[0]]
        fwer_sorted = sorted(fwer, key=lambda i: -bs["mean"][i])
        bs["fwer_sig"] = bs["mean"] > thr

        loo = leave_one_out(stack, top10[:5], ids)

        cats = sorted({r["category"] for r in used})
        by_cat = {}
        for h in top10[:3]:
            by_cat[_lh(h)] = {
                c: float(np.mean([stack[k].flatten()[h]
                                  for k, r in enumerate(used) if r["category"] == c]))
                for c in cats
            }

        flat_mean = bs["mean"]
        results[name] = {
            "n": len(used),
            "score_mean": float(flat_mean.mean()),
            "score_std": float(flat_mean.std()),
            "score_max": float(flat_mean.max()),
            "n_heads_ci_above_zero": len(sig),
            "heads_ci_above_zero": [_lh(i) for i in sig_sorted],
            "fwer_threshold": thr,
            "n_heads_fwer_sig": len(fwer),
            "heads_fwer_sig": [_lh(i) for i in fwer_sorted],
            "expected_false_ci_above_zero": round(0.025 * L * H, 1),
            "top10": [{"head": _lh(i), "mean": float(bs["mean"][i]),
                       "ci": [float(bs["lo"][i]), float(bs["hi"][i])],
                       "top5_freq": float(bs["top5_freq"][i])} for i in top10],
            "leave_one_out": loo,
            "top3_by_category": by_cat,
            "_bs": bs,
        }

        print(f"\n  {name}  (n={len(used)})")
        print(f"    heads with 95% CI > 0: {len(sig)} / {L * H}  "
              f"(~{0.025 * L * H:.0f} expected by chance -- descriptive only)")
        print(f"    FWER-significant heads (sign-flip max, alpha .05): {len(fwer)}"
              f"  threshold {thr:.4f}  {[_lh(i) for i in fwer_sorted[:8]]}")
        print(f"    {'head':<7} {'mean':>8} {'95% CI':>20} {'top5%':>7}")
        for t in results[name]["top10"][:5]:
            print(f"    {t['head']:<7} {t['mean']:>8.4f} "
                  f"[{t['ci'][0]:>8.4f}, {t['ci'][1]:>7.4f}] {t['top5_freq']:>6.0%}")
        for h, v in loo.items():
            if v["max_shift_frac"] > 0.25:
                print(f"    WARNING {h}: dropping {v['max_shift_concept']} moves its "
                      f"mean by {v['max_shift_frac']:.0%}")
    return results, stacks


# ---------------------------------------------------------------------------
# 3. Correlations
# ---------------------------------------------------------------------------

def shuffled_null(a: np.ndarray, b: np.ndarray, n: int, rng) -> np.ndarray:
    """Permute head scores within each layer of b; keeps layer structure."""
    out = np.empty(n)
    for k in range(n):
        bs = np.array([rng.permutation(row) for row in b])
        out[k] = pearsonr(a.flatten(), bs.flatten())[0]
    return out


def analyze_correlations(struct: dict, n_null: int, rng) -> dict:
    print("\n=== 3. correlations (on mean heatmaps) ===")
    names = list(struct)
    maps = {n: struct[n]["_bs"]["mean"].reshape(L, H) for n in names}
    top2 = np.argsort(-maps[GATING_PAIR].flatten())[:2] if GATING_PAIR in maps else []
    mask = np.ones(L * H, bool)
    mask[top2] = False
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = maps[names[i]], maps[names[j]]
            null = shuffled_null(a, b, n_null, rng)
            r = pearsonr(a.flatten(), b.flatten())[0]
            key = f"{names[i]} vs {names[j]}"
            out[key] = {
                "pearson": float(r),
                "spearman": float(spearmanr(a.flatten(), b.flatten())[0]),
                "pearson_wo_top2": float(pearsonr(a.flatten()[mask], b.flatten()[mask])[0]),
                "null_mean": float(null.mean()),
                "null_95": float(np.percentile(null, 95)),
                "p_vs_null": float((null >= r).mean()),
            }
            v = out[key]
            print(f"  {key}")
            print(f"    pearson {v['pearson']:.3f}  spearman {v['spearman']:.3f}  "
                  f"w/o top2 {v['pearson_wo_top2']:.3f}  "
                  f"null95 {v['null_95']:.3f}  p {v['p_vs_null']:.3f}")
    out["_top2_excluded"] = [_lh(int(i)) for i in top2]
    return out


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------

def make_figures(struct: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(struct)
    maps = {n: struct[n]["_bs"]["mean"].reshape(L, H) for n in names}
    vmax = max(np.percentile(np.abs(m), 99.5) for m in maps.values()) or 1e-3

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 7), squeeze=False)
    for ax, n in zip(axes[0], names):
        im = ax.imshow(maps[n], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto", origin="lower")
        ax.set_title(n)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(0, H, 2))
        ax.set_yticks(range(0, L, 3))
    fig.colorbar(im, ax=axes[0].tolist(), label="mean recovery", shrink=0.8)
    fig.savefig(outdir / "heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.2), squeeze=False)
    for ax, n in zip(axes[0], names):
        ax.hist(maps[n].flatten(), bins=50, color="0.35")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(n)
        ax.set_xlabel("mean recovery per head")
        ax.set_ylabel("heads")
    fig.tight_layout()
    fig.savefig(outdir / "histograms.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), squeeze=False)
    for ax, n in zip(axes[0], names):
        top = struct[n]["top10"]
        y = np.arange(len(top))[::-1]
        m = np.array([t["mean"] for t in top])
        lo = np.array([t["ci"][0] for t in top])
        hi = np.array([t["ci"][1] for t in top])
        ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="o", color="0.2", capsize=2)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([t["head"] for t in top])
        ax.set_title(n)
        ax.set_xlabel("mean recovery (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(outdir / "top_heads_ci.png", dpi=200)
    plt.close(fig)
    print(f"\n=== 4. figures -> {outdir} ===")


# ---------------------------------------------------------------------------
# 5. Mask export
# ---------------------------------------------------------------------------

def export_scores(struct: dict, outdir: Path) -> None:
    print("\n=== 5. mask export ===")
    for n, s in struct.items():
        bs = s["_bs"]
        np.savez(
            outdir / f"scores_{n}.npz",
            recovery=bs["mean"].reshape(L, H).astype(np.float32),
            recovery_clipped=np.clip(bs["mean"], 0, None).reshape(L, H).astype(np.float32),
            ci_lo=bs["lo"].reshape(L, H).astype(np.float32),
            ci_hi=bs["hi"].reshape(L, H).astype(np.float32),
            significant=(bs["lo"] > 0).reshape(L, H),
            fwer_significant=bs["fwer_sig"].reshape(L, H),
        )
    print(f"  wrote scores_<pair>.npz; gating pair is {GATING_PAIR}")
    g = struct.get(GATING_PAIR)
    if g is not None:
        n_sig = g["n_heads_fwer_sig"]
        verdict = ("GO: FWER-significant heads exist -- build the CRA mask from them"
                   if n_sig > 0 else
                   "NO-GO: no head survives family-wise correction -- defer CRA")
        print(f"  {GATING_PAIR}: {n_sig} FWER-significant heads "
              f"{g['heads_fwer_sig']} -> {verdict}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, default=Path("runs/full_trace_v2.json"))
    p.add_argument("--out", type=Path, default=Path("runs/analysis_v2"))
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--null", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = json.loads(args.trace.read_text(encoding="utf-8"))
    if data.get("version") != 2:
        raise SystemExit("expected a full_trace_v2.py output (version 2)")
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    validity = check_validity(data["summary"], data["per_concept"])
    struct, _ = analyze_structure(data["per_concept"], args.boot, rng)
    corr = analyze_correlations(struct, args.null, rng)
    make_figures(struct, args.out)
    export_scores(struct, args.out)

    clean = {n: {k: v for k, v in s.items() if k != "_bs"} for n, s in struct.items()}
    (args.out / "summary.json").write_text(json.dumps(
        {"validity": validity, "structure": clean, "correlations": corr,
         "bootstrap_B": args.boot, "null_n": args.null, "seed": args.seed},
        indent=2))
    print(f"\nsummary -> {args.out / 'summary.json'}")
    return 0 if all(v["contrast_ok"] for v in validity.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
