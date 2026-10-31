#!/usr/bin/env python3
"""
plot_entangle.py - figures for the head-interaction experiments (CPU only).

Reads the outputs of entangle.py and writes three PNGs (and PDFs) to
figures/stage1/entangle/:

  curve_<pair>.png      Joint patching curve: top-k joint recovery, the sum of
                        the same heads' single recoveries (additive prediction)
                        and random k heads. Bands are 95% bootstrap intervals
                        over the concepts.
  necessity_<pair>.png  Necessity against sufficiency for all 336 heads.
  pairs_<pair>.png      Largest pair interactions with 95% intervals, the
                        family-wise threshold, and the random-pair range.

Usage (from the repository root):
  python scripts/stage1/plot_entangle.py
  python scripts/stage1/plot_entangle.py --pair en_hi-latn --boot 2000

Component: Stage 1 head-interaction analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.ticker  # noqa: F401
import matplotlib.text  # noqa: F401

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import entangle as E  # noqa: E402  (same directory; also puts src/ on sys.path)

FIG_DIR = E.REPO_ROOT / "figures" / "stage1" / "entangle"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
C_JOINT, C_SUM, C_RAND, C_SIG, C_OTHER = "#1b6ca8", "#d1495b", "#888888", "#d1495b", "#9aa5b1"


def band(stack: np.ndarray, B: int, rng):
    """stack [N concepts, P points] -> mean, 2.5% and 97.5% bootstrap bounds."""
    return E.bootstrap(stack, B, rng)


def save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path)
    print(f"wrote {FIG_DIR / (stem + '.png')}")
    plt.close(fig)


def plot_curve(curve: dict, singles: dict, pair: str, B: int, rng) -> None:
    rows = [r for r in curve["per_concept"] if not r.get("skipped")]
    ks = curve["ks"]
    order = curve["top_order"]
    joint = np.array([[r["top"][str(k)] for k in ks] for r in rows])
    additive = np.array([[singles[r["concept_id"]][order[:k] if k <= len(order)
                                                    else list(range(E.N_HEADS))].sum()
                          for k in ks] for r in rows])
    rand_ks = [i for i, k in enumerate(ks) if str(k) in rows[0]["random"]]
    rand = np.array([[np.mean(r["random"][str(ks[i])]) for i in rand_ks] for r in rows])

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for data, idx, color, label, ls in (
            (joint, range(len(ks)), C_JOINT, "top-$k$ heads, patched together", "-"),
            (additive, range(len(ks)), C_SUM, "sum of the same heads' single recoveries", "--"),
            (rand, rand_ks, C_RAND, "$k$ random heads, patched together", ":")):
        m, lo, hi = band(data, B, rng)
        x = [ks[i] for i in idx]
        ax.plot(x, m, ls, color=color, marker="o", ms=3, lw=1.5, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.axhline(1.0, color="black", lw=0.5, alpha=0.4)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=7)
    ax.set_xlabel("number of heads patched, $k$")
    ax.set_ylabel(f"mean recovery over {len(rows)} concepts")
    ax.set_title("Joint patching vs. additive prediction")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    save(fig, f"curve_{pair}")


def _ha(a: float) -> str:
    c = np.cos(a)
    return "left" if c > 0.3 else "right" if c < -0.3 else "center"


def _va(a: float) -> str:
    sn = np.sin(a)
    return "bottom" if sn > 0.3 else "top" if sn < -0.3 else "center"


def place_labels(ax, xs, ys, names, fontsize=6.5):
    """Label points with thin leader lines. For each point, try offsets in
    rings of growing radius, outward from the origin first, and keep the first
    whose text box overlaps no other label, no labelled marker, and stays
    inside the axes."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(renderer)
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    cx, cy = ax.transData.transform((0.0, 0.0))
    radii = (6, 12, 20, 30)
    angles = np.deg2rad(np.arange(0, 360, 30))
    placed = []
    for i in np.argsort(-np.asarray(ys)):                     # top to bottom
        outward = np.arctan2(pts[i, 1] - cy, pts[i, 0] - cx)
        cands = sorted(((r, a) for r in radii for a in angles),
                       key=lambda ra: (ra[0], abs(np.angle(np.exp(1j * (ra[1] - outward))))))
        chosen = None
        for r, a in cands:
            ann = ax.annotate(names[i], xy=(xs[i], ys[i]),
                              xytext=(r * np.cos(a), r * np.sin(a)),
                              textcoords="offset points", fontsize=fontsize,
                              ha=_ha(a), va=_va(a),
                              arrowprops=dict(arrowstyle="-", lw=0.4, color="0.35",
                                              shrinkA=0, shrinkB=2))
            ann.update_positions(renderer)      # offsets are only resolved on update
            bb = matplotlib.text.Text.get_window_extent(ann, renderer).expanded(1.08, 1.25)
            ok = (bb.x0 >= axbox.x0 and bb.x1 <= axbox.x1 and
                  bb.y0 >= axbox.y0 and bb.y1 <= axbox.y1)
            ok = ok and not any(bb.overlaps(q) for q in placed)
            ok = ok and not any(bb.contains(*pts[j]) for j in range(len(names)) if j != i)
            if ok:
                chosen = (ann, bb)
                break
            ann.remove()
        if chosen is None:                                      # fall back to the last candidate
            r, a = cands[-1]
            ann = ax.annotate(names[i], xy=(xs[i], ys[i]),
                              xytext=(r * np.cos(a), r * np.sin(a)),
                              textcoords="offset points", fontsize=fontsize,
                              ha=_ha(a), va=_va(a),
                              arrowprops=dict(arrowstyle="-", lw=0.4, color="0.35"))
            ann.update_positions(renderer)
            bb = matplotlib.text.Text.get_window_extent(ann, renderer)
            chosen = (ann, bb)
        placed.append(chosen[1])


def plot_necessity(noise: dict, v2: dict, pair: str) -> None:
    rows = [r for r in noise["per_concept"] if not r.get("skipped")]
    nec = np.mean([np.array(r["necessity"]).flatten() for r in rows], axis=0)
    suff = np.array(v2["summary"][pair]["heatmap_mean"]).flatten()
    top_n = np.argsort(-nec)[:5].tolist()
    top_s = np.argsort(-suff)[:5].tolist()
    label = list(dict.fromkeys(top_n + top_s))

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(suff, nec, s=6, color=C_OTHER, alpha=0.6, lw=0)
    for i in label:
        ax.scatter(suff[i], nec[i], s=14, color=C_JOINT, lw=0, zorder=3)
    ax.axhline(0, color="black", lw=0.4, alpha=0.4)
    ax.axvline(0, color="black", lw=0.4, alpha=0.4)
    r = np.corrcoef(nec, suff)[0, 1]
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_xlabel("sufficiency (clean $\\to$ corrupted), mean recovery")
    ax.set_ylabel("necessity (corrupted $\\to$ clean), mean")
    ax.set_title(f"336 heads (correlation {r:+.2f})")
    place_labels(ax, suff[label], nec[label], [E.lh(i) for i in label])
    save(fig, f"necessity_{pair}")


def plot_pairs(analysis: dict, pair: str, top: int = 15) -> None:
    p = analysis["pairs"]
    rank = (lambda d: -abs(d["t"])) if p.get("stat", "mean") == "t" else (lambda d: -abs(d["I"]))
    items = sorted(p["all"], key=rank)[:top][::-1]
    sig = {d["pair"] for d in p["significant"]}
    fig, ax = plt.subplots(figsize=(4.6, 0.26 * top + 1.2))
    y = np.arange(len(items))
    I = np.array([d["I"] for d in items])
    lo = np.array([d["ci"][0] for d in items])
    hi = np.array([d["ci"][1] for d in items])
    robust = set()
    if p.get("stat", "mean") == "t" and "by_baseline" in p:
        bb = p["by_baseline"]
        robust = set(bb["v2"]["significant_t"]) & set(bb["same-run"]["significant_t"])
    C_WEAK = "#e8a33d"
    def colour(d):
        if robust:                                   # separate robust from baseline-dependent
            return C_SIG if d["pair"] in robust else (C_WEAK if d["pair"] in sig else C_OTHER)
        return C_SIG if d["pair"] in sig else C_OTHER
    colors = [colour(d) for d in items]
    ax.axvspan(*p["random_interval"], color=C_RAND, alpha=0.2, lw=0,
               label="random pairs (2.5-97.5%)")
    if p.get("stat", "mean") == "mean":
        ax.axvline(p["tau"], color="black", ls="--", lw=0.8,
                   label=f"family-wise threshold ({p['tau']:.4f})")
        ax.axvline(-p["tau"], color="black", ls="--", lw=0.8)
    else:                                   # threshold is on t, not on the interaction itself
        ax.plot([], [], "o", color=C_SIG, ms=4,
                label="significant under both baselines (max-$|t|$)")
        ax.plot([], [], "o", color=C_WEAK, ms=4, label="significant under same-run singles only")
        ax.plot([], [], "o", color=C_OTHER, ms=4, label="not significant")
    ax.errorbar(I, y, xerr=[I - lo, hi - I], fmt="none", ecolor="black", lw=0.8, capsize=1.5)
    ax.scatter(I, y, c=colors, s=18, zorder=3, lw=0)
    ax.set_yticks(y)
    ax.set_yticklabels([d["pair"] for d in items], fontsize=7)
    ax.set_xlabel("interaction $I(A,B)=R(A,B)-R(A)-R(B)$")
    ax.set_title(f"Pair interactions (singles: {p.get('baseline', 'v2')})")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    save(fig, f"pairs_{pair}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="en_hi-latn", choices=E.PAIR_NAMES)
    ap.add_argument("--scores", type=Path, default=E.SCORES)
    ap.add_argument("--curve", type=Path, default=None)
    ap.add_argument("--noise", type=Path, default=None)
    ap.add_argument("--analysis", type=Path, default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig-dir", type=Path, default=None,
                    help="output folder (default: figures/stage1/entangle)")
    a = ap.parse_args()
    if a.fig_dir is not None:
        global FIG_DIR
        FIG_DIR = a.fig_dir

    rng = np.random.default_rng(a.seed)
    v2 = E.load(a.scores)
    singles = E.v2_singles(v2, a.pair)
    curve = a.curve or E.default_path("curve", a.pair)
    noise = a.noise or E.default_path("noise", a.pair)
    analysis = a.analysis or E.default_path("analysis", a.pair)

    if curve.is_file():
        plot_curve(E.load(curve), singles, a.pair, a.boot, rng)
    else:
        print(f"(skipping curve: {curve} not found)")
    if noise.is_file():
        plot_necessity(E.load(noise), v2, a.pair)
    else:
        print(f"(skipping necessity: {noise} not found)")
    if analysis.is_file() and "pairs" in E.load(analysis):
        plot_pairs(E.load(analysis), a.pair)
    else:
        print(f"(skipping pairs: run `entangle.py analyze` first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
