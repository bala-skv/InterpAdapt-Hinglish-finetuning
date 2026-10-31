#!/usr/bin/env python3
"""
entangle.py - Stage 1 extension: do attention heads interact?

Single-head patching assumes heads act independently. This script tests that
assumption on the existing tracing set, with the same scoring code as the
v2 trace (`_encode` and `_logp_from_logits` are loaded from full_trace.py).

GPU commands (fp16, any CUDA GPU; no bitsandbytes needed)
  curve      Joint patching curve: patch the top-k heads by v2 score together,
             and k random heads, for k in 1,2,3,5,10,20,50,100,336.
  noise      Reverse-direction sweep: copy CORRUPTED activations into the
             CLEAN run, one head at a time (necessity).
  pairs      Patch every pair in a fixed shortlist together, plus random
             pairs as a control.

CPU commands (no torch needed)
  shortlist  Fix the candidate heads BEFORE running pairs (pre-registration).
  analyze    Interactions, bootstrap intervals, family-wise test, controls.

Definitions, per concept:
  R(S)   = (m_patch(S) - m_corr) / (m_clean - m_corr)    sufficiency of set S
  N(h)   = (m_clean - m_noised(h)) / (m_clean - m_corr)  necessity of head h
  I(A,B) = R({A,B}) - R({A}) - R({B})                    pair interaction
           > 0 synergy (needed together); < 0 redundancy (either suffices)

Every GPU command first self-tests on the first concept and exits if either
check fails:
  1. Re-injecting the target run's own activations into all 336 heads leaves
     m_corr unchanged (set patching alters nothing it should not).
  2. Single-head sets reproduce the committed v2 recovery for its top-3 heads
     (this script and full_trace.py measure the same thing).

Typical use (from the repository root):
  python scripts/stage1/entangle.py curve --limit 2      # smoke test
  python scripts/stage1/entangle.py curve
  python scripts/stage1/entangle.py noise
  python scripts/stage1/entangle.py shortlist             # then commit it
  python scripts/stage1/entangle.py pairs
  python scripts/stage1/entangle.py analyze

Outputs go to results/stage1/entangle/<command>_<pair>.json.
See docs/stage1_interactions.md.

Component: Stage 1 head-interaction analysis
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

REPO_ROOT = _bootstrap.REPO_ROOT
DATA = REPO_ROOT / "data" / "tracing" / "tracing_set_v1.json"
SCORES = REPO_ROOT / "results" / "stage1" / "full_trace_v2.json"
OUT_DIR = REPO_ROOT / "results" / "stage1" / "entangle"

L, H, D = 28, 12, 128          # verified layout (docs/reproducibility.md)
N_HEADS = L * H
KS = [1, 2, 3, 5, 10, 20, 50, 100, 336]
PAIR_NAMES = ("en_hi-latn", "hi-deva_hi-latn", "en_hi-deva")


def lh(i: int) -> str:
    return f"L{i // H}H{i % H}"


def flat(name: str) -> int:
    layer, head = name[1:].split("H")
    return int(layer) * H + int(head)


def default_path(kind: str, pair: str) -> Path:
    return OUT_DIR / f"{kind}_{pair}.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _git_commit() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def provenance(args, extra: dict | None = None) -> dict:
    """Configuration and environment for the run (docs/development.md rule 4)."""
    info = {
        "argv": sys.argv,
        "args": {k: str(v) for k, v in vars(args).items() if k != "fn"},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    info.update(extra or {})
    return info


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))
    print(f"wrote {path}")


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ===========================================================================
# GPU side
# ===========================================================================

def _load_full_trace():
    """Import scripts/stage1/full_trace.py by path: the v2 scoring code."""
    path = Path(__file__).with_name("full_trace.py")
    spec = importlib.util.spec_from_file_location("stage1_full_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tracer:
    """Loads the model once; measures contrasts with optional set patches."""

    def __init__(self, data_path: Path):
        import torch
        import transformers
        from interp_adapt.tracing import patching
        from interp_adapt.tracing.model_loading import LoadConfig, load_base_model
        from interp_adapt.tracing.set_patching import set_patched_logits

        assert (patching.NUM_LAYERS, patching.NUM_HEADS, patching.HEAD_DIM) == (L, H, D)
        self.torch = torch
        self.ft = _load_full_trace()
        self.capture_activations = patching.capture_activations
        self.set_patched_logits = set_patched_logits

        cfg = LoadConfig(precision="fp16")
        self.model, self.tok = load_base_model(cfg)
        self.model.eval()
        self.env = {
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model_revision": cfg.revision,
            "precision": cfg.precision,
            "data_sha256": _sha(data_path),
        }
        print(f"device: {self.env['device']}")
        self.concepts = json.loads(data_path.read_text(encoding="utf-8"))

    # -- scoring (one path for every number) ---------------------------------
    def _logits(self, enc, heads=(), stored=None):
        ids, _, pos = enc
        x = self.torch.tensor([ids], device=self.model.device)
        pairs = [(i // H, i % H) for i in heads]
        return self.set_patched_logits(self.model, x, pos, pairs, stored)

    def m(self, a, b, heads=(), stored=None) -> float:
        """Contrast on one prompt: logp(target a) - logp(target b)."""
        la = self.ft._logp_from_logits(self._logits(a, heads, stored), a[0], a[1])
        lb = self.ft._logp_from_logits(self._logits(b, heads, stored), b[0], b[1])
        return la - lb

    def capture(self, enc):
        ids, _, pos = enc
        x = self.torch.tensor([ids], device=self.model.device)
        return self.capture_activations(self.model, x, pos)

    def prepare(self, item: dict, pair: str) -> dict:
        """Encode the four sequences and compute m_clean, m_corr, gap."""
        src, tgt = self.ft.PAIRS[pair]
        sp, st = item[f"{src}_prompt"], item[f"{src}_target"]
        tp, tt = item[f"{tgt}_prompt"], item[f"{tgt}_target"]
        P = {
            "s_a": self.ft._encode(self.tok, sp, st),   # source prompt, source word
            "s_b": self.ft._encode(self.tok, sp, tt),   # source prompt, target word
            "t_a": self.ft._encode(self.tok, tp, st),   # target prompt, source word
            "t_b": self.ft._encode(self.tok, tp, tt),   # target prompt, target word
        }
        P["m_clean"] = self.m(P["s_a"], P["s_b"])
        P["m_corr"] = self.m(P["t_a"], P["t_b"])
        P["gap"] = P["m_clean"] - P["m_corr"]
        return P

    def recovery(self, P, heads, stored_src) -> float:
        m_patch = self.m(P["t_a"], P["t_b"], heads, stored_src)
        return (m_patch - P["m_corr"]) / P["gap"]

    def necessity(self, P, heads, stored_tgt) -> float:
        m_noised = self.m(P["s_a"], P["s_b"], heads, stored_tgt)
        return (P["m_clean"] - m_noised) / P["gap"]

    def items(self, limit):
        return self.concepts[:limit] if limit else self.concepts

    # -- self-test ------------------------------------------------------------
    def selftest(self, pair: str, v2: dict) -> dict:
        item = self.concepts[0]
        P = self.prepare(item, pair)
        stored_tgt = self.capture(P["t_a"])
        m_id = self.m(P["t_a"], P["t_b"], range(N_HEADS), stored_tgt)
        d1 = abs(m_id - P["m_corr"])
        print(f"[selftest 1] identity re-injection, all 336 heads: |diff| = {d1:.2e}")
        if d1 > 1e-4:
            raise SystemExit("selftest 1 FAILED: set patching alters the run")

        rows = {r["concept_id"]: r for r in v2["per_concept"][pair]}
        row = rows.get(item["concept_id"])
        if row is None or row.get("skipped"):
            raise SystemExit(f"selftest 2: {item['concept_id']} missing from the v2 trace")
        ref = np.array(row["recovery"]).flatten()
        stored_src = self.capture(P["s_a"])
        top3 = np.argsort(-np.array(v2["summary"][pair]["heatmap_mean"]).flatten())[:3]
        worst, checks = 0.0, {}
        for i in (int(j) for j in top3):
            r = self.recovery(P, [i], stored_src)
            worst = max(worst, abs(r - ref[i]))
            checks[lh(i)] = {"this_script": r, "v2": float(ref[i])}
            print(f"[selftest 2] {lh(i)}: this script {r:+.5f}   v2 {ref[i]:+.5f}")
        if worst > 2e-3:
            raise SystemExit(
                f"selftest 2 FAILED: max diff {worst:.2e} against the v2 trace. "
                "Small (~1e-3) suggests a different GPU; large means data or code differ.")
        print("selftests passed\n")
        return {"identity_diff": d1, "v2_max_diff": worst, "v2_checks": checks}


def cmd_curve(a):
    v2 = load(a.scores)
    T = Tracer(a.data)
    st = T.selftest(a.pair, v2)
    order = np.argsort(-np.array(v2["summary"][a.pair]["heatmap_mean"]).flatten())
    rng = np.random.default_rng(a.seed)
    rand = {k: [rng.choice(N_HEADS, k, replace=False).tolist() for _ in range(a.draws)]
            for k in KS if k < N_HEADS}
    rows, t0 = [], time.time()
    for item in T.items(a.limit):
        P = T.prepare(item, a.pair)
        row = {"concept_id": item["concept_id"], "category": item["category"],
               "gap": P["gap"], "top": {}, "random": {}}
        if P["gap"] < T.ft.MIN_CONTRAST:
            row["skipped"] = True
            rows.append(row)
            continue
        stored = T.capture(P["s_a"])
        for k in KS:
            row["top"][k] = T.recovery(P, order[:k].tolist(), stored)
            if k in rand:
                row["random"][k] = [T.recovery(P, s, stored) for s in rand[k]]
        rows.append(row)
        print(f"{item['concept_id']:<18} "
              + "  ".join(f"k{k}:{row['top'][k]:+.3f}" for k in KS))
    out = a.out or default_path("curve", a.pair)
    save(out, {"command": "curve", "pair": a.pair, "ks": KS, "draws": a.draws,
               "seed": a.seed, "top_order": order[:100].tolist(), "random_sets": rand,
               "selftest": st, "seconds": time.time() - t0,
               "provenance": provenance(a, T.env), "per_concept": rows})


def cmd_noise(a):
    v2 = load(a.scores)
    T = Tracer(a.data)
    st = T.selftest(a.pair, v2)
    rows, t0 = [], time.time()
    for item in T.items(a.limit):
        P = T.prepare(item, a.pair)
        row = {"concept_id": item["concept_id"], "category": item["category"],
               "gap": P["gap"]}
        if P["gap"] < T.ft.MIN_CONTRAST:
            row["skipped"] = True
            rows.append(row)
            continue
        stored_tgt = T.capture(P["t_a"])
        grid = [T.necessity(P, [i], stored_tgt) for i in range(N_HEADS)]
        row["necessity"] = np.array(grid).reshape(L, H).tolist()
        row["necessity_all"] = T.necessity(P, list(range(N_HEADS)), stored_tgt)
        rows.append(row)
        top = np.argsort(-np.array(grid))[:3]
        print(f"{item['concept_id']:<18} all={row['necessity_all']:+.3f}  top: "
              + ", ".join(f"{lh(int(i))} {grid[i]:+.3f}" for i in top))
    out = a.out or default_path("noise", a.pair)
    save(out, {"command": "noise", "pair": a.pair, "selftest": st,
               "seconds": time.time() - t0, "provenance": provenance(a, T.env),
               "per_concept": rows})


def new_tracer(data_path: Path):
    """Indirection so the resume logic can be tested without torch or a GPU."""
    return Tracer(data_path)


def _read_partial(path: Path, header: dict) -> dict:
    """Rows already finished by an interrupted run (one JSON object per line)."""
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    if json.loads(lines[0]) != json.loads(json.dumps(header)):
        raise SystemExit(
            f"{path} belongs to a different run (shortlist, seed or settings differ).\n"
            "Use the same arguments to resume, or pass --fresh to start over.")
    done = {}
    for ln in lines[1:]:
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:          # last line cut off by a kill
            break
        done[row["concept_id"]] = row
    return done


def cmd_pairs(a):
    """Patch every shortlist pair, plus random pairs as a control.

    Resumable: each finished concept is appended to <out>.partial.jsonl. If the
    job is killed, run the same command again and it continues from the next
    concept. The partial file is deleted when the run completes.
    """
    v2 = load(a.scores)
    sl_path = a.shortlist or default_path("shortlist", a.pair)
    sl = load(sl_path)
    if sl["pair"] != a.pair:
        raise SystemExit(f"shortlist is for {sl['pair']}, not {a.pair}")
    heads = [flat(n) for n in sl["heads"]]
    pairs = [tuple(sorted(p)) for p in combinations(heads, 2)]
    taken = set(pairs)
    rng = np.random.default_rng(a.seed)
    rand_pairs = []
    while len(rand_pairs) < a.random:
        p = tuple(sorted(rng.choice(N_HEADS, 2, replace=False).tolist()))
        if p not in taken:
            taken.add(p)
            rand_pairs.append(p)
    print(f"{len(pairs)} shortlist pairs, {len(rand_pairs)} random pairs, "
          f"{len(heads)} singles re-measured")

    out = a.out or default_path("pairs", a.pair)
    partial = out.with_name(out.stem + ".partial.jsonl")
    header = {"pair": a.pair, "shortlist_sha256": _sha(sl_path), "seed": a.seed,
              "random": a.random, "n_pairs": len(pairs), "limit": a.limit}
    if getattr(a, "fresh", False) and partial.is_file():
        partial.unlink()
    done = _read_partial(partial, header)
    partial.parent.mkdir(parents=True, exist_ok=True)
    # (re)write the partial file cleanly: header plus the rows that parsed
    partial.write_text("\n".join([json.dumps(header)] + [json.dumps(r) for r in done.values()]) + "\n",
                       encoding="utf-8")
    if done:
        print(f"resuming: {len(done)} concept(s) already in {partial.name}")

    T = new_tracer(a.data)
    st = T.selftest(a.pair, v2)
    items = T.items(a.limit)
    rows, n_new, t_session = [], 0, time.time()
    for item in items:
        cid = item["concept_id"]
        if cid in done:
            rows.append(done[cid])
            continue
        t1 = time.time()
        P = T.prepare(item, a.pair)
        row = {"concept_id": cid, "category": item["category"], "gap": P["gap"]}
        if P["gap"] < T.ft.MIN_CONTRAST:
            row["skipped"] = True
        else:
            stored = T.capture(P["s_a"])
            row["single"] = {lh(i): T.recovery(P, [i], stored) for i in heads}
            row["pair"] = {f"{lh(x)}+{lh(y)}": T.recovery(P, [x, y], stored)
                           for x, y in pairs}
            row["random_pair"] = {f"{lh(x)}+{lh(y)}": T.recovery(P, [x, y], stored)
                                  for x, y in rand_pairs}
        row["seconds"] = time.time() - t1
        with partial.open("a", encoding="utf-8") as fh:        # checkpoint
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        rows.append(row)
        n_new += 1
        eta = (time.time() - t_session) / n_new * (len(items) - len(rows))
        print(f"{cid:<18} {len(rows)}/{len(items)} done  {row['seconds']:.0f}s  "
              f"ETA {eta / 60:.0f} min", flush=True)
    save(out, {"command": "pairs", "pair": a.pair, "shortlist": sl,
               "shortlist_sha256": _sha(sl_path), "seed": a.seed, "selftest": st,
               "seconds": sum(r.get("seconds", 0.0) for r in rows),
               "provenance": provenance(a, T.env), "per_concept": rows})
    partial.unlink()


# ===========================================================================
# CPU side
# ===========================================================================

def v2_singles(v2: dict, pair: str) -> dict:
    """concept_id -> flat [336] array of v2 single-head recovery."""
    return {r["concept_id"]: np.array(r["recovery"]).flatten()
            for r in v2["per_concept"][pair] if not r.get("skipped")}


def cmd_shortlist(a):
    v2 = load(a.scores)
    noise_path = a.noise or default_path("noise", a.pair)
    noise = load(noise_path)
    if noise["pair"] != a.pair:
        raise SystemExit(f"noise file is for {noise['pair']}, not {a.pair}")
    suff = np.array(v2["summary"][a.pair]["heatmap_mean"]).flatten()
    nec = np.mean([np.array(r["necessity"]).flatten()
                   for r in noise["per_concept"] if not r.get("skipped")], axis=0)
    exclude = {flat(n) for n in a.exclude}
    excl_txt = f", excluding {sorted(a.exclude)}" if a.exclude else ""
    size = getattr(a, "size", None)
    if size:
        # exactly `size` heads: alternate the next-best by sufficiency and by
        # necessity, skipping repeats. The first 2k distinct heads always
        # contain the top-k of each list, so smaller shortlists are subsets.
        rank_s = [int(i) for i in np.argsort(-suff) if int(i) not in exclude]
        rank_n = [int(i) for i in np.argsort(-nec) if int(i) not in exclude]
        heads, top_s, top_n = [], [], []
        for r in range(len(rank_s)):
            for ranking, bucket in ((rank_s, top_s), (rank_n, top_n)):
                h = ranking[r]
                if h not in heads and len(heads) < size:
                    heads.append(h)
                    bucket.append(h)
            if len(heads) >= size:
                break
        rule = (f"{size} heads: alternately the next-best by v2 sufficiency "
                f"(heatmap_mean) and by mean necessity, skipping repeats{excl_txt}")
    else:
        top_s = [int(i) for i in np.argsort(-suff) if int(i) not in exclude][:a.top]
        top_n = [int(i) for i in np.argsort(-nec) if int(i) not in exclude][:a.top]
        heads = list(dict.fromkeys(top_s + top_n))
        rule = (f"top {a.top} by v2 sufficiency (heatmap_mean) union top {a.top} "
                f"by mean necessity{excl_txt}")
    out = {
        "pair": a.pair,
        "rule": rule,
        "heads": [lh(i) for i in heads],
        "from_sufficiency": [lh(i) for i in top_s],
        "from_necessity": [lh(i) for i in top_n],
        "necessary_not_sufficient": [lh(i) for i in top_n if i not in top_s],
        "n_pairs": len(heads) * (len(heads) - 1) // 2,
        "inputs_sha256": {"scores": _sha(a.scores), "noise": _sha(noise_path)},
        "provenance": provenance(a),
    }
    path = a.out or default_path("shortlist", a.pair)
    save(path, out)
    print(f"{len(heads)} heads -> {out['n_pairs']} pairs")
    n_concepts = sum(1 for r in noise["per_concept"] if not r.get("skipped"))
    passes = n_concepts * (2 * len(heads) + 2 * out["n_pairs"] + 2 * 190 + 4)
    print(f"estimated pairs run: about {passes / 23.0 / 3600:.1f} h "
          f"(23 forward passes/s measured on an RTX 2080 Ti, {n_concepts} concepts, 190 random pairs)")
    print("necessary but not sufficient:",
          ", ".join(out["necessary_not_sufficient"]) or "none")
    print(f"Commit {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path} "
          "before running `pairs`: it is the pre-registered candidate set.")


def bootstrap(stack: np.ndarray, B: int, rng):
    """stack [N, P] -> mean, 2.5th and 97.5th percentiles over concept resamples."""
    n = stack.shape[0]
    means = np.stack([stack[rng.integers(0, n, n)].mean(0) for _ in range(B)])
    return stack.mean(0), np.percentile(means, 2.5, 0), np.percentile(means, 97.5, 0)


def signflip_maxabs(stack: np.ndarray, B: int, rng, alpha: float = 0.05) -> float:
    """Two-sided family-wise threshold on |mean| over all columns.

    One random sign per concept, applied to every column, preserves the
    dependence between pairs measured on the same concept.
    """
    n = stack.shape[0]
    mx = np.empty(B)
    for b in range(B):
        s = rng.choice([-1.0, 1.0], size=n)
        mx[b] = np.abs((s[:, None] * stack).mean(0)).max()
    return float(np.quantile(mx, 1 - alpha))


def tstat(stack: np.ndarray) -> np.ndarray:
    """One-sample t statistic of each column (concepts are the observations)."""
    n = stack.shape[0]
    sd = stack.std(0, ddof=1)
    sd = np.where(sd == 0, np.inf, sd)
    return stack.mean(0) / (sd / np.sqrt(n))


def signflip_maxt(stack: np.ndarray, B: int, rng, alpha: float = 0.05) -> float:
    """Two-sided family-wise threshold on max |t| (Nichols and Holmes 2002).

    Standardising each pair by its own spread keeps a few very noisy pairs
    from inflating the threshold for all the others, which max |mean| does.
    The t statistic is recomputed on every sign-flipped dataset.
    """
    n = stack.shape[0]
    mx = np.empty(B)
    for b in range(B):
        s = rng.choice([-1.0, 1.0], size=n)
        mx[b] = np.abs(tstat(s[:, None] * stack)).max()
    return float(np.quantile(mx, 1 - alpha))


def analyze_curve(curve: dict, singles: dict, B: int, rng) -> dict:
    rows = [r for r in curve["per_concept"] if not r.get("skipped")]
    order = curve["top_order"]
    out = {}
    print("\n=== joint patching curve ===")
    print(f"{'k':>4} {'top-k joint':>12} {'sum singles':>12} {'random-k':>9} "
          f"{'joint - sum':>12}  95% CI")
    for k in curve["ks"]:
        ks = str(k)
        joint = np.array([r["top"][ks] for r in rows])
        idx = order[:k] if k <= len(order) else list(range(N_HEADS))
        additive = np.array([singles[r["concept_id"]][idx].sum() for r in rows])
        has_rand = ks in rows[0]["random"]
        rand = np.array([np.mean(r["random"][ks]) for r in rows]) if has_rand else None
        m, lo, hi = bootstrap((joint - additive)[:, None], B, rng)
        out[ks] = {"joint": float(joint.mean()), "additive": float(additive.mean()),
                   "random": None if rand is None else float(rand.mean()),
                   "joint_minus_additive": float(m[0]),
                   "joint_minus_additive_ci": [float(lo[0]), float(hi[0])]}
        rtxt = "-" if rand is None else f"{rand.mean():+.3f}"
        print(f"{k:>4} {joint.mean():>+12.3f} {additive.mean():>+12.3f} {rtxt:>9} "
              f"{m[0]:>+12.3f}  [{lo[0]:+.3f}, {hi[0]:+.3f}]")
    print("joint above the sum of singles (CI above 0) at small k suggests synergy;")
    print("top-k close to random-k suggests the single-head ranking carries little.")
    return out


def analyze_noise(noise: dict, v2: dict, pair: str) -> dict:
    rows = [r for r in noise["per_concept"] if not r.get("skipped")]
    nec = np.mean([np.array(r["necessity"]).flatten() for r in rows], axis=0)
    suff = np.array(v2["summary"][pair]["heatmap_mean"]).flatten()
    top = [int(i) for i in np.argsort(-nec)[:10]]
    n_all = float(np.mean([r["necessity_all"] for r in rows]))
    corr = float(np.corrcoef(nec, suff)[0, 1])
    print("\n=== reverse sweep (necessity) ===")
    print(f"mean necessity of all 336 heads together: {n_all:+.3f}")
    print(f"{'head':<7} {'necessity':>10} {'sufficiency':>12}")
    for i in top:
        print(f"{lh(i):<7} {nec[i]:>+10.4f} {suff[i]:>+12.4f}")
    print(f"correlation of necessity and sufficiency over 336 heads: {corr:+.3f}")
    return {"necessity_all_heads": n_all, "corr_with_sufficiency": corr,
            "top10": [{"head": lh(i), "necessity": float(nec[i]),
                       "sufficiency": float(suff[i])} for i in top],
            "heatmap_mean": nec.reshape(L, H).tolist()}


def analyze_pairs(pairs: dict, singles: dict, B: int, rng, baseline: str = "v2",
                  rng_t=None, stat: str = "mean") -> dict:
    """Pair interactions I(A,B) = R(A,B) - R(A) - R(B), per concept, then averaged.

    The singles R(A), R(B) can come from two places:
      "v2"       the committed v2 trace (measured earlier, possibly on another GPU)
      "same-run" the singles re-measured in this pairs run, on the same GPU
    Two family-wise tests are always run on each:
      "mean"     sign-flip on max |mean I| (one threshold for every pair)
      "t"        sign-flip on max |t|, each pair standardised by its own spread
    `baseline` and `stat` pick which results are listed as primary.
    Random control pairs only have v2 singles (their singles were not re-measured).
    """
    if baseline not in ("v2", "same-run"):
        raise SystemExit(f"baseline must be v2 or same-run, not {baseline!r}")
    if stat not in ("mean", "t"):
        raise SystemExit(f"stat must be mean or t, not {stat!r}")
    rng_t = rng_t if rng_t is not None else np.random.default_rng(1)
    rows = [r for r in pairs["per_concept"] if not r.get("skipped")]
    names = list(rows[0]["pair"])

    def interactions(key, source):
        names_k = list(rows[0][key])
        mat = np.empty((len(rows), len(names_k)))
        for c, r in enumerate(rows):
            for j, name in enumerate(names_k):
                x, y = name.split("+")
                if source == "same-run":
                    sx, sy = r["single"][x], r["single"][y]
                else:
                    v = singles[r["concept_id"]]
                    sx, sy = v[flat(x)], v[flat(y)]
                mat[c, j] = r[key][name] - sx - sy
        return names_k, mat

    diffs = [abs(v - singles[r["concept_id"]][flat(n)])
             for r in rows for n, v in r["single"].items()]
    print(f"\n=== pairs ===\nmax |re-measured single - v2| = {max(diffs):.2e}")

    _, I_rand = interactions("random_pair", "v2")
    r_lo, r_hi = np.percentile(I_rand.mean(0), [2.5, 97.5])

    by_baseline, full = {}, {}
    for source in ("v2", "same-run"):
        _, I = interactions("pair", source)
        mean, lo, hi = bootstrap(I, B, rng)
        tau = signflip_maxabs(I, B, rng)
        t = tstat(I)
        tau_t = signflip_maxt(I, B, rng_t)
        sig_mean = [j for j in range(len(names)) if abs(mean[j]) > tau]
        sig_t = [j for j in range(len(names)) if abs(t[j]) > tau_t]
        by_baseline[source] = {
            "tau": tau, "n_significant": len(sig_mean),
            "significant": [names[j] for j in sig_mean],
            "tau_t": tau_t, "n_significant_t": len(sig_t),
            "significant_t": [names[j] for j in sig_t],
            "mean_I_all_pairs": float(mean.mean())}
        full[source] = (I, mean, lo, hi, t, tau, tau_t, sig_mean, sig_t)
    shift = float((full["same-run"][0] - full["v2"][0]).mean())
    both = set(by_baseline["v2"]["significant"]) & set(by_baseline["same-run"]["significant"])
    both_t = set(by_baseline["v2"]["significant_t"]) & set(by_baseline["same-run"]["significant_t"])

    print(f"{len(names)} shortlist pairs, {I_rand.shape[1]} random pairs, {len(rows)} concepts")
    print(f"random-pair mean interaction, 2.5-97.5%: [{r_lo:+.4f}, {r_hi:+.4f}]")
    print(f"{'baseline':<10} {'max|mean| tau':>14} {'signif.':>8} {'max|t| tau':>11} {'signif.':>8} {'mean I':>9}")
    for source in ("v2", "same-run"):
        b = by_baseline[source]
        print(f"{source:<10} {b['tau']:>14.4f} {b['n_significant']:>8} {b['tau_t']:>11.2f} "
              f"{b['n_significant_t']:>8} {b['mean_I_all_pairs']:>+9.4f}")
    print(f"mean shift in I from using same-run singles: {shift:+.5f}")
    print(f"significant under both baselines: max|mean| {len(both)}, max|t| {len(both_t)}")

    I, mean, lo, hi, t, tau, tau_t, sig_mean, sig_t = full[baseline]
    sig = sig_t if stat == "t" else sig_mean
    other = "same-run" if baseline == "v2" else "v2"
    thr = f"max|t| threshold {tau_t:.2f}" if stat == "t" else f"max|mean| threshold {tau:.4f}"
    print(f"\nprimary: baseline {baseline}, statistic {stat}   ({thr})")
    key = (lambda j: -abs(t[j])) if stat == "t" else (lambda j: -abs(mean[j]))
    print(f"{'pair':<16} {'I':>8} {'95% CI':>20} {'t':>7}  {'I ('+other+')':>14}  kind")
    for j in sorted(range(len(names)), key=key)[:15]:
        kind = ("synergy" if mean[j] > 0 else "redundancy") if j in sig else "-"
        print(f"{names[j]:<16} {mean[j]:>+8.4f} [{lo[j]:+.4f}, {hi[j]:+.4f}] {t[j]:>+7.1f}  "
              f"{full[other][1][j]:>+14.4f}  {kind}")
    return {
        "baseline": baseline, "stat": stat, "n_pairs": len(names),
        "n_random_pairs": int(I_rand.shape[1]), "n_concepts": len(rows),
        "tau": tau, "tau_t": tau_t,
        "random_interval": [float(r_lo), float(r_hi)],
        "n_significant": len(sig),
        "significant": [{"pair": names[j], "I": float(mean[j]), "t": float(t[j]),
                         "ci": [float(lo[j]), float(hi[j])],
                         "kind": "synergy" if mean[j] > 0 else "redundancy"}
                        for j in sig],
        "all": [{"pair": n, "I": float(mean[j]), "t": float(t[j]),
                 "ci": [float(lo[j]), float(hi[j])],
                 "I_other_baseline": float(full[other][1][j])}
                for j, n in enumerate(names)],
        "by_baseline": by_baseline,
        "mean_shift_same_run_minus_v2": shift,
        "n_significant_under_both": len(both),
        "n_significant_under_both_t": len(both_t),
        "max_single_diff_vs_v2": float(max(diffs)),
    }


def cmd_analyze(a):
    v2 = load(a.scores)
    singles = v2_singles(v2, a.pair)
    rng = np.random.default_rng(a.seed)
    paths = {"curve": a.curve or default_path("curve", a.pair),
             "noise": a.noise or default_path("noise", a.pair),
             "pairs": a.pairs_file or default_path("pairs", a.pair)}
    result = {"pair": a.pair, "bootstrap_B": a.boot, "inputs": {}}
    for kind, path in paths.items():
        if not path.is_file():
            print(f"(skipping {kind}: {path} not found)")
            continue
        data = load(path)
        result["inputs"][kind] = {"path": str(path), "sha256": _sha(path)}
        if kind == "curve":
            result["curve"] = analyze_curve(data, singles, a.boot, rng)
        elif kind == "noise":
            result["noise"] = analyze_noise(data, v2, a.pair)
        else:
            result["pairs"] = analyze_pairs(data, singles, a.boot, rng, a.singles,
                                        np.random.default_rng(a.seed + 1),
                                        getattr(a, "stat", "mean"))
    result["provenance"] = provenance(a)
    save(a.out or default_path("analysis", a.pair), result)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, gpu=True):
        sp.add_argument("--pair", default="en_hi-latn", choices=PAIR_NAMES)
        sp.add_argument("--scores", type=Path, default=SCORES, help="v2 trace JSON")
        sp.add_argument("--out", type=Path, default=None,
                        help="default: results/stage1/entangle/<command>_<pair>.json")
        sp.add_argument("--seed", type=int, default=0)
        if gpu:
            sp.add_argument("--data", type=Path, default=DATA)
            sp.add_argument("--limit", type=int, default=None, help="first N concepts")

    sp = sub.add_parser("curve", help="joint patching curve (GPU)")
    common(sp)
    sp.add_argument("--draws", type=int, default=5, help="random sets per k")
    sp.set_defaults(fn=cmd_curve)

    sp = sub.add_parser("noise", help="reverse-direction sweep (GPU)")
    common(sp)
    sp.set_defaults(fn=cmd_noise)

    sp = sub.add_parser("pairs", help="pairs in the shortlist + random pairs (GPU)")
    common(sp)
    sp.add_argument("--shortlist", type=Path, default=None)
    sp.add_argument("--random", type=int, default=190, help="random control pairs")
    sp.add_argument("--fresh", action="store_true",
                    help="ignore and delete an existing .partial.jsonl instead of resuming")
    sp.set_defaults(fn=cmd_pairs)

    sp = sub.add_parser("shortlist", help="fix the candidate heads (CPU)")
    common(sp, gpu=False)
    sp.add_argument("--noise", type=Path, default=None)
    sp.add_argument("--top", type=int, default=10,
                    help="heads taken from each ranking (union, so up to 2x this many)")
    sp.add_argument("--size", type=int, default=None,
                    help="exactly this many heads, alternating the two rankings (overrides --top)")
    sp.add_argument("--exclude", nargs="*", default=[], help="heads to leave out, e.g. L0H10")
    sp.set_defaults(fn=cmd_shortlist)

    sp = sub.add_parser("analyze", help="interactions and statistics (CPU)")
    common(sp, gpu=False)
    sp.add_argument("--curve", type=Path, default=None)
    sp.add_argument("--noise", type=Path, default=None)
    sp.add_argument("--pairs-file", type=Path, default=None)
    sp.add_argument("--boot", type=int, default=2000)
    sp.add_argument("--stat", choices=("mean", "t"), default="mean",
                    help="which family-wise test is listed as primary (both are always run): "
                         "max |mean| (default) or the studentised max |t|")
    sp.add_argument("--singles", choices=("v2", "same-run"), default="v2",
                    help="which single-head values the pair interactions subtract "
                         "(both are always reported; this picks the primary)")
    sp.set_defaults(fn=cmd_analyze)

    args = p.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
