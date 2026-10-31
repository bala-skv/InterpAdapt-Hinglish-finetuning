#!/usr/bin/env python3
"""
run_concepts.py - run the whole Stage 1 study on a new concept set, in one command.

    python scripts/pipeline/run_concepts.py --concepts data/tracing/my_set.json --name my_set

Steps (in order). Each writes into results/concepts/<name>/ (figures into
figures/concepts/<name>/) and is SKIPPED if its outputs already exist, so the
same command can be re-run after an interruption. Use --force STEP to redo one.

  check           schema and sanity checks; freezes a copy of the concept set      CPU
  validate        log-prob floor and contrast check per concept and contrast      GPU
  heads           single-head patching, 336 heads x 3 contrasts                   GPU
  heads_analysis  bootstrap, family-wise test, leave-one-out, heatmaps            CPU
  mlp             per-layer MLP-output patching, 28 layers x 3 contrasts          GPU
  mlp_analysis    bootstrap and family-wise tests over the 28 layers              CPU
  curve           joint patching curve, top-k vs random-k heads                   GPU
  noise           reverse-direction sweep (necessity of each head)                GPU
  shortlist       fixes the candidate heads for the pair test                     CPU
  pairs           pair interactions (resumable, checkpoints every concept)        GPU
  interactions    pair statistics: both single-head baselines, both tests         CPU
  plots           interaction figures                                             CPU
  report          REPORT.md summarising every step that has finished              CPU

The interaction steps (curve .. plots) run for each contrast in
--entangle-pairs (default: the language contrast, en_hi-latn).

Every GPU step checks the GPU first (--require-gpu, default "2080 Ti": the
identity self-test is exact there and fails on a GTX 1080 Ti) and refuses to
mix GPU models within one run. Each step logs to results/concepts/<name>/logs/.

Useful options
  --dry-run              print the plan and a GPU-time estimate, run nothing
  --steps cpu|gpu|a,b,c  run a subset (default: all)
  --limit N              smoke test on the first N concepts, in a separate
                         run folder (<name>__smokeN) so it can never be
                         mistaken for the real run
  --drop-unusable        drop concepts that fail validation instead of stopping
  --shortlist-size S     heads in the pair test (S*(S-1)/2 pairs; default 50)

Component: Stage 1 pipeline (heads, MLP, interactions)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE1 = REPO / "scripts" / "stage1"
PAIR_NAMES = ("en_hi-latn", "hi-deva_hi-latn", "en_hi-deva")
CONDITIONS = ("en", "hi_deva", "hi_latn")
RATE = 23.0          # forward passes per second, measured on an RTX 2080 Ti
LOAD_S = 90          # rough model-load time per GPU step, seconds
N_LAYERS, N_HEADS = 28, 12

STEPS = [
    ("check", "cpu", "schema checks; freeze the concept set"),
    ("validate", "gpu", "log-prob floor and contrast per concept"),
    ("heads", "gpu", "single-head patching, 3 contrasts"),
    ("heads_analysis", "cpu", "bootstrap, family-wise test, heatmaps"),
    ("mlp", "gpu", "per-layer MLP patching, 3 contrasts"),
    ("mlp_analysis", "cpu", "bootstrap and family-wise tests, 28 layers"),
    ("curve", "gpu", "joint patching curve"),
    ("noise", "gpu", "necessity sweep"),
    ("shortlist", "cpu", "fix candidate heads"),
    ("pairs", "gpu", "pair interactions (resumable)"),
    ("interactions", "cpu", "pair statistics"),
    ("plots", "cpu", "interaction figures"),
    ("report", "cpu", "REPORT.md"),
]
STEP_NAMES = [s[0] for s in STEPS]
DEVICE = {s[0]: s[1] for s in STEPS}
ENTANGLE_STEPS = ("curve", "noise", "shortlist", "pairs", "interactions", "plots")
ALWAYS_RUN = ("check", "report")     # cheap; check is what detects an edited concept file


# ===========================================================================
# helpers
# ===========================================================================

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        names = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return names[0] if names else None
    except (OSError, subprocess.SubprocessError):
        return None


def load_entangle():
    """scripts/stage1/entangle.py as a module (its statistics helpers)."""
    if str(STAGE1) not in sys.path:
        sys.path.insert(0, str(STAGE1))
    spec = importlib.util.spec_from_file_location("stage1_entangle", STAGE1 / "entangle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_concepts(data) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a concept set in the tracing-set format."""
    errors, warnings = [], []
    if not isinstance(data, list) or not data:
        return ["the file must contain a non-empty JSON list of concepts"], warnings
    required = ["concept_id", "category"] + [f"{c}_{k}" for c in CONDITIONS
                                             for k in ("prompt", "target")]
    seen = Counter()
    for i, item in enumerate(data):
        where = f"entry {i}"
        if not isinstance(item, dict):
            errors.append(f"{where}: not a JSON object")
            continue
        cid = item.get("concept_id")
        if isinstance(cid, str) and cid:
            where = f"{cid}"
            seen[cid] += 1
            if not re.fullmatch(r"[A-Za-z0-9_\-]+", cid):
                warnings.append(f"{where}: concept_id should use only letters, digits, _ and -")
        for key in required:
            v = item.get(key)
            if not isinstance(v, str) or not v.strip():
                errors.append(f"{where}: missing or empty '{key}'")
        if any(f"{c}_{k}" not in item for c in CONDITIONS for k in ("prompt", "target")):
            continue
        for c in CONDITIONS:
            prompt, target = item[f"{c}_prompt"], item[f"{c}_target"]
            if isinstance(prompt, str) and prompt and not prompt.endswith(" "):
                warnings.append(f"{where}: {c}_prompt does not end with a space "
                                "(the v1 set puts the space at the end of the prompt)")
            if isinstance(target, str) and target.startswith(" "):
                warnings.append(f"{where}: {c}_target starts with a space")
        targets = {c: item[f"{c}_target"].strip() for c in CONDITIONS
                   if isinstance(item.get(f"{c}_target"), str)}
        for a, b in (("en", "hi_latn"), ("hi_deva", "hi_latn"), ("en", "hi_deva")):
            if a in targets and b in targets and targets[a] == targets[b]:
                errors.append(f"{where}: {a}_target and {b}_target are identical "
                              f"('{targets[a]}'), so that contrast is exactly zero")
    for cid, n in seen.items():
        if n > 1:
            errors.append(f"concept_id '{cid}' appears {n} times")
    cats = Counter(item.get("category") for item in data if isinstance(item, dict))
    small = sorted(c for c, n in cats.items() if isinstance(c, str) and n < 5)
    if small:
        warnings.append("categories with fewer than 5 concepts (per-category results will "
                        f"be noisy): {', '.join(small)}")
    if len(data) < 30:
        warnings.append(f"only {len(data)} concepts: the family-wise tests have little power "
                        "below about 30")
    return errors, warnings


def estimate_passes(step: str, n: int, shortlist: int, n_pairs: int) -> int:
    """Forward passes for one step (all its contrasts) on n concepts."""
    if step == "validate":
        return 12 * n
    if step == "heads":
        return 3 * n * (2 * 336 + 7)
    if step == "mlp":
        return 3 * n * (2 * 28 + 7)
    if step == "curve":
        return n_pairs * n * (5 + 2 * 9 + 2 * 8 * 5)
    if step == "noise":
        return n_pairs * n * (5 + 2 * 336 + 2)
    if step == "pairs":
        return n_pairs * n * (5 + 2 * shortlist + 2 * comb(shortlist, 2) + 2 * 190)
    return 0


# ===========================================================================
# the run
# ===========================================================================

class Run:
    def __init__(self, a):
        self.a = a
        self.name = a.name + (f"__smoke{a.limit}" if a.limit else "")
        root = a.results_root or (REPO / "results" / "concepts")
        froot = a.figures_root or (REPO / "figures" / "concepts")
        self.dir = Path(root) / self.name
        self.fig = Path(froot) / self.name
        self.logs = self.dir / "logs"
        self.frozen = self.dir / "concepts.json"
        self.used = self.dir / "concepts_used.json"
        self.heads = self.dir / "heads_trace.json"
        self.mlp = self.dir / "mlp_trace.json"
        self.validation = self.dir / "validation.json"
        self.heads_summary = self.dir / "heads_analysis" / "summary.json"
        self.mlp_summary = self.dir / "mlp_analysis.json"
        self.report = self.dir / "REPORT.md"
        self.manifest_path = self.dir / "manifest.json"
        self.manifest = load_json(self.manifest_path) if self.manifest_path.is_file() else {
            "name": self.name, "created_utc": utc(), "steps": {}}

    # -- paths ------------------------------------------------------------------
    def ent(self, pair: str, kind: str) -> Path:
        return self.dir / "entangle" / pair / f"{kind}.json"

    def outputs(self, step: str) -> list[Path]:
        P = self.a.entangle_pairs
        return {
            "check": [self.frozen],
            "validate": [self.validation, self.used],
            "heads": [self.heads],
            "heads_analysis": [self.heads_summary],
            "mlp": [self.mlp],
            "mlp_analysis": [self.mlp_summary],
            "curve": [self.ent(p, "curve") for p in P],
            "noise": [self.ent(p, "noise") for p in P],
            "shortlist": [self.ent(p, "shortlist") for p in P],
            "pairs": [self.ent(p, "pairs") for p in P],
            "interactions": [self.ent(p, "analysis") for p in P],
            "plots": [self.fig / "entangle" / p / f"pairs_{p}.png" for p in P],
            "report": [self.report],
        }[step]

    def requires(self, step: str) -> list[tuple[Path, str]]:
        P = self.a.entangle_pairs
        need = {
            "validate": [(self.frozen, "check")],
            "heads": [(self.used, "validate")],
            "heads_analysis": [(self.heads, "heads")],
            "mlp": [(self.used, "validate")],
            "mlp_analysis": [(self.mlp, "mlp")],
            "curve": [(self.used, "validate"), (self.heads, "heads")],
            "noise": [(self.used, "validate"), (self.heads, "heads")],
            "shortlist": [(self.heads, "heads")] + [(self.ent(p, "noise"), "noise") for p in P],
            "pairs": [(self.ent(p, "shortlist"), "shortlist") for p in P],
            "interactions": [(self.ent(p, k), k) for p in P for k in ("curve", "noise", "pairs")],
            "plots": [(self.ent(p, "analysis"), "interactions") for p in P],
        }
        return need.get(step, [])

    def done(self, step: str) -> bool:
        return all(p.is_file() for p in self.outputs(step))

    def n_concepts(self) -> int:
        for path in (self.used, self.frozen):
            if path.is_file():
                return len(load_json(path))
        data = load_json(self.a.concepts)
        return min(len(data), self.a.limit) if self.a.limit else len(data)

    def save_manifest(self) -> None:
        write_json(self.manifest_path, self.manifest)

    # -- running scripts --------------------------------------------------------
    def script(self, step: str, args: list, ok_codes=(0,)) -> int:
        cmd = [sys.executable] + [str(x) for x in args]
        self.logs.mkdir(parents=True, exist_ok=True)
        log = self.logs / f"{step}.log"
        print("$ " + " ".join(cmd[1:]), flush=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== {utc()}  {' '.join(cmd)}\n")
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
            rc = proc.wait()
        if rc not in ok_codes:
            raise SystemExit(f"step '{step}' failed (exit {rc}); see {log}")
        return rc

    def gpu_guard(self, step: str) -> None:
        name = gpu_name()
        need = self.a.require_gpu
        if need and (name is None or need.lower() not in name.lower()):
            raise SystemExit(
                f"step '{step}' needs a GPU matching '{need}', found: {name or 'none'}.\n"
                "The identity self-test is exact on an RTX 2080 Ti and fails on a GTX 1080 Ti.\n"
                "Use a 2080 Ti node, or pass --require-gpu '' to override.")
        prev = self.manifest.get("device")
        if prev and name and prev != name and not self.a.allow_gpu_change:
            raise SystemExit(f"this run used '{prev}' and this node has '{name}'. Mixing GPU "
                             "models within one run changes the numbers slightly; use the "
                             "same model, or pass --allow-gpu-change.")
        if name:
            self.manifest["device"] = name

    # -- the steps ----------------------------------------------------------------
    def step_check(self) -> None:
        data = load_json(self.a.concepts)
        if self.a.limit:
            data = data[: self.a.limit]
        errors, warnings = check_concepts(data)
        for w in warnings:
            print(f"  warning: {w}")
        if errors:
            for e in errors:
                print(f"  ERROR: {e}")
            raise SystemExit(f"{len(errors)} problem(s) in {self.a.concepts}; fix them and re-run")
        cats = Counter(item["category"] for item in data)
        print(f"  {len(data)} concepts in {len(cats)} categories: "
              + ", ".join(f"{c} {n}" for c, n in sorted(cats.items())))
        payload = json.dumps(data, ensure_ascii=False, indent=1)
        if self.frozen.is_file() and self.frozen.read_text(encoding="utf-8") != payload:
            if "check" not in self.a.force:
                raise SystemExit(f"{self.a.concepts} differs from the copy frozen for this run "
                                 f"({self.frozen}). Start a new run with a different --name, or "
                                 "pass --force check (later steps then describe the OLD set "
                                 "until you --force them too).")
            print("  warning: replacing the frozen concept set; outputs of later steps still "
                  "describe the old set until you re-run them with --force")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frozen.write_text(payload, encoding="utf-8")
        self.manifest["concepts_source"] = str(self.a.concepts)
        self.manifest["concepts_sha256"] = sha256(self.frozen)[:16]

    def step_validate(self) -> None:
        self.script("validate", [STAGE1 / "validate_tracing_set_v1.py", "--data", self.frozen,
                                 "--out", self.validation], ok_codes=(0, 1))
        v = load_json(self.validation)
        bad = sorted({r["concept_id"] for rows in v["pairs"].values() for r in rows
                      if not r["usable"]})
        data = load_json(self.frozen)
        if bad and not self.a.drop_unusable:
            raise SystemExit(
                f"{len(bad)} concept(s) fail the validation criteria (log-prob floor -10 per "
                f"token, contrast >= 0.3 on every contrast): {', '.join(bad)}.\n"
                "Rewrite them, or re-run with --drop-unusable to continue without them "
                "(see validation.json for the numbers).")
        keep = [c for c in data if c["concept_id"] not in set(bad)]
        write_json(self.used, keep)
        self.manifest["dropped_by_validation"] = bad
        print(f"  {len(keep)} of {len(data)} concepts used"
              + (f"; dropped: {', '.join(bad)}" if bad else ""))
        if len(keep) < 15:
            print("  warning: fewer than 15 usable concepts; the statistics will be weak")

    def step_heads(self) -> None:
        self.script("heads", [STAGE1 / "full_trace.py", "--component", "heads",
                              "--data", self.used, "--out", self.heads])

    def step_heads_analysis(self) -> None:
        out = self.heads_summary.parent
        self.script("heads_analysis", [STAGE1 / "analyze_trace.py", "--trace", self.heads,
                                       "--out", out, "--boot", self.a.boot,
                                       "--seed", self.a.seed])
        (self.fig / "heads").mkdir(parents=True, exist_ok=True)
        for png in out.glob("*.png"):
            shutil.copy(png, self.fig / "heads" / png.name)

    def step_mlp(self) -> None:
        self.script("mlp", [STAGE1 / "full_trace.py", "--component", "mlp",
                            "--data", self.used, "--out", self.mlp])

    def step_mlp_analysis(self) -> None:
        write_json(self.mlp_summary, mlp_analysis(load_json(self.mlp), self.a.boot,
                                                  self.a.seed, self.fig / "mlp"))
        print(f"  wrote {self.mlp_summary}")

    def _entangle(self, step: str, pair: str, extra: list) -> None:
        self.script(step, [STAGE1 / "entangle.py", step if step != "interactions" else "analyze",
                           "--pair", pair, "--scores", self.heads] + extra)

    def step_curve(self) -> None:
        for p in self.a.entangle_pairs:
            if self.ent(p, "curve").is_file() and "curve" not in self.a.force:
                continue
            self._entangle("curve", p, ["--data", self.used, "--out", self.ent(p, "curve")])

    def step_noise(self) -> None:
        for p in self.a.entangle_pairs:
            if self.ent(p, "noise").is_file() and "noise" not in self.a.force:
                continue
            self._entangle("noise", p, ["--data", self.used, "--out", self.ent(p, "noise")])

    def step_shortlist(self) -> None:
        for p in self.a.entangle_pairs:
            if self.ent(p, "shortlist").is_file() and "shortlist" not in self.a.force:
                continue
            extra = ["--noise", self.ent(p, "noise"), "--size", self.a.shortlist_size,
                     "--out", self.ent(p, "shortlist")]
            if self.a.exclude:
                extra += ["--exclude"] + list(self.a.exclude)
            self._entangle("shortlist", p, extra)

    def step_pairs(self) -> None:
        for p in self.a.entangle_pairs:
            out = self.ent(p, "pairs")
            if out.is_file() and "pairs" not in self.a.force:
                continue
            extra = ["--data", self.used, "--shortlist", self.ent(p, "shortlist"), "--out", out]
            if "pairs" in self.a.force:
                extra.append("--fresh")
            self._entangle("pairs", p, extra)

    def step_interactions(self) -> None:
        for p in self.a.entangle_pairs:
            self._entangle("interactions", p, [
                "--curve", self.ent(p, "curve"), "--noise", self.ent(p, "noise"),
                "--pairs-file", self.ent(p, "pairs"), "--singles", "same-run", "--stat", "t",
                "--boot", self.a.boot, "--seed", self.a.seed, "--out", self.ent(p, "analysis")])

    def step_plots(self) -> None:
        for p in self.a.entangle_pairs:
            self.script("plots", [STAGE1 / "plot_entangle.py", "--pair", p, "--scores", self.heads,
                                  "--curve", self.ent(p, "curve"), "--noise", self.ent(p, "noise"),
                                  "--analysis", self.ent(p, "analysis"),
                                  "--fig-dir", self.fig / "entangle" / p])

    def step_report(self) -> None:
        self.report.write_text(build_report(self), encoding="utf-8")
        print(f"  wrote {self.report}")

    # -- plan and execution -----------------------------------------------------
    def plan(self, steps: list[str]) -> None:
        n = self.n_concepts()
        k = len(self.a.entangle_pairs)
        S = self.a.shortlist_size
        print(f"\nrun '{self.name}': {n} concepts, interaction contrasts {', '.join(self.a.entangle_pairs)}, "
              f"shortlist {S} heads ({comb(S, 2)} pairs)")
        print(f"output: {self.dir}")
        print(f"{'step':<15} {'device':<6} {'status':<8} {'est. GPU time':>13}")
        total = 0.0
        for step in steps:
            status = "done" if (self.done(step) and step not in ALWAYS_RUN
                                and step not in self.a.force) else "to run"
            secs = 0.0
            if DEVICE[step] == "gpu":
                secs = estimate_passes(step, n, S, k) / RATE + LOAD_S
                if status == "to run":
                    total += secs
            est = f"{secs / 3600:.1f} h" if secs else "-"
            print(f"{step:<15} {DEVICE[step]:<6} {status:<8} {est:>13}")
        print(f"{'total still to run':<31} {total / 3600:>13.1f} h   "
              f"(at {RATE:.0f} forward passes/s on an RTX 2080 Ti)\n")

    def execute(self, steps: list[str]) -> None:
        for step in steps:
            if step not in ALWAYS_RUN and self.done(step) and step not in self.a.force:
                print(f"[skip] {step}: outputs exist")
                continue
            missing = [(p, s) for p, s in self.requires(step) if not p.is_file()]
            if missing:
                raise SystemExit(f"step '{step}' needs the output of step "
                                 f"'{missing[0][1]}' ({missing[0][0]}); run that first")
            if DEVICE[step] == "gpu":
                self.gpu_guard(step)
            print(f"\n=== {step}: {dict((s[0], s[2]) for s in STEPS)[step]}", flush=True)
            t0 = time.time()
            rec = {"started_utc": utc(), "status": "running"}
            self.manifest["steps"][step] = rec
            self.save_manifest()
            try:
                getattr(self, f"step_{step}")()
            except BaseException:
                rec.update(status="failed", seconds=round(time.time() - t0))
                self.save_manifest()
                raise
            rec.update(status="ok", finished_utc=utc(), seconds=round(time.time() - t0))
            if DEVICE[step] == "gpu":
                rec["device"] = self.manifest.get("device")
            self.manifest["git_commit"] = git_commit()
            self.save_manifest()


# ===========================================================================
# MLP statistics (28 layers)
# ===========================================================================

def mlp_analysis(trace: dict, B: int, seed: int, fig_dir: Path | None) -> dict:
    import numpy as np
    E = load_entangle()
    out = {"component": "mlp", "bootstrap_B": B, "seed": seed, "pairs": {}}
    for pair, rows in trace["per_concept"].items():
        rows = [r for r in rows if not r.get("skipped")]
        X = np.array([r["recovery"] for r in rows], dtype=float)      # [N, 28]
        rng = np.random.default_rng(seed)
        mean, lo, hi = E.bootstrap(X, B, rng)
        tau = E.signflip_maxabs(X, B, rng)
        t = E.tstat(X)
        tau_t = E.signflip_maxt(X, B, np.random.default_rng(seed + 1))
        out["pairs"][pair] = {
            "n_concepts": len(rows),
            "mean": mean.tolist(), "ci": [lo.tolist(), hi.tolist()], "t": t.tolist(),
            "tau_mean": tau, "significant_mean": [int(l) for l in range(X.shape[1]) if abs(mean[l]) > tau],
            "tau_t": tau_t, "significant_t": [int(l) for l in range(X.shape[1]) if abs(t[l]) > tau_t],
            "top5": [int(l) for l in np.argsort(-mean)[:5]],
        }
        if fig_dir is not None:
            try:
                _plot_mlp(pair, mean, lo, hi, out["pairs"][pair]["significant_t"], len(rows), fig_dir)
            except ImportError:
                print("  (matplotlib not installed: skipping the MLP figure)")
    return out


def _plot_mlp(pair, mean, lo, hi, sig, n, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(mean))
    colours = ["#d1495b" if l in sig else "#9aa5b1" for l in x]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar(x, mean, color=colours, width=0.75)
    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor="black", lw=0.7, capsize=1.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x[::3])
    ax.set_xlabel("layer")
    ax.set_ylabel(f"mean recovery ({n} concepts)")
    ax.set_title(f"MLP output patching, {pair} (red: significant, max-|t|)", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(fig_dir / f"mlp_{pair}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# REPORT.md
# ===========================================================================

def _fmt(v, nd=4):
    return f"{v:+.{nd}f}" if isinstance(v, (int, float)) else str(v)


def _ci(ci, nd=4):
    return f"[{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]"


def interaction_diagnostics(analysis: dict, pairs_json: dict) -> dict:
    """Same-layer shift and the curvature check, computed from saved outputs."""
    import numpy as np
    p = analysis["pairs"]
    allp = {d["pair"]: d for d in p["all"]}
    names = list(allp)
    I = np.array([allp[n]["I"] for n in names])
    layer = lambda h: int(re.match(r"L(\d+)H", h).group(1))
    same = np.array([layer(n.split("+")[0]) == layer(n.split("+")[1]) for n in names])
    bb = p["by_baseline"]
    robust = set(bb["v2"].get("significant_t", [])) & set(bb["same-run"].get("significant_t", []))
    rows = [r for r in pairs_json["per_concept"] if not r.get("skipped")]
    sm = {h: float(np.mean([r["single"][h] for r in rows])) for h in rows[0]["single"]}
    prod = np.array([sm[n.split("+")[0]] * sm[n.split("+")[1]] for n in names])
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    spearman = float(np.corrcoef(rk(prod), rk(I))[0, 1]) if len(I) > 2 else float("nan")
    return {
        "robust": sorted(robust, key=lambda n: -abs(allp[n]["t"])),
        "same_share_all": float(same.mean()),
        "same_in_robust": int(sum(1 for n in robust if same[names.index(n)])),
        "mean_I_same": float(I[same].mean()) if same.any() else float("nan"),
        "mean_I_cross": float(I[~same].mean()) if (~same).any() else float("nan"),
        "spearman_I_vs_product": spearman,
        "all": allp,
    }


def build_report(run: Run) -> str:
    L = []
    w = L.append
    m = run.manifest
    w(f"# Stage 1 report: `{run.name}`")
    w("")
    w(f"Generated {utc()} by `scripts/pipeline/run_concepts.py`; git `{git_commit() or '?'}`; "
      f"GPU `{m.get('device', '?')}`; concept file `{m.get('concepts_source', run.a.concepts)}` "
      f"(sha256 `{m.get('concepts_sha256', '?')}`).")
    w("")
    w("Numbers below are copied from the step outputs in this folder. Section 8 says how to read them.")

    # 1. concept set --------------------------------------------------------------
    used = load_json(run.used) if run.used.is_file() else (load_json(run.frozen) if run.frozen.is_file() else [])
    w("\n## 1. Concept set\n")
    if used:
        cats = Counter(c["category"] for c in used)
        w(f"{len(used)} concepts used, {len(cats)} categories.\n")
        w("| Category | Concepts |\n|---|---|")
        for c, n in sorted(cats.items(), key=lambda x: (-x[1], x[0])):
            w(f"| {c} | {n}{' (fewer than 5: noisy)' if n < 5 else ''} |")
        dropped = m.get("dropped_by_validation") or []
        if dropped:
            w(f"\nDropped by validation: {', '.join(dropped)}.")
    else:
        w("_Not run yet._")

    # 2. validation -----------------------------------------------------------------
    w("\n## 2. Validation\n")
    if run.validation.is_file():
        v = load_json(run.validation)["summary"]["pairs"]
        w("| Contrast | Usable | Mean contrast | Min contrast |\n|---|---|---|---|")
        for key, s in v.items():
            w(f"| {key} | {s['n_usable']}/{s['n_items']} | {s['mean_contrast']:.2f} | {s['min_contrast']:.2f} |")
    else:
        w("_Not run yet._")

    # 3. single heads ---------------------------------------------------------------
    w("\n## 3. Single heads\n")
    if run.heads_summary.is_file():
        st = load_json(run.heads_summary)["structure"]
        trace = load_json(run.heads) if run.heads.is_file() else {"summary": {}}
        for pair, s in st.items():
            cm = trace["summary"].get(pair, {}).get("contrast_mean")
            w(f"**{pair}** ({s['n']} concepts" + (f", mean contrast {cm:.2f}" if cm else "") + ")\n")
            w(f"- Family-wise threshold {s['fwer_threshold']:.4f}; significant: "
              f"{', '.join(s['heads_fwer_sig']) or 'none'}")
            w(f"- Heads with a 95% interval above zero: {s['n_heads_ci_above_zero']} "
              f"(about {s['expected_false_ci_above_zero']:.1f} expected by chance)\n")
            w("| Head | Mean recovery | 95% CI | In top 5 (bootstrap) | Leave-one-out max shift |\n|---|---|---|---|---|")
            loo = s.get("leave_one_out", {})
            for d in s["top10"][:5]:
                lo = loo.get(d["head"])
                lo_txt = (f"{lo['max_shift_frac']:.0%} ({lo['max_shift_concept']})"
                          + (" **> 25%**" if lo["max_shift_frac"] > 0.25 else "")) if lo else "-"
                w(f"| {d['head']} | {_fmt(d['mean'])} | {_ci(d['ci'])} | {d.get('top5_freq', 0):.0%} | {lo_txt} |")
            w("")
    else:
        w("_Not run yet._")

    # 4. per category -----------------------------------------------------------------
    w("\n## 4. Leading heads per category (descriptive)\n")
    if run.heads.is_file():
        import numpy as np
        trace = load_json(run.heads)
        w("Mean recovery of the three best heads within each category. No multiple-comparison "
          "correction; use this to generate hypotheses, not to claim them.\n")
        for pair, rows in trace["per_concept"].items():
            rows = [r for r in rows if not r.get("skipped")]
            by = defaultdict(list)
            for r in rows:
                by[r["category"]].append(np.array(r["recovery"]).flatten())
            w(f"**{pair}**\n")
            w("| Category | n | Top heads (mean recovery) |\n|---|---|---|")
            for cat in sorted(by, key=lambda c: (-len(by[c]), c)):
                mu = np.mean(by[cat], axis=0)
                top = np.argsort(-mu)[:3]
                cells = ", ".join(f"L{i // N_HEADS}H{i % N_HEADS} {mu[i]:+.3f}" for i in top)
                w(f"| {cat} | {len(by[cat])} | {cells} |")
            w("")
    else:
        w("_Not run yet._")

    # 5. MLP --------------------------------------------------------------------------
    w("\n## 5. MLP layers\n")
    if run.mlp_summary.is_file():
        ms = load_json(run.mlp_summary)["pairs"]
        w("| Contrast | max-\\|mean\\| threshold | Significant | max-\\|t\\| threshold | Significant | Top 5 layers (mean) |\n|---|---|---|---|---|---|")
        for pair, s in ms.items():
            top = ", ".join(f"L{l} {s['mean'][l]:+.3f}" for l in s["top5"])
            w(f"| {pair} | {s['tau_mean']:.4f} | {', '.join('L' + str(l) for l in s['significant_mean']) or 'none'} | "
              f"{s['tau_t']:.2f} | {', '.join('L' + str(l) for l in s['significant_t']) or 'none'} | {top} |")
        w(f"\nFigures: `figures/concepts/{run.name}/mlp/`.")
    else:
        w("_Not run yet._")

    # 6. interactions -------------------------------------------------------------------
    w("\n## 6. Head interactions\n")
    any_pair = False
    for pair in run.a.entangle_pairs:
        ap = run.ent(pair, "analysis")
        if not ap.is_file():
            continue
        any_pair = True
        a = load_json(ap)
        w(f"### {pair}\n")
        if "curve" in a:
            w("Joint patching curve (top-k heads patched together vs the sum of their single-head recoveries):\n")
            w("| k | Joint | Sum of singles | Random k | Joint - sum [95% CI] |\n|---|---|---|---|---|")
            for k, c in a["curve"].items():
                rnd = "-" if c["random"] is None else f"{c['random']:+.3f}"
                w(f"| {k} | {c['joint']:+.3f} | {c['additive']:+.3f} | {rnd} | "
                  f"{c['joint_minus_additive']:+.3f} {_ci(c['joint_minus_additive_ci'], 3)} |")
            w("")
        if "noise" in a:
            n = a["noise"]
            top = ", ".join(f"{d['head']} {d['necessity']:+.4f} (suff. {d['sufficiency']:+.4f})" for d in n["top10"][:5])
            w(f"Necessity: all heads together {n['necessity_all_heads']:+.3f}; correlation with "
              f"sufficiency {n['corr_with_sufficiency']:+.2f}; top five: {top}.\n")
        if "pairs" in a:
            p = a["pairs"]
            bb = p["by_baseline"]
            w(f"Pairs: {p['n_pairs']} shortlist pairs, {p['n_random_pairs']} random pairs, {p['n_concepts']} concepts. "
              f"Random-pair interactions (2.5-97.5%): {_ci(p['random_interval'])}.\n")
            w("| Single-head baseline | max-\\|mean\\| threshold | Significant | max-\\|t\\| threshold | Significant |\n|---|---|---|---|---|")
            for src in ("v2", "same-run"):
                b = bb[src]
                w(f"| {src} | {b['tau']:.4f} | {b['n_significant']} | {b.get('tau_t', float('nan')):.2f} | {b.get('n_significant_t', '-')} |")
            pp = run.ent(pair, "pairs")
            if pp.is_file():
                d = interaction_diagnostics(a, load_json(pp))
                w(f"\nRobust pairs (max-|t| under both baselines): **{len(d['robust'])}**\n")
                if d["robust"]:
                    w("| Pair | I | t | 95% CI |\n|---|---|---|---|")
                    for name in d["robust"][:30]:
                        x = d["all"][name]
                        w(f"| {name} | {x['I']:+.4f} | {x['t']:+.1f} | {_ci(x['ci'])} |")
                    if len(d["robust"]) > 30:
                        w(f"\n({len(d['robust']) - 30} more in `entangle/{pair}/analysis.json`.)")
                w(f"\nDiagnostics: same-layer pairs are {d['same_share_all']:.1%} of the shortlist pairs and "
                  f"{d['same_in_robust']} of the {len(d['robust'])} robust pairs; mean interaction "
                  f"{d['mean_I_same']:+.5f} for same-layer pairs vs {d['mean_I_cross']:+.5f} for cross-layer pairs. "
                  f"Spearman correlation of I with the product of the two single-head effects: "
                  f"{d['spearman_I_vs_product']:+.2f}.")
        w(f"\nFigures: `figures/concepts/{run.name}/entangle/{pair}/`.\n")
    if not any_pair:
        w("_Not run yet._")

    # 7. steps ------------------------------------------------------------------------
    w("\n## 7. Steps\n")
    w("| Step | Status | Minutes | GPU |\n|---|---|---|---|")
    for step in STEP_NAMES:
        r = m.get("steps", {}).get(step)
        if r:
            w(f"| {step} | {r['status']} | {r.get('seconds', 0) / 60:.1f} | {r.get('device', '')} |")

    # 8. how to read ----------------------------------------------------------------------
    w("\n## 8. How to read this report\n")
    w("- **Recovery** is the fraction of the clean-corrupted contrast restored by patching, at the last "
      "prompt token; 1 means fully restored.")
    w("- **Family-wise tests** control the chance of any false positive across all heads, layers or pairs "
      "tested. max-|mean| uses one threshold for everything (it was the test planned for the pairs); "
      "max-|t| standardises each item by its spread across concepts and was added after a few noisy "
      "early-layer pairs inflated the max-|mean| threshold. Report both.")
    w("- **Robust pairs** pass max-|t| with both single-head baselines (v2 trace of this run, and the "
      "singles re-measured inside the pairs run). They are the conservative claim.")
    w("- **A significant interaction** means joint patching is non-additive for that pair on these "
      "concepts. It is not by itself evidence of a mechanism; see the same-layer and curvature "
      "diagnostics.")
    w("- **Per-category tables** are descriptive: no correction, and categories with few concepts are noisy.")
    w("- **Shortlist**: pairs are tested only among the shortlist heads, so heads outside it are untested, "
      "not ruled out.")
    return "\n".join(L) + "\n"


# ===========================================================================
# CLI
# ===========================================================================

def parse_steps(spec: str) -> list[str]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(STEP_NAMES)
    if spec in ("gpu", "cpu"):
        return [s for s in STEP_NAMES if DEVICE[s] == spec]
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STEP_NAMES]
    if unknown:
        raise SystemExit(f"unknown step(s): {', '.join(unknown)}; choose from {', '.join(STEP_NAMES)}")
    return [s for s in STEP_NAMES if s in wanted]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--concepts", type=Path, required=True, help="concept set in the tracing-set JSON format")
    p.add_argument("--name", required=True, help="run name; outputs go to results/concepts/<name>/")
    p.add_argument("--steps", default="all", help="all | gpu | cpu | comma-separated step names")
    p.add_argument("--force", nargs="*", default=[], help="steps to redo even if their outputs exist")
    p.add_argument("--entangle-pairs", nargs="+", default=["en_hi-latn"], choices=PAIR_NAMES,
                   help="contrasts for the interaction steps (default: en_hi-latn)")
    p.add_argument("--shortlist-size", type=int, default=50, help="heads in the pair test (default 50)")
    p.add_argument("--exclude", nargs="*", default=[], help="heads to keep out of the shortlist, e.g. L0H10")
    p.add_argument("--drop-unusable", action="store_true",
                   help="drop concepts that fail validation instead of stopping")
    p.add_argument("--limit", type=int, default=None,
                   help="smoke test on the first N concepts, in run folder <name>__smokeN")
    p.add_argument("--require-gpu", default="2080 Ti",
                   help="substring the GPU name must contain for GPU steps ('' to allow any)")
    p.add_argument("--allow-gpu-change", action="store_true",
                   help="allow GPU steps of one run on different GPU models")
    p.add_argument("--dry-run", action="store_true", help="print the plan and estimates only")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-root", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--figures-root", type=Path, default=None, help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", a.name):
        raise SystemExit("--name may contain only letters, digits, '.', '_' and '-'")
    if not a.concepts.is_file():
        raise SystemExit(f"concept file not found: {a.concepts}")
    if a.shortlist_size < 2:
        raise SystemExit("--shortlist-size must be at least 2")
    unknown = [s for s in a.force if s not in STEP_NAMES]
    if unknown:
        raise SystemExit(f"unknown step(s) in --force: {', '.join(unknown)}")
    a.concepts = a.concepts.resolve()
    steps = parse_steps(a.steps)
    run = Run(a)
    run.plan(steps)
    if a.dry_run:
        return 0
    run.execute(steps)
    if "report" not in steps:
        run.step_report()
    print(f"\ndone. Report: {run.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
