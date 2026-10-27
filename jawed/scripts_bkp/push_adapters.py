#!/usr/bin/env python
"""Push the Stage-2 LoRA baseline and the four CRA arms to the Hugging Face Hub.

Each run directory holds a custom ``ckpt.pt`` (torch.save payload: adapter state
+ optimizer + scheduler + step + RNG) and a ``results.json``. The CRA arms use
the bespoke ``MaskedLoRALinear`` surface (NOT a PEFT adapter), so rather than try
to emit a standard PEFT layout we upload the checkpoint verbatim plus a generated
model card that records the metrics / head budget / GPU and the load recipe.

Only depends on ``huggingface_hub`` -- runs on any node (no GPU, no torch).

Usage (from jawed/, on a COMPUTE node after ``hf auth login``):

    python scripts/push_adapters.py --owner phdiiitmohammed
    python scripts/push_adapters.py --owner phdiiitmohammed --private --dry-run
    # override a specific arm's run dir:
    python scripts/push_adapters.py --owner phdiiitmohammed \
        --run topk=runs/cra-topk-20261001-163617

The default run dirs below are the GPU-matched (GTX 1080 Ti) set for the CRA arms
plus the final LoRA baseline. Edit DEFAULT_RUNS or pass --run NAME=PATH to change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# arm -> (default run dir relative to jawed/, hub repo slug suffix, one-line role)
DEFAULT_RUNS: dict[str, tuple[str, str, str]] = {
    "baseline": ("runs/lora-baseline-20260928-234551", "babyshark-lora-baseline",
                 "Uniform LoRA baseline (the reference the CRA must beat)."),
    "uniform":  ("runs/cra-uniform-20261002-010632", "babyshark-cra-uniform",
                 "CRA uniform arm (all heads active == full LoRA; sanity check)."),
    "random":   ("runs/cra-random-20261001-233726", "babyshark-cra-random",
                 "CRA matched-budget RANDOM-head control."),
    "topk":     ("runs/cra-topk-20261001-163617", "babyshark-cra-topk",
                 "CRA hard global top-k routing (causal head selection)."),
    "soft":     ("runs/cra-soft-20261001-150600", "babyshark-cra-soft",
                 "CRA soft global routing (M = clip(s,0)/global_max scaling)."),
}

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
# files we actually want on the Hub (skip optimizer-heavy temp files, wandb logs)
ALLOW = ["ckpt.pt", "results.json", "run.log", "wandb_run_id", "README.md",
         "budget.json", "config.yaml"]


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def build_card(arm: str, role: str, repo_id: str, res: dict) -> str:
    """Render a model-card README.md (YAML front-matter + body) from results.json."""
    mask = res.get("mask_mode", arm)
    cra_mf1 = _fmt(res.get("cra_macro_f1"))
    base_mf1 = _fmt(res.get("base_macro_f1"))
    cra_acc = _fmt(res.get("cra_acc"))
    base_acc = _fmt(res.get("base_acc"))
    heads = res.get("active_heads", "n/a")
    n_eval = res.get("n_eval", "n/a")
    gpu = res.get("gpu", "n/a")
    metric_name = "CRA macro-F1" if arm != "baseline" else "LoRA macro-F1"

    front = [
        "---",
        "library_name: peft" if arm == "baseline" else "library_name: pytorch",
        f"base_model: {BASE_MODEL}",
        "language:",
        "- hi",
        "- en",
        "tags:",
        "- lora",
        "- circuit-routing-adapter",
        "- interpretability",
        "- hinglish",
        "- sentiment-analysis",
        f"- mask:{mask}",
        "datasets:",
        "- satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2",
        "metrics:",
        "- f1",
        "- accuracy",
        "---",
        "",
    ]
    body = f"""# {repo_id.split('/')[-1]}

{role}

Part of **InterpAdapt** (BabyShark team, IIIT Hyderabad): interpretability-guided
circuit routing for Hindi-English LoRA fine-tuning on **{BASE_MODEL}** (base, fp16).
Evaluated on SAIL-2017 Romanized (Hinglish) code-mixed **sentiment** (3-class).

## Results (validation, n={n_eval}, fp16, {gpu})

| Metric | Base {BASE_MODEL.split('/')[-1]} | + adapter |
|---|---|---|
| Accuracy | {base_acc} | **{cra_acc}** |
| Macro-F1 | {base_mf1} | **{cra_mf1}** |

- **Mask mode:** `{mask}`   ·   **Active head-blocks:** {heads}
- Rank-8 LoRA on `q_proj` + `o_proj` (alpha 16, dropout 0.05), 600 steps, lr 1e-4, max_len 256, seed 0.
- Head scores: Stage-1 v2 causal trace (`en_hi-latn` mean recovery).

## Files

- `ckpt.pt` — `torch.save` payload: `adapter` state dict (+ optimizer/scheduler/step/RNG for resume).
- `results.json` — full eval metrics and run metadata.

## How to load

This is **not** a plain PEFT adapter{" (the baseline is PEFT-compatible)" if arm == "baseline" else " — it uses the custom `MaskedLoRALinear` routing surface"}.
Reload with the project code (https://github.com/bala-skv/InterpAdapt-Hinglish-finetuning):

```python
import torch
# from the jawed/ sub-project:
from circuit_routing.config import load_config
# build base + {"get_peft_model" if arm == "baseline" else "inject_masked_lora"} exactly as in
# scripts/{"train_lora_baseline.py" if arm == "baseline" else "train_cra_compare.py"}, then:
ckpt = torch.load("ckpt.pt", map_location="cpu")
# load ckpt["adapter"] into the model (see load_checkpoint in that script).
```

> Numbers are ADA `fp16`. CRA arms share seed 0 and data order; only the head mask differs.
"""
    return "\n".join(front) + body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--owner", required=True, help="HF user or org (e.g. phdiiitmohammed)")
    ap.add_argument("--run", action="append", default=[],
                    help="Override a run dir: NAME=PATH (repeatable). "
                         "NAME in {baseline,uniform,random,topk,soft}.")
    ap.add_argument("--only", nargs="+", choices=list(DEFAULT_RUNS),
                    help="Only push these arms (default: all).")
    ap.add_argument("--private", action="store_true", help="Create private repos.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen; do not create/upload.")
    args = ap.parse_args()

    overrides: dict[str, str] = {}
    for item in args.run:
        if "=" not in item:
            ap.error(f"--run expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        if name not in DEFAULT_RUNS:
            ap.error(f"unknown arm {name!r}; choose from {list(DEFAULT_RUNS)}")
        overrides[name] = path

    arms = args.only or list(DEFAULT_RUNS)

    if not args.dry_run:
        from huggingface_hub import HfApi, whoami
        api = HfApi()
        try:
            who = whoami()
            print(f"[auth] logged in as: {who.get('name')}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: not logged in ({exc}). Run `hf auth login` first.", file=sys.stderr)
            return 2
    else:
        api = None

    rc = 0
    for arm in arms:
        default_dir, slug, role = DEFAULT_RUNS[arm]
        run_dir = Path(overrides.get(arm, default_dir))
        repo_id = f"{args.owner}/{slug}"
        res_path = run_dir / "results.json"

        if not run_dir.is_dir():
            print(f"SKIP {arm}: run dir not found -> {run_dir}", file=sys.stderr)
            rc = 1
            continue
        if not (run_dir / "ckpt.pt").exists():
            print(f"SKIP {arm}: no ckpt.pt in {run_dir}", file=sys.stderr)
            rc = 1
            continue

        res = json.loads(res_path.read_text()) if res_path.exists() else {}
        card = build_card(arm, role, repo_id, res)
        (run_dir / "README.md").write_text(card, encoding="utf-8")

        mf1 = _fmt(res.get("cra_macro_f1" if arm != "baseline" else "lora_macro_f1")
                   or res.get("cra_macro_f1"))
        print(f"\n== {arm:8s} -> {repo_id}")
        print(f"   dir      : {run_dir}")
        print(f"   macro-F1 : {mf1}   active_heads={res.get('active_heads','?')}   gpu={res.get('gpu','?')}")

        if args.dry_run:
            print("   (dry-run: would create repo + upload README.md, ckpt.pt, results.json)")
            continue

        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(run_dir),
            allow_patterns=ALLOW,
            commit_message=f"{arm}: macro-F1 {mf1}, {res.get('active_heads','?')} active heads (ADA fp16)",
        )
        print(f"   pushed -> https://huggingface.co/{repo_id}")

    print("\nDone." if rc == 0 else "\nDone (with skips -- see stderr).")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
