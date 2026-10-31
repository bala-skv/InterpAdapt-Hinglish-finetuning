#!/usr/bin/env python3
"""Upload finished Stage-2 runs to the Hugging Face Hub: ONE model repo per arm,
one subfolder per seed (seed0/, seed1/, ...).

    huggingface-cli login            # or: hf auth login   (needs a WRITE token)
    python scripts/stage2/push_adapters.py --study ling_study --owner <hf-user-or-org> --dry-run
    python scripts/stage2/push_adapters.py --study ling_study --owner <hf-user-or-org>

Only runs with a DONE marker are uploaded. Per run we upload ckpt.pt (the adapter
tensors plus optimiser state, the format the trainer resumes from), results.json,
config.json, budget.json, preds.json, run.log, and a copy of the score files the masks
were built from. A model card (README.md) is generated for each arm repo from the
results and records the arm, budget, seeds, metrics, GPU and the W&B run ids.

The CRA uses the project's own masked-LoRA layers, not a PEFT adapter, so the card says
how to reload it with this repository's code. Needs only huggingface_hub (no torch).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DATASET = "satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2"
GITHUB = "https://github.com/bala-skv/InterpAdapt-Hinglish-finetuning"
UPLOAD = ["ckpt.pt", "results.json", "config.json", "budget.json", "preds.json", "run.log"]

ROLE = {
    "H-uniform": "Uniform LoRA on q_proj and o_proj in all 28 layers (full-budget reference).",
    "H-topk112": "CRA: LoRA only on the 112 attention heads with the highest causal recovery (hard top-k).",
    "H-random112": "CRA control: LoRA on 112 randomly chosen heads (same budget as H-topk112).",
    "H-soft": "CRA: LoRA scaled per head by its normalised causal recovery (soft routing).",
    "H-topk34": "CRA: LoRA on the 34 highest-recovery heads (about 10% of heads).",
    "H-random34": "CRA control: LoRA on 34 random heads (same budget as H-topk34).",
    "M-uniform": "LoRA on down_proj of all 28 MLP layers (MLP reference).",
    "M-topk3": "CRA: LoRA on down_proj of the 3 MLP layers with the highest causal recovery.",
    "M-random3": "CRA control: LoRA on down_proj of 3 random MLP layers.",
    "M-soft": "CRA: down_proj LoRA scaled per layer by normalised MLP recovery.",
    "HM-topk": "CRA: top-112 heads plus the top-3 MLP layers.",
    "HM-random": "CRA control: 112 random heads plus 3 random MLP layers.",
    "HM-topk34": "CRA: top-34 heads (about 10%) plus the top-3 MLP layers.",
    "HM-random34": "CRA control: 34 random heads plus 3 random MLP layers.",
}


def f(x, nd=4):
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def card(arm: str, repo_id: str, runs: dict, study: str, project: str, entity: str | None) -> str:
    seeds = sorted(runs)
    r0 = runs[seeds[0]]["res"]
    cfg = runs[seeds[0]]["cfg"].get("cra", {})
    rows = "\n".join(
        f"| {s} | {f(runs[s]['res']['cra_acc'])} | {f(runs[s]['res']['cra_macro_f1'])} | "
        f"{runs[s].get('wandb_id') or 'n/a'} |" for s in seeds)
    f1s = [runs[s]["res"]["cra_macro_f1"] for s in seeds]
    mean = sum(f1s) / len(f1s)
    wb = (f"https://wandb.ai/{entity}/{project}" if entity else f"W&B project `{project}`")
    return f"""---
library_name: pytorch
base_model: {BASE_MODEL}
language:
- hi
- en
tags:
- lora
- circuit-routing-adapter
- interpretability
- hinglish
- sentiment-analysis
- mask:{r0.get('mask_mode')}
datasets:
- {DATASET}
metrics:
- f1
- accuracy
---

# {repo_id.split('/')[-1]}

{ROLE.get(arm, arm)}

Part of **InterpAdapt** (BabyShark team, IIIT Hyderabad): interpretability-guided circuit
routing for Hindi-English LoRA fine-tuning of **{BASE_MODEL}** (fp16). Task: SAIL-2017
Hinglish sentiment, 3 classes. Head and MLP-layer scores come from activation patching on the
linguistically sampled concept set (`ling39`).

## Results (validation, n={r0.get('n_eval')}, {r0.get('gpu')})

Zero-shot base model macro-F1: {f(r0.get('base_macro_f1'))}.

| Seed | Accuracy | Macro-F1 | W&B run id |
|---|---|---|---|
{rows}

Mean macro-F1 over {len(seeds)} seed(s): **{mean:.4f}**.

- Mask mode `{r0.get('mask_mode')}`, tag `{r0.get('tag')}`; active attention heads: {r0.get('active_heads')};
  MLP wrappers: {r0.get('mlp_wrappers')}; effective trainable parameters: {r0.get('effective_params')}.
- Rank {cfg.get('rank', 8)}, alpha {cfg.get('alpha', 16)}, dropout {cfg.get('dropout', 0.05)}, 600 steps,
  effective batch 32, lr 1e-4, max length 256. Study `{study}`; logged to {wb}.

## Files (per seed folder)

`ckpt.pt` (adapter state + optimiser/scheduler; resume format), `results.json`, `config.json`,
`budget.json`, `preds.json` (per-example gold and predictions), `run.log`, score files used.

## How to load

Not a PEFT adapter: it uses the project's `MaskedLoRALinear` layers
([code]({GITHUB})). Build the model exactly as `scripts/stage2/train_cra_compare.py` does
(`build_cra_model` with this folder's `config.json`), then load `ckpt["adapter"]`
(see `load_checkpoint` in that script).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default="ling_study")
    ap.add_argument("--owner", required=True, help="HF user or organisation that owns the repos")
    ap.add_argument("--prefix", default="babyshark-cra", help="repo name prefix: <prefix>-<arm>")
    ap.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--only", nargs="+", help="only these arms (e.g. H-topk112 H-random112)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    study_dir = a.runs_root / a.study
    by_arm: dict[str, dict] = defaultdict(dict)
    for d in sorted(study_dir.iterdir()):
        m = re.fullmatch(r"(.+)-s(\d+)", d.name)
        if not (d.is_dir() and m and (d / "DONE").is_file()):
            continue
        res = json.loads((d / "results.json").read_text())
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").is_file() else {}
        wid = (d / "wandb_run_id").read_text().strip() if (d / "wandb_run_id").is_file() else None
        by_arm[m.group(1)][int(m.group(2))] = {"dir": d, "res": res, "cfg": cfg, "wandb_id": wid}
    if a.only:
        by_arm = {k: v for k, v in by_arm.items() if k in a.only}
    if not by_arm:
        print("nothing to upload (no finished runs match)", file=sys.stderr)
        return 1

    api = None
    if not a.dry_run:
        from huggingface_hub import HfApi, whoami
        try:
            print(f"[auth] logged in as {whoami().get('name')}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: not logged in ({exc}). Run `huggingface-cli login` with a WRITE token.", file=sys.stderr)
            return 2
        api = HfApi()

    for arm, runs in sorted(by_arm.items()):
        repo_id = f"{a.owner}/{a.prefix}-{arm}"
        any_cfg = next(iter(runs.values()))["cfg"]
        wb = any_cfg.get("wandb", {})
        readme = card(arm, repo_id, runs, a.study, wb.get("project", "babyshark-cra-ling"), wb.get("entity"))
        print(f"\n== {arm}: {len(runs)} seed(s) -> {repo_id}")
        for s in sorted(runs):
            print(f"   seed{s}: macro-F1 {f(runs[s]['res']['cra_macro_f1'])}  ({runs[s]['dir']})")
        if a.dry_run:
            continue
        api.create_repo(repo_id, repo_type="model", private=a.private, exist_ok=True)
        stage = a.runs_root / a.study / f"_hf_stage_{arm}"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        (stage / "README.md").write_text(readme, encoding="utf-8")
        for s, r in runs.items():
            dst = stage / f"seed{s}"
            dst.mkdir()
            for name in UPLOAD:
                if (r["dir"] / name).is_file():
                    shutil.copy2(r["dir"] / name, dst / name)
            for key in ("scores_file", "mlp_scores_file"):
                p = REPO_ROOT / (r["cfg"].get("cra", {}).get(key) or "")
                if p.is_file():
                    shutil.copy2(p, dst / p.name)
        api.upload_folder(repo_id=repo_id, folder_path=str(stage),
                          commit_message=f"{arm}: {len(runs)} seed(s), study {a.study}")
        shutil.rmtree(stage)
        print(f"   pushed -> https://huggingface.co/{repo_id}")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
