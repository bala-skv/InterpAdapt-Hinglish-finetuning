# InterpAdapt: Interpretability-Guided Circuit Routing for Hindi–English LoRA Fine-Tuning

Course project (BabyShark team). We localize the attention heads that causally
carry Hindi ↔ English behaviour in **Qwen2.5-1.5B (base)**, then route a LoRA
update through those heads — a **Circuit-Routing Adapter (CRA)** whose per-head
mask acts *inside* the LoRA update — and test it against matched-budget controls
on a code-mixed (Hinglish) task.

- **Stage 1 — Causal localization:** activation patching across all 336 query
  heads over three language/script conditions (English, Romanized Hindi,
  Devanagari) to separate *language* from *script*.
- **Stage 2 — Circuit-Routing Adapter:** soft-scaling / hard-top-k LoRA gated by
  the Stage-1 per-head scores, compared at matched budget against uniform LoRA,
  a random-head mask, and an inverse-causal mask.

## Stage-2 baseline (the "one real number")

Uniform LoRA (rank 8 on `q_proj`+`o_proj`, 0.089% of params, fp16, 600 steps) on
the frozen SAIL-2017 Romanized sentiment **validation** split (n=1260). Primary
metric is **macro-F1**. The base model is at chance; a tiny LoRA nearly doubles it —
this is the bar the Circuit-Routing Adapter must beat at matched budget.

| Metric | Base Qwen2.5-1.5B | + Uniform LoRA | Δ |
|---|---|---|---|
| Accuracy | 0.358 | **0.630** | +0.272 |
| Macro-F1 | 0.319 | **0.607** | +0.288 |

([W&B run `mn88apmt`](https://wandb.ai/phdiiitmohammed-iiit-hyderabad/babyshark-cra/runs/mn88apmt))

## Links (submission guidelines §3)

| Resource | Link |
|---|---|
| **Code (this repo)** | https://github.com/bala-skv/InterpAdapt-Hinglish-finetuning |
| **W&B — training runs** | https://wandb.ai/phdiiitmohammed-iiit-hyderabad/babyshark-cra |
| **W&B — LoRA baseline run** | https://wandb.ai/phdiiitmohammed-iiit-hyderabad/babyshark-cra/runs/mn88apmt |
| **HF dataset — SAIL-2017 Hinglish (Stage 2 task)** | https://huggingface.co/datasets/satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2 |
| **HF model — base checkpoint (pinned)** | https://huggingface.co/Qwen/Qwen2.5-1.5B (revision `8faed761d45a263340a0528343f099c05c9a4323`) |
| **HF adapters — trained LoRA / CRA** | _added after the runs finish (`model.save_pretrained` output pushed per arm)_ |

## Repository layout

| Path | Owner | Contents |
|---|---|---|
| [`phase0/`](phase0/) | Jawed | Phase-0 setup & de-risking: verify base checkpoint, memory pilot, hook smoke test. |
| [`bala/`](bala/) | Bala | Tracing-set construction + sequence scoring for Stage 1. |
| [`interp/`](interp/), [`stage1/`](stage1/) | Daniel | Activation-patching harness, sanity gate, heatmaps, score correlations. |
| [`jawed/`](jawed/) | Jawed | Stage-2 adapter surface (`MaskedLoRALinear`), LoRA baseline, CRA-vs-controls runner, SLURM. |

## Reproduce

Everything runs on a single small GPU on the cluster (ADA). It is not expected to
run on a locked-down workstation (no local `torch`).

```bash
# 1. Environment (from the jawed/ sub-project)
cd jawed
pip install -e .            # or: pip install -r requirements.txt

# 2. Phase-0 gates (verify base + memory pilot + hooks) -> GO/NO-GO
python scripts/run_phase0.py --config configs/phase0.yaml

# 3. Stage-2 uniform-LoRA baseline (the "one real number")
sbatch cluster/train_lora_baseline.slurm

# 4. Stage-2 CRA vs. matched-budget controls (one job per arm)
MASK_MODE=soft    sbatch cluster/train_cra_compare.slurm
MASK_MODE=uniform sbatch cluster/train_cra_compare.slurm
MASK_MODE=topk    sbatch cluster/train_cra_compare.slurm
MASK_MODE=random  sbatch cluster/train_cra_compare.slurm
```

Set `WANDB_API_KEY` (or `wandb login`) before submitting so runs log to the W&B
project above. `soft`/`topk`/`random` need Daniel's Stage-1 per-head scores at
`jawed/data/stage1_head_scores.json` (`{"scores": [[...], ...]}`, shape `[28][12]`).

### Environment & version parity

**Every number in this repo — Bala/Daniel's Stage-1 traces and Jawed's Stage-2
LoRA/CRA runs — was produced on ADA in `fp16`**, so the compute-capability axis
(RTX 2080 Ti / GTX 1080 Ti, `sm_75`/`sm_61`) is held fixed across the team and
no `bf16`/TF32 numerics leak in. The software stack is pinned two ways:

- **Floors** (`jawed/requirements.txt`): `torch>=2.1`, `transformers>=4.44,<5`
  (5.x renamed `from_pretrained(torch_dtype=)`→`dtype=`, which `model_loading.py`
  relies on), `peft>=0.12`, `bitsandbytes>=0.43`, `datasets>=2.14`.
- **Exact lock** (`jawed/requirements-lock.txt`): the frozen versions of the ADA
  `torch310` conda env the reported runs actually used (`torch 2.5.1+cu121`,
  `transformers 4.57.x`, `peft 0.21.x`, `bitsandbytes 0.50.x`, Python 3.10).
  Regenerate with `pip freeze > jawed/requirements-lock.txt` on the GPU node.

The base checkpoint is additionally pinned by **revision**
(`8faed761d45a263340a0528343f099c05c9a4323`), so tokenizer/weights are identical
regardless of the transformers version in the resolved range.

## Team

Bala · Daniel · Jawed · Satyam — IIIT Hyderabad.
