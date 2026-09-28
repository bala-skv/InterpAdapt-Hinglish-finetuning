# Phase 0 — Setup & De-risking

Runnable framework for **Phase 0** of *Interpretability-Guided Circuit Routing for
Hindi–English LoRA Fine-Tuning*. It de-risks the project **before** any Stage-1
tracing or Stage-2 training by validating the four things that most often sink it:

| Goal | Script | Answers |
|---|---|---|
| 🔴 Pin & verify the **base** checkpoint | `scripts/verify_model.py` | Is it Qwen2.5-1.5B *base* (not `-Instruct`)? Does the GQA layout match (28 layers, 12 q-heads, 2 kv-groups)? |
| 🔴 **Memory pilot** | `scripts/memory_pilot.py` | Does the 4–5 GB budget hold across `{4bit,8bit,bf16} × {256,512} × rank{4,8}`, measured **separately** for tracing vs. training? |
| 🟡 **Hook smoke test** | `scripts/hook_smoke_test.py` | Does per-head activation hooking work **under quantization**, with correct GQA shapes? |
| ⚙️ **Reproducibility** | `circuit_routing/seeding.py` | Fixed seeds (≥3), deterministic kernels, config snapshot per run. |

Run everything at once:

```bash
python scripts/run_phase0.py --config configs/phase0.yaml
```

## Stage-2 LoRA baseline — the "one real number"

The uniform-LoRA fine-tuning baseline the interpretability-guided Circuit-Routing
Adapter will be measured against. It fine-tunes the pinned **fp16** base with a
uniform LoRA on `q_proj` + `o_proj` (the head-aligned projections the per-head
routing mask will target) for English→Hindi translation, then reports one
comparable number: **base vs. LoRA mean per-token teacher-forced log-probability**
of the Hindi target on a held-out split (same metric as Stage-1 tracing).

```bash
# Cluster (self-chaining, resumable, saves gracefully before the ~6h kill):
sbatch cluster/train_lora_baseline.slurm

# Or directly (grep the log for the line starting `RESULT`):
python scripts/train_lora_baseline.py --config configs/lora_baseline.yaml
python scripts/train_lora_baseline.py --config configs/lora_baseline.yaml \
    --resume runs/lora-baseline-<timestamp>
```

Data source (config `data:` section): point `jsonl_train` at a local
`{"en","hi"}`-per-line file — this is where the finetuning task split drops in —
otherwise it falls back to the `cfilt/iitb-english-hindi` parallel corpus.
Prefetch both the base weights and the dataset on a login node before submitting
(the GPU node runs offline); see the NOTE in the SLURM template.


## Environment

This code targets a **single small GPU on the cluster** (CUDA + bitsandbytes).
It is not expected to run on the locked-down workstation (no local torch). Install
on the cluster:

```bash
pip install -e .          # or: pip install -r requirements.txt
```

## Layout

```
circuit_routing/
  configs/phase0.yaml          # single source of truth for the pilot grid
  circuit_routing/
    config.py                  # dataclasses + YAML loader
    seeding.py                 # set_seed + determinism
    logging_utils.py           # run dirs, logger, JSON/CSV result writers
    model_loading.py           # load + verify base checkpoint, quantization
    memory.py                  # peak-VRAM measurement utilities
    hooks.py                   # attention-head hooking (GQA-aware)
  scripts/
    verify_model.py            # checkpoint identity + structural report
    memory_pilot.py            # the go/no-go memory sweep
    hook_smoke_test.py         # hooking under quantization
    run_phase0.py              # orchestrates all checks + go/no-go summary
```

## Go / No-go

`run_phase0.py` exits non-zero if any gate fails, so it can gate CI or a cluster
job before Stage 1 begins:

- **base checkpoint** verified (not instruct),
- **at least one** pilot configuration fits under `pilot.vram_budget_gb` for
  both tracing and training,
- **hooks** capture per-head activations with the expected `[B, T, H, d]` shapes.
