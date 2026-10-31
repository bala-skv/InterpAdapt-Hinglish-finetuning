# Repository architecture

The repository is organized by technical responsibility by
technical responsibility.

## Canonical layout

- `src/circuit_routing/`: model, adapter, data, memory, seeding, and logging primitives.
- `src/interp_adapt/tracing/`: Stage 1 scoring and activation-patching primitives.
- `scripts/phase0/`: Phase 0 verification, hook smoke test, and memory pilot entry points.
- `scripts/stage1/`: tracing-set validation, causal tracing, and Stage 1 analysis.
- `scripts/stage2/`: LoRA/CRA training and Stage 1 score export for Stage 2.
- `configs/`: experiment configuration files.
- `cluster/`: SLURM/environment helpers.
- `data/`: frozen data references and small checked-in datasets.
- `results/`: machine-generated experiment outputs retained as evidence.
- `figures/`: presentation-ready figures.
- `legacy/`: superseded development implementations retained for provenance only.

## Pipeline

Qwen2.5-1.5B base checkpoint -> Stage 1 tracing -> per-head causal scores ->
Stage 2 circuit-routing/LoRA comparison.
