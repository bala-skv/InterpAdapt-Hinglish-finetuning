# Refactored branch handoff

Suggested branch name:

`refactor/repository-architecture`

## What this branch does

This branch converts the development-oriented repository into a technical
architecture while preserving the existing implementation and generated
research evidence. Active code is organized under `src/`, `scripts/`,
`configs/`, `cluster/`, `data/`, `results/`, and `figures/`.

Superseded implementations are retained under `legacy/` for rollback and
provenance. They are not part of the active execution path.

## What was intentionally not changed

- Model identity or pinned revision
- Stage 1 tracing data contents
- Stage 1 scoring protocol
- Stage 1 patching algorithm
- Stage 2 adapter algorithm
- Experiment hyperparameters
- Existing retained result files

## Validation already performed

- Active Python syntax compilation: PASS
- Active shell/SLURM syntax checks: PASS
- Dependency-free repository contract tests: PASS (4/4)
- Existing masked-LoRA unit tests: PASS
- Tracing-set generator reproduces the frozen JSON byte-for-byte: PASS
- Key retained Stage 1 result hashes preserved: PASS

## Required before merge

Run the canonical entry points on the target ADA environment. At minimum:

```bash
pip install -e .
python scripts/phase0/run_phase0.py --config configs/phase0.yaml
python scripts/stage1/validate_tracing_set_v1.py
```

Then run the agreed Stage 1 sanity/full-trace path and compare the resulting
artifacts against the pre-refactor baseline.

Do not merge to `main` until the target-environment checks pass.
