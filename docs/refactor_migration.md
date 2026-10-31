# Refactor migration map

This branch converts the development-oriented tree into a technical architecture.
The active implementation is now organized by package, stage, configuration,
cluster tooling, data, results, and documentation.

## Active implementation

| Previous location | Canonical location | Treatment |
|---|---|---|
| `jawed/circuit_routing/` | `src/circuit_routing/` | moved without algorithmic changes |
| `bala/model_loading.py` | `src/interp_adapt/tracing/model_loading.py` | canonical tracing loader |
| `bala/sequence_scoring.py` | `src/interp_adapt/tracing/sequence_scoring.py` | canonical Stage 1 scorer |
| `interp/patching.py` | `src/interp_adapt/tracing/patching.py` | canonical patching implementation |
| `interp/full_trace_v2.py` | `scripts/stage1/full_trace.py` | canonical v2 entry point |
| `interp/analyze_trace_v2.py` | `scripts/stage1/analyze_trace.py` | canonical analysis entry point |
| `bala/build_tracing_set_v1.py` | `scripts/stage1/build_tracing_set_v1.py` | canonical data builder |
| `bala/validate_tracing_set_v1.py` | `scripts/stage1/validate_tracing_set_v1.py` | canonical validator |
| `jawed/scripts/` | `scripts/phase0/` and `scripts/stage2/` | split by technical stage |
| `jawed/configs/` | `configs/` | canonical configuration directory |
| `jawed/cluster/` | `cluster/` | canonical cluster tooling |
| `bala/data/tracing_set_v1.json` | `data/tracing/tracing_set_v1.json` | frozen Stage 1 data |
| `interp/runs/analysis_v2/` | `results/stage1/analysis_v2/` | retained evidence |
| `jawed/data/stage1_head_scores.json` | `results/stage1/stage1_head_scores.json` | retained derived score artifact |

## Archived implementation

Older Stage 1 implementations, diagnostics, the old Phase 0 tree, and other
superseded working files are retained under `legacy/`. They are not imported by
the canonical entry points.

## Compatibility policy

The old owner-oriented paths are intentionally not kept as active compatibility
entry points on this branch. The main branch remains untouched and is the rollback
path. Before merging this branch, the canonical entry points below must be run on
the target environment.

## Canonical entry points

- `python scripts/phase0/run_phase0.py --config configs/phase0.yaml`
- `python scripts/phase0/verify_model.py --config configs/phase0.yaml`
- `python scripts/phase0/hook_smoke_test.py --config configs/phase0.yaml --quant 4bit`
- `python scripts/phase0/memory_pilot.py --config configs/phase0.yaml`
- `python scripts/stage1/validate_tracing_set_v1.py`
- `python scripts/stage1/full_trace.py --data data/tracing/tracing_set_v1.json`
- `python scripts/stage1/analyze_trace.py --trace results/stage1/full_trace_v2.json`
- `python scripts/stage2/train_lora_baseline.py --config configs/lora_baseline.yaml`
- `python scripts/stage2/train_cra_compare.py --config configs/cra_compare.yaml --mask-mode soft`
