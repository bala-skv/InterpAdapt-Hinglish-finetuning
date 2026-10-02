# Refactor validation record

## Scope

This branch reorganizes the repository without intentionally changing the
research algorithms, model checkpoint, data, metrics, or experiment settings.

## Static checks completed

- Python byte-compilation: PASS for all active `src/`, `scripts/`, and `cluster/` Python files.
- Shell syntax validation: PASS for all active `.sh` and `.slurm` files.
- Repository contract tests: PASS (4 tests).
- Masked-LoRA unit tests: PASS (all existing tests).
- Tracing-set generator reproduction: PASS; regenerated JSON is byte-identical to the checked-in tracing set.
- Frozen Stage 1 trace artifact: SHA-256 preserved from the pre-refactor snapshot.
- Frozen Stage 1 head-score artifact: SHA-256 preserved from the pre-refactor snapshot.

## Runtime validation boundary

The available validation environment does not contain the project's full
Transformers/ADA GPU stack, so full model execution cannot be claimed from this
artifact build. Before merging into the main branch, run the canonical Phase 0
and Stage 1 entry points on the target environment and compare the outputs with
the pre-refactor baseline.

## Merge gate

Do not merge this branch into `main` until the target-environment runtime checks
pass. The existing `main` branch is the rollback baseline.
