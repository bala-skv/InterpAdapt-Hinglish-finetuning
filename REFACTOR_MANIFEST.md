# Repository refactor manifest

## Scope

This branch reorganizes the repository around technical responsibilities while
preserving the existing experimental implementations and results.

## Functional invariants

- No model checkpoint or revision changes.
- No dataset contents are changed.
- No experimental hyperparameters are intentionally changed.
- No metric is changed.
- Stage 1 remains fp16 and uses the existing tracing/patching implementation.
- Stage 2 uses the existing circuit-routing implementation.
- Existing generated results are preserved.

## Canonical active locations

- `src/circuit_routing/`
- `src/interp_adapt/tracing/`
- `scripts/phase0/`
- `scripts/stage1/`
- `scripts/stage2/`
- `configs/`
- `cluster/`
- `data/`
- `results/`
- `figures/`

## Legacy policy

Superseded/duplicated implementations are retained under `legacy/` and are not
part of the active execution path. They are preserved for rollback and
provenance rather than deleted.

## Validation status

Static syntax validation is required before merge. GPU execution validation
must be performed on the project's target environment before the branch is
merged into the main branch.
