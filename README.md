# InterpAdapt: Interpretability-Guided Circuit Routing for Hindi-English LoRA Fine-Tuning

InterpAdapt studies whether causal activation-patching signals can identify
language-specific computation in Qwen2.5-1.5B and use those signals to route
parameter-efficient adaptation for Hindi-English code-mixed NLP.

## Research pipeline

### Stage 1 — Causal localization

Activation patching is evaluated across three matched conditions:

- English (`en`)
- Romanized Hindi (`hi_latn`)
- Devanagari Hindi (`hi_deva`)

The primary language contrast is `en <-> hi_latn`, where script is held
constant. The `hi_deva <-> hi_latn` contrast isolates script while language is
held constant. The `en <-> hi_deva` contrast contains both effects.

The current tracing set contains 38 validated concepts. The Stage 1 scoring
interface uses mean teacher-forced full-sequence target log-probability, and the
patch position is derived from the scoring interface.

### Stage 2 — Circuit-Routing Adapter

Stage 2 contains the LoRA baseline and matched-budget routing controls. The
Stage 1 per-head scores provide the causal signal used by the routing masks.

## Repository structure

```text
src/
  circuit_routing/           Core model, data, adapter and experiment utilities
  interp_adapt/tracing/      Stage 1 scoring and patching primitives
scripts/
  phase0/                    Phase 0 verification and memory gates
  stage1/                    Tracing, validation and analysis entry points
  stage2/                    LoRA/CRA training and score export
configs/                     Experiment configurations
cluster/                     SLURM and cluster environment helpers
data/                        Frozen small data artifacts and dataset references
results/                     Retained machine-generated results
figures/                     Presentation-ready figures
docs/                        Architecture, data and reproducibility documentation
legacy/                      Superseded implementations retained for provenance
```

## Reproducibility

The base model is pinned to:

`Qwen/Qwen2.5-1.5B @ 8faed761d45a263340a0528343f099c05c9a4323`

Stage 1 tracing uses fp16 and the verified 28-layer / 12-query-head / 2-KV-group
Qwen layout. See `docs/reproducibility.md` for the locked experiment contract.

## Data

Stage 1 tracing data is checked into `data/tracing/`.

The Stage 2 SAIL-2017 Romanized sentiment artifact is hosted on Hugging Face:

https://huggingface.co/datasets/satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2

The downstream-data provenance and cleaning policy are documented in
`data/downstream/README.md` and the linked dataset artifact.

## Running

Install the repository in editable mode:

```bash
pip install -e .
```

### Phase 0

```bash
python scripts/phase0/run_phase0.py --config configs/phase0.yaml
```

### Stage 1 tracing validation

```bash
python scripts/stage1/validate_tracing_set_v1.py
```

### Stage 1 full trace

```bash
python scripts/stage1/full_trace.py --data data/tracing/tracing_set_v1.json
```

### Stage 1 analysis

```bash
python scripts/stage1/analyze_trace.py \
  --trace results/stage1/full_trace_v2.json \
  --out results/stage1/analysis_v2
```

### Stage 2

See the configuration files in `configs/` and the SLURM entry points in
`cluster/`.

## Experimental results

Retained results and figures are stored under `results/` and `figures/`.
Experimental numbers should always be interpreted together with their recorded
configuration and execution environment.

## Provenance and legacy code

The `legacy/` directory preserves the pre-refactor working tree for provenance
and rollback. It is not part of the canonical execution path.
