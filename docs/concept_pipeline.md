# Running the Stage 1 study on a new concept set

`scripts/pipeline/run_concepts.py` runs the whole Stage 1 study on any concept
set in the tracing-set format, with one command:

- **single heads**: activation patching of all 336 attention heads on the three
  contrasts, with bootstrap intervals, the family-wise test, leave-one-out and
  heatmaps (`full_trace.py`, `analyze_trace.py`);
- **MLP layers**: the same patching for each layer's MLP output (Jawed's
  `--component mlp`), with bootstrap intervals and family-wise tests over the
  28 layers;
- **head interactions**: the joint patching curve, the necessity sweep, and pair
  interactions among a shortlist of heads (`entangle.py`), with both
  single-head baselines and both family-wise tests;
- **a report**: `REPORT.md`, which summarises every finished step, including
  which heads lead each category.

It reuses the team's scripts unchanged in method, so results on a new set are
directly comparable with the 38-concept results in `results/stage1/`.

## 1. Prepare the concept file

A JSON list, one object per concept, exactly like
`data/tracing/tracing_set_v1.json` (see `data/tracing/concepts_template.json`):

| Field | Meaning |
|---|---|
| `concept_id` | unique; letters, digits, `_` and `-` only |
| `category` | the group the concept belongs to (for a linguistic study, the feature it probes) |
| `en_prompt`, `en_target` | English sentence and the single word that completes it |
| `hi_deva_prompt`, `hi_deva_target` | the same in Devanagari Hindi |
| `hi_latn_prompt`, `hi_latn_target` | the same in Romanized Hindi |

Rules the `check` step enforces or warns about:

- the three targets must differ (identical English and Romanized targets give a
  contrast of exactly zero, which is why `school_study` was replaced);
- prompts end with a space and targets do not start with one, as in v1;
- at least about 30 concepts in total, and at least 5 per category, or the
  statistics (overall and per category) have little power.

The `validate` step then requires, as in v1, a mean per-token log-probability of
at least -8 for each target on its own prompt and a contrast of at least 0.5 on
every contrast. It stops and lists failing concepts; rewrite them, or pass
`--drop-unusable` to continue without them (the report lists what was dropped).

## 2. Run it

All commands from the repository root, in the `anlp` environment, **on an
RTX 2080 Ti node**. The pipeline refuses other GPUs by default: the identity
self-test is exact on a 2080 Ti and fails on a GTX 1080 Ti.

**a. Plan and estimate (no GPU needed):**

    python scripts/pipeline/run_concepts.py --concepts data/tracing/my_set.json --name my_set --dry-run

**b. Smoke test on 3 concepts (about 15 minutes):**

    python scripts/pipeline/run_concepts.py --concepts data/tracing/my_set.json --name my_set --limit 3

This writes to `results/concepts/my_set__smoke3/`, so it can never be mistaken
for the real run. Check that it finishes and that `REPORT.md` there looks sensible.

**c. The real run:**

    nohup python scripts/pipeline/run_concepts.py --concepts data/tracing/my_set.json --name my_set \
        > logs/concepts_my_set.log 2>&1 &

or as a batch job (resubmit the same command if it hits the time limit):

    CONCEPTS=data/tracing/my_set.json NAME=my_set CONDA_ENV=anlp sbatch cluster/run_concepts.slurm

Every finished step is skipped when you run the command again, and the pairs
step resumes from the last finished concept, so after an interruption you simply
run the same command. `--force STEP` redoes a step; `--steps cpu` re-runs only
the analysis and report.

### GPU time (RTX 2080 Ti, one interaction contrast)

| Concepts | Heads step | Total, shortlist 30 | Total, shortlist 50 (default) | Total, shortlist 100 |
|---|---|---|---|---|
| 38 | 0.9 h | 2.1 h | 2.9 h | 6.3 h |
| 60 | 1.5 h | 3.3 h | 4.5 h | 9.9 h |
| 100 | 2.5 h | 5.4 h | 7.3 h | 16.5 h |
| 150 | 3.7 h | 8.0 h | 10.9 h | 24.6 h |

The heads and MLP steps are not resumable: each must finish within one
allocation (above about 230 concepts the heads step exceeds 6 hours). The pairs
step is resumable, so its length only means more resubmissions.
`--entangle-pairs en_hi-latn hi-deva_hi-latn en_hi-deva` runs the interaction
steps on all three contrasts (about three times the interaction cost).

## 3. Outputs

    results/concepts/<name>/
      REPORT.md                 summary of everything below
      manifest.json             steps, timings, GPU, git commit, concept-set hash
      concepts.json             frozen copy of the input (every step uses this)
      concepts_used.json        after validation (minus dropped concepts)
      validation.json
      heads_trace.json          per concept: recovery of all 336 heads, 3 contrasts
      heads_analysis/           summary.json and heatmaps
      mlp_trace.json            per concept: recovery of all 28 MLP layers
      mlp_analysis.json
      entangle/<contrast>/      curve, noise, shortlist, pairs, analysis (.json)
      logs/                     one log per step (not committed)
    figures/concepts/<name>/    heads/, mlp/, entangle/<contrast>/

The input file is frozen at the first run. If you edit it, the next run stops
rather than mixing old and new concepts: use a new `--name`, or `--force check`
and then `--force` the later steps.

## 4. Reading REPORT.md

- **Recovery** is the fraction of the clean-corrupted contrast restored by
  patching at the last prompt token.
- **Family-wise tests** control the chance of any false positive across all
  heads, layers or pairs tested. `max-|mean|` uses one threshold for every item
  and was the planned test for pairs; `max-|t|` standardises each item by its
  spread across concepts and was added after a few noisy early-layer pairs
  inflated the `max-|mean|` threshold. Report both.
- **Robust pairs** pass `max-|t|` under both single-head baselines. They are the
  conservative claim. A significant interaction means joint patching is
  non-additive for that pair on these concepts, not that a mechanism is shown;
  see the same-layer and curvature diagnostics next to it.
- **Per-category tables** (section 4 of the report) are descriptive, with no
  correction. For a linguistic study they are the place to look for heads that
  respond to one feature and not others; confirm such a finding with a contrast
  or concept set built for it before claiming it.
- **MLP and head scores are not directly comparable per unit**: one MLP output
  is a whole layer's feed-forward block, one head is a twelfth of a layer's
  attention.
