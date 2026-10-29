# Stage 1 extension: head interactions

Single-head patching assumes heads act independently. These experiments test
that assumption on the existing tracing set (`data/tracing/tracing_set_v1.json`,
38 concepts) and the language contrast (`en_hi-latn`) by default.

The scoring protocol is unchanged: every number goes through `_encode` and
`_logp_from_logits` in `scripts/stage1/full_trace.py`, patches the o_proj input
at the last prompt token, and uses fp16 activations with fp32 log-softmax.

## Quantities (per concept)

| Symbol | Definition | Meaning |
|---|---|---|
| R(S) | (m_patch(S) − m_corr) / (m_clean − m_corr) | sufficiency: copy clean activations of set S into the corrupted run |
| N(h) | (m_clean − m_noised(h)) / (m_clean − m_corr) | necessity: copy corrupted activations of head h into the clean run |
| I(A,B) | R({A,B}) − R({A}) − R({B}) | pair interaction: > 0 synergy, < 0 redundancy |

Single-head values in I can come from two places, and `analyze` always
reports both:

- `v2`: the committed v2 trace (`results/stage1/full_trace_v2.json`), measured
  earlier and possibly on a different GPU.
- `same-run`: the shortlist singles re-measured inside the `pairs` run, on the
  same GPU and in the same session as the pair measurements.

If the two baselines differ, the interaction estimate depends on the GPU, so
compare the number of significant pairs and the mean shift printed by
`analyze` before interpreting any pair. `--singles same-run` makes the
same-run values the primary baseline (default: `v2`). The random control pairs
only have v2 singles, because their singles are not re-measured.

## Why the reverse sweep

In the clean-into-corrupted direction, a synergistic pair is invisible to
single-head patching: neither head is sufficient alone. Removing either head
from the clean run does break the behaviour, so such heads have high necessity
and low sufficiency. The shortlist is therefore the union of the top heads by
sufficiency and the top heads by necessity.

## Steps

| Step | Command | Device | Approx. forward passes |
|---|---|---|---|
| 1 | `entangle.py curve` | GPU | 3,800 |
| 2 | `entangle.py noise` | GPU | 25,600 |
| 3 | `entangle.py shortlist`, then commit the file | CPU | 0 |
| 4 | `entangle.py pairs` | GPU | 30,400 |
| 5 | `entangle.py analyze` | CPU | 0 |
| 6 | `plot_entangle.py` | CPU | 0 |

GPU steps: `CMD=<step> CONDA_ENV=<env> sbatch cluster/entangle.slurm`.
Add `EXTRA="--limit 2"` for a smoke test. Outputs go to
`results/stage1/entangle/<command>_<pair>.json` with provenance (git commit,
GPU, library versions, model revision, data hash) and self-test results.

## Larger shortlists and resuming

`shortlist --size N` picks exactly N heads by alternating the next-best head by
sufficiency and by necessity (repeats skipped). The first 2k distinct heads
always contain the top k of each ranking, so a smaller shortlist is a subset of
a larger one. Excluded heads (`--exclude`) are removed from both rankings.
The command prints an estimate of the `pairs` run time (21.5 forward passes per
second measured on an RTX 2080 Ti).

| Heads | Pairs | Approx. `pairs` run |
|---|---|---|
| 19 | 171 | 22 min (measured, with 190 random pairs) |
| 50 | 1,225 | 1.4 h |
| 100 | 4,950 | 5.2 h |

`pairs` checkpoints after every concept to `<out>.partial.jsonl`. If the job
is killed or the allocation ends, run the same command again and it continues
from the next concept; a partial file from a different run (other shortlist,
seed or settings) is refused unless `--fresh` is given. The partial file is
deleted when the run completes. The family-wise correction covers all pairs in
the shortlist, so a larger shortlist has a higher threshold.

## Self-tests (every GPU command, first concept)

1. Re-injecting the target run's own activations into all 336 heads leaves
   m_corr unchanged (|diff| ≤ 1e-4).
2. Single-head set patches reproduce the v2 recovery of its top-3 heads
   (|diff| ≤ 2e-3; small differences can come from a different GPU).

## Statistics

- Bootstrap: 2,000 resamples of concepts, percentile intervals.
- Family-wise tests for pairs (two-sided sign-flip, one random sign per concept
  shared across all pairs, 2,000 permutations, alpha 0.05). Both are always run:
  - `max |mean|`: one threshold on the mean interaction for every pair. This
    was the planned test. Its threshold is set by the noisiest pairs, so a few
    high-variance pairs (early layers in the 100-head run) raise it for all
    others (0.0032 for 171 pairs, 0.0358 for 4,950 pairs).
  - `max |t|`: each pair is standardised by its own spread across concepts
    (Nichols and Holmes 2002). Added after seeing the `max |mean|` result on
    the 100-head run; report it together with the planned test, not instead.
  `analyze --stat t` lists the `max |t|` result as primary; both are saved.
- Both single-head baselines (see above) are reported. Pairs that are
  significant under both baselines are the conservative set.
- Control: interactions of 190 random pairs outside the shortlist (v2 singles).
- Diagnostics worth reporting: the correlation of I with R(A)R(B) (generic
  curvature), and the share of significant pairs that are same-layer pairs.

The shortlist is computed from single-head scores only (`shortlist` never
reads pair results), and the `pairs` output records its hash, so the
family-wise correction covers a candidate set that cannot depend on the pair
results.

## Figures

`plot_entangle.py` writes PNG and PDF files to `figures/stage1/entangle/`:
the joint curve (top-k joint recovery, the sum of the same heads' single
recoveries, and random k heads, with 95% bootstrap bands over concepts), a
necessity-versus-sufficiency scatter over all 336 heads, and the largest pair
interactions with the family-wise threshold and the random-pair range.

## Limits

- One contrast (language) unless run with `--pair`; 38 concepts; last prompt
  token only; clean-into-corrupted for pairs.
- Triples and larger sets are covered only through the joint curve.
- A significant interaction is a measured effect on these concepts, not a
  mechanism.
