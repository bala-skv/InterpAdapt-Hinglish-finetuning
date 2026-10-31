# Stage 2 on the linguistic concept set: running the study

This runs the Circuit Routing Adapter (LoRA gated by head / MLP-layer causal scores) on
SAIL-2017 Hinglish sentiment, using the scores from the linguistic concept set
(`results/concepts/linguistic_v1_ada`, 39 concepts passing the pre-specified validation
rules), and compares every arm at matched budget against random and uniform controls, with
several seeds. Everything below is run on ADA from the repository root.

**What "MLP" means here.** Our MLP patching scores a whole layer's MLP output (28 numbers),
not individual neurons, so MLP routing is per *layer*: a LoRA on `down_proj` of the chosen
layers. Neuron-level routing would need neuron-level patching first; it is not done.

## 0. What will be run

Budget per run is about 45 minutes on an RTX 2080 Ti (33 min training at 3.3 s/step, about
12 min of evaluation, model load). Tiers can be submitted separately.

| Tier | Arm | Seeds | What it tests |
|---|---|---|---|
| 1 | `H-uniform` | 0,1,2 | full LoRA on q/o (reference) |
| 1 | `H-topk112` | 0,1,2 | **primary**: 112 highest-scoring heads |
| 1 | `H-random112` | 0,1,2 | **primary control**: 112 random heads (random set differs per seed) |
| 1 | `H-soft` | 0 | soft routing |
| 2 | `M-topk3`, `M-random3` | 0,1,2 | 3 MLP layers (down_proj): top-scoring vs random |
| 2 | `M-uniform`, `M-soft` | 0 | MLP references (all 28 layers; soft) |
| 3 | `HM-topk`, `HM-random` | 0,1,2 | 112 heads + 3 MLP layers together |
| 4 | `H-topk34`, `H-random34` | 0,1,2 | about 10% of heads (the regime where LoFiT finds selection matters) |

30 runs, about 22 GPU-hours; with 6 jobs in parallel about 4 hours. Tier 1 alone is 10 runs.

## 1. One-time setup

```bash
cd ~/babyshark2 && git pull --rebase origin main      # or copy the files from the zip
conda activate anlp
pip install --upgrade wandb huggingface_hub           # if `python -c "import wandb"` fails
python -m unittest discover -s tests                  # expect OK (GPU-free)
```

**Weights & Biases.** Create/login at https://wandb.ai, then:

```bash
wandb login                     # paste the key from https://wandb.ai/authorize
python - <<'EOF'
import wandb; print("W&B user:", wandb.Api().viewer.username, "| teams:", wandb.Api().viewer.teams)
EOF
export WANDB_ENTITY=<your username or team shown above>   # runs will appear under it
```

Use an entity you control; the old config pointed at Jawed's entity. In W&B, open the project
`babyshark-cra-ling` after the first run and set Settings -> Privacy to **public** (or add the
TAs). The course asks for the link in the README.

**Hugging Face.** Create a token with *write* access at https://huggingface.co/settings/tokens:

```bash
huggingface-cli login           # paste the token (or: hf auth login)
huggingface-cli whoami          # note the user name; this is your --owner
```

**Cluster settings** (copy what worked for Daniel's `run_concepts.slurm`; the values below are his):

```bash
export SBATCH_ACCOUNT=research SBATCH_QOS=medium SBATCH_CONSTRAINT=2080ti
export CONDA_ENV=anlp
```

`SBATCH_CONSTRAINT=2080ti` matters: every arm must run on the same GPU model (fp16 numbers
differ slightly between GPUs). The summary script warns if they do not.

## 2. Check the score files (CPU, 1 minute)

```bash
python scripts/stage2/build_scores_from_run.py --run results/concepts/linguistic_v1_ada \
    --keep results/concepts/linguistic_v1_ada/orig_rules/keep.json --name ling39
```

Expect: top heads L22H0, L21H2, L22H6; top layers L17, L22, L23; "rank-112 head score +0.00172".
This (re)creates `data/stage2_scores/ling39_heads.json` and `ling39_mlp.json`, which the config reads.

## 3. Smoke test (about 10 minutes), do not skip

Runs every code path (data download, model, W&B, evaluation, checkpoint) for 20 steps:

```bash
SMOKE=1 TIER=1,2 SEEDS="0" bash scripts/stage2/launch_study.sh       # submits tier 1+2, seed 0
squeue -u $USER | grep cra-
```

When they finish (`ls runs/smoke/*/DONE`): `grep RESULT runs/smoke/*/run.log` must show one line per
arm with `effective_params` > 0 and `gpu=...2080 Ti`; the W&B project must show the smoke runs with
loss curves. Numbers from smoke runs mean nothing (20 steps, 64 examples).
If anything fails, fix it now: see section 8. Then delete `runs/smoke`.

## 4. Launch

```bash
TIER=1 bash scripts/stage2/launch_study.sh           # core comparison, 10 jobs
# after it is queued, add the others (re-running the launcher never duplicates finished runs):
TIER=2,3,4 bash scripts/stage2/launch_study.sh
```

`DRYRUN=1` prints the jobs without submitting. Jobs chain themselves: if one hits the wall-time
limit it resubmits and resumes from its checkpoint. If a job *fails*, fix the cause and re-run
the same launcher command: finished runs are skipped, unfinished ones resume.

## 5. Monitor

```bash
squeue -u $USER | grep cra-
grep -h RESULT runs/ling_study/*/run.log             # one line per finished run
tail -n 3 runs/ling_study/H-topk112-s0/run.log       # step 10/600 ... loss
```

Healthy run: loss falls from about 2.4 to about 0.5 in the first 50 steps; eval accuracy at step 200
is clearly above the zero-shot base (about 0.36). In W&B compare `train/loss` and `eval/*` across arms.

## 6. Summarise (CPU)

```bash
python scripts/stage2/summarize_study.py --study ling_study
cat results/stage2/ling_study/summary.md
```

It prints each arm's macro-F1 (mean ± sd over seeds, every seed shown), the parameter accounting, and
the paired comparisons (top-k minus random, same seed) with a bootstrap interval over validation
examples.

**Reading rules, fixed before the results exist (do not change them afterwards):**
1. *Primary question:* `H-topk112` against `H-random112`. We say the method works **only if** all
   three per-seed differences are positive **and** the bootstrap interval excludes 0.
2. Everything else is secondary (MLP, heads+MLP, k=34). Report all of it, including arms that fail.
3. Do not change k, learning rate or steps after seeing results; a rerun with other settings is a
   new, separately labelled experiment.
4. Mixed signs or an interval containing 0 means "not distinguishable", never "works".

## 7. Upload to Hugging Face

```bash
python scripts/stage2/push_adapters.py --study ling_study --owner <your-hf-user> --dry-run
python scripts/stage2/push_adapters.py --study ling_study --owner <your-hf-user>
```

One model repo per arm (`<owner>/babyshark-cra-<arm>`), one folder per seed, with a generated model
card (metrics, budget, GPU, W&B run ids). Run `--dry-run` first and check the list. Then add to the
repository README: the W&B project URL and the Hugging Face repo URLs. (Do not commit `runs/`:
checkpoints are uploaded to the Hub instead.)

## 8. If something fails

| Symptom | Cause / fix |
|---|---|
| `wandb: ERROR api_key not configured` | run `wandb login`, or `export WANDB_MODE=offline` and later `wandb sync runs/ling_study/<run>/wandb/offline-run-*` |
| `wandb.init failed ... continuing without it` in `run.log` | the run still trains; fix login and `wandb sync` later |
| `Unknown keys` / `--set cra.k_totl: ... has no field` | typo in an override; the trainer refuses unknown names on purpose |
| `no space left` / HF cache | `export HF_HOME=/ssd_scratch/$USER/hf_cache` (the slurm script already does) |
| `CUDA out of memory` | submit again on a node with the same GPU model; do not lower batch size (it changes the comparison) |
| jobs pending for hours | check `squeue --start`; reduce parallelism by launching one tier at a time |
| summary warns "different GPUs" | rerun the odd runs with `SBATCH_CONSTRAINT=2080ti` so all arms match |
| HF `401` / `403` | token lacks *write* access, or `--owner` is not yours |

## 9. What goes into the report

`results/stage2/ling_study/summary.md` and `summary.json` replace the single-seed Table 5 numbers.
The Stage 2 section and the discussion must then be rewritten to match the outcome under the reading
rules above. `scripts/stage2/summarize_study.py` and the run folders are the source of every number.
