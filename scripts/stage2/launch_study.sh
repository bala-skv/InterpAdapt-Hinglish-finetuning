#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Submit the Stage-2 study on the linguistic concept set: one SLURM job per
# (arm, seed). Safe to re-run: runs that already have a DONE marker are skipped,
# and unfinished runs resume from their checkpoint.
#
#   DRYRUN=1 TIER=1 bash scripts/stage2/launch_study.sh          # print only
#   SMOKE=1  TIER=1 SEEDS="0" bash scripts/stage2/launch_study.sh  # 20-step test
#   TIER=1   bash scripts/stage2/launch_study.sh                  # the core comparison
#   TIER=1,2,3,4 bash scripts/stage2/launch_study.sh              # everything
#
# Environment (all optional except CONDA_ENV):
#   CONDA_ENV      conda env with torch/transformers/wandb/peft        (required)
#   TIER           1,2,3,4 or "all"                       (default 1)
#   SEEDS          seeds per arm that has seeds           (default "0 1 2")
#   STUDY          run folder name under runs/            (default ling_study; smoke: smoke)
#   WANDB_ENTITY   W&B user/team to log under             (default: your W&B default)
#   CONFIG         trainer config                         (default configs/cra_ling.yaml)
#   DRYRUN=1       print the jobs, submit nothing
#   LOCAL=1        run jobs here, one after another, instead of sbatch (use inside an
#                  interactive GPU session). SHARD=i/n then takes every n-th job, so n
#                  sessions split the work:  LOCAL=1 SHARD=0/4 TIER=1 bash ... (session 1),
#                  SHARD=1/4 (session 2), ... Finished runs are skipped; reruns resume.
#   SMOKE=1        20 steps, 64 eval examples, W&B tagged "smoke"
#
# Tiers (arm: what it tests; "x3" = three seeds):
#   1  H-uniform x3, H-topk112 x3, H-random112 x3, H-soft x1     heads at 1/3 budget vs random vs full LoRA
#   2  M-topk3 x3, M-random3 x3, M-uniform x1, M-soft x1         MLP layers (down_proj LoRA)
#   3  HM-topk x3, HM-random x3                                  heads + MLP together
#   4  H-topk34 x3, H-random34 x3                                heads at ~10% (LoFiT regime)
#
# Cluster settings propagate to the self-resubmitted jobs through the environment:
#   export SBATCH_ACCOUNT=research SBATCH_QOS=medium SBATCH_CONSTRAINT=2080ti
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOCAL="${LOCAL:-0}"
SHARD="${SHARD:-}"
if [[ "${LOCAL}" != "1" ]]; then : "${CONDA_ENV:?set CONDA_ENV=<env with torch, transformers, wandb>}"; fi
SH_I=0; SH_N=1
if [[ -n "${SHARD}" ]]; then SH_I="${SHARD%/*}"; SH_N="${SHARD#*/}"; fi
TIER="${TIER:-1}"; [[ "${TIER}" == "all" ]] && TIER="1,2,3,4"
SEEDS="${SEEDS:-0 1 2}"
CONFIG="${CONFIG:-configs/cra_ling.yaml}"
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then STUDY="${STUDY:-smoke}"; else STUDY="${STUDY:-ling_study}"; fi
DRYRUN="${DRYRUN:-0}"

COMMON="--set output_dir=runs/${STUDY}"
[[ -n "${WANDB_ENTITY:-}" ]] && COMMON+=" --set wandb.entity=${WANDB_ENTITY}"
if [[ "${SMOKE}" == "1" ]]; then
    COMMON+=" --set train.max_steps=20 --set train.warmup_steps=2 --set train.eval_every=10"
    COMMON+=" --set train.ckpt_every=10 --set data.train_size=512 --set data.eval_size=64"
fi

MLP="--set cra.target_modules=[] --set cra.mlp_modules=[down_proj]"
HM="--set cra.mlp_modules=[down_proj]"

# arm_name | mask_mode | seeds(0=single seed 0, S=all SEEDS) | extra args
ARMS_1=(
  "H-uniform|uniform|S|"
  "H-topk112|topk|S|--set cra.k_total=112"
  "H-random112|random|S|--set cra.k_total=112"
  "H-soft|soft|0|"
)
ARMS_2=(
  "M-topk3|topk|S|${MLP} --set cra.mlp_k_layers=3"
  "M-random3|random|S|${MLP} --set cra.mlp_k_layers=3"
  "M-uniform|uniform|0|${MLP}"
  "M-soft|soft|0|${MLP}"
)
ARMS_3=(
  "HM-topk|topk|S|${HM} --set cra.k_total=112 --set cra.mlp_k_layers=3"
  "HM-random|random|S|${HM} --set cra.k_total=112 --set cra.mlp_k_layers=3"
)
ARMS_4=(
  "H-topk34|topk|S|--set cra.k_total=34"
  "H-random34|random|S|--set cra.k_total=34"
)

SMOKE_TAG=""
if [[ "${SMOKE}" == "1" ]]; then SMOKE_TAG=",smoke"; fi
mkdir -p "runs/${STUDY}" logs
LEDGER="runs/${STUDY}/submitted.tsv"
n_sub=0; n_skip=0; idx=0; failed=""
if [[ "${LOCAL}" == "1" ]]; then
    export HF_HOME="${HF_HOME:-/ssd_scratch/${USER:-$(id -un)}/hf_cache}"; mkdir -p "${HF_HOME}"
    export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
fi
for tier in ${TIER//,/ }; do
    declare -n ARMS="ARMS_${tier}"
    for spec in "${ARMS[@]}"; do
        IFS='|' read -r arm mode seedspec extra <<< "${spec}"
        if [[ "${seedspec}" == "S" ]]; then seeds="${SEEDS}"; else seeds="0"; fi
        for seed in ${seeds}; do
            run="${arm}-s${seed}"
            run_dir="runs/${STUDY}/${run}"
            my_idx=${idx}; idx=$((idx + 1))
            if (( my_idx % SH_N != SH_I )); then continue; fi      # another session's job
            if [[ -f "${run_dir}/DONE" ]]; then
                echo "[skip] ${run}: DONE"; n_skip=$((n_skip + 1)); continue
            fi
            tags="[stage2,linguistic_v1,ling39,${arm},seed${seed}${SMOKE_TAG}]"
            EXTRA="${COMMON} ${extra} --set cra.tag=${arm} --set train.seed=${seed} --set cra.random_seed=${seed} --set wandb.tags=${tags}"
            echo "[submit] ${run}  (mode=${mode})"
            if [[ "${DRYRUN}" == "1" ]]; then
                echo "         EXTRA=${EXTRA}"; continue
            fi
            if [[ "${LOCAL}" == "1" ]]; then
                mkdir -p "${run_dir}"; export WANDB_DIR="${run_dir}"
                set -f
                # shellcheck disable=SC2086
                if python scripts/stage2/train_cra_compare.py --config "${CONFIG}" --mask-mode "${mode}" \
                       ${EXTRA} --resume "${run_dir}" 2>&1 | tee -a "${run_dir}/local.log"; then :; else failed+=" ${run}"; fi
                set +f
                [[ -f "${run_dir}/DONE" ]] || failed+=" ${run}"
                n_sub=$((n_sub + 1)); continue
            fi
            jid=$(CONFIG="${CONFIG}" MASK_MODE="${mode}" RUN_DIR="${run_dir}" EXTRA="${EXTRA}" \
                  CONDA_ENV="${CONDA_ENV}" sbatch --parsable --job-name="cra-${run}" \
                  --export=ALL cluster/train_cra_compare.slurm)
            printf '%s\t%s\t%s\n' "${jid}" "${run}" "$(date -Is)" >> "${LEDGER}"
            n_sub=$((n_sub + 1))
        done
    done
done
echo "handled ${n_sub} job(s), skipped ${n_skip} finished. Ledger: ${LEDGER}"
[[ -z "${failed}" ]] || echo "NOT DONE (rerun the same command to resume):${failed}"
[[ "${DRYRUN}" == "1" ]] || echo "watch: squeue -u \$USER | grep cra- ; summarise: python scripts/stage2/summarize_study.py --study ${STUDY}"
