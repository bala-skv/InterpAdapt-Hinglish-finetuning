#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Tier 5 (added after the first plan): heads + MLP layers at ~10% of heads, 1 seed.
#   HM-topk34   : LoRA on the 34 top-scoring heads + down_proj of the 3 top MLP layers
#   HM-random34 : the same budget with 34 random heads + 3 random MLP layers
# Run both with the SAME seed, one per GPU session (~35 min each on an RTX 2080 Ti):
#   bash scripts/stage2/run_hm34.sh topk   [seed]
#   bash scripts/stage2/run_hm34.sh random [seed]
# Needs: conda env active, WANDB_ENTITY exported. DRYRUN=1 prints the command only.
# Finished runs are skipped; an interrupted run resumes from its checkpoint.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODE="${1:?usage: run_hm34.sh topk|random [seed]}"; SEED="${2:-0}"
case "${MODE}" in topk|random) ;; *) echo "mode must be topk or random" >&2; exit 2 ;; esac
ARM="HM-${MODE}34"; STUDY="${STUDY:-ling_study}"; CONFIG="${CONFIG:-configs/cra_ling.yaml}"
RUN_DIR="runs/${STUDY}/${ARM}-s${SEED}"
if [[ -f "${RUN_DIR}/DONE" ]]; then echo "[skip] ${ARM}-s${SEED}: DONE"; exit 0; fi

EXTRA="--set output_dir=runs/${STUDY}"
if [[ -n "${WANDB_ENTITY:-}" ]]; then EXTRA+=" --set wandb.entity=${WANDB_ENTITY}"; fi
EXTRA+=" --set cra.mlp_modules=[down_proj] --set cra.k_total=34 --set cra.mlp_k_layers=3"
EXTRA+=" --set cra.tag=${ARM} --set train.seed=${SEED} --set cra.random_seed=${SEED}"
EXTRA+=" --set wandb.tags=[stage2,linguistic_v1,ling39,${ARM},seed${SEED}]"

echo "[run] ${ARM}-s${SEED}  (mode=${MODE})"
if [[ "${DRYRUN:-0}" == "1" ]]; then echo "      EXTRA=${EXTRA}"; exit 0; fi

export HF_HOME="${HF_HOME:-/ssd_scratch/${USER:-$(id -un)}/hf_cache}"; mkdir -p "${HF_HOME}"
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "${RUN_DIR}"; export WANDB_DIR="${RUN_DIR}"
set -f   # keep values like [down_proj] literal
# shellcheck disable=SC2086
python scripts/stage2/train_cra_compare.py --config "${CONFIG}" --mask-mode "${MODE}" \
    ${EXTRA} --resume "${RUN_DIR}" 2>&1 | tee -a "${RUN_DIR}/local.log" || true
set +f
if [[ -f "${RUN_DIR}/DONE" ]]; then grep -h RESULT "${RUN_DIR}/run.log" | cut -c1-200; else echo "NOT DONE: rerun the same command to resume"; fi
