#!/usr/bin/env bash
#SBATCH --job-name=cr-cra-compare
#SBATCH --output=logs/cra-%j.out
#SBATCH --error=logs/cra-%j.err
#SBATCH --partition=u22            # default GPU partition (sinfo: u22*); EDIT for ADA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00            # a gnode allocation is hard-capped ~6h -> chain jobs
#SBATCH --signal=B:USR1@120        # deliver SIGUSR1 ~2 min before the wall-time kill
# #SBATCH --constraint=2080ti      # EDIT/uncomment to pin sm_75 (no native bf16 on Turing)
# ---------------------------------------------------------------------------
# Stage-2 CRA vs. matched-budget controls -- ONE arm per submission.
#
# Pick the arm with MASK_MODE (uniform | soft | topk | random). Submit one job
# per arm and compare the `RESULT` lines (grep the .out) / W&B side by side:
#
#     MASK_MODE=soft    sbatch cluster/train_cra_compare.slurm
#     MASK_MODE=uniform sbatch cluster/train_cra_compare.slurm
#     MASK_MODE=topk    sbatch cluster/train_cra_compare.slurm
#     MASK_MODE=random  sbatch cluster/train_cra_compare.slurm
#
# soft/topk/random need Stage-1 scores at the path in cra.scores_file
# (default data/stage1_head_scores.json). uniform ignores it.
#
# RESUMABLE + SELF-CHAINING like the baseline: durable ckpt, SIGUSR1 graceful
# save, resubmit against the SAME run dir until a DONE marker exists.
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
REPO_ROOT="$(pwd)"
mkdir -p logs

CONFIG="${CONFIG:-configs/cra_compare.yaml}"
MASK_MODE="${MASK_MODE:-soft}"     # uniform | soft | topk | random
# One durable run dir per arm, reused across the whole chain.
RUN_DIR="${RUN_DIR:-runs/cra-${MASK_MODE}-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${RUN_DIR}"

echo "=============================================================="
echo " CRA[${MASK_MODE}] job ${SLURM_JOB_ID:-<local>} on $(hostname)"
echo "   repo root : ${REPO_ROOT}"
echo "   config    : ${CONFIG}"
echo "   mask mode : ${MASK_MODE}"
echo "   run dir   : ${RUN_DIR}"
echo "   started   : $(date -Is)"
echo "=============================================================="

# --- Environment (same pattern as train_lora_baseline.slurm) ----------------
CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "${CONDA_ENV}" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u
    echo "[env] conda env '${CONDA_ENV}' -> $(command -v python)"
else
    VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        echo "ERROR: ${VENV_DIR} not found and CONDA_ENV unset." >&2
        echo "       Run 'bash cluster/setup_env.sh' first, or set CONDA_ENV=<name>." >&2
        exit 2
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

# --- HF cache on node-local /ssd_scratch (roomy, wiped at end -> re-fetch).
# Durable state (RUN_DIR/ckpt.pt) stays on the repo/home2 so a chained resume
# survives the node wipe. Compute nodes here HAVE internet -> download in-job.
export HF_HOME="${HF_HOME:-/ssd_scratch/${USER}/hf_cache}"
mkdir -p "${HF_HOME}" || { echo "cannot create HF_HOME=${HF_HOME}" >&2; exit 4; }
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "  HF_HOME : ${HF_HOME}   OFFLINE(hub/ds): ${HF_HUB_OFFLINE}/${HF_DATASETS_OFFLINE}"

# --- Weights & Biases (guidelines-required link) ----------------------------
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}}"
if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-online}" == "online" ]]; then
    echo "[wandb] WARNING: WANDB_API_KEY unset -> W&B may run offline or skip." >&2
    echo "        export WANDB_API_KEY=... (or 'wandb login'), or set WANDB_MODE=offline." >&2
fi

echo "--- verifying CUDA -------------------------------------------"
python cluster/check_env.py || { echo "env check failed" >&2; exit 3; }

# --- Run training in the background so we can forward SIGUSR1 to it ----------
python scripts/stage2/train_cra_compare.py --config "${CONFIG}" --mask-mode "${MASK_MODE}" \
    --resume "${RUN_DIR}" &
PYPID=$!

_forward_usr1() {
    echo "[slurm] SIGUSR1 -> forwarding to python (pid ${PYPID}) for graceful save"
    kill -USR1 "${PYPID}" 2>/dev/null || true
}
trap _forward_usr1 USR1

set +e
wait "${PYPID}"
RC=$?
while [[ ${RC} -gt 128 ]]; do
    wait "${PYPID}"
    RC=$?
done
set -e
echo "[slurm] python exited rc=${RC}"

# --- Chain: resubmit against the SAME run dir until training is DONE ---------
if [[ -f "${RUN_DIR}/DONE" ]]; then
    echo "[slurm] DONE marker present -> CRA[${MASK_MODE}] complete. Results:"
    cat "${RUN_DIR}/results.json" || true
elif [[ ${RC} -eq 0 ]]; then
    echo "[slurm] not DONE yet -> resubmitting a resume job for ${RUN_DIR}"
    sbatch --export=ALL,RUN_DIR="${RUN_DIR}",CONFIG="${CONFIG}",MASK_MODE="${MASK_MODE}" \
        cluster/train_cra_compare.slurm
else
    echo "[slurm] python failed (rc=${RC}); NOT chaining. Inspect ${RUN_DIR}/run.log" >&2
    exit "${RC}"
fi
