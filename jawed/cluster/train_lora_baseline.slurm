#!/usr/bin/env bash
#SBATCH --job-name=cr-lora-baseline
#SBATCH --output=logs/lora-baseline-%j.out
#SBATCH --error=logs/lora-baseline-%j.err
#SBATCH --partition=u22            # default GPU partition (sinfo: u22*); EDIT for ADA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00            # a gnode allocation is hard-capped ~6h -> chain jobs
#SBATCH --signal=B:USR1@120        # deliver SIGUSR1 ~2 min before the wall-time kill
# #SBATCH --constraint=2080ti      # EDIT/uncomment to pin sm_75 (no native bf16 on Turing)
# #SBATCH --account=<your_account> # EDIT/uncomment if your cluster requires it
# ---------------------------------------------------------------------------
# Stage-2 uniform-LoRA fine-tuning baseline -- the "one real number".
#
# Produces base-vs-LoRA mean target log-prob on a held-out English->Hindi split
# (grep the .out for a line starting `RESULT`). fp16, LoRA on q_proj+o_proj.
#
# Submit from the repo root (the jawed/ sub-repo):
#     sbatch cluster/train_lora_baseline.slurm
#
# The job is RESUMABLE and SELF-CHAINING: it checkpoints to a durable run dir,
# saves gracefully on SIGUSR1 before the ~6h kill, and resubmits itself against
# the SAME run dir until a DONE marker exists. So a run longer than one
# allocation completes across chained jobs with no manual babysitting.
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
REPO_ROOT="$(pwd)"
mkdir -p logs

CONFIG="${CONFIG:-configs/lora_baseline.yaml}"
# One durable run dir, reused across the whole chain. The FIRST job mints it;
# every chained job inherits it via --export and resumes from its ckpt.pt.
RUN_DIR="${RUN_DIR:-runs/lora-baseline-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${RUN_DIR}"

echo "=============================================================="
echo " LoRA baseline job ${SLURM_JOB_ID:-<local>} on $(hostname)"
echo "   repo root : ${REPO_ROOT}"
echo "   config    : ${CONFIG}"
echo "   run dir   : ${RUN_DIR}"
echo "   started   : $(date -Is)"
echo "=============================================================="

# --- Environment (same pattern as memory_pilot.slurm) -----------------------
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

# --- HF cache. IMPORTANT: /home2 (NFS $HOME) is small/quota'd -- putting the
# multi-GB weights + dataset there fails with "Disk quota exceeded (errno 122)".
# The big, re-downloadable cache goes on NODE-LOCAL /ssd_scratch (fast, roomy,
# but wiped at allocation end -> fine, we re-fetch). Compute nodes here HAVE
# internet, so we download IN-JOB (online) rather than pre-warming home2.
# The small, durable state (RUN_DIR/ckpt.pt, a few MB) stays on the repo/home2
# so a chained resume survives the node wipe.
export HF_HOME="${HF_HOME:-/ssd_scratch/${USER}/hf_cache}"
mkdir -p "${HF_HOME}" || { echo "cannot create HF_HOME=${HF_HOME}" >&2; exit 4; }
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"        # online: fetch model+dataset in-job
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"  # classic downloader; xet errored on constrained FS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "  HF_HOME : ${HF_HOME}   OFFLINE(hub/ds): ${HF_HUB_OFFLINE}/${HF_DATASETS_OFFLINE}"
if command -v df >/dev/null 2>&1; then df -h "$(dirname "${HF_HOME}")" 2>/dev/null || true; fi
#
# NOTE: to run OFFLINE instead (no node network), pre-warm a SHARED cache and
# submit with HF_HOME pointing at it + HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1,
# or set data.jsonl_train in the config so no dataset download is needed.

echo "--- verifying CUDA -------------------------------------------"
python cluster/check_env.py || { echo "env check failed" >&2; exit 3; }

# --- Run training in the background so we can forward SIGUSR1 to it ----------
# The Python handler saves a checkpoint and exits 0; we then chain a resume job.
python scripts/train_lora_baseline.py --config "${CONFIG}" --resume "${RUN_DIR}" &
PYPID=$!

_forward_usr1() {
    echo "[slurm] SIGUSR1 -> forwarding to python (pid ${PYPID}) for graceful save"
    kill -USR1 "${PYPID}" 2>/dev/null || true
}
trap _forward_usr1 USR1

# `wait` is interrupted by the trap; loop until python actually exits.
set +e
wait "${PYPID}"
RC=$?
while [[ ${RC} -gt 128 ]]; do   # 128+signo means wait was interrupted, not exit
    wait "${PYPID}"
    RC=$?
done
set -e
echo "[slurm] python exited rc=${RC}"

# --- Chain: resubmit against the SAME run dir until training is DONE ---------
if [[ -f "${RUN_DIR}/DONE" ]]; then
    echo "[slurm] DONE marker present -> baseline complete. Results:"
    cat "${RUN_DIR}/results.json" || true
elif [[ ${RC} -eq 0 ]]; then
    echo "[slurm] not DONE yet -> resubmitting a resume job for ${RUN_DIR}"
    sbatch --export=ALL,RUN_DIR="${RUN_DIR}",CONFIG="${CONFIG}" \
        cluster/train_lora_baseline.slurm
else
    echo "[slurm] python failed (rc=${RC}); NOT chaining. Inspect ${RUN_DIR}/run.log" >&2
    exit "${RC}"
fi
