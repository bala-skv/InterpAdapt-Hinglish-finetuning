#!/usr/bin/env bash
#SBATCH --job-name=entangle
#SBATCH --output=logs/entangle-%j.out
#SBATCH --error=logs/entangle-%j.err
#SBATCH --partition=u22            # default GPU partition (sinfo: u22*); EDIT for ADA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
# #SBATCH --constraint=2080ti      # EDIT/uncomment to pin a 2080 Ti (faster fp16 than a 1080 Ti)
# ---------------------------------------------------------------------------
# Stage 1 head-interaction experiments (scripts/stage1/entangle.py).
# One GPU command per submission; run from the repository root:
#
#     CMD=curve CONDA_ENV=anlp EXTRA="--limit 2" sbatch cluster/entangle.slurm  # smoke test
#     CMD=curve CONDA_ENV=anlp sbatch cluster/entangle.slurm
#     CMD=noise CONDA_ENV=anlp sbatch cluster/entangle.slurm
#     # then, on the login node:
#     #   python scripts/stage1/entangle.py shortlist   and commit the shortlist
#     CMD=pairs CONDA_ENV=anlp sbatch cluster/entangle.slurm
#
# fp16 tracing does not use bitsandbytes, so this job does NOT call
# cluster/check_env.py (it tests 4-bit/bf16 kernels, which fail on a 1080 Ti).
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
REPO_ROOT="$(pwd)"
mkdir -p logs

CMD="${CMD:?set CMD=curve|noise|pairs}"
PAIR="${PAIR:-en_hi-latn}"
EXTRA="${EXTRA:-}"
case "${CMD}" in
    curve|noise|pairs) ;;
    *) echo "ERROR: CMD must be curve, noise or pairs (shortlist/analyze run on CPU)" >&2; exit 2 ;;
esac

echo "=============================================================="
echo " entangle ${CMD} [${PAIR}] job ${SLURM_JOB_ID:-<local>} on $(hostname)"
echo "   repo root : ${REPO_ROOT}"
echo "   extra     : ${EXTRA:-<none>}"
echo "   started   : $(date -Is)"
echo "=============================================================="

# --- Environment (same pattern as train_cra_compare.slurm) ------------------
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
        exit 2
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

export HF_HOME="${HF_HOME:-/ssd_scratch/${USER}/hf_cache}"
mkdir -p "${HF_HOME}" || { echo "cannot create HF_HOME=${HF_HOME}" >&2; exit 4; }
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "--- verifying CUDA -------------------------------------------"
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA not available")
p = torch.cuda.get_device_properties(0)
print(f"  {p.name}  (sm_{p.major}{p.minor}, {p.total_memory / 2**30:.1f} GiB)  torch {torch.__version__}")
PY

# shellcheck disable=SC2086
python scripts/stage1/entangle.py "${CMD}" --pair "${PAIR}" ${EXTRA}
echo "[slurm] finished $(date -Is)"
