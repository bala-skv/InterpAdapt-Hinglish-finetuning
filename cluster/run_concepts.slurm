#!/usr/bin/env bash
#SBATCH --job-name=concepts
#SBATCH --output=logs/concepts-%j.out
#SBATCH --error=logs/concepts-%j.err
#SBATCH --partition=u22            # default GPU partition (sinfo: u22*); EDIT for ADA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
# #SBATCH --constraint=2080ti      # EDIT/uncomment to pin an RTX 2080 Ti (the pipeline requires one)
# ---------------------------------------------------------------------------
# Runs the Stage 1 study on a concept set (scripts/pipeline/run_concepts.py).
# From the repository root:
#
#   CONCEPTS=data/tracing/my_set.json NAME=my_set CONDA_ENV=anlp \
#       EXTRA="--dry-run" sbatch cluster/run_concepts.slurm          # plan + GPU-time estimate
#   CONCEPTS=data/tracing/my_set.json NAME=my_set CONDA_ENV=anlp \
#       EXTRA="--limit 3" sbatch cluster/run_concepts.slurm          # smoke test, 3 concepts
#   CONCEPTS=data/tracing/my_set.json NAME=my_set CONDA_ENV=anlp sbatch cluster/run_concepts.slurm
#
# Every finished step is skipped on the next run and the pairs step resumes
# mid-way, so if a job hits its time limit, submit the SAME command again.
# Optional: STEPS=gpu|cpu|step1,step2 (default all), EXTRA="--shortlist-size 40 ..."
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
REPO_ROOT="$(pwd)"
mkdir -p logs

CONCEPTS="${CONCEPTS:?set CONCEPTS=path/to/concepts.json}"
NAME="${NAME:?set NAME=run_name}"
STEPS="${STEPS:-all}"
EXTRA="${EXTRA:-}"

echo "=============================================================="
echo " concepts pipeline [${NAME}] job ${SLURM_JOB_ID:-<local>} on $(hostname)"
echo "   concepts : ${CONCEPTS}"
echo "   steps    : ${STEPS}   extra: ${EXTRA:-<none>}"
echo "   started  : $(date -Is)"
echo "=============================================================="

# --- Environment (same pattern as the other job scripts) --------------------
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

# shellcheck disable=SC2086
python scripts/pipeline/run_concepts.py --concepts "${CONCEPTS}" --name "${NAME}" \
    --steps "${STEPS}" ${EXTRA}
echo "[slurm] finished $(date -Is)"
