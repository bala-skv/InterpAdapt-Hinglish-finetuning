#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage-2 CRA -- PRELIMINARY four-arm run on a cloud GPU (Colab / Kaggle).
#
# Runs the SAME uniform/soft/topk/random comparison as the ADA SLURM job, but in
# one shot on a single T4/P100, so we have an internal sanity check of the arm
# ordering while ADA is down. Numbers are NON-ADA + PRELIMINARY (frozen rule:
# keep them OUT of the official results table).
#
# ---- Usage (Colab or Kaggle, GPU + Internet ON) ---------------------------
#   # cell 1 -- get the code (Kaggle: enable Internet in the sidebar first)
#   !git clone https://github.com/bala-skv/InterpAdapt-Hinglish-finetuning.git
#   %cd InterpAdapt-Hinglish-finetuning/jawed
#
#   # cell 2 -- run all four arms serially, collect RESULT lines
#   !bash cluster/run_cloud_gpu.sh
#
#   # (optional) faster/lighter run:
#   !EVAL_SIZE=400 MAX_STEPS=400 bash cluster/run_cloud_gpu.sh
#   # (optional) log to W&B (preliminary, tagged non-ada): `wandb login` then
#   !WANDB=1 bash cluster/run_cloud_gpu.sh
#
# Reads the combined table from runs/cloud_summary.txt at the end -- paste that
# back for the (clearly-labeled preliminary) internal comparison.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- must run from jawed/ (repo path resolution + configs/ live here) --------
if [[ ! -f "scripts/train_cra_compare.py" ]]; then
    echo "ERROR: run this from the 'jawed/' dir, e.g.:" >&2
    echo "       cd InterpAdapt-Hinglish-finetuning/jawed && bash cluster/run_cloud_gpu.sh" >&2
    exit 2
fi

ARMS="${ARMS:-uniform soft topk random}"     # order: baseline sanity first
BASE_CONFIG="${CONFIG:-configs/cra_compare_cloud.yaml}"
PY="${PYTHON:-python}"

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# W&B off unless WANDB=1 (keeps preliminary non-ADA numbers out of the project).
if [[ "${WANDB:-0}" != "1" ]]; then
    export WANDB_MODE=disabled
fi

echo "=============================================================="
echo " CRA cloud run (PRELIMINARY / non-ADA)"
echo "   arms      : ${ARMS}"
echo "   base cfg  : ${BASE_CONFIG}"
echo "   python    : $(command -v ${PY})"
echo "   started   : $(date -Is 2>/dev/null || date)"
echo "=============================================================="

# --- deps: DO NOT touch torch (cloud image ships the right CUDA build) --------
echo "--- installing python deps (transformers/peft/accelerate/datasets) ---"
${PY} -m pip -q install "transformers>=4.44,<5" "peft>=0.12" "accelerate>=0.33" \
    "datasets>=2.14" "pyyaml>=6.0" wandb >/dev/null

# --- GPU sanity ---------------------------------------------------------------
${PY} - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible -- enable a GPU runtime"
p = torch.cuda.get_device_properties(0)
print(f"[gpu] {p.name}  cap={p.major}.{p.minor}  vram={p.total_memory/1024**3:.1f} GiB  torch={torch.__version__}")
PYEOF

# --- optional overrides (EVAL_SIZE / MAX_STEPS / TRAIN_SIZE) -> temp config ---
ACTIVE_CONFIG="${BASE_CONFIG}"
if [[ -n "${EVAL_SIZE:-}${MAX_STEPS:-}${TRAIN_SIZE:-}" ]]; then
    ACTIVE_CONFIG="configs/_cloud_active.yaml"
    EVAL_SIZE="${EVAL_SIZE:-}" MAX_STEPS="${MAX_STEPS:-}" TRAIN_SIZE="${TRAIN_SIZE:-}" \
    BASE_CONFIG="${BASE_CONFIG}" OUT="${ACTIVE_CONFIG}" ${PY} - <<'PYEOF'
import os, yaml
cfg = yaml.safe_load(open(os.environ["BASE_CONFIG"], encoding="utf-8"))
def seti(section, key, env):
    v = os.environ.get(env, "")
    if v.strip():
        cfg.setdefault(section, {})[key] = int(v)
        print(f"[override] {section}.{key} = {int(v)}")
seti("data", "eval_size", "EVAL_SIZE")
seti("data", "train_size", "TRAIN_SIZE")
if os.environ.get("MAX_STEPS", "").strip():
    ms = int(os.environ["MAX_STEPS"])
    cfg.setdefault("train", {})["max_steps"] = ms
    cfg["train"]["eval_every"] = ms      # keep "final eval only"
    cfg["train"]["ckpt_every"] = ms
    print(f"[override] train.max_steps/eval_every/ckpt_every = {ms}")
yaml.safe_dump(cfg, open(os.environ["OUT"], "w", encoding="utf-8"), sort_keys=False)
PYEOF
fi
echo "   active cfg: ${ACTIVE_CONFIG}"

# --- run each arm serially (single-GPU cap on cloud too) ----------------------
mkdir -p runs
SUMMARY="runs/cloud_summary.txt"
: > "${SUMMARY}"
{
    echo "# CRA PRELIMINARY (non-ADA) -- $(date -Is 2>/dev/null || date)"
    echo "# config=${ACTIVE_CONFIG}"
    echo "# NOTE: preliminary sanity of arm ordering; NOT for the official table."
} >> "${SUMMARY}"

FAILED=""
for mode in ${ARMS}; do
    echo ""
    echo ">>> arm: ${mode} ------------------------------------------------"
    if ${PY} scripts/train_cra_compare.py --config "${ACTIVE_CONFIG}" --mask-mode "${mode}"; then
        # the RESULT line is emitted to the per-run run.log; scoop the latest.
        line="$(grep -h '^.*RESULT mask_mode=' runs/cra-${mode}-*/run.log 2>/dev/null | tail -n1 || true)"
        line="${line#*RESULT }"
        [[ -n "${line}" ]] && echo "RESULT ${line}" | tee -a "${SUMMARY}"
    else
        echo "!!! arm ${mode} FAILED (see runs/cra-${mode}-*/run.log)" | tee -a "${SUMMARY}"
        FAILED="${FAILED} ${mode}"
    fi
done

echo ""
echo "=============================================================="
echo " DONE. Combined RESULT lines -> ${SUMMARY}"
echo "=============================================================="
cat "${SUMMARY}"
[[ -z "${FAILED}" ]] || { echo "arms failed:${FAILED}" >&2; exit 1; }
