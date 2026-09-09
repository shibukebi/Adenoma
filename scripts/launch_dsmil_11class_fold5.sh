#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data15/data15_5/yuexin2/anaconda3/envs/clam_latest/bin/python}"
READY_CSV="${READY_CSV:-/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/manifests_11class_fold5/joint_hp_yx_adenoma_11class_ready.csv}"
SPLIT_DIR="${SPLIT_DIR:-/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/manifests_11class_fold5/splits}"
FEATURE_ROOT="${FEATURE_ROOT:-/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/features}"
RESULT_ROOT="${RESULT_ROOT:-/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx}"
FOLD="${FOLD:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
SEED="${SEED:-2023}"
LR="${LR:-0.0001}"
REG="${REG:-0.00001}"
DROPOUT="${DROPOUT:-0.25}"
EMBED_DIM="${EMBED_DIM:-1024}"
DSMIL_HIDDEN_DIM="${DSMIL_HIDDEN_DIM:-512}"
DSMIL_ATTN_DIM="${DSMIL_ATTN_DIM:-128}"

declare -A GPU_BY_MAG=(
  ["2p5x"]="${DSMIL_2P5X_GPU:-0}"
  ["5x"]="${DSMIL_5X_GPU:-1}"
  ["10x"]="${DSMIL_10X_GPU:-3}"
  ["20x"]="${DSMIL_20X_GPU:-4}"
)

launch_job() {
  local mag="$1"
  local gpu="$2"
  local results_dir="${RESULT_ROOT}/DSMIL/11class/${mag}"

  mkdir -p "${results_dir}"
  printf 'Launching DSMIL %s on GPU %s -> %s\n' "${mag}" "${gpu}" "${results_dir}"
  setsid bash -c '
    set -euo pipefail
    project_root="$1"
    gpu="$2"
    shift 2
    cd "${project_root}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "$@"
  ' _ "${PROJECT_ROOT}" "${gpu}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_dsmil_single_11class.py" \
      --ready-csv "${READY_CSV}" \
      --split-dir "${SPLIT_DIR}" \
      --feature-dir "${FEATURE_ROOT}/${mag}" \
      --results-dir "${results_dir}" \
      --fold "${FOLD}" \
      --max-epochs "${MAX_EPOCHS}" \
      --seed "${SEED}" \
      --lr "${LR}" \
      --reg "${REG}" \
      --drop-out "${DROPOUT}" \
      --embed-dim "${EMBED_DIM}" \
      --hidden-dim "${DSMIL_HIDDEN_DIM}" \
      --attn-dim "${DSMIL_ATTN_DIM}" \
      --weighted-sample \
      --early-stopping \
      --task-mode adenoma_11class \
      --task-name "adenoma_11class_${mag}_dsmil" \
    >"${results_dir}/train.log" 2>&1 &
  printf '%s\n' "$!" >"${results_dir}/pid.txt"
}

for mag in 2p5x 5x 10x 20x; do
  launch_job "${mag}" "${GPU_BY_MAG[${mag}]}"
done

printf 'All DSMIL jobs launched. Check each result directory train.log and pid.txt for progress.\n'
