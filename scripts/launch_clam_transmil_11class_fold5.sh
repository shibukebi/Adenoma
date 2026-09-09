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
TRANSMIL_MODEL_DIM="${TRANSMIL_MODEL_DIM:-512}"

declare -A GPU_BY_JOB=(
  ["clam_2p5x"]="${CLAM_2P5X_GPU:-3}"
  ["clam_5x"]="${CLAM_5X_GPU:-4}"
  ["clam_10x"]="${CLAM_10X_GPU:-5}"
  ["clam_20x"]="${CLAM_20X_GPU:-6}"
  ["transmil_2p5x"]="${TRANSMIL_2P5X_GPU:-7}"
  ["transmil_5x"]="${TRANSMIL_5X_GPU:-3}"
  ["transmil_10x"]="${TRANSMIL_10X_GPU:-4}"
  ["transmil_20x"]="${TRANSMIL_20X_GPU:-5}"
)

launch_job() {
  local job_name="$1"
  local gpu="$2"
  local results_dir="$3"
  shift 3

  mkdir -p "${results_dir}"
  printf 'Launching %s on GPU %s -> %s\n' "${job_name}" "${gpu}" "${results_dir}"
  setsid bash -c '
    set -euo pipefail
    project_root="$1"
    gpu="$2"
    shift 2
    cd "${project_root}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "$@"
  ' _ "${PROJECT_ROOT}" "${gpu}" "$@" >"${results_dir}/train.log" 2>&1 &
  printf '%s\n' "$!" >"${results_dir}/pid.txt"
}

for mag in 2p5x 5x 10x 20x; do
  launch_job \
    "clam_${mag}" \
    "${GPU_BY_JOB[clam_${mag}]}" \
    "${RESULT_ROOT}/CLAM-SB/11class/${mag}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_clam_ssl_20x.py" \
      --dataset-csv "${READY_CSV}" \
      --ready-csv "${READY_CSV}" \
      --split-dir "${SPLIT_DIR}" \
      --feature-dir "${FEATURE_ROOT}/${mag}" \
      --results-dir "${RESULT_ROOT}/CLAM-SB/11class/${mag}" \
      --fold "${FOLD}" \
      --max-epochs "${MAX_EPOCHS}" \
      --seed "${SEED}" \
      --lr "${LR}" \
      --reg "${REG}" \
      --drop-out "${DROPOUT}" \
      --embed-dim "${EMBED_DIM}" \
      --weighted-sample \
      --early-stopping \
      --task-mode adenoma_11class \
      --task-name "adenoma_11class_${mag}_clam_sb" \
      --model-type clam_sb \
      --bag-loss ce \
      --inst-loss ce \
      --model-size small
done

for mag in 2p5x 5x 10x 20x; do
  launch_job \
    "transmil_${mag}" \
    "${GPU_BY_JOB[transmil_${mag}]}" \
    "${RESULT_ROOT}/TransMIL/11class/${mag}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_transmil_ssl.py" \
      --ready-csv "${READY_CSV}" \
      --split-dir "${SPLIT_DIR}" \
      --feature-dir "${FEATURE_ROOT}/${mag}" \
      --results-dir "${RESULT_ROOT}/TransMIL/11class/${mag}" \
      --fold "${FOLD}" \
      --max-epochs "${MAX_EPOCHS}" \
      --seed "${SEED}" \
      --lr "${LR}" \
      --reg "${REG}" \
      --drop-out "${DROPOUT}" \
      --embed-dim "${EMBED_DIM}" \
      --model-dim "${TRANSMIL_MODEL_DIM}" \
      --weighted-sample \
      --early-stopping \
      --task-mode adenoma_11class \
      --task-name "adenoma_11class_${mag}_transmil"
done

printf 'All jobs launched. Check each result directory train.log and pid.txt for progress.\n'
