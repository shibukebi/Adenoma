#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIST_ROOT="${MIST_ROOT:-${SCRIPT_DIR}}"
MIST_PYTHON="${MIST_PYTHON:-python}"
DATASET_ROOT="${MIST_DATASET_ROOT:-${MIST_ROOT}/datasets}"
RESULT_DIR="${RESULT_DIR:-${MIST_ROOT}/release/results/mist_11class_fold5}"

cd "${MIST_ROOT}"

if [[ ! -d "${DATASET_ROOT}/mist_hp_yx_11class_fold5_train" || ! -d "${DATASET_ROOT}/mist_hp_yx_11class_fold5_val" ]]; then
  printf 'Missing MIST dataset manifests under %s\n' "${DATASET_ROOT}" >&2
  exit 2
fi

mkdir -p "${RESULT_DIR}"

exec "${MIST_PYTHON}" train_ade.py \
  --dataset_train mist_hp_yx_11class_fold5_train \
  --dataset_val mist_hp_yx_11class_fold5_val \
  --num_classes 11 \
  --feats_size 5120 \
  --gpu_index 0 \
  --num_epochs 200 \
  --save_dir "${RESULT_DIR}"
