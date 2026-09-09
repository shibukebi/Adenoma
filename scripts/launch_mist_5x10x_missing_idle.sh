#!/usr/bin/env bash
set -euo pipefail

MIST_ROOT="${MIST_ROOT:-/data15/data15_5/yuexin2/MIST}"
MIST_PYTHON="${MIST_PYTHON:-/data15/data15_5/yuexin2/anaconda3/envs/patho-r1/bin/python}"
MIST_MANIFEST_ROOT="${MIST_MANIFEST_ROOT:-/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/manifests_11class_5fold}"
RESULT_ROOT="${RESULT_ROOT:-/data15/zhengke_usb2/yuexin_data/result/5fold_11class}"
MIST_EPOCHS="${MIST_EPOCHS:-200}"

# Format: fold:physical_gpu. train_ade.py resets CUDA_VISIBLE_DEVICES from
# --gpu_index internally, so this must be the real GPU id, not local cuda:0.
ASSIGNMENTS=(${ASSIGNMENTS:-2:3 3:4})

LAUNCH_DIR="${RESULT_ROOT}/launcher_logs/mist_5x10x_idle_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LAUNCH_DIR}"

for pair in "${ASSIGNMENTS[@]}"; do
  fold="${pair%%:*}"
  gpu="${pair##*:}"
  result_dir="${RESULT_ROOT}/MIST/11class/5x_10x/fold-${fold}"
  mkdir -p "${result_dir}"
  if [[ -f "${result_dir}/1.pth" || -f "${result_dir}/metrics.json" || -f "${result_dir}/test_metrics.json" ]]; then
    printf '[%s] SKIP existing MIST_5x_10x_fold%s -> %s\n' "$(date '+%F %T')" "${fold}" "${result_dir}" | tee -a "${LAUNCH_DIR}/launch_summary.txt"
    continue
  fi

  log="${LAUNCH_DIR}/MIST_5x_10x_fold${fold}_gpu${gpu}.log"
  cmd=(
    "${MIST_PYTHON}" "${MIST_ROOT}/train_ade.py"
    --dataset_train "${MIST_MANIFEST_ROOT}/5x_10x/fold-${fold}/train/mist_11class_5x_10x_fold${fold}_train.csv"
    --dataset_val "${MIST_MANIFEST_ROOT}/5x_10x/fold-${fold}/val/mist_11class_5x_10x_fold${fold}_val.csv"
    --num_classes 11
    --feats_size 5120
    --num_epochs "${MIST_EPOCHS}"
    --gpu_index "${gpu}"
    --save_dir "${result_dir}"
  )
  printf '%q ' "${cmd[@]}" >"${result_dir}/command.txt"
  printf '\n' >>"${result_dir}/command.txt"
  printf '[%s] START MIST_5x_10x_fold%s on GPU %s -> %s\n' "$(date '+%F %T')" "${fold}" "${gpu}" "${result_dir}" | tee -a "${LAUNCH_DIR}/launch_summary.txt"
  setsid "${cmd[@]}" >"${result_dir}/train.log" 2>&1 &
  printf '%s\n' "$!" >"${LAUNCH_DIR}/MIST_5x_10x_fold${fold}_gpu${gpu}.pid"
  printf 'PID %s log %s\n' "$!" "${result_dir}/train.log" | tee -a "${LAUNCH_DIR}/launch_summary.txt"
done

printf 'Launch dir: %s\n' "${LAUNCH_DIR}" | tee -a "${LAUNCH_DIR}/launch_summary.txt"
