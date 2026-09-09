#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MIST_ROOT="${MIST_ROOT:-/data15/data15_5/yuexin2/MIST}"

CLAM_PYTHON="${CLAM_PYTHON:-/data15/data15_5/yuexin2/anaconda3/envs/clam_latest/bin/python}"
MIST_PYTHON="${MIST_PYTHON:-/data15/data15_5/yuexin2/anaconda3/envs/patho-r1/bin/python}"

READY_CSV="${READY_CSV:-/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/manifests_11class_fold5/joint_hp_yx_adenoma_11class_ready.csv}"
SPLIT_DIR="${SPLIT_DIR:-/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/manifests_11class_fold5/splits}"
FEATURE_ROOT="${FEATURE_ROOT:-/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/features}"
MIST_MANIFEST_ROOT="${MIST_MANIFEST_ROOT:-/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/manifests_11class_5fold}"
RESULT_ROOT="${RESULT_ROOT:-/data15/zhengke_usb2/yuexin_data/result/5fold_11class}"

MAX_EPOCHS="${MAX_EPOCHS:-100}"
MIST_EPOCHS="${MIST_EPOCHS:-200}"
SEED="${SEED:-2023}"
LR="${LR:-0.0001}"
REG="${REG:-0.00001}"
DROPOUT="${DROPOUT:-0.25}"
EMBED_DIM="${EMBED_DIM:-1024}"
DSMIL_HIDDEN_DIM="${DSMIL_HIDDEN_DIM:-512}"
DSMIL_ATTN_DIM="${DSMIL_ATTN_DIM:-128}"
GPUS=(${GPUS:-1 3 5 6})

LAUNCH_DIR="${RESULT_ROOT}/launcher_logs/idle_accel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LAUNCH_DIR}"

for gpu in "${GPUS[@]}"; do
  worker="${LAUNCH_DIR}/worker_gpu${gpu}.sh"
  cat >"${worker}" <<'WORKER'
#!/usr/bin/env bash
set -uo pipefail

run_job() {
  local job_name="$1"
  local result_dir="$2"
  shift 2
  mkdir -p "${result_dir}"
  if [[ -f "${result_dir}/metrics.json" || -f "${result_dir}/test_metrics.json" || -f "${result_dir}/1.pth" ]]; then
    printf '[%s] SKIP existing %s -> %s\n' "$(date '+%F %T')" "${job_name}" "${result_dir}"
    return 0
  fi
  printf '[%s] START %s -> %s\n' "$(date '+%F %T')" "${job_name}" "${result_dir}"
  printf '%q ' "$@" >"${result_dir}/command.txt"
  printf '\n' >>"${result_dir}/command.txt"
  CUDA_VISIBLE_DEVICES="${WORKER_GPU}" "$@" >"${result_dir}/train.log" 2>&1
  local status=$?
  printf '[%s] EXIT %s status=%s\n' "$(date '+%F %T')" "${job_name}" "${status}"
  return 0
}
WORKER
done

job_idx=0
enqueue() {
  local job_name="$1"
  local result_dir="$2"
  shift 2
  local gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
  local worker="${LAUNCH_DIR}/worker_gpu${gpu}.sh"
  {
    printf '\nrun_job %q %q ' "${job_name}" "${result_dir}"
    printf '%q ' "$@"
    printf '\n'
  } >>"${worker}"
  job_idx=$((job_idx + 1))
}

for fold in 1 2 3; do
  enqueue "DSMIL_5x_fold${fold}" "${RESULT_ROOT}/DSMIL/11class/5x/fold-${fold}" \
    "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/train_dsmil_single_11class.py" \
      --ready-csv "${READY_CSV}" \
      --split-dir "${SPLIT_DIR}" \
      --feature-dir "${FEATURE_ROOT}/5x" \
      --results-dir "${RESULT_ROOT}/DSMIL/11class/5x/fold-${fold}" \
      --fold "${fold}" \
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
      --task-name "adenoma_11class_5x_dsmil_fold${fold}"
done

for combo in 2p5x_5x 5x_10x; do
  for fold in 1 2 3; do
    enqueue "MIST_${combo}_fold${fold}" "${RESULT_ROOT}/MIST/11class/${combo}/fold-${fold}" \
      "${MIST_PYTHON}" "${MIST_ROOT}/train_ade.py" \
        --dataset_train "${MIST_MANIFEST_ROOT}/${combo}/fold-${fold}/train/mist_11class_${combo}_fold${fold}_train.csv" \
        --dataset_val "${MIST_MANIFEST_ROOT}/${combo}/fold-${fold}/val/mist_11class_${combo}_fold${fold}_val.csv" \
        --num_classes 11 \
        --feats_size 5120 \
        --num_epochs "${MIST_EPOCHS}" \
        --gpu_index 0 \
        --save_dir "${RESULT_ROOT}/MIST/11class/${combo}/fold-${fold}"
  done
done

printf 'Queued %s idle-accel jobs across GPUs: %s\n' "${job_idx}" "${GPUS[*]}" | tee "${LAUNCH_DIR}/launch_summary.txt"
printf 'Launch dir: %s\n' "${LAUNCH_DIR}" | tee -a "${LAUNCH_DIR}/launch_summary.txt"

for gpu in "${GPUS[@]}"; do
  worker="${LAUNCH_DIR}/worker_gpu${gpu}.sh"
  log="${LAUNCH_DIR}/worker_gpu${gpu}.log"
  setsid env WORKER_GPU="${gpu}" bash "${worker}" >"${log}" 2>&1 &
  printf '%s\n' "$!" >"${LAUNCH_DIR}/worker_gpu${gpu}.pid"
  printf 'Started idle-accel worker GPU %s PID %s log %s\n' "${gpu}" "$!" "${log}" | tee -a "${LAUNCH_DIR}/launch_summary.txt"
done
