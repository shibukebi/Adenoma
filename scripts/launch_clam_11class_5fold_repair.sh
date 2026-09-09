#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLAM_PYTHON="${CLAM_PYTHON:-/data15/data15_5/yuexin2/anaconda3/envs/clam_latest/bin/python}"
READY_CSV="${READY_CSV:-/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/manifests_11class_fold5/joint_hp_yx_adenoma_11class_ready.csv}"
SPLIT_DIR="${SPLIT_DIR:-/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/manifests_11class_fold5/splits}"
FEATURE_ROOT="${FEATURE_ROOT:-/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/features}"
RESULT_ROOT="${RESULT_ROOT:-/data15/zhengke_usb2/yuexin_data/result/5fold_11class}"

FOLDS="${FOLDS:-0 1 2 3}"
MAGS="${MAGS:-2p5x 5x 10x 20x}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
SEED="${SEED:-2023}"
LR="${LR:-0.0001}"
REG="${REG:-0.00001}"
DROPOUT="${DROPOUT:-0.25}"
EMBED_DIM="${EMBED_DIM:-1024}"
GPUS=(${GPUS:-7})

LAUNCH_DIR="${RESULT_ROOT}/launcher_logs/clam_repair_$(date +%Y%m%d_%H%M%S)"
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
  if [[ -f "${result_dir}/metrics.json" ]]; then
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

for fold in ${FOLDS}; do
  for mag in ${MAGS}; do
    enqueue "CLAM-SB_${mag}_fold${fold}" "${RESULT_ROOT}/CLAM-SB/11class/${mag}/fold-${fold}" \
      "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/train_clam_ssl_20x.py" \
        --dataset-csv "${READY_CSV}" \
        --ready-csv "${READY_CSV}" \
        --split-dir "${SPLIT_DIR}" \
        --feature-dir "${FEATURE_ROOT}/${mag}" \
        --results-dir "${RESULT_ROOT}/CLAM-SB/11class/${mag}/fold-${fold}" \
        --fold "${fold}" \
        --max-epochs "${MAX_EPOCHS}" \
        --seed "${SEED}" \
        --lr "${LR}" \
        --reg "${REG}" \
        --drop-out "${DROPOUT}" \
        --embed-dim "${EMBED_DIM}" \
        --weighted-sample \
        --early-stopping \
        --allow-small-splits \
        --task-mode adenoma_11class \
        --task-name "adenoma_11class_${mag}_clam_sb_fold${fold}" \
        --model-type clam_sb \
        --bag-loss ce \
        --inst-loss ce \
        --model-size small
  done
done

printf 'Queued %s CLAM repair jobs across GPUs: %s\n' "${job_idx}" "${GPUS[*]}" | tee "${LAUNCH_DIR}/launch_summary.txt"
printf 'Launch dir: %s\n' "${LAUNCH_DIR}" | tee -a "${LAUNCH_DIR}/launch_summary.txt"

for gpu in "${GPUS[@]}"; do
  worker="${LAUNCH_DIR}/worker_gpu${gpu}.sh"
  log="${LAUNCH_DIR}/worker_gpu${gpu}.log"
  setsid env WORKER_GPU="${gpu}" bash "${worker}" >"${log}" 2>&1 &
  printf '%s\n' "$!" >"${LAUNCH_DIR}/worker_gpu${gpu}.pid"
  printf 'Started CLAM repair worker GPU %s PID %s log %s\n' "${gpu}" "$!" "${log}" | tee -a "${LAUNCH_DIR}/launch_summary.txt"
done
