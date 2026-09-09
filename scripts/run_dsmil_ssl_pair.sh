#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/dsmil_ssl_2p5x_5x.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

fold=5
results_dir=""
smoke=0
prepare_only=0
selected_gpu=""

while (($#)); do
    case "$1" in
        --fold) fold="$2"; shift 2 ;;
        --results-dir) results_dir="$2"; shift 2 ;;
        --smoke) smoke=1; shift ;;
        --prepare-only) prepare_only=1; shift ;;
        --gpu) selected_gpu="$2"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 1 ;;
    esac
done

if [[ -z "${results_dir}" ]]; then
    results_dir="${DSMIL_RESULTS_ROOT}/fold-${fold}"
fi
if [[ "${selected_gpu}" == "auto" ]]; then
    selected_gpu="$("${PROJECT_ROOT}/scripts/select_idle_gpu.sh")"
fi
if [[ -n "${selected_gpu}" ]]; then
    export CUDA_VISIBLE_DEVICES="${selected_gpu}"
fi

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/check_dsmil_env.py"
"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/prepare_dsmil_ssl_data.py" \
  --label-csv "${LABEL_CSV}" \
  --feature-dir-a "${DSMIL_FEATURE_DIR_A}" \
  --feature-dir-b "${DSMIL_FEATURE_DIR_B}" \
  --raw-fold-dir "${RAW_FOLD_DIR}" \
  --dataset-csv "${DSMIL_DATASET_CSV}" \
  --ready-csv "${DSMIL_READY_CSV}" \
  --split-dir "${DSMIL_SPLIT_DIR}" \
  --stream-a-name "${DSMIL_STREAM_A_NAME}" \
  --stream-b-name "${DSMIL_STREAM_B_NAME}" \
  --fold "${fold}"

if [[ "${prepare_only}" == "1" ]]; then
    exit 0
fi

cmd=(
  "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/train_dsmil_ssl.py"
  --ready-csv "${DSMIL_READY_CSV}"
  --split-dir "${DSMIL_SPLIT_DIR}"
  --feature-dir-a "${DSMIL_FEATURE_DIR_A}"
  --feature-dir-b "${DSMIL_FEATURE_DIR_B}"
  --results-dir "${results_dir}"
  --fold "${fold}"
  --seed "${SEED}"
  --lr "${LR}"
  --reg "${REG}"
  --drop-out "${DROPOUT}"
  --embed-dim "${EMBED_DIM}"
  --hidden-dim "${DSMIL_HIDDEN_DIM}"
  --attn-dim "${DSMIL_ATTN_DIM}"
  --task-name "${DSMIL_TASK_NAME}"
)
if [[ "${WEIGHTED_SAMPLE}" == "1" ]]; then cmd+=(--weighted-sample); fi
if [[ "${EARLY_STOPPING}" == "1" ]]; then cmd+=(--early-stopping); fi
if [[ "${smoke}" == "1" ]]; then cmd+=(--smoke); else cmd+=(--max-epochs "${MAX_EPOCHS}"); fi

printf 'selected_gpu: %s\n' "${selected_gpu:-default}"
printf 'Command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
