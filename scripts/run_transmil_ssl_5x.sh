#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/transmil_ssl_5x.env}"

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
    results_dir="${TRANSMIL_RESULTS_ROOT}/fold-${fold}"
fi
if [[ "${selected_gpu}" == "auto" ]]; then
    selected_gpu="$("${PROJECT_ROOT}/scripts/select_idle_gpu.sh")"
fi
if [[ -n "${selected_gpu}" ]]; then
    export CUDA_VISIBLE_DEVICES="${selected_gpu}"
fi

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/check_transmil_env.py"
"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/prepare_transmil_ssl_data.py" \
  --label-csv "${LABEL_CSV}" \
  --patch-dir "${LOWMAG_PATCH_H5_DIR}" \
  --feature-dir "${TRANSMIL_FEATURE_DIR}" \
  --raw-fold-dir "${RAW_FOLD_DIR}" \
  --dataset-csv "${CLAM_DATASET_CSV}" \
  --ready-csv "${TRANSMIL_READY_CSV}" \
  --split-dir "${TRANSMIL_SPLIT_DIR}" \
  --fold "${fold}"

if [[ "${prepare_only}" == "1" ]]; then
    exit 0
fi

cmd=(
  "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/train_transmil_ssl.py"
  --ready-csv "${TRANSMIL_READY_CSV}"
  --split-dir "${TRANSMIL_SPLIT_DIR}"
  --feature-dir "${TRANSMIL_FEATURE_DIR}"
  --results-dir "${results_dir}"
  --fold "${fold}"
  --seed "${SEED}"
  --lr "${LR}"
  --reg "${REG}"
  --drop-out "${DROPOUT}"
  --embed-dim "${EMBED_DIM}"
  --model-dim "${TRANSMIL_MODEL_DIM}"
  --task-name "${TRANSMIL_TASK_NAME}"
)
if [[ "${WEIGHTED_SAMPLE}" == "1" ]]; then cmd+=(--weighted-sample); fi
if [[ "${EARLY_STOPPING}" == "1" ]]; then cmd+=(--early-stopping); fi
if [[ "${smoke}" == "1" ]]; then cmd+=(--smoke); else cmd+=(--max-epochs "${MAX_EPOCHS}"); fi

printf 'selected_gpu: %s\n' "${selected_gpu:-default}"
printf 'Command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
