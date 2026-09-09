#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/hierarchical_transmil_20x.env}"

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
    results_dir="${HIERARCHICAL_RESULTS_ROOT}/fold-${fold}"
fi
if [[ "${selected_gpu}" == "auto" ]]; then
    selected_gpu="$("${PROJECT_ROOT}/scripts/select_idle_gpu.sh")"
fi
if [[ -n "${selected_gpu}" ]]; then
    export CUDA_VISIBLE_DEVICES="${selected_gpu}"
fi

stage1_dir="${results_dir}/stage1"
stage2_dir="${results_dir}/stage2"
hierarchical_dir="${results_dir}/hierarchical"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/build_ssl_binary_labels.py" \
  --input-csv "${MASTER_LABEL_INPUT_CSV}" \
  --output-csv "${LABEL_CSV}"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/prepare_clam_ssl_20x_data.py" \
  --label-csv "${LABEL_CSV}" \
  --patch-dir "${PATCH_DIR}" \
  --feature-dir "${FEATURE_DIR}" \
  --raw-fold-dir "${RAW_FOLD_DIR}" \
  --dataset-csv "${STAGE1_DATASET_CSV}" \
  --ready-csv "${STAGE1_READY_CSV}" \
  --split-dir "${STAGE1_SPLIT_DIR}" \
  --task-mode "${STAGE1_TASK_MODE}" \
  --fold "${fold}"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/prepare_clam_ssl_20x_data.py" \
  --label-csv "${LABEL_CSV}" \
  --patch-dir "${PATCH_DIR}" \
  --feature-dir "${FEATURE_DIR}" \
  --raw-fold-dir "${RAW_FOLD_DIR}" \
  --dataset-csv "${STAGE2_DATASET_CSV}" \
  --ready-csv "${STAGE2_READY_CSV}" \
  --split-dir "${STAGE2_SPLIT_DIR}" \
  --task-mode "${STAGE2_TASK_MODE}" \
  --fold "${fold}"

if [[ "${prepare_only}" == "1" ]]; then
    exit 0
fi

stage1_cmd=(
  "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/train_transmil_ssl.py"
  --ready-csv "${STAGE1_READY_CSV}"
  --split-dir "${STAGE1_SPLIT_DIR}"
  --feature-dir "${FEATURE_DIR}"
  --results-dir "${stage1_dir}"
  --fold "${fold}"
  --seed "${SEED}"
  --lr "${LR}"
  --reg "${REG}"
  --drop-out "${DROPOUT}"
  --embed-dim "${EMBED_DIM}"
  --model-dim "${TRANSMIL_MODEL_DIM}"
  --task-mode "${STAGE1_TASK_MODE}"
  --task-name "${STAGE1_TASK_NAME}"
)
if [[ "${WEIGHTED_SAMPLE}" == "1" ]]; then stage1_cmd+=(--weighted-sample); fi
if [[ "${EARLY_STOPPING}" == "1" ]]; then stage1_cmd+=(--early-stopping); fi
if [[ "${smoke}" == "1" ]]; then stage1_cmd+=(--smoke); else stage1_cmd+=(--max-epochs "${MAX_EPOCHS}"); fi
"${stage1_cmd[@]}"

stage2_cmd=(
  "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/train_transmil_ssl.py"
  --ready-csv "${STAGE2_READY_CSV}"
  --split-dir "${STAGE2_SPLIT_DIR}"
  --feature-dir "${FEATURE_DIR}"
  --results-dir "${stage2_dir}"
  --fold "${fold}"
  --seed "${SEED}"
  --lr "${LR}"
  --reg "${REG}"
  --drop-out "${DROPOUT}"
  --embed-dim "${EMBED_DIM}"
  --model-dim "${TRANSMIL_MODEL_DIM}"
  --task-mode "${STAGE2_TASK_MODE}"
  --task-name "${STAGE2_TASK_NAME}"
)
if [[ "${WEIGHTED_SAMPLE}" == "1" ]]; then stage2_cmd+=(--weighted-sample); fi
if [[ "${EARLY_STOPPING}" == "1" ]]; then stage2_cmd+=(--early-stopping); fi
if [[ "${smoke}" == "1" ]]; then stage2_cmd+=(--smoke); else stage2_cmd+=(--max-epochs "${MAX_EPOCHS}"); fi
"${stage2_cmd[@]}"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/score_transmil_task_split.py" \
  --ready-csv "${STAGE1_READY_CSV}" \
  --split-dir "${STAGE1_SPLIT_DIR}" \
  --feature-dir "${FEATURE_DIR}" \
  --checkpoint "${stage2_dir}/s_${fold}_checkpoint.pt" \
  --output-csv "${stage2_dir}/full_test_predictions.csv" \
  --fold "${fold}" \
  --split-name test \
  --task-mode "${STAGE2_TASK_MODE}" \
  --embed-dim "${EMBED_DIM}" \
  --model-dim "${TRANSMIL_MODEL_DIM}" \
  --drop-out "${DROPOUT}"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/merge_hierarchical_predictions.py" \
  --label-csv "${LABEL_CSV}" \
  --stage1-val-predictions "${stage1_dir}/val_predictions.csv" \
  --stage1-test-predictions "${stage1_dir}/predictions.csv" \
  --stage2-full-test-predictions "${stage2_dir}/full_test_predictions.csv" \
  --output-dir "${hierarchical_dir}"
