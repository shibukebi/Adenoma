#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH must point to a hierarchical ABMIL/CLAM env file}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

fold=5
results_dir=""
smoke=0
prepare_only=0

while (($#)); do
    case "$1" in
        --fold) fold="$2"; shift 2 ;;
        --results-dir) results_dir="$2"; shift 2 ;;
        --smoke) smoke=1; shift ;;
        --prepare-only) prepare_only=1; shift ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 1 ;;
    esac
done

if [[ -z "${results_dir}" ]]; then
    results_dir="${HIERARCHICAL_RESULTS_ROOT}/fold-${fold}"
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
  "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/train_clam_ssl_20x.py"
  --dataset-csv "${STAGE1_DATASET_CSV}"
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
  --task-mode "${STAGE1_TASK_MODE}"
  --model-type "${MODEL_TYPE}"
  --model-size "${MODEL_SIZE}"
  --bag-loss "${BAG_LOSS}"
  --task-name "${STAGE1_TASK_NAME}"
  --k-sample "${K_SAMPLE}"
  --inst-loss "${INST_LOSS}"
  --bag-weight "${BAG_WEIGHT}"
)
if [[ "${WEIGHTED_SAMPLE}" == "1" ]]; then stage1_cmd+=(--weighted-sample); fi
if [[ "${EARLY_STOPPING}" == "1" ]]; then stage1_cmd+=(--early-stopping); fi
if [[ "${ALLOW_SMALL_SPLITS}" == "1" ]]; then stage1_cmd+=(--allow-small-splits); fi
if [[ "${NO_INST_CLUSTER}" == "1" ]]; then stage1_cmd+=(--no-inst-cluster); fi
if [[ "${smoke}" == "1" ]]; then stage1_cmd+=(--smoke); else stage1_cmd+=(--max-epochs "${MAX_EPOCHS}"); fi
"${stage1_cmd[@]}"

stage2_cmd=(
  "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/train_clam_ssl_20x.py"
  --dataset-csv "${STAGE2_DATASET_CSV}"
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
  --task-mode "${STAGE2_TASK_MODE}"
  --model-type "${MODEL_TYPE}"
  --model-size "${MODEL_SIZE}"
  --bag-loss "${BAG_LOSS}"
  --task-name "${STAGE2_TASK_NAME}"
  --k-sample "${K_SAMPLE}"
  --inst-loss "${INST_LOSS}"
  --bag-weight "${BAG_WEIGHT}"
)
if [[ "${WEIGHTED_SAMPLE}" == "1" ]]; then stage2_cmd+=(--weighted-sample); fi
if [[ "${EARLY_STOPPING}" == "1" ]]; then stage2_cmd+=(--early-stopping); fi
if [[ "${ALLOW_SMALL_SPLITS}" == "1" ]]; then stage2_cmd+=(--allow-small-splits); fi
if [[ "${NO_INST_CLUSTER}" == "1" ]]; then stage2_cmd+=(--no-inst-cluster); fi
if [[ "${smoke}" == "1" ]]; then stage2_cmd+=(--smoke); else stage2_cmd+=(--max-epochs "${MAX_EPOCHS}"); fi
"${stage2_cmd[@]}"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/score_clam_task_split.py" \
  --ready-csv "${STAGE1_READY_CSV}" \
  --split-dir "${STAGE1_SPLIT_DIR}" \
  --feature-dir "${FEATURE_DIR}" \
  --checkpoint "${stage2_dir}/s_${fold}_checkpoint.pt" \
  --output-csv "${stage2_dir}/full_test_predictions.csv" \
  --fold "${fold}" \
  --split-name test \
  --task-mode "${STAGE2_TASK_MODE}" \
  --model-type "${MODEL_TYPE}" \
  --embed-dim "${EMBED_DIM}" \
  --drop-out "${DROPOUT}" \
  --model-size "${MODEL_SIZE}"

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/merge_hierarchical_predictions.py" \
  --label-csv "${LABEL_CSV}" \
  --stage1-val-predictions "${stage1_dir}/val_predictions.csv" \
  --stage1-test-predictions "${stage1_dir}/predictions.csv" \
  --stage2-full-test-predictions "${stage2_dir}/full_test_predictions.csv" \
  --output-dir "${hierarchical_dir}"
