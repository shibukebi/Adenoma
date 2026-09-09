#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_lowmag_5x.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

fold=0
results_dir=""
allow_small_splits="${ALLOW_SMALL_SPLITS}"
model_type="${MODEL_TYPE:-mil}"
bag_loss="${BAG_LOSS:-ce}"
model_size="${MODEL_SIZE:-}"
task_name="${TASK_NAME:-ssl_vs_others}"
k_sample="${K_SAMPLE:-8}"
inst_loss="${INST_LOSS:-ce}"
bag_weight="${BAG_WEIGHT:-0.7}"
no_inst_cluster="${NO_INST_CLUSTER:-0}"
smoke=0
prepare_only=0

while (($#)); do
    case "$1" in
        --results-dir)
            results_dir="$2"
            shift 2
            ;;
        --fold)
            fold="$2"
            shift 2
            ;;
        --allow-small-splits)
            allow_small_splits=1
            shift
            ;;
        --smoke)
            smoke=1
            shift
            ;;
        --prepare-only)
            prepare_only=1
            shift
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${results_dir}" ]]; then
    results_dir="${RESULTS_ROOT}/fold-${fold}"
fi

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/prepare_clam_ssl_5x_data.py" \
  --label-csv "${LABEL_CSV}" \
  --patch-dir "${LOWMAG_PATCH_H5_DIR}" \
  --feature-dir "${LOWMAG_FEATURE_DIR}" \
  --raw-fold-dir "${RAW_FOLD_DIR}" \
  --dataset-csv "${CLAM_DATASET_CSV}" \
  --ready-csv "${CLAM_READY_CSV}" \
  --split-dir "${CLAM_SPLIT_DIR}" \
  --fold "${fold}"

if [[ "${prepare_only}" == "1" ]]; then
    exit 0
fi

cmd=(
    "${CLAM_PYTHON}"
    "${PROJECT_ROOT}/scripts/train_clam_ssl_5x.py"
    --dataset-csv "${CLAM_DATASET_CSV}"
    --ready-csv "${CLAM_READY_CSV}"
    --split-dir "${CLAM_SPLIT_DIR}"
    --feature-dir "${LOWMAG_FEATURE_DIR}"
    --results-dir "${results_dir}"
    --fold "${fold}"
    --seed "${SEED}"
    --lr "${LR}"
    --reg "${REG}"
    --drop-out "${DROPOUT}"
    --embed-dim "${EMBED_DIM}"
    --model-type "${model_type}"
    --bag-loss "${bag_loss}"
    --task-name "${task_name}"
    --k-sample "${k_sample}"
    --inst-loss "${inst_loss}"
    --bag-weight "${bag_weight}"
)

if [[ -n "${model_size}" ]]; then
    cmd+=(--model-size "${model_size}")
fi

if [[ "${WEIGHTED_SAMPLE}" == "1" ]]; then
    cmd+=(--weighted-sample)
fi

if [[ "${EARLY_STOPPING}" == "1" ]]; then
    cmd+=(--early-stopping)
fi

if [[ "${allow_small_splits}" == "1" ]]; then
    cmd+=(--allow-small-splits)
fi

if [[ "${no_inst_cluster}" == "1" ]]; then
    cmd+=(--no-inst-cluster)
fi

if [[ "${smoke}" == "1" ]]; then
    cmd+=(--smoke)
else
    cmd+=(--max-epochs "${MAX_EPOCHS}")
fi

printf 'Command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
