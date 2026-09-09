#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_c_patho_r1.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

results_dir="${ROUTE_C_RESULTS_ROOT}/fold-0"
allow_small_splits=0
smoke=0

while (($#)); do
    case "$1" in
        --results-dir)
            results_dir="$2"
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
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 1
            ;;
    esac
done

"${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/prepare_route_c_clam_ssl_20x_data.py" \
  --label-csv "${ROUTE_C_LABEL_CSV}" \
  --route-manifest-csv "${ROUTE_C_OUTPUT_ROOT}/manifest_route_c.csv" \
  --feature-dir "${ROUTE_C_FEATURE_DIR}" \
  --raw-fold-dir "${ROUTE_C_RAW_FOLD_DIR}" \
  --dataset-csv "${ROUTE_C_DATASET_CSV}" \
  --ready-csv "${ROUTE_C_READY_CSV}" \
  --split-dir "${ROUTE_C_SPLIT_DIR}"

cmd=(
  "${CLAM_PYTHON}"
  "${PROJECT_ROOT}/scripts/train_clam_ssl_20x_route_c.py"
  --dataset-csv "${ROUTE_C_DATASET_CSV}"
  --ready-csv "${ROUTE_C_READY_CSV}"
  --split-dir "${ROUTE_C_SPLIT_DIR}"
  --feature-dir "${ROUTE_C_FEATURE_DIR}"
  --results-dir "${results_dir}"
  --seed 2023
  --lr 1e-4
  --reg 1e-5
  --drop-out 0.25
  --embed-dim 1024
  --weighted-sample
  --early-stopping
)

if [[ "${allow_small_splits}" == "1" ]]; then
  cmd+=(--allow-small-splits)
fi

if [[ "${smoke}" == "1" ]]; then
  cmd+=(--smoke)
else
  cmd+=(--max-epochs 100)
fi

printf 'Command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
