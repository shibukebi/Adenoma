#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/route_lowmag_10x.env}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

base_magnification="${BASE_MAGNIFICATION:-40}"
target_magnification="${TARGET_MAGNIFICATION:-10}"
physical_extent_patch_size="${PHYSICAL_EXTENT_PATCH_SIZE:-${PATCH_SIZE}}"

mode="full"
data_h5_dir="${LOWMAG_PATCH_H5_DIR}"
csv_path="${FULL_FEATURE_INPUT_CSV}"
feat_dir="${LOWMAG_FEATURE_DIR}"
model_name="${FEATURE_MODEL_NAME}"
encoder_init="${FEATURE_INIT_MODE}"
weights_path="${FEATURE_WEIGHTS_PATH}"
no_auto_skip=0
selected_gpu=""

while (($#)); do
    case "$1" in
        --smoke)
            mode="smoke"
            data_h5_dir="${LOWMAG_SMOKE_PATCH_H5_DIR}"
            csv_path="${SMOKE_FEATURE_INPUT_CSV}"
            feat_dir="${LOWMAG_SMOKE_FEATURE_DIR}"
            shift
            ;;
        --csv-path)
            csv_path="$2"
            shift 2
            ;;
        --data-h5-dir)
            data_h5_dir="$2"
            shift 2
            ;;
        --feat-dir)
            feat_dir="$2"
            shift 2
            ;;
        --gpu)
            selected_gpu="$2"
            shift 2
            ;;
        --model-name)
            model_name="$2"
            shift 2
            ;;
        --batch-size)
            FEATURE_BATCH_SIZE="$2"
            shift 2
            ;;
        --random-init)
            encoder_init="random"
            shift
            ;;
        --pretrained-init)
            encoder_init="pretrained"
            shift
            ;;
        --weights-path)
            weights_path="$2"
            shift 2
            ;;
        --no-auto-skip)
            no_auto_skip=1
            shift
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 1
            ;;
    esac
done

if [[ "${selected_gpu}" == "auto" ]]; then
    selected_gpu="$("${PROJECT_ROOT}/scripts/select_idle_gpu.sh")"
fi

if [[ -n "${selected_gpu}" ]]; then
    export CUDA_VISIBLE_DEVICES="${selected_gpu}"
fi

mkdir -p "${feat_dir}" "${LOG_DIR}"
timing_dir="${STAGE_TIMING_DIR}"
timing_path="${timing_dir}/feature_extraction_10x.json"
mkdir -p "${timing_dir}"

cmd=(
    "${CLAM_PYTHON}"
    "${PROJECT_ROOT}/scripts/extract_features_route_a.py"
    --data_h5_dir "${data_h5_dir}"
    --data_slide_dir "${SOURCE_DIR}"
    --csv_path "${csv_path}"
    --feat_dir "${feat_dir}"
    --model-name "${model_name}"
    --batch_size "${FEATURE_BATCH_SIZE}"
    --slide_ext .svs
    --target_patch_size "${FEATURE_TARGET_PATCH_SIZE}"
    --base-magnification "${base_magnification}"
    --target-magnification "${target_magnification}"
    --physical-extent-patch-size "${physical_extent_patch_size}"
    --encoder-init "${encoder_init}"
)

if [[ "${FEATURE_EXPECTED_PHYSICAL_LEVEL0_EXTENT:-0}" != "0" ]]; then
    cmd+=(--expected-physical-level0-extent "${FEATURE_EXPECTED_PHYSICAL_LEVEL0_EXTENT}")
fi

if [[ -n "${weights_path}" ]]; then
    cmd+=(--weights-path "${weights_path}")
fi

if [[ "${no_auto_skip}" == "1" ]]; then
    cmd+=(--no_auto_skip)
fi

printf 'Mode: %s\n' "${mode}"
printf 'data_h5_dir: %s\n' "${data_h5_dir}"
printf 'csv_path: %s\n' "${csv_path}"
printf 'feat_dir: %s\n' "${feat_dir}"
printf 'selected_gpu: %s\n' "${selected_gpu:-default}"
printf 'Command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

start_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
start_ts="$(date +%s)"

if "${cmd[@]}"; then
    status=0
else
    status=$?
fi

end_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
end_ts="$(date +%s)"
duration_sec=$((end_ts - start_ts))

cat > "${timing_path}" <<EOF
{
  "stage": "feature_extraction_10x",
  "mode": "${mode}",
  "data_h5_dir": "${data_h5_dir}",
  "csv_path": "${csv_path}",
  "feat_dir": "${feat_dir}",
  "start_time_utc": "${start_iso}",
  "end_time_utc": "${end_iso}",
  "duration_sec": ${duration_sec},
  "exit_code": ${status}
}
EOF

exit "${status}"
