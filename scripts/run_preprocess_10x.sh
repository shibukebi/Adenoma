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
physical_level0_patch_size="$("${CLAM_PYTHON}" -c "import sys; print(int(round(float(sys.argv[1]) * float(sys.argv[2]) / float(sys.argv[3]))))" "${physical_extent_patch_size}" "${base_magnification}" "${target_magnification}")"

if [[ -n "${PYISYNTAX_SITE_PACKAGES:-}" ]]; then
    export PYISYNTAX_SITE_PACKAGES
fi

mode="full"
active_source_dir="${SOURCE_DIR}"
save_dir="${LOWMAG_PREPROCESS_DIR}"
stitch_flag="${ENABLE_STITCH}"
process_list=""

while (($#)); do
    case "$1" in
        --smoke)
            mode="smoke"
            active_source_dir="${WORK_DIR}/smoke_input"
            save_dir="${LOWMAG_SMOKE_PREPROCESS_DIR}"
            shift
            ;;
        --save-dir)
            save_dir="$2"
            shift 2
            ;;
        --source-dir)
            active_source_dir="$2"
            shift 2
            ;;
        --process-list)
            process_list="$2"
            shift 2
            ;;
        --stitch)
            stitch_flag=1
            shift
            ;;
        --no-stitch)
            stitch_flag=0
            shift
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${save_dir}" "${LOG_DIR}"
timing_dir="${STAGE_TIMING_DIR}"
timing_path="${timing_dir}/preprocess_10x.json"
mkdir -p "${timing_dir}"

cmd=(
    "${CLAM_PYTHON}"
    "${CLAM_ROOT}/create_patches_fp.py"
    --source "${active_source_dir}"
    --save_dir "${save_dir}"
    --patch_size "${PATCH_SIZE}"
    --step_size "${STEP_SIZE}"
    --patch_level "${PATCH_LEVEL}"
    --physical_level0_patch_size "${physical_level0_patch_size}"
    --physical_level0_step_size "${physical_level0_patch_size}"
    --no_auto_skip
)

if [[ -n "${PREPROCESS_PRESET:-}" ]]; then
    cmd+=(--preset "${PREPROCESS_PRESET}")
fi

if [[ "${ENABLE_SEG}" == "1" ]]; then
    cmd+=(--seg)
fi

if [[ "${ENABLE_PATCH}" == "1" ]]; then
    cmd+=(--patch)
fi

if [[ "${stitch_flag}" == "1" ]]; then
    cmd+=(--stitch)
fi

if [[ -n "${process_list}" ]]; then
    cmd+=(--process_list "${process_list}")
fi

if [[ "${SAVE_SEGMENTATION_PKL:-0}" == "1" ]]; then
    cmd+=(--save_segmentation_pkl)
fi

if [[ -n "${SEGMENTATION_MASK_DIR:-}" ]]; then
    cmd+=(--segmentation_mask_dir "${SEGMENTATION_MASK_DIR}")
fi

printf 'Mode: %s\n' "${mode}"
printf 'Source: %s\n' "${active_source_dir}"
printf 'Save dir: %s\n' "${save_dir}"
printf 'Physical level0 patch size: %s\n' "${physical_level0_patch_size}"
printf 'Command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

start_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
start_ts="$(date +%s)"

cd "${CLAM_ROOT}"
if "${cmd[@]}"; then
    status=0
else
    status=$?
fi

if [[ "${status}" == "0" && "${ENABLE_PATCH}" == "1" ]]; then
    patch_h5_dir="${save_dir}/patches"
    if compgen -G "${patch_h5_dir}/*.h5" > /dev/null; then
        "${CLAM_PYTHON}" "${PROJECT_ROOT}/scripts/annotate_patch_h5_magnification.py" \
            --patch-h5-dir "${patch_h5_dir}" \
            --base-magnification "${base_magnification}" \
            --target-magnification "${target_magnification}" \
            --physical-extent-patch-size "${physical_extent_patch_size}"
    fi
fi

end_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
end_ts="$(date +%s)"
duration_sec=$((end_ts - start_ts))

cat > "${timing_path}" <<EOF
{
  "stage": "preprocess_10x",
  "mode": "${mode}",
  "source_dir": "${active_source_dir}",
  "save_dir": "${save_dir}",
  "start_time_utc": "${start_iso}",
  "end_time_utc": "${end_iso}",
  "duration_sec": ${duration_sec},
  "exit_code": ${status}
}
EOF

exit "${status}"
