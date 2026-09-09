#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/adenoma_yx_preprocess.env}"
ENV_STAIN_REFINE_SEGMENTATION="${STAIN_REFINE_SEGMENTATION:-}"
ENV_STAIN_REFINED_OUTPUT_DIR="${STAIN_REFINED_OUTPUT_DIR:-}"
ENV_STAIN_REFINED_SEGMENTATION_DIR="${STAIN_REFINED_SEGMENTATION_DIR:-}"
ENV_STAIN_REFINE_TARGET_DOWNSAMPLE="${STAIN_REFINE_TARGET_DOWNSAMPLE:-}"
ENV_STAIN_REFINE_COLOR_SPACE="${STAIN_REFINE_COLOR_SPACE:-}"
ENV_STAIN_REFINE_THRESHOLD="${STAIN_REFINE_THRESHOLD:-}"
ENV_STAIN_REFINE_EXTRA_ARGS="${STAIN_REFINE_EXTRA_ARGS:-}"
ENV_ENABLE_SEG="${ENABLE_SEG:-}"
ENV_ENABLE_PATCH="${ENABLE_PATCH:-}"
ENV_ENABLE_STITCH="${ENABLE_STITCH:-}"
ENV_SAVE_SEGMENTATION_PKL="${SAVE_SEGMENTATION_PKL:-}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

if [[ -n "${PYISYNTAX_SITE_PACKAGES:-}" ]]; then
    export PYISYNTAX_SITE_PACKAGES
fi

dataset_tag="${DATASET_TAG:-adenoma_yx}"
preprocess_run_name="${PREPROCESS_RUN_NAME:-${dataset_tag}_preprocess}"
smoke_run_name="${SMOKE_RUN_NAME:-${dataset_tag}_smoke}"
smoke_input_dir="${SMOKE_INPUT_DIR:-${WORK_DIR}/smoke_input}"
enable_seg="${ENV_ENABLE_SEG:-${ENABLE_SEG:-0}}"
enable_patch="${ENV_ENABLE_PATCH:-${ENABLE_PATCH:-0}}"
enable_stitch="${ENV_ENABLE_STITCH:-${ENABLE_STITCH:-0}}"
save_segmentation_pkl="${ENV_SAVE_SEGMENTATION_PKL:-${SAVE_SEGMENTATION_PKL:-0}}"
segmentation_mask_dir="${SEGMENTATION_MASK_DIR:-}"
stain_refine_segmentation="${ENV_STAIN_REFINE_SEGMENTATION:-${STAIN_REFINE_SEGMENTATION:-0}}"
stain_refined_output_dir="${ENV_STAIN_REFINED_OUTPUT_DIR:-${STAIN_REFINED_OUTPUT_DIR:-${RUN_ROOT}/${preprocess_run_name}_stain_refined}}"
stain_refined_segmentation_dir="${ENV_STAIN_REFINED_SEGMENTATION_DIR:-${STAIN_REFINED_SEGMENTATION_DIR:-${stain_refined_output_dir}/segmentations}}"
stain_refine_target_downsample="${ENV_STAIN_REFINE_TARGET_DOWNSAMPLE:-${STAIN_REFINE_TARGET_DOWNSAMPLE:-64}}"
stain_refine_color_space="${ENV_STAIN_REFINE_COLOR_SPACE:-${STAIN_REFINE_COLOR_SPACE:-hsv}}"
stain_refine_threshold="${ENV_STAIN_REFINE_THRESHOLD:-${STAIN_REFINE_THRESHOLD:-otsu}}"
stain_refine_extra_args="${ENV_STAIN_REFINE_EXTRA_ARGS:-${STAIN_REFINE_EXTRA_ARGS:-}}"
base_magnification="${BASE_MAGNIFICATION:-40}"
target_magnification="${TARGET_MAGNIFICATION:-}"
physical_extent_patch_size="${PHYSICAL_EXTENT_PATCH_SIZE:-${PATCH_SIZE}}"
physical_level0_patch_size="0"
if [[ -n "${target_magnification}" ]]; then
    physical_level0_patch_size="$("${CLAM_PYTHON}" -c "import sys; print(int(round(float(sys.argv[1]) * float(sys.argv[2]) / float(sys.argv[3]))))" "${physical_extent_patch_size}" "${base_magnification}" "${target_magnification}")"
fi

mode="full"
active_source_dir="${SOURCE_DIR}"
save_dir="${RUN_ROOT}/${preprocess_run_name}"
stitch_flag="${enable_stitch}"
process_list=""
no_auto_skip=0

while (($#)); do
    case "$1" in
        --smoke)
            mode="smoke"
            active_source_dir="${smoke_input_dir}"
            save_dir="${RUN_ROOT}/${smoke_run_name}"
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

mkdir -p "${save_dir}" "${LOG_DIR}"
timing_dir="${PROJECT_ROOT}/outputs/stage_timing_20x"
timing_path="${timing_dir}/${preprocess_run_name}.json"
mkdir -p "${timing_dir}"

base_cmd=(
    "${CLAM_PYTHON}"
    "${CLAM_ROOT}/create_patches_fp.py"
    --source "${active_source_dir}"
    --save_dir "${save_dir}"
    --patch_size "${PATCH_SIZE}"
    --step_size "${STEP_SIZE}"
    --patch_level "${PATCH_LEVEL}"
)

if [[ -n "${PREPROCESS_PRESET:-}" ]]; then
    base_cmd+=(--preset "${PREPROCESS_PRESET}")
fi

if [[ "${physical_level0_patch_size}" != "0" ]]; then
    base_cmd+=(--physical_level0_patch_size "${physical_level0_patch_size}")
    base_cmd+=(--physical_level0_step_size "${physical_level0_patch_size}")
fi

cmd=("${base_cmd[@]}")

if [[ "${enable_seg}" == "1" ]]; then
    cmd+=(--seg)
fi

if [[ "${enable_patch}" == "1" ]]; then
    cmd+=(--patch)
fi

if [[ "${stitch_flag}" == "1" ]]; then
    cmd+=(--stitch)
fi

if [[ -n "${process_list}" ]]; then
    cmd+=(--process_list "${process_list}")
fi

if [[ "${save_segmentation_pkl}" == "1" ]]; then
    cmd+=(--save_segmentation_pkl)
fi

if [[ "${no_auto_skip}" == "1" ]]; then
    cmd+=(--no_auto_skip)
fi

if [[ -n "${segmentation_mask_dir}" ]]; then
    cmd+=(--segmentation_mask_dir "${segmentation_mask_dir}")
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
if [[ "${stain_refine_segmentation}" == "1" ]]; then
    seg_cmd=("${base_cmd[@]}" --seg --save_segmentation_pkl --no_auto_skip)
    if [[ -n "${process_list}" ]]; then
        seg_cmd+=(--process_list "${process_list}")
    fi

    printf 'Stain refinement enabled. Initial CLAM segmentation command:\n'
    printf '  %q' "${seg_cmd[@]}"
    printf '\n'

    if "${seg_cmd[@]}"; then
        refine_cmd=(
            "${CLAM_PYTHON}"
            "${PROJECT_ROOT}/scripts/refine_clam_segmentation_masks.py"
            --source-dir "${active_source_dir}"
            --clam-segmentation-dir "${save_dir}/segmentations"
            --output-dir "${stain_refined_output_dir}"
            --slide-ext "${SLIDE_EXT:-.svs}"
            --target-downsample "${stain_refine_target_downsample}"
            --color-space "${stain_refine_color_space}"
            --threshold "${stain_refine_threshold}"
            --overwrite
        )
        if [[ -n "${process_list}" ]]; then
            refine_cmd+=(--manifest-csv "${process_list}")
        fi
        if [[ -n "${stain_refine_extra_args}" ]]; then
            # shellcheck disable=SC2206
            extra_args=(${stain_refine_extra_args})
            refine_cmd+=("${extra_args[@]}")
        fi

        printf 'Stain refinement command:\n'
        printf '  %q' "${refine_cmd[@]}"
        printf '\n'

        if "${refine_cmd[@]}"; then
            patch_cmd=("${base_cmd[@]}" --seg --segmentation_mask_dir "${stain_refined_segmentation_dir}")
            patch_cmd+=(--no_auto_skip)
            if [[ "${enable_patch}" == "1" ]]; then
                patch_cmd+=(--patch)
            fi
            if [[ "${stitch_flag}" == "1" ]]; then
                patch_cmd+=(--stitch)
            fi
            if [[ -n "${process_list}" ]]; then
                patch_cmd+=(--process_list "${process_list}")
            fi
            if [[ "${save_segmentation_pkl}" == "1" ]]; then
                patch_cmd+=(--save_segmentation_pkl)
            fi

            printf 'Refined segmentation patch command:\n'
            printf '  %q' "${patch_cmd[@]}"
            printf '\n'

            if "${patch_cmd[@]}"; then
                status=0
            else
                status=$?
            fi
        else
            status=$?
        fi
    else
        status=$?
    fi
else
    if "${cmd[@]}"; then
        status=0
    else
        status=$?
    fi
fi

if [[ "${status}" == "0" && -n "${target_magnification}" && "${enable_patch}" == "1" ]]; then
    patch_h5_dir="${save_dir}/patches"
    if compgen -G "${patch_h5_dir}/*.h5" > /dev/null; then
        annotate_cmd=(
            "${CLAM_PYTHON}"
            "${PROJECT_ROOT}/scripts/annotate_patch_h5_magnification.py"
            --patch-h5-dir "${patch_h5_dir}"
            --base-magnification "${base_magnification}"
            --target-magnification "${target_magnification}"
            --physical-extent-patch-size "${physical_extent_patch_size}"
        )
        printf 'Annotating patch h5 magnification metadata:\n'
        printf '  %q' "${annotate_cmd[@]}"
        printf '\n'
        "${annotate_cmd[@]}"
    fi
fi

end_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
end_ts="$(date +%s)"
duration_sec=$((end_ts - start_ts))

cat > "${timing_path}" <<EOF
{
  "stage": "${preprocess_run_name}",
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
