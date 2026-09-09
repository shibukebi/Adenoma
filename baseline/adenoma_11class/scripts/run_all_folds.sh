#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
READY_CSV="${READY_CSV:-}"
SPLIT_DIR="${SPLIT_DIR:-}"
FEATURE_ROOT="${FEATURE_ROOT:-}"
MIST_MANIFEST_ROOT="${MIST_MANIFEST_ROOT:-}"
RESULTS_ROOT="${RESULTS_ROOT:?Set RESULTS_ROOT to an external output directory}"
FOLDS="${FOLDS:-0 1 2 3 4}"
MODELS="${MODELS:-clam-sb transmil dsmil mist}"
DRY_RUN="${DRY_RUN:-0}"
GPU_INDEX="${GPU_INDEX:-}"

run_one() {
  local model="$1"
  local feature="$2"
  local fold="$3"
  local command=(
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
    --model "${model}"
    --feature "${feature}"
    --fold "${fold}"
    --results-root "${RESULTS_ROOT}"
    --python-bin "${PYTHON_BIN}"
  )
  if [[ "${model}" == "mist" ]]; then
    [[ -n "${MIST_MANIFEST_ROOT}" ]] || { printf 'Set MIST_MANIFEST_ROOT for MIST\n' >&2; exit 2; }
    command+=(--mist-manifest-root "${MIST_MANIFEST_ROOT}")
  else
    [[ -n "${READY_CSV}" && -n "${SPLIT_DIR}" && -n "${FEATURE_ROOT}" ]] || {
      printf 'Set READY_CSV, SPLIT_DIR and FEATURE_ROOT for %s\n' "${model}" >&2
      exit 2
    }
    command+=(--ready-csv "${READY_CSV}" --split-dir "${SPLIT_DIR}" --feature-root "${FEATURE_ROOT}")
  fi
  if [[ -n "${GPU_INDEX}" ]]; then
    command+=(--gpu-index "${GPU_INDEX}")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    command+=(--dry-run)
  fi
  "${command[@]}"
}

for fold in ${FOLDS}; do
  for model in ${MODELS}; do
    if [[ "${model}" == "mist" ]]; then
      for feature in 2p5x_5x 5x_10x; do
        run_one "${model}" "${feature}" "${fold}"
      done
    else
      for feature in 2p5x 5x 10x 20x; do
        run_one "${model}" "${feature}" "${fold}"
      done
    fi
  done
done
