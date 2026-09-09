#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAM_DIR="${CLAM_ROOT:-${ROOT_DIR}/CLAM}"
PATCH_GCN_DIR="${PATCH_GCN_ROOT:-${ROOT_DIR}/Patch-GCN}"
CLAM_REV="53e2409d4a8189c682c173382964a85f114f923c"
PATCH_GCN_REV="823addaee5b8f4cc2bec3ea8e5e0077b2a5115a4"

clone_or_checkout() {
  local url="$1"
  local directory="$2"
  local revision="$3"
  if [[ -d "${directory}/.git" ]]; then
    git -C "${directory}" fetch --quiet --tags origin
  else
    mkdir -p "$(dirname "${directory}")"
    git clone --quiet "${url}" "${directory}"
  fi
  git -C "${directory}" checkout --quiet "${revision}"
  printf '%s\t%s\t%s\n' "${directory}" "${revision}" "$(git -C "${directory}" rev-parse HEAD)"
}

clone_or_checkout https://github.com/mahmoodlab/CLAM.git "${CLAM_DIR}" "${CLAM_REV}"
clone_or_checkout https://github.com/mahmoodlab/Patch-GCN.git "${PATCH_GCN_DIR}" "${PATCH_GCN_REV}"
