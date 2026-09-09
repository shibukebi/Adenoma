#!/usr/bin/env bash
set -euo pipefail

exclude_gpus="${GPU_EXCLUDE:-}"
if [[ $# -gt 0 ]]; then
    exclude_gpus="$1"
fi

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', *' -v exclude="${exclude_gpus}" '
    BEGIN {
      split(exclude, arr, ",");
      for (i in arr) {
        if (arr[i] != "") {
          blocked[arr[i]] = 1;
        }
      }
    }
    {
      idx = $1;
      mem = $2 + 0;
      util = $3 + 0;
      if (!(idx in blocked)) {
        print idx, mem, util;
      }
    }
  ' \
  | sort -k2,2n -k3,3n \
  | awk 'NR == 1 { print $1 }'
