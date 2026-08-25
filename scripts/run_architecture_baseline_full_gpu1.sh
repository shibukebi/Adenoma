#!/usr/bin/env bash
set -euo pipefail

ADENOMA_ROOT=/data1/yuexin/Adenoma
ADENOMA_PYTHON=/data1/yuexin/.conda/envs/patho-r1/bin/python
ADENOMA_OUTPUT="$ADENOMA_ROOT/artifacts/architecture_baselines/frozen_conch_5x_mil_v1"
ADENOMA_CONFIG="$ADENOMA_ROOT/configs/architecture_baselines/frozen_conch_5x_mil_v1.yaml"
ADENOMA_CLI="$ADENOMA_ROOT/scripts/run_architecture_baselines.py"
ADENOMA_CASE_PATHS="$ADENOMA_OUTPUT/data_audit/local_provenance/case_paths.jsonl"
ADENOMA_LABELS="$ADENOMA_OUTPUT/data_audit/labels/canonical_labels.jsonl"
ADENOMA_MANIFEST="$ADENOMA_OUTPUT/manifests/five_x_patch_manifest.jsonl"
ADENOMA_TRAIN_LABELS="$ADENOMA_OUTPUT/splits/training_labels.jsonl"
ADENOMA_CONCH="$ADENOMA_ROOT/models/CONCH/pytorch_model.bin"
ADENOMA_LOGS="$ADENOMA_OUTPUT/logs"
ADENOMA_UNI_PID_FILE="$ADENOMA_LOGS/uni_prismnet_gpu1.pid"

export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$ADENOMA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=/data1/yuexin/.conda/envs/patho-r1/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/data1/yuexin/.conda/envs/patho-r1/lib/python3.10/site-packages/nvidia/cusparse/lib:/data1/yuexin/.conda/envs/patho-r1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

mkdir -p "$ADENOMA_LOGS"
cd "$ADENOMA_ROOT"
exec > >(tee -a "$ADENOMA_LOGS/full_pipeline_gpu1.log") 2>&1

stage() {
  printf '\n[%s] ===== %s =====\n' "$(date --iso-8601=seconds)" "$1"
}

require_file() {
  if [[ ! -s "$1" ]]; then
    printf 'Required artifact is missing or empty: %s\n' "$1" >&2
    exit 1
  fi
}

stop_registered_uni() {
  if [[ ! -s "$ADENOMA_UNI_PID_FILE" ]]; then
    printf 'UNI PID file is absent; nothing to stop.\n'
    return
  fi
  local pid
  pid="$(tr -d '[:space:]' < "$ADENOMA_UNI_PID_FILE")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || [[ ! -r "/proc/$pid/cmdline" ]]; then
    printf 'UNI PID file is stale: %s\n' "$pid"
    return
  fi
  local command_line
  command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  if [[ "$command_line" != *"uni_prismnet_roi_server.py"* ]] || [[ "$command_line" != *"--port 8400"* ]]; then
    printf 'Refusing to stop unrelated PID %s: %s\n' "$pid" "$command_line" >&2
    exit 1
  fi
  kill "$pid"
  for _attempt in 1 2 3 4 5 6; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if kill -0 "$pid" 2>/dev/null; then
    printf 'UNI server PID %s did not stop cleanly.\n' "$pid" >&2
    exit 1
  fi
  printf 'Stopped UNI/PathPrism PID %s.\n' "$pid"
}

start_uni() {
  nohup "$ADENOMA_PYTHON" "$ADENOMA_ROOT/scripts/uni_prismnet_roi_server.py" \
    --uni-weights-path "$ADENOMA_ROOT/models/UNI/weights/pytorch_model.bin" \
    --prismnet-path "$ADENOMA_ROOT/models/PathPrism/prismnet_linprobe.pt" \
    --device cuda:0 \
    --host 127.0.0.1 \
    --port 8400 \
    > "$ADENOMA_LOGS/uni_prismnet_gpu1_smoke.log" 2>&1 &
  printf '%s\n' "$!" > "$ADENOMA_UNI_PID_FILE"
  for _attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if curl --fail --silent http://127.0.0.1:8400/openapi.json >/dev/null; then
      printf 'UNI/PathPrism is ready on port 8400.\n'
      return
    fi
    sleep 5
  done
  printf 'UNI/PathPrism failed to become ready.\n' >&2
  exit 1
}

require_file "$ADENOMA_OUTPUT/resolved_runtime.json"
require_file "$ADENOMA_CASE_PATHS"
require_file "$ADENOMA_LABELS"

stage "Full 1,608-case Mucosa extraction in four deterministic shards (resume existing cases)"
curl --fail --silent http://127.0.0.1:8400/openapi.json >/dev/null
mucosa_pids=()
for shard_index in 0 1 2 3; do
  "$ADENOMA_PYTHON" "$ADENOMA_CLI" run-mucosa \
    --config "$ADENOMA_CONFIG" \
    --output-root "$ADENOMA_OUTPUT" \
    --case-paths "$ADENOMA_CASE_PATHS" \
    --yx-root /mnt/zhengke_usb2/yuexin_data/Adenoma_yx \
    --hp-root /mnt/zhengke_usb2/yuexin_data/Adenoma_hp \
    --pathprism-url http://127.0.0.1:8400/predict \
    --batch-size 16 \
    --timeout-seconds 240 \
    --min-tissue-coverage 0.05 \
    --mask-downsample 32 \
    --mucosa-threshold 0.30 \
    --shard-count 4 \
    --shard-index "$shard_index" \
    --resume \
    --skip-qc-panel \
    > "$ADENOMA_LOGS/mucosa_full_shard_${shard_index}.log" 2>&1 &
  mucosa_pids+=("$!")
  printf 'Started Mucosa shard %s as PID %s.\n' "$shard_index" "$!"
done

mucosa_worker_failure=0
for mucosa_pid in "${mucosa_pids[@]}"; do
  if ! wait "$mucosa_pid"; then
    printf 'Mucosa worker PID %s exited non-zero.\n' "$mucosa_pid" >&2
    mucosa_worker_failure=1
  fi
done
if [[ "$mucosa_worker_failure" -ne 0 ]]; then
  exit 1
fi

"$ADENOMA_PYTHON" -c 'import json,sys; from pathlib import Path; from adenoma_agent.architecture_baselines.mucosa import aggregate_mucosa_shards; p=aggregate_mucosa_shards(Path(sys.argv[1]),4); assert p["requested_cases"]==1608,p; assert not p["failed"],p["failed"][:10]; print(json.dumps({"requested":p["requested_cases"],"completed":len(p["completed"]),"resumed":len(p["skipped_complete"]),"failed":len(p["failed"])},sort_keys=True))' "$ADENOMA_OUTPUT/mucosa_by_slide"

stage "Formal 1,608-case Mucosa technical and eligibility ledger"
"$ADENOMA_PYTHON" -c 'import json,sys; from pathlib import Path; from adenoma_agent.architecture_baselines.mucosa import build_mucosa_eligibility_ledger; p=build_mucosa_eligibility_ledger(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),canonical_labels_jsonl=Path(sys.argv[4])); assert p["requested_cases"]==1608,p; print(json.dumps(p,sort_keys=True))' "$ADENOMA_CASE_PATHS" "$ADENOMA_OUTPUT/mucosa_by_slide" "$ADENOMA_OUTPUT/manifests" "$ADENOMA_LABELS"
require_file "$ADENOMA_OUTPUT/manifests/mucosa_eligibility_ledger.jsonl"
require_file "$ADENOMA_OUTPUT/manifests/mucosa_eligibility_summary.json"

stage "Merge and validate full 5x manifest"
"$ADENOMA_PYTHON" "$ADENOMA_CLI" validate-manifest \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --merge
"$ADENOMA_PYTHON" "$ADENOMA_CLI" validate-manifest \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --manifest "$ADENOMA_MANIFEST"
require_file "$ADENOMA_MANIFEST"
"$ADENOMA_PYTHON" -c 'import json,sys; from adenoma_agent.architecture_baselines.io import iter_jsonl; eligible={str(r["case_alias"]) for r in iter_jsonl(sys.argv[1]) if bool(r["eligible_for_training"])}; manifest={str(r["case_alias"]) for r in iter_jsonl(sys.argv[2]) if float(r["mucosa_coverage"])>=0.60}; assert eligible==manifest,{"ledger_only":sorted(eligible-manifest)[:10],"manifest_only":sorted(manifest-eligible)[:10]}; print("Eligibility/manifest aliases accepted:",len(eligible))' "$ADENOMA_OUTPUT/manifests/mucosa_eligibility_ledger.jsonl" "$ADENOMA_MANIFEST"

stage "Release UNI/PathPrism before CONCH and MIL"
stop_registered_uni
sleep 5

stage "Full Frozen CONCH cache"
"$ADENOMA_PYTHON" "$ADENOMA_CLI" embed-conch \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --manifest "$ADENOMA_MANIFEST" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --checkpoint "$ADENOMA_CONCH" \
  --device cuda:0 \
  --batch-size 32 \
  --minimum-coverage 0.30 \
  --resume
require_file "$ADENOMA_OUTPUT/conch_embeddings/cache_validation.json"

stage "Formal five-fold specimen-family split"
"$ADENOMA_PYTHON" "$ADENOMA_CLI" build-splits \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --manifest "$ADENOMA_MANIFEST" \
  --labels "$ADENOMA_LABELS" \
  --minimum-coverage 0.60 \
  --seed 17
require_file "$ADENOMA_TRAIN_LABELS"

stage "Baseline 1 annotation gate"
"$ADENOMA_PYTHON" "$ADENOMA_CLI" baseline1-status \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --manifest "$ADENOMA_MANIFEST"

stage "45 MIL runs: 3 models x 5 folds x 3 seeds"
"$ADENOMA_PYTHON" "$ADENOMA_CLI" train-mil \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --labels "$ADENOMA_TRAIN_LABELS" \
  --models mean_pool,abmil,dsmil \
  --folds all \
  --seeds 17,29,43 \
  --device cuda:0 \
  --minimum-coverage 0.60 \
  --hidden-dim 256 \
  --dropout 0.25 \
  --max-epochs 100 \
  --patience 15 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --resume
require_file "$ADENOMA_OUTPUT/runs/training_batch_summary.json"

stage "OOF evaluation and 2,000 family bootstraps"
"$ADENOMA_PYTHON" "$ADENOMA_CLI" evaluate \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --models mean_pool,abmil,dsmil \
  --seeds 17,29,43 \
  --minimum-coverage 0.60 \
  --bootstrap 2000 \
  --seed 17 \
  --top-k 10
require_file "$ADENOMA_OUTPUT/evaluation/evaluation_summary.json"

stage "Fresh real YX smoke"
start_uni
"$ADENOMA_PYTHON" "$ADENOMA_CLI" real-smoke \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --checkpoint "$ADENOMA_CONCH" \
  --device cuda:0 \
  --pathprism-url http://127.0.0.1:8400/predict \
  --mucosa-batch-size 16 \
  --conch-batch-size 32 \
  --seeds 17,29,43 \
  --top-k 10 \
  --resume
require_file "$ADENOMA_OUTPUT/real_smoke/inference_result.json"
stop_registered_uni

stage "Full pipeline complete; report decision awaits result audit"
printf 'Evaluation: %s\n' "$ADENOMA_OUTPUT/evaluation/evaluation_summary.json"
printf 'Real smoke: %s\n' "$ADENOMA_OUTPUT/real_smoke/inference_result.json"
