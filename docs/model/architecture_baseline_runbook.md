# Frozen CONCH 5x Architecture Baseline Runbook

This runbook executes the implemented CLI exactly as registered by
`scripts/run_architecture_baselines.py`. Run it from a terminal where
`nvidia-smi` and PyTorch can both access physical GPU 1. The shell exposes
that card as the only visible device, so PyTorch addresses it as logical
`cuda:0`.

The two WSI roots are read-only. Every generated experiment file, including
logs and PID files, is written below:

```text
/data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1
```

## 1. Shell environment

Start a fresh shell and define the formal paths. Do not change
`CUDA_VISIBLE_DEVICES` after preflight.

```bash
export ADENOMA_ROOT=/data1/yuexin/Adenoma
export ADENOMA_PYTHON=/data1/yuexin/.conda/envs/patho-r1/bin/python
export ADENOMA_CONFIG=/data1/yuexin/Adenoma/configs/architecture_baselines/frozen_conch_5x_mil_v1.yaml
export ADENOMA_OUTPUT=/data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1
export ADENOMA_PILOT_OUTPUT=/data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/pilot10
export ADENOMA_YX_ROOT=/mnt/zhengke_usb2/yuexin_data/Adenoma_yx
export ADENOMA_HP_ROOT=/mnt/zhengke_usb2/yuexin_data/Adenoma_hp
export ADENOMA_LABELS_XLSX=/data1/yuexin/Adenoma/data/label/Adenoma_filtered.xlsx
export ADENOMA_CONCH=/data1/yuexin/Adenoma/models/CONCH/pytorch_model.bin
export ADENOMA_CLI=/data1/yuexin/Adenoma/scripts/run_architecture_baselines.py
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/data1/yuexin/Adenoma/src${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=/data1/yuexin/.conda/envs/patho-r1/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/data1/yuexin/.conda/envs/patho-r1/lib/python3.10/site-packages/nvidia/cusparse/lib:/data1/yuexin/.conda/envs/patho-r1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
set -o pipefail
cd /data1/yuexin/Adenoma
mkdir -p /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs
```

The baseline environment needs the project test, contract, and architecture
extras. Check the environment first:

```bash
"$ADENOMA_PYTHON" --version
"$ADENOMA_PYTHON" -m pip check
"$ADENOMA_PYTHON" -m pytest --version
```

If an optional dependency is missing, install the declared extras before
continuing:

```bash
"$ADENOMA_PYTHON" -m pip install -e '/data1/yuexin/Adenoma[contracts,test,architecture-baseline]'
```

The official CONCH package is an environment-specific dependency and is not
replaced by the repository's zero-shot HTTP server. Verify the exact direct
encoder import before preflight:

```bash
"$ADENOMA_PYTHON" -c 'from conch.open_clip_custom import create_model_from_pretrained; print("CONCH direct encoder import: OK")'
```

## 2. Frozen AgentFlow and baseline tests

Run the frozen current gate. This command enforces exactly 11 Reviewer
contract tests, 17 behavioral-loop tests, and 81 focused regression tests,
with zero failures, errors, or skips.

```bash
"$ADENOMA_PYTHON" /data1/yuexin/Adenoma/scripts/run_agentflow_ci.py \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/agentflow_ci.log
```

Run the independent architecture baseline suite. The current suite contains
21 tests and should report `21 passed` with no skips.

```bash
"$ADENOMA_PYTHON" -m pytest -q --strict-markers \
  -m 'architecture_baseline and not gpu' \
  /data1/yuexin/Adenoma/tests/test_architecture_baseline_data.py \
  /data1/yuexin/Adenoma/tests/test_architecture_baseline_models.py \
  /data1/yuexin/Adenoma/tests/test_architecture_baseline_runtime.py \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/baseline_tests.log

git -C /data1/yuexin/Adenoma diff --check
```

Do not proceed to the real run if either gate is red.

## 3. Physical GPU 1 / logical cuda:0 preflight

The preflight rejects CPU fallback, checks `nvidia-smi`, verifies a real YX
WSI can be opened with physical metadata, checks the registered CONCH SHA-256,
and probes batches 8, 16, and 32. It writes the selected batch size to
`resolved_runtime.json`.

```bash
nvidia-smi

"$ADENOMA_PYTHON" "$ADENOMA_CLI" preflight \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --device cuda:0 \
  --checkpoint "$ADENOMA_CONCH" \
  --batch-candidates 8,16,32 \
  --yx-root "$ADENOMA_YX_ROOT" \
  --hp-root "$ADENOMA_HP_ROOT" \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/preflight.log

test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/resolved_runtime.json

export ADENOMA_CONCH_BATCH="$("$ADENOMA_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_conch_batch_size"])' /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/resolved_runtime.json)"
printf 'Resolved CONCH batch: %s\n' "$ADENOMA_CONCH_BATCH"
```

If preflight fails, keep the error log and stop. The formal run must not be
repeated with `--device cpu`.

## 4. Read-only data audit

This step inventories both sources but creates training labels only from
clean matched records. The HP source remains audit-only. Raw paths are stored
under `local_provenance/` and must not be copied into shareable reports.

```bash
"$ADENOMA_PYTHON" "$ADENOMA_CLI" audit \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --label-workbook "$ADENOMA_LABELS_XLSX" \
  --yx-root "$ADENOMA_YX_ROOT" \
  --hp-root "$ADENOMA_HP_ROOT" \
  --seed 17 \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/data_audit.log

export ADENOMA_CASE_PATHS=/data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/data_audit/local_provenance/case_paths.jsonl
export ADENOMA_CANONICAL_LABELS=/data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/data_audit/labels/canonical_labels.jsonl
export ADENOMA_MANIFEST=/data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/manifests/five_x_patch_manifest.jsonl
export ADENOMA_TRAIN_LABELS=/data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/splits/training_labels.jsonl

test -s "$ADENOMA_CASE_PATHS"
test -s "$ADENOMA_CANONICAL_LABELS"
test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/data_audit/audit_summary.json
```

## 5. Start UNI + PathPrism on physical GPU 1

The Mucosa extractor calls the existing UNI + PathPrism service at port 8400.
The server is intentionally launched with explicit `cuda:0` rather than
`auto`.

```bash
nohup "$ADENOMA_PYTHON" /data1/yuexin/Adenoma/scripts/uni_prismnet_roi_server.py \
  --uni-weights-path /data1/yuexin/Adenoma/models/UNI/weights/pytorch_model.bin \
  --prismnet-path /data1/yuexin/Adenoma/models/PathPrism/prismnet_linprobe.pt \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8400 \
  > /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/uni_prismnet_gpu0.log 2>&1 &

export ADENOMA_UNI_PID=$!
printf '%s\n' "$ADENOMA_UNI_PID" > /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/uni_prismnet_gpu0.pid

for attempt in 1 2 3 4 5 6; do
  curl --fail --silent http://127.0.0.1:8400/openapi.json >/dev/null && break
  sleep 5
done
curl --fail --silent http://127.0.0.1:8400/openapi.json >/dev/null
sed -n '1,40p' /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/uni_prismnet_gpu0.log
```

The startup log must contain `"device": "cuda:0"`. Stop the service after
all Mucosa and real-smoke steps are complete:

```bash
kill "$(sed -n '1p' /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/uni_prismnet_gpu0.pid)"
```

## 6. Ten-case Mucosa pilot

The pilot writes into the formal per-slide Mucosa directory so the later full
run can resume these ten completed slides. `--skip-qc-panel` must be kept the
same for pilot and full runs.

```bash
"$ADENOMA_PYTHON" "$ADENOMA_CLI" run-mucosa \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --yx-root "$ADENOMA_YX_ROOT" \
  --hp-root "$ADENOMA_HP_ROOT" \
  --pathprism-url http://127.0.0.1:8400/predict \
  --batch-size 16 \
  --timeout-seconds 240 \
  --min-tissue-coverage 0.05 \
  --mask-downsample 32 \
  --mucosa-threshold 0.30 \
  --limit-cases 10 \
  --resume \
  --skip-qc-panel \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/mucosa_pilot10.log

"$ADENOMA_PYTHON" "$ADENOMA_CLI" validate-manifest \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --merge \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/manifest_merge_pilot10.log

"$ADENOMA_PYTHON" "$ADENOMA_CLI" validate-manifest \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --manifest "$ADENOMA_MANIFEST" \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/manifest_validate_pilot10.log
```

Inspect `mucosa_by_slide/batch_summary.json`. Do not begin full preprocessing
until `failed` is empty or every failure has a documented data/dependency
reason.

## 7. Ten-case Frozen CONCH pilot

The pilot embedding cache is isolated under `pilot10/`. This prevents the
pilot manifest hash from being confused with the later full manifest hash.

```bash
"$ADENOMA_PYTHON" "$ADENOMA_CLI" embed-conch \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_PILOT_OUTPUT" \
  --manifest "$ADENOMA_MANIFEST" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --checkpoint "$ADENOMA_CONCH" \
  --device cuda:0 \
  --batch-size "$ADENOMA_CONCH_BATCH" \
  --minimum-coverage 0.30 \
  --limit-cases 10 \
  --resume \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/conch_pilot10.log

test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/pilot10/conch_embeddings/cache_validation.json
```

The pilot validation must report 512-dimensional embeddings, finite values,
and `labels_in_cache=false` for every slide.

## 8. Full YX Mucosa and merged manifest

Omitting `--limit-cases` selects all YX rows from the resolver. `--resume`
preserves completed per-slide outputs whose run fingerprint matches.

```bash
"$ADENOMA_PYTHON" "$ADENOMA_CLI" run-mucosa \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --yx-root "$ADENOMA_YX_ROOT" \
  --hp-root "$ADENOMA_HP_ROOT" \
  --pathprism-url http://127.0.0.1:8400/predict \
  --batch-size 16 \
  --timeout-seconds 240 \
  --min-tissue-coverage 0.05 \
  --mask-downsample 32 \
  --mucosa-threshold 0.30 \
  --resume \
  --skip-qc-panel \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/mucosa_full_yx.log

"$ADENOMA_PYTHON" "$ADENOMA_CLI" validate-manifest \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --merge \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/manifest_merge_full.log

"$ADENOMA_PYTHON" "$ADENOMA_CLI" validate-manifest \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --manifest "$ADENOMA_MANIFEST" \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/manifest_validate_full.log
```

This merge reads only completed per-slide manifests. Compare the merged case
count with the batch summary; failed or zero-patch slides must remain explicit.
The full GPU1 runner additionally writes
`manifests/mucosa_eligibility_ledger.jsonl` and rejects every technical/tile
error before merge. Clean slides without a patch at coverage `>=0.60` remain
explicit pre-registered eligibility exclusions rather than technical failures.

Mucosa is now complete, so stop UNI + PathPrism before CONCH embedding and MIL
training. This releases GPU memory and prevents the server from changing the
preflighted CONCH memory envelope.

```bash
kill "$(sed -n '1p' /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/uni_prismnet_gpu0.pid)"
```

## 9. Full Frozen CONCH cache

The formal cache uses the full merged manifest and the batch size selected by
preflight. It stores the label-free per-slide `features.npy`, `index.jsonl`,
and `metadata.json` artifacts.

```bash
"$ADENOMA_PYTHON" "$ADENOMA_CLI" embed-conch \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --manifest "$ADENOMA_MANIFEST" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --checkpoint "$ADENOMA_CONCH" \
  --device cuda:0 \
  --batch-size "$ADENOMA_CONCH_BATCH" \
  --minimum-coverage 0.30 \
  --resume \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/conch_full_yx.log

test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/conch_embeddings/cache_summary.json
test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/conch_embeddings/cache_validation.json
```

## 10. Five-fold grouped splits and Baseline 1 gate

Only cases with at least one patch at `mucosa_coverage >= 0.60` enter the
formal split. The command must fail if train, validation, or test lacks any of
the seven classes.

```bash
"$ADENOMA_PYTHON" "$ADENOMA_CLI" build-splits \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --manifest "$ADENOMA_MANIFEST" \
  --labels "$ADENOMA_CANONICAL_LABELS" \
  --minimum-coverage 0.60 \
  --seed 17 \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/build_splits.log

"$ADENOMA_PYTHON" "$ADENOMA_CLI" baseline1-status \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --manifest "$ADENOMA_MANIFEST" \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/baseline1_status.log

test -s "$ADENOMA_TRAIN_LABELS"
test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/splits/split_summary.json
test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/baseline1/annotation_status.json
```

The Baseline 1 output must be `annotation_blocked` with
`slide_label_broadcast=false`. Do not pass the slide-label JSONL as
`--annotations`.

## 11. Three models × five folds × three seeds

This produces 45 formal runs: MeanPool, ABMIL, and DSMIL over all five folds
with seeds 17, 29, and 43. Test predictions are written only after validation
checkpoint selection.

```bash
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
  --resume \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/train_mil_45runs.log

test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/runs/training_batch_summary.json
```

For recovery, rerun the identical command with `--resume`. A run is considered
complete only when its `test_predictions.jsonl` exists.

## 12. OOF evaluation and 2,000 family bootstraps

Evaluation averages the three seed probabilities within each fold, creates
complete OOF predictions, computes the majority baseline, performs family-
cluster bootstrap intervals, and applies Holm correction to the three
pre-registered model comparisons.

```bash
"$ADENOMA_PYTHON" "$ADENOMA_CLI" evaluate \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --models mean_pool,abmil,dsmil \
  --seeds 17,29,43 \
  --minimum-coverage 0.60 \
  --bootstrap 2000 \
  --seed 17 \
  --top-k 10 \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/evaluate_oof.log

test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/evaluation/evaluation_summary.json
```

## 13. Real YX smoke

Restart UNI + PathPrism on physical GPU 1 before this step. Use the same registered
weights, endpoint and explicit device as the preprocessing run.

```bash
nohup "$ADENOMA_PYTHON" /data1/yuexin/Adenoma/scripts/uni_prismnet_roi_server.py \
  --uni-weights-path /data1/yuexin/Adenoma/models/UNI/weights/pytorch_model.bin \
  --prismnet-path /data1/yuexin/Adenoma/models/PathPrism/prismnet_linprobe.pt \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8400 \
  > /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/uni_prismnet_gpu0_smoke.log 2>&1 &
printf '%s\n' "$!" > /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/uni_prismnet_gpu0.pid

for attempt in 1 2 3 4 5 6; do
  curl --fail --silent http://127.0.0.1:8400/openapi.json >/dev/null && break
  sleep 5
done
curl --fail --silent http://127.0.0.1:8400/openapi.json >/dev/null
sed -n '1,40p' /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/uni_prismnet_gpu0_smoke.log
```

The startup log must again contain `"device": "cuda:0"`. This command then
selects the first source-safe alias
from fold-0 test, then executes a fresh label-free
WSI → Mucosa → CONCH → selected MIL closure.

```bash
"$ADENOMA_PYTHON" "$ADENOMA_CLI" real-smoke \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --case-paths "$ADENOMA_CASE_PATHS" \
  --checkpoint "$ADENOMA_CONCH" \
  --device cuda:0 \
  --pathprism-url http://127.0.0.1:8400/predict \
  --mucosa-batch-size 16 \
  --conch-batch-size "$ADENOMA_CONCH_BATCH" \
  --seeds 17,29,43 \
  --top-k 10 \
  --resume \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/real_yx_smoke.log

test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/real_smoke/inference_result.json
```

The smoke result must state `ground_truth_used_before_prediction=false`.

## 14. Final A/B/C/D report

Choose a recommendation only after reviewing `evaluation_summary.json`, all
per-class errors, paired bootstrap comparisons, and the real-smoke artifact:

- **A** — Frozen CONCH signal is already strong; the simple model is
  sufficient for next-stage Agent integration.
- **B** — CONCH signal exists, but MIL is necessary.
- **C** — Weak signal exists; semantic-guided architecture supervision should
  be tested next.
- **D** — The current 5x formulation is insufficient; task, data, FOV, or
  labels need redesign before a more complex model.

Set exactly one allowed decision and write a result-specific rationale. The
report is kept inside the formal artifact root.

```bash
export ADENOMA_RECOMMENDATION=B
export ADENOMA_RECOMMENDATION_RATIONALE='Replace this text with the observed OOF metrics, confidence intervals, per-class behavior, paired comparisons, and real-smoke result supporting the selected decision.'

"$ADENOMA_PYTHON" "$ADENOMA_CLI" report \
  --config "$ADENOMA_CONFIG" \
  --output-root "$ADENOMA_OUTPUT" \
  --report-path /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/report/architecture_baseline_report.md \
  --recommendation "$ADENOMA_RECOMMENDATION" \
  --recommendation-rationale "$ADENOMA_RECOMMENDATION_RATIONALE" \
  2>&1 | tee /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/report.log

test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/report/architecture_baseline_report.md
test -s /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/report/report_manifest.json
```

Do not leave the example `B` or placeholder rationale unchanged. A conclusion
before all formal folds and the real smoke complete is `inconclusive`, not an
A/B/C/D result.

## 15. Stop the server and final checks

```bash
kill "$(sed -n '1p' /data1/yuexin/Adenoma/artifacts/architecture_baselines/frozen_conch_5x_mil_v1/logs/uni_prismnet_gpu0.pid)"

git -C /data1/yuexin/Adenoma diff --check
git -C /data1/yuexin/Adenoma status --short
```

Never run cleanup commands against either source root. Preserve failed-slide
boundaries, logs, resolved runtime, split manifests, cache metadata, training
manifests, OOF predictions, and report hashes as the reproducibility record.
