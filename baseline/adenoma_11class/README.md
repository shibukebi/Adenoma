# Adenoma 11-class five-fold benchmark

This directory defines the benchmark used to compare CLAM-SB, TransMIL,
DSMIL and MIST on the same 11-class adenoma task and the same five-fold case
partition.

## Scope

The benchmark contains 14 model/feature configurations:

| Model | Feature configurations |
| --- | --- |
| CLAM-SB | 2.5x, 5x, 10x, 20x |
| TransMIL | 2.5x, 5x, 10x, 20x |
| DSMIL | 2.5x, 5x, 10x, 20x |
| MIST | 2.5x + 5x, 5x + 10x |

The cancelled MIST 10x + 20x experiment is not part of this benchmark.

## Reproducibility contract

All configurations must share:

- the class-ID mapping in `class_mapping.json`;
- the case assignments in `splits/`;
- the same UNI feature version for each magnification;
- validation-based checkpoint selection;
- a test set that is evaluated only after checkpoint selection;
- five per-fold result records plus Mean and sample SD.

The report display order is `IP, HP, SSL, SSLD, TSA, TSAD, USA, TA, TAD,
TVA, TVAD`. This differs from the model class-ID order and must not be used to
reinterpret logits.

## Fold naming

The public benchmark uses `fold-0` through `fold-4`. Historically, the fifth
fold was trained first and stored under `fold5_hp+yx`, with checkpoint names
such as `s_5_checkpoint.pt`. The weight manifest records both identities:

```text
canonical fold: 4
historical fold: 5
```

Do not retrain or silently rename that fold without recording a new benchmark
version.

## Training one configuration

```bash
python baseline/adenoma_11class/scripts/train.py \
  --model clam-sb \
  --feature 10x \
  --fold 0 \
  --ready-csv /path/to/joint_11class_ready.csv \
  --split-dir /path/to/shared/splits \
  --feature-root /path/to/uni/features \
  --results-root /path/to/benchmark-runs \
  --gpu-index 0
```

Supported model names are `clam-sb`, `transmil`, `dsmil` and `mist`. MIST
additionally requires `--mist-manifest-root` and uses feature names
`2p5x_5x` or `5x_10x`.

Use `--dry-run` to inspect the resolved command without starting training.

## Running all folds

Set the required paths as environment variables and run:

```bash
READY_CSV=/path/to/ready.csv \
SPLIT_DIR=/path/to/splits \
FEATURE_ROOT=/path/to/features \
MIST_MANIFEST_ROOT=/path/to/mist/manifests \
RESULTS_ROOT=/path/to/benchmark-runs \
DRY_RUN=1 \
baseline/adenoma_11class/scripts/run_all_folds.sh
```

Remove `DRY_RUN=1` only after reviewing the commands and GPU allocation. The
wrapper is sequential by design; cluster scheduling should call `train.py`
once per job.

## Aggregating metrics

```bash
python baseline/adenoma_11class/scripts/evaluate.py \
  --results-root /path/to/benchmark-runs \
  --legacy-root /path/to/result/fold5_hp+yx \
  --output-dir baseline/adenoma_11class/results \
  --strict
```

The primary root may use either the new benchmark layout or the historical
`<model>/11class/<feature>/fold-N` layout. `--legacy-root` supplies the old
standalone fifth-fold results. This writes `per_fold_metrics.csv` and
`benchmark_summary.csv`. MIST test
inference must be completed before aggregation so that each MIST fold also has
a `metrics.json` in the standard result layout.

## Checkpoints

After the historical USB2 result storage is mounted:

```bash
python baseline/adenoma_11class/scripts/build_weights_manifest.py \
  --new-root /path/to/result/5fold_11class \
  --legacy-root /path/to/result/fold5_hp+yx \
  --copy-weights
```

The command searches all 70 expected checkpoints, standardizes their release
locations, preserves original filenames, and writes SHA-256 checksums. Copied
weights require Git LFS or an external GitHub Release.

## Data policy

Do not commit WSI files, UNI feature tensors, patch H5 files, review databases,
tile caches or raw local logs. For a public repository, anonymize slide IDs in
split and prediction CSVs. Keep the private ID mapping and source WSI locations
outside GitHub.
