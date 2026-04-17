# adenoma_agent

`adenoma_agent` is a companion project for `/data15/data15_5/yuexin2/adenoma`.

This revision focuses on a CPathAgent-style `SSA vs others` WSI workflow for `Adenoma_yx`.
It reuses the existing `adenoma` preprocessing and Route C selection scripts, then adds:

- `Trace -> Navigate -> Observe/Reason -> Audit -> End` orchestration
- `TraceCluster` outputs with `l_k / s_k / d_k`
- `(x, y, m, o)` navigation steps
- English observation logs and checklist-style pathological reports
- structured JSONL logging, replay, and SSA proxy evaluation

## Current MVP Scope

- Input: `adenoma/data/adenoma_yx_manifest.csv` plus `adenoma/data/adenoma_yx_labels.csv`
- Output:
  - trace clusters with `l_k / s_k / d_k`
  - `(x, y, m, o)` navigation trajectory
  - step-level observation and reasoning logs
  - checklist-style pathological report
  - `SSA vs others` final prediction
  - audit report

The MVP does not claim pixel-level semantic masks. It is an engineering approximation of the paper's WSI agent logic for `SSA vs others`.

## Layout

- `configs/`: scope, task matrix, metrics, budget, baselines, runtime defaults
- `src/adenoma_agent/`: package code
- `scripts/`: smoke, batch, replay, eval, test runners
- `tests/`: `unittest` coverage
- `artifacts/`: run outputs and caches

## Quick Start

Build a deterministic pilot subset:

```bash
PYTHONPATH=/data15/data15_5/yuexin2/adenoma_agent/src \
python3 -m adenoma_agent.cli build-pilot \
  --positives 12 \
  --negatives 12
```

Run the smoke test on the default slide:

```bash
/data15/data15_5/yuexin2/adenoma_agent/scripts/run_smoke.sh
```

Run a single case:

```bash
PYTHONPATH=/data15/data15_5/yuexin2/adenoma_agent/src \
python3 -m adenoma_agent.cli run-case \
  --case-id 138189_751666001 \
  --output-root /data15/data15_5/yuexin2/adenoma_agent/artifacts/runs/manual
```

Replay a finished case:

```bash
PYTHONPATH=/data15/data15_5/yuexin2/adenoma_agent/src \
python3 -m adenoma_agent.cli replay-case \
  --case-dir /data15/data15_5/yuexin2/adenoma_agent/artifacts/runs/smoke/138189_751666001
```

Aggregate a run:

```bash
PYTHONPATH=/data15/data15_5/yuexin2/adenoma_agent/src \
python3 -m adenoma_agent.cli eval-run \
  --run-dir /data15/data15_5/yuexin2/adenoma_agent/artifacts/runs/smoke
```

## Runtime Notes

- Main control code targets `Python 3.8+`
- The controller uses standard Python plus `PyYAML`, `Pillow`, and `numpy`
- WSI and `.h5` operations are delegated to the existing `CLAM` environment
- Trace overview generation is delegated to the existing Route C selection script
- Multimodal stages use a provider-agnostic backend chain:
  - `external_command`
  - `local_patho_r1`
  - `heuristic`

## Integration Sources

The MVP wraps these existing scripts instead of reimplementing them:

- `/data15/data15_5/yuexin2/adenoma/scripts/patho_r1_route_c_select.py`
- `/data15/data15_5/yuexin2/adenoma/scripts/route_c_boxes_to_h5.py`
- `/data15/data15_5/yuexin2/adenoma/scripts/export_patch_samples.py`
