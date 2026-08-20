# adenoma_agent

`adenoma_agent` is a contract-first, non-clinical WSI evidence workflow for colorectal
adenoma and polyp review. The current canonical workflow is defined by
[docs/Agent_workflow.md](docs/Agent_workflow.md):

```text
Raw WSI
  -> Mucosa Extractor
  -> five_x_patch_manifest.jsonl
  -> 5x Mucosal Architecture Predictor
  -> Spatial Evidence + ROI Manager
  -> immutable Evidence Snapshot
  -> single-step Planner
  -> ROI Reviewer
  -> append-only Ledger + resnapshot
  -> Chief + versioned Knowledge Base
  -> final or uncertain 11-class result
```

The `Trace -> Navigate -> Observe(+Chief)` experiment code may still exist for historical
artifact compatibility, but it no longer defines the current workflow or documentation.

## Current implementation

- `src/adenoma_agent/mucosa_extractor.py`: raw-WSI/tile-prediction bridge to tissue-context evidence and the coordinate-aligned 5x manifest.
- `src/adenoma_agent/agentflow/architecture_runtime.py`: formal 5x prediction contract and replaceable model port.
- `src/adenoma_agent/agentflow/spatial.py`: deterministic soft ratios, clusters, mixing, heterogeneity and ROI candidates.
- `src/adenoma_agent/agentflow/ledger.py`: append-only evidence records and immutable snapshots.
- `src/adenoma_agent/agentflow/planner.py`: pure reducer, hypothesis ranking, action binding and stop policy.
- `src/adenoma_agent/agentflow/reviewer.py`: Registry-governed Reviewer requests, observations and failure records.
- `src/adenoma_agent/agentflow/orchestrator.py`: crop/invoke/validate/retry/append/resnapshot loop.
- `src/adenoma_agent/agentflow/chief.py`: rule-based final/uncertain projection and versioned guideline retrieval.
- `scripts/run_agentflow_v1.py`: isolated AgentFlow v1 entrypoint.

The current control plane does not silently fabricate missing medical inference. If a
validated Architecture or Reviewer backend is unavailable, the run must fail explicitly or
append an invocation failure. Synthetic/scripted inputs require explicit opt-in and remain
non-clinical.

## Formal boundaries

- `five_x_patch_manifest.jsonl` is the only canonical Mucosa-to-Architecture interface.
- Architecture primary scale is `5x`; `10x/20x` are downstream Reviewer ROI scales.
- Planner reads one immutable Snapshot and emits one `PlanDecision`; it does not crop images or write the Ledger.
- Reviewer sees only a neutral task, an existing ROI candidate and Registry-approved context.
- `not_evaluable`, missing evidence and invocation failure are never negative evidence.
- High-grade/definite dysplasia is an independent axis and is not gated by an early morphology branch.
- Chief emits a final label only for `diagnostic_ready` evidence with no unresolved major conflict; otherwise it emits an uncertain result.

## Configuration

- [AgentFlow runtime](configs/agentflow/runtime_v1.yaml)
- [Structured Knowledge Base](configs/agentflow/knowledge_base_v1.json)
- [Reviewer Registry](configs/agentflow/reviewer_registry_v1.json)

## Usage

Build Mucosa Extractor outputs from a WSI:

```bash
python3 scripts/run_mucosa_extractor.py \
  --wsi /path/to/slide.svs \
  --output-dir artifacts/example/mucosa \
  --pathprism-url http://127.0.0.1:8400/predict
```

Run the control plane from formal 5x predictions:

```bash
PYTHONPATH=src python3 scripts/run_agentflow_v1.py \
  --case-id case_001 \
  --architecture-predictions-jsonl /path/to/architecture_predictions.jsonl \
  --output-dir artifacts/agentflow/case_001
```

The CLI also accepts `--mucosa-output-dir` or `--five-x-manifest`. A real Architecture
predictor and Reviewer backend must be configured before those inputs can produce medically
meaningful results.

## Documentation

- [Documentation map](docs/README.md)
- [Canonical workflow](docs/Agent_workflow.md)
- [Implementation and model gaps](docs/model/agentflow_architecture_and_model_gaps.md)
- [Mucosa Extractor v1](docs/model/mucosa_extractor_v1.md)
- [Reviewer/Chief contracts](docs/model/contracts/README.md)

## Model status

Still requiring training, calibration or strong task-level validation:

- 5x Mucosal Architecture Classifier on pathologist-annotated ROI data;
- high-recall 20x dysplasia/atypia hotspot proposal model;
- Reviewer VLM feature-level validation and calibration, especially dysplasia, SSL-vs-HP, TSA, villous extent and inflammatory mimics;
- clinical calibration of Mucosa thresholds, uncertainty and stop-policy thresholds;
- a curated, versioned guideline knowledge base.

See the [model-gap report](docs/model/agentflow_architecture_and_model_gaps.md) for the full
training and acceptance checklist.

## Repository layout

- `configs/agentflow/`: current AgentFlow contracts and runtime policy.
- `docs/`: canonical workflow, current contracts and research notes.
- `scripts/`: extraction, inference, experiment and validation entrypoints.
- `src/adenoma_agent/agentflow/`: current control-plane package.
- `tests/`: contract, orchestration and model-boundary tests.
- `artifacts/`: generated local outputs; not a source-of-truth contract.
