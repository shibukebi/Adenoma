# Frozen CONCH 5x Architecture Baseline — Report

Status: code/configuration baseline prepared; results are pending manual
execution on physical GPU 1 (logical `cuda:0`). No scientific performance
conclusion is made yet.

## Machine-readable state

```text
IMPLEMENTATION_READY=true
GPU_EXECUTION_COMPLETE=false
OOF_COMPLETE=false
LOCKED_BENCHMARK_COMPLETE=false
```

These flags describe implementation and execution state only; they do not
constitute a scientific or clinical result.

The formal manual procedure is
[`architecture_baseline_runbook.md`](architecture_baseline_runbook.md).

## Experiment identity

- Configuration: `configs/architecture_baselines/frozen_conch_5x_mil_v1.yaml`
- Primary cohort: YX `.svs`, seven slide-level morphology classes
- Representation: frozen CONCH ViT-B/16, pre-projection, unnormalized
- Models: MeanPool, ABMIL, DSMIL
- Split: seed-17 stratified grouped five-fold OOF protocol
- Device policy: `cuda:0` required; CPU fallback forbidden

## Current result state

| Component | Status | Result |
| --- | --- | --- |
| Data audit | implemented | See `architecture_baseline_data_audit.md` |
| Baseline 1 patch/ROI probe | annotation_blocked | No real patch/ROI labels available |
| Mucosa 5x preprocessing | pending execution | Must run on GPU-visible host |
| Frozen CONCH cache | pending execution | Must validate checkpoint and manifest hashes |
| MeanPool / ABMIL / DSMIL | pending execution | No metrics available yet |
| OOF evaluation | pending execution | Requires completed fold predictions |
| Real YX smoke | pending execution | Requires successful GPU preflight |

## Interpretation rules

Macro-F1, balanced accuracy, per-class metrics, confusion matrices, and
family-cluster bootstrap intervals will be reported only after test-fold
predictions are produced without leakage. Attention and instance scores are
reported as `candidate diagnostic relevance`, not validated pathology
explanations.

The experiment can support a representation-signal conclusion only when the
pre-registered baselines, split protocol, provenance checks, and GPU smoke
all pass. It cannot establish clinical probability, clinical utility, or
patch-level annotation performance.

## Pending result table

| Model | OOF macro-F1 | Balanced accuracy | 95% family-bootstrap CI | Status |
| --- | ---: | ---: | --- | --- |
| Majority baseline | — | — | — | pending |
| MeanPool | — | — | — | pending |
| ABMIL | — | — | — | pending |
| DSMIL | — | — | — | pending |

## Execution closure

The report must be updated after manual execution with the resolved GPU
environment, artifact hashes, fold/seed coverage, failed or skipped slides,
all metrics, and an explicit statement of whether the result is positive,
negative, or inconclusive for the narrow representation-signal question.
