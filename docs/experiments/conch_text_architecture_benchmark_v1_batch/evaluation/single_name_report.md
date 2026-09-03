# Frozen CONCH Architecture Evaluation v1

`MODEL_EVALUATION_STARTED=true`  
`MODEL_EVALUATION_COMPLETE=true`  
Benchmark: `conch_text_architecture_benchmark_v1`  
Checkpoint SHA-256: `40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c`  
Prompt SHA-256: `59dde335493321b6a75f23e2550b47e6241fef40e709813f8a718be903120ef4`

Only the frozen `single_class_name` set was executed. The matched ensemble and
morphology-rich sets were skipped because the frozen prompt configuration does
not define an aggregation rule; no aggregation was invented. K=5 was skipped
because no formal zero-shot K=5 protocol exists in the repository. K=10/K=20
were not ready and were not run.

## Table 1 — Primary benchmark (N=94)

| Prompt | Accuracy | Balanced Accuracy | Macro F1 | Serrated F1 | Tubular F1 | Villous F1 |
|---|---:|---:|---:|---:|---:|---:|
| single-name | 0.4681 | 0.4440 | 0.3698 | 0.6094 | 0.0000 | 0.5000 |
| matched ensemble | SKIPPED | SKIPPED | SKIPPED | — | — | — |
| morphology-rich | SKIPPED | SKIPPED | SKIPPED | — | — | — |

Macro-F1 95% family bootstrap CI: `[0.2518501381659276, 0.4597077761599376]`

## Table 2 — Recall and confusion

| Prompt | Serrated Recall | Tubular Recall | Villous Recall |
|---|---:|---:|---:|
| single-name | 0.9750 | 0.0000 | 0.3571 |

Confusion matrix (rows=ground truth, columns=prediction; order serrated,
tubular, villous): `[[39, 0, 1], [40, 0, 0], [9, 0, 5]]`

## Table 3 — Stress robustness (raw N=26; three-class metric N=11)

| Prompt | N | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| single-name | 11 | 0.1818 | 0.3333 | 0.1026 |
| matched ensemble | — | SKIPPED | SKIPPED | SKIPPED |
| morphology-rich | — | SKIPPED | SKIPPED | SKIPPED |

## Error analysis

Primary error counts by ground-truth class and predicted class are recorded in
`error_analysis.json`; row-level errors are in `error_cases.csv` and
`error_cases.jsonl`.

## Low-data and leakage

- K=5 executed: `false` (no formal protocol)
- K=10 executed: `false` (not ready)
- K=20 executed: `false` (not ready)
- label leakage: `0`
- model-output leakage: `0`
- family split leakage: `0`

## Artifacts

- evaluation manifest: `evaluation_manifest.json`
- raw predictions: `raw_predictions.csv`, `raw_predictions.jsonl`
- metrics: `metrics.json`
- prompt status: `prompt_set_status.csv`
