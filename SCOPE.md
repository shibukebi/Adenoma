# Scope

## MVP

The first deliverable is a runnable CPathAgent-style WSI agent stack for `SSA vs others` on `Adenoma_yx`.

The MVP is successful when we can:

- load cases from the aligned `manifest + labels.csv`
- emit `TraceCluster` outputs with `l_k / s_k / d_k`
- generate a finite `(x, y, m, o)` trajectory
- export step crops and English observation / reasoning logs
- emit a checklist-style pathological report and final `SSA vs others` prediction
- emit structured JSONL logs and a final audit report
- replay a case from saved artifacts

The MVP does not require:

- pixel-level ground-truth masks
- official multi-benchmark VQA results
- trainable agents
- LoRA or finetuning
- full ablation tables

## Full Phase

Future expansion targets:

- public VQA / classification dataset adapters
- trainable trace / navigation / observation modules
- pixel-level mask refinement if segmentation labels become available
- benchmark-aligned evaluation and ablations

## Deliverables

- companion project in `/data15/data15_5/yuexin2/adenoma_agent`
- reusable configs and CLI entrypoints
- smoke workflow on `138189_751666001.svs`
- `unittest` coverage for schema, coordinate mapping, FSM flow, replay, and aggregation
