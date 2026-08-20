# Scope

The current project scope is the non-clinical AgentFlow defined in
[docs/Agent_workflow.md](docs/Agent_workflow.md). Documentation and machine contracts use
the following canonical chain:

```text
Mucosa evidence
  -> 5x Architecture evidence
  -> Spatial evidence and ROI candidates
  -> immutable Ledger Snapshot
  -> one-step Planner decision
  -> one Reviewer invocation
  -> append evidence/failure and resnapshot
  -> Chief final-or-uncertain decision
```

## In scope

- build tissue-context evidence and a level-0-aligned `five_x_patch_manifest.jsonl`;
- accept only formal 5x Architecture predictions at the runtime boundary;
- compute deterministic soft ratios, clusters, co-localization, mixing, heterogeneity and uncertainty regions;
- generate deterministic `2.5x / 5x / 10x / 20x` ROI candidates without letting the Planner invent coordinates;
- maintain append-only evidence records and immutable snapshots;
- rank seven fixed morphology hypotheses while preserving an independent dysplasia axis;
- emit one pure `PlanDecision` per planning round;
- enforce Reviewer Registry, neutral-input, feature-ownership, scale and context-view rules;
- persist both valid Reviewer observations and terminal invocation failures with provenance;
- distinguish `diagnostic_ready`, `no_useful_action`, `budget_exhausted` and `non_diagnostic_or_quality_limited`;
- produce a structured Chief `final` or `uncertain` result with evidence IDs and conflicts;
- retrieve management guidance only from a versioned configured knowledge entry after a final diagnosis;
- support explicit scripted/synthetic fixtures for contract tests without presenting them as medical inference.

## Model work still required

- train and calibrate the 5x Mucosal Architecture Classifier on real pathologist labels;
- train or strongly validate a high-recall 20x dysplasia/atypia hotspot proposal model;
- validate and calibrate Reviewer VLM performance at feature and ROI level;
- calibrate Mucosa, uncertainty, spatial, action-score and stop-policy thresholds;
- curate versioned guideline entries and verify final-label management mappings.

## Explicitly out of scope

- claiming clinical or production readiness from contract tests or synthetic smoke runs;
- silently substituting heuristics or synthetic checkpoints for missing medical models;
- using 10x Architecture experiments as the canonical runtime input;
- allowing Planner, Reviewer or Chief to invent pixel coordinates;
- treating missing, `not_evaluable` or invocation-failure records as negative evidence;
- forcing unresolved dysplasia into a non-D class;
- allowing Reviewer to output a final 11-class diagnosis;
- allowing Chief to free-generate guideline citations or management recommendations;
- treating historical `Trace -> Navigate -> Observe(+Chief)` code or artifacts as the current contract.

## Completion criteria for the control plane

- all formal inputs and outputs validate against their dataclass/JSON Schema contracts;
- every evidence item can be traced to case, ROI, scale, model, prompt, action and snapshot;
- every planning round consumes exactly one immutable snapshot and selects at most one action;
- Reviewer failures are recorded and cannot bypass validation into evidence;
- stop reasons and unresolved questions survive to the Chief result;
- final labels are emitted only through the legal morphology plus dysplasia projection;
- targeted AgentFlow, Reviewer contract and regression tests pass in a supported Python environment.

Detailed implementation and training gaps are tracked in
[docs/model/agentflow_architecture_and_model_gaps.md](docs/model/agentflow_architecture_and_model_gaps.md).
