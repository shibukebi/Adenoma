# Agent Behavioral Loop Engineering Report

Date: 2026-08-20
Repository: `/data1/yuexin/Adenoma`
Governing specification: `docs/Agent_workflow.md`

This report covers the current-architecture audit, deterministic behavioral
loop implementation, Reviewer compatibility verification, and read-only real
case integration.  No Reviewer taxonomy or formal Reviewer wire contract was
changed, no 10x Architecture backbone was introduced, and no model was trained.

## 1. Initial Audit

The initial runtime already had a formal 5x boundary, Mucosa extraction,
Architecture experiment code, spatial/ROI primitives, an append-only Evidence
Ledger, six Reviewer contracts and an Orchestrator that could invoke scripted
or unavailable Reviewer backends.  It did not yet have one persistent
AgentState, deterministic hypothesis-relative evidence effects, belief update,
spatial correlation protection, adaptive planning causality, five-way stop, or
full state/trace replay.

| Component | Initial status | Initial runtime finding |
| --- | --- | --- |
| AgentState | missing | No unified persistent state |
| HypothesisGenerator | partial | Transient hypothesis ranking, no persistent competing state |
| ReviewerDispatcher | implemented | PlanningAgent and the single formal request shape existed |
| Reviewer Runtime | partial | Scripted/unavailable backends were connected |
| Evidence Analysis | implemented | ReviewerObservation could be adapted to feature evidence |
| Evidence Scoring | partial | No diagnostic weight, independence or hypothesis-relative effect ledger |
| EvidenceLedger | implemented | Append-only evidence JSONL and snapshots existed |
| ReviewerPlanner | partial | Did not consume persistent belief, failures or relations |
| BeliefUpdater | missing | No deterministic active-evidence recomputation |
| Contradiction/Heterogeneity | partial | Same-scope conflict logic did not separate cross-ROI heterogeneity |
| StopDecider | partial | Legacy four-reason stop only |
| Chief | partial | Did not receive a complete behavioral state summary |
| AgentTrace | partial | Artifact logs existed, but not a replayable state transition chain |

The complete before/after matrix is retained in
`docs/model/current_architecture_behavioral_audit.md`.

## 2. Reviewer Verification

All six existing logical Reviewers have executable runtime routing through the
same compatibility boundary:

1. `QualityMucosaReviewer`
2. `SerratedArchitectureReviewer`
3. `TSAReviewer`
4. `ConventionalArchitectureReviewer`
5. `DysplasiaReviewer`
6. `InflammatoryReactiveReviewer`

Dispatcher and request construction are in
`src/adenoma_agent/agentflow/planner.py` and
`src/adenoma_agent/agentflow/reviewer.py::build_reviewer_request`.
`ScriptedReviewerBackend` is the deterministic non-clinical test runtime;
`HttpReviewerBackend` is the production adapter for an existing shared
Qwen/PathReasoner `/predict` service; `UnavailableReviewerBackend` records an
explicit terminal dependency failure.

Request validation is performed by the existing Reviewer Registry and the
Draft 2020-12 request schema.  Response validation is performed by the
Registry and ReviewerObservationV1 schema before any finding can enter the
Evidence Ledger.  The HTTP adapter exposes only the neutral formal
`model_input`, allows one constrained JSON repair, and retains reviewer,
profile, prompt, backend, model, image hash and ROI provenance.

The executed runtime path is:

```text
ReviewerTaskRequestV1
→ backend invocation
→ Registry validation
→ JSON Schema validation
→ ReviewerObservationV1
→ Reviewer ledger record
→ feature-level EvidenceRecord
```

Reviewer contract regression result: **11/11 passed**.  TSA signature feature
ownership remains disjoint from HGD/dysplasia ownership.

## 3. Data Audit

Both source directories were read only.  No source file was moved, renamed,
modified or deleted.

| Source | Safe inventory | Nested layout | Auxiliary files |
| --- | ---: | --- | --- |
| `Adenoma_hp` | 2,496 `.isyntax` WSI files | No first-level subdirectories | No mask, annotation, manifest or metadata files within depth two |
| `Adenoma_yx` | 1,608 `.svs` WSI files | No first-level subdirectories | No mask, annotation, manifest or metadata files within depth two |

`data/label/Adenoma_filtered.xlsx` is 133,698 bytes.  Its loader returns only a
set of eligible slide stems; diagnosis/type/grade values never leave the
eligibility function.  Ground truth is not accepted by AgentState case context,
Reviewer requests, belief updates, trace events or Chief input.  Nested case
metadata is rejected so a label cannot be hidden under an allowed reference
field.

Real output uses neutral `CASE_nnn_nnn_hash` aliases and `SOURCE_nnn` codes;
cohort names such as `HP` are not exposed to inference.  Original source names,
paths and patch IDs are kept only in local provenance artifacts.

## 4. Implemented

- Added one immutable `AgentState` containing persistent hypotheses,
  discriminators, observation and evidence history, effects, relations,
  contradiction state, plans, failures, budget, scope and termination state.
- Added append-only `StateStore` and `AgentTraceStore` with atomic latest state,
  deterministic replay, score-change explanation and plan-transition replay.
- Initialized all seven existing morphology hypotheses at evidence score zero;
  softmax is explicitly an uncalibrated ranking, not a clinical probability.
- Added persistent DiscriminatorState projected from the existing Knowledge
  Base questions, Reviewer Registry ownership, task profiles and scales.
- Kept morphology Observation separate from hypothesis-relative
  `EvidenceEffect`.
- Added explicit `kb_rule_id`, support/oppose/neutral/uncertain direction,
  diagnostic weight, confidence, quality, independence and signed contribution.
- Implemented required/supportive/contradictory feature absence semantics.
  Supportive absence and not-evaluable findings are neutral; required absence
  opposes; contradictory presence opposes; invocation failure contributes zero.
- Preserved dysplasia as an independent axis.  HGD evidence does not change TSA
  morphology score or TSA subtype ownership.
- Added exact duplicate removal, bbox-IoU correlation down-weighting,
  source-cluster provenance and max-contribution cluster saturation.
- Added correction/supersede events and deterministic recomputation from active
  evidence.  Planner and Chief also reduce only active evidence.
- Added corroboration, true contradiction, spatial heterogeneity, duplicate,
  correlated and saturation relations.
- Added structured `ContradictionState` with feature, target ROI, evidence,
  affected hypotheses, severity and active/recheck/blocked status.  An adequate
  formal recheck appends an adjudication record that supersedes the old
  conflicting pair without overwriting history.
- Added explicit Planner priority components and conflict recheck binding to the
  corresponding ROI/bbox and legal existing Reviewer/profile.
- Added Reviewer availability/failure history.  Terminal failure cannot create
  a morphology observation or diagnostic evidence.
- Added branch-agnostic dysplasia assessment before sufficient-evidence stop.
- Added continue, sufficient-evidence, budget, unresolved and failure stop.
- Added Chief state summary and decision-trace sidecar; Chief cannot bypass an
  unresolved/failed loop to produce an independent classification.
- Added physically consistent SVS/iSyntax reader and ROI crop provenance.  No
  PIL fallback is allowed for unsupported WSI formats.
- Added real smoke Gate A and conditional Gate B.  Gate B runs only with real,
  embedding-traceable 5x predictions and a configured real Reviewer endpoint.

## 5. Files Changed

- `configs/agentflow/knowledge_base_v1.json` — added explicit evidence roles,
  absence semantics and heuristic diagnostic weights without changing the
  seven hypotheses.
- `configs/agentflow/runtime_v1.yaml` — records evidence, correlation, Planner,
  stop and persistence policy.
- `src/adenoma_agent/agentflow/contracts.py` — extends internal evidence,
  relation, plan and result contracts while preserving Reviewer wire shapes.
- `src/adenoma_agent/agentflow/state.py` — implements AgentState,
  ContradictionState, state persistence and trace replay.
- `src/adenoma_agent/agentflow/evidence.py` — implements deterministic effects,
  active-evidence belief updates, correlation, saturation and relations.
- `src/adenoma_agent/agentflow/knowledge.py` — compiles explicit
  hypothesis-relative feature rules from the existing KB.
- `src/adenoma_agent/agentflow/planner.py` — implements state-driven priority,
  adaptive re-planning, recheck binding and five-way stop.
- `src/adenoma_agent/agentflow/reviewer.py` — adds the shared HTTP backend and
  thin observation-to-evidence adapter, including incidental findings.
- `src/adenoma_agent/agentflow/orchestrator.py` — owns state transitions,
  Reviewer execution, validation, failure handling, belief update and trace.
- `src/adenoma_agent/agentflow/chief.py` — consumes final state and active
  evidence without changing Chief taxonomy.
- `src/adenoma_agent/agentflow/wsi_runtime.py` — adds safe deterministic cohort,
  physical ROI cropper and integration-boundary artifacts.
- `src/adenoma_agent/wsi.py` — provides one physical-metadata-aware WSI reader.
- `src/adenoma_agent/mucosa_extractor.py` — reuses the unified reader and emits
  physical 5x provenance.
- `scripts/run_agentflow_v1.py` — exposes the current AgentFlow and HTTP Reviewer
  configuration.
- `scripts/run_agentflow_real_smoke.py` — implements deterministic real Gate A
  and conditional real Gate B without synthetic fallback.
- `tests/test_agent_behavioral_loop.py` — adds the 15 required deterministic
  gates, rejected-hypothesis reactivation and recheck adjudication.
- `tests/test_agentflow_wsi_runtime.py` — validates physical cropping,
  dependency boundaries, safe aliases and label isolation.
- `docs/model/current_architecture_behavioral_audit.md` — records the initial
  and final implementation matrices.
- `docs/model/agent_behavioral_loop_integration_report.md` — this final report.

## 6. Final Runtime Workflow

| Runtime node | Code path |
| --- | --- |
| Case / canonical 5x input | `architecture_runtime.py`, `wsi_runtime.py` |
| HypothesisGenerator | `evidence.py::initialize_state` |
| AgentState | `state.py::AgentState`, `StateStore` |
| ReviewerPlanner | `planner.py::PlanningAgent` |
| Existing Reviewer Runtime | `reviewer.py`, invoked by `orchestrator.py` |
| ReviewerObservation | Existing `ReviewerObservationV1` in `contracts.py` |
| EvidenceAdapter | `reviewer.py::reviewer_observation_to_evidence` |
| EvidenceLedger | `ledger.py::EvidenceLedger` |
| EvidenceEffect / BeliefUpdater | `evidence.py::EvidenceEngine` |
| Contradiction / heterogeneity | `evidence.py`, `state.py::ContradictionState` |
| Re-plan / Stop | `planner.py::BehavioralStopPolicy` |
| Chief | `chief.py::RuleBasedChiefAgent` |
| Replay | `state.py::AgentTraceStore` |

Actual loop:

```text
Case
→ HypothesisGenerator
→ AgentState_t
→ unresolved discriminator
→ ReviewerPlanner
→ existing ReviewerTaskRequestV1
→ existing Reviewer runtime
→ ReviewerObservationV1
→ EvidenceAdapter
→ append-only EvidenceLedger
→ hypothesis-relative EvidenceEffect
→ deterministic BeliefUpdater
→ contradiction / spatial heterogeneity update
→ AgentState_t+1
→ evidence-caused re-plan or stop
→ Chief
```

## 7. Behavioral Validation

| Test | Result | Deterministic assertion |
| --- | --- | --- |
| 1 Competing hypotheses | passed | Ambiguous serrated evidence retains multiple non-rejected hypotheses |
| 2 Reviewer routing | passed | Crypt-base discriminator routes to the existing Serrated profile at 10x |
| 3 TSA ownership | passed | HGD creates a neutral H_TSA morphology effect |
| 4 Evidence direction | passed | One feature supports SSL, opposes HP and is neutral for TSA |
| 5 Quality weighting | passed | Quality 0.2 contributes one fifth of equivalent quality 1.0 evidence |
| 6 Duplicate evidence | passed | Exact duplicate is inactive and does not change score |
| 7 Adaptive re-planning | passed | Plan changes from `Q_SSL_HP_CRYPT_BASE` to `Q_DYSPLASIA` because of the first Reviewer evidence IDs |
| 8 True contradiction | passed | Same target present/absent creates ContradictionState and a legal feature recheck |
| 9 Reviewer failure | passed | Failure creates only invocation failure, blocks availability and ends in failure stop |
| 10 Stop | passed | Continue/budget/unresolved/failure/sufficient semantics do not use one absolute probability threshold |
| 11 Trace replay | passed | Replay explains H score before/after and request X→Y causality |
| 12 Supportive absence | passed | Supportive absence is neutral; required absence and contradictory presence oppose |
| 13 Heterogeneity | passed | Non-overlapping ROI present/absent is heterogeneity, not contradiction |
| 14 Cluster saturation | passed | Three same-cluster observations equal the maximum member contribution, not their sum |
| 15 Label leakage | passed | Flat or nested label metadata is rejected and absent from inference artifacts |

Additional test 16 verifies that append-only correction events can reactivate a
previously rejected hypothesis.  Test 17 verifies that an adequate legal
conflict recheck supersedes the old conflicting pair, clears the active
ContradictionState and changes the next plan instead of repeating to budget.

The central behavioral gate is satisfied: the second Planner event has a
different question, Reviewer, ROI and scale from the first event, and its
`caused_by_evidence_ids` point to the immediately preceding Reviewer evidence.

## 8. Real-Case Integration

### Adenoma_yx

- Cohort selection: deterministic sorted-first OpenSlide-readable, valid
  physical metadata, workbook-eligible `.svs`; public output uses a safe alias.
- Pipeline stage reached: WSI open → 40x/MPP validation → real UNI+PathPrism
  Mucosa → source-safe `five_x_patch_manifest.jsonl` → Architecture boundary.
- Mucosa provenance: an existing real result was reused only after the original
  artifact WSI and read-only source WSI had identical SHA-256; 29 safe 5x rows
  were materialized.
- Reviewer rounds: not run.
- Belief/ranking changed: not applicable because real Architecture evidence is
  unavailable.
- Next plan changed: not applicable.
- Stop reason: `architecture_predictions_unavailable` integration boundary.
- Chief reached: no.
- Blocking dependency: no real validated 5x Architecture checkpoint/prediction
  JSONL with embedding provenance.

### Adenoma_hp

- Cohort selection: deterministic first `.isyntax` is retained as a source-safe
  alias for dependency audit.
- Pipeline stage reached: source discovery; WSI open blocked.
- Reviewer rounds: 0.
- Belief/ranking changed: no.
- Next plan changed: no.
- Stop reason: dependency boundary at legal iSyntax reader.
- Chief reached: no.
- Blocking dependency: `pyisyntax` is not installed.  No PIL or synthetic
  fallback was used.

Artifacts are under `artifacts/agentflow_real_smoke_gate_a_v2/`.  The combined
boundary contains no raw source path, patient identifier or ground-truth
diagnosis.  Real paths exist only in local provenance files.

## 9. Remaining Gaps

| Item | Status | Exact gap |
| --- | --- | --- |
| Behavioral control plane | implemented | Deterministic Reviewer→evidence→belief→re-plan→stop→Chief loop is complete |
| Real 5x Architecture inference | dependency_blocked | No validated checkpoint or embedding-traceable prediction JSONL is available |
| Adenoma_hp iSyntax read | dependency_blocked | Legal `pyisyntax` reader is unavailable |
| Real Gate B execution | dependency_blocked | It needs both real Architecture predictions and a reachable Reviewer endpoint |
| Reviewer medical performance | partial | Existing shared backend still needs Reviewer/profile-level validation and calibration |
| Evidence/stop weights | partial | Heuristic Agent weights are not clinical likelihood ratios or calibrated stopping rules |
| Guideline retrieval content | not implemented | Chief retrieval interface exists, but authoritative guideline entries are empty |
| Clinical management recommendation | not implemented | Explicitly outside this task |

## 10. Tests

- Reviewer contracts: **11 passed, 0 failed, 0 skipped**.
- New behavioral loop: **17 passed, 0 failed, 0 skipped**.
- Required focused regression set covering AgentFlow, Reviewer, Mucosa,
  Architecture experiment/models, 11-class, dysplasia and WSI: **81 passed,
  0 failed, 0 skipped**.
- Full discovery: **182 total; 170 passed, 8 failed, 4 errors, 0 skipped**.
- The 12 non-passing full-suite cases are the same pre-existing legacy
  Trace/Grid/legacy-Chief vocabulary and CONCH fusion expectation mismatches
  recorded in the initial audit.  They are outside the current AgentFlow path.
- Architecture model tests require the project Python 3.10 environment and the
  NVIDIA nvJitLink/cuSparse library ordering used in the recorded command.
- Pillow emitted deprecation warnings only; no required test was skipped.

## 11. Git Status / Risks

The worktree was already heavily dirty before this task.  No `git reset`,
`git clean`, destructive checkout or source-data mutation was performed.

- Modified tracked files include existing README/config/legacy runtime and test
  changes that predate or overlap this work.
- `docs/`, `configs/agentflow/`, `src/adenoma_agent/agentflow/`, real smoke
  scripts and several test files remain untracked as a group in the current
  worktree.  This is a material handoff risk: they must be deliberately added
  to version control rather than assumed to be committed.
- The optional `jsonschema>=4.18` dependency is declared under the `contracts`
  extra and was installed in the `patho-r1` environment for validation.
- Real Architecture, Reviewer service and iSyntax dependencies remain external
  deployment risks.
- Source data remains read-only; local provenance artifacts contain source
  paths and must not be published with inference-facing artifacts.

## Conclusion

The deterministic implementation proves the required behavioral property:

> Reviewer feature evidence changes hypothesis scores and discriminator state,
> and those changes alter the next legal Reviewer action with replayable
> evidence causality.

The control plane is implemented and regression-tested.  Real Gate A reaches
the 5x Architecture boundary for the deterministic YX case.  Real Gate B and
the HP reader remain honestly dependency-blocked; no synthetic evidence was
used to claim otherwise.
