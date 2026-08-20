# Current Architecture and Behavioral Runtime Audit

Status values in this document are restricted to `implemented`, `partial`,
`schema_only`, `contract_only`, `missing`, `legacy/conflicting`, and
`dependency_blocked`.

## Initial implementation matrix

| Component | Status | Runtime Path | Contract/Schema | Tests | Remaining Gap |
| --- | --- | --- | --- | --- | --- |
| AgentState | missing | No persistent state object in the initial runtime | None | None | Add one replayable structured state and transition store |
| HypothesisGenerator | partial | `agentflow/planner.py::HypothesisEngine` | Seven hypotheses in the existing KB | AgentFlow v1 smoke | Initial implementation recomputes a transient `0.35`-based plausibility score |
| ReviewerDispatcher | implemented | `PlanningAgent` → `build_reviewer_request` | Registry and ReviewerTaskRequestV1 | Reviewer semantic tests | Preserve the single formal request boundary |
| Reviewer Runtime | partial | `AgentFlowOrchestrator` invokes a backend | Request, observation and ledger schemas | Scripted and failure-path tests | Only scripted/unavailable backends were connected initially |
| Evidence Analysis | implemented | `reviewer_observation_to_evidence` | ReviewerObservationV1 and EvidenceRecord | Contract and AgentFlow tests | Keep observation separate from diagnostic effects |
| Evidence Scoring | partial | `EvidenceReducer` and `HypothesisEngine` | KB feature lists | Missing dedicated behavioral tests | Add feature type, diagnostic weight, independence and traceable effects |
| EvidenceLedger | implemented | `agentflow/ledger.py` | EvidenceRecord JSONL envelope | Immutability and replay tests | Add correction/supersede semantics without overwrite |
| ReviewerPlanner | partial | `PlanningAgent` and `ActionBinder` | PlanDecision and Reviewer Registry | Scripted E2E | Consume persistent belief, relations, failures and discriminator state |
| BeliefUpdater | missing | None | None | None | Add deterministic active-evidence recomputation |
| Contradiction/Heterogeneity | partial | Same-scope conflict flag in `EvidenceReducer` | ConflictObject | Weak indirect coverage | Separate true contradiction from cross-ROI heterogeneity |
| StopDecider | partial | `StopPolicy` | Four legacy-compatible stop reasons | Diagnostic-ready smoke | Add explicit continue/sufficient/budget/unresolved/failure termination state |
| Chief | partial | `RuleBasedChiefAgent` | ChiefDecision | Scripted E2E | Consume AgentState summary and emit a complete decision trace |
| AgentTrace | partial | Ledger, plan and invocation JSONL files | Per-artifact contracts | Ledger replay only | Add state transitions, belief updates and plan-causality replay |

## Reviewer runtime verification

The initial current AgentFlow already executes the following real control path:

```text
PlanDecision
→ build_reviewer_request
→ Reviewer backend invocation
→ Reviewer Registry validation
→ Reviewer JSON Schema validation when jsonschema is installed
→ ReviewerObservation
→ reviewer_observation_to_evidence
→ append-only EvidenceLedger
```

The six logical Reviewer profiles and their feature ownership are therefore a
runtime compatibility boundary, not merely documentation.  The initial backend
implementations are `ScriptedReviewerBackend` for deterministic non-clinical
tests and `UnavailableReviewerBackend` for explicit failure.  A production HTTP
adapter remains an integration gap.

## Initial data and dependency audit

- `Adenoma_yx` contains 1,608 root-level `.svs` files.  System OpenSlide can
  open the first deterministic candidate and reports 40x with valid MPP.
- `Adenoma_hp` contains 2,496 root-level `.isyntax` files.  The current runtime
  lacks `pyisyntax`, so this source is initially `dependency_blocked`.
- The two source roots contain no local masks, annotations or manifests and are
  treated as read-only.
- `data/label/Adenoma_filtered.xlsx` is evaluation metadata.  It may be used
  for eligibility and post-hoc evaluation only and must never enter inference
  state, Reviewer input, belief update or Chief input.
- A real validated 5x Architecture checkpoint and a configured real Reviewer
  backend are not currently available.
- The project Python is 3.10.  The system Python 3.6 is unsupported.  The
  optional `jsonschema>=4.18` dependency is absent in the initial environment.
- CUDA is unavailable.  Torch imports only with the project environment's
  NVIDIA library ordering, so real model execution must support CPU or report
  an exact dependency boundary.

## Baseline tests

- Current AgentFlow v1: 8/8 passed.
- Reviewer semantic contract tests: 8/8 passed.
- Reviewer Draft 2020-12 schema tests: 3 skipped because `jsonschema>=4.18` is
  not installed.
- The broader legacy suite has pre-existing Trace/Grid/Chief failures.  These
  are recorded as legacy regression debt; the behavioral integration must not
  introduce additional failures.

This file is updated again after implementation with the final matrix, runtime
paths, behavioral gates and real-data integration boundaries.

## Final implementation matrix

| Component | Status | Runtime Path | Contract/Schema | Tests | Remaining Gap |
| --- | --- | --- | --- | --- | --- |
| AgentState | implemented | `agentflow/state.py::AgentState`, `StateStore` | `agent_state_v1` JSONL snapshots | behavioral 1, 7, 9, 11, 16 | Uncalibrated evidence rankings remain non-clinical |
| HypothesisGenerator | implemented | `EvidenceEngine.initialize_state`, deterministic state updates | Seven existing KB hypotheses plus independent scope status | behavioral 1, 3, 16 | No learned prior; intentionally starts at zero evidence score |
| ReviewerDispatcher | implemented | `PlanningAgent` → `build_reviewer_request` | Existing single ReviewerTaskRequestV1 | behavioral 2, 7, 8; Reviewer 11/11 | None in the control plane |
| Reviewer Runtime | implemented | `ScriptedReviewerBackend`, `HttpReviewerBackend`, Orchestrator validation/retry | Existing request/observation/ledger schemas | behavioral 7, 8, 9; Reviewer 11/11 | Real service endpoint is not currently available |
| Evidence Analysis | implemented | `reviewer_observation_to_evidence` | ReviewerObservationV1 → immutable EvidenceRecord | behavioral 3, 4, 12 | Reviewer models still require task-level validation |
| Evidence Scoring | implemented | `EvidenceEngine._effects` | Explicit `EvidenceEffect` with `kb_rule_id` | behavioral 4, 5, 12 | Weights are Agent heuristics, not likelihood ratios |
| EvidenceLedger | implemented | `agentflow/ledger.py` | Append-only JSONL and immutable snapshots | behavioral 6, 11, 16 | None for MVP semantics |
| ReviewerPlanner | implemented | `PlanningAgent._plan_from_state` | Existing PlanDecision and Reviewer Registry | behavioral 2, 7, 8, 10 | Priority is explicit heuristic, not learned EIG |
| BeliefUpdater | implemented | `EvidenceEngine.update` | Deterministic BeliefUpdate | behavioral 4-7, 11, 14, 16 | Softmax output is an uncalibrated ranking only |
| Contradiction/Heterogeneity | implemented | `EvidenceRelation`, `ContradictionState`, recheck ROI binding and append-only adjudication | Existing feature disagreement request field | behavioral 8, 13, 17 | Does not claim gland-level identity |
| StopDecider | implemented | `BehavioralStopPolicy` | Five termination statuses | behavioral 9, 10 | Clinical stopping thresholds require validation |
| Chief | implemented | `RuleBasedChiefAgent.decide` plus decision trace sidecar | Existing ChiefDecision contract | behavioral 7, 11, 15 | Guideline entries remain unconfigured |
| AgentTrace | implemented | `AgentTraceStore`, `state/snapshots.jsonl`, `trace/events.jsonl` | `agent_trace_event_v1` | behavioral 7, 11 | None for deterministic replay |

## Final runtime verification

The implemented runtime path is:

```text
Architecture/index evidence
→ persistent seven-hypothesis AgentState
→ unresolved DiscriminatorState
→ existing Reviewer request and runtime
→ diagnosis-free ReviewerObservation
→ immutable EvidenceRecord
→ hypothesis-relative EvidenceEffect
→ deterministic belief recomputation
→ contradiction/heterogeneity/discriminator update
→ evidence-caused re-plan or five-way stop
→ traceable Chief decision
```

The adaptive deterministic test records `Q_SSL_HP_CRYPT_BASE` in the first
plan, appends Reviewer feature evidence, then records `Q_DYSPLASIA` in the
second plan.  The second PlanDecision and replay event both carry the first
Reviewer evidence IDs in `caused_by_evidence_ids`.

## Final real-data boundaries

- `Adenoma_yx`: the source-safe deterministic case reached OpenSlide read,
  verified 40x/MPP metadata, a SHA-256-matched real UNI+PathPrism Mucosa
  result, and a de-identified canonical 5x manifest with 29 patches.  It stops
  at the missing real 5x Architecture prediction/checkpoint boundary.
- `Adenoma_hp`: a deterministic iSyntax case was retained for audit and stops
  explicitly at `pyisyntax` dependency availability.  No PIL fallback occurs.
- Public real-smoke identities use neutral `CASE_*` aliases and `SOURCE_*`
  codes.  Diagnostic cohort names are confined to local provenance.
- Gate Real-B is implemented in `run_agentflow_real_smoke.py` and runs only
  when real, embedding-traceable 5x predictions and a real Reviewer endpoint
  are supplied.  Neither dependency is currently present, so no synthetic
  evidence was substituted.

## Final validation summary

- Reviewer compatibility boundary: 11/11 passed.
- New deterministic behavioral tests: 17/17 passed.
- Required AgentFlow, Reviewer, Mucosa, Architecture, 11-class, dysplasia and
  WSI set: 81/81 passed.
- Full discovery: 182 total, 170 passed, 8 failed and 4 errored.  The 12
  non-passing tests are the same legacy Trace/Grid/legacy-Chief expectation
  mismatches recorded in the initial audit and are outside the current
  AgentFlow runtime.
