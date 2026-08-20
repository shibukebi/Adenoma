# AgentFlow v1 CI Baseline

This document freezes the current control-plane verification boundary. It does
not change the behavior of `AgentState`, `EvidenceEngine`, `PlanningAgent`,
`BehavioralStopPolicy`, or `Chief`.

## Baseline identity

- Commit message: `Freeze AgentFlow behavioral loop v1 baseline`
- Annotated tag: `agent-behavioral-loop-v1`
- Current workflow: `docs/Agent_workflow.md`

## Current marker

`tests/conftest.py` marks the explicit current AgentFlow test set with
`agentflow`. The formal current set is:

- `test_reviewer_contract_schemas.py`: 11 tests
- `test_agent_behavioral_loop.py`: 17 tests
- focused set: 81 tests across AgentFlow, Reviewer, Mucosa, Architecture,
  WSI, 11-class and dysplasia modules

Run the strict gate with:

```bash
python scripts/run_agentflow_ci.py
```

The command requires supported Python 3.10, `pytest`, `jsonschema`, and an
importable PyTorch Architecture environment. Failures, errors, or skips are
not accepted in the current baseline.

## Legacy marker

All tests outside the explicit current allowlist are marked `legacy`. Run:

```bash
python scripts/run_legacy_ci.py
```

Legacy failures are controlled by
`tests/legacy_expected_failures.txt`. A new failure fails CI; a repaired test
must be removed from that file in the same reviewed change.

## Required P0 gates

Every later change must keep all of the following green:

```text
Reviewer contract: 11/11
Behavioral control-plane: 17/17
Focused current regression: 81/81
```
