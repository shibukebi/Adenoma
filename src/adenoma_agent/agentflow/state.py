import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from adenoma_agent.agentflow.contracts import DictMixin, EvidenceEffect, EvidenceRelation


HYPOTHESIS_STATUSES = (
    "active",
    "strengthened",
    "weakened",
    "rejected",
    "resolved",
    "uncertain",
)
DISCRIMINATOR_STATUSES = (
    "unresolved",
    "resolved",
    "blocked",
    "conflicting",
    "spatially_heterogeneous",
    "unavailable",
)
TERMINATION_DECISIONS = (
    "continue",
    "sufficient_evidence_stop",
    "budget_stop",
    "unresolved_stop",
    "failure_stop",
)
DIAGNOSTIC_SCOPE_STATUSES = (
    "adequate",
    "incomplete",
    "out_of_scope_suspected",
)
CONTRADICTION_RESOLUTION_STATUSES = (
    "active",
    "recheck_planned",
    "resolved",
    "blocked",
)


def _bounded(value, name):
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError("{0} must be in [0, 1]".format(name))
    return value


@dataclass(frozen=True)
class HypothesisState(DictMixin):
    hypothesis_id: str
    pathway: str
    subtype: str
    evidence_score: float = 0.0
    ranking_score: float = 0.0
    status: str = "uncertain"
    supporting_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    contradicting_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    required_features: Tuple[str, ...] = field(default_factory=tuple)
    observed_required_features: Tuple[str, ...] = field(default_factory=tuple)
    missing_required_features: Tuple[str, ...] = field(default_factory=tuple)
    update_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id is required")
        if self.status not in HYPOTHESIS_STATUSES:
            raise ValueError("Unsupported hypothesis status: {0}".format(self.status))
        object.__setattr__(self, "evidence_score", float(self.evidence_score))
        object.__setattr__(self, "ranking_score", _bounded(self.ranking_score, "ranking_score"))
        for name in (
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
            "required_features",
            "observed_required_features",
            "missing_required_features",
            "update_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class DiscriminatorState(DictMixin):
    discriminator_id: str
    target_hypothesis_ids: Tuple[str, ...]
    feature_ids: Tuple[str, ...]
    preferred_reviewer: str
    task_profile: str
    preferred_scales: Tuple[float, ...]
    importance: float
    status: str = "unresolved"
    resolution_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    relation_ids: Tuple[str, ...] = field(default_factory=tuple)
    last_action_id: Optional[str] = None

    def __post_init__(self):
        if not self.discriminator_id or not self.feature_ids:
            raise ValueError("discriminator_id and feature_ids are required")
        if self.status not in DISCRIMINATOR_STATUSES:
            raise ValueError("Unsupported discriminator status: {0}".format(self.status))
        object.__setattr__(self, "target_hypothesis_ids", tuple(self.target_hypothesis_ids))
        object.__setattr__(self, "feature_ids", tuple(self.feature_ids))
        object.__setattr__(self, "preferred_scales", tuple(float(value) for value in self.preferred_scales))
        object.__setattr__(self, "importance", _bounded(self.importance, "importance"))
        object.__setattr__(self, "resolution_evidence_ids", tuple(self.resolution_evidence_ids))
        object.__setattr__(self, "relation_ids", tuple(self.relation_ids))

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class ContradictionState(DictMixin):
    contradiction_id: str
    feature_id: str
    target_roi_ids: Tuple[str, ...]
    evidence_ids: Tuple[str, ...]
    affected_hypothesis_ids: Tuple[str, ...]
    severity: float
    resolution_status: str = "active"
    relation_type: str = "true_contradiction"
    round_created: int = 0

    def __post_init__(self):
        if not self.contradiction_id or not self.feature_id:
            raise ValueError("contradiction_id and feature_id are required")
        if self.resolution_status not in CONTRADICTION_RESOLUTION_STATUSES:
            raise ValueError("Unsupported contradiction resolution status")
        object.__setattr__(self, "target_roi_ids", tuple(self.target_roi_ids))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "affected_hypothesis_ids", tuple(self.affected_hypothesis_ids))
        object.__setattr__(self, "severity", _bounded(self.severity, "severity"))
        object.__setattr__(self, "round_created", int(self.round_created))

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class BeliefUpdate(DictMixin):
    update_id: str
    round_id: int
    input_snapshot_id: str
    evidence_score_before: Mapping[str, float]
    evidence_score_after: Mapping[str, float]
    belief_before: Mapping[str, float]
    belief_after: Mapping[str, float]
    new_evidence_ids: Tuple[str, ...]
    active_evidence_ids: Tuple[str, ...]
    effects: Tuple[EvidenceEffect, ...]
    relations: Tuple[EvidenceRelation, ...]
    resolved_discriminator_ids: Tuple[str, ...] = field(default_factory=tuple)
    remaining_discriminator_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not self.update_id or not self.input_snapshot_id:
            raise ValueError("update_id and input_snapshot_id are required")
        if int(self.round_id) < 0:
            raise ValueError("round_id must be non-negative")
        object.__setattr__(self, "round_id", int(self.round_id))
        for name in ("evidence_score_before", "evidence_score_after", "belief_before", "belief_after"):
            object.__setattr__(self, name, {str(key): float(value) for key, value in dict(getattr(self, name)).items()})
        for name in (
            "new_evidence_ids",
            "active_evidence_ids",
            "resolved_discriminator_ids",
            "remaining_discriminator_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "relations", tuple(self.relations))

    @classmethod
    def from_dict(cls, payload):
        item = dict(payload)
        item["effects"] = tuple(
            value if isinstance(value, EvidenceEffect) else EvidenceEffect.from_dict(value)
            for value in item.get("effects", ())
        )
        item["relations"] = tuple(
            value if isinstance(value, EvidenceRelation) else EvidenceRelation.from_dict(value)
            for value in item.get("relations", ())
        )
        return cls(**item)


@dataclass(frozen=True)
class TerminationState(DictMixin):
    decision: str = "continue"
    reason: Optional[str] = None
    round_id: int = 0
    criteria: Mapping[str, Any] = field(default_factory=dict)
    unresolved_discriminator_ids: Tuple[str, ...] = field(default_factory=tuple)
    blocking_relation_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.decision not in TERMINATION_DECISIONS:
            raise ValueError("Unsupported termination decision: {0}".format(self.decision))
        if int(self.round_id) < 0:
            raise ValueError("round_id must be non-negative")
        if self.decision == "continue" and self.reason is not None:
            raise ValueError("continue termination state must not have a stop reason")
        if self.decision != "continue" and not self.reason:
            raise ValueError("stop termination state requires a reason")
        object.__setattr__(self, "round_id", int(self.round_id))
        object.__setattr__(self, "criteria", dict(self.criteria or {}))
        object.__setattr__(self, "unresolved_discriminator_ids", tuple(self.unresolved_discriminator_ids))
        object.__setattr__(self, "blocking_relation_ids", tuple(self.blocking_relation_ids))

    @property
    def stopped(self):
        return self.decision != "continue"

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class AgentState(DictMixin):
    state_id: str
    case_id: str
    state_version: int
    round_id: int
    ledger_snapshot_id: Optional[str]
    case_context: Mapping[str, Any]
    hypotheses: Tuple[HypothesisState, ...]
    discriminators: Tuple[DiscriminatorState, ...]
    observation_history: Tuple[str, ...] = field(default_factory=tuple)
    evidence_history: Tuple[str, ...] = field(default_factory=tuple)
    evidence_effects: Tuple[EvidenceEffect, ...] = field(default_factory=tuple)
    evidence_relations: Tuple[EvidenceRelation, ...] = field(default_factory=tuple)
    contradictions: Tuple[ContradictionState, ...] = field(default_factory=tuple)
    belief_history: Tuple[BeliefUpdate, ...] = field(default_factory=tuple)
    reviewer_plan: Optional[Mapping[str, Any]] = None
    review_history: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    tool_reviewer_failures: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    budget: Mapping[str, Any] = field(default_factory=dict)
    diagnostic_scope_status: str = "incomplete"
    termination_state: TerminationState = field(default_factory=TerminationState)
    active_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    superseded_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    previous_state_id: Optional[str] = None
    schema_version: str = "agent_state_v1"

    def __post_init__(self):
        if not self.state_id or not self.case_id:
            raise ValueError("state_id and case_id are required")
        if int(self.state_version) < 0 or int(self.round_id) < 0:
            raise ValueError("state_version and round_id must be non-negative")
        if self.diagnostic_scope_status not in DIAGNOSTIC_SCOPE_STATUSES:
            raise ValueError(
                "Unsupported diagnostic_scope_status: {0}".format(
                    self.diagnostic_scope_status
                )
            )
        hypotheses = tuple(self.hypotheses)
        if len({item.hypothesis_id for item in hypotheses}) != len(hypotheses):
            raise ValueError("AgentState hypothesis ids must be unique")
        ranking_total = sum(item.ranking_score for item in hypotheses)
        if hypotheses and abs(ranking_total - 1.0) > 1e-6:
            raise ValueError("Hypothesis ranking_score values must sum to 1")
        object.__setattr__(self, "state_version", int(self.state_version))
        object.__setattr__(self, "round_id", int(self.round_id))
        object.__setattr__(self, "case_context", dict(self.case_context or {}))
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "discriminators", tuple(self.discriminators))
        object.__setattr__(self, "observation_history", tuple(self.observation_history))
        object.__setattr__(self, "evidence_history", tuple(self.evidence_history))
        object.__setattr__(self, "evidence_effects", tuple(self.evidence_effects))
        object.__setattr__(self, "evidence_relations", tuple(self.evidence_relations))
        object.__setattr__(self, "contradictions", tuple(self.contradictions))
        object.__setattr__(self, "belief_history", tuple(self.belief_history))
        object.__setattr__(self, "reviewer_plan", dict(self.reviewer_plan) if self.reviewer_plan else None)
        object.__setattr__(self, "review_history", tuple(dict(value) for value in self.review_history))
        object.__setattr__(self, "tool_reviewer_failures", tuple(dict(value) for value in self.tool_reviewer_failures))
        object.__setattr__(self, "budget", dict(self.budget or {}))
        object.__setattr__(self, "active_evidence_ids", tuple(self.active_evidence_ids))
        object.__setattr__(self, "superseded_evidence_ids", tuple(self.superseded_evidence_ids))

    def hypothesis(self, hypothesis_id):
        for item in self.hypotheses:
            if item.hypothesis_id == hypothesis_id:
                return item
        raise KeyError(hypothesis_id)

    @classmethod
    def from_dict(cls, payload):
        item = dict(payload)
        item["hypotheses"] = tuple(
            value if isinstance(value, HypothesisState) else HypothesisState.from_dict(value)
            for value in item.get("hypotheses", ())
        )
        item["discriminators"] = tuple(
            value if isinstance(value, DiscriminatorState) else DiscriminatorState.from_dict(value)
            for value in item.get("discriminators", ())
        )
        item["evidence_effects"] = tuple(
            value if isinstance(value, EvidenceEffect) else EvidenceEffect.from_dict(value)
            for value in item.get("evidence_effects", ())
        )
        item["evidence_relations"] = tuple(
            value if isinstance(value, EvidenceRelation) else EvidenceRelation.from_dict(value)
            for value in item.get("evidence_relations", ())
        )
        item["contradictions"] = tuple(
            value if isinstance(value, ContradictionState) else ContradictionState.from_dict(value)
            for value in item.get("contradictions", ())
        )
        item["belief_history"] = tuple(
            value if isinstance(value, BeliefUpdate) else BeliefUpdate.from_dict(value)
            for value in item.get("belief_history", ())
        )
        termination = item.get("termination_state", {})
        item["termination_state"] = (
            termination if isinstance(termination, TerminationState) else TerminationState.from_dict(termination)
        )
        return cls(**item)


class StateStore(object):
    """Append-only full-state journal with deterministic JSONL replay."""

    def __init__(self, case_id, jsonl_path=None, latest_path=None, replay=True):
        self.case_id = str(case_id)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.latest_path = (
            Path(latest_path)
            if latest_path
            else (self.jsonl_path.with_name("latest.json") if self.jsonl_path else None)
        )
        self._states = []
        self._by_id = {}
        if self.jsonl_path and replay and self.jsonl_path.exists():
            self._replay()
            if self.latest is not None:
                self._write_latest(self.latest)

    @property
    def states(self):
        return tuple(self._states)

    @property
    def latest(self):
        return self._states[-1] if self._states else None

    def append(self, state):
        state = state if isinstance(state, AgentState) else AgentState.from_dict(state)
        if state.case_id != self.case_id:
            raise ValueError("AgentState case_id does not match StateStore")
        existing = self._by_id.get(state.state_id)
        if existing is not None:
            if existing.to_dict() != state.to_dict():
                raise ValueError("state_id already exists with different content: {0}".format(state.state_id))
            return existing
        if self.latest is not None:
            if state.previous_state_id != self.latest.state_id:
                raise ValueError("AgentState previous_state_id does not match latest persisted state")
            if state.state_version != self.latest.state_version + 1:
                raise ValueError("AgentState state_version must increase by one")
        elif state.previous_state_id is not None:
            raise ValueError("Initial AgentState cannot reference a previous state")
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            envelope = {
                "store_schema_version": "agent_state_store_jsonl_v1",
                "case_id": self.case_id,
                "state_id": state.state_id,
                "previous_state_id": state.previous_state_id,
                "state": state.to_dict(),
            }
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n")
        self._states.append(state)
        self._by_id[state.state_id] = state
        self._write_latest(state)
        return state

    def _write_latest(self, state):
        if not self.latest_path:
            return
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "store_schema_version": "agent_state_store_latest_v1",
            "case_id": self.case_id,
            "state_id": state.state_id,
            "state": state.to_dict(),
        }
        temporary_path = self.latest_path.with_name(self.latest_path.name + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        temporary_path.replace(self.latest_path)

    def _replay(self):
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                envelope = json.loads(line)
                if envelope.get("case_id") != self.case_id:
                    raise ValueError("StateStore case mismatch at line {0}".format(line_number))
                state = AgentState.from_dict(envelope["state"])
                existing = self._by_id.get(state.state_id)
                if existing is not None:
                    if existing.to_dict() != state.to_dict():
                        raise ValueError("Conflicting duplicate AgentState at line {0}".format(line_number))
                    continue
                if self._states:
                    previous = self._states[-1]
                    if state.previous_state_id != previous.state_id:
                        raise ValueError("Broken AgentState chain at line {0}".format(line_number))
                    if state.state_version != previous.state_version + 1:
                        raise ValueError("Non-contiguous AgentState version at line {0}".format(line_number))
                elif state.previous_state_id is not None:
                    raise ValueError("Initial AgentState references a previous state")
                self._states.append(state)
                self._by_id[state.state_id] = state

    @classmethod
    def from_jsonl(cls, case_id, jsonl_path):
        return cls(case_id=case_id, jsonl_path=jsonl_path, replay=True)

    @classmethod
    def for_output_dir(cls, case_id, output_dir, replay=True):
        state_dir = Path(output_dir) / "state"
        return cls(
            case_id=case_id,
            jsonl_path=state_dir / "snapshots.jsonl",
            latest_path=state_dir / "latest.json",
            replay=replay,
        )


class AgentTraceStore(object):
    """Append-only transition journal referencing full AgentState snapshots."""

    def __init__(self, case_id, jsonl_path=None, replay=True):
        self.case_id = str(case_id)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._events = []
        self._by_id = {}
        if self.jsonl_path and replay and self.jsonl_path.exists():
            self._replay()

    @property
    def events(self):
        return tuple(dict(item) for item in self._events)

    def append(self, event):
        payload = dict(event or {})
        event_id = str(payload.get("event_id", ""))
        if not event_id:
            raise ValueError("Agent trace event_id is required")
        if str(payload.get("case_id", "")) != self.case_id:
            raise ValueError("Agent trace case_id mismatch")
        existing = self._by_id.get(event_id)
        if existing is not None:
            if existing != payload:
                raise ValueError("Agent trace event_id already exists with different content")
            return dict(existing)
        required = ("round_id", "actor", "input_state_id", "output_state_id", "action")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError("Agent trace event lacks required fields: {0}".format(missing))
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self._events.append(payload)
        self._by_id[event_id] = payload
        return dict(payload)

    def hypothesis_updates(self, hypothesis_id):
        output = []
        for event in self._events:
            before = dict(event.get("hypothesis_scores_before", {}))
            after = dict(event.get("hypothesis_scores_after", {}))
            if hypothesis_id in before or hypothesis_id in after:
                output.append(
                    {
                        "event_id": event["event_id"],
                        "round_id": event["round_id"],
                        "before": before.get(hypothesis_id),
                        "after": after.get(hypothesis_id),
                        "evidence_ids": list(event.get("evidence_ids", [])),
                        "belief_update_id": event.get("belief_update_id"),
                    }
                )
        return tuple(output)

    def plan_transitions(self):
        """Replay reviewer choices and the evidence explicitly attributed to each choice."""

        output = []
        previous = None
        for event in self._events:
            if event.get("actor") != "ReviewerPlanner":
                continue
            current = {
                "event_id": event["event_id"],
                "round_id": event["round_id"],
                "plan_id": event.get("plan_id"),
                "question_id": event.get("selected_question_id"),
                "reviewer_id": event.get("selected_reviewer_id"),
                "roi_id": event.get("selected_roi_id"),
                "scale": event.get("selected_scale"),
                "termination_status": event.get("termination_status"),
                "priority_trace": dict(event.get("priority_trace", {})),
                "caused_by_evidence_ids": list(event.get("caused_by_evidence_ids", [])),
                "changed_from_previous": False,
            }
            if previous is not None:
                current["changed_from_previous"] = any(
                    current.get(name) != previous.get(name)
                    for name in ("question_id", "reviewer_id", "roi_id", "scale")
                )
                current["previous_plan_id"] = previous.get("plan_id")
            output.append(current)
            previous = current
        return tuple(output)

    def _replay(self):
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("case_id") != self.case_id:
                    raise ValueError("Agent trace case mismatch at line {0}".format(line_number))
                event_id = str(event.get("event_id", ""))
                if not event_id:
                    raise ValueError("Agent trace lacks event_id at line {0}".format(line_number))
                existing = self._by_id.get(event_id)
                if existing is not None:
                    if existing != event:
                        raise ValueError("Conflicting Agent trace event at line {0}".format(line_number))
                    continue
                self._events.append(event)
                self._by_id[event_id] = event

    @classmethod
    def for_output_dir(cls, case_id, output_dir, replay=True):
        return cls(
            case_id=case_id,
            jsonl_path=Path(output_dir) / "trace" / "events.jsonl",
            replay=replay,
        )
