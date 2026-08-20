from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


EVIDENCE_STATUSES = (
    "present",
    "absent",
    "uncertain",
    "not_evaluable",
    "measurement",
    "invocation_failure",
)
FEATURE_EVALUABILITY = ("adequate", "limited", "not_evaluable")
DYSPLASIA_STATES = ("unassessed", "not_evaluable", "not_supported", "supported", "conflicting")
STOP_REASONS = (
    "diagnostic_ready",
    "no_useful_action",
    "budget_exhausted",
    "non_diagnostic_or_quality_limited",
)
TERMINATION_STATUSES = (
    "continue",
    "sufficient_evidence_stop",
    "budget_stop",
    "unresolved_stop",
    "failure_stop",
)
EVIDENCE_EFFECT_DIRECTIONS = ("support", "oppose", "neutral", "uncertain")
EVIDENCE_RELATION_TYPES = (
    "corroboration",
    "exact_duplicate",
    "correlated_overlap",
    "cluster_saturation",
    "true_contradiction",
    "spatial_heterogeneity",
    "quality_disagreement",
    "correction",
    "supersede",
)


def _bounded(value, name):
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError("{0} must be in [0, 1]".format(name))
    return value


def _bbox(value):
    values = tuple(int(item) for item in (value or ()))
    if values and (
        len(values) != 4
        or values[0] < 0
        or values[1] < 0
        or values[2] <= values[0]
        or values[3] <= values[1]
    ):
        raise ValueError("level0_bbox must be empty or [x1, y1, x2, y2]")
    return values


class DictMixin(object):
    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ArchitecturePatchPrediction(DictMixin):
    """Architecture model output for one canonical 5x manifest patch."""

    patch_id: str
    slide_id: str
    level0_bbox: Tuple[int, int, int, int]
    mucosa_coverage: float
    evaluable: float
    architecture: Mapping[str, float]
    context: Mapping[str, float]
    uncertainty: float
    source_model: str
    scale: float = 5.0
    component_ids: Tuple[int, ...] = field(default_factory=tuple)
    dysplasia_risk: float = 0.0
    abnormal_epithelial_score: float = 0.0
    embedding_ref: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.patch_id or not self.slide_id:
            raise ValueError("patch_id and slide_id are required")
        object.__setattr__(self, "level0_bbox", _bbox(self.level0_bbox))
        if abs(float(self.scale) - 5.0) > 1e-6:
            raise ValueError("Agent_workflow canonical architecture input must be 5x")
        object.__setattr__(self, "mucosa_coverage", _bounded(self.mucosa_coverage, "mucosa_coverage"))
        object.__setattr__(self, "evaluable", _bounded(self.evaluable, "evaluable"))
        object.__setattr__(self, "uncertainty", _bounded(self.uncertainty, "uncertainty"))
        object.__setattr__(self, "dysplasia_risk", _bounded(self.dysplasia_risk, "dysplasia_risk"))
        object.__setattr__(
            self,
            "abnormal_epithelial_score",
            _bounded(self.abnormal_epithelial_score, "abnormal_epithelial_score"),
        )
        required_architecture = {"serrated", "tubular", "villous"}
        required_context = {"normal_mucosa_present", "reactive_inflammatory_present", "other_pattern_present"}
        if set(self.architecture) != required_architecture:
            raise ValueError("architecture must contain serrated, tubular, and villous scores")
        if set(self.context) != required_context:
            raise ValueError("context must contain normal, reactive/inflammatory, and other scores")
        object.__setattr__(
            self,
            "architecture",
            {key: _bounded(value, "architecture.{0}".format(key)) for key, value in self.architecture.items()},
        )
        object.__setattr__(
            self,
            "context",
            {key: _bounded(value, "context.{0}".format(key)) for key, value in self.context.items()},
        )
        object.__setattr__(self, "component_ids", tuple(int(item) for item in self.component_ids))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class SpatialEvidenceSummary(DictMixin):
    slide_id: str
    ratios: Mapping[str, float]
    spatial: Mapping[str, float]
    uncertainty: Mapping[str, float]
    source_patch_ids: Tuple[str, ...]
    schema_version: str = "spatial_evidence_v1"


@dataclass(frozen=True)
class ROICandidate(DictMixin):
    roi_id: str
    slide_id: str
    level0_bbox: Tuple[int, int, int, int]
    scale: float
    roi_semantics: str
    candidate_features: Tuple[str, ...]
    allowed_reviewers: Tuple[str, ...]
    suitability: float
    spatial_coverage: float
    estimated_cost: float
    source_patch_ids: Tuple[str, ...] = field(default_factory=tuple)
    source_cluster_id: Optional[str] = None
    image_path: Optional[str] = None
    image_sha256: Optional[str] = None
    pixel_dimensions: Tuple[int, ...] = field(default_factory=tuple)
    mpp: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.roi_id or not self.slide_id:
            raise ValueError("roi_id and slide_id are required")
        object.__setattr__(self, "level0_bbox", _bbox(self.level0_bbox))
        if float(self.scale) not in (2.5, 5.0, 10.0, 20.0):
            raise ValueError("Unsupported ROI scale: {0}".format(self.scale))
        object.__setattr__(self, "suitability", _bounded(self.suitability, "suitability"))
        object.__setattr__(self, "spatial_coverage", _bounded(self.spatial_coverage, "spatial_coverage"))
        object.__setattr__(self, "estimated_cost", _bounded(self.estimated_cost, "estimated_cost"))
        object.__setattr__(self, "candidate_features", tuple(self.candidate_features))
        object.__setattr__(self, "allowed_reviewers", tuple(self.allowed_reviewers))
        object.__setattr__(self, "source_patch_ids", tuple(self.source_patch_ids))
        if self.image_sha256 is not None:
            digest = str(self.image_sha256)
            if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
                raise ValueError("image_sha256 must be a 64-character hexadecimal digest")
        dimensions = tuple(int(value) for value in self.pixel_dimensions)
        if dimensions and (len(dimensions) != 2 or any(value <= 0 for value in dimensions)):
            raise ValueError("pixel_dimensions must be empty or [width, height]")
        object.__setattr__(self, "pixel_dimensions", dimensions)
        if self.mpp is not None and float(self.mpp) <= 0.0:
            raise ValueError("mpp must be positive when provided")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class EvidenceRecord(DictMixin):
    evidence_id: str
    case_id: str
    evidence_type: str
    feature: str
    status: str
    confidence: float
    source: str
    source_version: str
    schema_version: str = "evidence_record_v1"
    value: Optional[float] = None
    quality: float = 1.0
    feature_evaluability: str = "adequate"
    roi_id: Optional[str] = None
    scale: Optional[float] = None
    level0_bbox: Tuple[int, ...] = field(default_factory=tuple)
    patch_id: Optional[str] = None
    cluster_id: Optional[str] = None
    planner_action_id: Optional[str] = None
    input_snapshot_id: Optional[str] = None
    model_version: str = ""
    prompt_version: str = ""
    limitations: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    correction_of_evidence_id: Optional[str] = None
    supersedes_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not self.evidence_id or not self.case_id or not self.feature or not self.source:
            raise ValueError("evidence_id, case_id, feature, and source are required")
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError("Unsupported evidence status: {0}".format(self.status))
        object.__setattr__(self, "confidence", _bounded(self.confidence, "confidence"))
        object.__setattr__(self, "quality", _bounded(self.quality, "quality"))
        if self.feature_evaluability not in FEATURE_EVALUABILITY:
            raise ValueError("Unsupported feature_evaluability: {0}".format(self.feature_evaluability))
        if self.status == "absent" and self.feature_evaluability != "adequate":
            raise ValueError("absent evidence requires adequate feature evaluability")
        if self.status == "not_evaluable" and self.feature_evaluability != "not_evaluable":
            raise ValueError("not_evaluable status requires not_evaluable feature evaluability")
        if self.status == "measurement" and self.value is None:
            raise ValueError("measurement evidence requires value")
        object.__setattr__(self, "level0_bbox", _bbox(self.level0_bbox))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "supersedes_evidence_ids", tuple(self.supersedes_evidence_ids or ()))
        if self.correction_of_evidence_id == self.evidence_id:
            raise ValueError("Evidence cannot correct itself")
        if self.evidence_id in self.supersedes_evidence_ids:
            raise ValueError("Evidence cannot supersede itself")

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class EvidenceEffect(DictMixin):
    """Hypothesis-relative use of one active feature-level evidence record."""

    effect_id: str
    evidence_id: str
    hypothesis_id: str
    feature_id: str
    direction: str
    sign: int
    diagnostic_weight: float
    confidence: float
    quality: float
    independence: float
    kb_rule_id: str = ""
    contribution: Optional[float] = None
    cluster_id: Optional[str] = None
    active: bool = True
    suppressed_by_effect_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.effect_id or not self.evidence_id or not self.hypothesis_id or not self.feature_id:
            raise ValueError("effect_id, evidence_id, hypothesis_id, and feature_id are required")
        if self.direction not in EVIDENCE_EFFECT_DIRECTIONS:
            raise ValueError("Unsupported evidence effect direction: {0}".format(self.direction))
        if int(self.sign) not in (-1, 0, 1):
            raise ValueError("Evidence effect sign must be -1, 0, or 1")
        expected_sign = {"support": 1, "oppose": -1, "neutral": 0, "uncertain": 0}[self.direction]
        if int(self.sign) != expected_sign:
            raise ValueError("Evidence effect direction/sign mismatch")
        weight = float(self.diagnostic_weight)
        if weight < 0.0:
            raise ValueError("diagnostic_weight must be non-negative")
        confidence = _bounded(self.confidence, "confidence")
        quality = _bounded(self.quality, "quality")
        independence = _bounded(self.independence, "independence")
        expected = float(self.sign) * weight * confidence * quality * independence
        if not self.active:
            expected = 0.0
        if self.contribution is not None and abs(float(self.contribution) - expected) > 1e-9:
            raise ValueError("Evidence contribution must equal sign*diagnostic_weight*confidence*quality*independence")
        object.__setattr__(self, "diagnostic_weight", weight)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "independence", independence)
        object.__setattr__(self, "contribution", expected)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class EvidenceRelation(DictMixin):
    """A deterministic relationship between immutable Ledger records."""

    relation_id: str
    relation_type: str
    evidence_ids: Tuple[str, ...]
    feature_id: Optional[str] = None
    hypothesis_ids: Tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.relation_id:
            raise ValueError("relation_id is required")
        if self.relation_type not in EVIDENCE_RELATION_TYPES:
            raise ValueError("Unsupported evidence relation type: {0}".format(self.relation_type))
        evidence_ids = tuple(str(value) for value in self.evidence_ids)
        if not evidence_ids or len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Evidence relation requires unique evidence_ids")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "hypothesis_ids", tuple(str(value) for value in self.hypothesis_ids))
        object.__setattr__(self, "confidence", _bounded(self.confidence, "confidence"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class LedgerSnapshot(DictMixin):
    snapshot_id: str
    case_id: str
    records: Tuple[EvidenceRecord, ...]
    created_after_action_id: Optional[str] = None

    @property
    def evidence_ids(self):
        return tuple(record.evidence_id for record in self.records)

    def to_dict(self):
        return {
            "snapshot_id": self.snapshot_id,
            "case_id": self.case_id,
            "evidence_ids": list(self.evidence_ids),
            "created_after_action_id": self.created_after_action_id,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class ReducedFeature(DictMixin):
    feature: str
    resolved_status: str
    strength: float
    quality: float
    spatial_coverage: float
    source_agreement: float
    present_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    absent_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    uncertain_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    not_evaluable_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    measurement: Optional[float] = None
    redundant_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvidenceView(DictMixin):
    snapshot_id: str
    features: Mapping[str, ReducedFeature]
    conflicts: Tuple[str, ...]
    limitations: Tuple[str, ...]


@dataclass(frozen=True)
class HypothesisAssessment(DictMixin):
    hypothesis_id: str
    pathway: str
    subtype: str
    dysplasia_state: str
    dysplasia_applicability: str
    observed_supporting_evidence: Tuple[str, ...]
    observed_conflict_evidence: Tuple[str, ...]
    missing_required_evidence: Tuple[str, ...]
    plausibility_score: float
    required_evidence_coverage: float
    conflict_score: float
    decision_ready: bool


@dataclass(frozen=True)
class DiscriminativeQuestion(DictMixin):
    question_id: str
    target_feature: str
    question: str
    discriminates: Tuple[str, ...]
    expected_effect: Mapping[str, str]
    reviewer_id: str
    task_profile: str
    allowed_scales: Tuple[float, ...]
    priority: float = 0.5


@dataclass(frozen=True)
class ReviewerAction(DictMixin):
    action_id: str
    question_id: str
    reviewer_id: str
    task_profile: str
    roi_id: str
    scale: float
    target_features: Tuple[str, ...]
    goal: str
    score: float
    score_components: Mapping[str, float]


@dataclass(frozen=True)
class PlanDecision(DictMixin):
    plan_id: str
    snapshot_id: str
    decision: str
    ranked_hypotheses: Tuple[HypothesisAssessment, ...]
    selected_question: Optional[DiscriminativeQuestion]
    selected_action: Optional[ReviewerAction]
    stop: bool
    stop_reason: Optional[str]
    unresolved_questions: Tuple[str, ...] = field(default_factory=tuple)
    input_state_id: Optional[str] = None
    termination_status: str = "continue"
    selected_discriminator_id: Optional[str] = None
    priority_trace: Mapping[str, Any] = field(default_factory=dict)
    caused_by_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    active_relation_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.decision not in ("invoke_reviewer", "stop"):
            raise ValueError("decision must be invoke_reviewer or stop")
        if self.stop_reason is not None and self.stop_reason not in STOP_REASONS:
            raise ValueError("Unsupported stop_reason: {0}".format(self.stop_reason))
        if self.stop and self.decision != "stop":
            raise ValueError("stop=True requires decision=stop")
        if not self.stop and self.selected_action is None:
            raise ValueError("non-stop PlanDecision requires selected_action")
        termination_status = self.termination_status
        if self.stop and termination_status == "continue":
            termination_status = {
                "diagnostic_ready": "sufficient_evidence_stop",
                "budget_exhausted": "budget_stop",
                "no_useful_action": "unresolved_stop",
                "non_diagnostic_or_quality_limited": "unresolved_stop",
            }.get(self.stop_reason, "unresolved_stop")
            object.__setattr__(self, "termination_status", termination_status)
        if termination_status not in TERMINATION_STATUSES:
            raise ValueError("Unsupported termination_status: {0}".format(self.termination_status))
        if not self.stop and termination_status != "continue":
            raise ValueError("A reviewer action must use termination_status=continue")
        object.__setattr__(self, "priority_trace", dict(self.priority_trace or {}))
        object.__setattr__(self, "caused_by_evidence_ids", tuple(self.caused_by_evidence_ids))
        object.__setattr__(self, "active_relation_ids", tuple(self.active_relation_ids))


@dataclass(frozen=True)
class ReviewerTaskRequest(DictMixin):
    schema_version: str
    request_id: str
    action_id: str
    plan_id: str
    question_id: str
    snapshot_id: str
    reviewer: str
    model_input: Mapping[str, Any]
    primary_roi: ROICandidate
    context_views: Tuple[Tuple[str, ROICandidate], ...]
    registry_version: str
    prompt_id: str
    prompt_version: str

    def __post_init__(self):
        object.__setattr__(self, "model_input", dict(self.model_input or {}))
        object.__setattr__(self, "context_views", tuple(self.context_views))

    @property
    def planner_action_id(self):
        return self.action_id

    @property
    def reviewer_id(self):
        return self.reviewer

    @property
    def task_profile(self):
        return str(self.model_input.get("task_profile", ""))

    @property
    def target_features(self):
        return tuple(self.model_input.get("target_features", ()))

    def to_dict(self):
        """Serialize only the public ReviewerTaskRequestV1 envelope.

        The materialized ``ROICandidate`` objects and registry version remain
        Orchestrator-side provenance and are deliberately not model-visible.
        """

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "action_id": self.action_id,
            "plan_id": self.plan_id,
            "question_id": self.question_id,
            "snapshot_id": self.snapshot_id,
            "reviewer": self.reviewer,
            "model_input": dict(self.model_input),
        }


@dataclass(frozen=True)
class ReviewerFinding(DictMixin):
    feature_id: str
    status: str
    status_confidence: float
    feature_evaluability: str
    scope: str
    evidence_text: str
    limitations: Tuple[str, ...] = field(default_factory=tuple)
    quantitation: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        if self.status not in ("present", "absent", "uncertain", "not_evaluable"):
            raise ValueError("Unsupported reviewer finding status: {0}".format(self.status))
        object.__setattr__(self, "status_confidence", _bounded(self.status_confidence, "status_confidence"))
        if self.feature_evaluability not in FEATURE_EVALUABILITY:
            raise ValueError("Unsupported feature_evaluability")
        if self.status == "absent" and self.feature_evaluability != "adequate":
            raise ValueError("absent finding requires adequate evaluability")
        if self.status == "not_evaluable" and self.feature_evaluability != "not_evaluable":
            raise ValueError("not_evaluable finding requires not_evaluable evaluability")
        if self.feature_evaluability == "not_evaluable" and self.status != "not_evaluable":
            raise ValueError("not_evaluable feature evaluability requires not_evaluable status")
        if self.scope not in ("roi_overview", "roi_local"):
            raise ValueError("scope must be roi_overview or roi_local")
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if self.feature_id == "villous_component_extent_estimate" and self.quantitation is None:
            raise ValueError("villous_component_extent_estimate requires structured quantitation")
        if self.feature_id != "villous_component_extent_estimate" and self.quantitation is not None:
            raise ValueError("quantitation is only valid for villous_component_extent_estimate")
        if self.quantitation is not None:
            quantitation = dict(self.quantitation)
            if quantitation.get("metric") != "villous_component_extent":
                raise ValueError("Unsupported villous extent quantitation metric")
            if quantitation.get("category") not in (
                "less_than_25_percent",
                "25_to_75_percent",
                "greater_than_75_percent",
                "indeterminate",
            ):
                raise ValueError("Unsupported villous extent quantitation category")
            object.__setattr__(self, "quantitation", quantitation)

    def to_dict(self):
        payload = {
            "feature_id": self.feature_id,
            "status": self.status,
            "status_confidence": self.status_confidence,
            "feature_evaluability": self.feature_evaluability,
            "scope": self.scope,
            "evidence_text": self.evidence_text,
            "limitations": list(self.limitations),
        }
        if self.quantitation is not None:
            payload["quantitation"] = dict(self.quantitation)
        return payload


@dataclass(frozen=True)
class ReviewerObservation(DictMixin):
    schema_version: str
    observation_id: str
    request_id: str
    reviewer: str
    task_profile: str
    primary_roi_id: str
    primary_magnification: float
    target_features: Tuple[str, ...]
    quality: Mapping[str, Any]
    findings: Tuple[ReviewerFinding, ...]
    incidental_findings: Tuple[ReviewerFinding, ...]
    overall_evidence_strength: Mapping[str, Any]
    limitations: Tuple[str, ...]
    does_not_decide_final_diagnosis: bool
    model_version: str = ""
    prompt_version: str = ""

    def __post_init__(self):
        strength = dict(self.overall_evidence_strength or {})
        if strength.get("level") not in ("none", "weak", "moderate", "strong"):
            raise ValueError("overall_evidence_strength.level is invalid")
        strength["score"] = _bounded(strength.get("score", -1.0), "overall_evidence_strength.score")
        object.__setattr__(self, "overall_evidence_strength", strength)
        if not self.does_not_decide_final_diagnosis:
            raise ValueError("Reviewer must not decide final diagnosis")
        object.__setattr__(self, "target_features", tuple(self.target_features))
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "incidental_findings", tuple(self.incidental_findings))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "quality", dict(self.quality or {}))

    @property
    def reviewer_id(self):
        return self.reviewer

    def to_dict(self):
        """Serialize the diagnosis-free ReviewerObservationV1 payload.

        Model and prompt versions are retained internally for the Orchestrator
        ledger provenance, where the public contract places them.
        """

        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "request_id": self.request_id,
            "reviewer": self.reviewer,
            "task_profile": self.task_profile,
            "primary_roi_id": self.primary_roi_id,
            "primary_magnification": self.primary_magnification,
            "target_features": list(self.target_features),
            "quality": dict(self.quality),
            "findings": [finding.to_dict() for finding in self.findings],
            "incidental_findings": [finding.to_dict() for finding in self.incidental_findings],
            "overall_evidence_strength": dict(self.overall_evidence_strength),
            "limitations": list(self.limitations),
            "does_not_decide_final_diagnosis": self.does_not_decide_final_diagnosis,
        }


@dataclass(frozen=True)
class ConflictObject(DictMixin):
    conflict_id: str
    conflict_type: str
    competing_hypothesis_ids: Tuple[str, ...]
    conflicting_evidence_ids: Tuple[str, ...]
    missing_discriminative_evidence: Tuple[str, ...]
    recommended_actions: Tuple[Mapping[str, Any], ...]
    round_index: int
    max_rounds: int
    status: str = "active"


@dataclass(frozen=True)
class ChiefDecision(DictMixin):
    status: str
    final_label: Optional[str]
    final_diagnosis: Optional[str]
    diagnostic_confidence: float
    morphology_hypothesis_id: Optional[str]
    dysplasia_state: str
    supporting_evidence_ids: Tuple[str, ...]
    conflicting_evidence_ids: Tuple[str, ...]
    unresolved_questions: Tuple[str, ...]
    conflicts: Tuple[ConflictObject, ...]
    management_recommendation: Optional[Mapping[str, Any]]
    knowledge_version: str
    non_clinical: bool = True


@dataclass(frozen=True)
class AgentFlowResult(DictMixin):
    case_id: str
    final_snapshot_id: str
    plans: Tuple[PlanDecision, ...]
    chief_decision: ChiefDecision
    spatial_evidence: SpatialEvidenceSummary
    roi_candidates: Tuple[ROICandidate, ...]
    ledger_path: Optional[str]
    synthetic_stub: bool
    non_clinical: bool = True
    final_state_id: Optional[str] = None
    state_path: Optional[str] = None
    trace_path: Optional[str] = None
