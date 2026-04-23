from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CaseSpec:
    case_id: str
    slide_path: str
    task_type: str
    question: str
    label: Optional[str] = None
    serrated_target: Optional[int] = None
    ssl_like_target: Optional[int] = None
    dysplasia_proxy_target: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class TraceCluster:
    cluster_id: str
    cluster_bbox_thumb: Dict[str, int]
    cluster_bbox_level0: Dict[str, int]
    regions_thumb: List[Dict[str, int]]
    regions_level0: List[Dict[str, int]]
    l: str
    s: int
    d: bool
    review_stage: str
    crypt_disorder_risk: int
    dysplasia_review_needed: bool
    desc: str
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class NavigationStep:
    step_id: str
    x: int
    y: int
    m: float
    o: str
    review_goal: str
    stage_gate: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class ObservationRecord:
    step_id: str
    crop_path: str
    observation: str
    reasoning: str
    next_step: str
    level_1_findings: List[str]
    level_2_findings: List[str]
    level_3_findings: List[str]
    stage_decision: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class ReasoningState:
    hypotheses: List[str]
    supporting_evidence: List[str]
    conflicts: List[str]
    stop_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class InterventionEvent:
    override_roi: Optional[str] = None
    force_zoom: Optional[float] = None
    early_stop: Optional[int] = None
    operator_note: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class AuditReport:
    status: str
    warnings: List[str]
    errors: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class CaseResult:
    case_id: str
    serrated_target: Optional[int]
    ssl_like_target: Optional[int]
    dysplasia_proxy_target: Optional[int]
    hierarchical_prediction: Dict[str, Any]
    serrated_checklist: Dict[str, Any]
    ssl_like_crypt_checklist: Dict[str, Any]
    dysplasia_checklist: Dict[str, Any]
    integrated_report: str
    segmentation_artifact: Dict[str, Any]
    trace_clusters: List[Dict[str, Any]]
    trajectory: List[Dict[str, Any]]
    evidence_chain: List[Dict[str, Any]]
    cost: Dict[str, Any]
    timing: Dict[str, Any]
    audit: Dict[str, Any]
    label: Optional[str] = None
    status: str = "ok"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
