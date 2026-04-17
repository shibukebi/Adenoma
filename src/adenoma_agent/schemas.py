from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CaseSpec:
    case_id: str
    slide_path: str
    task_type: str
    question: str
    label: Optional[str] = None
    binary_target: Optional[int] = None
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
    criteria_hits: Dict[str, str]
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
    binary_target: Optional[int]
    final_binary_prediction: Dict[str, Any]
    report_checklist: Dict[str, Any]
    pathological_report: str
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
