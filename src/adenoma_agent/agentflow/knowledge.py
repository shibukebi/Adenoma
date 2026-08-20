import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from adenoma_agent.agentflow.contracts import DictMixin, DiscriminativeQuestion


@dataclass(frozen=True)
class HypothesisTemplate(DictMixin):
    hypothesis_id: str
    pathway: str
    subtype: str
    dysplasia_applicability: str
    supporting_evidence: Tuple[str, ...]
    contradictory_evidence: Tuple[str, ...]
    required_evidence: Tuple[str, ...]
    discriminative_features: Tuple[str, ...]
    final_label_mapping: Mapping[str, str]
    diagnostic_weights: Mapping[str, float] = field(default_factory=dict)
    absence_semantics: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        supporting = tuple(self.supporting_evidence)
        contradictory = tuple(self.contradictory_evidence)
        required = tuple(self.required_evidence)
        if set(supporting).intersection(contradictory):
            raise ValueError("A feature cannot be both supportive and contradictory for one hypothesis")
        weights = {str(key): float(value) for key, value in dict(self.diagnostic_weights or {}).items()}
        if any(value < 0.0 for value in weights.values()):
            raise ValueError("diagnostic_weights must be non-negative")
        object.__setattr__(self, "supporting_evidence", supporting)
        object.__setattr__(self, "contradictory_evidence", contradictory)
        object.__setattr__(self, "required_evidence", required)
        object.__setattr__(self, "discriminative_features", tuple(self.discriminative_features))
        object.__setattr__(self, "final_label_mapping", dict(self.final_label_mapping or {}))
        object.__setattr__(self, "diagnostic_weights", weights)
        object.__setattr__(self, "absence_semantics", dict(self.absence_semantics or {}))


@dataclass(frozen=True)
class FeatureRule(DictMixin):
    feature_id: str
    roles: Tuple[str, ...]
    diagnostic_weight: float
    present_effect: str
    adequate_absent_effect: str
    uncertain_effect: str = "uncertain"
    not_evaluable_effect: str = "unresolved"

    def __post_init__(self):
        allowed_roles = {"supportive", "required", "contradictory"}
        allowed_effects = {"support", "oppose", "neutral", "uncertain", "unresolved"}
        roles = tuple(self.roles)
        if not roles or not set(roles).issubset(allowed_roles):
            raise ValueError("FeatureRule roles are invalid")
        if self.present_effect not in allowed_effects or self.adequate_absent_effect not in allowed_effects:
            raise ValueError("FeatureRule effect is invalid")
        if self.uncertain_effect not in allowed_effects or self.not_evaluable_effect not in allowed_effects:
            raise ValueError("FeatureRule uncertainty/evaluability effect is invalid")
        if float(self.diagnostic_weight) < 0.0:
            raise ValueError("FeatureRule diagnostic_weight must be non-negative")
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "diagnostic_weight", float(self.diagnostic_weight))


class KnowledgeBase(object):
    def __init__(self, version, hypotheses, questions, guidelines, label_names, evidence_scoring=None):
        self.version = str(version)
        self.hypotheses = tuple(hypotheses)
        self.questions = tuple(questions)
        self.guidelines = dict(guidelines)
        self.label_names = dict(label_names)
        self.evidence_scoring = dict(evidence_scoring or {})
        self._hypothesis_by_id = {item.hypothesis_id: item for item in self.hypotheses}
        self._hypothesis_by_subtype = {item.subtype: item for item in self.hypotheses}
        if len(self.hypotheses) != 7:
            raise ValueError("Agent_workflow v1 requires exactly seven morphology hypotheses")
        if len(self._hypothesis_by_id) != len(self.hypotheses):
            raise ValueError("hypothesis_id values must be unique")

    def hypothesis(self, hypothesis_id):
        return self._hypothesis_by_id[hypothesis_id]

    def hypothesis_for_subtype(self, subtype):
        return self._hypothesis_by_subtype[subtype]

    def guideline_for_label(self, final_label):
        entry = self.guidelines.get(final_label)
        return dict(entry) if entry else None

    def diagnosis_name(self, final_label):
        return self.label_names.get(final_label, final_label)

    def feature_rules(self, hypothesis_id):
        """Compile explicit, hypothesis-relative rules without changing Reviewer contracts."""

        template = self.hypothesis(hypothesis_id)
        defaults = dict(self.evidence_scoring.get("absence_semantics", {}))
        defaults.update(template.absence_semantics)
        default_weight = float(self.evidence_scoring.get("default_diagnostic_weight", 1.0))
        role_weights = dict(
            self.evidence_scoring.get(
                "diagnostic_weights_by_role",
                {"required": 1.0, "contradictory": 0.75, "supportive": 0.50},
            )
        )
        features = sorted(
            set(template.supporting_evidence)
            | set(template.required_evidence)
            | set(template.contradictory_evidence)
        )
        output = []
        for feature_id in features:
            roles = []
            if feature_id in template.supporting_evidence:
                roles.append("supportive")
            if feature_id in template.required_evidence:
                roles.append("required")
            if feature_id in template.contradictory_evidence:
                roles.append("contradictory")
            if "contradictory" in roles:
                present_effect = "oppose"
                absent_effect = defaults.get("contradictory_adequate_absent", "neutral")
            elif "required" in roles:
                present_effect = "support"
                absent_effect = defaults.get("required_adequate_absent", "oppose")
            else:
                present_effect = "support"
                absent_effect = defaults.get("supportive_adequate_absent", "neutral")
            if "required" in roles:
                role_weight = float(role_weights.get("required", default_weight))
            elif "contradictory" in roles:
                role_weight = float(role_weights.get("contradictory", default_weight))
            else:
                role_weight = float(role_weights.get("supportive", default_weight))
            output.append(
                FeatureRule(
                    feature_id=feature_id,
                    roles=tuple(roles),
                    diagnostic_weight=float(template.diagnostic_weights.get(feature_id, role_weight)),
                    present_effect=present_effect,
                    adequate_absent_effect=absent_effect,
                    uncertain_effect=defaults.get("uncertain", "uncertain"),
                    not_evaluable_effect=defaults.get("not_evaluable", "unresolved"),
                )
            )
        return tuple(output)

    @classmethod
    def from_dict(cls, payload):
        hypotheses = []
        for item in payload.get("hypotheses", []):
            hypotheses.append(
                HypothesisTemplate(
                    hypothesis_id=item["hypothesis_id"],
                    pathway=item["pathway"],
                    subtype=item["subtype"],
                    dysplasia_applicability=item["dysplasia_applicability"],
                    supporting_evidence=tuple(item.get("supporting_evidence", [])),
                    contradictory_evidence=tuple(item.get("contradictory_evidence", [])),
                    required_evidence=tuple(item.get("required_evidence", [])),
                    discriminative_features=tuple(item.get("discriminative_features", [])),
                    final_label_mapping=dict(item.get("final_label_mapping", {})),
                    diagnostic_weights=dict(item.get("diagnostic_weights", {})),
                    absence_semantics=dict(item.get("absence_semantics", {})),
                )
            )
        questions = []
        for item in payload.get("questions", []):
            questions.append(
                DiscriminativeQuestion(
                    question_id=item["question_id"],
                    target_feature=item["target_feature"],
                    question=item["question"],
                    discriminates=tuple(item.get("discriminates", [])),
                    expected_effect=dict(item.get("expected_effect", {})),
                    reviewer_id=item["reviewer_id"],
                    task_profile=item["task_profile"],
                    allowed_scales=tuple(float(value) for value in item.get("allowed_scales", [])),
                    priority=float(item.get("priority", 0.5)),
                )
            )
        return cls(
            version=payload.get("version", "agentflow_kb_v1"),
            hypotheses=hypotheses,
            questions=questions,
            guidelines=payload.get("guidelines", {}),
            label_names=payload.get("label_names", {}),
            evidence_scoring=payload.get("evidence_scoring", {}),
        )

    @classmethod
    def from_json(cls, path):
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _repository_default_path():
    return Path(__file__).resolve().parents[3] / "configs" / "agentflow" / "knowledge_base_v1.json"


def default_knowledge_base(path=None):
    candidate = Path(path) if path else _repository_default_path()
    if candidate.exists():
        return KnowledgeBase.from_json(candidate)
    return KnowledgeBase.from_dict(_DEFAULT_KNOWLEDGE_BASE)


_DEFAULT_KNOWLEDGE_BASE = {
    "version": "agentflow_kb_v1",
    "evidence_scoring": {
        "initial_evidence_score": 0.0,
        "ranking_transform": "softmax_uncalibrated",
        "default_diagnostic_weight": 0.5,
        "diagnostic_weights_by_role": {
            "required": 1.0,
            "contradictory": 0.75,
            "supportive": 0.5,
        },
        "absence_semantics": {
            "supportive_adequate_absent": "neutral",
            "required_adequate_absent": "oppose",
            "contradictory_adequate_absent": "neutral",
            "uncertain": "uncertain",
            "not_evaluable": "unresolved",
        },
    },
    "hypotheses": [
        {
            "hypothesis_id": "H_SSL",
            "pathway": "serrated",
            "subtype": "ssl",
            "dysplasia_applicability": "class_defining",
            "supporting_evidence": [
                "serrated_architecture",
                "basal_crypt_dilation",
                "serration_to_crypt_base",
                "crypt_branching",
            ],
            "contradictory_evidence": [
                "surface_limited_serration",
                "straight_crypt_bases",
                "ectopic_crypt_formation",
            ],
            "required_evidence": ["basal_crypt_dilation", "serration_to_crypt_base"],
            "discriminative_features": ["serration_to_crypt_base", "basal_crypt_dilation"],
            "final_label_mapping": {"not_supported": "SSL", "supported": "SSLD"},
        },
        {
            "hypothesis_id": "H_HP",
            "pathway": "serrated",
            "subtype": "hp",
            "dysplasia_applicability": "not_class_defining",
            "supporting_evidence": [
                "serrated_architecture",
                "surface_limited_serration",
                "straight_crypt_bases",
            ],
            "contradictory_evidence": ["basal_crypt_dilation", "serration_to_crypt_base"],
            "required_evidence": ["surface_limited_serration", "straight_crypt_bases"],
            "discriminative_features": ["serration_to_crypt_base", "straight_crypt_bases"],
            "final_label_mapping": {"not_supported": "HP", "supported": "HP"},
        },
        {
            "hypothesis_id": "H_TSA",
            "pathway": "serrated",
            "subtype": "tsa",
            "dysplasia_applicability": "class_defining",
            "supporting_evidence": [
                "serrated_architecture",
                "ectopic_crypt_formation",
                "slit_like_serration",
                "villiform_or_filiform_serrated_architecture",
                "cytoplasmic_eosinophilia",
                "pencillate_nuclei",
            ],
            "contradictory_evidence": ["straight_crypt_bases", "pure_tubular_architecture"],
            "required_evidence": ["ectopic_crypt_formation", "slit_like_serration"],
            "discriminative_features": ["ectopic_crypt_formation", "cytoplasmic_eosinophilia"],
            "final_label_mapping": {"not_supported": "TSA", "supported": "TSAD"},
        },
        {
            "hypothesis_id": "H_USA",
            "pathway": "serrated",
            "subtype": "unclassified_serrated_adenoma",
            "dysplasia_applicability": "not_class_defining",
            "supporting_evidence": [
                "serrated_architecture",
                "mixed_serrated_villous_architecture",
                "serrated_villous_colocalization",
            ],
            "contradictory_evidence": ["straight_crypt_bases", "pure_tubular_architecture"],
            "required_evidence": ["mixed_serrated_villous_architecture"],
            "discriminative_features": ["mixed_serrated_villous_architecture"],
            "final_label_mapping": {
                "not_supported": "Unclassified serrated adenoma",
                "supported": "Unclassified serrated adenoma",
            },
        },
        {
            "hypothesis_id": "H_TA",
            "pathway": "conventional_adenoma",
            "subtype": "tubular_adenoma",
            "dysplasia_applicability": "class_defining",
            "supporting_evidence": ["tubular_architecture", "crowded_adenomatous_glands"],
            "contradictory_evidence": ["serration_to_crypt_base", "villous_component_present"],
            "required_evidence": ["tubular_architecture", "crowded_adenomatous_glands"],
            "discriminative_features": ["substantial_villous_component"],
            "final_label_mapping": {"not_supported": "Tubular adenoma", "supported": "TAD"},
        },
        {
            "hypothesis_id": "H_TVA",
            "pathway": "conventional_adenoma",
            "subtype": "tubulovillous_adenoma",
            "dysplasia_applicability": "class_defining",
            "supporting_evidence": [
                "tubular_architecture",
                "villous_architecture",
                "villous_component_present",
                "tubular_villous_mixing",
            ],
            "contradictory_evidence": ["surface_limited_serration"],
            "required_evidence": ["villous_component_present", "tubular_villous_mixing"],
            "discriminative_features": ["villous_component_present", "tubular_villous_mixing"],
            "final_label_mapping": {"not_supported": "Tubulovillous adenoma", "supported": "TVAD"},
        },
        {
            "hypothesis_id": "H_INFLAMMATORY",
            "pathway": "inflammatory",
            "subtype": "inflammatory_reactive",
            "dysplasia_applicability": "not_class_defining",
            "supporting_evidence": [
                "reactive_inflammatory_context",
                "erosion",
                "mixed_inflammation",
                "reactive_regenerative_change",
                "adenomatous_architecture_absent",
                "serrated_architecture_absent",
            ],
            "contradictory_evidence": [
                "serration_to_crypt_base",
                "ectopic_crypt_formation",
                "crowded_adenomatous_glands",
            ],
            "required_evidence": ["reactive_regenerative_change", "adenomatous_architecture_absent"],
            "discriminative_features": [
                "reactive_regenerative_change",
                "adenomatous_architecture_absent",
                "serrated_architecture_absent"
            ],
            "final_label_mapping": {"not_supported": "Inflammatory", "supported": "Inflammatory"},
        },
    ],
    "questions": [
        {
            "question_id": "Q_SSL_HP_CRYPT_BASE",
            "target_feature": "serration_to_crypt_base",
            "question": "Does serration extend to the crypt base, with basal crypt distortion or dilation?",
            "discriminates": ["ssl", "hp"],
            "expected_effect": {"present": "supports_ssl", "absent_in_adequate_roi": "supports_hp"},
            "reviewer_id": "SerratedArchitectureReviewer",
            "task_profile": "ssl_hp_discrimination",
            "allowed_scales": [5.0, 10.0],
            "priority": 1.0,
        },
        {
            "question_id": "Q_TSA_SIGNATURE",
            "target_feature": "ectopic_crypt_formation",
            "question": "Is ectopic crypt formation or another TSA-specific architecture present?",
            "discriminates": ["tsa", "ssl", "hp", "unclassified_serrated_adenoma"],
            "expected_effect": {"present": "supports_tsa", "absent_in_adequate_roi": "opposes_tsa"},
            "reviewer_id": "TSAReviewer",
            "task_profile": "tsa_architecture",
            "allowed_scales": [5.0, 10.0],
            "priority": 0.9,
        },
        {
            "question_id": "Q_CONVENTIONAL_VILLOUS",
            "target_feature": "villous_component_present",
            "question": "Is a substantial villous component present in this representative architecture ROI?",
            "discriminates": ["tubular_adenoma", "tubulovillous_adenoma"],
            "expected_effect": {"present": "supports_tubulovillous", "absent_in_adequate_roi": "supports_tubular"},
            "reviewer_id": "ConventionalArchitectureReviewer",
            "task_profile": "tubular_villous_resolution",
            "allowed_scales": [5.0, 10.0],
            "priority": 0.9,
        },
        {
            "question_id": "Q_INFLAMMATORY_MIMIC",
            "target_feature": "reactive_regenerative_change",
            "question": "Are the epithelial changes reactive/regenerative in an inflammatory context without neoplastic architecture?",
            "discriminates": ["inflammatory_reactive", "ssl", "tubular_adenoma"],
            "expected_effect": {"present": "supports_inflammatory", "absent_in_adequate_roi": "opposes_inflammatory"},
            "reviewer_id": "InflammatoryReactiveReviewer",
            "task_profile": "reactive_mimic_resolution",
            "allowed_scales": [5.0, 10.0],
            "priority": 0.7,
        },
        {
            "question_id": "Q_DYSPLASIA",
            "target_feature": "high_grade_or_definite_dysplasia",
            "question": "Is high-grade or definite dysplasia present in this hotspot?",
            "discriminates": ["dysplasia_supported", "dysplasia_not_supported"],
            "expected_effect": {"present": "supports_dysplasia", "absent_in_adequate_roi": "does_not_support_dysplasia"},
            "reviewer_id": "DysplasiaReviewer",
            "task_profile": "high_grade_dysplasia_assessment",
            "allowed_scales": [20.0],
            "priority": 1.0,
        },
        {
            "question_id": "Q_QUALITY",
            "target_feature": "reviewable_mucosa",
            "question": "Is this ROI sufficiently reviewable for the requested morphology assessment?",
            "discriminates": ["evaluable", "not_evaluable"],
            "expected_effect": {"present": "permits_specialist_review", "absent_in_adequate_roi": "requires_new_roi"},
            "reviewer_id": "QualityMucosaReviewer",
            "task_profile": "overview_evaluability",
            "allowed_scales": [2.5, 5.0],
            "priority": 0.6,
        },
    ],
    "label_names": {
        "SSL": "Sessile serrated lesion",
        "SSLD": "Sessile serrated lesion with high-grade/definite dysplasia",
        "HP": "Hyperplastic polyp",
        "TSA": "Traditional serrated adenoma",
        "TSAD": "Traditional serrated adenoma with high-grade/definite dysplasia",
        "Unclassified serrated adenoma": "Unclassified serrated adenoma",
        "Tubular adenoma": "Tubular adenoma",
        "TAD": "Tubular adenoma with high-grade/definite dysplasia",
        "Tubulovillous adenoma": "Tubulovillous adenoma",
        "TVAD": "Tubulovillous adenoma with high-grade/definite dysplasia",
        "Inflammatory": "Inflammatory/reactive polyp-like lesion",
    },
    "guidelines": {},
}
