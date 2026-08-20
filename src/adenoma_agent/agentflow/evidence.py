import hashlib
import math
import re
from collections import defaultdict
from dataclasses import replace
from typing import Mapping, Optional, Sequence, Tuple

from adenoma_agent.agentflow.contracts import (
    EvidenceEffect,
    EvidenceRecord,
    EvidenceRelation,
    LedgerSnapshot,
)
from adenoma_agent.agentflow.knowledge import FeatureRule, KnowledgeBase, default_knowledge_base
from adenoma_agent.agentflow.state import (
    AgentState,
    BeliefUpdate,
    ContradictionState,
    DiscriminatorState,
    HypothesisState,
    TerminationState,
)


def softmax_uncalibrated(scores):
    """Return deterministic ranking weights; these are not clinical probabilities."""

    values = [float(value) for value in scores]
    if not values:
        return tuple()
    offset = max(values)
    exponentials = [math.exp(value - offset) for value in values]
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


def bbox_iou(first, second):
    first = tuple(first or ())
    second = tuple(second or ())
    if len(first) != 4 or len(second) != 4:
        return 0.0
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection) / float(union) if union > 0 else 0.0


def _safe_token(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _stable_id(prefix, values):
    body = "\x1f".join(str(value) for value in values)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    return "{0}_{1}".format(prefix, digest)


class EvidenceEngine(object):
    """Recompute hypothesis-relative belief state from active immutable evidence.

    Reviewer observations remain diagnosis-free.  This engine is the thin,
    deterministic adapter that applies versioned feature rules after evidence
    has entered the Ledger.
    """

    def __init__(self, knowledge_base=None, policy=None):
        self.knowledge_base = knowledge_base or default_knowledge_base()
        configured = dict(self.knowledge_base.evidence_scoring or {})
        configured.update(dict(policy or {}))
        self.policy = configured
        self.overlap_correlation_iou = float(configured.get("overlap_correlation_iou", 0.50))
        self.true_contradiction_iou = float(configured.get("true_contradiction_iou", 0.80))
        self.minimum_correlated_independence = float(
            configured.get("minimum_correlated_independence", 0.10)
        )
        self.relation_quality_threshold = float(configured.get("relation_quality_threshold", 0.50))
        for name, value in (
            ("overlap_correlation_iou", self.overlap_correlation_iou),
            ("true_contradiction_iou", self.true_contradiction_iou),
            ("minimum_correlated_independence", self.minimum_correlated_independence),
            ("relation_quality_threshold", self.relation_quality_threshold),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError("{0} must be in [0, 1]".format(name))
        self._rules = {
            hypothesis.hypothesis_id: {
                rule.feature_id: rule
                for rule in self.knowledge_base.feature_rules(hypothesis.hypothesis_id)
            }
            for hypothesis in self.knowledge_base.hypotheses
        }

    def initialize_state(self, case_id, case_context=None, budget=None):
        templates = self.knowledge_base.hypotheses
        ranking = 1.0 / float(len(templates)) if templates else 0.0
        hypotheses = tuple(
            HypothesisState(
                hypothesis_id=template.hypothesis_id,
                pathway=template.pathway,
                subtype=template.subtype,
                evidence_score=0.0,
                ranking_score=ranking,
                status="uncertain",
                required_features=tuple(template.required_evidence),
                missing_required_features=tuple(template.required_evidence),
            )
            for template in templates
        )
        discriminators = tuple(self._initial_discriminator(question) for question in self.knowledge_base.questions)
        return AgentState(
            state_id="STATE_{0}_000000".format(_safe_token(case_id)),
            case_id=str(case_id),
            state_version=0,
            round_id=0,
            ledger_snapshot_id=None,
            case_context=dict(case_context or {}),
            hypotheses=hypotheses,
            discriminators=discriminators,
            budget=dict(budget or {}),
            diagnostic_scope_status=str(
                dict(case_context or {}).get("diagnostic_scope_status", "incomplete")
            ),
            termination_state=TerminationState(),
        )

    def update(self, state, snapshot, round_id=None):
        if not isinstance(state, AgentState):
            state = AgentState.from_dict(state)
        if not isinstance(snapshot, LedgerSnapshot):
            snapshot = LedgerSnapshot(
                snapshot_id=snapshot["snapshot_id"],
                case_id=snapshot["case_id"],
                records=tuple(
                    value if isinstance(value, EvidenceRecord) else EvidenceRecord.from_dict(value)
                    for value in snapshot.get("records", ())
                ),
                created_after_action_id=snapshot.get("created_after_action_id"),
            )
        if state.case_id != snapshot.case_id:
            raise ValueError("AgentState and LedgerSnapshot case_id mismatch")
        if state.termination_state.stopped:
            raise ValueError("Cannot update a terminated AgentState")
        current_round = int(state.round_id + 1 if round_id is None else round_id)
        if current_round <= state.round_id:
            raise ValueError("round_id must increase across AgentState transitions")

        active_records, superseded_ids, relations, independence_by_id = self._active_records(snapshot.records)
        relations.extend(self._evidence_disagreement_relations(active_records))
        effects = self._effects(active_records, independence_by_id)
        effects, saturation_relations = self._apply_cluster_saturation(effects)
        relations.extend(saturation_relations)
        relations = self._unique_relations(relations)

        scores_after = {item.hypothesis_id: 0.0 for item in state.hypotheses}
        for effect in effects:
            if effect.active:
                scores_after[effect.hypothesis_id] += effect.contribution
        ordered_ids = [item.hypothesis_id for item in state.hypotheses]
        ranking_values = softmax_uncalibrated([scores_after[item] for item in ordered_ids])
        ranking_after = dict(zip(ordered_ids, ranking_values))
        update_id = "UPDATE_{0}_{1:03d}".format(_safe_token(snapshot.snapshot_id), current_round)
        hypotheses = self._hypothesis_states(state, active_records, effects, scores_after, ranking_after, update_id)
        diagnostic_scope_status = state.diagnostic_scope_status
        if diagnostic_scope_status != "out_of_scope_suspected":
            diagnostic_scope_status = (
                "adequate"
                if any(
                    not item.missing_required_features and item.evidence_score > 0.0
                    for item in hypotheses
                )
                else "incomplete"
            )
        discriminators = self._discriminator_states(state.discriminators, active_records, relations)
        contradictions = self._contradiction_states(active_records, relations, current_round)
        resolved_ids = tuple(item.discriminator_id for item in discriminators if item.status == "resolved")
        remaining_ids = tuple(item.discriminator_id for item in discriminators if item.status != "resolved")
        previous_evidence_ids = set(state.evidence_history)
        new_evidence_ids = tuple(
            record.evidence_id for record in snapshot.records if record.evidence_id not in previous_evidence_ids
        )
        update = BeliefUpdate(
            update_id=update_id,
            round_id=current_round,
            input_snapshot_id=snapshot.snapshot_id,
            evidence_score_before={item.hypothesis_id: item.evidence_score for item in state.hypotheses},
            evidence_score_after=scores_after,
            belief_before={item.hypothesis_id: item.ranking_score for item in state.hypotheses},
            belief_after=ranking_after,
            new_evidence_ids=new_evidence_ids,
            active_evidence_ids=tuple(record.evidence_id for record in active_records),
            effects=effects,
            relations=relations,
            resolved_discriminator_ids=resolved_ids,
            remaining_discriminator_ids=remaining_ids,
        )
        state_id = "STATE_{0}_{1:06d}_{2}".format(
            _safe_token(state.case_id),
            state.state_version + 1,
            _safe_token(snapshot.snapshot_id),
        )
        return AgentState(
            state_id=state_id,
            case_id=state.case_id,
            state_version=state.state_version + 1,
            round_id=current_round,
            ledger_snapshot_id=snapshot.snapshot_id,
            case_context=state.case_context,
            hypotheses=hypotheses,
            discriminators=discriminators,
            observation_history=tuple(
                dict.fromkeys(
                    str(dict(record.metadata or {}).get("observation_id"))
                    for record in snapshot.records
                    if dict(record.metadata or {}).get("observation_id")
                )
            ),
            evidence_history=tuple(record.evidence_id for record in snapshot.records),
            evidence_effects=effects,
            evidence_relations=relations,
            contradictions=contradictions,
            belief_history=tuple(state.belief_history) + (update,),
            reviewer_plan=state.reviewer_plan,
            review_history=state.review_history,
            tool_reviewer_failures=state.tool_reviewer_failures,
            budget=state.budget,
            diagnostic_scope_status=diagnostic_scope_status,
            termination_state=state.termination_state,
            active_evidence_ids=tuple(record.evidence_id for record in active_records),
            superseded_evidence_ids=tuple(sorted(superseded_ids)),
            previous_state_id=state.state_id,
        )

    def _initial_discriminator(self, question):
        target_ids = []
        for subtype in question.discriminates:
            try:
                target_ids.append(self.knowledge_base.hypothesis_for_subtype(subtype).hypothesis_id)
            except KeyError:
                continue
        if not target_ids and question.question_id == "Q_DYSPLASIA":
            target_ids = [
                item.hypothesis_id
                for item in self.knowledge_base.hypotheses
                if item.dysplasia_applicability == "class_defining"
            ]
        if not target_ids:
            target_ids = [item.hypothesis_id for item in self.knowledge_base.hypotheses]
        return DiscriminatorState(
            discriminator_id=question.question_id,
            target_hypothesis_ids=tuple(target_ids),
            feature_ids=(question.target_feature,),
            preferred_reviewer=question.reviewer_id,
            task_profile=question.task_profile,
            preferred_scales=tuple(question.allowed_scales),
            importance=question.priority,
        )

    def _active_records(self, records):
        records = tuple(records)
        superseded = set()
        relations = []
        for record in records:
            targets = list(record.supersedes_evidence_ids)
            metadata = dict(record.metadata or {})
            metadata_targets = metadata.get("supersedes_evidence_ids", ())
            if isinstance(metadata_targets, str):
                metadata_targets = (metadata_targets,)
            targets.extend(metadata_targets or ())
            single_target = metadata.get("supersedes_evidence_id")
            if single_target:
                targets.append(single_target)
            correction_target = record.correction_of_evidence_id or metadata.get("correction_of_evidence_id")
            if correction_target:
                targets.append(correction_target)
                relations.append(
                    EvidenceRelation(
                        relation_id=_stable_id("REL_CORRECTION", (correction_target, record.evidence_id)),
                        relation_type="correction",
                        evidence_ids=(str(correction_target), record.evidence_id),
                        feature_id=record.feature,
                    )
                )
            supersede_targets = tuple(sorted(set(str(value) for value in targets if value)))
            superseded.update(supersede_targets)
            explicit_supersede = tuple(value for value in supersede_targets if value != correction_target)
            if explicit_supersede:
                relations.append(
                    EvidenceRelation(
                        relation_id=_stable_id("REL_SUPERSEDE", explicit_supersede + (record.evidence_id,)),
                        relation_type="supersede",
                        evidence_ids=explicit_supersede + (record.evidence_id,),
                        feature_id=record.feature,
                    )
                )

        candidates = [record for record in records if record.evidence_id not in superseded]
        unique = []
        exact_seen = {}
        independence = {}
        for record in candidates:
            key = self._exact_key(record)
            duplicate_of = exact_seen.get(key)
            if duplicate_of is not None:
                relations.append(
                    EvidenceRelation(
                        relation_id=_stable_id("REL_DUPLICATE", (duplicate_of.evidence_id, record.evidence_id)),
                        relation_type="exact_duplicate",
                        evidence_ids=(duplicate_of.evidence_id, record.evidence_id),
                        feature_id=record.feature,
                    )
                )
                continue
            exact_seen[key] = record
            unique.append(record)
            independence[record.evidence_id] = self._record_independence(record)

        for index, first in enumerate(unique):
            for second in unique[index + 1 :]:
                if first.feature != second.feature or first.status != second.status:
                    continue
                overlap = bbox_iou(first.level0_bbox, second.level0_bbox)
                if overlap < self.overlap_correlation_iou:
                    continue
                relation = EvidenceRelation(
                    relation_id=_stable_id("REL_OVERLAP", (first.evidence_id, second.evidence_id)),
                    relation_type="correlated_overlap",
                    evidence_ids=(first.evidence_id, second.evidence_id),
                    feature_id=first.feature,
                    confidence=overlap,
                    metadata={"iou": overlap},
                )
                relations.append(relation)
                independence[second.evidence_id] = min(
                    independence[second.evidence_id],
                    max(self.minimum_correlated_independence, 1.0 - overlap),
                )
        return tuple(unique), superseded, relations, independence

    def _effects(self, records, independence_by_id):
        output = []
        for record in records:
            cluster_id = self._cluster_id(record)
            for hypothesis in self.knowledge_base.hypotheses:
                rule = self._rules[hypothesis.hypothesis_id].get(record.feature)
                if rule is None:
                    direction = "neutral"
                    diagnostic_weight = 0.0
                    roles = tuple()
                    kb_rule_id = "{0}:no_relation:{1}:{2}".format(
                        self.knowledge_base.version,
                        hypothesis.hypothesis_id,
                        record.feature,
                    )
                else:
                    direction = self._direction(record, rule)
                    diagnostic_weight = rule.diagnostic_weight
                    roles = rule.roles
                    kb_rule_id = "{0}:feature_rule:{1}:{2}".format(
                        self.knowledge_base.version,
                        hypothesis.hypothesis_id,
                        record.feature,
                    )
                sign = {"support": 1, "oppose": -1, "neutral": 0, "uncertain": 0}[direction]
                output.append(
                    EvidenceEffect(
                        effect_id="EFFECT_{0}_{1}".format(
                            _safe_token(hypothesis.hypothesis_id),
                            _safe_token(record.evidence_id),
                        ),
                        evidence_id=record.evidence_id,
                        hypothesis_id=hypothesis.hypothesis_id,
                        feature_id=record.feature,
                        direction=direction,
                        sign=sign,
                        diagnostic_weight=diagnostic_weight,
                        confidence=record.confidence,
                        quality=(0.0 if record.feature_evaluability == "not_evaluable" else record.quality),
                        independence=independence_by_id.get(record.evidence_id, 1.0),
                        kb_rule_id=kb_rule_id,
                        cluster_id=cluster_id,
                        metadata={
                            "roles": list(roles),
                            "source": record.source,
                            "status": record.status,
                            "feature_evaluability": record.feature_evaluability,
                            "ranking_is_uncalibrated": True,
                        },
                    )
                )
        return tuple(output)

    @staticmethod
    def _direction(record, rule):
        if record.status == "present":
            return rule.present_effect if rule.present_effect in ("support", "oppose", "neutral") else "uncertain"
        if record.status == "absent" and record.feature_evaluability == "adequate":
            return (
                rule.adequate_absent_effect
                if rule.adequate_absent_effect in ("support", "oppose", "neutral")
                else "uncertain"
            )
        if record.status == "uncertain":
            return "uncertain"
        if record.status in ("not_evaluable", "invocation_failure"):
            return "neutral"
        return "neutral"

    def _apply_cluster_saturation(self, effects):
        output = list(effects)
        positions = defaultdict(list)
        for index, effect in enumerate(output):
            if effect.direction not in ("support", "oppose"):
                continue
            cluster = effect.cluster_id or "evidence:{0}".format(effect.evidence_id)
            key = (effect.hypothesis_id, effect.feature_id, cluster, effect.direction)
            positions[key].append(index)
        relations = []
        for key, indices in sorted(positions.items()):
            if len(indices) <= 1:
                continue
            winner_index = sorted(
                indices,
                key=lambda index: (-abs(output[index].contribution), output[index].effect_id),
            )[0]
            winner = output[winner_index]
            suppressed = []
            for index in indices:
                if index == winner_index:
                    continue
                suppressed.append(output[index].evidence_id)
                output[index] = replace(
                    output[index],
                    active=False,
                    suppressed_by_effect_id=winner.effect_id,
                    contribution=None,
                )
            evidence_ids = tuple(dict.fromkeys((winner.evidence_id,) + tuple(suppressed)))
            if len(evidence_ids) > 1:
                relations.append(
                    EvidenceRelation(
                        relation_id=_stable_id("REL_SATURATION", key + evidence_ids),
                        relation_type="cluster_saturation",
                        evidence_ids=evidence_ids,
                        feature_id=winner.feature_id,
                        hypothesis_ids=(winner.hypothesis_id,),
                        metadata={
                            "cluster_id": winner.cluster_id,
                            "direction": winner.direction,
                            "policy": "max_contribution_per_hypothesis_feature_cluster_direction",
                            "retained_effect_id": winner.effect_id,
                        },
                    )
                )
        return tuple(output), tuple(relations)

    def _evidence_disagreement_relations(self, records):
        by_feature = defaultdict(list)
        for record in records:
            if (
                record.status in ("present", "absent")
                and record.feature_evaluability == "adequate"
                and record.quality >= self.relation_quality_threshold
            ):
                by_feature[record.feature].append(record)
        output = []
        for feature, values in sorted(by_feature.items()):
            present = [record for record in values if record.status == "present"]
            absent = [record for record in values if record.status == "absent"]
            for same_status_records in (present, absent):
                for index, first in enumerate(same_status_records):
                    for second in same_status_records[index + 1 :]:
                        overlap = bbox_iou(first.level0_bbox, second.level0_bbox)
                        scale_compatible = bool(
                            first.scale is None
                            or second.scale is None
                            or abs(float(first.scale) - float(second.scale)) <= 1e-6
                        )
                        output.append(
                            EvidenceRelation(
                                relation_id=_stable_id(
                                    "REL_CORROBORATION",
                                    (first.evidence_id, second.evidence_id),
                                ),
                                relation_type="corroboration",
                                evidence_ids=(first.evidence_id, second.evidence_id),
                                feature_id=feature,
                                confidence=min(
                                    first.confidence * first.quality,
                                    second.confidence * second.quality,
                                ),
                                metadata={
                                    "status": first.status,
                                    "iou": overlap,
                                    "same_roi": bool(
                                        first.roi_id and first.roi_id == second.roi_id
                                    ),
                                    "scale_compatible": scale_compatible,
                                },
                            )
                        )
            for first in present:
                for second in absent:
                    overlap = bbox_iou(first.level0_bbox, second.level0_bbox)
                    same_cluster = bool(
                        self._cluster_id(first)
                        and self._cluster_id(first) == self._cluster_id(second)
                    )
                    same_roi = bool(first.roi_id and first.roi_id == second.roi_id)
                    scale_compatible = self._scales_compatible(first.scale, second.scale)
                    same_anatomical_view = scale_compatible and (
                        same_roi or overlap >= self.true_contradiction_iou
                    )
                    quality_feature = (
                        feature in ("reviewable_mucosa", "epithelium_present")
                        or first.source == "QualityMucosaReviewer"
                        or second.source == "QualityMucosaReviewer"
                    )
                    if quality_feature and same_anatomical_view:
                        relation_type = "quality_disagreement"
                    elif same_anatomical_view:
                        relation_type = "true_contradiction"
                    else:
                        relation_type = "spatial_heterogeneity"
                    output.append(
                        EvidenceRelation(
                            relation_id=_stable_id(
                                "REL_{0}".format(relation_type.upper()),
                                (first.evidence_id, second.evidence_id),
                            ),
                            relation_type=relation_type,
                            evidence_ids=(first.evidence_id, second.evidence_id),
                            feature_id=feature,
                            confidence=min(
                                first.confidence * first.quality,
                                second.confidence * second.quality,
                            ),
                            metadata={
                                "iou": overlap,
                                "same_roi": same_roi,
                                "same_cluster": same_cluster,
                                "scale_compatible": scale_compatible,
                                "first_cluster_id": self._cluster_id(first),
                                "second_cluster_id": self._cluster_id(second),
                            },
                        )
                    )
        return tuple(output)

    @staticmethod
    def _scales_compatible(first_scale, second_scale):
        if first_scale is None or second_scale is None:
            return True
        first_value = float(first_scale)
        second_value = float(second_scale)
        if first_value <= 0.0 or second_value <= 0.0:
            return False
        return max(first_value, second_value) / min(first_value, second_value) <= 2.0 + 1e-9

    def _hypothesis_states(self, state, records, effects, scores, rankings, update_id):
        determinate = defaultdict(set)
        for record in records:
            if record.status in ("present", "absent") and record.feature_evaluability == "adequate":
                determinate[record.feature].add(record.evidence_id)
        output = []
        for previous in state.hypotheses:
            supporting = tuple(
                sorted(
                    effect.evidence_id
                    for effect in effects
                    if effect.hypothesis_id == previous.hypothesis_id
                    and effect.active
                    and effect.direction == "support"
                )
            )
            contradicting = tuple(
                sorted(
                    effect.evidence_id
                    for effect in effects
                    if effect.hypothesis_id == previous.hypothesis_id
                    and effect.active
                    and effect.direction == "oppose"
                )
            )
            observed_required = tuple(
                feature for feature in previous.required_features if determinate.get(feature)
            )
            missing_required = tuple(
                feature for feature in previous.required_features if feature not in observed_required
            )
            delta = scores[previous.hypothesis_id] - previous.evidence_score
            if scores[previous.hypothesis_id] <= float(self.policy.get("rejection_score", -2.0)):
                status = "rejected"
            elif delta > 1e-12:
                status = "strengthened"
            elif delta < -1e-12:
                status = "weakened"
            elif scores[previous.hypothesis_id] == 0.0:
                status = "uncertain"
            else:
                status = "active"
            output.append(
                HypothesisState(
                    hypothesis_id=previous.hypothesis_id,
                    pathway=previous.pathway,
                    subtype=previous.subtype,
                    evidence_score=scores[previous.hypothesis_id],
                    ranking_score=rankings[previous.hypothesis_id],
                    status=status,
                    supporting_evidence_ids=supporting,
                    contradicting_evidence_ids=contradicting,
                    required_features=previous.required_features,
                    observed_required_features=observed_required,
                    missing_required_features=missing_required,
                    update_ids=tuple(previous.update_ids) + (update_id,),
                )
            )
        output.sort(key=lambda item: (-item.ranking_score, item.hypothesis_id))
        return tuple(output)

    def _discriminator_states(self, previous, records, relations):
        feature_evidence = defaultdict(list)
        for record in records:
            if record.status in ("present", "absent") and record.feature_evaluability == "adequate":
                feature_evidence[record.feature].append(record.evidence_id)
        feature_relations = defaultdict(list)
        for relation in relations:
            if relation.feature_id:
                feature_relations[relation.feature_id].append(relation)
        output = []
        for item in previous:
            relevant_relations = [
                relation
                for feature in item.feature_ids
                for relation in feature_relations.get(feature, ())
                if relation.relation_type
                in ("true_contradiction", "quality_disagreement", "spatial_heterogeneity")
            ]
            if any(
                relation.relation_type in ("true_contradiction", "quality_disagreement")
                for relation in relevant_relations
            ):
                status = "conflicting"
            elif any(relation.relation_type == "spatial_heterogeneity" for relation in relevant_relations):
                status = "spatially_heterogeneous"
            elif all(feature_evidence.get(feature) for feature in item.feature_ids):
                status = "resolved"
            else:
                status = "unresolved"
            evidence_ids = tuple(
                sorted(
                    set(
                        evidence_id
                        for feature in item.feature_ids
                        for evidence_id in feature_evidence.get(feature, ())
                    )
                )
            )
            output.append(
                replace(
                    item,
                    status=status,
                    resolution_evidence_ids=evidence_ids,
                    relation_ids=tuple(sorted(set(relation.relation_id for relation in relevant_relations))),
                )
            )
        return tuple(output)

    def _contradiction_states(self, records, relations, round_id):
        by_id = {record.evidence_id: record for record in records}
        output = []
        for relation in relations:
            if relation.relation_type not in ("true_contradiction", "quality_disagreement"):
                continue
            related = [by_id[value] for value in relation.evidence_ids if value in by_id]
            affected = tuple(
                sorted(
                    hypothesis_id
                    for hypothesis_id, rules in self._rules.items()
                    if relation.feature_id in rules
                )
            )
            output.append(
                ContradictionState(
                    contradiction_id=relation.relation_id,
                    feature_id=relation.feature_id,
                    target_roi_ids=tuple(
                        sorted(set(record.roi_id for record in related if record.roi_id))
                    ),
                    evidence_ids=relation.evidence_ids,
                    affected_hypothesis_ids=affected,
                    severity=relation.confidence,
                    resolution_status="active",
                    relation_type=relation.relation_type,
                    round_created=round_id,
                )
            )
        return tuple(output)

    @staticmethod
    def _cluster_id(record):
        metadata = dict(record.metadata or {})
        return (
            record.cluster_id
            or metadata.get("evidence_cluster_id")
            or metadata.get("source_cluster_id")
            or record.roi_id
        )

    @staticmethod
    def _record_independence(record):
        value = dict(record.metadata or {}).get("independence", 1.0)
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError("Evidence metadata independence must be in [0, 1]")
        return value

    @staticmethod
    def _exact_key(record):
        metadata = dict(record.metadata or {})
        return (
            record.feature,
            record.status,
            record.feature_evaluability,
            record.source,
            record.source_version,
            record.model_version,
            record.prompt_version,
            record.roi_id,
            record.scale,
            tuple(record.level0_bbox),
            record.value,
            metadata.get("image_sha256"),
        )

    @staticmethod
    def _unique_relations(relations):
        output = []
        seen = set()
        for relation in relations:
            if relation.relation_id in seen:
                continue
            seen.add(relation.relation_id)
            output.append(relation)
        output.sort(key=lambda item: item.relation_id)
        return tuple(output)
