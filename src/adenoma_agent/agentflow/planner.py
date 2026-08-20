from collections import Counter, defaultdict
from dataclasses import replace
from typing import Mapping, Optional, Sequence, Tuple

from adenoma_agent.agentflow.contracts import (
    DiscriminativeQuestion,
    EvidenceView,
    HypothesisAssessment,
    PlanDecision,
    ReducedFeature,
    ReviewerAction,
)
from adenoma_agent.agentflow.knowledge import KnowledgeBase, default_knowledge_base
from adenoma_agent.agentflow.evidence import bbox_iou, softmax_uncalibrated
from adenoma_agent.agentflow.reviewer import ReviewerContractError, ReviewerRegistry, default_reviewer_registry
from adenoma_agent.agentflow.state import AgentState


QUESTION_FEATURE_BUNDLES = {
    "Q_SSL_HP_CRYPT_BASE": (
        "serration_to_crypt_base",
        "basal_crypt_dilation",
        "surface_limited_serration",
        "straight_crypt_bases",
    ),
    "Q_TSA_SIGNATURE": ("ectopic_crypt_formation", "slit_like_serration"),
    "Q_CONVENTIONAL_VILLOUS": (
        "villous_component_present",
        "tubular_villous_mixing",
        "crowded_adenomatous_glands",
    ),
    "Q_INFLAMMATORY_MIMIC": (
        "reactive_regenerative_change",
        "adenomatous_architecture_absent",
        "serrated_architecture_absent",
    ),
    "Q_DYSPLASIA": ("high_grade_or_definite_dysplasia",),
    "Q_QUALITY": ("reviewable_mucosa",),
}


def _clamp(value):
    return max(0.0, min(1.0, float(value)))


class EvidenceReducer(object):
    """Reduces provenance-rich records without converting missing data to negatives."""

    def reduce(self, snapshot, active_evidence_ids=None):
        active_ids = None if active_evidence_ids is None else set(active_evidence_ids)
        grouped = defaultdict(list)
        limitations = []
        for record in snapshot.records:
            if active_ids is not None and record.evidence_id not in active_ids:
                continue
            grouped[record.feature].append(record)
            limitations.extend(record.limitations)
        features = {}
        conflicts = []
        for feature, records in grouped.items():
            unique = []
            redundant_ids = []
            seen = set()
            for record in records:
                key = (
                    record.source,
                    record.roi_id,
                    record.scale,
                    record.status,
                    record.value,
                )
                if key in seen:
                    redundant_ids.append(record.evidence_id)
                    continue
                seen.add(key)
                unique.append(record)
            present = [record for record in unique if record.status == "present"]
            absent = [
                record
                for record in unique
                if record.status == "absent" and record.feature_evaluability == "adequate"
            ]
            uncertain = [record for record in unique if record.status in ("uncertain", "invocation_failure")]
            not_evaluable = [record for record in unique if record.status == "not_evaluable"]
            measurements = [record for record in unique if record.status == "measurement"]
            present_locations = {record.roi_id or record.cluster_id or "case" for record in present}
            absent_locations = {record.roi_id or record.cluster_id or "case" for record in absent}
            same_scope_disagreement = bool(present_locations.intersection(absent_locations))
            if same_scope_disagreement:
                resolved_status = "conflicting"
                conflicts.append(feature)
            elif present:
                resolved_status = "present"
            elif absent:
                resolved_status = "absent"
            elif measurements:
                resolved_status = "measurement"
            elif not_evaluable and not uncertain:
                resolved_status = "not_evaluable"
            else:
                resolved_status = "uncertain"
            determinate = present + absent
            if determinate:
                strength = sum(record.confidence * record.quality for record in determinate) / len(determinate)
                quality = sum(record.quality for record in determinate) / len(determinate)
                status_counts = Counter(record.status for record in determinate)
                source_agreement = max(status_counts.values()) / float(len(determinate))
            elif measurements:
                strength = sum(record.confidence for record in measurements) / len(measurements)
                quality = sum(record.quality for record in measurements) / len(measurements)
                source_agreement = 1.0
            else:
                strength = 0.0
                quality = 0.0
                source_agreement = 0.0
            spatial_keys = {
                record.cluster_id or record.roi_id or record.patch_id or record.evidence_id
                for record in determinate + measurements
            }
            coverage = min(1.0, len(spatial_keys) / 3.0) if spatial_keys else 0.0
            measurement = None
            if measurements:
                denominator = sum(max(1e-6, record.confidence * record.quality) for record in measurements)
                measurement = sum(
                    float(record.value) * max(1e-6, record.confidence * record.quality)
                    for record in measurements
                ) / denominator
            features[feature] = ReducedFeature(
                feature=feature,
                resolved_status=resolved_status,
                strength=_clamp(strength),
                quality=_clamp(quality),
                spatial_coverage=_clamp(coverage),
                source_agreement=_clamp(source_agreement),
                present_evidence_ids=tuple(record.evidence_id for record in present),
                absent_evidence_ids=tuple(record.evidence_id for record in absent),
                uncertain_evidence_ids=tuple(record.evidence_id for record in uncertain),
                not_evaluable_evidence_ids=tuple(record.evidence_id for record in not_evaluable),
                measurement=measurement,
                redundant_evidence_ids=tuple(redundant_ids),
            )
        return EvidenceView(
            snapshot_id=snapshot.snapshot_id,
            features=features,
            conflicts=tuple(sorted(conflicts)),
            limitations=tuple(sorted(set(limitations))),
        )


def dysplasia_state_from_view(view):
    feature = view.features.get("high_grade_or_definite_dysplasia")
    if feature is None:
        return "unassessed"
    if feature.resolved_status == "present":
        return "supported"
    if feature.resolved_status == "absent":
        return "not_supported"
    if feature.resolved_status == "conflicting":
        return "conflicting"
    if feature.resolved_status == "not_evaluable":
        return "not_evaluable"
    return "unassessed"


class HypothesisEngine(object):
    def __init__(self, knowledge_base, min_plausibility=0.55, max_conflict=0.45):
        self.knowledge_base = knowledge_base
        self.min_plausibility = float(min_plausibility)
        self.max_conflict = float(max_conflict)

    def rank(self, view):
        dysplasia_state = dysplasia_state_from_view(view)
        preliminary = []
        for order, template in enumerate(self.knowledge_base.hypotheses):
            support_ids = []
            conflict_ids = []
            support_strength = 0.0
            conflict_strength = 0.0
            for rule in self.knowledge_base.feature_rules(template.hypothesis_id):
                feature = view.features.get(rule.feature_id)
                if feature is None:
                    continue
                if feature.resolved_status == "present":
                    if rule.present_effect == "support":
                        support_ids.extend(feature.present_evidence_ids)
                        support_strength += rule.diagnostic_weight * feature.strength
                    elif rule.present_effect == "oppose":
                        conflict_ids.extend(feature.present_evidence_ids)
                        conflict_strength += rule.diagnostic_weight * feature.strength
                elif feature.resolved_status == "absent":
                    if rule.adequate_absent_effect == "support":
                        support_ids.extend(feature.absent_evidence_ids)
                        support_strength += rule.diagnostic_weight * feature.strength
                    elif rule.adequate_absent_effect == "oppose":
                        conflict_ids.extend(feature.absent_evidence_ids)
                        conflict_strength += rule.diagnostic_weight * feature.strength
                elif feature.resolved_status == "conflicting":
                    conflict_ids.extend(feature.present_evidence_ids + feature.absent_evidence_ids)
                    conflict_strength += rule.diagnostic_weight * feature.strength
            missing_required = []
            observed_required = 0
            for feature_name in template.required_evidence:
                feature = view.features.get(feature_name)
                if feature is not None and feature.resolved_status in ("present", "absent", "conflicting"):
                    observed_required += 1
                if feature is None or feature.resolved_status != "present":
                    missing_required.append(feature_name)
            coverage = observed_required / float(max(1, len(template.required_evidence)))
            conflict_denominator = sum(
                rule.diagnostic_weight
                for rule in self.knowledge_base.feature_rules(template.hypothesis_id)
                if "contradictory" in rule.roles or "required" in rule.roles
            )
            conflict_score = _clamp(conflict_strength / max(1.0, conflict_denominator))
            raw_score = support_strength - conflict_strength
            decision_ready = (
                not missing_required
                and raw_score > 0.0
                and conflict_score <= self.max_conflict
            )
            preliminary.append(
                (
                    order,
                    raw_score,
                    HypothesisAssessment(
                        hypothesis_id=template.hypothesis_id,
                        pathway=template.pathway,
                        subtype=template.subtype,
                        dysplasia_state=dysplasia_state,
                        dysplasia_applicability=template.dysplasia_applicability,
                        observed_supporting_evidence=tuple(sorted(set(support_ids))),
                        observed_conflict_evidence=tuple(sorted(set(conflict_ids))),
                        missing_required_evidence=tuple(missing_required),
                        plausibility_score=0.0,
                        required_evidence_coverage=_clamp(coverage),
                        conflict_score=conflict_score,
                        decision_ready=decision_ready,
                    ),
                )
            )
        ranking_values = softmax_uncalibrated([item[1] for item in preliminary])
        ranked = [
            (item[0], replace(item[2], plausibility_score=ranking_values[index]))
            for index, item in enumerate(preliminary)
        ]
        ranked.sort(key=lambda item: (-item[1].plausibility_score, item[0]))
        return tuple(item[1] for item in ranked)


class DiscriminativeQuestionGenerator(object):
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def generate(self, ranked_hypotheses, view):
        if not ranked_hypotheses:
            return tuple()
        top = ranked_hypotheses[0]
        if self._dysplasia_required(ranked_hypotheses) and top.decision_ready:
            question = self._question("Q_DYSPLASIA")
            if self._question_unresolved(question, view):
                return (question,)
        top_subtypes = {item.subtype for item in ranked_hypotheses[:2]}
        candidates = []
        for question in self.knowledge_base.questions:
            if question.question_id in ("Q_DYSPLASIA", "Q_QUALITY"):
                continue
            unresolved = self._question_unresolved(question, view)
            if not unresolved:
                continue
            overlap = len(top_subtypes.intersection(set(question.discriminates)))
            missing_overlap = len(
                set(QUESTION_FEATURE_BUNDLES.get(question.question_id, (question.target_feature,))).intersection(
                    set(top.missing_required_evidence)
                )
            )
            if overlap or missing_overlap:
                score = question.priority + 0.25 * overlap + 0.20 * missing_overlap
                candidates.append((score, question.question_id, question))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in candidates)

    @staticmethod
    def _dysplasia_required(ranked_hypotheses):
        top_score = ranked_hypotheses[0].plausibility_score
        competitive_d_capable = any(
            item.dysplasia_applicability == "class_defining"
            and item.plausibility_score >= max(0.45, top_score - 0.15)
            for item in ranked_hypotheses
        )
        state = ranked_hypotheses[0].dysplasia_state
        return competitive_d_capable and state not in ("supported", "not_supported")

    @staticmethod
    def _question_unresolved(question, view):
        features = QUESTION_FEATURE_BUNDLES.get(question.question_id, (question.target_feature,))
        for feature_name in features:
            feature = view.features.get(feature_name)
            if feature is None or feature.resolved_status not in ("present", "absent"):
                return True
        return False

    def _question(self, question_id):
        for question in self.knowledge_base.questions:
            if question.question_id == question_id:
                return question
        raise KeyError(question_id)


class ActionBinder(object):
    def __init__(self, registry, weights=None):
        self.registry = registry
        self.weights = dict(
            weights
            or {
                "discriminative_value": 0.35,
                "roi_suitability": 0.25,
                "spatial_coverage": 0.20,
                "normalized_cost": 0.10,
                "evidence_redundancy": 0.10,
            }
        )

    def bind(self, question, roi_candidates, snapshot, plan_id, allow_recheck=False):
        target_features = QUESTION_FEATURE_BUNDLES.get(question.question_id, (question.target_feature,))
        actions = []
        for roi in roi_candidates:
            if float(roi.scale) not in question.allowed_scales:
                continue
            if roi.allowed_reviewers and question.reviewer_id not in roi.allowed_reviewers:
                continue
            if not set(target_features).issubset(set(roi.candidate_features)):
                continue
            redundancy = self._redundancy(
                snapshot,
                question.reviewer_id,
                roi,
                target_features,
            )
            if allow_recheck:
                redundancy = min(redundancy, 0.50)
            if redundancy >= 1.0:
                continue
            action_id = "ACTION_{0}_{1}_{2}".format(
                plan_id.replace("PLAN_", ""),
                question.question_id,
                roi.roi_id,
            )
            components = {
                "discriminative_value": _clamp(question.priority),
                "roi_suitability": roi.suitability,
                "spatial_coverage": roi.spatial_coverage,
                "normalized_cost": roi.estimated_cost,
                "evidence_redundancy": redundancy,
            }
            score = (
                self.weights["discriminative_value"] * components["discriminative_value"]
                + self.weights["roi_suitability"] * components["roi_suitability"]
                + self.weights["spatial_coverage"] * components["spatial_coverage"]
                - self.weights["normalized_cost"] * components["normalized_cost"]
                - self.weights["evidence_redundancy"] * components["evidence_redundancy"]
            )
            action = ReviewerAction(
                action_id=action_id,
                question_id=question.question_id,
                reviewer_id=question.reviewer_id,
                task_profile=question.task_profile,
                roi_id=roi.roi_id,
                scale=roi.scale,
                target_features=tuple(target_features),
                goal=question.question,
                score=_clamp(score),
                score_components=components,
            )
            try:
                self.registry.validate_action(action, roi)
            except ReviewerContractError:
                continue
            actions.append(action)
        actions.sort(key=lambda item: (-item.score, item.roi_id, item.action_id))
        return tuple(actions)

    @staticmethod
    def _redundancy(snapshot, reviewer_id, roi, target_features):
        feature_redundancy = {feature: 0.0 for feature in target_features}
        for record in snapshot.records:
            if record.source == reviewer_id and record.feature in target_features:
                redundancy = 0.0
                if record.roi_id == roi.roi_id:
                    redundancy = 1.0
                elif record.cluster_id and record.cluster_id == roi.source_cluster_id:
                    redundancy = 0.75
                else:
                    redundancy = bbox_iou(record.level0_bbox, roi.level0_bbox)
                feature_redundancy[record.feature] = max(
                    feature_redundancy[record.feature], redundancy
                )
            if (
                record.evidence_type == "reviewer_invocation_failure"
                and record.roi_id == roi.roi_id
            ):
                for feature in target_features:
                    feature_redundancy[feature] = 1.0
        return sum(feature_redundancy.values()) / float(max(1, len(target_features)))


class StopPolicy(object):
    def __init__(self, margin_threshold=0.10):
        self.margin_threshold = float(margin_threshold)

    def diagnostic_ready(self, ranked_hypotheses, view):
        if not ranked_hypotheses:
            return False
        top = ranked_hypotheses[0]
        second_score = ranked_hypotheses[1].plausibility_score if len(ranked_hypotheses) > 1 else 0.0
        if not top.decision_ready or top.plausibility_score - second_score < self.margin_threshold:
            return False
        if view.conflicts:
            return False
        dysplasia_required = any(
            item.dysplasia_applicability == "class_defining"
            and item.plausibility_score >= max(0.45, top.plausibility_score - 0.15)
            for item in ranked_hypotheses
        )
        if dysplasia_required and top.dysplasia_state in ("unassessed", "not_evaluable", "conflicting"):
            return False
        if top.dysplasia_applicability != "class_defining" and top.dysplasia_state == "supported":
            return False
        return True

    def stop_reason(self, ranked_hypotheses, view, actions, round_index, max_actions, roi_candidates):
        if self.diagnostic_ready(ranked_hypotheses, view):
            return "diagnostic_ready"
        if round_index >= max_actions:
            return "budget_exhausted"
        if not actions:
            if not roi_candidates or all(candidate.suitability < 0.25 for candidate in roi_candidates):
                return "non_diagnostic_or_quality_limited"
            return "no_useful_action"
        return None


class BehavioralStopPolicy(object):
    """Stop policy for uncalibrated evidence scores and structured state."""

    def __init__(self, evidence_margin_threshold=0.50, minimum_evidence_quality=0.50):
        self.evidence_margin_threshold = float(evidence_margin_threshold)
        self.minimum_evidence_quality = float(minimum_evidence_quality)

    def diagnostic_ready(self, state, ranked_hypotheses, view):
        if not ranked_hypotheses or state.diagnostic_scope_status != "adequate":
            return False
        top = ranked_hypotheses[0]
        top_state = state.hypothesis(top.hypothesis_id)
        second_score = (
            state.hypothesis(ranked_hypotheses[1].hypothesis_id).evidence_score
            if len(ranked_hypotheses) > 1
            else 0.0
        )
        if top_state.missing_required_features or top_state.evidence_score <= 0.0:
            return False
        if top_state.evidence_score - second_score < self.evidence_margin_threshold:
            return False
        if any(
            relation.relation_type in ("true_contradiction", "quality_disagreement")
            for relation in state.evidence_relations
        ):
            return False
        supporting_effects = [
            effect
            for effect in state.evidence_effects
            if effect.active
            and effect.hypothesis_id == top_state.hypothesis_id
            and effect.direction == "support"
        ]
        if not supporting_effects or max(effect.quality for effect in supporting_effects) < self.minimum_evidence_quality:
            return False
        required_quality = {
            feature_id: max(
                (
                    effect.quality
                    for effect in supporting_effects
                    if effect.feature_id == feature_id and effect.direction == "support"
                ),
                default=0.0,
            )
            for feature_id in top_state.required_features
        }
        if any(
            quality < self.minimum_evidence_quality
            for quality in required_quality.values()
        ):
            return False
        if top.dysplasia_state not in ("supported", "not_supported"):
            return False
        if top.dysplasia_applicability != "class_defining" and top.dysplasia_state == "supported":
            return False
        return True

    def decide(self, state, ranked, view, actions, round_index, max_actions, roi_candidates):
        if state.diagnostic_scope_status == "out_of_scope_suspected":
            return "unresolved_stop", "non_diagnostic_or_quality_limited"
        if self.diagnostic_ready(state, ranked, view):
            return "sufficient_evidence_stop", "diagnostic_ready"
        if int(round_index) >= int(max_actions):
            return "budget_stop", "budget_exhausted"
        if not actions:
            if state.tool_reviewer_failures:
                return "failure_stop", "no_useful_action"
            if not roi_candidates or all(candidate.suitability < 0.25 for candidate in roi_candidates):
                return "unresolved_stop", "non_diagnostic_or_quality_limited"
            return "unresolved_stop", "no_useful_action"
        return "continue", None


class PlanningAgent(object):
    """Pure, single-step planning kernel required by Agent_workflow."""

    def __init__(
        self,
        knowledge_base=None,
        registry=None,
        action_weights=None,
        margin_threshold=0.10,
        evidence_margin_threshold=0.50,
        minimum_evidence_quality=0.50,
    ):
        self.knowledge_base = knowledge_base or default_knowledge_base()
        self.registry = registry or default_reviewer_registry()
        self.reducer = EvidenceReducer()
        self.hypothesis_engine = HypothesisEngine(self.knowledge_base)
        self.question_generator = DiscriminativeQuestionGenerator(self.knowledge_base)
        self.action_binder = ActionBinder(self.registry, action_weights)
        self.stop_policy = StopPolicy(margin_threshold=margin_threshold)
        self.behavioral_stop_policy = BehavioralStopPolicy(
            evidence_margin_threshold=evidence_margin_threshold,
            minimum_evidence_quality=minimum_evidence_quality,
        )

    def plan(
        self,
        snapshot,
        roi_candidates,
        round_index=0,
        max_actions=8,
        agent_state=None,
        reviewer_availability=None,
    ):
        if agent_state is not None:
            return self._plan_from_state(
                snapshot,
                roi_candidates,
                agent_state,
                round_index=round_index,
                max_actions=max_actions,
                reviewer_availability=reviewer_availability,
            )
        return self._plan_legacy(snapshot, roi_candidates, round_index, max_actions)

    def _plan_legacy(self, snapshot, roi_candidates, round_index=0, max_actions=8):
        view = self.reducer.reduce(snapshot)
        ranked = self.hypothesis_engine.rank(view)
        plan_id = "PLAN_{0}_{1:03d}".format(snapshot.snapshot_id, int(round_index))
        if self.stop_policy.diagnostic_ready(ranked, view):
            return PlanDecision(
                plan_id=plan_id,
                snapshot_id=snapshot.snapshot_id,
                decision="stop",
                ranked_hypotheses=ranked,
                selected_question=None,
                selected_action=None,
                stop=True,
                stop_reason="diagnostic_ready",
                unresolved_questions=tuple(),
            )
        questions = self.question_generator.generate(ranked, view)
        selected_question = questions[0] if questions else None
        actions = tuple()
        for question in questions:
            candidate_actions = self.action_binder.bind(
                question,
                roi_candidates,
                snapshot,
                plan_id,
            )
            if candidate_actions:
                selected_question = question
                actions = candidate_actions
                break
        if not actions and self._needs_quality_review(snapshot):
            quality_question = self.question_generator._question("Q_QUALITY")
            quality_actions = self.action_binder.bind(
                quality_question,
                roi_candidates,
                snapshot,
                plan_id,
            )
            if quality_actions:
                selected_question = quality_question
                actions = quality_actions
        reason = self.stop_policy.stop_reason(
            ranked,
            view,
            actions,
            int(round_index),
            int(max_actions),
            roi_candidates,
        )
        unresolved = []
        if ranked:
            unresolved.extend(ranked[0].missing_required_evidence)
            if ranked[0].dysplasia_state in ("unassessed", "not_evaluable", "conflicting"):
                unresolved.append("high_grade_or_definite_dysplasia")
        unresolved.extend(view.conflicts)
        if reason is not None:
            return PlanDecision(
                plan_id=plan_id,
                snapshot_id=snapshot.snapshot_id,
                decision="stop",
                ranked_hypotheses=ranked,
                selected_question=selected_question,
                selected_action=None,
                stop=True,
                stop_reason=reason,
                unresolved_questions=tuple(sorted(set(unresolved))),
            )
        return PlanDecision(
            plan_id=plan_id,
            snapshot_id=snapshot.snapshot_id,
            decision="invoke_reviewer",
            ranked_hypotheses=ranked,
            selected_question=selected_question,
            selected_action=actions[0],
            stop=False,
            stop_reason=None,
            unresolved_questions=tuple(sorted(set(unresolved))),
        )

    def _plan_from_state(
        self,
        snapshot,
        roi_candidates,
        agent_state,
        round_index=0,
        max_actions=8,
        reviewer_availability=None,
    ):
        state = agent_state if isinstance(agent_state, AgentState) else AgentState.from_dict(agent_state)
        if state.ledger_snapshot_id != snapshot.snapshot_id:
            raise ValueError("Planning AgentState does not reference the current Ledger snapshot")
        availability = defaultdict(lambda: True)
        availability.update(dict(reviewer_availability or {}))
        view = self.reducer.reduce(snapshot, active_evidence_ids=state.active_evidence_ids)
        ranked = self._assess_state(state, view)
        plan_id = "PLAN_{0}_{1:03d}".format(snapshot.snapshot_id, int(round_index))
        caused_by = (
            tuple(state.belief_history[-1].new_evidence_ids)
            if state.belief_history
            else tuple()
        )
        active_relations = tuple(
            relation.relation_id
            for relation in state.evidence_relations
            if relation.relation_type
            in ("true_contradiction", "quality_disagreement", "spatial_heterogeneity")
        )
        if self.behavioral_stop_policy.diagnostic_ready(state, ranked, view):
            top_state = state.hypothesis(ranked[0].hypothesis_id)
            second_score = (
                state.hypothesis(ranked[1].hypothesis_id).evidence_score
                if len(ranked) > 1
                else 0.0
            )
            return PlanDecision(
                plan_id=plan_id,
                snapshot_id=snapshot.snapshot_id,
                decision="stop",
                ranked_hypotheses=ranked,
                selected_question=None,
                selected_action=None,
                stop=True,
                stop_reason="diagnostic_ready",
                unresolved_questions=tuple(),
                input_state_id=state.state_id,
                termination_status="sufficient_evidence_stop",
                selected_discriminator_id=None,
                priority_trace={
                    "reason": "critical_discriminator_coverage_and_margin",
                    "top_hypothesis_id": top_state.hypothesis_id,
                    "evidence_score_margin": top_state.evidence_score - second_score,
                    "diagnostic_scope_status": state.diagnostic_scope_status,
                },
                caused_by_evidence_ids=caused_by,
                active_relation_ids=active_relations,
            )

        question_rows = self._state_questions(
            state,
            ranked,
            view,
            roi_candidates,
            availability,
        )
        selected_question = None
        selected_discriminator = None
        selected_actions = tuple()
        selected_trace = {}
        for priority_tuple, discriminator, question, trace, allow_recheck in question_rows:
            if not availability[question.reviewer_id]:
                continue
            candidate_pool = (
                self._conflict_recheck_candidates(
                    snapshot,
                    state,
                    question,
                    roi_candidates,
                )
                if allow_recheck
                else roi_candidates
            )
            candidate_actions = self.action_binder.bind(
                question,
                candidate_pool,
                snapshot,
                plan_id,
                allow_recheck=allow_recheck,
            )
            if not candidate_actions:
                continue
            selected_question = question
            selected_discriminator = discriminator
            selected_actions = candidate_actions
            selected_trace = dict(trace)
            selected_trace["action_score"] = candidate_actions[0].score
            selected_trace["chosen_roi_id"] = candidate_actions[0].roi_id
            selected_trace["chosen_scale"] = candidate_actions[0].scale
            break

        if not selected_actions and self._needs_quality_review(snapshot):
            question = self.question_generator._question("Q_QUALITY")
            if availability[question.reviewer_id]:
                quality_actions = self.action_binder.bind(
                    question,
                    roi_candidates,
                    snapshot,
                    plan_id,
                )
                if quality_actions:
                    selected_question = question
                    selected_discriminator = next(
                        (
                            item
                            for item in state.discriminators
                            if item.discriminator_id == question.question_id
                        ),
                        None,
                    )
                    selected_actions = quality_actions
                    selected_trace = {
                        "reason": "quality_recovery",
                        "chosen_roi_id": quality_actions[0].roi_id,
                        "chosen_scale": quality_actions[0].scale,
                    }

        termination_status, legacy_reason = self.behavioral_stop_policy.decide(
            state,
            ranked,
            view,
            selected_actions,
            int(round_index),
            int(max_actions),
            roi_candidates,
        )
        unresolved = tuple(
            sorted(
                set(
                    feature
                    for discriminator in state.discriminators
                    if discriminator.status != "resolved"
                    for feature in discriminator.feature_ids
                )
            )
        )
        if termination_status != "continue":
            return PlanDecision(
                plan_id=plan_id,
                snapshot_id=snapshot.snapshot_id,
                decision="stop",
                ranked_hypotheses=ranked,
                selected_question=selected_question,
                selected_action=None,
                stop=True,
                stop_reason=legacy_reason,
                unresolved_questions=unresolved,
                input_state_id=state.state_id,
                termination_status=termination_status,
                selected_discriminator_id=(
                    selected_discriminator.discriminator_id if selected_discriminator else None
                ),
                priority_trace=selected_trace,
                caused_by_evidence_ids=caused_by,
                active_relation_ids=active_relations,
            )
        return PlanDecision(
            plan_id=plan_id,
            snapshot_id=snapshot.snapshot_id,
            decision="invoke_reviewer",
            ranked_hypotheses=ranked,
            selected_question=selected_question,
            selected_action=selected_actions[0],
            stop=False,
            stop_reason=None,
            unresolved_questions=unresolved,
            input_state_id=state.state_id,
            termination_status="continue",
            selected_discriminator_id=(
                selected_discriminator.discriminator_id if selected_discriminator else None
            ),
            priority_trace=selected_trace,
            caused_by_evidence_ids=caused_by,
            active_relation_ids=active_relations,
        )

    def _assess_state(self, state, view):
        dysplasia_state = dysplasia_state_from_view(view)
        output = []
        for item in state.hypotheses:
            template = self.knowledge_base.hypothesis(item.hypothesis_id)
            active_effects = [
                effect
                for effect in state.evidence_effects
                if effect.active and effect.hypothesis_id == item.hypothesis_id
            ]
            conflict_mass = sum(
                abs(effect.contribution)
                for effect in active_effects
                if effect.direction == "oppose"
            )
            total_mass = sum(abs(effect.contribution) for effect in active_effects)
            coverage = len(item.observed_required_features) / float(
                max(1, len(item.required_features))
            )
            required_supported = all(
                any(
                    effect.feature_id == feature_id
                    and effect.direction == "support"
                    and effect.active
                    for effect in active_effects
                )
                for feature_id in item.required_features
            )
            output.append(
                HypothesisAssessment(
                    hypothesis_id=item.hypothesis_id,
                    pathway=item.pathway,
                    subtype=item.subtype,
                    dysplasia_state=dysplasia_state,
                    dysplasia_applicability=template.dysplasia_applicability,
                    observed_supporting_evidence=item.supporting_evidence_ids,
                    observed_conflict_evidence=item.contradicting_evidence_ids,
                    missing_required_evidence=item.missing_required_features,
                    plausibility_score=item.ranking_score,
                    required_evidence_coverage=_clamp(coverage),
                    conflict_score=_clamp(conflict_mass / total_mass if total_mass else 0.0),
                    decision_ready=(
                        required_supported and item.evidence_score > 0.0
                    ),
                )
            )
        output.sort(key=lambda value: (-value.plausibility_score, value.hypothesis_id))
        return tuple(output)

    def _state_questions(self, state, ranked, view, roi_candidates, reviewer_availability):
        if not ranked:
            return tuple()
        top = ranked[0]
        top_pair_ids = {item.hypothesis_id for item in ranked[:2]}
        top_state = state.hypothesis(top.hypothesis_id)
        question_by_id = {item.question_id: item for item in self.knowledge_base.questions}
        active_conflict_features = {
            relation.feature_id
            for relation in state.evidence_relations
            if relation.relation_type in ("true_contradiction", "quality_disagreement")
        }
        rows = []
        for discriminator in state.discriminators:
            question = question_by_id.get(discriminator.discriminator_id)
            if question is None or question.question_id == "Q_QUALITY":
                continue
            is_dysplasia = question.question_id == "Q_DYSPLASIA"
            if is_dysplasia:
                if not (
                    top.decision_ready
                    and top.dysplasia_state not in ("supported", "not_supported")
                ):
                    continue
            elif not (
                top_pair_ids.intersection(set(discriminator.target_hypothesis_ids))
                or set(discriminator.feature_ids).intersection(
                    set(top_state.missing_required_features)
                )
            ):
                continue
            conflict_recheck = bool(
                set(
                    QUESTION_FEATURE_BUNDLES.get(
                        question.question_id,
                        tuple(discriminator.feature_ids),
                    )
                ).intersection(active_conflict_features)
            )
            if discriminator.status == "resolved" and not conflict_recheck:
                continue
            target_scores = [
                state.hypothesis(hypothesis_id).evidence_score
                for hypothesis_id in discriminator.target_hypothesis_ids
                if hypothesis_id in {item.hypothesis_id for item in state.hypotheses}
            ]
            ambiguity = (
                1.0 / (1.0 + abs(max(target_scores) - min(target_scores)))
                if len(target_scores) >= 2
                else 0.5
            )
            evidence_gap = {
                "unresolved": 1.0,
                "conflicting": 1.0,
                "spatially_heterogeneous": 0.60,
                "blocked": 0.0,
                "unavailable": 0.0,
                "resolved": 0.0,
            }[discriminator.status]
            target_features = QUESTION_FEATURE_BUNDLES.get(
                question.question_id,
                (question.target_feature,),
            )
            compatible_rois = [
                roi
                for roi in roi_candidates
                if float(roi.scale) in question.allowed_scales
                and (
                    not roi.allowed_reviewers
                    or question.reviewer_id in roi.allowed_reviewers
                )
                and set(target_features).issubset(set(roi.candidate_features))
            ]
            expected_quality = max(
                (
                    roi.suitability * max(0.25, roi.spatial_coverage)
                    for roi in compatible_rois
                ),
                default=0.0,
            )
            reviewer_available = bool(reviewer_availability[question.reviewer_id])
            components = {
                "diagnostic_ambiguity": _clamp(ambiguity),
                "discriminator_weight": discriminator.importance,
                "evidence_gap": evidence_gap,
                "reviewer_availability": 1.0 if reviewer_available else 0.0,
                "expected_evidence_quality": expected_quality,
            }
            priority = 1.0 if conflict_recheck else 0.0
            priority += (
                components["diagnostic_ambiguity"]
                * components["discriminator_weight"]
                * components["evidence_gap"]
                * components["reviewer_availability"]
                * components["expected_evidence_quality"]
            )
            trace = dict(components)
            trace.update(
                {
                    "priority": priority,
                    "reason": "true_contradiction_recheck" if conflict_recheck else "unresolved_discriminator",
                    "discriminator_id": discriminator.discriminator_id,
                    "reviewer": discriminator.preferred_reviewer,
                    "task_profile": discriminator.task_profile,
                    "preferred_scales": list(discriminator.preferred_scales),
                    "target_hypothesis_ids": list(discriminator.target_hypothesis_ids),
                }
            )
            rows.append(
                (
                    (-priority, question.question_id),
                    discriminator,
                    question,
                    trace,
                    conflict_recheck,
                )
            )
        rows.sort(key=lambda item: item[0])
        return tuple(rows)

    @staticmethod
    def _conflict_recheck_candidates(snapshot, state, question, roi_candidates):
        target_features = set(
            QUESTION_FEATURE_BUNDLES.get(question.question_id, (question.target_feature,))
        )
        active_ids = set(state.active_evidence_ids)
        records = {record.evidence_id: record for record in snapshot.records}
        related_records = []
        for relation in state.evidence_relations:
            if relation.relation_type not in ("true_contradiction", "quality_disagreement"):
                continue
            if relation.feature_id not in target_features:
                continue
            for evidence_id in relation.evidence_ids:
                record = records.get(evidence_id)
                if record is not None and evidence_id in active_ids:
                    related_records.append(record)
        if not related_records:
            return tuple()
        output = []
        for candidate in roi_candidates:
            for record in related_records:
                same_roi = bool(record.roi_id and record.roi_id == candidate.roi_id)
                overlap = bbox_iou(record.level0_bbox, candidate.level0_bbox)
                scale_compatible = bool(
                    record.scale is None
                    or max(float(record.scale), float(candidate.scale))
                    / min(float(record.scale), float(candidate.scale))
                    <= 2.0 + 1e-9
                )
                if scale_compatible and (same_roi or overlap >= 0.80):
                    output.append(candidate)
                    break
        return tuple(output)

    @staticmethod
    def _needs_quality_review(snapshot):
        return any(
            record.evidence_type == "reviewer_evidence"
            and record.status == "not_evaluable"
            for record in snapshot.records
        )
