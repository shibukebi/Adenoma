from adenoma_agent.agentflow.contracts import ChiefDecision, ConflictObject
from adenoma_agent.agentflow.knowledge import default_knowledge_base
from adenoma_agent.agentflow.planner import EvidenceReducer


class StructuredGuidelineRetriever(object):
    """Retrieves only versioned local entries; it never free-generates advice."""

    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def retrieve(self, final_label):
        entry = self.knowledge_base.guideline_for_label(final_label)
        if entry is None:
            return {
                "status": "not_configured",
                "source": None,
                "source_version": None,
                "recommendation": None,
                "knowledge_version": self.knowledge_base.version,
                "non_clinical": True,
            }
        payload = dict(entry)
        payload.setdefault("status", "retrieved")
        payload.setdefault("knowledge_version", self.knowledge_base.version)
        payload.setdefault("non_clinical", True)
        return payload


class RuleBasedChiefAgent(object):
    """Contract-first Chief used until a validated Chief model is integrated."""

    def __init__(self, knowledge_base=None, max_conflict_rounds=2):
        self.knowledge_base = knowledge_base or default_knowledge_base()
        self.reducer = EvidenceReducer()
        self.guideline_retriever = StructuredGuidelineRetriever(self.knowledge_base)
        self.max_conflict_rounds = int(max_conflict_rounds)

    def decide(self, final_plan, snapshot, round_index=0, agent_state=None):
        view = self.reducer.reduce(
            snapshot,
            active_evidence_ids=(agent_state.active_evidence_ids if agent_state is not None else None),
        )
        ranked = final_plan.ranked_hypotheses
        top = ranked[0] if ranked else None
        conflicts = self._conflicts(top, ranked, view, round_index, agent_state=agent_state)
        conflict_ids = []
        for conflict in conflicts:
            conflict_ids.extend(conflict.conflicting_evidence_ids)
        if top is None:
            return self._uncertain(
                top,
                final_plan,
                conflicts,
                conflict_ids,
                reason="No morphology hypothesis could be ranked.",
            )
        if agent_state is not None and agent_state.diagnostic_scope_status != "adequate":
            return self._uncertain(
                top,
                final_plan,
                conflicts,
                conflict_ids,
                reason="Diagnostic scope is incomplete or outside the seven-hypothesis morphology scope.",
            )
        if final_plan.termination_status != "sufficient_evidence_stop" or conflicts:
            return self._uncertain(
                top,
                final_plan,
                conflicts,
                conflict_ids,
                reason="Evidence is incomplete, conflicting, or the review budget stopped before diagnostic readiness.",
            )
        template = self.knowledge_base.hypothesis(top.hypothesis_id)
        dysplasia_state = top.dysplasia_state
        if dysplasia_state not in ("supported", "not_supported"):
            return self._uncertain(
                top,
                final_plan,
                conflicts,
                conflict_ids,
                reason="Dysplasia state is unresolved.",
            )
        final_label = template.final_label_mapping.get(dysplasia_state)
        if final_label is None:
            return self._uncertain(
                top,
                final_plan,
                conflicts,
                conflict_ids,
                reason="The morphology/dysplasia state has no legal 11-class projection.",
            )
        supporting = list(top.observed_supporting_evidence)
        dysplasia_feature = view.features.get("high_grade_or_definite_dysplasia")
        if dysplasia_feature is not None:
            if dysplasia_state == "supported":
                supporting.extend(dysplasia_feature.present_evidence_ids)
            elif dysplasia_state == "not_supported":
                supporting.extend(dysplasia_feature.absent_evidence_ids)
        return ChiefDecision(
            status="final",
            final_label=final_label,
            final_diagnosis=self.knowledge_base.diagnosis_name(final_label),
            diagnostic_confidence=min(
                0.99,
                max(0.0, top.required_evidence_coverage * (1.0 - top.conflict_score)),
            ),
            morphology_hypothesis_id=top.hypothesis_id,
            dysplasia_state=dysplasia_state,
            supporting_evidence_ids=tuple(sorted(set(supporting))),
            conflicting_evidence_ids=tuple(sorted(set(conflict_ids))),
            unresolved_questions=tuple(),
            conflicts=tuple(conflicts),
            management_recommendation=self.guideline_retriever.retrieve(final_label),
            knowledge_version=self.knowledge_base.version,
            non_clinical=True,
        )

    def _uncertain(self, top, final_plan, conflicts, conflict_ids, reason):
        unresolved = list(final_plan.unresolved_questions)
        unresolved.append(reason)
        supporting = tuple(top.observed_supporting_evidence) if top else tuple()
        return ChiefDecision(
            status="uncertain",
            final_label=None,
            final_diagnosis=None,
            diagnostic_confidence=(min(0.49, top.plausibility_score) if top else 0.0),
            morphology_hypothesis_id=(top.hypothesis_id if top else None),
            dysplasia_state=(top.dysplasia_state if top else "unassessed"),
            supporting_evidence_ids=tuple(sorted(set(supporting))),
            conflicting_evidence_ids=tuple(sorted(set(conflict_ids))),
            unresolved_questions=tuple(sorted(set(unresolved))),
            conflicts=tuple(conflicts),
            management_recommendation=None,
            knowledge_version=self.knowledge_base.version,
            non_clinical=True,
        )

    def _conflicts(self, top, ranked, view, round_index, agent_state=None):
        output = []
        state_conflicts = tuple(
            relation
            for relation in getattr(agent_state, "evidence_relations", ())
            if relation.relation_type in ("true_contradiction", "quality_disagreement")
        )
        if state_conflicts:
            for relation in state_conflicts:
                output.append(
                    ConflictObject(
                        conflict_id=relation.relation_id,
                        conflict_type=relation.relation_type,
                        competing_hypothesis_ids=tuple(
                            item.hypothesis_id for item in ranked[:2]
                        ),
                        conflicting_evidence_ids=tuple(relation.evidence_ids),
                        missing_discriminative_evidence=(relation.feature_id,),
                        recommended_actions=tuple(),
                        round_index=int(round_index),
                        max_rounds=self.max_conflict_rounds,
                    )
                )
        elif view.conflicts:
            evidence_ids = []
            for feature_name in view.conflicts:
                feature = view.features[feature_name]
                evidence_ids.extend(feature.present_evidence_ids)
                evidence_ids.extend(feature.absent_evidence_ids)
            output.append(
                ConflictObject(
                    conflict_id="CONFLICT_EVIDENCE_{0:03d}".format(int(round_index)),
                    conflict_type="evidence_disagreement",
                    competing_hypothesis_ids=tuple(item.hypothesis_id for item in ranked[:2]),
                    conflicting_evidence_ids=tuple(sorted(set(evidence_ids))),
                    missing_discriminative_evidence=tuple(view.conflicts),
                    recommended_actions=tuple(),
                    round_index=int(round_index),
                    max_rounds=self.max_conflict_rounds,
                )
            )
        if top is not None and top.dysplasia_applicability != "class_defining" and top.dysplasia_state == "supported":
            feature = view.features.get("high_grade_or_definite_dysplasia")
            output.append(
                ConflictObject(
                    conflict_id="CONFLICT_MORPH_DYS_{0:03d}".format(int(round_index)),
                    conflict_type="morphology_dysplasia_conflict",
                    competing_hypothesis_ids=(top.hypothesis_id,),
                    conflicting_evidence_ids=(tuple(feature.present_evidence_ids) if feature else tuple()),
                    missing_discriminative_evidence=tuple(top.missing_required_evidence),
                    recommended_actions=tuple(),
                    round_index=int(round_index),
                    max_rounds=self.max_conflict_rounds,
                )
            )
        return tuple(output)
