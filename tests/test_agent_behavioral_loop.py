import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from adenoma_agent.agentflow.contracts import ArchitecturePatchPrediction, EvidenceRecord
from adenoma_agent.agentflow.evidence import EvidenceEngine
from adenoma_agent.agentflow.ledger import EvidenceLedger
from adenoma_agent.agentflow.orchestrator import AgentFlowOrchestrator, VirtualROICropper
from adenoma_agent.agentflow.planner import PlanningAgent
from adenoma_agent.agentflow.reviewer import (
    ModelUnavailableError,
    ScriptedReviewerBackend,
    UnavailableReviewerBackend,
    build_reviewer_request,
    default_reviewer_registry,
)
from adenoma_agent.agentflow.spatial import ROIManager, SpatialEvidenceEvaluator
from adenoma_agent.agentflow.state import AgentTraceStore, StateStore


CASE_ID = "behavior_case"


def _prediction(patch_id="patch_serrated"):
    return ArchitecturePatchPrediction(
        patch_id=patch_id,
        slide_id="safe_slide",
        level0_bbox=(1024, 1024, 3072, 3072),
        mucosa_coverage=0.9,
        evaluable=0.95,
        architecture={"serrated": 0.95, "tubular": 0.02, "villous": 0.02},
        context={
            "normal_mucosa_present": 0.10,
            "reactive_inflammatory_present": 0.02,
            "other_pattern_present": 0.02,
        },
        uncertainty=0.60,
        source_model="deterministic-architecture-fixture",
        component_ids=(1,),
        dysplasia_risk=0.90,
        abnormal_epithelial_score=0.80,
        metadata={"synthetic_stub": True, "non_clinical": True},
    )


def _evidence(
    evidence_id,
    feature,
    status="present",
    case_id=CASE_ID,
    confidence=1.0,
    quality=1.0,
    evaluability="adequate",
    roi_id="ROI_10X_patch_serrated",
    bbox=(1536, 1536, 2560, 2560),
    scale=10.0,
    cluster_id=None,
    source="DeterministicReviewer",
    metadata=None,
    **kwargs
):
    return EvidenceRecord(
        evidence_id=evidence_id,
        case_id=case_id,
        evidence_type=kwargs.pop("evidence_type", "reviewer_evidence"),
        feature=feature,
        status=status,
        confidence=confidence,
        source=source,
        source_version=kwargs.pop("source_version", "v1"),
        quality=quality,
        feature_evaluability=evaluability,
        roi_id=roi_id,
        scale=scale,
        level0_bbox=bbox,
        cluster_id=cluster_id,
        metadata=dict(metadata or {}),
        **kwargs
    )


def _state_for(records=(), case_id=CASE_ID, case_context=None, engine=None):
    engine = engine or EvidenceEngine()
    ledger = EvidenceLedger(case_id)
    ledger.append_many(tuple(records))
    initial = engine.initialize_state(case_id, case_context=case_context)
    state = engine.update(initial, ledger.snapshot(), round_id=1)
    return engine, ledger, state


def _effect(state, hypothesis_id, evidence_id):
    return next(
        item
        for item in state.evidence_effects
        if item.hypothesis_id == hypothesis_id and item.evidence_id == evidence_id
    )


class AgentBehavioralLoopTest(unittest.TestCase):
    def test_01_competing_hypotheses_remain_persistent(self):
        _engine, _ledger, state = _state_for(
            (_evidence("E_SERRATED", "serrated_architecture", confidence=0.7),)
        )
        candidates = [
            item
            for item in state.hypotheses
            if item.status in ("active", "strengthened", "uncertain")
        ]
        self.assertGreaterEqual(len(candidates), 4)
        self.assertEqual(len(state.hypotheses), 7)
        self.assertAlmostEqual(sum(item.ranking_score for item in state.hypotheses), 1.0)
        self.assertLess(max(item.ranking_score for item in state.hypotheses), 0.5)
        self.assertTrue(all(item.evidence_score == 0.0 for item in EvidenceEngine().initialize_state("zero").hypotheses))

    def test_02_missing_discriminator_routes_to_existing_reviewer_profile_and_scale(self):
        prediction = _prediction()
        evaluator = SpatialEvidenceEvaluator()
        architecture_records = evaluator.to_evidence(
            CASE_ID,
            (prediction,),
            evaluator.evaluate((prediction,)),
        )
        _engine, ledger, state = _state_for(architecture_records)
        candidates = ROIManager().build_candidates((prediction,), slide_dimensions=(4096, 4096))
        plan = PlanningAgent().plan(
            ledger.snapshot(),
            candidates,
            agent_state=state,
            round_index=0,
            max_actions=4,
        )
        self.assertFalse(plan.stop)
        self.assertEqual(plan.selected_question.question_id, "Q_SSL_HP_CRYPT_BASE")
        self.assertEqual(plan.selected_action.reviewer_id, "SerratedArchitectureReviewer")
        self.assertEqual(plan.selected_action.task_profile, "ssl_hp_discrimination")
        self.assertEqual(plan.selected_action.scale, 10.0)
        self.assertEqual(plan.priority_trace["reason"], "unresolved_discriminator")
        self.assertEqual(plan.priority_trace["chosen_roi_id"], plan.selected_action.roi_id)

    def test_03_tsa_signature_and_dysplasia_ownership_do_not_leak(self):
        _engine, _ledger, state = _state_for(
            (_evidence("E_HGD", "high_grade_or_definite_dysplasia"),)
        )
        tsa_effect = _effect(state, "H_TSA", "E_HGD")
        self.assertEqual(tsa_effect.direction, "neutral")
        self.assertEqual(tsa_effect.diagnostic_weight, 0.0)
        self.assertIn(":no_relation:H_TSA:", tsa_effect.kb_rule_id)
        self.assertEqual(state.hypothesis("H_TSA").evidence_score, 0.0)
        self.assertNotIn(
            "high_grade_or_definite_dysplasia",
            state.hypothesis("H_TSA").supporting_evidence_ids,
        )

    def test_04_one_observation_has_hypothesis_relative_directions(self):
        _engine, _ledger, state = _state_for(
            (_evidence("E_BASAL", "basal_crypt_dilation"),)
        )
        ssl = _effect(state, "H_SSL", "E_BASAL")
        hp = _effect(state, "H_HP", "E_BASAL")
        tsa = _effect(state, "H_TSA", "E_BASAL")
        self.assertEqual((ssl.direction, hp.direction, tsa.direction), ("support", "oppose", "neutral"))
        self.assertGreater(ssl.contribution, 0.0)
        self.assertLess(hp.contribution, 0.0)
        self.assertEqual(tsa.contribution, 0.0)
        self.assertTrue(ssl.kb_rule_id and hp.kb_rule_id and tsa.kb_rule_id)

    def test_05_quality_weights_diagnostic_contribution(self):
        _engine, _ledger, high = _state_for(
            (_evidence("E_HIGH", "basal_crypt_dilation", quality=1.0),)
        )
        _engine, _ledger, low = _state_for(
            (_evidence("E_LOW", "basal_crypt_dilation", quality=0.2),)
        )
        high_value = _effect(high, "H_SSL", "E_HIGH").contribution
        low_value = _effect(low, "H_SSL", "E_LOW").contribution
        self.assertAlmostEqual(low_value, high_value * 0.2)
        self.assertLess(low_value, high_value)

    def test_06_exact_duplicate_has_zero_additional_belief(self):
        first = _evidence("E_DUP_1", "basal_crypt_dilation")
        second = _evidence("E_DUP_2", "basal_crypt_dilation")
        _engine, _ledger, single = _state_for((first,))
        _engine, _ledger, duplicated = _state_for((first, second))
        self.assertEqual(
            single.hypothesis("H_SSL").evidence_score,
            duplicated.hypothesis("H_SSL").evidence_score,
        )
        self.assertIn("E_DUP_1", duplicated.active_evidence_ids)
        self.assertNotIn("E_DUP_2", duplicated.active_evidence_ids)
        self.assertTrue(
            any(item.relation_type == "exact_duplicate" for item in duplicated.evidence_relations)
        )

    def test_07_reviewer_evidence_changes_the_next_plan(self):
        backend = ScriptedReviewerBackend(
            {
                "Q_SSL_HP_CRYPT_BASE": [
                    {
                        "serration_to_crypt_base": "present",
                        "basal_crypt_dilation": "present",
                        "surface_limited_serration": "absent",
                        "straight_crypt_bases": "absent",
                    }
                ],
                "Q_DYSPLASIA": [{"high_grade_or_definite_dysplasia": "absent"}],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run"
            result = AgentFlowOrchestrator(
                reviewer_backend=backend,
                cropper=VirtualROICropper(),
                max_actions=4,
            ).run(
                case_id=CASE_ID,
                predictions=(_prediction(),),
                slide_dimensions=(4096, 4096),
                output_dir=output,
            )
            questions = [
                plan.selected_question.question_id if plan.selected_question else None
                for plan in result.plans
            ]
            self.assertEqual(questions, ["Q_SSL_HP_CRYPT_BASE", "Q_DYSPLASIA", None])
            second_plan = result.plans[1]
            self.assertNotEqual(result.plans[0].selected_action.action_id, second_plan.selected_action.action_id)
            self.assertTrue(
                any(value.startswith("REV_REQ_") for value in second_plan.caused_by_evidence_ids)
            )
            self.assertEqual(second_plan.priority_trace["discriminator_id"], "Q_DYSPLASIA")
            replayed = AgentTraceStore(CASE_ID, output / "trace" / "events.jsonl", replay=True)
            transitions = replayed.plan_transitions()
            self.assertTrue(transitions[1]["changed_from_previous"])
            self.assertEqual(transitions[1]["question_id"], "Q_DYSPLASIA")
            self.assertTrue(transitions[1]["caused_by_evidence_ids"])
            self.assertEqual(result.chief_decision.status, "final")

    def test_08_true_contradiction_prioritizes_legal_feature_recheck(self):
        records = (
            _evidence(
                "E_PRESENT",
                "serration_to_crypt_base",
                status="present",
                metadata={"observation_id": "OBS_PRESENT"},
            ),
            _evidence(
                "E_ABSENT",
                "serration_to_crypt_base",
                status="absent",
                source="IndependentReviewer",
                metadata={"observation_id": "OBS_ABSENT"},
            ),
        )
        _engine, ledger, state = _state_for(records)
        relation = next(
            item for item in state.evidence_relations if item.relation_type == "true_contradiction"
        )
        self.assertEqual(relation.feature_id, "serration_to_crypt_base")
        contradiction = next(
            item for item in state.contradictions if item.contradiction_id == relation.relation_id
        )
        self.assertEqual(contradiction.resolution_status, "active")
        self.assertEqual(contradiction.target_roi_ids, ("ROI_10X_patch_serrated",))
        self.assertIn("H_SSL", contradiction.affected_hypothesis_ids)
        prediction = _prediction()
        unrelated = replace(
            _prediction("patch_unrelated"),
            level0_bbox=(4096, 4096, 6144, 6144),
            component_ids=(2,),
        )
        candidates = ROIManager().build_candidates(
            (prediction, unrelated), slide_dimensions=(8192, 8192)
        )
        plan = PlanningAgent().plan(
            ledger.snapshot(), candidates, agent_state=state, round_index=0, max_actions=3
        )
        self.assertEqual(plan.priority_trace["reason"], "true_contradiction_recheck")
        self.assertEqual(plan.selected_action.reviewer_id, "SerratedArchitectureReviewer")
        self.assertEqual(plan.selected_action.task_profile, "ssl_hp_discrimination")
        self.assertEqual(plan.selected_action.roi_id, "ROI_10X_patch_serrated")
        roi = next(item for item in candidates if item.roi_id == plan.selected_action.roi_id)
        disagreements = AgentFlowOrchestrator._feature_disagreements(
            state, ledger.snapshot(), plan.selected_action
        )
        request = build_reviewer_request(
            snapshot_id=plan.snapshot_id,
            plan_id=plan.plan_id,
            action=plan.selected_action,
            question=plan.selected_question,
            roi=roi,
            registry=default_reviewer_registry(),
            feature_disagreements=disagreements,
        )
        self.assertEqual(
            request.model_input["feature_disagreements"][0]["feature_id"],
            "serration_to_crypt_base",
        )
        self.assertNotEqual(request.reviewer, "ConflictReviewer")

    def test_09_reviewer_failure_never_fabricates_observation_or_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run"
            result = AgentFlowOrchestrator(
                reviewer_backend=UnavailableReviewerBackend("deterministic outage"),
                cropper=VirtualROICropper(),
                max_actions=3,
            ).run(
                case_id=CASE_ID,
                predictions=(_prediction(),),
                slide_dimensions=(4096, 4096),
                output_dir=output,
            )
            snapshot = EvidenceLedger.from_jsonl(CASE_ID, result.ledger_path).snapshot()
            reviewer_records = [
                item for item in snapshot.records if item.evidence_type == "reviewer_evidence"
            ]
            failures = [
                item for item in snapshot.records if item.evidence_type == "reviewer_invocation_failure"
            ]
            self.assertEqual(reviewer_records, [])
            self.assertEqual(len(failures), 1)
            self.assertEqual(result.plans[-1].termination_status, "failure_stop")
            state = StateStore.from_jsonl(CASE_ID, output / "state" / "snapshots.jsonl").latest
            self.assertEqual(state.termination_state.decision, "failure_stop")
            self.assertEqual(len(state.tool_reviewer_failures), 1)
            self.assertTrue(any(item.status == "blocked" for item in state.discriminators))

    def test_10_stop_policy_uses_coverage_margin_quality_scope_budget_and_availability(self):
        prediction = _prediction()
        evaluator = SpatialEvidenceEvaluator()
        records = evaluator.to_evidence(CASE_ID, (prediction,), evaluator.evaluate((prediction,)))
        _engine, ledger, state = _state_for(records)
        candidates = ROIManager().build_candidates((prediction,), slide_dimensions=(4096, 4096))
        planner = PlanningAgent()
        continuing = planner.plan(
            ledger.snapshot(), candidates, agent_state=state, round_index=0, max_actions=3
        )
        self.assertEqual(continuing.termination_status, "continue")
        self.assertFalse(continuing.stop)
        budget = planner.plan(
            ledger.snapshot(), candidates, agent_state=state, round_index=0, max_actions=0
        )
        self.assertEqual(budget.termination_status, "budget_stop")
        out_of_scope = replace(state, diagnostic_scope_status="out_of_scope_suspected")
        unresolved = planner.plan(
            ledger.snapshot(), candidates, agent_state=out_of_scope, round_index=0, max_actions=3
        )
        self.assertEqual(unresolved.termination_status, "unresolved_stop")
        unavailable = planner.plan(
            ledger.snapshot(),
            candidates,
            agent_state=state,
            round_index=0,
            max_actions=3,
            reviewer_availability={item.reviewer_id: False for item in default_reviewer_registry().profiles},
        )
        self.assertEqual(unavailable.termination_status, "unresolved_stop")

    def test_11_trace_replay_explains_score_and_request_changes(self):
        backend = ScriptedReviewerBackend(
            {
                "Q_SSL_HP_CRYPT_BASE": [
                    {
                        "serration_to_crypt_base": "present",
                        "basal_crypt_dilation": "present",
                        "surface_limited_serration": "absent",
                        "straight_crypt_bases": "absent",
                    }
                ],
                "Q_DYSPLASIA": [{"high_grade_or_definite_dysplasia": "present"}],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run"
            AgentFlowOrchestrator(
                reviewer_backend=backend,
                cropper=VirtualROICropper(),
                max_actions=4,
            ).run(
                case_id=CASE_ID,
                predictions=(_prediction(),),
                slide_dimensions=(4096, 4096),
                output_dir=output,
            )
            trace = AgentTraceStore(CASE_ID, output / "trace" / "events.jsonl", replay=True)
            ssl_updates = [row for row in trace.hypothesis_updates("H_SSL") if row["before"] != row["after"]]
            self.assertTrue(ssl_updates)
            self.assertTrue(ssl_updates[-1]["evidence_ids"])
            plans = trace.plan_transitions()
            self.assertEqual(plans[0]["question_id"], "Q_SSL_HP_CRYPT_BASE")
            self.assertEqual(plans[1]["question_id"], "Q_DYSPLASIA")
            self.assertTrue(plans[1]["changed_from_previous"])
            self.assertTrue(plans[1]["caused_by_evidence_ids"])

    def test_12_supportive_absence_is_neutral_unless_kb_marks_it_informative(self):
        records = (
            _evidence("E_SUPPORTIVE_ABSENT", "crypt_branching", status="absent"),
            _evidence("E_REQUIRED_ABSENT", "basal_crypt_dilation", status="absent", roi_id="ROI_B"),
            _evidence("E_CONTRADICTORY_PRESENT", "straight_crypt_bases", status="present", roi_id="ROI_C"),
            _evidence(
                "E_NOT_EVALUABLE",
                "serration_to_crypt_base",
                status="not_evaluable",
                evaluability="not_evaluable",
                quality=0.0,
                roi_id="ROI_D",
            ),
        )
        _engine, _ledger, state = _state_for(records)
        self.assertEqual(_effect(state, "H_SSL", "E_SUPPORTIVE_ABSENT").direction, "neutral")
        self.assertEqual(_effect(state, "H_SSL", "E_REQUIRED_ABSENT").direction, "oppose")
        self.assertEqual(_effect(state, "H_SSL", "E_CONTRADICTORY_PRESENT").direction, "oppose")
        self.assertEqual(_effect(state, "H_SSL", "E_NOT_EVALUABLE").direction, "neutral")

    def test_13_cross_roi_present_absent_is_heterogeneity_not_contradiction(self):
        records = (
            _evidence(
                "E_ROI_A",
                "serration_to_crypt_base",
                status="present",
                roi_id="ROI_A",
                bbox=(0, 0, 1000, 1000),
            ),
            _evidence(
                "E_ROI_B",
                "serration_to_crypt_base",
                status="absent",
                roi_id="ROI_B",
                bbox=(2000, 2000, 3000, 3000),
                source="IndependentReviewer",
            ),
        )
        _engine, _ledger, state = _state_for(records)
        relation_types = {item.relation_type for item in state.evidence_relations}
        self.assertIn("spatial_heterogeneity", relation_types)
        self.assertNotIn("true_contradiction", relation_types)
        discriminator = next(
            item for item in state.discriminators if item.discriminator_id == "Q_SSL_HP_CRYPT_BASE"
        )
        self.assertEqual(discriminator.status, "spatially_heterogeneous")

    def test_14_cluster_saturation_prevents_linear_belief_inflation(self):
        records = (
            _evidence(
                "E_CLUSTER_1",
                "basal_crypt_dilation",
                cluster_id="CLUSTER_SHARED",
                roi_id="ROI_A",
                bbox=(0, 0, 512, 512),
                source="ReviewerA",
            ),
            _evidence(
                "E_CLUSTER_2",
                "basal_crypt_dilation",
                cluster_id="CLUSTER_SHARED",
                roi_id="ROI_B",
                bbox=(1024, 0, 1536, 512),
                source="ReviewerB",
            ),
            _evidence(
                "E_CLUSTER_3",
                "basal_crypt_dilation",
                cluster_id="CLUSTER_SHARED",
                roi_id="ROI_C",
                bbox=(2048, 0, 2560, 512),
                source="ReviewerC",
            ),
        )
        _engine, _ledger, single = _state_for(records[:1])
        _engine, _ledger, saturated = _state_for(records)
        self.assertEqual(
            single.hypothesis("H_SSL").evidence_score,
            saturated.hypothesis("H_SSL").evidence_score,
        )
        self.assertEqual(
            len(
                [
                    item
                    for item in saturated.evidence_effects
                    if item.hypothesis_id == "H_SSL"
                    and item.feature_id == "basal_crypt_dilation"
                    and item.active
                ]
            ),
            1,
        )
        self.assertTrue(
            any(item.relation_type == "cluster_saturation" for item in saturated.evidence_relations)
        )

    def test_15_real_smoke_has_no_label_leakage(self):
        private_ground_truth = "SECRET_GROUND_TRUTH_HP"
        with self.assertRaisesRegex(ValueError, "outside the inference allowlist"):
            AgentFlowOrchestrator._sanitize_case_context(
                {"case_alias": "SAFE", "ground_truth_label": private_ground_truth}
            )
        with self.assertRaisesRegex(ValueError, "outside the inference allowlist"):
            AgentFlowOrchestrator._sanitize_case_context(
                {"case_alias": "SAFE", "source_name": "Adenoma_hp"}
            )
        with self.assertRaisesRegex(ValueError, "nested metadata is not inference-safe"):
            AgentFlowOrchestrator._sanitize_case_context(
                {
                    "case_alias": "SAFE",
                    "wsi_metadata_ref": {"ground_truth_label": private_ground_truth},
                }
            )
        backend = ScriptedReviewerBackend(
            {
                "Q_SSL_HP_CRYPT_BASE": [
                    {
                        "serration_to_crypt_base": "present",
                        "basal_crypt_dilation": "present",
                        "surface_limited_serration": "absent",
                        "straight_crypt_bases": "absent",
                    }
                ],
                "Q_DYSPLASIA": [{"high_grade_or_definite_dysplasia": "absent"}],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run"
            AgentFlowOrchestrator(
                reviewer_backend=backend,
                cropper=VirtualROICropper(),
                max_actions=4,
            ).run(
                case_id="SAFE_ALIAS",
                predictions=(_prediction(),),
                slide_dimensions=(4096, 4096),
                output_dir=output,
                case_context={
                    "case_alias": "SAFE_ALIAS",
                    "source_code": "SOURCE_001",
                    "source_format": "svs",
                    "diagnostic_scope_status": "incomplete",
                    "non_clinical": True,
                },
            )
            inference_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*.json*")
                if path.is_file()
            )
            self.assertNotIn(private_ground_truth, inference_text)
            self.assertNotIn("ground_truth_label", inference_text)
            self.assertNotIn('"diagnosis":', inference_text)

    def test_16_rejected_hypothesis_can_be_reactivated_by_corrections(self):
        originals = (
            _evidence(
                "E_OPPOSE_1",
                "basal_crypt_dilation",
                cluster_id="C1",
                roi_id="R1",
                bbox=(0, 0, 512, 512),
            ),
            _evidence(
                "E_OPPOSE_2",
                "basal_crypt_dilation",
                cluster_id="C2",
                roi_id="R2",
                bbox=(1024, 0, 1536, 512),
            ),
            _evidence(
                "E_OPPOSE_3",
                "serration_to_crypt_base",
                cluster_id="C3",
                roi_id="R3",
                bbox=(2048, 0, 2560, 512),
            ),
        )
        engine, ledger, rejected = _state_for(originals)
        self.assertEqual(rejected.hypothesis("H_HP").status, "rejected")
        corrections = tuple(
            _evidence(
                "CORR_{0}".format(index),
                record.feature,
                status="absent",
                roi_id=record.roi_id,
                bbox=record.level0_bbox,
                cluster_id=record.cluster_id,
                source="CorrectionReviewer",
                correction_of_evidence_id=record.evidence_id,
            )
            for index, record in enumerate(originals, 1)
        )
        ledger.append_many(corrections)
        reactivated = engine.update(rejected, ledger.snapshot(), round_id=2)
        self.assertGreater(reactivated.hypothesis("H_HP").evidence_score, rejected.hypothesis("H_HP").evidence_score)
        self.assertNotEqual(reactivated.hypothesis("H_HP").status, "rejected")
        self.assertTrue(set(record.evidence_id for record in originals).issubset(set(reactivated.superseded_evidence_ids)))
        active_view = PlanningAgent().reducer.reduce(
            ledger.snapshot(),
            active_evidence_ids=reactivated.active_evidence_ids,
        )
        self.assertNotIn("basal_crypt_dilation", active_view.conflicts)
        self.assertNotIn("serration_to_crypt_base", active_view.conflicts)

    def test_17_conflict_recheck_adjudicates_old_evidence_and_changes_plan(self):
        initial_conflict = (
            _evidence(
                "E_CONFLICT_PRESENT",
                "serration_to_crypt_base",
                status="present",
                metadata={"observation_id": "OBS_CONFLICT_PRESENT"},
            ),
            _evidence(
                "E_CONFLICT_ABSENT",
                "serration_to_crypt_base",
                status="absent",
                source="IndependentReviewer",
                metadata={"observation_id": "OBS_CONFLICT_ABSENT"},
            ),
        )
        backend = ScriptedReviewerBackend(
            {
                "Q_SSL_HP_CRYPT_BASE": [
                    {
                        "serration_to_crypt_base": "present",
                        "basal_crypt_dilation": "present",
                        "surface_limited_serration": "absent",
                        "straight_crypt_bases": "absent",
                    }
                ],
                "Q_DYSPLASIA": [{"high_grade_or_definite_dysplasia": "absent"}],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run"
            result = AgentFlowOrchestrator(
                reviewer_backend=backend,
                cropper=VirtualROICropper(),
                max_actions=4,
            ).run(
                case_id=CASE_ID,
                predictions=(_prediction(),),
                initial_evidence=initial_conflict,
                slide_dimensions=(4096, 4096),
                output_dir=output,
            )
            questions = [
                plan.selected_question.question_id if plan.selected_question else None
                for plan in result.plans
            ]
            self.assertEqual(questions, ["Q_SSL_HP_CRYPT_BASE", "Q_DYSPLASIA", None])
            final_state = StateStore.from_jsonl(
                CASE_ID, output / "state" / "snapshots.jsonl"
            ).latest
            self.assertFalse(
                any(
                    relation.relation_type == "true_contradiction"
                    for relation in final_state.evidence_relations
                )
            )
            self.assertTrue(
                {"E_CONFLICT_PRESENT", "E_CONFLICT_ABSENT"}.issubset(
                    set(final_state.superseded_evidence_ids)
                )
            )
            self.assertEqual(final_state.hypotheses[0].status, "resolved")
            ledger = EvidenceLedger.from_jsonl(CASE_ID, result.ledger_path).snapshot()
            adjudication = next(
                record
                for record in ledger.records
                if record.feature == "serration_to_crypt_base"
                and record.metadata.get("adjudicates_feature_disagreement")
            )
            self.assertEqual(
                set(adjudication.supersedes_evidence_ids),
                {"E_CONFLICT_PRESENT", "E_CONFLICT_ABSENT"},
            )


if __name__ == "__main__":
    unittest.main()
