import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from adenoma_agent.agentflow.architecture_runtime import (
    ArchitectureInferenceRuntime,
    ArchitectureModelUnavailableError,
    ScriptedArchitecturePredictor,
    load_five_x_manifest,
)
from adenoma_agent.agentflow.contracts import (
    ArchitecturePatchPrediction,
    EvidenceRecord,
    ReviewerAction,
    ReviewerFinding,
    ReviewerObservation,
    ROICandidate,
)
from adenoma_agent.agentflow.knowledge import default_knowledge_base
from adenoma_agent.agentflow.ledger import EvidenceLedger
from adenoma_agent.agentflow.orchestrator import AgentFlowOrchestrator, VirtualROICropper
from adenoma_agent.agentflow.planner import EvidenceReducer, HypothesisEngine
from adenoma_agent.agentflow.reviewer import (
    ModelUnavailableError,
    ReviewerContractError,
    ScriptedReviewerBackend,
    UnavailableReviewerBackend,
    build_reviewer_request,
    default_reviewer_registry,
)
from adenoma_agent.agentflow.spatial import ROIManager


def _write_jsonl(path, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _manifest_row(patch_id="patch_ssl", coverage=0.9, magnification="5x"):
    return {
        "slide_id": "slide_001",
        "patch_id": patch_id,
        "level0_bbox": [1024, 1024, 3072, 3072],
        "mucosa_coverage": coverage,
        "target_magnification": magnification,
        "source_component_ids": [1],
        "mean_uncertainty": 0.6,
        "status": "retained",
    }


def _ssl_prediction(patch_id="patch_ssl"):
    return ArchitecturePatchPrediction(
        patch_id=patch_id,
        slide_id="slide_001",
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
        source_model="scripted-architecture-v1",
        component_ids=(1,),
        dysplasia_risk=0.95,
        abnormal_epithelial_score=0.80,
        metadata={"synthetic_stub": True, "non_clinical": True},
    )


def _evidence(evidence_id, feature, status="present", evaluability="adequate", **kwargs):
    return EvidenceRecord(
        evidence_id=evidence_id,
        case_id="case_001",
        evidence_type=kwargs.pop("evidence_type", "test_evidence"),
        feature=feature,
        status=status,
        confidence=kwargs.pop("confidence", 0.9),
        source=kwargs.pop("source", "SyntheticFixture"),
        source_version=kwargs.pop("source_version", "v1"),
        quality=kwargs.pop("quality", 1.0),
        feature_evaluability=evaluability,
        roi_id=kwargs.pop("roi_id", "ROI_TEST"),
        scale=kwargs.pop("scale", 10.0),
        level0_bbox=kwargs.pop("level0_bbox", (0, 0, 1024, 1024)),
        metadata=kwargs.pop("metadata", {"fixture": True}),
        **kwargs
    )


class AgentFlowV1Test(unittest.TestCase):
    def test_five_x_manifest_is_the_strict_canonical_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "five_x_patch_manifest.jsonl"
            _write_jsonl(
                manifest,
                [
                    _manifest_row("kept", coverage=0.75, magnification="5x"),
                    _manifest_row("below_threshold", coverage=0.20, magnification=5.0),
                ],
            )
            rows = load_five_x_manifest(manifest, min_mucosa_coverage=0.30)
            self.assertEqual([row["patch_id"] for row in rows], ["kept"])
            self.assertEqual(rows[0]["level0_bbox"], (1024, 1024, 3072, 3072))

            _write_jsonl(manifest, [_manifest_row("wrong_scale", magnification="10x")])
            with self.assertRaisesRegex(ValueError, "only canonical 5x"):
                load_five_x_manifest(manifest)

            duplicate = _manifest_row("duplicate")
            _write_jsonl(manifest, [duplicate, dict(duplicate)])
            with self.assertRaisesRegex(ValueError, "Duplicate patch_id"):
                load_five_x_manifest(manifest)

            missing_bbox = _manifest_row("missing_bbox")
            missing_bbox.pop("level0_bbox")
            _write_jsonl(manifest, [missing_bbox])
            with self.assertRaisesRegex(ValueError, "lacks"):
                load_five_x_manifest(manifest)

        with self.assertRaisesRegex(ValueError, "canonical architecture input must be 5x"):
            replace(_ssl_prediction(), scale=10.0)

    def test_ledger_snapshots_are_immutable_and_jsonl_is_replayable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger" / "evidence.jsonl"
            ledger = EvidenceLedger("case_001", jsonl_path=ledger_path)
            first = _evidence("E001", "serrated_architecture")
            second = _evidence("E002", "basal_crypt_dilation")

            snapshot_v1 = ledger.append(first, created_after_action_id="ACTION_001")
            self.assertEqual(snapshot_v1.snapshot_id, "ledger_v000001")
            snapshot_v1.records[0].metadata["fixture"] = "mutated outside ledger"

            snapshot_v2 = ledger.append(second, created_after_action_id="ACTION_002")
            self.assertEqual(snapshot_v2.snapshot_id, "ledger_v000002")
            self.assertEqual(snapshot_v1.evidence_ids, ("E001",))
            self.assertEqual(snapshot_v2.evidence_ids, ("E001", "E002"))
            self.assertIs(ledger.snapshot().records[0].metadata["fixture"], True)

            replayed = EvidenceLedger.from_jsonl("case_001", ledger_path)
            replay_snapshot = replayed.snapshot()
            self.assertEqual(replay_snapshot.snapshot_id, snapshot_v2.snapshot_id)
            self.assertEqual(replay_snapshot.evidence_ids, snapshot_v2.evidence_ids)
            self.assertEqual(
                [record.to_dict() for record in replay_snapshot.records],
                [record.to_dict() for record in snapshot_v2.records],
            )

            unchanged = replayed.append(second, created_after_action_id="ACTION_002")
            self.assertEqual(unchanged.snapshot_id, "ledger_v000002")

    def test_missing_and_not_evaluable_are_not_negative_evidence(self):
        ledger = EvidenceLedger("case_001")
        reducer = EvidenceReducer()
        engine = HypothesisEngine(default_knowledge_base())

        empty_view = reducer.reduce(ledger.snapshot())
        empty_ssl = next(item for item in engine.rank(empty_view) if item.hypothesis_id == "H_SSL")
        self.assertIn("basal_crypt_dilation", empty_ssl.missing_required_evidence)
        self.assertEqual(empty_ssl.observed_conflict_evidence, tuple())

        ledger.append(
            _evidence(
                "E_NOT_EVALUABLE",
                "basal_crypt_dilation",
                status="not_evaluable",
                evaluability="not_evaluable",
                confidence=0.98,
                quality=0.0,
            )
        )
        view = reducer.reduce(ledger.snapshot())
        reduced = view.features["basal_crypt_dilation"]
        self.assertEqual(reduced.resolved_status, "not_evaluable")
        self.assertEqual(reduced.absent_evidence_ids, tuple())
        self.assertEqual(reduced.not_evaluable_evidence_ids, ("E_NOT_EVALUABLE",))

        ssl = next(item for item in engine.rank(view) if item.hypothesis_id == "H_SSL")
        self.assertIn("basal_crypt_dilation", ssl.missing_required_evidence)
        self.assertEqual(ssl.observed_conflict_evidence, tuple())

        with self.assertRaisesRegex(ValueError, "absent evidence requires adequate"):
            _evidence(
                "E_BAD_NEGATIVE",
                "basal_crypt_dilation",
                status="absent",
                evaluability="limited",
            )

    def test_twenty_x_roi_is_contained_by_its_ten_x_parent(self):
        candidates = ROIManager().build_candidates(
            (_ssl_prediction(),),
            slide_dimensions=(4096, 4096),
        )
        ten_x = next(item for item in candidates if item.roi_id == "ROI_10X_patch_ssl")
        two_point_five_x = next(item for item in candidates if item.roi_id == "ROI_2P5X_component_1")
        twenty_x_candidates = [
            item
            for item in candidates
            if item.scale == 20.0 and item.source_patch_ids == ("patch_ssl",)
        ]

        self.assertEqual(two_point_five_x.scale, 2.5)
        self.assertEqual(two_point_five_x.roi_semantics, "roi_overview_context")
        self.assertEqual(two_point_five_x.level0_bbox[2] - two_point_five_x.level0_bbox[0], 4096)
        self.assertEqual(ten_x.scale, 10.0)
        self.assertEqual(ten_x.level0_bbox[2] - ten_x.level0_bbox[0], 1024)
        self.assertEqual(len(twenty_x_candidates), 4)
        for twenty_x in twenty_x_candidates:
            self.assertEqual(twenty_x.level0_bbox[2] - twenty_x.level0_bbox[0], 512)
            self.assertGreaterEqual(twenty_x.level0_bbox[0], ten_x.level0_bbox[0])
            self.assertGreaterEqual(twenty_x.level0_bbox[1], ten_x.level0_bbox[1])
            self.assertLessEqual(twenty_x.level0_bbox[2], ten_x.level0_bbox[2])
            self.assertLessEqual(twenty_x.level0_bbox[3], ten_x.level0_bbox[3])
            self.assertEqual(tuple(twenty_x.metadata["parent_10x_bbox"]), ten_x.level0_bbox)

    def test_reviewer_contract_enforces_feature_coverage_and_prompt_neutrality(self):
        registry = default_reviewer_registry()
        knowledge = default_knowledge_base()
        question = next(item for item in knowledge.questions if item.question_id == "Q_SSL_HP_CRYPT_BASE")
        roi = ROICandidate(
            roi_id="ROI_10X_contract",
            slide_id="slide_001",
            level0_bbox=(1536, 1536, 2560, 2560),
            scale=10.0,
            roi_semantics="crypt_architecture_detail",
            candidate_features=("serration_to_crypt_base", "basal_crypt_dilation"),
            allowed_reviewers=("SerratedArchitectureReviewer",),
            suitability=0.9,
            spatial_coverage=0.8,
            estimated_cost=0.5,
            image_path="roi://ROI_10X_contract",
        )
        action = ReviewerAction(
            action_id="ACTION_CONTRACT",
            question_id=question.question_id,
            reviewer_id=question.reviewer_id,
            task_profile=question.task_profile,
            roi_id=roi.roi_id,
            scale=roi.scale,
            target_features=("serration_to_crypt_base", "basal_crypt_dilation"),
            goal=question.question,
            score=0.9,
            score_components={},
        )
        request = build_reviewer_request(
            snapshot_id="ledger_v000001",
            plan_id="PLAN_001",
            action=action,
            question=question,
            roi=roi,
            registry=registry,
        )
        request_payload = request.to_dict()
        self.assertEqual(
            set(request_payload),
            {
                "schema_version",
                "request_id",
                "action_id",
                "plan_id",
                "question_id",
                "snapshot_id",
                "reviewer",
                "model_input",
            },
        )
        self.assertEqual(
            set(request_payload["model_input"]),
            {"task_profile", "target_features", "primary_roi", "context_views"},
        )
        self.assertEqual(
            set(request_payload["model_input"]["primary_roi"]),
            {
                "roi_id",
                "image_id",
                "image_ref",
                "image_sha256",
                "level0_bbox",
                "magnification",
                "pixel_dimensions",
            },
        )
        serialized_model_input = json.dumps(request.model_input, sort_keys=True)
        for forbidden in registry.BLOCKED_MODEL_INPUT_KEYS:
            self.assertNotIn('"{0}"'.format(forbidden), serialized_model_input)

        backend = ScriptedReviewerBackend(
            {
                question.question_id: [
                    {
                        "serration_to_crypt_base": "present",
                        "basal_crypt_dilation": "present",
                    }
                ]
            }
        )
        observation = backend.invoke(request)
        registry.validate_observation(request, observation)
        observation_payload = observation.to_dict()
        self.assertEqual(
            set(observation_payload),
            {
                "schema_version",
                "observation_id",
                "request_id",
                "reviewer",
                "task_profile",
                "primary_roi_id",
                "primary_magnification",
                "target_features",
                "quality",
                "findings",
                "incidental_findings",
                "overall_evidence_strength",
                "limitations",
                "does_not_decide_final_diagnosis",
            },
        )
        self.assertEqual(
            set(observation_payload["quality"]),
            {
                "overall_evaluability",
                "status_confidence",
                "adequate_for_requested_features",
                "limitations",
            },
        )
        self.assertEqual(
            set(observation_payload["overall_evidence_strength"]),
            {"level", "score"},
        )
        with self.assertRaisesRegex(ReviewerContractError, "unknown or missing"):
            registry.validate_observation(
                request,
                replace(
                    observation,
                    quality=dict(observation.quality, unexpected_field=True),
                ),
            )
        with self.assertRaisesRegex(ReviewerContractError, "prompt_version"):
            registry.validate_observation(
                request,
                replace(observation, prompt_version="wrong-prompt-version"),
            )

        leaked_request = replace(
            request,
            model_input=dict(request.model_input, final_label="SSLD"),
        )
        with self.assertRaisesRegex(ReviewerContractError, "leaks planner state"):
            registry.validate_request(leaked_request)

        incomplete = ReviewerObservation(
            schema_version="reviewer_observation_v1",
            observation_id="OBS_INCOMPLETE",
            request_id=request.request_id,
            reviewer=request.reviewer,
            task_profile=request.task_profile,
            primary_roi_id=request.primary_roi.roi_id,
            primary_magnification=request.primary_roi.scale,
            target_features=tuple(request.target_features),
            quality={
                "overall_evaluability": "adequate",
                "status_confidence": 0.95,
                "adequate_for_requested_features": True,
                "limitations": [],
            },
            findings=(
                ReviewerFinding(
                    feature_id="serration_to_crypt_base",
                    status="present",
                    status_confidence=0.9,
                    feature_evaluability="adequate",
                    scope="roi_local",
                    evidence_text="Crypt-base serration is visible.",
                ),
            ),
            incidental_findings=tuple(),
            overall_evidence_strength={"level": "strong", "score": 0.9},
            limitations=tuple(),
            does_not_decide_final_diagnosis=True,
            model_version="fixture-v1",
            prompt_version="fixture-v1",
        )
        with self.assertRaisesRegex(ReviewerContractError, "exactly once"):
            registry.validate_observation(request, incomplete)

        with self.assertRaisesRegex(ValueError, "absent finding requires adequate"):
            ReviewerFinding(
                feature_id="basal_crypt_dilation",
                status="absent",
                status_confidence=0.9,
                feature_evaluability="limited",
                scope="roi_local",
                evidence_text="Not seen in a limited crop.",
            )

    def test_scripted_ssl_to_dysplasia_to_ssld_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest = _write_jsonl(
                tmpdir / "five_x_patch_manifest.jsonl",
                [_manifest_row()],
            )
            predictor = ScriptedArchitecturePredictor(
                {
                    "patch_ssl": {
                        "evaluable": 0.95,
                        "architecture": {"serrated": 0.95, "tubular": 0.02, "villous": 0.02},
                        "context": {
                            "normal_mucosa_present": 0.10,
                            "reactive_inflammatory_present": 0.02,
                            "other_pattern_present": 0.02,
                        },
                        "uncertainty": 0.60,
                        "dysplasia_risk": 0.95,
                        "abnormal_epithelial_score": 0.80,
                    }
                }
            )
            architecture_runtime = ArchitectureInferenceRuntime(predictor=predictor)
            reviewer_backend = ScriptedReviewerBackend(
                {
                    "Q_SSL_HP_CRYPT_BASE": [
                        {
                            "serration_to_crypt_base": "present",
                            "basal_crypt_dilation": "present",
                        }
                    ],
                    "Q_DYSPLASIA": [
                        {"high_grade_or_definite_dysplasia": "present"}
                    ],
                }
            )
            cropper = VirtualROICropper()
            orchestrator = AgentFlowOrchestrator(
                reviewer_backend=reviewer_backend,
                cropper=cropper,
                max_actions=4,
            )
            output_dir = tmpdir / "run"

            result = orchestrator.run_from_manifest(
                case_id="case_001",
                five_x_manifest_path=manifest,
                architecture_runtime=architecture_runtime,
                slide_dimensions=(4096, 4096),
                output_dir=output_dir,
            )

            self.assertEqual(
                [plan.selected_question.question_id if plan.selected_question else None for plan in result.plans],
                ["Q_SSL_HP_CRYPT_BASE", "Q_DYSPLASIA", None],
            )
            self.assertEqual(
                [plan.selected_action.scale for plan in result.plans if plan.selected_action],
                [10.0, 20.0],
            )
            self.assertEqual(
                [plan.snapshot_id for plan in result.plans],
                ["ledger_v000001", "ledger_v000002", "ledger_v000003"],
            )
            self.assertTrue(result.plans[-1].stop)
            self.assertEqual(result.plans[-1].stop_reason, "diagnostic_ready")
            self.assertEqual(result.plans[-1].ranked_hypotheses[0].hypothesis_id, "H_SSL")
            self.assertEqual(result.plans[-1].ranked_hypotheses[0].dysplasia_state, "supported")

            self.assertEqual(result.chief_decision.status, "final")
            self.assertEqual(result.chief_decision.final_label, "SSLD")
            self.assertEqual(result.chief_decision.dysplasia_state, "supported")
            self.assertTrue(result.chief_decision.non_clinical)
            self.assertTrue(result.synthetic_stub)
            self.assertEqual(len(reviewer_backend.requests), 2)
            self.assertEqual(
                [request.reviewer_id for request in reviewer_backend.requests],
                ["SerratedArchitectureReviewer", "DysplasiaReviewer"],
            )
            self.assertEqual(
                [request.snapshot_id for request in reviewer_backend.requests],
                ["ledger_v000001", "ledger_v000002"],
            )
            self.assertEqual(
                [item["roi_id"] for item in cropper.crops],
                ["ROI_10X_patch_ssl", "ROI_5X_component_1", "ROI_20X_patch_ssl_q00"],
            )
            self.assertEqual(
                [request.model_input["context_views"][0]["roi"]["roi_id"] for request in reviewer_backend.requests],
                ["ROI_5X_component_1", "ROI_10X_patch_ssl"],
            )

            replayed = EvidenceLedger.from_jsonl("case_001", result.ledger_path).snapshot()
            self.assertEqual(replayed.snapshot_id, result.final_snapshot_id)
            self.assertIn(
                "high_grade_or_definite_dysplasia",
                {record.feature for record in replayed.records},
            )
            run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["canonical_architecture_input_scale"], "5x")
            self.assertEqual(run_manifest["final_label"], "SSLD")
            self.assertTrue(run_manifest["synthetic_stub"])
            self.assertTrue(run_manifest["non_clinical"])

            invocation_records = [
                json.loads(line)
                for line in (output_dir / "reviewers" / "invocation_records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([item["record_type"] for item in invocation_records], ["evidence", "evidence"])
            for record in invocation_records:
                self.assertEqual(record["schema_version"], "reviewer_ledger_record_v1")
                self.assertEqual(record["reasoning_use"], {"status": "pending"})
                self.assertEqual(record["observation"]["reviewer"], record["reviewer"])
                self.assertEqual(record["provenance"]["action_id"], record["action_id"])
                self.assertEqual(len(record["provenance"]["image_sha256"]), 64)

    def test_missing_model_ports_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _write_jsonl(
                Path(tmpdir) / "five_x_patch_manifest.jsonl",
                [_manifest_row()],
            )
            with self.assertRaisesRegex(
                ArchitectureModelUnavailableError,
                "checkpoint is not configured",
            ):
                ArchitectureInferenceRuntime().run(manifest)

        with self.assertRaisesRegex(ModelUnavailableError, "not configured"):
            UnavailableReviewerBackend().invoke(None)

    def test_schema_invalid_reviewer_output_becomes_failure_not_evidence(self):
        class InvalidReviewerBackend(object):
            synthetic_stub = True
            model_id = "invalid-scripted-reviewer"
            model_version = "invalid-v1"
            prompt_version = None

            def invoke(self, request):
                return ReviewerObservation(
                    schema_version="reviewer_observation_v1",
                    observation_id="OBS_INVALID",
                    request_id=request.request_id,
                    reviewer=request.reviewer,
                    task_profile=request.task_profile,
                    primary_roi_id=request.primary_roi.roi_id,
                    primary_magnification=request.primary_roi.scale,
                    target_features=tuple(request.target_features),
                    quality={
                        "overall_evaluability": "adequate",
                        "status_confidence": 0.9,
                        "adequate_for_requested_features": True,
                        "limitations": [],
                    },
                    findings=(
                        ReviewerFinding(
                            feature_id=request.target_features[0],
                            status="present",
                            status_confidence=0.9,
                            feature_evaluability="adequate",
                            scope="roi_local",
                            evidence_text="Only one requested feature was returned.",
                        ),
                    ),
                    incidental_findings=tuple(),
                    overall_evidence_strength={"level": "moderate", "score": 0.6},
                    limitations=tuple(),
                    does_not_decide_final_diagnosis=True,
                    model_version=self.model_version,
                    prompt_version=request.prompt_version,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            result = AgentFlowOrchestrator(
                reviewer_backend=InvalidReviewerBackend(),
                cropper=VirtualROICropper(),
                max_actions=1,
            ).run(
                case_id="case_invalid_observation",
                predictions=(_ssl_prediction(),),
                slide_dimensions=(4096, 4096),
                output_dir=output_dir,
            )
            records = [
                json.loads(line)
                for line in (output_dir / "reviewers" / "invocation_records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["record_type"], "invocation_failure")
            self.assertEqual(records[0]["failure"]["stage"], "schema_validation")
            self.assertEqual(records[0]["failure"]["attempt_count"], 1)
            replayed = EvidenceLedger.from_jsonl(
                "case_invalid_observation",
                result.ledger_path,
            ).snapshot()
            self.assertIn(
                "reviewer_invocation_failure",
                {record.evidence_type for record in replayed.records},
            )
            self.assertNotIn(
                "serration_to_crypt_base",
                {
                    record.feature
                    for record in replayed.records
                    if record.evidence_type == "reviewer_evidence"
                },
            )


if __name__ == "__main__":
    unittest.main()
