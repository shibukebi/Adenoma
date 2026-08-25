import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from adenoma_agent.agents.observe_reason import ObserveReasonAgent
from adenoma_agent.schemas import CaseSpec, GlobalReviewRecord, ObservationRecord, TraceCluster, NavigationStep


class _FakeResponse(object):
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class ChiefHttpIntegrationTest(unittest.TestCase):
    def _agent(self, url, cache_root):
        bundle = {
            "runtime": {
                "cache": {"description_cache_root": cache_root},
                "chief_llm": {
                    "server_url": url,
                    "timeout_seconds": 10,
                    "model_name": "DeepSeek-R1-Distill-Qwen-32B",
                    "require_real_chief": True,
                },
            }
        }
        return ObserveReasonAgent(bundle=bundle, cropper_adapter=None, backend_chain=None)

    def _case_inputs(self):
        case_spec = CaseSpec(case_id="case_smoke", slide_path="/tmp/fake.svs", task_type="test", question="test")
        step = NavigationStep(
            step_id="step_00",
            x=100,
            y=120,
            m=5.0,
            region_size_level0=2048,
            need_to_see="ssl_suspicious_mucosa",
            review_goal="serrated_lesion_assessment",
            stage_gate="mucosa_or_serrated",
            metadata={"cluster_id": "cluster_ssl", "cell_id": "cell_0_0", "cell_priority": 4, "patch_id": [0, 0], "cluster_label": "ssl_suspicious_mucosa", "workflow_branch": "serrated"},
        )
        record = ObservationRecord(
            step_id="step_00",
            crop_path="/tmp/step_00.png",
            observation="Overview supports the serrated branch.",
            reasoning="Low-power morphology supports serrated lesion context.",
            next_step="Inspect abnormal crypt architecture.",
            level_1_findings=["serrated_lesion_context"],
            level_2_findings=[],
            level_3_findings=[],
            stage_decision="supports_serrated_lesion",
            confidence=0.8,
            metadata={"cluster_id": "cluster_ssl", "cell_id": "cell_0_0", "cell_priority": 4, "patch_id": [0, 0], "cluster_label": "ssl_suspicious_mucosa", "workflow_branch": "serrated", "review_goal": "serrated_lesion_assessment"},
        )
        trace_cluster = TraceCluster(
            cluster_id="cluster_ssl",
            cluster_bbox_thumb={},
            cluster_bbox_level0={},
            regions_thumb=[],
            regions_level0=[],
            l="ssl_suspicious_mucosa",
            s=4,
            d=True,
            review_stage="serrated_screening",
            crypt_disorder_risk=2,
            dysplasia_review_needed=False,
            desc="SSL-like cluster",
            metadata={"workflow_branch": "serrated"},
        )
        return case_spec, step, record, trace_cluster

    def test_call_chief_global_review_parses_http_response(self):
        response_payload = {
            "review_id": "global_review_0000",
            "source_step_id": "step_00",
            "decision": "continue",
            "continue_reason": "Need high-magnification crypt confirmation before stopping.",
            "chief_confidence": 0.78,
            "resolved_branch_state": {
                "serrated": "supported",
                "abnormal_crypt": "unresolved",
                "conventional": "opposed",
                "dysplasia": "unresolved",
            },
            "sufficient_evidence": [],
            "unresolved_questions": ["Need abnormal crypt confirmation."],
            "next_visual_target": {
                "target_cluster_id": "cluster_ssl",
                "target_branch": "serrated",
                "target_region_semantic": "ssl_suspicious_mucosa",
                "target_morphology_prompt": ["look for basal crypt dilatation"],
                "preferred_magnification": 10.0,
                "priority_reason": "Resolve remaining abnormal-crypt uncertainty.",
            },
            "branch_correction_reason": "",
            "request_id": "req-chief-smoke",
            "model_name": "DeepSeek-R1-Distill-Qwen-32B",
            "review_source": "chief_model",
            "raw_generated_text": "{\"decision\":\"continue\"}",
            "thought_text": "reasoning",
            "answer_candidate_text": "{\"decision\":\"continue\"}",
            "parse_error": "",
            "gpu_device_id": 0,
            "cuda_visible_devices": "0",
            "round_trip_ms": 1234,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._agent("http://127.0.0.1:8100/predict", tmpdir)
            case_spec, step, record, trace_cluster = self._case_inputs()
            with patch("adenoma_agent.agents.observe_reason.requests.post", return_value=_FakeResponse(200, response_payload)) as mocked_post:
                result = agent._call_chief_global_review(
                    case_spec=case_spec,
                    step=step,
                    record=record,
                    records=[record],
                    global_reviews=[],
                    trace_result={"clusters": [trace_cluster]},
                    pending_steps=[],
                    observe_dir=tmpdir,
                )
            self.assertEqual(result.review_id, "global_review_0000")
            self.assertEqual(result.decision, "continue")
            self.assertEqual(result.next_visual_target["preferred_magnification"], 10.0)
            self.assertEqual(result.metadata["chief_model_name"], "DeepSeek-R1-Distill-Qwen-32B")
            self.assertEqual(result.metadata["chief_request_id"], "req-chief-smoke")
            self.assertEqual(result.metadata["chief_gpu_device_id"], 0)
            self.assertEqual(result.metadata["start_mode"], "warm_start")
            self.assertEqual(result.metadata["review_source"], "chief_model")
            self.assertTrue(Path(result.metadata["chief_debug_response_json"]).exists())
            self.assertTrue(Path(result.metadata["chief_debug_raw_text"]).exists())
            self.assertTrue(Path(result.metadata["chief_debug_response_json"]).with_name("step_00_chief_thought_text.txt").exists())
            self.assertTrue(Path(result.metadata["chief_debug_response_json"]).with_name("step_00_chief_answer_candidate.txt").exists())
            _, kwargs = mocked_post.call_args
            self.assertEqual(kwargs["timeout"], 10)
            self.assertEqual(kwargs["json"]["case_id"], "case_smoke")
            self.assertEqual(kwargs["json"]["step"]["step_id"], "step_00")

    def test_call_chief_global_review_retargets_illegal_cluster_to_pending_step(self):
        response_payload = {
            "review_id": "global_review_0000",
            "source_step_id": "step_00",
            "decision": "continue",
            "continue_reason": "Need another target.",
            "chief_confidence": 0.78,
            "resolved_branch_state": {
                "serrated": "supported",
                "abnormal_crypt": "maybe",
                "conventional": "opposed",
                "dysplasia": "unresolved",
            },
            "sufficient_evidence": [],
            "unresolved_questions": ["Need abnormal crypt confirmation."],
            "next_visual_target": {
                "target_cluster_id": "cluster_not_real",
                "target_branch": "serrated",
                "target_region_semantic": "ssl_suspicious_mucosa",
                "target_morphology_prompt": "look for basal crypt dilatation",
                "preferred_magnification": 20.0,
            },
            "branch_correction_reason": "",
            "request_id": "req-chief-smoke",
            "review_source": "chief_model",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._agent("http://127.0.0.1:8100/predict", tmpdir)
            case_spec, step, record, trace_cluster = self._case_inputs()
            pending_step = NavigationStep(
                step_id="step_01",
                x=120,
                y=140,
                m=10.0,
                region_size_level0=1024,
                need_to_see="ssl_suspicious_mucosa",
                review_goal="abnormal crypt confirmation",
                stage_gate="abnormal_crypt",
                metadata={
                    "cluster_id": "cluster_pending",
                    "cell_id": "cell_0_1",
                    "patch_id": [0, 1],
                    "cluster_label": "ssl_suspicious_mucosa",
                    "workflow_branch": "serrated",
                },
            )
            with patch("adenoma_agent.agents.observe_reason.requests.post", return_value=_FakeResponse(200, response_payload)):
                result = agent._call_chief_global_review(
                    case_spec=case_spec,
                    step=step,
                    record=record,
                    records=[record],
                    global_reviews=[],
                    trace_result={"clusters": [trace_cluster]},
                    pending_steps=[pending_step],
                    observe_dir=tmpdir,
                )

            self.assertEqual(result.next_visual_target["target_cluster_id"], "cluster_pending")
            self.assertEqual(result.next_visual_target["preferred_magnification"], 10.0)
            self.assertIn("chief_invalid_target_cluster_retargeted", result.metadata["normalization_actions"])

    def test_call_chief_global_review_raises_on_http_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._agent("http://127.0.0.1:8100/predict", tmpdir)
            case_spec, step, record, _ = self._case_inputs()
            with patch("adenoma_agent.agents.observe_reason.requests.post", return_value=_FakeResponse(500, {"detail": "failure"})):
                with self.assertRaises(RuntimeError):
                    agent._call_chief_global_review(
                        case_spec=case_spec,
                        step=step,
                        record=record,
                        records=[record],
                        global_reviews=[],
                        trace_result={"clusters": []},
                        pending_steps=[],
                        observe_dir=tmpdir,
                    )

    def test_call_chief_global_review_rejects_fallback_review(self):
        response_payload = {
            "review_id": "global_review_fallback",
            "source_step_id": "step_00",
            "decision": "continue",
            "continue_reason": "fallback",
            "chief_confidence": 0.2,
            "resolved_branch_state": {
                "serrated": "unresolved",
                "abnormal_crypt": "unresolved",
                "conventional": "unresolved",
                "dysplasia": "unresolved",
            },
            "sufficient_evidence": [],
            "unresolved_questions": [],
            "next_visual_target": {
                "target_cluster_id": "cluster_ssl",
                "target_branch": "serrated",
                "target_region_semantic": "ssl_suspicious_mucosa",
                "target_morphology_prompt": ["look closer"],
                "preferred_magnification": 10.0,
                "priority_reason": "fallback",
            },
            "branch_correction_reason": "",
            "request_id": "req-chief-fallback",
            "model_name": "DeepSeek-R1-Distill-Qwen-32B",
            "review_source": "fallback_rule_based",
            "raw_generated_text": "I think this should continue.",
            "thought_text": "deliberation",
            "answer_candidate_text": "",
            "parse_error": "Chief model response did not contain a JSON object",
            "gpu_device_id": 0,
            "cuda_visible_devices": "0",
            "round_trip_ms": 10,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._agent("http://127.0.0.1:8100/predict", tmpdir)
            case_spec, step, record, trace_cluster = self._case_inputs()
            with patch("adenoma_agent.agents.observe_reason.requests.post", return_value=_FakeResponse(200, response_payload)):
                with self.assertRaises(RuntimeError):
                    agent._call_chief_global_review(
                        case_spec=case_spec,
                        step=step,
                        record=record,
                        records=[record],
                        global_reviews=[],
                        trace_result={"clusters": [trace_cluster]},
                        pending_steps=[],
                        observe_dir=tmpdir,
                    )


class ChiefBatchingIntegrationTest(unittest.TestCase):
    def test_chief_runs_once_after_same_grid_observation_unit(self):
        class _FakeCropper(object):
            def export_crops(self, case_spec, bundle_steps, output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                crops = []
                for item in bundle_steps:
                    role = item.metadata.get("image_role", "detail")
                    image_path = output_dir / "{0}_{1}.png".format(item.step_id, role)
                    Image.new("RGB", (32, 32), color=(180, 120, 150)).save(image_path)
                    crops.append(
                        {
                            "image_path": str(image_path),
                            "m": item.m,
                            "metadata": {"image_role": role},
                        }
                    )
                return {
                    "manifest": {"crops": crops},
                    "result": {"returncode": 0, "stdout": "", "stderr": "", "latency_ms": 1},
                }

        class _FakeBackend(object):
            def __init__(self):
                self.observe_step_calls = 0

            def invoke(self, stage, chain_names, request):
                if stage == "observe_step":
                    self.observe_step_calls += 1
                    if self.observe_step_calls == 1:
                        return {
                            "backend": "fake",
                            "attempts": [],
                            "output": {
                                "observation": "5x review",
                                "reasoning": "Supports serrated lesion.",
                                "next_step": "Go high mag.",
                                "level_1_findings": ["serrated_lesion_context"],
                                "level_2_findings": [],
                                "level_3_findings": [],
                                "stage_decision": "supports_serrated_lesion",
                                "confidence": 0.8,
                                "view_count": 2,
                                "serrated_hits": {},
                                "abnormal_crypt_hits": {},
                                "conventional_hits": {},
                                "serrated_dysplasia_hits": {},
                                "conventional_dysplasia_hits": {},
                                "dysplasia_hits": {},
                            },
                        }
                    return {
                        "backend": "fake",
                        "attempts": [],
                        "output": {
                            "observation": "10x review",
                            "reasoning": "Supports abnormal crypt.",
                            "next_step": "Consider dysplasia.",
                            "level_1_findings": [],
                            "level_2_findings": ["basal_dilatation"],
                            "level_3_findings": [],
                            "stage_decision": "supports_abnormal_crypt",
                            "confidence": 0.85,
                            "view_count": 2,
                            "serrated_hits": {},
                            "abnormal_crypt_hits": {},
                            "conventional_hits": {},
                            "serrated_dysplasia_hits": {},
                            "conventional_dysplasia_hits": {},
                            "dysplasia_hits": {},
                        },
                    }
                if stage == "observe_report":
                    return {
                        "backend": "fake",
                        "attempts": [],
                        "output": {
                            "hierarchical_prediction": {"final_case_assessment": {"label": "SSL"}},
                            "serrated_checklist": {},
                            "abnormal_crypt_checklist": {},
                            "conventional_adenoma_checklist": {},
                            "serrated_dysplasia_checklist": {},
                            "conventional_dysplasia_checklist": {},
                            "dysplasia_checklist": {},
                            "integrated_report": "report",
                        },
                    }
                raise AssertionError("unexpected stage")

        class _Logger(object):
            def log(self, *args, **kwargs):
                return None

        bundle = {
            "runtime": {
                "cache": {"description_cache_root": tempfile.mkdtemp()},
                "observe": {
                    "backend_chain": ["fake"],
                    "patho_r1_question": "q",
                },
                "chief_llm": {
                    "server_url": "http://127.0.0.1:8100/predict",
                    "timeout_seconds": 10,
                    "model_name": "chief",
                    "require_real_chief": True,
                },
            },
            "budget": {
                "magnification_to_region_size": {"2.5": 4096, "5.0": 2048, "10.0": 1024},
            },
        }
        agent = ObserveReasonAgent(bundle=bundle, cropper_adapter=_FakeCropper(), backend_chain=_FakeBackend())
        chief_calls = []

        def _fake_chief(case_spec, step, record, records, global_reviews, trace_result, pending_steps, observe_dir):
            chief_calls.append(step.step_id)
            return GlobalReviewRecord(
                review_id="global_review_0000",
                source_step_id=record.step_id,
                decision="early_stop",
                continue_reason="",
                chief_confidence=0.9,
                resolved_branch_state={
                    "serrated": "supported",
                    "abnormal_crypt": "supported",
                    "conventional": "opposed",
                    "dysplasia": "unresolved",
                },
                sufficient_evidence=["enough"],
                unresolved_questions=[],
                next_visual_target=None,
                branch_correction_reason="",
                metadata={"review_source": "chief_model"},
            )

        agent._call_chief_global_review = _fake_chief
        case_spec = CaseSpec(case_id="case_batch", slide_path="/tmp/fake.svs", task_type="test", question="test")
        trace_cluster = TraceCluster(
            cluster_id="cluster_ssl",
            cluster_bbox_thumb={},
            cluster_bbox_level0={},
            regions_thumb=[],
            regions_level0=[],
            l="ssl_suspicious_mucosa",
            s=4,
            d=True,
            review_stage="serrated_screening",
            crypt_disorder_risk=2,
            dysplasia_review_needed=False,
            desc="SSL-like cluster",
            metadata={"workflow_branch": "serrated"},
        )
        navigation_steps = [
            NavigationStep(
                step_id="step_00",
                x=100,
                y=100,
                m=5.0,
                region_size_level0=2048,
                need_to_see="ssl",
                review_goal="serrated_lesion_assessment",
                stage_gate="mucosa_or_serrated",
                metadata={"cluster_id": "cluster_ssl", "cell_id": "cell_0_0", "cell_priority": 4, "patch_id": [0, 0], "cluster_label": "ssl_suspicious_mucosa", "workflow_branch": "serrated", "action": "inspect"},
            ),
            NavigationStep(
                step_id="step_01",
                x=100,
                y=100,
                m=10.0,
                region_size_level0=1024,
                need_to_see="10x zoom-in for abnormal crypt assessment",
                review_goal="abnormal_crypt_assessment",
                stage_gate="abnormal_crypt",
                metadata={"cluster_id": "cluster_ssl", "cell_id": "cell_0_0", "cell_priority": 4, "patch_id": [0, 0], "cluster_label": "ssl_suspicious_mucosa", "workflow_branch": "serrated", "action": "inspect"},
            ),
            NavigationStep(
                step_id="step_02",
                x=140,
                y=140,
                m=10.0,
                region_size_level0=1024,
                need_to_see="second 10x zoom-in for abnormal crypt assessment",
                review_goal="abnormal_crypt_assessment",
                stage_gate="abnormal_crypt",
                metadata={"cluster_id": "cluster_ssl", "cell_id": "cell_0_0", "cell_priority": 4, "patch_id": [0, 0], "cluster_label": "ssl_suspicious_mucosa", "workflow_branch": "serrated", "action": "inspect", "intra_cell_target_index": 1, "intra_cell_target_count": 2},
            ),
            NavigationStep(
                step_id="step_03",
                x=100,
                y=100,
                m=5.0,
                region_size_level0=2048,
                need_to_see="stop",
                review_goal="integrated_impression",
                stage_gate="end",
                metadata={"action": "stop"},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = agent.run(
                case_spec=case_spec,
                trace_result={"clusters": [trace_cluster]},
                navigation_result={"steps": navigation_steps},
                case_dir=tmpdir,
                logger=_Logger(),
            )
        self.assertEqual(chief_calls, ["step_02"])
        self.assertEqual(len(result["records"]), 4)
        self.assertEqual(len(result["global_reviews"]), 1)


if __name__ == "__main__":
    unittest.main()
