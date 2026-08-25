import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adenoma_agent.multimodal import StageBackendChain


class LocalCPathAgentQwenBackendTest(unittest.TestCase):
    def test_local_cpathagent_qwen_runner_invokes_navigate_stage(self):
        bundle = {
            "runtime": {
                "paths": {
                    "cpathagent_qwen_python": "/data1/yuexin/.conda/envs/patho-r1/bin/python",
                    "cpathagent_qwen_runner_script": "/data1/yuexin/Adenoma/scripts/run_cpathagent_qwen.py",
                },
                "models": {
                    "cpathagent_qwen_model_id": "Qwen/Qwen3-14B",
                    "cpathagent_qwen_vision_encoder_id": "CPath-CLIP-compatible",
                    "cpathagent_qwen_adapter_path": "/tmp/cpathagent_adapter",
                    "cpathagent_qwen_cache_dir": "/tmp/cpathagent_cache",
                },
                "execution": {"cpathagent_qwen_cuda_visible_devices": ""},
                "backends": {
                    "local_cpathagent_qwen": {"enabled": True, "max_new_tokens": 256, "shim_mode": "heuristic"}
                },
                "navigate": {"overlap_threshold": 0.05},
                "observe": {
                    "serrated_criteria": [],
                    "abnormal_crypt_criteria": [],
                    "conventional_adenoma_criteria": [],
                    "dysplasia_criteria": [],
                },
                "trace": {"labels": ["background_artifact_stroma", "normal_mucosa", "conventional_adenoma_like", "inflammatory_polyp_like", "ssl_suspicious_mucosa"]},
            },
            "budget": {
                "max_navigation_steps": 4,
                "magnification_to_region_size": {"5.0": 2048, "20.0": 512},
            },
        }
        chain = StageBackendChain(bundle)
        request = {
            "images": [],
            "prompt": {"question": "plan navigation", "task": "cpathagent_navigation_planning"},
            "metadata": {
                "slide_dimensions_level0": [4000, 4000],
                "clusters": [
                    {
                        "cluster_id": "cluster_ssl",
                        "l": "ssl_suspicious_mucosa",
                        "s": 5,
                        "d": True,
                        "cluster_bbox_level0": {"x1": 400, "y1": 400, "x2": 600, "y2": 600},
                        "patches_level0": [{"patch_id": [0, 0], "x1": 400, "y1": 400, "x2": 600, "y2": 600}],
                    }
                ],
            },
        }
        response = chain.invoke("navigate", ["local_cpathagent_qwen"], request)
        self.assertEqual(response["backend"], "local_cpathagent_qwen")
        self.assertTrue(response["output"]["steps"])
        self.assertEqual(response["output"]["steps"][0]["metadata"]["source_group_id"], "cluster_ssl")
        self.assertEqual(response["output"]["runner_metadata"]["start_mode"], "cold_start_fallback")

    def test_local_cpathagent_qwen_uses_trace_stage_token_override(self):
        bundle = {
            "runtime": {
                "backends": {
                    "local_cpathagent_qwen": {
                        "enabled": True,
                        "max_new_tokens": 1024,
                        "shim_mode": "server",
                        "server_url": "http://127.0.0.1:8200/predict",
                        "timeout_seconds": 30,
                    }
                },
                "trace": {
                    "qwen_max_new_tokens": 4096,
                    "labels": [
                        "background_artifact_stroma",
                        "normal_mucosa",
                        "conventional_adenoma_like",
                        "inflammatory_polyp_like",
                        "ssl_suspicious_mucosa",
                    ],
                },
            },
            "budget": {},
        }
        chain = StageBackendChain(bundle)
        request = {
            "images": ["/tmp/fake_grid.jpg"],
            "prompt": {"question": "trace", "task": "ssl_others_dual_branch_trace_annotation"},
            "metadata": {
                "case_id": "case_001",
                "thumbnail_meta": {"thumbnail_size": [100, 100], "slide_dimensions_level0": [1000, 1000]},
                "proposals": [],
                "selector_mode": "grid_input",
            },
        }

        class _FakeResponse(object):
            status_code = 200
            text = ""

            def json(self):
                return {
                    "text": "{\"patches\":[]}",
                    "request_id": "req-1",
                    "gpu_device_id": 0,
                    "cuda_visible_devices": "0",
                    "model_id": "qwen",
                    "adapter_path": "adapter",
                }

        captured = {}

        def _fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return _FakeResponse()

        with patch("requests.post", side_effect=_fake_post):
            with self.assertRaises(Exception):
                chain.invoke("trace", ["local_cpathagent_qwen"], request)
        self.assertEqual(captured["json"]["max_new_tokens"], 4096)

    def test_local_cpathagent_qwen_parser_keeps_multiple_intra_cell_zoom_targets(self):
        bundle = {
            "runtime": {
                "backends": {
                    "local_cpathagent_qwen": {
                        "enabled": True,
                        "shim_mode": "server",
                        "server_url": "http://127.0.0.1:8200/predict",
                        "timeout_seconds": 30,
                    }
                },
                "navigate": {"overlap_threshold": 0.05},
            },
            "budget": {
                "magnification_to_region_size": {"5.0": 2048, "10.0": 1024},
            },
        }
        chain = StageBackendChain(bundle)
        request = {
            "images": ["/tmp/fake_grid.jpg"],
            "prompt": {"question": "plan navigation", "task": "cpathagent_navigation_planning"},
            "metadata": {
                "slide_dimensions_level0": [10000, 10000],
                "clusters": [
                    {
                        "cluster_id": "cluster_ssl",
                        "l": "ssl_suspicious_mucosa",
                        "s": 5,
                        "d": True,
                        "cluster_bbox_level0": {"x1": 2000, "y1": 2000, "x2": 6000, "y2": 6000},
                        "patches_level0": [
                            {"patch_id": [0, 0], "x1": 2000, "y1": 2000, "x2": 6000, "y2": 6000}
                        ],
                        "metadata": {"workflow_branch": "serrated"},
                    }
                ],
            },
        }

        class _FakeResponse(object):
            status_code = 200
            text = ""

            def json(self):
                return {
                    "text": json.dumps(
                        {
                            "steps": [
                                {
                                    "source_group_id": "cluster_ssl",
                                    "patch_id": [0, 0],
                                    "x": 4000,
                                    "y": 4000,
                                    "m": 5.0,
                                    "region_size_level0": 2048,
                                    "need_to_see": "overview",
                                    "review_goal": "serrated_lesion_assessment",
                                    "stage_gate": "mucosa_or_serrated",
                                },
                                {
                                    "source_group_id": "cluster_ssl",
                                    "patch_id": [0, 0],
                                    "x": 3000,
                                    "y": 3000,
                                    "m": 10.0,
                                    "region_size_level0": 1024,
                                    "need_to_see": "crypt base",
                                    "review_goal": "abnormal_crypt_assessment",
                                    "stage_gate": "abnormal_crypt",
                                },
                                {
                                    "source_group_id": "cluster_ssl",
                                    "patch_id": [0, 0],
                                    "x": 5000,
                                    "y": 5000,
                                    "m": 10.0,
                                    "region_size_level0": 1024,
                                    "need_to_see": "serrated edge",
                                    "review_goal": "abnormal_crypt_assessment",
                                    "stage_gate": "abnormal_crypt",
                                },
                            ]
                        }
                    ),
                    "request_id": "req-1",
                }

        with patch("requests.post", return_value=_FakeResponse()):
            response = chain.invoke("navigate", ["local_cpathagent_qwen"], request)

        inspect_steps = [step for step in response["output"]["steps"] if step["metadata"].get("action") == "inspect"]
        zoom_steps = [step for step in inspect_steps if step["m"] == 10.0]
        self.assertEqual(len(zoom_steps), 2)
        self.assertEqual([step["metadata"]["intra_cell_target_index"] for step in zoom_steps], [0, 1])
        self.assertEqual({step["metadata"]["intra_cell_target_count"] for step in zoom_steps}, {2})
        self.assertEqual({step["metadata"]["coordinate_source"] for step in zoom_steps}, {"model_proposed"})

    def test_local_cpathagent_qwen_parser_deduplicates_overlapping_zoom_targets(self):
        bundle = {
            "runtime": {
                "backends": {
                    "local_cpathagent_qwen": {
                        "enabled": True,
                        "shim_mode": "server",
                        "server_url": "http://127.0.0.1:8200/predict",
                        "timeout_seconds": 30,
                    }
                },
                "navigate": {"overlap_threshold": 0.80},
            },
            "budget": {
                "magnification_to_region_size": {"5.0": 2048, "10.0": 1024},
            },
        }
        chain = StageBackendChain(bundle)
        request = {
            "images": ["/tmp/fake_grid.jpg"],
            "prompt": {"question": "plan navigation", "task": "cpathagent_navigation_planning"},
            "metadata": {
                "slide_dimensions_level0": [10000, 10000],
                "clusters": [
                    {
                        "cluster_id": "cluster_ssl",
                        "l": "ssl_suspicious_mucosa",
                        "s": 5,
                        "d": True,
                        "cluster_bbox_level0": {"x1": 2000, "y1": 2000, "x2": 6000, "y2": 6000},
                        "patches_level0": [
                            {"patch_id": [0, 0], "x1": 2000, "y1": 2000, "x2": 6000, "y2": 6000}
                        ],
                        "metadata": {"workflow_branch": "serrated"},
                    }
                ],
            },
        }

        class _FakeResponse(object):
            status_code = 200
            text = ""

            def json(self):
                return {
                    "text": json.dumps(
                        {
                            "steps": [
                                {
                                    "source_group_id": "cluster_ssl",
                                    "patch_id": [0, 0],
                                    "x": 3000,
                                    "y": 3000,
                                    "m": 10.0,
                                    "region_size_level0": 1024,
                                    "need_to_see": "crypt base",
                                    "review_goal": "abnormal_crypt_assessment",
                                    "stage_gate": "abnormal_crypt",
                                },
                                {
                                    "source_group_id": "cluster_ssl",
                                    "patch_id": [0, 0],
                                    "x": 3020,
                                    "y": 3020,
                                    "m": 10.0,
                                    "region_size_level0": 1024,
                                    "need_to_see": "duplicate crypt base",
                                    "review_goal": "abnormal_crypt_assessment",
                                    "stage_gate": "abnormal_crypt",
                                },
                            ]
                        }
                    ),
                    "request_id": "req-1",
                }

        with patch("requests.post", return_value=_FakeResponse()):
            response = chain.invoke("navigate", ["local_cpathagent_qwen"], request)

        zoom_steps = [step for step in response["output"]["steps"] if step["metadata"].get("action") == "inspect" and step["m"] == 10.0]
        self.assertEqual(len(zoom_steps), 1)

    def test_local_cpathagent_qwen_repairs_observe_report_missing_checklists(self):
        bundle = {
            "runtime": {
                "backends": {
                    "local_cpathagent_qwen": {
                        "enabled": True,
                        "shim_mode": "server",
                        "server_url": "http://127.0.0.1:8200/predict",
                        "timeout_seconds": 30,
                    }
                },
                "chief_llm": {"server_url": "http://127.0.0.1:8300/predict", "timeout_seconds": 30},
                "output_repair": {
                    "enabled": True,
                    "backend": "deepseek_chief_http",
                    "stages": ["observe_report"],
                    "allow_light_clinical_fill": True,
                    "max_repair_attempts": 1,
                },
            },
            "budget": {},
        }
        chain = StageBackendChain(bundle)
        request = {
            "images": [],
            "prompt": {"question": "write final report", "task": "cpathagent_observe_report"},
            "metadata": {
                "case_id": "case_repair",
                "records": [
                    {
                        "step_id": "step_00",
                        "stage_decision": "continue",
                        "reasoning": "serrated architecture was not assessable in the available crop",
                    }
                ],
                "global_reviews": [],
            },
        }

        class _FakeResponse(object):
            status_code = 200
            text = ""

            def __init__(self, payload):
                self._payload = payload

            def json(self):
                return self._payload

        qwen_payload = {
            "text": json.dumps(
                {
                    "hierarchical_prediction": {"primary_pattern": "insufficient"},
                    "integrated_report": "Insufficient evidence for a definitive serrated diagnosis.",
                }
            ),
            "request_id": "qwen-1",
        }
        repaired_report = {
            "hierarchical_prediction": {"primary_pattern": "insufficient"},
            "serrated_checklist": {"overall_status": "not_assessed", "evidence": []},
            "abnormal_crypt_checklist": {"overall_status": "not_assessed", "evidence": []},
            "conventional_adenoma_checklist": {"overall_status": "not_assessed", "evidence": []},
            "serrated_dysplasia_checklist": {"overall_status": "not_assessed", "evidence": []},
            "conventional_dysplasia_checklist": {"overall_status": "not_assessed", "evidence": []},
            "dysplasia_checklist": {"overall_status": "not_assessed", "evidence": []},
            "integrated_report": "Insufficient evidence for a definitive serrated diagnosis.",
        }
        repair_payload = {
            "repaired_json": repaired_report,
            "repair_actions": ["filled_missing_checklist_with_not_assessed"],
            "unrecoverable_errors": [],
            "confidence": 0.9,
            "request_id": "repair-1",
            "model_name": "deepseek-r1-distill-qwen-14b",
        }

        def _fake_post(url, json=None, timeout=None):
            if url.endswith("/repair_output"):
                return _FakeResponse(repair_payload)
            return _FakeResponse(qwen_payload)

        with patch("requests.post", side_effect=_fake_post):
            response = chain.invoke("observe_report", ["local_cpathagent_qwen"], request)

        output = response["output"]
        self.assertEqual(output["serrated_checklist"]["overall_status"], "not_assessed")
        self.assertEqual(output["repair_metadata"]["repair_source"], "deepseek_output_repair")
        self.assertEqual(response["attempts"][0]["backend"], "deepseek_output_repair")
        self.assertEqual(response["attempts"][0]["status"], "ok")
        self.assertEqual(output["runner_metadata"]["repair_status"], "repaired")

    def test_local_cpathagent_qwen_failed_repair_falls_back_to_heuristic(self):
        bundle = {
            "runtime": {
                "backends": {
                    "local_cpathagent_qwen": {
                        "enabled": True,
                        "shim_mode": "server",
                        "server_url": "http://127.0.0.1:8200/predict",
                        "timeout_seconds": 30,
                    }
                },
                "chief_llm": {"server_url": "http://127.0.0.1:8300/predict", "timeout_seconds": 30},
                "output_repair": {"enabled": True, "stages": ["navigate"]},
                "navigate": {"overlap_threshold": 0.05},
                "trace": {"labels": ["ssl_suspicious_mucosa"]},
            },
            "budget": {"max_navigation_steps": 2, "magnification_to_region_size": {"5.0": 2048, "10.0": 1024}},
        }
        chain = StageBackendChain(bundle)
        request = {
            "images": ["/tmp/fake_grid.jpg"],
            "prompt": {"question": "plan navigation", "task": "cpathagent_navigation_planning"},
            "metadata": {
                "slide_dimensions_level0": [4000, 4000],
                "clusters": [
                    {
                        "cluster_id": "cluster_ssl",
                        "l": "ssl_suspicious_mucosa",
                        "s": 5,
                        "d": True,
                        "cluster_bbox_level0": {"x1": 400, "y1": 400, "x2": 600, "y2": 600},
                        "patches_level0": [{"patch_id": [0, 0], "x1": 400, "y1": 400, "x2": 600, "y2": 600}],
                    }
                ],
            },
        }

        class _FakeResponse(object):
            status_code = 200
            text = ""

            def __init__(self, payload):
                self._payload = payload

            def json(self):
                return self._payload

        def _fake_post(url, json=None, timeout=None):
            if url.endswith("/repair_output"):
                return _FakeResponse(
                    {
                        "repaired_json": {},
                        "repair_actions": [],
                        "unrecoverable_errors": ["no JSON object could be recovered"],
                        "confidence": 0.0,
                    }
                )
            return _FakeResponse({"text": "not json", "request_id": "qwen-bad"})

        with patch("requests.post", side_effect=_fake_post):
            response = chain.invoke("navigate", ["local_cpathagent_qwen", "heuristic"], request)

        self.assertEqual(response["backend"], "heuristic")
        self.assertEqual(response["attempts"][0]["backend"], "deepseek_output_repair")
        self.assertEqual(response["attempts"][0]["status"], "unrecoverable")
        self.assertEqual(response["attempts"][1]["backend"], "local_cpathagent_qwen")
        self.assertEqual(response["attempts"][-1]["backend"], "heuristic")


if __name__ == "__main__":
    unittest.main()
