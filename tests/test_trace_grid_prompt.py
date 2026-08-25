import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from adenoma_agent.agents.trace import TraceAgent
from adenoma_agent.multimodal import (
    HeuristicStageBackend,
    _build_trace_output_from_text,
    _build_trace_patho_r1_prompt,
    _normalize_trace_label,
    _run_trace_grid_with_coverage_retry,
    fuse_trace_patch_assignments_with_conch,
    request_conch_patch_predictions,
)
from adenoma_agent.schemas import CaseSpec
from adenoma_agent.trace_supervision import score_trace_case


class TraceGridPromptTest(unittest.TestCase):
    def _write_grid_metadata(self, root, valid_image=False):
        image_path = Path(root) / "case_001_tissuegrid125_ds32_level2_grid.jpg"
        if valid_image:
            image = Image.new("RGB", (221, 244), color=(240, 210, 210))
            image.paste((170, 120, 160), (0, 0, 125, 125))
            image.paste((110, 80, 100), (96, 0, 221, 125))
            image.paste((230, 220, 220), (0, 119, 125, 244))
            image.paste((70, 60, 60), (96, 119, 221, 244))
            image.save(image_path)
        else:
            image_path.write_bytes(b"fake-image")
        metadata_path = image_path.with_suffix(".json")
        metadata = {
            "thumbnail_mode": "tissue_grid32x_svs",
            "grid_rows": 2,
            "grid_cols": 2,
            "n_selected_cells": 4,
            "grid_cell_size_thumbnail": 125,
            "grid_stride_thumbnail": 119,
            "grid_cells": [
                {
                    "row_id": 0,
                    "col_id": 0,
                    "thumbnail_top_left_x": 0,
                    "thumbnail_top_left_y": 0,
                    "thumbnail_width": 125,
                    "thumbnail_height": 125,
                    "level0_top_left_x": 1000,
                    "level0_top_left_y": 2000,
                    "level0_width": 4000,
                    "level0_height": 4000,
                    "center_in_tissue": True,
                    "tissue_coverage_ratio": 0.8,
                    "is_selected": True,
                },
                {
                    "row_id": 0,
                    "col_id": 1,
                    "thumbnail_top_left_x": 96,
                    "thumbnail_top_left_y": 0,
                    "thumbnail_width": 125,
                    "thumbnail_height": 125,
                    "level0_top_left_x": 4808,
                    "level0_top_left_y": 2000,
                    "level0_width": 4000,
                    "level0_height": 4000,
                    "center_in_tissue": True,
                    "tissue_coverage_ratio": 0.7,
                    "is_selected": True,
                },
                {
                    "row_id": 1,
                    "col_id": 0,
                    "thumbnail_top_left_x": 0,
                    "thumbnail_top_left_y": 119,
                    "thumbnail_width": 125,
                    "thumbnail_height": 125,
                    "level0_top_left_x": 1000,
                    "level0_top_left_y": 5808,
                    "level0_width": 4000,
                    "level0_height": 4000,
                    "center_in_tissue": True,
                    "tissue_coverage_ratio": 0.9,
                    "is_selected": True,
                },
                {
                    "row_id": 1,
                    "col_id": 1,
                    "thumbnail_top_left_x": 96,
                    "thumbnail_top_left_y": 119,
                    "thumbnail_width": 125,
                    "thumbnail_height": 125,
                    "level0_top_left_x": 4808,
                    "level0_top_left_y": 5808,
                    "level0_width": 4000,
                    "level0_height": 4000,
                    "center_in_tissue": True,
                    "tissue_coverage_ratio": 0.6,
                    "is_selected": True,
                },
            ],
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return image_path

    def _build_request(self, image_path):
        return {
            "stage": "trace",
            "images": [str(image_path)],
            "prompt": {
                "question": "You are the Trace Agent in a hierarchical pathology workflow.",
                "task": "mucosa_serrated_abnormal_crypt_trace_annotation",
            },
            "metadata": {
                "case_id": "case_001",
                "thumbnail_meta": {"thumbnail_size": [221, 244], "slide_dimensions_level0": [20000, 20000]},
                "proposals": [],
                "selector_mode": "thumbnail_only",
            },
        }

    def test_build_trace_prompt_uses_grid_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            prompt = _build_trace_patho_r1_prompt(
                self._build_request(image_path),
                {"runtime": {"trace": {"labels": []}}},
            )
            self.assertIn('"patches"', prompt)
            self.assertNotIn('"groups": [', prompt)
            self.assertIn("authoritative structured input source", prompt)
            self.assertIn("Available selected grid cells:", prompt)
            self.assertIn("- patch_id=[0,0]", prompt)
            self.assertIn("Exact allowed patch vocabulary", prompt)
            self.assertIn("[[0,0],[0,1],[1,0],[1,1]]", prompt)
            self.assertIn("Assign every selected patch ID exactly once in the patches array.", prompt)
            self.assertIn("Every patch_id must be an exact copy of one item from the allowed patch vocabulary above.", prompt)
            self.assertIn("missing=[], duplicates=[], out_of_set=[]", prompt)
            self.assertIn('"region_semantic": "background_artifact_stroma"', prompt)
            self.assertIn('"region_semantic": "ssl_suspicious_mucosa"', prompt)
            self.assertNotIn("ssl_high_priority_mucosa", prompt)

    def test_build_trace_output_from_patch_assignments_aggregates_clusters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            request = self._build_request(image_path)
            bundle = {"runtime": {"trace": {"labels": []}}}
            text = json.dumps(
                {
                    "patches": [
                        {
                            "patch_id": [1, 1],
                            "name": "SSL-like mucosa",
                            "region_semantic": "ssl_suspicious_mucosa",
                            "description": "Lower right patch needs review.",
                            "require_high_magnification": True,
                            "severity_reasoning": "Serrated concern.",
                            "diagnostic_priority": 4,
                            "observation_points": ["serrated concern"],
                        },
                        {
                            "patch_id": [0, 0],
                            "name": "SSL-like mucosa",
                            "region_semantic": "ssl_suspicious_mucosa",
                            "description": "Upper left patch has same pattern.",
                            "require_high_magnification": True,
                            "severity_reasoning": "Serrated concern.",
                            "diagnostic_priority": 4,
                            "observation_points": ["lesion edge"],
                        },
                        {
                            "patch_id": [0, 1],
                            "name": "Background/stroma remainder",
                            "region_semantic": "background_artifact_stroma",
                            "description": "Low value.",
                            "require_high_magnification": False,
                            "severity_reasoning": "Discard.",
                            "diagnostic_priority": 0,
                            "observation_points": ["coverage"],
                        },
                        {
                            "patch_id": [1, 0],
                            "name": "Normal mucosa",
                            "region_semantic": "normal_mucosa",
                            "description": "Low-priority mucosa.",
                            "require_high_magnification": False,
                            "severity_reasoning": "Low concern.",
                            "diagnostic_priority": 1,
                            "observation_points": ["benign architecture"],
                        },
                    ]
                }
            )
            output = _build_trace_output_from_text(text, request, bundle)
            self.assertTrue(output["coverage_summary"]["coverage_ok"])
            self.assertEqual(output["coverage_summary"]["final_trace_schema"], "patch_assignments")
            self.assertEqual(output["coverage_summary"]["cluster_aggregation_mode"], "system_from_patch_assignments")
            self.assertEqual(len(output["clusters"]), 4)
            self.assertEqual(output["clusters"][0]["l"], "ssl_suspicious_mucosa")
            self.assertEqual(output["clusters"][0]["patch_ids_ordered"], [[0, 0]])
            self.assertTrue(output["clusters"][0]["metadata"]["patch_assignment_schema"])
            self.assertTrue(output["clusters"][0]["metadata"]["split_by_connectivity"])
            self.assertEqual(output["clusters"][0]["metadata"]["spatial_component_count"], 2)
            self.assertEqual(output["clusters"][1]["l"], "ssl_suspicious_mucosa")
            self.assertEqual(output["clusters"][1]["patch_ids_ordered"], [[1, 1]])
            self.assertTrue(output["clusters"][1]["metadata"]["split_by_connectivity"])
            self.assertIn("all_clusters", output)
            self.assertIn("patch_assignments", output)
            self.assertEqual(len(output["all_clusters"]), 4)
            self.assertEqual(len(output["patch_assignments"]["patches"]), 4)

    def test_trace_supervision_scores_semantic_consistency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            grid_meta = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
            payload = {
                "patches": [
                    {
                        "patch_id": [0, 0],
                        "name": "SSL edge",
                        "region_semantic": "ssl_suspicious_mucosa",
                        "description": "edge only",
                        "require_high_magnification": False,
                        "severity_reasoning": "edge",
                        "diagnostic_priority": 1,
                        "observation_points": [],
                    }
                ]
            }
            score = score_trace_case(payload, grid_meta)
            self.assertFalse(score["structure"]["coverage_ok"])
            self.assertLess(score["semantics"]["semantic_score"], 100)

    def test_fuse_trace_patch_assignments_with_conch_sets_consensus_fields(self):
        payload = {
            "patches": [
                {
                    "patch_id": [0, 0],
                    "region_semantic": "ssl_suspicious_mucosa",
                    "diagnostic_priority": 4,
                    "require_high_magnification": True,
                },
                {
                    "patch_id": [0, 1],
                    "region_semantic": "background_artifact_stroma",
                    "diagnostic_priority": 0,
                    "require_high_magnification": False,
                },
            ]
        }
        fused = fuse_trace_patch_assignments_with_conch(
            payload,
            {"0,0": "ssl_suspicious_mucosa", "0,1": "normal_mucosa"},
        )
        first = fused["patches"][0]
        second = fused["patches"][1]
        self.assertEqual(first["agreement_status"], "strong_agreement")
        self.assertEqual(first["score_origin"], "consensus_fusion")
        self.assertEqual(first["diagnostic_priority"], 5)
        self.assertEqual(second["agreement_status"], "low_value_disagreement")
        self.assertEqual(second["region_semantic"], "normal_mucosa")
        self.assertEqual(second["conch_region_semantic"], "normal_mucosa")
        self.assertEqual(second["pathoreasoner_r1_region_semantic"], "background_artifact_stroma")

    def test_fuse_trace_patch_assignments_with_conch_soft_falls_back_when_missing(self):
        payload = {
            "patches": [
                {
                    "patch_id": [0, 0],
                    "region_semantic": "conventional_adenoma_like",
                    "diagnostic_priority": 3,
                    "require_high_magnification": True,
                }
            ]
        }
        fused = fuse_trace_patch_assignments_with_conch(payload, {})
        patch = fused["patches"][0]
        self.assertEqual(patch["agreement_status"], "single_model_trace")
        self.assertEqual(patch["score_origin"], "normalized_diagnostic_priority")
        self.assertEqual(patch["conch_region_semantic"], "not_available_in_this_run")
        self.assertEqual(patch["pathoreasoner_r1_region_semantic"], "conventional_adenoma_like")

    def test_request_conch_patch_predictions_accepts_minimal_http_response(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "patch.png"
            Image.new("RGB", (32, 32), color=(200, 180, 180)).save(image_path)

            class _FakeResponse(object):
                status_code = 200

                def json(self):
                    return {"label": "ssl_suspicious_mucosa"}

            class _FakeRequests(object):
                Timeout = RuntimeError

                @staticmethod
                def post(url, json=None, timeout=None):
                    return _FakeResponse()

            bundle = {
                "runtime": {
                    "backends": {
                        "local_conch": {
                            "enabled": True,
                            "server_url": "http://127.0.0.1:8200/predict",
                            "timeout_seconds": 3,
                        }
                    },
                    "trace": {"conch_fail_policy": "soft_fail_skip_patch"},
                }
            }
            with mock.patch.dict("sys.modules", {"requests": _FakeRequests}):
                result = request_conch_patch_predictions(
                    [{"patch_id": [0, 0], "image_path": str(image_path)}],
                    bundle,
                )
        self.assertEqual(result["predictions"]["0,0"], "ssl_suspicious_mucosa")
        self.assertFalse(result["errors"])

    def test_trace_agent_run_fuses_grid_first_patch_assignments_with_conch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = self._write_grid_metadata(root, valid_image=True)
            request = self._build_request(image_path)
            bundle = {
                "runtime": {
                    "trace": {
                        "labels": [
                            "background_artifact_stroma",
                            "normal_mucosa",
                            "conventional_adenoma_like",
                            "inflammatory_polyp_like",
                            "ssl_suspicious_mucosa",
                        ],
                        "backend_chain": ["stub"],
                        "enable_conch_fusion": True,
                    }
                },
                "budget": {"max_trace_candidates": 4},
            }
            text = json.dumps(
                {
                    "patches": [
                        {
                            "patch_id": [0, 0],
                            "name": "SSL-like mucosa",
                            "region_semantic": "ssl_suspicious_mucosa",
                            "description": "Upper left patch has same pattern.",
                            "require_high_magnification": True,
                            "severity_reasoning": "Serrated concern.",
                            "diagnostic_priority": 4,
                            "observation_points": ["lesion edge"],
                        },
                        {
                            "patch_id": [0, 1],
                            "name": "Background/stroma remainder",
                            "region_semantic": "background_artifact_stroma",
                            "description": "Low value.",
                            "require_high_magnification": False,
                            "severity_reasoning": "Discard.",
                            "diagnostic_priority": 0,
                            "observation_points": ["coverage"],
                        },
                        {
                            "patch_id": [1, 0],
                            "name": "Normal mucosa",
                            "region_semantic": "normal_mucosa",
                            "description": "Low-priority mucosa.",
                            "require_high_magnification": False,
                            "severity_reasoning": "Low concern.",
                            "diagnostic_priority": 1,
                            "observation_points": ["benign architecture"],
                        },
                        {
                            "patch_id": [1, 1],
                            "name": "Conventional",
                            "region_semantic": "conventional_adenoma_like",
                            "description": "Conventional concern.",
                            "require_high_magnification": True,
                            "severity_reasoning": "Crowded glands.",
                            "diagnostic_priority": 3,
                            "observation_points": ["crowding"],
                        },
                    ]
                }
            )
            backend_output = _build_trace_output_from_text(text, request, bundle)

            class _BackendChain(object):
                def invoke(self, stage, chain_names, request_payload):
                    return {
                        "backend": "stub",
                        "output": backend_output,
                        "attempts": [],
                        "raw_text": text,
                    }

            class _Logger(object):
                def log(self, **kwargs):
                    return None

            case_spec = CaseSpec(
                case_id="case_001",
                slide_path=str(image_path),
                task_type="ssl_others_dual_branch_cpathagent_grid",
                question="trace question",
                input_mode="grid_thumbnail",
                grid_thumbnail_path=str(image_path),
                grid_metadata_path=str(image_path.with_suffix(".json")),
                metadata={},
            )
            agent = TraceAgent(bundle, selector_adapter=None, backend_chain=_BackendChain())
            with mock.patch(
                "adenoma_agent.agents.trace.request_conch_patch_predictions",
                return_value={
                    "enabled": True,
                    "predictions": {
                        "0,0": "ssl_suspicious_mucosa",
                        "0,1": "normal_mucosa",
                        "1,0": "background_artifact_stroma",
                        "1,1": "ssl_suspicious_mucosa",
                    },
                    "attempts": [],
                    "invalid_predictions": [],
                    "errors": [],
                },
            ):
                result = agent.run(case_spec, root / "case_output", _Logger())
            trace_payload = json.loads(result["trace_clusters_json"].read_text(encoding="utf-8"))
            fused_patches = trace_payload["patch_assignments"]["patches"]
            self.assertEqual(fused_patches[0]["agreement_status"], "strong_agreement")
            self.assertEqual(fused_patches[0]["score_origin"], "consensus_fusion")
            self.assertEqual(fused_patches[1]["agreement_status"], "low_value_disagreement")
            self.assertEqual(fused_patches[1]["conch_region_semantic"], "normal_mucosa")
            self.assertEqual(fused_patches[3]["agreement_status"], "risk_disagreement")
            self.assertEqual(trace_payload["conch_runtime_metadata"]["enabled"], True)

    def test_trace_agent_run_fuses_conch_only_trace_with_digepath(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = self._write_grid_metadata(root, valid_image=True)
            bundle = {
                "runtime": {
                    "trace": {
                        "mode": "conch_only",
                        "enable_digepath_fusion": True,
                        "digepath_high_confidence_threshold": 0.70,
                        "cluster_grid_size": 16,
                        "min_cell_tissue_fraction": 0.08,
                        "min_cluster_area_fraction": 0.01,
                    },
                    "backends": {
                        "local_digepath": {
                            "enabled": True,
                            "server_url": "http://127.0.0.1:8300/predict",
                            "timeout_seconds": 3,
                        }
                    },
                },
                "budget": {"max_trace_candidates": 4},
            }

            class _Logger(object):
                def log(self, **kwargs):
                    return None

            case_spec = CaseSpec(
                case_id="case_001",
                slide_path=str(image_path),
                task_type="ssl_others_dual_branch_cpathagent_grid",
                question="trace question",
                input_mode="grid_thumbnail",
                grid_thumbnail_path=str(image_path),
                grid_metadata_path=str(image_path.with_suffix(".json")),
                metadata={},
            )
            conch_runtime = {
                "enabled": True,
                "predictions": {
                    "0,0": "epithelial_neoplasia_suspicious",
                    "0,1": "background_or_artifact",
                    "1,0": "background_or_artifact",
                    "1,1": "mucus_rich_or_pale_context",
                },
                "prediction_details": {
                    "0,0": {"conch_region_semantic": "epithelial_neoplasia_suspicious", "conch_crc_label": "TUM", "conch_confidence": 0.91},
                    "0,1": {"conch_region_semantic": "background_or_artifact", "conch_crc_label": "BACK", "conch_confidence": 0.95},
                    "1,0": {"conch_region_semantic": "background_or_artifact", "conch_crc_label": "BACK", "conch_confidence": 0.96},
                    "1,1": {"conch_region_semantic": "mucus_rich_or_pale_context", "conch_crc_label": "MUC", "conch_confidence": 0.88},
                },
                "attempts": [],
                "invalid_predictions": [],
                "errors": [],
            }
            digepath_runtime = {
                "enabled": True,
                "predictions": {
                    "0,0": "epithelial_neoplasia_suspicious",
                    "0,1": "epithelial_neoplasia_suspicious",
                    "1,0": "background_or_artifact",
                    "1,1": "mucus_rich_or_pale_context",
                },
                "prediction_details": {
                    "0,0": {"digepath_class": "tumor_epithelium", "digepath_region_semantic": "epithelial_neoplasia_suspicious", "digepath_confidence": 0.93},
                    "0,1": {"digepath_class": "tumor_epithelium", "digepath_region_semantic": "epithelial_neoplasia_suspicious", "digepath_confidence": 0.90},
                    "1,0": {"digepath_class": "background", "digepath_region_semantic": "background_or_artifact", "digepath_confidence": 0.99},
                    "1,1": {"digepath_class": "mucus", "digepath_region_semantic": "mucus_rich_or_pale_context", "digepath_confidence": 0.86},
                },
                "attempts": [{"backend": "local_digepath", "patch_id": [0, 0], "status": 200, "latency_ms": 5}],
                "invalid_predictions": [],
                "errors": [],
                "high_confidence_threshold": 0.70,
            }
            agent = TraceAgent(bundle, selector_adapter=None, backend_chain=None)
            with mock.patch("adenoma_agent.agents.trace.request_conch_patch_predictions", return_value=conch_runtime):
                with mock.patch("adenoma_agent.agents.trace.request_digepath_patch_predictions", return_value=digepath_runtime) as digepath_mock:
                    result = agent.run(case_spec, root / "case_output", _Logger())
            self.assertTrue(digepath_mock.called)
            trace_payload = json.loads(result["trace_clusters_json"].read_text(encoding="utf-8"))
            fused_patches = trace_payload["patch_assignments"]["patches"]
            self.assertEqual(fused_patches[0]["agreement_status"], "conch_digepath_agree")
            self.assertEqual(fused_patches[0]["digepath_class"], "tumor_epithelium")
            self.assertEqual(fused_patches[1]["region_semantic"], "uncertain_reviewable_mucosa")
            self.assertEqual(fused_patches[1]["agreement_status"], "conch_digepath_disagree")
            self.assertEqual(fused_patches[2]["region_semantic"], "background_or_artifact")
            self.assertEqual(trace_payload["coverage_summary"]["digepath_fusion_enabled"], True)
            self.assertEqual(trace_payload["digepath_runtime_metadata"]["enabled"], True)

    def test_trace_agent_conch_only_soft_falls_back_when_digepath_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = self._write_grid_metadata(root, valid_image=True)
            bundle = {
                "runtime": {
                    "trace": {
                        "mode": "conch_only",
                        "enable_digepath_fusion": True,
                        "cluster_grid_size": 16,
                        "min_cell_tissue_fraction": 0.08,
                        "min_cluster_area_fraction": 0.01,
                    },
                    "backends": {"local_digepath": {"enabled": False}},
                },
                "budget": {"max_trace_candidates": 4},
            }

            class _Logger(object):
                def log(self, **kwargs):
                    return None

            case_spec = CaseSpec(
                case_id="case_001",
                slide_path=str(image_path),
                task_type="ssl_others_dual_branch_cpathagent_grid",
                question="trace question",
                input_mode="grid_thumbnail",
                grid_thumbnail_path=str(image_path),
                grid_metadata_path=str(image_path.with_suffix(".json")),
                metadata={},
            )
            conch_runtime = {
                "enabled": True,
                "predictions": {"0,0": "epithelial_neoplasia_suspicious"},
                "prediction_details": {
                    "0,0": {"conch_region_semantic": "epithelial_neoplasia_suspicious", "conch_crc_label": "TUM", "conch_confidence": 0.91}
                },
                "attempts": [],
                "invalid_predictions": [],
                "errors": [],
            }
            agent = TraceAgent(bundle, selector_adapter=None, backend_chain=None)
            with mock.patch("adenoma_agent.agents.trace.request_conch_patch_predictions", return_value=conch_runtime):
                with mock.patch("adenoma_agent.agents.trace.request_digepath_patch_predictions", return_value={"enabled": False, "attempts": [], "errors": []}):
                    result = agent.run(case_spec, root / "case_output", _Logger())
            trace_payload = json.loads(result["trace_clusters_json"].read_text(encoding="utf-8"))
            self.assertEqual(trace_payload["coverage_summary"]["final_groups_source"], "conch_only_patch_assignments")
            self.assertEqual(trace_payload["patch_assignments"]["patches"][0]["agreement_status"], "conch_only_trace")

    def test_classifier_patch_requests_use_clean_overview_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            grid_path = root / "case_001_tissuegrid64_ds32_level2_grid.jpg"
            clean_path = root / "case_001_tissuegrid64_ds32_level2.jpg"
            Image.new("RGB", (64, 64), color=(255, 0, 0)).save(grid_path)
            Image.new("RGB", (64, 64), color=(0, 255, 0)).save(clean_path)
            metadata_path = grid_path.with_suffix(".json")
            metadata_path.write_text(
                json.dumps(
                    {
                        "grid_cells": [
                            {
                                "row_id": 0,
                                "col_id": 0,
                                "is_selected": True,
                                "thumbnail_top_left_x": 0,
                                "thumbnail_top_left_y": 0,
                                "thumbnail_width": 32,
                                "thumbnail_height": 32,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            bundle = {"runtime": {"trace": {}}, "budget": {"max_trace_candidates": 4}}
            agent = TraceAgent(bundle, selector_adapter=None, backend_chain=None)
            requests = agent._collect_conch_patch_requests(grid_path, metadata_path, root / "crops")
            self.assertEqual(requests[0]["classifier_source_mode"], "clean_overview")
            self.assertEqual(Path(requests[0]["classifier_source_image"]), clean_path)
            with Image.open(requests[0]["image_path"]).convert("RGB") as crop:
                pixel = crop.getpixel((0, 0))
                self.assertGreaterEqual(pixel[1], 250)
                self.assertLessEqual(pixel[0], 5)

    def test_classifier_patch_requests_use_wsi_level0_crop_when_slide_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            grid_path = root / "case_001_tissuegrid64_ds32_level2_grid.jpg"
            clean_path = root / "case_001_tissuegrid64_ds32_level2.jpg"
            wsi_path = root / "case_001_wsi.png"
            Image.new("RGB", (64, 64), color=(255, 255, 255)).save(grid_path)
            Image.new("RGB", (64, 64), color=(0, 255, 0)).save(clean_path)
            wsi = Image.new("RGB", (128, 128), color=(0, 0, 255))
            wsi.paste((255, 0, 0), (16, 16, 48, 48))
            wsi.save(wsi_path)
            metadata_path = grid_path.with_suffix(".json")
            metadata_path.write_text(
                json.dumps(
                    {
                        "slide_path": str(wsi_path),
                        "grid_cell_size_level0": 32,
                        "grid_cells": [
                            {
                                "row_id": 0,
                                "col_id": 0,
                                "patch_id": [0, 0],
                                "is_selected": True,
                                "thumbnail_top_left_x": 0,
                                "thumbnail_top_left_y": 0,
                                "thumbnail_width": 32,
                                "thumbnail_height": 32,
                                "level0_top_left_x": 16,
                                "level0_top_left_y": 16,
                                "level0_width": 32,
                                "level0_height": 32,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            bundle = {
                "runtime": {"trace": {"classifier_crop_source": "wsi_first", "classifier_wsi_crop_output_size": 32}},
                "budget": {"max_trace_candidates": 4},
            }
            agent = TraceAgent(bundle, selector_adapter=None, backend_chain=None)
            requests = agent._collect_conch_patch_requests(grid_path, metadata_path, root / "crops", slide_path=wsi_path)
            self.assertEqual(requests[0]["classifier_source_mode"], "wsi_level0_5x")
            self.assertEqual(Path(requests[0]["classifier_source_image"]), wsi_path)
            self.assertEqual(requests[0]["classifier_crop_level0_bbox"], [16, 16, 48, 48])
            self.assertEqual(requests[0]["classifier_crop_output_size"], [32, 32])
            with Image.open(requests[0]["image_path"]).convert("RGB") as crop:
                pixel = crop.getpixel((0, 0))
                self.assertGreaterEqual(pixel[0], 250)
                self.assertLessEqual(pixel[1], 5)

    def test_build_trace_output_from_groups_maps_to_internal_clusters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            request = self._build_request(image_path)
            bundle = {
                "runtime": {
                    "trace": {
                        "labels": [
                            "background_artifact_stroma",
                            "normal_mucosa",
                            "conventional_adenoma_like",
                            "inflammatory_polyp_like",
                            "ssl_suspicious_mucosa",
                        ]
                    }
                }
            }
            text = json.dumps(
                {
                    "groups": [
                        {
                            "name": "SSL-suspicious mucosa",
                            "region_semantic": "ssl_high_priority_mucosa",
                            "description": "Pale mucosal region with lesion edge worth closer review.",
                            "id_list": [[0, 0], [1, 1]],
                            "require_high_magnification": True,
                            "severity_reasoning": "Surface pallor and region coherence support serrated suspicion.",
                            "diagnostic_priority": 5,
                            "observation_points": ["possible serrated surface pattern", "lesion edge localization"],
                        }
                    ]
                }
            )
            output = _build_trace_output_from_text(text, request, bundle)
            self.assertEqual(len(output["clusters"]), 2)
            cluster = output["clusters"][0]
            self.assertEqual(cluster["cluster_id"], "grid_group_00")
            self.assertEqual(cluster["l"], "ssl_suspicious_mucosa")
            self.assertEqual(cluster["s"], 5)
            self.assertTrue(cluster["d"])
            self.assertEqual(cluster["metadata"]["workflow_branch"], "serrated")
            self.assertTrue(cluster["metadata"]["serrated_dysplasia_suspected"])
            self.assertFalse(cluster["metadata"]["conventional_dysplasia_suspected"])
            self.assertEqual(cluster["cluster_bbox_thumb"], {"x1": 0, "y1": 0, "x2": 221, "y2": 244})
            self.assertEqual(cluster["metadata"]["grid_id_list"], [[0, 0], [1, 1]])
            self.assertEqual(cluster["patch_ids_ordered"], [[0, 0], [1, 1]])
            self.assertEqual(output["clusters"][1]["cluster_id"], "grid_group_fallback")
            self.assertEqual(output["clusters"][1]["l"], "background_artifact_stroma")
            self.assertEqual(output["clusters"][1]["patch_ids_ordered"], [[0, 1], [1, 0]])
            self.assertFalse(output["coverage_summary"]["coverage_ok"])
            self.assertEqual(output["coverage_summary"]["missing_patch_ids"], [[0, 1], [1, 0]])

    def test_build_trace_output_from_empty_groups_repairs_full_coverage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            request = self._build_request(image_path)
            bundle = {"runtime": {"trace": {"labels": []}}}
            output = _build_trace_output_from_text(json.dumps({"groups": []}), request, bundle)
            self.assertEqual(len(output["clusters"]), 1)
            self.assertEqual(output["clusters"][0]["cluster_id"], "grid_group_fallback")
            self.assertEqual(output["clusters"][0]["patch_ids_ordered"], [[0, 0], [0, 1], [1, 0], [1, 1]])
            self.assertEqual(output["clusters"][0]["s"], 0)
            self.assertEqual(output["coverage_summary"]["empty_groups"], [])
            self.assertEqual(output["coverage_summary"]["missing_patch_ids"], [[0, 0], [0, 1], [1, 0], [1, 1]])

    def test_build_trace_output_from_prose_with_explicit_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            request = self._build_request(image_path)
            bundle = {"runtime": {"trace": {"labels": []}}}
            text = (
                "<think>reasoning</think>"
                "<answer>The priority groups are the upper left (0,0) and lower right (1,1) grid cells, "
                "which warrant high-magnification review for serrated lesions.</answer>"
            )
            output = _build_trace_output_from_text(text, request, bundle)
            self.assertEqual(len(output["clusters"]), 2)
            self.assertEqual(output["clusters"][0]["metadata"]["grid_id_list"], [[0, 0], [1, 1]])
            self.assertTrue(output["clusters"][0]["d"])
            self.assertEqual(output["clusters"][1]["patch_ids_ordered"], [[0, 1], [1, 0]])

    def test_build_trace_output_from_prose_with_row_column_ranges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            request = self._build_request(image_path)
            bundle = {"runtime": {"trace": {"labels": []}}}
            text = (
                "<answer>The prioritized groups are the cells in grid rows 0 and 1, columns 1, "
                "and row 1, col 0. These regions show a pale, mucus-rich surface and irregular contour.</answer>"
            )
            output = _build_trace_output_from_text(text, request, bundle)
            self.assertEqual(len(output["clusters"]), 2)
            self.assertEqual(output["clusters"][0]["metadata"]["grid_id_list"], [[0, 1], [1, 0], [1, 1]])
            self.assertIn("mucus-rich surface pattern", output["clusters"][0]["evidence"])
            self.assertEqual(output["clusters"][1]["patch_ids_ordered"], [[0, 0]])

    def test_build_trace_output_from_prose_with_grid_object_fallback_uses_all_selected_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            request = self._build_request(image_path)
            bundle = {"runtime": {"trace": {"labels": []}}}
            text = (
                "<answer>The prioritized groups are the cells listed in the _grid_ object. "
                "The groups are grouped into a single coherent region.</answer>"
            )
            output = _build_trace_output_from_text(text, request, bundle)
            self.assertEqual(len(output["clusters"]), 1)
            self.assertEqual(
                output["clusters"][0]["metadata"]["grid_id_list"],
                [[0, 0], [0, 1], [1, 0], [1, 1]],
            )

    def test_build_trace_output_deduplicates_patch_ids_and_repairs_missing_ones(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            request = self._build_request(image_path)
            bundle = {"runtime": {"trace": {"labels": []}}}
            text = json.dumps(
                {
                    "groups": [
                        {
                            "name": "normal mucosa",
                            "region_semantic": "normal_mucosa",
                            "description": "Low-priority mucosa.",
                            "id_list": [[0, 0], [0, 1]],
                            "require_high_magnification": False,
                            "severity_reasoning": "Low concern.",
                            "diagnostic_priority": 1,
                            "observation_points": ["confirm benign architecture"],
                        },
                        {
                            "name": "follow-up patch",
                            "region_semantic": "ssl_suspicious_mucosa",
                            "description": "Check an overlapping list.",
                            "id_list": [[0, 1], [1, 0]],
                            "require_high_magnification": True,
                            "severity_reasoning": "Overlap should be repaired deterministically.",
                            "diagnostic_priority": 3,
                            "observation_points": ["compare repeated patch"],
                        },
                    ]
                }
            )
            output = _build_trace_output_from_text(text, request, bundle)
            self.assertEqual(output["clusters"][0]["patch_ids_ordered"], [[1, 0]])
            self.assertEqual(output["clusters"][0]["metadata"]["dropped_duplicate_patch_ids"], [[0, 1]])
            self.assertEqual(output["clusters"][1]["patch_ids_ordered"], [[0, 0], [0, 1]])
            self.assertEqual(output["clusters"][2]["patch_ids_ordered"], [[1, 1]])
            self.assertEqual(output["coverage_summary"]["duplicate_patch_ids"], [[0, 1]])
            self.assertEqual(output["coverage_summary"]["missing_patch_ids"], [[1, 1]])

    def test_load_trace_grid_metadata_rejects_non_tissue_grid_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "case_001_tissuegrid125_ds32_level2_grid.jpg"
            image_path.write_bytes(b"fake-image")
            image_path.with_suffix(".json").write_text(
                json.dumps({"thumbnail_mode": "grid32x", "grid_cells": []}),
                encoding="utf-8",
            )
            prompt = _build_trace_patho_r1_prompt(
                self._build_request(image_path),
                {"runtime": {"trace": {"labels": []}}},
            )
            self.assertIn("Candidate proposals in thumbnail pixel space:", prompt)

    def test_trace_label_mapping_separates_conventional_inflammatory_and_ssl(self):
        self.assertEqual(
            _normalize_trace_label(
                "others",
                "Non-SSL tubular adenoma-like region",
                "Crowded non-SSL tubular adenoma glands.",
                "Adenomatous pattern.",
                True,
                4,
            ),
            "conventional_adenoma_like",
        )
        self.assertEqual(
            _normalize_trace_label(
                "others",
                "Inflammatory polyp-like region",
                "Reactive inflamed mucosa.",
                "Inflammatory polyp pattern.",
                False,
                2,
            ),
            "inflammatory_polyp_like",
        )
        self.assertEqual(
            _normalize_trace_label(
                "serrated",
                "HP-like serrated region",
                "Equivocal serrated architecture.",
                "Needs SSL pathway review.",
                True,
                3,
            ),
            "ssl_suspicious_mucosa",
        )

    def test_run_trace_grid_with_retry_repairs_missing_patch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            request = self._build_request(image_path)
            bundle = {"runtime": {"trace": {"labels": []}}}
            responses = iter(
                [
                    json.dumps(
                        {
                            "groups": [
                                {
                                    "name": "suspicious mucosa",
                                    "region_semantic": "ssl_suspicious_mucosa",
                                    "description": "Initial pass missed one patch.",
                                    "id_list": [[0, 1], [1, 0], [1, 1]],
                                    "require_high_magnification": True,
                                    "severity_reasoning": "Needs review.",
                                    "diagnostic_priority": 4,
                                    "observation_points": ["serrated concern"],
                                }
                            ]
                        }
                    ),
                    json.dumps(
                        {
                            "groups": [
                                {
                                    "name": "suspicious mucosa",
                                    "region_semantic": "ssl_suspicious_mucosa",
                                    "description": "Corrected full assignment.",
                                    "id_list": [[0, 0], [0, 1], [1, 0], [1, 1]],
                                    "require_high_magnification": True,
                                    "severity_reasoning": "Needs review.",
                                    "diagnostic_priority": 4,
                                    "observation_points": ["serrated concern"],
                                }
                            ]
                        }
                    ),
                ]
            )
            result = _run_trace_grid_with_coverage_retry(lambda prompt: next(responses), request, bundle)
            self.assertEqual(len(result["trace_attempts"]), 2)
            self.assertFalse(result["trace_attempts"][0]["coverage_ok"])
            self.assertEqual(result["trace_attempts"][0]["missing_patch_ids"], [[0, 0]])
            self.assertTrue(result["trace_attempts"][1]["coverage_ok"])
            self.assertEqual(len(result["output"]["clusters"]), 1)
            self.assertTrue(result["output"]["coverage_summary"]["coverage_ok"])
            self.assertTrue(result["output"]["coverage_summary"]["coverage_repaired_by_retry"])
            self.assertFalse(result["output"]["coverage_summary"]["final_used_fallback"])
            self.assertEqual(result["output"]["clusters"][0]["patch_ids_ordered"], [[0, 0], [0, 1], [1, 0], [1, 1]])

    def test_run_trace_grid_with_retry_falls_back_after_second_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            request = self._build_request(image_path)
            bundle = {"runtime": {"trace": {"labels": []}}}
            responses = iter(
                [
                    json.dumps(
                        {
                            "groups": [
                                {
                                    "name": "normal mucosa",
                                    "region_semantic": "normal_mucosa",
                                    "description": "Misses [1,1].",
                                    "id_list": [[0, 0], [0, 1], [1, 0]],
                                    "require_high_magnification": False,
                                    "severity_reasoning": "Low concern.",
                                    "diagnostic_priority": 1,
                                    "observation_points": ["benign architecture"],
                                }
                            ]
                        }
                    ),
                    json.dumps(
                        {
                            "groups": [
                                {
                                    "name": "normal mucosa",
                                    "region_semantic": "normal_mucosa",
                                    "description": "Still misses [1,1].",
                                    "id_list": [[0, 0], [0, 1], [1, 0]],
                                    "require_high_magnification": False,
                                    "severity_reasoning": "Low concern.",
                                    "diagnostic_priority": 1,
                                    "observation_points": ["benign architecture"],
                                }
                            ]
                        }
                    ),
                    json.dumps(
                        {
                            "groups": [
                                {
                                    "name": "normal mucosa",
                                    "region_semantic": "normal_mucosa",
                                    "description": "Third try still misses [1,1].",
                                    "id_list": [[0, 0], [0, 1], [1, 0]],
                                    "require_high_magnification": False,
                                    "severity_reasoning": "Low concern.",
                                    "diagnostic_priority": 1,
                                    "observation_points": ["benign architecture"],
                                }
                            ]
                        }
                    ),
                ]
            )
            result = _run_trace_grid_with_coverage_retry(lambda prompt: next(responses), request, bundle)
            self.assertEqual(len(result["trace_attempts"]), 3)
            self.assertTrue(result["output"]["coverage_summary"]["retry_attempted"])
            self.assertTrue(result["output"]["coverage_summary"]["final_used_fallback"])
            self.assertEqual(result["output"]["clusters"][-1]["cluster_id"], "grid_group_fallback")
            self.assertEqual(result["output"]["clusters"][-1]["metadata"]["missing_patch_ids"], [[1, 1]])
            self.assertEqual(result["output"]["clusters"][-1]["metadata"]["coverage_repair_stage"], "post_retry")

    def test_run_trace_grid_with_retry_records_unexpected_patch_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            request = self._build_request(image_path)
            bundle = {"runtime": {"trace": {"labels": []}}}
            responses = iter(
                [
                    json.dumps(
                        {
                            "groups": [
                                {
                                    "name": "invalid patch",
                                    "region_semantic": "normal_mucosa",
                                    "description": "Includes an out-of-set patch.",
                                    "id_list": [[0, 0], [0, 1], [1, 0], [2, 2]],
                                    "require_high_magnification": False,
                                    "severity_reasoning": "Bad ID.",
                                    "diagnostic_priority": 1,
                                    "observation_points": ["out-of-set patch"],
                                }
                            ]
                        }
                    ),
                    json.dumps(
                        {
                            "groups": [
                                {
                                    "name": "corrected patch set",
                                    "region_semantic": "normal_mucosa",
                                    "description": "Corrected assignment.",
                                    "id_list": [[0, 0], [0, 1], [1, 0], [1, 1]],
                                    "require_high_magnification": False,
                                    "severity_reasoning": "Fixed.",
                                    "diagnostic_priority": 1,
                                    "observation_points": ["corrected"],
                                }
                            ]
                        }
                    ),
                ]
            )
            result = _run_trace_grid_with_coverage_retry(lambda prompt: next(responses), request, bundle)
            self.assertEqual(result["trace_attempts"][0]["unexpected_patch_ids"], [[2, 2]])
            self.assertEqual(result["trace_attempts"][0]["ignored_patch_ids"], [[2, 2]])
            self.assertTrue(result["output"]["coverage_summary"]["coverage_repaired_by_retry"])

    def test_run_trace_grid_with_second_retry_can_repair_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid_metadata(tmpdir)
            request = self._build_request(image_path)
            bundle = {"runtime": {"trace": {"labels": []}}}
            responses = iter(
                [
                    json.dumps(
                        {
                            "groups": [
                                {
                                    "name": "bad first pass",
                                    "region_semantic": "ssl_suspicious_mucosa",
                                    "description": "Includes an invalid patch.",
                                    "id_list": [[0, 0], [2, 2]],
                                    "require_high_magnification": True,
                                    "severity_reasoning": "First pass is wrong.",
                                    "diagnostic_priority": 4,
                                    "observation_points": ["invalid id"],
                                }
                            ]
                        }
                    ),
                    json.dumps(
                        {
                            "groups": [
                                {
                                    "name": "still wrong",
                                    "region_semantic": "ssl_suspicious_mucosa",
                                    "description": "Still misses one patch.",
                                    "id_list": [[0, 0], [0, 1], [1, 0]],
                                    "require_high_magnification": True,
                                    "severity_reasoning": "Needs one more fix.",
                                    "diagnostic_priority": 4,
                                    "observation_points": ["missing one patch"],
                                }
                            ]
                        }
                    ),
                    json.dumps(
                        {
                            "groups": [
                                {
                                    "name": "fixed",
                                    "region_semantic": "ssl_suspicious_mucosa",
                                    "description": "Complete corrected assignment.",
                                    "id_list": [[0, 0], [0, 1], [1, 0], [1, 1]],
                                    "require_high_magnification": True,
                                    "severity_reasoning": "Fully corrected.",
                                    "diagnostic_priority": 4,
                                    "observation_points": ["complete"],
                                }
                            ]
                        }
                    ),
                ]
            )
            result = _run_trace_grid_with_coverage_retry(lambda prompt: next(responses), request, bundle)
            self.assertEqual(len(result["trace_attempts"]), 3)
            self.assertFalse(result["trace_attempts"][0]["coverage_ok"])
            self.assertFalse(result["trace_attempts"][1]["coverage_ok"])
            self.assertTrue(result["trace_attempts"][2]["coverage_ok"])
            self.assertTrue(result["output"]["coverage_summary"]["coverage_ok"])
            self.assertTrue(result["output"]["coverage_summary"]["coverage_repaired_by_retry"])
            self.assertFalse(result["output"]["coverage_summary"]["final_used_fallback"])


class TraceNavigateHeuristicTest(unittest.TestCase):
    def test_navigate_uses_group_priority_then_patch_order(self):
        backend = HeuristicStageBackend()
        request = {
            "stage": "navigate",
            "metadata": {
                "slide_dimensions_level0": [4000, 4000],
                "clusters": [
                    {
                        "cluster_id": "cluster_low",
                        "l": "normal_mucosa",
                        "s": 1,
                        "d": False,
                        "cluster_bbox_level0": {"x1": 3000, "y1": 3000, "x2": 3200, "y2": 3200},
                        "patches_level0": [
                            {"patch_id": [2, 0], "x1": 3000, "y1": 3000, "x2": 3200, "y2": 3200}
                        ],
                    },
                    {
                        "cluster_id": "cluster_high",
                        "l": "ssl_suspicious_mucosa",
                        "s": 5,
                        "d": True,
                        "cluster_bbox_level0": {"x1": 400, "y1": 400, "x2": 1900, "y2": 1900},
                        "patches_level0": [
                            {"patch_id": [0, 0], "x1": 400, "y1": 400, "x2": 600, "y2": 600},
                            {"patch_id": [1, 1], "x1": 1600, "y1": 1600, "x2": 1800, "y2": 1800},
                        ],
                    },
                ],
            },
        }
        bundle = {
            "runtime": {"navigate": {"overlap_threshold": 0.05}},
            "budget": {
                "max_navigation_steps": 8,
                "magnification_to_region_size": {"2.5": 4096, "5.0": 2048, "10.0": 1024},
            },
        }
        output = backend.invoke(request, bundle)["output"]
        inspect_steps = [step for step in output["steps"] if step["metadata"].get("action") == "inspect"]
        self.assertEqual(inspect_steps[0]["m"], 2.5)
        self.assertEqual(inspect_steps[0]["review_goal"], "serrated_overview_assessment")
        self.assertIn("ssl_assessment", [step["review_goal"] for step in inspect_steps])
        self.assertIn("hp_assessment", [step["review_goal"] for step in inspect_steps])
        self.assertIn("tsa_assessment", [step["review_goal"] for step in inspect_steps])
        self.assertFalse([step for step in inspect_steps if step["m"] == 10.0])

    def test_navigate_routes_conventional_adenoma_branch_without_serrated_gate(self):
        backend = HeuristicStageBackend()
        request = {
            "stage": "navigate",
            "metadata": {
                "slide_dimensions_level0": [4000, 4000],
                "clusters": [
                    {
                        "cluster_id": "cluster_conventional",
                        "l": "conventional_adenoma_like",
                        "s": 4,
                        "d": True,
                        "cluster_bbox_level0": {"x1": 400, "y1": 400, "x2": 600, "y2": 600},
                        "patches_level0": [
                            {"patch_id": [0, 0], "x1": 400, "y1": 400, "x2": 600, "y2": 600}
                        ],
                    }
                ],
            },
        }
        bundle = {
            "runtime": {"navigate": {"overlap_threshold": 0.05}},
            "budget": {
                "max_navigation_steps": 4,
                "magnification_to_region_size": {"2.5": 4096, "5.0": 2048, "10.0": 1024},
            },
        }
        output = backend.invoke(request, bundle)["output"]
        inspect_steps = [step for step in output["steps"] if step["metadata"].get("action") == "inspect"]
        self.assertEqual(len(inspect_steps), 3)
        self.assertEqual(inspect_steps[0]["m"], 2.5)
        self.assertEqual(inspect_steps[0]["review_goal"], "conventional_overview_assessment")
        self.assertEqual(inspect_steps[1]["review_goal"], "conventional_architecture_assessment")
        self.assertEqual(inspect_steps[1]["stage_gate"], "conventional_architecture")
        self.assertEqual(inspect_steps[2]["review_goal"], "reactive_regenerative_assessment")
        self.assertEqual(inspect_steps[2]["stage_gate"], "reactive_regenerative")
        self.assertEqual(inspect_steps[0]["metadata"]["workflow_branch"], "conventional")

    def test_navigate_generates_multiple_intra_cell_zoom_targets_for_ssl(self):
        backend = HeuristicStageBackend()
        request = {
            "stage": "navigate",
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
                    }
                ],
            },
        }
        bundle = {
            "runtime": {"navigate": {"overlap_threshold": 0.05}},
            "budget": {
                "max_navigation_steps": 8,
                "max_intra_cell_zoom_targets": 3,
                "magnification_to_region_size": {"2.5": 4096, "5.0": 2048, "10.0": 1024},
            },
        }
        output = backend.invoke(request, bundle)["output"]
        inspect_steps = [step for step in output["steps"] if step["metadata"].get("action") == "inspect"]
        zoom_steps = [step for step in inspect_steps if step["m"] == 10.0]
        self.assertEqual(inspect_steps[0]["m"], 2.5)
        self.assertEqual(len([step for step in inspect_steps if step["m"] == 5.0]), 3)
        self.assertEqual(len(zoom_steps), 0)
        self.assertTrue(all(step["metadata"]["coordinate_source"] in {"heuristic_offset", "trace_anchor"} for step in zoom_steps))

    def test_navigate_does_not_multi_zoom_low_priority_normal_cell(self):
        backend = HeuristicStageBackend()
        request = {
            "stage": "navigate",
            "metadata": {
                "slide_dimensions_level0": [10000, 10000],
                "clusters": [
                    {
                        "cluster_id": "cluster_normal",
                        "l": "normal_mucosa",
                        "s": 1,
                        "d": False,
                        "cluster_bbox_level0": {"x1": 2000, "y1": 2000, "x2": 6000, "y2": 6000},
                        "patches_level0": [
                            {"patch_id": [0, 0], "x1": 2000, "y1": 2000, "x2": 6000, "y2": 6000}
                        ],
                    }
                ],
            },
        }
        bundle = {
            "runtime": {"navigate": {"overlap_threshold": 0.05}},
            "budget": {
                "max_navigation_steps": 8,
                "max_intra_cell_zoom_targets": 3,
                "magnification_to_region_size": {"2.5": 4096, "5.0": 2048, "10.0": 1024},
            },
        }
        output = backend.invoke(request, bundle)["output"]
        inspect_steps = [step for step in output["steps"] if step["metadata"].get("action") == "inspect"]
        self.assertEqual(len(inspect_steps), 1)
        self.assertEqual(inspect_steps[0]["m"], 2.5)
        self.assertEqual(inspect_steps[0]["review_goal"], "normal_overview_assessment")


if __name__ == "__main__":
    unittest.main()
