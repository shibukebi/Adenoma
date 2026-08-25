import json
import re
import tempfile
import unittest
from pathlib import Path

from adenoma_agent.dashboard import (
    build_dashboard_payload_from_harness_case,
    build_demo_dashboard_payload,
    export_dashboard,
)


class DashboardExportTest(unittest.TestCase):
    def test_export_dashboard_writes_html_and_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "dashboard"
            payload = build_demo_dashboard_payload()
            result_dir = export_dashboard(payload, output_dir)

            self.assertTrue((result_dir / "index.html").exists())
            self.assertTrue((result_dir / "dashboard_payload.json").exists())

            html = (result_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Agent 实验可视化看板", html)
            self.assertIn("entropy_disagreement_mode", html)
            self.assertIn("turbo_audit", html)
            self.assertIn("审计控制", html)
            self.assertIn("孤儿资产抽屉", html)
            self.assertIn("risk_disagreement", html)
            self.assertIn("dashboard-data", html)
            self.assertIn("column-resizer", html)
            self.assertIn("toggle-left-header", html)
            self.assertIn("GPS Navigation", html)
            self.assertIn("timeline-play", html)
            self.assertIn("Observation / Reasoning", html)
            self.assertIn("observation-card-list", html)
            match = re.search(r'<script id="dashboard-data" type="application/json">(.*?)</script>', html, flags=re.S)
            self.assertIsNotNone(match)
            self.assertNotIn("&quot;", match.group(1))

            snapshot = json.loads((result_dir / "dashboard_payload.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["case_id"], "demo_case_ssl_001")
            self.assertIn("contract_validation", snapshot)
            self.assertIn("orphan_assets", snapshot)

    def test_export_dashboard_normalizes_missing_and_orphan_states(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "dashboard"
            payload = {
                "case_id": "case_x",
                "slide_id": "slide_y",
                "selected_patch_ids": [[0, 0], [0, 1]],
                "grid_metadata": {
                    "grid_rows": 1,
                    "grid_cols": 2,
                    "patch_size": 256,
                    "overview_magnification": 5.0,
                    "selected_patch_count": 2,
                    "generated_at": "2026-06-01T00:00:00Z",
                    "model_version": "test",
                    "cropped_thumbnail_size": [220, 100],
                    "grid_cells": [
                        {
                            "patch_id": [0, 0],
                            "row_id": 0,
                            "col_id": 0,
                            "is_selected": True,
                            "thumbnail_top_left_x": 0,
                            "thumbnail_top_left_y": 0,
                            "thumbnail_width": 100,
                            "thumbnail_height": 100,
                        },
                        {
                            "patch_id": [0, 1],
                            "row_id": 0,
                            "col_id": 1,
                            "is_selected": True,
                            "thumbnail_top_left_x": 110,
                            "thumbnail_top_left_y": 0,
                            "thumbnail_width": 100,
                            "thumbnail_height": 100,
                        },
                    ],
                },
                "trace_rubric": {},
                "overview_image": {"width": 220, "height": 100, "image_url": ""},
                "patch_assignments": [
                    {
                        "patch_id": [0, 0],
                        "row": 0,
                        "col": 0,
                        "bbox_level0": [0, 0, 256, 256],
                        "score": 0.92,
                        "region_semantic": "ssl_suspicious_mucosa",
                        "diagnostic_priority": 4,
                        "require_high_magnification": True,
                        "agreement_status": "strong_agreement",
                        "conch_region_semantic": "ssl_suspicious_mucosa",
                        "pathoreasoner_r1_region_semantic": "ssl_suspicious_mucosa",
                        "fusion_reasoning": "stable consensus",
                        "high_mag_ref": "p00",
                    },
                    {
                        "patch_id": [2, 2],
                        "row": 2,
                        "col": 2,
                        "bbox_level0": [512, 512, 768, 768],
                        "score": 0.51,
                        "region_semantic": "conventional_adenoma_like",
                        "diagnostic_priority": 3,
                        "require_high_magnification": True,
                        "agreement_status": "risk_disagreement",
                        "conch_region_semantic": "conventional_adenoma_like",
                        "pathoreasoner_r1_region_semantic": "ssl_suspicious_mucosa",
                        "fusion_reasoning": "serrated vs conventional conflict",
                        "high_mag_ref": "p22",
                    },
                ],
                "high_mag_assets": [],
                "audit_log": [],
            }
            export_dashboard(payload, output_dir)
            snapshot = json.loads((output_dir / "dashboard_payload.json").read_text(encoding="utf-8"))
            violations = snapshot["contract_validation"]["violations"]
            violation_types = sorted(item["type"] for item in violations)
            self.assertIn("missing_patch", violation_types)
            self.assertIn("out_of_bounds_patch", violation_types)
            self.assertTrue(snapshot["orphan_assets"])

    def test_export_dashboard_copies_local_overview_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            output_dir = root / "dashboard"
            source_dir.mkdir()
            overview_svg = source_dir / "overview.svg"
            overview_svg.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="80"><rect width="100" height="80" fill="#eee"/></svg>',
                encoding="utf-8",
            )
            payload = build_demo_dashboard_payload()
            payload["overview_image"] = {
                "image_path": str(overview_svg),
                "width": 100,
                "height": 80,
            }
            export_dashboard(payload, output_dir, source_root=source_dir)
            self.assertTrue((output_dir / "assets" / "overview" / "overview.svg").exists())

    def test_build_dashboard_payload_from_harness_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "batch_run" / "case_001"
            (case_dir / "trace" / "grid_input").mkdir(parents=True)
            (case_dir / "observe" / "step_00").mkdir(parents=True)
            (case_dir / "navigation").mkdir(parents=True)

            (case_dir / "trace" / "grid_input" / "case_001_grid.jpg").write_bytes(b"fake-grid")
            (case_dir / "observe" / "step_00" / "step_00__detail_mag5.0_256.png").write_bytes(b"fake-detail")

            grid_meta = {
                "grid_rows": 1,
                "grid_cols": 2,
                "cropped_thumbnail_size": [220, 100],
                "selected_patch_count": 2,
                "grid_cells": [
                    {
                        "patch_id": [0, 0],
                        "row_id": 0,
                        "col_id": 0,
                        "is_selected": True,
                        "thumbnail_top_left_x": 0,
                        "thumbnail_top_left_y": 0,
                        "thumbnail_width": 100,
                        "thumbnail_height": 100,
                        "level0_top_left_x": 0,
                        "level0_top_left_y": 0,
                        "level0_width": 256,
                        "level0_height": 256,
                    },
                    {
                        "patch_id": [0, 1],
                        "row_id": 0,
                        "col_id": 1,
                        "is_selected": True,
                        "thumbnail_top_left_x": 110,
                        "thumbnail_top_left_y": 0,
                        "thumbnail_width": 100,
                        "thumbnail_height": 100,
                        "level0_top_left_x": 256,
                        "level0_top_left_y": 0,
                        "level0_width": 256,
                        "level0_height": 256,
                    },
                ],
            }
            (case_dir / "trace" / "grid_input" / "case_001_grid.json").write_text(json.dumps(grid_meta), encoding="utf-8")
            (case_dir / "trace" / "case_001_grid_input_boxes.json").write_text(
                json.dumps(
                    {
                        "boxes": [
                            {"patch_id": [0, 0], "score": 0.77},
                            {"patch_id": [0, 1], "score": 0.65},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "trace" / "trace_clusters.json").write_text(
                json.dumps(
                    {
                        "clusters": [
                            {
                                "cluster_id": "grid_group_00",
                                "l": "ssl_suspicious_mucosa",
                                "s": 4,
                                "d": True,
                                "review_stage": "serrated_screening",
                                "desc": "Possible serrated pattern.",
                                "patch_ids_ordered": [[0, 0], [0, 1]],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "observe" / "observation_records.json").write_text(
                json.dumps(
                    {
                        "observations": [
                            {
                                "step_id": "step_00",
                                "crop_path": "/stale/run/root/step_00__detail_mag5.0_256.png",
                                "confidence": 0.62,
                                "stage_decision": "supports_serrated_lesion",
                                "metadata": {
                                    "cluster_id": "grid_group_00",
                                    "backend": "local_cpathagent_qwen",
                                    "review_goal": "serrated_lesion_assessment",
                                    "bundle_image_paths": [
                                        "/stale/run/root/step_00__detail_mag5.0_256.png"
                                    ],
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "observe" / "pathological_report.json").write_text(
                json.dumps(
                    {
                        "integrated_report": {
                            "summary": "SSL-like mucosa requires closer review.",
                            "recommendations": ["Follow serrated branch."],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "navigation" / "navigation_steps.json").write_text(json.dumps({"steps": [{"step_id": "step_00"}]}), encoding="utf-8")
            (case_dir / "case_result.json").write_text(json.dumps({"case_id": "case_001"}), encoding="utf-8")
            (case_dir / "run_metadata.json").write_text(
                json.dumps({"case": {"case_id": "case_001", "slide_id": "slide_001", "input_mode": "grid_thumbnail"}}),
                encoding="utf-8",
            )

            payload = build_dashboard_payload_from_harness_case(case_dir)
            self.assertEqual(payload["case_id"], "case_001")
            self.assertEqual(payload["slide_id"], "slide_001")
            self.assertEqual(len(payload["patch_assignments"]), 2)
            self.assertEqual(payload["patch_assignments"][0]["score_origin"], "normalized_diagnostic_priority")
            self.assertEqual(payload["patch_assignments"][0]["agreement_status"], "single_model_trace")
            self.assertTrue(payload["high_mag_assets"])
            self.assertIn("step_00__detail_mag5.0_256.png", payload["high_mag_assets"][0]["image_path"])

    def test_build_dashboard_payload_from_harness_case_preserves_dual_model_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "batch_run" / "case_002"
            (case_dir / "trace" / "grid_input").mkdir(parents=True)
            (case_dir / "observe" / "step_00").mkdir(parents=True)
            (case_dir / "navigation").mkdir(parents=True)

            (case_dir / "trace" / "grid_input" / "case_002_grid.jpg").write_bytes(b"fake-grid")
            (case_dir / "observe" / "step_00" / "step_00__detail_mag5.0_256.png").write_bytes(b"fake-detail")

            grid_meta = {
                "grid_rows": 1,
                "grid_cols": 2,
                "cropped_thumbnail_size": [220, 100],
                "selected_patch_count": 2,
                "grid_cells": [
                    {
                        "patch_id": [0, 0],
                        "row_id": 0,
                        "col_id": 0,
                        "is_selected": True,
                        "thumbnail_top_left_x": 0,
                        "thumbnail_top_left_y": 0,
                        "thumbnail_width": 100,
                        "thumbnail_height": 100,
                        "level0_top_left_x": 0,
                        "level0_top_left_y": 0,
                        "level0_width": 256,
                        "level0_height": 256,
                    },
                    {
                        "patch_id": [0, 1],
                        "row_id": 0,
                        "col_id": 1,
                        "is_selected": True,
                        "thumbnail_top_left_x": 110,
                        "thumbnail_top_left_y": 0,
                        "thumbnail_width": 100,
                        "thumbnail_height": 100,
                        "level0_top_left_x": 256,
                        "level0_top_left_y": 0,
                        "level0_width": 256,
                        "level0_height": 256,
                    },
                ],
            }
            (case_dir / "trace" / "grid_input" / "case_002_grid.json").write_text(json.dumps(grid_meta), encoding="utf-8")
            (case_dir / "trace" / "trace_clusters.json").write_text(
                json.dumps(
                    {
                        "clusters": [
                            {
                                "cluster_id": "grid_group_00",
                                "l": "ssl_suspicious_mucosa",
                                "s": 5,
                                "d": True,
                                "review_stage": "serrated_screening",
                                "desc": "Consensus serrated cluster.",
                                "patch_ids_ordered": [[0, 0]],
                            },
                            {
                                "cluster_id": "grid_group_01",
                                "l": "normal_mucosa",
                                "s": 1,
                                "d": False,
                                "review_stage": "mucosa_screening",
                                "desc": "Consensus low-value cluster.",
                                "patch_ids_ordered": [[0, 1]],
                            },
                        ],
                        "patch_assignments": {
                            "patches": [
                                {
                                    "patch_id": [0, 0],
                                    "region_semantic": "ssl_suspicious_mucosa",
                                    "conch_region_semantic": "ssl_suspicious_mucosa",
                                    "pathoreasoner_r1_region_semantic": "ssl_suspicious_mucosa",
                                    "agreement_status": "strong_agreement",
                                    "diagnostic_priority": 5,
                                    "require_high_magnification": True,
                                    "score_origin": "consensus_fusion",
                                    "fusion_reasoning": "CONCH and PathoReasoner-R1 both selected ssl_suspicious_mucosa; consensus keeps ssl_suspicious_mucosa with priority 5.",
                                    "score": 1.0,
                                },
                                {
                                    "patch_id": [0, 1],
                                    "region_semantic": "normal_mucosa",
                                    "conch_region_semantic": "normal_mucosa",
                                    "pathoreasoner_r1_region_semantic": "background_artifact_stroma",
                                    "agreement_status": "low_value_disagreement",
                                    "diagnostic_priority": 1,
                                    "require_high_magnification": False,
                                    "score_origin": "consensus_fusion",
                                    "fusion_reasoning": "CONCH=normal_mucosa and PathoReasoner-R1=background_artifact_stroma disagree only in low-value territory; conservative fusion keeps normal_mucosa at priority 1.",
                                    "score": 0.2,
                                },
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "observe" / "observation_records.json").write_text(
                json.dumps(
                    {
                        "observations": [
                            {
                                "step_id": "step_00",
                                "crop_path": "/stale/run/root/step_00__detail_mag5.0_256.png",
                                "confidence": 0.62,
                                "stage_decision": "supports_serrated_lesion",
                                "metadata": {
                                    "cluster_id": "grid_group_00",
                                    "backend": "local_cpathagent_qwen",
                                    "review_goal": "serrated_lesion_assessment",
                                    "bundle_image_paths": [
                                        "/stale/run/root/step_00__detail_mag5.0_256.png"
                                    ],
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "observe" / "pathological_report.json").write_text(
                json.dumps(
                    {
                        "integrated_report": {
                            "summary": "Dual model consensus supported.",
                            "recommendations": ["Continue branch routing."],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "navigation" / "navigation_steps.json").write_text(json.dumps({"steps": [{"step_id": "step_00"}]}), encoding="utf-8")
            (case_dir / "case_result.json").write_text(json.dumps({"case_id": "case_002"}), encoding="utf-8")
            (case_dir / "run_metadata.json").write_text(
                json.dumps({"case": {"case_id": "case_002", "slide_id": "slide_002", "input_mode": "grid_thumbnail"}}),
                encoding="utf-8",
            )

            payload = build_dashboard_payload_from_harness_case(case_dir)
            self.assertEqual(payload["case_id"], "case_002")
            self.assertEqual(payload["trace_rubric"]["mode"], "dynamic_priority_consensus")
            self.assertEqual(len(payload["patch_assignments"]), 2)
            self.assertEqual(payload["patch_assignments"][0]["agreement_status"], "strong_agreement")
            self.assertEqual(payload["patch_assignments"][0]["score_origin"], "consensus_fusion")
            self.assertEqual(payload["patch_assignments"][0]["diagnostic_priority"], 5)
            self.assertEqual(payload["patch_assignments"][1]["agreement_status"], "low_value_disagreement")
            self.assertEqual(payload["patch_assignments"][1]["score_origin"], "consensus_fusion")
            self.assertEqual(payload["patch_assignments"][1]["conch_region_semantic"], "normal_mucosa")


if __name__ == "__main__":
    unittest.main()
