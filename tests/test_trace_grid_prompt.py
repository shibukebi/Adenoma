import json
import tempfile
import unittest
from pathlib import Path

from adenoma_agent.multimodal import (
    HeuristicStageBackend,
    _build_trace_output_from_text,
    _build_trace_patho_r1_prompt,
    _normalize_trace_label,
    _run_trace_grid_with_coverage_retry,
)
from adenoma_agent.trace_supervision import score_trace_case


class TraceGridPromptTest(unittest.TestCase):
    def _write_grid_metadata(self, root):
        image_path = Path(root) / "case_001_tissuegrid125_ds32_level2_grid.jpg"
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
            warning_names = [item["warning"] for item in score["warnings"] if "warning" in item]
            self.assertIn("ssl_priority_highmag_conflict", warning_names)

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
                "magnification_to_region_size": {"5.0": 128, "20.0": 64},
            },
        }
        output = backend.invoke(request, bundle)["output"]
        inspect_steps = [step for step in output["steps"] if step["metadata"].get("action") == "inspect"]
        self.assertEqual(
            [(step["metadata"]["cluster_id"], step["metadata"]["patch_id"], step["m"]) for step in inspect_steps],
            [
                ("cluster_high", [0, 0], 5.0),
                ("cluster_high", [0, 0], 20.0),
                ("cluster_high", [1, 1], 5.0),
                ("cluster_high", [1, 1], 20.0),
                ("cluster_low", [2, 0], 5.0),
            ],
        )

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
                "magnification_to_region_size": {"5.0": 128, "20.0": 64},
            },
        }
        output = backend.invoke(request, bundle)["output"]
        inspect_steps = [step for step in output["steps"] if step["metadata"].get("action") == "inspect"]
        self.assertEqual(len(inspect_steps), 1)
        self.assertEqual(inspect_steps[0]["review_goal"], "conventional_adenoma_assessment")
        self.assertEqual(inspect_steps[0]["stage_gate"], "conventional_adenoma")
        self.assertEqual(inspect_steps[0]["metadata"]["workflow_branch"], "conventional")


if __name__ == "__main__":
    unittest.main()
