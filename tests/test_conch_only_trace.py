import json
import tempfile
import unittest
from pathlib import Path

from adenoma_agent.multimodal import (
    build_conch_digepath_trace_output,
    build_conch_only_trace_output,
    parse_digepath_prediction_payload,
    parse_conch_prediction_payload,
)


class ConchOnlyTraceTest(unittest.TestCase):
    def _write_grid(self, root):
        image_path = Path(root) / "case_grid.jpg"
        image_path.write_bytes(b"fake")
        grid_path = image_path.with_suffix(".json")
        grid_path.write_text(
            json.dumps(
                {
                    "thumbnail_mode": "tissue_grid32x_svs",
                    "grid_rows": 1,
                    "grid_cols": 3,
                    "n_selected_cells": 3,
                    "grid_cells": [
                        {
                            "row_id": 0,
                            "col_id": 0,
                            "is_selected": True,
                            "thumbnail_top_left_x": 0,
                            "thumbnail_top_left_y": 0,
                            "thumbnail_width": 64,
                            "thumbnail_height": 64,
                            "level0_top_left_x": 0,
                            "level0_top_left_y": 0,
                            "level0_width": 2048,
                            "level0_height": 2048,
                            "tissue_coverage_ratio": 0.8,
                        },
                        {
                            "row_id": 0,
                            "col_id": 1,
                            "is_selected": True,
                            "thumbnail_top_left_x": 64,
                            "thumbnail_top_left_y": 0,
                            "thumbnail_width": 64,
                            "thumbnail_height": 64,
                            "level0_top_left_x": 2048,
                            "level0_top_left_y": 0,
                            "level0_width": 2048,
                            "level0_height": 2048,
                            "tissue_coverage_ratio": 0.7,
                        },
                        {
                            "row_id": 0,
                            "col_id": 2,
                            "is_selected": True,
                            "thumbnail_top_left_x": 128,
                            "thumbnail_top_left_y": 0,
                            "thumbnail_width": 64,
                            "thumbnail_height": 64,
                            "level0_top_left_x": 4096,
                            "level0_top_left_y": 0,
                            "level0_width": 2048,
                            "level0_height": 2048,
                            "tissue_coverage_ratio": 0.6,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return image_path

    def test_parse_crc100k_predictions_preserves_details(self):
        parsed = parse_conch_prediction_payload(
            {
                "predictions": [
                    {
                        "patch_id": [0, 0],
                        "label": "TUM",
                        "probabilities": {"TUM": 0.91, "NORM": 0.04},
                        "embedding_ref": "features.npy:0",
                    },
                    {"patch_id": [0, 1], "class": "MUC", "confidence": 0.72},
                ]
            }
        )
        self.assertEqual(parsed["predictions"]["0,0"], "epithelial_neoplasia_suspicious")
        self.assertEqual(parsed["predictions"]["0,1"], "mucus_rich_or_pale_context")
        self.assertEqual(parsed["prediction_details"]["0,0"]["conch_crc_label"], "TUM")
        self.assertEqual(parsed["prediction_details"]["0,0"]["conch_probs"]["TUM"], 0.91)
        self.assertEqual(parsed["prediction_details"]["0,0"]["embedding_ref"], "features.npy:0")

    def test_build_conch_only_trace_output_uses_neutral_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid(tmpdir)
            request = {"images": [str(image_path)], "metadata": {}}
            conch = parse_conch_prediction_payload(
                {
                    "predictions": [
                        {"patch_id": [0, 0], "label": "TUM", "confidence": 0.91},
                        {"patch_id": [0, 1], "label": "MUC", "confidence": 0.80},
                        {"patch_id": [0, 2], "label": "BACK", "confidence": 0.95},
                    ]
                }
            )
            conch["enabled"] = True
            conch["errors"] = []
            output = build_conch_only_trace_output(request, conch)
            labels = [patch["region_semantic"] for patch in output["patch_assignments"]["patches"]]
            self.assertEqual(
                labels,
                [
                    "epithelial_neoplasia_suspicious",
                    "mucus_rich_or_pale_context",
                    "background_or_artifact",
                ],
            )
            first = output["patch_assignments"]["patches"][0]
            self.assertEqual(first["candidate_branches"], ["serrated", "conventional"])
            self.assertEqual(first["routing_hint"], "needs_morphology_resolution")
            self.assertEqual(output["coverage_summary"]["trace_mode"], "conch_only")

    def test_parse_digepath_roi9_predictions_preserves_details(self):
        parsed = parse_digepath_prediction_payload(
            {
                "predictions": [
                    {
                        "patch_id": [0, 0],
                        "label": "tumor_epithelium",
                        "probabilities": {"tumor_epithelium": 0.93, "stroma": 0.04},
                    },
                    {"patch_id": [0, 1], "class": "mucus", "confidence": 0.82},
                    {"patch_id": [0, 2], "label": "not_a_crc100k_class"},
                ]
            }
        )
        self.assertEqual(parsed["predictions"]["0,0"], "epithelial_neoplasia_suspicious")
        self.assertEqual(parsed["predictions"]["0,1"], "mucus_rich_or_pale_context")
        self.assertEqual(parsed["prediction_details"]["0,0"]["digepath_class"], "tumor_epithelium")
        self.assertEqual(parsed["prediction_details"]["0,0"]["digepath_probs"]["tumor_epithelium"], 0.93)
        self.assertEqual(len(parsed["invalid_predictions"]), 1)

    def test_build_conch_digepath_trace_output_fuses_risk_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid(tmpdir)
            request = {"images": [str(image_path)], "metadata": {}}
            conch = parse_conch_prediction_payload(
                {
                    "predictions": [
                        {"patch_id": [0, 0], "label": "TUM", "confidence": 0.91},
                        {"patch_id": [0, 1], "label": "BACK", "confidence": 0.95},
                        {"patch_id": [0, 2], "label": "BACK", "confidence": 0.96},
                    ]
                }
            )
            conch["enabled"] = True
            conch["errors"] = []
            digepath = parse_digepath_prediction_payload(
                {
                    "predictions": [
                        {"patch_id": [0, 0], "label": "tumor_epithelium", "confidence": 0.93},
                        {"patch_id": [0, 1], "label": "tumor_epithelium", "confidence": 0.88},
                        {"patch_id": [0, 2], "label": "background", "confidence": 0.99},
                    ]
                }
            )
            digepath["enabled"] = True
            digepath["errors"] = []
            digepath["high_confidence_threshold"] = 0.70
            output = build_conch_digepath_trace_output(request, conch, digepath)
            patches = output["patch_assignments"]["patches"]
            self.assertEqual(patches[0]["region_semantic"], "epithelial_neoplasia_suspicious")
            self.assertEqual(patches[0]["agreement_status"], "conch_digepath_agree")
            self.assertEqual(patches[0]["score_origin"], "conch_digepath_fusion")
            self.assertEqual(patches[0]["digepath_class"], "tumor_epithelium")
            self.assertEqual(patches[1]["region_semantic"], "uncertain_reviewable_mucosa")
            self.assertEqual(patches[1]["agreement_status"], "conch_digepath_disagree")
            self.assertEqual(patches[2]["region_semantic"], "background_or_artifact")
            self.assertEqual(patches[2]["agreement_status"], "conch_digepath_agree")
            self.assertTrue(output["coverage_summary"]["digepath_fusion_enabled"])

    def test_build_conch_digepath_trace_output_gates_normal_agreement_by_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_grid(tmpdir)
            request = {"images": [str(image_path)], "metadata": {}}
            conch = parse_conch_prediction_payload(
                {
                    "predictions": [
                        {"patch_id": [0, 0], "label": "NORM", "confidence": 0.99},
                        {"patch_id": [0, 1], "label": "NORM", "confidence": 0.99},
                        {"patch_id": [0, 2], "label": "BACK", "confidence": 0.96},
                    ]
                }
            )
            conch["enabled"] = True
            conch["errors"] = []
            digepath = parse_digepath_prediction_payload(
                {
                    "predictions": [
                        {"patch_id": [0, 0], "label": "normal_colon_mucosa", "confidence": 0.95},
                        {"patch_id": [0, 1], "label": "normal_colon_mucosa", "confidence": 0.62},
                        {"patch_id": [0, 2], "label": "normal_colon_mucosa", "confidence": 0.96},
                    ]
                }
            )
            digepath["enabled"] = True
            digepath["errors"] = []
            digepath["normal_conch_confidence_threshold"] = 0.90
            digepath["normal_digepath_confidence_threshold"] = 0.90
            digepath["normal_gate_fallback_label"] = "uncertain_reviewable_mucosa"
            output = build_conch_digepath_trace_output(request, conch, digepath)
            patches = output["patch_assignments"]["patches"]
            self.assertEqual(patches[0]["region_semantic"], "reviewable_normal_mucosa")
            self.assertEqual(patches[1]["region_semantic"], "uncertain_reviewable_mucosa")
            self.assertEqual(patches[1]["agreement_status"], "conch_digepath_agree")
            self.assertEqual(patches[2]["region_semantic"], "uncertain_reviewable_mucosa")
            self.assertEqual(patches[2]["agreement_status"], "conch_digepath_disagree")


if __name__ == "__main__":
    unittest.main()
