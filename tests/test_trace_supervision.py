import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from adenoma_agent.trace_supervision import (
    TRACE_LABEL_RUBRIC,
    build_candidate_diff_view,
    build_image_only_teacher_prompt,
    candidate_label_agreement,
    export_candidate_review_package,
    parse_patch_assignment_response,
    score_trace_auto_review,
    score_trace_case,
    select_best_candidate,
)


class TraceSupervisionTest(unittest.TestCase):
    def _grid_meta(self):
        return {
            "grid_rows": 1,
            "grid_cols": 2,
            "n_selected_cells": 2,
            "grid_cells": [
                {"row_id": 0, "col_id": 0, "is_selected": True, "tissue_coverage_ratio": 0.8},
                {"row_id": 0, "col_id": 1, "is_selected": True, "tissue_coverage_ratio": 0.6},
            ],
        }

    def _payload(self, label0="normal_mucosa", label1="background_artifact_stroma"):
        return {
            "patches": [
                {
                    "patch_id": [0, 0],
                    "region_semantic": label0,
                    "name": "Reviewable normal mucosa",
                    "description": "Uniform crypt reviewable mucosa.",
                    "require_high_magnification": False,
                    "severity_reasoning": "Low-priority benign architecture check.",
                    "diagnostic_priority": 1,
                    "observation_points": ["low-priority non-lesional mucosa"],
                },
                {
                    "patch_id": [0, 1],
                    "region_semantic": label1,
                    "name": "Masked background",
                    "description": "Black masked low-value background or artifact.",
                    "require_high_magnification": False,
                    "severity_reasoning": "Coverage-preserving discard group.",
                    "diagnostic_priority": 0,
                    "observation_points": ["coverage-preserving discard group"],
                },
            ]
        }

    def test_no_report_prompt_uses_image_only_screening_contract(self):
        request = {
            "selected_patch_ids": [[0, 0], [0, 1]],
            "trace_rubric": TRACE_LABEL_RUBRIC,
            "agent_vocabulary": {"summary": "Use pathology workflow terminology."},
            "grid_metadata_summary": {"black_mask_hint": "Background has been black-masked."},
        }
        prompt = build_image_only_teacher_prompt(request)
        self.assertIn("screening-level only", prompt)
        self.assertIn("no final diagnosis", prompt)
        self.assertIn("image-only morphology", prompt)
        self.assertIn("[[0,0],[0,1]]", prompt)
        self.assertIn('"patches"', prompt)
        self.assertIn("black background is intentional masking", prompt)
        self.assertIn("Adenoma agent vocabulary", prompt)
        self.assertNotIn("teacher_uncertainty", prompt)
        self.assertNotIn("visual_evidence_strength", prompt)

    def test_parse_teacher_response_handles_code_fence_and_prose(self):
        text = "Here is the JSON:\n```json\n{\"patches\":[{\"patch_id\":[0,0]}]}\n```"
        parsed = parse_patch_assignment_response(text)
        self.assertFalse(parsed["parse_failure"])
        self.assertEqual(parsed["payload"]["patches"][0]["patch_id"], [0, 0])

    def test_parse_teacher_response_flags_non_json(self):
        parsed = parse_patch_assignment_response("not json at all")
        self.assertTrue(parsed["parse_failure"])
        self.assertEqual(parsed["payload"], {"patches": []})

    def test_candidate_agreement_and_diff_detect_label_disagreement(self):
        first = self._payload("normal_mucosa")
        second = self._payload("ssl_suspicious_mucosa")
        agreement = candidate_label_agreement([first, second])
        self.assertEqual(agreement["compared_patch_count"], 2)
        self.assertLess(agreement["agreement_rate"], 1.0)
        diff = build_candidate_diff_view([first, second])
        self.assertEqual(diff["disagreement_count"], 1)
        self.assertEqual(diff["rows"][0]["patch_id"], [0, 0])

    def test_export_candidate_review_package_writes_review_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "case_grid.jpg"
            meta_path = root / "case_grid.json"
            image_path.write_bytes(b"fake-image")
            meta_path.write_text(json.dumps(self._grid_meta()), encoding="utf-8")
            payload_a = self._payload("normal_mucosa")
            payload_b = self._payload("normal_mucosa")
            candidate_results = [
                {
                    "candidate_index": 0,
                    "payload": payload_a,
                    "score": score_trace_case(payload_a, self._grid_meta()),
                    "parse_failure": False,
                },
                {
                    "candidate_index": 1,
                    "payload": payload_b,
                    "score": score_trace_case(payload_b, self._grid_meta()),
                    "parse_failure": False,
                },
            ]
            selection = select_best_candidate(candidate_results)
            review_dir = export_candidate_review_package(
                "case_grid",
                image_path,
                meta_path,
                candidate_results,
                root / "review",
                selection=selection,
            )
            self.assertTrue((review_dir / "gemini_candidates.json").exists())
            self.assertTrue((review_dir / "candidate_diff.json").exists())
            self.assertTrue((review_dir / "index.html").exists())
            review_target = json.loads((review_dir / "review_target.json").read_text(encoding="utf-8"))
            self.assertFalse(review_target["report_available"])
            self.assertEqual(review_target["teacher_mode"], "image_only_screening_trace")
            self.assertIn("target", review_target)

    def test_auto_review_emits_slide_label_warning_and_field_consistency(self):
        grid_meta = self._grid_meta()
        payload = {
            "patches": [
                {
                    "patch_id": [0, 0],
                    "region_semantic": "normal_mucosa",
                    "name": "Benign mucosa",
                    "description": "Uniform crypt benign mucosa.",
                    "require_high_magnification": False,
                    "severity_reasoning": "Benign architecture.",
                    "diagnostic_priority": 2,
                    "observation_points": ["confirm benign architecture if sampled"],
                },
                {
                    "patch_id": [0, 1],
                    "region_semantic": "background_artifact_stroma",
                    "name": "Masked background",
                    "description": "Black masked low-value background.",
                    "require_high_magnification": False,
                    "severity_reasoning": "Discard group.",
                    "diagnostic_priority": 0,
                    "observation_points": ["coverage-preserving discard group"],
                },
            ]
        }
        candidate_results = [
            {
                "candidate_index": 0,
                "payload": payload,
                "score": score_trace_case(payload, grid_meta),
                "parse_failure": False,
            },
            {
                "candidate_index": 1,
                "payload": payload,
                "score": score_trace_case(payload, grid_meta),
                "parse_failure": False,
            },
        ]
        auto_review = score_trace_auto_review(
            candidate_results,
            grid_meta,
            slide_label_context={"label": "Sessile serrated adenoma", "serrated_target": 1, "metadata": {}},
        )
        selected = auto_review["selected_candidate"]
        self.assertEqual(auto_review["selection"]["review_status"], "needs_review")
        self.assertIn("slide_label_consistency_risk", auto_review["selection"]["review_reason"])
        warnings = [item["warning"] for item in selected["score"]["field_consistency"]["warnings"]]
        self.assertIn("normal_priority_too_high", warnings)
        slide_warnings = [item["warning"] for item in selected["score"]["slide_label_consistency"]["warnings"]]
        self.assertIn("ssl_absent_but_slide_positive_risk", slide_warnings)

    def test_export_candidate_review_package_includes_auto_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "case_grid.jpg"
            meta_path = root / "case_grid.json"
            image_path.write_bytes(b"fake-image")
            meta_path.write_text(json.dumps(self._grid_meta()), encoding="utf-8")
            payload_a = self._payload("normal_mucosa")
            payload_b = self._payload("ssl_suspicious_mucosa")
            candidate_results = [
                {
                    "candidate_index": 0,
                    "payload": payload_a,
                    "score": score_trace_case(payload_a, self._grid_meta()),
                    "parse_failure": False,
                },
                {
                    "candidate_index": 1,
                    "payload": payload_b,
                    "score": score_trace_case(payload_b, self._grid_meta()),
                    "parse_failure": False,
                },
            ]
            auto_review = score_trace_auto_review(candidate_results, self._grid_meta(), slide_label_context={"label": "Sessile serrated adenoma", "serrated_target": 1})
            review_dir = export_candidate_review_package(
                "case_grid",
                image_path,
                meta_path,
                candidate_results,
                root / "review",
                selection=auto_review["selection"],
                auto_review=auto_review,
            )
            review_target = json.loads((review_dir / "review_target.json").read_text(encoding="utf-8"))
            self.assertIn("auto_review", review_target)
            self.assertIn("candidate_agreement", review_target)
            html = (review_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Global Screening Review", html)
            self.assertIn("Patch comparison", html)

    def test_summarize_trace_auto_review_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "case_a"
            auto_checked = case_dir / "auto_checked"
            review_dir = case_dir / "review"
            auto_checked.mkdir(parents=True)
            review_dir.mkdir(parents=True)
            auto_review = {
                "selection": {"review_status": "needs_review", "review_reason": "candidate_disagreement", "selected_index": 0},
                "candidate_agreement": {"agreement_rate": 0.5},
                "candidates": [{"parse_failure": False}],
                "selected_candidate": {
                    "score": {
                        "total_score": 92,
                        "structure": {"coverage_ok": True, "missing_patch_ids": [], "duplicate_patch_ids": [], "unexpected_patch_ids": []},
                        "semantics": {"semantic_score": 90},
                        "field_consistency": {"field_consistency_score": 100},
                        "aggregation": {"aggregation_score": 100},
                        "slide_label_consistency": {"slide_label": "Sessile serrated adenoma", "warnings": [{"warning": "ssl_absent_but_slide_positive_risk"}]},
                    }
                },
            }
            (auto_checked / "auto_review.json").write_text(json.dumps(auto_review), encoding="utf-8")
            (review_dir / "review_target.json").write_text(json.dumps({"human_reviewed": False}), encoding="utf-8")
            output_json = root / "summary.json"
            subprocess.check_call(
                [
                    "python3",
                    "scripts/summarize_trace_auto_review.py",
                    "--run-root",
                    str(root),
                    "--output-json",
                    str(output_json),
                ],
                cwd="/data1/yuexin/Adenoma",
            )
            summary = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["case_count"], 1)
            self.assertEqual(summary["coverage_ok_count"], 1)
            self.assertEqual(summary["slide_label_risk_count"], 1)
            self.assertEqual(summary["review_reason_distribution"]["candidate_disagreement"], 1)

    def test_retrofit_trace_auto_review_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "case_a"
            auto_checked = case_dir / "auto_checked"
            review_dir = case_dir / "review"
            auto_checked.mkdir(parents=True)
            review_dir.mkdir(parents=True)
            image_path = review_dir / "case_a.jpg"
            meta_path = review_dir / "case_a.json"
            image_path.write_bytes(b"fake-image")
            meta_path.write_text(json.dumps(self._grid_meta()), encoding="utf-8")
            payload_a = self._payload("normal_mucosa")
            payload_b = self._payload("ssl_suspicious_mucosa")
            bundle = {
                "candidates": [
                    {
                        "candidate_index": 0,
                        "payload": payload_a,
                        "parse_failure": False,
                        "score": score_trace_case(payload_a, self._grid_meta()),
                    },
                    {
                        "candidate_index": 1,
                        "payload": payload_b,
                        "parse_failure": False,
                        "score": score_trace_case(payload_b, self._grid_meta()),
                    },
                ],
                "selection": {"selected_index": 0, "review_status": "needs_review"},
            }
            (auto_checked / "qwen_candidates.json").write_text(json.dumps(bundle), encoding="utf-8")
            (review_dir / "review_target.json").write_text(
                json.dumps(
                    {
                        "case_id": "case_a",
                        "teacher_mode": "image_only_screening_trace",
                        "report_available": False,
                        "review_status": "needs_review",
                        "human_reviewed": False,
                        "selected_candidate_index": 0,
                        "target": payload_a,
                    }
                ),
                encoding="utf-8",
            )
            subprocess.check_call(
                [
                    "python3",
                    "scripts/retrofit_trace_auto_review.py",
                    "--run-root",
                    str(root),
                ],
                cwd="/data1/yuexin/Adenoma",
            )
            self.assertTrue((auto_checked / "auto_review.json").exists())
            self.assertTrue((review_dir / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
