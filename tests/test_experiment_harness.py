import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from adenoma_agent.experiment_harness import (
    discover_grid_cases,
    publish_harness_dashboard,
    summarize_case_artifacts,
    write_experiment_summaries,
)
from adenoma_agent.utils import write_json


class ExperimentHarnessTests(unittest.TestCase):
    def test_discover_grid_cases_matches_wsi_and_25x_grid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wsi_dir = root / "wsi"
            grid_dir = root / "grid"
            wsi_dir.mkdir()
            grid_dir.mkdir()
            (wsi_dir / "caseA.svs").write_text("placeholder", encoding="utf-8")
            (wsi_dir / "caseB.svs").write_text("placeholder", encoding="utf-8")
            (grid_dir / "caseA_tissuegrid125_ds32_level2_grid.jpg").write_text("jpg", encoding="utf-8")
            write_json(grid_dir / "caseA_tissuegrid125_ds32_level2_grid.json", {"slide_id": "caseA"})

            cases = discover_grid_cases(wsi_dir, grid_dir)

        self.assertEqual([case["case_id"] for case in cases], ["caseA"])
        self.assertTrue(cases[0]["grid_thumbnail_path"].endswith("_grid.jpg"))

    def test_summary_marks_fallback_and_updates_case_result_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "caseA"
            (case_dir / "navigation").mkdir(parents=True)
            (case_dir / "observe/chief_reviews").mkdir(parents=True)
            write_json(
                case_dir / "case_result.json",
                {"case_id": "caseA", "status": "ok", "timing": {"total_runtime_ms": 12}, "metadata": {}},
            )
            write_json(
                case_dir / "navigation/navigation_steps.json",
                {"backend_attempts": [{"backend": "local_cpathagent_qwen", "status": "error"}, {"backend": "heuristic", "status": "ok"}]},
            )
            write_json(
                case_dir / "observe/chief_reviews/cellA_chief_response.json",
                {
                    "review_source": "chief_model",
                    "raw_generated_text": "prompt echo ```json {\"decision\":\"continue\"}```",
                    "answer_candidate_text": "{\"decision\":\"continue\"}",
                    "chief_confidence": 0.5,
                },
            )

            row = summarize_case_artifacts(case_dir)
            result = json.loads((case_dir / "case_result.json").read_text(encoding="utf-8"))

        self.assertEqual(row["quality_tier"], "partial_fallback")
        self.assertIn("navigation", row["fallback_stages"])
        self.assertGreater(row["normalization_count"], 0)
        self.assertEqual(result["metadata"]["harness_quality"]["quality_tier"], "partial_fallback")

    def test_write_experiment_summaries_serializes_tsv_without_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {
                    "case_id": "caseA",
                    "case_dir": str(Path(tmp) / "caseA"),
                    "final_status": "ok",
                    "recovery_status": "ok",
                    "quality_tier": "clean_real_model",
                    "fallback_stages": [],
                    "normalization_count": 0,
                    "normalization_actions": [],
                    "repair_count": 0,
                    "repair_stages": [],
                    "repair_actions": [],
                    "failed_stage_history": [],
                    "attempt_count": 1,
                    "case_result_path": str(Path(tmp) / "caseA/case_result.json"),
                    "total_runtime_ms": 5,
                }
            ]

            summary = write_experiment_summaries(Path(tmp), rows, [{"service": "mock"}], {"a": 1}, {"b": 2})

            tsv = (Path(tmp) / "experiment_summary.tsv").read_text(encoding="utf-8")

        self.assertEqual(summary["case_count"], 1)
        self.assertIn("clean_real_model", tsv)

    def test_publish_harness_dashboard_exports_and_records_frontend_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            output_dir = root / "dashboard"
            run_dir.mkdir()

            def _fake_export(src, dst):
                Path(dst).mkdir(parents=True, exist_ok=True)
                (Path(dst) / "index.html").write_text("<html></html>", encoding="utf-8")

            with patch("adenoma_agent.experiment_harness.export_dashboard_batch_from_harness_run", side_effect=_fake_export):
                with patch("adenoma_agent.experiment_harness.DashboardFrontendManager") as manager_cls:
                    manager_cls.return_value.switch_to.return_value = {
                        "service": "dashboard_frontend",
                        "status": "started",
                        "url": "http://127.0.0.1:8000/",
                        "health": {"ready": True, "status_code": 200},
                    }

                    payload = publish_harness_dashboard(
                        run_dir=run_dir,
                        output_dir=output_dir,
                        project_root=root,
                        switch_frontend=True,
                    )

            status = json.loads((output_dir / "dashboard_status.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["frontend"]["status"], "started")
        self.assertEqual(status["frontend"]["health"]["status_code"], 200)

    def test_crop_helper_writes_clean_rgb_png_and_cell_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slide_path = root / "slide.png"
            Image.new("RGB", (512, 512), (128, 64, 32)).save(slide_path)
            steps_path = root / "steps.json"
            manifest_path = root / "manifest.json"
            output_dir = root / "crops"
            write_json(
                steps_path,
                {
                    "steps": [
                        {
                            "step_id": "step_1",
                            "x": 256,
                            "y": 256,
                            "m": 10.0,
                            "region_size_level0": 128,
                            "metadata": {
                                "cell_id": "cell_1",
                                "patch_id": "patch_1",
                                "intra_cell_target_index": 1,
                                "coordinate_source": "model_proposed",
                            },
                        }
                    ]
                },
            )
            command = [
                "python3",
                "src/adenoma_agent/helpers/export_navigation_crops.py",
                "--slide-path",
                str(slide_path),
                "--steps-json",
                str(steps_path),
                "--output-dir",
                str(output_dir),
                "--manifest-json",
                str(manifest_path),
                "--output-size",
                "64",
            ]

            subprocess.check_call(command)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with Image.open(manifest["crops"][0]["image_path"]) as crop:
                crop_mode = crop.mode
                crop_size = crop.size

        self.assertEqual(crop_mode, "RGB")
        self.assertEqual(crop_size, (64, 64))
        self.assertEqual(manifest["crops"][0]["cell_id"], "cell_1")
        self.assertEqual(manifest["crops"][0]["level0_bbox"], [192, 192, 320, 320])

    def test_summary_marks_repaired_real_model_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "caseB"
            (case_dir / "observe").mkdir(parents=True)
            write_json(
                case_dir / "case_result.json",
                {"case_id": "caseB", "status": "ok", "timing": {"total_runtime_ms": 34}, "metadata": {}},
            )
            write_json(
                case_dir / "observe/pathological_report.json",
                {
                    "backend_attempts": [
                        {
                            "backend": "deepseek_output_repair",
                            "status": "ok",
                            "repair_stage": "observe_report",
                            "repair_actions": ["filled_missing_checklist_with_not_assessed"],
                        },
                        {"backend": "local_cpathagent_qwen", "status": "ok"},
                    ]
                },
            )

            row = summarize_case_artifacts(case_dir)
            result = json.loads((case_dir / "case_result.json").read_text(encoding="utf-8"))

        self.assertEqual(row["quality_tier"], "repaired_real_model")
        self.assertEqual(row["repair_count"], 1)
        self.assertEqual(row["repair_stages"], ["observe_report"])
        self.assertIn("filled_missing_checklist_with_not_assessed", row["repair_actions"])
        self.assertEqual(result["metadata"]["harness_quality"]["quality_tier"], "repaired_real_model")
        self.assertEqual(result["metadata"]["harness_quality"]["repair_count"], 1)


if __name__ == "__main__":
    unittest.main()
