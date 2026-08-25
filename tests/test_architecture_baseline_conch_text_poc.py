"""Critical non-model tests for Expert-Confirmed 5x Benchmark v1."""

from __future__ import annotations

import csv
import inspect
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image
from unittest.mock import patch

from adenoma_agent.conch_text_architecture_poc.io import sha256_file, write_json, write_text
from adenoma_agent.conch_text_architecture_poc.pipeline import (
    PRIMARY_CLASSES,
    ResumeRefusedError,
    _audit_public_package,
    _guideline_text,
    _select_sets,
    freeze_benchmark,
)
from adenoma_agent.conch_text_architecture_poc.backfill import (
    _mucosa_artifact,
    merge_mucosa_roots,
    selective_mucosa_backfill,
)
from adenoma_agent.conch_text_architecture_poc.workbook import (
    ANNOTATION_COLUMNS,
    DROPDOWNS,
    EXPERT_COLUMNS,
    LOCKED_COLUMNS,
    read_annotation_workbook,
    workbook_structure,
    write_annotation_workbook,
)


def _write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _build_fixture(root, prompt_confirmed=True):
    root = Path(root)
    artifact_root = root / "artifacts"
    package = artifact_root / "annotation_package_v1"
    images = package / "images"
    images.mkdir(parents=True)
    public_rows = []
    completed_rows = []
    row_number = 0
    for class_index, label in enumerate(PRIMARY_CLASSES):
        for class_row in range(10):
            row_number += 1
            roi_id = "ROI_{0:06d}".format(row_number)
            image_file = "images/{0}.png".format(roi_id)
            image_path = package / image_file
            Image.new("RGB", (16, 16), (80 + class_index * 50, 40 + class_row, 120)).save(str(image_path), format="PNG")
            public = {
                "annotation_instance_id": "ANN_V1_{0:06d}".format(row_number),
                "roi_id": roi_id,
                "safe_slide_id": "CASE_{0:06d}".format(row_number),
                "family_id": "FAMILY_{0}_{1:03d}".format(label.upper(), class_row),
                "grouping_resolution": "family",
                "image_file": image_file,
                "candidate_generation_order": row_number,
                "bbox": [0, 0, 2048, 2048],
                "magnification": 5.0,
                "MPP": [0.25, 0.25],
                "mpp_x": 0.25,
                "mpp_y": 0.25,
                "physical_fov_um": [512.0, 512.0],
                "image_sha256": sha256_file(image_path),
                "source_manifest_sha256": "a" * 64,
                "mucosa_boundary_sha256": "b" * 64,
                "mucosa_manifest_sha256": "c" * 64,
                "mucosa_artifact_sha256": "d" * 64,
                "source_wsi_identity_sha256": "e" * 64,
            }
            public_rows.append(public)
            completed_rows.append(
                {
                    **{key: public[key] for key in LOCKED_COLUMNS},
                    "architecture_label": label,
                    "evaluable": "yes",
                    "pure_or_mixed": "pure",
                    "expert_confidence": "high",
                    "architecture_components": "",
                    "exclusion_reason": "",
                    "notes": "",
                }
            )
    # A second high-confidence ROI from an existing family must become reserve.
    row_number += 1
    roi_id = "ROI_{0:06d}".format(row_number)
    image_path = images / (roi_id + ".png")
    Image.new("RGB", (16, 16), (100, 100, 100)).save(str(image_path), format="PNG")
    duplicate = {
        "annotation_instance_id": "ANN_V1_{0:06d}".format(row_number),
        "roi_id": roi_id,
        "safe_slide_id": "CASE_{0:06d}".format(row_number),
        "family_id": "FAMILY_SERRATED_000",
        "grouping_resolution": "family",
        "image_file": "images/{0}.png".format(roi_id),
        "candidate_generation_order": row_number,
        "bbox": [0, 0, 2048, 2048],
        "magnification": 5.0,
        "MPP": [0.25, 0.25],
        "mpp_x": 0.25,
        "mpp_y": 0.25,
        "physical_fov_um": [512.0, 512.0],
        "image_sha256": sha256_file(image_path),
        "source_manifest_sha256": "b" * 64,
        "mucosa_boundary_sha256": "b" * 64,
        "mucosa_manifest_sha256": "c" * 64,
        "mucosa_artifact_sha256": "d" * 64,
        "source_wsi_identity_sha256": "e" * 64,
    }
    public_rows.append(duplicate)
    completed_rows.append(
        {
            **{key: duplicate[key] for key in LOCKED_COLUMNS},
            "architecture_label": "serrated",
            "evaluable": "yes",
            "pure_or_mixed": "pure",
            "expert_confidence": "high",
            "architecture_components": "serrated",
            "exclusion_reason": "",
            "notes": "",
        }
    )
    # A difficult ROI is retained in stress and never enters primary.
    row_number += 1
    roi_id = "ROI_{0:06d}".format(row_number)
    image_path = images / (roi_id + ".png")
    Image.new("RGB", (16, 16), (150, 150, 150)).save(str(image_path), format="PNG")
    stress = {
        "annotation_instance_id": "ANN_V1_{0:06d}".format(row_number),
        "roi_id": roi_id,
        "safe_slide_id": "CASE_{0:06d}".format(row_number),
        "family_id": "FAMILY_STRESS_001",
        "grouping_resolution": "family",
        "image_file": "images/{0}.png".format(roi_id),
        "candidate_generation_order": row_number,
        "bbox": [0, 0, 2048, 2048],
        "magnification": 5.0,
        "MPP": [0.25, 0.25],
        "mpp_x": 0.25,
        "mpp_y": 0.25,
        "physical_fov_um": [512.0, 512.0],
        "image_sha256": sha256_file(image_path),
        "source_manifest_sha256": "c" * 64,
        "mucosa_boundary_sha256": "b" * 64,
        "mucosa_manifest_sha256": "c" * 64,
        "mucosa_artifact_sha256": "d" * 64,
        "source_wsi_identity_sha256": "e" * 64,
    }
    public_rows.append(stress)
    completed_rows.append(
        {
            **{key: stress[key] for key in LOCKED_COLUMNS},
            "architecture_label": "mixed",
            "evaluable": "yes",
            "pure_or_mixed": "mixed",
            "expert_confidence": "medium",
            "architecture_components": "tubular;villous",
            "exclusion_reason": "",
            "notes": "",
        }
    )
    blank_rows = [
        {**{key: row[key] for key in LOCKED_COLUMNS}, **{key: "" for key in EXPERT_COLUMNS}}
        for row in public_rows
    ]
    write_annotation_workbook(package / "annotation.xlsx", blank_rows)
    write_text(package / "annotation_guideline.md", _guideline_text())
    manifest = {
        "schema_version": "architecture_annotation_manifest_v1",
        "benchmark_version": "conch_text_architecture_benchmark_v1",
        "blinded": True,
        "candidate_rois": len(public_rows),
        "unique_candidate_wsi": len(public_rows),
        "rois": public_rows,
    }
    write_json(package / "annotation_manifest.json", manifest)
    completed = root / "annotation_completed.xlsx"
    write_annotation_workbook(completed, completed_rows)
    prompt = {
        "schema_version": "architecture_prompts_v1",
        "benchmark_version": "conch_text_architecture_benchmark_v1",
        "review_status": "confirmed" if prompt_confirmed else "pending_human_review",
        "reviewed_by": "fixture pathologist" if prompt_confirmed else "",
        "reviewed_at": "2026-08-23" if prompt_confirmed else "",
        "class_mapping": {label: label for label in PRIMARY_CLASSES},
        "prompt_sets": {
            "single_class_name": {label: [label] for label in PRIMARY_CLASSES},
            "matched_class_name_ensemble": {label: [label + " architecture"] for label in PRIMARY_CLASSES},
            "morphology_rich": {label: [label + " glandular morphology"] for label in PRIMARY_CLASSES},
        },
    }
    prompt_path = root / "architecture_prompts_v1.yaml"
    _write_yaml(prompt_path, prompt)
    config = {
        "schema_version": "conch_text_architecture_poc_v1",
        "benchmark_version": "conch_text_architecture_benchmark_v1",
        "paths": {
            "artifact_root": str(artifact_root),
            "annotation_package": str(package),
            "restricted_recruitment_provenance": str(artifact_root / "restricted_recruitment_provenance.json"),
            "audit_dir": str(artifact_root / "audit"),
            "mucosa_backfill_root": str(artifact_root / "mucosa_backfill"),
            "preparation_report": str(root / "preparation_report.md"),
            "benchmark_root": str(artifact_root / "benchmark_v1"),
            "freeze_report": str(root / "freeze_report.md"),
            "prompt_config": str(prompt_path),
            "baseline_root": str(root / "unused_baseline"),
            "case_paths": str(root / "unused_cases.jsonl"),
            "canonical_labels": str(root / "unused_labels.jsonl"),
            "per_slide_mucosa_root": str(root / "unused_mucosa"),
        },
        "mucosa_backfill": {
            "target_villous_wsi": 1,
            "uni_weights_path": str(root / "unused_uni.pt"),
            "prismnet_path": str(root / "unused_prismnet.pt"),
        },
        "freeze": {
            "primary_target_per_class": 10,
            "freeze_seed": 20260823,
            "split_seed": 17,
            "n_splits": 5,
            "validation_fold_offset": 1,
            "low_data_k": [5, 10, 20],
            "low_data_seeds": [17, 29],
            "primary_endpoint": "3-class macro_f1",
        },
        "model_arm_subset_contract": [
            "single_class_name_zero_shot",
            "matched_class_name_ensemble_zero_shot",
            "morphology_rich_zero_shot",
            "image_only_linear_probe",
            "correct_text_prior_linear_probe",
            "shuffled_text_prior_controls",
        ],
    }
    config_path = root / "config.yaml"
    _write_yaml(config_path, config)
    return {
        "root": root,
        "package": package,
        "completed": completed,
        "prompt": prompt_path,
        "config": config_path,
        "benchmark": artifact_root / "benchmark_v1",
        "manifest": manifest,
        "completed_rows": completed_rows,
    }


class ExpertConfirmedArchitectureBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = Path(tempfile.mkdtemp(prefix="architecture_poc_tests_"))
        cls.fixture = _build_fixture(cls.temporary)
        cls.result = freeze_benchmark(cls.fixture["completed"], cls.fixture["config"])
        cls.benchmark = cls.fixture["benchmark"]
        cls.freeze_manifest = json.loads((cls.benchmark / "freeze_manifest.json").read_text(encoding="utf-8"))
        with (cls.benchmark / "primary_rois.csv").open("r", encoding="utf-8", newline="") as handle:
            cls.primary = list(csv.DictReader(handle))
        with (cls.benchmark / "stress_rois.csv").open("r", encoding="utf-8", newline="") as handle:
            cls.stress = list(csv.DictReader(handle))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(cls.temporary))

    def test_annotation_workbook_contains_no_slide_diagnosis(self):
        self.assertEqual(_audit_public_package(self.fixture["package"], self.fixture["manifest"]), [])

    def test_annotation_workbook_contains_no_recruitment_stratum(self):
        rows, values = read_annotation_workbook(self.fixture["package"] / "annotation.xlsx")
        self.assertNotIn("recruitment_stratum", {key for row in rows for key in row})
        self.assertFalse(any("recruitment stratum" in value.lower() for value in values))

    def test_annotation_workbook_contains_no_model_outputs(self):
        rows, values = read_annotation_workbook(self.fixture["package"] / "annotation.xlsx")
        forbidden = ("conch_score", "model_prediction", "mil_attention", "classifier_score")
        self.assertFalse(any(field in row for row in rows for field in forbidden))
        self.assertFalse(any(any(field in value.lower() for field in forbidden) for value in values))

    def test_annotation_dropdown_values_are_valid(self):
        structure = workbook_structure(self.fixture["package"] / "annotation.xlsx")
        self.assertTrue(structure["protected"])
        self.assertEqual(structure["validation_count"], len(DROPDOWNS))
        self.assertEqual(structure["headers"], list(ANNOTATION_COLUMNS))

    def test_primary_contains_only_three_architecture_classes(self):
        self.assertEqual({row["architecture_label"] for row in self.primary}, set(PRIMARY_CLASSES))

    def test_primary_contains_only_high_confidence_pure_evaluable_rois(self):
        self.assertTrue(all(row["expert_confidence"] == "high" for row in self.primary))
        self.assertTrue(all(row["pure_or_mixed"] == "pure" for row in self.primary))
        self.assertTrue(all(row["evaluable"] == "yes" for row in self.primary))

    def test_no_duplicate_primary_roi(self):
        ids = [row["roi_id"] for row in self.primary]
        self.assertEqual(len(ids), len(set(ids)))

    def test_one_primary_roi_per_family_when_available(self):
        families = [row["family_id"] for row in self.primary]
        self.assertEqual(len(families), len(set(families)))

    def test_primary_has_no_slide_diagnosis_fields(self):
        self.assertFalse({"slide_diagnosis", "diagnosis", "original_slide_diagnosis"} & set(self.primary[0]))

    def test_recruitment_metadata_not_in_primary_manifest(self):
        self.assertFalse({"recruitment_stratum", "recruitment_diagnosis"} & set(self.primary[0]))

    def test_frozen_annotation_hash_matches(self):
        self.assertEqual(sha256_file(self.benchmark / "frozen_annotations.csv"), self.freeze_manifest["annotation_sha256"])

    def test_primary_image_hashes_match_reviewed_png(self):
        for row in self.primary:
            self.assertEqual(sha256_file((self.benchmark / row["image_file"]).resolve()), row["image_sha256"])

    def test_prompt_hash_is_frozen(self):
        self.assertEqual(sha256_file(self.benchmark / "architecture_prompts_v1.yaml"), self.freeze_manifest["prompt_config_sha256"])

    def test_fold_assignment_has_no_family_leakage(self):
        folds = json.loads((self.benchmark / "splits" / "folds.json").read_text(encoding="utf-8"))
        family_folds = {}
        primary_by_id = {row["roi_id"]: row for row in self.primary}
        for roi_id, fold in folds["assignments"].items():
            family_folds.setdefault(primary_by_id[roi_id]["family_id"], set()).add(fold)
        self.assertTrue(all(len(values) == 1 for values in family_folds.values()))

    def test_each_primary_roi_appears_in_exactly_one_fold(self):
        folds = json.loads((self.benchmark / "splits" / "folds.json").read_text(encoding="utf-8"))
        self.assertEqual(set(folds["assignments"]), {row["roi_id"] for row in self.primary})

    def test_low_data_subset_excludes_validation_and_test(self):
        with (self.benchmark / "splits" / "low_data_subsets.jsonl").open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        for row in records:
            self.assertFalse(set(row["roi_ids"]) & (set(row["validation_roi_ids"]) | set(row["test_roi_ids"])))

    def test_all_future_model_arms_reference_same_subset_hashes(self):
        contract = json.loads((self.benchmark / "splits" / "model_arm_subset_contract.json").read_text(encoding="utf-8"))
        hashes = {row["low_data_subsets_sha256"] for row in contract["arms"].values()}
        fold_hashes = {row["folds_sha256"] for row in contract["arms"].values()}
        self.assertEqual(len(hashes), 1)
        self.assertEqual(len(fold_hashes), 1)

    def test_freeze_resume_rejects_annotation_change(self):
        changed = self.fixture["root"] / "annotation_changed.xlsx"
        rows = [dict(row) for row in self.fixture["completed_rows"]]
        rows[0]["architecture_label"] = "tubular"
        write_annotation_workbook(changed, rows)
        with self.assertRaises(ResumeRefusedError):
            freeze_benchmark(changed, self.fixture["config"], resume=True)

    def test_freeze_resume_rejects_prompt_change(self):
        separate = Path(tempfile.mkdtemp(prefix="architecture_prompt_resume_"))
        try:
            fixture = _build_fixture(separate)
            freeze_benchmark(fixture["completed"], fixture["config"])
            prompt = yaml.safe_load(fixture["prompt"].read_text(encoding="utf-8"))
            prompt["prompt_sets"]["single_class_name"]["serrated"] = ["changed serrated"]
            _write_yaml(fixture["prompt"], prompt)
            with self.assertRaises(ResumeRefusedError):
                freeze_benchmark(fixture["completed"], fixture["config"], resume=True)
        finally:
            shutil.rmtree(str(separate))

    def test_no_conch_prediction_used_during_freeze(self):
        source = inspect.getsource(_select_sets).lower()
        self.assertNotIn("conch", source)
        self.assertNotIn("prediction", source)

    def test_no_model_score_used_for_primary_selection(self):
        source = inspect.getsource(_select_sets).lower()
        self.assertNotIn("model_score", source)
        self.assertNotIn("similarity", source)
        self.assertNotIn("attention", source)

    def test_stress_set_not_present_in_primary(self):
        self.assertFalse({row["roi_id"] for row in self.stress} & {row["roi_id"] for row in self.primary})

    def test_selective_mucosa_backfill_does_not_touch_full_baseline(self):
        roots = {"per_slide_mucosa_root": self.fixture["root"] / "full_baseline", "mucosa_backfill_root": self.fixture["root"] / "poc_backfill"}
        self.assertEqual(merge_mucosa_roots(roots), [])
        self.assertFalse((self.fixture["root"] / "full_baseline").exists())

    def test_existing_valid_mucosa_is_reused_read_only(self):
        case_root = self.fixture["root"] / "existing_mucosa" / "CASE_1"
        case_root.mkdir(parents=True)
        write_json(case_root / "baseline_boundary.json", {"status": "complete", "counts": {"errors": 0}})
        write_text(
            case_root / "five_x_patch_manifest.jsonl",
            json.dumps(
                {
                    "patch_id": "p1",
                    "grid_index": [0, 0],
                    "level0_bbox": [0, 0, 2048, 2048],
                    "clipped_level0_bbox": [0, 0, 2048, 2048],
                    "mucosa_coverage": 0.9,
                    "physical_provenance": {
                        "requested_magnification": 5.0,
                        "mpp_x": 0.25,
                        "mpp_y": 0.25,
                        "base_magnification": 40.0,
                        "fov_microns": [512.0, 512.0],
                        "output_pixel_dimensions": [256, 256],
                    },
                }
            )
            + "\n",
        )
        before = sha256_file(case_root / "five_x_patch_manifest.jsonl")
        artifact = _mucosa_artifact(self.fixture["root"] / "existing_mucosa", "CASE_1", 0.6)
        self.assertEqual(artifact["eligible_rows"], 1)
        self.assertEqual(sha256_file(case_root / "five_x_patch_manifest.jsonl"), before)

    def test_missing_mucosa_can_be_backfilled_in_poc_root(self):
        source = inspect.getsource(selective_mucosa_backfill)
        self.assertIn("mucosa_backfill_root", source)
        self.assertNotIn("frozen_conch_5x_mil_v1", source)

    def test_no_mucosa_threshold_relaxation_for_candidate_recruitment(self):
        config_text = Path("configs/conch_text_architecture_poc_v1.yaml").read_text(encoding="utf-8")
        self.assertIn("minimum_mucosa_coverage: 0.60", config_text)
        self.assertIn("mucosa_threshold: 0.30", config_text)

    def test_candidate_selection_does_not_use_conch_similarity(self):
        self.assertNotIn("conch", inspect.getsource(_select_sets).lower())

    def test_candidate_selection_does_not_use_model_prediction(self):
        source = inspect.getsource(_select_sets).lower()
        self.assertNotIn("prediction", source)
        self.assertNotIn("score", source)

    def test_recruitment_diagnosis_not_exported_to_annotation_workbook(self):
        rows, values = read_annotation_workbook(self.fixture["package"] / "annotation.xlsx")
        self.assertFalse(any("diagnosis" in key.lower() for row in rows for key in row))
        self.assertFalse(any("tubulovillous" in value.lower() for value in values))

    def test_recruitment_stratum_not_encoded_in_roi_id(self):
        self.assertFalse(any(stratum in row["roi_id"].lower() for row in self.fixture["manifest"]["rois"] for stratum in ("serrated", "tubular", "villous")))

    def test_annotation_images_are_deterministically_shuffled(self):
        orders = [int(row["candidate_generation_order"]) for row in self.fixture["manifest"]["rois"]]
        self.assertEqual(orders, list(range(1, len(orders) + 1)))
        self.assertNotEqual([row["architecture_label"] if "architecture_label" in row else "" for row in self.primary], [])

    def test_annotation_workbook_contains_only_safe_ids(self):
        rows, _ = read_annotation_workbook(self.fixture["package"] / "annotation.xlsx")
        self.assertTrue(all(row["roi_id"].startswith("ROI_") and row["safe_slide_id"].startswith("CASE_") for row in rows))

    def test_reviewed_png_sha_matches_manifest(self):
        self.test_primary_image_hashes_match_reviewed_png()

    def test_annotation_png_has_no_source_metadata(self):
        for row in self.fixture["manifest"]["rois"]:
            with (self.fixture["package"] / row["image_file"]).open("rb") as handle:
                image = Image.open(handle)
                self.assertEqual(dict(image.info), {})

    def test_candidate_roi_has_valid_5x_physical_scale(self):
        self.assertTrue(all(float(row["magnification"]) == 5.0 and row["physical_fov_um"] == [512.0, 512.0] for row in self.fixture["manifest"]["rois"]))

    def test_candidate_roi_has_valid_mucosa_provenance(self):
        self.assertTrue(all(row.get("mucosa_artifact_sha256") for row in self.fixture["manifest"]["rois"]))

    def test_one_default_candidate_roi_per_wsi(self):
        ids = [row["safe_slide_id"] for row in self.fixture["manifest"]["rois"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_duplicate_candidate_roi(self):
        ids = [row["roi_id"] for row in self.fixture["manifest"]["rois"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_duplicate_candidate_wsi(self):
        self.test_one_default_candidate_roi_per_wsi()

    def test_model_evaluation_not_started(self):
        self.assertFalse(self.freeze_manifest["model_evaluation_started"])
        self.assertFalse(self.freeze_manifest["zero_shot_performance_computed"])

    def test_prepare_annotation_stops_before_freeze(self):
        self.assertFalse(self.fixture["manifest"].get("flags", {}).get("BENCHMARK_FREEZE_COMPLETE", False))


if __name__ == "__main__":
    unittest.main()
