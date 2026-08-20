import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from adenoma_agent.agentflow.contracts import ROICandidate
from adenoma_agent.agentflow.wsi_runtime import (
    WSIROICropper,
    integration_boundary_payload,
    load_label_workbook_eligibility,
    select_deterministic_cohort,
)
from adenoma_agent.wsi import WSIPhysicalMetadataError, WSIReader, WSIReaderUnavailableError


class AgentFlowWSIRuntimeTest(unittest.TestCase):
    @staticmethod
    def _fixture_image(path, size=(128, 128)):
        image = Image.new("RGB", size, (255, 255, 255))
        image.paste(Image.new("RGB", (96, 96), (160, 70, 120)), (0, 0))
        image.save(path)
        return path

    def test_raster_fixture_crop_records_physical_provenance_and_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._fixture_image(Path(tmpdir) / "fixture.png")
            with WSIReader(path, base_magnification=40.0) as reader:
                crop = reader.crop_level0_bbox((0, 0, 64, 64), requested_magnification=20.0)
                output = crop.save(Path(tmpdir) / "crop.png")
                self.assertEqual(crop.pixel_dimensions, (32, 32))
                self.assertEqual(crop.provenance["source_level"], 0)
                self.assertEqual(crop.provenance["source_downsample"], 1.0)
                self.assertEqual(crop.provenance["base_magnification"], 40.0)
                self.assertEqual(crop.provenance["requested_magnification"], 20.0)
                self.assertEqual(crop.provenance["level0_bbox"], [0, 0, 64, 64])
                self.assertEqual(crop.provenance["fov_microns"], [16.0, 16.0])
                self.assertEqual(crop.provenance["output_pixel_dimensions"], [32, 32])
                self.assertEqual(crop.provenance["output_mpp"], [0.5, 0.5])
                self.assertEqual(
                    crop.image_sha256,
                    hashlib.sha256(output.read_bytes()).hexdigest(),
                )

    def test_missing_physical_metadata_blocks_magnification_crop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._fixture_image(Path(tmpdir) / "fixture.png")
            with WSIReader(path) as reader:
                with self.assertRaisesRegex(WSIPhysicalMetadataError, "physical metadata"):
                    reader.crop_level0_bbox((0, 0, 64, 64), requested_magnification=20.0)

    def test_explicit_output_pixels_must_match_requested_magnification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._fixture_image(Path(tmpdir) / "fixture.png")
            with WSIReader(path, base_magnification=40.0) as reader:
                with self.assertRaisesRegex(WSIPhysicalMetadataError, "inconsistent"):
                    reader.crop_level0_bbox(
                        (0, 0, 64, 64),
                        requested_magnification=20.0,
                        output_pixels=(64, 64),
                    )

    def test_isyntax_without_optional_reader_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.isyntax"
            path.write_text("<?xml version='1.0'?><isyntax/>", encoding="utf-8")
            with mock.patch.dict(sys.modules, {"isyntax": None}):
                with self.assertRaisesRegex(WSIReaderUnavailableError, "pyisyntax"):
                    WSIReader(path)

    def test_deterministic_cohort_uses_safe_alias_and_drops_eligibility_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hp = root / "hp"
            yx = root / "yx"
            hp.mkdir()
            yx.mkdir()
            for name in ("patient-a.isyntax", "patient-b.isyntax"):
                (hp / name).write_text("x", encoding="utf-8")
            for name in ("case-a.svs", "case-b.svs"):
                (yx / name).write_text("x", encoding="utf-8")
            private_labels = {
                "patient-a.isyntax": "exclude",
                "patient-b.isyntax": "eligible",
                "case-a.svs": "eligible",
                "case-b.svs": "exclude",
            }

            cohort = select_deterministic_cohort(
                {"hp": hp, "yx": yx},
                per_source=1,
                suffixes_by_source={"hp": (".isyntax",), "yx": (".svs",)},
                eligibility_predicate=lambda _source, path: private_labels[path.name] == "eligible",
            )
            self.assertEqual(len(cohort), 2)
            self.assertEqual([item.slide_path.name for item in cohort], ["patient-b.isyntax", "case-a.svs"])
            for item in cohort:
                inference = item.inference_payload()
                self.assertNotIn("path", inference)
                self.assertNotIn("label", inference)
                self.assertNotIn("source_name", inference)
                self.assertIn("source_code", inference)
                self.assertNotIn(item.slide_path.stem, item.case_alias)
                self.assertNotIn("hp", item.case_alias.lower())
                self.assertNotIn("yx", item.case_alias.lower())

            payload = integration_boundary_payload(
                cohort,
                {
                    cohort[0].case_alias: {
                        "status": "dependency_blocked",
                        "stages_reached": ["source_discovery"],
                        "integration_boundary": "wsi_open",
                        "dependency_blocked": [{"code": "pyisyntax_missing"}],
                    }
                },
            )
            self.assertFalse(payload["selection"]["labels_available_to_inference"])
            serialized_payload = json.dumps(payload, sort_keys=True)
            self.assertNotIn("Adenoma_hp", serialized_payload)
            self.assertNotIn("Adenoma_yx", serialized_payload)
            for row in payload["cases"]:
                self.assertNotIn("source_path", row["inference_identity"])
                self.assertIn("local_provenance_ref", row)
                self.assertNotIn("local_provenance", row)
                self.assertIn("reached_stage", row)
                self.assertIn("blocking_dependency", row)
                self.assertIn("recoverable_action", row)
                self.assertIn("exit_status", row)

    def test_yx_workbook_loader_returns_eligibility_only(self):
        workbook = Path(__file__).resolve().parents[1] / "data" / "label" / "Adenoma_filtered.xlsx"
        eligible = load_label_workbook_eligibility(workbook)
        self.assertIn("138189_751666001", eligible)
        self.assertTrue(all(isinstance(value, str) for value in eligible))

    def test_agentflow_wsi_cropper_materializes_roi_and_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            slide = self._fixture_image(root / "fixture.png")
            roi = ROICandidate(
                roi_id="ROI_20X_TEST",
                slide_id="SAFE_CASE",
                level0_bbox=(0, 0, 64, 64),
                scale=20.0,
                roi_semantics="cytology_hotspot",
                candidate_features=("high_grade_or_definite_dysplasia",),
                allowed_reviewers=("DysplasiaReviewer",),
                suitability=1.0,
                spatial_coverage=1.0,
                estimated_cost=0.5,
            )
            with WSIROICropper(slide, "SAFE_CASE", base_magnification=40.0) as cropper:
                artifact = cropper.crop(roi, root / "crops")
                self.assertEqual(artifact.pixel_dimensions, (32, 32))
                self.assertEqual(len(artifact.image_sha256), 64)
                self.assertTrue(Path(artifact.image_ref).exists())
                self.assertEqual(len(cropper.crop_events), 1)
                sidecar_path = Path(cropper.crop_events[0]["provenance_ref"])
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                self.assertEqual(sidecar["case_alias"], "SAFE_CASE")
                self.assertEqual(sidecar["crop"]["requested_magnification"], 20.0)
                self.assertNotIn("local_source_path", sidecar["crop"])
                self.assertIn("local_provenance_ref", sidecar)
                local_sidecar = json.loads(
                    (sidecar_path.parent / sidecar["local_provenance_ref"]).read_text(encoding="utf-8")
                )
                self.assertIn("source_path", local_sidecar)


if __name__ == "__main__":
    unittest.main()
