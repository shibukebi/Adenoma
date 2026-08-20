import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from adenoma_agent.mucosa_extractor import (
    CRC100K_LABELS,
    TISSUE_CONTEXT_LABELS,
    MucosaExtractorConfig,
    _WSIReader,
    _build_slide_arrays,
    _read_level0_bbox,
    run_mucosa_extractor,
    tissue_context_from_probabilities,
    validate_crc100k_probabilities,
)
from adenoma_agent.utils import read_json, read_jsonl, write_json


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _one_hot(label):
    return {name: 1.0 if name == label else 0.0 for name in CRC100K_LABELS}


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class MucosaExtractorTest(unittest.TestCase):
    def test_crc100k_mapping_keeps_mus_and_tum_separate(self):
        probabilities = {
            "ADI": 0.05,
            "BACK": 0.05,
            "DEB": 0.10,
            "LYM": 0.10,
            "MUC": 0.10,
            "MUS": 0.15,
            "NORM": 0.15,
            "STR": 0.10,
            "TUM": 0.20,
        }
        context = tissue_context_from_probabilities(probabilities)
        self.assertEqual(tuple(context), TISSUE_CONTEXT_LABELS)
        self.assertAlmostEqual(context["background_or_artifact"], 0.20)
        self.assertAlmostEqual(context["smooth_muscle_or_deep_tissue"], 0.15)
        self.assertAlmostEqual(context["abnormal_epithelial_candidate"], 0.20)
        self.assertAlmostEqual(sum(context.values()), 1.0)

    def test_probability_validation_rejects_bad_values_and_records_renormalization(self):
        with self.assertRaisesRegex(ValueError, "Missing CRC100K"):
            validate_crc100k_probabilities({"NORM": 1.0})
        negative = _one_hot("NORM")
        negative["ADI"] = -0.1
        with self.assertRaisesRegex(ValueError, "Negative"):
            validate_crc100k_probabilities(negative)
        nonfinite = _one_hot("NORM")
        nonfinite["ADI"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validate_crc100k_probabilities(nonfinite)
        doubled = {label: value * 2.0 for label, value in _one_hot("TUM").items()}
        normalized, provenance = validate_crc100k_probabilities(doubled)
        self.assertTrue(provenance["renormalized"])
        self.assertAlmostEqual(sum(normalized.values()), 1.0)

    def test_overlap_tiles_are_probability_averaged(self):
        rows = []
        for uid, bbox, label in (("a", [0, 0, 64, 64], "NORM"), ("b", [32, 0, 96, 64], "TUM")):
            raw = _one_hot(label)
            rows.append(
                {
                    "patch_uid": uid,
                    "clipped_level0_bbox": bbox,
                    "raw_probabilities": raw,
                    "tissue_context": tissue_context_from_probabilities(raw),
                    "uncertainty": 0.0,
                }
            )
        raw_maps, context_maps, _uncertainty, valid, count = _build_slide_arrays(rows, [0, 0, 96, 64], 16.0)
        norm_index = TISSUE_CONTEXT_LABELS.index("normal_epi_context")
        tum_index = TISSUE_CONTEXT_LABELS.index("abnormal_epithelial_candidate")
        self.assertTrue(np.allclose(context_maps[norm_index, 0:4, 2:4], 0.5))
        self.assertTrue(np.allclose(context_maps[tum_index, 0:4, 2:4], 0.5))
        self.assertTrue(np.all(count[0:4, 2:4] == 2.0))
        self.assertTrue(np.all(valid))
        self.assertEqual(raw_maps.shape[0], len(CRC100K_LABELS))

    def test_level0_boundary_padding_is_white(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.png"
            Image.new("RGB", (40, 40), (100, 40, 80)).save(path)
            reader = _WSIReader(path)
            try:
                crop = _read_level0_bbox(reader, [0, 0, 64, 64], (64, 64))
            finally:
                reader.close()
            pixels = np.asarray(crop)
            self.assertTrue(np.all(pixels[0:40, 0:40] == np.asarray([100, 40, 80])))
            self.assertTrue(np.all(pixels[40:, :] == 255))
            self.assertTrue(np.all(pixels[:, 40:] == 255))

    def _artifact_bundle(self, root):
        artifact = Path(root) / "artifact"
        artifact.mkdir()
        manifest = []
        predictions = []
        labels = ["NORM", "TUM", "LYM", "MUC", "STR", "MUS", "BACK"]
        boxes = []
        for index, label in enumerate(labels):
            x1 = index * 512
            bbox = [x1, 0, x1 + 512, 512]
            boxes.append(bbox)
            uid = "p{0}".format(index)
            manifest.append(
                {
                    "patch_uid": uid,
                    "patch_id": [0, index],
                    "slide_id": "slide 1",
                    "crop_id": "20x_256",
                    "level0_bbox": bbox,
                    "clipped_level0_bbox": bbox,
                    "tissue_coverage_ratio": 1.0,
                    "simulated_magnification": 20.0,
                    "target_patch_size": 256,
                    "base_magnification": 40.0,
                }
            )
            predictions.append(
                {
                    "patch_uid": uid,
                    "slide_id": "slide 1",
                    "crop_id": "20x_256",
                    "model": "uni_prismnet",
                    "raw_class": label,
                    "probabilities": _one_hot(label),
                }
            )
        _write_jsonl(artifact / "manifest.jsonl", manifest)
        _write_jsonl(artifact / "predictions.jsonl", predictions)
        write_json(
            artifact / "summary.json",
            {
                "planned": [
                    {
                        "slide_id": "slide 1",
                        "level0_crop_bbox": [0, 0, len(labels) * 512, 512],
                        "slide_dimensions_level0": [len(labels) * 512, 512],
                        "base_magnification": 40.0,
                    }
                ]
            },
        )
        return artifact

    def test_artifact_pipeline_outputs_v1_contract_and_mask_uses_only_norm_tum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = self._artifact_bundle(tmpdir)
            output = Path(tmpdir) / "output"
            manifest = run_mucosa_extractor(
                artifact_dir=artifact,
                output_dir=output,
                config=MucosaExtractorConfig(mask_downsample=32.0, mucosa_threshold=0.30, write_overlays=True),
            )
            self.assertEqual(manifest["schema_version"], "mucosa_extractor_v1_compact")
            self.assertEqual(manifest["counts"]["valid_tiles"], 7)
            self.assertEqual(tuple(manifest["channel_order"]["task_context_scores"]), TISSUE_CONTEXT_LABELS)
            slide = manifest["slides"][0]
            maps = np.load(slide["maps"])
            self.assertEqual(
                set(maps.files),
                {"raw_tissue_probabilities", "task_context_scores", "uncertainty", "mucosa_score", "mucosa_candidate_mask", "valid_mask"},
            )
            mask = np.asarray(maps["mucosa_candidate_mask"]) > 0
            self.assertTrue(mask[:, :32].any())
            self.assertFalse(mask[:, 32:].any())
            self.assertEqual(maps["raw_tissue_probabilities"].shape[0], len(CRC100K_LABELS))
            self.assertEqual(maps["task_context_scores"].shape[0], len(TISSUE_CONTEXT_LABELS))
            self.assertTrue(Path(slide["qc_panel"]).exists())
            inflammatory = slide["context_statistics"]["inflammatory_context"]
            self.assertTrue(inflammatory["presence"])
            self.assertIsNotNone(inflammatory["fraction_of_valid_tissue"])
            self.assertIsNotNone(inflammatory["fraction_near_mucosa_mask"])
            self.assertEqual(manifest["config"]["presence_calibration_status"], "unvalidated")
            tile_index = read_jsonl(output / "tile_index.jsonl")
            self.assertEqual(len(tile_index), 7)
            self.assertIn("raw_prediction", tile_index[0])
            self.assertIn("task_context", tile_index[0])
            self.assertFalse((output / "run_manifest.json").exists())
            self.assertFalse((output / "config_snapshot.json").exists())
            self.assertFalse((output / "summary.json").exists())
            self.assertFalse((output / "errors.jsonl").exists())
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"manifest.json", "tile_index.jsonl", "five_x_patch_manifest.jsonl", "slides"},
            )
            self.assertEqual({path.name for path in (output / "slides" / "slide_1").iterdir()}, {"maps.npz", "qc_panel.png"})
            five_x = read_jsonl(output / "five_x_patch_manifest.jsonl")
            self.assertTrue(five_x)
            self.assertTrue(all(row["level0_bbox"][0] % 2048 == 0 for row in five_x))
            self.assertTrue(all(row["level0_bbox"][1] % 2048 == 0 for row in five_x))
            self.assertTrue(all(row["status"] == "retained" for row in five_x))

    def test_invalid_tile_is_recorded_without_silent_background_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = self._artifact_bundle(tmpdir)
            predictions = read_jsonl(artifact / "predictions.jsonl")
            predictions[0]["probabilities"].pop("ADI")
            _write_jsonl(artifact / "predictions.jsonl", predictions)
            output = Path(tmpdir) / "output"
            manifest = run_mucosa_extractor(artifact_dir=artifact, output_dir=output)
            self.assertEqual(manifest["counts"]["errors"], 1)
            self.assertEqual(manifest["counts"]["valid_tiles"], 6)
            errors = read_jsonl(output / "errors.jsonl")
            self.assertEqual(errors[0]["error_type"], "InvalidProbabilities")

    def test_raw_wsi_http_inference_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wsi_path = Path(tmpdir) / "synthetic.png"
            image = Image.new("RGB", (128, 128), (255, 255, 255))
            image.paste(Image.new("RGB", (96, 96), (180, 90, 140)), (0, 0))
            image.save(wsi_path)
            output = Path(tmpdir) / "mucosa_extractor"
            calls = []

            def fake_post(_url, json, timeout):
                calls.append((json, timeout))
                return _FakeResponse(
                    {
                        "predictions": [
                            {
                                "image_path": path,
                                "crc_label": "NORM",
                                "confidence": 1.0,
                                "probabilities": _one_hot("NORM"),
                            }
                            for path in json["image_paths"]
                        ]
                    }
                )

            config = MucosaExtractorConfig(
                classifier_patch_size=64,
                base_magnification=20.0,
                min_tissue_coverage=0.05,
                mask_downsample=16.0,
                batch_size=2,
                write_overlays=False,
            )
            with patch("requests.post", side_effect=fake_post):
                first = run_mucosa_extractor(wsi_paths=[wsi_path], output_dir=output, config=config)
            self.assertGreater(first["counts"]["valid_tiles"], 0)
            self.assertTrue(calls)
            self.assertFalse((output / "crops").exists())
            self.assertFalse((output / "preprocess").exists())
            self.assertFalse((output / "errors.jsonl").exists())

            resumed_config = MucosaExtractorConfig(**{**config.__dict__, "resume": True})
            with patch("requests.post") as resumed_post:
                second = run_mucosa_extractor(wsi_paths=[wsi_path], output_dir=output, config=resumed_config)
            resumed_post.assert_not_called()
            self.assertEqual(second["counts"]["valid_tiles"], first["counts"]["valid_tiles"])

    def test_unavailable_pathprism_service_fails_explicitly_and_writes_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wsi_path = Path(tmpdir) / "synthetic.png"
            Image.new("RGB", (64, 64), (180, 90, 140)).save(wsi_path)
            output = Path(tmpdir) / "mucosa_extractor"
            config = MucosaExtractorConfig(
                classifier_patch_size=64,
                base_magnification=20.0,
                mask_downsample=16.0,
                write_overlays=False,
            )
            with patch("requests.post", side_effect=RuntimeError("service unavailable")):
                with self.assertRaisesRegex(RuntimeError, "failed for all pending tiles"):
                    run_mucosa_extractor(wsi_paths=[wsi_path], output_dir=output, config=config)
            errors = read_jsonl(output / "errors.jsonl")
            self.assertTrue(errors)
            self.assertEqual(errors[0]["stage"], "pathprism_inference")
            self.assertEqual(errors[0]["error_type"], "RuntimeError")

    def test_unreadable_wsi_creates_structured_optional_error_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "mucosa_extractor"
            missing = Path(tmpdir) / "missing.svs"
            with self.assertRaisesRegex(RuntimeError, "No readable WSI"):
                run_mucosa_extractor(
                    wsi_paths=[missing],
                    output_dir=output,
                    config=MucosaExtractorConfig(base_magnification=40.0),
                )
            errors = read_jsonl(output / "errors.jsonl")
            self.assertEqual(errors[0]["slide_id"], "missing")
            self.assertEqual(errors[0]["stage"], "wsi_reading")
            self.assertIn("error_type", errors[0])
            self.assertIn("message", errors[0])


if __name__ == "__main__":
    unittest.main()
