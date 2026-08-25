import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from adenoma_agent.architecture_baselines.cli import _require_explicit_cuda, build_parser
from adenoma_agent.architecture_baselines.conch import (
    CONCH_EMBEDDING_DIM,
    FrozenConchEncoder,
    validate_case_cache,
)
from adenoma_agent.architecture_baselines.io import (
    require_output_outside_sources,
    sha256_file,
    write_json,
    write_jsonl,
)
from adenoma_agent.architecture_baselines.manifest import (
    merge_sanitized_manifests,
    validate_manifest,
    validate_patch_row,
)
from adenoma_agent.architecture_baselines.mucosa import build_mucosa_eligibility_ledger
from adenoma_agent.wsi import WSIReader


def _physical():
    return {
        "wsi_backend": "fixture",
        "base_magnification": 40.0,
        "base_magnification_source": "explicit_override",
        "mpp_x": 0.25,
        "mpp_y": 0.25,
        "mpp_source": "explicit_override",
        "requested_magnification": 5.0,
        "fov_microns": [512.0, 512.0],
        "output_pixel_dimensions": [256, 256],
    }


class ArchitectureBaselineRuntimeTest(unittest.TestCase):
    def test_formal_mucosa_gate_rejects_tile_errors_but_records_clean_attrition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_paths = root / "case_paths.jsonl"
            labels = root / "labels.jsonl"
            aliases = ["CASE_A", "CASE_B", "CASE_C"]
            write_jsonl(
                case_paths,
                [
                    {
                        "case_alias": alias,
                        "source_code": "YX",
                        "source_path": "/read-only/{0}.svs".format(alias),
                        "family_id": "F_{0}".format(alias),
                    }
                    for alias in aliases
                ],
            )
            write_jsonl(
                labels,
                [
                    {
                        "case_alias": alias,
                        "source_code": "YX",
                        "label": "class_{0}".format(index),
                        "grade": "low",
                        "family_id": "F_{0}".format(alias),
                    }
                    for index, alias in enumerate(aliases)
                ],
            )
            mucosa = root / "mucosa"
            for alias, status, errors, coverages in (
                ("CASE_A", "complete", 0, [0.8]),
                ("CASE_B", "complete", 0, []),
                ("CASE_C", "complete_with_tile_errors", 1, [0.7]),
            ):
                case_dir = mucosa / alias
                case_dir.mkdir(parents=True)
                write_json(
                    case_dir / "baseline_boundary.json",
                    {
                        "status": status,
                        "counts": {
                            "errors": errors,
                            "valid_tiles": 2,
                            "retained_tiles": 1,
                            "five_x_patches": len(coverages),
                        },
                    },
                )
                write_jsonl(
                    case_dir / "five_x_patch_manifest.jsonl",
                    [{"mucosa_coverage": value} for value in coverages],
                )
                if errors:
                    write_jsonl(case_dir / "errors.jsonl", [{"error": "fixture"}])
            output = root / "formal_gate"
            with self.assertRaises(RuntimeError):
                build_mucosa_eligibility_ledger(case_paths, mucosa, output, labels)
            ledger = [json.loads(line) for line in (output / "mucosa_eligibility_ledger.jsonl").read_text().splitlines()]
            by_alias = {row["case_alias"]: row for row in ledger}
            self.assertTrue(by_alias["CASE_A"]["eligible_for_training"])
            self.assertEqual(by_alias["CASE_B"]["exclusion_reason"], "no_five_x_patch")
            self.assertEqual(by_alias["CASE_C"]["technical_outcome"], "technical_failure")

    def test_output_must_not_be_inside_read_only_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(ValueError):
                require_output_outside_sources(root / "source" / "artifacts", [root / "source"])
            require_output_outside_sources(root / "artifacts", [root / "source"])

    def test_manifest_is_sanitized_and_physically_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mucosa = root / "mucosa" / "CASE_SAFE"
            mucosa.mkdir(parents=True)
            source_row = {
                "slide_id": "raw-slide-identity",
                "patch_id": "raw-slide-identity__5x__r00000_c00000",
                "grid_index": [0, 0],
                "level0_bbox": [0, 0, 2048, 2048],
                "clipped_level0_bbox": [0, 0, 2048, 2048],
                "target_magnification": "5x",
                "mucosa_coverage": 0.8,
                "mean_uncertainty": 0.2,
                "source_component_ids": [1],
                "physical_provenance": _physical(),
            }
            low_coverage_row = dict(source_row)
            low_coverage_row.update(
                {
                    "patch_id": "raw-slide-identity__5x__r00000_c00001",
                    "grid_index": [0, 1],
                    "level0_bbox": [2048, 0, 4096, 2048],
                    "clipped_level0_bbox": [2048, 0, 4096, 2048],
                    "mucosa_coverage": 0.2,
                }
            )
            write_jsonl(mucosa / "five_x_patch_manifest.jsonl", [source_row, low_coverage_row])
            write_json(mucosa / "manifest.json", {"status": "complete"})
            paths = root / "case_paths.jsonl"
            write_jsonl(
                paths,
                [
                    {
                        "case_alias": "CASE_SAFE",
                        "source_code": "YX",
                        "source_path": "/private/raw-slide-identity.svs",
                        "slide_stem": "raw-slide-identity",
                        "source_format": "svs",
                    }
                ],
            )
            merged = root / "five_x_patch_manifest.jsonl"
            summary = merge_sanitized_manifests(paths, root / "mucosa", merged)
            row = json.loads(merged.read_text(encoding="utf-8"))
            self.assertEqual(row["case_alias"], "CASE_SAFE")
            self.assertNotIn("raw-slide-identity", merged.read_text(encoding="utf-8"))
            self.assertEqual(summary["patches"], 1)
            self.assertEqual(summary["minimum_mucosa_coverage"], 0.30)
            self.assertEqual(summary["source_manifests"][0]["excluded_below_minimum_coverage"], 1)
            validate_patch_row(row)
            validate_manifest(merged)

    def test_wsi_crop_keeps_5x_physical_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "fixture.png"
            Image.new("RGB", (2048, 2048), (120, 80, 160)).save(image_path)
            with WSIReader(image_path, base_magnification=40.0, mpp=0.25) as reader:
                crop = reader.crop_level0_bbox(
                    [0, 0, 2048, 2048], requested_magnification=5.0, output_pixels=(256, 256)
                )
            self.assertEqual(crop.image.size, (256, 256))
            self.assertEqual(crop.provenance["fov_microns"], [512.0, 512.0])
            self.assertEqual(crop.provenance["output_mpp"], [2.0, 2.0])
            self.assertEqual(len(crop.image_sha256), 64)

    def test_embedding_cache_has_no_label_leakage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "CASE_SAFE"
            case_dir.mkdir()
            np.save(case_dir / "features.npy", np.ones((2, 512), dtype=np.float16))
            write_jsonl(
                case_dir / "index.jsonl",
                [
                    {
                        "case_alias": "CASE_SAFE",
                        "patch_id": "p{0}".format(index),
                        "feature_row": index,
                        "grid_index": [0, index],
                        "level0_bbox": [index * 2048, 0, (index + 1) * 2048, 2048],
                        "mucosa_coverage": 0.8,
                        "image_sha256": "a" * 64,
                    }
                    for index in range(2)
                ],
            )
            metadata = {
                "case_alias": "CASE_SAFE",
                "item_count": 2,
                "embedding_dim": 512,
                "labels_in_cache": False,
                "features_sha256": sha256_file(case_dir / "features.npy"),
                "index_sha256": sha256_file(case_dir / "index.jsonl"),
            }
            write_json(case_dir / "metadata.json", metadata)
            self.assertEqual(validate_case_cache(case_dir)["items"], 2)
            serialized = (case_dir / "metadata.json").read_text(encoding="utf-8") + (
                case_dir / "index.jsonl"
            ).read_text(encoding="utf-8")
            for forbidden in ("ground_truth", "diagnosis", "class_label", "grade"):
                self.assertNotIn(forbidden, serialized.lower())
            valid_features_hash = metadata["features_sha256"]
            metadata["features_sha256"] = "0" * 64
            write_json(case_dir / "metadata.json", metadata)
            with self.assertRaises(ValueError):
                validate_case_cache(case_dir)
            metadata["features_sha256"] = valid_features_hash
            metadata["labels_in_cache"] = True
            write_json(case_dir / "metadata.json", metadata)
            with self.assertRaises(ValueError):
                validate_case_cache(case_dir)

    def test_conch_is_frozen(self):
        import torch

        class FakeConch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

        fake = FakeConch()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.bin"
            checkpoint.write_bytes(b"trusted-test-checkpoint")
            with mock.patch(
                "conch.open_clip_custom.create_model_from_pretrained",
                return_value=(fake, lambda image: torch.zeros(3, 448, 448)),
            ), mock.patch(
                "adenoma_agent.architecture_baselines.conch._require_cuda_device"
            ):
                encoder = FrozenConchEncoder(checkpoint, device="cpu", use_amp=False)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.model.parameters()))
        head = torch.nn.Linear(CONCH_EMBEDDING_DIM, 7)
        optimizer = torch.optim.AdamW(head.parameters())
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertFalse(any(id(parameter) in optimizer_ids for parameter in encoder.model.parameters()))

    def test_cli_exposes_all_registered_commands(self):
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
            _require_explicit_cuda("cuda:0")
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
            with self.assertRaises(RuntimeError):
                _require_explicit_cuda("cuda:0")
        with self.assertRaises(ValueError):
            _require_explicit_cuda("cuda:1")
        parser = build_parser()
        help_text = parser.format_help()
        for command in (
            "preflight",
            "audit",
            "run-mucosa",
            "validate-manifest",
            "embed-conch",
            "build-splits",
            "baseline1-status",
            "train-mil",
            "evaluate",
            "real-smoke",
            "report",
        ):
            self.assertIn(command, help_text)


if __name__ == "__main__":
    unittest.main()
