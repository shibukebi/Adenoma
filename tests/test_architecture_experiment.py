import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from adenoma_agent.architecture_experiment import (
    annotation_to_targets,
    build_architecture_roi_manifest,
    build_synthetic_smoke_annotations,
    read_jsonl,
    validate_parent_children,
    write_json,
    write_jsonl,
)


class ArchitectureExperimentTest(unittest.TestCase):
    def test_parent_children_require_exact_two_by_two_coverage(self):
        parent = [0, 0, 1024, 1024]
        children = [
            {"level0_bbox": [0, 0, 512, 512]},
            {"level0_bbox": [512, 0, 1024, 512]},
            {"level0_bbox": [0, 512, 512, 1024]},
            {"level0_bbox": [512, 512, 1024, 1024]},
        ]
        valid, reason = validate_parent_children(parent, children)
        self.assertTrue(valid)
        self.assertEqual(reason, "")
        children[-1]["level0_bbox"] = [513, 512, 1025, 1024]
        valid, reason = validate_parent_children(parent, children)
        self.assertFalse(valid)
        self.assertEqual(reason, "child_grid_does_not_cover_parent")

    def test_non_evaluable_and_uncertain_states_mask_morphology_losses(self):
        annotation = {
            "roi_id": "r1",
            "quality": {"state": "non_evaluable"},
            "architecture": {"serrated": "present", "tubular": "absent", "villous": "uncertain"},
            "context": {
                "normal_mucosa_present": "present",
                "reactive_inflammatory_present": "absent",
                "other_pattern_present": "uncertain",
            },
        }
        targets, masks = annotation_to_targets(annotation)
        self.assertEqual(targets[0], 0.0)
        self.assertEqual(masks[0], 1.0)
        self.assertTrue(np.all(masks[1:] == 0.0))

        annotation["quality"]["state"] = "evaluable"
        targets, masks = annotation_to_targets(annotation)
        self.assertEqual(targets.tolist(), [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(masks.tolist(), [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])

    def _artifact_bundle(self, root):
        artifact = Path(root) / "artifact"
        artifact.mkdir()
        crop_dir = artifact / "crops"
        crop_dir.mkdir()
        rows = []
        specs = [
            ("10x_1024", "parent", [0, 0, 1024, 1024], 0, 0),
            ("20x_512", "c00", [0, 0, 512, 512], 0, 0),
            ("20x_512", "c01", [512, 0, 1024, 512], 0, 1),
            ("20x_512", "c10", [0, 512, 512, 1024], 1, 0),
            ("20x_512", "c11", [512, 512, 1024, 1024], 1, 1),
        ]
        for crop_id, uid, bbox, row_id, col_id in specs:
            image_path = crop_dir / "{0}.png".format(uid)
            Image.new("RGB", (32, 32), (180, 120, 140)).save(image_path)
            rows.append(
                {
                    "patch_uid": uid,
                    "slide_id": "slide 1",
                    "crop_id": crop_id,
                    "level0_bbox": bbox,
                    "row_id": row_id,
                    "col_id": col_id,
                    "absolute_image_path": str(image_path),
                    "image_path": str(image_path.relative_to(artifact)),
                }
            )
        write_jsonl(artifact / "manifest.jsonl", rows)
        mask_dir = artifact / "mucosa_masks" / "slide_1"
        mask_dir.mkdir(parents=True)
        mask_path = mask_dir / "mucosa_candidate_mask.png"
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[:, :24] = 255
        Image.fromarray(mask, mode="L").save(mask_path)
        mask_json = mask_dir / "mucosa_mask.json"
        write_json(
            mask_json,
            {
                "pipeline": "test_mask",
                "processing": {"level0_window": [0, 0, 1024, 1024], "mask_downsample": 32.0},
                "artifacts": {"mucosa_candidate_mask": str(mask_path)},
                "model_policy": {},
            },
        )
        write_json(
            artifact / "mucosa_masks" / "mucosa_mask_summary.json",
            {"slides": [{"slide_id": "slide 1", "mucosa_mask_json": str(mask_json)}]},
        )
        return artifact

    def test_manifest_builder_projects_mask_and_writes_annotation_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = self._artifact_bundle(tmpdir)
            summary = build_architecture_roi_manifest(artifact, max_per_slide=10, seed=17)
            self.assertEqual(summary["counts"]["rois"], 1)
            rows = read_jsonl(Path(summary["output_dir"]) / "roi_manifest.jsonl")
            self.assertAlmostEqual(rows[0]["mucosa_coverage"], 0.75, places=3)
            self.assertEqual(len(rows[0]["children"]), 4)
            templates = read_jsonl(Path(summary["output_dir"]) / "annotation_template.jsonl")
            self.assertEqual(templates[0]["quality"]["state"], "unlabeled")

    def test_manifest_builder_prefers_mucosa_extractor_v1_over_legacy_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = self._artifact_bundle(tmpdir)
            slide_dir = artifact / "mucosa_extractor" / "slides" / "slide_1"
            slide_dir.mkdir(parents=True)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[:, :16] = 255
            maps_path = slide_dir / "maps.npz"
            np.savez_compressed(
                str(maps_path),
                mucosa_candidate_mask=mask,
            )
            write_json(
                artifact / "mucosa_extractor" / "manifest.json",
                {
                    "schema_version": "mucosa_extractor_v1_compact",
                    "slides": [
                        {
                            "slide_id": "slide 1",
                            "maps": str(maps_path),
                            "level0_window": [0, 0, 1024, 1024],
                            "mask_downsample": 32.0,
                        }
                    ],
                },
            )
            summary = build_architecture_roi_manifest(artifact, max_per_slide=10, seed=17)
            rows = read_jsonl(Path(summary["output_dir"]) / "roi_manifest.jsonl")
            self.assertEqual(summary["mask_source"], "mucosa_extractor_v1")
            self.assertAlmostEqual(rows[0]["mucosa_coverage"], 0.5, places=3)
            self.assertEqual(rows[0]["provenance"]["mask_source"], "mucosa_extractor_v1")

    def test_synthetic_annotations_are_explicit_and_trainable(self):
        rows = [{"roi_id": "r{0}".format(index), "slide_id": "s1"} for index in range(10)]
        annotations = build_synthetic_smoke_annotations(rows)
        self.assertTrue(all(row["synthetic_smoke_only"] for row in annotations))
        self.assertTrue(any(row["quality"]["state"] == "non_evaluable" for row in annotations))
        for annotation in annotations:
            targets, masks = annotation_to_targets(annotation)
            self.assertEqual(targets.shape, (7,))
            self.assertEqual(masks.shape, (7,))


if __name__ == "__main__":
    unittest.main()
