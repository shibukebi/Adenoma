import argparse
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from adenoma_agent.agents.trace import TraceAgent
from adenoma_agent.adapters.thumbnail_grid import ThumbnailGridPreprocessor
from adenoma_agent.cli import load_grid_case_from_args
from adenoma_agent.schemas import CaseSpec


class GridFirstEntryTest(unittest.TestCase):
    def _write_manifest_files(self, root):
        manifest_path = Path(root) / "manifest.csv"
        labels_path = Path(root) / "labels.csv"
        manifest_path.write_text(
            "slide_id,slide_path,slide_filename\ncase_001,/tmp/case_001.svs,case_001.svs\n",
            encoding="utf-8",
        )
        labels_path.write_text("slide_id,type,grade\ncase_001,Sessile serrated adenoma,high\n", encoding="utf-8")
        return manifest_path, labels_path

    def _write_grid_inputs(self, root):
        grid_thumbnail_path = Path(root) / "case_001_tissuegrid125_ds32_level2_grid.jpg"
        overview_path = Path(root) / "case_001_tissuegrid125_ds32_level2.jpg"
        Image.new("RGB", (240, 240), color=(220, 200, 200)).save(grid_thumbnail_path)
        Image.new("RGB", (240, 240), color=(220, 200, 200)).save(overview_path)
        metadata_path = grid_thumbnail_path.with_suffix(".json")
        metadata = {
            "slide_id": "case_001",
            "slide_path": "/tmp/case_001.svs",
            "thumbnail_mode": "tissue_grid32x_svs",
            "cropped_thumbnail_size": [240, 240],
            "level0_crop_bbox": [1000, 2000, 5000, 6000],
            "grid_rows": 1,
            "grid_cols": 1,
            "grid_cells": [
                {
                    "patch_id": [0, 0],
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
                    "tissue_coverage_ratio": 0.8,
                    "is_selected": True,
                }
            ],
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return grid_thumbnail_path, metadata_path

    def test_load_grid_case_from_args_uses_manifest_slide_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path, labels_path = self._write_manifest_files(tmpdir)
            grid_thumbnail_path, metadata_path = self._write_grid_inputs(tmpdir)
            bundle = {
                "runtime": {
                    "data": {
                        "manifest_csv": str(manifest_path),
                        "labels_csv": str(labels_path),
                        "serrated_labels": ["Sessile serrated adenoma"],
                        "abnormal_crypt_positive_labels": ["Sessile serrated adenoma"],
                        "dysplasia_positive_grades": ["high"],
                    }
                }
            }
            args = argparse.Namespace(
                grid_thumbnail_path=str(grid_thumbnail_path),
                grid_metadata_path=str(metadata_path),
                slide_path=None,
                case_id=None,
            )
            case_spec = load_grid_case_from_args(bundle, args)
            self.assertEqual(case_spec.input_mode, "grid_thumbnail")
            self.assertEqual(case_spec.slide_path, "/tmp/case_001.svs")
            self.assertEqual(case_spec.grid_thumbnail_path, str(grid_thumbnail_path))
            self.assertEqual(case_spec.grid_metadata_path, str(metadata_path))
            self.assertTrue(case_spec.overview_thumbnail_path.endswith("_tissuegrid125_ds32_level2.jpg"))

    def test_trace_agent_bypasses_selector_for_grid_input(self):
        class _FailingSelector(object):
            def select(self, *args, **kwargs):
                raise AssertionError("selector should not be called for grid-first input")

        class _FakeBackendChain(object):
            def invoke(self, stage, chain_names, request):
                self.last_request = request
                return {
                    "backend": "fake",
                    "attempts": [],
                    "output": {
                        "clusters": [
                            {
                                "cluster_id": "cluster_00",
                                "l": "ssl_suspicious_mucosa",
                                "s": 4,
                                "d": True,
                                "review_stage": "serrated_screening",
                                "crypt_disorder_risk": 4,
                                "dysplasia_review_needed": False,
                                "desc": "test cluster",
                                "evidence": ["test"],
                                "metadata": {},
                                "patch_ids_ordered": [],
                                "patches_thumb": [],
                                "patches_level0": [],
                                "group_bbox_thumb": {"x1": 0, "y1": 0, "x2": 240, "y2": 240},
                                "group_bbox_level0": {"x1": 0, "y1": 0, "x2": 6000, "y2": 6000},
                            }
                        ]
                    },
                }

        class _Logger(object):
            def log(self, *args, **kwargs):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            grid_thumbnail_path, metadata_path = self._write_grid_inputs(tmpdir)
            bundle = {
                "runtime": {
                    "trace": {
                        "cluster_grid_size": 4,
                        "min_cell_tissue_fraction": 0.08,
                        "min_cluster_area_fraction": 0.01,
                        "backend_chain": ["fake"],
                        "patho_r1_question": "trace question",
                    }
                },
                "budget": {"max_trace_candidates": 4},
            }
            backend_chain = _FakeBackendChain()
            agent = TraceAgent(bundle, _FailingSelector(), backend_chain)
            case_spec = CaseSpec(
                case_id="case_001",
                slide_path="/tmp/case_001.svs",
                task_type="grid",
                question="q",
                input_mode="grid_thumbnail",
                grid_thumbnail_path=str(grid_thumbnail_path),
                grid_metadata_path=str(metadata_path),
            )
            result = agent.run(case_spec, Path(tmpdir) / "case_dir", _Logger())
            self.assertEqual(result["selection"]["mode"], "grid_input")
            self.assertTrue(Path(result["selection"]["paths"]["thumbnail"]).exists())
            self.assertEqual(result["clusters"][0].l, "ssl_suspicious_mucosa")

    def test_thumbnail_grid_preprocessor_promotes_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "thumb_grid" / "case_001"
            artifact_dir.mkdir(parents=True)
            grid_thumbnail_path, metadata_path = self._write_grid_inputs(artifact_dir)
            bundle = {
                "runtime": {
                    "thumbnail_grid_preprocess": {
                        "enabled": True,
                        "generate_if_missing": False,
                        "output_root": str(Path(tmpdir) / "thumb_grid"),
                        "case_subdir": True,
                    }
                }
            }
            case_spec = CaseSpec(
                case_id="case_001",
                slide_path="/tmp/case_001.svs",
                task_type="wsi",
                question="q",
            )

            result = ThumbnailGridPreprocessor(bundle).prepare(case_spec, Path(tmpdir) / "case_dir")

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["case_spec"].input_mode, "grid_thumbnail")
            self.assertEqual(result["case_spec"].grid_thumbnail_path, str(grid_thumbnail_path))
            self.assertEqual(result["case_spec"].grid_metadata_path, str(metadata_path))
            self.assertEqual(result["thumbnail_mode"], "tissue_grid32x_svs")


if __name__ == "__main__":
    unittest.main()
