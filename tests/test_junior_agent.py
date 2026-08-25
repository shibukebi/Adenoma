import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from adenoma_agent.agents.junior import JuniorAgent
from adenoma_agent.agents.trace import TraceAgent
from adenoma_agent.schemas import CaseSpec
from adenoma_agent.utils import read_json


class _Logger(object):
    def log(self, *args, **kwargs):
        return None


class JuniorAgentTest(unittest.TestCase):
    def test_junior_agent_generates_roi_json_with_embedded_mask(self):
        bundle = {
            "runtime": {
                "junior": {
                    "patch_size": 128,
                    "stride": 64,
                    "threshold": 0.45,
                    "mucosa_probability_threshold": 0.4,
                    "necrosis_veto_threshold": 0.5,
                }
            },
            "budget": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "case_001.png"
            canvas = Image.new("RGB", (512, 512), color=(250, 250, 250))
            for y in range(80, 360):
                for x in range(60, 430):
                    canvas.putpixel((x, y), (205, 145, 175))
            canvas.save(image_path)

            case_spec = CaseSpec(
                case_id="case_001",
                slide_path=str(image_path),
                task_type="junior_test",
                question="q",
                metadata={},
            )
            result = JuniorAgent(bundle).run(case_spec, Path(tmpdir) / "run_case", _Logger())
            self.assertEqual(result["status"], "ok")
            self.assertGreaterEqual(result["roi_count"], 1)

            payload = read_json(result["junior_json"])
            self.assertEqual(payload["slide_id"], "case_001")
            self.assertEqual(payload["status"], "ok")
            self.assertGreaterEqual(len(payload["mucosa_rois"]), 1)
            roi = payload["mucosa_rois"][0]
            self.assertEqual(roi["mask_spec"]["encoding"], "base64_png")

            mask_bytes = base64.b64decode(roi["mask_spec"]["raw_mask_string"])
            mask = Image.open(io.BytesIO(mask_bytes))
            self.assertGreater(mask.size[0], 0)
            self.assertGreater(mask.size[1], 0)
            self.assertTrue(Path(payload["debug_artifacts"]["anatomical_mask"]).exists())
            self.assertTrue(Path(payload["debug_artifacts"]["final_roi_boxes"]).exists())


class TraceJuniorGuidanceIntegrationTest(unittest.TestCase):
    def test_trace_agent_ignores_junior_roi_proposals_in_main_pipeline(self):
        class _FakeSelector(object):
            def __init__(self, thumbnail_path):
                self.thumbnail_path = Path(thumbnail_path)

            def select(self, case_spec, output_dir, manual_boxes=None, preferred_mode=None):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                boxes_json = output_dir / "{0}_route_c_boxes.json".format(case_spec.case_id)
                payload = {
                    "mode": "heuristic",
                    "thumbnail_meta": {
                        "thumbnail_size": [400, 300],
                        "slide_dimensions_level0": [4000, 3000],
                    },
                    "boxes": [{"x1": 0, "y1": 0, "x2": 4000, "y2": 3000, "score": 0.9, "label": "route_c"}],
                }
                from adenoma_agent.utils import write_json

                write_json(boxes_json, payload)
                return {
                    "payload": payload,
                    "mode": "heuristic",
                    "cache_hit": False,
                    "paths": {
                        "thumbnail": self.thumbnail_path,
                        "raw_response": output_dir / "trace_raw_response.txt",
                        "boxes_json": boxes_json,
                        "visualization": self.thumbnail_path,
                    },
                    "attempts": [],
                }

        class _FakeBackendChain(object):
            def invoke(self, stage, chain_names, request):
                self.last_request = request
                proposal = request["metadata"]["proposals"][0]
                return {
                    "backend": "fake",
                    "attempts": [],
                    "output": {
                        "clusters": [
                            {
                                "cluster_id": proposal["cluster_id"],
                                "l": "ssl_suspicious_mucosa",
                                "s": 4,
                                "d": True,
                                "review_stage": "serrated_screening",
                                "crypt_disorder_risk": 4,
                                "dysplasia_review_needed": False,
                                "desc": "junior-guided cluster",
                                "evidence": ["junior roi"],
                                "metadata": {"workflow_branch": "serrated"},
                                "patch_ids_ordered": [],
                                "patches_thumb": [],
                                "patches_level0": [],
                                "group_bbox_thumb": proposal["cluster_bbox_thumb"],
                                "group_bbox_level0": proposal["cluster_bbox_level0"],
                            }
                        ]
                    },
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            thumbnail_path = Path(tmpdir) / "thumbnail.png"
            Image.new("RGB", (400, 300), color=(220, 190, 205)).save(thumbnail_path)
            junior_json_path = Path(tmpdir) / "junior_mucosa_rois.json"
            from adenoma_agent.utils import write_json

            write_json(
                junior_json_path,
                {
                    "slide_id": "case_001",
                    "status": "ok",
                    "mucosa_rois": [
                        {
                            "roi_id": "mucosa_zone_01",
                            "diagnostic_priority": 3,
                            "level_0_bounding_box": {"xmin": 800, "ymin": 600, "width": 1600, "height": 900},
                            "comment": "junior roi",
                            "mask_spec": {
                                "width_at_processing_level": 10,
                                "height_at_processing_level": 10,
                                "encoding": "base64_png",
                                "raw_mask_string": "",
                            },
                        }
                    ],
                },
            )
            bundle = {
                "runtime": {
                    "trace": {
                        "cluster_grid_size": 8,
                        "min_cell_tissue_fraction": 0.08,
                        "min_cluster_area_fraction": 0.01,
                        "backend_chain": ["fake"],
                        "patho_r1_question": "trace question",
                    }
                },
                "budget": {"max_trace_candidates": 4},
            }
            backend_chain = _FakeBackendChain()
            agent = TraceAgent(bundle, _FakeSelector(thumbnail_path), backend_chain)
            case_spec = CaseSpec(
                case_id="case_001",
                slide_path=str(thumbnail_path),
                task_type="trace_test",
                question="q",
                metadata={"junior_result_json": str(junior_json_path)},
            )
            result = agent.run(case_spec, Path(tmpdir) / "case_dir", _Logger())
            self.assertEqual(result["payload"]["mode"], "heuristic")
            self.assertEqual(result["clusters"][0].cluster_id, "cluster_00")
            self.assertFalse(backend_chain.last_request["metadata"]["junior_guidance"]["used"])
            self.assertEqual(backend_chain.last_request["metadata"]["proposals"][0]["cluster_id"], "cluster_00")

            updated_boxes = read_json(result["selection"]["paths"]["boxes_json"])
            self.assertEqual(updated_boxes["boxes"][0]["label"], "route_c")


if __name__ == "__main__":
    unittest.main()
