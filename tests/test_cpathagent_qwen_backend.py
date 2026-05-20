import json
import tempfile
import unittest
from pathlib import Path

from adenoma_agent.multimodal import StageBackendChain


class LocalCPathAgentQwenBackendTest(unittest.TestCase):
    def test_local_cpathagent_qwen_runner_invokes_navigate_stage(self):
        bundle = {
            "runtime": {
                "paths": {
                    "cpathagent_qwen_python": "/data1/yuexin/.conda/envs/patho-r1/bin/python",
                    "cpathagent_qwen_runner_script": "/data1/yuexin/Adenoma/scripts/run_cpathagent_qwen.py",
                },
                "models": {
                    "cpathagent_qwen_model_id": "Qwen/Qwen3-14B",
                    "cpathagent_qwen_vision_encoder_id": "CPath-CLIP-compatible",
                    "cpathagent_qwen_adapter_path": "/tmp/cpathagent_adapter",
                    "cpathagent_qwen_cache_dir": "/tmp/cpathagent_cache",
                },
                "execution": {"cpathagent_qwen_cuda_visible_devices": ""},
                "backends": {
                    "local_cpathagent_qwen": {"enabled": True, "max_new_tokens": 256, "shim_mode": "heuristic"}
                },
                "navigate": {"overlap_threshold": 0.05},
                "observe": {
                    "serrated_criteria": [],
                    "abnormal_crypt_criteria": [],
                    "conventional_adenoma_criteria": [],
                    "dysplasia_criteria": [],
                },
                "trace": {"labels": ["background_artifact_stroma", "normal_mucosa", "conventional_adenoma_like", "inflammatory_polyp_like", "ssl_suspicious_mucosa"]},
            },
            "budget": {
                "max_navigation_steps": 4,
                "magnification_to_region_size": {"5.0": 128, "20.0": 64},
            },
        }
        chain = StageBackendChain(bundle)
        request = {
            "images": [],
            "prompt": {"question": "plan navigation", "task": "cpathagent_navigation_planning"},
            "metadata": {
                "slide_dimensions_level0": [4000, 4000],
                "clusters": [
                    {
                        "cluster_id": "cluster_ssl",
                        "l": "ssl_suspicious_mucosa",
                        "s": 5,
                        "d": True,
                        "cluster_bbox_level0": {"x1": 400, "y1": 400, "x2": 600, "y2": 600},
                        "patches_level0": [{"patch_id": [0, 0], "x1": 400, "y1": 400, "x2": 600, "y2": 600}],
                    }
                ],
            },
        }
        response = chain.invoke("navigate", ["local_cpathagent_qwen"], request)
        self.assertEqual(response["backend"], "local_cpathagent_qwen")
        self.assertTrue(response["output"]["steps"])
        self.assertEqual(response["output"]["steps"][0]["metadata"]["source_group_id"], "cluster_ssl")
        self.assertEqual(response["output"]["runner_metadata"]["start_mode"], "cold_start_fallback")


if __name__ == "__main__":
    unittest.main()
