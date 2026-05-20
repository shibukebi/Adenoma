import unittest

from adenoma_agent.contract_review import (
    review_global_screening_payload,
    review_navigation_payload,
    review_observation_report_payload,
    review_observation_step_payload,
)


class ContractReviewTest(unittest.TestCase):
    def _grid_meta(self):
        return {
            "grid_cells": [
                {"row_id": 0, "col_id": 0, "patch_id": [0, 0], "is_selected": True},
                {"row_id": 0, "col_id": 1, "patch_id": [0, 1], "is_selected": True},
            ]
        }

    def test_review_global_screening_payload_passes_valid_patch_assignments(self):
        payload = {
            "patches": [
                {
                    "patch_id": [0, 0],
                    "region_semantic": "normal_mucosa",
                    "name": "Normal mucosa",
                    "description": "Uniform crypt architecture.",
                    "require_high_magnification": False,
                    "severity_reasoning": "Low-priority non-lesional mucosa.",
                    "diagnostic_priority": 1,
                    "observation_points": ["confirm benign architecture if sampled"],
                },
                {
                    "patch_id": [0, 1],
                    "region_semantic": "background_artifact_stroma",
                    "name": "Background",
                    "description": "Masked background.",
                    "require_high_magnification": False,
                    "severity_reasoning": "Coverage preserving discard group.",
                    "diagnostic_priority": 0,
                    "observation_points": ["coverage-preserving discard group"],
                },
            ]
        }
        result = review_global_screening_payload(payload, self._grid_meta())
        self.assertTrue(result["ok"])

    def test_review_navigation_payload_rejects_bad_action_and_magnification(self):
        payload = {
            "steps": [
                {
                    "step_id": "step_00",
                    "x": 100,
                    "y": 120,
                    "m": 3.0,
                    "region_size_level0": 999,
                    "need_to_see": "inspect",
                    "review_goal": "serrated_lesion_assessment",
                    "stage_gate": "mucosa_or_serrated",
                    "metadata": {
                        "cluster_id": "grid_group_00",
                        "source_group_id": "grid_group_00",
                        "cluster_label": "ssl_suspicious_mucosa",
                        "cluster_priority": 4,
                        "patch_id": [0, 0],
                        "workflow_branch": "serrated",
                        "action": "zoom_in",
                    },
                }
            ]
        }
        result = review_navigation_payload(payload)
        self.assertFalse(result["ok"])
        self.assertTrue(any("m must be one of" in item for item in result["errors"]))
        self.assertTrue(any("action must be inspect or stop" in item for item in result["errors"]))

    def test_review_observation_step_payload_rejects_empty_observations(self):
        result = review_observation_step_payload({"observations": []})
        self.assertFalse(result["ok"])

    def test_review_observation_report_payload_accepts_list_based_checklists(self):
        payload = {
            "hierarchical_prediction": {"ssl_branch": {}},
            "serrated_checklist": ["cluster_id=grid_group_00"],
            "abnormal_crypt_checklist": [],
            "conventional_adenoma_checklist": [],
            "serrated_dysplasia_checklist": [],
            "conventional_dysplasia_checklist": [],
            "dysplasia_checklist": [],
            "integrated_report": {"summary": "summary", "recommendations": "recommendations"},
        }
        result = review_observation_report_payload(payload)
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
