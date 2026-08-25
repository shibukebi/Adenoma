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
                    "region_semantic": "normal",
                    "name": "Normal mucosa",
                    "description": "Uniform crypt architecture.",
                    "require_high_magnification": False,
                    "severity_reasoning": "Low-priority non-lesional mucosa.",
                    "diagnostic_priority": 1,
                    "observation_points": ["confirm benign architecture if sampled"],
                },
                {
                    "patch_id": [0, 1],
                    "region_semantic": "background",
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

    def test_review_global_screening_payload_passes_dual_model_patch_assignments(self):
        payload = {
            "patches": [
                {
                    "patch_id": [0, 0],
                    "region_semantic": "serrated",
                    "conch_region_semantic": "serrated",
                    "pathoreasoner_r1_region_semantic": "serrated",
                    "agreement_status": "strong_agreement",
                    "name": "Consensus SSL",
                    "description": "Pale serrated mucosa with mucus cap.",
                    "require_high_magnification": True,
                    "severity_reasoning": "Both models favored the highest-value serrated route.",
                    "diagnostic_priority": 4,
                    "score_origin": "consensus_fusion",
                    "fusion_reasoning": "CONCH and PathoReasoner-R1 both selected serrated; consensus keeps serrated with priority 4.",
                    "observation_points": ["mucus cap", "crypt branching"],
                },
                {
                    "patch_id": [0, 1],
                    "region_semantic": "normal",
                    "conch_region_semantic": "normal",
                    "pathoreasoner_r1_region_semantic": "background",
                    "agreement_status": "low_value_disagreement",
                    "name": "Conservative normal",
                    "description": "Normal mucosa wins over background.",
                    "require_high_magnification": False,
                    "severity_reasoning": "Low-value disagreement kept the more reviewable label.",
                    "diagnostic_priority": 1,
                    "score_origin": "consensus_fusion",
                    "fusion_reasoning": "CONCH=normal and PathoReasoner-R1=background disagree only in low-value territory; conservative fusion keeps normal at priority 1.",
                    "observation_points": ["confirm benign architecture if sampled"],
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
                        "cluster_label": "serrated",
                        "cluster_priority": 4,
                        "cell_priority": 4,
                        "cell_id": "cell_0_0",
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

    def test_review_navigation_payload_accepts_multiple_intra_cell_zoom_targets(self):
        payload = {
            "steps": [
                {
                    "step_id": "step_00",
                    "x": 4000,
                    "y": 4000,
                    "m": 5.0,
                    "region_size_level0": 2048,
                    "need_to_see": "overview",
                    "review_goal": "serrated_lesion_assessment",
                    "stage_gate": "mucosa_or_serrated",
                    "metadata": {
                        "cluster_id": "grid_group_00",
                        "source_group_id": "grid_group_00",
                        "cluster_label": "serrated",
                        "cluster_priority": 5,
                        "cell_priority": 5,
                        "cell_id": "cell_0_0",
                        "patch_id": [0, 0],
                        "region_size_level0": 2048,
                        "workflow_branch": "serrated",
                        "action": "inspect",
                        "intra_cell_target_index": 0,
                        "intra_cell_target_role": "overview",
                        "intra_cell_target_count": 1,
                        "coordinate_source": "model_proposed",
                    },
                },
                {
                    "step_id": "step_01",
                    "x": 3000,
                    "y": 3000,
                    "m": 10.0,
                    "region_size_level0": 1024,
                    "need_to_see": "crypt base",
                    "review_goal": "abnormal_crypt_assessment",
                    "stage_gate": "abnormal_crypt",
                    "metadata": {
                        "cluster_id": "grid_group_00",
                        "source_group_id": "grid_group_00",
                        "cluster_label": "serrated",
                        "cluster_priority": 5,
                        "cell_priority": 5,
                        "cell_id": "cell_0_0",
                        "patch_id": [0, 0],
                        "region_size_level0": 1024,
                        "workflow_branch": "serrated",
                        "action": "inspect",
                        "intra_cell_target_index": 0,
                        "intra_cell_target_role": "crypt_base",
                        "intra_cell_target_count": 2,
                        "coordinate_source": "model_proposed",
                    },
                },
                {
                    "step_id": "step_02",
                    "x": 5000,
                    "y": 5000,
                    "m": 10.0,
                    "region_size_level0": 1024,
                    "need_to_see": "serrated edge",
                    "review_goal": "abnormal_crypt_assessment",
                    "stage_gate": "abnormal_crypt",
                    "metadata": {
                        "cluster_id": "grid_group_00",
                        "source_group_id": "grid_group_00",
                        "cluster_label": "serrated",
                        "cluster_priority": 5,
                        "cell_priority": 5,
                        "cell_id": "cell_0_0",
                        "patch_id": [0, 0],
                        "region_size_level0": 1024,
                        "workflow_branch": "serrated",
                        "action": "inspect",
                        "intra_cell_target_index": 1,
                        "intra_cell_target_role": "serrated_edge",
                        "intra_cell_target_count": 2,
                        "coordinate_source": "model_proposed",
                    },
                },
            ]
        }
        result = review_navigation_payload(payload)
        self.assertTrue(result["ok"], result["errors"])

    def test_review_observation_step_payload_rejects_empty_observations(self):
        result = review_observation_step_payload({"observations": [], "global_reviews": []})
        self.assertFalse(result["ok"])

    def test_review_observation_step_payload_accepts_global_reviews(self):
        payload = {
            "observations": [
                {
                    "step_id": "step_00",
                    "crop_path": "/tmp/step_00.png",
                    "observation": "Overview supports serrated lesion context.",
                    "reasoning": "Low-power features support the serrated branch.",
                    "next_step": "Inspect crypt architecture.",
                    "level_1_findings": ["serrated_lesion_context"],
                    "level_2_findings": [],
                    "level_3_findings": [],
                    "stage_decision": "supports_serrated_lesion",
                    "confidence": 0.8,
                    "metadata": {
                        "review_goal": "serrated_lesion_assessment",
                        "stage_gate": "mucosa_or_serrated",
                        "workflow_branch": "serrated",
                    },
                },
                {
                    "step_id": "step_01",
                    "crop_path": "/tmp/step_01.png",
                    "observation": "10x zoom-in supports crypt-level follow-up.",
                    "reasoning": "Higher magnification adds branch-specific morphology.",
                    "next_step": "Ask Chief to consolidate this cell.",
                    "level_1_findings": [],
                    "level_2_findings": ["basal_crypt_cue"],
                    "level_3_findings": [],
                    "stage_decision": "supports_abnormal_crypt",
                    "confidence": 0.76,
                    "metadata": {
                        "review_goal": "abnormal_crypt_assessment",
                        "stage_gate": "abnormal_crypt",
                        "workflow_branch": "serrated",
                    },
                }
            ],
            "global_reviews": [
                {
                    "review_id": "global_review_0000",
                    "source_step_id": "step_00",
                    "decision": "continue",
                    "continue_reason": "Need crypt-level confirmation before ending the scan.",
                    "chief_confidence": 0.71,
                    "resolved_branch_state": {
                        "serrated": "supported",
                        "abnormal_crypt": "unresolved",
                        "conventional": "opposed",
                        "dysplasia": "unresolved",
                    },
                    "sufficient_evidence": [],
                    "unresolved_questions": ["Need abnormal crypt confirmation."],
                    "next_visual_target": {
                        "target_cluster_id": "cluster_ssl",
                        "target_branch": "serrated",
                        "target_region_semantic": "serrated",
                        "target_morphology_prompt": ["look for basal crypt dilatation"],
                        "preferred_magnification": 10.0,
                        "priority_reason": "Resolve remaining abnormal-crypt uncertainty.",
                    },
                    "branch_correction_reason": "",
                }
            ],
        }
        result = review_observation_step_payload(payload)
        self.assertTrue(result["ok"])

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
