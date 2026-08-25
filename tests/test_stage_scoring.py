import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from adenoma_agent.stage_scoring import (
    score_case,
    score_navigate,
    score_observation,
    score_observe_report,
    score_observe_step,
    score_trace,
)
from adenoma_agent.trace_supervision import FIXED_DIAGNOSTIC_PRIORITY, TRACE_LABEL_RUBRIC
from adenoma_agent.utils import write_json


class StageScoringTest(unittest.TestCase):
    def _grid_meta(self):
        return {
            "grid_rows": 1,
            "grid_cols": 2,
            "grid_cells": [
                {"row_id": 0, "col_id": 0, "patch_id": [0, 0], "is_selected": True},
                {"row_id": 0, "col_id": 1, "patch_id": [0, 1], "is_selected": True},
            ],
        }

    def _trace_payload(self):
        return {
            "patches": [
                {
                    "patch_id": [0, 0],
                    "region_semantic": "serrated",
                    "name": "Serrated mucosa",
                    "description": "Pale serrated mucosa with mucus cap.",
                    "require_high_magnification": True,
                    "severity_reasoning": "Serrated concern with basal crypt abnormality.",
                    "diagnostic_priority": 4,
                    "observation_points": ["mucus cap", "crypt branching"],
                },
                {
                    "patch_id": [0, 1],
                    "region_semantic": "background",
                    "name": "Background",
                    "description": "Low-value background only.",
                    "require_high_magnification": False,
                    "severity_reasoning": "Coverage-preserving discard group.",
                    "diagnostic_priority": 0,
                    "observation_points": ["coverage-preserving discard group"],
                },
            ]
        }

    def _trace_clusters_payload(self):
        return {
            "clusters": [
                {
                    "cluster_id": "cluster_ssl",
                    "l": "serrated",
                    "s": 4,
                    "d": True,
                    "desc": "SSL cluster",
                    "review_stage": "serrated_screening",
                    "patch_ids_ordered": [[0, 0]],
                    "metadata": {"workflow_branch": "serrated"},
                },
                {
                    "cluster_id": "cluster_bg",
                    "l": "background",
                    "s": 0,
                    "d": False,
                    "desc": "Background cluster",
                    "review_stage": "background_screening",
                    "patch_ids_ordered": [[0, 1]],
                    "metadata": {"workflow_branch": "background"},
                },
            ]
        }

    def _navigate_payload(self):
        return {
            "steps": [
                {
                    "step_id": "step_00",
                    "x": 100,
                    "y": 120,
                    "m": 5.0,
                    "region_size_level0": 2048,
                    "need_to_see": "Inspect serrated mucosal context.",
                    "review_goal": "serrated_lesion_assessment",
                    "stage_gate": "mucosa_or_serrated",
                    "metadata": {
                        "cluster_id": "cluster_ssl",
                        "source_group_id": "cluster_ssl",
                        "cluster_label": "serrated",
                        "cluster_priority": 4,
                        "cell_id": "cell_0_0",
                        "cell_priority": 4,
                        "patch_id": [0, 0],
                        "region_size_level0": 2048,
                        "workflow_branch": "serrated",
                        "action": "inspect",
                    },
                },
                {
                    "step_id": "step_01",
                    "x": 100,
                    "y": 120,
                    "m": 10.0,
                    "region_size_level0": 1024,
                    "need_to_see": "Inspect abnormal crypt architecture.",
                    "review_goal": "abnormal_crypt_assessment",
                    "stage_gate": "abnormal_crypt",
                    "metadata": {
                        "cluster_id": "cluster_ssl",
                        "source_group_id": "cluster_ssl",
                        "cluster_label": "serrated",
                        "cluster_priority": 4,
                        "cell_id": "cell_0_0",
                        "cell_priority": 4,
                        "patch_id": [0, 0],
                        "region_size_level0": 1024,
                        "workflow_branch": "serrated",
                        "action": "inspect",
                    },
                },
                {
                    "step_id": "step_02",
                    "x": 100,
                    "y": 120,
                    "m": 5.0,
                    "region_size_level0": 2048,
                    "need_to_see": "Stop navigation and consolidate.",
                    "review_goal": "integrated_impression",
                    "stage_gate": "end",
                    "metadata": {
                        "cluster_id": "cluster_ssl",
                        "source_group_id": "cluster_ssl",
                        "cluster_label": "serrated",
                        "cluster_priority": 4,
                        "cell_id": "cell_0_0",
                        "cell_priority": 4,
                        "patch_id": [0, 0],
                        "region_size_level0": 2048,
                        "workflow_branch": "serrated",
                        "action": "stop",
                    },
                },
            ]
        }

    def _observe_step_payload(self):
        return {
            "observations": [
                {
                    "step_id": "step_00",
                    "crop_path": "/tmp/step_00.png",
                    "observation": "Overview supports serrated lesion context.",
                    "reasoning": "The low-power view supports the serrated branch.",
                    "next_step": "Proceed to abnormal crypt review.",
                    "level_1_findings": ["serrated_lesion_context"],
                    "level_2_findings": [],
                    "level_3_findings": [],
                    "stage_decision": "supports_serrated_lesion",
                    "confidence": 0.75,
                    "metadata": {
                        "review_goal": "serrated_lesion_assessment",
                        "stage_gate": "mucosa_or_serrated",
                        "workflow_branch": "serrated",
                    },
                },
                {
                    "step_id": "step_01",
                    "crop_path": "/tmp/step_01.png",
                    "observation": "Crypt architecture supports abnormal crypt.",
                    "reasoning": "Crypt branching and mucus cap are supporting features.",
                    "next_step": "Proceed to serrated dysplasia review.",
                    "level_1_findings": [],
                    "level_2_findings": ["crypt_branching", "mucus_cap"],
                    "level_3_findings": [],
                    "stage_decision": "supports_abnormal_crypt",
                    "confidence": 0.82,
                    "metadata": {
                        "review_goal": "abnormal_crypt_assessment",
                        "stage_gate": "abnormal_crypt",
                        "workflow_branch": "serrated",
                    },
                },
            ],
            "global_reviews": [
                {
                    "review_id": "global_review_0000",
                    "source_step_id": "step_00",
                    "decision": "continue",
                    "continue_reason": "Need high-magnification crypt confirmation before stopping.",
                    "chief_confidence": 0.73,
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
                        "priority_reason": "Resolve highest-value remaining branch uncertainty.",
                    },
                    "branch_correction_reason": "",
                },
                {
                    "review_id": "global_review_0001",
                    "source_step_id": "step_01",
                    "decision": "early_stop",
                    "continue_reason": "",
                    "chief_confidence": 0.81,
                    "resolved_branch_state": {
                        "serrated": "supported",
                        "abnormal_crypt": "supported",
                        "conventional": "opposed",
                        "dysplasia": "unresolved",
                    },
                    "sufficient_evidence": [
                        "Serrated lesion context is supported.",
                        "Abnormal crypt architecture is supported.",
                    ],
                    "unresolved_questions": [],
                    "next_visual_target": None,
                    "branch_correction_reason": "",
                },
            ],
        }

    def _observe_report_payload(self):
        return {
            "hierarchical_prediction": {
                "serrated_lesion_assessment": {"label": "serrated_lesion", "positive": True, "score": 0.9},
                "abnormal_crypt_assessment": {"label": "abnormal_crypt_supported", "positive": True, "score": 0.8},
                "conventional_adenoma_assessment": {"label": "conventional_adenoma_not_supported_or_indeterminate", "positive": False, "score": 0.1},
                "serrated_dysplasia_assessment": {"label": "serrated_dysplasia_not_supported_or_indeterminate", "positive": False, "score": 0.2},
                "conventional_dysplasia_assessment": {"label": "conventional_dysplasia_not_supported_or_indeterminate", "positive": False, "score": 0.2},
                "dysplasia_assessment": {"label": "dysplasia_not_supported_or_not_entered", "positive": False, "score": 0.2},
                "final_case_assessment": {"label": "SSL", "positive": True, "score": 0.8},
            },
            "serrated_checklist": {"serrated_lesion_context": {"status": "supporting", "evidence_steps": ["step_00"]}},
            "abnormal_crypt_checklist": {"crypt_branching": {"status": "supporting", "evidence_steps": ["step_01"]}},
            "conventional_adenoma_checklist": {},
            "serrated_dysplasia_checklist": {},
            "conventional_dysplasia_checklist": {},
            "dysplasia_checklist": {},
            "integrated_report": {"summary": "SSL branch supported.", "recommendations": "No dysplasia support."},
        }

    def test_trace_scoring_passes_valid_payload(self):
        result = score_trace(self._trace_payload(), self._grid_meta())
        self.assertFalse(result["hard_fail"])
        self.assertGreaterEqual(result["score"], 80)
        self.assertEqual(result["status"], "pass")

    def test_trace_scoring_penalizes_invalid_serrated_priority_and_fuzzy_observation(self):
        payload = self._trace_payload()
        payload["patches"][0]["diagnostic_priority"] = 1
        payload["patches"][0]["observation_points"] = ["suspicious area"]
        result = score_trace(payload, self._grid_meta())
        rule_ids = [item["rule_id"] for item in result["violations"]]
        self.assertIn("trace.matrix.serrated_priority", rule_ids)
        self.assertIn("trace.observation_points.fuzzy", rule_ids)
        self.assertLess(result["score"], 80)

    def test_trace_scoring_hard_fails_on_missing_patch_coverage(self):
        payload = {"patches": [self._trace_payload()["patches"][0]]}
        result = score_trace(payload, self._grid_meta())
        self.assertTrue(result["hard_fail"])
        self.assertEqual(result["status"], "fail")

    def test_navigate_scoring_penalizes_branch_mismatch(self):
        payload = self._navigate_payload()
        payload["steps"][0]["metadata"]["workflow_branch"] = "background"
        result = score_navigate(payload, self._trace_clusters_payload())
        rule_ids = [item["rule_id"] for item in result["violations"]]
        self.assertIn("navigate.matrix.serrated_branch", rule_ids)
        self.assertLess(result["score"], 80)

    def test_navigate_scoring_hard_fails_on_unknown_cluster(self):
        payload = self._navigate_payload()
        payload["steps"][0]["metadata"]["cluster_id"] = "missing_cluster"
        payload["steps"][0]["metadata"]["source_group_id"] = "missing_cluster"
        result = score_navigate(payload, self._trace_clusters_payload())
        self.assertTrue(result["hard_fail"])
        self.assertEqual(result["status"], "fail")

    def test_observe_step_scoring_penalizes_support_without_findings(self):
        payload = self._observe_step_payload()
        payload["observations"][1]["level_2_findings"] = []
        result = score_observe_step(payload)
        rule_ids = [item["rule_id"] for item in result["violations"]]
        self.assertIn("observe_step.matrix.support_requires_findings", rule_ids)
        self.assertLessEqual(result["score"], 80)

    def test_observe_report_scoring_penalizes_positive_without_support(self):
        step_payload = self._observe_step_payload()
        report_payload = self._observe_report_payload()
        report_payload["hierarchical_prediction"]["serrated_dysplasia_assessment"]["positive"] = True
        report_payload["hierarchical_prediction"]["serrated_dysplasia_assessment"]["label"] = "serrated_dysplasia_supported"
        result = score_observe_report(report_payload, step_payload)
        rule_ids = [item["rule_id"] for item in result["violations"]]
        self.assertIn("observe_report.matrix.positive_without_support", rule_ids)
        self.assertLessEqual(result["score"], 80)

    def test_observe_step_scoring_penalizes_branch_redirect_without_reason(self):
        payload = self._observe_step_payload()
        payload["global_reviews"][0]["next_visual_target"]["target_branch"] = "conventional"
        payload["global_reviews"][0]["branch_correction_reason"] = ""
        result = score_observe_step(payload)
        rule_ids = [item["rule_id"] for item in result["violations"]]
        self.assertIn("observe_step.matrix.stage_decision_mapping", rule_ids)

    def test_observe_report_scoring_penalizes_missing_final_early_stop(self):
        step_payload = self._observe_step_payload()
        step_payload["global_reviews"][-1]["decision"] = "continue"
        step_payload["global_reviews"][-1]["continue_reason"] = "Need more evidence."
        step_payload["global_reviews"][-1]["next_visual_target"] = {
            "target_cluster_id": "cluster_ssl",
            "target_branch": "serrated",
            "target_region_semantic": "serrated",
            "target_morphology_prompt": ["look for additional corroboration"],
            "preferred_magnification": 10.0,
            "priority_reason": "More evidence requested.",
        }
        report_payload = self._observe_report_payload()
        result = score_observe_report(report_payload, step_payload)
        rule_ids = [item["rule_id"] for item in result["violations"]]
        self.assertIn("observe_report.matrix.missing_hierarchy_key", rule_ids)

    def test_observation_stage_uses_60_40_weighting(self):
        step_payload = self._observe_step_payload()
        report_payload = self._observe_report_payload()
        result = score_observation(step_payload, report_payload)
        self.assertEqual(result["metrics"]["observe_step_weight"], 0.6)
        self.assertEqual(result["metrics"]["observe_report_weight"], 0.4)
        self.assertGreaterEqual(result["score"], 80)

    def test_case_scoring_aggregates_stage_scores(self):
        result = score_case(
            trace_payload=self._trace_payload(),
            navigate_payload=self._navigate_payload(),
            observe_step_payload=self._observe_step_payload(),
            observe_report_payload=self._observe_report_payload(),
            grid_meta=self._grid_meta(),
        )
        self.assertIn("trace_score", result)
        self.assertIn("navigate_score", result)
        self.assertIn("observation_score", result)
        self.assertIn("weights", result)
        self.assertGreaterEqual(result["overall_score"], 85)

    def test_trace_rubric_priority_conflict_is_fixed_to_document_value(self):
        self.assertNotIn("inflammatory_polyp_like", TRACE_LABEL_RUBRIC)
        self.assertEqual(TRACE_LABEL_RUBRIC["normal"]["default_priority"], 1)
        self.assertEqual(FIXED_DIAGNOSTIC_PRIORITY["normal"], 1)

    def test_score_agent_stage_cli_case_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_json = root / "trace.json"
            navigate_json = root / "navigate.json"
            observe_step_json = root / "observe_step.json"
            observe_report_json = root / "observe_report.json"
            grid_json = root / "grid.json"
            write_json(trace_json, self._trace_payload())
            write_json(navigate_json, self._navigate_payload())
            write_json(observe_step_json, self._observe_step_payload())
            write_json(observe_report_json, self._observe_report_payload())
            write_json(grid_json, self._grid_meta())
            output_json = root / "score.json"
            subprocess.check_call(
                [
                    "python3",
                    "scripts/score_agent_stage_json.py",
                    "--stage",
                    "case",
                    "--trace-json",
                    str(trace_json),
                    "--navigate-json",
                    str(navigate_json),
                    "--observe-step-json",
                    str(observe_step_json),
                    "--observe-report-json",
                    str(observe_report_json),
                    "--grid-metadata-json",
                    str(grid_json),
                    "--output-json",
                    str(output_json),
                ],
                cwd="/data1/yuexin/Adenoma",
            )
            result = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertIn("overall_score", result)
            self.assertIn("overall_status", result)


if __name__ == "__main__":
    unittest.main()
