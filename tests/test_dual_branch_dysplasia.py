import unittest

from adenoma_agent.multimodal import HeuristicStageBackend


class DualBranchDysplasiaTest(unittest.TestCase):
    def _bundle(self):
        return {
            "runtime": {
                "observe": {
                    "serrated_criteria": [
                        "serrated_surface_pattern",
                        "mucus_rich_surface",
                        "serrated_lesion_context",
                    ],
                    "abnormal_crypt_criteria": [
                        "basal_dilatation",
                        "crypt_branching",
                        "horizontal_growth",
                        "boot_l_t_shaped_crypt",
                        "serration_to_base",
                        "mucus_cap",
                        "abnormal_maturation",
                    ],
                    "conventional_adenoma_criteria": [
                        "tubular_architecture",
                        "villous_component",
                        "high_villous_component",
                        "tubular_or_tubulovillous_architecture",
                        "crowded_adenomatous_glands",
                        "pencillate_hyperchromatic_nuclei",
                    ],
                    "dysplasia_criteria": [
                        "nuclear_enlargement_stratification",
                        "hyperchromasia",
                        "mitotic_activity_atypia",
                        "architectural_crowding",
                        "high_grade_focus",
                        "marked_cytologic_atypia",
                    ],
                }
            }
        }

    def test_conventional_dysplasia_maps_to_others_dysplasia_only(self):
        backend = HeuristicStageBackend()
        request = {
            "stage": "observe_report",
            "metadata": {
                "records": [
                    {
                        "step_id": "step_00",
                        "metadata": {
                            "conventional_hits": {
                                "tubular_or_tubulovillous_architecture": "supporting",
                                "crowded_adenomatous_glands": "supporting",
                            },
                            "conventional_dysplasia_hits": {
                                "nuclear_enlargement_stratification": "supporting",
                                "hyperchromasia": "supporting",
                                "architectural_crowding": "supporting",
                            },
                        },
                    }
                ],
                "trace_clusters": [
                    {"cluster_id": "cluster_00", "l": "conventional_adenoma_like", "s": 4}
                ],
            },
        }
        output = backend.invoke(request, self._bundle())["output"]
        hierarchy = output["hierarchical_prediction"]
        self.assertFalse(hierarchy["serrated_dysplasia_assessment"]["positive"])
        self.assertTrue(hierarchy["conventional_dysplasia_assessment"]["positive"])
        self.assertEqual(hierarchy["final_case_assessment"]["label"], "TAD")
        self.assertEqual(hierarchy["final_case_assessment"]["legacy_label"], "Others+dysplasia")
        self.assertEqual(hierarchy["final_11_class"], "TAD")

    def test_serrated_dysplasia_maps_to_ssl_dysplasia_only(self):
        backend = HeuristicStageBackend()
        request = {
            "stage": "observe_report",
            "metadata": {
                "records": [
                    {
                        "step_id": "step_00",
                        "metadata": {
                            "serrated_hits": {
                                "serrated_surface_pattern": "supporting",
                                "mucus_rich_surface": "supporting",
                                "serrated_lesion_context": "supporting",
                            },
                            "abnormal_crypt_hits": {
                                "basal_dilatation": "supporting",
                                "crypt_branching": "supporting",
                                "horizontal_growth": "supporting",
                            },
                            "serrated_dysplasia_hits": {
                                "nuclear_enlargement_stratification": "supporting",
                                "hyperchromasia": "supporting",
                                "architectural_crowding": "supporting",
                            },
                        },
                    }
                ],
                "trace_clusters": [
                    {"cluster_id": "cluster_00", "l": "ssl_suspicious_mucosa", "s": 5}
                ],
            },
        }
        output = backend.invoke(request, self._bundle())["output"]
        hierarchy = output["hierarchical_prediction"]
        self.assertTrue(hierarchy["serrated_dysplasia_assessment"]["positive"])
        self.assertFalse(hierarchy["conventional_dysplasia_assessment"]["positive"])
        self.assertEqual(hierarchy["final_case_assessment"]["label"], "SSLD")
        self.assertEqual(hierarchy["final_case_assessment"]["legacy_label"], "SSL+dysplasia")
        self.assertEqual(hierarchy["final_11_class"], "SSLD")


if __name__ == "__main__":
    unittest.main()
