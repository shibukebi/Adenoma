import unittest

from adenoma_agent.multimodal import HeuristicStageBackend, _normalize_trace_label, _trace_branch_for_label


class ElevenClassWorkflowTest(unittest.TestCase):
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
                    "ssl_criteria": [
                        "basal_crypt_dilatation",
                        "crypt_branching",
                        "horizontal_or_boot_shaped_crypt",
                        "serration_to_base",
                    ],
                    "hp_criteria": [
                        "surface_limited_serration",
                        "straight_crypt_bases",
                        "lacks_basal_architectural_distortion",
                    ],
                    "tsa_criteria": [
                        "villiform_or_filiform_architecture",
                        "eosinophilic_cytoplasm",
                        "ectopic_crypt_formation",
                        "slit_like_serration",
                    ],
                    "tsa_cytological_atypia_criteria": [
                        "cytoplasmic_eosinophilia",
                        "pencillate_nuclei",
                    ],
                    "inflammatory_criteria": [
                        "erosion",
                        "granulation_tissue",
                        "mixed_inflammation",
                        "reactive_regenerative_change",
                        "lacks_adenomatous_or_serrated_architecture",
                    ],
                }
            }
        }

    def _report(self, trace_label, record_metadata):
        request = {
            "stage": "observe_report",
            "metadata": {
                "records": [{"step_id": "step_00", "metadata": record_metadata}],
                "trace_clusters": [{"cluster_id": "cluster_00", "l": trace_label, "s": 4}],
            },
        }
        return HeuristicStageBackend().invoke(request, self._bundle())["output"]

    def _navigate(self, trace_label, priority=4, high_mag=True):
        request = {
            "stage": "navigate",
            "metadata": {
                "slide_dimensions_level0": [10000, 10000],
                "clusters": [
                    {
                        "cluster_id": "cluster_00",
                        "l": trace_label,
                        "s": priority,
                        "d": high_mag,
                        "cluster_bbox_level0": {"x1": 2000, "y1": 2000, "x2": 6000, "y2": 6000},
                        "patches_level0": [
                            {"patch_id": [0, 0], "x1": 2000, "y1": 2000, "x2": 6000, "y2": 6000}
                        ],
                    }
                ],
            },
        }
        bundle = self._bundle()
        bundle["runtime"]["navigate"] = {"overlap_threshold": 0.05}
        bundle["budget"] = {
            "max_navigation_steps": 8,
            "max_intra_cell_zoom_targets": 3,
            "magnification_to_region_size": {"2.5": 4096, "5.0": 2048, "10.0": 1024},
        }
        return HeuristicStageBackend().invoke(request, bundle)["output"]

    def _observe_step(self, review_goal, trace_label="serrated", priority=4, pale_fraction=0.02, tissue_fraction=0.7, background_fraction=0.1, metadata=None):
        request = {
            "stage": "observe_step",
            "metadata": {
                "image_stats": {
                    "pale_fraction": pale_fraction,
                    "tissue_fraction": tissue_fraction,
                    "background_fraction": background_fraction,
                },
                "image_stats_bundle": [],
                "step": {
                    "review_goal": review_goal,
                    "stage_gate": "serrated_overview",
                    "metadata": metadata or {},
                },
                "cluster": {
                    "cluster_id": "cluster_00",
                    "l": trace_label,
                    "s": priority,
                    "d": True,
                    "crypt_disorder_risk": priority,
                    "metadata": metadata or {},
                },
            },
        }
        return HeuristicStageBackend().invoke(request, self._bundle())["output"]

    def test_trace_labels_route_to_expected_branches(self):
        self.assertEqual(_trace_branch_for_label("serrated"), "serrated")
        self.assertEqual(_trace_branch_for_label("conventional"), "conventional_adenoma")
        self.assertEqual(_trace_branch_for_label("normal"), "normal")
        self.assertEqual(_trace_branch_for_label("background"), "background")
        self.assertEqual(_normalize_trace_label("ssl_like_mucosa", "", "", "", True, 4), "serrated")
        self.assertEqual(_normalize_trace_label("tubular_adenoma_like", "", "", "", True, 3), "conventional")
        self.assertEqual(_normalize_trace_label("inflammatory_polyp_like", "", "", "", False, 1), "normal")
        self.assertEqual(_trace_branch_for_label("ssl_like_mucosa"), "serrated")
        self.assertEqual(_trace_branch_for_label("tubular_adenoma_like"), "conventional_adenoma")

    def test_serrated_initial_navigation_uses_2p5_overview_and_5x_ssl_hp_tsa_only(self):
        output = self._navigate("serrated", priority=5, high_mag=True)
        inspect_steps = [step for step in output["steps"] if step["metadata"].get("action") == "inspect"]
        self.assertEqual(inspect_steps[0]["m"], 2.5)
        self.assertEqual(inspect_steps[0]["review_goal"], "serrated_overview_assessment")
        self.assertIn("ssl_assessment", [step["review_goal"] for step in inspect_steps if step["m"] == 5.0])
        self.assertIn("hp_assessment", [step["review_goal"] for step in inspect_steps if step["m"] == 5.0])
        self.assertIn("tsa_assessment", [step["review_goal"] for step in inspect_steps if step["m"] == 5.0])
        self.assertFalse([step for step in inspect_steps if step["m"] == 10.0])

    def test_conventional_initial_navigation_uses_2p5_overview_and_5x_architecture_reactive_only(self):
        output = self._navigate("conventional", priority=4, high_mag=True)
        inspect_steps = [step for step in output["steps"] if step["metadata"].get("action") == "inspect"]
        self.assertEqual([step["m"] for step in inspect_steps], [2.5, 5.0, 5.0])
        self.assertEqual(
            [step["review_goal"] for step in inspect_steps],
            [
                "conventional_overview_assessment",
                "conventional_architecture_assessment",
                "reactive_regenerative_assessment",
            ],
        )

    def test_normal_navigation_uses_2p5_overview(self):
        output = self._navigate("normal", priority=1, high_mag=False)
        inspect_steps = [step for step in output["steps"] if step["metadata"].get("action") == "inspect"]
        self.assertEqual(len(inspect_steps), 1)
        self.assertEqual(inspect_steps[0]["m"], 2.5)
        self.assertEqual(inspect_steps[0]["review_goal"], "normal_overview_assessment")

    def test_serrated_overview_negative_defaults_to_conventional_recovery_hint(self):
        output = self._observe_step("serrated_overview_assessment", priority=4, pale_fraction=0.02, tissue_fraction=0.7)
        self.assertEqual(output["stage_decision"], "serrated_overview_not_supported_or_indeterminate")
        self.assertEqual(output["branch_recovery_hint"], "conventional")

    def test_serrated_overview_negative_can_direct_to_normal_recovery_hint(self):
        output = self._observe_step(
            "serrated_overview_assessment",
            priority=4,
            pale_fraction=0.02,
            tissue_fraction=0.2,
            background_fraction=0.7,
        )
        self.assertEqual(output["stage_decision"], "serrated_overview_not_supported_or_indeterminate")
        self.assertEqual(output["branch_recovery_hint"], "normal")

    def test_hp_final_class(self):
        output = self._report(
            "serrated",
            {
                "serrated_hits": {"serrated_lesion_context": "supporting", "serrated_surface_pattern": "supporting"},
                "hp_hits": {"surface_limited_serration": "supporting", "straight_crypt_bases": "supporting"},
            },
        )
        self.assertEqual(output["hierarchical_prediction"]["final_11_class"], "HP")

    def test_hp_assessment_populates_hp_hits_without_dysplasia_suffix(self):
        output = self._observe_step(
            "hp_assessment",
            priority=3,
            pale_fraction=0.14,
            tissue_fraction=0.7,
        )
        self.assertEqual(output["stage_decision"], "hp_architecture_supported")
        self.assertEqual(output["hp_hits"]["surface_limited_serration"], "supporting")
        self.assertEqual(output["hp_hits"]["straight_crypt_bases"], "supporting")

        report = self._report(
            "serrated",
            {
                "serrated_hits": {"serrated_lesion_context": "supporting", "serrated_surface_pattern": "supporting"},
                "hp_hits": {
                    "surface_limited_serration": "supporting",
                    "straight_crypt_bases": "supporting",
                    "lacks_basal_architectural_distortion": "supporting",
                },
                "serrated_dysplasia_hits": {
                    "nuclear_enlargement_stratification": "supporting",
                    "hyperchromasia": "supporting",
                },
            },
        )
        self.assertEqual(report["hierarchical_prediction"]["final_11_class"], "HP")

    def test_reactive_regenerative_assessment_supports_inflammatory_when_architecture_absent(self):
        output = self._observe_step(
            "reactive_regenerative_assessment",
            trace_label="conventional",
            priority=2,
            pale_fraction=0.02,
            tissue_fraction=0.7,
        )
        self.assertEqual(output["stage_decision"], "reactive_regenerative_supported")
        self.assertEqual(output["inflammatory_hits"]["reactive_regenerative_change"], "supporting")

        report = self._report(
            "conventional",
            {
                "inflammatory_hits": {
                    "erosion": "supporting",
                    "mixed_inflammation": "supporting",
                    "reactive_regenerative_change": "supporting",
                    "lacks_adenomatous_or_serrated_architecture": "supporting",
                },
            },
        )
        self.assertEqual(report["hierarchical_prediction"]["final_11_class"], "Inflammatory")

    def test_reactive_regenerative_conflict_does_not_block_conventional_architecture(self):
        report = self._report(
            "conventional",
            {
                "conventional_hits": {
                    "tubular_architecture": "supporting",
                    "tubular_or_tubulovillous_architecture": "supporting",
                    "crowded_adenomatous_glands": "supporting",
                },
                "inflammatory_hits": {
                    "reactive_regenerative_change": "supporting",
                    "mixed_inflammation": "supporting",
                },
            },
        )
        self.assertEqual(report["hierarchical_prediction"]["final_11_class"], "Tubular adenoma")
        self.assertIn(
            "conventional_architecture_with_reactive_regenerative_mimic",
            report["hierarchical_prediction"]["final_case_assessment"]["conflicts"],
        )

    def test_tsa_with_dysplasia_final_class(self):
        output = self._report(
            "serrated",
            {
                "serrated_hits": {"serrated_lesion_context": "supporting", "serrated_surface_pattern": "supporting"},
                "abnormal_crypt_hits": {
                    "basal_dilatation": "supporting",
                    "crypt_branching": "supporting",
                    "horizontal_growth": "supporting",
                },
                "tsa_hits": {"ectopic_crypt_formation": "supporting", "eosinophilic_cytoplasm": "supporting"},
                "serrated_dysplasia_hits": {
                    "nuclear_enlargement_stratification": "supporting",
                    "hyperchromasia": "supporting",
                    "architectural_crowding": "supporting",
                },
            },
        )
        self.assertEqual(output["hierarchical_prediction"]["final_11_class"], "TSAD")

    def test_tubulovillous_with_dysplasia_final_class(self):
        output = self._report(
            "conventional",
            {
                "conventional_hits": {
                    "tubular_or_tubulovillous_architecture": "supporting",
                    "villous_component": "supporting",
                    "crowded_adenomatous_glands": "supporting",
                },
                "conventional_dysplasia_hits": {
                    "nuclear_enlargement_stratification": "supporting",
                    "hyperchromasia": "supporting",
                    "architectural_crowding": "supporting",
                },
            },
        )
        self.assertEqual(output["hierarchical_prediction"]["final_11_class"], "TVAD")

    def test_low_grade_like_dysplasia_does_not_add_d_suffix(self):
        output = self._report(
            "conventional",
            {
                "conventional_hits": {
                    "tubular_architecture": "supporting",
                    "tubular_or_tubulovillous_architecture": "supporting",
                    "crowded_adenomatous_glands": "supporting",
                },
                "conventional_dysplasia_hits": {
                    "nuclear_enlargement_stratification": "supporting",
                    "hyperchromasia": "supporting",
                },
            },
        )
        hierarchy = output["hierarchical_prediction"]
        self.assertEqual(hierarchy["final_11_class"], "Tubular adenoma")
        self.assertFalse(hierarchy["final_case_assessment"]["high_grade_or_definite_dysplasia"])

    def test_unclassified_serrated_adenoma_requires_serrated_mixed_morphology(self):
        output = self._report(
            "serrated",
            {
                "serrated_hits": {"serrated_lesion_context": "supporting", "serrated_surface_pattern": "supporting"},
                "conventional_hits": {"villous_component": "supporting"},
            },
        )
        self.assertEqual(output["hierarchical_prediction"]["final_11_class"], "Unclassified serrated adenoma")

    def test_insufficient_evidence_does_not_default_to_inflammatory_or_usa(self):
        output = self._report("normal", {})
        hierarchy = output["hierarchical_prediction"]
        self.assertIsNone(hierarchy["final_11_class"])
        self.assertEqual(hierarchy["classification_status"], "insufficient_evidence")
        self.assertNotEqual(hierarchy["final_case_assessment"]["label"], "Inflammatory")
        self.assertNotEqual(hierarchy["final_case_assessment"]["label"], "Unclassified serrated adenoma")

    def test_inflammatory_final_class(self):
        output = self._report(
            "conventional",
            {
                "inflammatory_hits": {
                    "erosion": "supporting",
                    "mixed_inflammation": "supporting",
                    "reactive_regenerative_change": "supporting",
                }
            },
        )
        self.assertEqual(output["hierarchical_prediction"]["final_11_class"], "Inflammatory")

    def test_normal_path_can_resolve_to_inflammatory(self):
        output = self._report(
            "normal",
            {
                "inflammatory_hits": {
                    "erosion": "supporting",
                    "mixed_inflammation": "supporting",
                    "reactive_regenerative_change": "supporting",
                }
            },
        )
        self.assertEqual(output["hierarchical_prediction"]["final_11_class"], "Inflammatory")

    def test_serrated_without_subtype_support_is_insufficient(self):
        output = self._report(
            "serrated",
            {"serrated_hits": {"serrated_lesion_context": "supporting"}},
        )
        hierarchy = output["hierarchical_prediction"]
        self.assertIsNone(hierarchy["final_11_class"])
        self.assertEqual(hierarchy["classification_status"], "insufficient_evidence")

    def test_background_path_is_non_diagnostic(self):
        output = self._report("background", {})
        hierarchy = output["hierarchical_prediction"]
        self.assertIsNone(hierarchy["final_11_class"])
        self.assertEqual(hierarchy["classification_status"], "non_diagnostic")


if __name__ == "__main__":
    unittest.main()
