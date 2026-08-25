import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adenoma_agent.architecture_baselines.evaluation import (
    classification_metrics,
    explain_weak_mil_prediction,
    family_cluster_bootstrap,
    holm_adjust,
    paired_family_bootstrap,
    patch_ranking_summary,
)

try:
    import torch

    from adenoma_agent.architecture_baselines.models import (
        DSMIL,
        GatedABMIL,
        LinearProbe,
        MeanPoolMIL,
        SmallMLP,
    )
    from adenoma_agent.architecture_baselines.training import (
        _output_loss,
        load_cached_bags,
        train_baseline,
        train_only_class_weights,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ArchitectureBaselineEvaluationTest(unittest.TestCase):
    def _predictions(self):
        rows = []
        for index in range(14):
            label = index % 7
            probabilities = [0.01] * 7
            probabilities[label] = 0.94
            rows.append(
                {
                    "case_alias": "case_{0}".format(index),
                    "family_id": "family_{0}".format(index // 2),
                    "label": label,
                    "probabilities": probabilities,
                }
            )
        return rows

    def test_metrics_bootstrap_and_holm_are_json_safe(self):
        predictions = self._predictions()
        metrics = classification_metrics(predictions)
        self.assertEqual(metrics["n_cases"], 14)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        bootstrap = family_cluster_bootstrap(predictions, n_bootstrap=20, seed=3)
        self.assertEqual(bootstrap["n_bootstrap"], 20)
        paired = paired_family_bootstrap(predictions, predictions, n_bootstrap=20, seed=3)
        self.assertAlmostEqual(paired["point_delta"], 0.0)
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.10})
        self.assertLessEqual(adjusted["a"]["holm_adjusted_p_value"], adjusted["b"]["holm_adjusted_p_value"])
        json.dumps({"metrics": metrics, "bootstrap": bootstrap, "paired": paired, "holm": adjusted})

    def test_patch_ranking_reports_concentration_and_spatial_components(self):
        patches = [
            {"patch_id": "p0", "grid_index": [0, 0], "level0_bbox": [0, 0, 10, 10]},
            {"patch_id": "p1", "grid_index": [0, 1], "level0_bbox": [10, 0, 20, 10]},
            {"patch_id": "p2", "grid_index": [5, 5], "level0_bbox": [50, 50, 60, 60]},
        ]
        summary = patch_ranking_summary(patches, [0.8, 0.15, 0.05], top_k=3)
        self.assertEqual(summary["top"][0]["patch_id"], "p0")
        self.assertEqual(summary["spatial_diversity"]["component_count"], 2)
        self.assertGreater(summary["concentration"]["top_1_percent"]["signal_mass"], 0.79)

    def test_weak_mil_explanation_keeps_candidate_relevance_boundary(self):
        bag = {
            "case_alias": "case_1",
            "patches": [
                {"patch_id": "p0", "grid_index": [0, 0], "level0_bbox": [0, 0, 10, 10]},
                {"patch_id": "p1", "grid_index": [0, 1], "level0_bbox": [10, 0, 20, 10]},
            ],
        }
        prediction = {
            "case_alias": "case_1",
            "prediction": 2,
            "class_attention": [[0.8, 0.2, 0.9], [0.2, 0.8, 0.1]],
            "instance_scores": [[0.0, 0.1, 2.0], [0.1, 0.0, 1.0]],
        }
        summary = explain_weak_mil_prediction(bag, prediction)
        self.assertEqual(summary["dsmil_attention"]["top"][0]["patch_id"], "p0")
        self.assertIn("not_pathologist_confirmed", summary["interpretation"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for architecture baseline model tests")
class ArchitectureBaselineModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.bag = torch.randn(2, 5, 512)

    def test_all_models_share_output_contract(self):
        models = [
            MeanPoolMIL(),
            GatedABMIL(),
            DSMIL(),
        ]
        for model in models:
            output = model(self.bag)
            self.assertEqual(set(("logits", "attention", "instance_scores")).issubset(output), True)
            self.assertEqual(tuple(output["logits"].shape), (2, 7))
            self.assertEqual(tuple(output["attention"].shape), (2, 5))
            self.assertEqual(tuple(output["instance_scores"].shape), (2, 5, 7))
        dsmil_output = models[-1](self.bag)
        self.assertEqual(tuple(dsmil_output["class_attention"].shape), (2, 5, 7))
        self.assertTrue(torch.allclose(dsmil_output["class_attention"].sum(dim=1), torch.ones(2, 7)))
        self.assertEqual(tuple(dsmil_output["max_instance_logits"].shape), (2, 7))

    def test_patch_probes_preserve_item_axes(self):
        patches = torch.randn(4, 512)
        self.assertEqual(tuple(LinearProbe()(patches)["logits"].shape), (4, 7))
        self.assertEqual(tuple(SmallMLP()(patches)["logits"].shape), (4, 7))

    def test_dsmil_uses_equal_bag_and_max_instance_losses(self):
        model = DSMIL()
        output = model(self.bag[:1])
        label = torch.tensor([2])
        weights = torch.ones(7)
        expected = 0.5 * torch.nn.functional.cross_entropy(output["logits"], label, weight=weights)
        expected += 0.5 * torch.nn.functional.cross_entropy(output["max_instance_logits"], label, weight=weights)
        self.assertTrue(torch.allclose(_output_loss(output, label, weights), expected))

    def test_slide_labels_cannot_be_applied_to_patch_probe_logits(self):
        output = LinearProbe()(self.bag[:1])
        with self.assertRaises(ValueError):
            _output_loss(output, torch.tensor([1]), torch.ones(7))

    def test_class_weights_are_fitted_from_train_rows_only(self):
        weights = train_only_class_weights([0, 0, 1, 2, 3, 4, 5, 6])
        self.assertLess(float(weights[0]), float(weights[1]))
        with self.assertRaises(ValueError):
            train_only_class_weights([0, 1])

    def test_loader_and_training_write_checkpoint_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_root = root / "cache"
            label_path = root / "labels.jsonl"
            label_rows = []
            for class_index in range(7):
                alias = "case_{0}".format(class_index)
                case_dir = cache_root / alias
                case_dir.mkdir(parents=True)
                np.save(str(case_dir / "features.npy"), np.full((2, 512), class_index, dtype=np.float16))
                index_rows = [
                    {
                        "case_alias": alias,
                        "patch_id": "{0}_{1}".format(alias, patch_index),
                        "grid_index": [0, patch_index],
                        "level0_bbox": [patch_index, 0, patch_index + 1, 1],
                        "mucosa_coverage": 0.8,
                        "image_sha256": "a" * 64,
                    }
                    for patch_index in range(2)
                ]
                (case_dir / "index.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in index_rows), encoding="utf-8"
                )
                (case_dir / "metadata.json").write_text(json.dumps({"embedding_dim": 512}), encoding="utf-8")
                label_rows.append({"case_alias": alias, "family_id": "family_{0}".format(class_index), "label_index": class_index})
            label_path.write_text("".join(json.dumps(row) + "\n" for row in label_rows), encoding="utf-8")
            bags = load_cached_bags(cache_root, label_path)
            self.assertEqual(len(bags), 7)
            summary = train_baseline(
                MeanPoolMIL(),
                bags,
                bags,
                root / "run",
                "mean_pool",
                max_epochs=1,
                patience=1,
                device="cpu",
            )
            self.assertTrue(Path(summary["checkpoint"]).is_file())
            self.assertTrue(Path(summary["val_predictions"]).is_file())


if __name__ == "__main__":
    unittest.main()
