import unittest


try:
    import torch

    from adenoma_agent.architecture_models import build_model, masked_multitask_bce
except Exception:
    torch = None


@unittest.skipIf(torch is None, "PyTorch architecture environment is not available")
class ArchitectureModelsTest(unittest.TestCase):
    def test_all_variants_emit_seven_logits(self):
        direct = torch.randn(2, 16)
        tokens = torch.randn(2, 4, 16)
        coordinates = torch.tensor(
            [[[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [0.5, 0.5, 1.0]]] * 2
        )
        for variant in ("A", "B", "C", "D0", "D1", "D2"):
            model = build_model(variant, feature_dim=16, hidden_dim=32, dropout=0.0)
            features = direct if variant in ("B", "C") else tokens
            logits = model(features, coordinates)
            self.assertEqual(tuple(logits.shape), (2, 7), variant)

    def test_spatial_models_use_coordinates(self):
        torch.manual_seed(17)
        features = torch.randn(1, 4, 16)
        coordinates = torch.tensor(
            [[[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [0.5, 0.5, 1.0]]]
        )
        permuted = coordinates[:, [3, 2, 1, 0], :]
        for variant in ("D1", "D2"):
            model = build_model(variant, feature_dim=16, hidden_dim=32, dropout=0.0).eval()
            first = model(features, coordinates)
            second = model(features, permuted)
            self.assertFalse(torch.allclose(first, second), variant)

    def test_masked_loss_backpropagates(self):
        model = build_model("D2", feature_dim=16, hidden_dim=32, dropout=0.0)
        features = torch.randn(2, 4, 16)
        coordinates = torch.randn(2, 4, 3)
        targets = torch.randint(0, 2, (2, 7), dtype=torch.float32)
        masks = torch.ones(2, 7)
        masks[0, 1:] = 0.0
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loss = masked_multitask_bce(model(features, coordinates), targets, masks)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))


if __name__ == "__main__":
    unittest.main()
