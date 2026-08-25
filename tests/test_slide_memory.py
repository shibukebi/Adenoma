import math
import unittest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from adenoma_agent.slide_memory import SlideMemoryTracker


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for SlideMemoryTracker behavior tests")
class SlideMemoryTrackerTest(unittest.TestCase):
    def test_process_patch_returns_required_fields(self):
        tracker = SlideMemoryTracker(feature_dim=4, hidden_dim=4, warm_up_steps=1)
        result = tracker.process_patch(torch.zeros(1, 4))

        self.assertEqual(set(result), {"raw_surprise", "calibrated_alarm_level", "trigger_trace_agent"})
        self.assertTrue(math.isfinite(result["raw_surprise"]))
        self.assertTrue(math.isfinite(result["calibrated_alarm_level"]))
        self.assertIs(type(result["trigger_trace_agent"]), bool)

    def test_reset_slide_reinitializes_parameters_and_state(self):
        torch.manual_seed(123)
        tracker = SlideMemoryTracker(feature_dim=4, hidden_dim=4, warm_up_steps=2)
        before = [parameter.detach().clone() for parameter in tracker.model.parameters()]
        tracker.process_patch(torch.ones(4))

        self.assertEqual(tracker.step_count, 1)
        self.assertEqual(len(tracker.warmup_surprises), 1)
        self.assertEqual(len(tracker.surprise_history), 1)

        tracker.reset_slide()
        after = [parameter.detach().clone() for parameter in tracker.model.parameters()]

        self.assertEqual(tracker.step_count, 0)
        self.assertEqual(tracker.warmup_surprises, [])
        self.assertEqual(len(tracker.surprise_history), 0)
        self.assertIsNone(tracker.surprise_threshold)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))

    def test_warmup_always_updates_and_records_surprise(self):
        tracker = SlideMemoryTracker(feature_dim=4, hidden_dim=4, lr=0.1, warm_up_steps=2)
        before = [parameter.detach().clone() for parameter in tracker.model.parameters()]
        tracker.process_patch(torch.ones(1, 4))
        after = [parameter.detach().clone() for parameter in tracker.model.parameters()]

        self.assertEqual(tracker.step_count, 1)
        self.assertEqual(len(tracker.warmup_surprises), 1)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))

    def test_familiar_patch_applies_decay_after_warmup(self):
        tracker = SlideMemoryTracker(feature_dim=4, hidden_dim=4, lr=0.1, warm_up_steps=1, decay_rate=0.5)
        tracker.process_patch(torch.zeros(1, 4))
        tracker.surprise_threshold = 1.0e9
        before = [parameter.detach().clone() for parameter in tracker.model.parameters()]

        result = tracker.process_patch(torch.zeros(1, 4))
        after = [parameter.detach().clone() for parameter in tracker.model.parameters()]

        self.assertFalse(result["trigger_trace_agent"])
        for old, new in zip(before, after):
            self.assertTrue(torch.allclose(new, old * 0.5))

    def test_sustained_high_surprise_triggers_trace_agent(self):
        tracker = SlideMemoryTracker(
            feature_dim=4,
            hidden_dim=4,
            warm_up_steps=0,
            window_size=3,
            rapid_decay_slope=-0.05,
        )
        tracker.surprise_threshold = 0.01
        tracker.surprise_history.extend([1.0, 1.05])

        result = tracker.process_patch(torch.full((1, 4), 5.0))

        self.assertTrue(result["trigger_trace_agent"])
        self.assertGreater(result["calibrated_alarm_level"], 0.0)
        self.assertGreaterEqual(tracker.last_slope, tracker.rapid_decay_slope)

    def test_rapid_negative_slope_suppresses_alarm(self):
        tracker = SlideMemoryTracker(
            feature_dim=4,
            hidden_dim=4,
            warm_up_steps=0,
            window_size=4,
            rapid_decay_slope=-0.05,
        )
        tracker.surprise_threshold = 0.01
        tracker.surprise_history.extend([10.0, 8.0, 6.0])

        result = tracker.process_patch(torch.full((1, 4), 5.0))

        self.assertFalse(result["trigger_trace_agent"])
        self.assertEqual(result["calibrated_alarm_level"], 0.0)
        self.assertLessEqual(tracker.last_slope, tracker.rapid_decay_slope)

    def test_invalid_feature_shape_raises(self):
        tracker = SlideMemoryTracker(feature_dim=4, hidden_dim=4)

        with self.assertRaises(ValueError):
            tracker.process_patch(torch.zeros(2, 4))
        with self.assertRaises(ValueError):
            tracker.process_patch(torch.zeros(5))
        with self.assertRaises(ValueError):
            tracker.process_patch(torch.zeros(1, 1, 4))


if __name__ == "__main__":
    unittest.main()
