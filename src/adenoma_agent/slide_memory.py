import math
from collections import deque


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "SlideMemoryTracker requires PyTorch. Install torch to use slide-specific memory tracking."
        ) from exc
    return torch


class SlideMemoryTracker(object):
    """Online per-slide memory that scores CONCH feature surprise."""

    def __init__(
        self,
        feature_dim=768,
        hidden_dim=768,
        lr=0.01,
        warm_up_steps=50,
        decay_rate=0.98,
        threshold_lambda=2.0,
        max_grad_norm=1.0,
        window_size=8,
        rapid_decay_slope=-0.05,
        huber_delta=1.0,
        device=None,
    ):
        self.torch = _require_torch()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.warm_up_steps = int(warm_up_steps)
        self.decay_rate = float(decay_rate)
        self.threshold_lambda = float(threshold_lambda)
        self.max_grad_norm = float(max_grad_norm)
        self.window_size = int(window_size)
        self.rapid_decay_slope = float(rapid_decay_slope)
        self.huber_delta = float(huber_delta)
        self.device = self.torch.device(device or "cpu")

        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.warm_up_steps < 0:
            raise ValueError("warm_up_steps must be non-negative")
        if not 0.0 <= self.decay_rate <= 1.0:
            raise ValueError("decay_rate must be in [0, 1]")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")

        self.reset_slide()

    def reset_slide(self):
        torch = self.torch
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.feature_dim),
        ).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.HuberLoss(delta=self.huber_delta)
        self.step_count = 0
        self.warmup_surprises = []
        self.surprise_history = deque(maxlen=self.window_size)
        self.surprise_threshold = None
        self.last_slope = 0.0
        return self

    def process_patch(self, conch_feature):
        feature = self._prepare_feature(conch_feature)
        raw_surprise, loss = self._loss_and_raw_surprise(feature)
        self.surprise_history.append(raw_surprise)

        is_warmup = self.step_count < self.warm_up_steps
        if is_warmup:
            self.warmup_surprises.append(raw_surprise)
            self._clip_and_step()
            trigger_trace_agent = False
            calibrated_alarm_level = 0.0
        else:
            threshold = self._surprise_threshold()
            is_high_surprise = raw_surprise > threshold
            if is_high_surprise:
                self._clip_and_step()
            else:
                self._decay_parameters()

            suppressed = False
            if is_high_surprise and len(self.surprise_history) >= min(2, self.window_size):
                self.last_slope = self._surprise_decay_slope()
                suppressed = self.last_slope <= self.rapid_decay_slope
            else:
                self.last_slope = self._surprise_decay_slope()

            trigger_trace_agent = bool(is_high_surprise and not suppressed)
            calibrated_alarm_level = 0.0
            if is_high_surprise and not suppressed:
                calibrated_alarm_level = max(0.0, raw_surprise / max(threshold, 1e-12) - 1.0)

        self.optimizer.zero_grad(set_to_none=True)
        self.step_count += 1
        return {
            "raw_surprise": float(raw_surprise),
            "calibrated_alarm_level": float(calibrated_alarm_level),
            "trigger_trace_agent": bool(trigger_trace_agent),
        }

    def _prepare_feature(self, conch_feature):
        torch = self.torch
        if not torch.is_tensor(conch_feature):
            raise ValueError("conch_feature must be a torch.Tensor")
        feature = conch_feature.detach().to(device=self.device, dtype=torch.float32)
        if feature.ndim == 1:
            if int(feature.shape[0]) != self.feature_dim:
                raise ValueError("conch_feature must have shape ({0},) or (1, {0})".format(self.feature_dim))
            feature = feature.unsqueeze(0)
        elif feature.ndim == 2:
            if int(feature.shape[0]) != 1 or int(feature.shape[1]) != self.feature_dim:
                raise ValueError("conch_feature must have shape ({0},) or (1, {0})".format(self.feature_dim))
        else:
            raise ValueError("conch_feature must have shape ({0},) or (1, {0})".format(self.feature_dim))
        return feature

    def _loss_and_raw_surprise(self, feature):
        self.optimizer.zero_grad(set_to_none=True)
        reconstruction = self.model(feature)
        loss = self.loss_fn(reconstruction, feature)
        loss.backward()
        return self._gradient_norm(), loss

    def _gradient_norm(self):
        total = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            grad_norm = parameter.grad.detach().norm(2).item()
            total += grad_norm * grad_norm
        return math.sqrt(total)

    def _clip_and_step(self):
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def _decay_parameters(self):
        with self.torch.no_grad():
            for parameter in self.model.parameters():
                parameter.mul_(self.decay_rate)

    def _surprise_threshold(self):
        if self.surprise_threshold is None:
            if self.warmup_surprises:
                values = self.torch.tensor(self.warmup_surprises, dtype=self.torch.float32)
                mean = float(values.mean().item())
                std = float(values.std(unbiased=False).item()) if len(self.warmup_surprises) > 1 else 0.0
                self.surprise_threshold = mean + self.threshold_lambda * std
            else:
                self.surprise_threshold = 0.0
        return float(self.surprise_threshold)

    def _surprise_decay_slope(self):
        values = list(self.surprise_history)
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = float(n - 1) / 2.0
        y_mean = sum(values) / float(n)
        numerator = 0.0
        denominator = 0.0
        for index, value in enumerate(values):
            x_delta = float(index) - x_mean
            numerator += x_delta * (float(value) - y_mean)
            denominator += x_delta * x_delta
        if denominator == 0.0:
            return 0.0
        return numerator / denominator
