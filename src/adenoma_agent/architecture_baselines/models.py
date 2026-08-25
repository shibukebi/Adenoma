"""Small heads over frozen CONCH embeddings for architecture baselines.

The models in this module never instantiate or own a foundation encoder.  A
caller supplies a batch of cached ``[batch, patches, 512]`` embeddings.  Every
forward method returns the same mapping so training and explanation code does
not need model-specific branches.
"""

from typing import Dict

import torch
from torch import nn


DEFAULT_FEATURE_DIM = 512
NUM_CLASSES = 7


def _require_bag(features: torch.Tensor) -> None:
    if features.ndim != 3:
        raise ValueError("MIL features must have shape [batch, patches, feature_dim]")
    if features.shape[1] < 1:
        raise ValueError("MIL bags must contain at least one patch")


def _output(
    logits: torch.Tensor,
    attention: torch.Tensor = None,
    instance_scores: torch.Tensor = None,
    **extra,
) -> Dict[str, torch.Tensor]:
    """Construct the common public model-output contract."""

    result = {
        "logits": logits,
        "attention": attention,
        "instance_scores": instance_scores,
    }
    result.update(extra)
    return result


class LinearProbe(nn.Module):
    """A single linear multiclass classifier for annotated patch embeddings."""

    def __init__(self, feature_dim: int = DEFAULT_FEATURE_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.architecture_baseline_config = {
            "feature_dim": int(feature_dim),
            "num_classes": int(num_classes),
        }
        self.classifier = nn.Linear(int(feature_dim), int(num_classes))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.ndim not in (2, 3):
            raise ValueError("LinearProbe features must be [items, dim] or [batch, items, dim]")
        logits = self.classifier(features)
        return _output(logits=logits, attention=None, instance_scores=logits)


class SmallMLP(nn.Module):
    """A deliberately small non-linear probe for annotated patch embeddings."""

    def __init__(
        self,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.architecture_baseline_config = {
            "feature_dim": int(feature_dim),
            "num_classes": int(num_classes),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
        }
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.ndim not in (2, 3):
            raise ValueError("SmallMLP features must be [items, dim] or [batch, items, dim]")
        logits = self.classifier(features)
        return _output(logits=logits, attention=None, instance_scores=logits)


class MeanPoolMIL(nn.Module):
    """The required simple weak-supervision pooling control."""

    def __init__(self, feature_dim: int = DEFAULT_FEATURE_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.architecture_baseline_config = {
            "feature_dim": int(feature_dim),
            "num_classes": int(num_classes),
        }
        self.normalization = nn.LayerNorm(int(feature_dim))
        self.classifier = nn.Linear(int(feature_dim), int(num_classes))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        _require_bag(features)
        instance_scores = self.classifier(self.normalization(features))
        logits = self.classifier(self.normalization(features.mean(dim=1)))
        patch_count = features.shape[1]
        attention = features.new_full((features.shape[0], patch_count), 1.0 / float(patch_count))
        return _output(logits=logits, attention=attention, instance_scores=instance_scores)


class GatedABMIL(nn.Module):
    """Feature-only gated-attention MIL (ABMIL) for slide weak supervision."""

    def __init__(
        self,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.architecture_baseline_config = {
            "feature_dim": int(feature_dim),
            "num_classes": int(num_classes),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
        }
        self.projection = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.attention_v = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.attention_u = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.attention_w = nn.Linear(int(hidden_dim), 1)
        self.classifier = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(int(hidden_dim), int(num_classes)))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        _require_bag(features)
        hidden = self.projection(features)
        raw_attention = self.attention_w(
            torch.tanh(self.attention_v(hidden)) * torch.sigmoid(self.attention_u(hidden))
        ).squeeze(-1)
        attention = torch.softmax(raw_attention, dim=1)
        pooled = torch.sum(hidden * attention.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
        # Applying the same bag classifier per patch yields a useful ranking
        # without adding an untrained, explanation-only parameter branch.
        instance_scores = self.classifier(hidden)
        return _output(logits=logits, attention=attention, instance_scores=instance_scores)


class DSMIL(nn.Module):
    """A compact seven-class dual-stream MIL implementation.

    ``logits`` are the bag-stream slide logits. ``max_instance_logits`` are
    the class-specific maxima of the instance stream and are consumed by the
    training loss with equal 0.5 weight.
    """

    def __init__(
        self,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.architecture_baseline_config = {
            "feature_dim": int(feature_dim),
            "num_classes": self.num_classes,
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
        }
        self.projection = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.instance_classifier = nn.Linear(int(hidden_dim), self.num_classes)
        self.query = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.value = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.bag_classifiers = nn.ModuleList(
            [nn.Linear(int(hidden_dim), 1) for _ in range(self.num_classes)]
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        _require_bag(features)
        hidden = self.projection(features)
        instance_scores = self.instance_classifier(hidden)
        max_instance_logits, selected_indices = instance_scores.max(dim=1)
        query = self.query(hidden)
        value = self.value(hidden)
        bag_logits = []
        class_attention = []
        for class_index, classifier in enumerate(self.bag_classifiers):
            indices = selected_indices[:, class_index]
            selected_hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
            critical_query = self.query(selected_hidden)
            similarity = torch.sum(query * critical_query.unsqueeze(1), dim=-1)
            similarity = similarity / float(query.shape[-1]) ** 0.5
            attention_for_class = torch.softmax(similarity, dim=1)
            bag_representation = torch.sum(value * attention_for_class.unsqueeze(-1), dim=1)
            bag_logits.append(classifier(bag_representation).squeeze(-1))
            class_attention.append(attention_for_class)
        logits = torch.stack(bag_logits, dim=1)
        predicted_class = logits.argmax(dim=1)
        class_attention = torch.stack(class_attention, dim=-1)
        attention = class_attention.gather(
            2,
            predicted_class.view(-1, 1, 1).expand(-1, class_attention.shape[1], 1),
        ).squeeze(-1)
        return _output(
            logits=logits,
            attention=attention,
            instance_scores=instance_scores,
            max_instance_logits=max_instance_logits,
            selected_instance_indices=selected_indices,
            class_attention=class_attention,
        )


def build_model(
    name: str,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    num_classes: int = NUM_CLASSES,
    hidden_dim: int = 256,
    dropout: float = 0.25,
) -> nn.Module:
    """Build a named baseline with the frozen-CONCH default feature dimension."""

    normalized = str(name).strip().lower().replace("-", "_")
    common = {"feature_dim": feature_dim, "num_classes": num_classes}
    if normalized in ("linear", "linear_probe"):
        return LinearProbe(**common)
    if normalized in ("small_mlp", "mlp"):
        return SmallMLP(hidden_dim=hidden_dim, dropout=dropout, **common)
    if normalized in ("mean_pool", "meanpool", "mean_pool_mil"):
        return MeanPoolMIL(**common)
    if normalized in ("abmil", "gated_abmil", "gated_attention"):
        return GatedABMIL(hidden_dim=hidden_dim, dropout=dropout, **common)
    if normalized == "dsmil":
        return DSMIL(hidden_dim=hidden_dim, dropout=dropout, **common)
    raise ValueError("Unsupported architecture baseline model: {0}".format(name))
