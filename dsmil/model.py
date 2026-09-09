from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BagClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 2, attn_dim: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.query = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.Tanh(),
        )
        self.value = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.class_conv = nn.Conv1d(self.n_classes, self.n_classes, kernel_size=input_dim, bias=True)

    def forward(self, feats: torch.Tensor, instance_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = self.value(feats)
        queries = self.query(feats)

        sorted_indices = torch.argsort(instance_logits, dim=0, descending=True)
        top_indices = sorted_indices[0]
        critical_feats = feats.index_select(0, top_indices)
        critical_queries = self.query(critical_feats)

        attn_scores = torch.matmul(queries, critical_queries.transpose(0, 1))
        attn_scores = attn_scores / math.sqrt(max(1, critical_queries.shape[-1]))
        attention = torch.softmax(attn_scores, dim=0)

        bag_repr = torch.einsum("nc,nk->ck", attention, values).unsqueeze(0)
        bag_logits = self.class_conv(bag_repr).view(1, -1)
        return bag_logits, attention, bag_repr


class SingleStreamDSMIL(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        n_classes: int = 2,
        attn_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.instance_classifier = nn.Linear(hidden_dim, n_classes)
        self.bag_classifier = BagClassifier(input_dim=hidden_dim, n_classes=n_classes, attn_dim=attn_dim, dropout=dropout)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.dim() == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        if features.dim() != 2:
            raise ValueError(f"Expected bag features shaped [N, D], got {tuple(features.shape)}")

        hidden = self.feature_proj(features.float())
        instance_logits = self.instance_classifier(hidden)
        bag_logits, attention, bag_repr = self.bag_classifier(hidden, instance_logits)
        max_instance_logits, max_instance_indices = torch.max(instance_logits, dim=0)
        return {
            "hidden": hidden,
            "instance_logits": instance_logits,
            "bag_logits": bag_logits,
            "attention": attention,
            "bag_repr": bag_repr,
            "max_instance_logits": max_instance_logits.unsqueeze(0),
            "max_instance_indices": max_instance_indices,
        }


class DualStreamDSMIL(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        n_classes: int = 2,
        attn_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.stream_a = SingleStreamDSMIL(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            attn_dim=attn_dim,
            dropout=dropout,
        )
        self.stream_b = SingleStreamDSMIL(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            attn_dim=attn_dim,
            dropout=dropout,
        )

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        label: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        stream_a = self.stream_a(features_a)
        stream_b = self.stream_b(features_b)

        fused_bag_logits = 0.5 * (stream_a["bag_logits"] + stream_b["bag_logits"])
        fused_instance_logits = 0.5 * (stream_a["max_instance_logits"] + stream_b["max_instance_logits"])
        logits = 0.5 * (fused_bag_logits + fused_instance_logits)

        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.argmax(y_prob, dim=1)

        results = {
            "bag_logits": fused_bag_logits,
            "max_instance_logits": fused_instance_logits,
            "stream_outputs": {
                "stream_a": stream_a,
                "stream_b": stream_b,
            },
        }
        if return_features:
            pooled_a = stream_a["bag_repr"].mean(dim=1)
            pooled_b = stream_b["bag_repr"].mean(dim=1)
            results["features"] = torch.cat([pooled_a, pooled_b], dim=1)

        return logits, y_prob, y_hat, None, results
