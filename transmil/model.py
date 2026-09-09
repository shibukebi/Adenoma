from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8, num_landmarks: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=max(1, dim // heads),
            heads=heads,
            num_landmarks=num_landmarks,
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        batch, _, channels = x.shape
        cls_token, feat_token = x[:, :1], x[:, 1:]
        feat_map = feat_token.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)
        feat_map = feat_map + self.proj(feat_map) + self.proj1(feat_map) + self.proj2(feat_map)
        feat_token = feat_map.flatten(2).transpose(1, 2)
        return torch.cat((cls_token, feat_token), dim=1)


class TransMIL(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        model_dim: int = 512,
        n_classes: int = 2,
        dropout: float = 0.25,
        heads: int = 8,
        num_landmarks: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.n_classes = n_classes

        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.layer1 = TransLayer(dim=model_dim, heads=heads, num_landmarks=num_landmarks, dropout=dropout)
        self.pos_layer = PPEG(dim=model_dim)
        self.layer2 = TransLayer(dim=model_dim, heads=heads, num_landmarks=num_landmarks, dropout=dropout)
        self.norm = nn.LayerNorm(model_dim)
        self.fc2 = nn.Linear(model_dim, n_classes)

    def forward(self, data: torch.Tensor, label: torch.Tensor | None = None, return_features: bool = False):
        if data.dim() == 2:
            data = data.unsqueeze(0)
        h = self.fc1(data.float())
        batch_size, num_instances, _ = h.shape

        grid_h = math.ceil(math.sqrt(num_instances))
        grid_w = grid_h
        add_length = grid_h * grid_w - num_instances
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, grid_h, grid_w)
        h = self.layer2(h)

        features = self.norm(h)[:, 0]
        logits = self.fc2(features)
        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.argmax(y_prob, dim=1)
        results = {}
        if return_features:
            results["features"] = features
        return logits, y_prob, y_hat, None, results
