from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnNetGated(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.25, n_classes: int = 1):
        super().__init__()
        self.attn_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attn_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attn_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.attn_a(x)
        b = self.attn_b(x)
        scores = self.attn_c(a * b)
        return scores, x


class SpatialGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.25, residual: bool = False):
        super().__init__()
        self.msg_linear = nn.Linear(in_dim, out_dim)
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = bool(residual and in_dim == out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.sparse.mm(adjacency, self.msg_linear(x))
        out = self.self_linear(x) + aggregated
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        if self.residual:
            out = out + x
        return out


class PatchGCN(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 4,
        n_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("PatchGCN requires at least 2 graph layers")

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_classes = n_classes

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.initial_conv = SpatialGraphConv(hidden_dim, hidden_dim, dropout=dropout, residual=False)
        self.layers = nn.ModuleList(
            [SpatialGraphConv(hidden_dim, hidden_dim, dropout=dropout, residual=True) for _ in range(num_layers - 1)]
        )

        pooled_dim = hidden_dim * num_layers
        self.path_phi = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.path_attention_head = AttnNetGated(
            input_dim=pooled_dim,
            hidden_dim=pooled_dim,
            dropout=dropout,
            n_classes=1,
        )
        self.path_rho = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(pooled_dim, n_classes)

    @staticmethod
    def build_sparse_adjacency(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        adjacency = torch.sparse_coo_tensor(
            indices=edge_index,
            values=edge_weight,
            size=(num_nodes, num_nodes),
            device=device,
        )
        return adjacency.coalesce()

    def forward(
        self,
        data: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        label: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        if data.ndim != 2:
            raise ValueError(f"Expected node features with shape [N, C], got {tuple(data.shape)}")

        x = self.fc(data.float())
        adjacency = self.build_sparse_adjacency(
            edge_index=edge_index.long(),
            edge_weight=edge_weight.float(),
            num_nodes=x.size(0),
            device=x.device,
        )

        features = []
        h = self.initial_conv(x, adjacency)
        features.append(h)
        for layer in self.layers:
            h = layer(h, adjacency)
            features.append(h)

        h = torch.cat(features, dim=1)
        h = self.path_phi(h)
        attention_scores, h = self.path_attention_head(h)
        attention_scores = torch.transpose(attention_scores, 1, 0)
        attention_raw = attention_scores
        attention_scores = F.softmax(attention_scores, dim=1)

        pooled = torch.mm(attention_scores, h)
        pooled = self.path_rho(pooled)

        logits = self.classifier(pooled)
        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.argmax(y_prob, dim=1)

        results = {}
        if return_features:
            results["features"] = pooled
        return logits, y_prob, y_hat, attention_raw, results
