import torch
from torch import nn
import torch.nn.functional as F


OUTPUT_DIM = 7


class MultiHeadOutputMixin(object):
    @staticmethod
    def split_logits(logits):
        return {
            "quality": logits[:, :1],
            "architecture": logits[:, 1:4],
            "context": logits[:, 4:7],
        }


class DirectHead(nn.Module, MultiHeadOutputMixin):
    def __init__(self, feature_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, OUTPUT_DIM),
        )

    def forward(self, features, coordinates=None):
        return self.head(features)


class LocalMaxModel(nn.Module, MultiHeadOutputMixin):
    def __init__(self, feature_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.patch_head = DirectHead(feature_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, features, coordinates=None):
        batch, tokens, feature_dim = features.shape
        logits = self.patch_head(features.reshape(batch * tokens, feature_dim))
        return logits.reshape(batch, tokens, OUTPUT_DIM).max(dim=1).values


class MeanPoolModel(nn.Module, MultiHeadOutputMixin):
    def __init__(self, feature_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.head = DirectHead(feature_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, features, coordinates=None):
        return self.head(features.mean(dim=1))


class GatedAttentionModel(nn.Module, MultiHeadOutputMixin):
    def __init__(self, feature_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.feature_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU())
        self.coordinate_projection = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.attention_v = nn.Linear(hidden_dim, hidden_dim)
        self.attention_u = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, OUTPUT_DIM))

    def forward(self, features, coordinates):
        hidden = self.feature_projection(features) + self.coordinate_projection(coordinates)
        attention = self.attention_w(torch.tanh(self.attention_v(hidden)) * torch.sigmoid(self.attention_u(hidden)))
        attention = torch.softmax(attention, dim=1)
        pooled = torch.sum(attention * hidden, dim=1)
        return self.output(pooled)


class SpatialTransformerModel(nn.Module, MultiHeadOutputMixin):
    def __init__(self, feature_dim, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim))
        self.coordinate_projection = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, OUTPUT_DIM))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, features, coordinates):
        hidden = self.feature_projection(features) + self.coordinate_projection(coordinates)
        cls = self.cls_token.expand(hidden.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, hidden], dim=1))
        return self.output(encoded[:, 0])


def build_model(variant, feature_dim, hidden_dim=256, dropout=0.1):
    variant = str(variant).upper()
    if variant in ("B", "C"):
        return DirectHead(feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    if variant == "A":
        return LocalMaxModel(feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    if variant == "D0":
        return MeanPoolModel(feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    if variant == "D1":
        return GatedAttentionModel(feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    if variant == "D2":
        return SpatialTransformerModel(feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError("Unsupported architecture model variant: {0}".format(variant))


def masked_multitask_bce(logits, targets, masks, positive_weights=None):
    if logits.shape != targets.shape or logits.shape != masks.shape:
        raise ValueError("logits, targets, and masks must have identical shapes")
    weight = None
    if positive_weights is not None:
        positive_weights = positive_weights.to(device=logits.device, dtype=logits.dtype)
        weight = torch.where(targets > 0.5, positive_weights.unsqueeze(0), torch.ones_like(targets))
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weight is not None:
        loss = loss * weight
    loss = loss * masks
    denominator = masks.sum().clamp_min(1.0)
    return loss.sum() / denominator
