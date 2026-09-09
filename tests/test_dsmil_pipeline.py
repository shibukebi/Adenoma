from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dsmil.dataset import build_dual_stream_split, collate_dual_stream  # noqa: E402
from dsmil.model import DualStreamDSMIL  # noqa: E402


def test_dual_stream_split_and_model_forward(tmp_path: Path) -> None:
    feature_dir_a = tmp_path / "features_a" / "pt_files"
    feature_dir_b = tmp_path / "features_b" / "pt_files"
    feature_dir_a.mkdir(parents=True)
    feature_dir_b.mkdir(parents=True)

    torch.save(torch.randn(12, 1024), feature_dir_a / "slide_a.pt")
    torch.save(torch.randn(20, 1024), feature_dir_b / "slide_a.pt")

    slide_data = pd.DataFrame(
        [
            {"slide_id": "slide_a", "label": 1},
        ]
    )
    dataset = build_dual_stream_split(
        slide_data=slide_data,
        ids=["slide_a"],
        feature_dir_a=feature_dir_a.parent,
        feature_dir_b=feature_dir_b.parent,
    )

    batch = collate_dual_stream([dataset[0]])
    features_a, features_b, label = batch
    assert features_a.shape == (12, 1024)
    assert features_b.shape == (20, 1024)
    assert label.tolist() == [1]

    model = DualStreamDSMIL(embed_dim=1024, hidden_dim=256, n_classes=2, attn_dim=64, dropout=0.1)
    logits, probs, preds, attention, results = model(features_a, features_b, return_features=True)
    assert logits.shape == (1, 2)
    assert probs.shape == (1, 2)
    assert preds.shape == (1,)
    assert attention is None
    assert results["bag_logits"].shape == (1, 2)
    assert results["max_instance_logits"].shape == (1, 2)
    assert results["features"].shape == (1, 512)
