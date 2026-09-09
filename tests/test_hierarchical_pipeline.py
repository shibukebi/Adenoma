from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "CLAM"))

from clam_experiment_utils import (  # noqa: E402
    FINAL_LABEL_DICT,
    build_hierarchical_label_row,
    merge_hierarchical_predictions,
    prepare_task_row,
    select_routing_threshold,
)
from models.model_abmil import ABMIL  # noqa: E402


def test_build_hierarchical_label_row_ssl_high_maps_to_dysplasia() -> None:
    row = build_hierarchical_label_row("slide_a", "Sessile serrated adenoma", "high")
    assert row["ssl_label_name"] == "SSL"
    assert row["ssl_label"] == 1
    assert row["dysplasia_label_name"] == "dysplasia"
    assert row["dysplasia_label"] == 1
    assert row["final_label_name"] == "SSL-with-dysplasia"
    assert row["final_label"] == FINAL_LABEL_DICT["SSL-with-dysplasia"]


def test_prepare_task_row_stage2_filters_non_ssl() -> None:
    raw_row = {"slide_id": "slide_b", "type": "Hyperplastic polyps", "grade": "low"}
    assert prepare_task_row(raw_row, "dysplasia_binary") is None


def test_select_routing_threshold_prefers_perfect_cutoff() -> None:
    predictions_df = pd.DataFrame(
        {
            "label": [1, 1, 0, 0],
            "prob_ssl": [0.9, 0.4, 0.3, 0.2],
        }
    )
    payload = select_routing_threshold(predictions_df, positive_score_col="prob_ssl", positive_label=1)
    assert np.isclose(payload["threshold"], 0.4)
    assert np.isclose(payload["f2"], 1.0)


def test_merge_hierarchical_predictions_handles_routed_false_positive() -> None:
    stage1_test_df = pd.DataFrame(
        [
            {
                "slide_id": "slide_c",
                "label": 0,
                "label_name": "others",
                "pred": 1,
                "pred_name": "SSL",
                "prob_others": 0.2,
                "prob_ssl": 0.8,
            },
            {
                "slide_id": "slide_d",
                "label": 1,
                "label_name": "SSL",
                "pred": 1,
                "pred_name": "SSL",
                "prob_others": 0.1,
                "prob_ssl": 0.9,
            },
        ]
    )
    stage2_full_test_df = pd.DataFrame(
        [
            {
                "slide_id": "slide_c",
                "label": -1,
                "label_name": "",
                "pred": 1,
                "pred_name": "dysplasia",
                "prob_no_dysplasia": 0.25,
                "prob_dysplasia": 0.75,
            },
            {
                "slide_id": "slide_d",
                "label": 0,
                "label_name": "no_dysplasia",
                "pred": 0,
                "pred_name": "no_dysplasia",
                "prob_no_dysplasia": 0.8,
                "prob_dysplasia": 0.2,
            },
        ]
    )
    label_df = pd.DataFrame(
        [
            build_hierarchical_label_row("slide_c", "Hyperplastic polyps", "low"),
            build_hierarchical_label_row("slide_d", "Sessile serrated adenoma", "low"),
        ]
    )

    merged_df = merge_hierarchical_predictions(stage1_test_df, stage2_full_test_df, label_df, routing_threshold=0.5)
    assert merged_df.loc[merged_df["slide_id"] == "slide_c", "final_pred_name"].item() == "SSL-with-dysplasia"
    assert merged_df.loc[merged_df["slide_id"] == "slide_d", "final_pred_name"].item() == "SSL-no-dysplasia"


def test_abmil_forward_shapes() -> None:
    model = ABMIL(embed_dim=1024, n_classes=2, dropout=0.25)
    features = torch.randn(32, 1024)
    logits, probs, preds, attention, results = model(features, return_features=True)
    assert logits.shape == (1, 2)
    assert probs.shape == (1, 2)
    assert preds.shape == (1, 1)
    assert attention.shape == (1, 32)
    assert results["features"].shape == (1, 512)
