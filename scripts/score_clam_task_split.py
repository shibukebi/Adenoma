#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAM_ROOT = PROJECT_ROOT.parent / "CLAM"
sys.path.insert(0, str(CLAM_ROOT))

from utils.eval_utils import initiate_model  # noqa: E402

from clam_experiment_utils import (  # noqa: E402
    attach_task_metadata,
    get_task_spec,
    load_split_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an arbitrary manifest split with a trained CLAM/MIL/ABMIL checkpoint.")
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split-name", choices=["train", "val", "test"], default="test")
    parser.add_argument("--task-mode", choices=["ssl_binary", "dysplasia_binary"], required=True)
    parser.add_argument("--model-type", choices=["mil", "abmil", "clam_sb", "clam_mb"], required=True)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--drop-out", type=float, default=0.25)
    parser.add_argument("--model-size", default="small")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_spec = get_task_spec(args.task_mode)
    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str, "case_id": str, "label": str})
    split_ids = load_split_ids(Path(args.split_dir), args.fold)[args.split_name]
    subset_df = ready_df[ready_df["slide_id"].isin(split_ids)].copy()
    subset_df = subset_df.set_index("slide_id").reindex(split_ids).reset_index()

    model = initiate_model(
        SimpleNamespace(
            drop_out=args.drop_out,
            n_classes=2,
            embed_dim=args.embed_dim,
            model_size=args.model_size,
            model_type=args.model_type,
        ),
        args.checkpoint,
    )
    device = next(model.parameters()).device

    rows = []
    for row in subset_df.itertuples(index=False):
        feature_path = Path(args.feature_dir) / "pt_files" / f"{row.slide_id}.pt"
        features = torch.load(feature_path, map_location="cpu").to(device)
        with torch.inference_mode():
            logits, y_prob, y_hat, _, _ = model(features)
        prob = y_prob.detach().cpu().numpy().reshape(-1)
        label_name = getattr(row, task_spec["label_name_col"], "")
        label_value = getattr(row, task_spec["label_value_col"], -1)
        label_name = "" if pd.isna(label_name) else str(label_name)
        label_value = -1 if pd.isna(label_value) else int(label_value)
        pred = int(y_hat.item())

        rows.append(
            {
                "slide_id": str(row.slide_id),
                "fold": args.fold,
                "split": args.split_name,
                "label": label_value,
                "label_name": label_name,
                "pred": pred,
                "pred_name": task_spec["inv_label_dict"][pred],
                task_spec["negative_score_col"]: float(prob[0]),
                task_spec["positive_score_col"]: float(prob[1]),
            }
        )

    predictions_df = pd.DataFrame(rows)
    predictions_df = attach_task_metadata(predictions_df, subset_df)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"output_csv={output_path}")
    print(f"rows={len(predictions_df)}")


if __name__ == "__main__":
    main()
