#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAM_ROOT = PROJECT_ROOT.parent / "CLAM"
sys.path.insert(0, str(CLAM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clam_experiment_utils import attach_task_metadata, get_task_spec, load_split_ids  # noqa: E402
from dsmil.dataset import build_dual_stream_split  # noqa: E402
from dsmil.model import DualStreamDSMIL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an arbitrary manifest split with a trained dual-stream DSMIL checkpoint.")
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--feature-dir-a", required=True)
    parser.add_argument("--feature-dir-b", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--split-name", choices=["train", "val", "test"], default="test")
    parser.add_argument("--task-mode", choices=["ssl_binary", "dysplasia_binary"], required=True)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--drop-out", type=float, default=0.25)
    return parser.parse_args()


def build_slide_frame(ready_df: pd.DataFrame, label_dict: dict[str, int]) -> pd.DataFrame:
    slide_data = ready_df.copy()
    if "label_value" in slide_data.columns:
        slide_data["label"] = slide_data["label_value"].astype(int)
    else:
        slide_data["label"] = slide_data["label_name"].map(label_dict).astype(int)
    slide_data["slide_id"] = slide_data["slide_id"].astype(str)
    return slide_data


def main() -> None:
    args = parse_args()
    task_spec = get_task_spec(args.task_mode)
    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str, "case_id": str, "label": str})
    slide_data = build_slide_frame(ready_df, task_spec["label_dict"])
    split_ids = load_split_ids(Path(args.split_dir), args.fold)[args.split_name]

    dataset = build_dual_stream_split(
        slide_data=slide_data,
        ids=split_ids,
        feature_dir_a=args.feature_dir_a,
        feature_dir_b=args.feature_dir_b,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamDSMIL(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_classes=2,
        attn_dim=args.attn_dim,
        dropout=args.drop_out,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    rows = []
    for index in range(len(dataset)):
        data_a, data_b, label = dataset[index]
        data_a = data_a.to(device)
        data_b = data_b.to(device)
        with torch.inference_mode():
            logits, y_prob, y_hat, _, _ = model(data_a, data_b)
        prob = y_prob.detach().cpu().numpy().reshape(-1)
        pred = int(y_hat.item())
        rows.append(
            {
                "slide_id": str(dataset.slide_data.iloc[index]["slide_id"]),
                "fold": args.fold,
                "split": args.split_name,
                "label": int(label),
                "label_name": task_spec["inv_label_dict"][int(label)],
                "pred": pred,
                "pred_name": task_spec["inv_label_dict"][pred],
                task_spec["negative_score_col"]: float(prob[0]),
                task_spec["positive_score_col"]: float(prob[1]),
            }
        )

    predictions_df = pd.DataFrame(rows)
    predictions_df = attach_task_metadata(predictions_df, ready_df)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"output_csv={output_path}")
    print(f"rows={len(predictions_df)}")


if __name__ == "__main__":
    main()
