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

from dataset_modules.dataset_generic import Generic_MIL_Dataset  # noqa: E402
from clam_experiment_utils import attach_task_metadata, get_task_spec, load_split_ids  # noqa: E402
from patch_gcn.dataset import build_patch_gcn_split  # noqa: E402
from patch_gcn.model import PatchGCN  # noqa: E402
from patch_gcn.training import build_loader, load_torch_checkpoint, move_sample_to_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an arbitrary manifest split with a trained PatchGCN checkpoint.")
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--split-name", choices=["train", "val", "test"], default="test")
    parser.add_argument("--task-mode", choices=["ssl_binary", "dysplasia_binary"], required=True)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--drop-out", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_spec = get_task_spec(args.task_mode)
    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str, "case_id": str, "label": str})
    split_ids = load_split_ids(Path(args.split_dir), args.fold)[args.split_name]

    mil_dataset = Generic_MIL_Dataset(
        csv_path=args.ready_csv,
        data_dir=args.feature_dir,
        shuffle=False,
        seed=2023,
        print_info=True,
        label_dict=task_spec["label_dict"],
        ignore=[],
        patient_strat=False,
        label_col="label",
    )
    dataset = build_patch_gcn_split(
        slide_data=mil_dataset.slide_data,
        ids=split_ids,
        feature_dir=args.feature_dir,
        patch_dir=args.patch_dir,
        graph_dir=args.graph_dir,
        k_neighbors=args.k_neighbors,
    )
    loader = build_loader(dataset, training=False, weighted=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchGCN(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_classes=2,
        dropout=args.drop_out,
    )
    model.load_state_dict(load_torch_checkpoint(Path(args.checkpoint)))
    model = model.to(device)
    model.eval()

    rows = []
    for sample in loader:
        slide_id = str(sample["slide_id"])
        features, edge_index, edge_weight, label = move_sample_to_device(sample)
        with torch.inference_mode():
            logits, y_prob, y_hat, _, _ = model(features, edge_index, edge_weight)
        prob = y_prob.detach().cpu().numpy().reshape(-1)
        label_value = int(label.item())
        label_name = task_spec["inv_label_dict"].get(label_value, "")
        pred = int(y_hat.item())
        rows.append(
            {
                "slide_id": slide_id,
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
    predictions_df = attach_task_metadata(predictions_df, ready_df)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"output_csv={output_path}")
    print(f"rows={len(predictions_df)}")


if __name__ == "__main__":
    main()
