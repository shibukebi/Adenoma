#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAM_ROOT = PROJECT_ROOT / "CLAM"
if not CLAM_ROOT.exists():
    CLAM_ROOT = PROJECT_ROOT.parent / "CLAM"
sys.path.insert(0, str(CLAM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dataset_modules.dataset_generic import Generic_MIL_Dataset, Generic_Split  # noqa: E402
from clam_experiment_utils import get_task_spec, load_split_ids  # noqa: E402
from transmil.model import TransMIL  # noqa: E402
from transmil.training import TrainConfig, run_transmil_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TransMIL experiment for either SSL-vs-others or dysplasia-vs-no_dysplasia.")
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--drop-out", type=float, default=0.25)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--weighted-sample", action="store_true", default=False)
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--task-mode", choices=["ssl_binary", "dysplasia_binary", "adenoma_11class"], default="ssl_binary")
    parser.add_argument("--task-name", default=None)
    return parser.parse_args()


def build_split(df, ids: list[str], data_dir: str, num_classes: int) -> Generic_Split:
    subset = df[df["slide_id"].isin(ids)].reset_index(drop=True)
    return Generic_Split(subset, data_dir=data_dir, num_classes=num_classes)


def main() -> None:
    args = parse_args()
    task_spec = get_task_spec(args.task_mode)
    task_name = args.task_name or str(task_spec["task_name"])
    n_classes = int(task_spec.get("n_classes", len(task_spec["label_dict"])))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str, "case_id": str, "label": str})
    dataset = Generic_MIL_Dataset(
        csv_path=args.ready_csv,
        data_dir=args.feature_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict=task_spec["label_dict"],
        ignore=[],
        patient_strat=False,
        label_col="label",
    )
    split_ids = load_split_ids(Path(args.split_dir), args.fold)
    split_sizes = {split_name: len(ids) for split_name, ids in split_ids.items()}
    label_counts = {}
    for split_name, ids in split_ids.items():
        subset = ready_df[ready_df["slide_id"].isin(ids)]
        label_counts[split_name] = subset["label_name"].value_counts().to_dict()

    train_split = build_split(dataset.slide_data, split_ids["train"], args.feature_dir, n_classes)
    val_split = build_split(dataset.slide_data, split_ids["val"], args.feature_dir, n_classes)
    test_split = build_split(dataset.slide_data, split_ids["test"], args.feature_dir, n_classes)

    model = TransMIL(embed_dim=args.embed_dim, model_dim=args.model_dim, n_classes=n_classes, dropout=args.drop_out)

    config = TrainConfig(
        results_dir=results_dir,
        max_epochs=args.max_epochs,
        seed=args.seed,
        lr=args.lr,
        reg=args.reg,
        weighted_sample=args.weighted_sample,
        early_stopping=args.early_stopping,
        smoke=args.smoke,
        task_name=task_name,
        task_mode=args.task_mode,
        task_spec=task_spec,
        model_name="transmil",
        n_classes=n_classes,
        fold=args.fold,
        split_sizes=split_sizes,
        label_counts=label_counts,
        ready_df=ready_df,
    )
    (results_dir / "transmil_model_config.json").write_text(
        json.dumps(
            {
                "embed_dim": args.embed_dim,
                "model_dim": args.model_dim,
                "drop_out": args.drop_out,
                "fold": args.fold,
                "task_name": task_name,
                "task_mode": args.task_mode,
                "task_spec": task_spec,
                "n_classes": n_classes,
                "split_sizes": split_sizes,
                "label_counts": label_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    run_transmil_training(model, train_split, val_split, test_split, config)


if __name__ == "__main__":
    main()
