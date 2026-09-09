#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

if not hasattr(np, "Inf"):
    np.Inf = np.inf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAM_ROOT = PROJECT_ROOT / "CLAM"
if not CLAM_ROOT.exists():
    CLAM_ROOT = PROJECT_ROOT.parent / "CLAM"
sys.path.insert(0, str(CLAM_ROOT))

from dataset_modules.dataset_generic import Generic_MIL_Dataset, Generic_Split  # noqa: E402
from utils.core_utils import summary as summarize_split  # noqa: E402
from utils.core_utils import train  # noqa: E402
from utils.eval_utils import initiate_model  # noqa: E402
from utils.file_utils import save_pkl  # noqa: E402
from utils.utils import get_simple_loader  # noqa: E402

from clam_experiment_utils import (  # noqa: E402
    attach_task_metadata,
    build_dynamic_count_metrics,
    build_inference_timing_metrics,
    build_prediction_frame_from_results,
    collect_runtime_resource_metrics,
    compute_binary_metrics,
    compute_multiclass_metrics,
    get_task_spec,
    load_split_ids,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CLAM/ABMIL experiment for either SSL-vs-others or dysplasia-vs-no_dysplasia.")
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--drop-out", type=float, default=0.25)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--weighted-sample", action="store_true", default=False)
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--allow-small-splits", action="store_true", default=False)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--task-mode", choices=["ssl_binary", "dysplasia_binary", "adenoma_11class"], default="ssl_binary")
    parser.add_argument("--model-type", choices=["mil", "abmil", "clam_sb", "clam_mb"], default="mil")
    parser.add_argument("--bag-loss", default="ce")
    parser.add_argument("--model-size", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--experiment-code", default=None)
    parser.add_argument("--k-sample", type=int, default=8)
    parser.add_argument("--inst-loss", default="ce")
    parser.add_argument("--bag-weight", type=float, default=0.7)
    parser.add_argument("--no-inst-cluster", action="store_true", default=False)
    return parser.parse_args()


def build_split(df: pd.DataFrame, ids: list[str], data_dir: str, num_classes: int) -> Generic_Split:
    subset = df[df["slide_id"].isin(ids)].reset_index(drop=True)
    return Generic_Split(subset, data_dir=data_dir, num_classes=num_classes)


def evaluate_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    task_spec: dict[str, object],
    val_split: Generic_Split,
    test_split: Generic_Split,
    ready_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_classes = int(task_spec.get("n_classes", len(task_spec["label_dict"])))
    eval_args = SimpleNamespace(
        drop_out=args.drop_out,
        n_classes=n_classes,
        embed_dim=args.embed_dim,
        model_size=args.model_size,
        model_type=args.model_type,
    )
    model = initiate_model(eval_args, str(checkpoint_path))

    val_loader = get_simple_loader(val_split)
    import time as _time
    val_t0 = _time.perf_counter()
    val_results, _, _, _ = summarize_split(model, val_loader, n_classes)
    val_elapsed = _time.perf_counter() - val_t0
    val_predictions = build_prediction_frame_from_results(val_results, args.fold, "val", task_spec)
    val_predictions = attach_task_metadata(val_predictions, ready_df)

    test_loader = get_simple_loader(test_split)
    test_t0 = _time.perf_counter()
    test_results, _, _, _ = summarize_split(model, test_loader, n_classes)
    test_elapsed = _time.perf_counter() - test_t0
    test_predictions = build_prediction_frame_from_results(test_results, args.fold, "test", task_spec)
    test_predictions = attach_task_metadata(test_predictions, ready_df)
    val_predictions.attrs["inference_timing"] = build_inference_timing_metrics("val", val_elapsed, len(val_predictions))
    test_predictions.attrs["inference_timing"] = build_inference_timing_metrics("test", test_elapsed, len(test_predictions))
    return val_predictions, test_predictions


def main() -> None:
    args = parse_args()
    task_spec = get_task_spec(args.task_mode)
    task_name = args.task_name or str(task_spec["task_name"])
    n_classes = int(task_spec.get("n_classes", len(task_spec["label_dict"])))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    stats_path = Path(args.split_dir) / f"flod-{args.fold}_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if not stats.get("formal_ready", False) and not args.allow_small_splits:
            raise RuntimeError(
                f"Split is not marked formal_ready in {stats_path}. "
                "Pass --allow-small-splits for smoke/debug runs."
            )

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

    model_size = args.model_size
    if args.model_type in {"abmil", "clam_sb", "clam_mb"} and model_size is None:
        model_size = "small"
        args.model_size = model_size

    experiment_code = args.experiment_code or f"{task_name}_{args.model_type}_fold{args.fold}"
    no_inst_cluster = args.no_inst_cluster or args.model_type in {"mil", "abmil"}

    train_args = SimpleNamespace(
        results_dir=str(results_dir),
        log_data=False,
        testing=False,
        early_stopping=args.early_stopping,
        bag_loss=args.bag_loss,
        n_classes=n_classes,
        drop_out=args.drop_out,
        model_size=model_size,
        model_type=args.model_type,
        subtyping=bool(task_spec.get("subtyping", n_classes > 2)),
        B=args.k_sample,
        inst_loss=args.inst_loss,
        no_inst_cluster=no_inst_cluster,
        bag_weight=args.bag_weight,
        max_epochs=2 if args.smoke else args.max_epochs,
        lr=args.lr,
        reg=args.reg,
        weighted_sample=args.weighted_sample,
        opt="adam",
        seed=args.seed,
        embed_dim=args.embed_dim,
        exp_code=experiment_code,
        label_frac=1.0,
    )

    config_snapshot = {
        "dataset_csv": args.dataset_csv,
        "ready_csv": args.ready_csv,
        "split_dir": args.split_dir,
        "feature_dir": args.feature_dir,
        "results_dir": str(results_dir),
        "checkpoint_path": str(results_dir / f"s_{args.fold}_checkpoint.pt"),
        "fold": args.fold,
        "max_epochs": train_args.max_epochs,
        "seed": args.seed,
        "lr": args.lr,
        "reg": args.reg,
        "drop_out": args.drop_out,
        "embed_dim": args.embed_dim,
        "weighted_sample": args.weighted_sample,
        "early_stopping": args.early_stopping,
        "allow_small_splits": args.allow_small_splits,
        "smoke": args.smoke,
        "task_mode": args.task_mode,
        "task_name": task_name,
        "task_spec": task_spec,
        "label_dict": task_spec["label_dict"],
        "split_sizes": split_sizes,
        "label_counts": label_counts,
        "model_type": args.model_type,
        "bag_loss": args.bag_loss,
        "model_size": model_size,
        "k_sample": args.k_sample,
        "inst_loss": args.inst_loss,
        "bag_weight": args.bag_weight,
        "no_inst_cluster": no_inst_cluster,
        "n_classes": n_classes,
    }
    (results_dir / "experiment_config.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results_dict, test_auc, val_auc, test_acc, val_acc = train(
        (train_split, val_split, test_split),
        args.fold,
        train_args,
    )
    save_pkl(results_dir / f"split_{args.fold}_results.pkl", results_dict)

    summary_df = pd.DataFrame(
        [
            {
                "fold": args.fold,
                "model_type": args.model_type,
                "task_name": task_name,
                "task_mode": args.task_mode,
                "test_auc": test_auc,
                "val_auc": val_auc,
                "test_acc": test_acc,
                "val_acc": val_acc,
            }
        ]
    )
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    checkpoint_path = results_dir / f"s_{args.fold}_checkpoint.pt"
    val_predictions_df, test_predictions_df = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        args=args,
        task_spec=task_spec,
        val_split=val_split,
        test_split=test_split,
        ready_df=ready_df,
    )
    val_predictions_df.to_csv(results_dir / "val_predictions.csv", index=False)
    test_predictions_df.to_csv(results_dir / "predictions.csv", index=False)

    if n_classes > 2:
        metrics, cm_df = compute_multiclass_metrics(
            test_predictions_df,
            label_dict=task_spec["label_dict"],
        )
    else:
        metrics, cm_df = compute_binary_metrics(
            test_predictions_df,
            positive_label=task_spec["label_dict"][task_spec["positive_name"]],
            positive_name=task_spec["positive_name"],
            negative_name=task_spec["negative_name"],
            positive_score_col=task_spec["positive_score_col"],
            negative_score_col=task_spec["negative_score_col"],
        )
    metrics.update(
        {
            "model_type": args.model_type,
            "task_name": task_name,
            "task_mode": args.task_mode,
            "fold": args.fold,
            "bag_loss": args.bag_loss,
            **build_dynamic_count_metrics(label_counts, split_sizes, task_spec),
        }
    )
    metrics.update(val_predictions_df.attrs.get("inference_timing", {}))
    metrics.update(test_predictions_df.attrs.get("inference_timing", {}))
    metrics.update(collect_runtime_resource_metrics())

    training_efficiency_path = results_dir / "training_efficiency.json"
    if training_efficiency_path.exists():
        training_efficiency = read_json(training_efficiency_path)
        metrics.update(training_efficiency)

    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cm_df.to_csv(results_dir / "confusion_matrix.csv")

    print(
        json.dumps(
            {"results_dir": str(results_dir), "metrics": metrics},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
