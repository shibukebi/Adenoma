#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAM_ROOT = PROJECT_ROOT / "CLAM"
if not CLAM_ROOT.exists():
    CLAM_ROOT = PROJECT_ROOT.parent / "CLAM"
sys.path.insert(0, str(CLAM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clam_experiment_utils import (  # noqa: E402
    attach_task_metadata,
    build_dynamic_count_metrics,
    build_inference_timing_metrics,
    build_prediction_frame,
    collect_runtime_resource_metrics,
    compute_binary_metrics,
    compute_multiclass_metrics,
    get_task_spec,
    load_split_ids,
)
from dsmil.dataset import build_single_stream_split, collate_single_stream  # noqa: E402
from dsmil.model import SingleStreamDSMIL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-stream DSMIL on adenoma 11-class UNI features.")
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
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--weighted-sample", action="store_true", default=False)
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--task-mode", choices=["ssl_binary", "dysplasia_binary", "adenoma_11class"], default="adenoma_11class")
    parser.add_argument("--task-name", default=None)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_slide_frame(ready_df: pd.DataFrame, label_dict: dict[str, int]) -> pd.DataFrame:
    slide_data = ready_df.copy()
    if "label_value" in slide_data.columns:
        slide_data["label"] = slide_data["label_value"].astype(int)
    else:
        slide_data["label"] = slide_data["label_name"].map(label_dict).astype(int)
    slide_data["slide_id"] = slide_data["slide_id"].astype(str)
    return slide_data


def build_weights(dataset) -> torch.DoubleTensor:
    labels = dataset.slide_data["label"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=dataset.num_classes)
    total = float(len(labels))
    weights = [total / max(1, counts[label]) for label in labels]
    return torch.DoubleTensor(weights)


def build_loader(dataset, training: bool, weighted: bool, device: torch.device) -> DataLoader:
    kwargs = {"num_workers": 4, "pin_memory": True} if device.type == "cuda" else {}
    if training:
        sampler = WeightedRandomSampler(build_weights(dataset), len(dataset)) if weighted else RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate_single_stream, **kwargs)


class AccuracyLogger:
    def __init__(self, n_classes: int) -> None:
        self.data = [{"count": 0, "correct": 0} for _ in range(n_classes)]

    def log(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        y_hat_i = int(y_hat)
        y_i = int(y)
        self.data[y_i]["count"] += 1
        self.data[y_i]["correct"] += int(y_hat_i == y_i)

    def get_summary(self, idx: int):
        count = self.data[idx]["count"]
        correct = self.data[idx]["correct"]
        return (None if count == 0 else float(correct) / float(count)), correct, count


class EarlyStopper:
    def __init__(self, patience: int = 20, stop_epoch: int = 20) -> None:
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch: int, val_loss: float, model: nn.Module, ckpt_path: Path) -> None:
        score = -val_loss
        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), ckpt_path)
            return
        self.counter += 1
        if self.counter >= self.patience and epoch >= self.stop_epoch:
            self.early_stop = True


def forward_logits(model: nn.Module, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    results = model(features)
    logits = 0.5 * (results["bag_logits"] + results["max_instance_logits"])
    return logits, results


def compute_loss(logits: torch.Tensor, results: dict[str, torch.Tensor], label: torch.Tensor, loss_fn: nn.Module):
    bag_loss = loss_fn(results["bag_logits"], label)
    max_loss = loss_fn(results["max_instance_logits"], label)
    total_loss = 0.5 * (bag_loss + max_loss)
    return total_loss, {"bag_loss": float(bag_loss.item()), "max_loss": float(max_loss.item())}


def run_epoch(epoch: int, loader: DataLoader, model: nn.Module, optimizer, loss_fn, n_classes: int, device: torch.device):
    model.train()
    total_loss = 0.0
    total_error = 0.0
    acc_logger = AccuracyLogger(n_classes)
    for batch_idx, (features, label) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits, results = forward_logits(model, features)
        y_prob = torch.softmax(logits, dim=1)
        y_hat = torch.argmax(y_prob, dim=1)
        loss, loss_parts = compute_loss(logits, results, label, loss_fn)
        total_loss += float(loss.item())
        total_error += float(y_hat.item() != label.item())
        acc_logger.log(y_hat, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"epoch {epoch} batch {batch_idx + 1}/{len(loader)}, loss {loss.item():.4f}, "
                f"bag_loss {loss_parts['bag_loss']:.4f}, max_loss {loss_parts['max_loss']:.4f}, "
                f"label {label.item()}, bag_size {features.size(0)}",
                flush=True,
            )

    total_loss /= max(1, len(loader))
    total_error /= max(1, len(loader))
    print(f"Epoch {epoch}, train_loss {total_loss:.4f}, train_error {total_error:.4f}", flush=True)
    for idx in range(n_classes):
        acc, correct, count = acc_logger.get_summary(idx)
        print(f"class {idx}: acc {acc}, correct {correct}/{count}", flush=True)
    return total_loss, total_error


def summarize_split(name: str, loader: DataLoader, model: nn.Module, n_classes: int, loss_fn, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    probs = np.zeros((len(loader), n_classes), dtype=np.float32)
    labels = np.zeros(len(loader), dtype=np.int64)
    slide_ids = loader.dataset.slide_data["slide_id"].tolist()

    with torch.inference_mode():
        for batch_idx, (features, label) in enumerate(loader):
            features = features.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logits, results = forward_logits(model, features)
            y_prob = torch.softmax(logits, dim=1)
            y_hat = torch.argmax(y_prob, dim=1)
            loss, _ = compute_loss(logits, results, label, loss_fn)
            total_loss += float(loss.item())
            total_error += float(y_hat.item() != label.item())
            probs[batch_idx] = y_prob.squeeze(0).detach().cpu().numpy()
            labels[batch_idx] = int(label.item())

    total_loss /= max(1, len(loader))
    total_error /= max(1, len(loader))
    print(f"{name} loss {total_loss:.4f}, {name} error {total_error:.4f}", flush=True)
    return {"split": name, "loss": total_loss, "error": total_error, "probs": probs, "labels": labels, "slide_ids": slide_ids}


def write_epoch_timing(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "duration_sec", "stopped_after_epoch"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed)

    task_spec = get_task_spec(args.task_mode)
    task_name = args.task_name or str(task_spec["task_name"])
    n_classes = int(task_spec.get("n_classes", len(task_spec["label_dict"])))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str, "case_id": str, "label": str})
    slide_data = build_slide_frame(ready_df, task_spec["label_dict"])
    split_ids = load_split_ids(Path(args.split_dir), args.fold)
    split_sizes = {split_name: len(ids) for split_name, ids in split_ids.items()}
    label_counts = {}
    for split_name, ids in split_ids.items():
        subset = ready_df[ready_df["slide_id"].isin(ids)]
        label_counts[split_name] = subset["label_name"].value_counts().to_dict()

    train_split = build_single_stream_split(slide_data, split_ids["train"], args.feature_dir, n_classes)
    val_split = build_single_stream_split(slide_data, split_ids["val"], args.feature_dir, n_classes)
    test_split = build_single_stream_split(slide_data, split_ids["test"], args.feature_dir, n_classes)

    model = SingleStreamDSMIL(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_classes=n_classes,
        attn_dim=args.attn_dim,
        dropout=args.drop_out,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    train_loader = build_loader(train_split, training=True, weighted=args.weighted_sample, device=device)
    val_loader = build_loader(val_split, training=False, weighted=False, device=device)
    test_loader = build_loader(test_split, training=False, weighted=False, device=device)

    checkpoint_path = results_dir / f"s_{args.fold}_checkpoint.pt"
    early_stopper = EarlyStopper() if args.early_stopping else None
    epoch_rows = []
    train_start = time.perf_counter()
    max_epochs = 2 if args.smoke else args.max_epochs
    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()
        run_epoch(epoch, train_loader, model, optimizer, loss_fn, n_classes, device)
        val_summary = summarize_split("val", val_loader, model, n_classes, loss_fn, device)
        stop = False
        if early_stopper:
            early_stopper(epoch, val_summary["loss"], model, checkpoint_path)
            stop = early_stopper.early_stop
        epoch_rows.append(
            {"epoch": epoch, "duration_sec": time.perf_counter() - epoch_start, "stopped_after_epoch": bool(stop)}
        )
        if stop:
            break

    total_training_time = time.perf_counter() - train_start
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        torch.save(model.state_dict(), checkpoint_path)

    val_eval_start = time.perf_counter()
    val_summary = summarize_split("val", val_loader, model, n_classes, loss_fn, device)
    val_eval_time = time.perf_counter() - val_eval_start
    test_eval_start = time.perf_counter()
    test_summary = summarize_split("test", test_loader, model, n_classes, loss_fn, device)
    test_eval_time = time.perf_counter() - test_eval_start

    val_predictions = build_prediction_frame(
        slide_ids=val_summary["slide_ids"],
        probs=val_summary["probs"],
        labels=val_summary["labels"],
        fold=args.fold,
        split_name="val",
        task_spec=task_spec,
    )
    val_predictions = attach_task_metadata(val_predictions, ready_df)
    predictions = build_prediction_frame(
        slide_ids=test_summary["slide_ids"],
        probs=test_summary["probs"],
        labels=test_summary["labels"],
        fold=args.fold,
        split_name="test",
        task_spec=task_spec,
    )
    predictions = attach_task_metadata(predictions, ready_df)

    if n_classes > 2:
        metrics, cm_df = compute_multiclass_metrics(predictions, label_dict=task_spec["label_dict"])
    else:
        metrics, cm_df = compute_binary_metrics(
            predictions,
            positive_label=task_spec["label_dict"][task_spec["positive_name"]],
            positive_name=task_spec["positive_name"],
            negative_name=task_spec["negative_name"],
            positive_score_col=task_spec["positive_score_col"],
            negative_score_col=task_spec["negative_score_col"],
        )
    metrics.update(
        {
            "model_type": "dsmil_single_stream",
            "task_name": task_name,
            "task_mode": args.task_mode,
            "fold": args.fold,
            **build_dynamic_count_metrics(label_counts, split_sizes, task_spec),
        }
    )
    metrics.update(build_inference_timing_metrics("val", val_eval_time, len(val_predictions)))
    metrics.update(build_inference_timing_metrics("test", test_eval_time, len(predictions)))
    metrics.update(collect_runtime_resource_metrics())

    config_snapshot = {
        "ready_csv": args.ready_csv,
        "split_dir": args.split_dir,
        "feature_dir": args.feature_dir,
        "results_dir": str(results_dir),
        "checkpoint_path": str(checkpoint_path),
        "fold": args.fold,
        "max_epochs": max_epochs,
        "seed": args.seed,
        "lr": args.lr,
        "reg": args.reg,
        "drop_out": args.drop_out,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "attn_dim": args.attn_dim,
        "weighted_sample": args.weighted_sample,
        "early_stopping": args.early_stopping,
        "smoke": args.smoke,
        "task_mode": args.task_mode,
        "task_name": task_name,
        "task_spec": task_spec,
        "n_classes": n_classes,
        "split_sizes": split_sizes,
        "label_counts": label_counts,
        "model_type": "dsmil_single_stream",
    }
    (results_dir / "experiment_config.json").write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "dsmil_model_config.json").write_text(
        json.dumps(
            {
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "attn_dim": args.attn_dim,
                "drop_out": args.drop_out,
                "n_classes": n_classes,
                "fold": args.fold,
                "task_name": task_name,
                "task_mode": args.task_mode,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    val_predictions.to_csv(results_dir / "val_predictions.csv", index=False)
    predictions.to_csv(results_dir / "predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": args.fold,
                "model_type": "dsmil_single_stream",
                "task_name": task_name,
                "test_auc": metrics.get("auc", metrics.get("auc_ovr_macro")),
                "test_acc": metrics["accuracy"],
            }
        ]
    ).to_csv(results_dir / "summary.csv", index=False)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    cm_df.to_csv(results_dir / "confusion_matrix.csv")
    write_epoch_timing(results_dir / "epoch_timing.csv", epoch_rows)
    (results_dir / "training_efficiency.json").write_text(
        json.dumps(
            {
                "total_training_time_sec": total_training_time,
                "epochs_completed": len(epoch_rows),
                "avg_epoch_time_sec": float(np.mean([row["duration_sec"] for row in epoch_rows])) if epoch_rows else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"results_dir": str(results_dir), "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
