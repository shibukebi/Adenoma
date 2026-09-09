from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.utils import collate_MIL, calculate_error  # noqa: E402
from clam_experiment_utils import (  # noqa: E402
    attach_task_metadata,
    build_inference_timing_metrics,
    build_dynamic_count_metrics,
    build_prediction_frame,
    collect_runtime_resource_metrics,
    compute_binary_metrics,
    compute_multiclass_metrics,
    read_json,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    results_dir: Path
    max_epochs: int
    seed: int
    lr: float
    reg: float
    weighted_sample: bool
    early_stopping: bool
    smoke: bool
    task_name: str
    task_mode: str
    task_spec: dict
    model_name: str
    n_classes: int
    fold: int
    split_sizes: dict[str, int]
    label_counts: dict[str, dict[str, int]]
    ready_df: pd.DataFrame


class AccuracyLogger:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.data = [{"count": 0, "correct": 0} for _ in range(n_classes)]

    def log(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        y_hat_i = int(y_hat)
        y_i = int(y)
        self.data[y_i]["count"] += 1
        self.data[y_i]["correct"] += int(y_hat_i == y_i)

    def get_summary(self, idx: int):
        count = self.data[idx]["count"]
        correct = self.data[idx]["correct"]
        acc = None if count == 0 else float(correct) / float(count)
        return acc, correct, count


class EarlyStopper:
    def __init__(self, patience: int = 20, stop_epoch: int = 20):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch: int, val_loss: float, model: nn.Module, ckpt_path: Path):
        score = -val_loss
        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), ckpt_path)
            return
        self.counter += 1
        if self.counter >= self.patience and epoch >= self.stop_epoch:
            self.early_stop = True


def build_weights(dataset) -> torch.DoubleTensor:
    labels = dataset.slide_data["label"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=dataset.num_classes)
    total = float(len(labels))
    weights = [total / counts[label] for label in labels]
    return torch.DoubleTensor(weights)


def build_loader(dataset, training: bool, weighted: bool) -> DataLoader:
    kwargs = {"num_workers": 4} if device.type == "cuda" else {}
    if training:
        if weighted:
            sampler = WeightedRandomSampler(build_weights(dataset), len(dataset))
        else:
            sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate_MIL, **kwargs)


def summarize_split(name: str, loader: DataLoader, model: nn.Module, n_classes: int, loss_fn: nn.Module):
    model.eval()
    acc_logger = AccuracyLogger(n_classes)
    total_loss = 0.0
    total_error = 0.0
    probs = np.zeros((len(loader), n_classes), dtype=np.float32)
    labels = np.zeros(len(loader), dtype=np.int64)
    slide_ids = loader.dataset.slide_data["slide_id"].tolist()

    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)
            logits, y_prob, y_hat, _, _ = model(data)
            loss = loss_fn(logits, label)
            total_loss += float(loss.item())
            total_error += calculate_error(y_hat, label)
            probs[batch_idx] = y_prob.cpu().numpy()
            labels[batch_idx] = int(label.item())
            acc_logger.log(y_hat, label)

    total_loss /= max(1, len(loader))
    total_error /= max(1, len(loader))
    return {
        "split": name,
        "loss": total_loss,
        "error": total_error,
        "probs": probs,
        "labels": labels,
        "slide_ids": slide_ids,
        "acc_logger": acc_logger,
    }


def run_epoch(epoch: int, loader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, n_classes: int):
    model.train()
    total_loss = 0.0
    total_error = 0.0
    acc_logger = AccuracyLogger(n_classes)

    for batch_idx, (data, label) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)
        logits, _, y_hat, _, _ = model(data)
        loss = loss_fn(logits, label)
        total_loss += float(loss.item())
        total_error += calculate_error(y_hat, label)
        acc_logger.log(y_hat, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 20 == 0:
            print(f"batch {batch_idx}, loss: {loss.item():.4f}, label: {label.item()}, bag_size: {data.size(0)}")

    total_loss /= max(1, len(loader))
    total_error /= max(1, len(loader))
    print(f"Epoch: {epoch}, train_loss: {total_loss:.4f}, train_error: {total_error:.4f}")
    for idx in range(n_classes):
        acc, correct, count = acc_logger.get_summary(idx)
        print(f"class {idx}: acc {acc}, correct {correct}/{count}")
    return total_loss, total_error


def write_epoch_timing(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "duration_sec", "stopped_after_epoch"])
        writer.writeheader()
        writer.writerows(rows)


def run_transmil_training(
    model: nn.Module,
    train_dataset,
    val_dataset,
    test_dataset,
    config: TrainConfig,
) -> dict:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.reg)
    train_loader = build_loader(train_dataset, training=True, weighted=config.weighted_sample)
    val_loader = build_loader(val_dataset, training=False, weighted=False)
    test_loader = build_loader(test_dataset, training=False, weighted=False)

    early_stopper = EarlyStopper() if config.early_stopping else None
    checkpoint_path = config.results_dir / f"s_{config.fold}_checkpoint.pt"

    train_start = time.perf_counter()
    epoch_rows = []
    early_stop_epoch = None

    for epoch in range(2 if config.smoke else config.max_epochs):
        epoch_start = time.perf_counter()
        run_epoch(epoch, train_loader, model, optimizer, loss_fn, n_classes=config.n_classes)
        val_summary = summarize_split("val", val_loader, model, config.n_classes, loss_fn)
        print(f"Val Set, val_loss: {val_summary['loss']:.4f}, val_error: {val_summary['error']:.4f}")
        stop = False
        if early_stopper:
            early_stopper(epoch, val_summary["loss"], model, checkpoint_path)
            stop = early_stopper.early_stop
        epoch_rows.append(
            {
                "epoch": epoch,
                "duration_sec": time.perf_counter() - epoch_start,
                "stopped_after_epoch": bool(stop),
            }
        )
        if stop:
            early_stop_epoch = epoch
            break

    total_training_time = time.perf_counter() - train_start
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        torch.save(model.state_dict(), checkpoint_path)

    val_eval_start = time.perf_counter()
    val_summary = summarize_split("val", val_loader, model, config.n_classes, loss_fn)
    val_eval_time = time.perf_counter() - val_eval_start
    test_eval_start = time.perf_counter()
    test_summary = summarize_split("test", test_loader, model, config.n_classes, loss_fn)
    test_eval_time = time.perf_counter() - test_eval_start
    val_predictions = build_prediction_frame(
        slide_ids=val_summary["slide_ids"],
        probs=val_summary["probs"],
        labels=val_summary["labels"],
        fold=config.fold,
        split_name="val",
        task_spec=config.task_spec,
    )
    val_predictions = attach_task_metadata(val_predictions, config.ready_df)
    predictions = build_prediction_frame(
        slide_ids=test_summary["slide_ids"],
        probs=test_summary["probs"],
        labels=test_summary["labels"],
        fold=config.fold,
        split_name="test",
        task_spec=config.task_spec,
    )
    predictions = attach_task_metadata(predictions, config.ready_df)
    if config.n_classes > 2:
        metrics, cm_df = compute_multiclass_metrics(
            predictions,
            label_dict=config.task_spec["label_dict"],
        )
    else:
        metrics, cm_df = compute_binary_metrics(
            predictions,
            positive_label=config.task_spec["label_dict"][config.task_spec["positive_name"]],
            positive_name=config.task_spec["positive_name"],
            negative_name=config.task_spec["negative_name"],
            positive_score_col=config.task_spec["positive_score_col"],
            negative_score_col=config.task_spec["negative_score_col"],
        )
    metrics.update(
        {
            "model_type": config.model_name,
            "task_name": config.task_name,
            "task_mode": config.task_mode,
            "fold": config.fold,
            **build_dynamic_count_metrics(config.label_counts, config.split_sizes, config.task_spec),
        }
    )
    metrics.update(build_inference_timing_metrics("val", val_eval_time, len(val_predictions)))
    metrics.update(build_inference_timing_metrics("test", test_eval_time, len(predictions)))
    metrics.update(collect_runtime_resource_metrics())

    (config.results_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "task_name": config.task_name,
                "task_mode": config.task_mode,
                "task_spec": config.task_spec,
                "n_classes": config.n_classes,
                "model_type": config.model_name,
                "fold": config.fold,
                "results_dir": str(config.results_dir),
                "max_epochs": 2 if config.smoke else config.max_epochs,
                "lr": config.lr,
                "reg": config.reg,
                "weighted_sample": config.weighted_sample,
                "early_stopping": config.early_stopping,
                "split_sizes": config.split_sizes,
                "label_counts": config.label_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    val_predictions.to_csv(config.results_dir / "val_predictions.csv", index=False)
    predictions.to_csv(config.results_dir / "predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": config.fold,
                "model_type": config.model_name,
                "task_name": config.task_name,
                "test_auc": metrics.get("auc", metrics.get("auc_ovr_macro")),
                "test_acc": metrics["accuracy"],
            }
        ]
    ).to_csv(config.results_dir / "summary.csv", index=False)
    (config.results_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    cm_df.to_csv(config.results_dir / "confusion_matrix.csv")
    write_epoch_timing(config.results_dir / "epoch_timing.csv", epoch_rows)
    (config.results_dir / "training_efficiency.json").write_text(
        json.dumps(
            {
                "model_type": config.model_name,
                "total_training_time_sec": total_training_time,
                "test_total_inference_time_sec": test_eval_time,
                "test_num_cases": int(len(predictions)),
                "test_mean_inference_time_sec_per_case": (test_eval_time / len(predictions)) if len(predictions) > 0 else None,
                "max_epochs_requested": 2 if config.smoke else config.max_epochs,
                "epochs_completed": len(epoch_rows),
                "final_epoch": epoch_rows[-1]["epoch"] if epoch_rows else None,
                "early_stopping_enabled": bool(config.early_stopping),
                "early_stop_epoch": early_stop_epoch,
                **collect_runtime_resource_metrics(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"results_dir": str(config.results_dir), "metrics": metrics}, ensure_ascii=False, indent=2))
    return metrics
