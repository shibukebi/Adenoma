#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


SSL_TYPE = "Sessile serrated adenoma"
SSL_NEGATIVE_NAME = "others"
SSL_POSITIVE_NAME = "SSL"
DYSPLASIA_NEGATIVE_NAME = "no_dysplasia"
DYSPLASIA_POSITIVE_NAME = "dysplasia"

ADENOMA_11_CLASS_NAMES = [
    "ssl",
    "hp",
    "TSA",
    "USA",
    "TA",
    "TVA",
    "IP",
    "ssl with highgrade dysplasia",
    "TSA with highgrade dysplasia",
    "TA with highgrade dysplasia",
    "TVA with highgrade dysplasia",
]
ADENOMA_11_LABEL_DICT = {name: index for index, name in enumerate(ADENOMA_11_CLASS_NAMES)}

FINAL_LABEL_DICT = {
    "others": 0,
    "SSL-no-dysplasia": 1,
    "SSL-with-dysplasia": 2,
}
INV_FINAL_LABEL_DICT = {value: key for key, value in FINAL_LABEL_DICT.items()}

TASK_SPECS = {
    "ssl_binary": {
        "task_mode": "ssl_binary",
        "task_name": "ssl_vs_others",
        "label_name_col": "ssl_label_name",
        "label_value_col": "ssl_label",
        "label_dict": {SSL_NEGATIVE_NAME: 0, SSL_POSITIVE_NAME: 1},
        "positive_name": SSL_POSITIVE_NAME,
        "negative_name": SSL_NEGATIVE_NAME,
        "positive_score_col": "prob_ssl",
        "negative_score_col": "prob_others",
        "default_min_positive_per_split": 5,
    },
    "dysplasia_binary": {
        "task_mode": "dysplasia_binary",
        "task_name": "dysplasia_vs_no_dysplasia",
        "label_name_col": "dysplasia_label_name",
        "label_value_col": "dysplasia_label",
        "label_dict": {DYSPLASIA_NEGATIVE_NAME: 0, DYSPLASIA_POSITIVE_NAME: 1},
        "positive_name": DYSPLASIA_POSITIVE_NAME,
        "negative_name": DYSPLASIA_NEGATIVE_NAME,
        "positive_score_col": "prob_dysplasia",
        "negative_score_col": "prob_no_dysplasia",
        "default_min_positive_per_split": 1,
    },
    "adenoma_11class": {
        "task_mode": "adenoma_11class",
        "task_name": "adenoma_11class",
        "label_name_col": "label_name",
        "label_value_col": "label_value",
        "label_dict": ADENOMA_11_LABEL_DICT,
        "class_names": ADENOMA_11_CLASS_NAMES,
        "n_classes": len(ADENOMA_11_CLASS_NAMES),
        "default_min_positive_per_split": 1,
        "subtyping": True,
    },
}

LABEL_DICT = TASK_SPECS["ssl_binary"]["label_dict"]
INV_LABEL_DICT = {value: key for key, value in LABEL_DICT.items()}

HIERARCHICAL_LABEL_COLUMNS = [
    "slide_id",
    "type",
    "grade",
    "label_name",
    "label",
    "ssl_label_name",
    "ssl_label",
    "dysplasia_label_name",
    "dysplasia_label",
    "is_ssl_for_stage2",
    "final_label_name",
    "final_label",
]

TASK_METADATA_COLUMNS = [
    "type",
    "grade",
    "ssl_label_name",
    "ssl_label",
    "dysplasia_label_name",
    "dysplasia_label",
    "is_ssl_for_stage2",
    "final_label_name",
    "final_label",
]


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def prob_column_name(label_name: str) -> str:
    return f"prob_{sanitize_name(label_name)}"


def read_id_list(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row[0].strip() for row in csv.reader(handle) if row and row[0].strip()]


def resolve_fold_index(split_dir: Path, fold: int) -> int:
    requested = int(fold)
    if (split_dir / f"flod-{requested}-train.csv").exists():
        return requested

    available = sorted(
        {
            int(path.stem.split("-")[1])
            for path in split_dir.glob("flod-*-train.csv")
            if len(path.stem.split("-")) >= 3 and path.stem.split("-")[1].isdigit()
        }
    )
    if not available:
        raise FileNotFoundError(f"No split files matching flod-*-train.csv found in {split_dir}")

    if requested == len(available):
        # Backward compatibility: treat fold=N as requesting the last fold in an N-fold CV set indexed from 0.
        return available[-1]
    raise FileNotFoundError(
        f"Requested fold={requested} not found in {split_dir}. Available folds: {available}"
    )


def load_split_ids(split_dir: Path, fold: int) -> dict[str, list[str]]:
    resolved_fold = resolve_fold_index(split_dir, fold)
    return {
        split_name: read_id_list(split_dir / f"flod-{resolved_fold}-{split_name}.csv")
        for split_name in ("train", "val", "test")
    }


def collect_runtime_resource_metrics() -> dict[str, object]:
    payload: dict[str, object] = {
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "device_name": None,
        "max_memory_allocated_bytes": None,
        "max_memory_reserved_bytes": None,
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        payload["torch_cuda_available"] = cuda_available
        payload["torch_cuda_device_count"] = int(torch.cuda.device_count())
        if cuda_available:
            device_index = int(torch.cuda.current_device())
            payload["cuda_device_index"] = device_index
            payload["device_name"] = torch.cuda.get_device_name(device_index)
            payload["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device_index))
            payload["max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device_index))
    except Exception as exc:
        payload["resource_probe_error"] = str(exc)
    return payload


def build_inference_timing_metrics(
    split_name: str,
    wall_time_sec: float,
    n_cases: int,
) -> dict[str, object]:
    n_cases = int(n_cases)
    total_time = float(wall_time_sec)
    return {
        f"{split_name}_total_inference_time_sec": total_time,
        f"{split_name}_num_cases": n_cases,
        f"{split_name}_mean_inference_time_sec_per_case": (total_time / n_cases) if n_cases > 0 else None,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_grade(raw_grade: str) -> str:
    grade = str(raw_grade).strip().lower()
    if grade not in {"high", "low"}:
        raise ValueError(f"Unsupported grade={raw_grade!r}; expected 'high' or 'low'")
    return grade


def build_hierarchical_label_row(slide_id: str, lesion_type: str, grade: str) -> dict[str, object]:
    normalized_grade = normalize_grade(grade)
    is_ssl = str(lesion_type).strip() == SSL_TYPE
    ssl_label = int(is_ssl)
    ssl_label_name = SSL_POSITIVE_NAME if is_ssl else SSL_NEGATIVE_NAME

    if is_ssl:
        dysplasia_label = int(normalized_grade == "high")
        dysplasia_label_name = DYSPLASIA_POSITIVE_NAME if dysplasia_label == 1 else DYSPLASIA_NEGATIVE_NAME
        final_label_name = "SSL-with-dysplasia" if dysplasia_label == 1 else "SSL-no-dysplasia"
    else:
        dysplasia_label = -1
        dysplasia_label_name = ""
        final_label_name = "others"

    return {
        "slide_id": str(slide_id).strip(),
        "type": str(lesion_type).strip(),
        "grade": normalized_grade,
        "label_name": ssl_label_name,
        "label": ssl_label,
        "ssl_label_name": ssl_label_name,
        "ssl_label": ssl_label,
        "dysplasia_label_name": dysplasia_label_name,
        "dysplasia_label": dysplasia_label,
        "is_ssl_for_stage2": int(is_ssl),
        "final_label_name": final_label_name,
        "final_label": FINAL_LABEL_DICT[final_label_name],
    }


def enrich_label_row(row: dict) -> dict[str, object]:
    slide_id = str(row.get("slide_id", "")).strip()
    lesion_type = str(row.get("type", "")).strip()
    grade = normalize_grade(row.get("grade", ""))
    base = build_hierarchical_label_row(slide_id, lesion_type, grade)

    enriched = dict(row)
    for key, value in base.items():
        enriched[key] = value
    return enriched


def get_task_spec(task_mode: str) -> dict[str, object]:
    if task_mode not in TASK_SPECS:
        raise KeyError(f"Unknown task_mode={task_mode!r}")
    spec = dict(TASK_SPECS[task_mode])
    spec["inv_label_dict"] = {value: key for key, value in spec["label_dict"].items()}
    spec["n_classes"] = int(spec.get("n_classes", len(spec["label_dict"])))
    return spec


def get_default_min_positive_per_split(task_mode: str) -> int:
    return int(get_task_spec(task_mode)["default_min_positive_per_split"])


def prepare_task_row(row: dict, task_mode: str) -> dict[str, object] | None:
    spec = get_task_spec(task_mode)
    enriched = enrich_label_row(row)

    if task_mode == "dysplasia_binary" and int(enriched["is_ssl_for_stage2"]) != 1:
        return None

    label_name = str(enriched[spec["label_name_col"]]).strip()
    label_value = int(enriched[spec["label_value_col"]])
    if not label_name:
        return None

    record = {
        "case_id": enriched["slide_id"],
        "slide_id": enriched["slide_id"],
        "label": label_name,
        "label_name": label_name,
        "label_value": label_value,
        "task_mode": task_mode,
    }
    for column in TASK_METADATA_COLUMNS:
        record[column] = enriched[column]
    return record


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_binary_metrics(
    predictions_df: pd.DataFrame,
    positive_label: int = 1,
    positive_name: str = SSL_POSITIVE_NAME,
    negative_name: str = SSL_NEGATIVE_NAME,
    positive_score_col: str | None = None,
    negative_score_col: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    positive_score_col = positive_score_col or prob_column_name(positive_name)
    negative_score_col = negative_score_col or prob_column_name(negative_name)

    y_true = predictions_df["label"].to_numpy(dtype=int)
    y_pred = predictions_df["pred"].to_numpy(dtype=int)
    y_score = predictions_df[positive_score_col].to_numpy(dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    positive_key = sanitize_name(positive_name)
    negative_key = sanitize_name(negative_name)
    metrics = {
        "auc": safe_roc_auc(y_true, y_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1[positive_label]),
        "recall": float(recall[positive_label]),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        f"{positive_key}_precision": float(precision[positive_label]),
        f"{positive_key}_recall": float(recall[positive_label]),
        f"{positive_key}_f1": float(f1[positive_label]),
        f"{positive_key}_support": int(support[positive_label]),
        f"{negative_key}_precision": float(precision[0]),
        f"{negative_key}_recall": float(recall[0]),
        f"{negative_key}_f1": float(f1[0]),
        f"{negative_key}_support": int(support[0]),
        "predicted_positive": int((y_pred == positive_label).sum()),
        "predicted_negative": int((y_pred != positive_label).sum()),
        "positive_score_col": positive_score_col,
        "negative_score_col": negative_score_col,
    }

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{negative_name}", f"true_{positive_name}"],
        columns=[f"pred_{negative_name}", f"pred_{positive_name}"],
    )
    return metrics, cm_df


def compute_multiclass_metrics(
    predictions_df: pd.DataFrame,
    true_col: str = "label",
    pred_col: str = "pred",
    label_dict: dict[str, int] | None = None,
) -> tuple[dict, pd.DataFrame]:
    label_dict = label_dict or FINAL_LABEL_DICT
    ordered_labels = sorted(label_dict.items(), key=lambda item: item[1])
    label_names = [name for name, _ in ordered_labels]
    label_ids = [value for _, value in ordered_labels]

    y_true = predictions_df[true_col].to_numpy(dtype=int)
    y_pred = predictions_df[pred_col].to_numpy(dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_samples": int(len(predictions_df)),
    }
    prob_cols = [prob_column_name(label_name) for label_name in label_names]
    if all(column in predictions_df.columns for column in prob_cols):
        y_score = predictions_df[prob_cols].to_numpy(dtype=float)
        try:
            metrics["auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_score, labels=label_ids, multi_class="ovr", average="macro")
            )
            metrics["auc_ovr_weighted"] = float(
                roc_auc_score(y_true, y_score, labels=label_ids, multi_class="ovr", average="weighted")
            )
        except ValueError:
            metrics["auc_ovr_macro"] = float("nan")
            metrics["auc_ovr_weighted"] = float("nan")

    for index, label_name in enumerate(label_names):
        key = sanitize_name(label_name)
        metrics[f"{key}_precision"] = float(precision[index])
        metrics[f"{key}_recall"] = float(recall[index])
        metrics[f"{key}_f1"] = float(f1[index])
        metrics[f"{key}_support"] = int(support[index])
        score_col = prob_column_name(label_name)
        if score_col in predictions_df.columns:
            binary_true = (y_true == label_ids[index]).astype(int)
            if len(np.unique(binary_true)) == 2:
                metrics[f"{key}_auc"] = float(roc_auc_score(binary_true, predictions_df[score_col].to_numpy(dtype=float)))
            else:
                metrics[f"{key}_auc"] = float("nan")

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in label_names],
        columns=[f"pred_{name}" for name in label_names],
    )
    return metrics, cm_df


def add_split_column(predictions_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    enriched = predictions_df.copy()
    enriched["split"] = split_name
    return enriched


def add_prediction_annotations(
    predictions_df: pd.DataFrame,
    positive_name: str = SSL_POSITIVE_NAME,
    negative_name: str = SSL_NEGATIVE_NAME,
    positive_score_col: str | None = None,
    negative_score_col: str | None = None,
) -> pd.DataFrame:
    positive_score_col = positive_score_col or prob_column_name(positive_name)
    negative_score_col = negative_score_col or prob_column_name(negative_name)

    df = predictions_df.copy()
    df["is_correct"] = df["label"] == df["pred"]
    df["confidence"] = np.where(df["pred"] == 1, df[positive_score_col], df[negative_score_col])

    outcome_groups = []
    for _, row in df.iterrows():
        if row["label_name"] == positive_name and row["pred_name"] == positive_name:
            outcome_groups.append(f"{positive_name}_tp")
        elif row["label_name"] == positive_name and row["pred_name"] != positive_name:
            outcome_groups.append(f"{positive_name}_fn")
        elif row["label_name"] == negative_name and row["pred_name"] == positive_name:
            outcome_groups.append(f"{negative_name}_fp")
        else:
            outcome_groups.append(f"{negative_name}_tn")
    df["outcome_group"] = outcome_groups
    return df


def attach_task_metadata(predictions_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [column for column in ["slide_id", *TASK_METADATA_COLUMNS] if column in source_df.columns]
    if len(metadata_columns) <= 1:
        return predictions_df
    metadata_df = source_df[metadata_columns].drop_duplicates(subset=["slide_id"])
    return predictions_df.merge(metadata_df, on="slide_id", how="left")


def build_prediction_frame(
    slide_ids: list[str],
    probs: np.ndarray,
    labels: np.ndarray,
    fold: int,
    split_name: str,
    task_spec: dict[str, object],
) -> pd.DataFrame:
    n_classes = int(task_spec.get("n_classes", len(task_spec["label_dict"])))
    if n_classes > 2:
        inv_label_dict = dict(task_spec["inv_label_dict"])
        ordered_labels = sorted(dict(task_spec["label_dict"]).items(), key=lambda item: item[1])
        rows = []
        for index, slide_id in enumerate(slide_ids):
            prob = np.asarray(probs[index]).reshape(-1)
            label = int(labels[index])
            pred = int(np.argmax(prob))
            row = {
                "slide_id": str(slide_id),
                "fold": fold,
                "label": label,
                "label_name": inv_label_dict.get(label, ""),
                "pred": pred,
                "pred_name": inv_label_dict.get(pred, ""),
                "is_correct": label == pred,
                "confidence": float(prob[pred]),
            }
            for label_name, label_value in ordered_labels:
                row[prob_column_name(label_name)] = float(prob[label_value])
            rows.append(row)
        return add_split_column(pd.DataFrame(rows), split_name)

    positive_name = str(task_spec["positive_name"])
    negative_name = str(task_spec["negative_name"])
    positive_score_col = str(task_spec["positive_score_col"])
    negative_score_col = str(task_spec["negative_score_col"])
    inv_label_dict = dict(task_spec["inv_label_dict"])

    rows = []
    for index, slide_id in enumerate(slide_ids):
        prob = np.asarray(probs[index]).reshape(-1)
        label = int(labels[index])
        pred = int(np.argmax(prob))
        rows.append(
            {
                "slide_id": str(slide_id),
                "fold": fold,
                "label": label,
                "label_name": inv_label_dict.get(label, ""),
                "pred": pred,
                "pred_name": inv_label_dict.get(pred, ""),
                negative_score_col: float(prob[0]),
                positive_score_col: float(prob[1]),
            }
        )
    predictions_df = pd.DataFrame(rows)
    predictions_df = add_split_column(predictions_df, split_name)
    predictions_df = add_prediction_annotations(
        predictions_df,
        positive_name=positive_name,
        negative_name=negative_name,
        positive_score_col=positive_score_col,
        negative_score_col=negative_score_col,
    )
    return predictions_df


def build_prediction_frame_from_results(
    results_dict: dict[str, dict],
    fold: int,
    split_name: str,
    task_spec: dict[str, object],
) -> pd.DataFrame:
    slide_ids = []
    probs = []
    labels = []
    for slide_id, payload in sorted(results_dict.items()):
        slide_ids.append(str(slide_id))
        probs.append(np.asarray(payload["prob"]).reshape(-1))
        labels.append(int(payload["label"]))
    return build_prediction_frame(
        slide_ids=slide_ids,
        probs=np.asarray(probs, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        fold=fold,
        split_name=split_name,
        task_spec=task_spec,
    )


def summarize_numeric(values: pd.Series | list[float] | np.ndarray) -> dict[str, float | int]:
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }

    return {
        "count": int(series.size),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
    }


def count_patch_coords(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as handle:
        if "coords" not in handle:
            raise KeyError(f"'coords' dataset not found in {h5_path}")
        return int(handle["coords"].shape[0])


def build_patch_count_table(
    ready_df: pd.DataFrame,
    split_ids: dict[str, list[str]],
    patch_dir: Path,
) -> pd.DataFrame:
    split_lookup = {
        slide_id: split_name
        for split_name, ids in split_ids.items()
        for slide_id in ids
    }
    rows = []
    for row in ready_df.itertuples(index=False):
        slide_id = str(row.slide_id)
        h5_path = patch_dir / f"{slide_id}.h5"
        if not h5_path.exists():
            continue
        rows.append(
            {
                "slide_id": slide_id,
                "label_name": str(row.label_name),
                "split": split_lookup.get(slide_id, "unspecified"),
                "patch_count": count_patch_coords(h5_path),
            }
        )
    return pd.DataFrame(rows)


def collect_file_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit_idx = -1
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def bytes_to_mb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0 * 1024.0)


def build_dynamic_count_metrics(
    label_counts: dict[str, dict[str, int]],
    split_sizes: dict[str, int],
    task_spec: dict[str, object],
) -> dict[str, int]:
    if "positive_name" not in task_spec or "negative_name" not in task_spec:
        metrics = {}
        for split_name, size in split_sizes.items():
            metrics[f"n_{split_name}"] = int(size)
            for label_name in task_spec.get("class_names", task_spec.get("label_dict", {}).keys()):
                key = sanitize_name(str(label_name))
                metrics[f"{key}_{split_name}"] = int(label_counts.get(split_name, {}).get(label_name, 0))
        return metrics

    positive_key = sanitize_name(str(task_spec["positive_name"]))
    negative_key = sanitize_name(str(task_spec["negative_name"]))

    metrics = {}
    for split_name, size in split_sizes.items():
        metrics[f"n_{split_name}"] = int(size)
        metrics[f"{positive_key}_{split_name}"] = int(label_counts.get(split_name, {}).get(task_spec["positive_name"], 0))
        metrics[f"{negative_key}_{split_name}"] = int(label_counts.get(split_name, {}).get(task_spec["negative_name"], 0))
    return metrics


def select_routing_threshold(
    predictions_df: pd.DataFrame,
    positive_score_col: str = "prob_ssl",
    positive_label: int = 1,
) -> dict[str, float]:
    if predictions_df.empty or positive_score_col not in predictions_df.columns:
        return {
            "threshold": 0.5,
            "f2": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "n_thresholds_scanned": 0,
        }

    y_true = predictions_df["label"].to_numpy(dtype=int)
    y_score = predictions_df[positive_score_col].to_numpy(dtype=float)
    thresholds = sorted(set(float(value) for value in y_score))
    thresholds = sorted(set([0.0, 0.5, 1.0, *thresholds]))

    best = None
    beta_sq = 4.0
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = int(((y_true == positive_label) & (y_pred == positive_label)).sum())
        fp = int(((y_true != positive_label) & (y_pred == positive_label)).sum())
        fn = int(((y_true == positive_label) & (y_pred != positive_label)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision == 0.0 and recall == 0.0:
            f2 = 0.0
        else:
            f2 = (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)

        candidate = {
            "threshold": float(threshold),
            "f2": float(f2),
            "precision": float(precision),
            "recall": float(recall),
        }
        if best is None:
            best = candidate
            continue

        if candidate["f2"] > best["f2"]:
            best = candidate
        elif math.isclose(candidate["f2"], best["f2"], rel_tol=0.0, abs_tol=1e-12):
            if candidate["recall"] > best["recall"]:
                best = candidate
            elif math.isclose(candidate["recall"], best["recall"], rel_tol=0.0, abs_tol=1e-12):
                if candidate["threshold"] < best["threshold"]:
                    best = candidate

    best = best or {
        "threshold": 0.5,
        "f2": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
    }
    best["n_thresholds_scanned"] = len(thresholds)
    return best


def merge_hierarchical_predictions(
    stage1_test_df: pd.DataFrame,
    stage2_full_test_df: pd.DataFrame,
    label_df: pd.DataFrame,
    routing_threshold: float,
) -> pd.DataFrame:
    stage1 = stage1_test_df.copy()
    stage2 = stage2_full_test_df.copy()
    labels = label_df.copy()

    labels["slide_id"] = labels["slide_id"].astype(str)
    stage1["slide_id"] = stage1["slide_id"].astype(str)
    stage2["slide_id"] = stage2["slide_id"].astype(str)

    label_columns = [
        "slide_id",
        "final_label_name",
        "final_label",
        "ssl_label_name",
        "ssl_label",
        "dysplasia_label_name",
        "dysplasia_label",
    ]
    labels = labels[label_columns].drop_duplicates(subset=["slide_id"])
    duplicate_label_columns = [column for column in label_columns if column != "slide_id"]
    stage1 = stage1.drop(columns=[column for column in duplicate_label_columns if column in stage1.columns], errors="ignore")
    stage2 = stage2.drop(columns=[column for column in duplicate_label_columns if column in stage2.columns], errors="ignore")

    stage1 = stage1.rename(
        columns={
            "label": "stage1_label",
            "label_name": "stage1_label_name",
            "pred": "stage1_pred",
            "pred_name": "stage1_pred_name",
            "prob_ssl": "stage1_prob_ssl",
            "prob_others": "stage1_prob_others",
        }
    )
    stage2 = stage2.rename(
        columns={
            "label": "stage2_label",
            "label_name": "stage2_label_name",
            "pred": "stage2_pred",
            "pred_name": "stage2_pred_name",
            "prob_dysplasia": "stage2_prob_dysplasia",
            "prob_no_dysplasia": "stage2_prob_no_dysplasia",
        }
    )

    merged = stage1.merge(labels, on="slide_id", how="left")
    merged = merged.merge(
        stage2[
            [
                "slide_id",
                "stage2_label",
                "stage2_label_name",
                "stage2_pred",
                "stage2_pred_name",
                "stage2_prob_dysplasia",
                "stage2_prob_no_dysplasia",
            ]
        ],
        on="slide_id",
        how="left",
    )

    merged["routing_threshold"] = float(routing_threshold)
    merged["routed_to_stage2"] = merged["stage1_prob_ssl"] >= float(routing_threshold)

    merged["final_pred_name"] = "others"
    routed_mask = merged["routed_to_stage2"].astype(bool)
    merged.loc[routed_mask & (merged["stage2_pred"] == 1), "final_pred_name"] = "SSL-with-dysplasia"
    merged.loc[routed_mask & (merged["stage2_pred"] == 0), "final_pred_name"] = "SSL-no-dysplasia"
    merged["final_pred"] = merged["final_pred_name"].map(FINAL_LABEL_DICT).astype(int)

    merged = merged.rename(
        columns={
            "final_label_name": "true_final_label_name",
            "final_label": "true_final_label",
        }
    )
    return merged
