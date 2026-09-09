#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from clam_experiment_utils import (
    bytes_to_gb,
    bytes_to_mb,
    collect_file_size_bytes,
    format_size,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ROC/AUC/Time/Size deliverables for a CLAM fold-0 experiment."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--positive-class-name", default="SSL")
    parser.add_argument("--negative-class-name", default="others")
    return parser.parse_args()


def load_confusion_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    return df.to_numpy(dtype=int)


def compute_mean_epoch_time(epoch_timing_path: Path) -> float:
    epoch_df = pd.read_csv(epoch_timing_path)
    return float(epoch_df["duration_sec"].mean())


def build_core_size_bytes(results_dir: Path, analysis_dir: Path) -> tuple[int, list[str]]:
    top_level_files = [
        "s_0_checkpoint.pt",
        "predictions.csv",
        "metrics.json",
        "summary.csv",
        "confusion_matrix.csv",
        "training_efficiency.json",
        "epoch_timing.csv",
        "experiment_config.json",
        "split_0_results.pkl",
        "splits_0.csv",
    ]
    included = []
    total_bytes = 0
    for filename in top_level_files:
        path = results_dir / filename
        if path.exists():
            included.append(str(path))
            total_bytes += collect_file_size_bytes(path)

    # Count the existing experiment analysis package, but exclude the new ROC deliverables.
    analysis_exclude = {
        "roc_curve_test_fold0.png",
        "roc_time_size_summary.csv",
        "roc_time_size_summary.json",
        "roc_time_size_report.md",
    }
    for child in analysis_dir.iterdir():
        if child.name in analysis_exclude:
            continue
        if child.is_file():
            included.append(str(child))
            total_bytes += collect_file_size_bytes(child)
        elif child.is_dir():
            included.append(str(child))
            total_bytes += collect_file_size_bytes(child)

    return total_bytes, included


def build_size_block(num_bytes: int) -> dict[str, object]:
    return {
        "bytes": int(num_bytes),
        "mb": bytes_to_mb(num_bytes),
        "gb": bytes_to_gb(num_bytes),
        "human_readable": format_size(num_bytes),
    }


def write_summary_csv(path: Path, summary: dict) -> None:
    flat_row = {
        "results_dir": summary["results_dir"],
        "analysis_dir": summary["analysis_dir"],
        "positive_class_name": summary["positive_class_name"],
        "negative_class_name": summary["negative_class_name"],
        "n_test": summary["test_set"]["n_test"],
        "n_positive": summary["test_set"]["n_positive"],
        "n_negative": summary["test_set"]["n_negative"],
        "auc_recomputed": summary["roc_auc"]["auc_recomputed"],
        "auc_metrics_json": summary["roc_auc"]["auc_metrics_json"],
        "auc_abs_diff": summary["roc_auc"]["auc_abs_diff"],
        "training_total_sec": summary["time"]["training_total_sec"],
        "mean_epoch_sec": summary["time"]["mean_epoch_sec"],
        "epochs_completed": summary["time"]["epochs_completed"],
        "heatmap_mean_per_slide_sec": summary["time"]["heatmap_mean_per_slide_sec"],
        "core_results_bytes": summary["size"]["core_results"]["bytes"],
        "core_results_mb": summary["size"]["core_results"]["mb"],
        "core_results_gb": summary["size"]["core_results"]["gb"],
        "heatmaps_bytes": summary["size"]["heatmaps"]["bytes"],
        "heatmaps_mb": summary["size"]["heatmaps"]["mb"],
        "heatmaps_gb": summary["size"]["heatmaps"]["gb"],
        "fold0_total_bytes": summary["size"]["fold0_total"]["bytes"],
        "fold0_total_mb": summary["size"]["fold0_total"]["mb"],
        "fold0_total_gb": summary["size"]["fold0_total"]["gb"],
        "total_ge_core": summary["size_checks"]["total_ge_core"],
        "total_ge_heatmaps": summary["size_checks"]["total_ge_heatmaps"],
        "confusion_matches_predictions": summary["validations"]["confusion_matches_predictions"],
        "auc_matches_metrics_json": summary["validations"]["auc_matches_metrics_json"],
        "test_count_matches_metrics": summary["validations"]["test_count_matches_metrics"],
    }
    pd.DataFrame([flat_row]).to_csv(path, index=False)


def write_markdown_report(path: Path, summary: dict) -> None:
    lines = []
    lines.append("# 2.5X CLAM-SB Fold-0 ROC / AUC / Time / Size")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- 结果目录: `{summary['results_dir']}`")
    lines.append(f"- 测试样本数: {summary['test_set']['n_test']}")
    lines.append(f"- 正类 `{summary['positive_class_name']}` 数量: {summary['test_set']['n_positive']}")
    lines.append(f"- 负类 `{summary['negative_class_name']}` 数量: {summary['test_set']['n_negative']}")
    lines.append("")
    lines.append("## ROC / AUC")
    lines.append(f"- 重算 AUC: {summary['roc_auc']['auc_recomputed']:.4f}")
    lines.append(f"- `metrics.json` AUC: {summary['roc_auc']['auc_metrics_json']:.4f}")
    lines.append(f"- 绝对差值: {summary['roc_auc']['auc_abs_diff']:.8f}")
    lines.append(f"- ROC 图: `{summary['roc_auc']['roc_curve_png']}`")
    lines.append("")
    lines.append("## Time")
    lines.append(f"- 训练总时长: {summary['time']['training_total_sec']:.4f} s")
    lines.append(f"- 平均单 epoch 时长: {summary['time']['mean_epoch_sec']:.4f} s")
    lines.append(f"- 已完成 epoch 数: {summary['time']['epochs_completed']}")
    if summary["time"]["heatmap_mean_per_slide_sec"] is not None:
        lines.append(f"- heatmap 平均单 slide 时长: {summary['time']['heatmap_mean_per_slide_sec']:.4f} s")
    else:
        lines.append("- heatmap 平均单 slide 时长: N/A")
    lines.append("")
    lines.append("## Size")
    lines.append(
        f"- 核心结果包: {summary['size']['core_results']['bytes']} bytes "
        f"({summary['size']['core_results']['human_readable']})"
    )
    lines.append(
        f"- heatmaps 输出目录: {summary['size']['heatmaps']['bytes']} bytes "
        f"({summary['size']['heatmaps']['human_readable']})"
    )
    lines.append(
        f"- 整个 `fold-0` 目录: {summary['size']['fold0_total']['bytes']} bytes "
        f"({summary['size']['fold0_total']['human_readable']})"
    )
    lines.append("")
    lines.append("## Validation Checks")
    lines.append(f"- ROC AUC 与 `metrics.json` 一致: {summary['validations']['auc_matches_metrics_json']}")
    lines.append(f"- `predictions.csv` 测试样本数与 `metrics.json` 一致: {summary['validations']['test_count_matches_metrics']}")
    lines.append(f"- `confusion_matrix.csv` 与 `predictions.csv` 一致: {summary['validations']['confusion_matches_predictions']}")
    lines.append(f"- `fold-0` 总大小 >= 核心结果包: {summary['size_checks']['total_ge_core']}")
    lines.append(f"- `fold-0` 总大小 >= heatmaps 输出目录: {summary['size_checks']['total_ge_heatmaps']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- 本统计口径仅针对最新 2.5X CLAM-SB rerun 的 `fold-0` 测试集。")
    lines.append("- `core_results` 包含顶层关键结果文件和既有 `analysis/` 目录，但不把本次新增的 ROC 交付文件本身计入 Size，避免自引用放大。")
    lines.append("- `heatmaps` 目录大小单独统计，用于和核心结果包区分展示。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_json(results_dir / "metrics.json")
    predictions_df = pd.read_csv(results_dir / "predictions.csv")
    confusion_from_file = load_confusion_csv(results_dir / "confusion_matrix.csv")
    training_efficiency = read_json(results_dir / "training_efficiency.json")
    heatmap_summary_path = results_dir / "heatmaps" / "heatmap_summary.json"
    heatmap_summary = read_json(heatmap_summary_path) if heatmap_summary_path.exists() else None

    y_true = predictions_df["label"].to_numpy(dtype=int)
    y_score = predictions_df["prob_ssl"].to_numpy(dtype=float)
    y_pred = predictions_df["pred"].to_numpy(dtype=int)

    auc_recomputed = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    cm_from_predictions = confusion_matrix(y_true, y_pred, labels=[0, 1])

    roc_curve_png = analysis_dir / "roc_curve_test_fold0.png"
    plt.figure(figsize=(6.2, 6.0))
    plt.plot(fpr, tpr, color="#1565c0", linewidth=2.2, label=f"ROC (AUC = {auc_recomputed:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.2, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("2.5X CLAM-SB Fold-0 Test ROC")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(roc_curve_png, dpi=180)
    plt.close()

    mean_epoch_sec = compute_mean_epoch_time(results_dir / "epoch_timing.csv")
    heatmap_mean_per_slide_sec = None
    if heatmap_summary is not None:
        heatmap_mean_per_slide_sec = float(heatmap_summary["mean_duration_sec"])

    core_results_bytes, core_components = build_core_size_bytes(results_dir, analysis_dir)
    heatmaps_bytes = collect_file_size_bytes(results_dir / "heatmaps")
    fold0_total_bytes = collect_file_size_bytes(results_dir)

    summary = {
        "results_dir": str(results_dir),
        "analysis_dir": str(analysis_dir),
        "positive_class_name": args.positive_class_name,
        "negative_class_name": args.negative_class_name,
        "roc_auc": {
            "auc_recomputed": auc_recomputed,
            "auc_metrics_json": float(metrics["auc"]),
            "auc_abs_diff": abs(auc_recomputed - float(metrics["auc"])),
            "roc_curve_png": str(roc_curve_png),
            "roc_points": int(len(fpr)),
        },
        "test_set": {
            "n_test": int(len(predictions_df)),
            "n_positive": int((y_true == 1).sum()),
            "n_negative": int((y_true == 0).sum()),
        },
        "time": {
            "training_total_sec": float(training_efficiency["total_training_time_sec"]),
            "mean_epoch_sec": mean_epoch_sec,
            "epochs_completed": int(training_efficiency["epochs_completed"]),
            "heatmap_mean_per_slide_sec": heatmap_mean_per_slide_sec,
        },
        "size": {
            "core_results": build_size_block(core_results_bytes),
            "heatmaps": build_size_block(heatmaps_bytes),
            "fold0_total": build_size_block(fold0_total_bytes),
        },
        "size_checks": {
            "total_ge_core": bool(fold0_total_bytes >= core_results_bytes),
            "total_ge_heatmaps": bool(fold0_total_bytes >= heatmaps_bytes),
        },
        "validations": {
            "auc_matches_metrics_json": bool(abs(auc_recomputed - float(metrics["auc"])) <= 1e-8),
            "test_count_matches_metrics": bool(len(predictions_df) == int(metrics["n_test"])),
            "confusion_matches_predictions": bool(np.array_equal(cm_from_predictions, confusion_from_file)),
        },
        "components": {
            "core_results_included_paths": core_components,
        },
    }

    summary_json_path = analysis_dir / "roc_time_size_summary.json"
    summary_csv_path = analysis_dir / "roc_time_size_summary.csv"
    report_path = analysis_dir / "roc_time_size_report.md"

    write_json(summary_json_path, summary)
    write_summary_csv(summary_csv_path, summary)
    write_markdown_report(report_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
