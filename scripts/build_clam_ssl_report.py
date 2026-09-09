#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clam_experiment_utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a markdown report for CLAM SSL experiments.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--selection-csv", default=None)
    parser.add_argument("--heatmap-dir", default=None)
    parser.add_argument("--magnification", default=None)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def metric_line(name: str, value: object) -> str:
    if isinstance(value, float):
        return f"- {name}: {value:.4f}"
    return f"- {name}: {value}"


def load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else results_dir / "analysis"
    heatmap_dir = Path(args.heatmap_dir) if args.heatmap_dir else None
    output_path = Path(args.output_path) if args.output_path else analysis_dir / "clam_ssl_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = read_json(results_dir / "experiment_config.json")
    metrics = read_json(results_dir / "metrics.json")
    magnification = args.magnification
    if not magnification:
        task_name = str(config.get("task_name", ""))
        if "20x" in task_name.lower():
            magnification = "20X"
        elif "5x" in task_name.lower():
            magnification = "5X"
        elif "2p5x" in task_name.lower() or "2.5x" in task_name.lower():
            magnification = "2.5X"
        else:
            magnification = "Unknown"

    performance_df = load_optional_csv(analysis_dir / "performance_summary.csv")
    efficiency_df = load_optional_csv(analysis_dir / "efficiency_summary.csv")
    patch_summary_df = load_optional_csv(analysis_dir / "patch_count_summary.csv")
    selection_df = load_optional_csv(Path(args.selection_csv)) if args.selection_csv else None
    heatmap_summary = read_json(heatmap_dir / "heatmap_summary.json") if heatmap_dir and (heatmap_dir / "heatmap_summary.json").exists() else None

    lines = []
    lines.append(f"# {magnification} CLAM_SB Attention/Heatmap Report")
    lines.append("")
    lines.append("## 实验设置")
    lines.append(metric_line("task_name", config.get("task_name")))
    lines.append(metric_line("model_type", config.get("model_type")))
    lines.append(metric_line("bag_loss", config.get("bag_loss")))
    lines.append(metric_line("embed_dim", config.get("embed_dim")))
    lines.append(metric_line("drop_out", config.get("drop_out")))
    lines.append(metric_line("lr", config.get("lr")))
    lines.append(metric_line("reg", config.get("reg")))
    lines.append(metric_line("seed", config.get("seed")))
    lines.append(metric_line("max_epochs", config.get("max_epochs")))
    lines.append(metric_line("n_train", metrics.get("n_train")))
    lines.append(metric_line("n_val", metrics.get("n_val")))
    lines.append(metric_line("n_test", metrics.get("n_test")))
    lines.append("")

    lines.append("## 效率统计")
    if efficiency_df is not None and not efficiency_df.empty:
        major_stages = efficiency_df[~efficiency_df["stage"].str.startswith("training_epoch_")].copy()
        for row in major_stages.itertuples(index=False):
            lines.append(metric_line(row.stage, row.duration_sec))
    else:
        lines.append("- 暂无 efficiency_summary.csv，请先运行 summarize_clam_ssl_experiment.py。")

    if patch_summary_df is not None and not patch_summary_df.empty:
        lines.append("")
        lines.append("Patch count summary:")
        for row in patch_summary_df[patch_summary_df["group_type"] == "split"].itertuples(index=False):
            lines.append(
                f"- {row.group_name}: count={int(row.count)}, mean={row.mean:.2f}, "
                f"median={row.median:.2f}, p90={row.p90:.2f}, p95={row.p95:.2f}"
            )
    lines.append("")

    lines.append("## 性能统计")
    for key in [
        "auc",
        "accuracy",
        "f1",
        "recall",
        "specificity",
        "macro_f1",
    ]:
        lines.append(metric_line(key, metrics.get(key)))

    if performance_df is not None and len(performance_df) > 1:
        lines.append("")
        lines.append("Baseline comparison:")
        for row in performance_df.itertuples(index=False):
            lines.append(
                f"- {row.experiment}: auc={getattr(row, 'auc', float('nan')):.4f}, "
                f"accuracy={getattr(row, 'accuracy', float('nan')):.4f}, "
                f"f1={getattr(row, 'f1', float('nan')):.4f}"
            )
    lines.append("")

    lines.append("## SSL 专项分析")
    for key in [
        "ssl_precision",
        "ssl_recall",
        "ssl_f1",
        "others_precision",
        "others_recall",
        "others_f1",
        "tp",
        "fn",
        "fp",
        "tn",
    ]:
        lines.append(metric_line(key, metrics.get(key)))

    if selection_df is not None and not selection_df.empty:
        lines.append("")
        lines.append("Representative sample counts:")
        counts = selection_df["selection_category"].value_counts().to_dict()
        for key, value in counts.items():
            lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append(f"## {magnification} Attention Heatmap 质性结果")
    if heatmap_summary is not None:
        lines.append(metric_line("slides_requested", heatmap_summary.get("slides_requested")))
        lines.append(metric_line("slides_succeeded", heatmap_summary.get("slides_succeeded")))
        lines.append(metric_line("slides_failed", heatmap_summary.get("slides_failed")))
        lines.append(metric_line("mean_duration_sec", heatmap_summary.get("mean_duration_sec")))
        lines.append(metric_line("heatmap_dir", str(heatmap_dir)))
    else:
        lines.append("- 尚未生成 heatmap summary。")
    lines.append("")

    lines.append("## 后续跨倍率对比计划")
    lines.append("- 复用同一套训练入口，仅替换 feature_dir / split_dir / checkpoint 路径。")
    lines.append("- 复用同一套 heatmap 包装脚本与样本筛选规则，保持 SSL 为正类。")
    lines.append("- 使用相同报告框架对比 2.5X、5X、20X 的效率、性能与 attention 可解释性差异。")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
