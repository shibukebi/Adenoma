#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clam_experiment_utils import (
    build_patch_count_table,
    load_split_ids,
    read_json,
    summarize_numeric,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CLAM SSL experiment metrics and efficiency.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--magnification", default="2.5X")
    parser.add_argument("--heatmap-dir", default=None)
    parser.add_argument("--stage-timing-dir", default=None)
    parser.add_argument("--baseline-results-dir", action="append", default=[])
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def build_patch_summary(patch_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if patch_df.empty:
        return pd.DataFrame(
            columns=["group_type", "group_name", "count", "mean", "median", "min", "max", "p25", "p75", "p90", "p95"]
        )

    for split_name in sorted(patch_df["split"].unique()):
        stats = summarize_numeric(patch_df.loc[patch_df["split"] == split_name, "patch_count"])
        rows.append({"group_type": "split", "group_name": split_name, **stats})

    for (split_name, label_name), subset in patch_df.groupby(["split", "label_name"]):
        stats = summarize_numeric(subset["patch_count"])
        rows.append({"group_type": "split_label", "group_name": f"{split_name}:{label_name}", **stats})

    rows.append({"group_type": "overall", "group_name": "all", **summarize_numeric(patch_df["patch_count"])})
    return pd.DataFrame(rows)


def collect_efficiency_rows(
    results_dir: Path,
    stage_timing_dir: Path | None,
    heatmap_dir: Path | None,
    magnification: str,
) -> pd.DataFrame:
    rows = []

    if stage_timing_dir and stage_timing_dir.exists():
        for json_path in sorted(stage_timing_dir.glob("*.json")):
            payload = read_json(json_path)
            rows.append(
                {
                    "stage": payload.get("stage", json_path.stem),
                    "magnification": magnification,
                    "duration_sec": payload.get("duration_sec"),
                    "exit_code": payload.get("exit_code", 0),
                    "source": str(json_path),
                }
            )

    training_efficiency_path = results_dir / "training_efficiency.json"
    if training_efficiency_path.exists():
        payload = read_json(training_efficiency_path)
        rows.append(
            {
                "stage": "training_total",
                "magnification": magnification,
                "duration_sec": payload.get("total_training_time_sec"),
                "exit_code": 0,
                "source": str(training_efficiency_path),
            }
        )

    epoch_timing_path = results_dir / "epoch_timing.csv"
    if epoch_timing_path.exists():
        epoch_df = pd.read_csv(epoch_timing_path)
        for row in epoch_df.itertuples(index=False):
            rows.append(
                {
                    "stage": f"training_epoch_{int(row.epoch)}",
                    "magnification": magnification,
                    "duration_sec": float(row.duration_sec),
                    "exit_code": 0,
                    "source": str(epoch_timing_path),
                }
            )

    if heatmap_dir:
        heatmap_timing_path = heatmap_dir / "heatmap_timing.csv"
        if heatmap_timing_path.exists():
            heatmap_df = pd.read_csv(heatmap_timing_path)
            for row in heatmap_df.itertuples(index=False):
                rows.append(
                    {
                        "stage": f"heatmap_{row.slide_id}",
                        "magnification": magnification,
                        "duration_sec": float(row.duration_sec) if pd.notna(row.duration_sec) else float("nan"),
                        "exit_code": int(row.exit_code),
                        "source": str(heatmap_timing_path),
                    }
                )
            if not heatmap_df.empty:
                rows.append(
                    {
                        "stage": "heatmap_mean_per_slide",
                        "magnification": magnification,
                        "duration_sec": float(heatmap_df["duration_sec"].dropna().mean()),
                        "exit_code": 0,
                        "source": str(heatmap_timing_path),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    heatmap_dir = Path(args.heatmap_dir) if args.heatmap_dir else None
    stage_timing_dir = Path(args.stage_timing_dir) if args.stage_timing_dir else results_dir.parent / "stage_timing"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str})
    split_ids = load_split_ids(Path(args.split_dir), args.fold)
    patch_df = build_patch_count_table(ready_df, split_ids, Path(args.patch_dir))
    patch_df.to_csv(output_dir / "patch_counts_by_slide.csv", index=False)

    patch_summary_df = build_patch_summary(patch_df)
    patch_summary_df.to_csv(output_dir / "patch_count_summary.csv", index=False)

    current_metrics = read_json(results_dir / "metrics.json")
    performance_rows = [{"experiment": results_dir.name, "results_dir": str(results_dir), **current_metrics}]
    for baseline_dir_str in args.baseline_results_dir:
        baseline_dir = Path(baseline_dir_str)
        metrics_path = baseline_dir / "metrics.json"
        if metrics_path.exists():
            performance_rows.append(
                {
                    "experiment": baseline_dir.name,
                    "results_dir": str(baseline_dir),
                    **read_json(metrics_path),
                }
            )
    performance_df = pd.DataFrame(performance_rows)
    performance_df.to_csv(output_dir / "performance_summary.csv", index=False)

    efficiency_df = collect_efficiency_rows(results_dir, stage_timing_dir, heatmap_dir, args.magnification)
    efficiency_df.to_csv(output_dir / "efficiency_summary.csv", index=False)

    summary = {
        "results_dir": str(results_dir),
        "analysis_dir": str(output_dir),
        "patch_count_rows": int(len(patch_df)),
        "performance_rows": int(len(performance_df)),
        "efficiency_rows": int(len(efficiency_df)),
        "patch_count_summary_csv": str(output_dir / "patch_count_summary.csv"),
        "performance_summary_csv": str(output_dir / "performance_summary.csv"),
        "efficiency_summary_csv": str(output_dir / "efficiency_summary.csv"),
    }
    write_json(output_dir / "analysis_summary.json", summary)
    print(pd.DataFrame([summary]).to_csv(index=False))


if __name__ == "__main__":
    main()
