#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from clam_experiment_utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CLAM-SB AUC bar chart across magnifications.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def metric_specs() -> list[dict[str, str]]:
    return [
        {
            "label": "2.5X",
            "metrics_json": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_2p5x_clam_sb_rerun_20260420_233727/fold-0/metrics.json",
            "color": "#1f77b4",
        },
        {
            "label": "5X",
            "metrics_json": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_5x_clam_sb_rerun_20260421_185110/fold-0/metrics.json",
            "color": "#2ca02c",
        },
        {
            "label": "20X",
            "metrics_json": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_clam_sb_rerun_20260421_110952/fold-0/metrics.json",
            "color": "#d62728",
        },
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in metric_specs():
        metrics_path = Path(spec["metrics_json"])
        payload = read_json(metrics_path)
        rows.append(
            {
                "label": spec["label"],
                "metrics_json": str(metrics_path),
                "auc": float(payload["auc"]),
                "n_test": int(payload["n_test"]),
                "n_positive_ssl": int(payload["ssl_test"]),
                "n_negative_others": int(payload["others_test"]),
                "color": spec["color"],
            }
        )

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    bars = ax.bar(df["label"], df["auc"], color=df["color"], width=0.62, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Magnification")
    ax.set_title("CLAM-SB Fold-0 AUC Comparison")
    ax.grid(axis="y", alpha=0.25)

    for bar, (_, row) in zip(bars, df.iterrows()):
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.4f}\n(n={int(row['n_test'])})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    png_path = output_dir / "auc_bar_clamsb_2p5x_5x_20x_fold0.png"
    pdf_path = output_dir / "auc_bar_clamsb_2p5x_5x_20x_fold0.pdf"
    csv_path = output_dir / "auc_bar_clamsb_2p5x_5x_20x_fold0.csv"
    json_path = output_dir / "auc_bar_clamsb_2p5x_5x_20x_fold0.json"

    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    df.drop(columns=["color"]).to_csv(csv_path, index=False)
    write_json(
        json_path,
        {
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "summary_csv": str(csv_path),
            "runs": df.drop(columns=["color"]).to_dict(orient="records"),
            "note": "AUC bars are computed from the current CLAM-SB fold-0 rerun metrics. n_test is shown above each bar.",
        },
    )

    print(df.drop(columns=["color"]).to_csv(index=False))
    print(str(png_path))


if __name__ == "__main__":
    main()
