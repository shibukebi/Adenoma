#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from clam_experiment_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CLAM-SB ROC comparison plot across magnifications.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def run_specs() -> list[dict[str, str]]:
    return [
        {
            "label": "CLAM-SB 2.5X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_2p5x_clam_sb_rerun_20260420_233727/fold-0/predictions.csv",
            "color": "#1f77b4",
            "linestyle": "-",
        },
        {
            "label": "CLAM-SB 5X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_5x_clam_sb_rerun_20260421_185110/fold-0/predictions.csv",
            "color": "#2ca02c",
            "linestyle": "-",
        },
        {
            "label": "CLAM-SB 20X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_clam_sb_rerun_20260421_110952/fold-0/predictions.csv",
            "color": "#d62728",
            "linestyle": "-",
        },
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    plt.figure(figsize=(7.0, 6.2))

    for spec in run_specs():
        predictions_path = Path(spec["predictions_csv"])
        df = pd.read_csv(predictions_path)
        y_true = df["label"].to_numpy(dtype=int)
        y_score = df["prob_ssl"].to_numpy(dtype=float)
        auc_value = float(roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)

        plt.plot(
            fpr,
            tpr,
            label=f"{spec['label']} (AUC={auc_value:.4f})",
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=2.2,
        )

        summary_rows.append(
            {
                "label": spec["label"],
                "predictions_csv": str(predictions_path),
                "n_test": int(len(df)),
                "n_positive_ssl": int((df["label"] == 1).sum()),
                "n_negative_others": int((df["label"] == 0).sum()),
                "auc": auc_value,
            }
        )

    plt.plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=1.2, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fold-0 ROC Comparison: CLAM-SB @ 2.5X / 5X / 20X")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()

    png_path = output_dir / "roc_comparison_clamsb_2p5x_5x_20x_fold0.png"
    pdf_path = output_dir / "roc_comparison_clamsb_2p5x_5x_20x_fold0.pdf"
    csv_path = output_dir / "roc_comparison_clamsb_2p5x_5x_20x_fold0.csv"
    json_path = output_dir / "roc_comparison_clamsb_2p5x_5x_20x_fold0.json"

    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False)
    write_json(
        json_path,
        {
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "summary_csv": str(csv_path),
            "runs": summary_rows,
            "note": "SSL is treated as the positive class for all three ROC curves.",
        },
    )

    print(summary_df.to_csv(index=False))
    print(str(png_path))


if __name__ == "__main__":
    main()
