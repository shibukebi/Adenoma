#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from clam_experiment_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fold-5 precision-recall comparison plots.")
    parser.add_argument(
        "--output-dir",
        default="/data15/data15_5/yuexin2/adenoma/outputs/pr_comparisons/fold5",
        help="Directory to write PR comparison figures and summaries.",
    )
    return parser.parse_args()


def run_specs() -> list[dict[str, str]]:
    return [
        {
            "family": "CLAM-SB",
            "magnification": "1X",
            "label": "CLAM-SB 1X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_1x_uni_clam_sb/fold-5/predictions.csv",
        },
        {
            "family": "CLAM-SB",
            "magnification": "2.5X",
            "label": "CLAM-SB 2.5X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_2p5x_uni_clam_sb/fold-5/predictions.csv",
        },
        {
            "family": "CLAM-SB",
            "magnification": "5X",
            "label": "CLAM-SB 5X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_5x_uni_clam_sb/fold-5/predictions.csv",
        },
        {
            "family": "CLAM-SB",
            "magnification": "20X",
            "label": "CLAM-SB 20X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_uni_clam_sb/fold-5_rerun_gpu0_20260424/predictions.csv",
        },
        {
            "family": "TransMIL",
            "magnification": "1X",
            "label": "TransMIL 1X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_1x_uni/fold-5/predictions.csv",
        },
        {
            "family": "TransMIL",
            "magnification": "2.5X",
            "label": "TransMIL 2.5X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_2p5x_uni/fold-5/predictions.csv",
        },
        {
            "family": "TransMIL",
            "magnification": "5X",
            "label": "TransMIL 5X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_5x_uni/fold-5/predictions.csv",
        },
        {
            "family": "TransMIL",
            "magnification": "20X",
            "label": "TransMIL 20X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_20x_uni/fold-5/predictions.csv",
        },
    ]


COLOR_MAP = {
    "1X": "#8c564b",
    "2.5X": "#1f77b4",
    "5X": "#2ca02c",
    "20X": "#d62728",
}

LINESTYLE_MAP = {
    "CLAM-SB": "-",
    "TransMIL": "--",
}


def load_predictions(spec: dict[str, str]) -> dict:
    predictions_path = Path(spec["predictions_csv"])
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    df = pd.read_csv(predictions_path)
    required_columns = {"label", "prob_ssl"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{predictions_path} is missing required columns: {sorted(missing_columns)}")

    y_true = df["label"].to_numpy(dtype=int)
    y_score = df["prob_ssl"].to_numpy(dtype=float)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = float(average_precision_score(y_true, y_score))
    positive_prevalence = float((df["label"] == 1).mean())

    return {
        **spec,
        "predictions_path": predictions_path,
        "dataframe": df,
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision,
        "n_test": int(len(df)),
        "n_positive_ssl": int((df["label"] == 1).sum()),
        "n_negative_others": int((df["label"] == 0).sum()),
        "positive_prevalence": positive_prevalence,
    }


def build_summary_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        rows.append(
            {
                "family": record["family"],
                "magnification": record["magnification"],
                "label": record["label"],
                "predictions_csv": str(record["predictions_path"]),
                "n_test": record["n_test"],
                "n_positive_ssl": record["n_positive_ssl"],
                "n_negative_others": record["n_negative_others"],
                "positive_prevalence": record["positive_prevalence"],
                "average_precision": record["average_precision"],
            }
        )
    return rows


def build_curve_points(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        for recall, precision in zip(record["recall"], record["precision"]):
            rows.append(
                {
                    "family": record["family"],
                    "magnification": record["magnification"],
                    "label": record["label"],
                    "recall": float(recall),
                    "precision": float(precision),
                    "average_precision": record["average_precision"],
                }
            )
    return pd.DataFrame(rows)


def save_figure_bundle(
    records: list[dict],
    output_dir: Path,
    stem: str,
    title: str,
    points_csv: Path,
    family_only_solid: bool = False,
) -> None:
    summary_rows = build_summary_rows(records)
    summary_df = pd.DataFrame(summary_rows)

    prevalences = {round(row["positive_prevalence"], 12) for row in summary_rows}
    if len(prevalences) != 1:
        raise ValueError(f"Inconsistent positive prevalence across runs for {stem}: {sorted(prevalences)}")
    baseline = next(iter(prevalences))

    plt.figure(figsize=(8.2, 6.6))
    for record in records:
        linestyle = "-" if family_only_solid else LINESTYLE_MAP[record["family"]]
        plt.plot(
            record["recall"],
            record["precision"],
            label=f"{record['label']} (AP={record['average_precision']:.4f})",
            color=COLOR_MAP[record["magnification"]],
            linestyle=linestyle,
            linewidth=2.2,
        )

    plt.axhline(
        baseline,
        color="#666666",
        linestyle=":",
        linewidth=1.4,
        label=f"Positive prevalence ({baseline:.4f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower left")
    plt.tight_layout()

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"

    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()

    summary_df.to_csv(csv_path, index=False)
    write_json(
        json_path,
        {
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "summary_csv": str(csv_path),
            "curve_points_csv": str(points_csv),
            "positive_prevalence": baseline,
            "runs": summary_rows,
            "note": "SSL is treated as the positive class and AP is average precision over prob_ssl.",
        },
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [load_predictions(spec) for spec in run_specs()]
    points_df = build_curve_points(records)
    points_csv = output_dir / "pr_curve_points_fold5.csv"
    points_df.to_csv(points_csv, index=False)

    save_figure_bundle(
        records,
        output_dir,
        stem="pr_comparison_clamsb_vs_transmil_1x_2p5x_5x_20x_fold5",
        title="Fold-5 PR Comparison: CLAM-SB vs TransMIL @ 1X / 2.5X / 5X / 20X",
        points_csv=points_csv,
        family_only_solid=False,
    )
    save_figure_bundle(
        [record for record in records if record["family"] == "CLAM-SB"],
        output_dir,
        stem="pr_comparison_clamsb_1x_2p5x_5x_20x_fold5",
        title="Fold-5 PR Comparison: CLAM-SB @ 1X / 2.5X / 5X / 20X",
        points_csv=points_csv,
        family_only_solid=True,
    )
    save_figure_bundle(
        [record for record in records if record["family"] == "TransMIL"],
        output_dir,
        stem="pr_comparison_transmil_1x_2p5x_5x_20x_fold5",
        title="Fold-5 PR Comparison: TransMIL @ 1X / 2.5X / 5X / 20X",
        points_csv=points_csv,
        family_only_solid=True,
    )

    print(pd.DataFrame(build_summary_rows(records)).to_csv(index=False))
    print(str(output_dir))


if __name__ == "__main__":
    main()
