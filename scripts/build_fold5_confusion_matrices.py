#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from clam_experiment_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fold-5 confusion matrix comparison figure.")
    parser.add_argument(
        "--output-dir",
        default="/data15/data15_5/yuexin2/adenoma/outputs/fold5_error_analysis/confusion_matrices",
        help="Directory to write confusion matrix comparison outputs.",
    )
    return parser.parse_args()


def run_specs() -> list[dict[str, str]]:
    return [
        {
            "family": "CLAM-SB",
            "magnification": "1X",
            "label": "CLAM-SB 1X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_1x_uni_clam_sb/fold-5/predictions.csv",
            "confusion_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_1x_uni_clam_sb/fold-5/confusion_matrix.csv",
        },
        {
            "family": "CLAM-SB",
            "magnification": "2.5X",
            "label": "CLAM-SB 2.5X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_2p5x_uni_clam_sb/fold-5/predictions.csv",
            "confusion_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_2p5x_uni_clam_sb/fold-5/confusion_matrix.csv",
        },
        {
            "family": "CLAM-SB",
            "magnification": "5X",
            "label": "CLAM-SB 5X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_5x_uni_clam_sb/fold-5/predictions.csv",
            "confusion_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_5x_uni_clam_sb/fold-5/confusion_matrix.csv",
        },
        {
            "family": "CLAM-SB",
            "magnification": "20X",
            "label": "CLAM-SB 20X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_uni_clam_sb/fold-5_rerun_gpu0_20260424/predictions.csv",
            "confusion_csv": "/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_uni_clam_sb/fold-5_rerun_gpu0_20260424/confusion_matrix.csv",
        },
        {
            "family": "TransMIL",
            "magnification": "1X",
            "label": "TransMIL 1X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_1x_uni/fold-5/predictions.csv",
            "confusion_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_1x_uni/fold-5/confusion_matrix.csv",
        },
        {
            "family": "TransMIL",
            "magnification": "2.5X",
            "label": "TransMIL 2.5X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_2p5x_uni/fold-5/predictions.csv",
            "confusion_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_2p5x_uni/fold-5/confusion_matrix.csv",
        },
        {
            "family": "TransMIL",
            "magnification": "5X",
            "label": "TransMIL 5X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_5x_uni/fold-5/predictions.csv",
            "confusion_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_5x_uni/fold-5/confusion_matrix.csv",
        },
        {
            "family": "TransMIL",
            "magnification": "20X",
            "label": "TransMIL 20X",
            "predictions_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_20x_uni/fold-5/predictions.csv",
            "confusion_csv": "/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_20x_uni/fold-5/confusion_matrix.csv",
        },
    ]


def load_confusion_from_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    return df.to_numpy(dtype=int)


def recompute_confusion(predictions_path: Path) -> np.ndarray:
    df = pd.read_csv(predictions_path)
    return confusion_matrix(df["label"].astype(int), df["pred"].astype(int), labels=[0, 1])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.6))
    cmap = plt.get_cmap("Blues")

    for idx, spec in enumerate(run_specs()):
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        predictions_path = Path(spec["predictions_csv"])
        confusion_path = Path(spec["confusion_csv"])
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {predictions_path}")
        if not confusion_path.exists():
            raise FileNotFoundError(f"Missing confusion matrix csv: {confusion_path}")

        computed_cm = recompute_confusion(predictions_path)
        file_cm = load_confusion_from_csv(confusion_path)
        if computed_cm.shape != (2, 2):
            raise ValueError(f"Unexpected confusion matrix shape for {predictions_path}: {computed_cm.shape}")

        im = ax.imshow(computed_cm, cmap=cmap, vmin=0)
        total = computed_cm.sum()
        for i in range(2):
            for j in range(2):
                count = int(computed_cm[i, j])
                pct = 100.0 * count / total if total else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", fontsize=10)

        ax.set_xticks([0, 1], ["pred others", "pred SSL"])
        ax.set_yticks([0, 1], ["true others", "true SSL"])
        ax.set_title(spec["label"])

        df = pd.read_csv(predictions_path)
        records.append(
            {
                "family": spec["family"],
                "magnification": spec["magnification"],
                "label": spec["label"],
                "predictions_csv": str(predictions_path),
                "confusion_csv": str(confusion_path),
                "tn": int(computed_cm[0, 0]),
                "fp": int(computed_cm[0, 1]),
                "fn": int(computed_cm[1, 0]),
                "tp": int(computed_cm[1, 1]),
                "n_test": int(len(df)),
                "n_positive_ssl": int((df["label"] == 1).sum()),
                "n_negative_others": int((df["label"] == 0).sum()),
                "matches_saved_confusion": bool(np.array_equal(computed_cm, file_cm)),
            }
        )

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Count")
    fig.suptitle("Fold-5 Confusion Matrices: CLAM-SB vs TransMIL", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = output_dir / "confusion_matrices_clamsb_vs_transmil_fold5.png"
    pdf_path = output_dir / "confusion_matrices_clamsb_vs_transmil_fold5.pdf"
    csv_path = output_dir / "confusion_matrices_clamsb_vs_transmil_fold5.csv"
    json_path = output_dir / "confusion_matrices_clamsb_vs_transmil_fold5.json"

    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    pd.DataFrame(records).to_csv(csv_path, index=False)
    write_json(
        json_path,
        {
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "summary_csv": str(csv_path),
            "runs": records,
            "note": "Confusion matrices were recomputed from predictions.csv and validated against each stored confusion_matrix.csv.",
        },
    )

    print(pd.DataFrame(records).to_csv(index=False))
    print(str(output_dir))


if __name__ == "__main__":
    main()
