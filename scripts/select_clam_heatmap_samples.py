#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clam_experiment_utils import add_prediction_annotations, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select representative slides for CLAM heatmaps.")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--positive-class-name", default="SSL")
    parser.add_argument("--negative-class-name", default="others")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def ranked_subset(df: pd.DataFrame, sort_by: str, ascending: bool, top_k: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = df.sort_values(sort_by, ascending=ascending).head(top_k).copy()
    ranked["rank_within_category"] = range(1, len(ranked) + 1)
    return ranked


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions_csv)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json) if args.summary_json else output_csv.with_suffix(".summary.json")

    predictions_df = pd.read_csv(predictions_path)
    predictions_df = add_prediction_annotations(
        predictions_df,
        positive_name=args.positive_class_name,
        negative_name=args.negative_class_name,
    )

    categories = {
        "high_conf_correct_ssl": ranked_subset(
            predictions_df[predictions_df["outcome_group"] == f"{args.positive_class_name}_tp"],
            sort_by="prob_ssl",
            ascending=False,
            top_k=args.top_k,
        ),
        "low_conf_correct_ssl": ranked_subset(
            predictions_df[predictions_df["outcome_group"] == f"{args.positive_class_name}_tp"],
            sort_by="prob_ssl",
            ascending=True,
            top_k=args.top_k,
        ),
        "ssl_false_negative": ranked_subset(
            predictions_df[predictions_df["outcome_group"] == f"{args.positive_class_name}_fn"],
            sort_by="prob_ssl",
            ascending=False,
            top_k=args.top_k,
        ),
        "others_false_positive": ranked_subset(
            predictions_df[predictions_df["outcome_group"] == f"{args.negative_class_name}_fp"],
            sort_by="prob_ssl",
            ascending=False,
            top_k=args.top_k,
        ),
        "others_true_negative_reference": ranked_subset(
            predictions_df[predictions_df["outcome_group"] == f"{args.negative_class_name}_tn"],
            sort_by="prob_ssl",
            ascending=False,
            top_k=args.top_k,
        ),
    }

    selected_frames = []
    for category_name, df in categories.items():
        if df.empty:
            continue
        tagged = df.copy()
        tagged["selection_category"] = category_name
        selected_frames.append(tagged)

    if selected_frames:
        selected_df = pd.concat(selected_frames, ignore_index=True)
        if "fold" not in selected_df.columns:
            selected_df["fold"] = pd.NA
        if "split" not in selected_df.columns:
            selected_df["split"] = "test"
        selected_df = selected_df[
            [
                "selection_category",
                "rank_within_category",
                "slide_id",
                "fold",
                "split",
                "label",
                "label_name",
                "pred",
                "pred_name",
                "prob_others",
                "prob_ssl",
                "confidence",
                "is_correct",
                "outcome_group",
            ]
        ].sort_values(["selection_category", "rank_within_category", "slide_id"])
    else:
        selected_df = pd.DataFrame(
            columns=[
                "selection_category",
                "rank_within_category",
                "slide_id",
                "fold",
                "split",
                "label",
                "label_name",
                "pred",
                "pred_name",
                "prob_others",
                "prob_ssl",
                "confidence",
                "is_correct",
                "outcome_group",
            ]
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(output_csv, index=False)

    summary = {
        "predictions_csv": str(predictions_path),
        "output_csv": str(output_csv),
        "top_k_per_category": args.top_k,
        "available_counts": {
            key: int(len(value))
            for key, value in categories.items()
        },
        "selected_total": int(len(selected_df)),
        "selected_counts": selected_df["selection_category"].value_counts().to_dict() if not selected_df.empty else {},
    }
    write_json(summary_json, summary)
    print(selected_df.to_csv(index=False))


if __name__ == "__main__":
    main()
