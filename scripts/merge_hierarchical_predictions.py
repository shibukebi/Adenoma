#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from clam_experiment_utils import (
    FINAL_LABEL_DICT,
    compute_multiclass_metrics,
    enrich_label_row,
    merge_hierarchical_predictions,
    select_routing_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge stage-1 SSL predictions and stage-2 dysplasia predictions into a three-class hierarchical result.")
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--stage1-val-predictions", required=True)
    parser.add_argument("--stage1-test-predictions", required=True)
    parser.add_argument("--stage2-full-test-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_hierarchical_labels(path: Path) -> pd.DataFrame:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [enrich_label_row(row) for row in csv.DictReader(handle)]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_df = load_hierarchical_labels(Path(args.label_csv))
    stage1_val_df = pd.read_csv(args.stage1_val_predictions, dtype={"slide_id": str})
    stage1_test_df = pd.read_csv(args.stage1_test_predictions, dtype={"slide_id": str})
    stage2_full_test_df = pd.read_csv(args.stage2_full_test_predictions, dtype={"slide_id": str})

    threshold_payload = select_routing_threshold(stage1_val_df, positive_score_col="prob_ssl", positive_label=1)
    threshold = float(threshold_payload["threshold"])

    merged_df = merge_hierarchical_predictions(
        stage1_test_df=stage1_test_df,
        stage2_full_test_df=stage2_full_test_df,
        label_df=label_df,
        routing_threshold=threshold,
    )

    if set(stage1_test_df["slide_id"]) - set(stage2_full_test_df["slide_id"]):
        missing = sorted(set(stage1_test_df["slide_id"]) - set(stage2_full_test_df["slide_id"]))
        raise RuntimeError(f"Stage-2 full-test predictions are missing slide_ids: {missing[:10]}")

    metrics, cm_df = compute_multiclass_metrics(
        merged_df,
        true_col="true_final_label",
        pred_col="final_pred",
        label_dict=FINAL_LABEL_DICT,
    )
    metrics.update(
        {
            "routing_threshold": threshold,
            "routing_threshold_f2": threshold_payload["f2"],
            "routing_threshold_precision": threshold_payload["precision"],
            "routing_threshold_recall": threshold_payload["recall"],
            "n_routed_to_stage2": int(merged_df["routed_to_stage2"].sum()),
        }
    )

    merged_df.to_csv(output_dir / "predictions.csv", index=False)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "routing_threshold.json").write_text(
        json.dumps(threshold_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
