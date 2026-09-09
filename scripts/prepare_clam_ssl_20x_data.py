#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from clam_experiment_utils import (
    TASK_METADATA_COLUMNS,
    get_default_min_positive_per_split,
    get_task_spec,
    prepare_task_row,
    read_id_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CLAM/TransMIL manifests for either the stage-1 SSL task or the stage-2 dysplasia task."
    )
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--raw-fold-dir", required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--task-mode", choices=["ssl_binary", "dysplasia_binary"], default="ssl_binary")
    parser.add_argument("--min-positive-per-split", type=int, default=None)
    parser.add_argument("--fold", type=int, default=0)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_dict_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_id_csv(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for value in values:
            writer.writerow([value])


def count_labels(ids: list[str], label_name_map: dict[str, str]) -> dict[str, int]:
    return dict(Counter(label_name_map[slide_id] for slide_id in ids))


def main() -> None:
    args = parse_args()
    task_spec = get_task_spec(args.task_mode)
    min_positive_per_split = (
        args.min_positive_per_split
        if args.min_positive_per_split is not None
        else get_default_min_positive_per_split(args.task_mode)
    )

    label_csv = Path(args.label_csv)
    patch_dir = Path(args.patch_dir)
    feature_dir = Path(args.feature_dir)
    raw_fold_dir = Path(args.raw_fold_dir)
    dataset_csv = Path(args.dataset_csv)
    ready_csv = Path(args.ready_csv)
    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    patch_ids = {path.stem for path in patch_dir.glob("*.h5")}
    feat_h5_ids = {path.stem for path in (feature_dir / "h5_files").glob("*.h5")}
    feat_pt_ids = {path.stem for path in (feature_dir / "pt_files").glob("*.pt")}

    label_rows = read_csv_rows(label_csv)
    dataset_rows = []
    ready_rows = []

    for row in label_rows:
        record = prepare_task_row(row, args.task_mode)
        if record is None:
            continue
        dataset_rows.append(record)
        slide_id = str(record["slide_id"])
        if slide_id in patch_ids and slide_id in feat_h5_ids and slide_id in feat_pt_ids:
            ready_rows.append(record)

    fieldnames = ["case_id", "slide_id", "label", "label_name", "label_value", "task_mode", *TASK_METADATA_COLUMNS]
    dataset_rows.sort(key=lambda row: row["slide_id"])
    ready_rows.sort(key=lambda row: row["slide_id"])
    write_dict_csv(dataset_csv, dataset_rows, fieldnames)
    write_dict_csv(ready_csv, ready_rows, fieldnames)

    ready_ids = {row["slide_id"] for row in ready_rows}
    label_name_map = {row["slide_id"]: row["label_name"] for row in dataset_rows}

    split_stats = {
        "task_mode": args.task_mode,
        "task_name": task_spec["task_name"],
        "dataset_rows": len(dataset_rows),
        "ready_rows": len(ready_rows),
        "missing_patch_h5": len([row for row in dataset_rows if row["slide_id"] not in patch_ids]),
        "missing_feature_h5": len([row for row in dataset_rows if row["slide_id"] not in feat_h5_ids]),
        "missing_feature_pt": len([row for row in dataset_rows if row["slide_id"] not in feat_pt_ids]),
        "ready_label_counts": count_labels(sorted(ready_ids), label_name_map),
        "positive_name": task_spec["positive_name"],
        "negative_name": task_spec["negative_name"],
        "min_positive_per_split": int(min_positive_per_split),
        "splits": {},
        "formal_ready": True,
        "warnings": [],
    }

    positive_name = str(task_spec["positive_name"])
    negative_name = str(task_spec["negative_name"])
    for split_name in ["train", "val", "test"]:
        raw_ids = read_id_list(raw_fold_dir / f"flod-{args.fold}-{split_name}.csv")
        filtered_ids = sorted([slide_id for slide_id in raw_ids if slide_id in ready_ids])
        write_id_csv(split_dir / f"flod-{args.fold}-{split_name}.csv", filtered_ids)
        label_counts = count_labels(filtered_ids, label_name_map) if filtered_ids else {}
        split_stats["splits"][split_name] = {
            "raw_count": len(raw_ids),
            "filtered_count": len(filtered_ids),
            "dropped_count": len(raw_ids) - len(filtered_ids),
            "label_counts": label_counts,
        }
        if label_counts.get(positive_name, 0) < min_positive_per_split:
            split_stats["formal_ready"] = False
            split_stats["warnings"].append(
                f"{split_name} split has only {label_counts.get(positive_name, 0)} {positive_name} samples (< {min_positive_per_split})"
            )
        if label_counts.get(negative_name, 0) == 0 or label_counts.get(positive_name, 0) == 0:
            split_stats["formal_ready"] = False
            split_stats["warnings"].append(f"{split_name} split is missing at least one class")

    stats_path = split_dir / f"flod-{args.fold}_stats.json"
    stats_path.write_text(json.dumps(split_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"task_mode={args.task_mode}")
    print(f"dataset_csv={dataset_csv}")
    print(f"ready_csv={ready_csv}")
    print(f"stats_json={stats_path}")
    print(json.dumps(split_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
