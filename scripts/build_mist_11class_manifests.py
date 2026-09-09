#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


CLASS_NAMES = [
    "ssl",
    "hp",
    "TSA",
    "USA",
    "TA",
    "TVA",
    "IP",
    "ssl with highgrade dysplasia",
    "TSA with highgrade dysplasia",
    "TA with highgrade dysplasia",
    "TVA with highgrade dysplasia",
]

TYPE_TO_LOW_CLASS = {
    "Sessile serrated adenoma": "ssl",
    "Hyperplastic polyps": "hp",
    "Traditional serrated adenoma": "TSA",
    "Unclassified serrated adenoma": "USA",
    "Tubular adenoma": "TA",
    "Tubulovillous adenoma": "TVA",
    "Inflammatory polyp": "IP",
}

TYPE_TO_HIGH_CLASS = {
    "Sessile serrated adenoma": "ssl with highgrade dysplasia",
    "Traditional serrated adenoma": "TSA with highgrade dysplasia",
    "Tubular adenoma": "TA with highgrade dysplasia",
    "Tubulovillous adenoma": "TVA with highgrade dysplasia",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIST manifests for the 11-class hp+yx fold experiment.")
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--fold", type=int, default=4, help="Zero-based fold index. fold=4 is the fifth fold.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-prefix", default="mist_hp_yx_11class_fold5")
    return parser.parse_args()


def read_split_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row and row[0].strip():
                ids.append(row[0].strip())
    return ids


def map_class(row: pd.Series) -> str:
    lesion_type = str(row["type"])
    grade = str(row["grade"]).lower()
    if grade == "high" and lesion_type in TYPE_TO_HIGH_CLASS:
        return TYPE_TO_HIGH_CLASS[lesion_type]
    if lesion_type in TYPE_TO_LOW_CLASS:
        return TYPE_TO_LOW_CLASS[lesion_type]
    raise ValueError(f"Unsupported type/grade combination: {lesion_type!r}, {grade!r}")


def find_feature_path(feature_root: Path, slide_id: str) -> Path:
    matches = list((feature_root / "features").glob(f"*/{slide_id}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one feature CSV for {slide_id}, found {len(matches)}")
    return matches[0]


def write_manifest(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str, "case_id": str})
    feature_root = Path(args.feature_root)
    split_dir = Path(args.split_dir)
    output_root = Path(args.output_root)
    class_to_id = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    ready_df["mist_11class_name"] = ready_df.apply(map_class, axis=1)
    ready_df["mist_11class_label"] = ready_df["mist_11class_name"].map(class_to_id).astype(int)
    slide_data = ready_df.set_index("slide_id", drop=False)

    split_outputs: dict[str, str] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_path = split_dir / f"flod-{args.fold}-{split_name}.csv"
        split_ids = read_split_ids(split_path)
        rows: list[tuple[str, int]] = []
        labels: list[str] = []
        for slide_id in split_ids:
            if slide_id not in slide_data.index:
                continue
            row = slide_data.loc[slide_id]
            feature_path = find_feature_path(feature_root, slide_id)
            rows.append((str(feature_path), int(row["mist_11class_label"])))
            labels.append(str(row["mist_11class_name"]))

        dataset_name = f"{args.dataset_prefix}_{split_name}"
        manifest_path = output_root / dataset_name / f"{dataset_name}.csv"
        write_manifest(manifest_path, rows)
        split_outputs[split_name] = str(manifest_path)
        split_counts[split_name] = pd.Series(labels).value_counts().reindex(CLASS_NAMES, fill_value=0).astype(int).to_dict()

    metadata = {
        "ready_csv": str(args.ready_csv),
        "feature_root": str(feature_root),
        "split_dir": str(split_dir),
        "fold": int(args.fold),
        "fold_name": "fold5" if args.fold == 4 else f"fold{args.fold}",
        "class_names": CLASS_NAMES,
        "class_to_id": class_to_id,
        "split_manifests": split_outputs,
        "split_counts": split_counts,
    }
    metadata_path = output_root / f"{args.dataset_prefix}_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
