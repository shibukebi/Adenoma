#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Route C priority manifest ordered for fold-0 training readiness."
    )
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_id_list(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[0].strip() for row in csv.reader(handle) if row and row[0].strip()}


def main() -> None:
    args = parse_args()
    manifest_rows = read_manifest(Path(args.manifest_csv))
    label_map = {
        row["slide_id"]: row["label_name"]
        for row in csv.DictReader(open(args.label_csv, newline="", encoding="utf-8"))
    }

    train_ids = read_id_list(Path(args.fold_dir) / "flod-0-train.csv")
    val_ids = read_id_list(Path(args.fold_dir) / "flod-0-val.csv")
    test_ids = read_id_list(Path(args.fold_dir) / "flod-0-test.csv")

    prioritized = []
    for row in manifest_rows:
        slide_id = row["slide_id"]
        split = "unassigned"
        if slide_id in val_ids:
            split = "val"
        elif slide_id in test_ids:
            split = "test"
        elif slide_id in train_ids:
            split = "train"

        label_name = label_map.get(slide_id, "")
        if split == "val":
            priority_bucket = 0
        elif split == "test":
            priority_bucket = 1
        elif split == "train" and label_name == "SSL":
            priority_bucket = 2
        elif split == "train":
            priority_bucket = 3
        else:
            priority_bucket = 4

        enriched = dict(row)
        enriched["label_name"] = label_name
        enriched["split_name"] = split
        enriched["priority_bucket"] = priority_bucket
        prioritized.append(enriched)

    prioritized.sort(key=lambda row: (int(row["priority_bucket"]), row["slide_id"]))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["slide_id", "slide_filename", "slide_path", "label_name", "split_name", "priority_bucket"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prioritized)

    counts = {}
    for bucket in range(5):
        counts[str(bucket)] = sum(1 for row in prioritized if int(row["priority_bucket"]) == bucket)
    print(f"output_csv={output_path}")
    print(f"bucket_counts={counts}")


if __name__ == "__main__":
    main()
