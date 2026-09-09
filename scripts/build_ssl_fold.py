#!/usr/bin/env python3
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one stratified MAG-GLTrans fold for the SSL/others task."
    )
    parser.add_argument(
        "--label-csv",
        default="/data15/data15_5/yuexin2/adenoma/data/adenoma_yx_ssl_others_labels.csv",
        help="Binary label csv with slide_id and label_name columns",
    )
    parser.add_argument(
        "--output-dir",
        default="/data15/data15_5/yuexin2/adenoma/data/mag_gltrans_folds/adenoma_yx_ssl",
        help="Output directory for flod-{fold}-{train,val,test}.csv",
    )
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to materialize")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")
    return parser.parse_args()


def split_class_items(items: List[str], val_ratio: float, test_ratio: float) -> Tuple[List[str], List[str], List[str]]:
    total = len(items)
    n_val = round(total * val_ratio)
    n_test = round(total * test_ratio)
    if n_val + n_test >= total:
        n_test = max(1, n_test)
        n_val = max(1, total - n_test - 1)
    n_train = total - n_val - n_test
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def write_list(path: Path, values: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for value in values:
            writer.writerow([value])


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed + args.fold)
    output_dir = Path(args.output_dir)

    grouped: Dict[str, List[str]] = defaultdict(list)
    with open(args.label_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["label_name"]].append(row["slide_id"])

    for key in grouped:
        rng.shuffle(grouped[key])

    split_map = {"train": [], "val": [], "test": []}
    class_summary = {}

    for label_name, items in grouped.items():
        train, val, test = split_class_items(items, args.val_ratio, args.test_ratio)
        split_map["train"].extend(train)
        split_map["val"].extend(val)
        split_map["test"].extend(test)
        class_summary[label_name] = {"train": len(train), "val": len(val), "test": len(test)}

    for split_name in split_map:
        split_map[split_name].sort()

    write_list(output_dir / f"flod-{args.fold}-train.csv", split_map["train"])
    write_list(output_dir / f"flod-{args.fold}-val.csv", split_map["val"])
    write_list(output_dir / f"flod-{args.fold}-test.csv", split_map["test"])

    print(f"seed={args.seed}")
    print(f"fold={args.fold}")
    print(f"output_dir={output_dir}")
    print(f"class_summary={class_summary}")
    print(
        "split_sizes="
        + str({name: len(values) for name, values in split_map.items()})
    )
    print(
        "split_counts="
        + str(
            {
                name: Counter(
                    "SSL" if value in set(grouped.get("SSL", [])) else "others"
                    for value in values
                )
                for name, values in split_map.items()
            }
        )
    )


if __name__ == "__main__":
    main()
