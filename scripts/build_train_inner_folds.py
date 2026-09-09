#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a canonical train split into stratified inner folds."
    )
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--train-ids-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stratify-column", action="append", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--slide-column", default="slide_id")
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def read_id_list(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row[0].strip() for row in csv.reader(handle) if row and row[0].strip()]


def write_id_csv(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for value in values:
            writer.writerow([value])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stratify_key(row: dict[str, str], columns: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in columns)


def count_by_columns(rows: list[dict[str, str]], columns: list[str]) -> dict[str, int]:
    counter = Counter(" | ".join(f"{column}={row.get(column, '')}" for column in columns) for row in rows)
    return dict(sorted(counter.items()))


def build_inner_folds(
    rows: list[dict[str, str]],
    *,
    slide_column: str,
    stratify_columns: list[str],
    folds: int,
    seed: int,
) -> list[list[dict[str, str]]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[stratify_key(row, stratify_columns)].append(row)

    fold_rows: list[list[dict[str, str]]] = [[] for _ in range(folds)]
    for key in sorted(grouped):
        items = sorted(grouped[key], key=lambda row: str(row[slide_column]))
        rng.shuffle(items)
        for idx, row in enumerate(items):
            fold_rows[idx % folds].append(row)

    for fold in fold_rows:
        fold.sort(key=lambda row: str(row[slide_column]))
    return fold_rows


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be >= 2")

    ready_csv = Path(args.ready_csv)
    train_ids_csv = Path(args.train_ids_csv)
    output_dir = Path(args.output_dir)

    rows, fieldnames = read_rows(ready_csv)
    missing_columns = [column for column in [args.slide_column, *args.stratify_column] if column not in fieldnames]
    if missing_columns:
        raise KeyError(f"{ready_csv} is missing columns: {missing_columns}")

    row_by_id = {str(row[args.slide_column]): row for row in rows}
    train_ids = read_id_list(train_ids_csv)
    train_rows = [row_by_id[slide_id] for slide_id in train_ids if slide_id in row_by_id]
    dropped_train_ids = [slide_id for slide_id in train_ids if slide_id not in row_by_id]

    fold_rows = build_inner_folds(
        train_rows,
        slide_column=args.slide_column,
        stratify_columns=args.stratify_column,
        folds=args.folds,
        seed=args.seed,
    )

    all_train_ids = {str(row[args.slide_column]) for row in train_rows}
    stats = {
        "ready_csv": str(ready_csv),
        "train_ids_csv": str(train_ids_csv),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "folds": args.folds,
        "slide_column": args.slide_column,
        "stratify_columns": args.stratify_column,
        "source_train_count": len(train_ids),
        "matched_train_count": len(train_rows),
        "dropped_train_ids": dropped_train_ids,
        "source_train_distribution": count_by_columns(train_rows, args.stratify_column),
        "folds_summary": {},
    }

    covered_ids: set[str] = set()
    for fold_idx, val_rows in enumerate(fold_rows):
        val_ids = sorted(str(row[args.slide_column]) for row in val_rows)
        train_fold_ids = sorted(all_train_ids - set(val_ids))
        covered_ids.update(val_ids)

        write_id_csv(output_dir / f"flod-{fold_idx}.csv", val_ids)
        write_id_csv(output_dir / f"flod-{fold_idx}-train.csv", train_fold_ids)
        write_id_csv(output_dir / f"flod-{fold_idx}-val.csv", val_ids)

        stats["folds_summary"][f"flod-{fold_idx}"] = {
            "val_count": len(val_ids),
            "train_count": len(train_fold_ids),
            "val_distribution": count_by_columns(val_rows, args.stratify_column),
        }

    stats["covered_unique_ids"] = len(covered_ids)
    stats["overlap_or_missing_count"] = len(covered_ids ^ all_train_ids)
    stats["formal_ready"] = (
        not dropped_train_ids
        and stats["covered_unique_ids"] == len(all_train_ids)
        and stats["overlap_or_missing_count"] == 0
    )
    write_json(output_dir / "train_5fold_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
