#!/usr/bin/env python3
"""
按文件大小尽量均衡地为 CLAM 预处理生成分片 process_list。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build size-balanced preprocess shards.")
    parser.add_argument("--manifest-csv", type=Path, required=True, help="Input manifest csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store shard csv files.")
    parser.add_argument("--num-shards", type=int, default=8, help="Number of shards to create.")
    parser.add_argument(
        "--filename-column",
        default="slide_filename",
        help="Manifest column used as CLAM slide_id filename.",
    )
    return parser.parse_args()


def load_rows(manifest_csv: Path, filename_column: str) -> list[dict]:
    with manifest_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"Manifest is empty: {manifest_csv}")

    required_columns = {filename_column, "slide_path"}
    missing = required_columns.difference(rows[0].keys())
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")

    for row in rows:
        slide_path = Path(row["slide_path"])
        row["slide_id"] = row[filename_column]
        row["size_bytes"] = slide_path.stat().st_size

    return rows


def assign_shards(rows: list[dict], num_shards: int) -> tuple[list[list[dict]], list[int]]:
    shards = [[] for _ in range(num_shards)]
    shard_sizes = [0 for _ in range(num_shards)]

    for row in sorted(rows, key=lambda item: item["size_bytes"], reverse=True):
        shard_idx = min(range(num_shards), key=lambda idx: shard_sizes[idx])
        row = dict(row)
        row["shard_id"] = shard_idx
        row["process"] = 1
        shards[shard_idx].append(row)
        shard_sizes[shard_idx] += row["size_bytes"]

    return shards, shard_sizes


def write_outputs(output_dir: Path, shards: list[list[dict]], shard_sizes: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slide_id",
        "slide_filename",
        "slide_path",
        "size_bytes",
        "shard_id",
        "process",
    ]

    summary = []
    for shard_idx, rows in enumerate(shards):
        shard_csv = output_dir / f"process_list_shard_{shard_idx}.csv"
        with shard_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        summary.append(
            {
                "shard_id": shard_idx,
                "num_slides": len(rows),
                "total_size_bytes": shard_sizes[shard_idx],
                "process_list": str(shard_csv),
            }
        )

    summary_path = output_dir / "shard_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    for item in summary:
        total_gib = item["total_size_bytes"] / (1024 ** 3)
        print(
            f"shard {item['shard_id']}: {item['num_slides']} slides, "
            f"{total_gib:.2f} GiB -> {item['process_list']}"
        )
    print(f"Wrote summary: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")

    rows = load_rows(args.manifest_csv, args.filename_column)
    shards, shard_sizes = assign_shards(rows, args.num_shards)
    write_outputs(args.output_dir, shards, shard_sizes)


if __name__ == "__main__":
    main()
