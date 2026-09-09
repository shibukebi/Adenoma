#!/usr/bin/env python3
"""
按 patch h5 文件大小为特征提取构建均衡分片。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build size-balanced feature extraction shards.")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV with slide_id column.")
    parser.add_argument("--patch-dir", type=Path, required=True, help="Patch h5 directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store shard csv files.")
    parser.add_argument("--num-shards", type=int, required=True, help="Number of shards.")
    return parser.parse_args()


def load_rows(input_csv: Path, patch_dir: Path) -> list[dict]:
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"Input csv is empty: {input_csv}")
    if "slide_id" not in rows[0]:
        raise SystemExit(f"Input csv missing slide_id column: {input_csv}")

    loaded = []
    for row in rows:
        slide_id = row["slide_id"]
        patch_path = patch_dir / f"{slide_id}.h5"
        if not patch_path.is_file():
            continue
        loaded.append(
            {
                "slide_id": slide_id,
                "patch_h5_path": str(patch_path),
                "size_bytes": patch_path.stat().st_size,
            }
        )
    if not loaded:
        raise SystemExit("No valid patch h5 files found for feature extraction shards.")
    return loaded


def assign_shards(rows: list[dict], num_shards: int) -> tuple[list[list[dict]], list[int]]:
    shards = [[] for _ in range(num_shards)]
    shard_sizes = [0 for _ in range(num_shards)]
    for row in sorted(rows, key=lambda item: item["size_bytes"], reverse=True):
        shard_idx = min(range(num_shards), key=lambda idx: shard_sizes[idx])
        row = dict(row)
        row["shard_id"] = shard_idx
        shards[shard_idx].append(row)
        shard_sizes[shard_idx] += row["size_bytes"]
    return shards, shard_sizes


def write_outputs(output_dir: Path, shards: list[list[dict]], shard_sizes: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for shard_idx, rows in enumerate(shards):
        shard_csv = output_dir / f"feature_shard_{shard_idx}.csv"
        with shard_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["slide_id", "patch_h5_path", "size_bytes", "shard_id"])
            writer.writeheader()
            writer.writerows(rows)

        summary.append(
            {
                "shard_id": shard_idx,
                "num_slides": len(rows),
                "total_size_bytes": shard_sizes[shard_idx],
                "csv_path": str(shard_csv),
            }
        )

    summary_path = output_dir / "feature_shard_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    for item in summary:
        total_gib = item["total_size_bytes"] / (1024 ** 3)
        print(
            f"shard {item['shard_id']}: {item['num_slides']} slides, "
            f"{total_gib:.2f} GiB -> {item['csv_path']}"
        )
    print(f"Wrote summary: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    rows = load_rows(args.input_csv, args.patch_dir)
    shards, shard_sizes = assign_shards(rows, args.num_shards)
    write_outputs(args.output_dir, shards, shard_sizes)


if __name__ == "__main__":
    main()
