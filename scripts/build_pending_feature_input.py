#!/usr/bin/env python3
"""
从已完成的 patch h5 中构建待提特征的 slide_id 清单，并排除已存在的 pt 特征。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pending feature extraction CSV.")
    parser.add_argument("--patch-dir", required=True, help="Directory containing patch h5 files.")
    parser.add_argument("--output-csv", required=True, help="Output CSV with slide_id column.")
    parser.add_argument(
        "--pt-dir",
        default=None,
        help="Optional pt feature directory; existing pt files will be excluded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_dir = Path(args.patch_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    patch_ids = {path.stem for path in patch_dir.glob("*.h5")}
    done_ids = set()
    if args.pt_dir:
        pt_dir = Path(args.pt_dir)
        if pt_dir.exists():
            done_ids = {path.stem for path in pt_dir.glob("*.pt")}

    pending_ids = sorted(patch_ids - done_ids)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id"])
        for slide_id in pending_ids:
            writer.writerow([slide_id])

    print(f"patch_count={len(patch_ids)}")
    print(f"done_pt_count={len(done_ids)}")
    print(f"pending_count={len(pending_ids)}")
    print(f"output_csv={output_csv}")


if __name__ == "__main__":
    main()
