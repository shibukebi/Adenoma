#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a feature input CSV from currently available patch h5 files."
    )
    parser.add_argument("--patch-dir", required=True, help="Directory containing *.h5 patch coordinate files")
    parser.add_argument("--output-csv", required=True, help="Where to write slide_id CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_dir = Path(args.patch_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    slide_ids = sorted(path.stem for path in patch_dir.glob("*.h5"))
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id"])
        for slide_id in slide_ids:
            writer.writerow([slide_id])

    print(f"output_csv={output_csv}")
    print(f"rows={len(slide_ids)}")


if __name__ == "__main__":
    main()
