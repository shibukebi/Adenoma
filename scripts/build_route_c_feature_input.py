#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Route C feature extraction input CSV from manifest_route_c.csv.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_csv = Path(args.manifest_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") in {"success_patho_r1", "success_fallback"} and int(row.get("coords_count", 0)) > 0:
                rows.append(row["slide_id"])

    rows = sorted(set(rows))
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id"])
        for slide_id in rows:
            writer.writerow([slide_id])

    print(f"output_csv={output_csv}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
