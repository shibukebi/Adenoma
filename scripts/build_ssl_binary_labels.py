#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from clam_experiment_utils import HIERARCHICAL_LABEL_COLUMNS, build_hierarchical_label_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the hierarchical adenoma master label table used by stage-1 SSL and stage-2 dysplasia tasks."
    )
    parser.add_argument(
        "--input-csv",
        default="/data15/data15_5/yuexin2/adenoma/data/adenoma_yx_labels.csv",
        help="Input label csv containing slide_id,type,grade",
    )
    parser.add_argument(
        "--output-csv",
        default="/data15/data15_5/yuexin2/adenoma/data/adenoma_yx_ssl_others_labels.csv",
        help="Output hierarchical label csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            rows.append(
                build_hierarchical_label_row(
                    slide_id=raw_row["slide_id"],
                    lesion_type=raw_row["type"],
                    grade=raw_row["grade"],
                )
            )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HIERARCHICAL_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    ssl_counts = Counter(row["ssl_label_name"] for row in rows)
    dysplasia_counts = Counter(row["dysplasia_label_name"] for row in rows if int(row["is_ssl_for_stage2"]) == 1)
    final_counts = Counter(row["final_label_name"] for row in rows)
    print(f"rows={len(rows)}")
    print(f"ssl_counts={dict(ssl_counts)}")
    print(f"dysplasia_counts={dict(dysplasia_counts)}")
    print(f"final_counts={dict(final_counts)}")
    print(f"output={output_csv}")


if __name__ == "__main__":
    main()
