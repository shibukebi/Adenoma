#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LABEL_DICT = {"others": 0, "SSL": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fold-specific manifest for the official-style TransMIL route.")
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_id_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str, "label": str})
    label_map = ready_df.set_index("slide_id")["label_name"].map(LABEL_DICT).to_dict()
    rows = {}
    for split in ("train", "val", "test"):
        ids = read_id_list(Path(args.split_dir) / f"flod-{args.fold}-{split}.csv")
        rows[f"{split}"] = ids
        rows[f"{split}_label"] = [label_map[sid] for sid in ids]
    max_len = max(len(v) for v in rows.values())
    for key, values in rows.items():
        if len(values) < max_len:
            values.extend([""] * (max_len - len(values)))
    df = pd.DataFrame(rows)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
