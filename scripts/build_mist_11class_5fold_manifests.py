#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_ids(path: Path) -> set[str]:
    df = pd.read_csv(path, header=None)
    return set(df.iloc[:, 0].astype(str))


def build_feature_index(feature_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in feature_root.glob("features/**/*.csv"):
        index[path.stem] = path
    return index


def write_manifest(
    ready_df: pd.DataFrame,
    split_ids: set[str],
    feature_index: dict[str, Path],
    out_csv: Path,
) -> None:
    rows = []
    missing = []
    for _, row in ready_df[ready_df["slide_id"].astype(str).isin(split_ids)].iterrows():
        slide_id = str(row["slide_id"])
        feature_path = feature_index.get(slide_id)
        if feature_path is None:
            missing.append(slide_id)
            continue
        rows.append({"path": str(feature_path), "label": int(row["label_value"])})
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} MIST feature CSVs for {out_csv}: {missing[:10]}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MIST path,label manifests for adenoma 11-class 5-fold runs.")
    parser.add_argument(
        "--ready-csv",
        default="/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/manifests_11class_fold5/joint_hp_yx_adenoma_11class_ready.csv",
    )
    parser.add_argument(
        "--split-dir",
        default="/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/manifests_11class_fold5/splits",
    )
    parser.add_argument(
        "--mist-root",
        default="/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST",
    )
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str})
    split_dir = Path(args.split_dir)
    mist_root = Path(args.mist_root)
    output_root = Path(args.output_root) if args.output_root else mist_root / "manifests_11class_5fold"

    specs = [
        ("2p5x_5x", mist_root / "adenoma_uni_hp_yx_2p5x_5x_low4cat"),
        ("5x_10x", mist_root / "adenoma_uni_hp_yx_5x_10x_low4cat"),
    ]
    for combo, feature_root in specs:
        feature_index = build_feature_index(feature_root)
        if not feature_index:
            raise FileNotFoundError(f"No feature CSVs found under {feature_root / 'features'}")
        for fold in range(5):
            for split in ("train", "val", "test"):
                ids = read_ids(split_dir / f"flod-{fold}-{split}.csv")
                out_csv = output_root / combo / f"fold-{fold}" / split / f"mist_11class_{combo}_fold{fold}_{split}.csv"
                write_manifest(ready_df, ids, feature_index, out_csv)
    print(f"Wrote MIST manifests to {output_root}")


if __name__ == "__main__":
    main()
