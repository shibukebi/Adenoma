#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_READY_CSV = "/data15/data15_5/yuexin2/adenoma/data/dsmil_ssl_others_2p5x_5x_uni_ready.csv"
DEFAULT_LOW_DIR = "/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_2p5x_uni"
DEFAULT_HIGH_DIR = "/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_5x_uni"
DEFAULT_OUT_DIR = "/data15/data15_5/yuexin2/MIST/datasets/adenoma_uni_2p5x_5x_cat"
DEFAULT_SPLIT_DIR = "/data15/data15_5/yuexin2/adenoma/data/dsmil_ssl_splits_2p5x_5x_uni"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MIST-style concatenated UNI features. Each output row is "
            "[high_mag_feature, low_mag_weight * matched_low_mag_feature]."
        )
    )
    parser.add_argument("--ready-csv", default=DEFAULT_READY_CSV)
    parser.add_argument("--low-feature-dir", default=DEFAULT_LOW_DIR, help="2.5x UNI feature directory")
    parser.add_argument("--high-feature-dir", default=DEFAULT_HIGH_DIR, help="5x UNI feature directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-name", default=None, help="Manifest name. Defaults to output directory name.")
    parser.add_argument("--low-weight", type=float, default=0.25)
    parser.add_argument("--float-format", default="%.6f")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument(
        "--no-split-manifests",
        action="store_true",
        help="Only write the all-slide dataset manifest, not train/val/test manifest folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-slide CSV files instead of reusing them.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "class"


def read_h5_feature_payload(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(path, "r") as handle:
        if "features" not in handle or "coords" not in handle:
            raise KeyError(f"{path} must contain 'features' and 'coords' datasets")
        features = handle["features"][:].astype(np.float32, copy=False)
        coords = handle["coords"][:].astype(np.int64, copy=False)
        extent = int(handle["coords"].attrs.get("physical_level_0_extent", 0))
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features in {path}, got {features.shape}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords shaped [N, 2] in {path}, got {coords.shape}")
    if len(features) != len(coords):
        raise ValueError(f"Feature/coord length mismatch in {path}: {len(features)} vs {len(coords)}")
    if extent <= 0:
        raise ValueError(f"Missing positive physical_level_0_extent in {path}")
    return features, coords, extent


def match_high_to_low(
    low_coords: np.ndarray,
    low_extent: int,
    high_coords: np.ndarray,
    high_extent: int,
) -> tuple[np.ndarray, dict[str, int]]:
    low_centers = low_coords + low_extent / 2.0
    high_points = high_coords + high_extent / 2.0
    parent_indices = np.empty(len(high_coords), dtype=np.int64)
    stats = {
        "unique_parent": 0,
        "ambiguous_parent_resolved_by_nearest_center": 0,
        "no_parent_resolved_by_nearest_center": 0,
    }

    for idx, point in enumerate(high_points):
        x, y = point
        contains = (
            (low_coords[:, 0] <= x)
            & (x < low_coords[:, 0] + low_extent)
            & (low_coords[:, 1] <= y)
            & (y < low_coords[:, 1] + low_extent)
        )
        candidates = np.flatnonzero(contains)
        if len(candidates) == 1:
            parent_indices[idx] = candidates[0]
            stats["unique_parent"] += 1
        elif len(candidates) > 1:
            deltas = low_centers[candidates] - point
            nearest = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
            parent_indices[idx] = candidates[nearest]
            stats["ambiguous_parent_resolved_by_nearest_center"] += 1
        else:
            deltas = low_centers - point
            parent_indices[idx] = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
            stats["no_parent_resolved_by_nearest_center"] += 1

    return parent_indices, stats


def write_manifest(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        writer.writerows(rows)


def read_split_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row and row[0].strip():
                ids.append(row[0].strip())
    return ids


def main() -> None:
    args = parse_args()
    ready_csv = Path(args.ready_csv)
    low_h5_dir = Path(args.low_feature_dir) / "h5_files"
    high_h5_dir = Path(args.high_feature_dir) / "h5_files"
    output_dir = Path(args.output_dir)
    dataset_name = args.dataset_name or output_dir.name
    feature_root = output_dir / "features"
    stats_dir = output_dir / "stats"
    feature_root.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    ready_df = pd.read_csv(ready_csv, dtype={"slide_id": str, "case_id": str})
    manifest_rows: list[tuple[str, int]] = []
    per_slide_stats: list[dict[str, object]] = []
    totals = {
        "slides": 0,
        "low_patches": 0,
        "high_patches": 0,
        "unique_parent": 0,
        "ambiguous_parent_resolved_by_nearest_center": 0,
        "no_parent_resolved_by_nearest_center": 0,
        "written": 0,
        "reused": 0,
    }

    for row_idx, row in ready_df.iterrows():
        slide_id = str(row["slide_id"])
        label_value = int(row["label_value"])
        label_name = safe_name(str(row.get("label_name", label_value)))
        class_dir = feature_root / f"class_{label_value}_{label_name}"
        output_csv = class_dir / f"{slide_id}.csv"
        manifest_rows.append((str(output_csv), label_value))

        low_path = low_h5_dir / f"{slide_id}.h5"
        high_path = high_h5_dir / f"{slide_id}.h5"
        low_features, low_coords, low_extent = read_h5_feature_payload(low_path)
        high_features, high_coords, high_extent = read_h5_feature_payload(high_path)

        parent_indices, match_stats = match_high_to_low(low_coords, low_extent, high_coords, high_extent)
        totals["slides"] += 1
        totals["low_patches"] += int(len(low_features))
        totals["high_patches"] += int(len(high_features))
        for key, value in match_stats.items():
            totals[key] += int(value)

        if output_csv.exists() and not args.overwrite:
            totals["reused"] += 1
        else:
            class_dir.mkdir(parents=True, exist_ok=True)
            concat_features = np.concatenate(
                [high_features, float(args.low_weight) * low_features[parent_indices]],
                axis=1,
            )
            pd.DataFrame(concat_features).to_csv(
                output_csv,
                index=False,
                float_format=args.float_format,
            )
            totals["written"] += 1

        per_slide_stats.append(
            {
                "slide_id": slide_id,
                "label_value": label_value,
                "label_name": str(row.get("label_name", "")),
                "low_patches": int(len(low_features)),
                "high_patches": int(len(high_features)),
                **match_stats,
                "output_csv": str(output_csv),
            }
        )

        if (row_idx + 1) % 50 == 0 or row_idx + 1 == len(ready_df):
            print(f"processed {row_idx + 1}/{len(ready_df)} slides")

    manifest_path = output_dir / f"{dataset_name}.csv"
    write_manifest(manifest_path, manifest_rows)

    stats_csv = stats_dir / "match_stats_by_slide.csv"
    pd.DataFrame(per_slide_stats).to_csv(stats_csv, index=False)
    summary = {
        "ready_csv": str(ready_csv),
        "low_feature_dir": str(args.low_feature_dir),
        "high_feature_dir": str(args.high_feature_dir),
        "output_dir": str(output_dir),
        "dataset_name": dataset_name,
        "low_weight": float(args.low_weight),
        "feature_dim": 2048,
        **totals,
    }
    summary_path = stats_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    split_outputs: dict[str, str] = {}
    if not args.no_split_manifests and args.split_dir:
        slide_to_manifest = {Path(path).stem: (path, label) for path, label in manifest_rows}
        split_dir = Path(args.split_dir)
        for split_name in ("train", "val", "test"):
            split_path = split_dir / f"flod-{args.fold}-{split_name}.csv"
            if not split_path.exists():
                continue
            split_ids = read_split_ids(split_path)
            split_rows = [slide_to_manifest[slide_id] for slide_id in split_ids if slide_id in slide_to_manifest]
            split_dataset_name = f"{dataset_name}_fold{args.fold}_{split_name}"
            split_manifest_path = output_dir.parent / split_dataset_name / f"{split_dataset_name}.csv"
            write_manifest(split_manifest_path, split_rows)
            split_outputs[split_name] = str(split_manifest_path)

    if split_outputs:
        split_summary_path = stats_dir / "split_manifests.json"
        split_summary_path.write_text(json.dumps(split_outputs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if split_outputs:
        print("split_manifests=" + json.dumps(split_outputs, ensure_ascii=False))
    print(f"manifest={manifest_path}")
    print(f"stats_csv={stats_csv}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
