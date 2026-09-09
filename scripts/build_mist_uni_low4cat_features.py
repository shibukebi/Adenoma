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
DEFAULT_OUT_DIR = "/data15/data15_5/yuexin2/MIST/datasets/adenoma_uni_2p5x_5x_low4cat"
DEFAULT_SPLIT_DIR = "/data15/data15_5/yuexin2/adenoma/data/dsmil_ssl_splits_2p5x_5x_uni"


QUADRANTS = (
    ("tl", 0, 0),
    ("tr", 1, 0),
    ("bl", 0, 1),
    ("br", 1, 1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build low-patch-centered MIST features. Each 2.5x row is "
            "[2.5x_feature, 5x_tl, 5x_tr, 5x_bl, 5x_br]."
        )
    )
    parser.add_argument("--ready-csv", default=DEFAULT_READY_CSV)
    parser.add_argument("--low-feature-dir", default=DEFAULT_LOW_DIR, help="2.5x UNI feature directory")
    parser.add_argument("--high-feature-dir", default=DEFAULT_HIGH_DIR, help="5x UNI feature directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-name", default=None, help="Manifest name. Defaults to output directory name.")
    parser.add_argument("--float-format", default="%.6f")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument(
        "--low-extent",
        type=int,
        default=None,
        help="Optional level-0 physical extent for each 2.5x patch when h5 attrs are missing.",
    )
    parser.add_argument(
        "--high-extent",
        type=int,
        default=None,
        help="Optional level-0 physical extent for each 5x patch when h5 attrs are missing.",
    )
    parser.add_argument(
        "--zero-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill missing 5x quadrants with zeros. Enabled by default.",
    )
    parser.add_argument(
        "--no-split-manifests",
        action="store_true",
        help="Only write the all-slide dataset manifest, not train/val/test manifest folders.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "class"


def read_h5_feature_payload(path: Path, fallback_extent: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
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
    if extent <= 0 and fallback_extent is not None:
        extent = int(fallback_extent)
    if extent <= 0:
        raise ValueError(f"Missing positive physical_level_0_extent in {path}")
    return features, coords, extent


def assign_quadrant_children(
    low_coord: np.ndarray,
    low_extent: int,
    high_coords: np.ndarray,
    high_extent: int,
) -> tuple[list[int | None], dict[str, int]]:
    low_x, low_y = low_coord.astype(np.float64)
    half = low_extent / 2.0
    high_centers = high_coords.astype(np.float64) + high_extent / 2.0
    inside_low = (
        (low_x <= high_centers[:, 0])
        & (high_centers[:, 0] < low_x + low_extent)
        & (low_y <= high_centers[:, 1])
        & (high_centers[:, 1] < low_y + low_extent)
    )

    child_indices: list[int | None] = []
    stats = {
        "quadrants_filled": 0,
        "quadrants_missing": 0,
        "quadrants_multi_resolved_by_nearest_center": 0,
        "candidate_high_patches": int(inside_low.sum()),
    }
    for _name, qx, qy in QUADRANTS:
        q_x0 = low_x + qx * half
        q_y0 = low_y + qy * half
        q_x1 = q_x0 + half
        q_y1 = q_y0 + half
        in_quadrant = (
            inside_low
            & (q_x0 <= high_centers[:, 0])
            & (high_centers[:, 0] < q_x1)
            & (q_y0 <= high_centers[:, 1])
            & (high_centers[:, 1] < q_y1)
        )
        candidates = np.flatnonzero(in_quadrant)
        if len(candidates) == 0:
            child_indices.append(None)
            stats["quadrants_missing"] += 1
            continue

        if len(candidates) > 1:
            target_center = np.array([q_x0 + half / 2.0, q_y0 + half / 2.0])
            deltas = high_centers[candidates] - target_center
            nearest = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
            child_indices.append(int(candidates[nearest]))
            stats["quadrants_multi_resolved_by_nearest_center"] += 1
        else:
            child_indices.append(int(candidates[0]))
        stats["quadrants_filled"] += 1

    return child_indices, stats


def build_low4_features(
    low_features: np.ndarray,
    low_coords: np.ndarray,
    low_extent: int,
    high_features: np.ndarray,
    high_coords: np.ndarray,
    high_extent: int,
    zero_missing: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    feature_dim = low_features.shape[1]
    zero = np.zeros(feature_dim, dtype=np.float32)
    rows: list[np.ndarray] = []
    totals = {
        "low_rows": int(len(low_features)),
        "quadrants_total": int(len(low_features) * len(QUADRANTS)),
        "quadrants_filled": 0,
        "quadrants_missing": 0,
        "quadrants_multi_resolved_by_nearest_center": 0,
        "low_patches_with_all_four": 0,
        "low_patches_with_any_missing": 0,
        "candidate_high_patch_assignments": 0,
    }

    for low_feature, low_coord in zip(low_features, low_coords):
        child_indices, stats = assign_quadrant_children(low_coord, low_extent, high_coords, high_extent)
        high_parts: list[np.ndarray] = []
        for child_idx in child_indices:
            if child_idx is None:
                if not zero_missing:
                    raise ValueError("Missing 5x quadrant and --no-zero-missing was requested")
                high_parts.append(zero)
            else:
                high_parts.append(high_features[child_idx])

        totals["quadrants_filled"] += stats["quadrants_filled"]
        totals["quadrants_missing"] += stats["quadrants_missing"]
        totals["quadrants_multi_resolved_by_nearest_center"] += stats[
            "quadrants_multi_resolved_by_nearest_center"
        ]
        totals["candidate_high_patch_assignments"] += stats["candidate_high_patches"]
        if stats["quadrants_missing"] == 0:
            totals["low_patches_with_all_four"] += 1
        else:
            totals["low_patches_with_any_missing"] += 1

        rows.append(np.concatenate([low_feature, *high_parts], axis=0))

    return np.stack(rows, axis=0).astype(np.float32, copy=False), totals


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
        "output_rows": 0,
        "quadrants_total": 0,
        "quadrants_filled": 0,
        "quadrants_missing": 0,
        "quadrants_multi_resolved_by_nearest_center": 0,
        "low_patches_with_all_four": 0,
        "low_patches_with_any_missing": 0,
        "candidate_high_patch_assignments": 0,
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
        low_features, low_coords, low_extent = read_h5_feature_payload(low_path, args.low_extent)
        high_features, high_coords, high_extent = read_h5_feature_payload(high_path, args.high_extent)

        if output_csv.exists() and not args.overwrite:
            feature_rows = len(low_features)
            slide_stats = {
                "low_rows": int(feature_rows),
                "quadrants_total": int(feature_rows * len(QUADRANTS)),
                "quadrants_filled": 0,
                "quadrants_missing": 0,
                "quadrants_multi_resolved_by_nearest_center": 0,
                "low_patches_with_all_four": 0,
                "low_patches_with_any_missing": 0,
                "candidate_high_patch_assignments": 0,
            }
            totals["reused"] += 1
        else:
            class_dir.mkdir(parents=True, exist_ok=True)
            concat_features, slide_stats = build_low4_features(
                low_features=low_features,
                low_coords=low_coords,
                low_extent=low_extent,
                high_features=high_features,
                high_coords=high_coords,
                high_extent=high_extent,
                zero_missing=args.zero_missing,
            )
            pd.DataFrame(concat_features).to_csv(
                output_csv,
                index=False,
                float_format=args.float_format,
            )
            totals["written"] += 1

        totals["slides"] += 1
        totals["low_patches"] += int(len(low_features))
        totals["high_patches"] += int(len(high_features))
        totals["output_rows"] += int(slide_stats["low_rows"])
        for key in (
            "quadrants_total",
            "quadrants_filled",
            "quadrants_missing",
            "quadrants_multi_resolved_by_nearest_center",
            "low_patches_with_all_four",
            "low_patches_with_any_missing",
            "candidate_high_patch_assignments",
        ):
            totals[key] += int(slide_stats[key])

        per_slide_stats.append(
            {
                "slide_id": slide_id,
                "label_value": label_value,
                "label_name": str(row.get("label_name", "")),
                "low_patches": int(len(low_features)),
                "high_patches": int(len(high_features)),
                **slide_stats,
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
        "row_rule": "[2.5x_UNI, 5x_tl, 5x_tr, 5x_bl, 5x_br]",
        "feature_dim": 5120,
        "zero_missing": bool(args.zero_missing),
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
