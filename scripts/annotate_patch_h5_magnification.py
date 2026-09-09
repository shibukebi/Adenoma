#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate CLAM patch h5 files with physical magnification metadata."
    )
    parser.add_argument("--patch-h5-dir", required=True)
    parser.add_argument("--base-magnification", type=float, default=40.0)
    parser.add_argument("--target-magnification", type=float, required=True)
    parser.add_argument(
        "--physical-extent-patch-size",
        type=int,
        default=0,
        help="Patch size used in the magnification formula. Defaults to h5 coords patch_size.",
    )
    return parser.parse_args()


def compute_extent(base_magnification: float, target_magnification: float, patch_size: int) -> int:
    if base_magnification <= 0:
        raise ValueError("--base-magnification must be positive.")
    if target_magnification <= 0:
        raise ValueError("--target-magnification must be positive.")
    if patch_size <= 0:
        raise ValueError("patch size for physical extent must be positive.")
    return int(round(float(patch_size) * float(base_magnification) / float(target_magnification)))


def annotate_file(
    h5_path: Path,
    *,
    base_magnification: float,
    target_magnification: float,
    physical_extent_patch_size: int,
) -> None:
    with h5py.File(h5_path, "a") as handle:
        if "coords" not in handle:
            raise KeyError(f"{h5_path} does not contain a coords dataset.")
        coords = handle["coords"]
        h5_patch_size = int(coords.attrs.get("patch_size", 0))
        extent_patch_size = int(physical_extent_patch_size or h5_patch_size)
        physical_extent = compute_extent(base_magnification, target_magnification, extent_patch_size)
        scale_to_level0 = float(base_magnification) / float(target_magnification)

        attrs = {
            "base_magnification": float(base_magnification),
            "target_magnification": float(target_magnification),
            "scale_to_level0": scale_to_level0,
            "physical_extent_patch_size": extent_patch_size,
            "physical_level_0_extent": physical_extent,
            "coordinate_space": "level0",
            "level_index_for_coordinate_mapping": "forbidden",
            "source_level_downsample_for_coordinate_mapping": "forbidden",
            "magnification_rule": "physical_level0_extent = physical_extent_patch_size * base_magnification / target_magnification",
        }
        for key, value in attrs.items():
            coords.attrs[key] = value
            handle.attrs[key] = value


def main() -> None:
    args = parse_args()
    patch_h5_dir = Path(args.patch_h5_dir)
    h5_paths = sorted(patch_h5_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found under {patch_h5_dir}")

    for h5_path in h5_paths:
        annotate_file(
            h5_path,
            base_magnification=args.base_magnification,
            target_magnification=args.target_magnification,
            physical_extent_patch_size=args.physical_extent_patch_size,
        )
    print(f"Annotated {len(h5_paths)} h5 files in {patch_h5_dir}")


if __name__ == "__main__":
    main()
