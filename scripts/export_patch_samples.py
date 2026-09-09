#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import openslide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a subset of CLAM patch coordinates as PNG images."
    )
    parser.add_argument("--slide-path", required=True, help="Path to the source WSI (.svs)")
    parser.add_argument("--coords-h5", required=True, help="Path to CLAM patch coordinate .h5")
    parser.add_argument("--output-dir", required=True, help="Directory to save exported PNG patches")
    parser.add_argument(
        "--max-patches",
        type=int,
        default=64,
        help="Maximum number of patches to export (default: 64)",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["uniform", "head"],
        default="uniform",
        help="How to choose patches from the coordinate list",
    )
    return parser.parse_args()


def choose_indices(total: int, max_patches: int, sample_mode: str) -> np.ndarray:
    count = min(total, max_patches)
    if count <= 0:
        return np.array([], dtype=np.int64)
    if sample_mode == "head" or count == total:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, total - 1, num=count, dtype=np.int64)


def main() -> None:
    args = parse_args()
    slide_path = Path(args.slide_path)
    coords_h5 = Path(args.coords_h5)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(coords_h5, "r") as handle:
        coords = handle["coords"][:]
        patch_level = int(handle["coords"].attrs["patch_level"])
        patch_size = int(handle["coords"].attrs["patch_size"])

    selected = choose_indices(len(coords), args.max_patches, args.sample_mode)
    if len(selected) == 0:
        raise SystemExit("No coordinates available in the provided .h5 file.")

    slide = openslide.open_slide(str(slide_path))
    manifest_path = output_dir / "patch_manifest.csv"

    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "slide_id",
                "coord_index",
                "x",
                "y",
                "patch_level",
                "patch_size",
                "image_path",
            ]
        )

        for coord_index in selected:
            x, y = coords[coord_index]
            image = slide.read_region((int(x), int(y)), patch_level, (patch_size, patch_size)).convert("RGB")
            image_name = f"{slide_path.stem}_idx{coord_index:05d}_x{x}_y{y}.png"
            image_path = output_dir / image_name
            image.save(image_path)
            writer.writerow(
                [slide_path.stem, int(coord_index), int(x), int(y), patch_level, patch_size, str(image_path)]
            )

    slide.close()
    print(f"Exported {len(selected)} patches to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
