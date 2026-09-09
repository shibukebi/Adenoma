#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CLAM .pt feature files into MAG-GLTrans .npy feature dictionaries."
    )
    parser.add_argument("--pt-dir", required=True, help="Directory containing CLAM pt_files")
    parser.add_argument("--coords-dir", required=True, help="Directory containing CLAM patch .h5 coordinate files")
    parser.add_argument("--output-dir", required=True, help="Directory to save MAG-GLTrans .npy files")
    return parser.parse_args()


def load_coords(coords_path: Path) -> np.ndarray:
    with h5py.File(coords_path, "r") as handle:
        coords = handle["coords"][:]
    return coords.astype(np.int64)


def main() -> None:
    args = parse_args()
    pt_dir = Path(args.pt_dir)
    coords_dir = Path(args.coords_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for pt_path in sorted(pt_dir.glob("*.pt")):
        slide_id = pt_path.stem
        coords_path = coords_dir / f"{slide_id}.h5"
        if not coords_path.exists():
            print(f"skip {slide_id}: missing coords file {coords_path}")
            skipped += 1
            continue

        features = torch.load(pt_path, map_location="cpu")
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        coords = load_coords(coords_path)

        if len(features) != len(coords):
            print(
                f"skip {slide_id}: feature/coord length mismatch "
                f"({len(features)} vs {len(coords)})"
            )
            skipped += 1
            continue

        out_path = output_dir / f"{slide_id}.npy"
        torch.save({"feature": features, "index": coords}, str(out_path))
        converted += 1

    print(f"converted={converted}")
    print(f"skipped={skipped}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
