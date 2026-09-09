#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Route C selected boxes on a thumbnail into a CLAM-compatible coords .h5 file."
    )
    parser.add_argument("--boxes-json", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=256)
    parser.add_argument("--patch-level", type=int, default=0)
    parser.add_argument("--min-box-size", type=int, default=32, help="Ignore tiny thumbnail boxes")
    return parser.parse_args()


def load_boxes_payload(boxes_json: Path) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    payload = json.loads(boxes_json.read_text(encoding="utf-8"))
    meta = payload["thumbnail_meta"]
    boxes = payload.get("boxes", [])
    return payload, meta, boxes


def boxes_to_coords(
    boxes: List[Dict[str, Any]],
    meta: Dict[str, Any],
    patch_size: int,
    step_size: int,
    min_box_size: int,
) -> np.ndarray:
    slide_w, slide_h = meta["slide_dimensions_level0"]
    thumb_w, thumb_h = meta["thumbnail_size"]
    scale_x = slide_w / thumb_w
    scale_y = slide_h / thumb_h

    coords = []
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue
        sx1 = max(0, int(round(x1 * scale_x)))
        sy1 = max(0, int(round(y1 * scale_y)))
        sx2 = min(slide_w, int(round(x2 * scale_x)))
        sy2 = min(slide_h, int(round(y2 * scale_y)))

        max_x = max(sx1, sx2 - patch_size + 1)
        max_y = max(sy1, sy2 - patch_size + 1)
        for y in range(sy1, max_y, step_size):
            for x in range(sx1, max_x, step_size):
                coords.append((x, y))

    return np.array(sorted(set(coords)), dtype=np.int32)


def write_coords_h5(
    output_h5: Path,
    coords: np.ndarray,
    meta: Dict[str, Any],
    patch_size: int,
    patch_level: int,
    source: str,
) -> int:
    if len(coords) == 0:
        return 0

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as handle:
        chunk_rows = max(1, min(len(coords), 1024))
        dset = handle.create_dataset(
            "coords",
            data=coords,
            maxshape=(None, 2),
            chunks=(chunk_rows, 2),
            dtype=np.int32,
        )
        dset.attrs["patch_level"] = patch_level
        dset.attrs["patch_size"] = patch_size
        dset.attrs["thumbnail_width"] = meta["thumbnail_size"][0]
        dset.attrs["thumbnail_height"] = meta["thumbnail_size"][1]
        dset.attrs["slide_width"] = meta["slide_dimensions_level0"][0]
        dset.attrs["slide_height"] = meta["slide_dimensions_level0"][1]
        dset.attrs["source"] = source
    return len(coords)


def convert_boxes_json_to_h5(
    boxes_json: Path,
    output_h5: Path,
    patch_size: int = 256,
    step_size: int = 256,
    patch_level: int = 0,
    min_box_size: int = 32,
) -> Tuple[int, Dict[str, Any]]:
    payload, meta, boxes = load_boxes_payload(boxes_json)
    coords = boxes_to_coords(boxes, meta, patch_size, step_size, min_box_size)
    coords_count = write_coords_h5(
        output_h5=output_h5,
        coords=coords,
        meta=meta,
        patch_size=patch_size,
        patch_level=patch_level,
        source=payload.get("mode_used", "route_c_patho_r1"),
    )
    return coords_count, payload


def main() -> None:
    args = parse_args()
    coords_count, _ = convert_boxes_json_to_h5(
        boxes_json=Path(args.boxes_json),
        output_h5=Path(args.output_h5),
        patch_size=args.patch_size,
        step_size=args.step_size,
        patch_level=args.patch_level,
        min_box_size=args.min_box_size,
    )
    print(f"output_h5={args.output_h5}")
    print(f"coords_count={coords_count}")
    if coords_count == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
