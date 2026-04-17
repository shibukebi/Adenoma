#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import openslide
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Export navigation crops from a slide using level-0 center coordinates.")
    parser.add_argument("--slide-path", required=True)
    parser.add_argument("--steps-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(args.steps_json).read_text(encoding="utf-8"))
    slide = openslide.open_slide(args.slide_path)

    crops = []
    for step in payload.get("steps", []):
        region_size = int(step.get("metadata", {}).get("region_size_level0", args.output_size))
        half = int(region_size // 2)
        x = int(step["x"])
        y = int(step["y"])
        top_left = (x - half, y - half)
        region = slide.read_region(top_left, int(step.get("level", 0)), (region_size, region_size)).convert("RGB")
        if region.size != (args.output_size, args.output_size):
            region = region.resize((args.output_size, args.output_size), Image.BILINEAR)
        image_name = "{0}_mag{1:.1f}_{2}.png".format(step["step_id"], float(step["m"]), region_size)
        image_path = output_dir / image_name
        region.save(image_path)
        crops.append(
            {
                "step_id": step["step_id"],
                "image_path": str(image_path),
                "x": x,
                "y": y,
                "m": float(step["m"]),
                "o": step.get("o"),
                "region_size_level0": region_size,
                "metadata": step.get("metadata", {}),
            }
        )

    slide.close()
    Path(args.manifest_json).write_text(json.dumps({"crops": crops}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
