#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    import openslide
except Exception:
    openslide = None


def parse_args():
    parser = argparse.ArgumentParser(description="Export navigation crops from a slide using level-0 center coordinates.")
    parser.add_argument("--slide-path", required=True)
    parser.add_argument("--steps-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-size", type=int, default=256)
    return parser.parse_args()


class SlideReader(object):
    def __init__(self, slide_path):
        self.slide_path = str(slide_path)
        self.slide = None
        self.image = None
        self.mode = "pil"
        if openslide is not None:
            try:
                self.slide = openslide.open_slide(self.slide_path)
                self.mode = "openslide"
                return
            except Exception:
                self.slide = None
        self.image = Image.open(self.slide_path)

    def read_region(self, top_left, level, size):
        if self.slide is not None:
            return self.slide.read_region(top_left, int(level), size).convert("RGB")
        x, y = top_left
        width, height = size
        source = self.image.convert("RGB")
        crop = Image.new("RGB", size, (255, 255, 255))
        source_box = (
            max(0, x),
            max(0, y),
            min(source.width, x + width),
            min(source.height, y + height),
        )
        if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
            paste_at = (max(0, -x), max(0, -y))
            crop.paste(source.crop(source_box), paste_at)
        return crop

    def close(self):
        if self.slide is not None:
            self.slide.close()
        if self.image is not None:
            self.image.close()


def clean_rgb(image):
    rgb = Image.new("RGB", image.size, (255, 255, 255))
    rgb.paste(image.convert("RGB"))
    return rgb


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(args.steps_json).read_text(encoding="utf-8"))
    slide = SlideReader(args.slide_path)

    crops = []
    for step in payload.get("steps", []):
        metadata = step.get("metadata", {})
        region_size = int(step.get("region_size_level0", step.get("metadata", {}).get("region_size_level0", args.output_size)))
        half = int(region_size // 2)
        x = int(step["x"])
        y = int(step["y"])
        top_left = (x - half, y - half)
        region = clean_rgb(slide.read_region(top_left, int(step.get("level", 0)), (region_size, region_size)))
        if region.size != (args.output_size, args.output_size):
            region = region.resize((args.output_size, args.output_size), Image.BILINEAR)
        image_name = "{0}_mag{1:.1f}_{2}.png".format(step["step_id"], float(step["m"]), region_size)
        image_path = output_dir / image_name
        region.save(image_path, format="PNG")
        with Image.open(image_path) as check:
            check.verify()
        level0_bbox = [x - half, y - half, x + half, y + half]
        crops.append(
            {
                "step_id": step["step_id"],
                "image_path": str(image_path),
                "source_slide_path": str(args.slide_path),
                "x": x,
                "y": y,
                "m": float(step["m"]),
                "need_to_see": step.get("need_to_see"),
                "region_size_level0": region_size,
                "level0_bbox": level0_bbox,
                "cell_id": metadata.get("cell_id"),
                "patch_id": metadata.get("patch_id"),
                "intra_cell_target_index": metadata.get("intra_cell_target_index"),
                "coordinate_source": metadata.get("coordinate_source"),
                "metadata": metadata,
            }
        )

    slide.close()
    Path(args.manifest_json).write_text(json.dumps({"crops": crops}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
