#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LABEL_STYLES = {
    "epithelial_neoplasia_suspicious": {
        "color": (220, 38, 38),
        "abbr": "EPI",
        "name": "epithelial_neoplasia_suspicious",
    },
    "mucus_rich_or_pale_context": {
        "color": (0, 166, 166),
        "abbr": "MUC",
        "name": "mucus_rich_or_pale_context",
    },
    "inflammatory_or_stromal_context": {
        "color": (126, 58, 166),
        "abbr": "INF",
        "name": "inflammatory_or_stromal_context",
    },
    "reviewable_normal_mucosa": {
        "color": (34, 150, 84),
        "abbr": "NOR",
        "name": "reviewable_normal_mucosa",
    },
    "background_or_artifact": {
        "color": (82, 96, 122),
        "abbr": "BG",
        "name": "background_or_artifact",
    },
    "uncertain_reviewable_mucosa": {
        "color": (245, 158, 11),
        "abbr": "UNC",
        "name": "uncertain_reviewable_mucosa",
    },
    "unassigned_or_other": {
        "color": (20, 20, 20),
        "abbr": "OTH",
        "name": "unassigned_or_other",
    },
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _patch_key(patch_id):
    if not isinstance(patch_id, (list, tuple)) or len(patch_id) != 2:
        return None
    return "{0},{1}".format(int(patch_id[0]), int(patch_id[1]))


def _font(size=14):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    return draw.textsize(text, font=font)


def _draw_legend(styles, output_path):
    font = _font(16)
    title_font = _font(18)
    row_h = 30
    width = 620
    height = 42 + row_h * len(styles)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "Trace region_semantic color legend", fill=(20, 20, 20), font=title_font)
    y = 42
    for label, style in styles.items():
        color = tuple(style["color"])
        draw.rounded_rectangle((16, y + 5, 42, y + 25), radius=3, fill=color, outline=(255, 255, 255), width=1)
        draw.text((54, y + 6), "{0}  {1}".format(style["abbr"], label), fill=(20, 20, 20), font=font)
        y += row_h
    image.save(output_path)


def render_case(trace_json_path, output_dir, alpha=96, draw_text=True):
    trace_json_path = Path(trace_json_path)
    trace_dir = trace_json_path.parent
    grid_input_dir = trace_dir / "grid_input"
    grid_jsons = sorted(grid_input_dir.glob("*_grid.json"))
    grid_images = sorted(grid_input_dir.glob("*_grid.jpg"))
    if not grid_jsons or not grid_images:
        raise FileNotFoundError("Missing grid_input *_grid.json or *_grid.jpg under {0}".format(trace_dir))

    grid_json_path = grid_jsons[0]
    grid_image_path = grid_images[0]
    trace_payload = _read_json(trace_json_path)
    grid_payload = _read_json(grid_json_path)
    assignments = trace_payload.get("patch_assignments", {}).get("patches", [])
    label_by_patch = {
        _patch_key(item.get("patch_id")): str(item.get("region_semantic") or "unassigned_or_other")
        for item in assignments
        if _patch_key(item.get("patch_id")) is not None
    }

    base = Image.open(grid_image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_font = _font(13)
    counts = {label: 0 for label in LABEL_STYLES}
    missing = []

    for cell in grid_payload.get("grid_cells", []):
        if not cell.get("is_selected"):
            continue
        patch_id = cell.get("patch_id", [cell.get("row_id"), cell.get("col_id")])
        key = _patch_key(patch_id)
        label = label_by_patch.get(key, "unassigned_or_other")
        if label not in LABEL_STYLES:
            label = "unassigned_or_other"
        if key not in label_by_patch:
            missing.append(patch_id)
        counts[label] = counts.get(label, 0) + 1
        style = LABEL_STYLES[label]
        color = tuple(style["color"])
        x1 = int(round(float(cell.get("thumbnail_top_left_x", 0) or 0)))
        y1 = int(round(float(cell.get("thumbnail_top_left_y", 0) or 0)))
        x2 = int(round(x1 + float(cell.get("thumbnail_width", 0) or 0)))
        y2 = int(round(y1 + float(cell.get("thumbnail_height", 0) or 0)))
        x1 = max(0, min(base.size[0] - 1, x1))
        y1 = max(0, min(base.size[1] - 1, y1))
        x2 = max(x1 + 1, min(base.size[0], x2))
        y2 = max(y1 + 1, min(base.size[1], y2))
        draw.rectangle((x1, y1, x2, y2), fill=color + (int(alpha),), outline=color + (255,), width=3)
        if draw_text:
            text = style["abbr"]
            tw, th = _text_size(draw, text, text_font)
            tx = x1 + 4
            ty = y1 + 4
            draw.rounded_rectangle((tx - 2, ty - 1, tx + tw + 4, ty + th + 3), radius=2, fill=(255, 255, 255, 210))
            draw.text((tx, ty), text, fill=color + (255,), font=text_font)

    rendered = Image.alpha_composite(base, overlay).convert("RGB")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "{0}_trace_region_overlay.jpg".format(grid_image_path.stem.replace("_grid", ""))
    rendered.save(output_path, quality=95)
    return {
        "case_id": trace_json_path.parents[1].name,
        "trace_json": str(trace_json_path),
        "grid_image": str(grid_image_path),
        "grid_metadata": str(grid_json_path),
        "overlay_image": str(output_path),
        "counts": {key: value for key, value in counts.items() if value},
        "missing_patch_ids": missing,
    }


def render_run(trace_run_dir, output_dir, alpha=96, draw_text=True):
    trace_run_dir = Path(trace_run_dir)
    output_dir = Path(output_dir)
    trace_jsons = sorted(trace_run_dir.glob("*/trace/trace_clusters.json"))
    if not trace_jsons:
        raise FileNotFoundError("No */trace/trace_clusters.json found under {0}".format(trace_run_dir))
    cases = [render_case(path, output_dir, alpha=alpha, draw_text=draw_text) for path in trace_jsons]
    legend_path = output_dir / "trace_region_overlay_legend.png"
    _draw_legend(LABEL_STYLES, legend_path)
    manifest = {
        "trace_run_dir": str(trace_run_dir),
        "output_dir": str(output_dir),
        "legend_image": str(legend_path),
        "styles": LABEL_STYLES,
        "cases": cases,
    }
    manifest_path = output_dir / "trace_region_overlay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Render region_semantic color overlays on Trace grid thumbnails.")
    parser.add_argument("--trace-run-dir", required=True, help="Run directory containing */trace/trace_clusters.json.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=int, default=96, help="Fill alpha from 0 to 255.")
    parser.add_argument("--no-text", action="store_true", help="Do not draw label abbreviations inside cells.")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = render_run(
        args.trace_run_dir,
        args.output_dir,
        alpha=max(0, min(255, int(args.alpha))),
        draw_text=not args.no_text,
    )
    print("manifest={0}".format(manifest_path))


if __name__ == "__main__":
    main()
