#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAM_ROOT = PROJECT_ROOT / "CLAM"
sys.path.insert(0, str(CLAM_ROOT))

extra_site_packages = os.environ.get("PYISYNTAX_SITE_PACKAGES", "").strip()
if extra_site_packages:
    for site_path in extra_site_packages.split(":"):
        if site_path and site_path not in sys.path:
            sys.path.append(site_path)

from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stain-filtered binary Tissue Masks and CLAM-compatible segmentation pkl files."
    )
    parser.add_argument("--source-dir", required=True, help="Directory containing WSI files.")
    parser.add_argument("--output-dir", required=True, help="Output root for masks, debug images, and segmentations.")
    parser.add_argument("--slide-ext", default=".isyntax")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many slides; 0 means all.")
    parser.add_argument("--manifest-csv", default=None, help="Optional CSV with slide_id/slide_path/path/name column.")
    parser.add_argument("--target-downsample", type=float, default=64.0)
    parser.add_argument("--color-space", choices=["hsv", "lab"], default="hsv")
    parser.add_argument("--threshold", choices=["otsu", "adaptive"], default="otsu")
    parser.add_argument("--adaptive-block-size", type=int, default=51)
    parser.add_argument("--adaptive-c", type=float, default=-5.0)
    parser.add_argument("--min-tissue-area", type=float, default=5000.0, help="Minimum contour area at mask level.")
    parser.add_argument("--min-hole-area", type=float, default=500.0, help="Minimum hole area at mask level.")
    parser.add_argument("--max-holes", type=int, default=32)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--close-kernel", type=int, default=9)
    parser.add_argument("--dilate-kernel", type=int, default=0)
    parser.add_argument(
        "--disable-he-filter",
        action="store_true",
        default=False,
        help="Disable the H&E hue/value stain-artifact suppression filter.",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def read_manifest(path: Path, slide_ext: str) -> list[Path]:
    rows: list[Path] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = (
                row.get("slide_path")
                or row.get("path")
                or row.get("full_path")
                or row.get("slide_id")
                or row.get("name")
            )
            if not value:
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = candidate.with_suffix(slide_ext) if candidate.suffix == "" else candidate
            rows.append(candidate)
    return rows


def collect_slides(source_dir: Path, slide_ext: str, manifest_csv: str | None, limit: int) -> list[Path]:
    if manifest_csv:
        candidates = read_manifest(Path(manifest_csv), slide_ext)
        slides = []
        for candidate in candidates:
            path = candidate if candidate.is_absolute() else source_dir / candidate.name
            if path.suffix == "":
                path = path.with_suffix(slide_ext)
            if path.exists():
                slides.append(path)
    else:
        slides = sorted(source_dir.glob(f"*{slide_ext}"))
    if limit > 0:
        slides = slides[:limit]
    return slides


def choose_level(wsi, target_downsample: float) -> int:
    if hasattr(wsi, "get_best_level_for_downsample"):
        return int(wsi.get_best_level_for_downsample(target_downsample))
    downsamples = [float(value) for value in getattr(wsi, "level_downsamples", [1.0])]
    return int(np.argmin([abs(value - target_downsample) for value in downsamples]))


def ensure_odd_block_size(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def build_initial_mask(
    rgb: np.ndarray,
    color_space: str,
    threshold: str,
    block_size: int,
    c_value: float,
    he_filter: bool,
) -> np.ndarray:
    if color_space == "hsv":
        converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        signal = converted[:, :, 1]
    else:
        converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        signal = 255 - converted[:, :, 0]

    signal = cv2.GaussianBlur(signal, (5, 5), 0)
    if threshold == "otsu":
        _, mask = cv2.threshold(signal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        block_size = ensure_odd_block_size(block_size)
        mask = cv2.adaptiveThreshold(
            signal,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value,
        )

    # Bright, low-saturation background suppression keeps pen/stain specks from becoming tissue.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    background = (value > 235) & (saturation < 35)
    mask[background] = 0
    if he_filter:
        hue = hsv[:, :, 0]
        # H&E tissue is usually pink/purple in HSV. This suppresses green pen marks,
        # scanner borders, black labels, and strongly saturated non-H&E artifacts.
        he_like = ((hue >= 105) | (hue <= 25)) & (saturation >= 10) & (value >= 45)
        mask[~he_like] = 0
    return mask


def morph_filter(mask: np.ndarray, open_kernel: int, close_kernel: int, dilate_kernel: int) -> np.ndarray:
    filtered = mask.copy()
    if open_kernel > 0:
        kernel = np.ones((open_kernel, open_kernel), np.uint8)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    if close_kernel > 0:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    if dilate_kernel > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        filtered = cv2.dilate(filtered, kernel, iterations=1)
    return filtered


def contours_from_mask(
    mask: np.ndarray,
    min_tissue_area: float,
    min_hole_area: float,
    max_holes: int,
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], []
    hierarchy = np.squeeze(hierarchy, axis=0)

    tissue_contours: list[np.ndarray] = []
    hole_contours: list[list[np.ndarray]] = []
    for idx, contour in enumerate(contours):
        parent = int(hierarchy[idx][3])
        if parent != -1:
            continue
        area = cv2.contourArea(contour)
        if area < min_tissue_area:
            continue

        holes = []
        child = int(hierarchy[idx][2])
        while child != -1:
            hole = contours[child]
            if cv2.contourArea(hole) >= min_hole_area:
                holes.append(hole)
            child = int(hierarchy[child][0])
        holes = sorted(holes, key=cv2.contourArea, reverse=True)[:max_holes]
        tissue_contours.append(contour)
        hole_contours.append(holes)
    return tissue_contours, hole_contours


def scale_contour(contour: np.ndarray, downsample: float) -> np.ndarray:
    scaled = np.rint(contour.astype(np.float64) * float(downsample)).astype(np.int32)
    return scaled


def save_debug_image(rgb: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    overlay = rgb.copy()
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    debug = np.concatenate([rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), overlay], axis=1)
    Image.fromarray(debug).save(output_path)


def process_slide(slide_path: Path, args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    slide_id = slide_path.stem
    mask_dir = output_dir / "masks"
    debug_dir = output_dir / "debug"
    segmentation_dir = output_dir / "segmentations"
    mask_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    mask_path = mask_dir / f"{slide_id}_tissue_mask.png"
    debug_path = debug_dir / f"{slide_id}_stain_filter_debug.png"
    pkl_path = segmentation_dir / f"{slide_id}.pkl"
    if mask_path.exists() and pkl_path.exists() and not args.overwrite:
        return {"slide_id": slide_id, "status": "already_exists", "mask_path": str(mask_path), "pkl_path": str(pkl_path)}

    wsi_object = WholeSlideImage(str(slide_path))
    wsi = wsi_object.getOpenSlide()
    level = choose_level(wsi, args.target_downsample)
    downsample = float(wsi.level_downsamples[level])
    width, height = map(int, wsi.level_dimensions[level])
    region = wsi.read_region((0, 0), level, (width, height)).convert("RGB")
    rgb = np.asarray(region)

    initial = build_initial_mask(
        rgb,
        color_space=args.color_space,
        threshold=args.threshold,
        block_size=args.adaptive_block_size,
        c_value=args.adaptive_c,
        he_filter=not args.disable_he_filter,
    )
    mask = morph_filter(initial, args.open_kernel, args.close_kernel, args.dilate_kernel)
    tissue, holes = contours_from_mask(mask, args.min_tissue_area, args.min_hole_area, args.max_holes)

    tissue_level0 = [scale_contour(contour, downsample) for contour in tissue]
    holes_level0 = [[scale_contour(hole, downsample) for hole in slide_holes] for slide_holes in holes]

    Image.fromarray(mask).save(mask_path)
    save_debug_image(rgb, mask, debug_path)
    with pkl_path.open("wb") as handle:
        pickle.dump({"tissue": tissue_level0, "holes": holes_level0}, handle)

    if hasattr(wsi, "close"):
        wsi.close()

    tissue_area = int((mask > 0).sum())
    return {
        "slide_id": slide_id,
        "status": "processed",
        "slide_path": str(slide_path),
        "seg_level": level,
        "downsample": downsample,
        "mask_width": width,
        "mask_height": height,
        "tissue_pixels": tissue_area,
        "num_tissue_contours": len(tissue),
        "num_holes": sum(len(item) for item in holes),
        "mask_path": str(mask_path),
        "debug_path": str(debug_path),
        "pkl_path": str(pkl_path),
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    slides = collect_slides(source_dir, args.slide_ext, args.manifest_csv, args.limit)
    if not slides:
        raise RuntimeError(f"No slides found in {source_dir} with extension {args.slide_ext}")

    rows = []
    for index, slide_path in enumerate(slides, start=1):
        print(f"[{index}/{len(slides)}] {slide_path}")
        try:
            rows.append(process_slide(slide_path, args, output_dir))
        except Exception as exc:
            rows.append({"slide_id": slide_path.stem, "status": "failed", "slide_path": str(slide_path), "error": str(exc)})
            print(f"failed: {slide_path}: {exc}", file=sys.stderr)

    write_summary(output_dir / "tissue_mask_summary.csv", rows)
    failed = [row for row in rows if row.get("status") == "failed"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
