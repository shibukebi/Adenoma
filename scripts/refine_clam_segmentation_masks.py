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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

extra_site_packages = os.environ.get("PYISYNTAX_SITE_PACKAGES", "").strip()
if extra_site_packages:
    for site_path in extra_site_packages.split(":"):
        if site_path and site_path not in sys.path:
            sys.path.append(site_path)

from generate_tissue_masks import (  # noqa: E402
    build_initial_mask,
    choose_level,
    collect_slides,
    contours_from_mask,
    morph_filter,
    save_debug_image,
    scale_contour,
    write_summary,
)
from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine existing CLAM segmentation pkl files with stain-filtered Tissue Masks."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--clam-segmentation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slide-ext", default=".isyntax")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--target-downsample", type=float, default=64.0)
    parser.add_argument("--color-space", choices=["hsv", "lab"], default="hsv")
    parser.add_argument("--threshold", choices=["otsu", "adaptive"], default="otsu")
    parser.add_argument(
        "--refine-mode",
        choices=["tissue_mask", "dark_stain", "gray_border"],
        default="tissue_mask",
        help="tissue_mask intersects CLAM with a newly estimated tissue mask; dark_stain removes low-S/low-V artifacts; gray_border removes low-S gray artifacts connected to image borders.",
    )
    parser.add_argument(
        "--dark-stain-saturation-max",
        type=int,
        default=40,
        help="Maximum HSV saturation for dark-stain artifact pixels in dark_stain mode.",
    )
    parser.add_argument(
        "--dark-stain-value-max",
        type=int,
        default=80,
        help="Maximum HSV value for dark-stain artifact pixels in dark_stain mode.",
    )
    parser.add_argument(
        "--gray-border-saturation-max",
        type=int,
        default=45,
        help="Maximum HSV saturation for gray border artifact pixels in gray_border mode.",
    )
    parser.add_argument(
        "--gray-border-value-min",
        type=int,
        default=50,
        help="Minimum HSV value for gray border artifact pixels in gray_border mode.",
    )
    parser.add_argument(
        "--gray-border-value-max",
        type=int,
        default=230,
        help="Maximum HSV value for gray border artifact pixels in gray_border mode.",
    )
    parser.add_argument(
        "--gray-border-combine",
        choices=["all", "any"],
        default="all",
        help="Combine gray-border saturation/value tests with AND (all) or OR (any).",
    )
    parser.add_argument(
        "--gray-border-padding",
        type=int,
        default=12,
        help="Mask-level border width used to seed connected gray-border artifact components.",
    )
    parser.add_argument(
        "--gray-border-seed",
        choices=["image", "clam"],
        default="image",
        help="Seed gray-border artifact components from the image boundary or from the CLAM mask boundary.",
    )
    parser.add_argument("--adaptive-block-size", type=int, default=51)
    parser.add_argument("--adaptive-c", type=float, default=-5.0)
    parser.add_argument("--min-tissue-area", type=float, default=5000.0)
    parser.add_argument("--min-hole-area", type=float, default=500.0)
    parser.add_argument("--max-holes", type=int, default=32)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--close-kernel", type=int, default=9)
    parser.add_argument("--dilate-kernel", type=int, default=0)
    parser.add_argument("--disable-he-filter", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def load_clam_segmentation(path: Path) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    tissue = payload.get("tissue", [])
    holes = payload.get("holes", [])
    return tissue, holes


def rasterize_clam_mask(
    tissue_level0: list[np.ndarray],
    holes_level0: list[list[np.ndarray]],
    width: int,
    height: int,
    downsample: float,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    inv_downsample = 1.0 / float(downsample)

    tissue_scaled = [np.rint(contour.astype(np.float64) * inv_downsample).astype(np.int32) for contour in tissue_level0]
    cv2.drawContours(mask, tissue_scaled, contourIdx=-1, color=255, thickness=-1)

    for slide_holes in holes_level0:
        holes_scaled = [np.rint(hole.astype(np.float64) * inv_downsample).astype(np.int32) for hole in slide_holes]
        if holes_scaled:
            cv2.drawContours(mask, holes_scaled, contourIdx=-1, color=0, thickness=-1)
    return mask


def save_refine_debug(rgb: np.ndarray, clam_mask: np.ndarray, stain_mask: np.ndarray, refined_mask: np.ndarray, path: Path) -> None:
    overlay = rgb.copy()
    overlay[refined_mask > 0] = (0.55 * overlay[refined_mask > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    panel = np.concatenate(
        [
            rgb,
            cv2.cvtColor(clam_mask, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(stain_mask, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2RGB),
            overlay,
        ],
        axis=1,
    )
    Image.fromarray(panel).save(path)


def build_dark_stain_keep_mask(
    rgb: np.ndarray,
    saturation_max: int,
    value_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    artifact = (saturation <= int(saturation_max)) & (value <= int(value_max))
    keep_mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    keep_mask[artifact] = 0
    artifact_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    artifact_mask[artifact] = 255
    return keep_mask, artifact_mask


def build_gray_border_keep_mask(
    rgb: np.ndarray,
    clam_mask: np.ndarray,
    saturation_max: int,
    value_min: int,
    value_max: int,
    border_padding: int,
    seed_mode: str,
    combine_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturation_candidate = saturation <= int(saturation_max)
    value_candidate = (value >= int(value_min)) & (value <= int(value_max))
    if combine_mode == "any":
        gray_candidate = (saturation_candidate | value_candidate).astype(np.uint8)
    else:
        gray_candidate = (saturation_candidate & value_candidate).astype(np.uint8)

    height, width = gray_candidate.shape
    pad = max(1, min(int(border_padding), height, width))
    seed = np.zeros_like(gray_candidate, dtype=np.uint8)
    if seed_mode == "clam":
        kernel = np.ones((pad * 2 + 1, pad * 2 + 1), dtype=np.uint8)
        clam_binary = (clam_mask > 0).astype(np.uint8)
        eroded = cv2.erode(clam_binary, kernel, iterations=1)
        clam_boundary = (clam_binary > 0) & (eroded == 0)
        seed[clam_boundary] = gray_candidate[clam_boundary]
    else:
        seed[:pad, :] = gray_candidate[:pad, :]
        seed[-pad:, :] = gray_candidate[-pad:, :]
        seed[:, :pad] = gray_candidate[:, :pad]
        seed[:, -pad:] = gray_candidate[:, -pad:]

    num_labels, labels = cv2.connectedComponents(gray_candidate, connectivity=8)
    seed_labels = np.unique(labels[seed > 0])
    seed_labels = seed_labels[seed_labels != 0]
    artifact = np.isin(labels, seed_labels).astype(np.uint8) if num_labels > 1 else np.zeros_like(gray_candidate)

    keep_mask = np.full((height, width), 255, dtype=np.uint8)
    keep_mask[artifact > 0] = 0
    artifact_mask = np.zeros((height, width), dtype=np.uint8)
    artifact_mask[artifact > 0] = 255
    return keep_mask, artifact_mask


def process_slide(slide_path: Path, args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    slide_id = slide_path.stem
    clam_pkl_path = Path(args.clam_segmentation_dir) / f"{slide_id}.pkl"
    mask_dir = output_dir / "masks"
    debug_dir = output_dir / "debug"
    segmentation_dir = output_dir / "segmentations"
    mask_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    mask_path = mask_dir / f"{slide_id}_tissue_mask_refined.png"
    debug_path = debug_dir / f"{slide_id}_clam_stain_refine_debug.png"
    pkl_path = segmentation_dir / f"{slide_id}.pkl"
    if mask_path.exists() and pkl_path.exists() and not args.overwrite:
        return {
            "slide_id": slide_id,
            "status": "already_exists",
            "mask_path": str(mask_path),
            "pkl_path": str(pkl_path),
        }
    if not clam_pkl_path.exists():
        return {
            "slide_id": slide_id,
            "status": "missing_clam_segmentation",
            "slide_path": str(slide_path),
            "clam_pkl_path": str(clam_pkl_path),
        }

    wsi_object = WholeSlideImage(str(slide_path))
    wsi = wsi_object.getOpenSlide()
    level = choose_level(wsi, args.target_downsample)
    downsample = float(wsi.level_downsamples[level])
    width, height = map(int, wsi.level_dimensions[level])
    rgb = np.asarray(wsi.read_region((0, 0), level, (width, height)).convert("RGB"))

    clam_tissue, clam_holes = load_clam_segmentation(clam_pkl_path)
    clam_mask = rasterize_clam_mask(clam_tissue, clam_holes, width, height, downsample)
    artifact_pixels = 0
    if args.refine_mode == "dark_stain":
        stain_mask, artifact_mask = build_dark_stain_keep_mask(
            rgb,
            args.dark_stain_saturation_max,
            args.dark_stain_value_max,
        )
        artifact_pixels = int(((artifact_mask > 0) & (clam_mask > 0)).sum())
        refined_mask = cv2.bitwise_and(clam_mask, stain_mask)
    elif args.refine_mode == "gray_border":
        stain_mask, artifact_mask = build_gray_border_keep_mask(
            rgb,
            clam_mask,
            args.gray_border_saturation_max,
            args.gray_border_value_min,
            args.gray_border_value_max,
            args.gray_border_padding,
            args.gray_border_seed,
            args.gray_border_combine,
        )
        artifact_pixels = int(((artifact_mask > 0) & (clam_mask > 0)).sum())
        refined_mask = cv2.bitwise_and(clam_mask, stain_mask)
    else:
        stain_mask = build_initial_mask(
            rgb,
            color_space=args.color_space,
            threshold=args.threshold,
            block_size=args.adaptive_block_size,
            c_value=args.adaptive_c,
            he_filter=not args.disable_he_filter,
        )
        stain_mask = morph_filter(stain_mask, args.open_kernel, args.close_kernel, args.dilate_kernel)
        refined_mask = cv2.bitwise_and(clam_mask, stain_mask)
        refined_mask = morph_filter(refined_mask, args.open_kernel, args.close_kernel, 0)

    tissue, holes = contours_from_mask(refined_mask, args.min_tissue_area, args.min_hole_area, args.max_holes)
    tissue_level0 = [scale_contour(contour, downsample) for contour in tissue]
    holes_level0 = [[scale_contour(hole, downsample) for hole in slide_holes] for slide_holes in holes]

    Image.fromarray(refined_mask).save(mask_path)
    save_refine_debug(rgb, clam_mask, stain_mask, refined_mask, debug_path)
    save_debug_image(rgb, refined_mask, debug_dir / f"{slide_id}_refined_overlay.png")
    with pkl_path.open("wb") as handle:
        pickle.dump({"tissue": tissue_level0, "holes": holes_level0}, handle)

    if hasattr(wsi, "close"):
        wsi.close()

    return {
        "slide_id": slide_id,
        "status": "processed",
        "slide_path": str(slide_path),
        "clam_pkl_path": str(clam_pkl_path),
        "seg_level": level,
        "downsample": downsample,
        "mask_width": width,
        "mask_height": height,
        "clam_tissue_pixels": int((clam_mask > 0).sum()),
        "stain_tissue_pixels": int((stain_mask > 0).sum()),
        "dark_stain_artifact_pixels_in_clam": artifact_pixels,
        "refined_tissue_pixels": int((refined_mask > 0).sum()),
        "refine_mode": args.refine_mode,
        "dark_stain_saturation_max": args.dark_stain_saturation_max,
        "dark_stain_value_max": args.dark_stain_value_max,
        "gray_border_saturation_max": args.gray_border_saturation_max,
        "gray_border_value_min": args.gray_border_value_min,
        "gray_border_value_max": args.gray_border_value_max,
        "gray_border_padding": args.gray_border_padding,
        "gray_border_seed": args.gray_border_seed,
        "gray_border_combine": args.gray_border_combine,
        "num_clam_tissue_contours": len(clam_tissue),
        "num_refined_tissue_contours": len(tissue),
        "num_refined_holes": sum(len(item) for item in holes),
        "mask_path": str(mask_path),
        "debug_path": str(debug_path),
        "pkl_path": str(pkl_path),
    }


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

    write_summary(output_dir / "refined_tissue_mask_summary.csv", rows)
    failures = [row for row in rows if row.get("status") not in {"processed", "already_exists"}]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
