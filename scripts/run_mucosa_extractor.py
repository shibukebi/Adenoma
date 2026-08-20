#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from adenoma_agent.mucosa_extractor import MucosaExtractorConfig, run_mucosa_extractor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build Mucosa Extractor v1 evidence from raw WSI or saved PathPrism artifacts.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wsi", action="append", default=[], help="WSI path; repeat for multiple slides.")
    source.add_argument("--wsi-dir", default="", help="Directory containing WSI files.")
    source.add_argument("--artifact-dir", default="", help="Directory containing manifest.jsonl and predictions.jsonl.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover WSI files under --wsi-dir.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--pathprism-url", default="http://127.0.0.1:8400/predict")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--base-magnification", type=float, default=0.0)
    parser.add_argument("--min-tissue-coverage", type=float, default=0.05)
    parser.add_argument("--mask-downsample", type=float, default=32.0)
    parser.add_argument("--mucosa-threshold", type=float, default=0.30)
    parser.add_argument("--presence-area-threshold", type=float, default=0.005)
    parser.add_argument("--presence-peak-threshold", type=float, default=0.50)
    parser.add_argument("--five-x-patch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-qc-panel", action="store_true", help="Do not write the per-slide six-panel QC image.")
    parser.add_argument("--skip-overlays", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def discover_wsi(root, recursive=False):
    from adenoma_agent.mucosa_extractor import SUPPORTED_WSI_SUFFIXES

    root = Path(root)
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_WSI_SUFFIXES)


def main():
    args = parse_args()
    wsi_paths = [Path(path).resolve() for path in args.wsi]
    if args.wsi_dir:
        wsi_paths.extend(discover_wsi(args.wsi_dir, recursive=args.recursive))
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else None
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif artifact_dir:
        output_dir = artifact_dir / "mucosa_extractor"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "artifacts" / "mucosa_extractor_{0}".format(stamp)
    config = MucosaExtractorConfig(
        pathprism_url=args.pathprism_url,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        base_magnification=args.base_magnification,
        min_tissue_coverage=args.min_tissue_coverage,
        mask_downsample=args.mask_downsample,
        mucosa_threshold=args.mucosa_threshold,
        presence_area_threshold=args.presence_area_threshold,
        presence_peak_threshold=args.presence_peak_threshold,
        five_x_patch_size=args.five_x_patch_size,
        resume=bool(args.resume),
        write_overlays=not bool(args.skip_qc_panel or args.skip_overlays),
    )
    summary = run_mucosa_extractor(
        wsi_paths=wsi_paths or None,
        artifact_dir=artifact_dir,
        output_dir=output_dir,
        config=config,
    )
    print(json.dumps({"event": "mucosa_extractor_complete", **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
