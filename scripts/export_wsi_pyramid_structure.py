#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import openslide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export level/downsample pyramid structures for yx and hp WSI manifests."
    )
    parser.add_argument(
        "--yx-manifest",
        default="/data15/data15_5/yuexin2/adenoma/data/adenoma_yx_manifest.csv",
    )
    parser.add_argument(
        "--hp-manifest",
        default="/data15/data15_5/yuexin2/adenoma/data/adenoma_hp_manifest.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="/data15/data15_5/yuexin2/adenoma/data/wsi_pyramid_architecture",
    )
    parser.add_argument(
        "--pyisyntax-site-packages",
        default=os.environ.get("PYISYNTAX_SITE_PACKAGES", ""),
        help="Optional site-packages path that provides `from isyntax import ISyntax`.",
    )
    return parser.parse_args()


def maybe_load_isyntax(extra_site_packages: str):
    if extra_site_packages:
        import sys

        for site_path in extra_site_packages.split(":"):
            site_path = site_path.strip()
            if site_path and site_path not in sys.path:
                sys.path.append(site_path)

    try:
        from isyntax import ISyntax
    except Exception:
        return None
    return ISyntax


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_openslide_pyramid(path: str) -> dict[str, Any]:
    slide = openslide.OpenSlide(path)
    try:
        level_dimensions = [tuple(map(int, dim)) for dim in slide.level_dimensions]
        level_downsamples = [float(value) for value in slide.level_downsamples]
        properties = dict(slide.properties)
    finally:
        slide.close()

    return {
        "reader": "openslide",
        "level_dimensions": level_dimensions,
        "level_downsamples": level_downsamples,
        "properties": {
            key: properties.get(key)
            for key in [
                "openslide.vendor",
                "openslide.objective-power",
                "openslide.mpp-x",
                "openslide.mpp-y",
            ]
            if properties.get(key) is not None
        },
    }


def read_isyntax_pyramid(path: str, isyntax_cls) -> dict[str, Any]:
    if isyntax_cls is None:
        raise RuntimeError("isyntax package is unavailable; pass --pyisyntax-site-packages or set PYISYNTAX_SITE_PACKAGES.")

    slide = isyntax_cls.open(path)
    try:
        level_dimensions = [tuple(map(int, dim)) for dim in slide.level_dimensions]
        level_downsamples = [float(value) for value in slide.level_downsamples]
        properties = {}
        for attr in ("mpp_x", "mpp_y"):
            value = getattr(slide, attr, None)
            if value is not None:
                properties[attr] = value
    finally:
        slide.close()

    return {
        "reader": "isyntax",
        "level_dimensions": level_dimensions,
        "level_downsamples": level_downsamples,
        "properties": properties,
    }


def signature(level_dimensions: list[tuple[int, int]], level_downsamples: list[float]) -> str:
    parts = []
    for idx, (dim, downsample) in enumerate(zip(level_dimensions, level_downsamples)):
        parts.append(f"L{idx}:{dim[0]}x{dim[1]}@{downsample:g}")
    return "; ".join(parts)


def build_rows(dataset: str, manifest_rows: list[dict[str, str]], isyntax_cls) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slide_records = []
    level_records = []

    for idx, row in enumerate(manifest_rows, start=1):
        slide_id = row["slide_id"]
        slide_path = row["slide_path"]
        slide_filename = row.get("slide_filename", Path(slide_path).name)
        ext = Path(slide_path).suffix.lower()

        print(f"[{dataset}] {idx}/{len(manifest_rows)} {slide_id}", flush=True)
        try:
            if ext == ".isyntax":
                payload = read_isyntax_pyramid(slide_path, isyntax_cls)
            else:
                payload = read_openslide_pyramid(slide_path)

            level_dimensions = payload["level_dimensions"]
            level_downsamples = payload["level_downsamples"]
            level0_width, level0_height = level_dimensions[0]

            slide_record = {
                "dataset": dataset,
                "slide_id": slide_id,
                "slide_filename": slide_filename,
                "slide_path": slide_path,
                "reader": payload["reader"],
                "level_count": len(level_dimensions),
                "pyramid_signature": signature(level_dimensions, level_downsamples),
                "properties": payload["properties"],
                "levels": [],
                "status": "ok",
                "error": "",
            }

            for level, ((width, height), reported_downsample) in enumerate(zip(level_dimensions, level_downsamples)):
                estimated_downsample_x = level0_width / float(width)
                estimated_downsample_y = level0_height / float(height)
                level_record = {
                    "dataset": dataset,
                    "slide_id": slide_id,
                    "slide_filename": slide_filename,
                    "slide_path": slide_path,
                    "reader": payload["reader"],
                    "level_count": len(level_dimensions),
                    "level": level,
                    "width": width,
                    "height": height,
                    "reported_downsample": reported_downsample,
                    "estimated_downsample_x": estimated_downsample_x,
                    "estimated_downsample_y": estimated_downsample_y,
                    "status": "ok",
                    "error": "",
                }
                level_records.append(level_record)
                slide_record["levels"].append(
                    {
                        "level": level,
                        "width": width,
                        "height": height,
                        "reported_downsample": reported_downsample,
                        "estimated_downsample_x": estimated_downsample_x,
                        "estimated_downsample_y": estimated_downsample_y,
                    }
                )
            slide_records.append(slide_record)
        except Exception as exc:
            message = str(exc)
            slide_records.append(
                {
                    "dataset": dataset,
                    "slide_id": slide_id,
                    "slide_filename": slide_filename,
                    "slide_path": slide_path,
                    "reader": "",
                    "level_count": 0,
                    "pyramid_signature": "",
                    "properties": {},
                    "levels": [],
                    "status": "error",
                    "error": message,
                }
            )
            level_records.append(
                {
                    "dataset": dataset,
                    "slide_id": slide_id,
                    "slide_filename": slide_filename,
                    "slide_path": slide_path,
                    "reader": "",
                    "level_count": 0,
                    "level": "",
                    "width": "",
                    "height": "",
                    "reported_downsample": "",
                    "estimated_downsample_x": "",
                    "estimated_downsample_y": "",
                    "status": "error",
                    "error": message,
                }
            )
    return slide_records, level_records


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "slide_id",
        "slide_filename",
        "slide_path",
        "reader",
        "level_count",
        "level",
        "width",
        "height",
        "reported_downsample",
        "estimated_downsample_x",
        "estimated_downsample_y",
        "status",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_nested_json(path: Path, slide_records: list[dict[str, Any]]) -> None:
    payload: dict[str, Any] = {"datasets": defaultdict(list)}
    for record in slide_records:
        payload["datasets"][record["dataset"]].append(record)
    payload["datasets"] = dict(payload["datasets"])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_md(path: Path, slide_records: list[dict[str, Any]]) -> None:
    by_dataset = defaultdict(list)
    for record in slide_records:
        by_dataset[record["dataset"]].append(record)

    lines = ["# WSI Pyramid Architecture Summary", ""]
    for dataset in sorted(by_dataset):
        records = by_dataset[dataset]
        ok_records = [record for record in records if record["status"] == "ok"]
        error_records = [record for record in records if record["status"] != "ok"]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- slides: {len(records)}",
                f"- ok: {len(ok_records)}",
                f"- errors: {len(error_records)}",
                "",
                "| count | level_count | pyramid signature |",
                "| --- | --- | --- |",
            ]
        )
        counts = Counter(record["pyramid_signature"] for record in ok_records)
        for sig, count in counts.most_common():
            level_count = sig.count("L")
            lines.append(f"| {count} | {level_count} | `{sig}` |")
        if error_records:
            lines.extend(["", "Errors:", ""])
            for record in error_records:
                lines.append(f"- `{record['slide_id']}`: {record['error']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    isyntax_cls = maybe_load_isyntax(args.pyisyntax_site_packages)

    manifests = {
        "yx": read_manifest(Path(args.yx_manifest)),
        "hp": read_manifest(Path(args.hp_manifest)),
    }

    all_slide_records = []
    all_level_records = []
    for dataset, rows in manifests.items():
        slide_records, level_records = build_rows(dataset, rows, isyntax_cls)
        all_slide_records.extend(slide_records)
        all_level_records.extend(level_records)

    write_flat_csv(output_prefix.with_suffix(".levels.csv"), all_level_records)
    write_nested_json(output_prefix.with_suffix(".nested.json"), all_slide_records)
    write_summary_md(output_prefix.with_suffix(".summary.md"), all_slide_records)

    print(f"wrote {output_prefix.with_suffix('.levels.csv')}")
    print(f"wrote {output_prefix.with_suffix('.nested.json')}")
    print(f"wrote {output_prefix.with_suffix('.summary.md')}")


if __name__ == "__main__":
    main()
