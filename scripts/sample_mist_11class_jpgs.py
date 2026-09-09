#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image


CLASS_NAMES = {
    0: "ssl",
    1: "hp",
    2: "TSA",
    3: "USA",
    4: "TA",
    5: "TVA",
    6: "IP",
    7: "ssl_with_highgrade_dysplasia",
    8: "TSA_with_highgrade_dysplasia",
    9: "TA_with_highgrade_dysplasia",
    10: "TVA_with_highgrade_dysplasia",
}

DEFAULT_MANIFESTS = [
    "/data15/data15_5/yuexin2/MIST/datasets/mist_hp_yx_11class_fold5_train/mist_hp_yx_11class_fold5_train.csv",
    "/data15/data15_5/yuexin2/MIST/datasets/mist_hp_yx_11class_fold5_val/mist_hp_yx_11class_fold5_val.csv",
    "/data15/data15_5/yuexin2/MIST/datasets/mist_hp_yx_11class_fold5_test/mist_hp_yx_11class_fold5_test.csv",
]
DEFAULT_OUT_DIR = "/data15/data15_5/yuexin2/adenoma/outputs/mist_11class_random_jpg_samples"
DEFAULT_CACHE_DB = "/data15/data15_5/yuexin2/adenoma/challenge_review/cache/wsi/cache_index.sqlite3"
DEFAULT_CACHE_ROOT = "/data15/data15_5/yuexin2/adenoma/challenge_review/cache/wsi"
DEFAULT_YX_ROOTS = [
    "/data15/zhengke_usb2/yuexin_data/Adenoma_yx",
    "/data15/zhengke_usb/Adenoma_yx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample 10 WSI thumbnails per MIST 11-class label as JPG.")
    parser.add_argument("--manifest", action="append", default=DEFAULT_MANIFESTS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache-db", default=DEFAULT_CACHE_DB)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--yx-root", action="append", default=DEFAULT_YX_ROOTS)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max-size", type=int, default=1200)
    parser.add_argument("--svs-timeout", type=int, default=60)
    return parser.parse_args()


def load_cache_thumbnails(cache_db: Path, cache_root: Path) -> dict[str, Path]:
    if not cache_db.exists():
        return {}
    with sqlite3.connect(cache_db) as connection:
        rows = connection.execute(
            "select slide_id, relative_path from entries where kind='thumbnail'"
        ).fetchall()
    thumbnails: dict[str, Path] = {}
    for slide_id, relative_path in rows:
        path = cache_root / relative_path
        if path.exists():
            thumbnails[str(slide_id)] = path
    return thumbnails


def write_jpg_from_cache(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").save(dst, "JPEG", quality=92, optimize=True)


def write_jpg_from_svs(src: Path, dst: Path, max_size: int, timeout: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(".tmp.jpg")
    cmd = ["vipsthumbnail", str(src), "-s", str(max_size), "-o", str(tmp_dst)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        tmp_dst.unlink(missing_ok=True)
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(message or f"vipsthumbnail exited with code {result.returncode}")
    tmp_dst.replace(dst)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataframes = [pd.read_csv(path) for path in args.manifest]
    df = pd.concat(dataframes, ignore_index=True).drop_duplicates("path")
    df["slide_id"] = df["path"].map(lambda value: Path(str(value)).stem)
    df["label"] = df["label"].astype(int)

    cache_thumbnails = load_cache_thumbnails(Path(args.cache_db), Path(args.cache_root))
    yx_slides: dict[str, Path] = {}
    for root in args.yx_root:
        for path in Path(root).glob("*.svs"):
            current = yx_slides.get(path.stem)
            if current is None or path.stat().st_size > current.stat().st_size:
                yx_slides[path.stem] = path

    rng = random.Random(args.seed)
    manifest_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for label, class_name in CLASS_NAMES.items():
        class_df = df[df["label"] == label].copy()
        records = class_df.to_dict(orient="records")
        rng.shuffle(records)
        records = sorted(
            records,
            key=lambda record: 0 if str(record["slide_id"]) in cache_thumbnails else 1,
        )

        selected = 0
        for record in records:
            if selected >= args.per_class:
                break
            slide_id = str(record["slide_id"])
            safe_slide_id = slide_id.replace("/", "_")
            dst = out_dir / f"class_{label:02d}_{class_name}_{selected + 1:02d}_{safe_slide_id}.jpg"
            try:
                if slide_id in cache_thumbnails:
                    write_jpg_from_cache(cache_thumbnails[slide_id], dst)
                    source_kind = "cached_thumbnail"
                    source_path = cache_thumbnails[slide_id]
                elif slide_id in yx_slides and yx_slides[slide_id].stat().st_size > 20 * 1024 * 1024:
                    write_jpg_from_svs(yx_slides[slide_id], dst, args.max_size, args.svs_timeout)
                    source_kind = "svs_thumbnail"
                    source_path = yx_slides[slide_id]
                else:
                    continue
            except Exception as exc:
                failures.append(
                    {
                        "label": label,
                        "class_name": class_name,
                        "slide_id": slide_id,
                        "error": repr(exc),
                    }
                )
                continue

            selected += 1
            print(
                f"class={label:02d} {class_name} selected={selected}/{args.per_class} "
                f"slide_id={slide_id} source={source_kind}",
                flush=True,
            )
            manifest_rows.append(
                {
                    "label": label,
                    "class_name": class_name,
                    "index_in_class": selected,
                    "slide_id": slide_id,
                    "output_jpg": str(dst),
                    "source_kind": source_kind,
                    "source_path": str(source_path),
                }
            )

        if selected < args.per_class:
            raise RuntimeError(
                f"Only exported {selected}/{args.per_class} images for class {label} {class_name}"
            )

    manifest_path = out_dir / "sample_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    failures_path = out_dir / "failed_candidates.csv"
    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["label", "class_name", "slide_id", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)

    print(f"exported={len(manifest_rows)}")
    print(f"manifest={manifest_path}")
    print(f"failures={len(failures)}")
    print(f"failed_candidates={failures_path}")


if __name__ == "__main__":
    main()
