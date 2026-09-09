from __future__ import annotations

import csv
import argparse
import os
import shutil
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / "data" / "challenge_review.sqlite3"
OUTPUT = ROOT / "data" / "challenge_14of14_wrong_wsi_links"


def copy_wsi(source: Path, destination: Path) -> None:
    if destination.exists() and not destination.is_symlink():
        if destination.stat().st_size == source.stat().st_size:
            return
        raise RuntimeError(f"Refusing to replace existing file: {destination}")
    if destination.is_symlink() and destination.resolve() != source.resolve():
        raise RuntimeError(f"Unexpected symbolic link target: {destination}")

    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"Remove stale partial file before retrying: {temporary}")
    with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=16 * 1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if temporary.stat().st_size != source.stat().st_size:
        temporary.unlink()
        raise RuntimeError(f"Copy size mismatch: {source}")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--copy-wsi",
        action="store_true",
        help="Replace existing links with independent WSI file copies.",
    )
    arguments = parser.parse_args()
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT slide_id, original_label, pathology_type, grade, source, cv_fold,
               wsi_path, wsi_format, hardness_rank, consensus_wrong_class,
               consensus_wrong_count, mean_wrong_confidence
        FROM slides
        WHERE wrong_configurations = 14
        ORDER BY slide_id
        """
    ).fetchall()
    connection.close()

    if len(rows) != 193:
        raise RuntimeError(f"Expected 193 14/14 wrong slides, found {len(rows)}")
    missing = [row["wsi_path"] for row in rows if not Path(row["wsi_path"]).is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} source WSI files; no export created")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        source = Path(row["wsi_path"])
        destination = OUTPUT / source.name
        if arguments.copy_wsi:
            copy_wsi(source, destination)
            print(f"[{index}/{len(rows)}] copied {source.name}", flush=True)
            continue
        if destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"Refusing to replace existing path: {destination}")
        destination.symlink_to(source)

    manifest = OUTPUT / "manifest_14of14_wrong.csv"
    temporary_manifest = manifest.with_suffix(".csv.tmp")
    with temporary_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)
    temporary_manifest.replace(manifest)

    export_type = "WSI copies" if arguments.copy_wsi else "WSI links"
    print(f"Created {len(rows)} {export_type} in {OUTPUT}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
