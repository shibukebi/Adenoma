#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from patho_r1_route_c_select import load_patho_r1_stack, process_slide  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Route C selection and coords generation.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--coords-root", required=True)
    parser.add_argument("--mode", choices=["patho-r1", "heuristic", "manual"], default="patho-r1")
    parser.add_argument("--fallback-mode", choices=["none", "heuristic"], default="heuristic")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--thumbnail-max-size", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=256)
    parser.add_argument("--patch-level", type=int, default=0)
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["slide_id"] for row in csv.DictReader(handle) if row.get("slide_id")}


def read_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slide_id",
        "slide_path",
        "mode_requested",
        "mode_used",
        "status",
        "box_count",
        "coords_count",
        "failure_reason",
        "selection_dir",
        "boxes_json",
        "coords_h5",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest_rows = read_manifest(Path(args.manifest_csv))
    manifest_path = Path(args.output_root) / "manifest_route_c.csv"
    processed_ids = read_processed_ids(manifest_path) if args.resume else set()
    output_rows = read_existing_rows(manifest_path) if args.resume else []

    selected_rows = manifest_rows[args.start_index:]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    model = None
    processor = None
    if args.mode == "patho-r1":
        model, processor, resolved = load_patho_r1_stack(
            model_id=args.model_id,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
        print(f"loaded_model_source={resolved}")

    for row in selected_rows:
        slide_id = row["slide_id"]
        slide_path = row["slide_path"]
        if slide_id in processed_ids:
            continue

        selection_dir = Path(args.output_root) / slide_id
        coords_h5 = Path(args.coords_root) / "patches" / f"{slide_id}.h5"

        result = process_slide(
            slide_path=slide_path,
            output_dir=str(selection_dir),
            mode=args.mode,
            fallback_mode=args.fallback_mode,
            model=model,
            processor=processor,
            model_id=args.model_id,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            thumbnail_max_size=args.thumbnail_max_size,
            max_boxes=args.max_boxes,
            max_new_tokens=args.max_new_tokens,
        )

        boxes_json = selection_dir / f"{slide_id}_route_c_boxes.json"
        coords_count = 0
        status = result["status"]
        if status.startswith("success"):
            try:
                from route_c_boxes_to_h5 import convert_boxes_json_to_h5
                coords_count, _ = convert_boxes_json_to_h5(
                    boxes_json=boxes_json,
                    output_h5=coords_h5,
                    patch_size=args.patch_size,
                    step_size=args.step_size,
                    patch_level=args.patch_level,
                )
                if coords_count == 0:
                    status = "failed_no_box"
            except Exception as exc:
                status = "failed_parse_error"
                result["failure_reason"] = str(exc)

        output_rows.append(
            {
                "slide_id": slide_id,
                "slide_path": slide_path,
                "mode_requested": result["mode_requested"],
                "mode_used": result["mode_used"],
                "status": status,
                "box_count": len(result.get("boxes", [])),
                "coords_count": coords_count,
                "failure_reason": result.get("failure_reason", ""),
                "selection_dir": str(selection_dir),
                "boxes_json": str(boxes_json),
                "coords_h5": str(coords_h5) if coords_count > 0 else "",
            }
        )
        write_manifest(manifest_path, output_rows)

    print(f"manifest={manifest_path}")
    print(f"rows_written={len(output_rows)}")


if __name__ == "__main__":
    main()
