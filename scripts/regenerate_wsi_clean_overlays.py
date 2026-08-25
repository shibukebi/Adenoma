#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from run_wsi_grid_5class_experiment import (  # noqa: E402
    iter_jsonl,
    read_json,
    render_overlays,
    write_json,
    write_output_contract,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate five-class overlays using clean WSI overview bases.")
    parser.add_argument("output_dir", help="Completed WSI grid five-class experiment output directory.")
    return parser.parse_args()


def load_jsonl(path):
    return list(iter_jsonl(path) or [])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Missing summary.json in {0}".format(output_dir))

    summary = read_json(summary_path)
    planned = summary.get("planned") or []
    if not planned:
        raise RuntimeError("summary.json does not contain planned slide metadata.")

    manifest_rows = load_jsonl(output_dir / "manifest.jsonl")
    predictions = load_jsonl(output_dir / "predictions.jsonl")
    fused_rows = load_jsonl(output_dir / "fused_5class_assignments.jsonl")
    if not manifest_rows or not predictions or not fused_rows:
        raise RuntimeError("manifest.jsonl, predictions.jsonl, and fused_5class_assignments.jsonl must be populated.")

    report = render_overlays(output_dir, planned, manifest_rows, predictions, fused_rows)
    summary["overlay_report"] = report
    summary["overlays_dir"] = str(output_dir / "overlays")
    write_json(summary_path, summary)
    write_json(output_dir / "overlays" / "_overlay_report.json", report)
    write_output_contract(output_dir)
    print(json.dumps({"event": "clean_wsi_overlays_regenerated", **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
