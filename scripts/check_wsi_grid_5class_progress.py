#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize WSI grid five-class experiment progress.")
    parser.add_argument("output_dir")
    return parser.parse_args()


def count_jsonl(path):
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    crop_counts = Counter()
    if crops_dir.exists():
        for crop_path in crops_dir.glob("*/*/*.png"):
            crop_counts[crop_path.parts[-3]] += 1
    summary = read_json(output_dir / "summary.json")
    run_config = read_json(output_dir / "run_config.json")
    payload = {
        "output_dir": str(output_dir),
        "run_config": run_config,
        "summary_exists": bool(summary),
        "crop_png_count": int(sum(crop_counts.values())),
        "crop_png_counts_by_crop_id": dict(sorted(crop_counts.items())),
        "manifest_rows": count_jsonl(output_dir / "manifest.jsonl"),
        "prediction_rows": count_jsonl(output_dir / "predictions.jsonl"),
        "fused_rows": count_jsonl(output_dir / "fused_5class_assignments.jsonl"),
        "error_rows": count_jsonl(output_dir / "errors.jsonl"),
        "skipped_wsi_rows": count_jsonl(output_dir / "skipped_wsi.jsonl"),
    }
    if summary:
        payload["summary_counts"] = {
            "n_manifest": summary.get("n_manifest"),
            "n_predictions": summary.get("n_predictions"),
            "n_fused": summary.get("n_fused"),
            "n_errors": summary.get("n_errors"),
            "n_skipped_wsi": summary.get("n_skipped_wsi"),
            "manifest_counts_by_crop_id": summary.get("manifest_counts_by_crop_id", {}),
            "prediction_counts_by_model": summary.get("prediction_counts_by_model", {}),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
