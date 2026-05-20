#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adenoma_agent.trace_supervision import (  # noqa: E402
    TRACE_LABEL_RUBRIC,
    build_image_only_teacher_prompt,
    selected_patch_ids_from_grid,
)
from adenoma_agent.utils import ensure_dir, read_csv_rows, read_json, write_json  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Build hybrid trace teacher request packages from grid thumbnails.")
    parser.add_argument("--grid-dir", required=True, help="Directory containing *_grid.jpg and matching .json files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pathreasoner-root", default=None, help="Optional trace experiment root with prior trace_clusters.json files.")
    parser.add_argument("--labels-csv", default="data/adenoma_yx_labels.csv", help="Optional slide-level labels CSV for weak consistency checks.")
    return parser


def _pathreasoner_trace_path(root, case_name):
    if not root:
        return None
    direct = Path(root) / case_name / "trace" / "trace_clusters.json"
    if direct.exists():
        return direct
    legacy = Path(root) / case_name / (case_name + "_trace_clusters.json")
    if legacy.exists():
        return legacy
    return None


def _load_label_lookup(labels_csv):
    path = Path(labels_csv)
    if not path.exists():
        return {}
    lookup = {}
    for row in read_csv_rows(path):
        slide_id = row.get("slide_id")
        if slide_id:
            lookup[slide_id] = row
    return lookup


def _slide_label_context(label_row):
    label = (label_row or {}).get("type")
    grade = (label_row or {}).get("grade")
    lower_label = str(label or "").lower()
    return {
        "label": label,
        "grade": grade,
        "serrated_target": 1 if any(token in lower_label for token in ("sessile serrated", "traditional serrated", "serrated adenoma", "hyperplastic")) else 0 if label else None,
        "abnormal_crypt_target": 1 if "sessile serrated adenoma" in lower_label else 0 if label else None,
        "dysplasia_proxy_target": 1 if str(grade or "").strip().lower() == "high" else 0 if grade is not None else None,
        "metadata": {"type": label, "grade": grade},
    }


def _load_agent_vocabulary():
    vocab_path = REPO_ROOT / "docs" / "model" / "vocab" / "adenoma_agent_vocabulary.md"
    if not vocab_path.exists():
        return None
    return {
        "source": str(vocab_path),
        "summary": (
            "Use pathology workflow terminology consistent with the adenoma agent vocabulary. "
            "Keep region names, morphology descriptions, and review wording aligned with the five-label semantic space."
        ),
    }


def main():
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    label_lookup = _load_label_lookup(args.labels_csv)
    grid_paths = sorted(Path(args.grid_dir).glob("*_grid.jpg"))
    if args.limit is not None:
        grid_paths = grid_paths[: args.limit]
    index_rows = []
    for grid_path in grid_paths:
        meta_path = grid_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        case_name = grid_path.stem
        grid_meta = read_json(meta_path)
        selected_patch_ids = [list(item) for item in selected_patch_ids_from_grid(grid_meta)]
        prior_path = _pathreasoner_trace_path(args.pathreasoner_root, case_name)
        slide_id = str(grid_meta.get("slide_id") or case_name.split("_tissuegrid", 1)[0])
        slide_label_context = _slide_label_context(label_lookup.get(slide_id))
        request = {
            "case_id": case_name,
            "report_available": False,
            "teacher_mode": "image_only_screening_trace",
            "teacher_model": "gemini-2.5-pro",
            "candidate_count": 2,
            "generation_config": {
                "temperature": 0.8,
                "max_output_tokens": 3000,
            },
            "grid_thumbnail_path": str(grid_path),
            "grid_metadata_path": str(meta_path),
            "slide_label_context": slide_label_context,
            "selected_patch_ids": selected_patch_ids,
            "grid_metadata_summary": {
                "grid_rows": grid_meta.get("grid_rows"),
                "grid_cols": grid_meta.get("grid_cols"),
                "n_selected_cells": grid_meta.get("n_selected_cells", len(selected_patch_ids)),
                "thumbnail_mode": grid_meta.get("thumbnail_mode"),
                "selected_cells": [
                    {
                        "patch_id": [cell.get("row_id"), cell.get("col_id")],
                        "thumbnail_bbox": [
                            cell.get("thumbnail_top_left_x"),
                            cell.get("thumbnail_top_left_y"),
                            cell.get("thumbnail_width"),
                            cell.get("thumbnail_height"),
                        ],
                        "level0_bbox": [
                            cell.get("level0_top_left_x"),
                            cell.get("level0_top_left_y"),
                            cell.get("level0_width"),
                            cell.get("level0_height"),
                        ],
                        "tissue_coverage_ratio": cell.get("tissue_coverage_ratio"),
                    }
                    for cell in grid_meta.get("grid_cells", [])
                    if cell.get("is_selected")
                ],
                "black_mask_hint": "Background has been black-masked to make real tissue stand out.",
            },
            "trace_rubric": TRACE_LABEL_RUBRIC,
            "agent_vocabulary": _load_agent_vocabulary(),
            "no_report_prompt_block": (
                "No pathology report is available. This is image-only morphology supervision. "
                "Produce screening-level only trace labels; no final diagnosis."
            ),
            "teacher_instruction": (
                "Return one JSON object with {'patches': [...]} only. Assign every selected patch exactly once. "
                "Use possible/suspicious/review-worthy/check wording instead of final diagnostic claims. "
                "Use the adenoma trace rubric for region_semantic, description, diagnostic_priority, "
                "require_high_magnification, and observation_points."
            ),
            "optional_context": {
                "pathreasoner_trace_clusters_json": str(prior_path) if prior_path else None,
            },
            "teacher_output_stub": {
                "patches": [
                    {
                        "patch_id": patch_id,
                        "region_semantic": "",
                        "name": "",
                        "description": "",
                        "require_high_magnification": False,
                        "severity_reasoning": "",
                        "diagnostic_priority": 0,
                        "observation_points": [],
                    }
                    for patch_id in selected_patch_ids
                ]
            },
        }
        request["prompt_text"] = build_image_only_teacher_prompt(request)
        case_dir = ensure_dir(output_dir / case_name)
        request_path = write_json(case_dir / "teacher_request.json", request)
        index_rows.append({"case_id": case_name, "teacher_request_json": str(request_path)})
    write_json(output_dir / "teacher_request_index.json", {"rows": index_rows})
    print(json.dumps({"count": len(index_rows), "index": str(output_dir / "teacher_request_index.json")}))


if __name__ == "__main__":
    main()
