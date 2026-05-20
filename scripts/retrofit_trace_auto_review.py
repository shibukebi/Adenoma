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

from adenoma_agent.trace_supervision import export_candidate_review_package, score_trace_auto_review  # noqa: E402
from adenoma_agent.utils import read_json, write_json  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Retrofit auto_review and visual review artifacts onto an existing trace teacher run.")
    parser.add_argument("--run-root", required=True, help="Existing run root containing per-case directories.")
    parser.add_argument("--case-id", action="append", default=[], help="Optional specific case id(s) to retrofit. Repeatable.")
    return parser


def _candidate_results_from_bundle(bundle):
    candidates = []
    for item in bundle.get("candidates", []):
        if not isinstance(item, dict):
            continue
        candidates.append(
            {
                "candidate_index": item.get("candidate_index"),
                "payload": item.get("payload", {"patches": []}),
                "parse_failure": bool(item.get("parse_failure")),
                "parse_error": item.get("parse_error"),
                "score": item.get("score", {}),
                "raw_response_path": item.get("raw_response_path"),
                "generation_config": item.get("generation_config", {}),
            }
        )
    return candidates


def _grid_paths(case_dir):
    review_dir = case_dir / "review"
    image_paths = sorted(review_dir.glob("*.jpg"))
    meta_paths = [item for item in sorted(review_dir.glob("*.json")) if item.name != "review_target.json"]
    if not image_paths or not meta_paths:
        raise RuntimeError("Missing copied review image or metadata in {0}".format(case_dir))
    return image_paths[0], meta_paths[0]


def _retrofit_case(case_dir):
    checked_dir = case_dir / "auto_checked"
    review_dir = case_dir / "review"
    bundle_path = checked_dir / "qwen_candidates.json"
    if not bundle_path.exists():
        bundle_path = checked_dir / "gemini_candidates.json"
    if not bundle_path.exists():
        raise RuntimeError("No candidate bundle found in {0}".format(case_dir))
    bundle = read_json(bundle_path)
    candidate_results = _candidate_results_from_bundle(bundle)
    if not candidate_results:
        raise RuntimeError("No candidates found in {0}".format(bundle_path))
    image_path, meta_path = _grid_paths(case_dir)
    grid_meta = read_json(meta_path)
    review_target = read_json(review_dir / "review_target.json")
    slide_label_context = review_target.get("auto_review", {}).get("selected_candidate", {}).get("score", {}).get("slide_label_consistency")
    auto_review = score_trace_auto_review(candidate_results, grid_meta, slide_label_context=slide_label_context)
    selection = auto_review["selection"]
    selected_index = selection.get("selected_index")
    selected_payload = {"patches": []}
    if selected_index is not None and 0 <= int(selected_index) < len(auto_review["candidates"]):
        selected_payload = auto_review["candidates"][int(selected_index)]["payload"]
    write_json(checked_dir / "auto_review.json", auto_review)
    write_json(checked_dir / "candidate_agreement.json", auto_review.get("candidate_agreement", {}))
    write_json(bundle_path, {"candidates": auto_review["candidates"], "selection": selection, "auto_review": auto_review})
    write_json(checked_dir / "selected_candidate.json", selected_payload)
    export_candidate_review_package(
        case_dir.name,
        image_path,
        meta_path,
        candidate_results,
        review_dir,
        selection=selection,
        auto_review=auto_review,
    )
    return {
        "case_id": case_dir.name,
        "selected_candidate_index": selected_index,
        "review_status": selection.get("review_status"),
        "review_reason": selection.get("review_reason"),
    }


def main():
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    target_case_ids = set(args.case_id or [])
    rows = []
    for case_dir in sorted([item for item in run_root.iterdir() if item.is_dir()]):
        if target_case_ids and case_dir.name not in target_case_ids:
            continue
        rows.append(_retrofit_case(case_dir))
    summary = {"run_root": str(run_root), "count": len(rows), "rows": rows}
    write_json(run_root / "auto_review_retrofit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
