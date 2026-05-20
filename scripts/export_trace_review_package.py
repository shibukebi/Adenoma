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
    export_candidate_review_package,
    export_review_package,
    read_assignment_payload,
    select_best_candidate,
)
from adenoma_agent.utils import read_json  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Export a human-review package for one trace supervision case.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--grid-thumbnail-path", required=True)
    parser.add_argument("--grid-metadata-path", required=True)
    parser.add_argument("--teacher-assignment-json", default=None)
    parser.add_argument("--gemini-candidates-json", default=None, help="JSON with a candidates array from run_gemini_trace_teacher.")
    parser.add_argument("--pathreasoner-assignment-json", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    pathreasoner = read_assignment_payload(args.pathreasoner_assignment_json) if args.pathreasoner_assignment_json else None
    if args.gemini_candidates_json:
        candidate_bundle = read_json(args.gemini_candidates_json)
        candidate_results = candidate_bundle.get("candidates", []) if isinstance(candidate_bundle, dict) else []
        selection = candidate_bundle.get("selection") if isinstance(candidate_bundle, dict) else None
        selection = selection or select_best_candidate(candidate_results)
        output_dir = export_candidate_review_package(
            args.case_id,
            args.grid_thumbnail_path,
            args.grid_metadata_path,
            candidate_results,
            Path(args.output_dir),
            selection=selection,
            pathreasoner_payload=pathreasoner,
        )
    else:
        if not args.teacher_assignment_json:
            raise SystemExit("--teacher-assignment-json or --gemini-candidates-json is required")
        teacher = read_assignment_payload(args.teacher_assignment_json)
        output_dir = export_review_package(
            args.case_id,
            args.grid_thumbnail_path,
            args.grid_metadata_path,
            teacher,
            Path(args.output_dir),
            pathreasoner_payload=pathreasoner,
        )
    print(json.dumps({"review_dir": str(output_dir), "review_target": str(Path(output_dir) / "review_target.json")}))


if __name__ == "__main__":
    main()
