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

from adenoma_agent.stage_scoring import (  # noqa: E402
    score_case_from_paths,
    score_navigate_from_path,
    score_observe_report_from_path,
    score_observe_step_from_path,
    score_trace_from_path,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Score adenoma agent stage JSON outputs with the official rubric.")
    parser.add_argument("--stage", required=True, choices=["trace", "navigate", "observe_step", "observe_report", "case"])
    parser.add_argument("--json", default=None, help="Primary stage JSON path.")
    parser.add_argument("--trace-json", default=None, help="Trace JSON path for navigate/case scoring.")
    parser.add_argument("--navigate-json", default=None, help="Navigate JSON path for case scoring.")
    parser.add_argument("--observe-step-json", default=None, help="Observe-step JSON path for observe_report/case scoring.")
    parser.add_argument("--observe-report-json", default=None, help="Observe-report JSON path for case scoring.")
    parser.add_argument("--grid-metadata-json", default=None, help="Grid metadata JSON path for trace/case scoring.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the score JSON.")
    return parser


def require_arg(value, flag_name):
    if value:
        return value
    raise SystemExit("Missing required argument: {0}".format(flag_name))


def main():
    args = build_parser().parse_args()
    if args.stage == "trace":
        result = score_trace_from_path(
            require_arg(args.json, "--json"),
            require_arg(args.grid_metadata_json, "--grid-metadata-json"),
        )
    elif args.stage == "navigate":
        result = score_navigate_from_path(
            require_arg(args.json, "--json"),
            require_arg(args.trace_json, "--trace-json"),
        )
    elif args.stage == "observe_step":
        result = score_observe_step_from_path(require_arg(args.json, "--json"))
    elif args.stage == "observe_report":
        result = score_observe_report_from_path(
            require_arg(args.json, "--json"),
            require_arg(args.observe_step_json, "--observe-step-json"),
        )
    elif args.stage == "case":
        result = score_case_from_paths(
            trace_json=require_arg(args.trace_json, "--trace-json"),
            navigate_json=require_arg(args.navigate_json, "--navigate-json"),
            observe_step_json=require_arg(args.observe_step_json, "--observe-step-json"),
            observe_report_json=require_arg(args.observe_report_json, "--observe-report-json"),
            grid_metadata_json=require_arg(args.grid_metadata_json, "--grid-metadata-json"),
        )
    else:
        raise SystemExit("Unsupported stage: {0}".format(args.stage))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(1 if result.get("hard_fail") else 0)


if __name__ == "__main__":
    main()
