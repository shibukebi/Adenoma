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

from adenoma_agent.contract_review import (  # noqa: E402
    review_global_screening_payload,
    review_navigation_payload,
    review_observation_report_payload,
    review_observation_step_payload,
)
from adenoma_agent.utils import read_json  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Validate stage JSON outputs for the adenoma agent pipeline.")
    parser.add_argument("--stage", required=True, choices=["trace", "navigate", "observe_step", "observe_report"])
    parser.add_argument("--json", required=True, help="Path to the stage JSON artifact.")
    parser.add_argument("--grid-metadata-json", default=None, help="Optional grid metadata JSON for trace/global-screening review.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the validation result.")
    return parser


def main():
    args = build_parser().parse_args()
    payload = read_json(args.json)
    grid_meta = read_json(args.grid_metadata_json) if args.grid_metadata_json else None
    if args.stage == "trace":
        result = review_global_screening_payload(payload, grid_meta=grid_meta)
    elif args.stage == "navigate":
        result = review_navigation_payload(payload)
    elif args.stage == "observe_step":
        result = review_observation_step_payload(payload)
    elif args.stage == "observe_report":
        result = review_observation_report_payload(payload)
    else:
        raise SystemExit("Unsupported stage: {0}".format(args.stage))
    result.update({"stage": args.stage, "json_path": str(Path(args.json))})
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
