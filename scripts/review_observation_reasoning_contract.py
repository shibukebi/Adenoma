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
    review_observation_report_payload,
    review_observation_step_payload,
)
from adenoma_agent.utils import read_json  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Review the Observation / Reasoning stage against the current contract.")
    parser.add_argument("--observations-json", default=None, help="Path to observation_records.json")
    parser.add_argument("--report-json", default=None, help="Path to pathological_report.json")
    parser.add_argument("--output-json", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    if not args.observations_json and not args.report_json:
        raise SystemExit("At least one of --observations-json or --report-json is required.")
    result = {"stage": "observation_reasoning"}
    ok = True
    if args.observations_json:
        observations_payload = read_json(args.observations_json)
        result["observe_step"] = review_observation_step_payload(observations_payload)
        result["observe_step"]["json_path"] = str(Path(args.observations_json))
        ok = ok and result["observe_step"]["ok"]
    if args.report_json:
        report_payload = read_json(args.report_json)
        result["observe_report"] = review_observation_report_payload(report_payload)
        result["observe_report"]["json_path"] = str(Path(args.report_json))
        ok = ok and result["observe_report"]["ok"]
    result["ok"] = ok
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
