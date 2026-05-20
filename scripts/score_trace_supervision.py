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

from adenoma_agent.trace_supervision import read_assignment_payload, score_trace_case  # noqa: E402
from adenoma_agent.utils import read_json, write_json  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Score a trace supervision patch-assignment target.")
    parser.add_argument("--assignment-json", required=True)
    parser.add_argument("--grid-metadata-json", required=True)
    parser.add_argument("--clusters-json", default=None)
    parser.add_argument("--pathreasoner-assignment-json", default=None)
    parser.add_argument("--output-json", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    target = read_assignment_payload(args.assignment_json)
    grid_meta = read_json(args.grid_metadata_json)
    clusters = None
    if args.clusters_json:
        clusters_payload = read_json(args.clusters_json)
        clusters = clusters_payload.get("clusters", clusters_payload.get("selected_clusters", []))
    pathreasoner = read_assignment_payload(args.pathreasoner_assignment_json) if args.pathreasoner_assignment_json else None
    score = score_trace_case(target, grid_meta, clusters=clusters, pathreasoner_payload=pathreasoner)
    write_json(args.output_json, score)
    print(json.dumps({"output_json": args.output_json, "total_score": score["total_score"], "review_recommended": score["review_recommended"]}))


if __name__ == "__main__":
    main()
