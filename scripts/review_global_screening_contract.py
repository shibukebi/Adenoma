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

from adenoma_agent.contract_review import review_global_screening_payload  # noqa: E402
from adenoma_agent.utils import read_json  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Review the Global Screening / Trace stage against the current contract.")
    parser.add_argument("--json", required=True, help="Path to the global screening JSON artifact.")
    parser.add_argument("--grid-metadata-json", default=None, help="Optional grid metadata JSON for exact patch coverage checks.")
    parser.add_argument("--output-json", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    payload = read_json(args.json)
    grid_meta = read_json(args.grid_metadata_json) if args.grid_metadata_json else None
    result = review_global_screening_payload(payload, grid_meta=grid_meta)
    result.update({"stage": "global_screening", "json_path": str(Path(args.json))})
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
