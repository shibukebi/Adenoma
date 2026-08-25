#!/usr/bin/env python3
import argparse
from pathlib import Path

from adenoma_agent.dashboard import (
    build_dashboard_payload_from_harness_case,
    build_demo_dashboard_payload,
    export_dashboard,
    export_dashboard_batch_from_harness_run,
    export_dashboard_from_json,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Export a static agent experiment visualization dashboard.")
    parser.add_argument(
        "--payload-json",
        default=None,
        help="Path to a dashboard payload JSON. If omitted together with --demo, demo payload is used.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where index.html and payload snapshot will be written.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate the dashboard from an internal demo payload.",
    )
    parser.add_argument(
        "--harness-case-dir",
        default=None,
        help="Path to one harness case output directory containing trace/observe/navigation JSON artifacts.",
    )
    parser.add_argument(
        "--harness-run-dir",
        default=None,
        help="Path to a harness run directory containing multiple case output directories.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if args.harness_run_dir:
        run_dir = Path(args.harness_run_dir).resolve()
        if not run_dir.exists():
            raise SystemExit("Harness run directory not found: {0}".format(run_dir))
        export_dashboard_batch_from_harness_run(run_dir, output_dir)
        print("Dashboard batch exported to {0}".format(output_dir / "index.html"))
        return

    if args.harness_case_dir:
        case_dir = Path(args.harness_case_dir).resolve()
        if not case_dir.exists():
            raise SystemExit("Harness case directory not found: {0}".format(case_dir))
        export_dashboard(build_dashboard_payload_from_harness_case(case_dir), output_dir, source_root=case_dir)
        print("Dashboard exported to {0}".format(output_dir / "index.html"))
        return

    if args.demo or not args.payload_json:
        export_dashboard(build_demo_dashboard_payload(), output_dir)
        print("Dashboard exported to {0}".format(output_dir / "index.html"))
        return

    payload_json = Path(args.payload_json).resolve()
    if not payload_json.exists():
        raise SystemExit("Payload JSON not found: {0}".format(payload_json))
    export_dashboard_from_json(payload_json, output_dir)
    print("Dashboard exported to {0}".format(output_dir / "index.html"))


if __name__ == "__main__":
    main()
