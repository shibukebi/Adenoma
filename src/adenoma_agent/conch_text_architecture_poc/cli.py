"""CLI for the strictly separated prepare/freeze architecture workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .pipeline import (
    DEFAULT_CONFIG,
    BenchmarkWorkflowError,
    _load_config,
    freeze_benchmark,
    prepare_annotation,
    status_report,
)
from .backfill import audit_villous_recruitment, selective_mucosa_backfill


def _print_terminal(payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        else:
            rendered = str(value)
        print("{0}={1}".format(key, rendered))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expert-confirmed 5x architecture annotation preparation and benchmark freeze"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-annotation", help="materialize blinded ROI PNGs and annotation.xlsx")
    prepare.add_argument("--resume", action="store_true", help="verify and reuse an identical existing package")

    commands.add_parser("audit-villous", help="read-only villous source/Mucosa recruitment audit")

    backfill = commands.add_parser("backfill-villous", help="selectively run isolated POC-local Mucosa backfill")
    backfill.add_argument("--pathprism-url", default=None)

    freeze = commands.add_parser("freeze", help="validate completed expert annotations and freeze benchmark_v1")
    freeze.add_argument("--annotations", type=Path, required=True)
    freeze.add_argument("--resume", action="store_true", help="verify immutable benchmark identity; never rewrite v1")

    commands.add_parser("status", help="report the current legal gate without changing state")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare-annotation":
            result = prepare_annotation(args.config, resume=bool(args.resume))
        elif args.command == "audit-villous":
            config, paths = _load_config(args.config)
            result = audit_villous_recruitment(config, paths)["summary"]
        elif args.command == "backfill-villous":
            config, paths = _load_config(args.config)
            audit = audit_villous_recruitment(config, paths)
            result = selective_mucosa_backfill(config, paths, audit, pathprism_url=args.pathprism_url)["summary"]
        elif args.command == "freeze":
            result = freeze_benchmark(args.annotations, args.config, resume=bool(args.resume))
        else:
            result = status_report(args.config)
    except BenchmarkWorkflowError as exc:
        _print_terminal(
            {
                "FREEZE_FAIL": args.command == "freeze",
                "BENCHMARK_FREEZE_COMPLETE": False,
                "MODEL_EVALUATION_STARTED": False,
                "FINAL_GATE": exc.gate,
                "ERROR": str(exc),
                "ISSUES": exc.issues,
            }
        )
        return 2
    except Exception as exc:
        _print_terminal(
            {
                "FREEZE_FAIL": args.command == "freeze",
                "BENCHMARK_FREEZE_COMPLETE": False,
                "MODEL_EVALUATION_STARTED": False,
                "FINAL_GATE": getattr(exc, "gate", "PROVENANCE_BLOCKED"),
                "ERROR": str(exc),
                "ISSUES": getattr(exc, "issues", []),
            }
        )
        return 2
    _print_terminal(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
