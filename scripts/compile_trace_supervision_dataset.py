#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adenoma_agent.trace_supervision import build_supervision_record, read_assignment_payload, write_jsonl  # noqa: E402
from adenoma_agent.utils import read_json, write_json  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Compile reviewed trace targets into canonical supervision JSONL.")
    parser.add_argument("--review-root", required=True, help="Directory containing per-case review_target.json files.")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--val-jsonl", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--exclude-case-id", action="append", default=[], help="Case id to keep out of train/val compilation. Repeatable.")
    parser.add_argument("--output-summary-json", default=None)
    parser.add_argument("--include-auto-pass", action="store_true")
    return parser


def _case_paths(review_path):
    payload = read_json(review_path)
    case_dir = review_path.parent
    image_paths = list(case_dir.glob("*_grid.jpg"))
    if not image_paths:
        image_paths = list(case_dir.glob("*.jpg"))
    meta_paths = list(case_dir.glob("*_grid.json"))
    if not meta_paths:
        meta_paths = list(case_dir.glob("*.json"))
        excluded_names = {
            "review_target.json",
            "auto_score.json",
            "teacher_patch_assignments.json",
            "pathreasoner_patch_assignments.json",
            "gemini_candidates.json",
            "candidate_diff.json",
        }
        meta_paths = [item for item in meta_paths if item.name not in excluded_names]
    return payload, image_paths[0] if image_paths else None, meta_paths[0] if meta_paths else None


def main():
    args = build_parser().parse_args()
    if not args.output_jsonl and not (args.train_jsonl and args.val_jsonl):
        raise SystemExit("Provide --output-jsonl or both --train-jsonl and --val-jsonl")
    rows = []
    skipped = []
    excluded_case_ids = set(args.exclude_case_id or [])
    for review_path in sorted(Path(args.review_root).glob("**/review_target.json")):
        payload, image_path, meta_path = _case_paths(review_path)
        case_id = payload.get("case_id", review_path.parent.name)
        if case_id in excluded_case_ids:
            skipped.append({"review_target": str(review_path), "reason": "excluded_case_id", "case_id": case_id})
            continue
        if not image_path or not meta_path:
            skipped.append({"review_target": str(review_path), "reason": "missing_grid_image_or_metadata"})
            continue
        status = payload.get("review_status", "")
        if status == "needs_review" and not payload.get("human_reviewed", False):
            skipped.append({"review_target": str(review_path), "reason": "needs_review_not_human_reviewed"})
            continue
        if status == "auto_pass" and not args.include_auto_pass:
            skipped.append({"review_target": str(review_path), "reason": "auto_pass_excluded"})
            continue
        target = read_assignment_payload(review_path)
        human_reviewed = bool(payload.get("human_reviewed", False))
        training_weight = 1.0 if human_reviewed else 0.5
        source = {
            "review_target_json": str(review_path),
            "review_status": status,
            "human_reviewed": human_reviewed,
            "teacher_mode": payload.get("teacher_mode"),
            "report_available": payload.get("report_available"),
            "selected_candidate_index": payload.get("selected_candidate_index"),
            "selection_reason": payload.get("selection_reason"),
            "candidate_agreement": payload.get("candidate_agreement"),
        }
        record = build_supervision_record(
            case_id,
            image_path,
            meta_path,
            target,
            source=source,
            target_groups=payload.get("target_groups", []),
        )
        record["training_weight"] = training_weight
        rows.append(record)
    summary = {"compiled_count": len(rows), "skipped": skipped}
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, rows)
        summary["output_jsonl"] = args.output_jsonl
    if args.train_jsonl and args.val_jsonl:
        shuffled = list(rows)
        random.Random(args.seed).shuffle(shuffled)
        val_count = int(round(len(shuffled) * max(0.0, min(1.0, args.val_fraction))))
        if len(shuffled) > 1:
            val_count = max(1, val_count)
        val_rows = shuffled[:val_count]
        train_rows = shuffled[val_count:]
        write_jsonl(args.train_jsonl, train_rows)
        write_jsonl(args.val_jsonl, val_rows)
        summary.update(
            {
                "train_jsonl": args.train_jsonl,
                "val_jsonl": args.val_jsonl,
                "train_count": len(train_rows),
                "val_count": len(val_rows),
                "val_fraction": args.val_fraction,
                "seed": args.seed,
            }
        )
    if args.output_summary_json:
        write_json(args.output_summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
