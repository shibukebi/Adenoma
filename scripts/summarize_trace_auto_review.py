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

from adenoma_agent.utils import write_json  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize auto_checked trace review results into dashboard-style metrics.")
    parser.add_argument("--run-root", required=True, help="Root containing per-case auto_checked directories.")
    parser.add_argument("--output-json", required=True)
    return parser


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_rate(numerator, denominator):
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _agreement_bucket(value):
    if value is None:
        return "unknown"
    if value < 0.5:
        return "<0.5"
    if value < 0.7:
        return "0.5-0.7"
    if value < 0.85:
        return "0.7-0.85"
    return ">=0.85"


def _case_row(case_dir):
    auto_review_path = case_dir / "auto_checked" / "auto_review.json"
    review_target_path = case_dir / "review" / "review_target.json"
    if not auto_review_path.exists():
        return None
    auto_review = _load_json(auto_review_path)
    review_target = _load_json(review_target_path) if review_target_path.exists() else {}
    selected = auto_review.get("selected_candidate") or {}
    score = selected.get("score", {})
    structure = score.get("structure", {})
    slide_label_consistency = score.get("slide_label_consistency", {})
    slide_warnings = slide_label_consistency.get("warnings", [])
    return {
        "case_id": case_dir.name,
        "review_status": (auto_review.get("selection") or {}).get("review_status"),
        "review_reason": (auto_review.get("selection") or {}).get("review_reason"),
        "selected_candidate_index": (auto_review.get("selection") or {}).get("selected_index"),
        "coverage_ok": structure.get("coverage_ok"),
        "missing_count": len(structure.get("missing_patch_ids") or []),
        "duplicate_count": len(structure.get("duplicate_patch_ids") or []),
        "unexpected_count": len(structure.get("unexpected_patch_ids") or []),
        "parse_failures": sum(1 for item in auto_review.get("candidates", []) if item.get("parse_failure")),
        "agreement_rate": (auto_review.get("candidate_agreement") or {}).get("agreement_rate"),
        "agreement_bucket": _agreement_bucket((auto_review.get("candidate_agreement") or {}).get("agreement_rate")),
        "total_score": score.get("total_score"),
        "semantic_score": (score.get("semantics") or {}).get("semantic_score"),
        "field_consistency_score": (score.get("field_consistency") or {}).get("field_consistency_score"),
        "aggregation_score": (score.get("aggregation") or {}).get("aggregation_score"),
        "slide_label": slide_label_consistency.get("slide_label"),
        "slide_label_warning_count": len(slide_warnings),
        "slide_label_warnings": [item.get("warning") for item in slide_warnings],
        "human_reviewed": bool(review_target.get("human_reviewed", False)),
    }


def main():
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    rows = []
    for case_dir in sorted([item for item in run_root.iterdir() if item.is_dir()]):
        row = _case_row(case_dir)
        if row:
            rows.append(row)

    completed = len(rows)
    coverage_ok_count = sum(1 for row in rows if row["coverage_ok"] is True)
    parse_failure_case_count = sum(1 for row in rows if row["parse_failures"] > 0)
    parse_failure_total = sum(row["parse_failures"] for row in rows)
    slide_label_risk_count = sum(1 for row in rows if row["slide_label_warning_count"] > 0)
    needs_review_count = sum(1 for row in rows if row["review_status"] == "needs_review")
    auto_pass_count = sum(1 for row in rows if row["review_status"] == "auto_pass")
    human_reviewed_count = sum(1 for row in rows if row["human_reviewed"])
    agreement_values = [row["agreement_rate"] for row in rows if isinstance(row["agreement_rate"], (int, float))]
    total_scores = [row["total_score"] for row in rows if isinstance(row["total_score"], (int, float))]
    agreement_distribution = {}
    for row in rows:
        agreement_distribution[row["agreement_bucket"]] = agreement_distribution.get(row["agreement_bucket"], 0) + 1
    slide_label_warning_distribution = {}
    review_reason_distribution = {}
    for row in rows:
        for warning in row["slide_label_warnings"]:
            slide_label_warning_distribution[warning] = slide_label_warning_distribution.get(warning, 0) + 1
        for reason in str(row.get("review_reason") or "").split(","):
            reason = reason.strip()
            if reason:
                review_reason_distribution[reason] = review_reason_distribution.get(reason, 0) + 1
    summary = {
        "run_root": str(run_root),
        "case_count": completed,
        "coverage_ok_rate": _safe_rate(coverage_ok_count, completed),
        "coverage_ok_count": coverage_ok_count,
        "coverage_fail_count": completed - coverage_ok_count,
        "parse_failure_case_count": parse_failure_case_count,
        "parse_failure_total": parse_failure_total,
        "slide_label_risk_count": slide_label_risk_count,
        "needs_review_count": needs_review_count,
        "auto_pass_count": auto_pass_count,
        "human_reviewed_count": human_reviewed_count,
        "agreement_mean": _safe_rate(sum(agreement_values), len(agreement_values)) if agreement_values else None,
        "agreement_min": min(agreement_values) if agreement_values else None,
        "agreement_max": max(agreement_values) if agreement_values else None,
        "score_mean": _safe_rate(sum(total_scores), len(total_scores)) if total_scores else None,
        "score_min": min(total_scores) if total_scores else None,
        "score_max": max(total_scores) if total_scores else None,
        "agreement_distribution": agreement_distribution,
        "slide_label_warning_distribution": slide_label_warning_distribution,
        "review_reason_distribution": review_reason_distribution,
        "cases": rows,
    }
    write_json(args.output_json, summary)
    print(json.dumps({"output_json": args.output_json, "case_count": completed, "coverage_ok_rate": summary["coverage_ok_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
