import csv
from pathlib import Path

from adenoma_agent.utils import ensure_dir, read_json, write_json


def evaluate_run(run_dir):
    run_dir = Path(run_dir)
    case_results = []
    for case_dir in sorted(run_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        result_path = case_dir / "case_result.json"
        if result_path.exists():
            case_results.append(read_json(result_path))

    case_count = len(case_results)
    avg_steps = 0.0
    avg_runtime = 0.0
    avg_cost = 0.0
    warn_cases = 0
    fail_cases = 0
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    proxy_cases = 0
    checklist_total = 0.0
    for result in case_results:
        avg_steps += float(result.get("audit", {}).get("metrics", {}).get("trajectory_length", 0))
        avg_runtime += float(result.get("timing", {}).get("total_runtime_ms", 0))
        avg_cost += float(result.get("cost", {}).get("estimated_case_cost_units", 0))
        checklist_total += float(result.get("audit", {}).get("metrics", {}).get("report_checklist_completeness", 0.0))
        if result.get("status") == "warn":
            warn_cases += 1
        if result.get("status") == "fail":
            fail_cases += 1
        if result.get("binary_target") is not None and "positive" in result.get("final_binary_prediction", {}):
            proxy_cases += 1
            target = int(result["binary_target"])
            pred = int(bool(result["final_binary_prediction"]["positive"]))
            if pred == 1 and target == 1:
                confusion["tp"] += 1
            elif pred == 0 and target == 0:
                confusion["tn"] += 1
            elif pred == 1 and target == 0:
                confusion["fp"] += 1
            elif pred == 0 and target == 1:
                confusion["fn"] += 1

    if case_count:
        avg_steps /= case_count
        avg_runtime /= case_count
        avg_cost /= case_count
        checklist_total /= case_count

    accuracy = None
    recall = None
    specificity = None
    if proxy_cases:
        accuracy = round(float(confusion["tp"] + confusion["tn"]) / float(proxy_cases), 4)
        recall = round(float(confusion["tp"]) / float(max(1, confusion["tp"] + confusion["fn"])), 4)
        specificity = round(float(confusion["tn"]) / float(max(1, confusion["tn"] + confusion["fp"])), 4)

    summary = {
        "run_dir": str(run_dir),
        "case_count": case_count,
        "warn_cases": warn_cases,
        "fail_cases": fail_cases,
        "avg_trajectory_length": round(avg_steps, 4),
        "avg_runtime_ms": round(avg_runtime, 4),
        "avg_cost_units": round(avg_cost, 4),
        "avg_report_checklist_completeness": round(checklist_total, 4),
        "proxy_case_count": proxy_cases,
        "ssa_binary_accuracy": accuracy,
        "ssa_recall": recall,
        "ssa_specificity": specificity,
        "confusion_matrix": confusion,
    }
    prediction_csv = run_dir / "case_predictions.csv"
    ensure_dir(prediction_csv.parent)
    with prediction_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "label", "binary_target", "pred_label", "pred_positive", "pred_score", "status"])
        for result in case_results:
            pred = result.get("final_binary_prediction", {})
            writer.writerow(
                [
                    result.get("case_id"),
                    result.get("label"),
                    result.get("binary_target"),
                    pred.get("label"),
                    pred.get("positive"),
                    pred.get("score"),
                    result.get("status"),
                ]
            )
    write_json(run_dir / "evaluation_summary.json", summary)
    return summary
