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
    serrated_confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    ssl_like_confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    dysplasia_confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    serrated_cases = 0
    ssl_like_cases = 0
    dysplasia_cases = 0
    serrated_checklist_total = 0.0
    ssl_like_checklist_total = 0.0
    dysplasia_checklist_total = 0.0
    for result in case_results:
        avg_steps += float(result.get("audit", {}).get("metrics", {}).get("trajectory_length", 0))
        avg_runtime += float(result.get("timing", {}).get("total_runtime_ms", 0))
        avg_cost += float(result.get("cost", {}).get("estimated_case_cost_units", 0))
        serrated_checklist_total += float(result.get("audit", {}).get("metrics", {}).get("serrated_checklist_completeness", 0.0))
        ssl_like_checklist_total += float(result.get("audit", {}).get("metrics", {}).get("ssl_like_checklist_completeness", 0.0))
        dysplasia_checklist_total += float(result.get("audit", {}).get("metrics", {}).get("dysplasia_checklist_completeness", 0.0))
        if result.get("status") == "warn":
            warn_cases += 1
        if result.get("status") == "fail":
            fail_cases += 1
        hierarchy = result.get("hierarchical_prediction", {})
        serrated_pred = hierarchy.get("serrated_lesion_assessment", {}).get("positive")
        ssl_like_pred = hierarchy.get("ssl_like_architecture_assessment", {}).get("positive")
        dysplasia_pred = hierarchy.get("dysplasia_assessment", {}).get("positive")
        if result.get("serrated_target") is not None and serrated_pred is not None:
            serrated_cases += 1
            _update_confusion(serrated_confusion, int(result["serrated_target"]), int(bool(serrated_pred)))
        if result.get("ssl_like_target") is not None and ssl_like_pred is not None:
            ssl_like_cases += 1
            _update_confusion(ssl_like_confusion, int(result["ssl_like_target"]), int(bool(ssl_like_pred)))
        if result.get("dysplasia_proxy_target") is not None and dysplasia_pred is not None:
            dysplasia_cases += 1
            _update_confusion(dysplasia_confusion, int(result["dysplasia_proxy_target"]), int(bool(dysplasia_pred)))

    if case_count:
        avg_steps /= case_count
        avg_runtime /= case_count
        avg_cost /= case_count
        serrated_checklist_total /= case_count
        ssl_like_checklist_total /= case_count
        dysplasia_checklist_total /= case_count

    summary = {
        "run_dir": str(run_dir),
        "case_count": case_count,
        "warn_cases": warn_cases,
        "fail_cases": fail_cases,
        "avg_trajectory_length": round(avg_steps, 4),
        "avg_runtime_ms": round(avg_runtime, 4),
        "avg_cost_units": round(avg_cost, 4),
        "avg_serrated_checklist_completeness": round(serrated_checklist_total, 4),
        "avg_ssl_like_checklist_completeness": round(ssl_like_checklist_total, 4),
        "avg_dysplasia_checklist_completeness": round(dysplasia_checklist_total, 4),
        "serrated_case_count": serrated_cases,
        "ssl_like_case_count": ssl_like_cases,
        "dysplasia_case_count": dysplasia_cases,
        "serrated_accuracy": _accuracy(serrated_confusion, serrated_cases),
        "ssl_like_accuracy": _accuracy(ssl_like_confusion, ssl_like_cases),
        "dysplasia_proxy_accuracy": _accuracy(dysplasia_confusion, dysplasia_cases),
        "serrated_recall": _recall(serrated_confusion),
        "ssl_like_recall": _recall(ssl_like_confusion),
        "dysplasia_proxy_recall": _recall(dysplasia_confusion),
        "serrated_specificity": _specificity(serrated_confusion),
        "ssl_like_specificity": _specificity(ssl_like_confusion),
        "dysplasia_proxy_specificity": _specificity(dysplasia_confusion),
        "serrated_confusion_matrix": serrated_confusion,
        "ssl_like_confusion_matrix": ssl_like_confusion,
        "dysplasia_confusion_matrix": dysplasia_confusion,
    }
    prediction_csv = run_dir / "case_predictions.csv"
    ensure_dir(prediction_csv.parent)
    with prediction_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "label",
                "serrated_target",
                "ssl_like_target",
                "dysplasia_proxy_target",
                "serrated_pred_label",
                "serrated_pred_positive",
                "serrated_pred_score",
                "ssl_like_pred_label",
                "ssl_like_pred_positive",
                "ssl_like_pred_score",
                "dysplasia_pred_label",
                "dysplasia_pred_positive",
                "dysplasia_pred_score",
                "status",
            ]
        )
        for result in case_results:
            pred = result.get("hierarchical_prediction", {})
            serrated_pred = pred.get("serrated_lesion_assessment", {})
            ssl_like_pred = pred.get("ssl_like_architecture_assessment", {})
            dysplasia_pred = pred.get("dysplasia_assessment", {})
            writer.writerow(
                [
                    result.get("case_id"),
                    result.get("label"),
                    result.get("serrated_target"),
                    result.get("ssl_like_target"),
                    result.get("dysplasia_proxy_target"),
                    serrated_pred.get("label"),
                    serrated_pred.get("positive"),
                    serrated_pred.get("score"),
                    ssl_like_pred.get("label"),
                    ssl_like_pred.get("positive"),
                    ssl_like_pred.get("score"),
                    dysplasia_pred.get("label"),
                    dysplasia_pred.get("positive"),
                    dysplasia_pred.get("score"),
                    result.get("status"),
                ]
            )
    write_json(run_dir / "evaluation_summary.json", summary)
    return summary


def _update_confusion(confusion, target, pred):
    if pred == 1 and target == 1:
        confusion["tp"] += 1
    elif pred == 0 and target == 0:
        confusion["tn"] += 1
    elif pred == 1 and target == 0:
        confusion["fp"] += 1
    elif pred == 0 and target == 1:
        confusion["fn"] += 1


def _accuracy(confusion, case_count):
    if not case_count:
        return None
    return round(float(confusion["tp"] + confusion["tn"]) / float(case_count), 4)


def _recall(confusion):
    return round(float(confusion["tp"]) / float(max(1, confusion["tp"] + confusion["fn"])), 4)


def _specificity(confusion):
    return round(float(confusion["tn"]) / float(max(1, confusion["tn"] + confusion["fp"])), 4)
