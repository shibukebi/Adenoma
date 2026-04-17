from pathlib import Path

from adenoma_agent.utils import read_json, read_jsonl, write_text


def build_replay_report(case_dir):
    case_dir = Path(case_dir)
    events_path = case_dir / "events.jsonl"
    result_path = case_dir / "case_result.json"
    events = read_jsonl(events_path)
    result = read_json(result_path) if result_path.exists() else {}

    lines = []
    lines.append("# Replay for {0}".format(case_dir.name))
    lines.append("")
    lines.append("## Timeline")
    lines.append("")
    for event in events:
        lines.append(
            "- [{0}] {1}/{2} status={3} input={4} output={5}".format(
                event["timestamp"],
                event["state"],
                event["agent"],
                event["status"],
                event.get("input_ref"),
                event.get("output_ref"),
            )
        )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append("- status: {0}".format(result.get("status")))
    lines.append("- trace_clusters: {0}".format(len(result.get("trace_clusters", []))))
    lines.append("- trajectory_steps: {0}".format(len(result.get("trajectory", []))))
    lines.append("- target: {0}".format(result.get("binary_target")))
    lines.append("- prediction: {0}".format(result.get("final_binary_prediction", {}).get("label")))
    lines.append("- prediction_score: {0}".format(result.get("final_binary_prediction", {}).get("score")))
    lines.append("- warning_count: {0}".format(len(result.get("audit", {}).get("warnings", []))))
    lines.append("- error_count: {0}".format(len(result.get("audit", {}).get("errors", []))))
    lines.append("")
    lines.append("## Checklist")
    lines.append("")
    for criterion, payload in sorted(result.get("report_checklist", {}).items()):
        lines.append("- {0}: {1}".format(criterion, payload.get("status")))
    return "\n".join(lines) + "\n"


def write_replay_report(case_dir):
    report = build_replay_report(case_dir)
    path = Path(case_dir) / "replay.md"
    write_text(path, report)
    return path
