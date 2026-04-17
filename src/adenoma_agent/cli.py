import argparse
from pathlib import Path

from adenoma_agent.adapters.manifest import AdenomaManifestAdapter
from adenoma_agent.config import load_bundle
from adenoma_agent.eval import evaluate_run
from adenoma_agent.orchestrator import AdenomaAgentOrchestrator
from adenoma_agent.replay import build_replay_report, write_replay_report
from adenoma_agent.schemas import CaseSpec, InterventionEvent
from adenoma_agent.utils import ensure_dir, write_json


def build_parser():
    parser = argparse.ArgumentParser(description="adenoma_agent MVP CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_case = subparsers.add_parser("run-case", help="Run one case through the MVP pipeline.")
    run_case.add_argument("--case-id", default=None)
    run_case.add_argument("--slide-path", default=None)
    run_case.add_argument("--output-root", required=True)
    run_case.add_argument("--runtime-config", default=None)
    run_case.add_argument("--budget-config", default=None)
    run_case.add_argument("--trace-mode", default=None, choices=["auto", "patho-r1", "heuristic", "manual"])
    run_case.add_argument("--override-box-thumb", default=None, help="Manual thumbnail box spec x1,y1,x2,y2;...")
    run_case.add_argument("--force-zoom", type=float, default=None)
    run_case.add_argument("--early-stop-after", type=int, default=None)
    run_case.add_argument("--operator-note", default=None)

    run_batch = subparsers.add_parser("run-batch", help="Run a batch of cases from the manifest.")
    run_batch.add_argument("--output-root", required=True)
    run_batch.add_argument("--runtime-config", default=None)
    run_batch.add_argument("--budget-config", default=None)
    run_batch.add_argument("--limit", type=int, default=None)
    run_batch.add_argument("--trace-mode", default=None, choices=["auto", "patho-r1", "heuristic", "manual"])

    build_pilot = subparsers.add_parser("build-pilot", help="Build a deterministic SSA-vs-others pilot subset.")
    build_pilot.add_argument("--runtime-config", default=None)
    build_pilot.add_argument("--budget-config", default=None)
    build_pilot.add_argument("--positives", type=int, default=12)
    build_pilot.add_argument("--negatives", type=int, default=12)
    build_pilot.add_argument("--output-json", default=None)

    replay_case = subparsers.add_parser("replay-case", help="Build a replay summary for a finished case.")
    replay_case.add_argument("--case-dir", required=True)
    replay_case.add_argument("--write", action="store_true")

    eval_run = subparsers.add_parser("eval-run", help="Aggregate a finished run directory.")
    eval_run.add_argument("--run-dir", required=True)
    return parser


def load_case_from_args(bundle, args):
    adapter = AdenomaManifestAdapter(
        bundle["runtime"]["data"]["manifest_csv"],
        bundle["runtime"]["data"]["labels_csv"],
        positive_label=bundle["runtime"]["data"]["binary_positive_label"],
    )
    if args.case_id:
        return adapter.get_case(args.case_id)
    if args.slide_path:
        slide_path = str(Path(args.slide_path))
        case_id = Path(slide_path).stem
        return CaseSpec(
            case_id=case_id,
            slide_path=slide_path,
            task_type="ssa_vs_others_huge_region_agent",
            question=(
                "Review this whole-slide image for SSA versus others. Focus on serration to crypt base, mucus cap, "
                "abnormal maturation, basal dilatation, crypt branching, horizontal growth, and boot/L/T-shaped crypts."
            ),
            label=None,
            binary_target=None,
            metadata={},
        )
    raise SystemExit("Either --case-id or --slide-path is required.")


def command_run_case(args):
    bundle = load_bundle(args.runtime_config, args.budget_config)
    case_spec = load_case_from_args(bundle, args)
    orchestrator = AdenomaAgentOrchestrator(bundle)
    interventions = InterventionEvent(
        override_roi=args.override_box_thumb,
        force_zoom=args.force_zoom,
        early_stop=args.early_stop_after,
        operator_note=args.operator_note,
    )
    output_root = ensure_dir(args.output_root)
    result = orchestrator.run_case(case_spec, output_root, interventions=interventions, trace_mode=args.trace_mode)
    print("case_dir={0}".format(result["case_dir"]))
    print("case_result={0}".format(result["case_result_path"]))


def command_run_batch(args):
    bundle = load_bundle(args.runtime_config, args.budget_config)
    adapter = AdenomaManifestAdapter(
        bundle["runtime"]["data"]["manifest_csv"],
        bundle["runtime"]["data"]["labels_csv"],
        positive_label=bundle["runtime"]["data"]["binary_positive_label"],
    )
    orchestrator = AdenomaAgentOrchestrator(bundle)
    output_root = ensure_dir(args.output_root)
    summary = {"completed": [], "failed": []}
    cases = adapter.list_cases()
    if args.limit is not None:
        cases = cases[: args.limit]
    for case_spec in cases:
        try:
            result = orchestrator.run_case(case_spec, output_root, trace_mode=args.trace_mode)
            summary["completed"].append(str(result["case_result_path"]))
        except Exception as exc:
            summary["failed"].append({"case_id": case_spec.case_id, "error": str(exc)})
    summary_path = write_json(Path(output_root) / "batch_summary.json", summary)
    print("batch_summary={0}".format(summary_path))


def command_replay_case(args):
    if args.write:
        path = write_replay_report(args.case_dir)
        print("replay_report={0}".format(path))
    else:
        print(build_replay_report(args.case_dir))


def command_build_pilot(args):
    bundle = load_bundle(args.runtime_config, args.budget_config)
    adapter = AdenomaManifestAdapter(
        bundle["runtime"]["data"]["manifest_csv"],
        bundle["runtime"]["data"]["labels_csv"],
        positive_label=bundle["runtime"]["data"]["binary_positive_label"],
    )
    output_json = args.output_json or str(
        Path(bundle["runtime"]["data"]["pilot_root"])
        / "ssa_vs_others_pilot_p{0}_n{1}.json".format(args.positives, args.negatives)
    )
    payload = adapter.build_pilot_subset(output_json, positives=args.positives, negatives=args.negatives)
    print("pilot_subset={0}".format(output_json))
    print(payload["selected_case_count"])


def command_eval_run(args):
    summary = evaluate_run(args.run_dir)
    print("evaluation_summary={0}".format(Path(args.run_dir) / "evaluation_summary.json"))
    print(summary)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run-case":
        command_run_case(args)
    elif args.command == "run-batch":
        command_run_batch(args)
    elif args.command == "build-pilot":
        command_build_pilot(args)
    elif args.command == "replay-case":
        command_replay_case(args)
    elif args.command == "eval-run":
        command_eval_run(args)
    else:
        raise SystemExit("Unknown command: {0}".format(args.command))


if __name__ == "__main__":
    main()
