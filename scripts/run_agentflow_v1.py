#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from adenoma_agent.agentflow.architecture_runtime import (  # noqa: E402
    ArchitectureInferenceRuntime,
    ScriptedArchitecturePredictor,
)
from adenoma_agent.agentflow.contracts import ArchitecturePatchPrediction  # noqa: E402
from adenoma_agent.agentflow.chief import RuleBasedChiefAgent  # noqa: E402
from adenoma_agent.agentflow.evidence import EvidenceEngine  # noqa: E402
from adenoma_agent.agentflow.knowledge import default_knowledge_base  # noqa: E402
from adenoma_agent.agentflow.mucosa_bridge import MucosaEvidenceBridge  # noqa: E402
from adenoma_agent.agentflow.orchestrator import (  # noqa: E402
    AgentFlowOrchestrator,
    ExistingImageROICropper,
    VirtualROICropper,
)
from adenoma_agent.agentflow.reviewer import (  # noqa: E402
    HttpReviewerBackend,
    ScriptedReviewerBackend,
    UnavailableReviewerBackend,
    default_reviewer_registry,
)
from adenoma_agent.agentflow.planner import PlanningAgent  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run the contract-first Agent_workflow v1 control plane.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "agentflow" / "runtime_v1.yaml"),
    )
    parser.add_argument("--case-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--architecture-predictions-jsonl", default="")
    source.add_argument("--five-x-manifest", default="")
    source.add_argument(
        "--mucosa-output-dir",
        default="",
        help="Compact Mucosa Extractor output containing manifest/tile_index/five_x_patch_manifest.",
    )
    parser.add_argument(
        "--architecture-script-json",
        default="",
        help="Explicit non-clinical patch_id -> scripted architecture output mapping.",
    )
    parser.add_argument(
        "--reviewer-script-json",
        default="",
        help="Explicit non-clinical question/reviewer -> response queue mapping.",
    )
    parser.add_argument(
        "--reviewer-endpoint",
        default="",
        help="Existing generic Qwen/PathReasoner /predict endpoint used through ReviewerObservationV1.",
    )
    parser.add_argument("--reviewer-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--reviewer-model-id", default="shared_reviewer_http")
    parser.add_argument("--reviewer-model-version", default="unreported")
    parser.add_argument("--allow-synthetic-stub", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slide-width", type=int, default=0)
    parser.add_argument("--slide-height", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=None)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def repository_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_predictions(path):
    output = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                output.append(ArchitecturePatchPrediction.from_dict(json.loads(line)))
    return tuple(output)


def main():
    args = parse_args()
    runtime_config = read_yaml(args.config)
    architecture_config = runtime_config.get("architecture", {})
    planner_config = runtime_config.get("planner", {})
    reviewer_config = runtime_config.get("reviewers", {})
    chief_config = runtime_config.get("chief", {})
    registry = default_reviewer_registry(
        repository_path(
            reviewer_config.get(
                "registry",
                "configs/agentflow/reviewer_registry_v1.json",
            )
        )
    )
    knowledge_base = default_knowledge_base(
        repository_path(
            chief_config.get(
                "knowledge_base",
                "configs/agentflow/knowledge_base_v1.json",
            )
        )
    )
    planner = PlanningAgent(
        knowledge_base=knowledge_base,
        registry=registry,
        action_weights=planner_config.get("action_weights"),
        margin_threshold=float(planner_config.get("top_hypothesis_margin", 0.10)),
        evidence_margin_threshold=float(planner_config.get("evidence_score_margin", 0.50)),
        minimum_evidence_quality=float(
            planner_config.get("minimum_required_evidence_quality", 0.50)
        ),
    )
    chief_agent = RuleBasedChiefAgent(
        knowledge_base=knowledge_base,
        max_conflict_rounds=int(planner_config.get("max_conflict_resolution_rounds", 2)),
    )
    uses_stub = bool(args.architecture_script_json or args.reviewer_script_json)
    if uses_stub and not args.allow_synthetic_stub:
        raise SystemExit("Synthetic inputs require --allow-synthetic-stub")
    initial_evidence = tuple()
    five_x_manifest = args.five_x_manifest
    if args.mucosa_output_dir:
        bridge_payload = MucosaEvidenceBridge().load(args.case_id, args.mucosa_output_dir)
        five_x_manifest = bridge_payload["five_x_manifest_path"]
        initial_evidence = bridge_payload["evidence_records"]
    if args.architecture_predictions_jsonl:
        predictions = read_predictions(args.architecture_predictions_jsonl)
        if any(bool(item.metadata.get("synthetic_stub")) for item in predictions) and not args.allow_synthetic_stub:
            raise SystemExit("Synthetic architecture predictions require --allow-synthetic-stub")
        missing_embeddings = [
            item.patch_id
            for item in predictions
            if not item.embedding_ref and not bool(item.metadata.get("synthetic_stub"))
        ]
        if missing_embeddings:
            raise SystemExit(
                "Production architecture predictions require embedding_ref provenance: {0}".format(
                    missing_embeddings
                )
            )
    else:
        if args.architecture_script_json:
            predictor = ScriptedArchitecturePredictor(read_json(args.architecture_script_json))
            architecture_runtime = ArchitectureInferenceRuntime(
                predictor=predictor,
                min_mucosa_coverage=float(
                    architecture_config.get("inference_min_mucosa_coverage", 0.30)
                ),
            )
        else:
            architecture_runtime = ArchitectureInferenceRuntime(
                min_mucosa_coverage=float(
                    architecture_config.get("inference_min_mucosa_coverage", 0.30)
                )
            )
        predictions = architecture_runtime.run(five_x_manifest)
    if args.reviewer_script_json:
        reviewer_backend = ScriptedReviewerBackend(read_json(args.reviewer_script_json))
        cropper = VirtualROICropper()
    elif args.reviewer_endpoint:
        reviewer_backend = HttpReviewerBackend(
            endpoint=args.reviewer_endpoint,
            timeout_seconds=args.reviewer_timeout_seconds,
            model_id=args.reviewer_model_id,
            model_version=args.reviewer_model_version,
        )
        cropper = ExistingImageROICropper()
    else:
        reviewer_backend = UnavailableReviewerBackend()
        cropper = ExistingImageROICropper()
    dimensions = None
    if args.slide_width > 0 and args.slide_height > 0:
        dimensions = (args.slide_width, args.slide_height)
    result = AgentFlowOrchestrator(
        planner=planner,
        reviewer_backend=reviewer_backend,
        registry=registry,
        chief_agent=chief_agent,
        cropper=cropper,
        max_actions=int(
            args.max_actions
            if args.max_actions is not None
            else planner_config.get("max_actions", 8)
        ),
        max_retries_per_reviewer=int(reviewer_config.get("max_retries_per_reviewer", 1)),
        max_schema_repairs=int(reviewer_config.get("max_schema_repairs", 1)),
        require_contract_validator=bool(
            reviewer_config.get("json_schema_validation", {}).get(
                "required_in_production",
                True,
            )
        )
        and not bool(getattr(reviewer_backend, "synthetic_stub", False)),
        evidence_engine=EvidenceEngine(
            knowledge_base=knowledge_base,
            policy=runtime_config.get("evidence_engine", {}),
        ),
    ).run(
        case_id=args.case_id,
        predictions=predictions,
        initial_evidence=initial_evidence,
        slide_dimensions=dimensions,
        output_dir=Path(args.output_dir).resolve(),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
