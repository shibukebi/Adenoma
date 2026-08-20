#!/usr/bin/env python3
"""Audit a deterministic real-slide cohort up to the first honest boundary.

This entrypoint never substitutes scripted Architecture or Reviewer evidence.
It can run the real Mucosa extractor when explicitly requested, then stops at
the first unavailable production dependency and records that boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from adenoma_agent.agentflow.wsi_runtime import (  # noqa: E402
    WSIROICropper,
    load_label_workbook_eligibility,
    select_deterministic_cohort,
    write_integration_boundary,
)
from adenoma_agent.agentflow.contracts import ArchitecturePatchPrediction  # noqa: E402
from adenoma_agent.agentflow.architecture_runtime import load_five_x_manifest  # noqa: E402
from adenoma_agent.agentflow.orchestrator import AgentFlowOrchestrator  # noqa: E402
from adenoma_agent.agentflow.reviewer import HttpReviewerBackend  # noqa: E402
from adenoma_agent.agentflow.state import AgentTraceStore, StateStore  # noqa: E402
from adenoma_agent.mucosa_extractor import MucosaExtractorConfig, run_mucosa_extractor  # noqa: E402
from adenoma_agent.wsi import WSIPhysicalMetadataError, WSIReader, WSIReaderUnavailableError  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit deterministic Adenoma_hp/Adenoma_yx real cases without synthetic evidence."
    )
    parser.add_argument(
        "--hp-root",
        default="/mnt/zhengke_usb2/yuexin_data/Adenoma_hp",
    )
    parser.add_argument(
        "--yx-root",
        default="/mnt/zhengke_usb2/yuexin_data/Adenoma_yx",
    )
    parser.add_argument("--per-source", type=int, default=1)
    parser.add_argument(
        "--label-workbook",
        default=str(REPO_ROOT / "data" / "label" / "Adenoma_filtered.xlsx"),
        help="Evaluation metadata used only as a boolean Adenoma_yx eligibility filter.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-magnification", type=float, default=0.0)
    parser.add_argument("--mpp", type=float, default=0.0)
    parser.add_argument("--run-mucosa", action="store_true")
    parser.add_argument(
        "--existing-mucosa-output-dir",
        default="",
        help="Existing real UNI+PathPrism Mucosa output; imported only after source WSI SHA-256 verification.",
    )
    parser.add_argument("--pathprism-url", default="http://127.0.0.1:8400/predict")
    parser.add_argument("--mucosa-batch-size", type=int, default=64)
    parser.add_argument(
        "--architecture-predictions-dir",
        default="",
        help="Optional directory containing real <case_alias>.jsonl Architecture predictions.",
    )
    parser.add_argument("--reviewer-endpoint", default="")
    parser.add_argument("--reviewer-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--reviewer-model-id", default="shared_reviewer_http")
    parser.add_argument("--reviewer-model-version", default="unreported")
    parser.add_argument("--max-actions", type=int, default=8)
    return parser.parse_args()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_existing_mucosa_case(case, existing_output_dir, output_dir):
    existing_output_dir = Path(existing_output_dir).resolve()
    source_manifest_path = existing_output_dir / "manifest.json"
    source_five_x_path = existing_output_dir / "five_x_patch_manifest.jsonl"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    matching_slides = [
        item
        for item in source_manifest.get("slides", [])
        if str(item.get("slide_id")) == case.slide_path.stem
    ]
    if len(matching_slides) != 1:
        raise ValueError("Existing Mucosa output does not contain exactly one matching real WSI")
    artifact_wsi_path = Path(matching_slides[0].get("wsi_path", ""))
    if not artifact_wsi_path.is_file():
        raise FileNotFoundError("Existing Mucosa source WSI is unavailable for identity verification")
    source_sha256 = _sha256(case.slide_path)
    artifact_sha256 = _sha256(artifact_wsi_path)
    if source_sha256 != artifact_sha256:
        raise ValueError("Existing Mucosa output belongs to a different WSI")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_rows = []
    local_patch_map = []
    with source_five_x_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("slide_id")) != case.slide_path.stem:
                continue
            safe_patch_id = "{0}__5x__{1:06d}".format(case.case_alias, len(safe_rows))
            safe_row = dict(row)
            safe_row["slide_id"] = case.case_alias
            safe_row["patch_id"] = safe_patch_id
            safe_rows.append(safe_row)
            local_patch_map.append(
                {"safe_patch_id": safe_patch_id, "source_patch_id": row.get("patch_id")}
            )
    if not safe_rows:
        raise ValueError("Existing Mucosa output has no 5x rows for the selected WSI")
    five_x_path = output_dir / "five_x_patch_manifest.jsonl"
    with five_x_path.open("w", encoding="utf-8") as handle:
        for row in safe_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    public_manifest = {
        "schema_version": "mucosa_extractor_safe_import_v1",
        "status": "complete",
        "case_alias": case.case_alias,
        "source_code": case.source_code,
        "model": dict(source_manifest.get("model", {})),
        "canonical_output": "five_x_patch_manifest.jsonl",
        "counts": {"slides": 1, "five_x_patches": len(safe_rows), "errors": 0},
        "source_wsi_sha256_verified": True,
        "non_clinical": True,
    }
    _write_json(output_dir / "manifest.json", public_manifest)
    _write_json(
        output_dir / "local_provenance.json",
        {
            "schema_version": "mucosa_safe_import_local_provenance_v1",
            "case_alias": case.case_alias,
            "source_name": case.source_name,
            "source_wsi_path": str(case.slide_path.resolve()),
            "artifact_wsi_path": str(artifact_wsi_path.resolve()),
            "source_mucosa_output": str(existing_output_dir),
            "source_wsi_sha256": source_sha256,
            "patch_id_map": local_patch_map,
        },
    )
    return public_manifest


def _alias_input(case, case_dir):
    input_dir = case_dir / "local_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    alias_path = input_dir / "{0}{1}".format(case.case_alias, case.slide_path.suffix.lower())
    if alias_path.exists() or alias_path.is_symlink():
        if alias_path.resolve() != case.slide_path.resolve():
            raise RuntimeError("Existing alias points to a different source: {0}".format(alias_path))
    else:
        alias_path.symlink_to(case.slide_path.resolve())
    return alias_path


def _metadata_payload(case, reader, alias_path):
    return {
        "schema_version": "real_wsi_metadata_v1",
        "case_alias": case.case_alias,
        "source_code": case.source_code,
        "source_format": case.source_format,
        "inference_input": {
            "case_alias": case.case_alias,
            "slide_alias_path": str(alias_path.absolute()),
        },
        "reader": {
            "backend": reader.backend_name,
            "dimensions_level0": list(reader.dimensions),
            "level_dimensions": [list(row) for row in reader.level_dimensions],
            "level_downsamples": list(reader.level_downsamples),
            "base_magnification": reader.base_magnification,
            "base_magnification_source": reader.base_magnification_source,
            "mpp_x": reader.mpp_x,
            "mpp_y": reader.mpp_y,
            "mpp_source": reader.mpp_source,
        },
    }


def _read_architecture_predictions(path):
    output = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            prediction = ArchitecturePatchPrediction.from_dict(json.loads(line))
            if bool(prediction.metadata.get("synthetic_stub")):
                raise ValueError(
                    "Real smoke rejects synthetic Architecture prediction at line {0}".format(
                        line_number
                    )
                )
            if not prediction.embedding_ref:
                raise ValueError(
                    "Real Architecture prediction requires embedding_ref at line {0}".format(
                        line_number
                    )
                )
            output.append(prediction)
    if not output:
        raise ValueError("Architecture prediction JSONL is empty")
    return tuple(output)


def _validate_predictions_against_manifest(predictions, manifest_path, case_alias):
    manifest_rows = load_five_x_manifest(manifest_path)
    by_patch = {item.patch_id: item for item in predictions}
    manifest_by_patch = {str(item["patch_id"]): item for item in manifest_rows}
    if set(by_patch) != set(manifest_by_patch):
        raise ValueError("Architecture predictions do not exactly match the canonical 5x manifest")
    for patch_id, prediction in by_patch.items():
        row = manifest_by_patch[patch_id]
        if prediction.slide_id != case_alias:
            raise ValueError("Real smoke Architecture slide_id must be the source-safe case alias")
        if tuple(prediction.level0_bbox) != tuple(row["level0_bbox"]):
            raise ValueError("Architecture prediction bbox does not match the 5x manifest")
        if abs(float(prediction.mucosa_coverage) - float(row["mucosa_coverage"])) > 1e-6:
            raise ValueError("Architecture prediction coverage does not match the 5x manifest")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible_yx_stems = load_label_workbook_eligibility(args.label_workbook)

    def eligible(source_name, path):
        if source_name != "Adenoma_yx":
            return True
        return path.stem in eligible_yx_stems

    def physically_readable(source_name, path):
        try:
            with WSIReader(
                path,
                base_magnification=(args.base_magnification or None),
                mpp=(args.mpp or None),
                allow_raster_fixture=False,
            ) as candidate_reader:
                candidate_reader.require_physical_metadata()
            return True
        except WSIReaderUnavailableError:
            # Preserve the first deterministic iSyntax case when the legal
            # reader itself is missing so the source receives an explicit
            # dependency_blocked boundary rather than disappearing.
            return source_name == "Adenoma_hp" and path.suffix.lower() == ".isyntax"
        except (WSIPhysicalMetadataError, OSError, RuntimeError, ValueError):
            return False

    cohort = select_deterministic_cohort(
        {"Adenoma_hp": Path(args.hp_root), "Adenoma_yx": Path(args.yx_root)},
        per_source=args.per_source,
        eligibility_predicate=eligible,
        validation_predicate=physically_readable,
        suffixes_by_source={"Adenoma_hp": (".isyntax",), "Adenoma_yx": (".svs",)},
    )
    case_results = {}
    boundary_path = output_dir / "integration_boundary.json"
    architecture_dir = Path(args.architecture_predictions_dir).resolve() if args.architecture_predictions_dir else None

    for case in cohort:
        case_dir = output_dir / "cases" / case.case_alias
        case_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "dependency_blocked",
            "stages_reached": ["source_discovery"],
            "integration_boundary": "wsi_open",
            "dependency_blocked": [],
            "artifacts": {},
        }
        case_results[case.case_alias] = result
        try:
            alias_path = _alias_input(case, case_dir)
            with WSIReader(
                alias_path,
                base_magnification=(args.base_magnification or None),
                mpp=(args.mpp or None),
                allow_raster_fixture=False,
            ) as reader:
                result["stages_reached"].append("wsi_open")
                metadata_path = case_dir / "wsi_metadata.json"
                _write_json(metadata_path, _metadata_payload(case, reader, alias_path))
                result["artifacts"]["wsi_metadata"] = str(metadata_path)
                try:
                    reader.require_physical_metadata()
                except WSIPhysicalMetadataError as exc:
                    result["integration_boundary"] = "physical_metadata"
                    result["dependency_blocked"].append(
                        {"code": "physical_metadata_unavailable", "message": str(exc)}
                    )
                    continue
                result["stages_reached"].append("physical_metadata")
                result["integration_boundary"] = "mucosa_extraction"
        except (WSIReaderUnavailableError, WSIPhysicalMetadataError) as exc:
            result["dependency_blocked"].append(
                {"code": "wsi_reader_unavailable", "message": str(exc)}
            )
            write_integration_boundary(boundary_path, cohort, case_results)
            continue
        except Exception as exc:
            result["status"] = "failed"
            result["dependency_blocked"].append(
                {"code": exc.__class__.__name__, "message": str(exc)}
            )
            write_integration_boundary(boundary_path, cohort, case_results)
            continue

        if not args.run_mucosa and not args.existing_mucosa_output_dir:
            result["dependency_blocked"].append(
                {
                    "code": "mucosa_not_requested",
                    "message": "Pass --run-mucosa with a reachable real PathPrism service to continue.",
                }
            )
            write_integration_boundary(boundary_path, cohort, case_results)
            continue

        try:
            mucosa_dir = case_dir / "mucosa_extractor"
            if args.existing_mucosa_output_dir:
                manifest = _import_existing_mucosa_case(
                    case,
                    args.existing_mucosa_output_dir,
                    mucosa_dir,
                )
            else:
                manifest = run_mucosa_extractor(
                    wsi_paths=[alias_path],
                    output_dir=mucosa_dir,
                    config=MucosaExtractorConfig(
                        pathprism_url=args.pathprism_url,
                        batch_size=args.mucosa_batch_size,
                        base_magnification=args.base_magnification,
                        mpp=args.mpp,
                        resume=True,
                    ),
                )
            result["stages_reached"].extend(["mucosa_extraction", "five_x_manifest"])
            result["artifacts"]["mucosa_manifest"] = str(mucosa_dir / "manifest.json")
            result["artifacts"]["five_x_manifest"] = str(mucosa_dir / "five_x_patch_manifest.jsonl")
            result["artifacts"]["mucosa_summary"] = {
                "valid_tiles": int(manifest.get("counts", {}).get("valid_tiles", 0)),
                "five_x_candidates": int(
                    manifest.get("counts", {}).get(
                        "five_x_candidates",
                        manifest.get("counts", {}).get("five_x_patches", 0),
                    )
                ),
                "source_wsi_sha256_verified": bool(
                    manifest.get("source_wsi_sha256_verified", False)
                ),
            }
        except Exception as exc:
            result["integration_boundary"] = "mucosa_extraction"
            result["dependency_blocked"].append(
                {"code": exc.__class__.__name__, "message": str(exc)}
            )
            write_integration_boundary(boundary_path, cohort, case_results)
            continue

        prediction_path = architecture_dir / "{0}.jsonl".format(case.case_alias) if architecture_dir else None
        if prediction_path is None or not prediction_path.exists():
            result["integration_boundary"] = "architecture_model"
            result["dependency_blocked"].append(
                {
                    "code": "architecture_predictions_unavailable",
                    "message": "A real validated 5x Architecture prediction JSONL was not supplied.",
                }
            )
        else:
            result["stages_reached"].append("architecture_predictions_available")
            result["artifacts"]["architecture_predictions"] = str(prediction_path)
            if not args.reviewer_endpoint:
                result["integration_boundary"] = "reviewer_backend"
                result["dependency_blocked"].append(
                    {
                        "code": "reviewer_backend_unavailable",
                        "message": "Pass a real Reviewer /predict endpoint to execute Gate Real-B.",
                    }
                )
                write_integration_boundary(boundary_path, cohort, case_results)
                continue
            try:
                predictions = _read_architecture_predictions(prediction_path)
                _validate_predictions_against_manifest(
                    predictions,
                    mucosa_dir / "five_x_patch_manifest.jsonl",
                    case.case_alias,
                )
                agentflow_dir = case_dir / "agentflow"
                reviewer_backend = HttpReviewerBackend(
                    endpoint=args.reviewer_endpoint,
                    timeout_seconds=args.reviewer_timeout_seconds,
                    model_id=args.reviewer_model_id,
                    model_version=args.reviewer_model_version,
                )
                with WSIROICropper(
                    alias_path,
                    case.case_alias,
                    base_magnification=(args.base_magnification or None),
                    mpp=(args.mpp or None),
                ) as cropper:
                    agentflow_result = AgentFlowOrchestrator(
                        reviewer_backend=reviewer_backend,
                        cropper=cropper,
                        max_actions=args.max_actions,
                        require_contract_validator=True,
                    ).run(
                        case_id=case.case_alias,
                        predictions=predictions,
                        slide_dimensions=cropper.slide_dimensions,
                        output_dir=agentflow_dir,
                        case_context={
                            "case_alias": case.case_alias,
                            "source_code": case.source_code,
                            "source_format": case.source_format,
                            "slide_id": case.case_alias,
                            "wsi_metadata_ref": str(metadata_path),
                            "five_x_manifest_ref": str(
                                mucosa_dir / "five_x_patch_manifest.jsonl"
                            ),
                            "architecture_predictions_ref": str(prediction_path),
                            "diagnostic_scope_status": "incomplete",
                            "non_clinical": True,
                        },
                    )
                final_state = StateStore.from_jsonl(
                    case.case_alias,
                    agentflow_dir / "state" / "snapshots.jsonl",
                ).latest
                trace = AgentTraceStore(
                    case.case_alias,
                    agentflow_dir / "trace" / "events.jsonl",
                    replay=True,
                )
                plan_transitions = trace.plan_transitions()
                reviewer_updates = [
                    event
                    for event in trace.events
                    if event.get("actor") == "EvidenceUpdater"
                ]
                belief_changed = any(
                    event.get("hypothesis_scores_before")
                    != event.get("hypothesis_scores_after")
                    for event in reviewer_updates
                )
                next_plan_changed = any(
                    item.get("changed_from_previous")
                    and item.get("caused_by_evidence_ids")
                    for item in plan_transitions[1:]
                )
                result["stages_reached"].extend(
                    [
                        "reviewer_runtime",
                        "evidence_adapter",
                        "belief_update",
                        "adaptive_replanning",
                        "stop",
                        "chief",
                    ]
                )
                result["integration_boundary"] = "chief"
                result["status"] = "implemented"
                result["artifacts"]["agentflow_run"] = str(agentflow_dir)
                result["artifacts"]["behavior_summary"] = {
                    "reviewer_rounds": len(final_state.review_history),
                    "belief_or_ranking_changed": belief_changed,
                    "next_plan_changed": next_plan_changed,
                    "stop_reason": final_state.termination_state.reason,
                    "termination_status": final_state.termination_state.decision,
                    "chief_reached": True,
                    "chief_status": agentflow_result.chief_decision.status,
                    "final_state_id": final_state.state_id,
                    "ground_truth_available_to_inference": False,
                }
                if final_state.tool_reviewer_failures:
                    result["status"] = "dependency_blocked"
                    result["dependency_blocked"].append(
                        {
                            "code": "reviewer_runtime_failure",
                            "message": "The real Reviewer runtime failed; see local invocation audit.",
                        }
                    )
            except Exception as exc:
                result["status"] = "dependency_blocked"
                result["integration_boundary"] = "reviewer_behavior"
                result["dependency_blocked"].append(
                    {"code": exc.__class__.__name__, "message": str(exc)}
                )
        write_integration_boundary(boundary_path, cohort, case_results)

    write_integration_boundary(boundary_path, cohort, case_results)
    print(json.dumps({"integration_boundary": str(boundary_path), "case_count": len(cohort)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
