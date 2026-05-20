from pathlib import Path

from adenoma_agent.trace_supervision import (
    FIXED_DIAGNOSTIC_PRIORITY,
    TRACE_LABEL_RUBRIC,
    score_patch_field_consistency,
    score_patch_semantics,
    selected_patch_ids_from_grid,
    validate_patch_assignments,
)


ALLOWED_NAVIGATION_MAGNIFICATIONS = {
    5.0: 256,
    20.0: 64,
}

ALLOWED_NAVIGATION_ACTIONS = {"inspect", "stop"}


def _append_error(errors, message):
    if message not in errors:
        errors.append(message)


def _append_warning(warnings, message):
    if message not in warnings:
        warnings.append(message)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_patch_id(value):
    if not isinstance(value, list) or len(value) != 2:
        return None
    if not all(isinstance(item, int) for item in value):
        return None
    return tuple(value)


def review_global_screening_payload(payload, grid_meta=None):
    errors = []
    warnings = []
    metrics = {}

    if isinstance(payload, dict) and isinstance(payload.get("patches"), list):
        structure = validate_patch_assignments(payload, grid_meta or {"grid_cells": []})
        semantics = score_patch_semantics(payload)
        field_consistency = score_patch_field_consistency(payload)

        if not structure["coverage_ok"]:
            _append_error(errors, "global_screening.coverage_ok must be true")
        if structure["missing_patch_ids"]:
            _append_error(errors, "global_screening contains missing patch assignments")
        if structure["duplicate_patch_ids"]:
            _append_error(errors, "global_screening contains duplicate patch assignments")
        if structure["unexpected_patch_ids"]:
            _append_error(errors, "global_screening contains unexpected patch assignments")
        if structure["ignored_patch_ids"]:
            _append_error(errors, "global_screening contains ignored or invalid patch assignments")

        for warning in semantics["warnings"]:
            if warning.get("warning") == "unknown_region_semantic":
                _append_error(errors, "global_screening contains unknown region_semantic")
            else:
                _append_warning(
                    warnings,
                    "semantic_warning:{0}:{1}".format(
                        warning.get("patch_id"),
                        warning.get("warning"),
                    ),
                )

        for warning in field_consistency["warnings"]:
            if warning.get("warning") == "invalid_priority":
                _append_error(errors, "global_screening contains invalid diagnostic_priority values")
            else:
                _append_warning(
                    warnings,
                    "field_warning:{0}:{1}".format(
                        warning.get("patch_id"),
                        warning.get("warning"),
                    ),
                )

        metrics = {
            "mode": "patch_assignments",
            "selected_patch_count": structure["selected_patch_count"],
            "covered_patch_count": structure["covered_patch_count"],
            "assignment_count": structure["assignment_count"],
            "label_counts": semantics["label_counts"],
            "semantic_score": semantics["semantic_score"],
            "field_consistency_score": field_consistency["field_consistency_score"],
        }
        return {"ok": not errors, "errors": errors, "warnings": warnings, "metrics": metrics}

    _append_error(errors, "global_screening payload must contain patches")
    return {"ok": False, "errors": errors, "warnings": warnings, "metrics": metrics}


def review_navigation_payload(payload):
    errors = []
    warnings = []
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(steps, list):
        _append_error(errors, "navigation.steps must be a list")
        steps = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            _append_error(errors, "navigation.steps[{0}] must be an object".format(index))
            continue
        if not isinstance(step.get("step_id"), str) or not step.get("step_id"):
            _append_error(errors, "navigation.steps[{0}].step_id must be a non-empty string".format(index))
        for key in ("x", "y"):
            if not isinstance(step.get(key), int):
                _append_error(errors, "navigation.steps[{0}].{1} must be an integer".format(index, key))
        magnification = step.get("m")
        if not _is_number(magnification):
            _append_error(errors, "navigation.steps[{0}].m must be numeric".format(index))
            magnification = None
        elif float(magnification) not in ALLOWED_NAVIGATION_MAGNIFICATIONS:
            _append_error(errors, "navigation.steps[{0}].m must be one of 5.0, 20.0".format(index))
        if not isinstance(step.get("region_size_level0"), int):
            _append_error(errors, "navigation.steps[{0}].region_size_level0 must be an integer".format(index))
        elif magnification is not None:
            expected_size = ALLOWED_NAVIGATION_MAGNIFICATIONS.get(float(magnification))
            if expected_size is not None and step.get("region_size_level0") != expected_size:
                _append_error(errors, "navigation.steps[{0}].region_size_level0 does not match fixed mapping".format(index))
        for key in ("need_to_see", "review_goal", "stage_gate"):
            if not isinstance(step.get(key), str) or not str(step.get(key)).strip():
                _append_error(errors, "navigation.steps[{0}].{1} must be a non-empty string".format(index, key))
        metadata = step.get("metadata")
        if not isinstance(metadata, dict):
            _append_error(errors, "navigation.steps[{0}].metadata must be an object".format(index))
            continue
        required_metadata = (
            "cluster_id",
            "source_group_id",
            "cluster_label",
            "cluster_priority",
            "patch_id",
            "region_size_level0",
            "workflow_branch",
            "action",
        )
        for key in required_metadata:
            if key not in metadata:
                _append_error(errors, "navigation.steps[{0}].metadata.{1} is required".format(index, key))
        if "patch_id" in metadata and _normalize_patch_id(metadata.get("patch_id")) is None:
            _append_error(errors, "navigation.steps[{0}].metadata.patch_id must be [int, int]".format(index))
        if "region_size_level0" in metadata and not isinstance(metadata.get("region_size_level0"), int):
            _append_error(errors, "navigation.steps[{0}].metadata.region_size_level0 must be an integer".format(index))
        elif "region_size_level0" in metadata and step.get("region_size_level0") != metadata.get("region_size_level0"):
            _append_error(errors, "navigation.steps[{0}].metadata.region_size_level0 must match top-level region_size_level0".format(index))
        if "action" in metadata and metadata.get("action") not in ALLOWED_NAVIGATION_ACTIONS:
            _append_error(errors, "navigation.steps[{0}].metadata.action must be inspect or stop".format(index))
        if step.get("stage_gate") == "end" and metadata.get("action") != "stop":
            _append_error(errors, "navigation.steps[{0}] with stage_gate=end must use action=stop".format(index))
        if step.get("stage_gate") != "end" and metadata.get("action") != "inspect":
            _append_error(errors, "navigation.steps[{0}] non-terminal step must use action=inspect".format(index))
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": {"step_count": len(steps)},
    }


def review_observation_step_payload(payload):
    errors = []
    warnings = []
    observations = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(observations, list):
        _append_error(errors, "observation.observations must be a list")
        observations = []
    if not observations:
        _append_error(errors, "observation.observations must not be empty")
    for index, record in enumerate(observations):
        if not isinstance(record, dict):
            _append_error(errors, "observation.observations[{0}] must be an object".format(index))
            continue
        for key in ("step_id", "crop_path", "observation", "reasoning", "next_step", "stage_decision"):
            if not isinstance(record.get(key), str):
                _append_error(errors, "observation.observations[{0}].{1} must be a string".format(index, key))
        for key in ("level_1_findings", "level_2_findings", "level_3_findings"):
            if not isinstance(record.get(key), list):
                _append_error(errors, "observation.observations[{0}].{1} must be a list".format(index, key))
        if not _is_number(record.get("confidence")):
            _append_error(errors, "observation.observations[{0}].confidence must be numeric".format(index))
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            _append_error(errors, "observation.observations[{0}].metadata must be an object".format(index))
        elif "review_goal" not in metadata or "stage_gate" not in metadata:
            _append_error(errors, "observation.observations[{0}].metadata must include review_goal and stage_gate".format(index))
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": {"observation_count": len(observations)},
    }


def review_observation_report_payload(payload):
    errors = []
    warnings = []
    hierarchy = payload.get("hierarchical_prediction")
    if not isinstance(hierarchy, dict):
        _append_error(errors, "observation_report.hierarchical_prediction must be an object")
    for key in (
        "serrated_checklist",
        "abnormal_crypt_checklist",
        "conventional_adenoma_checklist",
        "serrated_dysplasia_checklist",
        "conventional_dysplasia_checklist",
        "dysplasia_checklist",
    ):
        value = payload.get(key)
        if not isinstance(value, (dict, list)):
            _append_error(errors, "observation_report.{0} must be an object or list".format(key))
    integrated_report = payload.get("integrated_report")
    if not isinstance(integrated_report, (str, dict)):
        _append_error(errors, "observation_report.integrated_report must be a string or object")
    elif isinstance(integrated_report, str) and not integrated_report.strip():
        _append_error(errors, "observation_report.integrated_report must not be empty")
    elif isinstance(integrated_report, dict) and not any(str(value).strip() for value in integrated_report.values()):
        _append_error(errors, "observation_report.integrated_report object must not be empty")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "metrics": {}}
