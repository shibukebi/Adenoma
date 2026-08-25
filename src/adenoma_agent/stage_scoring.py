from adenoma_agent.contract_review import (
    ALLOWED_NAVIGATION_MAGNIFICATIONS,
    review_navigation_payload,
    review_observation_report_payload,
    review_observation_step_payload,
)
from adenoma_agent.trace_supervision import (
    FIXED_DIAGNOSTIC_PRIORITY,
    TRACE_LABEL_RUBRIC,
    _normalize_global_screening_label,
    read_assignment_payload,
    selected_patch_ids_from_grid,
    validate_patch_assignments,
)
from adenoma_agent.utils import read_json


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


MINOR_PENALTY = 5
MODERATE_PENALTY = 10
MAJOR_PENALTY = 20
CRITICAL_ENUM_PENALTY = 30
TEXT_STANDARDIZATION_CAP = 20

TRACE_WEIGHT = 0.40
NAVIGATE_WEIGHT = 0.25
OBSERVATION_WEIGHT = 0.35
OBSERVE_STEP_WEIGHT = 0.60
OBSERVE_REPORT_WEIGHT = 0.40

ALLOWED_WORKFLOW_BRANCHES = {"serrated", "conventional", "conventional_adenoma", "normal", "non_serrated", "background", "unresolved"}
ALLOWED_REVIEW_GOALS = {
    "serrated_overview_assessment",
    "ssl_assessment",
    "hp_assessment",
    "tsa_assessment",
    "ssl_dysplasia_assessment",
    "tsa_cytological_atypia_assessment",
    "tsa_dysplasia_assessment",
    "conventional_overview_assessment",
    "conventional_architecture_assessment",
    "reactive_regenerative_assessment",
    "normal_overview_assessment",
    "inflammatory_reactive_assessment",
    "serrated_lesion_assessment",
    "abnormal_crypt_assessment",
    "conventional_adenoma_assessment",
    "non_serrated_overview_assessment",
    "morphology_resolution_assessment",
    "serrated_dysplasia_assessment",
    "conventional_dysplasia_assessment",
    "integrated_impression",
}
ALLOWED_STAGE_GATES = {
    "mucosa_or_serrated",
    "serrated_overview",
    "ssl_architecture",
    "hp_architecture",
    "tsa_architecture",
    "tsa_cytology",
    "abnormal_crypt",
    "conventional_overview",
    "conventional_architecture",
    "conventional_adenoma",
    "normal_overview",
    "inflammatory_reactive",
    "reactive_regenerative",
    "non_serrated_context",
    "morphology_resolution",
    "dysplasia",
    "end",
}
TRACE_POSITIVE_TERMS = {"serrated", "dysplasia", "adenoma", "abnormal crypt", "crypt branching", "mucus cap"}
BACKGROUND_FORBIDDEN_TERMS = {
    "epitheli",
    "gland",
    "adenoma",
    "dysplasia",
    "serrated",
    "crypt",
    "mucosa",
}
STANDARD_OBSERVATION_POINT_SYNONYMS = {
    "basal crypt dilatation": {
        "basal crypt dilatation",
        "basal dilatation",
        "basal crypt dilation",
        "crypt dilatation",
    },
    "crypt branching": {
        "crypt branching",
        "branching",
        "branched crypt",
    },
    "horizontal growth": {
        "horizontal growth",
        "horizontal crypt growth",
    },
    "serration to crypt base": {
        "serration to crypt base",
        "serration to base",
    },
    "mucus cap": {
        "mucus cap",
        "mucous cap",
    },
    "abnormal maturation": {
        "abnormal maturation",
        "maturation abnormality",
    },
    "tubular or tubulovillous architecture": {
        "tubular or tubulovillous architecture",
        "tubular architecture",
        "tubulovillous architecture",
        "villous architecture",
    },
    "adenomatous gland crowding": {
        "adenomatous gland crowding",
        "gland crowding",
        "crowded adenomatous glands",
    },
    "conventional dysplasia branch review": {
        "conventional dysplasia branch review",
        "conventional dysplasia review",
    },
    "reactive changes": {
        "reactive changes",
        "reactive change",
    },
    "inflammation": {
        "inflammation",
        "inflamed",
    },
    "erosion": {
        "erosion",
        "eroded surface",
    },
    "granulation tissue": {
        "granulation tissue",
        "granulation",
    },
    "exclude dysplasia if uncertain": {
        "exclude dysplasia if uncertain",
        "exclude dysplasia",
    },
    "confirm benign architecture if sampled": {
        "confirm benign architecture if sampled",
        "confirm benign architecture",
        "benign architecture",
    },
    "low-priority non-lesional mucosa": {
        "low-priority non-lesional mucosa",
        "low priority non-lesional mucosa",
        "non-lesional mucosa",
    },
    "coverage-preserving discard group": {
        "coverage-preserving discard group",
        "coverage preserving discard group",
        "discard group",
    },
    "low-value background or artifact": {
        "low-value background or artifact",
        "low value background or artifact",
        "background or artifact",
    },
}
FUZZY_OBSERVATION_TERMS = {"suspicious area", "maybe lesion", "unclear abnormality", "weird patch", "possible issue"}
TRACE_RULE_TO_LAYER = {
    "trace.region_semantic.enum": "semantic_vocabulary",
    "trace.observation_points.non_standard": "semantic_vocabulary",
    "trace.observation_points.fuzzy": "semantic_vocabulary",
    "trace.matrix.serrated_priority": "clinical_logic",
    "trace.matrix.serrated_high_mag": "clinical_logic",
    "trace.matrix.serrated_observation_focus": "clinical_logic",
    "trace.matrix.conventional_priority": "clinical_logic",
    "trace.matrix.conventional_high_mag": "clinical_logic",
    "trace.matrix.normal_priority": "clinical_logic",
    "trace.matrix.normal_positive_terms": "clinical_logic",
    "trace.matrix.background_priority": "clinical_logic",
    "trace.matrix.background_high_mag": "clinical_logic",
    "trace.matrix.background_epithelial_terms": "clinical_logic",
}
NAV_RULE_TO_LAYER = {
    "navigate.cluster_label.enum": "semantic_vocabulary",
    "navigate.workflow_branch.enum": "semantic_vocabulary",
    "navigate.review_goal.enum": "semantic_vocabulary",
    "navigate.stage_gate.enum": "semantic_vocabulary",
    "navigate.region_size_mapping": "clinical_logic",
    "navigate.matrix.trace_reference_mismatch": "contextual_spatial",
    "navigate.matrix.serrated_branch": "clinical_logic",
    "navigate.matrix.serrated_review_goal": "clinical_logic",
    "navigate.matrix.serrated_abnormal_without_high_mag": "clinical_logic",
    "navigate.matrix.conventional_branch": "clinical_logic",
    "navigate.matrix.conventional_review_goal": "clinical_logic",
    "navigate.matrix.non_serrated_branch": "clinical_logic",
    "navigate.matrix.non_serrated_review_goal": "clinical_logic",
    "navigate.matrix.background_goal": "clinical_logic",
    "navigate.matrix.dysplasia_gate_order": "clinical_logic",
}
OBS_RULE_TO_LAYER = {
    "observe_step.review_goal.enum": "semantic_vocabulary",
    "observe_step.stage_decision.enum": "semantic_vocabulary",
    "observe_step.matrix.stage_decision_mapping": "clinical_logic",
    "observe_step.matrix.support_requires_findings": "clinical_logic",
    "observe_step.matrix.background_positive_leak": "clinical_logic",
    "observe_report.matrix.missing_hierarchy_key": "clinical_logic",
    "observe_report.matrix.positive_without_support": "clinical_logic",
    "observe_report.matrix.supported_but_denied": "clinical_logic",
    "observe_report.matrix.final_case_alignment": "clinical_logic",
}
OBSERVE_STEP_DECISION_MAP = {
    "serrated_overview_assessment": {
        "supports_serrated_overview",
        "serrated_overview_not_supported_or_indeterminate",
    },
    "ssl_assessment": {
        "ssl_architecture_supported",
        "ssl_architecture_not_supported_or_indeterminate",
    },
    "hp_assessment": {
        "hp_architecture_supported",
        "hp_architecture_not_supported_or_indeterminate",
    },
    "tsa_assessment": {
        "tsa_architecture_supported",
        "tsa_architecture_not_supported_or_indeterminate",
    },
    "ssl_dysplasia_assessment": {
        "ssl_dysplasia_supported",
        "ssl_dysplasia_not_supported_or_indeterminate",
    },
    "tsa_cytological_atypia_assessment": {
        "tsa_cytological_atypia_supported",
        "tsa_cytological_atypia_not_supported_or_indeterminate",
    },
    "tsa_dysplasia_assessment": {
        "tsa_dysplasia_supported",
        "tsa_dysplasia_not_supported_or_indeterminate",
    },
    "conventional_overview_assessment": {
        "supports_conventional_overview",
        "conventional_overview_not_supported_or_indeterminate",
    },
    "conventional_architecture_assessment": {
        "supports_conventional_architecture",
        "conventional_architecture_not_supported_or_indeterminate",
    },
    "normal_overview_assessment": {
        "supports_normal_overview",
        "normal_overview_not_supported_or_indeterminate",
    },
    "inflammatory_reactive_assessment": {
        "inflammatory_reactive_supported",
        "inflammatory_reactive_not_supported_or_indeterminate",
    },
    "reactive_regenerative_assessment": {
        "reactive_regenerative_supported",
        "reactive_regenerative_not_supported_or_indeterminate",
    },
    "serrated_lesion_assessment": {
        "supports_serrated_lesion",
        "leans_non_serrated_or_indeterminate",
    },
    "abnormal_crypt_assessment": {
        "supports_abnormal_crypt",
        "serrated_but_no_support_for_abnormal_crypt",
    },
    "conventional_adenoma_assessment": {
        "supports_conventional_adenoma",
        "conventional_adenoma_indeterminate_or_opposed",
    },
    "non_serrated_overview_assessment": {
        "supports_non_serrated_overview",
        "background_or_low_value",
    },
    "serrated_dysplasia_assessment": {
        "serrated_dysplasia_supported",
        "serrated_dysplasia_not_supported_or_indeterminate",
    },
    "conventional_dysplasia_assessment": {
        "conventional_dysplasia_supported",
        "conventional_dysplasia_not_supported_or_indeterminate",
    },
}
POSITIVE_STAGE_DECISIONS = {
    "supports_serrated_overview",
    "ssl_architecture_supported",
    "hp_architecture_supported",
    "tsa_architecture_supported",
    "ssl_dysplasia_supported",
    "tsa_cytological_atypia_supported",
    "tsa_dysplasia_supported",
    "supports_conventional_overview",
    "supports_conventional_architecture",
    "reactive_regenerative_supported",
    "supports_normal_overview",
    "inflammatory_reactive_supported",
    "supports_serrated_lesion",
    "supports_abnormal_crypt",
    "supports_conventional_adenoma",
    "supports_non_serrated_overview",
    "serrated_dysplasia_supported",
    "conventional_dysplasia_supported",
}


def _init_layer_penalties():
    return {
        "syntax_schema": 0,
        "contextual_spatial": 0,
        "semantic_vocabulary": 0,
        "clinical_logic": 0,
    }


def _lower_text(value):
    return str(value or "").strip().lower()


def _stringify(value):
    if isinstance(value, list):
        return " ".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_stringify(item) for item in value.values())
    return str(value or "")


def _payload_text(record):
    return _lower_text(
        " ".join(
            [
                _stringify(record.get("name")),
                _stringify(record.get("description")),
                _stringify(record.get("severity_reasoning")),
                _stringify(record.get("observation_points")),
            ]
        )
    )


def _patch_key(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except Exception:
            return None
    return None


def _make_violation(rule_id, severity, penalty, message, layer, location=None, details=None):
    payload = {
        "rule_id": rule_id,
        "severity": severity,
        "penalty": int(penalty),
        "layer": layer,
        "message": message,
    }
    if location:
        payload["location"] = location
    if details is not None:
        payload["details"] = details
    return payload


def _add_penalty(summary, rule_id, severity, penalty, message, layer=None, location=None, details=None):
    layer = layer or "clinical_logic"
    violation = _make_violation(rule_id, severity, penalty, message, layer, location=location, details=details)
    summary["violations"].append(violation)
    summary["penalty_points"] += int(penalty)
    summary["layer_penalties"][layer] += int(penalty)


def _add_text_penalty(summary, tracker, key, rule_id, message, location=None, details=None):
    used = int(tracker.get(key, 0))
    if used >= TEXT_STANDARDIZATION_CAP:
        return
    penalty = min(MINOR_PENALTY, TEXT_STANDARDIZATION_CAP - used)
    tracker[key] = used + penalty
    _add_penalty(
        summary,
        rule_id=rule_id,
        severity="minor",
        penalty=penalty,
        message=message,
        layer=TRACE_RULE_TO_LAYER.get(rule_id, "semantic_vocabulary"),
        location=location,
        details=details,
    )


def _finalize_stage(summary):
    if summary["hard_fail"]:
        summary["score"] = 0
        summary["status"] = "fail"
        return summary
    summary["score"] = max(0, 100 - int(summary["penalty_points"]))
    if summary["score"] < 60:
        summary["status"] = "fail"
    elif summary["score"] < 80:
        summary["status"] = "review"
    else:
        summary["status"] = "pass"
    return summary


def _new_stage_summary(stage):
    return {
        "stage": stage,
        "hard_fail": False,
        "score": 0,
        "penalty_points": 0,
        "layer_penalties": _init_layer_penalties(),
        "violations": [],
        "metrics": {},
        "status": "pass",
    }


def _record_contract_errors(summary, contract_result):
    if contract_result.get("ok", False):
        return
    summary["hard_fail"] = True
    for message in contract_result.get("errors", []):
        summary["violations"].append(
            _make_violation(
                rule_id="contract.hard_fail",
                severity="hard_fail",
                penalty=100,
                message=message,
                layer="syntax_schema",
            )
        )


def _canonical_observation_point(value):
    text = _lower_text(value)
    for canonical, synonyms in STANDARD_OBSERVATION_POINT_SYNONYMS.items():
        if any(item in text for item in synonyms):
            return canonical
    return None


def _contains_any(text, tokens):
    return any(token in text for token in tokens)


def _trace_patch_lookup(payload):
    lookup = {}
    for patch in payload.get("patches", []) if isinstance(payload, dict) else []:
        if not isinstance(patch, dict):
            continue
        key = _patch_key(patch.get("patch_id"))
        if key is not None:
            lookup[key] = patch
    return lookup


def _is_numeric_non_bool(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _trace_schema_errors(payload):
    errors = []
    if not isinstance(payload, dict):
        errors.append("trace payload must be an object")
        return errors
    patches = payload.get("patches")
    if not isinstance(patches, list):
        errors.append("trace.patches must be a list")
        return errors
    required_string_fields = ("region_semantic", "name", "description", "severity_reasoning")
    for index, patch in enumerate(patches):
        location = "patches[{0}]".format(index)
        if not isinstance(patch, dict):
            errors.append("{0} must be an object".format(location))
            continue
        if _patch_key(patch.get("patch_id")) is None:
            errors.append("{0}.patch_id must be [int, int]".format(location))
        for key in required_string_fields:
            if not isinstance(patch.get(key), str):
                errors.append("{0}.{1} must be a string".format(location, key))
        if not isinstance(patch.get("require_high_magnification"), bool):
            errors.append("{0}.require_high_magnification must be a boolean".format(location))
        if not _is_numeric_non_bool(patch.get("diagnostic_priority")):
            errors.append("{0}.diagnostic_priority must be numeric".format(location))
        if not isinstance(patch.get("observation_points"), list):
            errors.append("{0}.observation_points must be a list".format(location))
    return errors


def _trace_cluster_lookup(payload):
    lookup = {}
    for cluster in payload.get("clusters", []) if isinstance(payload, dict) else []:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if cluster_id:
            lookup[cluster_id] = cluster
    return lookup


def _check_trace_patch_logic(summary, patch, index, text_penalties):
    label = _normalize_global_screening_label(patch.get("region_semantic"))
    location = "patches[{0}]".format(index)
    if label not in TRACE_LABEL_RUBRIC:
        _add_penalty(
            summary,
            "trace.region_semantic.enum",
            "critical_enum",
            CRITICAL_ENUM_PENALTY,
            "region_semantic must match the 4-label trace whitelist",
            layer="semantic_vocabulary",
            location=location,
            details={"region_semantic": label},
        )
        return

    priority = patch.get("diagnostic_priority")
    require_high_mag = patch.get("require_high_magnification")
    observation_points = patch.get("observation_points", [])
    if not isinstance(observation_points, list):
        observation_points = [observation_points]
    canonical_points = [item for item in (_canonical_observation_point(value) for value in observation_points) if item]
    text_blob = _payload_text(patch)

    for obs_index, item in enumerate(observation_points):
        item_text = _lower_text(item)
        if item_text and not _canonical_observation_point(item):
            _add_text_penalty(
                summary,
                text_penalties,
                "trace.observation_points.non_standard",
                "trace.observation_points.non_standard",
                "observation_points should use the standardized vocabulary or explicit synonym list",
                location="{0}.observation_points[{1}]".format(location, obs_index),
                details={"value": item},
            )
        if item_text and any(token in item_text for token in FUZZY_OBSERVATION_TERMS):
            _add_text_penalty(
                summary,
                text_penalties,
                "trace.observation_points.fuzzy",
                "trace.observation_points.fuzzy",
                "observation_points contains a fuzzy non-rubric term",
                location="{0}.observation_points[{1}]".format(location, obs_index),
                details={"value": item},
            )

    if label == "serrated":
        if priority != 4:
            _add_penalty(summary, "trace.matrix.serrated_priority", "major", MAJOR_PENALTY, "Serrated-route patches must use diagnostic_priority=4", location=location, details={"diagnostic_priority": priority})
        if require_high_mag is not True:
            _add_penalty(summary, "trace.matrix.serrated_high_mag", "major", MAJOR_PENALTY, "Serrated-route patches must require high magnification", location=location)
        ssl_focus = {
            "basal crypt dilatation",
            "crypt branching",
            "horizontal growth",
            "serration to crypt base",
            "mucus cap",
            "abnormal maturation",
        }
        if not set(canonical_points) & ssl_focus:
            _add_penalty(summary, "trace.matrix.serrated_observation_focus", "major", MAJOR_PENALTY, "Serrated-route patches must include at least one serrated observation focus", location=location, details={"observation_points": observation_points})
    elif label == "conventional":
        if priority != 3:
            _add_penalty(summary, "trace.matrix.conventional_priority", "major", MAJOR_PENALTY, "Conventional-route patches must use diagnostic_priority=3", location=location, details={"diagnostic_priority": priority})
        if require_high_mag is not True:
            _add_penalty(summary, "trace.matrix.conventional_high_mag", "major", MAJOR_PENALTY, "Conventional-route patches must require high magnification", location=location)
    elif label == "normal":
        if priority is None or int(priority) > 1:
            _add_penalty(summary, "trace.matrix.normal_priority", "major", MAJOR_PENALTY, "Normal-route mucosa must use diagnostic_priority<=1", location=location, details={"diagnostic_priority": priority})
        if _contains_any(text_blob, TRACE_POSITIVE_TERMS):
            _add_penalty(summary, "trace.matrix.normal_positive_terms", "major", MAJOR_PENALTY, "Normal-route mucosa must not contain serrated/dysplasia positive terms", location=location)
    elif label == "background":
        if priority != 0:
            _add_penalty(summary, "trace.matrix.background_priority", "major", MAJOR_PENALTY, "Background patches must use diagnostic_priority=0", location=location, details={"diagnostic_priority": priority})
        if require_high_mag is not False:
            _add_penalty(summary, "trace.matrix.background_high_mag", "major", MAJOR_PENALTY, "Background patches must not require high magnification", location=location)
        if _contains_any(text_blob, BACKGROUND_FORBIDDEN_TERMS):
            _add_penalty(summary, "trace.matrix.background_epithelial_terms", "major", MAJOR_PENALTY, "Background patches must not contain epithelial, glandular, or lesion terms", location=location)


def score_trace(payload, grid_meta=None):
    summary = _new_stage_summary("trace")
    grid_meta = grid_meta or {"grid_cells": []}
    schema_errors = _trace_schema_errors(payload)
    structure = validate_patch_assignments(payload, grid_meta)
    summary["metrics"].update(
        {
            "selected_patch_ids": [list(item) for item in selected_patch_ids_from_grid(grid_meta)],
            "selected_patch_count": structure.get("selected_patch_count", 0),
            "covered_patch_count": structure.get("covered_patch_count", 0),
            "assignment_count": structure.get("assignment_count", 0),
            "coverage_ok": structure.get("coverage_ok", False),
        }
    )
    if schema_errors:
        summary["hard_fail"] = True
        for message in schema_errors:
            summary["violations"].append(
                _make_violation(
                    "trace.hard_fail.schema",
                    "hard_fail",
                    100,
                    message,
                    "syntax_schema",
                )
            )
    if not structure.get("coverage_ok", False):
        summary["hard_fail"] = True
        summary["violations"].append(
            _make_violation(
                "trace.hard_fail.coverage",
                "hard_fail",
                100,
                "trace patch assignments must cover every selected patch exactly once",
                "contextual_spatial",
                details={
                    "missing_patch_ids": structure.get("missing_patch_ids", []),
                    "duplicate_patch_ids": structure.get("duplicate_patch_ids", []),
                    "unexpected_patch_ids": structure.get("unexpected_patch_ids", []),
                    "ignored_patch_ids": structure.get("ignored_patch_ids", []),
                },
            )
        )
    if summary["hard_fail"]:
        return _finalize_stage(summary)

    text_penalties = {}
    patches = payload.get("patches", []) if isinstance(payload, dict) else []
    for index, patch in enumerate(patches):
        if isinstance(patch, dict):
            _check_trace_patch_logic(summary, patch, index, text_penalties)
    return _finalize_stage(summary)


def _build_trace_context(trace_payload):
    trace_payload = trace_payload if isinstance(trace_payload, dict) else {}
    patch_lookup = _trace_patch_lookup(trace_payload)
    cluster_lookup = _trace_cluster_lookup(trace_payload)
    cluster_by_patch = {}
    for cluster_id, cluster in cluster_lookup.items():
        for patch_id in cluster.get("patch_ids_ordered", []) if isinstance(cluster, dict) else []:
            key = _patch_key(patch_id)
            if key is not None:
                cluster_by_patch[key] = cluster
                if key not in patch_lookup:
                    patch_lookup[key] = {
                        "patch_id": list(key),
                        "region_semantic": cluster.get("l"),
                        "require_high_magnification": bool(cluster.get("d", False)),
                        "diagnostic_priority": cluster.get("s"),
                    }
    return {
        "patch_lookup": patch_lookup,
        "cluster_lookup": cluster_lookup,
        "cluster_by_patch": cluster_by_patch,
    }


def _trace_cluster_for_step(step, trace_context):
    metadata = step.get("metadata", {}) if isinstance(step, dict) else {}
    cluster_id = str(metadata.get("cluster_id") or metadata.get("source_group_id") or "").strip()
    cluster = trace_context["cluster_lookup"].get(cluster_id)
    if cluster is not None:
        return cluster
    patch_key = _patch_key(metadata.get("patch_id"))
    if patch_key is not None:
        cluster = trace_context["cluster_by_patch"].get(patch_key)
        if cluster is not None:
            return cluster
        patch = trace_context["patch_lookup"].get(patch_key)
        if patch is not None:
            return {
                "l": patch.get("region_semantic"),
                "d": bool(patch.get("require_high_magnification", False)),
                "s": patch.get("diagnostic_priority"),
            }
    return None


def _trace_payload_has_clusters(trace_payload):
    return isinstance(trace_payload, dict) and isinstance(trace_payload.get("clusters"), list)


def _expected_branch_for_label(label):
    label = _normalize_global_screening_label(label)
    if label in {
        "epithelial_neoplasia_suspicious",
        "mucus_rich_or_pale_context",
        "uncertain_reviewable_mucosa",
        "inflammatory_or_stromal_context",
    }:
        return "unresolved"
    if label == "reviewable_normal_mucosa":
        return "normal"
    if label == "background_or_artifact":
        return "background"
    if label == "serrated":
        return "serrated"
    if label == "conventional":
        return "conventional"
    if label == "normal":
        return "normal"
    if label == "background":
        return "background"
    return None


def _check_navigation_step(summary, step, index, trace_context, seen_branch_gate):
    metadata = step.get("metadata", {}) if isinstance(step, dict) else {}
    location = "steps[{0}]".format(index)
    if metadata.get("action") == "stop":
        return
    cluster_id = str(metadata.get("cluster_id") or "").strip()
    source_group_id = str(metadata.get("source_group_id") or "").strip()
    patch_id = metadata.get("patch_id")
    patch_key = _patch_key(patch_id)
    has_cluster_truth = bool(trace_context["cluster_lookup"])
    if has_cluster_truth and cluster_id and cluster_id not in trace_context["cluster_lookup"]:
        summary["hard_fail"] = True
        summary["violations"].append(
            _make_violation(
                "navigate.hard_fail.trace_reference_missing",
                "hard_fail",
                100,
                "Navigation step cluster_id does not exist in trace truth",
                "contextual_spatial",
                location=location,
                details={"cluster_id": cluster_id},
            )
        )
        return
    if has_cluster_truth and source_group_id and source_group_id not in trace_context["cluster_lookup"]:
        summary["hard_fail"] = True
        summary["violations"].append(
            _make_violation(
                "navigate.hard_fail.trace_reference_missing",
                "hard_fail",
                100,
                "Navigation step source_group_id does not exist in trace truth",
                "contextual_spatial",
                location=location,
                details={"source_group_id": source_group_id},
            )
        )
        return
    if patch_key is None or patch_key not in trace_context["patch_lookup"]:
        summary["hard_fail"] = True
        summary["violations"].append(
            _make_violation(
                "navigate.hard_fail.trace_reference_missing",
                "hard_fail",
                100,
                "Navigation step patch_id does not exist in trace truth",
                "contextual_spatial",
                location=location,
                details={"patch_id": patch_id},
            )
        )
        return
    cluster = _trace_cluster_for_step(step, trace_context)
    if cluster is None and has_cluster_truth:
        summary["hard_fail"] = True
        summary["violations"].append(
            _make_violation(
                "navigate.hard_fail.trace_reference_missing",
                "hard_fail",
                100,
                "Navigation step could not be resolved to a trace cluster",
                "contextual_spatial",
                location=location,
                details={"metadata": metadata},
            )
        )
        return
    if cluster is None:
        cluster = {"l": metadata.get("cluster_label"), "d": False}

    cluster_label = _normalize_global_screening_label(metadata.get("cluster_label")) or str(metadata.get("cluster_label") or "").strip()
    workflow_branch = str(metadata.get("workflow_branch") or "").strip()
    review_goal = str(step.get("review_goal") or "").strip()
    stage_gate = str(step.get("stage_gate") or "").strip()
    if cluster_label not in TRACE_LABEL_RUBRIC:
        _add_penalty(summary, "navigate.cluster_label.enum", "major", MAJOR_PENALTY, "cluster_label must match the trace label whitelist", layer="semantic_vocabulary", location=location, details={"cluster_label": cluster_label})
    if workflow_branch not in ALLOWED_WORKFLOW_BRANCHES:
        _add_penalty(summary, "navigate.workflow_branch.enum", "major", MAJOR_PENALTY, "workflow_branch must be in the navigation whitelist", layer="semantic_vocabulary", location=location, details={"workflow_branch": workflow_branch})
    if review_goal not in ALLOWED_REVIEW_GOALS:
        _add_penalty(summary, "navigate.review_goal.enum", "major", MAJOR_PENALTY, "review_goal must be in the navigation whitelist", layer="semantic_vocabulary", location=location, details={"review_goal": review_goal})
    if stage_gate not in ALLOWED_STAGE_GATES:
        _add_penalty(summary, "navigate.stage_gate.enum", "major", MAJOR_PENALTY, "stage_gate must be in the navigation whitelist", layer="semantic_vocabulary", location=location, details={"stage_gate": stage_gate})

    magnification = float(step.get("m", 0.0))
    expected_region_size = ALLOWED_NAVIGATION_MAGNIFICATIONS.get(float(magnification))
    if expected_region_size is not None and step.get("region_size_level0") != expected_region_size:
        _add_penalty(summary, "navigate.region_size_mapping", "major", MAJOR_PENALTY, "region_size_level0 must match the fixed magnification mapping", location=location, details={"m": magnification, "region_size_level0": step.get("region_size_level0"), "expected": expected_region_size})

    trace_label = _normalize_global_screening_label(cluster.get("l") or cluster_label) or str(cluster.get("l") or cluster_label or "").strip()
    expected_branch = _expected_branch_for_label(trace_label)
    neutral_trace_label = trace_label in {
        "epithelial_neoplasia_suspicious",
        "mucus_rich_or_pale_context",
        "uncertain_reviewable_mucosa",
        "inflammatory_or_stromal_context",
    }
    if expected_branch and workflow_branch != expected_branch and not neutral_trace_label:
        _add_penalty(summary, "navigate.matrix.trace_reference_mismatch", "major", MAJOR_PENALTY, "Navigation branch must stay aligned with the trace cluster label", layer="contextual_spatial", location=location, details={"trace_label": trace_label, "workflow_branch": workflow_branch})

    cluster_id = str(metadata.get("cluster_id") or metadata.get("source_group_id") or "").strip()
    if stage_gate in {"abnormal_crypt", "ssl_architecture", "hp_architecture", "tsa_architecture"} and cluster_id:
        seen_branch_gate[("serrated", cluster_id)] = True
    if stage_gate in {"conventional_adenoma", "conventional_architecture", "reactive_regenerative"} and cluster_id:
        seen_branch_gate[("conventional", cluster_id)] = True

    if neutral_trace_label:
        if review_goal != "morphology_resolution_assessment":
            _add_penalty(summary, "navigate.matrix.neutral_review_goal", "minor", MINOR_PENALTY, "Neutral CONCH trace labels should start with morphology resolution before branch-specific review", location=location, details={"review_goal": review_goal})
    elif trace_label == "serrated":
        if workflow_branch != "serrated":
            _add_penalty(summary, "navigate.matrix.serrated_branch", "major", MAJOR_PENALTY, "Serrated trace steps must stay in the serrated branch unless Chief correction is recorded", location=location)
        if review_goal not in {
            "serrated_overview_assessment",
            "ssl_assessment",
            "hp_assessment",
            "tsa_assessment",
            "ssl_dysplasia_assessment",
            "tsa_cytological_atypia_assessment",
            "tsa_dysplasia_assessment",
            "serrated_lesion_assessment",
            "abnormal_crypt_assessment",
            "serrated_dysplasia_assessment",
        }:
            _add_penalty(summary, "navigate.matrix.serrated_review_goal", "major", MAJOR_PENALTY, "Serrated trace steps must use serrated review goals only", location=location, details={"review_goal": review_goal})
    elif trace_label == "conventional":
        if workflow_branch not in {"conventional", "conventional_adenoma"}:
            _add_penalty(summary, "navigate.matrix.conventional_branch", "major", MAJOR_PENALTY, "Conventional trace steps must stay in the conventional branch unless Chief correction is recorded", location=location)
        if review_goal not in {
            "conventional_overview_assessment",
            "conventional_architecture_assessment",
            "reactive_regenerative_assessment",
            "conventional_adenoma_assessment",
            "conventional_dysplasia_assessment",
            "non_serrated_overview_assessment",
        }:
            _add_penalty(summary, "navigate.matrix.conventional_review_goal", "major", MAJOR_PENALTY, "Conventional trace steps must use conventional/non-serrated review goals", location=location, details={"review_goal": review_goal})
    elif trace_label == "normal":
        if workflow_branch not in {"normal", "non_serrated"}:
            _add_penalty(summary, "navigate.matrix.normal_branch", "major", MAJOR_PENALTY, "Normal trace steps must stay in normal/non_serrated review", location=location)
        if review_goal not in {"normal_overview_assessment", "inflammatory_reactive_assessment", "non_serrated_overview_assessment"}:
            _add_penalty(summary, "navigate.matrix.normal_review_goal", "major", MAJOR_PENALTY, "Normal trace steps must use non_serrated_overview_assessment only", location=location, details={"review_goal": review_goal})
    elif trace_label == "background":
        if review_goal != "non_serrated_overview_assessment":
            _add_penalty(summary, "navigate.matrix.background_goal", "major", MAJOR_PENALTY, "Background steps must not enter lesion or dysplasia goals", location=location, details={"review_goal": review_goal})

    if stage_gate == "dysplasia" and cluster_id:
        branch = workflow_branch
        gate_source = str(metadata.get("gate_source") or "").strip()
        if gate_source == "abnormal_crypt":
            branch = "serrated"
        elif gate_source in {"ssl_architecture", "tsa_architecture"}:
            branch = "serrated"
        elif gate_source == "conventional_adenoma":
            branch = "conventional"
        if not seen_branch_gate.get((branch, cluster_id)):
            _add_penalty(summary, "navigate.matrix.dysplasia_gate_order", "major", MAJOR_PENALTY, "Dysplasia steps must occur after their branch gate", location=location, details={"cluster_id": cluster_id, "branch": branch})


def score_navigate(payload, trace_payload):
    summary = _new_stage_summary("navigate")
    contract = review_navigation_payload(payload)
    summary["metrics"].update(contract.get("metrics", {}))
    _record_contract_errors(summary, contract)
    if summary["hard_fail"]:
        return _finalize_stage(summary)

    trace_context = _build_trace_context(trace_payload if isinstance(trace_payload, dict) else {"clusters": []})
    seen_branch_gate = {}
    goals_by_cluster = {}
    for index, step in enumerate(payload.get("steps", []) if isinstance(payload, dict) else []):
        if isinstance(step, dict):
            metadata = step.get("metadata", {}) if isinstance(step.get("metadata", {}), dict) else {}
            if metadata.get("action") != "stop":
                cluster_id = str(metadata.get("cluster_id") or metadata.get("source_group_id") or "").strip()
                if cluster_id:
                    goals_by_cluster.setdefault(cluster_id, set()).add(str(step.get("review_goal") or "").strip())
            _check_navigation_step(summary, step, index, trace_context, seen_branch_gate)
            if summary["hard_fail"]:
                break
    if not summary["hard_fail"]:
        for cluster_id, cluster in trace_context["cluster_lookup"].items():
            goals = goals_by_cluster.get(cluster_id, set())
            trace_label = _normalize_global_screening_label(cluster.get("l")) if isinstance(cluster, dict) else None
            if trace_label == "serrated" and "serrated_overview_assessment" in goals and "hp_assessment" not in goals:
                _add_penalty(summary, "navigate.matrix.missing_hp_assessment", "moderate", MODERATE_PENALTY, "New serrated navigation must include 5x hp_assessment alongside SSL/TSA assessment", location="cluster:{0}".format(cluster_id), details={"review_goals": sorted(goals)})
            if trace_label == "conventional" and "conventional_overview_assessment" in goals and "reactive_regenerative_assessment" not in goals:
                _add_penalty(summary, "navigate.matrix.missing_reactive_regenerative_assessment", "moderate", MODERATE_PENALTY, "New conventional navigation must include 5x reactive_regenerative_assessment alongside architecture assessment", location="cluster:{0}".format(cluster_id), details={"review_goals": sorted(goals)})
    return _finalize_stage(summary)


def _allowed_stage_decisions_for_goal(review_goal):
    return OBSERVE_STEP_DECISION_MAP.get(review_goal, set())


def _finding_key_for_goal(review_goal):
    if review_goal in {
        "serrated_overview_assessment",
        "serrated_lesion_assessment",
        "ssl_assessment",
        "hp_assessment",
        "tsa_assessment",
        "conventional_overview_assessment",
        "conventional_architecture_assessment",
        "reactive_regenerative_assessment",
        "normal_overview_assessment",
        "inflammatory_reactive_assessment",
        "conventional_adenoma_assessment",
        "non_serrated_overview_assessment",
    }:
        return "level_1_findings"
    if review_goal == "abnormal_crypt_assessment":
        return "level_2_findings"
    if review_goal == "tsa_cytological_atypia_assessment":
        return "level_1_findings"
    if review_goal in {
        "ssl_dysplasia_assessment",
        "tsa_dysplasia_assessment",
        "serrated_dysplasia_assessment",
        "conventional_dysplasia_assessment",
    }:
        return "level_3_findings"
    return None


def _flatten_checklist_support(payload):
    if isinstance(payload, dict):
        support = 0
        assessed = 0
        for value in payload.values():
            if isinstance(value, dict):
                if value.get("status") == "supporting":
                    support += 1
                if value.get("status") != "not_assessed":
                    assessed += 1
        return {"support": support, "assessed": assessed}
    if isinstance(payload, list):
        return {"support": len(payload), "assessed": len(payload)}
    return {"support": 0, "assessed": 0}


def _normalize_hierarchy_branch(value):
    return value if isinstance(value, dict) else {}


def _check_observe_step(summary, record, index):
    location = "observations[{0}]".format(index)
    metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
    review_goal = str(metadata.get("review_goal") or "").strip()
    stage_decision = str(record.get("stage_decision") or "").strip()
    if review_goal not in OBSERVE_STEP_DECISION_MAP:
        _add_penalty(summary, "observe_step.review_goal.enum", "major", MAJOR_PENALTY, "observe_step metadata.review_goal must be in the observation whitelist", layer="semantic_vocabulary", location=location, details={"review_goal": review_goal})
        return
    allowed = _allowed_stage_decisions_for_goal(review_goal)
    if stage_decision not in allowed:
        _add_penalty(summary, "observe_step.matrix.stage_decision_mapping", "major", MAJOR_PENALTY, "stage_decision must match the allowed set for its review_goal", location=location, details={"review_goal": review_goal, "stage_decision": stage_decision, "allowed": sorted(allowed)})
    if stage_decision in POSITIVE_STAGE_DECISIONS:
        finding_key = _finding_key_for_goal(review_goal)
        findings = record.get(finding_key, []) if finding_key else []
        if not isinstance(findings, list) or not findings:
            _add_penalty(summary, "observe_step.matrix.support_requires_findings", "major", MAJOR_PENALTY, "Supportive stage_decision must have a non-empty corresponding finding level", location=location, details={"review_goal": review_goal, "stage_decision": stage_decision, "finding_key": finding_key})
    if stage_decision in {"supports_non_serrated_overview", "background_or_low_value"}:
        positive_text = _lower_text(
            " ".join(
                [
                    _stringify(record.get("level_2_findings", [])),
                    _stringify(record.get("level_3_findings", [])),
                    _stringify(record.get("reasoning", "")),
                    _stringify(record.get("observation", "")),
                ]
            )
        )
        if _contains_any(positive_text, {"dysplasia", "adenoma", "abnormal crypt", "crypt branching", "serration"}):
            _add_penalty(summary, "observe_step.matrix.background_positive_leak", "major", MAJOR_PENALTY, "background/non-serrated records must not emit positive dysplasia, adenoma, or abnormal-crypt evidence", location=location)


def _check_global_review(summary, review, index, observations):
    location = "global_reviews[{0}]".format(index)
    decision = str(review.get("decision") or "").strip()
    if decision not in {"continue", "early_stop"}:
        _add_penalty(summary, "observe_step.stage_decision.enum", "major", MAJOR_PENALTY, "global_review decision must be continue or early_stop", layer="semantic_vocabulary", location=location, details={"decision": decision})
    if not _is_number(review.get("chief_confidence")):
        _add_penalty(summary, "observe_step.matrix.stage_decision_mapping", "major", MAJOR_PENALTY, "chief_confidence must be numeric", location=location)
    if not isinstance(review.get("resolved_branch_state"), dict):
        _add_penalty(summary, "observe_step.matrix.stage_decision_mapping", "major", MAJOR_PENALTY, "resolved_branch_state must be an object", location=location)
    if index < len(observations):
        expected_step_id = observations[index].get("step_id")
        if review.get("source_step_id") != expected_step_id:
            _add_penalty(summary, "observe_step.matrix.stage_decision_mapping", "major", MAJOR_PENALTY, "global_review.source_step_id must align with the corresponding observation step_id", location=location, details={"source_step_id": review.get("source_step_id"), "expected": expected_step_id})
    if decision == "continue":
        nxt = review.get("next_visual_target")
        if not isinstance(review.get("continue_reason"), str) or not review.get("continue_reason", "").strip():
            _add_penalty(summary, "observe_step.matrix.support_requires_findings", "major", MAJOR_PENALTY, "continue decision must include continue_reason", location=location)
        if not isinstance(nxt, dict):
            _add_penalty(summary, "observe_step.matrix.support_requires_findings", "major", MAJOR_PENALTY, "continue decision must include next_visual_target", location=location)
        else:
            if nxt.get("preferred_magnification") not in ALLOWED_NAVIGATION_MAGNIFICATIONS:
                _add_penalty(summary, "navigate.region_size_mapping", "major", MAJOR_PENALTY, "Chief preferred_magnification must be one of the navigation magnification whitelist", location=location, details={"preferred_magnification": nxt.get("preferred_magnification")})
            if not isinstance(nxt.get("target_morphology_prompt"), list) or not nxt.get("target_morphology_prompt"):
                _add_penalty(summary, "observe_step.matrix.support_requires_findings", "major", MAJOR_PENALTY, "Chief continue decision must include target_morphology_prompt", location=location)
            if nxt.get("target_branch") != observations[index].get("metadata", {}).get("workflow_branch") and not str(review.get("branch_correction_reason") or "").strip():
                _add_penalty(summary, "observe_step.matrix.stage_decision_mapping", "major", MAJOR_PENALTY, "branch_correction_reason is required when Chief redirects to a different branch", location=location)
    if decision == "early_stop":
        evidence = review.get("sufficient_evidence")
        if not isinstance(evidence, list) or not evidence:
            _add_penalty(summary, "observe_step.matrix.support_requires_findings", "major", MAJOR_PENALTY, "early_stop decision must include sufficient_evidence", location=location)


def score_observe_step(payload):
    summary = _new_stage_summary("observe_step")
    contract = review_observation_step_payload(payload)
    summary["metrics"].update(contract.get("metrics", {}))
    _record_contract_errors(summary, contract)
    if summary["hard_fail"]:
        return _finalize_stage(summary)
    observations = payload.get("observations", []) if isinstance(payload, dict) else []
    global_reviews = payload.get("global_reviews", []) if isinstance(payload, dict) else []
    for index, record in enumerate(observations):
        if isinstance(record, dict):
            _check_observe_step(summary, record, index)
    for index, review in enumerate(global_reviews):
        if isinstance(review, dict):
            _check_global_review(summary, review, index, observations)
    return _finalize_stage(summary)


def _build_support_index(observe_step_payload):
    index = {
        "serrated_lesion_assessment": False,
        "abnormal_crypt_assessment": False,
        "conventional_adenoma_assessment": False,
        "serrated_dysplasia_assessment": False,
        "conventional_dysplasia_assessment": False,
    }
    for record in observe_step_payload.get("observations", []) if isinstance(observe_step_payload, dict) else []:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
        review_goal = str(metadata.get("review_goal") or "").strip()
        stage_decision = str(record.get("stage_decision") or "").strip()
        if review_goal == "serrated_lesion_assessment" and stage_decision == "supports_serrated_lesion":
            index["serrated_lesion_assessment"] = True
        elif review_goal == "abnormal_crypt_assessment" and stage_decision == "supports_abnormal_crypt":
            index["abnormal_crypt_assessment"] = True
        elif review_goal == "conventional_adenoma_assessment" and stage_decision == "supports_conventional_adenoma":
            index["conventional_adenoma_assessment"] = True
        elif review_goal == "serrated_dysplasia_assessment" and stage_decision == "serrated_dysplasia_supported":
            index["serrated_dysplasia_assessment"] = True
        elif review_goal == "conventional_dysplasia_assessment" and stage_decision == "conventional_dysplasia_supported":
            index["conventional_dysplasia_assessment"] = True
    return index


def _build_chief_support_index(observe_step_payload):
    index = {
        "serrated_lesion_assessment": False,
        "abnormal_crypt_assessment": False,
        "conventional_adenoma_assessment": False,
        "serrated_dysplasia_assessment": False,
        "conventional_dysplasia_assessment": False,
    }
    if not isinstance(observe_step_payload, dict):
        return index
    reviews = observe_step_payload.get("global_reviews", [])
    if not isinstance(reviews, list) or not reviews:
        return index
    final_review = reviews[-1] if isinstance(reviews[-1], dict) else {}
    evidence_text = _lower_text(_stringify(final_review.get("sufficient_evidence", [])))
    if _contains_any(evidence_text, {"serrated lesion", "serrated context", "ssl"}):
        index["serrated_lesion_assessment"] = True
    if _contains_any(evidence_text, {"abnormal crypt", "crypt branching", "basal crypt", "boot-shaped"}):
        index["abnormal_crypt_assessment"] = True
    if _contains_any(evidence_text, {"conventional adenoma", "adenomatous glands", "tubular architecture", "tubulovillous"}):
        index["conventional_adenoma_assessment"] = True
    if _contains_any(evidence_text, {"serrated dysplasia"}):
        index["serrated_dysplasia_assessment"] = True
    if _contains_any(evidence_text, {"conventional dysplasia"}):
        index["conventional_dysplasia_assessment"] = True
    return index


def _check_hierarchy_key(summary, hierarchy, key):
    branch = _normalize_hierarchy_branch(hierarchy.get(key))
    if not branch:
        _add_penalty(summary, "observe_report.matrix.missing_hierarchy_key", "major", MAJOR_PENALTY, "hierarchical_prediction must contain the required normalized branch", location="hierarchical_prediction", details={"missing_key": key})
    return branch


def score_observe_report(payload, observe_step_payload):
    summary = _new_stage_summary("observe_report")
    contract = review_observation_report_payload(payload)
    summary["metrics"].update(contract.get("metrics", {}))
    _record_contract_errors(summary, contract)
    if summary["hard_fail"]:
        return _finalize_stage(summary)

    hierarchy = payload.get("hierarchical_prediction", {}) if isinstance(payload, dict) else {}
    step_support = _build_support_index(observe_step_payload)
    chief_support = _build_chief_support_index(observe_step_payload)
    global_reviews = observe_step_payload.get("global_reviews", []) if isinstance(observe_step_payload, dict) else []
    if not global_reviews or str(global_reviews[-1].get("decision") or "").strip() != "early_stop":
        _add_penalty(summary, "observe_report.matrix.missing_hierarchy_key", "major", MAJOR_PENALTY, "observe_report must only be generated after a final global_review decision=early_stop", location="global_reviews")
    serrated_branch = _check_hierarchy_key(summary, hierarchy, "serrated_lesion_assessment")
    abnormal_branch = _check_hierarchy_key(summary, hierarchy, "abnormal_crypt_assessment")
    conventional_branch = _check_hierarchy_key(summary, hierarchy, "conventional_adenoma_assessment")
    serrated_dysplasia_branch = _check_hierarchy_key(summary, hierarchy, "serrated_dysplasia_assessment")
    conventional_dysplasia_branch = _check_hierarchy_key(summary, hierarchy, "conventional_dysplasia_assessment")
    dysplasia_branch = _check_hierarchy_key(summary, hierarchy, "dysplasia_assessment")
    final_case_branch = _check_hierarchy_key(summary, hierarchy, "final_case_assessment")

    checklist_support = {
        "serrated_lesion_assessment": _flatten_checklist_support(payload.get("serrated_checklist")),
        "abnormal_crypt_assessment": _flatten_checklist_support(payload.get("abnormal_crypt_checklist")),
        "conventional_adenoma_assessment": _flatten_checklist_support(payload.get("conventional_adenoma_checklist")),
        "serrated_dysplasia_assessment": _flatten_checklist_support(payload.get("serrated_dysplasia_checklist")),
        "conventional_dysplasia_assessment": _flatten_checklist_support(payload.get("conventional_dysplasia_checklist")),
        "dysplasia_assessment": _flatten_checklist_support(payload.get("dysplasia_checklist")),
    }

    branch_expectations = [
        ("serrated_lesion_assessment", serrated_branch),
        ("abnormal_crypt_assessment", abnormal_branch),
        ("conventional_adenoma_assessment", conventional_branch),
        ("serrated_dysplasia_assessment", serrated_dysplasia_branch),
        ("conventional_dysplasia_assessment", conventional_dysplasia_branch),
    ]
    for branch_key, branch_payload in branch_expectations:
        positive = bool(branch_payload.get("positive"))
        has_step_support = bool(step_support.get(branch_key))
        has_chief_support = bool(chief_support.get(branch_key))
        has_checklist_support = bool(checklist_support.get(branch_key, {}).get("support"))
        if positive and not (has_step_support and has_chief_support and has_checklist_support):
            _add_penalty(summary, "observe_report.matrix.positive_without_support", "major", MAJOR_PENALTY, "Positive report branches must be supported by step-level stage_decision, Chief sufficient_evidence, and checklist evidence", location="hierarchical_prediction.{0}".format(branch_key), details={"branch": branch_key, "step_support": has_step_support, "chief_support": has_chief_support, "checklist_support": has_checklist_support})
        if not positive and has_step_support and has_checklist_support:
            _add_penalty(summary, "observe_report.matrix.supported_but_denied", "major", MAJOR_PENALTY, "Report branches must not deny a branch already supported upstream", location="hierarchical_prediction.{0}".format(branch_key), details={"branch": branch_key})

    overall_dysplasia_positive = bool(dysplasia_branch.get("positive"))
    if overall_dysplasia_positive and not (
        bool(serrated_dysplasia_branch.get("positive")) or bool(conventional_dysplasia_branch.get("positive"))
    ):
        _add_penalty(summary, "observe_report.matrix.positive_without_support", "major", MAJOR_PENALTY, "Overall dysplasia assessment must be supported by a positive serrated or conventional dysplasia branch", location="hierarchical_prediction.dysplasia_assessment")

    final_label = str(final_case_branch.get("label") or "").strip()
    classification_status = str(hierarchy.get("classification_status") or final_case_branch.get("classification_status") or "classified").strip()
    if classification_status == "classified" and not final_label:
        _add_penalty(summary, "observe_report.matrix.final_case_alignment", "major", MAJOR_PENALTY, "Classified reports must include a final 11-class label", location="hierarchical_prediction.final_case_assessment", details={"classification_status": classification_status})
    if final_label in ("SSL+dysplasia", "SSLD", "TSAD"):
        if not (bool(serrated_branch.get("positive")) and bool(serrated_dysplasia_branch.get("positive"))):
            _add_penalty(summary, "observe_report.matrix.final_case_alignment", "major", MAJOR_PENALTY, "Serrated final labels with D suffix require both serrated and high-grade/definite dysplasia branches to be positive", location="hierarchical_prediction.final_case_assessment", details={"label": final_label})
    elif final_label in ("Others+dysplasia", "TAD", "TVAD"):
        if not (bool(conventional_branch.get("positive")) and bool(conventional_dysplasia_branch.get("positive"))):
            _add_penalty(summary, "observe_report.matrix.final_case_alignment", "major", MAJOR_PENALTY, "Conventional final labels with D suffix require both conventional adenoma and high-grade/definite dysplasia branches to be positive", location="hierarchical_prediction.final_case_assessment", details={"label": final_label})

    summary["metrics"]["step_support"] = step_support
    summary["metrics"]["chief_support"] = chief_support
    summary["metrics"]["checklist_support"] = checklist_support
    return _finalize_stage(summary)


def score_observation(observe_step_payload, observe_report_payload):
    step_score = score_observe_step(observe_step_payload)
    report_score = score_observe_report(observe_report_payload, observe_step_payload)
    stage_summary = _new_stage_summary("observation")
    stage_summary["metrics"] = {
        "observe_step_score": step_score["score"],
        "observe_report_score": report_score["score"],
        "observe_step_weight": OBSERVE_STEP_WEIGHT,
        "observe_report_weight": OBSERVE_REPORT_WEIGHT,
    }
    stage_summary["hard_fail"] = bool(step_score["hard_fail"] or report_score["hard_fail"])
    stage_summary["violations"] = (
        [{"source_stage": "observe_step", **item} for item in step_score["violations"]]
        + [{"source_stage": "observe_report", **item} for item in report_score["violations"]]
    )
    for layer_name in stage_summary["layer_penalties"]:
        stage_summary["layer_penalties"][layer_name] = int(step_score["layer_penalties"].get(layer_name, 0)) + int(
            report_score["layer_penalties"].get(layer_name, 0)
        )
    if stage_summary["hard_fail"]:
        stage_summary["penalty_points"] = 100
        stage_summary["score"] = 0
        stage_summary["status"] = "fail"
    else:
        stage_summary["score"] = int(
            round(
                float(step_score["score"]) * OBSERVE_STEP_WEIGHT
                + float(report_score["score"]) * OBSERVE_REPORT_WEIGHT
            )
        )
        stage_summary["penalty_points"] = max(0, 100 - stage_summary["score"])
        if stage_summary["score"] < 60:
            stage_summary["status"] = "fail"
        elif stage_summary["score"] < 80:
            stage_summary["status"] = "review"
        else:
            stage_summary["status"] = "pass"
    stage_summary["observe_step"] = step_score
    stage_summary["observe_report"] = report_score
    return stage_summary


def score_case(trace_payload, navigate_payload, observe_step_payload, observe_report_payload, grid_meta=None):
    trace_score = score_trace(trace_payload, grid_meta=grid_meta or {"grid_cells": []})
    navigate_score = score_navigate(navigate_payload, trace_payload)
    observation_score = score_observation(observe_step_payload, observe_report_payload)
    overall_score = round(
        TRACE_WEIGHT * float(trace_score["score"])
        + NAVIGATE_WEIGHT * float(navigate_score["score"])
        + OBSERVATION_WEIGHT * float(observation_score["score"]),
        4,
    )
    hard_fail = bool(trace_score["hard_fail"] or navigate_score["hard_fail"] or observation_score["hard_fail"])
    if hard_fail or min(trace_score["score"], navigate_score["score"], observation_score["score"]) < 60 or overall_score < 70:
        overall_status = "fail"
    elif min(trace_score["score"], navigate_score["score"], observation_score["score"]) >= 80 and overall_score >= 85:
        overall_status = "pass"
    else:
        overall_status = "review"
    return {
        "stage": "case",
        "hard_fail": hard_fail,
        "trace_score": trace_score,
        "navigate_score": navigate_score,
        "observation_score": observation_score,
        "overall_score": overall_score,
        "weights": {
            "trace": TRACE_WEIGHT,
            "navigate": NAVIGATE_WEIGHT,
            "observation": OBSERVATION_WEIGHT,
            "observe_step_within_observation": OBSERVE_STEP_WEIGHT,
            "observe_report_within_observation": OBSERVE_REPORT_WEIGHT,
        },
        "overall_status": overall_status,
    }


def read_trace_payload(path):
    return read_assignment_payload(path)


def score_trace_from_path(json_path, grid_metadata_json):
    return score_trace(read_trace_payload(json_path), grid_meta=read_json(grid_metadata_json))


def score_navigate_from_path(json_path, trace_json):
    return score_navigate(read_json(json_path), read_json(trace_json))


def score_observe_step_from_path(json_path):
    return score_observe_step(read_json(json_path))


def score_observe_report_from_path(json_path, observe_step_json):
    return score_observe_report(read_json(json_path), read_json(observe_step_json))


def score_case_from_paths(trace_json, navigate_json, observe_step_json, observe_report_json, grid_metadata_json):
    return score_case(
        trace_payload=read_trace_payload(trace_json),
        navigate_payload=read_json(navigate_json),
        observe_step_payload=read_json(observe_step_json),
        observe_report_payload=read_json(observe_report_json),
        grid_meta=read_json(grid_metadata_json),
    )
