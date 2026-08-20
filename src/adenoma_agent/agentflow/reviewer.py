import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Tuple

import requests

from adenoma_agent.agentflow.contracts import (
    DictMixin,
    EvidenceRecord,
    ReviewerAction,
    ReviewerFinding,
    ReviewerObservation,
    ReviewerTaskRequest,
)


class ReviewerContractError(ValueError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


def _extract_json_payload(text):
    value = str(text or "").strip()
    if not value:
        raise ReviewerContractError("Reviewer backend returned an empty response")
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(value)
    first = value.find("{")
    last = value.rfind("}")
    if first >= 0 and last > first:
        candidates.append(value[first : last + 1])
    last_error = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception as exc:
            last_error = exc
            continue
        if not isinstance(payload, Mapping):
            last_error = TypeError("Reviewer response JSON must be an object")
            continue
        return dict(payload)
    raise ReviewerContractError("Reviewer response is not valid JSON: {0}".format(last_error))


def _reviewer_observation_from_payload(payload, request, model_version="", prompt_version=""):
    def findings(section):
        output = []
        for row in payload.get(section, ()):
            if not isinstance(row, Mapping):
                raise ReviewerContractError("{0} entries must be JSON objects".format(section))
            output.append(
                ReviewerFinding(
                    feature_id=row["feature_id"],
                    status=row["status"],
                    status_confidence=row["status_confidence"],
                    feature_evaluability=row["feature_evaluability"],
                    scope=row["scope"],
                    evidence_text=row["evidence_text"],
                    limitations=tuple(row.get("limitations", ())),
                    quantitation=row.get("quantitation"),
                )
            )
        return tuple(output)

    try:
        return ReviewerObservation(
            schema_version=payload.get("schema_version", "reviewer_observation_v1"),
            observation_id=payload.get("observation_id", "OBS_{0}".format(request.request_id)),
            request_id=payload.get("request_id", request.request_id),
            reviewer=payload.get("reviewer", request.reviewer),
            task_profile=payload.get("task_profile", request.task_profile),
            primary_roi_id=payload.get("primary_roi_id", request.primary_roi.roi_id),
            primary_magnification=payload.get(
                "primary_magnification", request.primary_roi.scale
            ),
            target_features=tuple(payload.get("target_features", request.target_features)),
            quality=dict(payload["quality"]),
            findings=findings("findings"),
            incidental_findings=findings("incidental_findings"),
            overall_evidence_strength=dict(payload["overall_evidence_strength"]),
            limitations=tuple(payload.get("limitations", ())),
            does_not_decide_final_diagnosis=payload.get(
                "does_not_decide_final_diagnosis", True
            ),
            model_version=str(model_version or payload.get("model_version", "")),
            prompt_version=str(prompt_version or payload.get("prompt_version", request.prompt_version)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewerContractError("Invalid ReviewerObservation payload: {0}".format(exc)) from exc


def validate_repair_preserves_semantics(original, repaired):
    """Reject schema repair that invents or changes morphology findings."""

    def finding_rows(value):
        if isinstance(value, ReviewerObservation):
            payload = value.to_dict()
        elif isinstance(value, Mapping):
            payload = value
        else:
            raise ReviewerContractError("Schema repair inputs must be observation-like")
        rows = []
        for section in ("findings", "incidental_findings"):
            for finding in payload.get(section, ()):
                if isinstance(finding, ReviewerFinding):
                    finding = finding.to_dict()
                rows.append(json.dumps(finding, ensure_ascii=False, sort_keys=True))
        return sorted(rows)

    if finding_rows(original) != finding_rows(repaired):
        raise ReviewerContractError(
            "Schema repair may not add, remove, or alter morphology findings"
        )


@dataclass(frozen=True)
class ReviewerProfile(DictMixin):
    reviewer_id: str
    task_profile: str
    allowed_scales: Tuple[float, ...]
    target_feature_allowlist: Tuple[str, ...]
    incidental_feature_allowlist: Tuple[str, ...]
    context_policies: Tuple[Mapping[str, Any], ...]
    prompt_id: str = "reviewer"
    prompt_version: str = "reviewer_v1"

    @property
    def allowed_context_roles(self):
        return tuple(policy.get("view_role", "") for policy in self.context_policies)


class ReviewerRegistry(object):
    BLOCKED_MODEL_INPUT_KEYS = {
        "hypothesis",
        "hypotheses",
        "hypothesis_ranking",
        "top_hypothesis",
        "ranked_hypotheses",
        "candidate_final_label",
        "final_diagnosis",
        "final_label",
        "final_label_mapping",
        "expected_answer",
        "expected_effect",
        "branch_context",
        "branch_preference",
        "action_score",
        "diagnostic_preference",
        "guideline_recommendation",
    }
    FORBIDDEN_DIAGNOSIS_TOKENS = (
        "SSLD",
        "TSAD",
        "TAD",
        "TVAD",
        "Sessile serrated lesion",
        "Hyperplastic polyp",
        "Traditional serrated adenoma",
        "Unclassified serrated adenoma",
        "Tubular adenoma",
        "Tubulovillous adenoma",
        "Inflammatory/reactive polyp",
    )

    def __init__(self, version, profiles, retry_policy=None):
        self.version = str(version)
        self.profiles = tuple(profiles)
        self.retry_policy = dict(
            retry_policy
            or {
                "transport_or_model_retries": 1,
                "schema_repair_attempts": 1,
                "max_conflict_resolution_rounds": 2,
                "retry_same_roi_when_not_evaluable": False,
            }
        )
        self._by_key = {(item.reviewer_id, item.task_profile): item for item in profiles}
        if len(self._by_key) != len(self.profiles):
            raise ValueError("Reviewer registry profile keys must be unique")

    def profile(self, reviewer_id, task_profile):
        try:
            return self._by_key[(reviewer_id, task_profile)]
        except KeyError:
            raise ReviewerContractError(
                "Unknown reviewer/task profile: {0}/{1}".format(reviewer_id, task_profile)
            )

    def validate_action(self, action, roi):
        profile = self.profile(action.reviewer_id, action.task_profile)
        if not action.target_features or len(action.target_features) != len(set(action.target_features)):
            raise ReviewerContractError("Action target features must be non-empty and unique")
        if float(action.scale) not in profile.allowed_scales:
            raise ReviewerContractError("Action scale is incompatible with reviewer profile")
        if action.roi_id != roi.roi_id or float(action.scale) != float(roi.scale):
            raise ReviewerContractError("Action ROI provenance does not match selected candidate")
        if roi.allowed_reviewers and action.reviewer_id not in roi.allowed_reviewers:
            raise ReviewerContractError("ROI does not allow selected reviewer")
        unknown = set(action.target_features) - set(profile.target_feature_allowlist)
        if unknown:
            raise ReviewerContractError("Target features are not in reviewer allowlist: {0}".format(sorted(unknown)))

    def validate_request(self, request):
        profile = self.profile(request.reviewer_id, request.task_profile)
        if request.schema_version != "reviewer_task_request_v1":
            raise ReviewerContractError("Reviewer request schema_version mismatch")
        if request.registry_version != self.version:
            raise ReviewerContractError("Reviewer registry version mismatch")
        if request.prompt_id != profile.prompt_id or request.prompt_version != profile.prompt_version:
            raise ReviewerContractError("Reviewer prompt provenance does not match registry profile")
        if float(request.primary_roi.scale) not in profile.allowed_scales:
            raise ReviewerContractError("Primary ROI scale is incompatible with task profile")
        unknown = set(request.target_features) - set(profile.target_feature_allowlist)
        if unknown:
            raise ReviewerContractError("Request contains disallowed target features")
        if not request.target_features or len(request.target_features) != len(set(request.target_features)):
            raise ReviewerContractError("Request target features must be non-empty and unique")
        blocked = self._find_blocked_keys(request.model_input)
        if blocked:
            raise ReviewerContractError("model_input leaks planner state: {0}".format(sorted(blocked)))
        allowed_model_input_keys = {
            "task_profile",
            "target_features",
            "primary_roi",
            "context_views",
            "feature_disagreements",
        }
        unknown_model_input_keys = set(request.model_input) - allowed_model_input_keys
        if unknown_model_input_keys:
            raise ReviewerContractError(
                "model_input contains fields outside the public contract: {0}".format(
                    sorted(unknown_model_input_keys)
                )
            )
        primary = request.model_input.get("primary_roi", {})
        required = (
            "roi_id",
            "image_id",
            "image_ref",
            "image_sha256",
            "level0_bbox",
            "magnification",
            "pixel_dimensions",
        )
        if any(key not in primary for key in required):
            raise ReviewerContractError("model_input primary_roi lacks required provenance")
        if request.model_input.get("task_profile") != request.task_profile:
            raise ReviewerContractError("model_input task_profile does not match orchestration metadata")
        if tuple(request.model_input.get("target_features", [])) != tuple(request.target_features):
            raise ReviewerContractError("model_input target_features do not match orchestration metadata")
        if primary.get("roi_id") != request.primary_roi.roi_id:
            raise ReviewerContractError("model_input primary ROI id mismatch")
        if tuple(primary.get("level0_bbox", [])) != tuple(request.primary_roi.level0_bbox):
            raise ReviewerContractError("model_input primary ROI bbox mismatch")
        if float(primary.get("magnification")) != float(request.primary_roi.scale):
            raise ReviewerContractError("model_input primary ROI scale mismatch")
        digest = str(primary.get("image_sha256", ""))
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise ReviewerContractError("model_input primary ROI image_sha256 is invalid")
        dimensions = primary.get("pixel_dimensions", [])
        if len(dimensions) != 2 or any(int(value) <= 0 for value in dimensions):
            raise ReviewerContractError("model_input primary ROI pixel_dimensions are invalid")
        contexts = request.model_input.get("context_views", [])
        if len(contexts) > 3:
            raise ReviewerContractError("At most three context views are allowed")
        policies = {policy.get("view_role"): policy for policy in profile.context_policies}
        role_counts = {}
        if len(contexts) != len(request.context_views):
            raise ReviewerContractError("Context view orchestration provenance mismatch")
        for context_index, context in enumerate(contexts):
            role = context.get("view_role")
            policy = policies.get(role)
            if policy is None:
                raise ReviewerContractError("Context role is incompatible with task profile")
            role_counts[role] = role_counts.get(role, 0) + 1
            if role_counts[role] > int(policy.get("max_count", 0)):
                raise ReviewerContractError("Context role exceeds registry max_count")
            context_roi = context.get("roi", {})
            if float(context_roi.get("magnification", -1)) not in tuple(
                float(value) for value in policy.get("allowed_magnifications", ())
            ):
                raise ReviewerContractError("Context scale is incompatible with task profile")
            for key in required:
                if key not in context_roi:
                    raise ReviewerContractError("Context ROI lacks required provenance")
            context_digest = str(context_roi.get("image_sha256", ""))
            if len(context_digest) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in context_digest
            ):
                raise ReviewerContractError("Context ROI image_sha256 is invalid")
            context_dimensions = context_roi.get("pixel_dimensions", [])
            if len(context_dimensions) != 2 or any(int(value) <= 0 for value in context_dimensions):
                raise ReviewerContractError("Context ROI pixel_dimensions are invalid")
            internal_role, internal_roi = request.context_views[context_index]
            if role != internal_role or context_roi.get("roi_id") != internal_roi.roi_id:
                raise ReviewerContractError("Context ROI identity or role provenance mismatch")
            if tuple(context_roi.get("level0_bbox", ())) != tuple(internal_roi.level0_bbox):
                raise ReviewerContractError("Context ROI bbox provenance mismatch")
            if float(context_roi.get("magnification")) != float(internal_roi.scale):
                raise ReviewerContractError("Context ROI magnification provenance mismatch")
            expected_hash = internal_roi.image_sha256
            if expected_hash and context_roi.get("image_sha256") != expected_hash:
                raise ReviewerContractError("Context ROI image hash provenance mismatch")
        disagreements = request.model_input.get("feature_disagreements")
        if disagreements is not None:
            if not isinstance(disagreements, list) or not disagreements:
                raise ReviewerContractError("feature_disagreements must be a non-empty list")
            requested = set(request.target_features)
            for disagreement in disagreements:
                if disagreement.get("feature_id") not in requested:
                    raise ReviewerContractError("Disagreement feature must be a requested target")

    def validate_observation(self, request, observation):
        self.validate_request(request)
        profile = self.profile(request.reviewer_id, request.task_profile)
        if observation.schema_version != "reviewer_observation_v1":
            raise ReviewerContractError("Reviewer observation schema_version mismatch")
        if not observation.observation_id:
            raise ReviewerContractError("Reviewer observation_id is required")
        if observation.request_id != request.request_id:
            raise ReviewerContractError("Observation request_id mismatch")
        if observation.reviewer_id != request.reviewer_id or observation.task_profile != request.task_profile:
            raise ReviewerContractError("Observation reviewer/profile mismatch")
        if observation.primary_roi_id != request.primary_roi.roi_id:
            raise ReviewerContractError("Observation primary ROI mismatch")
        if float(observation.primary_magnification) != float(request.primary_roi.scale):
            raise ReviewerContractError("Observation primary magnification mismatch")
        if tuple(observation.target_features) != tuple(request.target_features):
            raise ReviewerContractError("Observation target_features mismatch")
        observed = [finding.feature_id for finding in observation.findings]
        if sorted(observed) != sorted(request.target_features) or len(observed) != len(set(observed)):
            raise ReviewerContractError("Every requested feature must be returned exactly once")
        incidental_ids = [finding.feature_id for finding in observation.incidental_findings]
        if len(incidental_ids) != len(set(incidental_ids)):
            raise ReviewerContractError("Incidental findings must not contain duplicate features")
        incidental = set(incidental_ids)
        if not incidental.issubset(set(profile.incidental_feature_allowlist)):
            raise ReviewerContractError("Observation contains disallowed incidental findings")
        if incidental.intersection(set(observed)):
            raise ReviewerContractError("Target and incidental findings must not overlap")
        quality = observation.quality
        expected_quality_keys = {
            "overall_evaluability",
            "status_confidence",
            "adequate_for_requested_features",
            "limitations",
        }
        if set(quality) != expected_quality_keys:
            raise ReviewerContractError("Observation quality block has unknown or missing fields")
        for key in (
            "overall_evaluability",
            "status_confidence",
            "adequate_for_requested_features",
            "limitations",
        ):
            if key not in quality:
                raise ReviewerContractError("Observation quality block is missing {0}".format(key))
        confidence = float(quality.get("status_confidence", -1.0))
        if confidence < 0.0 or confidence > 1.0:
            raise ReviewerContractError("quality.status_confidence must be in [0, 1]")
        if quality.get("overall_evaluability") not in ("adequate", "limited", "not_evaluable"):
            raise ReviewerContractError("quality.overall_evaluability is invalid")
        if not isinstance(quality.get("adequate_for_requested_features"), bool):
            raise ReviewerContractError("quality.adequate_for_requested_features must be boolean")
        if not observation.model_version or not observation.prompt_version:
            raise ReviewerContractError("Reviewer observation requires model and prompt versions")
        if observation.prompt_version != profile.prompt_version:
            raise ReviewerContractError("Observation prompt_version does not match registry profile")
        text = " ".join(
            finding.evidence_text
            for finding in tuple(observation.findings) + tuple(observation.incidental_findings)
        )
        if any(token in text for token in self.FORBIDDEN_DIAGNOSIS_TOKENS):
            raise ReviewerContractError("Reviewer finding must not output a D-suffix diagnosis")

    def _find_blocked_keys(self, value):
        found = set()
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) in self.BLOCKED_MODEL_INPUT_KEYS:
                    found.add(str(key))
                found.update(self._find_blocked_keys(child))
        elif isinstance(value, (list, tuple)):
            for child in value:
                found.update(self._find_blocked_keys(child))
        return found

    @classmethod
    def from_dict(cls, payload):
        profiles = []
        if payload.get("profiles"):
            for item in payload.get("profiles", []):
                profiles.append(
                    ReviewerProfile(
                        reviewer_id=item["reviewer_id"],
                        task_profile=item["task_profile"],
                        allowed_scales=tuple(float(value) for value in item.get("allowed_scales", [])),
                        target_feature_allowlist=tuple(item.get("target_feature_allowlist", [])),
                        incidental_feature_allowlist=tuple(item.get("incidental_feature_allowlist", [])),
                        context_policies=tuple(
                            {
                                "view_role": role,
                                "allowed_magnifications": [2.5, 5.0, 10.0, 20.0],
                                "max_count": 1,
                            }
                            for role in item.get("allowed_context_roles", [])
                        ),
                    )
                )
            version = payload.get("version", "reviewer_registry_v1")
        else:
            for reviewer in payload.get("reviewers", []):
                reviewer_id = reviewer["reviewer"]
                for item in reviewer.get("task_profiles", []):
                    profiles.append(
                        ReviewerProfile(
                            reviewer_id=reviewer_id,
                            task_profile=item["task_profile"],
                            allowed_scales=tuple(
                                float(value) for value in item.get("allowed_primary_magnifications", [])
                            ),
                            target_feature_allowlist=tuple(item.get("target_feature_allowlist", [])),
                            incidental_feature_allowlist=tuple(item.get("incidental_feature_allowlist", [])),
                            context_policies=tuple(
                                dict(policy) for policy in item.get("allowed_context_views", [])
                            ),
                            prompt_id=reviewer.get("prompt_id", "reviewer"),
                            prompt_version=reviewer.get("prompt_version", "reviewer_v1"),
                        )
                    )
            version = payload.get("registry_version", "reviewer_registry_v1")
        return cls(version, profiles, retry_policy=payload.get("retry_policy"))

    @classmethod
    def from_json(cls, path):
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _default_registry_path():
    return Path(__file__).resolve().parents[3] / "configs" / "agentflow" / "reviewer_registry_v1.json"


def default_reviewer_registry(path=None):
    candidate = Path(path) if path else _default_registry_path()
    if candidate.exists():
        return ReviewerRegistry.from_json(candidate)
    return ReviewerRegistry.from_dict(_DEFAULT_REGISTRY)


def build_reviewer_request(
    snapshot_id,
    plan_id,
    action,
    question,
    roi,
    registry,
    context_views=(),
    feature_disagreements=(),
):
    registry.validate_action(action, roi)
    profile = registry.profile(action.reviewer_id, action.task_profile)

    def roi_image(item):
        image_ref = item.image_path or "roi://{0}".format(item.roi_id)
        payload = {
            "roi_id": item.roi_id,
            "image_id": str(item.metadata.get("image_id", "IMG_{0}".format(item.roi_id))),
            "image_ref": image_ref,
            "image_sha256": item.image_sha256 or hashlib.sha256(image_ref.encode("utf-8")).hexdigest(),
            "level0_bbox": list(item.level0_bbox),
            "magnification": item.scale,
            "pixel_dimensions": list(item.pixel_dimensions or (512, 512)),
        }
        parent_roi_id = item.metadata.get("parent_roi_id") or item.metadata.get("parent_10x_roi_id")
        if parent_roi_id:
            payload["parent_roi_id"] = str(parent_roi_id)
        if item.mpp is not None:
            payload["mpp"] = float(item.mpp)
        return payload

    normalized_contexts = []
    for context in context_views:
        if not isinstance(context, (tuple, list)) or len(context) != 2:
            raise ReviewerContractError(
                "Context views must be explicit (view_role, ROICandidate) pairs"
            )
        role, item = context
        normalized_contexts.append((str(role), item))

    model_input = {
        "task_profile": action.task_profile,
        "target_features": list(action.target_features),
        "primary_roi": roi_image(roi),
        "context_views": [
            {"view_role": role, "roi": roi_image(item)}
            for role, item in normalized_contexts
        ],
    }
    if feature_disagreements:
        model_input["feature_disagreements"] = [
            dict(item) for item in feature_disagreements
        ]
    request = ReviewerTaskRequest(
        schema_version="reviewer_task_request_v1",
        request_id="REQ_{0}".format(action.action_id),
        action_id=action.action_id,
        plan_id=plan_id,
        question_id=action.question_id,
        snapshot_id=snapshot_id,
        reviewer=action.reviewer_id,
        model_input=model_input,
        primary_roi=roi,
        context_views=tuple(normalized_contexts),
        registry_version=registry.version,
        prompt_id=profile.prompt_id,
        prompt_version=profile.prompt_version,
    )
    registry.validate_request(request)
    return request


def reviewer_observation_to_evidence(
    case_id,
    request,
    observation,
    adjudicated_evidence_ids_by_feature=None,
):
    quality_reviewability = str(observation.quality.get("overall_evaluability", "limited"))
    quality_value = 1.0 if quality_reviewability == "adequate" else (0.5 if quality_reviewability == "limited" else 0.0)
    records = []
    adjudication_targets = {
        str(feature_id): tuple(evidence_ids)
        for feature_id, evidence_ids in dict(
            adjudicated_evidence_ids_by_feature or {}
        ).items()
    }
    for finding_group, findings in (
        ("target", observation.findings),
        ("incidental", observation.incidental_findings),
    ):
        for finding in findings:
            evidence_id = (
                "REV_{0}_{1}".format(request.request_id, finding.feature_id)
                if finding_group == "target"
                else "REV_{0}_INCIDENTAL_{1}".format(request.request_id, finding.feature_id)
            )
            superseded_ids = (
                adjudication_targets.get(finding.feature_id, tuple())
                if finding_group == "target"
                and finding.status in ("present", "absent")
                and finding.feature_evaluability == "adequate"
                else tuple()
            )
            records.append(EvidenceRecord(
                evidence_id=evidence_id,
                case_id=case_id,
                evidence_type="reviewer_evidence",
                feature=finding.feature_id,
                status=finding.status,
                confidence=finding.status_confidence,
                source=observation.reviewer,
                source_version=observation.model_version or "unspecified",
                quality=quality_value,
                feature_evaluability=finding.feature_evaluability,
                roi_id=request.primary_roi.roi_id,
                scale=request.primary_roi.scale,
                level0_bbox=request.primary_roi.level0_bbox,
                patch_id=(
                    request.primary_roi.source_patch_ids[0]
                    if len(request.primary_roi.source_patch_ids) == 1
                    else None
                ),
                cluster_id=request.primary_roi.source_cluster_id,
                planner_action_id=request.planner_action_id,
                input_snapshot_id=request.snapshot_id,
                model_version=observation.model_version,
                prompt_version=observation.prompt_version,
                limitations=tuple(finding.limitations) + tuple(observation.limitations),
                metadata={
                    "request_id": request.request_id,
                    "observation_id": observation.observation_id,
                    "reviewer_ledger_record_id": "RLR_{0}".format(observation.observation_id),
                    "task_profile": request.task_profile,
                    "scope": finding.scope,
                    "evidence_text": finding.evidence_text,
                    "registry_version": request.registry_version,
                    "image_ref": request.model_input["primary_roi"]["image_ref"],
                    "image_sha256": request.model_input["primary_roi"]["image_sha256"],
                    "pixel_dimensions": request.model_input["primary_roi"]["pixel_dimensions"],
                    "source_patch_ids": list(request.primary_roi.source_patch_ids),
                    "source_cluster_id": request.primary_roi.source_cluster_id,
                    "roi_metadata": dict(request.primary_roi.metadata),
                    "finding_group": finding_group,
                    "adjudicates_feature_disagreement": bool(superseded_ids),
                },
                supersedes_evidence_ids=superseded_ids,
            ))
    return records


def invocation_failure_evidence(case_id, request, error, attempt_count):
    return EvidenceRecord(
        evidence_id="FAIL_{0}".format(request.request_id),
        case_id=case_id,
        evidence_type="reviewer_invocation_failure",
        feature="reviewer_invocation:{0}".format(request.reviewer),
        status="invocation_failure",
        confidence=1.0,
        source="AgentFlowOrchestrator",
        source_version="v1",
        quality=0.0,
        feature_evaluability="not_evaluable",
        roi_id=request.primary_roi.roi_id,
        scale=request.primary_roi.scale,
        level0_bbox=request.primary_roi.level0_bbox,
        patch_id=(
            request.primary_roi.source_patch_ids[0]
            if len(request.primary_roi.source_patch_ids) == 1
            else None
        ),
        cluster_id=request.primary_roi.source_cluster_id,
        planner_action_id=request.planner_action_id,
        input_snapshot_id=request.snapshot_id,
        limitations=(str(error),),
        metadata={
            "request_id": request.request_id,
            "reviewer_ledger_record_id": "RLR_FAIL_{0}".format(request.request_id),
            "attempt_count": int(attempt_count),
        },
    )


def _ledger_provenance(
    case_id,
    request,
    registry,
    model_id,
    model_version,
    prompt_id,
    prompt_version,
    recorded_at=None,
):
    primary = request.model_input["primary_roi"]
    provenance = {
        "case_id": case_id,
        "request_id": request.request_id,
        "action_id": request.action_id,
        "snapshot_id": request.snapshot_id,
        "planner_action_id": request.action_id,
        "reviewer": request.reviewer,
        "model_id": str(model_id or "unconfigured_reviewer_backend"),
        "model_version": str(model_version or "unavailable"),
        "prompt_id": str(prompt_id or "reviewer"),
        "prompt_version": str(prompt_version or "unavailable"),
        "registry_version": registry.version,
        "roi_id": primary["roi_id"],
        "image_ref": primary["image_ref"],
        "image_sha256": primary["image_sha256"],
        "level0_bbox": list(primary["level0_bbox"]),
        "magnification": primary["magnification"],
        "source_models": ["{0}@{1}".format(model_id or "unconfigured", model_version or "unavailable")],
        "recorded_at": recorded_at or datetime.now(timezone.utc).isoformat(),
    }
    if primary.get("mpp") is not None:
        provenance["mpp"] = primary["mpp"]
    return provenance


def reviewer_ledger_evidence_record(
    case_id,
    request,
    observation,
    registry,
    model_id,
    model_version,
    prompt_id,
    prompt_version,
    recorded_at=None,
):
    return {
        "schema_version": "reviewer_ledger_record_v1",
        "record_id": "RLR_{0}".format(observation.observation_id),
        "record_type": "evidence",
        "request_id": request.request_id,
        "action_id": request.action_id,
        "plan_id": request.plan_id,
        "question_id": request.question_id,
        "snapshot_id": request.snapshot_id,
        "reviewer": request.reviewer,
        "task_profile": request.task_profile,
        "provenance": _ledger_provenance(
            case_id,
            request,
            registry,
            model_id,
            model_version,
            prompt_id,
            prompt_version,
            recorded_at=recorded_at,
        ),
        "observation": observation.to_dict(),
        "reasoning_use": {"status": "pending"},
    }


def reviewer_ledger_failure_record(
    case_id,
    request,
    registry,
    error,
    stage,
    attempt_count,
    repair_attempted,
    model_id,
    model_version,
    prompt_id,
    prompt_version,
    recorded_at=None,
):
    return {
        "schema_version": "reviewer_ledger_record_v1",
        "record_id": "RLR_FAIL_{0}".format(request.request_id),
        "record_type": "invocation_failure",
        "request_id": request.request_id,
        "action_id": request.action_id,
        "plan_id": request.plan_id,
        "question_id": request.question_id,
        "snapshot_id": request.snapshot_id,
        "reviewer": request.reviewer,
        "task_profile": request.task_profile,
        "provenance": _ledger_provenance(
            case_id,
            request,
            registry,
            model_id,
            model_version,
            prompt_id,
            prompt_version,
            recorded_at=recorded_at,
        ),
        "failure": {
            "stage": stage,
            "error_code": error.__class__.__name__,
            "message": str(error) or error.__class__.__name__,
            "retryable": False,
            "attempt_count": int(attempt_count),
            "repair_attempted": bool(repair_attempted),
        },
    }


class HttpReviewerBackend(object):
    """Adapter for the existing generic Qwen/PathReasoner HTTP service.

    Only the formal diagnosis-free Reviewer ``model_input`` is exposed to the
    model.  Contract validation remains owned by the Registry and Orchestrator.
    """

    synthetic_stub = False

    def __init__(
        self,
        endpoint,
        timeout_seconds=120,
        max_new_tokens=1200,
        model_id="shared_reviewer_http",
        model_version="unreported",
        prompt_version=None,
        session=None,
    ):
        self.endpoint = str(endpoint).strip()
        if not self.endpoint:
            raise ValueError("Reviewer HTTP endpoint is required")
        self.timeout_seconds = float(timeout_seconds)
        self.max_new_tokens = int(max_new_tokens)
        self.model_id = str(model_id)
        self.model_version = str(model_version)
        self.prompt_version = prompt_version
        self.session = session or requests.Session()
        self.requests = []
        self._repair_used = set()

    def invoke(self, request):
        self.requests.append(request)
        raw_text, response_metadata = self._call(
            request,
            self._prompt(request),
        )
        model_version = str(response_metadata.get("model_id") or self.model_version)
        self.model_version = model_version
        try:
            payload = _extract_json_payload(raw_text)
            return _reviewer_observation_from_payload(
                payload,
                request,
                model_version=model_version,
                prompt_version=self.prompt_version or request.prompt_version,
            )
        except ReviewerContractError as exc:
            return self._repair_text_once(request, raw_text, exc, model_version)

    def repair(self, request, candidate_observation, error):
        if request.request_id in self._repair_used:
            raise ReviewerContractError("Reviewer JSON repair limit exhausted")
        original = candidate_observation.to_dict()
        repaired = self._repair_text_once(
            request,
            json.dumps(original, ensure_ascii=False, sort_keys=True),
            error,
            candidate_observation.model_version or self.model_version,
        )
        validate_repair_preserves_semantics(candidate_observation, repaired)
        return repaired

    def _repair_text_once(self, request, raw_text, error, model_version):
        if request.request_id in self._repair_used:
            raise ReviewerContractError(
                "Reviewer output remained invalid after one constrained repair: {0}".format(error)
            )
        self._repair_used.add(request.request_id)
        prompt = (
            "Repair the JSON envelope below so it conforms to ReviewerObservationV1. "
            "Do not add, remove, reinterpret, or change any morphology finding, feature status, "
            "confidence, ROI identity, reviewer, task profile, or target feature. Return one JSON "
            "object only.\n\nFORMAL REQUEST:\n{0}\n\nVALIDATION ERROR:\n{1}\n\n"
            "ORIGINAL OUTPUT:\n{2}"
        ).format(
            json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True),
            str(error),
            raw_text,
        )
        repaired_text, metadata = self._call(request, prompt)
        payload = _extract_json_payload(repaired_text)
        return _reviewer_observation_from_payload(
            payload,
            request,
            model_version=str(metadata.get("model_id") or model_version),
            prompt_version=self.prompt_version or request.prompt_version,
        )

    def _call(self, request, prompt):
        image_paths = []
        primary = request.model_input.get("primary_roi", {})
        primary_ref = str(primary.get("image_ref", ""))
        if primary_ref and not primary_ref.startswith("roi://"):
            image_paths.append(primary_ref)
        for context in request.model_input.get("context_views", ()):  # formal neutral contexts
            image_ref = str(context.get("roi", {}).get("image_ref", ""))
            if image_ref and not image_ref.startswith("roi://") and image_ref not in image_paths:
                image_paths.append(image_ref)
        try:
            response = self.session.post(
                self.endpoint,
                json={
                    "image_paths": image_paths,
                    "prompt": prompt,
                    "max_new_tokens": self.max_new_tokens,
                    "stage": "reviewer",
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise ModelUnavailableError("Reviewer HTTP request failed: {0}".format(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise ReviewerContractError("Reviewer HTTP response is not JSON: {0}".format(exc)) from exc
        if not isinstance(payload, Mapping) or "text" not in payload:
            raise ReviewerContractError("Reviewer HTTP response lacks text")
        return str(payload["text"]), dict(payload)

    @staticmethod
    def _prompt(request):
        output_contract = {
            "schema_version": "reviewer_observation_v1",
            "observation_id": "OBS_<request_id>",
            "request_id": request.request_id,
            "reviewer": request.reviewer,
            "task_profile": request.task_profile,
            "primary_roi_id": request.primary_roi.roi_id,
            "primary_magnification": request.primary_roi.scale,
            "target_features": list(request.target_features),
            "quality": {
                "overall_evaluability": "adequate|limited|not_evaluable",
                "status_confidence": "number 0..1",
                "adequate_for_requested_features": "boolean",
                "limitations": [],
            },
            "findings": [
                {
                    "feature_id": "exact requested feature id",
                    "status": "present|absent|uncertain|not_evaluable",
                    "status_confidence": "number 0..1",
                    "feature_evaluability": "adequate|limited|not_evaluable",
                    "scope": "roi_overview|roi_local",
                    "evidence_text": "short morphology-only observation",
                    "limitations": [],
                }
            ],
            "incidental_findings": [],
            "overall_evidence_strength": {
                "level": "none|weak|moderate|strong",
                "score": "number 0..1",
            },
            "limitations": [],
            "does_not_decide_final_diagnosis": True,
        }
        return (
            "Act only as the named pathology Reviewer. Describe visible morphology features; do not "
            "rank hypotheses, choose a diagnosis, select another ROI, or propose a next action. Every "
            "target feature must appear exactly once in findings. An absent finding is allowed only "
            "when that feature is adequately evaluable. Return JSON only.\n\nMODEL INPUT:\n{0}\n\n"
            "OUTPUT CONTRACT TEMPLATE:\n{1}"
        ).format(
            json.dumps(request.model_input, ensure_ascii=False, sort_keys=True),
            json.dumps(output_contract, ensure_ascii=False, sort_keys=True),
        )


class ScriptedReviewerBackend(object):
    """Explicitly non-clinical backend for contract and orchestration tests."""

    synthetic_stub = True

    def __init__(
        self,
        scripts,
        model_id="scripted-reviewer",
        model_version="scripted-reviewer-v1",
        prompt_version=None,
    ):
        self.scripts = {key: list(value) for key, value in scripts.items()}
        self.model_id = model_id
        self.model_version = model_version
        self.prompt_version = prompt_version
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        queue = self.scripts.get(request.question_id) or self.scripts.get(request.reviewer)
        if not queue:
            raise ModelUnavailableError("No scripted response for {0}".format(request.question_id))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, ReviewerObservation):
            return item
        if not isinstance(item, Mapping):
            raise TypeError("Scripted response must be ReviewerObservation, mapping, or Exception")
        findings = []
        for feature in request.target_features:
            value = item.get(feature, item.get("default", "uncertain"))
            if isinstance(value, str):
                value = {"status": value}
            status = value.get("status", "uncertain")
            evaluability = value.get(
                "feature_evaluability",
                "not_evaluable" if status == "not_evaluable" else "adequate",
            )
            findings.append(
                ReviewerFinding(
                    feature_id=feature,
                    status=status,
                    status_confidence=float(value.get("confidence", 0.9)),
                    feature_evaluability=evaluability,
                    scope=value.get("scope", "roi_local"),
                    evidence_text=value.get("evidence_text", "Scripted non-clinical observation for {0}.".format(feature)),
                    limitations=tuple(value.get("limitations", [])),
                    quantitation=(
                        value.get(
                            "quantitation",
                            {
                                "metric": "villous_component_extent",
                                "category": "indeterminate",
                            },
                        )
                        if feature == "villous_component_extent_estimate"
                        else None
                    ),
                )
            )
        reviewability = item.get("overall_evaluability", item.get("reviewability", "adequate"))
        strength = item.get("overall_evidence_strength", 0.9)
        if isinstance(strength, Mapping):
            strength_payload = dict(strength)
        else:
            strength_score = float(strength)
            strength_payload = {
                "level": (
                    "strong"
                    if strength_score >= 0.75
                    else "moderate"
                    if strength_score >= 0.5
                    else "weak"
                    if strength_score > 0.0
                    else "none"
                ),
                "score": strength_score,
            }
        return ReviewerObservation(
            schema_version="reviewer_observation_v1",
            observation_id="OBS_{0}".format(request.request_id),
            request_id=request.request_id,
            reviewer=request.reviewer,
            task_profile=request.task_profile,
            primary_roi_id=request.primary_roi.roi_id,
            primary_magnification=request.primary_roi.scale,
            target_features=tuple(request.target_features),
            quality={
                "overall_evaluability": reviewability,
                "status_confidence": float(item.get("quality_confidence", 0.95)),
                "adequate_for_requested_features": bool(
                    item.get("adequate_for_requested_features", reviewability != "not_evaluable")
                ),
                "limitations": list(item.get("quality_limitations", [])),
            },
            findings=tuple(findings),
            incidental_findings=tuple(),
            overall_evidence_strength=strength_payload,
            limitations=tuple(item.get("limitations", [])),
            does_not_decide_final_diagnosis=True,
            model_version=self.model_version,
            prompt_version=self.prompt_version or request.prompt_version,
        )


class UnavailableReviewerBackend(object):
    synthetic_stub = False

    def __init__(self, reason="Reviewer model/backend is not configured"):
        self.reason = reason
        self.model_id = "unconfigured_reviewer_backend"
        self.model_version = "unavailable"
        self.prompt_version = None

    def invoke(self, request):
        raise ModelUnavailableError(self.reason)


_DEFAULT_REGISTRY = {
    "version": "reviewer_registry_v1",
    "profiles": [
        {
            "reviewer_id": "QualityMucosaReviewer",
            "task_profile": "overview_evaluability",
            "allowed_scales": [2.5, 5.0],
            "target_feature_allowlist": ["reviewable_mucosa"],
            "incidental_feature_allowlist": [],
            "allowed_context_roles": ["overview", "parent"],
        },
        {
            "reviewer_id": "SerratedArchitectureReviewer",
            "task_profile": "ssl_hp_discrimination",
            "allowed_scales": [5.0, 10.0],
            "target_feature_allowlist": [
                "serration_to_crypt_base",
                "basal_crypt_dilation",
                "surface_limited_serration",
                "straight_crypt_bases",
                "crypt_branching",
                "horizontal_or_boot_shaped_crypt",
            ],
            "incidental_feature_allowlist": ["serration_present"],
            "allowed_context_roles": ["overview", "parent"],
        },
        {
            "reviewer_id": "TSAReviewer",
            "task_profile": "tsa_architecture",
            "allowed_scales": [5.0, 10.0],
            "target_feature_allowlist": [
                "ectopic_crypt_formation",
                "slit_like_serration",
                "villiform_or_filiform_serrated_architecture",
                "cytoplasmic_eosinophilia",
                "pencillate_nuclei",
            ],
            "incidental_feature_allowlist": [],
            "allowed_context_roles": ["overview", "parent"],
        },
        {
            "reviewer_id": "ConventionalArchitectureReviewer",
            "task_profile": "tubular_villous_resolution",
            "allowed_scales": [5.0, 10.0],
            "target_feature_allowlist": [
                "villous_component_present",
                "tubular_villous_mixing",
                "crowded_adenomatous_glands",
            ],
            "incidental_feature_allowlist": [],
            "allowed_context_roles": ["overview", "parent", "peer"],
        },
        {
            "reviewer_id": "DysplasiaReviewer",
            "task_profile": "high_grade_dysplasia_assessment",
            "allowed_scales": [20.0],
            "target_feature_allowlist": ["high_grade_or_definite_dysplasia"],
            "incidental_feature_allowlist": [
                "nuclear_stratification",
                "hyperchromasia",
                "loss_of_polarity",
                "mitotic_activity",
            ],
            "allowed_context_roles": ["parent", "peer"],
        },
        {
            "reviewer_id": "InflammatoryReactiveReviewer",
            "task_profile": "reactive_mimic_resolution",
            "allowed_scales": [5.0, 10.0],
            "target_feature_allowlist": [
                "reactive_regenerative_change",
                "adenomatous_architecture_absent",
                "serrated_architecture_absent",
                "dysplasia_not_evaluable_due_to_reactive_change",
                "erosion",
                "mixed_inflammation",
            ],
            "incidental_feature_allowlist": ["granulation_tissue"],
            "allowed_context_roles": ["overview", "parent"],
        },
    ],
}
