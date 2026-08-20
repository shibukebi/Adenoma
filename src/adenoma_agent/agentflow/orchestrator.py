import json
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from adenoma_agent.agentflow.architecture_runtime import ArchitectureInferenceRuntime, write_architecture_predictions
from adenoma_agent.agentflow.chief import RuleBasedChiefAgent
from adenoma_agent.agentflow.contracts import AgentFlowResult
from adenoma_agent.agentflow.evidence import EvidenceEngine
from adenoma_agent.agentflow.ledger import EvidenceLedger
from adenoma_agent.agentflow.planner import PlanningAgent
from adenoma_agent.agentflow.reviewer import (
    ModelUnavailableError,
    ReviewerContractError,
    UnavailableReviewerBackend,
    build_reviewer_request,
    default_reviewer_registry,
    invocation_failure_evidence,
    reviewer_ledger_evidence_record,
    reviewer_ledger_failure_record,
    reviewer_observation_to_evidence,
    validate_repair_preserves_semantics,
)
from adenoma_agent.agentflow.schema_validation import ReviewerJsonSchemaValidator
from adenoma_agent.agentflow.spatial import ROIManager, SpatialEvidenceEvaluator
from adenoma_agent.agentflow.state import AgentTraceStore, StateStore, TerminationState


class ROICropperError(RuntimeError):
    pass


@dataclass(frozen=True)
class CropArtifact(object):
    image_ref: str
    image_sha256: str
    pixel_dimensions: tuple
    mpp: object = None
    hash_scope: str = "image_content"


class ROICropper(object):
    synthetic_stub = False

    def crop(self, roi, output_dir):
        raise NotImplementedError


class ExistingImageROICropper(ROICropper):
    """Production-safe default: accepts only a pre-existing ROI image."""

    def crop(self, roi, output_dir):
        if roi.image_path and Path(roi.image_path).exists():
            path = Path(roi.image_path).resolve()
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            with Image.open(path) as image:
                dimensions = tuple(int(value) for value in image.size)
            return CropArtifact(
                image_ref=str(path),
                image_sha256=digest.hexdigest(),
                pixel_dimensions=dimensions,
                mpp=roi.mpp,
            )
        raise ROICropperError(
            "ROI {0} has no materialized image; configure a WSI cropper adapter".format(roi.roi_id)
        )


class VirtualROICropper(ROICropper):
    """Explicitly synthetic cropper for no-weight/no-WSI control-flow tests."""

    synthetic_stub = True

    def __init__(self):
        self.crops = []

    def crop(self, roi, output_dir):
        image_ref = "roi://{0}".format(roi.roi_id)
        digest = hashlib.sha256(image_ref.encode("utf-8")).hexdigest()
        artifact = CropArtifact(
            image_ref=image_ref,
            image_sha256=digest,
            pixel_dimensions=(512, 512),
            mpp=roi.mpp,
            hash_scope="reference_only_synthetic",
        )
        self.crops.append(
            {
                "roi_id": roi.roi_id,
                "image_ref": image_ref,
                "image_sha256": digest,
                "pixel_dimensions": artifact.pixel_dimensions,
            }
        )
        return artifact


class AgentFlowOrchestrator(object):
    """Side-effect owner for the Planner -> Reviewer -> Ledger loop."""

    def __init__(
        self,
        planner=None,
        reviewer_backend=None,
        registry=None,
        chief_agent=None,
        spatial_evaluator=None,
        roi_manager=None,
        cropper=None,
        max_actions=8,
        max_retries_per_reviewer=None,
        max_schema_repairs=None,
        schema_validator=None,
        require_contract_validator=False,
        evidence_engine=None,
    ):
        self.registry = registry or default_reviewer_registry()
        self.planner = planner or PlanningAgent(registry=self.registry)
        self.reviewer_backend = reviewer_backend or UnavailableReviewerBackend()
        self.chief_agent = chief_agent or RuleBasedChiefAgent(self.planner.knowledge_base)
        self.spatial_evaluator = spatial_evaluator or SpatialEvidenceEvaluator()
        self.roi_manager = roi_manager or ROIManager()
        self.cropper = cropper or ExistingImageROICropper()
        self.max_actions = int(max_actions)
        retry_policy = self.registry.retry_policy
        self.max_retries_per_reviewer = int(
            retry_policy.get("transport_or_model_retries", 1)
            if max_retries_per_reviewer is None
            else max_retries_per_reviewer
        )
        self.max_schema_repairs = int(
            retry_policy.get("schema_repair_attempts", 1)
            if max_schema_repairs is None
            else max_schema_repairs
        )
        self.schema_validator = schema_validator or ReviewerJsonSchemaValidator(
            require_dependency=require_contract_validator
        )
        self.evidence_engine = evidence_engine or EvidenceEngine(self.planner.knowledge_base)

    def run_from_manifest(
        self,
        case_id,
        five_x_manifest_path,
        architecture_runtime=None,
        initial_evidence=(),
        slide_dimensions=None,
        output_dir=None,
        case_context=None,
        reviewer_availability=None,
    ):
        runtime = architecture_runtime or ArchitectureInferenceRuntime()
        predictions = runtime.run(five_x_manifest_path)
        return self.run(
            case_id=case_id,
            predictions=predictions,
            initial_evidence=initial_evidence,
            slide_dimensions=slide_dimensions,
            output_dir=output_dir,
            case_context=case_context,
            reviewer_availability=reviewer_availability,
        )

    def run(
        self,
        case_id,
        predictions,
        initial_evidence=(),
        slide_dimensions=None,
        output_dir=None,
        case_context=None,
        reviewer_availability=None,
    ):
        predictions = tuple(predictions)
        if not predictions:
            raise ValueError("AgentFlow requires architecture predictions")
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            ledger_path = output_path / "ledger" / "evidence.jsonl"
        else:
            ledger_path = None
        ledger = EvidenceLedger(case_id=case_id, jsonl_path=ledger_path, replay=True)
        spatial = self.spatial_evaluator.evaluate(predictions)
        roi_candidates = self.roi_manager.build_candidates(predictions, slide_dimensions=slide_dimensions)
        architecture_evidence = self.spatial_evaluator.to_evidence(case_id, predictions, spatial)
        ledger.append_many(tuple(initial_evidence) + tuple(architecture_evidence))
        state_store = (
            StateStore.for_output_dir(case_id, output_path, replay=True)
            if output_path
            else StateStore(case_id)
        )
        trace_store = (
            AgentTraceStore.for_output_dir(case_id, output_path, replay=True)
            if output_path
            else AgentTraceStore(case_id)
        )
        sanitized_context = self._sanitize_case_context(case_context)
        state = state_store.latest
        if state is None:
            state = self.evidence_engine.initialize_state(
                case_id=case_id,
                case_context=sanitized_context,
                budget={"max_actions": self.max_actions, "actions_used": 0},
            )
            state_store.append(state)
        current_snapshot = ledger.snapshot()
        if state.ledger_snapshot_id != current_snapshot.snapshot_id:
            previous_state = state
            state = self.evidence_engine.update(
                previous_state,
                current_snapshot,
                round_id=previous_state.round_id + 1,
            )
            state_store.append(state)
            trace_store.append(
                self._state_trace_event(
                    case_id=case_id,
                    event_id="EVENT_INITIAL_{0}".format(state.state_version),
                    round_id=state.round_id,
                    actor="EvidenceEngine",
                    input_state=previous_state,
                    output_state=state,
                    action="initialize_from_architecture_and_case_evidence",
                    belief_update=(state.belief_history[-1] if state.belief_history else None),
                )
            )
        availability = defaultdict(lambda: True)
        availability.update(dict(reviewer_availability or {}))
        if output_path:
            write_architecture_predictions(output_path / "architecture" / "patch_evidence.jsonl", predictions)
            self._write_json(
                output_path / "architecture" / "manifest.json",
                {
                    "schema_version": "architecture_inference_manifest_v1",
                    "canonical_input_scale": "5x",
                    "prediction_count": len(predictions),
                    "source_models": sorted(set(row.source_model for row in predictions)),
                    "all_embeddings_traceable": all(bool(row.embedding_ref) for row in predictions),
                    "non_clinical": True,
                },
            )
            self._write_json(output_path / "spatial" / "spatial_evidence.json", spatial.to_dict())
            self._write_jsonl(
                output_path / "spatial" / "roi_candidates.jsonl",
                [candidate.to_dict() for candidate in roi_candidates],
            )
            self._write_jsonl(
                output_path / "spatial" / "clusters.jsonl",
                [
                    candidate.to_dict()
                    for candidate in roi_candidates
                    if candidate.scale == 5.0 and candidate.source_cluster_id
                ],
            )
        plans = []
        final_plan = None
        materialized_rois = {}
        for round_index in range(self.max_actions + 1):
            snapshot = ledger.snapshot()
            plan = self.planner.plan(
                snapshot,
                roi_candidates,
                round_index=round_index,
                max_actions=self.max_actions,
                agent_state=state,
                reviewer_availability=availability,
            )
            plans.append(plan)
            final_plan = plan
            previous_state = state
            termination = TerminationState(
                decision=plan.termination_status,
                reason=(plan.stop_reason if plan.stop else None),
                round_id=previous_state.round_id,
                criteria=dict(plan.priority_trace),
                unresolved_discriminator_ids=tuple(
                    item.discriminator_id
                    for item in previous_state.discriminators
                    if item.status != "resolved"
                ),
                blocking_relation_ids=tuple(plan.active_relation_ids),
            )
            state = replace(
                previous_state,
                state_id="STATE_{0}_{1:06d}_PLAN".format(
                    case_id,
                    previous_state.state_version + 1,
                ),
                state_version=previous_state.state_version + 1,
                reviewer_plan=plan.to_dict(),
                hypotheses=tuple(
                    replace(item, status="resolved")
                    if plan.termination_status == "sufficient_evidence_stop"
                    and plan.ranked_hypotheses
                    and item.hypothesis_id == plan.ranked_hypotheses[0].hypothesis_id
                    else item
                    for item in previous_state.hypotheses
                ),
                contradictions=tuple(
                    replace(item, resolution_status="recheck_planned")
                    if plan.priority_trace.get("reason") == "true_contradiction_recheck"
                    and item.contradiction_id in set(plan.active_relation_ids)
                    else item
                    for item in previous_state.contradictions
                ),
                budget={
                    "max_actions": self.max_actions,
                    "actions_used": int(round_index),
                    "remaining_actions": max(0, self.max_actions - int(round_index)),
                },
                termination_state=termination,
                previous_state_id=previous_state.state_id,
            )
            state_store.append(state)
            trace_store.append(
                self._state_trace_event(
                    case_id=case_id,
                    event_id="EVENT_PLAN_{0:03d}_{1}".format(round_index, plan.plan_id),
                    round_id=previous_state.round_id,
                    actor="ReviewerPlanner",
                    input_state=previous_state,
                    output_state=state,
                    action=("stop" if plan.stop else "invoke_reviewer"),
                    plan=plan,
                )
            )
            if output_path:
                self._append_jsonl(output_path / "planning" / "plan_decisions.jsonl", plan.to_dict())
            if plan.stop:
                break
            ledger.assert_current(plan.snapshot_id)
            roi = self._roi_by_id(roi_candidates, plan.selected_action.roi_id)
            prepared_roi = self._materialize_roi(roi, output_path, materialized_rois)
            profile = self.registry.profile(
                plan.selected_action.reviewer_id,
                plan.selected_action.task_profile,
            )
            prepared_context_views = []
            for view_role, context_roi in self._select_context_views(roi, roi_candidates, profile):
                prepared_context_views.append(
                    (
                        view_role,
                        self._materialize_roi(context_roi, output_path, materialized_rois),
                    )
                )
            feature_disagreements = self._feature_disagreements(
                state,
                snapshot,
                plan.selected_action,
            )
            request = build_reviewer_request(
                snapshot_id=plan.snapshot_id,
                plan_id=plan.plan_id,
                action=plan.selected_action,
                question=plan.selected_question,
                roi=prepared_roi,
                registry=self.registry,
                context_views=tuple(prepared_context_views),
                feature_disagreements=feature_disagreements,
            )
            self.schema_validator.validate_request(request.to_dict())
            if output_path:
                self._append_jsonl(output_path / "reviewers" / "requests.jsonl", request.to_dict())
            observation = None
            last_error = None
            failure_stage = "model"
            repair_attempted = False
            attempts = self.max_retries_per_reviewer + 1
            attempts_used = 0
            invocation_started = time.monotonic()
            for _attempt in range(attempts):
                attempts_used = _attempt + 1
                try:
                    candidate_observation = self.reviewer_backend.invoke(request)
                except Exception as exc:
                    last_error = exc
                    if isinstance(exc, ModelUnavailableError):
                        failure_stage = "model"
                    elif isinstance(exc, (ReviewerContractError, ValueError, TypeError, AttributeError)):
                        failure_stage = "schema_validation"
                        break
                    else:
                        failure_stage = "transport"
                    continue
                try:
                    self.registry.validate_observation(request, candidate_observation)
                    self.schema_validator.validate_observation(candidate_observation.to_dict())
                except (ReviewerContractError, ValueError, TypeError, AttributeError) as exc:
                    last_error = exc
                    failure_stage = "schema_validation"
                    repair = getattr(self.reviewer_backend, "repair", None)
                    if callable(repair) and self.max_schema_repairs > 0:
                        repair_attempted = True
                        try:
                            repaired_observation = repair(request, candidate_observation, exc)
                            validate_repair_preserves_semantics(
                                candidate_observation,
                                repaired_observation,
                            )
                            self.registry.validate_observation(request, repaired_observation)
                            self.schema_validator.validate_observation(repaired_observation.to_dict())
                            observation = repaired_observation
                            last_error = None
                        except Exception as repair_exc:
                            last_error = repair_exc
                    break
                observation = candidate_observation
                last_error = None
                break
            if observation is None:
                failure = invocation_failure_evidence(case_id, request, last_error, attempts_used)
                ledger.append(failure, created_after_action_id=plan.selected_action.action_id)
                if output_path:
                    profile = self.registry.profile(request.reviewer, request.task_profile)
                    formal_failure_record = reviewer_ledger_failure_record(
                        case_id=case_id,
                        request=request,
                        registry=self.registry,
                        error=last_error,
                        stage=failure_stage,
                        attempt_count=attempts_used,
                        repair_attempted=repair_attempted,
                        model_id=getattr(
                            self.reviewer_backend,
                            "model_id",
                            self.reviewer_backend.__class__.__name__,
                        ),
                        model_version=getattr(self.reviewer_backend, "model_version", "unavailable"),
                        prompt_id=profile.prompt_id,
                        prompt_version=getattr(
                            self.reviewer_backend,
                            "prompt_version",
                            profile.prompt_version,
                        )
                        or profile.prompt_version,
                    )
                    self.schema_validator.validate_ledger_record(formal_failure_record)
                    self._append_jsonl(
                        output_path / "reviewers" / "invocation_records.jsonl",
                        formal_failure_record,
                    )
                    self._append_jsonl(
                        output_path / "reviewers" / "invocation_audit.jsonl",
                        {
                            "request_id": request.request_id,
                            "action_id": request.action_id,
                            "status": "invocation_failure",
                            "failure_stage": failure_stage,
                            "attempt_count": attempts_used,
                            "repair_attempted": repair_attempted,
                            "latency_ms": int(round((time.monotonic() - invocation_started) * 1000.0)),
                            "error": str(last_error),
                        },
                    )
                if isinstance(last_error, ModelUnavailableError):
                    availability[request.reviewer] = False
                    if isinstance(self.reviewer_backend, UnavailableReviewerBackend):
                        for registry_profile in self.registry.profiles:
                            availability[registry_profile.reviewer_id] = False
                previous_state = state
                failure_row = {
                    "request_id": request.request_id,
                    "action_id": request.action_id,
                    "reviewer": request.reviewer,
                    "task_profile": request.task_profile,
                    "roi_id": request.primary_roi.roi_id,
                    "failure_stage": failure_stage,
                    "attempt_count": attempts_used,
                    "error": str(last_error),
                }
                state = self.evidence_engine.update(
                    previous_state,
                    ledger.snapshot(),
                    round_id=previous_state.round_id + 1,
                )
                state = replace(
                    state,
                    discriminators=tuple(
                        replace(item, status="blocked")
                        if item.preferred_reviewer == request.reviewer
                        and item.status not in ("resolved", "conflicting")
                        else item
                        for item in state.discriminators
                    ),
                    contradictions=tuple(
                        replace(item, resolution_status="blocked")
                        if item.feature_id in set(request.target_features)
                        else item
                        for item in state.contradictions
                    ),
                    tool_reviewer_failures=tuple(previous_state.tool_reviewer_failures)
                    + (failure_row,),
                    review_history=tuple(previous_state.review_history)
                    + (dict(failure_row, status="invocation_failure"),),
                )
                state_store.append(state)
                trace_store.append(
                    self._state_trace_event(
                        case_id=case_id,
                        event_id="EVENT_FAILURE_{0}".format(request.request_id),
                        round_id=state.round_id,
                        actor="ReviewerRuntime",
                        input_state=previous_state,
                        output_state=state,
                        action="reviewer_invocation_failure",
                        plan=plan,
                        request_id=request.request_id,
                        evidence_ids=(failure.evidence_id,),
                        belief_update=state.belief_history[-1],
                    )
                )
                continue
            reviewer_records = reviewer_observation_to_evidence(
                case_id,
                request,
                observation,
                adjudicated_evidence_ids_by_feature=self._adjudication_targets(
                    state,
                    snapshot,
                    request,
                ),
            )
            ledger.append_many(reviewer_records, created_after_action_id=plan.selected_action.action_id)
            if output_path:
                self._append_jsonl(output_path / "reviewers" / "observations.jsonl", observation.to_dict())
                profile = self.registry.profile(request.reviewer, request.task_profile)
                formal_evidence_record = reviewer_ledger_evidence_record(
                    case_id=case_id,
                    request=request,
                    observation=observation,
                    registry=self.registry,
                    model_id=getattr(
                        self.reviewer_backend,
                        "model_id",
                        self.reviewer_backend.__class__.__name__,
                    ),
                    model_version=observation.model_version,
                    prompt_id=profile.prompt_id,
                    prompt_version=observation.prompt_version or profile.prompt_version,
                )
                self.schema_validator.validate_ledger_record(formal_evidence_record)
                self._append_jsonl(
                    output_path / "reviewers" / "invocation_records.jsonl",
                    formal_evidence_record,
                )
                self._append_jsonl(
                    output_path / "reviewers" / "invocation_audit.jsonl",
                    {
                        "request_id": request.request_id,
                        "action_id": request.action_id,
                        "status": "evidence",
                        "attempt_count": attempts_used,
                        "repair_attempted": repair_attempted,
                        "latency_ms": int(round((time.monotonic() - invocation_started) * 1000.0)),
                    },
                )
            previous_state = state
            review_row = {
                "request_id": request.request_id,
                "observation_id": observation.observation_id,
                "reviewer_ledger_record_id": "RLR_{0}".format(observation.observation_id),
                "action_id": request.action_id,
                "reviewer": request.reviewer,
                "task_profile": request.task_profile,
                "roi_id": request.primary_roi.roi_id,
                "magnification": request.primary_roi.scale,
                "status": "evidence",
                "evidence_ids": [record.evidence_id for record in reviewer_records],
            }
            state = self.evidence_engine.update(
                previous_state,
                ledger.snapshot(),
                round_id=previous_state.round_id + 1,
            )
            state = replace(
                state,
                review_history=tuple(previous_state.review_history) + (review_row,),
            )
            state_store.append(state)
            trace_store.append(
                self._state_trace_event(
                    case_id=case_id,
                    event_id="EVENT_OBSERVATION_{0}".format(observation.observation_id),
                    round_id=state.round_id,
                    actor="EvidenceUpdater",
                    input_state=previous_state,
                    output_state=state,
                    action="reviewer_observation_to_belief_update",
                    plan=plan,
                    request_id=request.request_id,
                    observation_ids=(observation.observation_id,),
                    evidence_ids=tuple(record.evidence_id for record in reviewer_records),
                    belief_update=state.belief_history[-1],
                )
            )
        if final_plan is None:
            raise RuntimeError("Planning loop produced no PlanDecision")
        final_snapshot = ledger.snapshot()
        chief_decision = self.chief_agent.decide(
            final_plan,
            final_snapshot,
            round_index=len(plans) - 1,
            agent_state=state,
        )
        chief_trace = self._chief_decision_trace(state, final_plan, chief_decision)
        trace_store.append(
            self._state_trace_event(
                case_id=case_id,
                event_id="EVENT_CHIEF_{0}".format(final_plan.plan_id),
                round_id=state.round_id,
                actor="Chief",
                input_state=state,
                output_state=state,
                action="chief_final_decision",
                plan=final_plan,
            )
        )
        synthetic_stub = bool(
            getattr(self.reviewer_backend, "synthetic_stub", False)
            or getattr(self.cropper, "synthetic_stub", False)
            or any(bool(row.metadata.get("synthetic_stub")) for row in predictions)
        )
        result = AgentFlowResult(
            case_id=case_id,
            final_snapshot_id=final_snapshot.snapshot_id,
            plans=tuple(plans),
            chief_decision=chief_decision,
            spatial_evidence=spatial,
            roi_candidates=tuple(roi_candidates),
            ledger_path=(str(ledger_path) if ledger_path else None),
            synthetic_stub=synthetic_stub,
            non_clinical=True,
            final_state_id=state.state_id,
            state_path=(str(state_store.jsonl_path) if state_store.jsonl_path else None),
            trace_path=(str(trace_store.jsonl_path) if trace_store.jsonl_path else None),
        )
        if output_path:
            self._write_json(output_path / "chief" / "final_decision.json", chief_decision.to_dict())
            self._write_json(output_path / "chief" / "decision_trace.json", chief_trace)
            self._write_json(
                output_path / "chief" / "guideline_retrieval.json",
                chief_decision.management_recommendation
                or {
                    "status": "not_run",
                    "reason": "Diagnosis is uncertain; guideline retrieval is only allowed after diagnosis.",
                    "knowledge_version": chief_decision.knowledge_version,
                    "non_clinical": True,
                },
            )
            self._write_json(
                output_path / "run_manifest.json",
                {
                    "schema_version": "agentflow_run_v1",
                    "case_id": case_id,
                    "canonical_architecture_input_scale": "5x",
                    "final_snapshot_id": final_snapshot.snapshot_id,
                    "final_state_id": state.state_id,
                    "termination_status": state.termination_state.decision,
                    "termination_reason": state.termination_state.reason,
                    "diagnostic_scope_status": state.diagnostic_scope_status,
                    "plan_count": len(plans),
                    "roi_candidate_count": len(roi_candidates),
                    "evidence_count": len(final_snapshot.records),
                    "final_status": chief_decision.status,
                    "final_label": chief_decision.final_label,
                    "synthetic_stub": synthetic_stub,
                    "non_clinical": True,
                },
            )
        return result

    @staticmethod
    def _sanitize_case_context(case_context):
        allowed_types = {
            "case_alias": str,
            "source_code": str,
            "source_format": str,
            "slide_id": str,
            "diagnostic_scope_status": str,
            "wsi_metadata_ref": str,
            "five_x_manifest_ref": str,
            "architecture_predictions_ref": str,
            "non_clinical": bool,
        }
        payload = dict(case_context or {})
        unknown = set(payload) - set(allowed_types)
        if unknown:
            raise ValueError(
                "case_context contains fields outside the inference allowlist: {0}".format(
                    sorted(unknown)
                )
            )
        for key, value in payload.items():
            expected_type = allowed_types[key]
            if not isinstance(value, expected_type):
                raise ValueError(
                    "case_context.{0} must be a flat {1}; nested metadata is not inference-safe".format(
                        key,
                        expected_type.__name__,
                    )
                )
        if payload.get("diagnostic_scope_status") not in (
            None,
            "adequate",
            "incomplete",
            "out_of_scope_suspected",
        ):
            raise ValueError("case_context diagnostic_scope_status is invalid")
        payload.setdefault("diagnostic_scope_status", "incomplete")
        payload.setdefault("non_clinical", True)
        return payload

    @staticmethod
    def _state_trace_event(
        case_id,
        event_id,
        round_id,
        actor,
        input_state,
        output_state,
        action,
        plan=None,
        request_id=None,
        observation_ids=(),
        evidence_ids=(),
        belief_update=None,
    ):
        return {
            "schema_version": "agent_trace_event_v1",
            "event_id": str(event_id),
            "case_id": str(case_id),
            "round_id": int(round_id),
            "actor": str(actor),
            "input_state_id": input_state.state_id,
            "output_state_id": output_state.state_id,
            "action": str(action),
            "plan_id": (plan.plan_id if plan else None),
            "reviewer_request_id": request_id,
            "observation_ids": list(observation_ids),
            "evidence_ids": list(evidence_ids),
            "belief_update_id": (belief_update.update_id if belief_update else None),
            "hypothesis_scores_before": {
                item.hypothesis_id: item.evidence_score for item in input_state.hypotheses
            },
            "hypothesis_scores_after": {
                item.hypothesis_id: item.evidence_score for item in output_state.hypotheses
            },
            "ranking_before": {
                item.hypothesis_id: item.ranking_score for item in input_state.hypotheses
            },
            "ranking_after": {
                item.hypothesis_id: item.ranking_score for item in output_state.hypotheses
            },
            "termination_status": output_state.termination_state.decision,
            "termination_reason": output_state.termination_state.reason,
            "caused_by_evidence_ids": (
                list(plan.caused_by_evidence_ids) if plan else list(evidence_ids)
            ),
            "selected_discriminator_id": (
                plan.selected_discriminator_id if plan else None
            ),
            "selected_question_id": (
                plan.selected_question.question_id
                if plan and plan.selected_question
                else None
            ),
            "selected_reviewer_id": (
                plan.selected_action.reviewer_id
                if plan and plan.selected_action
                else None
            ),
            "selected_task_profile": (
                plan.selected_action.task_profile
                if plan and plan.selected_action
                else None
            ),
            "selected_roi_id": (
                plan.selected_action.roi_id
                if plan and plan.selected_action
                else None
            ),
            "selected_scale": (
                plan.selected_action.scale
                if plan and plan.selected_action
                else None
            ),
            "priority_trace": (dict(plan.priority_trace) if plan else {}),
        }

    @staticmethod
    def _feature_disagreements(state, snapshot, action):
        records = {record.evidence_id: record for record in snapshot.records}
        output = []
        for relation in state.evidence_relations:
            if relation.relation_type not in ("true_contradiction", "quality_disagreement"):
                continue
            if relation.feature_id not in set(action.target_features):
                continue
            related = [records[evidence_id] for evidence_id in relation.evidence_ids if evidence_id in records]
            if len(related) < 2:
                continue
            observation_ids = []
            for record in related:
                observation_id = dict(record.metadata or {}).get("observation_id") or record.evidence_id
                if observation_id not in observation_ids:
                    observation_ids.append(str(observation_id))
            if len(observation_ids) < 2:
                continue
            row = {
                "feature_id": relation.feature_id,
                "disagreement_kind": (
                    "quality_disagreement"
                    if relation.relation_type == "quality_disagreement"
                    else "status_disagreement"
                ),
                "source_observation_ids": observation_ids,
            }
            if row["disagreement_kind"] == "status_disagreement":
                row["observed_statuses"] = [record.status for record in related]
            else:
                row["observed_evaluabilities"] = [
                    record.feature_evaluability for record in related
                ]
            output.append(row)
        return tuple(output)

    @staticmethod
    def _adjudication_targets(state, snapshot, request):
        if not request.model_input.get("feature_disagreements"):
            return {}
        active_ids = set(state.active_evidence_ids)
        snapshot_ids = set(snapshot.evidence_ids)
        output = defaultdict(list)
        requested = set(request.target_features)
        for relation in state.evidence_relations:
            if relation.relation_type not in ("true_contradiction", "quality_disagreement"):
                continue
            if relation.feature_id not in requested:
                continue
            for evidence_id in relation.evidence_ids:
                if evidence_id in active_ids and evidence_id in snapshot_ids:
                    output[relation.feature_id].append(evidence_id)
        return {
            feature_id: tuple(sorted(set(evidence_ids)))
            for feature_id, evidence_ids in output.items()
        }

    @staticmethod
    def _chief_decision_trace(state, plan, decision):
        return {
            "schema_version": "chief_decision_trace_v1",
            "state_id": state.state_id,
            "ledger_snapshot_id": state.ledger_snapshot_id,
            "termination_status": state.termination_state.decision,
            "termination_reason": state.termination_state.reason,
            "diagnostic_scope_status": state.diagnostic_scope_status,
            "final_hypotheses": [item.to_dict() for item in state.hypotheses],
            "belief_update_ids": [item.update_id for item in state.belief_history],
            "reviewer_ledger_record_ids": [
                row.get("reviewer_ledger_record_id")
                for row in state.review_history
                if row.get("reviewer_ledger_record_id")
            ],
            "review_request_ids": [
                row.get("request_id") for row in state.review_history if row.get("request_id")
            ],
            "contradiction_ids": [
                relation.relation_id
                for relation in state.evidence_relations
                if relation.relation_type in ("true_contradiction", "quality_disagreement")
            ],
            "heterogeneity_ids": [
                relation.relation_id
                for relation in state.evidence_relations
                if relation.relation_type == "spatial_heterogeneity"
            ],
            "unresolved_discriminator_ids": [
                item.discriminator_id
                for item in state.discriminators
                if item.status != "resolved"
            ],
            "final_plan_id": plan.plan_id,
            "chief_status": decision.status,
            "chief_final_label": decision.final_label,
            "supporting_evidence_ids": list(decision.supporting_evidence_ids),
            "conflicting_evidence_ids": list(decision.conflicting_evidence_ids),
            "non_clinical": True,
        }

    def _prepare_roi(self, roi, output_path):
        artifact = self.cropper.crop(roi, (output_path / "reviewers" / "crops") if output_path else None)
        if isinstance(artifact, str):
            artifact = CropArtifact(
                image_ref=artifact,
                image_sha256=hashlib.sha256(artifact.encode("utf-8")).hexdigest(),
                pixel_dimensions=roi.pixel_dimensions or (512, 512),
                mpp=roi.mpp,
                hash_scope="reference_only_adapter",
            )
        metadata = dict(roi.metadata)
        metadata["image_hash_scope"] = artifact.hash_scope
        return replace(
            roi,
            image_path=artifact.image_ref,
            image_sha256=artifact.image_sha256,
            pixel_dimensions=artifact.pixel_dimensions,
            mpp=artifact.mpp,
            metadata=metadata,
        )

    def _materialize_roi(self, roi, output_path, cache):
        if roi.roi_id not in cache:
            cache[roi.roi_id] = self._prepare_roi(roi, output_path)
        return cache[roi.roi_id]

    @staticmethod
    def _select_context_views(primary_roi, candidates, profile):
        policies = {policy.get("view_role"): policy for policy in profile.context_policies}
        parent_policy = policies.get("parent_architecture")
        parent_roi_id = primary_roi.metadata.get("parent_roi_id")
        if not parent_policy or not parent_roi_id:
            return tuple()
        parent = AgentFlowOrchestrator._roi_by_id(candidates, parent_roi_id)
        allowed_scales = tuple(
            float(value) for value in parent_policy.get("allowed_magnifications", ())
        )
        if float(parent.scale) not in allowed_scales:
            return tuple()
        return (("parent_architecture", parent),)

    @staticmethod
    def _roi_by_id(candidates, roi_id):
        for candidate in candidates:
            if candidate.roi_id == roi_id:
                return candidate
        raise RuntimeError("Planner selected ROI outside candidate pool: {0}".format(roi_id))

    @staticmethod
    def _write_json(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path, rows):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _append_jsonl(path, row):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
