from pathlib import Path

import numpy as np
import requests
from PIL import Image

from adenoma_agent.cache import JsonCache
from adenoma_agent.schemas import GlobalReviewRecord, NavigationStep, ObservationRecord, ReasoningState
from adenoma_agent.utils import ensure_dir, write_json


class ObserveReasonAgent(object):
    def __init__(self, bundle, cropper_adapter, backend_chain):
        self.bundle = bundle
        self.cropper_adapter = cropper_adapter
        self.backend_chain = backend_chain
        self.description_cache = JsonCache(bundle["runtime"]["cache"]["description_cache_root"])

    def run(self, case_spec, trace_result, navigation_result, case_dir, logger):
        observe_dir = Path(case_dir) / "observe"
        planned_review_steps = [step for step in navigation_result["steps"] if step.metadata.get("action") != "stop"]
        stop_steps = [step for step in navigation_result["steps"] if step.metadata.get("action") == "stop"]
        cluster_lookup = {cluster.cluster_id: cluster for cluster in trace_result["clusters"]}

        records = []
        global_reviews = []
        support_summaries = []
        conflicts = []
        trajectory_steps = []
        crop_manifests = []
        crop_results = []
        pending_steps = list(planned_review_steps)
        dysplasia_added = set()

        step_index = 0
        while step_index < len(pending_steps):
            step = pending_steps[step_index]
            step_index += 1
            bundle_steps = self._build_view_bundle_steps(step)
            export_result = self.cropper_adapter.export_crops(case_spec, bundle_steps, observe_dir / step.step_id)
            crop_manifests.append(export_result["manifest"])
            crop_results.append(export_result["result"])
            crops = self._order_view_bundle_crops(export_result["manifest"].get("crops", []), step)
            if not crops:
                continue
            crop = crops[0]
            cluster = cluster_lookup.get(step.metadata.get("cluster_id"))
            image_stats_bundle = []
            for item in crops:
                stats = self._image_stats(item["image_path"])
                image_stats_bundle.append(
                    {
                        "image_path": item["image_path"],
                        "image_role": item.get("metadata", {}).get("image_role", "detail"),
                        "magnification": float(item.get("m", step.m)),
                        **stats,
                    }
                )
            image_stats = {
                "background_fraction": image_stats_bundle[0]["background_fraction"],
                "tissue_fraction": image_stats_bundle[0]["tissue_fraction"],
                "pale_fraction": image_stats_bundle[0]["pale_fraction"],
            }
            cache_payload = {
                "case_id": case_spec.case_id,
                "agent_version": "dual_branch_v2_cpathagent_multiview",
                "step_id": step.step_id,
                "m": step.m,
                "cluster_id": step.metadata.get("cluster_id"),
                "image_stats": image_stats,
                "image_roles": [item["image_role"] for item in image_stats_bundle],
                "backend_chain": self.bundle["runtime"]["observe"]["backend_chain"],
            }
            cached = self.description_cache.load("observe_step", cache_payload, "record.json")
            if cached is not None and (
                "level_1_findings" not in cached
                or "conventional_hits" not in cached
                or "serrated_dysplasia_hits" not in cached
                or "conventional_dysplasia_hits" not in cached
                or "ssl_hits" not in cached
                or "hp_hits" not in cached
                or "tsa_hits" not in cached
                or "tsa_cytological_atypia_hits" not in cached
                or "inflammatory_hits" not in cached
            ):
                cached = None
            if cached is None:
                backend_response = self.backend_chain.invoke(
                    "observe_step",
                    self.bundle["runtime"]["observe"]["backend_chain"],
                    {
                        "images": [item["image_path"] for item in image_stats_bundle],
                        "prompt": {
                            "question": self.bundle["runtime"]["observe"]["patho_r1_question"],
                            "task": "ssl_others_multi_view_observe_step",
                        },
                        "metadata": {
                            "case_id": case_spec.case_id,
                            "step": step.to_dict(),
                            "cluster": cluster.to_dict() if cluster else {},
                            "image_stats": image_stats,
                            "image_stats_bundle": image_stats_bundle,
                        },
                    },
                )
                step_output = backend_response["output"]
                payload = {
                    "backend": backend_response["backend"],
                    "backend_attempts": backend_response["attempts"],
                    **step_output,
                }
                self.description_cache.save("observe_step", cache_payload, "record.json", payload)
            else:
                payload = cached
            record = ObservationRecord(
                step_id=step.step_id,
                crop_path=crop["image_path"],
                observation=payload["observation"],
                reasoning=payload["reasoning"],
                next_step=payload["next_step"],
                level_1_findings=payload["level_1_findings"],
                level_2_findings=payload["level_2_findings"],
                level_3_findings=payload["level_3_findings"],
                stage_decision=payload["stage_decision"],
                confidence=float(payload["confidence"]),
                metadata={
                    "backend": payload.get("backend"),
                    "background_fraction": image_stats["background_fraction"],
                    "tissue_fraction": image_stats["tissue_fraction"],
                    "pale_fraction": image_stats["pale_fraction"],
                    "cluster_id": step.metadata.get("cluster_id"),
                    "cell_id": self._step_cell_id(step),
                    "cell_priority": step.metadata.get("cell_priority", step.metadata.get("cluster_priority")),
                    "patch_id": step.metadata.get("patch_id"),
                    "cluster_label": step.metadata.get("cluster_label"),
                    "magnification": step.m,
                    "workflow_branch": step.metadata.get("workflow_branch"),
                    "need_to_see": step.need_to_see,
                    "review_goal": step.review_goal,
                    "stage_gate": step.stage_gate,
                    "view_bundle_id": step.step_id,
                    "view_count": int(payload.get("view_count", len(image_stats_bundle))),
                    "image_roles": [item["image_role"] for item in image_stats_bundle],
                    "bundle_image_paths": [item["image_path"] for item in image_stats_bundle],
                    "serrated_hits": payload.get("serrated_hits", {}),
                    "abnormal_crypt_hits": payload.get("abnormal_crypt_hits", {}),
                    "conventional_hits": payload.get("conventional_hits", {}),
                    "serrated_dysplasia_hits": payload.get("serrated_dysplasia_hits", {}),
                    "conventional_dysplasia_hits": payload.get("conventional_dysplasia_hits", {}),
                    "dysplasia_hits": payload.get("dysplasia_hits", {}),
                    "ssl_hits": payload.get("ssl_hits", {}),
                    "hp_hits": payload.get("hp_hits", {}),
                    "tsa_hits": payload.get("tsa_hits", {}),
                    "tsa_cytological_atypia_hits": payload.get("tsa_cytological_atypia_hits", {}),
                    "inflammatory_hits": payload.get("inflammatory_hits", {}),
                    "branch_recovery_hint": payload.get("branch_recovery_hint", "none"),
                    "branch_recovery_reason": payload.get("branch_recovery_reason", ""),
                },
            )
            records.append(record)
            trajectory_steps.append(step)
            support_summaries.append(record.reasoning)
            if image_stats["background_fraction"] > 0.7:
                conflicts.append("Background-heavy crop at {0}".format(step.step_id))
            logger.log(
                state="OBSERVE",
                agent="ObserveReasonAgent",
                input_ref=crop["image_path"],
                output_ref=record.step_id,
                payload=record.to_dict(),
            )
            if (
                step.review_goal == "serrated_overview_assessment"
                and record.stage_decision == "serrated_overview_not_supported_or_indeterminate"
                and ("serrated_recovery", step.metadata.get("cluster_id"), self._step_cell_id(step)) not in dysplasia_added
            ):
                self._suppress_pending_same_cell(
                    pending_steps,
                    step_index,
                    step,
                    {"ssl_assessment", "hp_assessment", "tsa_assessment"},
                )
                hint = str(record.metadata.get("branch_recovery_hint") or "none").strip()
                target = "normal_overview" if hint == "normal" else "conventional_overview"
                pending_steps.insert(
                    step_index,
                    self._build_recovery_step(
                        step,
                        cluster,
                        len(trajectory_steps) + len(stop_steps),
                        target=target,
                        recovery_hint=hint,
                        recovery_reason=record.metadata.get("branch_recovery_reason", ""),
                    ),
                )
                dysplasia_added.add(("serrated_recovery", step.metadata.get("cluster_id"), self._step_cell_id(step)))
            if (
                step.review_goal == "conventional_overview_assessment"
                and step.metadata.get("recovery_source") == "serrated_overview_negative"
                and record.stage_decision == "supports_conventional_overview"
                and ("recovery_conventional_architecture", step.metadata.get("cluster_id"), self._step_cell_id(step)) not in dysplasia_added
            ):
                insert_at = step_index
                for target in ("conventional_architecture", "reactive_regenerative"):
                    pending_steps.insert(
                        insert_at,
                        self._build_recovery_step(
                            step,
                            cluster,
                            len(trajectory_steps) + len(stop_steps) + (insert_at - step_index),
                            target=target,
                            recovery_hint="conventional",
                            recovery_reason=record.metadata.get("branch_recovery_reason", ""),
                        ),
                    )
                    insert_at += 1
                dysplasia_added.add(("recovery_conventional_architecture", step.metadata.get("cluster_id"), self._step_cell_id(step)))
            if (
                step.review_goal == "conventional_overview_assessment"
                and step.metadata.get("recovery_source") == "serrated_overview_negative"
                and record.stage_decision == "conventional_overview_not_supported_or_indeterminate"
                and ("recovery_normal_overview", step.metadata.get("cluster_id"), self._step_cell_id(step)) not in dysplasia_added
            ):
                pending_steps.insert(
                    step_index,
                    self._build_recovery_step(
                        step,
                        cluster,
                        len(trajectory_steps) + len(stop_steps),
                        target="normal_overview",
                        recovery_hint="normal",
                        recovery_reason="Conventional recovery overview was not supported after serrated overview was negative.",
                    ),
                )
                dysplasia_added.add(("recovery_normal_overview", step.metadata.get("cluster_id"), self._step_cell_id(step)))
            if (
                step.review_goal in {"abnormal_crypt_assessment", "ssl_assessment"}
                and record.stage_decision in {"supports_abnormal_crypt", "ssl_architecture_supported"}
                and ("serrated", step.metadata.get("cluster_id")) not in dysplasia_added
            ):
                pending_steps.insert(
                    step_index,
                    self._build_dysplasia_step(
                        step,
                        cluster,
                        len(trajectory_steps) + len(stop_steps),
                        branch="ssl",
                    ),
                )
                dysplasia_added.add(("serrated", step.metadata.get("cluster_id")))
            if (
                step.review_goal == "tsa_assessment"
                and record.stage_decision == "tsa_architecture_supported"
                and ("tsa", step.metadata.get("cluster_id")) not in dysplasia_added
            ):
                insert_at = step_index
                for target_branch in ("tsa_cytology", "tsa"):
                    pending_steps.insert(
                        insert_at,
                        self._build_dysplasia_step(
                            step,
                            cluster,
                            len(trajectory_steps) + len(stop_steps) + (insert_at - step_index),
                            branch=target_branch,
                        ),
                    )
                    insert_at += 1
                dysplasia_added.add(("tsa", step.metadata.get("cluster_id")))
            if (
                step.review_goal in {"conventional_adenoma_assessment", "conventional_architecture_assessment"}
                and record.stage_decision in {"supports_conventional_adenoma", "supports_conventional_architecture"}
                and ("conventional", step.metadata.get("cluster_id")) not in dysplasia_added
            ):
                pending_steps.insert(
                    step_index,
                    self._build_dysplasia_step(
                        step,
                        cluster,
                        len(trajectory_steps) + len(stop_steps),
                        branch="conventional",
                    ),
                )
                dysplasia_added.add(("conventional", step.metadata.get("cluster_id")))
            if not self._should_run_chief_review(step, pending_steps[step_index:]):
                continue
            global_review = self._call_chief_global_review(
                case_spec=case_spec,
                step=step,
                record=record,
                records=records,
                global_reviews=global_reviews,
                trace_result=trace_result,
                pending_steps=pending_steps[step_index:],
                observe_dir=observe_dir,
            )
            if global_review.decision == "continue" and not pending_steps[step_index:]:
                global_review.decision = "early_stop"
                global_review.continue_reason = ""
                global_review.next_visual_target = None
                if not global_review.sufficient_evidence:
                    global_review.sufficient_evidence = [
                        "No additional pending observation targets remain after step {0}.".format(record.step_id),
                        "Chief finalization was forced because the observation queue is exhausted.",
                    ]
            global_reviews.append(global_review)
            logger.log(
                state="OBSERVE_GLOBAL_REVIEW",
                agent="ChiefPathologist",
                input_ref=record.step_id,
                output_ref=global_review.review_id,
                payload=global_review.to_dict(),
            )
            if global_review.decision == "early_stop":
                pending_steps = []
                stop_steps = stop_steps[:1]
                break
            if global_review.decision == "continue":
                pending_steps = pending_steps[:step_index] + self._rerank_pending_steps(
                    pending_steps[step_index:],
                    global_review,
                )

        trajectory_steps.extend(stop_steps)
        combined_manifest = {"crops": []}
        for manifest in crop_manifests:
            combined_manifest["crops"].extend(manifest.get("crops", []))
        combined_manifest_json = write_json(observe_dir / "crop_manifest.json", combined_manifest)
        combined_result = {
            "returncode": 0 if all(result.get("returncode", 1) == 0 for result in crop_results) else 1,
            "stdout": "\n".join(result.get("stdout", "") for result in crop_results if result.get("stdout")),
            "stderr": "\n".join(result.get("stderr", "") for result in crop_results if result.get("stderr")),
            "latency_ms": sum(int(result.get("latency_ms", 0)) for result in crop_results),
        }

        if not global_reviews or global_reviews[-1].decision != "early_stop":
            raise RuntimeError("Observation cannot generate observe_report before a final real Chief early_stop review.")

        report_response = self.backend_chain.invoke(
            "observe_report",
            self.bundle["runtime"]["observe"]["backend_chain"],
            {
                "images": [],
                "prompt": {
                    "question": "Synthesize an 11-class hierarchical colorectal polyp report with serrated subtype, conventional adenoma subtype, inflammatory classification, and branch-specific dysplasia status.",
                    "task": "mucosa_11_class_hierarchical_polyp_report",
                },
                "metadata": {
                    "case_id": case_spec.case_id,
                    "records": [record.to_dict() for record in records],
                    "global_reviews": [item.to_dict() for item in global_reviews],
                    "trace_clusters": [cluster.to_dict() for cluster in trace_result["clusters"]],
                },
            },
        )
        report_output = dict(report_response["output"])
        hierarchy = report_output.get("hierarchical_prediction", {})
        if not isinstance(hierarchy, dict):
            hierarchy = {}
        report_output["hierarchical_prediction"] = hierarchy

        for key in (
            "serrated_checklist",
            "abnormal_crypt_checklist",
            "conventional_adenoma_checklist",
            "serrated_dysplasia_checklist",
            "conventional_dysplasia_checklist",
            "dysplasia_checklist",
            "ssl_checklist",
            "hp_checklist",
            "tsa_checklist",
            "tsa_cytological_atypia_checklist",
            "conventional_architecture_checklist",
            "inflammatory_checklist",
        ):
            value = report_output.get(key, [])
            if not isinstance(value, (dict, list)):
                report_output[key] = []

        if not isinstance(report_output.get("integrated_report", ""), (dict, str)):
            report_output["integrated_report"] = str(report_output.get("integrated_report", ""))
        reasoning_state = ReasoningState(
            hypotheses=["mucosa_serrated_abnormal_crypt_dysplasia_report_ready"],
            supporting_evidence=support_summaries[:8],
            conflicts=conflicts,
            stop_reason="trajectory_completed",
            metadata={"observation_count": len(records), "trajectory_length": len(trajectory_steps)},
        )
        observation_json = write_json(
            observe_dir / "observation_records.json",
            {
                "observations": [record.to_dict() for record in records],
                "global_reviews": [item.to_dict() for item in global_reviews],
            },
        )
        report_json = write_json(
            observe_dir / "pathological_report.json",
            {
                "hierarchical_prediction": report_output["hierarchical_prediction"],
                "serrated_checklist": report_output["serrated_checklist"],
                "abnormal_crypt_checklist": report_output["abnormal_crypt_checklist"],
                "conventional_adenoma_checklist": report_output.get("conventional_adenoma_checklist", {}),
                "serrated_dysplasia_checklist": report_output.get("serrated_dysplasia_checklist", {}),
                "conventional_dysplasia_checklist": report_output.get("conventional_dysplasia_checklist", {}),
                "dysplasia_checklist": report_output["dysplasia_checklist"],
                "ssl_checklist": report_output.get("ssl_checklist", {}),
                "hp_checklist": report_output.get("hp_checklist", {}),
                "tsa_checklist": report_output.get("tsa_checklist", {}),
                "tsa_cytological_atypia_checklist": report_output.get("tsa_cytological_atypia_checklist", {}),
                "conventional_architecture_checklist": report_output.get("conventional_architecture_checklist", {}),
                "inflammatory_checklist": report_output.get("inflammatory_checklist", {}),
                "final_case_assessment": report_output["hierarchical_prediction"].get("final_case_assessment", {}),
                "integrated_report": report_output["integrated_report"],
                "backend_attempts": report_response["attempts"],
            },
        )
        reasoning_json = write_json(observe_dir / "reasoning_state.json", reasoning_state.to_dict())
        return {
            "records": records,
            "global_reviews": global_reviews,
            "reasoning_state": reasoning_state,
            "observation_json": observation_json,
            "report_json": report_json,
            "reasoning_json": reasoning_json,
            "crop_manifest_json": combined_manifest_json,
            "crop_result": combined_result,
            "trajectory_steps": trajectory_steps,
            "hierarchical_prediction": report_output["hierarchical_prediction"],
            "serrated_checklist": report_output["serrated_checklist"],
            "abnormal_crypt_checklist": report_output["abnormal_crypt_checklist"],
            "conventional_adenoma_checklist": report_output.get("conventional_adenoma_checklist", {}),
            "serrated_dysplasia_checklist": report_output.get("serrated_dysplasia_checklist", {}),
            "conventional_dysplasia_checklist": report_output.get("conventional_dysplasia_checklist", {}),
            "dysplasia_checklist": report_output["dysplasia_checklist"],
            "ssl_checklist": report_output.get("ssl_checklist", {}),
            "hp_checklist": report_output.get("hp_checklist", {}),
            "tsa_checklist": report_output.get("tsa_checklist", {}),
            "tsa_cytological_atypia_checklist": report_output.get("tsa_cytological_atypia_checklist", {}),
            "conventional_architecture_checklist": report_output.get("conventional_architecture_checklist", {}),
            "inflammatory_checklist": report_output.get("inflammatory_checklist", {}),
            "final_case_assessment": report_output["hierarchical_prediction"].get("final_case_assessment", {}),
            "integrated_report": report_output["integrated_report"],
            "report_backend_attempts": report_response["attempts"],
        }

    def _call_chief_global_review(self, case_spec, step, record, records, global_reviews, trace_result, pending_steps, observe_dir):
        chief_cfg = self.bundle["runtime"].get("chief_llm", {})
        server_url = str(chief_cfg.get("server_url", "")).strip()
        timeout_seconds = int(chief_cfg.get("timeout_seconds", 180))
        require_real_chief = bool(chief_cfg.get("require_real_chief", False))
        if not server_url:
            raise RuntimeError("chief_llm.server_url is not configured")
        current_cell_id = self._step_cell_id(step)
        current_cell_records = [
            item
            for item in records
            if item.metadata.get("cell_id") == current_cell_id
            or (
                not item.metadata.get("cell_id")
                and item.metadata.get("patch_id") == step.metadata.get("patch_id")
            )
        ]
        payload = {
            "case_id": case_spec.case_id,
            "current_cell_id": current_cell_id,
            "step": step.to_dict(),
            "record": record.to_dict(),
            "current_cell_observations": [item.to_dict() for item in current_cell_records],
            "observations": [item.to_dict() for item in records],
            "global_reviews": [item.to_dict() for item in global_reviews],
            "trace_clusters": [cluster.to_dict() for cluster in trace_result["clusters"]],
            "pending_steps": [item.to_dict() for item in pending_steps],
        }
        try:
            response = requests.post(server_url, json=payload, timeout=timeout_seconds)
        except Exception as exc:
            raise RuntimeError("Chief HTTP service request failed: {0}".format(str(exc)))
        if response.status_code != 200:
            raise RuntimeError("Chief HTTP service returned HTTP {0}: {1}".format(response.status_code, response.text))
        data = self._normalize_chief_review_payload(response.json(), step, record, trace_result, pending_steps)
        debug_paths = self._write_chief_debug_artifacts(observe_dir, step.step_id, data)
        self._validate_chief_review_payload(data, step, record, trace_result, pending_steps, require_real_chief)
        review_source = str(data.get("review_source", "chief_model")).strip() or "chief_model"
        return GlobalReviewRecord(
            review_id=str(data.get("review_id", "global_review_{0:04d}".format(len(global_reviews)))),
            source_step_id=str(data.get("source_step_id", record.step_id)),
            decision=str(data.get("decision", "continue")),
            continue_reason=str(data.get("continue_reason", "")),
            chief_confidence=float(data.get("chief_confidence", 0.0)),
            resolved_branch_state=dict(data.get("resolved_branch_state", {})),
            sufficient_evidence=list(data.get("sufficient_evidence", [])),
            unresolved_questions=list(data.get("unresolved_questions", [])),
            next_visual_target=data.get("next_visual_target"),
            branch_correction_reason=str(data.get("branch_correction_reason", "")),
            metadata={
                "case_id": case_spec.case_id,
                "cluster_id": step.metadata.get("cluster_id"),
                "review_goal": record.metadata.get("review_goal"),
                "chief_model_name": chief_cfg.get("model_name"),
                "chief_request_id": data.get("request_id"),
                "chief_gpu_device_id": data.get("gpu_device_id"),
                "chief_cuda_visible_devices": data.get("cuda_visible_devices"),
                "chief_round_trip_ms": data.get("round_trip_ms"),
                "normalization_actions": data.get("normalization_actions", []),
                "start_mode": "warm_start",
                "review_source": review_source,
                "chief_debug_response_json": str(debug_paths["response_json"]),
                "chief_debug_raw_text": str(debug_paths["raw_text"]),
            },
        )

    def _normalize_chief_review_payload(self, data, step, record, trace_result, pending_steps):
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        actions = list(normalized.get("normalization_actions", [])) if isinstance(normalized.get("normalization_actions"), list) else []

        def add_action(action):
            if action not in actions:
                actions.append(action)

        if str(normalized.get("source_step_id", "")).strip() != str(record.step_id):
            normalized["source_step_id"] = str(record.step_id)
            add_action("chief_source_step_id_defaulted")
        if not str(normalized.get("review_id", "")).strip():
            normalized["review_id"] = "global_review_{0}".format(record.step_id)
            add_action("chief_review_id_defaulted")
        decision = str(normalized.get("decision", "")).strip()
        if decision not in {"continue", "early_stop"}:
            decision = "continue" if pending_steps else "early_stop"
            normalized["decision"] = decision
            add_action("chief_decision_defaulted")

        branch_state = normalized.get("resolved_branch_state", {})
        if not isinstance(branch_state, dict):
            branch_state = {}
            add_action("chief_resolved_branch_state_defaulted")
        fixed_branch_state = {}
        for key in ("serrated", "abnormal_crypt", "conventional", "dysplasia"):
            value = str(branch_state.get(key, "")).strip()
            if value not in {"supported", "opposed", "unresolved"}:
                value = "unresolved"
                add_action("chief_resolved_branch_state_normalized")
            fixed_branch_state[key] = value
        normalized["resolved_branch_state"] = fixed_branch_state

        if decision == "continue":
            if not str(normalized.get("continue_reason", "")).strip():
                normalized["continue_reason"] = "Chief requested additional visual evidence after adapter normalization."
                add_action("chief_continue_reason_defaulted")
            target = normalized.get("next_visual_target")
            cluster_ids = {cluster.cluster_id for cluster in trace_result.get("clusters", [])}
            pending_cluster_ids = {item.metadata.get("cluster_id") for item in pending_steps}
            valid_target = isinstance(target, dict) and target.get("target_cluster_id") in cluster_ids.union(pending_cluster_ids)
            if not valid_target:
                if pending_steps:
                    target = self._target_from_pending_step(pending_steps[0], target)
                    add_action("chief_invalid_target_cluster_retargeted")
                else:
                    normalized["decision"] = "early_stop"
                    normalized["next_visual_target"] = None
                    normalized["sufficient_evidence"] = list(normalized.get("sufficient_evidence") or [record.reasoning])
                    normalized["continue_reason"] = ""
                    add_action("chief_continue_without_valid_target_forced_early_stop")
                    normalized["normalization_actions"] = actions
                    return normalized
            else:
                target = dict(target)

            if target.get("target_branch") not in {"serrated", "conventional", "non_serrated"}:
                target["target_branch"] = step.metadata.get("workflow_branch") or "non_serrated"
                add_action("chief_target_branch_normalized")
            if target.get("target_region_semantic") not in {
                "serrated",
                "conventional",
                "normal",
                "background",
            }:
                target["target_region_semantic"] = step.metadata.get("cluster_label") or "normal"
                add_action("chief_target_region_semantic_normalized")
            try:
                preferred = float(target.get("preferred_magnification"))
            except Exception:
                preferred = 10.0
                add_action("chief_preferred_magnification_normalized")
            if preferred not in {2.5, 5.0, 10.0}:
                preferred = 10.0 if preferred > 5.0 else (2.5 if preferred < 5.0 else 5.0)
                add_action("chief_preferred_magnification_normalized")
            target["preferred_magnification"] = preferred
            prompts = target.get("target_morphology_prompt")
            if isinstance(prompts, str):
                target["target_morphology_prompt"] = [prompts]
                add_action("chief_target_prompt_normalized")
            elif not isinstance(prompts, list):
                target["target_morphology_prompt"] = ["review the next planned cell-level morphology target"]
                add_action("chief_target_prompt_defaulted")
            if not str(target.get("priority_reason", "")).strip():
                target["priority_reason"] = "Adapter selected the next valid pending navigation target."
                add_action("chief_priority_reason_defaulted")
            normalized["next_visual_target"] = target
        else:
            if normalized.get("next_visual_target") is not None:
                normalized["next_visual_target"] = None
                add_action("chief_early_stop_target_cleared")
            if not isinstance(normalized.get("sufficient_evidence"), list) or not normalized.get("sufficient_evidence"):
                normalized["sufficient_evidence"] = [record.reasoning]
                add_action("chief_sufficient_evidence_defaulted")

        target = normalized.get("next_visual_target")
        current_branch = str(step.metadata.get("workflow_branch", "")).strip()
        if isinstance(target, dict) and str(target.get("target_branch", "")).strip() != current_branch:
            if not str(normalized.get("branch_correction_reason", "")).strip():
                normalized["branch_correction_reason"] = "Chief target branch differs from current step branch after adapter normalization."
                add_action("chief_branch_correction_reason_defaulted")
        normalized["normalization_actions"] = actions
        return normalized

    def _target_from_pending_step(self, pending_step, original_target=None):
        original_target = original_target if isinstance(original_target, dict) else {}
        branch = pending_step.metadata.get("workflow_branch") or original_target.get("target_branch") or "non_serrated"
        if branch not in {"serrated", "conventional", "non_serrated"}:
            branch = "non_serrated"
        semantic = pending_step.metadata.get("cluster_label") or original_target.get("target_region_semantic") or "normal"
        return {
            "target_cluster_id": pending_step.metadata.get("cluster_id"),
            "target_branch": branch,
            "target_region_semantic": semantic,
            "target_morphology_prompt": original_target.get("target_morphology_prompt") or [pending_step.review_goal],
            "preferred_magnification": float(pending_step.m),
            "priority_reason": original_target.get("priority_reason") or "Retargeted to the next valid pending navigation step.",
        }

    def _write_chief_debug_artifacts(self, observe_dir, step_id, payload):
        chief_debug_dir = ensure_dir(Path(observe_dir) / "chief_reviews")
        response_json = write_json(chief_debug_dir / "{0}_chief_response.json".format(step_id), payload)
        raw_text_path = chief_debug_dir / "{0}_chief_raw_text.txt".format(step_id)
        raw_text_path.write_text(str((payload or {}).get("raw_generated_text", "") or ""), encoding="utf-8")
        thought_text_path = chief_debug_dir / "{0}_chief_thought_text.txt".format(step_id)
        thought_text_path.write_text(str((payload or {}).get("thought_text", "") or ""), encoding="utf-8")
        answer_text_path = chief_debug_dir / "{0}_chief_answer_candidate.txt".format(step_id)
        answer_text_path.write_text(str((payload or {}).get("answer_candidate_text", "") or ""), encoding="utf-8")
        parse_error_path = chief_debug_dir / "{0}_chief_parse_error.txt".format(step_id)
        parse_error_path.write_text(str((payload or {}).get("parse_error", "") or ""), encoding="utf-8")
        return {
            "response_json": response_json,
            "raw_text": raw_text_path,
            "thought_text": thought_text_path,
            "answer_text": answer_text_path,
            "parse_error": parse_error_path,
        }

    def _validate_chief_review_payload(self, data, step, record, trace_result, pending_steps, require_real_chief):
        if not isinstance(data, dict):
            raise RuntimeError("Chief HTTP service returned a non-object JSON payload.")
        if require_real_chief and str(data.get("review_source", "chief_model")).strip() != "chief_model":
            raise RuntimeError("Chief review must come from the real Chief model, but got review_source={0}".format(data.get("review_source")))
        if str(data.get("review_id", "")).strip() == "global_review_fallback":
            raise RuntimeError("Chief fallback reviews are not allowed in the main experiment pipeline.")
        if str(data.get("source_step_id", "")).strip() != str(record.step_id):
            raise RuntimeError("Chief review source_step_id must match the observation step_id.")
        decision = str(data.get("decision", "")).strip()
        if decision not in {"continue", "early_stop"}:
            raise RuntimeError("Chief review decision must be continue or early_stop.")
        branch_state = data.get("resolved_branch_state", {})
        if not isinstance(branch_state, dict):
            raise RuntimeError("Chief resolved_branch_state must be an object.")
        for key in ("serrated", "abnormal_crypt", "conventional", "dysplasia"):
            value = str(branch_state.get(key, "")).strip()
            if value not in {"supported", "opposed", "unresolved"}:
                raise RuntimeError("Chief resolved_branch_state.{0} is invalid.".format(key))
        if decision == "continue":
            if not str(data.get("continue_reason", "")).strip():
                raise RuntimeError("Chief continue review must include continue_reason.")
            target = data.get("next_visual_target")
            if not isinstance(target, dict):
                raise RuntimeError("Chief continue review must include next_visual_target.")
            cluster_ids = {cluster.cluster_id for cluster in trace_result.get("clusters", [])}
            pending_cluster_ids = {item.metadata.get("cluster_id") for item in pending_steps}
            target_cluster_id = target.get("target_cluster_id")
            if target_cluster_id not in cluster_ids and target_cluster_id not in pending_cluster_ids:
                raise RuntimeError("Chief next_visual_target.target_cluster_id must refer to an existing trace/pending cluster.")
            if target.get("target_branch") not in {"serrated", "conventional", "non_serrated"}:
                raise RuntimeError("Chief next_visual_target.target_branch is invalid.")
            if target.get("target_region_semantic") not in {
                "serrated",
                "conventional",
                "normal",
                "background",
            }:
                raise RuntimeError("Chief next_visual_target.target_region_semantic is invalid.")
            if float(target.get("preferred_magnification")) not in {2.5, 5.0, 10.0}:
                raise RuntimeError("Chief next_visual_target.preferred_magnification must be 2.5, 5.0, or 10.0.")
        else:
            if data.get("next_visual_target") is not None:
                raise RuntimeError("Chief early_stop review must set next_visual_target to null.")
            if not isinstance(data.get("sufficient_evidence"), list) or not data.get("sufficient_evidence"):
                raise RuntimeError("Chief early_stop review must include non-empty sufficient_evidence.")
        target = data.get("next_visual_target")
        current_branch = str(step.metadata.get("workflow_branch", "")).strip()
        if isinstance(target, dict) and str(target.get("target_branch", "")).strip() != current_branch:
            if not str(data.get("branch_correction_reason", "")).strip():
                raise RuntimeError("Chief branch correction requires branch_correction_reason.")

    def _build_next_visual_target(self, cluster, record, pending_steps):
        target_cluster_id = record.metadata.get("cluster_id")
        target_branch = "serrated"
        target_region_semantic = "serrated"
        target_prompt = ["look for additional corroborating morphology"]
        preferred_magnification = 10.0
        priority_reason = "Need higher-value morphology to resolve the remaining branch uncertainty."

        if record.stage_decision in {"supports_serrated_overview", "supports_serrated_lesion"}:
            target_branch = "serrated"
            target_region_semantic = "serrated"
            target_prompt = [
                "perform 5x SSL architectural distortion assessment",
                "perform 5x TSA architecture and low-power cytology assessment",
            ]
            preferred_magnification = 5.0
            priority_reason = "Current uncertainty is concentrated in SSL/TSA subtype assessment after serrated overview."
        elif record.stage_decision == "ssl_architecture_supported":
            target_branch = "serrated"
            target_region_semantic = "serrated"
            target_prompt = [
                "check high-grade or definite dysplasia at the SSL crypt base",
            ]
            preferred_magnification = 10.0
            priority_reason = "SSL architecture is supported; targeted dysplasia review is next."
        elif record.stage_decision == "tsa_architecture_supported":
            target_branch = "serrated"
            target_region_semantic = "serrated"
            target_prompt = [
                "confirm TSA cytological atypia at 10x",
                "check high-grade or definite dysplasia in the TSA-suspicious region",
            ]
            preferred_magnification = 10.0
            priority_reason = "TSA architecture is supported; targeted cytology and dysplasia review are next."
        elif record.stage_decision in {"supports_conventional_overview", "supports_conventional_architecture", "supports_conventional_adenoma"}:
            target_branch = "conventional"
            target_region_semantic = "conventional"
            target_prompt = [
                "look for gland crowding",
                "check tubular or tubulovillous architecture",
            ]
            preferred_magnification = 5.0 if record.stage_decision == "supports_conventional_overview" else 10.0
            priority_reason = "Current uncertainty is concentrated in conventional architecture or high-grade dysplasia confirmation."
        elif record.stage_decision in {"supports_normal_overview", "supports_non_serrated_overview", "background_or_low_value"}:
            target_branch = "non_serrated"
            target_region_semantic = "normal"
            target_prompt = [
                "confirm benign architecture",
                "exclude hidden higher-priority lesion cues",
            ]
            preferred_magnification = 5.0
            priority_reason = "Remaining scan value is low-power exclusion rather than crypt-level confirmation."

        if not pending_steps:
            return None
        for step in pending_steps:
            if step.metadata.get("cluster_id") == target_cluster_id:
                return {
                    "target_cluster_id": target_cluster_id,
                    "target_branch": target_branch,
                    "target_region_semantic": target_region_semantic,
                    "target_morphology_prompt": target_prompt,
                    "preferred_magnification": preferred_magnification,
                    "priority_reason": priority_reason,
                }
        next_step = pending_steps[0]
        return {
            "target_cluster_id": next_step.metadata.get("cluster_id"),
            "target_branch": next_step.metadata.get("workflow_branch"),
            "target_region_semantic": next_step.metadata.get("cluster_label"),
            "target_morphology_prompt": target_prompt,
            "preferred_magnification": preferred_magnification,
            "priority_reason": priority_reason,
        }

    def _rerank_pending_steps(self, pending_steps, global_review):
        target = global_review.next_visual_target or {}
        target_cluster_id = target.get("target_cluster_id")
        target_branch = target.get("target_branch")
        if not pending_steps:
            return pending_steps

        def key(step):
            cluster_match = 0 if step.metadata.get("cluster_id") == target_cluster_id else 1
            branch_match = 0 if step.metadata.get("workflow_branch") == target_branch else 1
            return (cluster_match, branch_match, step.step_id)

        return sorted(pending_steps, key=key)

    def _suppress_pending_same_cell(self, pending_steps, start_index, parent_step, review_goals):
        parent_cell_id = self._step_cell_id(parent_step)
        kept = pending_steps[:start_index]
        for candidate in pending_steps[start_index:]:
            if self._step_cell_id(candidate) == parent_cell_id and candidate.review_goal in review_goals:
                continue
            kept.append(candidate)
        pending_steps[:] = kept

    def _should_run_chief_review(self, step, pending_steps):
        current_cell_id = self._step_cell_id(step)
        if not current_cell_id:
            return True
        for candidate in pending_steps:
            if candidate.metadata.get("action") == "stop":
                continue
            return self._step_cell_id(candidate) != current_cell_id
        return True

    def _step_cell_id(self, step):
        cell_id = step.metadata.get("cell_id")
        if cell_id:
            return str(cell_id)
        patch_id = step.metadata.get("patch_id")
        if isinstance(patch_id, list) and len(patch_id) == 2:
            return "cell_{0}_{1}".format(patch_id[0], patch_id[1])
        return str(step.metadata.get("cluster_id", ""))

    def _build_view_bundle_steps(self, step):
        mag_to_region = self.bundle["budget"].get("magnification_to_region_size", {})
        role = "overview" if float(step.m) <= 5.0 else "detail"
        bundle_id = step.step_id
        magnification = float(step.m)
        region_size = int(mag_to_region.get(str(magnification), step.region_size_level0))
        return [
            NavigationStep(
                step_id="{0}__{1}".format(bundle_id, role),
                x=step.x,
                y=step.y,
                m=magnification,
                region_size_level0=region_size,
                need_to_see=step.need_to_see,
                review_goal=step.review_goal,
                stage_gate=step.stage_gate,
                metadata={
                    **step.metadata,
                    "region_size_level0": region_size,
                    "view_bundle_id": bundle_id,
                    "image_role": role,
                    "source_step_id": step.step_id,
                },
            )
        ]

    def _order_view_bundle_crops(self, crops, step):
        role_order = {
            "serrated_lesion_assessment": ["overview", "local", "detail"],
            "non_serrated_overview_assessment": ["overview", "local", "detail"],
            "abnormal_crypt_assessment": ["local", "detail", "overview"],
            "conventional_adenoma_assessment": ["local", "detail", "overview"],
            "serrated_dysplasia_assessment": ["detail", "local", "overview"],
            "conventional_dysplasia_assessment": ["detail", "local", "overview"],
        }.get(step.review_goal, ["local", "detail", "overview"])
        rank = {role: index for index, role in enumerate(role_order)}
        return sorted(crops, key=lambda item: rank.get(item.get("metadata", {}).get("image_role", "detail"), 99))

    def _build_dysplasia_step(self, parent_step, cluster, step_index, branch):
        cluster_id = parent_step.metadata.get("cluster_id")
        region_size = int(self.bundle["budget"].get("magnification_to_region_size", {}).get("10.0", 1024))
        if branch == "conventional":
            review_goal = "conventional_dysplasia_assessment"
            need_to_see = "Perform conventional adenoma-branch dysplasia review; keep this evidence separate from SSL dysplasia."
            gate_source = "conventional_adenoma"
            workflow_branch = "conventional"
        elif branch == "ssl":
            review_goal = "ssl_dysplasia_assessment"
            need_to_see = "Re-cut a targeted 10x SSL crypt-base view to check high-grade or definite dysplasia after SSL architecture support."
            gate_source = "ssl_architecture"
            workflow_branch = "serrated"
        elif branch == "tsa_cytology":
            review_goal = "tsa_cytological_atypia_assessment"
            need_to_see = "Re-cut a targeted 10x TSA cytology view to confirm cytoplasmic eosinophilia and pencillate nuclei."
            gate_source = "tsa_architecture"
            workflow_branch = "serrated"
        elif branch == "tsa":
            review_goal = "tsa_dysplasia_assessment"
            need_to_see = "Re-cut a targeted 10x TSA view to check high-grade or definite dysplasia after TSA support."
            gate_source = "tsa_architecture"
            workflow_branch = "serrated"
        else:
            review_goal = "serrated_dysplasia_assessment"
            need_to_see = "Perform serrated-branch dysplasia review only after abnormal crypt support has been established."
            gate_source = "abnormal_crypt"
            workflow_branch = "serrated"
        return NavigationStep(
            step_id="step_dyn_{0:02d}_{1}_{2}".format(int(step_index), branch, cluster_id or "cluster"),
            x=parent_step.x,
            y=parent_step.y,
            m=10.0,
            region_size_level0=region_size,
            need_to_see=need_to_see,
            review_goal=review_goal,
            stage_gate="dysplasia",
            metadata={
                **parent_step.metadata,
                "cluster_id": cluster_id,
                "cluster_label": cluster.l if cluster else parent_step.metadata.get("cluster_label"),
                "cluster_priority": cluster.s if cluster else parent_step.metadata.get("cluster_priority"),
                "region_size_level0": region_size,
                "action": "inspect",
                "generated_by": "ObserveReasonAgent",
                "gate_source_step": parent_step.step_id,
                "gate_source": gate_source,
                "workflow_branch": workflow_branch,
            },
        )

    def _build_recovery_step(self, parent_step, cluster, step_index, target, recovery_hint, recovery_reason):
        cluster_id = parent_step.metadata.get("cluster_id")
        mag_to_region = self.bundle["budget"].get("magnification_to_region_size", {})
        if target == "conventional_architecture":
            magnification = 5.0
            region_size = int(mag_to_region.get("5.0", 2048))
            review_goal = "conventional_architecture_assessment"
            stage_gate = "conventional_architecture"
            workflow_branch = "conventional"
            need_to_see = "Recovery review: assess conventional adenoma architecture after conventional overview support."
        elif target == "reactive_regenerative":
            magnification = 5.0
            region_size = int(mag_to_region.get("5.0", 2048))
            review_goal = "reactive_regenerative_assessment"
            stage_gate = "reactive_regenerative"
            workflow_branch = "conventional"
            need_to_see = "Recovery review: assess reactive/regenerative mimic features after conventional overview support."
        elif target == "normal_overview":
            magnification = 2.5
            region_size = int(mag_to_region.get("2.5", 4096))
            review_goal = "normal_overview_assessment"
            stage_gate = "normal_overview"
            workflow_branch = "normal"
            need_to_see = "Recovery review: assess normal or low-priority mucosa after serrated overview was not supported."
        else:
            magnification = 2.5
            region_size = int(mag_to_region.get("2.5", 4096))
            review_goal = "conventional_overview_assessment"
            stage_gate = "conventional_overview"
            workflow_branch = "conventional"
            need_to_see = "Recovery review: assess conventional adenoma candidate after serrated overview was not supported."
        return NavigationStep(
            step_id="step_dyn_{0:02d}_{1}_{2}".format(int(step_index), target, cluster_id or "cluster"),
            x=parent_step.x,
            y=parent_step.y,
            m=magnification,
            region_size_level0=region_size,
            need_to_see=need_to_see,
            review_goal=review_goal,
            stage_gate=stage_gate,
            metadata={
                **parent_step.metadata,
                "cluster_id": cluster_id,
                "cluster_label": cluster.l if cluster else parent_step.metadata.get("cluster_label"),
                "cluster_priority": cluster.s if cluster else parent_step.metadata.get("cluster_priority"),
                "region_size_level0": region_size,
                "action": "inspect",
                "generated_by": "ObserveReasonAgent",
                "recovery_source": "serrated_overview_negative",
                "branch_recovery_hint": recovery_hint or "none",
                "branch_recovery_reason": recovery_reason or "Serrated overview was not supported; alternate-route visual evidence is required before Chief branch correction.",
                "branch_correction_reason": recovery_reason or "Serrated overview was not supported; alternate-route visual evidence is required before Chief branch correction.",
                "gate_source_step": parent_step.step_id,
                "gate_source": "serrated_overview",
                "workflow_branch": workflow_branch,
            },
        )

    def _image_stats(self, image_path):
        image = Image.open(image_path).convert("RGB")
        arr = np.array(image, dtype=np.float32)
        mean_rgb = arr.mean(axis=2)
        sat = arr.max(axis=2) - arr.min(axis=2)
        tissue_mask = (mean_rgb < 235.0) & (sat > 8.0)
        pale_mask = (mean_rgb > 165.0) & (mean_rgb < 235.0) & (sat < 28.0) & tissue_mask
        background_fraction = float((mean_rgb > 235.0).mean())
        tissue_fraction = float(tissue_mask.mean())
        pale_fraction = float(pale_mask.mean())
        return {
            "background_fraction": round(background_fraction, 4),
            "tissue_fraction": round(tissue_fraction, 4),
            "pale_fraction": round(pale_fraction, 4),
        }
