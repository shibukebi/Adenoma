import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from adenoma_agent.trace_supervision import LESION_TRACE_LABELS, connected_components_for_patch_ids
from adenoma_agent.utils import env_with_cuda_visible_devices, read_json, run_command, write_json


class BackendUnavailableError(RuntimeError):
    pass


class BackendExecutionError(RuntimeError):
    pass


class FatalBackendExecutionError(BackendExecutionError):
    pass


class MultimodalStageBackend(object):
    name = "base"
    supported_stages = ()

    def invoke(self, request, bundle):
        raise NotImplementedError


class ExternalCommandStageBackend(MultimodalStageBackend):
    name = "external_command"
    supported_stages = ("trace", "navigate", "observe_step", "observe_report")

    def invoke(self, request, bundle):
        config = bundle["runtime"].get("backends", {}).get("external_command", {})
        if not config.get("enabled") or not config.get("command_prefix"):
            raise BackendUnavailableError("external_command backend is disabled")
        with tempfile.TemporaryDirectory() as tmpdir:
            request_path = Path(tmpdir) / "request.json"
            response_path = Path(tmpdir) / "response.json"
            write_json(request_path, request)
            command = list(config["command_prefix"]) + [
                "--stage",
                request["stage"],
                "--request-json",
                str(request_path),
                "--response-json",
                str(response_path),
            ]
            result = run_command(command, timeout=300)
            if result["returncode"] != 0:
                raise BackendExecutionError(result["stderr"] or result["stdout"])
            if not response_path.exists():
                raise BackendExecutionError("No response_json produced by external command backend")
            return {
                "backend": self.name,
                "output": read_json(response_path),
                "raw_text": result["stdout"],
                "latency_ms": result["latency_ms"],
            }


class LocalPathoR1StageBackend(MultimodalStageBackend):
    name = "local_patho_r1"
    supported_stages = ("trace", "observe_step")

    def invoke(self, request, bundle):
        config = bundle["runtime"].get("backends", {}).get("local_patho_r1", {})
        if not config.get("enabled"):
            raise BackendUnavailableError("local_patho_r1 backend is disabled")
        image_path = request["images"][0]
        if request["stage"] == "trace":
            def _generate_trace_text(prompt_text):
                command = [
                    bundle["runtime"]["paths"]["patho_r1_python"],
                    bundle["runtime"]["paths"]["patho_r1_patch_qa_script"],
                    "--image",
                    image_path,
                    "--prompt",
                    prompt_text,
                    "--model-id",
                    bundle["runtime"]["models"]["patho_r1_model_id"],
                    "--max-new-tokens",
                    str(config.get("max_new_tokens", 256)),
                ]
                patho_r1_env = env_with_cuda_visible_devices(
                    bundle["runtime"].get("execution", {}).get("patho_r1_cuda_visible_devices")
                )
                result = run_command(command, timeout=300, env_overrides=patho_r1_env)
                if result["returncode"] != 0:
                    raise BackendExecutionError(result["stderr"] or result["stdout"])
                text = (result["stdout"] or "").strip()
                if not text:
                    raise BackendExecutionError("Empty Patho-R1 response")
                return {"text": text, "latency_ms": result["latency_ms"]}

            if _load_trace_grid_metadata(request):
                trace_runs = []

                def _generate_trace_text_only(prompt_text):
                    result = _generate_trace_text(prompt_text)
                    trace_runs.append(result)
                    return result["text"]

                trace_response = _run_trace_grid_with_coverage_retry(_generate_trace_text_only, request, bundle)
                return {
                    "backend": self.name,
                    "output": trace_response["output"],
                    "raw_text": trace_response["raw_text"],
                    "raw_texts": trace_response["raw_texts"],
                    "trace_attempts": trace_response["trace_attempts"],
                    "latency_ms": sum(int(item.get("latency_ms", 0)) for item in trace_runs),
                }
            question = _build_trace_patho_r1_prompt(request, bundle)
        elif request["stage"] == "observe_step":
            question = request["prompt"]["question"]
        else:
            raise BackendUnavailableError("local_patho_r1 only supports trace and observe_step")
        command = [
            bundle["runtime"]["paths"]["patho_r1_python"],
            bundle["runtime"]["paths"]["patho_r1_patch_qa_script"],
            "--image",
            image_path,
            "--prompt",
            question,
            "--model-id",
            bundle["runtime"]["models"]["patho_r1_model_id"],
            "--max-new-tokens",
            str(config.get("max_new_tokens", 256)),
        ]
        patho_r1_env = env_with_cuda_visible_devices(
            bundle["runtime"].get("execution", {}).get("patho_r1_cuda_visible_devices")
        )
        result = run_command(command, timeout=300, env_overrides=patho_r1_env)
        if result["returncode"] != 0:
            raise BackendExecutionError(result["stderr"] or result["stdout"])
        text = (result["stdout"] or "").strip()
        if not text:
            raise BackendExecutionError("Empty Patho-R1 response")
        if request["stage"] == "trace":
            output = _build_trace_output_from_text(text, request, bundle)
        else:
            output = _build_text_driven_output(text, request, bundle)
        return {
            "backend": self.name,
            "output": output,
            "raw_text": text,
            "latency_ms": result["latency_ms"],
        }


class LocalCPathAgentQwenStageBackend(MultimodalStageBackend):
    name = "local_cpathagent_qwen"
    supported_stages = ("trace", "navigate", "observe_step", "observe_report")

    def invoke(self, request, bundle):
        config = bundle["runtime"].get("backends", {}).get("local_cpathagent_qwen", {})
        if not config.get("enabled"):
            raise BackendUnavailableError("local_cpathagent_qwen backend is disabled")
        shim_mode = str(config.get("shim_mode", "server")).strip().lower()
        if shim_mode == "heuristic":
            response = HeuristicStageBackend().invoke(request, bundle)
            output = dict(response["output"])
            output["runner_metadata"] = {
                "backend_name": self.name,
                "start_mode": "cold_start_fallback",
                "gpu_device_id": None,
                "server_url": None,
                "round_trip_ms": 0,
            }
            return {
                "backend": self.name,
                "output": output,
                "raw_text": response.get("raw_text", ""),
                "raw_texts": [],
                "trace_attempts": [],
                "latency_ms": response.get("latency_ms", 0),
                "runtime_metadata": output["runner_metadata"],
            }

        try:
            import requests
        except Exception as exc:
            raise BackendUnavailableError("requests is required for local_cpathagent_qwen server mode") from exc

        server_url = str(config.get("server_url", "http://127.0.0.1:8000/predict")).strip()
        timeout_seconds = int(config.get("timeout_seconds", 180))
        prompt_text = _build_local_cpathagent_qwen_prompt(request, bundle)
        response_payload = {
            "image_path": request.get("images", [None])[0] if request.get("images") else None,
            "image_paths": list(request.get("images", [])),
            "prompt": prompt_text,
            "max_new_tokens": int(config.get("max_new_tokens", 512)),
            "stage": request["stage"],
        }
        started = time.time()
        try:
            api_response = requests.post(server_url, json=response_payload, timeout=timeout_seconds)
            round_trip_ms = int(round((time.time() - started) * 1000.0))
        except requests.Timeout as exc:
            raise FatalBackendExecutionError(
                "local_cpathagent_qwen API timed out after {0}s for stage {1}".format(
                    timeout_seconds, request["stage"]
                )
            ) from exc
        except Exception as exc:
            raise FatalBackendExecutionError(
                "local_cpathagent_qwen API request failed for stage {0}: {1}".format(
                    request["stage"], str(exc)
                )
            ) from exc

        if api_response.status_code != 200:
            raise FatalBackendExecutionError(
                "local_cpathagent_qwen API returned HTTP {0}: {1}".format(
                    api_response.status_code, api_response.text
                )
            )
        try:
            api_payload = api_response.json()
        except Exception as exc:
            raise FatalBackendExecutionError("local_cpathagent_qwen API returned non-JSON response") from exc

        generated_text = str(api_payload.get("text", "") or "")
        if not generated_text.strip():
            raise FatalBackendExecutionError("local_cpathagent_qwen API returned empty text")

        output = _parse_local_cpathagent_qwen_output(request["stage"], generated_text, request, bundle)
        runner_metadata = {
            "backend_name": self.name,
            "start_mode": "warm_start",
            "gpu_device_id": api_payload.get("gpu_device_id"),
            "cuda_visible_devices": api_payload.get("cuda_visible_devices"),
            "server_url": server_url,
            "round_trip_ms": round_trip_ms,
            "model_id": api_payload.get("model_id"),
            "adapter_path": api_payload.get("adapter_path"),
            "server_request_id": api_payload.get("request_id"),
        }
        output["runner_metadata"] = runner_metadata
        return {
            "backend": self.name,
            "output": output,
            "raw_text": generated_text,
            "raw_texts": [],
            "trace_attempts": output.get("trace_attempts", []),
            "latency_ms": round_trip_ms,
            "runtime_metadata": runner_metadata,
        }


class HeuristicStageBackend(MultimodalStageBackend):
    name = "heuristic"
    supported_stages = ("trace", "navigate", "observe_step", "observe_report")

    def invoke(self, request, bundle):
        stage = request["stage"]
        if stage == "trace":
            output = self._trace_output(request, bundle)
        elif stage == "navigate":
            output = self._navigate_output(request, bundle)
        elif stage == "observe_step":
            output = self._observe_step_output(request, bundle)
        elif stage == "observe_report":
            output = self._observe_report_output(request, bundle)
        else:
            raise BackendUnavailableError("Unsupported heuristic stage: {0}".format(stage))
        return {"backend": self.name, "output": output, "raw_text": "", "latency_ms": 0}

    def _trace_output(self, request, bundle):
        trace_labels = bundle["runtime"]["trace"]["labels"]
        serrated_criteria = list(bundle["runtime"]["observe"].get("serrated_criteria", []))
        abnormal_crypt_criteria = list(bundle["runtime"]["observe"].get("abnormal_crypt_criteria", []))
        grid_meta = _load_trace_grid_metadata(request)
        if grid_meta:
            image = Image.open(request["images"][0]).convert("RGB")
            arr = np.array(image, dtype=np.float32)
            grid_lookup = _grid_lookup_from_metadata(grid_meta)
            bucket_specs = {
                TRACE_CONVENTIONAL_LABEL: {"priority": 4, "d": True, "name": "conventional adenoma-like mucosa"},
                TRACE_SERRATED_LABEL: {"priority": 4, "d": True, "name": "SSL-suspicious mucosa"},
                TRACE_INFLAMMATORY_LABEL: {"priority": 2, "d": False, "name": "inflammatory polyp-like mucosa"},
                TRACE_NORMAL_LABEL: {"priority": 1, "d": False, "name": "normal mucosa"},
                TRACE_BACKGROUND_LABEL: {"priority": 0, "d": False, "name": "background/artifact/stroma"},
            }
            buckets = {
                label: {"patch_ids": [], "cells": [], "evidence": []}
                for label in bucket_specs
            }
            for cell in _selected_grid_cells(grid_meta):
                bbox = _grid_cell_thumb_bbox(cell)
                crop = arr[int(bbox["y1"]) : int(bbox["y2"]), int(bbox["x1"]) : int(bbox["x2"])]
                mean_rgb = crop.mean(axis=2) if crop.size else np.zeros((1, 1), dtype=np.float32)
                sat = crop.max(axis=2) - crop.min(axis=2) if crop.size else np.zeros((1, 1), dtype=np.float32)
                tissue_mask = (mean_rgb < 235.0) & (sat > 8.0)
                red = crop[:, :, 0] if crop.size else np.zeros((1, 1), dtype=np.float32)
                green = crop[:, :, 1] if crop.size else np.zeros((1, 1), dtype=np.float32)
                blue = crop[:, :, 2] if crop.size else np.zeros((1, 1), dtype=np.float32)
                computed_tissue_fraction = float(tissue_mask.mean()) if crop.size else 0.0
                tissue_fraction = max(float(cell.get("tissue_coverage_ratio", 0.0)), computed_tissue_fraction)
                pale_fraction = float(
                    (((mean_rgb > 165.0) & (mean_rgb < 235.0) & (sat < 28.0) & tissue_mask).mean())
                ) if crop.size else 0.0
                dark_fraction = float((((mean_rgb < 150.0) & tissue_mask).mean())) if crop.size else 0.0
                red_dominance = float(
                    (((red > green * 1.06) & (red > blue * 1.02) & tissue_mask).mean())
                ) if crop.size else 0.0
                artifact_fraction = float(
                    (
                        (
                            (sat > 75.0)
                            & (
                                (crop[:, :, 0] > crop[:, :, 1] * 1.25)
                                | (crop[:, :, 2] > crop[:, :, 1] * 1.25)
                                | (crop[:, :, 1] > crop[:, :, 0] * 1.25)
                            )
                        ).mean()
                    )
                ) if crop.size else 0.0
                if tissue_fraction < 0.10 or artifact_fraction > 0.28:
                    label = TRACE_BACKGROUND_LABEL
                    reason = "low tissue coverage or artifact-dominant patch"
                elif pale_fraction > 0.18:
                    label = TRACE_SERRATED_LABEL
                    reason = "marked pale/mucus-rich pattern raises SSL review priority"
                elif pale_fraction > 0.10:
                    label = TRACE_SERRATED_LABEL
                    reason = "mild pale/mucus-rich pattern is suspicious for serration"
                elif dark_fraction > 0.22:
                    label = TRACE_CONVENTIONAL_LABEL
                    reason = "dark crowded glandular pattern raises conventional adenoma concern"
                elif red_dominance > 0.18:
                    label = TRACE_INFLAMMATORY_LABEL
                    reason = "eosinophilic/reactive appearance favors inflammatory polyp-like mucosa"
                else:
                    label = TRACE_NORMAL_LABEL
                    reason = "reviewable mucosa without strong serrated or adenomatous cues"
                row_col = (int(cell["row_id"]), int(cell["col_id"]))
                buckets[label]["patch_ids"].append(row_col)
                buckets[label]["cells"].append(cell)
                if reason not in buckets[label]["evidence"]:
                    buckets[label]["evidence"].append(reason)

            output_clusters = []
            for index, label in enumerate(
                (
                    TRACE_CONVENTIONAL_LABEL,
                    TRACE_SERRATED_LABEL,
                    TRACE_INFLAMMATORY_LABEL,
                    TRACE_NORMAL_LABEL,
                    TRACE_BACKGROUND_LABEL,
                )
            ):
                patch_ids = buckets[label]["patch_ids"]
                if not patch_ids:
                    continue
                patch_ids_ordered, patches_thumb, patches_level0, selected_cells = _grid_cell_sequence_from_ids(
                    patch_ids,
                    grid_lookup,
                )
                cluster_payload = _build_trace_cluster_payload(
                    cluster_id="heuristic_grid_group_{0:02d}".format(index),
                    label=label,
                    priority=bucket_specs[label]["priority"],
                    require_high_magnification=bucket_specs[label]["d"],
                    desc="; ".join(buckets[label]["evidence"]),
                    evidence=buckets[label]["evidence"],
                    patch_ids_ordered=patch_ids_ordered,
                    patches_thumb=patches_thumb,
                    patches_level0=patches_level0,
                    selected_cells=selected_cells,
                    grid_meta=grid_meta,
                    metadata={
                        "source": "heuristic_grid_trace",
                        "group_name": bucket_specs[label]["name"],
                        "severity_reasoning": "; ".join(buckets[label]["evidence"]),
                        "group_output_index": index,
                        "serrated_criteria_focus": serrated_criteria,
                        "abnormal_crypt_criteria_focus": abnormal_crypt_criteria,
                    },
                )
                if cluster_payload["l"] not in trace_labels:
                    raise BackendExecutionError("Heuristic trace produced unsupported label: {0}".format(cluster_payload["l"]))
                output_clusters.append(cluster_payload)
            return {"clusters": _sort_trace_clusters(output_clusters)}

        clusters = []
        for proposal in request["metadata"]["proposals"]:
            tissue_fraction = float(proposal["metadata"].get("tissue_fraction", 0.0))
            pale_fraction = float(proposal["metadata"].get("pale_fraction", 0.0))
            artifact_fraction = float(proposal["metadata"].get("artifact_fraction", 0.0))
            route_c_overlap = float(proposal["metadata"].get("route_c_hint_overlap", 0.0))
            area_fraction = float(proposal["metadata"].get("area_fraction", 0.0))
            dark_fraction = max(0.0, min(1.0, tissue_fraction * (0.30 + artifact_fraction)))
            inflammatory_score = max(0.0, min(1.0, (1.0 - pale_fraction) * max(0.0, tissue_fraction - 0.20)))
            if tissue_fraction < 0.08:
                label = TRACE_BACKGROUND_LABEL
                priority = 0
                need_high_mag_review = False
                crypt_disorder_risk = 0
                review_stage = "mucosa_screening"
                reasons = ["low tissue fraction on overview screening"]
            elif artifact_fraction > 0.28:
                label = TRACE_BACKGROUND_LABEL
                priority = 0
                need_high_mag_review = False
                crypt_disorder_risk = 0
                review_stage = "mucosa_screening"
                reasons = ["high artifact-like color fraction"]
            else:
                serrated_score = 1
                reasons = ["mucosal tissue retained after overview filtering"]
                if pale_fraction > 0.10:
                    serrated_score += 1
                    reasons.append("surface pallor / mucus-rich pattern supports an SSL impression")
                if route_c_overlap > 0.05:
                    serrated_score += 1
                    reasons.append("region overlaps a route-C low-resolution hint")
                if 0.02 <= area_fraction <= 0.35:
                    serrated_score += 1
                    reasons.append("region size is suitable for structured lesion review")
                if pale_fraction > 0.18 and serrated_score >= 3:
                    label = TRACE_SERRATED_LABEL
                    priority = 5
                    need_high_mag_review = True
                    review_stage = _trace_review_stage_for_label(label)
                    crypt_disorder_risk = min(5, serrated_score + 1)
                elif dark_fraction > 0.16 and tissue_fraction > 0.35:
                    label = TRACE_CONVENTIONAL_LABEL
                    priority = 4
                    need_high_mag_review = True
                    crypt_disorder_risk = 0
                    review_stage = _trace_review_stage_for_label(label)
                    reasons.append("crowded darker gland-rich pattern raises conventional adenoma concern")
                elif inflammatory_score > 0.22 and route_c_overlap < 0.08:
                    label = TRACE_INFLAMMATORY_LABEL
                    priority = 2
                    need_high_mag_review = False
                    crypt_disorder_risk = 0
                    review_stage = _trace_review_stage_for_label(label)
                    reasons.append("reactive/inflammatory appearance is favored over an adenomatous or SSL pattern")
                elif serrated_score >= 2:
                    label = TRACE_SERRATED_LABEL
                    priority = min(4, max(2, int(serrated_score)))
                    need_high_mag_review = True
                    review_stage = _trace_review_stage_for_label(label)
                    crypt_disorder_risk = min(5, serrated_score + (1 if pale_fraction > 0.18 else 0))
                else:
                    label = TRACE_NORMAL_LABEL
                    priority = 1
                    need_high_mag_review = False
                    crypt_disorder_risk = 0
                    review_stage = _trace_review_stage_for_label(label)
                if label == TRACE_SERRATED_LABEL:
                    reasons.append("cluster should enter serrated branch review")
                elif label == TRACE_CONVENTIONAL_LABEL:
                    reasons.append("cluster should enter conventional adenoma review")

            if label not in trace_labels:
                raise BackendExecutionError("Heuristic trace produced unsupported label: {0}".format(label))
            normalized_metadata = _trace_label_metadata(
                label,
                priority,
                need_high_mag_review,
                {
                    **proposal["metadata"],
                    "mucosa_retained": label != TRACE_BACKGROUND_LABEL,
                    "serrated_criteria_focus": serrated_criteria,
                    "abnormal_crypt_criteria_focus": abnormal_crypt_criteria,
                    "conventional_subtype_hint": (
                        "tubulovillous_adenoma_like" if label == TRACE_CONVENTIONAL_LABEL and area_fraction > 0.18 else
                        "tubular_adenoma_like" if label == TRACE_CONVENTIONAL_LABEL else
                        proposal["metadata"].get("conventional_subtype_hint")
                    ),
                    "inflammatory_subtype_hint": (
                        "inflammatory_polyp_like" if label == TRACE_INFLAMMATORY_LABEL else
                        proposal["metadata"].get("inflammatory_subtype_hint")
                    ),
                    "serrated_family_hint": (
                        "ssl_like" if label == TRACE_SERRATED_LABEL and priority >= 5 else
                        "equivocal_serrated" if label == TRACE_SERRATED_LABEL else
                        proposal["metadata"].get("serrated_family_hint")
                    ),
                },
            )
            clusters.append(
                {
                    "cluster_id": proposal["cluster_id"],
                    "l": label,
                    "s": priority,
                    "d": need_high_mag_review,
                    "review_stage": review_stage,
                    "crypt_disorder_risk": crypt_disorder_risk,
                    "dysplasia_review_needed": bool(
                        normalized_metadata.get("serrated_dysplasia_suspected")
                        or normalized_metadata.get("conventional_dysplasia_suspected")
                    ),
                    "desc": "; ".join(reasons),
                    "evidence": reasons,
                    "metadata": normalized_metadata,
                    "patch_ids_ordered": [],
                    "patches_thumb": [],
                    "patches_level0": [],
                    "group_bbox_thumb": dict(proposal["cluster_bbox_thumb"]),
                    "group_bbox_level0": dict(proposal["cluster_bbox_level0"]),
                }
            )
        return {"clusters": _sort_trace_clusters(clusters)}

    def _navigate_output(self, request, bundle):
        slide_dims = request["metadata"]["slide_dimensions_level0"]
        overlap_threshold = float(bundle["runtime"]["navigate"].get("overlap_threshold", 0.30))
        mag_to_region = bundle["budget"].get("magnification_to_region_size", {})
        clusters = _sort_trace_clusters(request["metadata"]["clusters"])

        steps = []
        prior_windows = []
        step_index = 0
        max_steps = int(bundle["budget"].get("max_navigation_steps", 8))
        for cluster in clusters:
            if int(cluster["s"]) <= 0 or cluster["l"] == TRACE_BACKGROUND_LABEL:
                continue
            if step_index >= max_steps:
                break
            patch_sequence = list(cluster.get("patches_level0") or [])
            if not patch_sequence:
                bbox = cluster.get("group_bbox_level0") or cluster["cluster_bbox_level0"]
                patch_sequence = [
                    {
                        "patch_id": [],
                        "x1": int(bbox["x1"]),
                        "y1": int(bbox["y1"]),
                        "x2": int(bbox["x2"]),
                        "y2": int(bbox["y2"]),
                    }
                ]
            branch = _trace_branch_for_label(cluster["l"])
            if cluster["l"] == TRACE_NORMAL_LABEL:
                step_specs = [
                    (
                        5.0,
                        "non_serrated_overview_assessment",
                        "non_serrated_context",
                        "Confirm this low-priority mucosal patch is non-lesional and does not hide meaningful serrated or adenomatous change.",
                    )
                ]
            elif cluster["l"] in (TRACE_SERRATED_LABEL, TRACE_LEGACY_SSL_HIGH_LABEL):
                low_power_note = (
                    "Inspect this highest-priority SSL-suspicious patch first and confirm lesion context at 5x."
                    if int(cluster.get("s", 0)) >= 5
                    else "Confirm that this patch belongs to the serrated lesion pathway and merits directed follow-up at 5x."
                )
                step_specs = [
                    (
                        5.0,
                        "serrated_lesion_assessment",
                        "mucosa_or_serrated",
                        low_power_note,
                    )
                ]
                if cluster["d"]:
                    step_specs.append(
                        (
                            20.0,
                            "abnormal_crypt_assessment",
                            "abnormal_crypt",
                            "Search for abnormal crypt architecture, including basal dilatation, branching, horizontal growth, and serration extending toward the crypt base.",
                        )
                    )
            elif cluster["l"] == TRACE_CONVENTIONAL_LABEL:
                step_specs = [
                    (
                        5.0,
                        "conventional_adenoma_assessment",
                        "conventional_adenoma",
                        "Review gland architecture for a conventional adenoma pattern, including tubular or tubulovillous crowding.",
                    )
                ]
            elif cluster["l"] == TRACE_INFLAMMATORY_LABEL:
                step_specs = [
                    (
                        5.0,
                        "non_serrated_overview_assessment",
                        "non_serrated_context",
                        "Confirm inflammatory polyp-like or reactive features and keep this region out of the dysplasia branch unless later evidence contradicts the overview.",
                    )
                ]
            else:
                step_specs = [
                    (
                        5.0,
                        "non_serrated_overview_assessment",
                        "non_serrated_context",
                        "Confirm that this retained patch is background or low-value tissue only.",
                    )
                ]

            for patch_index, patch in enumerate(patch_sequence):
                if step_index >= max_steps:
                    break
                anchor_x = patch.get("anchor_x")
                anchor_y = patch.get("anchor_y")
                if anchor_x is not None and anchor_y is not None:
                    center_x = int(anchor_x)
                    center_y = int(anchor_y)
                else:
                    center_x = int(round((int(patch["x1"]) + int(patch["x2"])) / 2.0))
                    center_y = int(round((int(patch["y1"]) + int(patch["y2"])) / 2.0))
                for magnification, review_goal, stage_gate, need_to_see in step_specs:
                    if step_index >= max_steps:
                        break
                    region_size = int(mag_to_region.get(str(float(magnification)), 256))
                    from adenoma_agent.utils import bbox_overlap_ratio, clamp_center_point, normalized_point

                    step_bbox = {
                        "x1": center_x - region_size // 2,
                        "y1": center_y - region_size // 2,
                        "x2": center_x + region_size // 2,
                        "y2": center_y + region_size // 2,
                    }
                    should_skip = False
                    for prior_item in prior_windows:
                        if abs(float(prior_item["m"]) - float(magnification)) > 1e-6:
                            continue
                        if bbox_overlap_ratio(step_bbox, prior_item["bbox"]) > overlap_threshold:
                            should_skip = True
                            break
                    if should_skip:
                        continue
                    fixed_x, fixed_y = clamp_center_point(center_x, center_y, region_size, slide_dims)
                    step_bbox = {
                        "x1": fixed_x - region_size // 2,
                        "y1": fixed_y - region_size // 2,
                        "x2": fixed_x + region_size // 2,
                        "y2": fixed_y + region_size // 2,
                    }
                    prior_windows.append({"bbox": step_bbox, "m": float(magnification)})
                    steps.append(
                        {
                            "step_id": "step_{0:02d}".format(step_index),
                            "x": fixed_x,
                            "y": fixed_y,
                            "m": float(magnification),
                            "region_size_level0": region_size,
                            "need_to_see": need_to_see,
                            "review_goal": review_goal,
                            "stage_gate": stage_gate,
                            "metadata": {
                                "cluster_id": cluster["cluster_id"],
                                "source_group_id": cluster["cluster_id"],
                                "cluster_label": cluster["l"],
                                "cluster_priority": cluster["s"],
                                "patch_id": list(patch.get("patch_id", [])),
                                "patch_index": patch_index,
                                "cluster_patch_count": len(patch_sequence),
                                "region_size_level0": region_size,
                                "normalized_center": normalized_point(fixed_x, fixed_y, slide_dims),
                                "anchor_source": patch.get("anchor_source"),
                                "action": "inspect",
                                "workflow_branch": branch,
                            },
                        }
                    )
                    step_index += 1

        if not steps:
            steps.append(
                {
                    "step_id": "step_00",
                    "x": 0,
                    "y": 0,
                    "m": 5.0,
                    "region_size_level0": 256,
                    "need_to_see": "Stop navigation because no reviewable lesion cluster was retained.",
                    "review_goal": "integrated_impression",
                    "stage_gate": "end",
                    "metadata": {"action": "stop", "region_size_level0": 256},
                }
            )
        else:
            last = steps[-1]
            steps.append(
                {
                    "step_id": "step_{0:02d}".format(len(steps)),
                    "x": last["x"],
                    "y": last["y"],
                    "m": 5.0,
                    "region_size_level0": 256,
                    "need_to_see": "Stop navigation and consolidate serrated, conventional adenoma, inflammatory, and branch-specific dysplasia evidence gathered so far.",
                    "review_goal": "integrated_impression",
                    "stage_gate": "end",
                    "metadata": {"action": "stop", "region_size_level0": 256},
                }
            )
        return {"steps": steps}

    def _observe_step_output(self, request, bundle):
        stats = request["metadata"]["image_stats"]
        image_stats_bundle = list(request["metadata"].get("image_stats_bundle", []))
        step = request["metadata"]["step"]
        cluster = request["metadata"].get("cluster", {})
        serrated_criteria = list(bundle["runtime"]["observe"].get("serrated_criteria", []))
        abnormal_crypt_criteria = list(bundle["runtime"]["observe"].get("abnormal_crypt_criteria", []))
        conventional_criteria = list(bundle["runtime"]["observe"].get("conventional_adenoma_criteria", []))
        dysplasia_criteria = list(bundle["runtime"]["observe"].get("dysplasia_criteria", []))
        background_fraction = float(stats.get("background_fraction", 0.0))
        pale_fraction = float(stats.get("pale_fraction", 0.0))
        tissue_fraction = float(stats.get("tissue_fraction", 0.0))
        cluster_priority = int(cluster.get("s", 0))
        crypt_disorder_risk = int(cluster.get("crypt_disorder_risk", cluster_priority))
        review_goal = step.get("review_goal")
        view_count = max(1, len(image_stats_bundle) or len(request.get("images", [])))

        cluster_metadata = cluster.get("metadata", {}) if isinstance(cluster.get("metadata", {}), dict) else {}
        conventional_subtype_hint = cluster_metadata.get("conventional_subtype_hint")
        serrated_dysplasia_suspected = bool(cluster_metadata.get("serrated_dysplasia_suspected", False))
        conventional_dysplasia_suspected = bool(cluster_metadata.get("conventional_dysplasia_suspected", False))

        serrated_hits = _blank_hits(serrated_criteria)
        abnormal_crypt_hits = _blank_hits(abnormal_crypt_criteria)
        conventional_hits = _blank_hits(conventional_criteria)
        serrated_dysplasia_hits = _blank_hits(dysplasia_criteria)
        conventional_dysplasia_hits = _blank_hits(dysplasia_criteria)

        if review_goal == "serrated_lesion_assessment":
            if cluster.get("l") in (TRACE_SERRATED_LABEL, TRACE_LEGACY_SSL_HIGH_LABEL):
                serrated_hits["serrated_lesion_context"] = "supporting"
                serrated_hits["serrated_surface_pattern"] = "supporting" if pale_fraction > 0.10 or cluster_priority >= 5 else "uncertain"
                serrated_hits["mucus_rich_surface"] = "supporting" if pale_fraction > 0.18 or cluster_priority >= 5 else "uncertain"
            elif cluster.get("l") == TRACE_NORMAL_LABEL:
                serrated_hits["serrated_lesion_context"] = "opposing"
                serrated_hits["serrated_surface_pattern"] = "opposing" if pale_fraction < 0.08 else "uncertain"
                serrated_hits["mucus_rich_surface"] = "opposing" if pale_fraction < 0.08 else "uncertain"
        elif review_goal == "abnormal_crypt_assessment":
            if pale_fraction > 0.18 and cluster_priority >= 4:
                abnormal_crypt_hits["serration_to_base"] = "supporting"
                abnormal_crypt_hits["mucus_cap"] = "supporting"
                abnormal_crypt_hits["abnormal_maturation"] = "supporting" if pale_fraction > 0.18 else "uncertain"
            else:
                abnormal_crypt_hits["serration_to_base"] = "uncertain"
                abnormal_crypt_hits["mucus_cap"] = "uncertain"
                abnormal_crypt_hits["abnormal_maturation"] = "uncertain"
            if crypt_disorder_risk >= 5 and tissue_fraction > 0.60 and pale_fraction > 0.15:
                abnormal_crypt_hits["basal_dilatation"] = "supporting"
                abnormal_crypt_hits["crypt_branching"] = "supporting"
                abnormal_crypt_hits["horizontal_growth"] = "supporting"
                abnormal_crypt_hits["boot_l_t_shaped_crypt"] = "supporting"
            elif crypt_disorder_risk >= 3:
                abnormal_crypt_hits["basal_dilatation"] = "uncertain"
                abnormal_crypt_hits["crypt_branching"] = "uncertain"
                abnormal_crypt_hits["horizontal_growth"] = "uncertain"
                abnormal_crypt_hits["boot_l_t_shaped_crypt"] = "uncertain"
        elif review_goal == "conventional_adenoma_assessment":
            if cluster.get("l") == TRACE_CONVENTIONAL_LABEL:
                conventional_hits["tubular_or_tubulovillous_architecture"] = "supporting"
                conventional_hits["crowded_adenomatous_glands"] = "supporting" if tissue_fraction > 0.45 else "uncertain"
                if conventional_subtype_hint == "tubulovillous_adenoma_like":
                    conventional_hits["pencillate_hyperchromatic_nuclei"] = "supporting"
                elif tissue_fraction > 0.30:
                    conventional_hits["pencillate_hyperchromatic_nuclei"] = "uncertain"
            elif cluster.get("l") == TRACE_INFLAMMATORY_LABEL:
                conventional_hits["tubular_or_tubulovillous_architecture"] = "opposing"
                conventional_hits["crowded_adenomatous_glands"] = "opposing"
                conventional_hits["pencillate_hyperchromatic_nuclei"] = "uncertain"
        elif review_goal == "serrated_dysplasia_assessment":
            if cluster_priority >= 5 and tissue_fraction > 0.70 and pale_fraction < 0.12:
                serrated_dysplasia_hits["nuclear_enlargement_stratification"] = "supporting"
                serrated_dysplasia_hits["hyperchromasia"] = "supporting"
                serrated_dysplasia_hits["architectural_crowding"] = "uncertain"
                serrated_dysplasia_hits["mitotic_activity_atypia"] = "uncertain"
            elif tissue_fraction > 0.40 or serrated_dysplasia_suspected:
                serrated_dysplasia_hits["nuclear_enlargement_stratification"] = "uncertain"
                serrated_dysplasia_hits["hyperchromasia"] = "uncertain"
                serrated_dysplasia_hits["architectural_crowding"] = "uncertain"
                serrated_dysplasia_hits["mitotic_activity_atypia"] = "uncertain"
        elif review_goal == "conventional_dysplasia_assessment":
            if conventional_dysplasia_suspected and tissue_fraction > 0.55:
                conventional_dysplasia_hits["nuclear_enlargement_stratification"] = "supporting"
                conventional_dysplasia_hits["hyperchromasia"] = "supporting"
                conventional_dysplasia_hits["architectural_crowding"] = (
                    "supporting" if conventional_subtype_hint == "tubulovillous_adenoma_like" else "uncertain"
                )
                conventional_dysplasia_hits["mitotic_activity_atypia"] = "uncertain"
            elif tissue_fraction > 0.40:
                conventional_dysplasia_hits["nuclear_enlargement_stratification"] = "uncertain"
                conventional_dysplasia_hits["hyperchromasia"] = "uncertain"
                conventional_dysplasia_hits["architectural_crowding"] = "uncertain"
                conventional_dysplasia_hits["mitotic_activity_atypia"] = "uncertain"

        dysplasia_hits = _combine_hits_maps(serrated_dysplasia_hits, conventional_dysplasia_hits)

        if review_goal == "serrated_lesion_assessment":
            level_1_findings = _supporting_findings_from_hits(serrated_hits)
            level_2_findings = []
            level_3_findings = []
        elif review_goal == "abnormal_crypt_assessment":
            level_1_findings = []
            level_2_findings = _supporting_findings_from_hits(abnormal_crypt_hits)
            level_3_findings = []
        elif review_goal == "conventional_adenoma_assessment":
            level_1_findings = _supporting_findings_from_hits(conventional_hits)
            level_2_findings = []
            level_3_findings = []
        elif review_goal == "serrated_dysplasia_assessment":
            level_1_findings = []
            level_2_findings = []
            level_3_findings = _supporting_findings_from_hits(serrated_dysplasia_hits)
        elif review_goal == "conventional_dysplasia_assessment":
            level_1_findings = []
            level_2_findings = []
            level_3_findings = _supporting_findings_from_hits(conventional_dysplasia_hits)
        else:
            level_1_findings = []
            level_2_findings = []
            level_3_findings = []

        if background_fraction > 0.7:
            observation = "The crop is background-heavy and provides limited diagnostic tissue."
        elif review_goal == "serrated_lesion_assessment":
            observation = "Low magnification preserves the overall mucosal context for serrated pathway screening."
        elif review_goal == "abnormal_crypt_assessment":
            observation = "Intermediate magnification targets crypt architecture and abnormal serration distribution."
        elif review_goal == "conventional_adenoma_assessment":
            observation = "This view reviews gland architecture for a conventional adenoma pattern before lineage-specific dysplasia assessment."
        elif review_goal == "conventional_dysplasia_assessment":
            observation = "High magnification focuses on dysplasia within a conventional adenoma-like region, supported by the multi-view bundle."
        elif review_goal == "serrated_dysplasia_assessment":
            observation = "High magnification focuses on dysplasia after serrated abnormal crypt support has been established, using multi-view context."
        else:
            observation = "This overview confirms a low-priority non-serrated or inflammatory region."

        if review_goal == "serrated_lesion_assessment":
            stage_decision = "supports_serrated_lesion" if level_1_findings else "leans_non_serrated_or_indeterminate"
            reasoning = "This view decides whether retained mucosa belongs to the serrated pathway before abnormal crypt review."
            next_step = (
                "Proceed to abnormal crypt review." if cluster.get("d") else "Consolidate as a non-serrated or low-priority serrated mucosal region."
            )
        elif review_goal == "abnormal_crypt_assessment":
            stage_decision = (
                "supports_abnormal_crypt"
                if level_2_findings
                else "serrated_but_no_support_for_abnormal_crypt"
            )
            reasoning = "This view evaluates whether the crypt pattern supports abnormal crypt architecture within the serrated pathway."
            next_step = (
                "Proceed to serrated dysplasia review."
                if stage_decision == "supports_abnormal_crypt"
                else "Do not enter dysplasia because abnormal crypt support is not established."
            )
        elif review_goal == "conventional_adenoma_assessment":
            stage_decision = (
                "supports_conventional_adenoma"
                if level_1_findings
                else "conventional_adenoma_indeterminate_or_opposed"
            )
            reasoning = "This view evaluates whether the region belongs to the conventional adenoma branch before its own dysplasia review."
            next_step = "Proceed to conventional dysplasia review for this adenoma-like branch."
        elif review_goal == "serrated_dysplasia_assessment":
            stage_decision = (
                "serrated_dysplasia_supported"
                if level_3_findings
                else "serrated_dysplasia_not_supported_or_indeterminate"
            )
            reasoning = "This view evaluates dysplasia specifically within the serrated branch after abnormal crypt support."
            next_step = "Integrate serrated, abnormal crypt, and serrated dysplasia evidence into the final report."
        elif review_goal == "conventional_dysplasia_assessment":
            stage_decision = (
                "conventional_dysplasia_supported"
                if level_3_findings
                else "conventional_dysplasia_not_supported_or_indeterminate"
            )
            reasoning = "This view evaluates dysplasia specifically within the conventional adenoma branch."
            next_step = "Integrate conventional adenoma and branch-specific dysplasia evidence into the final report."
        else:
            stage_decision = "supports_non_serrated_overview" if cluster.get("l") != TRACE_BACKGROUND_LABEL else "background_or_low_value"
            reasoning = "This view confirms that a retained patch belongs to a low-priority non-serrated or inflammatory context."
            next_step = "Keep this region out of the dysplasia branch unless later evidence contradicts the overview."

        support_count = len(level_1_findings) + len(level_2_findings) + len(level_3_findings)
        confidence = min(
            0.95,
            max(
                0.05,
                0.20
                + 0.20 * tissue_fraction
                + 0.10 * pale_fraction
                + 0.08 * support_count
                + 0.04 * cluster_priority
                + 0.01 * max(0, view_count - 1),
            ),
        )
        return {
            "observation": observation,
            "reasoning": reasoning,
            "next_step": next_step,
            "level_1_findings": level_1_findings,
            "level_2_findings": level_2_findings,
            "level_3_findings": level_3_findings,
            "stage_decision": stage_decision,
            "serrated_hits": serrated_hits,
            "abnormal_crypt_hits": abnormal_crypt_hits,
            "conventional_hits": conventional_hits,
            "serrated_dysplasia_hits": serrated_dysplasia_hits,
            "conventional_dysplasia_hits": conventional_dysplasia_hits,
            "dysplasia_hits": dysplasia_hits,
            "view_count": view_count,
            "confidence": round(confidence, 4),
        }

    def _observe_report_output(self, request, bundle):
        serrated_criteria = list(bundle["runtime"]["observe"].get("serrated_criteria", []))
        abnormal_crypt_criteria = list(bundle["runtime"]["observe"].get("abnormal_crypt_criteria", []))
        conventional_criteria = list(bundle["runtime"]["observe"].get("conventional_adenoma_criteria", []))
        dysplasia_criteria = list(bundle["runtime"]["observe"].get("dysplasia_criteria", []))
        records = request["metadata"]["records"]
        trace_clusters = request["metadata"]["trace_clusters"]

        serrated_checklist = _aggregate_hits(records, "serrated_hits", serrated_criteria)
        abnormal_crypt_checklist = _aggregate_hits(records, "abnormal_crypt_hits", abnormal_crypt_criteria)
        conventional_adenoma_checklist = _aggregate_hits(records, "conventional_hits", conventional_criteria)
        serrated_dysplasia_checklist = _aggregate_hits(records, "serrated_dysplasia_hits", dysplasia_criteria)
        conventional_dysplasia_checklist = _aggregate_hits(records, "conventional_dysplasia_hits", dysplasia_criteria)
        dysplasia_checklist = _merge_checklists(
            serrated_dysplasia_checklist,
            conventional_dysplasia_checklist,
        )

        serrated_assessment = _serrated_assessment(trace_clusters, serrated_checklist)
        abnormal_crypt_assessment = _abnormal_crypt_assessment(serrated_assessment, abnormal_crypt_checklist)
        conventional_adenoma_assessment = _conventional_adenoma_assessment(
            trace_clusters,
            conventional_adenoma_checklist,
        )
        serrated_dysplasia_assessment = _branch_dysplasia_assessment(
            abnormal_crypt_assessment,
            serrated_dysplasia_checklist,
            gate_label="not_entered_due_to_crypt_gate",
            supported_label="serrated_dysplasia_supported",
            negative_label="serrated_dysplasia_not_supported",
            indeterminate_label="serrated_dysplasia_indeterminate",
        )
        conventional_dysplasia_assessment = _branch_dysplasia_assessment(
            conventional_adenoma_assessment,
            conventional_dysplasia_checklist,
            gate_label="not_entered_due_to_conventional_gate",
            supported_label="conventional_dysplasia_supported",
            negative_label="conventional_dysplasia_not_supported",
            indeterminate_label="conventional_dysplasia_indeterminate",
        )
        dysplasia_assessment = _overall_dysplasia_assessment(
            serrated_dysplasia_assessment,
            conventional_dysplasia_assessment,
        )
        final_case_assessment = _final_case_assessment(
            serrated_assessment,
            serrated_dysplasia_assessment,
            conventional_adenoma_assessment,
            conventional_dysplasia_assessment,
        )
        integrated_impression = _integrated_impression(
            serrated_assessment,
            abnormal_crypt_assessment,
            serrated_dysplasia_assessment,
            conventional_adenoma_assessment,
            conventional_dysplasia_assessment,
            final_case_assessment,
        )

        lines = []
        lines.append("Integrated Pathological Report")
        lines.append("Task: mucosa -> serrated branch or conventional branch -> branch-specific dysplasia")
        lines.append("")
        lines.append("Serrated lesion assessment:")
        lines.append("- Impression: {0}".format(serrated_assessment["label"]))
        lines.append("- Supporting findings: {0}".format(_render_supporting_lines(serrated_checklist)))
        lines.append("")
        lines.append("Abnormal crypt assessment:")
        lines.append("- Impression: {0}".format(abnormal_crypt_assessment["label"]))
        lines.append("- Supporting findings: {0}".format(_render_supporting_lines(abnormal_crypt_checklist)))
        lines.append("")
        lines.append("Conventional adenoma assessment:")
        lines.append("- Impression: {0}".format(conventional_adenoma_assessment["label"]))
        lines.append("- Supporting findings: {0}".format(_render_supporting_lines(conventional_adenoma_checklist)))
        lines.append("")
        lines.append("Serrated-branch dysplasia assessment:")
        lines.append("- Impression: {0}".format(serrated_dysplasia_assessment["label"]))
        lines.append("- Supporting findings: {0}".format(_render_supporting_lines(serrated_dysplasia_checklist)))
        lines.append("")
        lines.append("Conventional-branch dysplasia assessment:")
        lines.append("- Impression: {0}".format(conventional_dysplasia_assessment["label"]))
        lines.append("- Supporting findings: {0}".format(_render_supporting_lines(conventional_dysplasia_checklist)))
        lines.append("")
        lines.append("Final case classification:")
        lines.append("- Label: {0}".format(final_case_assessment["label"]))
        if final_case_assessment.get("coexisting_labels"):
            lines.append("- Coexisting labels: {0}".format(", ".join(final_case_assessment["coexisting_labels"])))
        lines.append("")
        lines.append("Integrated impression:")
        lines.append("- {0}".format(integrated_impression))
        return {
            "hierarchical_prediction": {
                "serrated_lesion_assessment": serrated_assessment,
                "abnormal_crypt_assessment": abnormal_crypt_assessment,
                "conventional_adenoma_assessment": conventional_adenoma_assessment,
                "serrated_dysplasia_assessment": serrated_dysplasia_assessment,
                "conventional_dysplasia_assessment": conventional_dysplasia_assessment,
                "dysplasia_assessment": dysplasia_assessment,
                "final_case_assessment": final_case_assessment,
                "integrated_impression": integrated_impression,
            },
            "serrated_checklist": serrated_checklist,
            "abnormal_crypt_checklist": abnormal_crypt_checklist,
            "conventional_adenoma_checklist": conventional_adenoma_checklist,
            "serrated_dysplasia_checklist": serrated_dysplasia_checklist,
            "conventional_dysplasia_checklist": conventional_dysplasia_checklist,
            "dysplasia_checklist": dysplasia_checklist,
            "integrated_report": "\n".join(lines),
        }


class StageBackendChain(object):
    def __init__(self, bundle):
        self.bundle = bundle
        self.backends = {
            "external_command": ExternalCommandStageBackend(),
            "local_cpathagent_qwen": LocalCPathAgentQwenStageBackend(),
            "local_patho_r1": LocalPathoR1StageBackend(),
            "heuristic": HeuristicStageBackend(),
        }

    def invoke(self, stage, chain_names, request):
        attempts = []
        request = {**request, "stage": stage}
        for backend_name in chain_names:
            backend = self.backends[backend_name]
            try:
                response = backend.invoke(request, self.bundle)
                response["attempts"] = attempts + [
                    {"backend": backend_name, "status": "ok", "latency_ms": response.get("latency_ms", 0)}
                ]
                return response
            except BackendUnavailableError as exc:
                attempts.append({"backend": backend_name, "status": "unavailable", "error": str(exc)})
            except FatalBackendExecutionError as exc:
                attempts.append({"backend": backend_name, "status": "fatal_error", "error": str(exc)})
                raise FatalBackendExecutionError(
                    "Fatal backend failure for stage {0}: {1}".format(stage, attempts)
                )
            except Exception as exc:
                attempts.append({"backend": backend_name, "status": "error", "error": str(exc)})
        raise BackendExecutionError("All backends failed for stage {0}: {1}".format(stage, attempts))


def _build_local_cpathagent_qwen_prompt(request, bundle):
    stage = request["stage"]
    if stage == "trace":
        return _build_trace_patho_r1_prompt(request, bundle)
    if stage == "navigate":
        return _build_cpathagent_qwen_navigate_prompt(request)
    if stage == "observe_step":
        return request["prompt"]["question"]
    if stage == "observe_report":
        return _build_cpathagent_qwen_observe_report_prompt(request)
    raise ValueError("Unsupported stage: {0}".format(stage))


def _parse_local_cpathagent_qwen_output(stage, generated_text, request, bundle):
    if stage == "trace":
        return _build_trace_output_from_text(generated_text, request, bundle)
    if stage == "navigate":
        return _parse_cpathagent_qwen_navigate_output(generated_text, request, bundle)
    if stage == "observe_step":
        return _build_text_driven_output(generated_text, request, bundle)
    if stage == "observe_report":
        return _parse_cpathagent_qwen_observe_report_output(generated_text, request, bundle)
    raise ValueError("Unsupported stage: {0}".format(stage))


def _extract_first_json_object_from_text(text):
    start = str(text or "").find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _first_json_dict_from_text(text):
    blob = _extract_first_json_object_from_text(text)
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None


def _build_cpathagent_qwen_navigate_prompt(request):
    clusters = request["metadata"].get("clusters", [])
    lines = [
        "You are the Navigation Planning Agent in a pathology workflow.",
        "Given the overview image and grouped trace regions, produce a pathology viewing path as JSON only.",
        "Prioritize higher-s groups first. Preserve branch semantics.",
        "Use only 5x and 20x navigation magnifications.",
        "For ssl_suspicious_mucosa: plan 5x overview and, if d=true, add 20x abnormal_crypt_assessment.",
        "For conventional_adenoma_like: plan 5x conventional_adenoma_assessment.",
        "For inflammatory_polyp_like or normal_mucosa: 5x overview only when s>0.",
        "",
        "Return JSON:",
        '{ "steps": [ { "source_group_id": "grid_group_00", "patch_id": [0, 0], "x": 100, "y": 200, "m": 5.0, "region_size_level0": 256, "need_to_see": "what to inspect", "review_goal": "serrated_lesion_assessment", "stage_gate": "mucosa_or_serrated" }, { "source_group_id": "grid_group_00", "patch_id": [0, 0], "x": 100, "y": 200, "m": 20.0, "region_size_level0": 64, "need_to_see": "higher magnification", "review_goal": "abnormal_crypt_assessment", "stage_gate": "abnormal_crypt" } ] }',
        "",
        "Clusters:",
    ]
    for cluster in clusters:
        lines.append(
            "- cluster_id={cluster_id}, label={label}, s={priority}, d={need_high_mag}, patch_ids={patch_ids}, centers={centers}".format(
                cluster_id=cluster.get("cluster_id"),
                label=cluster.get("l"),
                priority=cluster.get("s"),
                need_high_mag=cluster.get("d"),
                patch_ids=cluster.get("patch_ids_ordered", []),
                centers=cluster.get("metadata", {}).get("representative_centers_level0", []),
            )
        )
    return "\n".join(lines)


def _parse_cpathagent_qwen_navigate_output(text, request, bundle):
    parsed = _first_json_dict_from_text(text) or {}
    steps = parsed.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    if not steps:
        raise BackendUnavailableError("local_cpathagent_qwen navigate returned no valid steps")

    clusters = {cluster["cluster_id"]: cluster for cluster in request["metadata"].get("clusters", [])}
    mag_to_region = bundle.get("budget", {}).get("magnification_to_region_size", {})
    normalized_steps = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        source_group_id = step.get("source_group_id") or step.get("cluster_id")
        cluster = clusters.get(source_group_id, {})
        patch_id = step.get("patch_id") or []
        x = step.get("x")
        y = step.get("y")
        if (x is None or y is None) and patch_id:
            for patch in cluster.get("patches_level0", []):
                if list(patch.get("patch_id", [])) == list(patch_id):
                    anchor_x = patch.get("anchor_x")
                    anchor_y = patch.get("anchor_y")
                    if anchor_x is not None and anchor_y is not None:
                        x = int(anchor_x)
                        y = int(anchor_y)
                    else:
                        x = int(round((int(patch["x1"]) + int(patch["x2"])) / 2.0))
                        y = int(round((int(patch["y1"]) + int(patch["y2"])) / 2.0))
                    break
        if x is None or y is None:
            bbox = cluster.get("group_bbox_level0") or cluster.get("cluster_bbox_level0") or {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
            x = int(round((int(bbox["x1"]) + int(bbox["x2"])) / 2.0))
            y = int(round((int(bbox["y1"]) + int(bbox["y2"])) / 2.0))
        magnification = float(step.get("m", 5.0))
        region_size = int(step.get("region_size_level0", mag_to_region.get(str(magnification), 256)))
        normalized_steps.append(
            {
                "step_id": "step_{0:02d}".format(index),
                "x": int(x),
                "y": int(y),
                "m": magnification,
                "region_size_level0": region_size,
                "need_to_see": step.get("need_to_see", step.get("o", "Inspect the planned pathology region.")),
                "review_goal": step.get("review_goal", "serrated_lesion_assessment"),
                "stage_gate": step.get("stage_gate", "mucosa_or_serrated"),
                "metadata": {
                    "cluster_id": source_group_id,
                    "source_group_id": source_group_id,
                    "cluster_label": cluster.get("l"),
                    "cluster_priority": cluster.get("s"),
                    "patch_id": list(patch_id),
                    "region_size_level0": region_size,
                    "workflow_branch": cluster.get("metadata", {}).get("workflow_branch"),
                    "action": "inspect",
                },
            }
        )
    if not normalized_steps:
        raise BackendUnavailableError("local_cpathagent_qwen navigate could not normalize any valid steps")
    last = normalized_steps[-1]
    normalized_steps.append(
        {
            "step_id": "step_{0:02d}".format(len(normalized_steps)),
            "x": last["x"],
            "y": last["y"],
            "m": 5.0,
            "region_size_level0": 256,
            "need_to_see": "Stop navigation and consolidate the gathered evidence.",
            "review_goal": "integrated_impression",
            "stage_gate": "end",
            "metadata": {"action": "stop", "region_size_level0": 256},
        }
    )
    return {"steps": normalized_steps}


def _build_cpathagent_qwen_observe_report_prompt(request):
    lines = [
        request["prompt"]["question"],
        "",
        "Summarize the following pathology reasoning records into a structured JSON report.",
        "Return JSON only with these keys:",
        "hierarchical_prediction, serrated_checklist, abnormal_crypt_checklist, conventional_adenoma_checklist, serrated_dysplasia_checklist, conventional_dysplasia_checklist, dysplasia_checklist, integrated_report",
        "",
        "Trace clusters:",
    ]
    for cluster in request["metadata"].get("trace_clusters", []):
        lines.append(
            "- cluster_id={cluster_id}, label={label}, s={priority}, desc={desc}".format(
                cluster_id=cluster.get("cluster_id"),
                label=cluster.get("l"),
                priority=cluster.get("s"),
                desc=cluster.get("desc", ""),
            )
        )
    lines.append("")
    lines.append("Observation records:")
    for record in request["metadata"].get("records", []):
        lines.append(
            "- step_id={step_id}, review_goal={review_goal}, stage_decision={stage_decision}, reasoning={reasoning}".format(
                step_id=record.get("step_id"),
                review_goal=record.get("metadata", {}).get("review_goal"),
                stage_decision=record.get("stage_decision"),
                reasoning=record.get("reasoning", ""),
            )
        )
    return "\n".join(lines)


def _parse_cpathagent_qwen_observe_report_output(text, request, bundle):
    parsed = _first_json_dict_from_text(text)
    if isinstance(parsed, dict) and "hierarchical_prediction" in parsed and "integrated_report" in parsed:
        hierarchy = parsed.get("hierarchical_prediction", {})
        if isinstance(hierarchy, str):
            nested = _first_json_dict_from_text(hierarchy)
            hierarchy = nested if isinstance(nested, dict) else {"raw_text": hierarchy}
        elif not isinstance(hierarchy, dict):
            hierarchy = {}
        parsed["hierarchical_prediction"] = hierarchy

        checklist_keys = (
            "serrated_checklist",
            "abnormal_crypt_checklist",
            "conventional_adenoma_checklist",
            "serrated_dysplasia_checklist",
            "conventional_dysplasia_checklist",
            "dysplasia_checklist",
        )
        for key in checklist_keys:
            value = parsed.get(key, [])
            if not isinstance(value, (dict, list)):
                parsed[key] = []

        integrated_report = parsed.get("integrated_report", "")
        if not isinstance(integrated_report, (dict, str)):
            parsed["integrated_report"] = str(integrated_report)
        return parsed
    raise FatalBackendExecutionError("local_cpathagent_qwen observe_report returned invalid JSON structure")


def _blank_hits(criteria):
    return {criterion: "not_assessed" for criterion in criteria}


TRACE_BACKGROUND_LABEL = "background_artifact_stroma"
TRACE_NORMAL_LABEL = "normal_mucosa"
TRACE_CONVENTIONAL_LABEL = "conventional_adenoma_like"
TRACE_INFLAMMATORY_LABEL = "inflammatory_polyp_like"
TRACE_SERRATED_LABEL = "ssl_suspicious_mucosa"
TRACE_LEGACY_SSL_HIGH_LABEL = "ssl_high_priority_mucosa"
TRACE_ALLOWED_LABELS = (
    TRACE_BACKGROUND_LABEL,
    TRACE_NORMAL_LABEL,
    TRACE_CONVENTIONAL_LABEL,
    TRACE_INFLAMMATORY_LABEL,
    TRACE_SERRATED_LABEL,
)


def _trace_branch_for_label(label):
    if label in (TRACE_SERRATED_LABEL, TRACE_LEGACY_SSL_HIGH_LABEL):
        return "serrated"
    if label == TRACE_CONVENTIONAL_LABEL:
        return "conventional"
    if label == TRACE_INFLAMMATORY_LABEL:
        return "inflammatory"
    if label == TRACE_NORMAL_LABEL:
        return "normal"
    return "background"


def _trace_review_stage_for_label(label):
    branch = _trace_branch_for_label(label)
    if branch == "serrated":
        return "serrated_screening"
    if branch == "conventional":
        return "conventional_adenoma_screening"
    if branch == "inflammatory":
        return "inflammatory_polyp_screening"
    return "mucosa_screening"


def _trace_label_metadata(label, priority, require_high_magnification, metadata=None):
    base_metadata = dict(metadata or {})
    branch = _trace_branch_for_label(label)
    if label == TRACE_CONVENTIONAL_LABEL:
        conventional_hint = base_metadata.get("conventional_subtype_hint") or "tubular_adenoma_like"
        inflammatory_hint = None
        serrated_hint = None
    elif label == TRACE_INFLAMMATORY_LABEL:
        conventional_hint = None
        inflammatory_hint = base_metadata.get("inflammatory_subtype_hint") or "inflammatory_polyp_like"
        serrated_hint = None
    elif label in (TRACE_SERRATED_LABEL, TRACE_LEGACY_SSL_HIGH_LABEL):
        conventional_hint = None
        inflammatory_hint = None
        serrated_hint = base_metadata.get("serrated_family_hint") or (
            "ssl_like" if int(priority) >= 5 else "equivocal_serrated"
        )
    else:
        conventional_hint = None
        inflammatory_hint = None
        serrated_hint = None
    serrated_dysplasia_suspected = bool(base_metadata.get("serrated_dysplasia_suspected", False))
    conventional_dysplasia_suspected = bool(base_metadata.get("conventional_dysplasia_suspected", False))
    if label in (TRACE_SERRATED_LABEL, TRACE_LEGACY_SSL_HIGH_LABEL) and int(priority) >= 5:
        serrated_dysplasia_suspected = True
    if label == TRACE_CONVENTIONAL_LABEL:
        conventional_dysplasia_suspected = True
    return {
        **base_metadata,
        "workflow_branch": branch,
        "region_semantic": label,
        "serrated_dysplasia_suspected": serrated_dysplasia_suspected,
        "conventional_dysplasia_suspected": conventional_dysplasia_suspected,
        "dysplasia_suspected": serrated_dysplasia_suspected or conventional_dysplasia_suspected,
        "conventional_subtype_hint": conventional_hint,
        "inflammatory_subtype_hint": inflammatory_hint,
        "serrated_family_hint": serrated_hint,
        "requires_high_magnification": bool(require_high_magnification),
    }


def _load_trace_grid_metadata(request):
    image_paths = request.get("images", [])
    if not image_paths:
        return None
    image_path = Path(image_paths[0])
    candidates = []
    if image_path.name.endswith("_grid.jpg"):
        candidates.append(image_path.with_suffix(".json"))
    if image_path.suffix:
        candidates.append(image_path.with_name("{0}_grid.json".format(image_path.stem)))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            try:
                metadata = read_json(candidate)
            except Exception:
                return None
            if (
                isinstance(metadata, dict)
                and metadata.get("thumbnail_mode") == "tissue_grid32x_svs"
                and isinstance(metadata.get("grid_cells"), list)
            ):
                return metadata
    return None


def _trace_prompt_header(request):
    header = str(request["prompt"]["question"]).strip()
    return header or (
        "You are the Trace Agent in a hierarchical pathology workflow. Analyze a gridded thumbnail image "
        "of a colorectal whole-slide image, group all selected grid cells into a complete pathology reading plan, "
        "and prioritize the regions most relevant to SSL versus conventional adenoma workflow routing."
    )


def _normalize_trace_priority(value, default_value=0):
    try:
        priority = int(value)
    except Exception:
        priority = int(default_value)
    return max(0, min(5, priority))


def _normalize_trace_label(region_semantic, name, description, severity_reasoning, require_high_magnification, diagnostic_priority):
    tokens = [
        str(region_semantic or ""),
        str(name or ""),
        str(description or ""),
        str(severity_reasoning or ""),
    ]
    haystack = " ".join(tokens).lower()
    exact_map = {
        "background": TRACE_BACKGROUND_LABEL,
        "artifact": TRACE_BACKGROUND_LABEL,
        "background_artifact_stroma": TRACE_BACKGROUND_LABEL,
        "normal": TRACE_NORMAL_LABEL,
        "normal_mucosa": TRACE_NORMAL_LABEL,
        "non_serrated_mucosa": TRACE_NORMAL_LABEL,
        "conventional_adenoma_like": TRACE_CONVENTIONAL_LABEL,
        "inflammatory_polyp_like": TRACE_INFLAMMATORY_LABEL,
        "serrated_suspicious_mucosa": TRACE_SERRATED_LABEL,
        "ssl_suspicious_mucosa": TRACE_SERRATED_LABEL,
        "ssl_high_priority_mucosa": TRACE_SERRATED_LABEL,
    }
    normalized_exact = exact_map.get(str(region_semantic or "").strip().lower())
    if normalized_exact:
        return normalized_exact
    if any(
        token in haystack
        for token in ("background", "artifact", "blank", "stroma", "muscle", "muscularis")
    ):
        return TRACE_BACKGROUND_LABEL
    if any(token in haystack for token in ("normal mucosa", "non-lesional", "non lesional", "benign mucosa")):
        return TRACE_NORMAL_LABEL
    if any(
        token in haystack
        for token in (
            "tubular adenoma",
            "tubulovillous adenoma",
            "conventional adenoma",
            "non-ssl adenoma",
            "non ssl adenoma",
            "non-ssl adenomatous",
            "non ssl adenomatous",
            "adenomatous",
            "villous",
            "tubular",
            "ta-like",
            "tva-like",
        )
    ):
        return TRACE_CONVENTIONAL_LABEL
    if any(
        token in haystack
        for token in (
            "inflammatory polyp",
            "inflammatory",
            "reactive polyp",
            "granulation",
            "prolapse-type",
        )
    ):
        return TRACE_INFLAMMATORY_LABEL
    if any(
        token in haystack
        for token in (
            "ssl",
            "sessile serrated",
            "highly suspicious",
            "high-priority ssl",
            "high priority ssl",
            "classic ssl",
            "high priority serrated",
        )
    ):
        return TRACE_SERRATED_LABEL
    if any(
        token in haystack
        for token in (
            "serrated",
            "hyperplastic",
            "tsa",
            "traditional serrated",
            "mucus cap",
            "mucous cap",
            "hp-like",
            "equivocal serrated",
            "suspicious",
        )
    ):
        return TRACE_SERRATED_LABEL
    diagnostic_priority = _normalize_trace_priority(diagnostic_priority, default_value=0)
    if require_high_magnification and diagnostic_priority >= 4:
        return TRACE_SERRATED_LABEL
    if diagnostic_priority >= 4:
        return TRACE_SERRATED_LABEL
    if require_high_magnification or diagnostic_priority >= 3:
        return TRACE_SERRATED_LABEL
    if diagnostic_priority <= 0:
        return TRACE_BACKGROUND_LABEL
    return TRACE_NORMAL_LABEL


def _normalize_patch_id_item(item):
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return None
    try:
        return (int(item[0]), int(item[1]))
    except Exception:
        return None


def _grid_cell_thumb_bbox(cell):
    return {
        "x1": int(cell["thumbnail_top_left_x"]),
        "y1": int(cell["thumbnail_top_left_y"]),
        "x2": int(cell["thumbnail_top_left_x"]) + int(cell["thumbnail_width"]),
        "y2": int(cell["thumbnail_top_left_y"]) + int(cell["thumbnail_height"]),
    }


def _grid_cell_level0_bbox(cell):
    return {
        "x1": int(cell["level0_top_left_x"]),
        "y1": int(cell["level0_top_left_y"]),
        "x2": int(cell["level0_top_left_x"]) + int(cell["level0_width"]),
        "y2": int(cell["level0_top_left_y"]) + int(cell["level0_height"]),
    }


def _grid_cell_patch_payload(cell, bbox):
    anchor_x = cell.get("level0_anchor_x")
    anchor_y = cell.get("level0_anchor_y")
    try:
        anchor_x = int(anchor_x) if anchor_x is not None else None
        anchor_y = int(anchor_y) if anchor_y is not None else None
    except Exception:
        anchor_x = None
        anchor_y = None
    return {
        "patch_id": [int(cell["row_id"]), int(cell["col_id"])],
        "row_id": int(cell["row_id"]),
        "col_id": int(cell["col_id"]),
        "x1": int(bbox["x1"]),
        "y1": int(bbox["y1"]),
        "x2": int(bbox["x2"]),
        "y2": int(bbox["y2"]),
        "tissue_coverage_ratio": round(float(cell.get("tissue_coverage_ratio", 0.0)), 4),
        "anchor_x": anchor_x,
        "anchor_y": anchor_y,
        "anchor_source": cell.get("anchor_source"),
        "center_in_tissue": bool(cell.get("center_in_tissue", False)),
        "centroid_in_tissue": bool(cell.get("centroid_in_tissue", False)),
    }


def _merge_bboxes(boxes):
    if not boxes:
        return None
    return {
        "x1": min(int(box["x1"]) for box in boxes),
        "y1": min(int(box["y1"]) for box in boxes),
        "x2": max(int(box["x2"]) for box in boxes),
        "y2": max(int(box["y2"]) for box in boxes),
    }


def _extract_answer_text(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(text or "").strip()


def _parse_number_list(text):
    if not text:
        return []
    values = []
    for token in re.findall(r"\d+", str(text)):
        value = int(token)
        if value not in values:
            values.append(value)
    return values


def _selected_patch_ids(grid_meta):
    patch_ids = []
    for cell in grid_meta.get("grid_cells", []):
        if not bool(cell.get("is_selected", False)):
            continue
        row_col = (int(cell["row_id"]), int(cell["col_id"]))
        if row_col not in patch_ids:
            patch_ids.append(row_col)
    return patch_ids


def _quadrant_patch_ids(grid_meta, quadrant_name):
    quadrant = str(quadrant_name or "").lower().strip()
    selected_ids = _selected_patch_ids(grid_meta)
    if not selected_ids:
        return []
    max_row = max(row for row, _ in selected_ids)
    max_col = max(col for _, col in selected_ids)
    row_mid = float(max_row) / 2.0
    col_mid = float(max_col) / 2.0
    matches = []
    for row_id, col_id in selected_ids:
        is_upper = row_id <= row_mid
        is_lower = row_id >= row_mid
        is_left = col_id <= col_mid
        is_right = col_id >= col_mid
        if quadrant == "upper left" and is_upper and is_left:
            matches.append((row_id, col_id))
        elif quadrant == "upper right" and is_upper and is_right:
            matches.append((row_id, col_id))
        elif quadrant == "lower left" and is_lower and is_left:
            matches.append((row_id, col_id))
        elif quadrant == "lower right" and is_lower and is_right:
            matches.append((row_id, col_id))
    return matches


def _selected_grid_cells(grid_meta):
    cells = []
    seen = set()
    for cell in grid_meta.get("grid_cells", []):
        if not isinstance(cell, dict) or not bool(cell.get("is_selected", False)):
            continue
        row_col = (int(cell["row_id"]), int(cell["col_id"]))
        if row_col in seen:
            continue
        seen.add(row_col)
        cells.append(cell)
    return cells


def _grid_lookup_from_metadata(grid_meta):
    return {
        (int(cell["row_id"]), int(cell["col_id"])): cell
        for cell in _selected_grid_cells(grid_meta)
    }


def _grid_cell_sequence_from_ids(id_list, grid_lookup):
    patch_ids_ordered = []
    patches_thumb = []
    patches_level0 = []
    selected_cells = []
    for row_id, col_id in id_list:
        cell = grid_lookup.get((int(row_id), int(col_id)))
        if not cell:
            continue
        thumb_bbox = _grid_cell_thumb_bbox(cell)
        level0_bbox = _grid_cell_level0_bbox(cell)
        patch_ids_ordered.append([int(row_id), int(col_id)])
        patches_thumb.append(_grid_cell_patch_payload(cell, thumb_bbox))
        patches_level0.append(_grid_cell_patch_payload(cell, level0_bbox))
        selected_cells.append(cell)
    return patch_ids_ordered, patches_thumb, patches_level0, selected_cells


def _representative_centers_level0(patches_level0):
    centers = []
    for patch in patches_level0:
        anchor_x = patch.get("anchor_x")
        anchor_y = patch.get("anchor_y")
        if anchor_x is not None and anchor_y is not None:
            x_value = int(anchor_x)
            y_value = int(anchor_y)
        else:
            x_value = int(round((int(patch["x1"]) + int(patch["x2"])) / 2.0))
            y_value = int(round((int(patch["y1"]) + int(patch["y2"])) / 2.0))
        centers.append(
            {
                "patch_id": list(patch.get("patch_id", [])),
                "x": x_value,
                "y": y_value,
                "anchor_source": patch.get("anchor_source"),
            }
        )
    return centers


def _build_trace_cluster_payload(
    cluster_id,
    label,
    priority,
    require_high_magnification,
    desc,
    evidence,
    patch_ids_ordered,
    patches_thumb,
    patches_level0,
    selected_cells,
    grid_meta,
    metadata,
):
    group_bbox_thumb = _merge_bboxes(patches_thumb) or {}
    group_bbox_level0 = _merge_bboxes(patches_level0) or {}
    label = _normalize_trace_label(label, label, desc, metadata.get("severity_reasoning", ""), require_high_magnification, priority)
    priority = _normalize_trace_priority(priority, default_value=0)
    if label == TRACE_BACKGROUND_LABEL:
        priority = 0
        require_high_magnification = False
    branch = _trace_branch_for_label(label)
    if label == TRACE_CONVENTIONAL_LABEL:
        require_high_magnification = True
    crypt_disorder_risk = min(5, max(priority, 0)) if branch == "serrated" else 0
    normalized_evidence = [str(value) for value in evidence if str(value).strip()]
    tissue_values = [float(cell.get("tissue_coverage_ratio", 0.0)) for cell in selected_cells]
    normalized_metadata = _trace_label_metadata(label, priority, require_high_magnification, metadata)
    return {
        "cluster_id": cluster_id,
        "cluster_bbox_thumb": dict(group_bbox_thumb),
        "cluster_bbox_level0": dict(group_bbox_level0),
        "regions_thumb": [
            {"x1": int(patch["x1"]), "y1": int(patch["y1"]), "x2": int(patch["x2"]), "y2": int(patch["y2"])}
            for patch in patches_thumb
        ],
        "regions_level0": [
            {"x1": int(patch["x1"]), "y1": int(patch["y1"]), "x2": int(patch["x2"]), "y2": int(patch["y2"])}
            for patch in patches_level0
        ],
        "group_bbox_thumb": dict(group_bbox_thumb),
        "group_bbox_level0": dict(group_bbox_level0),
        "patch_ids_ordered": [list(item) for item in patch_ids_ordered],
        "patches_thumb": list(patches_thumb),
        "patches_level0": list(patches_level0),
        "l": label,
        "s": priority,
        "d": bool(require_high_magnification),
        "review_stage": _trace_review_stage_for_label(label),
        "crypt_disorder_risk": crypt_disorder_risk,
        "dysplasia_review_needed": bool(
            normalized_metadata.get("serrated_dysplasia_suspected")
            or normalized_metadata.get("conventional_dysplasia_suspected")
        ),
        "desc": desc,
        "evidence": normalized_evidence,
        "metadata": {
            **normalized_metadata,
            "grid_id_list": [list(item) for item in patch_ids_ordered],
            "grid_cell_count": len(patch_ids_ordered),
            "grid_rows": int(grid_meta.get("grid_rows", 0) or 0),
            "grid_cols": int(grid_meta.get("grid_cols", 0) or 0),
            "grid_thumbnail_mode": grid_meta.get("thumbnail_mode"),
            "tissue_coverage_mean": round(sum(tissue_values) / float(max(1, len(tissue_values))), 4),
            "representative_centers_level0": _representative_centers_level0(patches_level0),
        },
    }


def _trace_observation_points_from_text(text):
    lower_text = str(text or "").lower()
    points = []
    keyword_map = [
        ("pale", "pale surface appearance"),
        ("mucus-rich", "mucus-rich surface pattern"),
        ("mucus", "mucus-rich surface pattern"),
        ("irregular contour", "mucosal contour irregularity"),
        ("crypt crowding", "crypt crowding suggestive of serration"),
        ("serrated edge", "lesion edge suspicious for serration"),
        ("lesion edge", "lesion edge localization"),
        ("higher magnification", "higher-magnification follow-up"),
        ("high-magnification", "higher-magnification follow-up"),
    ]
    for needle, label in keyword_map:
        if needle in lower_text and label not in points:
            points.append(label)
    return points


def _trace_prose_priority(text):
    lower_text = str(text or "").lower()
    if "highest diagnostic significance" in lower_text or "highest likelihood" in lower_text:
        return 5
    if "not yet suspicious" in lower_text or "equivocal" in lower_text:
        return 3
    return 4


def _trace_prose_requires_high_magnification(text):
    lower_text = str(text or "").lower()
    if "no high-magnification" in lower_text or "no high magnification" in lower_text:
        return False
    return any(
        phrase in lower_text
        for phrase in (
            "warranting high-magnification review",
            "warrant high-magnification review",
            "warrant review at higher magnification",
            "require higher magnification",
            "requires higher magnification",
            "review at higher magnification",
            "closer inspection",
            "closer review",
            "further evaluation",
        )
    )


def _build_groups_from_trace_prose(text, request):
    grid_meta = _load_trace_grid_metadata(request)
    if not grid_meta:
        return None
    answer_text = _extract_answer_text(text)
    if not answer_text:
        return None
    lower_text = answer_text.lower()
    patch_ids = []

    for row_id, col_id in re.findall(r"[\(\[]\s*(\d+)\s*,\s*(\d+)\s*[\)\]]", answer_text):
        row_col = (int(row_id), int(col_id))
        if row_col not in patch_ids:
            patch_ids.append(row_col)

    for row_id, col_id in re.findall(r"row\s+(\d+)\s*,?\s*col(?:umn)?\s+(\d+)", lower_text):
        row_col = (int(row_id), int(col_id))
        if row_col not in patch_ids:
            patch_ids.append(row_col)

    row_col_chunks = re.findall(
        r"rows?\s+([0-9,\sand]+?)\s*,?\s*columns?\s+([0-9,\sand]+?)(?=[\.;]|, and row| and row|$)",
        lower_text,
        flags=re.IGNORECASE,
    )
    for row_chunk, col_chunk in row_col_chunks:
        row_ids = _parse_number_list(row_chunk)
        col_ids = _parse_number_list(col_chunk)
        for row_id in row_ids:
            for col_id in col_ids:
                row_col = (row_id, col_id)
                if row_col not in patch_ids:
                    patch_ids.append(row_col)

    if not patch_ids:
        for quadrant in ("upper left", "upper right", "lower left", "lower right"):
            if quadrant in lower_text:
                for row_col in _quadrant_patch_ids(grid_meta, quadrant):
                    if row_col not in patch_ids:
                        patch_ids.append(row_col)

    if not patch_ids and (
        "cells listed in the _grid_ object" in lower_text
        or "single coherent region" in lower_text
        or "cells listed in the grid object" in lower_text
    ):
        patch_ids = _selected_patch_ids(grid_meta)

    if not patch_ids and (
        "prioritized groups are those" in lower_text
        or "these are marked for further review" in lower_text
        or "marked for further review" in lower_text
    ):
        patch_ids = _selected_patch_ids(grid_meta)

    selected_lookup = set(_selected_patch_ids(grid_meta))
    normalized_patch_ids = []
    for row_col in patch_ids:
        if row_col in selected_lookup and row_col not in normalized_patch_ids:
            normalized_patch_ids.append(row_col)
    if not normalized_patch_ids:
        return None
    selected_patch_order = _selected_patch_ids(grid_meta)
    normalized_patch_ids = [row_col for row_col in selected_patch_order if row_col in normalized_patch_ids]

    observation_points = _trace_observation_points_from_text(answer_text)
    require_high_magnification = _trace_prose_requires_high_magnification(answer_text)
    diagnostic_priority = _trace_prose_priority(answer_text)
    return {
        "groups": [
            {
                "name": "serrated-suspicious mucosa",
                "region_semantic": TRACE_SERRATED_LABEL,
                "description": answer_text,
                "id_list": [[row_id, col_id] for row_id, col_id in normalized_patch_ids],
                "require_high_magnification": require_high_magnification,
                "severity_reasoning": answer_text,
                "diagnostic_priority": diagnostic_priority,
                "observation_points": observation_points,
            }
        ]
    }


def _sort_trace_clusters(output_clusters):
    indexed = list(enumerate(output_clusters))
    indexed.sort(key=lambda item: (-int(item[1].get("s", 0)), item[0]))
    return [item[1] for item in indexed]


def _jsonable_trace_item(item):
    if isinstance(item, (list, tuple)):
        return [value for value in item]
    if isinstance(item, dict):
        return dict(item)
    return str(item)


def _append_unique_patch_id(rows, row_col):
    payload = [int(row_col[0]), int(row_col[1])]
    if payload not in rows:
        rows.append(payload)


def _selected_patch_vocab_text(grid_meta):
    return json.dumps(
        [[int(row_id), int(col_id)] for row_id, col_id in _selected_patch_ids(grid_meta)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_trace_json_payload(text):
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        json_blob = _extract_first_json_object(text)
        if json_blob:
            try:
                parsed = json.loads(json_blob)
            except Exception:
                parsed = None
    return parsed


def _normalize_trace_assignment_name(name):
    value = re.sub(r"\s+", " ", str(name or "").strip())
    return value or "normal mucosa"


def _extract_trace_groups_payload(text, request):
    parsed = _parse_trace_json_payload(text)

    if isinstance(parsed, dict) and isinstance(parsed.get("groups"), list):
        return {"parsed": parsed, "groups_payload": parsed, "groups_source": "json", "parse_failure": False}

    prose_groups = _build_groups_from_trace_prose(text, request)
    if isinstance(prose_groups, dict) and isinstance(prose_groups.get("groups"), list):
        return {"parsed": parsed, "groups_payload": prose_groups, "groups_source": "prose", "parse_failure": False}

    return {"parsed": parsed, "groups_payload": {"groups": []}, "groups_source": "parse_failure", "parse_failure": True}


def _extract_trace_patch_assignments_payload(text, request):
    parsed = _parse_trace_json_payload(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("patches"), list):
        return {
            "parsed": parsed,
            "assignment_payload": parsed,
            "groups_payload": None,
            "trace_schema": "patch_assignments",
            "groups_source": "patch_assignments",
            "parse_failure": False,
        }
    groups_extracted = _extract_trace_groups_payload(text, request)
    return {
        "parsed": groups_extracted.get("parsed"),
        "assignment_payload": {"patches": []},
        "groups_payload": groups_extracted.get("groups_payload", {"groups": []}),
        "trace_schema": "groups_legacy" if groups_extracted.get("groups_source") == "json" else groups_extracted.get("groups_source", "parse_failure"),
        "groups_source": groups_extracted.get("groups_source", "parse_failure"),
        "parse_failure": bool(groups_extracted.get("parse_failure", False)),
    }


def _validate_trace_groups_payload(groups_payload, request, groups_source="json", parse_failure=False):
    grid_meta = _load_trace_grid_metadata(request)
    if not grid_meta:
        return None
    groups = groups_payload.get("groups", [])
    if not isinstance(groups, list):
        return None
    selected_patch_ids = _selected_patch_ids(grid_meta)
    selected_lookup = set(selected_patch_ids)
    covered_ids = set()
    duplicate_patch_ids = []
    ignored_patch_ids = []
    unexpected_patch_ids = []
    empty_groups = []
    group_records = []
    for index, group in enumerate(groups):
        record = {
            "group_index": index,
            "group": group if isinstance(group, dict) else {},
            "normalized_ids": [],
            "duplicate_patch_ids": [],
            "ignored_patch_ids": [],
            "unexpected_patch_ids": [],
            "is_empty": False,
        }
        if not isinstance(group, dict):
            record["is_empty"] = True
            empty_groups.append(index)
            group_records.append(record)
            continue
        id_list = group.get("id_list", [])
        if not isinstance(id_list, list):
            record["is_empty"] = True
            empty_groups.append(index)
            group_records.append(record)
            continue
        local_seen = set()
        for item in id_list:
            row_col = _normalize_patch_id_item(item)
            if row_col is None:
                jsonable_item = _jsonable_trace_item(item)
                if jsonable_item not in record["ignored_patch_ids"]:
                    record["ignored_patch_ids"].append(jsonable_item)
                if jsonable_item not in ignored_patch_ids:
                    ignored_patch_ids.append(jsonable_item)
                continue
            if row_col not in selected_lookup:
                _append_unique_patch_id(record["ignored_patch_ids"], row_col)
                _append_unique_patch_id(ignored_patch_ids, row_col)
                _append_unique_patch_id(record["unexpected_patch_ids"], row_col)
                _append_unique_patch_id(unexpected_patch_ids, row_col)
                continue
            if row_col in local_seen or row_col in covered_ids:
                _append_unique_patch_id(record["duplicate_patch_ids"], row_col)
                _append_unique_patch_id(duplicate_patch_ids, row_col)
                continue
            local_seen.add(row_col)
            covered_ids.add(row_col)
            record["normalized_ids"].append((int(row_col[0]), int(row_col[1])))
        if not record["normalized_ids"]:
            record["is_empty"] = True
            if index not in empty_groups:
                empty_groups.append(index)
        group_records.append(record)
    missing_patch_ids = [[int(row_id), int(col_id)] for row_id, col_id in selected_patch_ids if (row_id, col_id) not in covered_ids]
    covered_patch_count = len(covered_ids)
    coverage_ok = not parse_failure and not (
        missing_patch_ids or duplicate_patch_ids or ignored_patch_ids or unexpected_patch_ids or empty_groups
    )
    return {
        "coverage_ok": coverage_ok,
        "missing_patch_ids": missing_patch_ids,
        "duplicate_patch_ids": duplicate_patch_ids,
        "ignored_patch_ids": ignored_patch_ids,
        "unexpected_patch_ids": unexpected_patch_ids,
        "empty_groups": list(empty_groups),
        "selected_patch_count": len(selected_patch_ids),
        "covered_patch_count": covered_patch_count,
        "groups_source": groups_source,
        "trace_schema": "groups_legacy" if groups_source == "json" else groups_source,
        "parse_failure": bool(parse_failure),
        "group_records": group_records,
    }


def _validate_trace_patch_assignments_payload(assignment_payload, request, groups_source="patch_assignments", parse_failure=False):
    grid_meta = _load_trace_grid_metadata(request)
    if not grid_meta:
        return None
    assignments = assignment_payload.get("patches", [])
    if not isinstance(assignments, list):
        assignments = []
    selected_patch_ids = _selected_patch_ids(grid_meta)
    selected_lookup = set(selected_patch_ids)
    covered_ids = set()
    duplicate_patch_ids = []
    ignored_patch_ids = []
    unexpected_patch_ids = []
    empty_assignments = []
    assignment_records = []
    for index, assignment in enumerate(assignments):
        record = {
            "assignment_index": index,
            "assignment": assignment if isinstance(assignment, dict) else {},
            "normalized_id": None,
            "duplicate_patch_ids": [],
            "ignored_patch_ids": [],
            "unexpected_patch_ids": [],
            "is_empty": False,
        }
        if not isinstance(assignment, dict):
            record["is_empty"] = True
            empty_assignments.append(index)
            assignment_records.append(record)
            continue
        row_col = _normalize_patch_id_item(assignment.get("patch_id"))
        if row_col is None:
            jsonable_item = _jsonable_trace_item(assignment.get("patch_id"))
            record["ignored_patch_ids"].append(jsonable_item)
            if jsonable_item not in ignored_patch_ids:
                ignored_patch_ids.append(jsonable_item)
            record["is_empty"] = True
            empty_assignments.append(index)
            assignment_records.append(record)
            continue
        if row_col not in selected_lookup:
            _append_unique_patch_id(record["ignored_patch_ids"], row_col)
            _append_unique_patch_id(ignored_patch_ids, row_col)
            _append_unique_patch_id(record["unexpected_patch_ids"], row_col)
            _append_unique_patch_id(unexpected_patch_ids, row_col)
            record["is_empty"] = True
            empty_assignments.append(index)
            assignment_records.append(record)
            continue
        if row_col in covered_ids:
            _append_unique_patch_id(record["duplicate_patch_ids"], row_col)
            _append_unique_patch_id(duplicate_patch_ids, row_col)
            record["is_empty"] = True
            empty_assignments.append(index)
            assignment_records.append(record)
            continue
        covered_ids.add(row_col)
        record["normalized_id"] = (int(row_col[0]), int(row_col[1]))
        assignment_records.append(record)
    missing_patch_ids = [[int(row_id), int(col_id)] for row_id, col_id in selected_patch_ids if (row_id, col_id) not in covered_ids]
    covered_patch_count = len(covered_ids)
    coverage_ok = not parse_failure and not (
        missing_patch_ids or duplicate_patch_ids or ignored_patch_ids or unexpected_patch_ids or empty_assignments
    )
    return {
        "coverage_ok": coverage_ok,
        "missing_patch_ids": missing_patch_ids,
        "duplicate_patch_ids": duplicate_patch_ids,
        "ignored_patch_ids": ignored_patch_ids,
        "unexpected_patch_ids": unexpected_patch_ids,
        "empty_groups": list(empty_assignments),
        "empty_assignments": list(empty_assignments),
        "selected_patch_count": len(selected_patch_ids),
        "covered_patch_count": covered_patch_count,
        "assignment_count": len(assignments),
        "groups_source": groups_source,
        "trace_schema": "patch_assignments",
        "parse_failure": bool(parse_failure),
        "assignment_records": assignment_records,
    }


def _validate_trace_extracted_payload(extracted, request):
    if extracted.get("trace_schema") == "patch_assignments":
        return _validate_trace_patch_assignments_payload(
            extracted.get("assignment_payload", {"patches": []}),
            request,
            groups_source=extracted.get("groups_source", "patch_assignments"),
            parse_failure=extracted.get("parse_failure", False),
        )
    return _validate_trace_groups_payload(
        extracted.get("groups_payload", {"groups": []}),
        request,
        groups_source=extracted.get("groups_source", "parse_failure"),
        parse_failure=extracted.get("parse_failure", False),
    )


def _build_trace_coverage_summary(
    validation,
    coverage_repaired_by_retry=False,
    coverage_repair_applied=False,
    coverage_repair_stage="no_retry_needed",
    source_attempt_index=0,
    retry_attempted=False,
):
    if not validation:
        return {}
    return {
        "coverage_ok": bool(validation.get("coverage_ok", False)),
        "missing_patch_ids": [list(item) for item in validation.get("missing_patch_ids", [])],
        "duplicate_patch_ids": [list(item) for item in validation.get("duplicate_patch_ids", [])],
        "ignored_patch_ids": list(validation.get("ignored_patch_ids", [])),
        "unexpected_patch_ids": [list(item) for item in validation.get("unexpected_patch_ids", [])],
        "empty_groups": list(validation.get("empty_groups", [])),
        "empty_assignments": list(validation.get("empty_assignments", [])),
        "selected_patch_count": int(validation.get("selected_patch_count", 0)),
        "covered_patch_count": int(validation.get("covered_patch_count", 0)),
        "assignment_count": int(validation.get("assignment_count", 0)),
        "groups_source": validation.get("groups_source", "unknown"),
        "trace_schema": validation.get("trace_schema", validation.get("groups_source", "unknown")),
        "parse_failure": bool(validation.get("parse_failure", False)),
        "coverage_repaired_by_retry": bool(coverage_repaired_by_retry),
        "coverage_repair_applied": bool(coverage_repair_applied),
        "coverage_repair_stage": str(coverage_repair_stage or "no_retry_needed"),
        "source_attempt_index": int(source_attempt_index),
        "retry_attempted": bool(retry_attempted),
    }


def _build_trace_clusters_from_groups(
    parsed,
    request,
    validation=None,
    apply_fallback=True,
    coverage_repaired_by_retry=False,
    coverage_repair_stage="no_retry_needed",
    source_attempt_index=0,
    retry_attempted=False,
):
    grid_meta = _load_trace_grid_metadata(request)
    if not grid_meta:
        return None
    groups = parsed.get("groups", [])
    if not isinstance(groups, list):
        return None
    validation = validation or _validate_trace_groups_payload(parsed, request)
    if validation is None:
        return None
    grid_lookup = _grid_lookup_from_metadata(grid_meta)
    output_clusters = []
    for index, group_record in enumerate(validation.get("group_records", [])):
        group = group_record.get("group", {})
        normalized_ids = list(group_record.get("normalized_ids", []))
        if not normalized_ids:
            continue
        patch_ids_ordered, patches_thumb, patches_level0, selected_cells = _grid_cell_sequence_from_ids(normalized_ids, grid_lookup)
        if not patch_ids_ordered:
            continue
        name = str(group.get("name", "normal mucosa"))
        region_semantic = group.get("region_semantic", "")
        description = str(group.get("description", "")).strip()
        severity_reasoning = str(group.get("severity_reasoning", "")).strip()
        require_high_magnification = bool(group.get("require_high_magnification", False))
        diagnostic_priority = _normalize_trace_priority(
            group.get("diagnostic_priority", 0),
            default_value=4 if require_high_magnification else 1,
        )
        label = _normalize_trace_label(
            region_semantic=region_semantic,
            name=name,
            description=description,
            severity_reasoning=severity_reasoning,
            require_high_magnification=require_high_magnification,
            diagnostic_priority=diagnostic_priority,
        )
        observation_points = group.get("observation_points", [])
        if not isinstance(observation_points, list):
            observation_points = [str(observation_points)]
        extra_metadata = {}
        for key in (
            "serrated_dysplasia_suspected",
            "conventional_dysplasia_suspected",
            "conventional_subtype_hint",
            "inflammatory_subtype_hint",
            "serrated_family_hint",
        ):
            if key in group:
                extra_metadata[key] = group.get(key)
        desc = description or severity_reasoning or name
        output_clusters.append(
            _build_trace_cluster_payload(
                cluster_id="grid_group_{0:02d}".format(index),
                label=label,
                priority=diagnostic_priority,
                require_high_magnification=require_high_magnification,
                desc=desc,
                evidence=observation_points or ([severity_reasoning] if severity_reasoning else []),
                patch_ids_ordered=patch_ids_ordered,
                patches_thumb=patches_thumb,
                patches_level0=patches_level0,
                selected_cells=selected_cells,
                grid_meta=grid_meta,
                metadata={
                    **extra_metadata,
                    "source": "patho_r1_trace_groups",
                    "group_name": name,
                    "region_semantic": label,
                    "severity_reasoning": severity_reasoning,
                    "group_output_index": index,
                    "dropped_duplicate_patch_ids": list(group_record.get("duplicate_patch_ids", [])),
                    "ignored_patch_ids": list(group_record.get("ignored_patch_ids", [])),
                    "unexpected_patch_ids": list(group_record.get("unexpected_patch_ids", [])),
                    "coverage_ok": bool(validation.get("coverage_ok", False)),
                    "coverage_repaired_by_retry": bool(coverage_repaired_by_retry),
                    "coverage_repair_applied": False,
                    "coverage_repair_stage": str(coverage_repair_stage or "no_retry_needed"),
                    "source_attempt_index": int(source_attempt_index),
                    "retry_attempted": bool(retry_attempted),
                },
            )
        )
    missing_patch_ids = [tuple(item) for item in validation.get("missing_patch_ids", [])]
    coverage_repair_applied = bool(apply_fallback and missing_patch_ids)
    if coverage_repair_applied:
        patch_ids_ordered, patches_thumb, patches_level0, selected_cells = _grid_cell_sequence_from_ids(missing_patch_ids, grid_lookup)
        output_clusters.append(
            _build_trace_cluster_payload(
                cluster_id="grid_group_fallback",
                label=TRACE_BACKGROUND_LABEL,
                priority=0,
                require_high_magnification=False,
                desc="Selected patches omitted by the trace response were repaired into a fallback discard group.",
                evidence=["trace_output_missing_patch_repair"],
                patch_ids_ordered=patch_ids_ordered,
                patches_thumb=patches_thumb,
                patches_level0=patches_level0,
                selected_cells=selected_cells,
                grid_meta=grid_meta,
                metadata={
                    "source": "trace_output_missing_patch_repair",
                    "group_name": "fallback discard group",
                    "region_semantic": TRACE_BACKGROUND_LABEL,
                    "severity_reasoning": "Missing selected patches were automatically covered to preserve complete grid coverage.",
                    "group_output_index": len(groups),
                    "missing_patch_ids": [list(item) for item in missing_patch_ids],
                    "dropped_duplicate_patch_ids": [list(item) for item in validation.get("duplicate_patch_ids", [])],
                    "ignored_patch_ids": list(validation.get("ignored_patch_ids", [])),
                    "unexpected_patch_ids": [list(item) for item in validation.get("unexpected_patch_ids", [])],
                    "coverage_ok": False,
                    "coverage_repaired_by_retry": bool(coverage_repaired_by_retry),
                    "coverage_repair_applied": True,
                    "coverage_repair_stage": str(coverage_repair_stage or "no_retry_needed"),
                    "source_attempt_index": int(source_attempt_index),
                    "retry_attempted": bool(retry_attempted),
                },
            )
        )
    return {
        "clusters": _sort_trace_clusters(output_clusters),
        "coverage_summary": _build_trace_coverage_summary(
            validation,
            coverage_repaired_by_retry=coverage_repaired_by_retry,
            coverage_repair_applied=coverage_repair_applied,
            coverage_repair_stage=coverage_repair_stage,
            source_attempt_index=source_attempt_index,
            retry_attempted=retry_attempted,
        ),
    }


def _build_trace_clusters_from_patch_assignments(
    assignment_payload,
    request,
    validation=None,
    apply_fallback=True,
    coverage_repaired_by_retry=False,
    coverage_repair_stage="no_retry_needed",
    source_attempt_index=0,
    retry_attempted=False,
):
    grid_meta = _load_trace_grid_metadata(request)
    if not grid_meta:
        return None
    assignments = assignment_payload.get("patches", [])
    if not isinstance(assignments, list):
        return None
    validation = validation or _validate_trace_patch_assignments_payload(assignment_payload, request)
    if validation is None:
        return None
    grid_lookup = _grid_lookup_from_metadata(grid_meta)
    selected_order = {row_col: index for index, row_col in enumerate(_selected_patch_ids(grid_meta))}
    grouped = []
    grouped_lookup = {}
    for record in validation.get("assignment_records", []):
        row_col = record.get("normalized_id")
        if row_col is None:
            continue
        assignment = record.get("assignment", {})
        name = _normalize_trace_assignment_name(assignment.get("name", "normal mucosa"))
        description = str(assignment.get("description", "")).strip()
        severity_reasoning = str(assignment.get("severity_reasoning", "")).strip()
        require_high_magnification = bool(assignment.get("require_high_magnification", False))
        diagnostic_priority = _normalize_trace_priority(
            assignment.get("diagnostic_priority", 0),
            default_value=4 if require_high_magnification else 1,
        )
        label = _normalize_trace_label(
            region_semantic=assignment.get("region_semantic", ""),
            name=name,
            description=description,
            severity_reasoning=severity_reasoning,
            require_high_magnification=require_high_magnification,
            diagnostic_priority=diagnostic_priority,
        )
        key = (label, bool(require_high_magnification), int(diagnostic_priority), name.lower())
        if key not in grouped_lookup:
            grouped_lookup[key] = {
                "label": label,
                "name": name,
                "require_high_magnification": bool(require_high_magnification),
                "diagnostic_priority": int(diagnostic_priority),
                "records": [],
                "patch_ids": [],
                "descriptions": [],
                "severity_reasoning": [],
                "evidence": [],
                "extra_metadata": {},
            }
            grouped.append(grouped_lookup[key])
        group = grouped_lookup[key]
        group["records"].append(record)
        group["patch_ids"].append((int(row_col[0]), int(row_col[1])))
        if description and description not in group["descriptions"]:
            group["descriptions"].append(description)
        if severity_reasoning and severity_reasoning not in group["severity_reasoning"]:
            group["severity_reasoning"].append(severity_reasoning)
        observation_points = assignment.get("observation_points", [])
        if not isinstance(observation_points, list):
            observation_points = [str(observation_points)]
        for item in observation_points:
            value = str(item).strip()
            if value and value not in group["evidence"]:
                group["evidence"].append(value)
        for extra_key in (
            "serrated_dysplasia_suspected",
            "conventional_dysplasia_suspected",
            "conventional_subtype_hint",
            "inflammatory_subtype_hint",
            "serrated_family_hint",
        ):
            if extra_key in assignment and extra_key not in group["extra_metadata"]:
                group["extra_metadata"][extra_key] = assignment.get(extra_key)
    output_clusters = []
    cluster_index = 0
    for group in grouped:
        normalized_group_ids = sorted(group["patch_ids"], key=lambda row_col: selected_order.get(tuple(row_col), 10**9))
        if group["label"] in LESION_TRACE_LABELS:
            components = connected_components_for_patch_ids(normalized_group_ids)
            split_by_connectivity = len(components) > 1
        else:
            components = [normalized_group_ids]
            split_by_connectivity = False
        component_count = len(components)
        semantic_merge_key = "|".join(
            [
                str(group["label"]),
                str(bool(group["require_high_magnification"])),
                str(int(group["diagnostic_priority"])),
                str(group["name"]).lower(),
            ]
        )
        for component_index, component_ids in enumerate(components):
            normalized_ids = sorted(component_ids, key=lambda row_col: selected_order.get(tuple(row_col), 10**9))
            patch_ids_ordered, patches_thumb, patches_level0, selected_cells = _grid_cell_sequence_from_ids(normalized_ids, grid_lookup)
            if not patch_ids_ordered:
                continue
            component_patch_lookup = set(tuple(item) for item in normalized_ids)
            component_records = [
                record for record in group["records"] if tuple(record.get("normalized_id") or ()) in component_patch_lookup
            ]
            severity_reasoning = "; ".join(group["severity_reasoning"][:3])
            desc = (group["descriptions"][0] if group["descriptions"] else "") or severity_reasoning or group["name"]
            evidence = group["evidence"] or ([severity_reasoning] if severity_reasoning else [])
            output_clusters.append(
                _build_trace_cluster_payload(
                    cluster_id="grid_group_{0:02d}".format(cluster_index),
                    label=group["label"],
                    priority=group["diagnostic_priority"],
                    require_high_magnification=group["require_high_magnification"],
                    desc=desc,
                    evidence=evidence,
                    patch_ids_ordered=patch_ids_ordered,
                    patches_thumb=patches_thumb,
                    patches_level0=patches_level0,
                    selected_cells=selected_cells,
                    grid_meta=grid_meta,
                    metadata={
                        **group["extra_metadata"],
                        "source": "pathreasoner_patch_assignments",
                        "patch_assignment_schema": True,
                        "cluster_aggregation_mode": "system_from_patch_assignments",
                        "group_name": group["name"],
                        "region_semantic": group["label"],
                        "severity_reasoning": severity_reasoning,
                        "assignment_count": len(component_records),
                        "assignment_output_indices": [
                            int(record.get("assignment_index", 0)) for record in component_records
                        ],
                        "semantic_merge_key": semantic_merge_key,
                        "spatial_component_id": int(component_index),
                        "spatial_component_count": int(component_count),
                        "is_spatially_contiguous": True,
                        "split_by_connectivity": bool(split_by_connectivity),
                        "coverage_ok": bool(validation.get("coverage_ok", False)),
                        "coverage_repaired_by_retry": bool(coverage_repaired_by_retry),
                        "coverage_repair_applied": False,
                        "coverage_repair_stage": str(coverage_repair_stage or "no_retry_needed"),
                        "source_attempt_index": int(source_attempt_index),
                        "retry_attempted": bool(retry_attempted),
                    },
                )
            )
            cluster_index += 1
    missing_patch_ids = [tuple(item) for item in validation.get("missing_patch_ids", [])]
    coverage_repair_applied = bool(apply_fallback and missing_patch_ids)
    if coverage_repair_applied:
        patch_ids_ordered, patches_thumb, patches_level0, selected_cells = _grid_cell_sequence_from_ids(missing_patch_ids, grid_lookup)
        output_clusters.append(
            _build_trace_cluster_payload(
                cluster_id="grid_group_fallback",
                label=TRACE_BACKGROUND_LABEL,
                priority=0,
                require_high_magnification=False,
                desc="Selected patches omitted by the trace response were repaired into a fallback discard group.",
                evidence=["trace_output_missing_patch_repair"],
                patch_ids_ordered=patch_ids_ordered,
                patches_thumb=patches_thumb,
                patches_level0=patches_level0,
                selected_cells=selected_cells,
                grid_meta=grid_meta,
                metadata={
                    "source": "trace_output_missing_patch_repair",
                    "patch_assignment_schema": True,
                    "cluster_aggregation_mode": "system_from_patch_assignments",
                    "group_name": "fallback discard group",
                    "region_semantic": TRACE_BACKGROUND_LABEL,
                    "severity_reasoning": "Missing selected patches were automatically covered to preserve complete grid coverage.",
                    "group_output_index": len(grouped),
                    "missing_patch_ids": [list(item) for item in missing_patch_ids],
                    "dropped_duplicate_patch_ids": [list(item) for item in validation.get("duplicate_patch_ids", [])],
                    "ignored_patch_ids": list(validation.get("ignored_patch_ids", [])),
                    "unexpected_patch_ids": [list(item) for item in validation.get("unexpected_patch_ids", [])],
                    "coverage_ok": False,
                    "coverage_repaired_by_retry": bool(coverage_repaired_by_retry),
                    "coverage_repair_applied": True,
                    "coverage_repair_stage": str(coverage_repair_stage or "no_retry_needed"),
                    "source_attempt_index": int(source_attempt_index),
                    "retry_attempted": bool(retry_attempted),
                },
            )
        )
    coverage_summary = _build_trace_coverage_summary(
        validation,
        coverage_repaired_by_retry=coverage_repaired_by_retry,
        coverage_repair_applied=coverage_repair_applied,
        coverage_repair_stage=coverage_repair_stage,
        source_attempt_index=source_attempt_index,
        retry_attempted=retry_attempted,
    )
    coverage_summary.update(
        {
            "final_trace_schema": "patch_assignments",
            "cluster_aggregation_mode": "system_from_patch_assignments",
        }
    )
    sorted_clusters = _sort_trace_clusters(output_clusters)
    return {
        "clusters": sorted_clusters,
        "all_clusters": sorted_clusters,
        "patch_assignments": assignment_payload,
        "coverage_summary": coverage_summary,
    }


def _build_trace_clusters_from_extracted_payload(
    extracted,
    request,
    validation=None,
    apply_fallback=True,
    coverage_repaired_by_retry=False,
    coverage_repair_stage="no_retry_needed",
    source_attempt_index=0,
    retry_attempted=False,
):
    if extracted.get("trace_schema") == "patch_assignments":
        return _build_trace_clusters_from_patch_assignments(
            extracted.get("assignment_payload", {"patches": []}),
            request,
            validation=validation,
            apply_fallback=apply_fallback,
            coverage_repaired_by_retry=coverage_repaired_by_retry,
            coverage_repair_stage=coverage_repair_stage,
            source_attempt_index=source_attempt_index,
            retry_attempted=retry_attempted,
        )
    output = _build_trace_clusters_from_groups(
        extracted.get("groups_payload", {"groups": []}),
        request,
        validation=validation,
        apply_fallback=apply_fallback,
        coverage_repaired_by_retry=coverage_repaired_by_retry,
        coverage_repair_stage=coverage_repair_stage,
        source_attempt_index=source_attempt_index,
        retry_attempted=retry_attempted,
    )
    if output and "coverage_summary" in output:
        output["coverage_summary"].setdefault("final_trace_schema", extracted.get("trace_schema", "groups_legacy"))
    if output is not None:
        output.setdefault("all_clusters", list(output.get("clusters", [])))
        output.setdefault("patch_assignments", extracted.get("assignment_payload", {"patches": []}))
    return output


def _build_trace_patho_r1_prompt(request, bundle):
    header = _trace_prompt_header(request)
    grid_meta = _load_trace_grid_metadata(request)
    if not grid_meta:
        proposals = request["metadata"].get("proposals", [])
        lines = [header, "", "Candidate proposals in thumbnail pixel space:"]
        for proposal in proposals:
            bbox = proposal["cluster_bbox_thumb"]
            meta = proposal.get("metadata", {})
            lines.append(
                "- {cluster_id}: bbox=({x1},{y1},{x2},{y2}), tissue_fraction={tissue:.4f}, pale_fraction={pale:.4f}, artifact_fraction={artifact:.4f}, area_fraction={area:.4f}".format(
                    cluster_id=proposal["cluster_id"],
                    x1=bbox["x1"],
                    y1=bbox["y1"],
                    x2=bbox["x2"],
                    y2=bbox["y2"],
                    tissue=float(meta.get("tissue_fraction", 0.0)),
                    pale=float(meta.get("pale_fraction", 0.0)),
                    artifact=float(meta.get("artifact_fraction", 0.0)),
                    area=float(meta.get("area_fraction", 0.0)),
                )
            )
        lines.extend(
            [
                "",
                "Return JSON only in the form:",
                '{',
                '  "clusters": [',
                '    {',
                '      "cluster_id": "cluster_00",',
                '      "l": "ssl_suspicious_mucosa",',
                '      "s": 4,',
                '      "d": true,',
                '      "review_stage": "serrated_screening",',
                '      "desc": "short reason",',
                '      "evidence": ["reason 1", "reason 2"]',
                '    }',
                '  ]',
                '}',
                "",
                'Allowed labels for "l": background_artifact_stroma, normal_mucosa, conventional_adenoma_like, inflammatory_polyp_like, ssl_suspicious_mucosa.',
                'Only use cluster_id values from the provided candidate proposals.',
            ]
        )
        return "\n".join(lines)

    lines = [
        header,
        "",
        "You are reviewing a colorectal whole-slide thumbnail that has already been cropped to tissue and divided into a regular grid with visible grid IDs.",
        "Simulate a pathologist's global screening pass.",
        "Focus on workflow routing for SSL versus conventional adenoma versus inflammatory/background regions.",
        "Do not issue a final diagnosis, tumor grade, or broad differential diagnosis.",
        "",
        "Task:",
        "1. Assign every selected patch ID exactly once in the patches array.",
        "2. Classify each patch independently using region_semantic, diagnostic_priority, and require_high_magnification.",
        "3. Use the same name/region_semantic/priority for patches that should later be merged by the system.",
        "4. Do not create groups or id_list; the system will aggregate patch assignments into clusters.",
        "5. Use the region_semantic field to classify each patch into the workflow trace label set.",
        "",
        "Structured input contract:",
        "- The accompanying *_grid.json is the only authoritative structured input source.",
        "- The JPEG images are for visual review only; do not infer hidden grid identities beyond the JSON-listed cells.",
        "- Background outside the tissue has been masked in black to emphasize the true tissue regions; do not treat black masked areas as additional tissue evidence.",
        "- Use only selected cells where is_selected=true from the JSON metadata listed below.",
        "- This task is INVALID unless patches contains exactly one assignment for every selected patch ID.",
        "- Before writing JSON, mentally enumerate all selected patch IDs and verify missing=[], duplicates=[], out_of_set=[].",
        "- Do not omit any selected patch ID. If a patch looks low value, still assign it to background_artifact_stroma.",
        "- If your generation format uses <think> and <answer> tags, keep reasoning in <think> and put exactly one raw JSON object in <answer>.",
        "",
        "Workflow trace label set for region_semantic:",
        "- background_artifact_stroma: background, artifact, muscle, stroma, or other discardable low-value regions.",
        "- normal_mucosa: reviewable but low-priority non-lesional mucosa.",
        "- conventional_adenoma_like: non-SSL adenomatous mucosa, mainly tubular adenoma or tubulovillous adenoma patterns.",
        "- inflammatory_polyp_like: inflammatory or reactive polyp-like mucosa that should stay outside the dysplasia branch by default.",
        "- ssl_suspicious_mucosa: mucosa suspicious for SSL/serrated architecture and worth directed review; encode urgency using diagnostic_priority rather than a separate SSL label.",
        "",
        "Screening guidance:",
        "- Look for mucosal regions that may warrant closer review for serrated architecture, conventional adenoma architecture, or inflammatory/reactive polyp context.",
        "- Helpful cues may include pale or mucus-rich surface appearance, contour irregularity, broad lesion shape, crypt crowding suggestive of serration, or a lesion edge worth higher-magnification inspection.",
        "- For others, keep tubular adenoma, tubulovillous adenoma, and inflammatory polyp separate from normal mucosa whenever the thumbnail pattern supports that distinction.",
        "- Do not report dysplasia, mitoses, final tumor type, or unrelated pathology.",
        "",
        "Grid metadata:",
        "- thumbnail_mode={thumbnail_mode}",
        "- grid_rows={grid_rows}, grid_cols={grid_cols}, selected_cells={selected_cells}",
        "- grid_cell_size_thumbnail={grid_cell_size_thumbnail}, grid_stride_thumbnail={grid_stride_thumbnail}",
        "- Use only patch IDs that are listed below.",
        "- Exact allowed patch vocabulary (copy patch_id entries only from this list): {selected_patch_vocab}",
        "",
        "Available selected grid cells:",
    ]
    lines = [line.format(
        thumbnail_mode=grid_meta.get("thumbnail_mode", "unknown"),
        grid_rows=int(grid_meta.get("grid_rows", 0) or 0),
        grid_cols=int(grid_meta.get("grid_cols", 0) or 0),
        selected_cells=int(grid_meta.get("n_selected_cells", 0) or 0),
        grid_cell_size_thumbnail=int(grid_meta.get("grid_cell_size_thumbnail", 0) or 0),
        grid_stride_thumbnail=int(grid_meta.get("grid_stride_thumbnail", 0) or 0),
        selected_patch_vocab=_selected_patch_vocab_text(grid_meta),
    ) for line in lines]
    for cell in grid_meta.get("grid_cells", []):
        if not cell.get("is_selected", True):
            continue
        lines.append(
            "- patch_id=[{row},{col}], row_id={row}, col_id={col}, center_in_tissue={center_in_tissue}, tissue_coverage_ratio={tissue:.4f}, thumbnail_bbox=({x1},{y1},{x2},{y2})".format(
                row=int(cell["row_id"]),
                col=int(cell["col_id"]),
                center_in_tissue=bool(cell.get("center_in_tissue", False)),
                tissue=float(cell.get("tissue_coverage_ratio", 0.0)),
                x1=int(cell["thumbnail_top_left_x"]),
                y1=int(cell["thumbnail_top_left_y"]),
                x2=int(cell["thumbnail_top_left_x"]) + int(cell["thumbnail_width"]),
                y2=int(cell["thumbnail_top_left_y"]) + int(cell["thumbnail_height"]),
            )
        )
    lines.extend(
        [
            "",
            "Return JSON only in the form:",
            "{",
            '  "patches": [',
            "    {",
            '      "patch_id": [0, 0],',
            '      "name": "SSL-like mucosa near lesion edge",',
            '      "region_semantic": "ssl_suspicious_mucosa",',
            '      "description": "brief visual summary of the mucosal region and why it may warrant review",',
            '      "require_high_magnification": true,',
            '      "severity_reasoning": "brief reason for the assigned diagnostic priority",',
            '      "diagnostic_priority": 4,',
            '      "observation_points": ["possible serrated surface pattern", "mucosal edge worth closer review"]',
            "    },",
            "    {",
            '      "patch_id": [1, 0],',
            '      "name": "Background/stroma remainder",',
            '      "region_semantic": "background_artifact_stroma",',
            '      "description": "low-value residual selected patch that still must be covered",',
            '      "require_high_magnification": false,',
            '      "severity_reasoning": "discard/background coverage for selected patch completeness",',
            '      "diagnostic_priority": 0,',
            '      "observation_points": ["coverage-preserving discard group"]',
            "    }",
            "  ]",
            "}",
            "",
            "Rules:",
            '- `patches` must contain one object per selected patch.',
            '- `patch_id` must contain exactly one [row, col] pair.',
            "- Treat each [row, col] pair as the patch_id primary key.",
            "- Cover every selected patch ID exactly once across the patches array. Any missing patch_id makes the answer invalid.",
            "- Do not invent patch IDs.",
            "- Every patch_id must be an exact copy of one item from the allowed patch vocabulary above.",
            "- Do not repeat the same patch ID.",
            "- Only use patch IDs from cells where is_selected=true.",
            "- Self-check before finalizing: missing=[], duplicates=[], out_of_set=[].",
            "- region_semantic must be one of: background_artifact_stroma, normal_mucosa, conventional_adenoma_like, inflammatory_polyp_like, ssl_suspicious_mucosa.",
            "- diagnostic_priority must be an integer from 0 to 5, where 5 is the highest priority and 0 is discard/background.",
            "- Do not output explanatory prose before or after the JSON object.",
            "- Output one JSON object only and nothing else.",
        ]
    )
    return "\n".join(lines)


def _extract_first_json_object(text):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _trace_group_repair_snapshot(validation):
    rows = []
    if validation.get("trace_schema") == "patch_assignments":
        for record in validation.get("assignment_records", []):
            assignment = record.get("assignment", {})
            rows.append(
                {
                    "assignment_index": int(record.get("assignment_index", 0)),
                    "patch_id": list(record.get("normalized_id") or []),
                    "name": str(assignment.get("name", "")),
                    "region_semantic": str(assignment.get("region_semantic", "")),
                    "duplicate_patch_ids": [list(item) for item in record.get("duplicate_patch_ids", [])],
                    "ignored_patch_ids": list(record.get("ignored_patch_ids", [])),
                    "unexpected_patch_ids": [list(item) for item in record.get("unexpected_patch_ids", [])],
                    "is_empty_after_validation": bool(record.get("is_empty", False)),
                }
            )
        return rows
    for group_record in validation.get("group_records", []):
        group = group_record.get("group", {})
        rows.append(
            {
                "group_index": int(group_record.get("group_index", 0)),
                "name": str(group.get("name", "")),
                "region_semantic": str(group.get("region_semantic", "")),
                "valid_selected_patch_ids": [
                    [int(row_id), int(col_id)] for row_id, col_id in group_record.get("normalized_ids", [])
                ],
                "duplicate_patch_ids": [list(item) for item in group_record.get("duplicate_patch_ids", [])],
                "ignored_patch_ids": list(group_record.get("ignored_patch_ids", [])),
                "unexpected_patch_ids": [list(item) for item in group_record.get("unexpected_patch_ids", [])],
                "is_empty_after_validation": bool(group_record.get("is_empty", False)),
            }
        )
    return rows


def _max_trace_structure_retries(bundle):
    trace_config = bundle.get("runtime", {}).get("trace", {})
    try:
        value = int(trace_config.get("max_structure_repair_attempts", 2))
    except Exception:
        value = 2
    return max(0, min(value, 3))


def _build_trace_coverage_retry_prompt(request, bundle, response_payload, validation, retry_index):
    base_prompt = _build_trace_patho_r1_prompt(request, bundle)
    grid_meta = _load_trace_grid_metadata(request) or {}
    selected_patch_ids = _selected_patch_ids(grid_meta)
    selected_patch_vocab = _selected_patch_vocab_text(grid_meta)
    prior_schema = validation.get("trace_schema", "unknown")
    lines = [
        base_prompt,
        "",
        "Coverage repair instruction (attempt {0}):".format(int(retry_index)),
        "- Your previous response failed the exact-coverage contract.",
        "- Return a corrected COMPLETE patch assignment JSON object.",
        "- Do not describe the fix. Do not output prose. Output one JSON object only.",
        '- Rewrite the full payload from scratch as {"patches":[...]} using only the exact selected patch list below.',
        "- Every patch_id must be copied verbatim from the exact allowed patch vocabulary below.",
        "- Reuse valid semantic labels when possible, but ensure every selected patch_id appears exactly once.",
        "- Any low-value leftover patch must be assigned to background_artifact_stroma rather than omitted.",
        "- Remove any assignment that would be empty after validation.",
        "- Do not invent neighbors, inferred cells, or interpolated patch IDs.",
        "",
        "Exact allowed patch vocabulary (copy only from this list):",
        selected_patch_vocab,
        "",
        "Selected patch IDs:",
        json.dumps([[int(row_id), int(col_id)] for row_id, col_id in selected_patch_ids], ensure_ascii=False),
        "Coverage validator findings:",
        "missing_patch_ids={0}".format(json.dumps(validation.get("missing_patch_ids", []), ensure_ascii=False)),
        "duplicate_patch_ids={0}".format(json.dumps(validation.get("duplicate_patch_ids", []), ensure_ascii=False)),
        "unexpected_patch_ids={0}".format(json.dumps(validation.get("unexpected_patch_ids", []), ensure_ascii=False)),
        "ignored_patch_ids={0}".format(json.dumps(validation.get("ignored_patch_ids", []), ensure_ascii=False)),
        "empty_assignments={0}".format(json.dumps(validation.get("empty_assignments", validation.get("empty_groups", [])), ensure_ascii=False)),
        "",
        "Previous response schema:",
        str(prior_schema),
        "",
        "Validated snapshot of the previous response:",
        json.dumps(_trace_group_repair_snapshot(validation), ensure_ascii=False, indent=2),
        "",
        "Previous response as parsed JSON:",
        json.dumps(response_payload, ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines)


def _trace_evaluation_score(evaluation):
    validation = evaluation.get("validation") or {}
    issue_count = sum(
        len(validation.get(key, []))
        for key in ("missing_patch_ids", "duplicate_patch_ids", "unexpected_patch_ids", "empty_groups")
    )
    issue_count += len(validation.get("ignored_patch_ids", []))
    groups_source = validation.get("groups_source", "")
    trace_schema = validation.get("trace_schema", "")
    return (
        1 if validation.get("coverage_ok") else 0,
        int(validation.get("covered_patch_count", 0)),
        -int(issue_count),
        2 if trace_schema == "patch_assignments" else (1 if groups_source == "json" else 0),
        -int(evaluation.get("attempt_index", 0)),
    )


def _evaluate_trace_grid_text(text, request):
    extracted = _extract_trace_patch_assignments_payload(text, request)
    validation = _validate_trace_extracted_payload(extracted, request)
    return {
        "text": text,
        "assignment_payload": extracted.get("assignment_payload", {"patches": []}),
        "groups_payload": extracted.get("groups_payload", {"groups": []}),
        "response_payload": extracted.get("assignment_payload") if extracted.get("trace_schema") == "patch_assignments" else extracted.get("groups_payload", {"groups": []}),
        "trace_schema": extracted.get("trace_schema", "parse_failure"),
        "groups_source": extracted["groups_source"],
        "parse_failure": extracted["parse_failure"],
        "validation": validation,
    }


def _attempt_metadata_from_trace_evaluation(evaluation, attempt_type, prompt_text):
    validation = evaluation.get("validation") or {}
    return {
        "attempt_index": int(evaluation.get("attempt_index", 0)),
        "attempt_type": str(attempt_type),
        "coverage_ok": bool(validation.get("coverage_ok", False)),
        "missing_patch_ids": [list(item) for item in validation.get("missing_patch_ids", [])],
        "duplicate_patch_ids": [list(item) for item in validation.get("duplicate_patch_ids", [])],
        "ignored_patch_ids": list(validation.get("ignored_patch_ids", [])),
        "unexpected_patch_ids": [list(item) for item in validation.get("unexpected_patch_ids", [])],
        "empty_groups": list(validation.get("empty_groups", [])),
        "empty_assignments": list(validation.get("empty_assignments", [])),
        "covered_patch_count": int(validation.get("covered_patch_count", 0)),
        "selected_patch_count": int(validation.get("selected_patch_count", 0)),
        "assignment_count": int(validation.get("assignment_count", 0)),
        "groups_source": validation.get("groups_source", evaluation.get("groups_source", "unknown")),
        "trace_schema": validation.get("trace_schema", evaluation.get("trace_schema", "unknown")),
        "parse_failure": bool(validation.get("parse_failure", evaluation.get("parse_failure", False))),
        "prompt_text": prompt_text,
        "raw_text": evaluation.get("text", ""),
    }


def _run_trace_grid_with_coverage_retry(generate_text_fn, request, bundle):
    prompt_text = _build_trace_patho_r1_prompt(request, bundle)
    evaluations = []
    raw_texts = []
    initial_text = generate_text_fn(prompt_text)
    initial_evaluation = _evaluate_trace_grid_text(initial_text, request)
    initial_evaluation["attempt_index"] = 0
    evaluations.append(initial_evaluation)
    raw_texts.append({"attempt_index": 0, "attempt_type": "initial", "prompt_text": prompt_text, "text": initial_text})
    current_evaluation = initial_evaluation
    max_retries = _max_trace_structure_retries(bundle)
    for retry_index in range(1, max_retries + 1):
        if current_evaluation["validation"].get("coverage_ok", False):
            break
        retry_prompt = _build_trace_coverage_retry_prompt(
            request,
            bundle,
            current_evaluation.get("response_payload", current_evaluation.get("groups_payload", {})),
            current_evaluation["validation"],
            retry_index=retry_index,
        )
        retry_text = generate_text_fn(retry_prompt)
        retry_evaluation = _evaluate_trace_grid_text(retry_text, request)
        retry_evaluation["attempt_index"] = retry_index
        evaluations.append(retry_evaluation)
        raw_texts.append(
            {
                "attempt_index": retry_index,
                "attempt_type": "coverage_retry" if retry_index == 1 else "coverage_retry_{0}".format(retry_index),
                "prompt_text": retry_prompt,
                "text": retry_text,
            }
        )
        current_evaluation = retry_evaluation
    best_evaluation = max(evaluations, key=_trace_evaluation_score)
    retry_attempted = len(evaluations) > 1
    coverage_repaired_by_retry = retry_attempted and best_evaluation.get("attempt_index", 0) > 0 and best_evaluation["validation"].get("coverage_ok", False)
    coverage_repair_stage = "post_retry" if retry_attempted else "no_retry_needed"
    output = _build_trace_clusters_from_extracted_payload(
        {
            "trace_schema": best_evaluation.get("trace_schema", "groups_legacy"),
            "assignment_payload": best_evaluation.get("assignment_payload", {"patches": []}),
            "groups_payload": best_evaluation.get("groups_payload", {"groups": []}),
        },
        request,
        validation=best_evaluation["validation"],
        apply_fallback=not best_evaluation["validation"].get("coverage_ok", False),
        coverage_repaired_by_retry=coverage_repaired_by_retry,
        coverage_repair_stage=coverage_repair_stage,
        source_attempt_index=best_evaluation.get("attempt_index", 0),
        retry_attempted=retry_attempted,
    ) or {"clusters": [], "coverage_summary": {}}
    output.setdefault("all_clusters", list(output.get("clusters", [])))
    output.setdefault("patch_assignments", best_evaluation.get("assignment_payload", {"patches": []}))
    output["coverage_summary"].update(
        {
            "initial_coverage_ok": bool(initial_evaluation["validation"].get("coverage_ok", False)),
            "retry_attempted": retry_attempted,
            "final_used_fallback": bool(output["coverage_summary"].get("coverage_repair_applied", False)),
            "final_groups_source": best_evaluation["validation"].get("groups_source", best_evaluation.get("groups_source", "unknown")),
            "final_trace_schema": best_evaluation["validation"].get("trace_schema", best_evaluation.get("trace_schema", "unknown")),
        }
    )
    return {
        "output": output,
        "trace_attempts": [
            _attempt_metadata_from_trace_evaluation(
                evaluation,
                raw_texts[index]["attempt_type"],
                raw_texts[index]["prompt_text"],
            )
            for index, evaluation in enumerate(evaluations)
        ],
        "raw_text": best_evaluation.get("text", initial_text),
        "raw_texts": raw_texts,
    }


def _build_trace_output_from_text(
    text,
    request,
    bundle,
    apply_fallback=True,
    coverage_repaired_by_retry=False,
    coverage_repair_stage="no_retry_needed",
    source_attempt_index=0,
    retry_attempted=False,
):
    grid_meta = _load_trace_grid_metadata(request)
    extracted = _extract_trace_patch_assignments_payload(text, request)
    parsed = extracted.get("parsed")
    if grid_meta:
        validation = _validate_trace_extracted_payload(extracted, request)
        output = _build_trace_clusters_from_extracted_payload(
            extracted,
            request,
            validation=validation,
            apply_fallback=apply_fallback,
            coverage_repaired_by_retry=coverage_repaired_by_retry,
            coverage_repair_stage=coverage_repair_stage,
            source_attempt_index=source_attempt_index,
            retry_attempted=retry_attempted,
        ) or {"clusters": [], "coverage_summary": {}}
        output.setdefault("all_clusters", list(output.get("clusters", [])))
        output.setdefault("patch_assignments", extracted.get("assignment_payload", {"patches": []}))
        output["coverage_summary"].update(
            {
                "initial_coverage_ok": bool(validation.get("coverage_ok", False)),
                "final_used_fallback": bool(output["coverage_summary"].get("coverage_repair_applied", False)),
                "final_groups_source": validation.get("groups_source", extracted["groups_source"]),
                "final_trace_schema": validation.get("trace_schema", extracted.get("trace_schema", "unknown")),
            }
        )
        return output
    output_clusters = []
    proposals = request["metadata"].get("proposals", [])
    proposal_lookup = {proposal["cluster_id"]: proposal for proposal in proposals}
    proposal_lookup_ci = {proposal["cluster_id"].lower(): proposal["cluster_id"] for proposal in proposals}
    if not output_clusters and isinstance(parsed, dict) and isinstance(parsed.get("clusters"), list):
        for item in parsed.get("clusters", []):
            cluster_id = str(item.get("cluster_id", ""))
            normalized_cluster_id = proposal_lookup_ci.get(cluster_id.lower())
            if normalized_cluster_id not in proposal_lookup:
                continue
            label = _normalize_trace_label(
                region_semantic=item.get("l", TRACE_NORMAL_LABEL),
                name=item.get("l", TRACE_NORMAL_LABEL),
                description=item.get("desc", ""),
                severity_reasoning="",
                require_high_magnification=bool(item.get("d", False)),
                diagnostic_priority=item.get("s", 0),
            )
            if label not in bundle["runtime"]["trace"]["labels"]:
                label = TRACE_NORMAL_LABEL
            evidence = item.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = [str(evidence)]
            priority = _normalize_trace_priority(item.get("s", 0), default_value=0)
            require_high_magnification = bool(item.get("d", False))
            if label == TRACE_CONVENTIONAL_LABEL:
                require_high_magnification = True
            metadata = _trace_label_metadata(
                label,
                priority,
                require_high_magnification,
                {
                    "source": "patho_r1_trace",
                    "serrated_dysplasia_suspected": item.get("serrated_dysplasia_suspected", False),
                    "conventional_dysplasia_suspected": item.get("conventional_dysplasia_suspected", False),
                    "conventional_subtype_hint": item.get("conventional_subtype_hint"),
                    "inflammatory_subtype_hint": item.get("inflammatory_subtype_hint"),
                    "serrated_family_hint": item.get("serrated_family_hint"),
                },
            )
            output_clusters.append(
                {
                    "cluster_id": normalized_cluster_id,
                    "l": label,
                    "s": priority,
                    "d": require_high_magnification,
                    "review_stage": item.get("review_stage", _trace_review_stage_for_label(label)),
                    "crypt_disorder_risk": (
                        _normalize_trace_priority(item.get("crypt_disorder_risk", priority), default_value=0)
                        if _trace_branch_for_label(label) == "serrated"
                        else 0
                    ),
                    "dysplasia_review_needed": bool(
                        metadata.get("serrated_dysplasia_suspected")
                        or metadata.get("conventional_dysplasia_suspected")
                    ),
                    "desc": item.get("desc", "Patho-R1 trace selection."),
                    "evidence": [str(value) for value in evidence],
                    "metadata": metadata,
                    "patch_ids_ordered": [],
                    "patches_thumb": [],
                    "patches_level0": [],
                    "group_bbox_thumb": dict(proposal_lookup[normalized_cluster_id]["cluster_bbox_thumb"]),
                    "group_bbox_level0": dict(proposal_lookup[normalized_cluster_id]["cluster_bbox_level0"]),
                }
            )
    if not output_clusters:
        fallback_cluster_ids = _extract_trace_cluster_ids_from_text(text, proposal_lookup_ci)
        for cluster_id in fallback_cluster_ids:
            metadata = _trace_label_metadata(
                TRACE_SERRATED_LABEL,
                4,
                True,
                {"source": "patho_r1_trace_free_form", "serrated_family_hint": "equivocal_serrated"},
            )
            output_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "l": TRACE_SERRATED_LABEL,
                    "s": 4,
                    "d": True,
                    "review_stage": _trace_review_stage_for_label(TRACE_SERRATED_LABEL),
                    "crypt_disorder_risk": 4,
                    "dysplasia_review_needed": bool(metadata.get("serrated_dysplasia_suspected")),
                    "desc": "Patho-R1 selected this proposal from free-form trace reasoning output.",
                    "evidence": ["patho_r1_free_form_trace_response"],
                    "metadata": metadata,
                    "patch_ids_ordered": [],
                    "patches_thumb": [],
                    "patches_level0": [],
                    "group_bbox_thumb": dict(proposal_lookup[cluster_id]["cluster_bbox_thumb"]),
                    "group_bbox_level0": dict(proposal_lookup[cluster_id]["cluster_bbox_level0"]),
                }
            )
    if not output_clusters:
        raise BackendExecutionError("Patho-R1 trace response did not select any valid proposal cluster_id")
    return {"clusters": _sort_trace_clusters(output_clusters)}


def _extract_trace_cluster_ids_from_text(text, proposal_lookup_ci):
    matches = []
    for match in re.findall(r"cluster[_\-\s]*\d+", text, flags=re.IGNORECASE):
        normalized = match.lower().replace("-", "_").replace(" ", "")
        cluster_id = proposal_lookup_ci.get(normalized)
        if cluster_id and cluster_id not in matches:
            matches.append(cluster_id)
    return matches


def _build_text_driven_output(text, request, bundle):
    review_goal = request["metadata"]["step"].get("review_goal")
    serrated_criteria = list(bundle["runtime"]["observe"].get("serrated_criteria", []))
    abnormal_crypt_criteria = list(bundle["runtime"]["observe"].get("abnormal_crypt_criteria", []))
    conventional_criteria = list(bundle["runtime"]["observe"].get("conventional_adenoma_criteria", []))
    dysplasia_criteria = list(bundle["runtime"]["observe"].get("dysplasia_criteria", []))
    lower_text = text.lower()
    serrated_hits = _blank_hits(serrated_criteria)
    abnormal_crypt_hits = _blank_hits(abnormal_crypt_criteria)
    conventional_hits = _blank_hits(conventional_criteria)
    serrated_dysplasia_hits = _blank_hits(dysplasia_criteria)
    conventional_dysplasia_hits = _blank_hits(dysplasia_criteria)

    keyword_map = {
        "serrated_surface_pattern": ["serrated", "serration"],
        "mucus_rich_surface": ["mucus", "mucin"],
        "serrated_lesion_context": ["serrated lesion", "serrated polyp", "mucosal lesion"],
        "basal_dilatation": ["dilat", "dilated", "dilation"],
        "crypt_branching": ["branch", "branched"],
        "horizontal_growth": ["horizontal"],
        "boot_l_t_shaped_crypt": ["boot", "l-shaped", "t-shaped"],
        "serration_to_base": ["serration", "to the base"],
        "mucus_cap": ["mucus cap", "mucus", "mucin"],
        "abnormal_maturation": ["maturation", "abnormal maturation", "dysmaturation"],
        "nuclear_enlargement_stratification": ["nuclear enlargement", "stratification"],
        "hyperchromasia": ["hyperchrom", "hyperchromasia"],
        "mitotic_activity_atypia": ["mitotic", "atypia", "atypical"],
        "architectural_crowding": ["crowding"],
        "tubular_or_tubulovillous_architecture": ["tubular", "tubulovillous", "villous", "adenoma"],
        "crowded_adenomatous_glands": ["crowded", "adenomatous", "gland"],
        "pencillate_hyperchromatic_nuclei": ["pencillate", "hyperchrom", "nuclei"],
    }
    if review_goal == "serrated_lesion_assessment":
        for criterion in serrated_hits:
            serrated_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal == "abnormal_crypt_assessment":
        for criterion in abnormal_crypt_hits:
            abnormal_crypt_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal == "conventional_adenoma_assessment":
        for criterion in conventional_hits:
            conventional_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal == "serrated_dysplasia_assessment":
        for criterion in serrated_dysplasia_hits:
            serrated_dysplasia_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal == "conventional_dysplasia_assessment":
        for criterion in conventional_dysplasia_hits:
            conventional_dysplasia_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    else:
        pass
    dysplasia_hits = _combine_hits_maps(serrated_dysplasia_hits, conventional_dysplasia_hits)

    if review_goal == "serrated_lesion_assessment":
        stage_decision = "supports_serrated_lesion"
        next_step = "Proceed to abnormal crypt review if the lesion remains within the serrated pathway."
    elif review_goal == "abnormal_crypt_assessment":
        stage_decision = "supports_abnormal_crypt"
        next_step = "Proceed to serrated dysplasia review only if abnormal crypt support is convincing."
    elif review_goal == "conventional_adenoma_assessment":
        stage_decision = "supports_conventional_adenoma"
        next_step = "Proceed to conventional dysplasia review within the conventional adenoma branch."
    elif review_goal == "serrated_dysplasia_assessment":
        stage_decision = "serrated_dysplasia_supported"
        next_step = "Integrate the serrated branch impression and finalize the report."
    elif review_goal == "conventional_dysplasia_assessment":
        stage_decision = "conventional_dysplasia_supported"
        next_step = "Integrate the conventional branch impression and finalize the report."
    else:
        stage_decision = "supports_non_serrated_overview"
        next_step = "Integrate the layered impression and finalize the report."

    return {
        "observation": text.splitlines()[-1][:320],
        "reasoning": "Local Patho-R1 textual evidence was used to summarize the requested diagnostic layer.",
        "next_step": next_step,
        "level_1_findings": _supporting_findings_from_hits(serrated_hits)
        + _supporting_findings_from_hits(conventional_hits),
        "level_2_findings": _supporting_findings_from_hits(abnormal_crypt_hits),
        "level_3_findings": _supporting_findings_from_hits(dysplasia_hits),
        "stage_decision": stage_decision,
        "serrated_hits": serrated_hits,
        "abnormal_crypt_hits": abnormal_crypt_hits,
        "conventional_hits": conventional_hits,
        "serrated_dysplasia_hits": serrated_dysplasia_hits,
        "conventional_dysplasia_hits": conventional_dysplasia_hits,
        "dysplasia_hits": dysplasia_hits,
        "confidence": 0.62,
    }


def _supporting_findings_from_hits(hits):
    return [criterion for criterion, status in hits.items() if status == "supporting"]


def _aggregate_hits(records, hits_key, criteria):
    checklist = {criterion: {"status": "not_assessed", "evidence_steps": []} for criterion in criteria}
    for record in records:
        hits = record.get("metadata", {}).get(hits_key, {})
        for criterion, status in hits.items():
            if criterion not in checklist or status == "not_assessed":
                continue
            current = checklist[criterion]["status"]
            if status == "supporting":
                checklist[criterion]["status"] = "supporting"
            elif status == "opposing" and current == "not_assessed":
                checklist[criterion]["status"] = "opposing"
            elif status == "uncertain" and current == "not_assessed":
                checklist[criterion]["status"] = "uncertain"
            elif current != "supporting":
                checklist[criterion]["status"] = status
            checklist[criterion]["evidence_steps"].append(record["step_id"])
    return checklist


def _combine_hit_status(left, right):
    order = {"not_assessed": 0, "opposing": 1, "uncertain": 2, "supporting": 3}
    return left if order.get(left, 0) >= order.get(right, 0) else right


def _combine_hits_maps(left, right):
    keys = list(left.keys())
    for key in right:
        if key not in keys:
            keys.append(key)
    return {key: _combine_hit_status(left.get(key, "not_assessed"), right.get(key, "not_assessed")) for key in keys}


def _merge_checklists(left, right):
    keys = list(left.keys())
    for key in right:
        if key not in keys:
            keys.append(key)
    merged = {}
    for key in keys:
        left_item = left.get(key, {"status": "not_assessed", "evidence_steps": []})
        right_item = right.get(key, {"status": "not_assessed", "evidence_steps": []})
        merged[key] = {
            "status": _combine_hit_status(left_item.get("status", "not_assessed"), right_item.get("status", "not_assessed")),
            "evidence_steps": list(left_item.get("evidence_steps", [])) + [
                step for step in right_item.get("evidence_steps", []) if step not in left_item.get("evidence_steps", [])
            ],
        }
    return merged


def _serrated_assessment(trace_clusters, checklist):
    support_count = len([value for value in checklist.values() if value["status"] == "supporting"])
    oppose_count = len([value for value in checklist.values() if value["status"] == "opposing"])
    trace_support = any(
        cluster.get("l") in (TRACE_SERRATED_LABEL, TRACE_LEGACY_SSL_HIGH_LABEL)
        for cluster in trace_clusters
    )
    positive = trace_support or support_count >= 2
    score = min(0.95, max(0.05, 0.20 + 0.16 * support_count + 0.10 * int(trace_support) - 0.08 * oppose_count))
    return {
        "label": "serrated_lesion" if positive else "non_serrated_lesion",
        "positive": positive,
        "score": round(score, 4),
    }


def _abnormal_crypt_assessment(serrated_assessment, checklist):
    support_count = len([value for value in checklist.values() if value["status"] == "supporting"])
    uncertain_count = len([value for value in checklist.values() if value["status"] == "uncertain"])
    structural_support = len(
        [
            key
            for key, value in checklist.items()
            if key in ("basal_dilatation", "crypt_branching", "horizontal_growth", "boot_l_t_shaped_crypt")
            and value["status"] == "supporting"
        ]
    )
    positive = serrated_assessment["positive"] and support_count >= 3 and structural_support >= 1
    if not serrated_assessment["positive"]:
        label = "not_applicable_non_serrated"
    elif positive:
        label = "abnormal_crypt_supported"
    elif support_count == 0 and uncertain_count == 0:
        label = "serrated_but_no_abnormal_crypt"
    elif support_count == 0 and uncertain_count > 0:
        label = "indeterminate_abnormal_crypt"
    else:
        label = "serrated_but_no_abnormal_crypt"
    score = min(0.95, max(0.05, 0.18 + 0.10 * support_count + 0.08 * structural_support))
    return {
        "label": label,
        "positive": positive,
        "score": round(score, 4),
    }


def _conventional_adenoma_assessment(trace_clusters, checklist):
    support_count = len([value for value in checklist.values() if value["status"] == "supporting"])
    oppose_count = len([value for value in checklist.values() if value["status"] == "opposing"])
    trace_support = any(cluster.get("l") == TRACE_CONVENTIONAL_LABEL for cluster in trace_clusters)
    positive = trace_support or support_count >= 2
    if positive:
        label = "conventional_adenoma_supported"
    elif oppose_count >= 2:
        label = "conventional_adenoma_opposed"
    else:
        label = "conventional_adenoma_not_supported_or_indeterminate"
    score = min(0.95, max(0.05, 0.18 + 0.16 * support_count + 0.14 * int(trace_support) - 0.08 * oppose_count))
    return {
        "label": label,
        "positive": positive,
        "score": round(score, 4),
    }


def _branch_dysplasia_assessment(branch_gate_assessment, checklist, gate_label, supported_label, negative_label, indeterminate_label):
    if not branch_gate_assessment["positive"]:
        return {
            "label": gate_label,
            "positive": False,
            "score": 0.0,
        }
    support_count = len([value for value in checklist.values() if value["status"] == "supporting"])
    assessed_count = len([value for value in checklist.values() if value["status"] != "not_assessed"])
    positive = support_count >= 2
    if positive:
        label = supported_label
    elif assessed_count == 0:
        label = indeterminate_label
    else:
        label = negative_label
    score = min(0.95, max(0.05, 0.15 + 0.12 * support_count))
    return {
        "label": label,
        "positive": positive,
        "score": round(score, 4),
    }


def _overall_dysplasia_assessment(serrated_dysplasia_assessment, conventional_dysplasia_assessment):
    serrated_positive = bool(serrated_dysplasia_assessment.get("positive"))
    conventional_positive = bool(conventional_dysplasia_assessment.get("positive"))
    if serrated_positive and conventional_positive:
        label = "serrated_and_conventional_dysplasia_supported"
    elif serrated_positive:
        label = "serrated_dysplasia_supported"
    elif conventional_positive:
        label = "conventional_dysplasia_supported"
    else:
        label = "dysplasia_not_supported_or_not_entered"
    score = max(
        float(serrated_dysplasia_assessment.get("score", 0.0)),
        float(conventional_dysplasia_assessment.get("score", 0.0)),
    )
    return {
        "label": label,
        "positive": serrated_positive or conventional_positive,
        "score": round(score, 4),
        "serrated_positive": serrated_positive,
        "conventional_positive": conventional_positive,
    }


def _final_case_assessment(
    serrated_assessment,
    serrated_dysplasia_assessment,
    conventional_adenoma_assessment,
    conventional_dysplasia_assessment,
):
    labels = []
    if serrated_assessment["positive"]:
        labels.append("SSL+dysplasia" if serrated_dysplasia_assessment["positive"] else "SSL")
    if conventional_adenoma_assessment["positive"]:
        labels.append("Others+dysplasia" if conventional_dysplasia_assessment["positive"] else "Others")
    if not labels:
        labels.append("Others")
    primary_label = labels[0]
    if "SSL+dysplasia" in labels:
        primary_label = "SSL+dysplasia"
    elif "SSL" in labels:
        primary_label = "SSL"
    elif "Others+dysplasia" in labels:
        primary_label = "Others+dysplasia"
    return {
        "label": primary_label,
        "positive": primary_label != "Others" or bool(conventional_adenoma_assessment["positive"]),
        "coexisting_labels": labels,
        "serrated_branch_positive": bool(serrated_assessment["positive"]),
        "serrated_dysplasia_positive": bool(serrated_dysplasia_assessment["positive"]),
        "conventional_branch_positive": bool(conventional_adenoma_assessment["positive"]),
        "conventional_dysplasia_positive": bool(conventional_dysplasia_assessment["positive"]),
    }


def _integrated_impression(
    serrated_assessment,
    abnormal_crypt_assessment,
    serrated_dysplasia_assessment,
    conventional_adenoma_assessment,
    conventional_dysplasia_assessment,
    final_case_assessment,
):
    pieces = []
    if serrated_assessment["positive"]:
        if abnormal_crypt_assessment["positive"] and serrated_dysplasia_assessment["positive"]:
            pieces.append("SSL branch supports abnormal crypt architecture with serrated-branch dysplasia")
        elif abnormal_crypt_assessment["positive"]:
            pieces.append("SSL branch supports abnormal crypt architecture without convincing serrated-branch dysplasia")
        else:
            pieces.append("SSL branch is present but abnormal crypt architecture is not convincingly supported")
    if conventional_adenoma_assessment["positive"]:
        if conventional_dysplasia_assessment["positive"]:
            pieces.append("conventional adenoma branch supports dysplasia, mapped to Others+dysplasia")
        else:
            pieces.append("conventional adenoma branch is present without supported conventional dysplasia")
    if not pieces:
        pieces.append("no SSL or conventional adenoma branch is convincingly supported")
    return "{0}. Final workflow label: {1}.".format("; ".join(pieces), final_case_assessment["label"])


def _render_supporting_lines(checklist):
    findings = []
    for criterion, payload in checklist.items():
        if payload["status"] == "supporting":
            findings.append("{0} ({1})".format(criterion, ", ".join(payload["evidence_steps"])))
    return "; ".join(findings) if findings else "No decisive supporting item recorded."
