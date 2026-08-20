import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from adenoma_agent.trace_supervision import (
    GLOBAL_SCREENING_LEGACY_SCORE_ORIGIN,
    GLOBAL_SCREENING_SINGLE_MODEL_STATUS,
    LESION_TRACE_LABELS,
    TRACE_LABEL_RUBRIC,
    connected_components_for_patch_ids,
    derive_global_screening_fusion,
)
from adenoma_agent.utils import env_with_cuda_visible_devices, read_json, run_command, write_json


class BackendUnavailableError(RuntimeError):
    pass


class BackendExecutionError(RuntimeError):
    pass


class FatalBackendExecutionError(BackendExecutionError):
    pass


def _patch_key(patch_id):
    if not isinstance(patch_id, (list, tuple)) or len(patch_id) != 2:
        return None
    try:
        return "{0},{1}".format(int(patch_id[0]), int(patch_id[1]))
    except Exception:
        return None


def _normalize_conch_label(label):
    label = str(label or "").strip()
    crc_map = {
        "ADI": "background_or_artifact",
        "BACK": "background_or_artifact",
        "DEB": "background_or_artifact",
        "MUS": "background_or_artifact",
        "NORM": "reviewable_normal_mucosa",
        "LYM": "inflammatory_or_stromal_context",
        "STR": "inflammatory_or_stromal_context",
        "MUC": "mucus_rich_or_pale_context",
        "TUM": "epithelial_neoplasia_suspicious",
    }
    normalized = crc_map.get(label.upper(), label)
    return normalized if normalized in TRACE_LABEL_RUBRIC else ""


DIGEPATH_ROI9_CLASS_NAMES = [
    "normal_colon_mucosa",
    "stroma",
    "tumor_epithelium",
    "smooth_muscle",
    "mucus",
    "lymphocytes",
    "debris",
    "background",
    "adipose",
]

DIGEPATH_ROI9_CLASS_TO_TRACE_LABEL = {
    "normal_colon_mucosa": "reviewable_normal_mucosa",
    "stroma": "inflammatory_or_stromal_context",
    "tumor_epithelium": "epithelial_neoplasia_suspicious",
    "smooth_muscle": "background_or_artifact",
    "mucus": "mucus_rich_or_pale_context",
    "lymphocytes": "inflammatory_or_stromal_context",
    "debris": "background_or_artifact",
    "background": "background_or_artifact",
    "adipose": "background_or_artifact",
}

_DIGEPATH_CLASS_ALIASES = {
    "normal": "normal_colon_mucosa",
    "normal_mucosa": "normal_colon_mucosa",
    "colon_mucosa": "normal_colon_mucosa",
    "tumor": "tumor_epithelium",
    "tumour": "tumor_epithelium",
    "tumor_epithelial": "tumor_epithelium",
    "tumour_epithelium": "tumor_epithelium",
    "muscle": "smooth_muscle",
    "smooth-muscle": "smooth_muscle",
    "lymphocyte": "lymphocytes",
    "lymphocytic": "lymphocytes",
    "mucin": "mucus",
    "muc": "mucus",
    "adi": "adipose",
    "back": "background",
    "deb": "debris",
    "mus": "smooth_muscle",
    "norm": "normal_colon_mucosa",
    "lym": "lymphocytes",
    "str": "stroma",
    "tum": "tumor_epithelium",
}


def _normalize_digepath_class(label):
    value = str(label or "").strip().lower().replace(" ", "_")
    value = _DIGEPATH_CLASS_ALIASES.get(value, value)
    return value if value in DIGEPATH_ROI9_CLASS_TO_TRACE_LABEL else ""


def _trace_label_from_digepath_class(label):
    digepath_class = _normalize_digepath_class(label)
    return DIGEPATH_ROI9_CLASS_TO_TRACE_LABEL.get(digepath_class, "")


def _normalize_digepath_trace_label(label):
    value = str(label or "").strip()
    if value in TRACE_LABEL_RUBRIC:
        return value
    return _trace_label_from_digepath_class(value)


def _crc_label_from_conch_prediction(prediction):
    if isinstance(prediction, str):
        value = prediction.strip().upper()
        return value if value in {"ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"} else ""
    if not isinstance(prediction, dict):
        return ""
    for key in ("crc_label", "conch_crc_label", "label", "class", "prediction", "region_semantic"):
        value = str(prediction.get(key) or "").strip().upper()
        if value in {"ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"}:
            return value
    return ""


def _prediction_scores(prediction):
    if not isinstance(prediction, dict):
        return {}
    for key in ("probabilities", "probs", "scores", "logits"):
        value = prediction.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _prediction_confidence(prediction, label):
    if not isinstance(prediction, dict):
        return 1.0 if label else 0.0
    for key in ("confidence", "score", "probability"):
        if key in prediction:
            try:
                return float(prediction.get(key))
            except Exception:
                pass
    scores = _prediction_scores(prediction)
    if scores:
        candidates = []
        for key, value in scores.items():
            try:
                candidates.append((str(key), float(value)))
            except Exception:
                continue
        if candidates:
            return max(score for _key, score in candidates)
    return 1.0 if label else 0.0


def _patch_id_from_prediction(prediction):
    if not isinstance(prediction, dict):
        return None
    patch_id = prediction.get("patch_id")
    if patch_id is None and "row" in prediction and "col" in prediction:
        patch_id = [prediction.get("row"), prediction.get("col")]
    return patch_id


def _label_from_conch_prediction(prediction):
    if isinstance(prediction, str):
        return _normalize_conch_label(prediction)
    if not isinstance(prediction, dict):
        return ""
    for key in ("region_semantic", "label", "class", "prediction"):
        label = _normalize_conch_label(prediction.get(key))
        if label:
            return label
    return ""


def _digepath_class_from_prediction(prediction):
    if isinstance(prediction, str):
        return _normalize_digepath_class(prediction)
    if not isinstance(prediction, dict):
        return ""
    for key in ("digepath_class", "pred_class", "class_name", "label", "class", "prediction"):
        value = _normalize_digepath_class(prediction.get(key))
        if value:
            return value
    return ""


def _label_from_digepath_prediction(prediction):
    if isinstance(prediction, str):
        return _normalize_digepath_trace_label(prediction)
    if not isinstance(prediction, dict):
        return ""
    for key in ("region_semantic", "digepath_region_semantic", "trace_label"):
        label = _normalize_digepath_trace_label(prediction.get(key))
        if label:
            return label
    digepath_class = _digepath_class_from_prediction(prediction)
    return _trace_label_from_digepath_class(digepath_class)


def parse_conch_prediction_payload(api_payload, requested_patch_ids=None):
    requested_patch_ids = requested_patch_ids or []
    predictions = {}
    prediction_details = {}
    invalid = []
    raw_predictions = []
    if isinstance(api_payload, dict) and isinstance(api_payload.get("predictions"), list):
        raw_predictions = list(api_payload.get("predictions", []))
    elif isinstance(api_payload, dict):
        raw_predictions = [api_payload]
    elif isinstance(api_payload, list):
        raw_predictions = list(api_payload)

    default_patch_id = requested_patch_ids[0] if len(requested_patch_ids) == 1 else None
    for item in raw_predictions:
        patch_id = _patch_id_from_prediction(item) if isinstance(item, dict) else default_patch_id
        if patch_id is None:
            patch_id = default_patch_id
        key = _patch_key(patch_id)
        label = _label_from_conch_prediction(item)
        if key and label:
            predictions[key] = label
            crc_label = _crc_label_from_conch_prediction(item)
            detail = {
                "conch_region_semantic": label,
                "conch_crc_label": crc_label,
                "conch_probs": _prediction_scores(item),
                "conch_confidence": _prediction_confidence(item, label),
                "conch_raw_prediction": item,
            }
            if isinstance(item, dict):
                if item.get("embedding_ref") is not None:
                    detail["embedding_ref"] = item.get("embedding_ref")
                if item.get("feature_ref") is not None:
                    detail["feature_ref"] = item.get("feature_ref")
            prediction_details[key] = detail
        else:
            invalid.append({"patch_id": patch_id, "payload": item})
    return {"predictions": predictions, "prediction_details": prediction_details, "invalid_predictions": invalid}


def parse_digepath_prediction_payload(api_payload, requested_patch_ids=None):
    requested_patch_ids = requested_patch_ids or []
    predictions = {}
    prediction_details = {}
    invalid = []
    raw_predictions = []
    if isinstance(api_payload, dict) and isinstance(api_payload.get("predictions"), list):
        raw_predictions = list(api_payload.get("predictions", []))
    elif isinstance(api_payload, dict):
        raw_predictions = [api_payload]
    elif isinstance(api_payload, list):
        raw_predictions = list(api_payload)

    default_patch_id = requested_patch_ids[0] if len(requested_patch_ids) == 1 else None
    for item in raw_predictions:
        patch_id = _patch_id_from_prediction(item) if isinstance(item, dict) else default_patch_id
        if patch_id is None:
            patch_id = default_patch_id
        key = _patch_key(patch_id)
        label = _label_from_digepath_prediction(item)
        digepath_class = _digepath_class_from_prediction(item)
        if key and label:
            predictions[key] = label
            detail = {
                "digepath_class": digepath_class,
                "digepath_region_semantic": label,
                "digepath_probs": _prediction_scores(item),
                "digepath_confidence": _prediction_confidence(item, label),
                "digepath_raw_prediction": item,
            }
            prediction_details[key] = detail
        else:
            invalid.append({"patch_id": patch_id, "payload": item})
    return {"predictions": predictions, "prediction_details": prediction_details, "invalid_predictions": invalid}


def request_conch_patch_predictions(patch_requests, bundle):
    config = bundle["runtime"].get("backends", {}).get("local_conch", {})
    if not config.get("enabled"):
        return {
            "enabled": False,
            "predictions": {},
            "prediction_details": {},
            "attempts": [],
            "invalid_predictions": [],
            "errors": [],
        }
    server_url = str(config.get("server_url", "http://127.0.0.1:8200/predict")).strip()
    timeout_seconds = int(config.get("timeout_seconds", 60))
    fail_policy = str(bundle["runtime"].get("trace", {}).get("conch_fail_policy", "soft_fail_skip_patch"))
    try:
        import requests
    except Exception as exc:
        return {
            "enabled": True,
            "predictions": {},
            "prediction_details": {},
            "attempts": [],
            "invalid_predictions": [],
            "errors": ["requests is required for local_conch HTTP mode: {0}".format(exc)],
            "server_url": server_url,
            "fail_policy": fail_policy,
        }

    predictions = {}
    prediction_details = {}
    attempts = []
    invalid_predictions = []
    errors = []
    metadata_keys = (
        "classifier_source_image",
        "classifier_source_mode",
        "classifier_crop_level0_bbox",
        "classifier_crop_level0_size",
        "classifier_crop_output_size",
        "classifier_crop_view",
    )
    for patch in patch_requests:
        patch_id = list(patch.get("patch_id", []))
        image_path = str(patch.get("image_path", ""))
        classifier_metadata = {key: patch.get(key) for key in metadata_keys if patch.get(key) is not None}
        classifier_source_image = str(classifier_metadata.get("classifier_source_image", ""))
        classifier_source_mode = str(classifier_metadata.get("classifier_source_mode", ""))
        key = _patch_key(patch_id)
        if not key or not image_path:
            invalid_predictions.append({"patch_id": patch_id, "payload": {"image_path": image_path}})
            continue
        request_payload = {
            "image_path": image_path,
            "image_paths": [image_path],
            "patch_id": patch_id,
            "task": "global_screening_patch_classification",
            "labels": list(TRACE_LABEL_RUBRIC.keys()),
            "crc100k_labels": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
        }
        request_payload.update(classifier_metadata)
        started = time.time()
        try:
            response = requests.post(server_url, json=request_payload, timeout=timeout_seconds)
            round_trip_ms = int(round((time.time() - started) * 1000.0))
            attempt = {"patch_id": patch_id, "status": response.status_code, "latency_ms": round_trip_ms}
            attempt.update(classifier_metadata)
            attempts.append(attempt)
            if response.status_code != 200:
                errors.append("CONCH HTTP {0} for patch {1}: {2}".format(response.status_code, patch_id, response.text))
                continue
            parsed = parse_conch_prediction_payload(response.json(), requested_patch_ids=[patch_id])
            for detail in parsed.get("prediction_details", {}).values():
                detail.update(classifier_metadata)
            predictions.update(parsed["predictions"])
            prediction_details.update(parsed.get("prediction_details", {}))
            invalid_predictions.extend(parsed["invalid_predictions"])
        except requests.Timeout:
            errors.append("CONCH request timed out after {0}s for patch {1}".format(timeout_seconds, patch_id))
        except Exception as exc:
            errors.append("CONCH request failed for patch {0}: {1}".format(patch_id, str(exc)))
    return {
        "enabled": True,
        "server_url": server_url,
        "timeout_seconds": timeout_seconds,
        "fail_policy": fail_policy,
        "predictions": predictions,
        "prediction_details": prediction_details,
        "attempts": attempts,
        "invalid_predictions": invalid_predictions,
        "errors": errors,
    }


def request_digepath_patch_predictions(patch_requests, bundle):
    config = bundle["runtime"].get("backends", {}).get("local_digepath", {})
    if not config.get("enabled"):
        return {
            "enabled": False,
            "predictions": {},
            "prediction_details": {},
            "attempts": [],
            "invalid_predictions": [],
            "errors": [],
        }
    server_url = str(config.get("server_url", "http://127.0.0.1:8300/predict")).strip()
    timeout_seconds = int(config.get("timeout_seconds", 60))
    trace_config = bundle["runtime"].get("trace", {})
    fail_policy = str(trace_config.get("digepath_fail_policy", "soft_fail_skip_patch"))
    high_confidence_threshold = float(trace_config.get("digepath_high_confidence_threshold", 0.70))
    normal_conch_confidence_threshold = float(trace_config.get("normal_conch_confidence_threshold", 0.90))
    normal_digepath_confidence_threshold = float(trace_config.get("normal_digepath_confidence_threshold", 0.90))
    normal_gate_fallback_label = str(trace_config.get("normal_gate_fallback_label", TRACE_NEUTRAL_UNCERTAIN_LABEL))
    try:
        import requests
    except Exception as exc:
        return {
            "enabled": True,
            "predictions": {},
            "prediction_details": {},
            "attempts": [],
            "invalid_predictions": [],
            "errors": ["requests is required for local_digepath HTTP mode: {0}".format(exc)],
            "server_url": server_url,
            "fail_policy": fail_policy,
            "high_confidence_threshold": high_confidence_threshold,
            "normal_conch_confidence_threshold": normal_conch_confidence_threshold,
            "normal_digepath_confidence_threshold": normal_digepath_confidence_threshold,
            "normal_gate_fallback_label": normal_gate_fallback_label,
        }

    predictions = {}
    prediction_details = {}
    attempts = []
    invalid_predictions = []
    errors = []
    metadata_keys = (
        "classifier_source_image",
        "classifier_source_mode",
        "classifier_crop_level0_bbox",
        "classifier_crop_level0_size",
        "classifier_crop_output_size",
        "classifier_crop_view",
    )
    for patch in patch_requests:
        patch_id = list(patch.get("patch_id", []))
        image_path = str(patch.get("image_path", ""))
        classifier_metadata = {key: patch.get(key) for key in metadata_keys if patch.get(key) is not None}
        classifier_source_image = str(classifier_metadata.get("classifier_source_image", ""))
        classifier_source_mode = str(classifier_metadata.get("classifier_source_mode", ""))
        key = _patch_key(patch_id)
        if not key or not image_path:
            invalid_predictions.append({"patch_id": patch_id, "payload": {"image_path": image_path}})
            continue
        request_payload = {
            "image_path": image_path,
            "image_paths": [image_path],
            "patch_id": patch_id,
            "task": "digepath_roi9_patch_classification",
            "class_names": list(DIGEPATH_ROI9_CLASS_NAMES),
        }
        request_payload.update(classifier_metadata)
        started = time.time()
        try:
            response = requests.post(server_url, json=request_payload, timeout=timeout_seconds)
            round_trip_ms = int(round((time.time() - started) * 1000.0))
            attempt = {"backend": "local_digepath", "patch_id": patch_id, "status": response.status_code, "latency_ms": round_trip_ms}
            attempt.update(classifier_metadata)
            attempts.append(attempt)
            if response.status_code != 200:
                errors.append("DIgePath HTTP {0} for patch {1}: {2}".format(response.status_code, patch_id, response.text))
                continue
            parsed = parse_digepath_prediction_payload(response.json(), requested_patch_ids=[patch_id])
            for detail in parsed.get("prediction_details", {}).values():
                detail.update(classifier_metadata)
            predictions.update(parsed["predictions"])
            prediction_details.update(parsed.get("prediction_details", {}))
            invalid_predictions.extend(parsed["invalid_predictions"])
        except requests.Timeout:
            errors.append("DIgePath request timed out after {0}s for patch {1}".format(timeout_seconds, patch_id))
        except Exception as exc:
            errors.append("DIgePath request failed for patch {0}: {1}".format(patch_id, str(exc)))
    return {
        "enabled": True,
        "server_url": server_url,
        "timeout_seconds": timeout_seconds,
        "fail_policy": fail_policy,
        "high_confidence_threshold": high_confidence_threshold,
        "normal_conch_confidence_threshold": normal_conch_confidence_threshold,
        "normal_digepath_confidence_threshold": normal_digepath_confidence_threshold,
        "normal_gate_fallback_label": normal_gate_fallback_label,
        "predictions": predictions,
        "prediction_details": prediction_details,
        "attempts": attempts,
        "invalid_predictions": invalid_predictions,
        "errors": errors,
    }


def fuse_trace_patch_assignments_with_conch(patho_payload, conch_predictions):
    fused_payload = dict(patho_payload or {})
    patches = []
    for patch in list((patho_payload or {}).get("patches", [])):
        fused_patch = dict(patch)
        patho_label = _normalize_conch_label(
            fused_patch.get("pathoreasoner_r1_region_semantic") or fused_patch.get("region_semantic")
        )
        key = _patch_key(fused_patch.get("patch_id"))
        conch_label = _normalize_conch_label((conch_predictions or {}).get(key))
        if patho_label:
            fused_patch["pathoreasoner_r1_region_semantic"] = patho_label
        if conch_label and patho_label:
            expected = derive_global_screening_fusion(conch_label, patho_label)
            if expected:
                fused_patch.update(expected)
        else:
            fused_patch["agreement_status"] = GLOBAL_SCREENING_SINGLE_MODEL_STATUS
            fused_patch["score_origin"] = GLOBAL_SCREENING_LEGACY_SCORE_ORIGIN
            fused_patch["conch_region_semantic"] = "not_available_in_this_run"
            if patho_label:
                fused_patch["pathoreasoner_r1_region_semantic"] = patho_label
            if not str(fused_patch.get("fusion_reasoning", "")).strip():
                fused_patch["fusion_reasoning"] = "Single-model trace output retained because CONCH classification was unavailable."
        patches.append(fused_patch)
    fused_payload["patches"] = patches
    return fused_payload


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
        max_new_tokens = _resolve_qwen_stage_max_new_tokens(bundle, request["stage"], config)
        response_payload = {
            "image_path": request.get("images", [None])[0] if request.get("images") else None,
            "image_paths": list(request.get("images", [])),
            "prompt": prompt_text,
            "max_new_tokens": int(max_new_tokens),
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

        output, repair_attempts = _parse_or_repair_local_cpathagent_qwen_output(
            request["stage"],
            generated_text,
            request,
            bundle,
            prompt_text,
        )
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
        if repair_attempts:
            runner_metadata["repair_attempts"] = repair_attempts
            runner_metadata["repair_status"] = "repaired"
        output["runner_metadata"] = runner_metadata
        return {
            "backend": self.name,
            "output": output,
            "raw_text": generated_text,
            "raw_texts": [],
            "trace_attempts": output.get("trace_attempts", []),
            "repair_attempts": repair_attempts,
            "latency_ms": round_trip_ms,
            "runtime_metadata": runner_metadata,
        }


def _resolve_qwen_stage_max_new_tokens(bundle, stage, backend_config):
    default_value = int(backend_config.get("max_new_tokens", 512))
    runtime = bundle.get("runtime", {})
    if stage == "trace":
        return int(runtime.get("trace", {}).get("qwen_max_new_tokens", default_value))
    if stage == "navigate":
        return int(runtime.get("navigate", {}).get("qwen_max_new_tokens", default_value))
    if stage == "observe_step":
        return int(runtime.get("observe", {}).get("observe_step_qwen_max_new_tokens", default_value))
    if stage == "observe_report":
        return int(runtime.get("observe", {}).get("observe_report_qwen_max_new_tokens", default_value))
    return default_value


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
        max_intra_cell_zoom_targets = int(bundle["budget"].get("max_intra_cell_zoom_targets", 3))
        for cluster in clusters:
            if int(cluster["s"]) <= 0 or cluster["l"] in {TRACE_BACKGROUND_LABEL, TRACE_NEUTRAL_BACKGROUND_LABEL}:
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
            if cluster["l"] in {TRACE_NEUTRAL_EPITHELIAL_LABEL, TRACE_NEUTRAL_MUCUS_LABEL, TRACE_NEUTRAL_UNCERTAIN_LABEL}:
                step_specs = [
                    (
                        5.0,
                        "morphology_resolution_assessment",
                        "morphology_resolution",
                        "Resolve this CONCH-selected epithelial/mucus-rich review target without committing to serrated or conventional branch at Trace.",
                    ),
                    (
                        10.0,
                        "morphology_resolution_assessment",
                        "morphology_resolution",
                        "Inspect epithelial architecture at higher magnification and decide whether serrated, conventional adenoma, inflammatory/reactive, or benign morphology is supported.",
                    ),
                ]
            elif cluster["l"] in {TRACE_NORMAL_LABEL, TRACE_NEUTRAL_NORMAL_LABEL}:
                step_specs = [
                    (
                        2.5,
                        "normal_overview_assessment",
                        "normal_overview",
                        "Confirm this low-priority mucosal patch is reviewable normal mucosa and does not hide meaningful serrated or adenomatous change.",
                    )
                ]
            elif cluster["l"] in TRACE_SERRATED_LABELS:
                low_power_note = (
                    "Inspect this highest-priority serrated candidate first and confirm serrated mucosa context at 2.5x."
                    if int(cluster.get("s", 0)) >= 5
                    else "Confirm that this patch belongs to the serrated mucosa pathway and merits directed follow-up."
                )
                step_specs = [
                    (
                        2.5,
                        "serrated_overview_assessment",
                        "serrated_overview",
                        low_power_note,
                    ),
                    (
                        5.0,
                        "ssl_assessment",
                        "ssl_architecture",
                        "Assess SSL architectural distortion at 5x, focusing on basal crypt deformation and crypt branching.",
                    ),
                    (
                        5.0,
                        "hp_assessment",
                        "hp_architecture",
                        "Assess HP architecture at 5x, focusing on surface-limited serration, straight crypt bases, and lack of basal architectural distortion.",
                    ),
                    (
                        5.0,
                        "tsa_assessment",
                        "tsa_architecture",
                        "Assess TSA architecture and low-power cytologic pattern at 5x, including ectopic crypt foci, slit-like serration, global eosinophilic color shift, and epithelial banding.",
                    ),
                ]
            elif cluster["l"] in TRACE_CONVENTIONAL_LABELS:
                step_specs = [
                    (
                        2.5,
                        "conventional_overview_assessment",
                        "conventional_overview",
                        "Confirm non-serrated lesion context and conventional adenoma candidacy at 2.5x.",
                    ),
                    (
                        5.0,
                        "conventional_architecture_assessment",
                        "conventional_architecture",
                        "Review conventional adenoma architecture at 5x and estimate tubular versus tubulovillous/villous component.",
                    ),
                    (
                        5.0,
                        "reactive_regenerative_assessment",
                        "reactive_regenerative",
                        "Review reactive/regenerative mimic features at 5x, including erosion, inflammation, regenerative change, and lack of adenomatous or serrated architecture.",
                    ),
                ]
            elif cluster["l"] in {TRACE_INFLAMMATORY_LABEL, TRACE_NEUTRAL_INFLAMMATORY_LABEL}:
                step_specs = [
                    (
                        2.5,
                        "normal_overview_assessment",
                        "normal_overview",
                        "Confirm low-priority mucosa and decide whether inflammatory/reactive follow-up is needed.",
                    ),
                    (
                        5.0,
                        "inflammatory_reactive_assessment",
                        "inflammatory_reactive",
                        "Confirm inflammatory polyp-like or reactive features and keep this region out of the dysplasia branch unless later evidence contradicts the overview.",
                    )
                ]
            else:
                step_specs = [
                    (
                        2.5,
                        "normal_overview_assessment",
                        "normal_overview",
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
                planned_targets = []
                for magnification, review_goal, stage_gate, need_to_see in step_specs:
                    if float(magnification) == 10.0 and _cluster_needs_multi_zoom(cluster) and review_goal in {"abnormal_crypt_assessment", "conventional_adenoma_assessment"}:
                        role_points = _intra_cell_candidate_points(center_x, center_y, patch, max_intra_cell_zoom_targets)
                        for target_index, (target_x, target_y) in enumerate(role_points):
                            planned_targets.append(
                                {
                                    "x": target_x,
                                    "y": target_y,
                                    "m": magnification,
                                    "review_goal": review_goal,
                                    "stage_gate": stage_gate,
                                    "need_to_see": need_to_see,
                                    "target_index": target_index,
                                    "target_count": len(role_points),
                                    "coordinate_source": "trace_anchor" if target_index == 0 and anchor_x is not None and anchor_y is not None else "heuristic_offset",
                                }
                            )
                    else:
                        planned_targets.append(
                            {
                                "x": center_x,
                                "y": center_y,
                                "m": magnification,
                                "review_goal": review_goal,
                                "stage_gate": stage_gate,
                                "need_to_see": need_to_see,
                                "target_index": 0,
                                "target_count": 1,
                                "coordinate_source": "trace_anchor" if anchor_x is not None and anchor_y is not None else "patch_center_fallback",
                            }
                        )
                for target in planned_targets:
                    if step_index >= max_steps:
                        break
                    magnification = float(target["m"])
                    region_size = int(mag_to_region.get(str(float(magnification)), 256))
                    from adenoma_agent.utils import bbox_overlap_ratio, clamp_center_point, normalized_point

                    step_bbox = {
                        "x1": int(target["x"]) - region_size // 2,
                        "y1": int(target["y"]) - region_size // 2,
                        "x2": int(target["x"]) + region_size // 2,
                        "y2": int(target["y"]) + region_size // 2,
                    }
                    should_skip = False
                    patch_id = list(patch.get("patch_id", []))
                    for prior_item in prior_windows:
                        if prior_item.get("patch_id") != patch_id:
                            continue
                        if abs(float(prior_item["m"]) - float(magnification)) > 1e-6:
                            continue
                        if prior_item.get("review_goal") != target["review_goal"]:
                            continue
                        if bbox_overlap_ratio(step_bbox, prior_item["bbox"]) > overlap_threshold:
                            should_skip = True
                            break
                    if should_skip:
                        continue
                    fixed_x, fixed_y = clamp_center_point(target["x"], target["y"], region_size, slide_dims)
                    step_bbox = {
                        "x1": fixed_x - region_size // 2,
                        "y1": fixed_y - region_size // 2,
                        "x2": fixed_x + region_size // 2,
                        "y2": fixed_y + region_size // 2,
                    }
                    prior_windows.append({"bbox": step_bbox, "m": float(magnification), "patch_id": patch_id, "review_goal": target["review_goal"]})
                    steps.append(
                        {
                            "step_id": "step_{0:02d}".format(step_index),
                            "x": fixed_x,
                            "y": fixed_y,
                            "m": float(magnification),
                            "region_size_level0": region_size,
                            "need_to_see": target["need_to_see"],
                            "review_goal": target["review_goal"],
                            "stage_gate": target["stage_gate"],
                            "metadata": {
                                "cluster_id": cluster["cluster_id"],
                                "source_group_id": cluster["cluster_id"],
                                "cluster_label": cluster["l"],
                                "cluster_priority": cluster["s"],
                                "cell_priority": cluster["s"],
                                "cell_id": _navigation_cell_id(patch.get("patch_id", []), cluster["cluster_id"]),
                                "patch_id": list(patch.get("patch_id", [])),
                                "patch_index": patch_index,
                                "cluster_patch_count": len(patch_sequence),
                                "intra_cell_target_index": int(target["target_index"]),
                                "intra_cell_target_role": _navigation_target_role(cluster["l"], target["review_goal"], target["target_index"]),
                                "intra_cell_target_count": int(target["target_count"]),
                                "coordinate_source": target["coordinate_source"],
                                "region_size_level0": region_size,
                                "normalized_center": normalized_point(fixed_x, fixed_y, slide_dims),
                                "anchor_source": patch.get("anchor_source"),
                                "action": "inspect",
                                "workflow_branch": branch,
                                "routing_hint": cluster.get("metadata", {}).get("routing_hint"),
                                "candidate_branches": cluster.get("metadata", {}).get("candidate_branches", []),
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
                    "m": 2.5,
                    "region_size_level0": 4096,
                    "need_to_see": "Stop navigation because no reviewable lesion cluster was retained.",
                    "review_goal": "integrated_impression",
                    "stage_gate": "end",
                    "metadata": {"action": "stop", "region_size_level0": 4096},
                }
            )
        else:
            last = steps[-1]
            steps.append(
                {
                    "step_id": "step_{0:02d}".format(len(steps)),
                    "x": last["x"],
                    "y": last["y"],
                    "m": 2.5,
                    "region_size_level0": 4096,
                    "need_to_see": "Stop navigation and consolidate serrated, conventional adenoma, inflammatory, and branch-specific dysplasia evidence gathered so far.",
                    "review_goal": "integrated_impression",
                    "stage_gate": "end",
                    "metadata": {"action": "stop", "region_size_level0": 4096},
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
        ssl_criteria = list(bundle["runtime"]["observe"].get("ssl_criteria", []))
        hp_criteria = list(bundle["runtime"]["observe"].get("hp_criteria", []))
        tsa_criteria = list(bundle["runtime"]["observe"].get("tsa_criteria", []))
        tsa_cytology_criteria = list(bundle["runtime"]["observe"].get("tsa_cytological_atypia_criteria", []))
        inflammatory_criteria = list(bundle["runtime"]["observe"].get("inflammatory_criteria", []))
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
        metadata_recovery_hint = str(step.get("metadata", {}).get("branch_recovery_hint") or cluster_metadata.get("branch_recovery_hint") or "none").strip()

        serrated_hits = _blank_hits(serrated_criteria)
        abnormal_crypt_hits = _blank_hits(abnormal_crypt_criteria)
        conventional_hits = _blank_hits(conventional_criteria)
        serrated_dysplasia_hits = _blank_hits(dysplasia_criteria)
        conventional_dysplasia_hits = _blank_hits(dysplasia_criteria)
        ssl_hits = _blank_hits(ssl_criteria)
        hp_hits = _blank_hits(hp_criteria)
        tsa_hits = _blank_hits(tsa_criteria)
        tsa_cytology_hits = _blank_hits(tsa_cytology_criteria)
        inflammatory_hits = _blank_hits(inflammatory_criteria)
        branch_recovery_hint = metadata_recovery_hint if metadata_recovery_hint in {"conventional", "normal"} else "none"
        branch_recovery_reason = ""

        if review_goal in {"serrated_overview_assessment", "serrated_lesion_assessment"}:
            if cluster.get("l") in TRACE_SERRATED_LABELS:
                serrated_context_supported = (
                    review_goal != "serrated_overview_assessment"
                    or pale_fraction > 0.10
                    or cluster_priority >= 5
                    or metadata_recovery_hint == "serrated"
                )
                serrated_hits["serrated_lesion_context"] = "supporting" if serrated_context_supported else "uncertain"
                serrated_hits["serrated_surface_pattern"] = "supporting" if pale_fraction > 0.10 or cluster_priority >= 5 else "uncertain"
                serrated_hits["mucus_rich_surface"] = "supporting" if pale_fraction > 0.18 or cluster_priority >= 5 else "uncertain"
                if cluster.get("l") in (TRACE_LEGACY_SSL_LABEL, TRACE_LEGACY_SSL_HIGH_LABEL):
                    for key in ssl_hits:
                        ssl_hits[key] = "supporting" if key in ("basal_crypt_dilatation", "crypt_branching", "serration_to_base") else "uncertain"
                elif cluster.get("l") == TRACE_HP_LABEL:
                    for key in hp_hits:
                        hp_hits[key] = "supporting"
                elif cluster.get("l") == TRACE_TSA_LABEL:
                    for key in tsa_hits:
                        tsa_hits[key] = "supporting"
            elif cluster.get("l") == TRACE_NORMAL_LABEL:
                serrated_hits["serrated_lesion_context"] = "opposing"
                serrated_hits["serrated_surface_pattern"] = "opposing" if pale_fraction < 0.08 else "uncertain"
                serrated_hits["mucus_rich_surface"] = "opposing" if pale_fraction < 0.08 else "uncertain"
            if review_goal == "serrated_overview_assessment":
                if metadata_recovery_hint in {"conventional", "normal"}:
                    branch_recovery_hint = metadata_recovery_hint
                    branch_recovery_reason = "Junior overview metadata suggests {0} after serrated evidence was insufficient.".format(metadata_recovery_hint)
                elif tissue_fraction > 0.50 and pale_fraction < 0.10:
                    branch_recovery_hint = "conventional"
                    branch_recovery_reason = "Serrated overview is not supported and tissue-rich non-pale mucosa warrants conventional overview recovery."
                elif tissue_fraction < 0.40 or background_fraction > 0.55:
                    branch_recovery_hint = "normal"
                    branch_recovery_reason = "Serrated overview is not supported and the crop appears low-priority or non-lesional."
        elif review_goal in {"ssl_assessment", "abnormal_crypt_assessment"}:
            if "basal_crypt_deformation" in ssl_hits:
                ssl_hits["basal_crypt_deformation"] = "supporting" if crypt_disorder_risk >= 4 and tissue_fraction > 0.50 else "uncertain"
            if "crypt_branching" in ssl_hits:
                ssl_hits["crypt_branching"] = "supporting" if crypt_disorder_risk >= 4 else "uncertain"
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
        elif review_goal == "hp_assessment":
            hp_context = cluster.get("l") == TRACE_HP_LABEL or (
                cluster.get("l") in TRACE_SERRATED_LABELS
                and pale_fraction > 0.10
                and crypt_disorder_risk < 4
            )
            if "surface_limited_serration" in hp_hits:
                hp_hits["surface_limited_serration"] = "supporting" if hp_context or pale_fraction > 0.12 else "uncertain"
            if "straight_crypt_bases" in hp_hits:
                hp_hits["straight_crypt_bases"] = "supporting" if hp_context and crypt_disorder_risk < 4 else "uncertain"
            if "lacks_basal_architectural_distortion" in hp_hits:
                hp_hits["lacks_basal_architectural_distortion"] = "supporting" if hp_context and crypt_disorder_risk < 4 else "uncertain"
        elif review_goal == "tsa_assessment":
            if "ectopic_crypt_foci" in tsa_hits:
                tsa_hits["ectopic_crypt_foci"] = "supporting" if cluster.get("l") == TRACE_TSA_LABEL or cluster_priority >= 5 else "uncertain"
            if "slit_like_serration" in tsa_hits:
                tsa_hits["slit_like_serration"] = "supporting" if pale_fraction > 0.12 or cluster.get("l") == TRACE_TSA_LABEL else "uncertain"
            if "global_color_shift" in tsa_hits:
                tsa_hits["global_color_shift"] = "supporting" if pale_fraction > 0.12 else "uncertain"
            if "epithelial_banding_pattern" in tsa_hits:
                tsa_hits["epithelial_banding_pattern"] = "supporting" if tissue_fraction > 0.55 and cluster_priority >= 4 else "uncertain"
        elif review_goal in {"conventional_overview_assessment", "conventional_architecture_assessment", "conventional_adenoma_assessment"}:
            conventional_context = cluster.get("l") in TRACE_CONVENTIONAL_LABELS or metadata_recovery_hint == "conventional" or step.get("metadata", {}).get("recovery_source") == "serrated_overview_negative"
            if conventional_context:
                if "tubular_architecture" in conventional_hits:
                    conventional_hits["tubular_architecture"] = "supporting"
                if "villous_component" in conventional_hits:
                    conventional_hits["villous_component"] = "supporting" if cluster.get("l") == TRACE_TUBULOVILLOUS_LABEL or conventional_subtype_hint == "tubulovillous_adenoma_like" else "opposing"
                if "high_villous_component" in conventional_hits:
                    conventional_hits["high_villous_component"] = "supporting" if cluster_metadata.get("villous_component_category") == ">75%" else "uncertain"
                conventional_hits["tubular_or_tubulovillous_architecture"] = "supporting"
                conventional_hits["crowded_adenomatous_glands"] = "supporting" if tissue_fraction > 0.45 else "uncertain"
                if cluster.get("l") == TRACE_TUBULOVILLOUS_LABEL or conventional_subtype_hint == "tubulovillous_adenoma_like":
                    conventional_hits["pencillate_hyperchromatic_nuclei"] = "supporting"
                elif tissue_fraction > 0.30:
                    conventional_hits["pencillate_hyperchromatic_nuclei"] = "uncertain"
            elif cluster.get("l") == TRACE_INFLAMMATORY_LABEL:
                conventional_hits["tubular_or_tubulovillous_architecture"] = "opposing"
                conventional_hits["crowded_adenomatous_glands"] = "opposing"
                conventional_hits["pencillate_hyperchromatic_nuclei"] = "uncertain"
                if "villous_component" in conventional_hits:
                    conventional_hits["villous_component"] = "opposing"
                if "high_villous_component" in conventional_hits:
                    conventional_hits["high_villous_component"] = "opposing"
                for key in inflammatory_hits:
                    inflammatory_hits[key] = "supporting"
        elif review_goal == "reactive_regenerative_assessment":
            reactive_context = (
                cluster.get("l") == TRACE_INFLAMMATORY_LABEL
                or cluster_metadata.get("reactive_regenerative_suspected")
                or step.get("metadata", {}).get("reactive_regenerative_suspected")
                or (tissue_fraction > 0.35 and pale_fraction < 0.08 and cluster_priority <= 3)
            )
            for key in inflammatory_hits:
                if key == "lacks_adenomatous_or_serrated_architecture":
                    inflammatory_hits[key] = "supporting" if reactive_context and cluster_priority <= 3 else "uncertain"
                else:
                    inflammatory_hits[key] = "supporting" if reactive_context else "uncertain"
        elif review_goal in {"ssl_dysplasia_assessment", "tsa_dysplasia_assessment", "serrated_dysplasia_assessment"}:
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
        elif review_goal == "tsa_cytological_atypia_assessment":
            if "cytoplasmic_eosinophilia" in tsa_cytology_hits:
                tsa_cytology_hits["cytoplasmic_eosinophilia"] = "supporting" if pale_fraction > 0.12 else "uncertain"
            if "pencillate_nuclei" in tsa_cytology_hits:
                tsa_cytology_hits["pencillate_nuclei"] = "supporting" if tissue_fraction > 0.55 else "uncertain"
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

        if review_goal in {"serrated_overview_assessment", "serrated_lesion_assessment"}:
            level_1_findings = _supporting_findings_from_hits(serrated_hits)
            level_2_findings = []
            level_3_findings = []
        elif review_goal in {"ssl_assessment", "abnormal_crypt_assessment"}:
            level_1_findings = _supporting_findings_from_hits(ssl_hits)
            level_2_findings = _supporting_findings_from_hits(abnormal_crypt_hits)
            level_3_findings = []
        elif review_goal == "tsa_assessment":
            level_1_findings = _supporting_findings_from_hits(tsa_hits)
            level_2_findings = []
            level_3_findings = []
        elif review_goal == "hp_assessment":
            level_1_findings = _supporting_findings_from_hits(hp_hits)
            level_2_findings = []
            level_3_findings = []
        elif review_goal in {"conventional_overview_assessment", "conventional_architecture_assessment", "conventional_adenoma_assessment"}:
            level_1_findings = _supporting_findings_from_hits(conventional_hits)
            level_2_findings = []
            level_3_findings = []
        elif review_goal == "reactive_regenerative_assessment":
            level_1_findings = _supporting_findings_from_hits(inflammatory_hits)
            level_2_findings = []
            level_3_findings = []
        elif review_goal in {"ssl_dysplasia_assessment", "tsa_dysplasia_assessment", "serrated_dysplasia_assessment"}:
            level_1_findings = []
            level_2_findings = []
            level_3_findings = _supporting_findings_from_hits(serrated_dysplasia_hits)
        elif review_goal == "tsa_cytological_atypia_assessment":
            level_1_findings = _supporting_findings_from_hits(tsa_cytology_hits)
            level_2_findings = []
            level_3_findings = []
        elif review_goal == "conventional_dysplasia_assessment":
            level_1_findings = []
            level_2_findings = []
            level_3_findings = _supporting_findings_from_hits(conventional_dysplasia_hits)
        else:
            level_1_findings = []
            level_2_findings = []
            level_3_findings = []
            if cluster.get("l") == TRACE_INFLAMMATORY_LABEL:
                for key in inflammatory_hits:
                    inflammatory_hits[key] = "supporting"

        if background_fraction > 0.7:
            observation = "The crop is background-heavy and provides limited diagnostic tissue."
        elif review_goal in {"serrated_overview_assessment", "serrated_lesion_assessment"}:
            observation = "2.5x overview confirms whether the retained mucosa belongs to the serrated pathway."
        elif review_goal in {"ssl_assessment", "abnormal_crypt_assessment"}:
            observation = "5x SSL assessment targets basal crypt deformation and crypt branching."
        elif review_goal == "tsa_assessment":
            observation = "5x TSA assessment targets ectopic crypt foci, slit-like serration, and low-power cytological pattern."
        elif review_goal == "hp_assessment":
            observation = "5x HP assessment targets surface-limited serration, straight crypt bases, and lack of basal architectural distortion."
        elif review_goal in {"conventional_overview_assessment", "conventional_architecture_assessment", "conventional_adenoma_assessment"}:
            observation = "This view reviews non-serrated lesion context and conventional adenoma architecture before dysplasia assessment."
        elif review_goal == "reactive_regenerative_assessment":
            observation = "5x reactive/regenerative assessment targets inflammatory injury-repair features and adenoma mimics."
        elif review_goal == "conventional_dysplasia_assessment":
            observation = "High magnification focuses on dysplasia within a conventional adenoma-like region, supported by the multi-view bundle."
        elif review_goal in {"ssl_dysplasia_assessment", "tsa_dysplasia_assessment", "serrated_dysplasia_assessment"}:
            observation = "High magnification focuses on high-grade or definite dysplasia after serrated subtype support has been established."
        elif review_goal == "tsa_cytological_atypia_assessment":
            observation = "High magnification confirms TSA cytological atypia with eosinophilic cytoplasm and pencillate nuclei."
        else:
            observation = "This overview confirms a low-priority non-serrated or inflammatory region."

        if review_goal in {"serrated_overview_assessment", "serrated_lesion_assessment"}:
            stage_decision = "supports_serrated_lesion" if level_1_findings else "leans_non_serrated_or_indeterminate"
            reasoning = "This view decides whether retained mucosa belongs to the serrated pathway before abnormal crypt review."
            next_step = (
                "Proceed to 5x SSL and TSA assessment." if cluster.get("d") else "Consolidate as a non-serrated or low-priority serrated mucosal region."
            )
            if review_goal == "serrated_overview_assessment":
                stage_decision = "supports_serrated_overview" if level_1_findings else "serrated_overview_not_supported_or_indeterminate"
                if stage_decision == "supports_serrated_overview":
                    branch_recovery_hint = "none"
                    branch_recovery_reason = ""
        elif review_goal in {"ssl_assessment", "abnormal_crypt_assessment"}:
            stage_decision = (
                "ssl_architecture_supported" if review_goal == "ssl_assessment" and level_1_findings else
                "supports_abnormal_crypt"
                if level_2_findings
                else ("ssl_architecture_not_supported_or_indeterminate" if review_goal == "ssl_assessment" else "serrated_but_no_support_for_abnormal_crypt")
            )
            reasoning = "This view evaluates whether the crypt pattern supports SSL architectural distortion within the serrated pathway."
            next_step = (
                "Proceed to 10x SSL dysplasia review."
                if stage_decision in {"supports_abnormal_crypt", "ssl_architecture_supported"}
                else "Do not enter SSL dysplasia review because SSL architectural support is not established."
            )
        elif review_goal == "tsa_assessment":
            stage_decision = "tsa_architecture_supported" if level_1_findings else "tsa_architecture_not_supported_or_indeterminate"
            reasoning = "This view evaluates TSA architecture and low-power cytological atypia cues; cytology support alone is not TSAD."
            next_step = "Proceed to 10x TSA cytology confirmation and dysplasia review if TSA support persists."
        elif review_goal == "hp_assessment":
            stage_decision = "hp_architecture_supported" if level_1_findings else "hp_architecture_not_supported_or_indeterminate"
            reasoning = "This view evaluates HP morphology within the serrated pathway without assigning any dysplasia suffix."
            next_step = "Use HP evidence only if SSL and TSA support remain absent; do not trigger dysplasia review from HP alone."
        elif review_goal in {"conventional_overview_assessment", "conventional_architecture_assessment", "conventional_adenoma_assessment"}:
            stage_decision = (
                "supports_conventional_overview" if review_goal == "conventional_overview_assessment" and level_1_findings else
                "supports_conventional_architecture" if review_goal == "conventional_architecture_assessment" and level_1_findings else
                "supports_conventional_adenoma"
                if level_1_findings
                else (
                    "conventional_overview_not_supported_or_indeterminate" if review_goal == "conventional_overview_assessment" else
                    "conventional_architecture_not_supported_or_indeterminate" if review_goal == "conventional_architecture_assessment" else
                    "conventional_adenoma_indeterminate_or_opposed"
                )
            )
            reasoning = "This view evaluates whether the region belongs to the conventional adenoma branch before its own dysplasia review."
            next_step = "Proceed to conventional dysplasia review for this adenoma-like branch."
        elif review_goal == "reactive_regenerative_assessment":
            stage_decision = "reactive_regenerative_supported" if level_1_findings else "reactive_regenerative_not_supported_or_indeterminate"
            reasoning = "This view evaluates inflammatory/reactive mimic evidence within the conventional pathway."
            next_step = "Use this as inflammatory/reactive support when conventional architecture is not established; otherwise record it as a conflict."
        elif review_goal in {"ssl_dysplasia_assessment", "tsa_dysplasia_assessment", "serrated_dysplasia_assessment"}:
            stage_decision = (
                "ssl_dysplasia_supported" if review_goal == "ssl_dysplasia_assessment" and level_3_findings else
                "tsa_dysplasia_supported" if review_goal == "tsa_dysplasia_assessment" and level_3_findings else
                "serrated_dysplasia_supported"
                if level_3_findings
                else (
                    "ssl_dysplasia_not_supported_or_indeterminate" if review_goal == "ssl_dysplasia_assessment" else
                    "tsa_dysplasia_not_supported_or_indeterminate" if review_goal == "tsa_dysplasia_assessment" else
                    "serrated_dysplasia_not_supported_or_indeterminate"
                )
            )
            reasoning = "This view evaluates dysplasia specifically within the serrated branch after abnormal crypt support."
            next_step = "Integrate serrated, abnormal crypt, and serrated dysplasia evidence into the final report."
        elif review_goal == "tsa_cytological_atypia_assessment":
            stage_decision = "tsa_cytological_atypia_supported" if level_1_findings else "tsa_cytological_atypia_not_supported_or_indeterminate"
            reasoning = "This view confirms TSA cytological atypia; it supports TSA lineage but does not by itself establish TSAD."
            next_step = "Proceed to TSA dysplasia review only if high-grade or definite dysplasia remains suspected."
        elif review_goal == "conventional_dysplasia_assessment":
            stage_decision = (
                "conventional_dysplasia_supported"
                if level_3_findings
                else "conventional_dysplasia_not_supported_or_indeterminate"
            )
            reasoning = "This view evaluates dysplasia specifically within the conventional adenoma branch."
            next_step = "Integrate conventional adenoma and branch-specific dysplasia evidence into the final report."
        else:
            if review_goal == "normal_overview_assessment":
                stage_decision = "supports_normal_overview" if cluster.get("l") != TRACE_BACKGROUND_LABEL else "normal_overview_not_supported_or_indeterminate"
            elif review_goal == "inflammatory_reactive_assessment":
                stage_decision = "inflammatory_reactive_supported" if _supporting_findings_from_hits(inflammatory_hits) else "inflammatory_reactive_not_supported_or_indeterminate"
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
            "ssl_hits": ssl_hits,
            "hp_hits": hp_hits,
            "tsa_hits": tsa_hits,
            "tsa_cytological_atypia_hits": tsa_cytology_hits,
            "inflammatory_hits": inflammatory_hits,
            "branch_recovery_hint": branch_recovery_hint,
            "branch_recovery_reason": branch_recovery_reason,
            "view_count": view_count,
            "confidence": round(confidence, 4),
        }

    def _observe_report_output(self, request, bundle):
        serrated_criteria = list(bundle["runtime"]["observe"].get("serrated_criteria", []))
        abnormal_crypt_criteria = list(bundle["runtime"]["observe"].get("abnormal_crypt_criteria", []))
        conventional_criteria = list(bundle["runtime"]["observe"].get("conventional_adenoma_criteria", []))
        dysplasia_criteria = list(bundle["runtime"]["observe"].get("dysplasia_criteria", []))
        ssl_criteria = list(bundle["runtime"]["observe"].get("ssl_criteria", []))
        hp_criteria = list(bundle["runtime"]["observe"].get("hp_criteria", []))
        tsa_criteria = list(bundle["runtime"]["observe"].get("tsa_criteria", []))
        tsa_cytology_criteria = list(bundle["runtime"]["observe"].get("tsa_cytological_atypia_criteria", []))
        inflammatory_criteria = list(bundle["runtime"]["observe"].get("inflammatory_criteria", []))
        records = request["metadata"]["records"]
        trace_clusters = request["metadata"]["trace_clusters"]
        global_reviews = request["metadata"].get("global_reviews", [])
        resolved_branch = None
        branch_correction_reason = ""
        for review in reversed(global_reviews if isinstance(global_reviews, list) else []):
            metadata = review.get("metadata", {}) if isinstance(review, dict) else {}
            state = review.get("resolved_branch_state", {}) if isinstance(review, dict) else {}
            branch_correction_reason = str(review.get("branch_correction_reason") or metadata.get("branch_correction_reason") or branch_correction_reason or "").strip()
            for candidate_branch in ("serrated", "conventional_adenoma", "conventional", "normal", "background"):
                if str(state.get(candidate_branch, "")).strip().lower() == "supported":
                    resolved_branch = "conventional_adenoma" if candidate_branch == "conventional" else candidate_branch
                    break
            if resolved_branch:
                break
        trace_branch_for_correction = _case_trace_branch(trace_clusters)
        if trace_branch_for_correction == "serrated" and resolved_branch in {"conventional_adenoma", "normal"}:
            alternate_supported = False
            for record in records:
                metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
                review_goal = str(metadata.get("review_goal") or "").strip()
                stage_decision = str(record.get("stage_decision") or metadata.get("stage_decision") or "").strip()
                if resolved_branch == "conventional_adenoma" and review_goal in {"conventional_overview_assessment", "conventional_architecture_assessment"} and stage_decision in {"supports_conventional_overview", "supports_conventional_architecture", "supports_conventional_adenoma"}:
                    alternate_supported = True
                    break
                if resolved_branch == "normal" and review_goal == "normal_overview_assessment" and stage_decision == "supports_normal_overview":
                    alternate_supported = True
                    break
            if not alternate_supported:
                resolved_branch = None
                branch_correction_reason = ""

        serrated_checklist = _aggregate_hits(records, "serrated_hits", serrated_criteria)
        abnormal_crypt_checklist = _aggregate_hits(records, "abnormal_crypt_hits", abnormal_crypt_criteria)
        conventional_adenoma_checklist = _aggregate_hits(records, "conventional_hits", conventional_criteria)
        serrated_dysplasia_checklist = _aggregate_hits(records, "serrated_dysplasia_hits", dysplasia_criteria)
        conventional_dysplasia_checklist = _aggregate_hits(records, "conventional_dysplasia_hits", dysplasia_criteria)
        ssl_checklist = _aggregate_hits(records, "ssl_hits", ssl_criteria)
        hp_checklist = _aggregate_hits(records, "hp_hits", hp_criteria)
        tsa_checklist = _aggregate_hits(records, "tsa_hits", tsa_criteria)
        tsa_cytological_atypia_checklist = _aggregate_hits(records, "tsa_cytological_atypia_hits", tsa_cytology_criteria)
        inflammatory_checklist = _aggregate_hits(records, "inflammatory_hits", inflammatory_criteria)
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
        serrated_dysplasia_gate_assessment = dict(abnormal_crypt_assessment)
        if _supporting_count_from_checklist(ssl_checklist) >= 2 or _supporting_count_from_checklist(tsa_checklist) >= 2:
            serrated_dysplasia_gate_assessment.update(
                {
                    "label": "serrated_subtype_gate_supported",
                    "positive": True,
                    "score": max(float(abnormal_crypt_assessment.get("score", 0.0)), 0.8),
                }
            )
        serrated_dysplasia_assessment = _branch_dysplasia_assessment(
            serrated_dysplasia_gate_assessment,
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
        ssl_assessment = {
            "label": "ssl_architecture_supported" if _supporting_count_from_checklist(ssl_checklist) >= 2 else "ssl_architecture_not_supported_or_indeterminate",
            "positive": _supporting_count_from_checklist(ssl_checklist) >= 2,
            "score": round(min(1.0, _supporting_count_from_checklist(ssl_checklist) / 2.0), 4),
        }
        hp_assessment = {
            "label": "hp_architecture_supported" if _supporting_count_from_checklist(hp_checklist) >= 2 else "hp_architecture_not_supported_or_indeterminate",
            "positive": _supporting_count_from_checklist(hp_checklist) >= 2,
            "score": round(min(1.0, _supporting_count_from_checklist(hp_checklist) / 2.0), 4),
        }
        tsa_assessment = {
            "label": "tsa_architecture_supported" if _supporting_count_from_checklist(tsa_checklist) >= 2 else "tsa_architecture_not_supported_or_indeterminate",
            "positive": _supporting_count_from_checklist(tsa_checklist) >= 2,
            "score": round(min(1.0, _supporting_count_from_checklist(tsa_checklist) / 2.0), 4),
        }
        reactive_regenerative_assessment = {
            "label": "reactive_regenerative_supported" if _supporting_count_from_checklist(inflammatory_checklist) >= 2 else "reactive_regenerative_not_supported_or_indeterminate",
            "positive": _supporting_count_from_checklist(inflammatory_checklist) >= 2,
            "score": round(min(1.0, _supporting_count_from_checklist(inflammatory_checklist) / 2.0), 4),
        }
        tsa_cytological_atypia_assessment = {
            "label": "tsa_cytological_atypia_supported" if _supporting_count_from_checklist(tsa_cytological_atypia_checklist) >= 1 else "tsa_cytological_atypia_not_supported_or_indeterminate",
            "positive": _supporting_count_from_checklist(tsa_cytological_atypia_checklist) >= 1,
            "score": round(min(1.0, _supporting_count_from_checklist(tsa_cytological_atypia_checklist) / 2.0), 4),
        }
        final_case_assessment = _final_case_assessment(
            serrated_assessment,
            serrated_dysplasia_assessment,
            conventional_adenoma_assessment,
            conventional_dysplasia_assessment,
        )
        final_11_case_assessment = _class11_assessment(
            trace_clusters,
            serrated_assessment,
            serrated_dysplasia_assessment,
            conventional_adenoma_assessment,
            conventional_dysplasia_assessment,
            ssl_checklist,
            hp_checklist,
            tsa_checklist,
            inflammatory_checklist,
            conventional_adenoma_checklist,
            resolved_branch=resolved_branch,
            branch_correction_reason=branch_correction_reason,
        )
        final_case_assessment = {
            **final_case_assessment,
            **final_11_case_assessment,
            "legacy_label": final_case_assessment.get("label"),
            "coexisting_candidates": final_case_assessment.get("coexisting_labels", []),
        }
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
        lines.append("- Branch: {0}".format(final_case_assessment["branch"]))
        lines.append("- Subtype: {0}".format(final_case_assessment["subtype"]))
        lines.append("- Classification status: {0}".format(final_case_assessment.get("classification_status", "classified")))
        if final_case_assessment.get("non_diagnostic_reason"):
            lines.append("- Non-diagnostic reason: {0}".format(final_case_assessment["non_diagnostic_reason"]))
        lines.append("- High-grade/definite dysplasia: {0}".format(final_case_assessment.get("high_grade_or_definite_dysplasia", False)))
        lines.append("- Villous component category: {0}".format(final_case_assessment.get("villous_component_category", "not_assessed")))
        if final_case_assessment.get("coexisting_labels"):
            lines.append("- Coexisting labels: {0}".format(", ".join(final_case_assessment["coexisting_labels"])))
        lines.append("")
        lines.append("Integrated impression:")
        lines.append("- {0}".format(integrated_impression))
        return {
            "hierarchical_prediction": {
                "serrated_lesion_assessment": serrated_assessment,
                "ssl_assessment": ssl_assessment,
                "hp_assessment": hp_assessment,
                "tsa_assessment": tsa_assessment,
                "tsa_cytological_atypia_assessment": tsa_cytological_atypia_assessment,
                "reactive_regenerative_assessment": reactive_regenerative_assessment,
                "abnormal_crypt_assessment": abnormal_crypt_assessment,
                "conventional_adenoma_assessment": conventional_adenoma_assessment,
                "serrated_dysplasia_assessment": serrated_dysplasia_assessment,
                "conventional_dysplasia_assessment": conventional_dysplasia_assessment,
                "dysplasia_assessment": dysplasia_assessment,
                "final_case_assessment": final_case_assessment,
                "primary_branch": final_case_assessment["branch"],
                "subtype_prediction": {
                    "label": final_case_assessment["subtype"],
                    "confidence": final_case_assessment["confidence"],
                },
                "dysplasia_status": {
                    "positive": final_case_assessment.get("high_grade_or_definite_dysplasia", False),
                    "high_grade_or_definite": final_case_assessment.get("high_grade_or_definite_dysplasia", False),
                    "serrated_positive": bool(serrated_dysplasia_assessment.get("positive")),
                    "conventional_positive": bool(conventional_dysplasia_assessment.get("positive")),
                },
                "final_11_class": final_case_assessment["label"],
                "classification_status": final_case_assessment.get("classification_status", "classified"),
                "non_diagnostic_reason": final_case_assessment.get("non_diagnostic_reason", ""),
                "coexisting_candidates": final_case_assessment.get("coexisting_candidates", []),
                "class_scores": final_case_assessment.get("class_scores", {}),
                "decision_path": [
                    "trace_region_semantic",
                    "branch_specific_navigation",
                    "subtype_checklist_aggregation",
                    "branch_dysplasia_gate",
                    "final_11_class_mapping",
                ],
                "integrated_impression": integrated_impression,
            },
            "serrated_checklist": serrated_checklist,
            "abnormal_crypt_checklist": abnormal_crypt_checklist,
            "conventional_adenoma_checklist": conventional_adenoma_checklist,
            "serrated_dysplasia_checklist": serrated_dysplasia_checklist,
            "conventional_dysplasia_checklist": conventional_dysplasia_checklist,
            "dysplasia_checklist": dysplasia_checklist,
            "ssl_checklist": ssl_checklist,
            "hp_checklist": hp_checklist,
            "tsa_checklist": tsa_checklist,
            "tsa_cytological_atypia_checklist": tsa_cytological_atypia_checklist,
            "conventional_architecture_checklist": conventional_adenoma_checklist,
            "inflammatory_checklist": inflammatory_checklist,
            "reactive_regenerative_checklist": inflammatory_checklist,
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
                response["attempts"] = attempts + list(response.get("repair_attempts", [])) + [
                    {"backend": backend_name, "status": "ok", "latency_ms": response.get("latency_ms", 0)}
                ]
                return response
            except BackendUnavailableError as exc:
                attempts.extend(list(getattr(exc, "repair_attempts", [])))
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


def _parse_or_repair_local_cpathagent_qwen_output(stage, generated_text, request, bundle, prompt_text):
    repair_attempts = []
    parse_error = ""
    output = None
    repair_required = False
    try:
        output = _parse_local_cpathagent_qwen_output(stage, generated_text, request, bundle)
        repair_reason = _stage_output_repair_reason(stage, output)
        if not repair_reason:
            return output, repair_attempts
        parse_error = repair_reason
        repair_required = True
    except Exception as exc:
        parse_error = str(exc)
        repair_required = True
    if not _output_repair_enabled(bundle, stage):
        if output is not None:
            return output, repair_attempts
        raise BackendUnavailableError(parse_error or "local_cpathagent_qwen output could not be parsed")
    repair_result = _call_deepseek_output_repair(stage, generated_text, request, bundle, prompt_text, parse_error, output)
    repair_attempts.append(repair_result["attempt"])
    if repair_result["attempt"].get("status") != "ok":
        if output is not None and not repair_required:
            output.setdefault("repair_metadata", repair_result["attempt"])
            return output, repair_attempts
        exc = BackendUnavailableError("DeepSeek output repair failed for stage {0}: {1}".format(stage, repair_result["attempt"].get("error")))
        exc.repair_attempts = repair_attempts
        raise exc
    repaired_text = json.dumps(repair_result["repaired_json"], ensure_ascii=False)
    try:
        repaired_output = _parse_local_cpathagent_qwen_output(stage, repaired_text, request, bundle)
    except Exception as exc:
        repair_attempts[-1]["status"] = "invalid_repair"
        repair_attempts[-1]["validation_error"] = str(exc)
        if output is not None and not repair_required:
            output.setdefault("repair_metadata", repair_attempts[-1])
            return output, repair_attempts
        wrapped = BackendUnavailableError("DeepSeek repaired output did not pass deterministic parser: {0}".format(str(exc)))
        wrapped.repair_attempts = repair_attempts
        raise wrapped
    repaired_reason = _stage_output_repair_reason(stage, repaired_output)
    if repaired_reason:
        repair_attempts[-1]["status"] = "invalid_repair"
        repair_attempts[-1]["validation_error"] = repaired_reason
        wrapped = BackendUnavailableError("DeepSeek repaired output did not satisfy contract: {0}".format(repaired_reason))
        wrapped.repair_attempts = repair_attempts
        raise wrapped
    repaired_output["repair_metadata"] = {
        "repair_source": "deepseek_output_repair",
        "repair_actions": repair_result.get("repair_actions", []),
        "repair_confidence": repair_result.get("confidence", 0.0),
        "repair_reason": parse_error,
        "repair_request_id": repair_result["attempt"].get("request_id"),
    }
    return repaired_output, repair_attempts


def _output_repair_config(bundle):
    return bundle.get("runtime", {}).get("output_repair", {}) if isinstance(bundle.get("runtime", {}), dict) else {}


def _output_repair_enabled(bundle, stage):
    cfg = _output_repair_config(bundle)
    if not cfg.get("enabled", False):
        return False
    stages = cfg.get("stages", ["trace", "navigate", "observe_step", "observe_report"])
    return stage in set(stages)


def _stage_output_repair_reason(stage, output):
    if not isinstance(output, dict):
        return "{0} output is not an object".format(stage)
    if stage == "trace":
        clusters = output.get("clusters")
        patches = output.get("patches")
        if clusters is not None and isinstance(clusters, list) and not clusters:
            return "trace clusters are empty"
        if patches is not None and isinstance(patches, list) and not patches:
            return "trace patches are empty"
    elif stage == "navigate":
        steps = output.get("steps")
        if not isinstance(steps, list) or not steps:
            return "navigation steps are missing"
        for step in steps:
            if not isinstance(step, dict):
                return "navigation contains non-object step"
            metadata = step.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if float(step.get("m", 0.0)) not in {5.0, 10.0} and metadata.get("action") != "stop":
                return "navigation contains illegal magnification"
            if metadata.get("action") != "stop" and not metadata.get("cell_id"):
                return "navigation step missing cell_id"
    elif stage == "observe_step":
        required = ("observation", "reasoning", "next_step", "stage_decision", "confidence")
        for key in required:
            if key not in output:
                return "observe_step missing {0}".format(key)
    elif stage == "observe_report":
        if not isinstance(output.get("hierarchical_prediction"), dict):
            return "observe_report missing hierarchical_prediction"
        for key in ("serrated_checklist", "abnormal_crypt_checklist", "dysplasia_checklist"):
            value = output.get(key)
            if value in (None, {}, []):
                return "observe_report missing or empty {0}".format(key)
    return ""


def _repair_endpoint_from_runtime(bundle):
    cfg = _output_repair_config(bundle)
    server_url = str(cfg.get("server_url") or bundle.get("runtime", {}).get("chief_llm", {}).get("server_url") or "").strip()
    if not server_url:
        return ""
    if server_url.endswith("/predict"):
        return server_url[: -len("/predict")] + "/repair_output"
    return server_url.rstrip("/") + "/repair_output"


def _call_deepseek_output_repair(stage, generated_text, request, bundle, prompt_text, parse_error, parsed_output=None):
    import requests

    cfg = _output_repair_config(bundle)
    server_url = _repair_endpoint_from_runtime(bundle)
    timeout_seconds = int(cfg.get("timeout_seconds", bundle.get("runtime", {}).get("chief_llm", {}).get("timeout_seconds", 180)))
    payload = {
        "task": "output_repair",
        "stage": stage,
        "raw_generated_text": str(generated_text or ""),
        "prompt_excerpt": str(prompt_text or "")[:4000],
        "parse_or_contract_error": str(parse_error or ""),
        "parsed_output": parsed_output if isinstance(parsed_output, dict) else None,
        "request_metadata": request.get("metadata", {}),
        "schema_hint": _output_repair_schema_hint(stage),
        "allow_light_clinical_fill": bool(cfg.get("allow_light_clinical_fill", True)),
    }
    started = time.time()
    attempt = {
        "backend": "deepseek_output_repair",
        "status": "error",
        "repair_stage": stage,
        "server_url": server_url,
    }
    try:
        response = requests.post(server_url, json=payload, timeout=timeout_seconds)
        attempt["latency_ms"] = int(round((time.time() - started) * 1000.0))
        if response.status_code != 200:
            attempt["error"] = "HTTP {0}: {1}".format(response.status_code, response.text)
            return {"attempt": attempt, "repaired_json": {}, "repair_actions": [], "confidence": 0.0}
        data = response.json()
    except Exception as exc:
        attempt["latency_ms"] = int(round((time.time() - started) * 1000.0))
        attempt["error"] = str(exc)
        return {"attempt": attempt, "repaired_json": {}, "repair_actions": [], "confidence": 0.0}
    unrecoverable = data.get("unrecoverable_errors", [])
    repaired_json = data.get("repaired_json", {})
    if unrecoverable or not isinstance(repaired_json, dict) or not repaired_json:
        attempt["status"] = "unrecoverable"
        attempt["error"] = "; ".join(str(item) for item in unrecoverable) or "empty repaired_json"
    else:
        attempt["status"] = "ok"
    attempt["request_id"] = data.get("request_id")
    attempt["model_name"] = data.get("model_name")
    attempt["repair_actions"] = data.get("repair_actions", [])
    attempt["repair_confidence"] = data.get("confidence", 0.0)
    attempt["raw_response_path"] = ""
    return {
        "attempt": attempt,
        "repaired_json": repaired_json if isinstance(repaired_json, dict) else {},
        "repair_actions": data.get("repair_actions", []),
        "confidence": data.get("confidence", 0.0),
    }


def _output_repair_schema_hint(stage):
    if stage == "navigate":
        return {
            "root": "steps",
            "required_step_fields": ["source_group_id", "patch_id", "x", "y", "m", "region_size_level0", "need_to_see", "review_goal", "stage_gate"],
            "allowed_magnifications": [5.0, 10.0],
        }
    if stage == "observe_report":
        return {
            "required_root_fields": [
                "hierarchical_prediction",
                "serrated_checklist",
                "abnormal_crypt_checklist",
                "conventional_adenoma_checklist",
                "serrated_dysplasia_checklist",
                "conventional_dysplasia_checklist",
                "dysplasia_checklist",
                "integrated_report",
            ]
        }
    if stage == "observe_step":
        return {"required_root_fields": ["observation", "reasoning", "next_step", "stage_decision", "confidence"]}
    return {"required_root_fields": ["clusters", "patches"]}


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
        "Navigation must use only 2.5x, 5x, and 10x magnifications.",
        "Treat 2.5x as the pathway overview and 5x as the subtype/architecture assessment field. Do not schedule 10x in this initial navigation pass; 10x is requested later by Chief reasoning as a targeted tool call.",
        "For serrated trace clusters: plan 2.5x serrated_overview_assessment, 5x ssl_assessment, 5x hp_assessment, and 5x tsa_assessment. Leave 10x SSL dysplasia, TSA cytology, and TSA dysplasia targets for Chief-triggered follow-up.",
        "For conventional trace clusters: plan 2.5x conventional_overview_assessment, 5x conventional_architecture_assessment, and 5x reactive_regenerative_assessment. Leave 10x conventional dysplasia targets for Chief-triggered follow-up.",
        "For normal trace clusters: plan 2.5x normal_overview_assessment and add 5x inflammatory_reactive_assessment only when a clear inflammatory/reactive clue exists. For background trace clusters: stop or discard without diagnostic observation.",
        "Optional per-step fields intra_cell_target_index, intra_cell_target_role, intra_cell_target_count, and coordinate_source are encouraged; the runner will normalize them if absent.",
        "",
        "Return JSON:",
        '{ "steps": [ { "source_group_id": "grid_group_00", "patch_id": [0, 0], "x": 100, "y": 200, "m": 2.5, "region_size_level0": 4096, "need_to_see": "2.5x overview for serrated mucosal context", "review_goal": "serrated_overview_assessment", "stage_gate": "serrated_overview", "intra_cell_target_index": 0, "intra_cell_target_role": "overview" }, { "source_group_id": "grid_group_00", "patch_id": [0, 0], "x": 100, "y": 200, "m": 5.0, "region_size_level0": 2048, "need_to_see": "5x SSL architectural distortion assessment", "review_goal": "ssl_assessment", "stage_gate": "ssl_architecture", "intra_cell_target_index": 0, "intra_cell_target_role": "ssl_architecture" }, { "source_group_id": "grid_group_00", "patch_id": [0, 0], "x": 118, "y": 218, "m": 5.0, "region_size_level0": 2048, "need_to_see": "5x HP architecture assessment", "review_goal": "hp_assessment", "stage_gate": "hp_architecture", "intra_cell_target_index": 1, "intra_cell_target_role": "hp_architecture" }, { "source_group_id": "grid_group_00", "patch_id": [0, 0], "x": 135, "y": 235, "m": 5.0, "region_size_level0": 2048, "need_to_see": "5x TSA architecture and low-power cytology assessment", "review_goal": "tsa_assessment", "stage_gate": "tsa_architecture", "intra_cell_target_index": 2, "intra_cell_target_role": "tsa_architecture" } ] }',
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
    overlap_threshold = float(bundle.get("runtime", {}).get("navigate", {}).get("overlap_threshold", 0.30))
    normalized_steps = []
    prior_windows = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_metadata = step.get("metadata", {}) if isinstance(step.get("metadata"), dict) else {}
        source_group_id = step.get("source_group_id") or step.get("cluster_id") or step_metadata.get("source_group_id") or step_metadata.get("cluster_id")
        cluster = clusters.get(source_group_id, {})
        patch_id = step.get("patch_id") or step_metadata.get("patch_id") or []
        x = step.get("x")
        y = step.get("y")
        coordinate_source = "model_proposed" if x is not None and y is not None else ""
        if (x is None or y is None) and patch_id:
            for patch in cluster.get("patches_level0", []):
                if list(patch.get("patch_id", [])) == list(patch_id):
                    anchor_x = patch.get("anchor_x")
                    anchor_y = patch.get("anchor_y")
                    if anchor_x is not None and anchor_y is not None:
                        x = int(anchor_x)
                        y = int(anchor_y)
                        coordinate_source = "trace_anchor"
                    else:
                        x = int(round((int(patch["x1"]) + int(patch["x2"])) / 2.0))
                        y = int(round((int(patch["y1"]) + int(patch["y2"])) / 2.0))
                        coordinate_source = "patch_center_fallback"
                    break
        if x is None or y is None:
            bbox = cluster.get("group_bbox_level0") or cluster.get("cluster_bbox_level0") or {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
            x = int(round((int(bbox["x1"]) + int(bbox["x2"])) / 2.0))
            y = int(round((int(bbox["y1"]) + int(bbox["y2"])) / 2.0))
            coordinate_source = "patch_center_fallback"
        magnification = float(step.get("m", 5.0))
        if abs(magnification - 20.0) < 1e-6:
            magnification = 10.0
        region_size = int(step.get("region_size_level0", mag_to_region.get(str(magnification), 2048)))
        if int(region_size) == 512 and abs(magnification - 10.0) < 1e-6:
            region_size = int(mag_to_region.get("10.0", 1024))
        from adenoma_agent.utils import bbox_from_center, bbox_overlap_ratio

        cell_id = step_metadata.get("cell_id") or _navigation_cell_id(patch_id, source_group_id)
        step_bbox = bbox_from_center(int(x), int(y), region_size)
        should_skip = False
        for prior_item in prior_windows:
            if prior_item.get("cell_id") != cell_id:
                continue
            if abs(float(prior_item["m"]) - float(magnification)) > 1e-6:
                continue
            if prior_item.get("review_goal") != step.get("review_goal"):
                continue
            if bbox_overlap_ratio(step_bbox, prior_item["bbox"]) > overlap_threshold:
                should_skip = True
                break
        if should_skip:
            continue
        prior_windows.append({"bbox": step_bbox, "m": magnification, "cell_id": cell_id, "review_goal": step.get("review_goal")})
        normalized_steps.append(
            {
                "step_id": "step_{0:02d}".format(len(normalized_steps)),
                "x": int(x),
                "y": int(y),
                "m": magnification,
                "region_size_level0": region_size,
                "need_to_see": step.get("need_to_see", step.get("o", "Inspect the planned pathology region.")),
                "review_goal": step.get("review_goal", "serrated_overview_assessment"),
                "stage_gate": step.get("stage_gate", "serrated_overview"),
                "metadata": {
                    "cluster_id": source_group_id,
                    "source_group_id": source_group_id,
                    "cluster_label": cluster.get("l"),
                    "cluster_priority": cluster.get("s"),
                    "cell_priority": cluster.get("s"),
                    "cell_id": cell_id,
                    "patch_id": list(patch_id),
                    "intra_cell_target_index": int(step.get("intra_cell_target_index", step_metadata.get("intra_cell_target_index", 0)) or 0),
                    "intra_cell_target_role": str(step.get("intra_cell_target_role", step_metadata.get("intra_cell_target_role", _navigation_target_role(cluster.get("l"), step.get("review_goal", "serrated_overview_assessment"), 0))) or "model_selected"),
                    "intra_cell_target_count": int(step.get("intra_cell_target_count", step_metadata.get("intra_cell_target_count", 1)) or 1),
                    "coordinate_source": str(step.get("coordinate_source", step_metadata.get("coordinate_source", coordinate_source)) or "model_proposed"),
                    "region_size_level0": region_size,
                    "workflow_branch": cluster.get("metadata", {}).get("workflow_branch") or _trace_branch_for_label(cluster.get("l")),
                    "action": "inspect",
                },
            }
        )
    if not normalized_steps:
        raise BackendUnavailableError("local_cpathagent_qwen navigate could not normalize any valid steps")
    focused_by_cell = {}
    for step in normalized_steps:
        metadata = step.get("metadata", {})
        if float(step.get("m", 0.0)) != 10.0:
            continue
        key = (metadata.get("cell_id"), tuple(metadata.get("patch_id", [])))
        focused_by_cell.setdefault(key, []).append(step)
    for items in focused_by_cell.values():
        total = len(items)
        for target_index, item in enumerate(items):
            metadata = item["metadata"]
            metadata["intra_cell_target_index"] = int(target_index)
            metadata["intra_cell_target_count"] = total
            if not metadata.get("intra_cell_target_role") or metadata.get("intra_cell_target_role") in {"overview", "model_selected"}:
                metadata["intra_cell_target_role"] = _navigation_target_role(metadata.get("cluster_label"), item.get("review_goal"), target_index)
    for step in normalized_steps:
        metadata = step.get("metadata", {})
        if float(step.get("m", 0.0)) == 5.0:
            metadata["intra_cell_target_index"] = 0
            metadata["intra_cell_target_role"] = metadata.get("intra_cell_target_role") or "overview"
            metadata["intra_cell_target_count"] = int(metadata.get("intra_cell_target_count", 1) or 1)
    last = normalized_steps[-1]
    normalized_steps.append(
        {
            "step_id": "step_{0:02d}".format(len(normalized_steps)),
            "x": last["x"],
            "y": last["y"],
            "m": 5.0,
            "region_size_level0": 2048,
            "need_to_see": "Stop navigation and consolidate the gathered evidence.",
            "review_goal": "integrated_impression",
            "stage_gate": "end",
            "metadata": {"action": "stop", "region_size_level0": 2048},
        }
    )
    return {"steps": normalized_steps}


def _build_cpathagent_qwen_observe_report_prompt(request):
    lines = [
        request["prompt"]["question"],
        "",
        "Summarize the following pathology reasoning records into a structured JSON report.",
        "Return JSON only with these keys:",
        "hierarchical_prediction, serrated_checklist, abnormal_crypt_checklist, conventional_adenoma_checklist, serrated_dysplasia_checklist, conventional_dysplasia_checklist, dysplasia_checklist, ssl_checklist, hp_checklist, tsa_checklist, inflammatory_checklist, integrated_report",
        "hierarchical_prediction must include primary_branch, subtype_prediction, dysplasia_status, final_11_class, classification_status, non_diagnostic_reason, coexisting_candidates, class_scores, decision_path, and final_case_assessment.",
        "When classification_status=classified, final_case_assessment.label must be one of: SSL, SSLD, HP, TSA, TSAD, Unclassified serrated adenoma, Tubular adenoma, TAD, Tubulovillous adenoma, TVAD, Inflammatory.",
        "The D suffix means high-grade/definite dysplasia, not any low-grade dysplasia. Do not map insufficient evidence to Inflammatory or Unclassified serrated adenoma.",
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
            "ssl_checklist",
            "hp_checklist",
            "tsa_checklist",
            "inflammatory_checklist",
        )
        for key in checklist_keys:
            value = parsed.get(key, [])
            if key not in parsed:
                parsed[key] = {}
            elif not isinstance(value, (dict, list)):
                parsed[key] = []

        integrated_report = parsed.get("integrated_report", "")
        if not isinstance(integrated_report, (dict, str)):
            parsed["integrated_report"] = str(integrated_report)
        return parsed
    raise BackendUnavailableError("local_cpathagent_qwen observe_report returned invalid JSON structure")


def _blank_hits(criteria):
    return {criterion: "not_assessed" for criterion in criteria}


TRACE_BACKGROUND_LABEL = "background"
TRACE_NORMAL_LABEL = "normal"
TRACE_NEUTRAL_BACKGROUND_LABEL = "background_or_artifact"
TRACE_NEUTRAL_NORMAL_LABEL = "reviewable_normal_mucosa"
TRACE_NEUTRAL_INFLAMMATORY_LABEL = "inflammatory_or_stromal_context"
TRACE_NEUTRAL_MUCUS_LABEL = "mucus_rich_or_pale_context"
TRACE_NEUTRAL_EPITHELIAL_LABEL = "epithelial_neoplasia_suspicious"
TRACE_NEUTRAL_UNCERTAIN_LABEL = "uncertain_reviewable_mucosa"
TRACE_LEGACY_CONVENTIONAL_LABEL = "conventional_adenoma_like"
TRACE_CONVENTIONAL_LABEL = "conventional"
TRACE_TUBULAR_LABEL = "tubular_adenoma_like"
TRACE_TUBULOVILLOUS_LABEL = "tubulovillous_adenoma_like"
TRACE_INFLAMMATORY_LABEL = "inflammatory_polyp_like"
TRACE_LEGACY_SSL_LABEL = "ssl_suspicious_mucosa"
TRACE_SSL_LIKE_LABEL = "ssl_like_mucosa"
TRACE_SERRATED_LABEL = "serrated"
TRACE_LEGACY_SSL_HIGH_LABEL = "ssl_high_priority_mucosa"
TRACE_HP_LABEL = "hp_like_mucosa"
TRACE_TSA_LABEL = "tsa_like_mucosa"
TRACE_UNCLASSIFIED_SERRATED_LABEL = "unclassified_serrated_like_mucosa"
TRACE_SERRATED_LABELS = (
    TRACE_SERRATED_LABEL,
    TRACE_SSL_LIKE_LABEL,
    TRACE_LEGACY_SSL_LABEL,
    TRACE_LEGACY_SSL_HIGH_LABEL,
    TRACE_HP_LABEL,
    TRACE_TSA_LABEL,
    TRACE_UNCLASSIFIED_SERRATED_LABEL,
)
TRACE_CONVENTIONAL_LABELS = (
    TRACE_CONVENTIONAL_LABEL,
    TRACE_LEGACY_CONVENTIONAL_LABEL,
    TRACE_TUBULAR_LABEL,
    TRACE_TUBULOVILLOUS_LABEL,
)
TRACE_ALLOWED_LABELS = (
    TRACE_BACKGROUND_LABEL,
    TRACE_NORMAL_LABEL,
    TRACE_CONVENTIONAL_LABEL,
    TRACE_SERRATED_LABEL,
    TRACE_NEUTRAL_BACKGROUND_LABEL,
    TRACE_NEUTRAL_NORMAL_LABEL,
    TRACE_NEUTRAL_INFLAMMATORY_LABEL,
    TRACE_NEUTRAL_MUCUS_LABEL,
    TRACE_NEUTRAL_EPITHELIAL_LABEL,
    TRACE_NEUTRAL_UNCERTAIN_LABEL,
)


def _navigation_cell_id(patch_id, fallback):
    values = list(patch_id or [])
    return "cell_{0}_{1}".format(*values[:2]) if len(values) == 2 else str(fallback or "")


def _navigation_target_role(label, review_goal, target_index):
    if float(target_index) == 0 and str(review_goal).endswith("_assessment") and "overview" in str(review_goal):
        return "overview"
    review_goal_roles = {
        "ssl_assessment": "ssl_architecture",
        "hp_assessment": "hp_architecture",
        "tsa_assessment": "tsa_architecture",
        "conventional_architecture_assessment": "conventional_architecture",
        "reactive_regenerative_assessment": "reactive_regenerative",
    }
    if str(review_goal) in review_goal_roles:
        return review_goal_roles[str(review_goal)]
    if str(review_goal) in {"serrated_lesion_assessment", "conventional_adenoma_assessment", "non_serrated_overview_assessment"} and int(target_index) == 0:
        return "overview"
    roles_by_label = {
        TRACE_SERRATED_LABEL: ("crypt_base", "serrated_edge", "dysplasia_hotspot"),
        TRACE_LEGACY_SSL_LABEL: ("crypt_base", "serrated_edge", "dysplasia_hotspot"),
        TRACE_LEGACY_SSL_HIGH_LABEL: ("crypt_base", "serrated_edge", "dysplasia_hotspot"),
        TRACE_HP_LABEL: ("surface_serration", "crypt_base_exclusion", "model_selected"),
        TRACE_TSA_LABEL: ("tsa_architecture", "ectopic_crypt_focus", "dysplasia_hotspot"),
        TRACE_UNCLASSIFIED_SERRATED_LABEL: ("serrated_architecture", "subtype_ambiguity", "dysplasia_hotspot"),
        TRACE_CONVENTIONAL_LABEL: ("gland_crowding", "dysplasia_hotspot", "architecture_transition"),
        TRACE_TUBULAR_LABEL: ("tubular_architecture", "dysplasia_hotspot", "architecture_transition"),
        TRACE_TUBULOVILLOUS_LABEL: ("villous_component", "dysplasia_hotspot", "architecture_transition"),
    }
    roles = roles_by_label.get(label, ("model_selected", "model_selected", "model_selected"))
    return roles[min(int(target_index), len(roles) - 1)]


def _cluster_needs_multi_zoom(cluster):
    metadata = cluster.get("metadata", {}) if isinstance(cluster.get("metadata"), dict) else {}
    label = cluster.get("l")
    if bool(cluster.get("d")) or bool(metadata.get("requires_high_magnification")):
        return label in set(TRACE_SERRATED_LABELS + TRACE_CONVENTIONAL_LABELS)
    if label in set(TRACE_SERRATED_LABELS + TRACE_CONVENTIONAL_LABELS) and int(cluster.get("s", 0)) >= 4:
        return True
    return False


def _intra_cell_candidate_points(center_x, center_y, patch, max_targets):
    x1 = int(patch.get("x1", center_x))
    y1 = int(patch.get("y1", center_y))
    x2 = int(patch.get("x2", center_x))
    y2 = int(patch.get("y2", center_y))
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    offset_x = max(1, int(round(width * 0.25)))
    offset_y = max(1, int(round(height * 0.25)))
    candidates = [
        (center_x, center_y),
        (center_x - offset_x, center_y - offset_y),
        (center_x + offset_x, center_y + offset_y),
        (center_x + offset_x, center_y - offset_y),
        (center_x - offset_x, center_y + offset_y),
    ]
    unique = []
    for x_value, y_value in candidates:
        point = (int(x_value), int(y_value))
        if point not in unique:
            unique.append(point)
        if len(unique) >= int(max_targets):
            break
    return unique


def _trace_branch_for_label(label):
    if label in {
        TRACE_NEUTRAL_EPITHELIAL_LABEL,
        TRACE_NEUTRAL_MUCUS_LABEL,
        TRACE_NEUTRAL_UNCERTAIN_LABEL,
    }:
        return "unresolved"
    if label == TRACE_NEUTRAL_INFLAMMATORY_LABEL:
        return "unresolved"
    if label == TRACE_NEUTRAL_NORMAL_LABEL:
        return "normal"
    if label == TRACE_NEUTRAL_BACKGROUND_LABEL:
        return "background"
    if label in TRACE_SERRATED_LABELS:
        return "serrated"
    if label in TRACE_CONVENTIONAL_LABELS:
        return "conventional_adenoma"
    if label == TRACE_INFLAMMATORY_LABEL:
        return "normal"
    if label == TRACE_NORMAL_LABEL:
        return "normal"
    return "background"


def _trace_review_stage_for_label(label):
    branch = _trace_branch_for_label(label)
    if branch == "unresolved":
        return "morphology_resolution_screening"
    if branch == "serrated":
        return "serrated_screening"
    if branch == "conventional_adenoma":
        return "conventional_adenoma_screening"
    if branch == "inflammatory":
        return "inflammatory_polyp_screening"
    return "mucosa_screening"


def _trace_label_metadata(label, priority, require_high_magnification, metadata=None):
    base_metadata = dict(metadata or {})
    branch = _trace_branch_for_label(label)
    if label in {TRACE_NEUTRAL_EPITHELIAL_LABEL, TRACE_NEUTRAL_MUCUS_LABEL, TRACE_NEUTRAL_UNCERTAIN_LABEL}:
        conventional_hint = None
        inflammatory_hint = None
        serrated_hint = None
        base_metadata.setdefault("candidate_branches", ["serrated", "conventional"])
        base_metadata.setdefault("routing_hint", "needs_morphology_resolution")
    elif label == TRACE_NEUTRAL_INFLAMMATORY_LABEL:
        conventional_hint = None
        inflammatory_hint = "inflammatory_or_stromal_context"
        serrated_hint = None
        base_metadata.setdefault("candidate_branches", ["inflammatory", "serrated", "conventional"])
        base_metadata.setdefault("routing_hint", "exclude_hidden_epithelial_lesion")
    elif label in TRACE_CONVENTIONAL_LABELS:
        conventional_hint = base_metadata.get("conventional_subtype_hint")
        if not conventional_hint:
            conventional_hint = "tubulovillous_adenoma_like" if label == TRACE_TUBULOVILLOUS_LABEL else "tubular_adenoma_like"
        inflammatory_hint = None
        serrated_hint = None
    elif label == TRACE_INFLAMMATORY_LABEL:
        conventional_hint = None
        inflammatory_hint = base_metadata.get("inflammatory_subtype_hint") or "inflammatory_polyp_like"
        serrated_hint = None
    elif label in TRACE_SERRATED_LABELS:
        conventional_hint = None
        inflammatory_hint = None
        serrated_hint = base_metadata.get("serrated_family_hint")
        if not serrated_hint:
            serrated_hint = {
                TRACE_SERRATED_LABEL: "ssl_like",
                TRACE_LEGACY_SSL_LABEL: "ssl_like",
                TRACE_LEGACY_SSL_HIGH_LABEL: "ssl_like",
                TRACE_HP_LABEL: "hp_like",
                TRACE_TSA_LABEL: "tsa_like",
                TRACE_UNCLASSIFIED_SERRATED_LABEL: "unclassified_serrated_like",
            }.get(label, "equivocal_serrated")
    else:
        conventional_hint = None
        inflammatory_hint = None
        serrated_hint = None
    serrated_dysplasia_suspected = bool(base_metadata.get("serrated_dysplasia_suspected", False))
    conventional_dysplasia_suspected = bool(base_metadata.get("conventional_dysplasia_suspected", False))
    if label in TRACE_SERRATED_LABELS and int(priority) >= 5 and label != TRACE_HP_LABEL:
        serrated_dysplasia_suspected = True
    if label in TRACE_CONVENTIONAL_LABELS:
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
        "background_or_artifact": TRACE_NEUTRAL_BACKGROUND_LABEL,
        "normal": TRACE_NORMAL_LABEL,
        "normal_mucosa": TRACE_NORMAL_LABEL,
        "non_serrated_mucosa": TRACE_NORMAL_LABEL,
        "reviewable_normal_mucosa": TRACE_NEUTRAL_NORMAL_LABEL,
        "inflammatory_or_stromal_context": TRACE_NEUTRAL_INFLAMMATORY_LABEL,
        "mucus_rich_or_pale_context": TRACE_NEUTRAL_MUCUS_LABEL,
        "epithelial_neoplasia_suspicious": TRACE_NEUTRAL_EPITHELIAL_LABEL,
        "uncertain_reviewable_mucosa": TRACE_NEUTRAL_UNCERTAIN_LABEL,
        "adi": TRACE_NEUTRAL_BACKGROUND_LABEL,
        "back": TRACE_NEUTRAL_BACKGROUND_LABEL,
        "deb": TRACE_NEUTRAL_BACKGROUND_LABEL,
        "mus": TRACE_NEUTRAL_BACKGROUND_LABEL,
        "norm": TRACE_NEUTRAL_NORMAL_LABEL,
        "lym": TRACE_NEUTRAL_INFLAMMATORY_LABEL,
        "str": TRACE_NEUTRAL_INFLAMMATORY_LABEL,
        "muc": TRACE_NEUTRAL_MUCUS_LABEL,
        "tum": TRACE_NEUTRAL_EPITHELIAL_LABEL,
        "conventional_adenoma_like": TRACE_CONVENTIONAL_LABEL,
        "tubular_adenoma_like": TRACE_CONVENTIONAL_LABEL,
        "tubulovillous_adenoma_like": TRACE_CONVENTIONAL_LABEL,
        "inflammatory_polyp_like": TRACE_NORMAL_LABEL,
        "serrated_suspicious_mucosa": TRACE_SERRATED_LABEL,
        "ssl_suspicious_mucosa": TRACE_SERRATED_LABEL,
        "ssl_like_mucosa": TRACE_SERRATED_LABEL,
        "ssl_high_priority_mucosa": TRACE_SERRATED_LABEL,
        "hp_like_mucosa": TRACE_SERRATED_LABEL,
        "tsa_like_mucosa": TRACE_SERRATED_LABEL,
        "unclassified_serrated_like_mucosa": TRACE_SERRATED_LABEL,
    }
    normalized_exact = exact_map.get(str(region_semantic or "").strip().lower())
    if normalized_exact:
        return normalized_exact
    if any(
        token in haystack
        for token in ("background", "artifact", "blank", "stroma", "muscle", "muscularis")
    ):
        return TRACE_NEUTRAL_BACKGROUND_LABEL if "background_or_artifact" in str(region_semantic or "") else TRACE_BACKGROUND_LABEL
    if any(token in haystack for token in ("epithelial neoplasia", "neoplasia suspicious", "tum", "neoplastic epithelium")):
        return TRACE_NEUTRAL_EPITHELIAL_LABEL
    if any(token in haystack for token in ("mucus-rich", "mucus rich", "mucin", "muc ")):
        return TRACE_NEUTRAL_MUCUS_LABEL
    if any(token in haystack for token in ("uncertain reviewable", "ambiguous mucosa")):
        return TRACE_NEUTRAL_UNCERTAIN_LABEL
    if any(token in haystack for token in ("lymphocyte", "stromal context", "inflammatory context")):
        return TRACE_NEUTRAL_INFLAMMATORY_LABEL
    if any(token in haystack for token in ("normal mucosa", "non-lesional", "non lesional", "benign mucosa")):
        return TRACE_NORMAL_LABEL
    if any(
        token in haystack
        for token in (
            "tubulovillous adenoma",
            "tubulovillous",
            "tva-like",
        )
    ):
        return TRACE_CONVENTIONAL_LABEL
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
            "tubular",
            "ta-like",
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
        return TRACE_NORMAL_LABEL
    if any(
        token in haystack
        for token in (
            "traditional serrated adenoma",
            "tsa",
            "ectopic crypt",
            "eosinophilic cytoplasm",
            "filiform",
            "villiform",
        )
    ):
        return TRACE_SERRATED_LABEL
    if any(
        token in haystack
        for token in (
            "hyperplastic polyp",
            "hyperplastic",
            "hp-like",
            "surface-limited serration",
            "straight crypt",
        )
    ):
        return TRACE_SERRATED_LABEL
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
            "mucus cap",
            "mucous cap",
            "unclassified serrated",
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
    if label in {TRACE_BACKGROUND_LABEL, TRACE_NEUTRAL_BACKGROUND_LABEL}:
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
            "candidate_branches",
            "routing_hint",
            "conch_region_semantic",
            "conch_crc_label",
            "conch_probs",
            "conch_confidence",
            "conch_raw_prediction",
            "digepath_class",
            "digepath_region_semantic",
            "digepath_probs",
            "digepath_confidence",
            "digepath_raw_prediction",
            "agreement_status",
            "score_origin",
            "fusion_reasoning",
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


def _conch_trace_defaults(label):
    rubric = TRACE_LABEL_RUBRIC.get(label, {})
    priority = int(rubric.get("default_priority", 0))
    high_mag = bool(rubric.get("default_high_mag", False))
    if label == TRACE_NEUTRAL_EPITHELIAL_LABEL:
        return priority, True, "CONCH TUM-like epithelial evidence; keep serrated and conventional branches open for morphology resolution."
    if label == TRACE_NEUTRAL_MUCUS_LABEL:
        return priority, True, "CONCH MUC-like mucus-rich context; prioritize morphology review without assigning SSL at Trace."
    if label == TRACE_NEUTRAL_UNCERTAIN_LABEL:
        return priority, True, "CONCH output was unavailable or low-confidence; retain as uncertain reviewable mucosa."
    if label == TRACE_NEUTRAL_INFLAMMATORY_LABEL:
        return priority, False, "CONCH LYM/STR-like inflammatory or stromal context; review only to exclude hidden epithelial lesion."
    if label == TRACE_NEUTRAL_NORMAL_LABEL:
        return priority, False, "CONCH NORM-like reviewable normal mucosa."
    return 0, False, "CONCH BACK/DEB/ADI/MUS-like low-value background or artifact."


def _conch_assignment_name(label):
    return {
        TRACE_NEUTRAL_EPITHELIAL_LABEL: "Epithelial neoplasia-suspicious mucosa",
        TRACE_NEUTRAL_MUCUS_LABEL: "Mucus-rich or pale review context",
        TRACE_NEUTRAL_UNCERTAIN_LABEL: "Uncertain reviewable mucosa",
        TRACE_NEUTRAL_INFLAMMATORY_LABEL: "Inflammatory or stromal context",
        TRACE_NEUTRAL_NORMAL_LABEL: "Reviewable normal mucosa",
        TRACE_NEUTRAL_BACKGROUND_LABEL: "Background or artifact",
    }.get(label, "CONCH-screened patch")


def _conch_observation_points(label):
    rubric = TRACE_LABEL_RUBRIC.get(label, {})
    points = list(rubric.get("observation_points", []))
    return points or ["CONCH-derived global screening evidence"]


def _trace_label_rank(label):
    return {
        TRACE_NEUTRAL_BACKGROUND_LABEL: 0,
        TRACE_NEUTRAL_NORMAL_LABEL: 1,
        TRACE_NEUTRAL_INFLAMMATORY_LABEL: 2,
        TRACE_NEUTRAL_UNCERTAIN_LABEL: 3,
        TRACE_NEUTRAL_MUCUS_LABEL: 4,
        TRACE_NEUTRAL_EPITHELIAL_LABEL: 4,
    }.get(str(label or ""), -1)


def _is_high_value_trace_label(label):
    return str(label or "") in {TRACE_NEUTRAL_EPITHELIAL_LABEL, TRACE_NEUTRAL_MUCUS_LABEL}


def _fuse_conch_digepath_label(
    conch_label,
    conch_confidence,
    digepath_label,
    digepath_confidence,
    high_confidence_threshold,
    normal_conch_threshold=0.90,
    normal_digepath_threshold=0.90,
    normal_gate_fallback_label=TRACE_NEUTRAL_UNCERTAIN_LABEL,
):
    conch_label = _normalize_conch_label(conch_label) or TRACE_NEUTRAL_UNCERTAIN_LABEL
    digepath_label = _normalize_digepath_trace_label(digepath_label)
    digepath_available = bool(digepath_label)
    if not digepath_available:
        return conch_label, "conch_only_trace", "conch_crc100k_neutral_mapping"
    if conch_label == digepath_label:
        if conch_label == TRACE_NEUTRAL_NORMAL_LABEL and (
            float(conch_confidence or 0.0) < float(normal_conch_threshold)
            or float(digepath_confidence or 0.0) < float(normal_digepath_threshold)
        ):
            fallback = _normalize_conch_label(normal_gate_fallback_label) or TRACE_NEUTRAL_UNCERTAIN_LABEL
            return fallback, "conch_digepath_agree", "conch_digepath_fusion"
        return conch_label, "conch_digepath_agree", "conch_digepath_fusion"
    if _is_high_value_trace_label(conch_label):
        return conch_label, "conch_digepath_disagree", "conch_digepath_fusion"
    if _is_high_value_trace_label(digepath_label):
        if conch_label == TRACE_NEUTRAL_BACKGROUND_LABEL:
            return TRACE_NEUTRAL_UNCERTAIN_LABEL, "conch_digepath_disagree", "conch_digepath_fusion"
        if float(digepath_confidence or 0.0) >= float(high_confidence_threshold):
            return digepath_label, "conch_digepath_disagree", "conch_digepath_fusion"
        return TRACE_NEUTRAL_UNCERTAIN_LABEL, "conch_digepath_disagree", "conch_digepath_fusion"
    if conch_label == TRACE_NEUTRAL_UNCERTAIN_LABEL or digepath_label == TRACE_NEUTRAL_UNCERTAIN_LABEL:
        return TRACE_NEUTRAL_UNCERTAIN_LABEL, "conch_digepath_disagree", "conch_digepath_fusion"
    if conch_label == TRACE_NEUTRAL_BACKGROUND_LABEL and digepath_label == TRACE_NEUTRAL_BACKGROUND_LABEL:
        return TRACE_NEUTRAL_BACKGROUND_LABEL, "conch_digepath_agree", "conch_digepath_fusion"
    if TRACE_NEUTRAL_NORMAL_LABEL in {conch_label, digepath_label} and TRACE_NEUTRAL_BACKGROUND_LABEL in {
        conch_label,
        digepath_label,
    }:
        fallback = _normalize_conch_label(normal_gate_fallback_label) or TRACE_NEUTRAL_UNCERTAIN_LABEL
        return fallback, "conch_digepath_disagree", "conch_digepath_fusion"
    label = max([conch_label, digepath_label], key=_trace_label_rank)
    return label, "conch_digepath_disagree", "conch_digepath_fusion"


def _conch_digepath_fusion_reason(conch_label, conch_confidence, digepath_label, digepath_class, digepath_confidence, fused_label, agreement_status):
    if agreement_status == "conch_only_trace":
        return "DIgePath fusion was unavailable; CONCH-only neutral tissue evidence was retained."
    if (
        conch_label == TRACE_NEUTRAL_NORMAL_LABEL
        and digepath_label == TRACE_NEUTRAL_NORMAL_LABEL
        and fused_label != TRACE_NEUTRAL_NORMAL_LABEL
    ):
        return (
            "CONCH and DIgePath both favored normal mucosa, but normal confirmation did not meet the configured confidence gate; "
            "Trace retained {0} to avoid over-suppressing reviewable ROI."
        ).format(fused_label)
    if TRACE_NEUTRAL_NORMAL_LABEL in {conch_label, digepath_label} and TRACE_NEUTRAL_BACKGROUND_LABEL in {
        conch_label,
        digepath_label,
    }:
        return (
            "CONCH ({0}, confidence {1:.3f}) and DIgePath ({2}/{3}, confidence {4:.3f}) split between normal and background-like evidence; "
            "Trace retained {5} instead of absorbing the patch into normal."
        ).format(
            conch_label or "unknown",
            float(conch_confidence or 0.0),
            digepath_class or "unknown_class",
            digepath_label or "unknown",
            float(digepath_confidence or 0.0),
            fused_label,
        )
    status_text = "agreed" if agreement_status == "conch_digepath_agree" else "disagreed"
    return (
        "CONCH ({0}, confidence {1:.3f}) and DIgePath ({2}/{3}, confidence {4:.3f}) {5}; "
        "risk-prioritized fusion selected {6} for Trace."
    ).format(
        conch_label or "unknown",
        float(conch_confidence or 0.0),
        digepath_class or "unknown_class",
        digepath_label or "unknown",
        float(digepath_confidence or 0.0),
        status_text,
        fused_label,
    )


def build_conch_only_trace_output(request, conch_runtime_metadata):
    grid_meta = _load_trace_grid_metadata(request)
    if not grid_meta:
        return {"clusters": [], "all_clusters": [], "patch_assignments": {"patches": []}, "coverage_summary": {}}
    predictions = dict((conch_runtime_metadata or {}).get("predictions", {}))
    details = dict((conch_runtime_metadata or {}).get("prediction_details", {}))
    patches = []
    for row_id, col_id in _selected_patch_ids(grid_meta):
        patch_id = [int(row_id), int(col_id)]
        key = _patch_key(patch_id)
        detail = dict(details.get(key, {}))
        label = _normalize_conch_label(predictions.get(key) or detail.get("conch_region_semantic"))
        confidence = float(detail.get("conch_confidence", 0.0) or 0.0)
        if not label or confidence < 0.01:
            label = TRACE_NEUTRAL_UNCERTAIN_LABEL
        priority, high_mag, reason = _conch_trace_defaults(label)
        patch = {
            "patch_id": patch_id,
            "name": _conch_assignment_name(label),
            "region_semantic": label,
            "description": reason,
            "require_high_magnification": bool(high_mag),
            "severity_reasoning": reason,
            "diagnostic_priority": int(priority),
            "observation_points": _conch_observation_points(label),
            "conch_region_semantic": label,
            "conch_crc_label": detail.get("conch_crc_label", ""),
            "conch_probs": detail.get("conch_probs", {}),
            "conch_confidence": confidence,
            "conch_raw_prediction": detail.get("conch_raw_prediction", {}),
            "classifier_source_image": detail.get("classifier_source_image", ""),
            "classifier_source_mode": detail.get("classifier_source_mode", ""),
            "classifier_crop_level0_bbox": detail.get("classifier_crop_level0_bbox", []),
            "classifier_crop_level0_size": detail.get("classifier_crop_level0_size", []),
            "classifier_crop_output_size": detail.get("classifier_crop_output_size", []),
            "classifier_crop_view": detail.get("classifier_crop_view", ""),
            "agreement_status": "conch_only_trace",
            "score_origin": "conch_crc100k_neutral_mapping",
            "pathoreasoner_r1_region_semantic": "not_used_in_conch_only_trace",
            "fusion_reasoning": "CONCH-only Trace retained neutral tissue evidence; morphology branch resolution is deferred downstream.",
        }
        if label in {TRACE_NEUTRAL_EPITHELIAL_LABEL, TRACE_NEUTRAL_MUCUS_LABEL, TRACE_NEUTRAL_UNCERTAIN_LABEL}:
            patch["candidate_branches"] = ["serrated", "conventional"]
            patch["routing_hint"] = "needs_morphology_resolution"
        if detail.get("embedding_ref") is not None:
            patch["embedding_ref"] = detail.get("embedding_ref")
        if detail.get("feature_ref") is not None:
            patch["feature_ref"] = detail.get("feature_ref")
        patches.append(patch)
    assignment_payload = {"patches": patches}
    validation = _validate_trace_patch_assignments_payload(assignment_payload, request, groups_source="conch_only_patch_assignments")
    output = _build_trace_clusters_from_patch_assignments(
        assignment_payload,
        request,
        validation=validation,
        apply_fallback=True,
    ) or {"clusters": [], "all_clusters": [], "patch_assignments": assignment_payload, "coverage_summary": {}}
    output.setdefault("patch_assignments", assignment_payload)
    output.setdefault("all_clusters", list(output.get("clusters", [])))
    output["coverage_summary"].update(
        {
            "trace_mode": "conch_only",
            "final_trace_schema": "patch_assignments",
            "final_groups_source": "conch_only_patch_assignments",
            "conch_enabled": bool((conch_runtime_metadata or {}).get("enabled")),
            "conch_error_count": len((conch_runtime_metadata or {}).get("errors", [])),
        }
    )
    return output


def build_conch_digepath_trace_output(request, conch_runtime_metadata, digepath_runtime_metadata):
    grid_meta = _load_trace_grid_metadata(request)
    if not grid_meta:
        return {"clusters": [], "all_clusters": [], "patch_assignments": {"patches": []}, "coverage_summary": {}}
    conch_predictions = dict((conch_runtime_metadata or {}).get("predictions", {}))
    conch_details = dict((conch_runtime_metadata or {}).get("prediction_details", {}))
    digepath_predictions = dict((digepath_runtime_metadata or {}).get("predictions", {}))
    digepath_details = dict((digepath_runtime_metadata or {}).get("prediction_details", {}))
    high_confidence_threshold = float((digepath_runtime_metadata or {}).get("high_confidence_threshold", 0.70))
    normal_conch_threshold = float((digepath_runtime_metadata or {}).get("normal_conch_confidence_threshold", 0.90))
    normal_digepath_threshold = float((digepath_runtime_metadata or {}).get("normal_digepath_confidence_threshold", 0.90))
    normal_gate_fallback_label = str(
        (digepath_runtime_metadata or {}).get("normal_gate_fallback_label", TRACE_NEUTRAL_UNCERTAIN_LABEL)
    )
    patches = []
    for row_id, col_id in _selected_patch_ids(grid_meta):
        patch_id = [int(row_id), int(col_id)]
        key = _patch_key(patch_id)
        conch_detail = dict(conch_details.get(key, {}))
        digepath_detail = dict(digepath_details.get(key, {}))
        conch_label = _normalize_conch_label(conch_predictions.get(key) or conch_detail.get("conch_region_semantic"))
        conch_confidence = float(conch_detail.get("conch_confidence", 0.0) or 0.0)
        if not conch_label or conch_confidence < 0.01:
            conch_label = TRACE_NEUTRAL_UNCERTAIN_LABEL
        digepath_label = _normalize_digepath_trace_label(
            digepath_predictions.get(key) or digepath_detail.get("digepath_region_semantic")
        )
        digepath_class = _normalize_digepath_class(digepath_detail.get("digepath_class"))
        if not digepath_class and digepath_label:
            for class_name, mapped_label in DIGEPATH_ROI9_CLASS_TO_TRACE_LABEL.items():
                if mapped_label == digepath_label:
                    digepath_class = class_name
                    break
        digepath_confidence = float(digepath_detail.get("digepath_confidence", 0.0) or 0.0)
        if digepath_label and digepath_confidence < 0.01:
            digepath_label = ""
        fused_label, agreement_status, score_origin = _fuse_conch_digepath_label(
            conch_label,
            conch_confidence,
            digepath_label,
            digepath_confidence,
            high_confidence_threshold,
            normal_conch_threshold=normal_conch_threshold,
            normal_digepath_threshold=normal_digepath_threshold,
            normal_gate_fallback_label=normal_gate_fallback_label,
        )
        priority, high_mag, _reason = _conch_trace_defaults(fused_label)
        reason = _conch_digepath_fusion_reason(
            conch_label,
            conch_confidence,
            digepath_label,
            digepath_class,
            digepath_confidence,
            fused_label,
            agreement_status,
        )
        patch = {
            "patch_id": patch_id,
            "name": _conch_assignment_name(fused_label),
            "region_semantic": fused_label,
            "description": reason,
            "require_high_magnification": bool(high_mag),
            "severity_reasoning": reason,
            "diagnostic_priority": int(priority),
            "observation_points": _conch_observation_points(fused_label),
            "conch_region_semantic": conch_label,
            "conch_crc_label": conch_detail.get("conch_crc_label", ""),
            "conch_probs": conch_detail.get("conch_probs", {}),
            "conch_confidence": conch_confidence,
            "conch_raw_prediction": conch_detail.get("conch_raw_prediction", {}),
            "digepath_class": digepath_class,
            "digepath_region_semantic": digepath_label or "not_available_in_this_run",
            "digepath_probs": digepath_detail.get("digepath_probs", {}),
            "digepath_confidence": digepath_confidence,
            "digepath_raw_prediction": digepath_detail.get("digepath_raw_prediction", {}),
            "classifier_source_image": conch_detail.get("classifier_source_image")
            or digepath_detail.get("classifier_source_image", ""),
            "classifier_source_mode": conch_detail.get("classifier_source_mode")
            or digepath_detail.get("classifier_source_mode", ""),
            "classifier_crop_level0_bbox": conch_detail.get("classifier_crop_level0_bbox")
            or digepath_detail.get("classifier_crop_level0_bbox", []),
            "classifier_crop_level0_size": conch_detail.get("classifier_crop_level0_size")
            or digepath_detail.get("classifier_crop_level0_size", []),
            "classifier_crop_output_size": conch_detail.get("classifier_crop_output_size")
            or digepath_detail.get("classifier_crop_output_size", []),
            "classifier_crop_view": conch_detail.get("classifier_crop_view")
            or digepath_detail.get("classifier_crop_view", ""),
            "agreement_status": agreement_status,
            "score_origin": score_origin,
            "pathoreasoner_r1_region_semantic": "not_used_in_conch_digepath_trace",
            "fusion_reasoning": reason,
        }
        if fused_label in {TRACE_NEUTRAL_EPITHELIAL_LABEL, TRACE_NEUTRAL_MUCUS_LABEL, TRACE_NEUTRAL_UNCERTAIN_LABEL}:
            patch["candidate_branches"] = ["serrated", "conventional"]
            patch["routing_hint"] = "needs_morphology_resolution"
        if conch_detail.get("embedding_ref") is not None:
            patch["embedding_ref"] = conch_detail.get("embedding_ref")
        if conch_detail.get("feature_ref") is not None:
            patch["feature_ref"] = conch_detail.get("feature_ref")
        patches.append(patch)
    assignment_payload = {"patches": patches}
    validation = _validate_trace_patch_assignments_payload(
        assignment_payload,
        request,
        groups_source="conch_digepath_patch_assignments",
    )
    output = _build_trace_clusters_from_patch_assignments(
        assignment_payload,
        request,
        validation=validation,
        apply_fallback=True,
    ) or {"clusters": [], "all_clusters": [], "patch_assignments": assignment_payload, "coverage_summary": {}}
    output.setdefault("patch_assignments", assignment_payload)
    output.setdefault("all_clusters", list(output.get("clusters", [])))
    output["coverage_summary"].update(
        {
            "trace_mode": "conch_only",
            "final_trace_schema": "patch_assignments",
            "final_groups_source": "conch_digepath_patch_assignments",
            "conch_enabled": bool((conch_runtime_metadata or {}).get("enabled")),
            "conch_error_count": len((conch_runtime_metadata or {}).get("errors", [])),
            "digepath_fusion_enabled": True,
            "digepath_enabled": bool((digepath_runtime_metadata or {}).get("enabled")),
            "digepath_error_count": len((digepath_runtime_metadata or {}).get("errors", [])),
        }
    )
    return output


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
                '      "l": "serrated",',
                '      "s": 4,',
                '      "d": true,',
                '      "review_stage": "serrated_screening",',
                '      "desc": "short reason",',
                '      "evidence": ["reason 1", "reason 2"]',
                '    }',
                '  ]',
                '}',
                "",
                'Allowed labels for "l": serrated, conventional, normal, background.',
                'Only use cluster_id values from the provided candidate proposals.',
            ]
        )
        return "\n".join(lines)

    lines = [
        header,
        "",
        "You are reviewing a colorectal whole-slide thumbnail that has already been cropped to tissue and divided into a regular grid with visible grid IDs.",
        "Simulate a pathologist's global screening pass.",
        "Focus on coarse workflow routing only: serrated, conventional, normal, or background.",
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
        "- Do not omit any selected patch ID. If a patch looks low value, still assign it to background.",
        "- If your generation format uses <think> and <answer> tags, keep reasoning in <think> and put exactly one raw JSON object in <answer>.",
        "",
        "Workflow trace label set for region_semantic:",
        "- serrated: mucosa suspicious for any serrated pathway lesion; do not subtype at Trace.",
        "- conventional: non-serrated adenomatous or non-serrated lesion-suspicious mucosa; do not subtype at Trace.",
        "- normal: reviewable low-priority/non-lesional mucosa, including possible reactive/inflammatory mucosa for downstream confirmation.",
        "- background: background, artifact, muscle, stroma, or other discardable low-value regions.",
        "",
        "Screening guidance:",
        "- Look for mucosal regions that may warrant closer review for serrated architecture, conventional adenoma architecture, or inflammatory/reactive polyp context.",
        "- Helpful cues may include pale or mucus-rich surface appearance, contour irregularity, broad lesion shape, crypt crowding suggestive of serration, or a lesion edge worth higher-magnification inspection.",
        "- Do not subtype serrated or conventional lesions at Trace; leave SSL/HP/TSA/tubular/tubulovillous/inflammatory resolution to downstream review.",
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
            '      "region_semantic": "serrated",',
            '      "description": "brief visual summary of the mucosal region and why it may warrant review",',
            '      "require_high_magnification": true,',
            '      "severity_reasoning": "brief reason for the assigned diagnostic priority",',
            '      "diagnostic_priority": 4,',
            '      "observation_points": ["possible serrated surface pattern", "mucosal edge worth closer review"]',
            "    },",
            "    {",
            '      "patch_id": [1, 0],',
            '      "name": "Background/stroma remainder",',
            '      "region_semantic": "background",',
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
            "- region_semantic must be one of: serrated, conventional, normal, background.",
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
        "- Any low-value leftover patch must be assigned to background rather than omitted.",
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
    ssl_criteria = list(bundle["runtime"]["observe"].get("ssl_criteria", []))
    hp_criteria = list(bundle["runtime"]["observe"].get("hp_criteria", []))
    tsa_criteria = list(bundle["runtime"]["observe"].get("tsa_criteria", []))
    tsa_cytology_criteria = list(bundle["runtime"]["observe"].get("tsa_cytological_atypia_criteria", []))
    inflammatory_criteria = list(bundle["runtime"]["observe"].get("inflammatory_criteria", []))
    lower_text = text.lower()
    serrated_hits = _blank_hits(serrated_criteria)
    abnormal_crypt_hits = _blank_hits(abnormal_crypt_criteria)
    conventional_hits = _blank_hits(conventional_criteria)
    serrated_dysplasia_hits = _blank_hits(dysplasia_criteria)
    conventional_dysplasia_hits = _blank_hits(dysplasia_criteria)
    ssl_hits = _blank_hits(ssl_criteria)
    hp_hits = _blank_hits(hp_criteria)
    tsa_hits = _blank_hits(tsa_criteria)
    tsa_cytology_hits = _blank_hits(tsa_cytology_criteria)
    inflammatory_hits = _blank_hits(inflammatory_criteria)
    branch_recovery_hint = "none"
    branch_recovery_reason = ""

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
        "high_grade_focus": ["high grade", "high-grade", "hgd"],
        "marked_cytologic_atypia": ["marked atypia", "marked cytologic", "severe atypia"],
        "tubular_or_tubulovillous_architecture": ["tubular", "tubulovillous", "villous", "adenoma"],
        "tubular_architecture": ["tubular", "adenoma"],
        "villous_component": ["villous", "tubulovillous"],
        "high_villous_component": [">75%", "greater than 75", "predominantly villous", "villous adenoma"],
        "crowded_adenomatous_glands": ["crowded", "adenomatous", "gland"],
        "pencillate_hyperchromatic_nuclei": ["pencillate", "hyperchrom", "nuclei"],
        "basal_crypt_dilatation": ["basal", "dilat", "dilated", "dilation"],
        "basal_crypt_deformation": ["basal", "deformation", "boot", "l-shaped", "t-shaped", "horizontal"],
        "horizontal_or_boot_shaped_crypt": ["horizontal", "boot", "l-shaped", "t-shaped"],
        "surface_limited_serration": ["surface-limited", "surface limited", "upper crypt", "hyperplastic"],
        "straight_crypt_bases": ["straight crypt", "straight bases"],
        "lacks_basal_architectural_distortion": ["lacks basal", "no basal", "without basal"],
        "villiform_or_filiform_architecture": ["villiform", "filiform", "traditional serrated", "tsa"],
        "eosinophilic_cytoplasm": ["eosinophilic"],
        "ectopic_crypt_formation": ["ectopic crypt"],
        "ectopic_crypt_foci": ["ectopic crypt", "ectopic crypt foci", "perpendicular bud"],
        "slit_like_serration": ["slit-like", "slit like"],
        "global_color_shift": ["pink", "eosinophilic", "color shift"],
        "epithelial_banding_pattern": ["banding", "palisading", "dark-purple band", "dark purple band"],
        "cytoplasmic_eosinophilia": ["eosinophilic cytoplasm", "bright pink", "cytoplasmic eosinophilia"],
        "pencillate_nuclei": ["pencillate", "rod-shaped", "palisaded nuclei"],
        "erosion": ["erosion", "eroded"],
        "granulation_tissue": ["granulation"],
        "mixed_inflammation": ["mixed inflammation", "inflammatory", "inflamed"],
        "reactive_regenerative_change": ["reactive", "regenerative"],
        "lacks_adenomatous_or_serrated_architecture": ["lacks adenomatous", "no adenomatous", "no serrated"],
    }
    if review_goal in {"serrated_overview_assessment", "serrated_lesion_assessment"}:
        for criterion in serrated_hits:
            serrated_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
        conventional_cues = ("conventional", "adenomatous", "adenoma", "gland crowding", "tubular", "villous", "non-serrated")
        normal_cues = ("normal", "benign", "low-priority", "low priority", "non-lesional", "no lesion", "unremarkable")
        if any(cue in lower_text for cue in conventional_cues):
            branch_recovery_hint = "conventional"
            branch_recovery_reason = "Text observation suggests conventional/non-serrated adenomatous evidence after serrated overview."
        elif any(cue in lower_text for cue in normal_cues):
            branch_recovery_hint = "normal"
            branch_recovery_reason = "Text observation suggests normal or non-lesional mucosa after serrated overview."
    elif review_goal in {"ssl_assessment", "abnormal_crypt_assessment"}:
        for criterion in ssl_hits:
            ssl_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
        for criterion in abnormal_crypt_hits:
            abnormal_crypt_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal == "hp_assessment":
        for criterion in hp_hits:
            hp_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal == "tsa_assessment":
        for criterion in tsa_hits:
            tsa_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal in {"conventional_overview_assessment", "conventional_architecture_assessment", "conventional_adenoma_assessment"}:
        for criterion in conventional_hits:
            conventional_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal == "reactive_regenerative_assessment":
        for criterion in inflammatory_hits:
            inflammatory_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal in {"ssl_dysplasia_assessment", "tsa_dysplasia_assessment", "serrated_dysplasia_assessment"}:
        for criterion in serrated_dysplasia_hits:
            serrated_dysplasia_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal == "tsa_cytological_atypia_assessment":
        for criterion in tsa_cytology_hits:
            tsa_cytology_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal == "conventional_dysplasia_assessment":
        for criterion in conventional_dysplasia_hits:
            conventional_dysplasia_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    else:
        for criterion in inflammatory_hits:
            inflammatory_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
        pass
    dysplasia_hits = _combine_hits_maps(serrated_dysplasia_hits, conventional_dysplasia_hits)

    if review_goal == "serrated_overview_assessment":
        if _supporting_findings_from_hits(serrated_hits):
            stage_decision = "supports_serrated_overview"
            branch_recovery_hint = "none"
            branch_recovery_reason = ""
            next_step = "Proceed to 5x SSL and TSA assessment if serrated overview remains supported."
        else:
            stage_decision = "serrated_overview_not_supported_or_indeterminate"
            next_step = "Suppress SSL/TSA assessment and trigger alternate-route recovery review."
    elif review_goal == "serrated_lesion_assessment":
        stage_decision = "supports_serrated_lesion"
        next_step = "Proceed to abnormal crypt review if the lesion remains within the serrated pathway."
    elif review_goal == "ssl_assessment":
        stage_decision = "ssl_architecture_supported"
        next_step = "Proceed to 10x SSL dysplasia review if SSL architecture is convincing."
    elif review_goal == "hp_assessment":
        stage_decision = "hp_architecture_supported" if _supporting_findings_from_hits(hp_hits) else "hp_architecture_not_supported_or_indeterminate"
        next_step = "Use HP evidence only if SSL and TSA support remain absent; do not trigger dysplasia review from HP alone."
    elif review_goal == "abnormal_crypt_assessment":
        stage_decision = "supports_abnormal_crypt"
        next_step = "Proceed to serrated dysplasia review only if abnormal crypt support is convincing."
    elif review_goal == "tsa_assessment":
        stage_decision = "tsa_architecture_supported"
        next_step = "Proceed to 10x TSA cytology confirmation and dysplasia review if TSA support persists."
    elif review_goal == "conventional_overview_assessment":
        stage_decision = "supports_conventional_overview"
        next_step = "Proceed to 5x conventional architecture assessment."
    elif review_goal in {"conventional_architecture_assessment", "conventional_adenoma_assessment"}:
        stage_decision = "supports_conventional_architecture" if review_goal == "conventional_architecture_assessment" else "supports_conventional_adenoma"
        next_step = "Proceed to conventional dysplasia review within the conventional adenoma branch."
    elif review_goal == "reactive_regenerative_assessment":
        stage_decision = "reactive_regenerative_supported" if _supporting_findings_from_hits(inflammatory_hits) else "reactive_regenerative_not_supported_or_indeterminate"
        next_step = "Use this as inflammatory/reactive support when conventional architecture is not established; otherwise record it as a conflict."
    elif review_goal in {"ssl_dysplasia_assessment", "tsa_dysplasia_assessment", "serrated_dysplasia_assessment"}:
        stage_decision = (
            "ssl_dysplasia_supported" if review_goal == "ssl_dysplasia_assessment" else
            "tsa_dysplasia_supported" if review_goal == "tsa_dysplasia_assessment" else
            "serrated_dysplasia_supported"
        )
        next_step = "Integrate the serrated branch impression and finalize the report."
    elif review_goal == "tsa_cytological_atypia_assessment":
        stage_decision = "tsa_cytological_atypia_supported"
        next_step = "Use this as TSA lineage support only; do not equate it with TSAD."
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
        + _supporting_findings_from_hits(conventional_hits)
        + _supporting_findings_from_hits(ssl_hits)
        + _supporting_findings_from_hits(tsa_hits)
        + _supporting_findings_from_hits(tsa_cytology_hits),
        "level_2_findings": _supporting_findings_from_hits(abnormal_crypt_hits),
        "level_3_findings": _supporting_findings_from_hits(dysplasia_hits),
        "stage_decision": stage_decision,
        "serrated_hits": serrated_hits,
        "abnormal_crypt_hits": abnormal_crypt_hits,
        "conventional_hits": conventional_hits,
        "serrated_dysplasia_hits": serrated_dysplasia_hits,
        "conventional_dysplasia_hits": conventional_dysplasia_hits,
        "dysplasia_hits": dysplasia_hits,
        "ssl_hits": ssl_hits,
        "hp_hits": hp_hits,
        "tsa_hits": tsa_hits,
        "tsa_cytological_atypia_hits": tsa_cytology_hits,
        "inflammatory_hits": inflammatory_hits,
        "branch_recovery_hint": branch_recovery_hint,
        "branch_recovery_reason": branch_recovery_reason,
        "confidence": 0.62,
    }


def _supporting_findings_from_hits(hits):
    return [criterion for criterion, status in hits.items() if status == "supporting"]


def _supporting_count_from_checklist(checklist):
    return len([value for value in checklist.values() if isinstance(value, dict) and value.get("status") == "supporting"])


def _checklist_supports(checklist, key):
    return isinstance(checklist, dict) and checklist.get(key, {}).get("status") == "supporting"


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
        cluster.get("l") in TRACE_SERRATED_LABELS
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
    trace_support = any(cluster.get("l") in TRACE_CONVENTIONAL_LABELS for cluster in trace_clusters)
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
    nuclear_support = _checklist_supports(checklist, "nuclear_enlargement_stratification") or _checklist_supports(checklist, "hyperchromasia")
    high_grade_support = (
        _checklist_supports(checklist, "architectural_crowding")
        or _checklist_supports(checklist, "mitotic_activity_atypia")
        or _checklist_supports(checklist, "high_grade_focus")
        or _checklist_supports(checklist, "marked_cytologic_atypia")
    )
    positive = support_count >= 2 and nuclear_support and high_grade_support
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


def _trace_subtype_vote(trace_clusters):
    votes = {
        "SSL": 0,
        "HP": 0,
        "TSA": 0,
        "Unclassified serrated adenoma": 0,
        "Tubular adenoma": 0,
        "Tubulovillous adenoma": 0,
        "Inflammatory": 0,
    }
    for cluster in trace_clusters:
        label = cluster.get("l")
        priority = max(1, int(cluster.get("s", 1) or 1))
        metadata = cluster.get("metadata", {}) if isinstance(cluster.get("metadata"), dict) else {}
        if label in (TRACE_LEGACY_SSL_LABEL, TRACE_LEGACY_SSL_HIGH_LABEL):
            votes["SSL"] += priority
        elif label == TRACE_HP_LABEL:
            votes["HP"] += priority
        elif label == TRACE_TSA_LABEL:
            votes["TSA"] += priority
        elif label == TRACE_UNCLASSIFIED_SERRATED_LABEL:
            votes["Unclassified serrated adenoma"] += priority
        elif label == TRACE_TUBULOVILLOUS_LABEL or metadata.get("conventional_subtype_hint") == "tubulovillous_adenoma_like":
            votes["Tubulovillous adenoma"] += priority
        elif label == TRACE_TUBULAR_LABEL:
            votes["Tubular adenoma"] += priority
        elif label == TRACE_INFLAMMATORY_LABEL:
            votes["Inflammatory"] += priority
    return votes


def _case_trace_branch(trace_clusters):
    branch_scores = {}
    for cluster in trace_clusters:
        branch = _trace_branch_for_label(cluster.get("l"))
        priority = max(1, int(cluster.get("s", 1) or 1))
        branch_scores[branch] = branch_scores.get(branch, 0) + priority
    if not branch_scores:
        return "background"
    return max(branch_scores, key=lambda key: (branch_scores[key], key != "background"))


def _class11_assessment(
    trace_clusters,
    serrated_assessment,
    serrated_dysplasia_assessment,
    conventional_adenoma_assessment,
    conventional_dysplasia_assessment,
    ssl_checklist,
    hp_checklist,
    tsa_checklist,
    inflammatory_checklist,
    conventional_adenoma_checklist,
    resolved_branch=None,
    branch_correction_reason="",
):
    votes = _trace_subtype_vote(trace_clusters)
    trace_branch = _case_trace_branch(trace_clusters)
    active_branch = str(resolved_branch or trace_branch or "").strip() or trace_branch
    correction_reason = str(branch_correction_reason or "").strip()
    if active_branch != trace_branch and not correction_reason:
        active_branch = trace_branch
        correction_reason = ""
    votes["SSL"] += _supporting_count_from_checklist(ssl_checklist) * 2
    votes["HP"] += _supporting_count_from_checklist(hp_checklist) * 2
    votes["TSA"] += _supporting_count_from_checklist(tsa_checklist) * 2
    votes["Inflammatory"] += _supporting_count_from_checklist(inflammatory_checklist) * 2
    if conventional_adenoma_checklist.get("villous_component", {}).get("status") == "supporting":
        votes["Tubulovillous adenoma"] += 3
    if conventional_adenoma_checklist.get("tubular_architecture", {}).get("status") == "supporting":
        votes["Tubular adenoma"] += 2
    if conventional_adenoma_checklist.get("tubular_or_tubulovillous_architecture", {}).get("status") == "supporting":
        votes["Tubular adenoma"] += 1
    if conventional_adenoma_checklist.get("high_villous_component", {}).get("status") == "supporting":
        votes["Tubulovillous adenoma"] += 4

    serrated_positive = bool(serrated_assessment.get("positive"))
    conventional_positive = bool(conventional_adenoma_assessment.get("positive"))
    inflammatory_supported = votes["Inflammatory"] > 0
    inflammatory_positive = inflammatory_supported and not serrated_positive and not conventional_positive
    conflicts = []
    if serrated_positive and conventional_positive:
        conflicts.append("serrated_and_conventional_branches_both_supported")
    if conventional_positive and inflammatory_supported:
        conflicts.append("conventional_architecture_with_reactive_regenerative_mimic")

    villous_support = (
        _checklist_supports(conventional_adenoma_checklist, "villous_component")
        or _checklist_supports(conventional_adenoma_checklist, "high_villous_component")
        or votes["Tubulovillous adenoma"] > 0
    )
    villous_component_category = "not_assessed"
    if _checklist_supports(conventional_adenoma_checklist, "high_villous_component"):
        villous_component_category = ">75%"
    elif _checklist_supports(conventional_adenoma_checklist, "villous_component") or votes["Tubulovillous adenoma"] > votes["Tubular adenoma"]:
        villous_component_category = "25-75%"
    elif conventional_positive:
        villous_component_category = "<25%"

    if trace_branch == "background" and active_branch == "background":
        return {
            "label": None,
            "final_11_class": None,
            "branch": None,
            "subtype": None,
            "trace_branch": trace_branch,
            "resolved_branch": active_branch,
            "branch_correction_reason": correction_reason,
            "dysplasia_positive": False,
            "high_grade_or_definite_dysplasia": False,
            "villous_component_category": villous_component_category,
            "confidence": 0.0,
            "positive": False,
            "classification_status": "non_diagnostic",
            "non_diagnostic_reason": "Trace branch is background/low-value tissue and no diagnostic lesion branch is available.",
            "supporting_checklists": {"ssl": [], "hp": [], "tsa": [], "inflammatory": [], "conventional_adenoma": []},
            "conflicts": ["background_trace_branch"],
            "class_scores": {key: round(float(value), 4) for key, value in votes.items()},
        }

    if not serrated_positive and not conventional_positive and not inflammatory_positive:
        return {
            "label": None,
            "final_11_class": None,
            "branch": None,
            "subtype": None,
            "trace_branch": trace_branch,
            "resolved_branch": active_branch,
            "branch_correction_reason": correction_reason,
            "dysplasia_positive": False,
            "high_grade_or_definite_dysplasia": False,
            "villous_component_category": villous_component_category,
            "confidence": 0.0,
            "positive": False,
            "classification_status": "insufficient_evidence",
            "non_diagnostic_reason": "No serrated, conventional adenoma, or inflammatory/reactive branch has sufficient supporting evidence.",
            "supporting_checklists": {
                "ssl": [],
                "hp": [],
                "tsa": [],
                "inflammatory": [],
                "conventional_adenoma": [],
            },
            "conflicts": ["no_diagnostic_branch_supported"],
            "class_scores": {key: round(float(value), 4) for key, value in votes.items()},
        }

    ssl_core_support = _supporting_count_from_checklist(ssl_checklist)
    hp_core_support = _supporting_count_from_checklist(hp_checklist)
    tsa_core_support = _supporting_count_from_checklist(tsa_checklist)
    serrated_subtype_supported = ssl_core_support >= 2 or hp_core_support >= 2 or tsa_core_support >= 2 or villous_support or votes["Unclassified serrated adenoma"] > 0 or votes["SSL"] > 0 or votes["HP"] > 0 or votes["TSA"] > 0
    conventional_subtype_supported = _supporting_count_from_checklist(conventional_adenoma_checklist) > 0 or votes["Tubular adenoma"] > 0 or votes["Tubulovillous adenoma"] > 0

    if active_branch == "serrated" and serrated_positive and not serrated_subtype_supported:
        return {
            "label": None,
            "final_11_class": None,
            "branch": "serrated",
            "subtype": None,
            "trace_branch": trace_branch,
            "resolved_branch": active_branch,
            "branch_correction_reason": correction_reason,
            "dysplasia_positive": False,
            "high_grade_or_definite_dysplasia": False,
            "villous_component_category": villous_component_category,
            "confidence": 0.0,
            "positive": False,
            "classification_status": "insufficient_evidence",
            "non_diagnostic_reason": "Trace branch is serrated, but SSL/HP/TSA/unclassified serrated subtype evidence is insufficient.",
            "supporting_checklists": {"ssl": [], "hp": [], "tsa": [], "inflammatory": [], "conventional_adenoma": []},
            "conflicts": ["serrated_trace_without_subtype_support"],
            "class_scores": {key: round(float(value), 4) for key, value in votes.items()},
        }

    if active_branch == "serrated" and serrated_positive:
        branch = "serrated"
        serrated_scores = {key: votes[key] for key in ("SSL", "HP", "TSA", "Unclassified serrated adenoma")}
        if tsa_core_support >= 2:
            subtype = "TSA"
        elif ssl_core_support >= 2:
            subtype = "SSL"
        elif hp_core_support >= 2 and ssl_core_support == 0 and tsa_core_support == 0:
            subtype = "HP"
        elif villous_support or serrated_scores["Unclassified serrated adenoma"] > 0:
            subtype = "Unclassified serrated adenoma"
        else:
            subtype = max(serrated_scores, key=lambda key: (serrated_scores[key], key == "SSL"))
            if serrated_scores[subtype] <= 0:
                subtype = "Unclassified serrated adenoma"
        dysplasia_positive = bool(serrated_dysplasia_assessment.get("positive"))
        if subtype == "SSL" and dysplasia_positive:
            label = "SSLD"
        elif subtype == "TSA" and dysplasia_positive:
            label = "TSAD"
        else:
            label = subtype
    elif active_branch in ("conventional_adenoma", "conventional") and conventional_positive and conventional_subtype_supported:
        branch = "conventional_adenoma"
        subtype = "Tubulovillous adenoma" if villous_component_category in ("25-75%", ">75%") else "Tubular adenoma"
        dysplasia_positive = bool(conventional_dysplasia_assessment.get("positive"))
        if subtype == "Tubulovillous adenoma" and dysplasia_positive:
            label = "TVAD"
        elif dysplasia_positive:
            label = "TAD"
        else:
            label = subtype
    elif active_branch in ("conventional_adenoma", "conventional", "normal") and inflammatory_supported and not conventional_subtype_supported:
        branch = "inflammatory"
        subtype = "Inflammatory"
        dysplasia_positive = False
        label = "Inflammatory"
    else:
        return {
            "label": None,
            "final_11_class": None,
            "branch": active_branch if active_branch not in ("background", "") else None,
            "subtype": None,
            "trace_branch": trace_branch,
            "resolved_branch": active_branch,
            "branch_correction_reason": correction_reason,
            "dysplasia_positive": False,
            "high_grade_or_definite_dysplasia": False,
            "villous_component_category": villous_component_category,
            "confidence": 0.0,
            "positive": False,
            "classification_status": "insufficient_evidence",
            "non_diagnostic_reason": "Resolved branch does not have enough subtype or inflammatory evidence for final 11-class classification.",
            "supporting_checklists": {
                "ssl": _supporting_findings_from_hits({key: value.get("status", "not_assessed") for key, value in ssl_checklist.items()}),
                "hp": _supporting_findings_from_hits({key: value.get("status", "not_assessed") for key, value in hp_checklist.items()}),
                "tsa": _supporting_findings_from_hits({key: value.get("status", "not_assessed") for key, value in tsa_checklist.items()}),
                "inflammatory": _supporting_findings_from_hits({key: value.get("status", "not_assessed") for key, value in inflammatory_checklist.items()}),
                "conventional_adenoma": _supporting_findings_from_hits({key: value.get("status", "not_assessed") for key, value in conventional_adenoma_checklist.items()}),
            },
            "conflicts": conflicts + ["resolved_branch_without_final_class_support"],
            "class_scores": {key: round(float(value), 4) for key, value in votes.items()},
        }

    max_score = max(votes.values()) if votes else 0
    confidence = min(0.95, max(0.05, 0.20 + 0.08 * float(max_score)))
    return {
        "label": label,
        "final_11_class": label,
        "branch": branch,
        "subtype": subtype,
        "trace_branch": trace_branch,
        "resolved_branch": active_branch,
        "branch_correction_reason": correction_reason,
        "dysplasia_positive": dysplasia_positive,
        "high_grade_or_definite_dysplasia": dysplasia_positive,
        "villous_component_category": villous_component_category,
        "confidence": round(confidence, 4),
        "positive": True,
        "classification_status": "classified",
        "non_diagnostic_reason": "",
        "supporting_checklists": {
            "ssl": _supporting_findings_from_hits({key: value.get("status", "not_assessed") for key, value in ssl_checklist.items()}),
            "hp": _supporting_findings_from_hits({key: value.get("status", "not_assessed") for key, value in hp_checklist.items()}),
            "tsa": _supporting_findings_from_hits({key: value.get("status", "not_assessed") for key, value in tsa_checklist.items()}),
            "inflammatory": _supporting_findings_from_hits({key: value.get("status", "not_assessed") for key, value in inflammatory_checklist.items()}),
            "conventional_adenoma": _supporting_findings_from_hits({key: value.get("status", "not_assessed") for key, value in conventional_adenoma_checklist.items()}),
        },
        "conflicts": conflicts,
        "class_scores": {key: round(float(value), 4) for key, value in votes.items()},
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
