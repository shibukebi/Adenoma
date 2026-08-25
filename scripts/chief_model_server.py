#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
import traceback
import uuid
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def _current_env_cuda_libs():
    prefix = Path(sys.prefix)
    version_dir = "python{0}.{1}".format(sys.version_info.major, sys.version_info.minor)
    candidates = [
        prefix / "lib" / version_dir / "site-packages" / "nvidia" / "nvjitlink" / "lib",
        prefix / "lib" / version_dir / "site-packages" / "nvidia" / "cusparse" / "lib",
        prefix / "lib",
    ]
    return [str(path) for path in candidates if path.exists()]


def ensure_torch_runtime():
    if os.environ.get("_CHIEF_SERVER_LD_READY") == "1":
        return
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    new_parts = list(_current_env_cuda_libs())
    for part in current:
        if part not in new_parts:
            new_parts.append(part)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(new_parts)
    env["_CHIEF_SERVER_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


ensure_torch_runtime()

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # noqa: E402


REQUIRED_RESPONSE_KEYS = {
    "review_id",
    "source_step_id",
    "decision",
    "continue_reason",
    "chief_confidence",
    "resolved_branch_state",
    "sufficient_evidence",
    "unresolved_questions",
    "next_visual_target",
}

WRAPPER_RESPONSE_KEYS = (
    "early_stop",
    "continue",
    "result",
    "response",
    "review",
    "global_review",
    "chief_review",
    "decision_payload",
    "output",
)


class ChiefGenerationError(RuntimeError):
    def __init__(self, message: str, raw_text: str = "", think_text: str = "", answer_text: str = ""):
        super(ChiefGenerationError, self).__init__(message)
        self.raw_text = str(raw_text or "")
        self.think_text = str(think_text or "")
        self.answer_text = str(answer_text or "")


def _bytes_to_gb(value: int) -> float:
    return round(float(value) / float(1024**3), 3)


def _gpu_snapshot():
    available = bool(torch.cuda.is_available())
    payload = {
        "cuda_available": available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": int(torch.cuda.device_count()) if available else 0,
    }
    if not available:
        return payload
    current_index = 0
    payload["device_name"] = torch.cuda.get_device_name(current_index)
    payload["device_id"] = current_index
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(current_index)
        payload["vram_free_gb"] = _bytes_to_gb(free_bytes)
        payload["vram_total_gb"] = _bytes_to_gb(total_bytes)
        payload["vram_used_gb"] = _bytes_to_gb(total_bytes - free_bytes)
    except Exception as exc:
        payload["vram_error"] = str(exc)
    return payload


def _log_json(event_type: str, payload: dict):
    row = {"timestamp": int(time.time()), "event": event_type, **payload}
    print(json.dumps(row, ensure_ascii=False), flush=True)


def _append_error_log(error_log_path: str, event_type: str, payload: dict):
    path = Path(error_log_path or os.environ.get("CHIEF_ERROR_LOG_PATH") or REPO_ROOT / "artifacts" / "chief_model_server_errors.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": int(time.time()), "event": event_type, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _extract_first_json_object(text: str):
    text = str(text or "")
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


def _looks_like_review_payload(value):
    return isinstance(value, dict) and bool(REQUIRED_RESPONSE_KEYS.intersection(value.keys()))


def _coerce_string_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_branch_state(value):
    if not isinstance(value, dict):
        value = {}
    normalized = {}
    aliases = {
        "serrated": ("serrated", "ssl", "serrated_branch"),
        "abnormal_crypt": ("abnormal_crypt", "crypt", "abnormal_crypt_architecture"),
        "conventional": ("conventional", "adenoma", "conventional_adenoma"),
        "dysplasia": ("dysplasia", "serrated_dysplasia", "conventional_dysplasia"),
    }
    allowed = {"supported", "opposed", "unresolved"}
    for target_key, source_keys in aliases.items():
        raw = ""
        for source_key in source_keys:
            if source_key in value:
                raw = str(value.get(source_key) or "").strip().lower()
                break
        normalized[target_key] = raw if raw in allowed else "unresolved"
    return normalized


def _normalize_target_branch(value):
    text = str(value or "").strip()
    if text in {"serrated", "conventional", "non_serrated"}:
        return text
    if text in {"inflammatory", "background", "normal"}:
        return "non_serrated"
    return "non_serrated"


def _normalize_next_visual_target(value, request: "ChiefReviewRequest"):
    if not isinstance(value, dict):
        return value
    metadata = value.get("metadata", {}) if isinstance(value.get("metadata"), dict) else {}
    cluster_id = value.get("target_cluster_id") or metadata.get("cluster_id") or metadata.get("source_group_id")
    target_branch = value.get("target_branch") or metadata.get("workflow_branch")
    region_semantic = value.get("target_region_semantic") or metadata.get("cluster_label")
    preferred_magnification = value.get("preferred_magnification", value.get("m"))
    if not cluster_id and value.get("step_id"):
        for pending in request.pending_steps or []:
            if str(pending.get("step_id")) == str(value.get("step_id")):
                pending_meta = pending.get("metadata", {}) if isinstance(pending.get("metadata"), dict) else {}
                cluster_id = pending_meta.get("cluster_id") or pending_meta.get("source_group_id")
                target_branch = target_branch or pending_meta.get("workflow_branch")
                region_semantic = region_semantic or pending_meta.get("cluster_label")
                preferred_magnification = preferred_magnification or pending.get("m")
                break
    valid_cluster_ids = {
        str(cluster.get("cluster_id") or "")
        for cluster in (request.trace_clusters or [])
        if isinstance(cluster, dict)
    }
    valid_cluster_ids.update(
        str((pending.get("metadata", {}) if isinstance(pending.get("metadata"), dict) else {}).get("cluster_id") or "")
        for pending in (request.pending_steps or [])
        if isinstance(pending, dict)
    )
    if str(cluster_id or "") not in valid_cluster_ids and request.pending_steps:
        pending = request.pending_steps[0]
        pending_meta = pending.get("metadata", {}) if isinstance(pending.get("metadata"), dict) else {}
        cluster_id = pending_meta.get("cluster_id") or pending_meta.get("source_group_id")
        target_branch = pending_meta.get("workflow_branch")
        region_semantic = pending_meta.get("cluster_label")
        preferred_magnification = pending.get("m")
    try:
        preferred_magnification = float(preferred_magnification)
    except Exception:
        preferred_magnification = 5.0
    if preferred_magnification not in {2.5, 5.0, 10.0}:
        preferred_magnification = 10.0 if preferred_magnification > 5.0 else (2.5 if preferred_magnification < 5.0 else 5.0)
    return {
        "target_cluster_id": str(cluster_id or ""),
        "target_branch": _normalize_target_branch(target_branch),
        "target_region_semantic": str(region_semantic or "normal_mucosa"),
        "target_morphology_prompt": _coerce_string_list(
            value.get("target_morphology_prompt")
            or value.get("need_to_see")
            or value.get("review_goal")
            or "inspect the next planned cell-level observation target"
        ),
        "preferred_magnification": preferred_magnification,
        "priority_reason": str(value.get("priority_reason") or value.get("need_to_see") or "Chief selected the next planned observation target."),
    }


def _unwrap_review_payload(parsed):
    if not isinstance(parsed, dict):
        return parsed
    if _looks_like_review_payload(parsed):
        return dict(parsed)
    for key in WRAPPER_RESPONSE_KEYS:
        nested = parsed.get(key)
        if isinstance(nested, dict):
            unwrapped = _unwrap_review_payload(nested)
            if isinstance(unwrapped, dict):
                merged = dict(unwrapped)
                for outer_key, outer_value in parsed.items():
                    if outer_key not in WRAPPER_RESPONSE_KEYS and outer_key not in merged:
                        merged[outer_key] = outer_value
                if _looks_like_review_payload(merged) or key in {"early_stop", "continue"}:
                    return merged
    return dict(parsed)


def _normalize_chief_response_payload(parsed, request: "ChiefReviewRequest"):
    payload = _unwrap_review_payload(parsed)
    if not isinstance(payload, dict):
        payload = {}
    payload = dict(payload)
    record = request.record or {}
    decision = str(payload.get("decision") or "").strip()
    if not decision:
        if isinstance(parsed, dict) and "early_stop" in parsed:
            decision = "early_stop"
        elif isinstance(parsed, dict) and "continue" in parsed:
            decision = "continue"
    if decision not in {"continue", "early_stop"}:
        decision = "early_stop" if payload.get("next_visual_target") is None and payload.get("sufficient_evidence") else "continue"
    payload["decision"] = decision
    payload["review_id"] = str(payload.get("review_id") or "global_review_{0}".format(str(record.get("step_id") or request.step.get("step_id") or "unknown")))
    payload["source_step_id"] = str(payload.get("source_step_id") or record.get("step_id") or request.step.get("step_id") or "")
    payload["continue_reason"] = str(payload.get("continue_reason") or payload.get("reason") or "")
    try:
        payload["chief_confidence"] = float(payload.get("chief_confidence", payload.get("confidence", 0.5)))
    except Exception:
        payload["chief_confidence"] = 0.5
    payload["resolved_branch_state"] = _normalize_branch_state(payload.get("resolved_branch_state") or payload.get("branch_state"))
    payload["sufficient_evidence"] = _coerce_string_list(payload.get("sufficient_evidence") or payload.get("evidence"))
    payload["unresolved_questions"] = _coerce_string_list(payload.get("unresolved_questions") or payload.get("remaining_questions"))
    if decision == "early_stop":
        payload["next_visual_target"] = None
        if not payload["sufficient_evidence"]:
            payload["sufficient_evidence"] = ["Chief model selected early_stop after reviewing the accumulated observation memory."]
    else:
        if not payload["continue_reason"]:
            payload["continue_reason"] = "Chief model selected continue after reviewing the accumulated observation memory."
        if payload.get("next_visual_target") is None and request.pending_steps:
            payload["next_visual_target"] = request.pending_steps[0]
        elif payload.get("next_visual_target") is None:
            payload["decision"] = "early_stop"
            decision = "early_stop"
            payload["continue_reason"] = ""
            payload["sufficient_evidence"] = payload["sufficient_evidence"] or [
                "Chief model found no remaining planned cell-level observation targets."
            ]
        payload["next_visual_target"] = _normalize_next_visual_target(payload.get("next_visual_target"), request)
    payload["branch_correction_reason"] = str(payload.get("branch_correction_reason") or "")
    target = payload.get("next_visual_target")
    step_metadata = request.step.get("metadata", {}) if isinstance(request.step.get("metadata"), dict) else {}
    current_branch = str(step_metadata.get("workflow_branch") or "").strip()
    if decision == "continue" and isinstance(target, dict) and target.get("target_branch") != current_branch and not payload["branch_correction_reason"]:
        payload["branch_correction_reason"] = "Chief selected a pending target from another workflow branch after completing the current cell review."
    return payload


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


class ChiefReviewRequest(BaseModel):
    case_id: str
    step: Dict[str, Any]
    record: Dict[str, Any]
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    global_reviews: List[Dict[str, Any]] = Field(default_factory=list)
    trace_clusters: List[Dict[str, Any]] = Field(default_factory=list)
    pending_steps: List[Dict[str, Any]] = Field(default_factory=list)


class ChiefReviewResponse(BaseModel):
    review_id: str
    source_step_id: str
    decision: str
    continue_reason: str
    chief_confidence: float
    resolved_branch_state: Dict[str, str]
    sufficient_evidence: List[str]
    unresolved_questions: List[str]
    next_visual_target: Optional[Dict[str, Any]] = None
    branch_correction_reason: str = ""
    request_id: str
    model_name: str
    review_source: str
    raw_generated_text: str = ""
    thought_text: str = ""
    answer_candidate_text: str = ""
    parse_error: str = ""
    prompt_token_count: int = 0
    prompt_truncated: bool = False
    gpu_device_id: Optional[int] = None
    cuda_visible_devices: Optional[str] = None
    round_trip_ms: int


class OutputRepairResponse(BaseModel):
    repaired_json: Dict[str, Any]
    repair_actions: List[str]
    unrecoverable_errors: List[str]
    confidence: float
    request_id: str
    model_name: str
    repair_source: str
    raw_generated_text: str = ""
    thought_text: str = ""
    answer_candidate_text: str = ""
    parse_error: str = ""
    prompt_token_count: int = 0
    prompt_truncated: bool = False
    gpu_device_id: Optional[int] = None
    cuda_visible_devices: Optional[str] = None
    round_trip_ms: int


def parse_args():
    parser = argparse.ArgumentParser(description="Chief Pathologist FastAPI server backed by DeepSeek-R1-Distill-Qwen.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--fallback-model-path", default="")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--error-log-path", default=os.environ.get("CHIEF_ERROR_LOG_PATH", str(REPO_ROOT / "artifacts" / "chief_model_server_errors.log")))
    return parser.parse_args()


def _load_model_bundle(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def _build_prompt(request: ChiefReviewRequest):
    system_prompt = (
        "You are the Chief Pathologist. Your job is not to describe a single patch, but to read the full accumulated "
        "observation memory and decide whether the scan should continue or early_stop. "
        "You may correct the branch direction if the existing evidence justifies it. "
        "Return JSON only. "
        "The JSON object must use this root-level schema: "
        "{review_id: string, source_step_id: string, decision: 'continue'|'early_stop', continue_reason: string, "
        "chief_confidence: number, resolved_branch_state: {serrated: string, abnormal_crypt: string, conventional: string, dysplasia: string}, "
        "sufficient_evidence: string[], unresolved_questions: string[], next_visual_target: object|null, branch_correction_reason: string}. "
        "DO NOT wrap the fields inside any nested structures like 'early_stop' or 'result'. "
        "All keys (review_id, decision, etc.) MUST reside at the root level of the JSON object."
    )
    user_prompt = (
        "Decide whether to continue or early_stop.\n"
        "Rules:\n"
        "- Return exactly one JSON object with all required keys at the root level.\n"
        "- DO NOT nest the response inside early_stop, result, response, output, or any other wrapper key.\n"
        "- If continue: provide continue_reason and next_visual_target.\n"
        "- If early_stop: provide sufficient_evidence and set next_visual_target to null.\n"
        "- Do not output pixel coordinates.\n"
        "- If you change branch direction, fill branch_correction_reason.\n"
        "- preferred_magnification must be 2.5, 5.0, or 10.0.\n\n"
        "Current context JSON:\n"
        + json.dumps(_model_to_dict(request), ensure_ascii=False, indent=2)
    )
    return system_prompt, user_prompt


def _build_output_repair_prompt(payload: Dict[str, Any]):
    system_prompt = (
        "You are a JSON contract repair adapter for a colorectal pathology agent. "
        "Your job is to repair structure, normalize field names, and fill only clearly missing required containers. "
        "You may perform light clinical fill only when the original text or supplied records explicitly mention the finding. "
        "Do not invent pathology evidence. Do not change final diagnosis or stage decisions unless the value is only a malformed synonym. "
        "Return exactly one JSON object with root keys: repaired_json, repair_actions, unrecoverable_errors, confidence."
    )
    user_prompt = (
        "Repair the following model output so it can be parsed by the requested stage contract.\n"
        "Rules:\n"
        "- repaired_json must contain only the repaired stage payload, not wrapper prose.\n"
        "- If evidence is absent, fill checklist containers with empty objects or not_assessed values, never supporting findings.\n"
        "- Navigation magnification must be 5.0 or 10.0; convert 20.0 to 10.0.\n"
        "- Preserve cell_id, patch_id, workflow_branch, action, coordinates, and original diagnostic intent when present.\n"
        "- If the output cannot be repaired without inventing evidence, put reasons in unrecoverable_errors.\n\n"
        "Repair request JSON:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return system_prompt, user_prompt


def _normalize_repair_response_payload(parsed):
    if not isinstance(parsed, dict):
        parsed = {}
    repaired = parsed.get("repaired_json")
    if repaired is None:
        for key in ("output", "result", "response", "json"):
            if isinstance(parsed.get(key), dict):
                repaired = parsed.get(key)
                break
    if repaired is None and any(key in parsed for key in ("steps", "hierarchical_prediction", "clusters", "patches", "observation")):
        repaired = dict(parsed)
        for key in ("repair_actions", "unrecoverable_errors", "confidence"):
            repaired.pop(key, None)
    if not isinstance(repaired, dict):
        repaired = {}
    actions = parsed.get("repair_actions", [])
    if isinstance(actions, str):
        actions = [actions]
    if not isinstance(actions, list):
        actions = []
    errors = parsed.get("unrecoverable_errors", [])
    if isinstance(errors, str):
        errors = [errors]
    if not isinstance(errors, list):
        errors = []
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    return {
        "repaired_json": repaired,
        "repair_actions": [str(item) for item in actions if str(item).strip()],
        "unrecoverable_errors": [str(item) for item in errors if str(item).strip()],
        "confidence": max(0.0, min(1.0, confidence)),
    }


def _normalize_generation_text(text: str):
    text = str(text or "")
    return text.replace("Ċ", "\n").replace("Ġ", " ").strip()


def _split_think_and_answer(text: str):
    normalized = _normalize_generation_text(text)
    think_parts = [item.strip() for item in re.findall(r"<think>(.*?)</think>", normalized, flags=re.S) if item.strip()]
    thought_text = "\n\n".join(think_parts).strip()
    answer_text = re.sub(r"<think>.*?</think>", "", normalized, flags=re.S).strip()
    return normalized, thought_text, answer_text


def _strip_markdown_fences(text: str):
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.lower().startswith("json\n"):
        stripped = stripped[5:].strip()
    return stripped


def _select_answer_candidate(text: str):
    stripped = str(text or "").strip()
    for marker in ("<｜Assistant｜>", "<|assistant|>", "Assistant:"):
        if marker in stripped:
            stripped = stripped.rsplit(marker, 1)[-1].strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.S | re.I)
    if fenced:
        stripped = fenced[-1].strip()
    return _strip_markdown_fences(stripped)


def _fallback_json(request: ChiefReviewRequest):
    record = request.record or {}
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    stage_decision = str(record.get("stage_decision") or "").strip()
    cluster_id = str(metadata.get("cluster_id") or "")
    workflow_branch = str(metadata.get("workflow_branch") or "non_serrated")
    cluster_label = str(metadata.get("cluster_label") or metadata.get("need_to_see") or "normal_mucosa")
    remaining = list(request.pending_steps or [])
    resolved_branch_state = {
        "serrated": "unresolved",
        "abnormal_crypt": "unresolved",
        "conventional": "unresolved",
        "dysplasia": "unresolved",
    }
    if stage_decision == "supports_serrated_lesion":
        resolved_branch_state["serrated"] = "supported"
        resolved_branch_state["conventional"] = "opposed"
    elif stage_decision == "supports_abnormal_crypt":
        resolved_branch_state["serrated"] = "supported"
        resolved_branch_state["abnormal_crypt"] = "supported"
    elif stage_decision == "supports_conventional_adenoma":
        resolved_branch_state["conventional"] = "supported"
        resolved_branch_state["serrated"] = "opposed"
    elif stage_decision in {"serrated_dysplasia_supported", "conventional_dysplasia_supported"}:
        resolved_branch_state["dysplasia"] = "supported"
    elif stage_decision in {"supports_non_serrated_overview", "background_or_low_value"}:
        resolved_branch_state["serrated"] = "opposed"
        resolved_branch_state["conventional"] = "opposed"
    if remaining:
        next_step = remaining[0]
        next_meta = next_step.get("metadata", {}) if isinstance(next_step.get("metadata"), dict) else {}
        preferred_magnification = float(next_step.get("m") or 5.0)
        if preferred_magnification not in {2.5, 5.0, 10.0}:
            preferred_magnification = 10.0 if stage_decision in {"supports_abnormal_crypt", "ssl_architecture_supported", "tsa_architecture_supported", "supports_conventional_architecture", "supports_conventional_adenoma"} else 5.0
        next_visual_target = {
            "target_cluster_id": str(next_meta.get("cluster_id") or cluster_id),
            "target_branch": str(next_meta.get("workflow_branch") or workflow_branch),
            "target_region_semantic": str(next_meta.get("cluster_label") or cluster_label),
            "target_morphology_prompt": [
                "seek corroborating colorectal pathology morphology",
                "resolve the highest-value remaining branch uncertainty",
            ],
            "preferred_magnification": preferred_magnification,
            "priority_reason": "Pending observation targets remain, so Chief keeps the scan active.",
        }
        return {
            "review_id": "global_review_fallback",
            "source_step_id": str(record.get("step_id") or ""),
            "decision": "continue",
            "continue_reason": "Fallback Chief decision: unresolved observation targets remain in the queue.",
            "chief_confidence": 0.51,
            "resolved_branch_state": resolved_branch_state,
            "sufficient_evidence": [],
            "unresolved_questions": ["Need to inspect the highest-priority remaining target before finalizing."],
            "next_visual_target": next_visual_target,
            "branch_correction_reason": "",
        }
    return {
        "review_id": "global_review_fallback",
        "source_step_id": str(record.get("step_id") or ""),
        "decision": "early_stop",
        "continue_reason": "",
        "chief_confidence": 0.55,
        "resolved_branch_state": resolved_branch_state,
        "sufficient_evidence": [
            "Fallback Chief decision: no pending observation targets remain.",
            "The current observation memory is sufficient for screening-level synthesis.",
        ],
        "unresolved_questions": [],
        "next_visual_target": None,
        "branch_correction_reason": "",
    }


def _generate_json(tokenizer, model, system_prompt: str, user_prompt: str, max_new_tokens: int, max_model_len: int):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = system_prompt + "\n\n" + user_prompt
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_model_len),
    )
    prompt_token_count = int(encoded["input_ids"].shape[1])
    prompt_truncated = bool(prompt_token_count >= int(max_model_len))
    inputs = encoded.to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    worker = Thread(target=model.generate, kwargs=generation_kwargs)
    worker.start()
    chunks = []
    for chunk in streamer:
        chunks.append(chunk)
    worker.join()
    raw_text = "".join(chunks)
    normalized_text, thought_text, answer_text = _split_think_and_answer(raw_text)
    candidate_text = _select_answer_candidate(answer_text)
    blob = _extract_first_json_object(candidate_text) or _extract_first_json_object(normalized_text)
    if not blob:
        raise ChiefGenerationError(
            "Chief model response did not contain a JSON object",
            raw_text=normalized_text,
            think_text=thought_text,
            answer_text=candidate_text,
        )
    try:
        return (
            json.loads(blob),
            normalized_text,
            thought_text,
            candidate_text,
            prompt_token_count,
            prompt_truncated,
        )
    except Exception as exc:
        raise ChiefGenerationError(
            "Chief model response contained invalid JSON: {0}".format(str(exc)),
            raw_text=normalized_text,
            think_text=thought_text,
            answer_text=candidate_text,
        )


def create_app(args):
    before = _gpu_snapshot()
    _log_json("startup_before_load", before)

    model_path = args.model_path
    loaded_model_name = model_path
    try:
        tokenizer, model = _load_model_bundle(model_path)
    except Exception as primary_exc:
        fallback = str(args.fallback_model_path or "").strip()
        if not fallback:
            raise
        _log_json("startup_primary_load_failed", {"model_path": model_path, "error": str(primary_exc), "fallback_model_path": fallback})
        tokenizer, model = _load_model_bundle(fallback)
        loaded_model_name = fallback

    after = _gpu_snapshot()
    _log_json(
        "startup_after_load",
        {
            **after,
            "model_name": loaded_model_name,
            "torch_cuda_is_available": bool(torch.cuda.is_available()),
        },
    )

    app = FastAPI(title="chief_pathologist_server")

    @app.middleware("http")
    async def log_unhandled_500(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            tb = traceback.format_exc()
            payload = {
                "path": str(request.url.path),
                "method": str(request.method),
                "error": str(exc),
                "traceback": tb,
            }
            _log_json("chief_unhandled_exception", payload)
            _append_error_log(args.error_log_path, "chief_unhandled_exception", payload)
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "error": str(exc)})

    @app.post("/predict", response_model=ChiefReviewResponse)
    def predict(request: ChiefReviewRequest):
        request_id = str(uuid.uuid4())
        system_prompt, user_prompt = _build_prompt(request)
        started = time.time()
        try:
            parsed, raw_generated_text, thought_text, answer_candidate_text, prompt_token_count, prompt_truncated = _generate_json(
                tokenizer,
                model,
                system_prompt,
                user_prompt,
                int(args.max_new_tokens),
                int(args.max_model_len),
            )
            parsed = _normalize_chief_response_payload(parsed, request)
            parsed["review_source"] = "chief_model"
            parsed["parse_error"] = ""
            parsed["thought_text"] = thought_text
            parsed["answer_candidate_text"] = answer_candidate_text
            parsed["prompt_token_count"] = prompt_token_count
            parsed["prompt_truncated"] = prompt_truncated
        except Exception as exc:
            raw_generated_text = getattr(exc, "raw_text", "")
            thought_text = getattr(exc, "think_text", "")
            answer_candidate_text = getattr(exc, "answer_text", "")
            _log_json(
                "chief_predict_error",
                {
                    "request_id": request_id,
                    "error": str(exc),
                    "fallback_mode": "rule_based",
                    "raw_generated_text": raw_generated_text,
                    "thought_text": thought_text,
                    "answer_candidate_text": answer_candidate_text,
                },
            )
            parsed = _fallback_json(request)
            parsed = _normalize_chief_response_payload(parsed, request)
            parsed["review_source"] = "fallback_rule_based"
            parsed["parse_error"] = str(exc)
            parsed["thought_text"] = thought_text
            parsed["answer_candidate_text"] = answer_candidate_text
            parsed["prompt_token_count"] = 0
            parsed["prompt_truncated"] = False
        elapsed_ms = int(round((time.time() - started) * 1000.0))
        gpu = _gpu_snapshot()
        parsed["request_id"] = request_id
        parsed["model_name"] = loaded_model_name
        parsed["gpu_device_id"] = gpu.get("device_id")
        parsed["cuda_visible_devices"] = gpu.get("cuda_visible_devices")
        parsed["round_trip_ms"] = elapsed_ms
        parsed["raw_generated_text"] = raw_generated_text
        _log_json(
            "chief_predict_ok",
            {
                "request_id": request_id,
                "model_name": loaded_model_name,
                "round_trip_ms": elapsed_ms,
                "gpu_device_id": parsed["gpu_device_id"],
                "cuda_visible_devices": parsed["cuda_visible_devices"],
                "review_source": parsed.get("review_source"),
            },
        )
        try:
            return ChiefReviewResponse(**parsed)
        except Exception as exc:
            tb = traceback.format_exc()
            _log_json(
                "chief_response_validation_error",
                {
                    "request_id": request_id,
                    "error": str(exc),
                    "traceback": tb,
                    "parsed": parsed,
                },
            )
            _append_error_log(
                args.error_log_path,
                "chief_response_validation_error",
                {
                    "request_id": request_id,
                    "error": str(exc),
                    "traceback": tb,
                    "parsed": parsed,
                },
            )
            raise

    @app.post("/chief_review", response_model=ChiefReviewResponse)
    def chief_review(request: ChiefReviewRequest):
        return predict(request)

    @app.post("/repair_output", response_model=OutputRepairResponse)
    async def repair_output(request: Request):
        request_id = str(uuid.uuid4())
        payload = await request.json()
        system_prompt, user_prompt = _build_output_repair_prompt(payload if isinstance(payload, dict) else {"raw_payload": payload})
        started = time.time()
        parse_error = ""
        try:
            parsed, raw_generated_text, thought_text, answer_candidate_text, prompt_token_count, prompt_truncated = _generate_json(
                tokenizer,
                model,
                system_prompt,
                user_prompt,
                int(args.max_new_tokens),
                int(args.max_model_len),
            )
            normalized = _normalize_repair_response_payload(parsed)
            repair_source = "repair_model"
        except Exception as exc:
            raw_generated_text = getattr(exc, "raw_text", "")
            thought_text = getattr(exc, "think_text", "")
            answer_candidate_text = getattr(exc, "answer_text", "")
            prompt_token_count = 0
            prompt_truncated = False
            parse_error = str(exc)
            normalized = {
                "repaired_json": {},
                "repair_actions": [],
                "unrecoverable_errors": [str(exc)],
                "confidence": 0.0,
            }
            repair_source = "repair_error"
        elapsed_ms = int(round((time.time() - started) * 1000.0))
        gpu = _gpu_snapshot()
        normalized.update(
            {
                "request_id": request_id,
                "model_name": loaded_model_name,
                "repair_source": repair_source,
                "raw_generated_text": raw_generated_text,
                "thought_text": thought_text,
                "answer_candidate_text": answer_candidate_text,
                "parse_error": parse_error,
                "prompt_token_count": prompt_token_count,
                "prompt_truncated": prompt_truncated,
                "gpu_device_id": gpu.get("device_id"),
                "cuda_visible_devices": gpu.get("cuda_visible_devices"),
                "round_trip_ms": elapsed_ms,
            }
        )
        _log_json(
            "repair_predict_ok",
            {
                "request_id": request_id,
                "model_name": loaded_model_name,
                "round_trip_ms": elapsed_ms,
                "repair_source": repair_source,
                "stage": (payload or {}).get("stage") if isinstance(payload, dict) else None,
            },
        )
        return OutputRepairResponse(**normalized)

    return app


def main():
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
