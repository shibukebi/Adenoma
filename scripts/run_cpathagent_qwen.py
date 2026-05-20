#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adenoma_agent.multimodal import (  # noqa: E402
    HeuristicStageBackend,
    _build_text_driven_output,
    _build_trace_output_from_text,
    _run_trace_grid_with_coverage_retry,
)
from adenoma_agent.utils import read_json, write_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Real ms-swift-based local runner for the CPathAgent-style Qwen backend.")
    parser.add_argument("--stage", required=True, choices=["trace", "navigate", "observe_step", "observe_report"])
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--response-json", required=True)
    parser.add_argument("--bundle-json", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--backbone-model-id", default="")
    parser.add_argument("--vision-encoder-id", default="")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--projector-type", default="two_layer_mlp")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def _extract_first_json_object(text: str):
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


def _first_json_dict(text: str):
    blob = _extract_first_json_object(text)
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None


def _build_multimodal_messages(images: List[str], prompt_text: str):
    content = []
    for image_path in images:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def _build_navigate_prompt(request, bundle):
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


def _parse_navigate_output(text, request, bundle):
    parsed = _first_json_dict(text) or {}
    steps = parsed.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    if not steps:
        return HeuristicStageBackend().invoke({**request, "stage": "navigate"}, bundle)["output"]

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
                    x = int(round((int(patch["x1"]) + int(patch["x2"])) / 2.0))
                    y = int(round((int(patch["y1"]) + int(patch["y2"])) / 2.0))
                    break
        if x is None or y is None:
            bbox = cluster.get("group_bbox_level0") or cluster.get("cluster_bbox_level0") or {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
            x = int(round((int(bbox["x1"]) + int(bbox["x2"])) / 2.0))
            y = int(round((int(bbox["y1"]) + int(bbox["y2"])) / 2.0))
        magnification = float(step.get("m", 5.0))
        normalized_steps.append(
            {
                "step_id": "step_{0:02d}".format(index),
                "x": int(x),
                "y": int(y),
                "m": magnification,
                "region_size_level0": int(mag_to_region.get(str(magnification), 256)),
                "need_to_see": step.get("need_to_see", step.get("o", "Inspect the planned pathology region.")),
                "review_goal": step.get("review_goal", "serrated_lesion_assessment"),
                "stage_gate": step.get("stage_gate", "mucosa_or_serrated"),
                "metadata": {
                    "cluster_id": source_group_id,
                    "source_group_id": source_group_id,
                    "cluster_label": cluster.get("l"),
                    "cluster_priority": cluster.get("s"),
                    "patch_id": list(patch_id),
                    "region_size_level0": int(mag_to_region.get(str(magnification), 256)),
                    "workflow_branch": cluster.get("metadata", {}).get("workflow_branch"),
                    "action": "inspect",
                },
            }
        )
    if normalized_steps:
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


def _build_observe_report_prompt(request, bundle):
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


def _parse_observe_report_output(text, request, bundle):
    parsed = _first_json_dict(text)
    if isinstance(parsed, dict) and "hierarchical_prediction" in parsed and "integrated_report" in parsed:
        return parsed
    fallback = HeuristicStageBackend().invoke({**request, "stage": "observe_report"}, bundle)["output"]
    fallback["integrated_report"] = text.strip() or fallback["integrated_report"]
    return fallback


def _build_stage_messages(stage, request, bundle):
    if stage == "trace":
        prompt_text = request["prompt"]["question"]
    elif stage == "observe_step":
        prompt_text = request["prompt"]["question"]
    elif stage == "navigate":
        prompt_text = _build_navigate_prompt(request, bundle)
    elif stage == "observe_report":
        prompt_text = _build_observe_report_prompt(request, bundle)
    else:
        raise ValueError("Unsupported stage: {0}".format(stage))
    return _build_multimodal_messages(request.get("images", []), prompt_text), prompt_text


def _real_stage_output(stage, generated_text, request, bundle):
    if stage == "trace":
        return _build_trace_output_from_text(generated_text, request, bundle)
    if stage == "navigate":
        return _parse_navigate_output(generated_text, request, bundle)
    if stage == "observe_step":
        return _build_text_driven_output(generated_text, request, bundle)
    if stage == "observe_report":
        return _parse_observe_report_output(generated_text, request, bundle)
    raise ValueError("Unsupported stage: {0}".format(stage))


def _should_use_heuristic(shim_mode):
    return shim_mode == "heuristic"


def _validate_real_model_args(args, request):
    if not args.model_id:
        raise RuntimeError("local_cpathagent_qwen requires --model-id for real inference")


def main():
    args = parse_args()
    request = read_json(args.request_json)
    bundle = read_json(args.bundle_json)
    request = {**request, "stage": args.stage}

    shim_mode = (
        bundle.get("runtime", {})
        .get("backends", {})
        .get("local_cpathagent_qwen", {})
        .get("shim_mode", "swift")
    )

    if _should_use_heuristic(shim_mode):
        response = HeuristicStageBackend().invoke(request, bundle)
        payload = {
            **response["output"],
            "runner_metadata": {
                "backend_name": "local_cpathagent_qwen",
                "shim_mode": shim_mode,
                "model_id": args.model_id,
                "vision_encoder_id": args.vision_encoder_id,
                "adapter_path": args.adapter_path,
                "cache_dir": args.cache_dir,
                "max_new_tokens": int(args.max_new_tokens),
            },
        }
        write_json(args.response_json, payload)
        print(json.dumps({"backend": "local_cpathagent_qwen", "shim_mode": shim_mode, "stage": args.stage, "model_id": args.model_id}))
        return

    _validate_real_model_args(args, request)
    from adenoma_agent.qwen_inference import QwenInference  # noqa: E402

    qwen_engine = QwenInference(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        device_map="auto",
        max_batch_size=1,
        use_hf=True,
        download_model=True,
    )
    messages, prompt_text = _build_stage_messages(args.stage, request, bundle)
    if args.stage == "trace" and request.get("metadata", {}).get("thumbnail_meta", {}).get("grid_metadata_path"):
        def _generate_trace_text(prompt_override):
            override_messages = _build_multimodal_messages(request.get("images", []), prompt_override)
            return qwen_engine.generate(override_messages, max_new_tokens=int(args.max_new_tokens), temperature=0.0)

        trace_response = _run_trace_grid_with_coverage_retry(_generate_trace_text, request, bundle)
        payload = trace_response["output"]
        generated_text = trace_response["raw_text"]
        payload["trace_attempts"] = trace_response["trace_attempts"]
        payload["raw_text"] = trace_response["raw_text"]
        payload["raw_texts"] = trace_response["raw_texts"]
    else:
        generated_text = qwen_engine.generate(messages, max_new_tokens=int(args.max_new_tokens), temperature=0.0)
        payload = _real_stage_output(args.stage, generated_text, request, bundle)
    payload["runner_metadata"] = {
        "backend_name": "local_cpathagent_qwen",
        "shim_mode": shim_mode,
        "model_id": args.model_id,
        "backbone_model_id": args.backbone_model_id or args.model_id,
        "vision_encoder_id": args.vision_encoder_id,
        "adapter_path": args.adapter_path,
        "projector_type": args.projector_type,
        "cache_dir": args.cache_dir,
        "max_new_tokens": int(args.max_new_tokens),
        "generated_text": generated_text,
        "prompt_text": prompt_text,
    }
    write_json(args.response_json, payload)
    print(json.dumps({"backend": "local_cpathagent_qwen", "shim_mode": shim_mode, "stage": args.stage, "model_id": args.model_id}))


if __name__ == "__main__":
    main()
