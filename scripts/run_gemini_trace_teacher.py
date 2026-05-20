#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adenoma_agent.trace_supervision import (  # noqa: E402
    build_image_only_teacher_prompt,
    export_candidate_review_package,
    parse_patch_assignment_response,
    score_trace_auto_review,
    score_trace_case,
    select_best_candidate,
)
from adenoma_agent.utils import ensure_dir, read_json, write_json  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Run Gemini image-only trace teacher generation.")
    parser.add_argument("--teacher-request-json", action="append", default=[], help="One teacher_request.json path. Repeatable.")
    parser.add_argument("--teacher-request-root", default=None, help="Root containing per-case teacher_request.json files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write prompts and package skeletons without calling Gemini.")
    return parser


def _collect_request_paths(args):
    paths = [Path(item) for item in args.teacher_request_json]
    if args.teacher_request_root:
        root = Path(args.teacher_request_root)
        index_path = root / "teacher_request_index.json"
        if index_path.exists():
            index = read_json(index_path)
            paths.extend(Path(row["teacher_request_json"]) for row in index.get("rows", []))
        else:
            paths.extend(sorted(root.glob("**/teacher_request.json")))
    deduped = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def _load_pil_image(path):
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to send grid thumbnails to Gemini. Install pillow in this environment.") from exc
    return Image.open(path)


def _call_gemini(prompt_text, image_path, model, api_key, temperature, max_output_tokens):
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError(
            "google-genai is required for Gemini calls. Install it and set GEMINI_API_KEY before running without --dry-run."
        ) from exc
    client = genai.Client(api_key=api_key)
    image = _load_pil_image(image_path)
    response = client.models.generate_content(
        model=model,
        contents=[prompt_text, image],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        ),
    )
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        parts = getattr(getattr(candidates[0], "content", None), "parts", []) or []
        return "\n".join(str(getattr(part, "text", "") or "") for part in parts)
    return str(response)


def _candidate_result(candidate_index, raw_text, grid_meta):
    parsed = parse_patch_assignment_response(raw_text)
    payload = parsed["payload"]
    score = score_trace_case(payload, grid_meta)
    return {
        "candidate_index": candidate_index,
        "payload": payload,
        "parse_failure": parsed["parse_failure"],
        "parse_error": parsed.get("parse_error"),
        "score": score,
    }


def _dry_run_payload(request):
    patches = []
    for patch_id in request.get("selected_patch_ids", []):
        patches.append(
            {
                "patch_id": patch_id,
                "region_semantic": "background_artifact_stroma",
                "name": "Dry-run placeholder",
                "description": "Placeholder assignment for dry-run packaging only.",
                "require_high_magnification": False,
                "severity_reasoning": "No Gemini call was made.",
                "diagnostic_priority": 0,
                "observation_points": ["dry-run placeholder"],
            }
        )
    return {"patches": patches}


def _process_request(path, args):
    request = read_json(path)
    case_id = request.get("case_id", path.parent.name)
    case_output_dir = ensure_dir(Path(args.output_dir) / case_id)
    raw_dir = ensure_dir(case_output_dir / "gemini_image_only_raw")
    checked_dir = ensure_dir(case_output_dir / "auto_checked")
    review_dir = ensure_dir(case_output_dir / "review")
    prompt_text = request.get("prompt_text") or build_image_only_teacher_prompt(request)
    write_json(case_output_dir / "teacher_request.json", request)
    (case_output_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")

    generation = request.get("generation_config", {})
    candidate_count = args.candidate_count or int(request.get("candidate_count", 2))
    temperature = args.temperature if args.temperature is not None else float(generation.get("temperature", 0.8))
    max_output_tokens = args.max_output_tokens or int(generation.get("max_output_tokens", 3000))
    model = args.model or request.get("teacher_model", "gemini-2.5-pro")
    image_path = request["grid_thumbnail_path"]
    grid_meta = read_json(request["grid_metadata_path"])
    slide_label_context = request.get("slide_label_context")
    api_key = os.environ.get(args.api_key_env, "")

    candidate_results = []
    for candidate_index in range(candidate_count):
        if args.dry_run:
            raw_text = json.dumps(_dry_run_payload(request), ensure_ascii=False)
        else:
            if not api_key:
                raise RuntimeError("{0} is not set. Provide the Gemini key or use --dry-run.".format(args.api_key_env))
            raw_text = _call_gemini(prompt_text, image_path, model, api_key, temperature, max_output_tokens)
        raw_path = raw_dir / "candidate_{0:02d}.txt".format(candidate_index)
        raw_path.write_text(raw_text, encoding="utf-8")
        result = _candidate_result(candidate_index, raw_text, grid_meta)
        result["raw_response_path"] = str(raw_path)
        result["generation_config"] = {
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "teacher_mode": "image_only_screening_trace",
            "report_available": False,
        }
        write_json(raw_dir / "candidate_{0:02d}.json".format(candidate_index), result["payload"])
        write_json(raw_dir / "candidate_{0:02d}_score.json".format(candidate_index), result["score"])
        candidate_results.append(result)

    auto_review = score_trace_auto_review(candidate_results, grid_meta, slide_label_context=slide_label_context)
    selection = auto_review["selection"]
    selected_index = selection.get("selected_index")
    selected_payload = {"patches": []}
    if selected_index is not None:
        selected_payload = auto_review["candidates"][int(selected_index)]["payload"]
    write_json(checked_dir / "gemini_candidates.json", {"candidates": auto_review["candidates"], "selection": selection, "auto_review": auto_review})
    write_json(checked_dir / "selected_candidate.json", selected_payload)
    write_json(checked_dir / "candidate_agreement.json", auto_review.get("candidate_agreement", {}))
    write_json(checked_dir / "auto_review.json", auto_review)
    export_candidate_review_package(
        case_id,
        request["grid_thumbnail_path"],
        request["grid_metadata_path"],
        candidate_results,
        review_dir,
        selection=selection,
        auto_review=auto_review,
    )
    return {
        "case_id": case_id,
        "output_dir": str(case_output_dir),
        "review_target": str(review_dir / "review_target.json"),
        "selected_candidate_index": selected_index,
        "review_status": selection.get("review_status"),
        "dry_run": bool(args.dry_run),
    }


def main():
    args = build_parser().parse_args()
    request_paths = _collect_request_paths(args)
    rows = [_process_request(path, args) for path in request_paths]
    summary = {"count": len(rows), "rows": rows}
    ensure_dir(args.output_dir)
    write_json(Path(args.output_dir) / "gemini_teacher_run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
