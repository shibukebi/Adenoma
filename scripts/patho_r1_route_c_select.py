#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openslide
from PIL import Image, ImageDraw


SYSTEM_PROMPT_TEMPLATE = """You are a pathology region proposal model.
Find up to {max_boxes} candidate thumbnail regions that may contain diagnostically relevant colorectal polyp tissue with high recall, especially SSL-related serrated or adenomatous epithelium.
Output ONLY one of the following:
1. Up to {max_boxes} lines in the exact format:
BOX x1 y1 x2 y2 score
2. The exact token NO_BOX if no suspicious region is visible.
Rules:
- Use integer thumbnail pixel coordinates.
- 0 <= x1 < x2 <= {width}
- 0 <= y1 < y2 <= {height}
- score must be a float between 0 and 1
- Do not output JSON.
- Do not explain your reasoning.
- Do not output any extra text."""

USER_PROMPT_TEMPLATE = """The image is a pathology thumbnail of size {width}x{height} pixels.
Identify up to {max_boxes} rectangular candidate regions with high recall.
Prefer including suspicious tissue rather than missing possible SSL-related regions.
Return only BOX lines or NO_BOX."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route C region selection on WSI thumbnails using Patho-R1 or fallback heuristics."
    )
    parser.add_argument("--slide-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["patho-r1", "heuristic", "manual"], default="heuristic")
    parser.add_argument("--model-id", default="WenchuanZhang/Patho-R1-3B")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--thumbnail-max-size", type=int, default=1024)
    parser.add_argument("--manual-boxes", default=None, help="Format: x1,y1,x2,y2;x1,y1,x2,y2")
    parser.add_argument("--fallback-mode", choices=["none", "heuristic"], default="none")
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    return parser.parse_args()


def generate_thumbnail(slide_path: Path, out_dir: Path, max_size: int) -> Tuple[Path, Dict[str, Any]]:
    slide = openslide.open_slide(str(slide_path))
    thumb = slide.get_thumbnail((max_size, max_size)).convert("RGB")
    thumb_path = out_dir / f"{slide_path.stem}_thumbnail.jpg"
    thumb.save(thumb_path, "JPEG", quality=90)
    meta = {
        "slide_id": slide_path.stem,
        "slide_path": str(slide_path),
        "slide_dimensions_level0": list(slide.dimensions),
        "thumbnail_size": list(thumb.size),
        "level_count": slide.level_count,
    }
    slide.close()
    return thumb_path, meta


def heuristic_boxes(image: Image.Image, max_boxes: int) -> List[Dict[str, Any]]:
    arr = np.array(image)
    sat = arr.max(axis=2) - arr.min(axis=2)
    mean_rgb = arr.mean(axis=2)
    mask = (mean_rgb < 235) & (sat > 8)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        w, h = image.size
        return [{
            "x1": w // 4,
            "y1": h // 4,
            "x2": 3 * w // 4,
            "y2": 3 * h // 4,
            "label": "fallback_center",
            "score": 0.1,
        }]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    margin = 10
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.size[0] - 1, x2 + margin)
    y2 = min(image.size[1] - 1, y2 + margin)
    return [{
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "label": "heuristic_tissue",
        "score": 1.0,
    }][:max_boxes]


def manual_boxes(spec: str) -> List[Dict[str, Any]]:
    boxes = []
    for idx, chunk in enumerate(spec.split(";")):
        chunk = chunk.strip()
        if not chunk:
            continue
        x1, y1, x2, y2 = [int(v) for v in chunk.split(",")]
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": f"manual_{idx}", "score": 1.0})
    if not boxes:
        raise ValueError("No valid manual boxes provided.")
    return boxes


def repo_id_to_cache_dir(repo_id: str) -> Path:
    model_dir = repo_id.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_dir}"


def resolve_model_source(model_id: str, local_files_only: bool) -> str:
    candidate = Path(model_id)
    if candidate.exists():
        return str(candidate)
    if local_files_only and "/" in model_id:
        cache_root = repo_id_to_cache_dir(model_id) / "snapshots"
        if cache_root.exists():
            snapshots = sorted([path for path in cache_root.iterdir() if path.is_dir()])
            if snapshots:
                return str(snapshots[-1])
    return model_id


def load_patho_r1_stack(model_id: str, cache_dir: Optional[str] = None, local_files_only: bool = False):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    resolved = resolve_model_source(model_id, local_files_only=local_files_only)
    kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    processor_kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
        processor_kwargs["cache_dir"] = cache_dir
    if local_files_only:
        kwargs["local_files_only"] = True
        processor_kwargs["local_files_only"] = True

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(resolved, **kwargs)
    processor = AutoProcessor.from_pretrained(resolved, **processor_kwargs)
    return model, processor, resolved


def build_messages(thumb_path: Path, width: int, height: int, max_boxes: int) -> List[Dict[str, Any]]:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(max_boxes=max_boxes, width=width - 1, height=height - 1)
    user_prompt = USER_PROMPT_TEMPLATE.format(width=width, height=height, max_boxes=max_boxes)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(thumb_path)},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def normalize_text(text: Any) -> str:
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        return "\n".join(normalize_text(item) for item in text)
    if isinstance(text, dict):
        if "text" in text:
            return normalize_text(text["text"])
        if "generated_text" in text:
            return normalize_text(text["generated_text"])
        return json.dumps(text, ensure_ascii=False)
    return str(text)


def run_patho_r1_generation(
    thumb_path: Path,
    model,
    processor,
    width: int,
    height: int,
    max_boxes: int,
    max_new_tokens: int,
) -> str:
    from qwen_vl_utils import process_vision_info

    messages = build_messages(thumb_path, width, height, max_boxes)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return normalize_text(output_text[0]) if output_text else ""


def clamp_box(box: Dict[str, Any], width: int, height: int) -> Optional[Dict[str, Any]]:
    x1, y1, x2, y2 = int(round(box["x1"])), int(round(box["y1"])), int(round(box["x2"])), int(round(box["y2"]))
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    score = float(box.get("score", 1.0))
    score = max(0.0, min(score, 1.0))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": box.get("label", "adenoma_region"), "score": score}


def parse_boxes_from_text(text: str, width: int, height: int, max_boxes: int) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    if not text:
        return boxes
    if "NO_BOX" in text:
        return boxes

    box_pattern = re.compile(
        r"BOX\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+([01](?:\.\d+)?|\d\.\d+|\d+)",
        flags=re.I,
    )
    for match in box_pattern.finditer(text):
        box = {
            "x1": float(match.group(1)),
            "y1": float(match.group(2)),
            "x2": float(match.group(3)),
            "y2": float(match.group(4)),
            "score": float(match.group(5)),
            "label": "adenoma_region",
        }
        clamped = clamp_box(box, width, height)
        if clamped is not None:
            boxes.append(clamped)
        if len(boxes) >= max_boxes:
            return boxes

    if boxes:
        return boxes[:max_boxes]

    # Best-effort JSON fallback for malformed but recoverable responses.
    raw = re.search(r"(\{.*)", text, flags=re.S)
    if raw:
        candidate = raw.group(1)
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(candidate)
            raw_boxes = obj.get("boxes") or obj.get("boxed") or []
            for raw_box in raw_boxes:
                clamped = clamp_box(raw_box, width, height)
                if clamped is not None:
                    boxes.append(clamped)
                if len(boxes) >= max_boxes:
                    break
        except Exception:
            pass

    return boxes[:max_boxes]


def draw_boxes(image: Image.Image, boxes: List[Dict[str, Any]], out_path: Path) -> None:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    for box in boxes:
        draw.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]], outline=(255, 0, 0), width=4)
    vis.save(out_path)


def process_slide(
    slide_path: str,
    output_dir: str,
    mode: str = "patho-r1",
    fallback_mode: str = "none",
    model=None,
    processor=None,
    model_id: str = "WenchuanZhang/Patho-R1-3B",
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    thumbnail_max_size: int = 1024,
    manual_box_spec: Optional[str] = None,
    max_boxes: int = 3,
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    slide_path_obj = Path(slide_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thumb_path, meta = generate_thumbnail(slide_path_obj, out_dir, thumbnail_max_size)
    thumb = Image.open(thumb_path).convert("RGB")
    width, height = thumb.size

    raw_response = ""
    failure_reason = ""
    mode_used = mode
    boxes: List[Dict[str, Any]] = []
    resolved_model_source = None

    try:
        if mode == "heuristic":
            boxes = heuristic_boxes(thumb, max_boxes=max_boxes)
            raw_response = json.dumps({"boxes": boxes}, ensure_ascii=False, indent=2)
            status = "success_fallback"
            mode_used = "heuristic"
        elif mode == "manual":
            boxes = manual_boxes(manual_box_spec or "")
            raw_response = json.dumps({"boxes": boxes}, ensure_ascii=False, indent=2)
            status = "success_manual"
            mode_used = "manual"
        else:
            if model is None or processor is None:
                model, processor, resolved_model_source = load_patho_r1_stack(
                    model_id=model_id, cache_dir=cache_dir, local_files_only=local_files_only
                )
            else:
                resolved_model_source = resolve_model_source(model_id, local_files_only=local_files_only)

            print(f"start_generation slide_id={slide_path_obj.stem} max_new_tokens={max_new_tokens}")
            start = time.time()
            raw_response = run_patho_r1_generation(
                thumb_path=thumb_path,
                model=model,
                processor=processor,
                width=width,
                height=height,
                max_boxes=max_boxes,
                max_new_tokens=max_new_tokens,
            )
            elapsed = time.time() - start
            print(f"finish_generation slide_id={slide_path_obj.stem} seconds={elapsed:.2f}")
            boxes = parse_boxes_from_text(raw_response, width, height, max_boxes=max_boxes)
            if boxes:
                status = "success_patho_r1"
                mode_used = "patho-r1"
            else:
                raise ValueError("No valid BOX lines parsed from Patho-R1 output.")
    except Exception as exc:
        failure_reason = str(exc)
        if fallback_mode == "heuristic":
            boxes = heuristic_boxes(thumb, max_boxes=max_boxes)
            raw_response = raw_response or failure_reason
            status = "success_fallback"
            mode_used = "heuristic"
        else:
            status = "failed_parse_error" if raw_response else "failed_model_error"
            mode_used = mode

    result = {
        "mode_requested": mode,
        "mode_used": mode_used,
        "status": status,
        "failure_reason": failure_reason,
        "model_id": model_id,
        "resolved_model_source": resolved_model_source,
        "thumbnail_path": str(thumb_path),
        "thumbnail_meta": meta,
        "boxes": boxes,
    }

    raw_path = out_dir / f"{slide_path_obj.stem}_route_c_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    json_path = out_dir / f"{slide_path_obj.stem}_route_c_boxes.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    vis_path = out_dir / f"{slide_path_obj.stem}_route_c_boxes.png"
    draw_boxes(thumb, boxes, vis_path)

    print(f"thumbnail={thumb_path}")
    print(f"raw_response={raw_path}")
    print(f"boxes_json={json_path}")
    print(f"boxes_visualization={vis_path}")
    print(f"status={status}")
    print(f"mode_used={mode_used}")
    print(f"box_count={len(boxes)}")
    if failure_reason:
        print(f"failure_reason={failure_reason}")

    return result


def main() -> None:
    args = parse_args()
    result = process_slide(
        slide_path=args.slide_path,
        output_dir=args.output_dir,
        mode=args.mode,
        fallback_mode=args.fallback_mode,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        thumbnail_max_size=args.thumbnail_max_size,
        manual_box_spec=args.manual_boxes,
        max_boxes=args.max_boxes,
        max_new_tokens=args.max_new_tokens,
    )
    if result["status"].startswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
