#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from classify_wsi_fixed_fov_patches import WSIReader, read_padded_bbox  # noqa: E402
from adenoma_agent.tissue_context import (  # noqa: E402
    normalized_entropy,
    tissue_context_from_probabilities,
    tissue_context_label,
)


FIVE_CLASS_LABELS = [
    "epithelial_neoplasia_suspicious",
    "mucus_rich_or_pale_context",
    "reviewable_normal_mucosa",
    "inflammatory_or_stromal_context",
    "background_or_artifact",
]
FIVE_CLASS_PRIORITY = {
    "background_or_artifact": 0,
    "reviewable_normal_mucosa": 1,
    "inflammatory_or_stromal_context": 2,
    "mucus_rich_or_pale_context": 3,
    "epithelial_neoplasia_suspicious": 4,
}
FIVE_CLASS_COLORS = {
    "epithelial_neoplasia_suspicious": (230, 40, 40),
    "mucus_rich_or_pale_context": (0, 190, 210),
    "reviewable_normal_mucosa": (60, 175, 80),
    "inflammatory_or_stromal_context": (245, 150, 45),
    "background_or_artifact": (130, 130, 130),
}
CRC100K_TO_FIVE_CLASS = {
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
DIGEPATH_TO_FIVE_CLASS = {
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
MODEL_NAMES = ["uni_prismnet", "conch_zeroshot", "digepath"]
SUPPORTED_WSI_SUFFIXES = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run WSI_10sample 10x/20x/40x selected-grid five-class experiment.")
    parser.add_argument("--input-dir", default=str(REPO_ROOT / "data" / "WSI_10sample"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--crop-ids", default="10x_1024,20x_512,40x_256")
    parser.add_argument("--include-all-cells", action="store_true")
    parser.add_argument("--limit-slides", type=int, default=0)
    parser.add_argument("--limit-patches-per-crop-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--progress-every-batches", type=int, default=20)
    parser.add_argument("--uni-url", default="http://127.0.0.1:8400/predict")
    parser.add_argument("--conch-url", default="http://127.0.0.1:8200/predict")
    parser.add_argument("--digepath-url", default="http://127.0.0.1:8300/predict")
    parser.add_argument("--skip-uni", action="store_true")
    parser.add_argument("--skip-conch", action="store_true")
    parser.add_argument("--skip-digepath", action="store_true")
    parser.add_argument("--skip-overlays", action="store_true")
    parser.add_argument("--allow-pil-fallback", action="store_true")
    parser.add_argument("--skip-unreadable-wsi", action="store_true")
    return parser.parse_args()


def safe_name(value):
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unnamed"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_crop_ids(value):
    crop_ids = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not crop_ids:
        raise ValueError("At least one crop_id is required.")
    return crop_ids


def grid_paths(input_dir, limit_slides=0):
    paths = sorted(Path(input_dir).glob("*_grid.json"))
    if limit_slides:
        paths = paths[: int(limit_slides)]
    return paths


def wsi_candidate_paths(input_dir, grid_payload):
    input_dir = Path(input_dir)
    slide_id = str(grid_payload.get("slide_id", "")).strip()
    seen = set()
    candidates = []

    def add_candidate(path):
        path = Path(path)
        key = str(path)
        if key not in seen and path.exists() and path.suffix.lower() in SUPPORTED_WSI_SUFFIXES:
            seen.add(key)
            candidates.append(path)

    for suffix in sorted(SUPPORTED_WSI_SUFFIXES):
        candidate = input_dir / "{0}{1}".format(slide_id, suffix)
        add_candidate(candidate)
    source = Path(str(grid_payload.get("slide_path", "")))
    add_candidate(source)
    matches = sorted(input_dir.glob("{0}.*".format(slide_id)))
    for candidate in matches:
        add_candidate(candidate)
    return candidates


def probe_wsi_readable(path, allow_pil_fallback=False):
    reader = None
    try:
        reader = WSIReader(path, allow_pil_fallback=allow_pil_fallback)
        return {
            "path": str(path),
            "readable": True,
            "reader_mode": getattr(reader, "mode", ""),
            "dimensions": list(getattr(reader, "dimensions", [])),
            "error": "",
        }
    except Exception as exc:
        return {
            "path": str(path),
            "readable": False,
            "reader_mode": "",
            "dimensions": [],
            "error": str(exc),
        }
    finally:
        if reader is not None:
            reader.close()


def resolve_wsi_path(input_dir, grid_payload, allow_pil_fallback=False):
    slide_id = str(grid_payload.get("slide_id", "")).strip()
    probes = []
    for candidate in wsi_candidate_paths(input_dir, grid_payload):
        probe = probe_wsi_readable(candidate, allow_pil_fallback=allow_pil_fallback)
        probes.append(probe)
        if probe["readable"]:
            if len(probes) > 1:
                print(
                    json.dumps(
                        {
                            "event": "wsi_candidate_fallback",
                            "slide_id": slide_id,
                            "selected_wsi_path": str(candidate),
                            "candidate_probes": probes,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            return candidate
    raise RuntimeError(
        "Could not resolve a readable WSI for slide_id={0}; candidate probes={1}".format(
            slide_id, json.dumps(probes, ensure_ascii=False)
        )
    )


def resolve_grid_jpg(grid_path):
    candidate = Path(str(grid_path).replace("_grid.json", "_grid.jpg"))
    return candidate if candidate.exists() else None


def selected_grid_cells(grid_payload, crop_ids, include_all_cells=False, limit_per_crop_id=0):
    counts = Counter()
    rows = []
    crop_id_set = set(crop_ids)
    for cell in grid_payload.get("grid_cells", []):
        crop_id = str(cell.get("crop_id", ""))
        if crop_id not in crop_id_set:
            continue
        if not include_all_cells and not bool(cell.get("is_selected")):
            continue
        if limit_per_crop_id and counts[crop_id] >= int(limit_per_crop_id):
            continue
        counts[crop_id] += 1
        rows.append(cell)
    return rows


def patch_uid(slide_id, crop_id, row_id, col_id):
    return "{0}__{1}__r{2:04d}_c{3:04d}".format(safe_name(slide_id), safe_name(crop_id), int(row_id), int(col_id))


def collect_manifest_rows(
    input_dir,
    grid_paths_list,
    crop_ids,
    include_all_cells=False,
    limit_per_crop_id=0,
    allow_pil_fallback=False,
    skip_unreadable_wsi=False,
):
    rows = []
    planned = []
    skipped = []
    for grid_path in grid_paths_list:
        payload = read_json(grid_path)
        slide_id = str(payload.get("slide_id") or grid_path.name.split("_wsicrop", 1)[0])
        try:
            wsi_path = resolve_wsi_path(input_dir, payload, allow_pil_fallback=allow_pil_fallback)
        except Exception as exc:
            if not skip_unreadable_wsi:
                raise
            candidate_probes = [
                probe_wsi_readable(path, allow_pil_fallback=allow_pil_fallback)
                for path in wsi_candidate_paths(input_dir, payload)
            ]
            skipped_row = {
                "slide_id": slide_id,
                "grid_json": str(grid_path),
                "reason": "unreadable_wsi",
                "error": str(exc),
                "candidate_probes": candidate_probes,
            }
            skipped.append(skipped_row)
            print(json.dumps({"event": "skip_unreadable_wsi", **skipped_row}, ensure_ascii=False), flush=True)
            continue
        grid_jpg = resolve_grid_jpg(grid_path)
        cells = selected_grid_cells(payload, crop_ids, include_all_cells=include_all_cells, limit_per_crop_id=limit_per_crop_id)
        counts_by_crop_id = Counter(str(cell.get("crop_id", "")) for cell in cells)
        planned.append(
            {
                "slide_id": slide_id,
                "grid_json": str(grid_path),
                "grid_jpg": str(grid_jpg or ""),
                "wsi_path": str(wsi_path),
                "slide_dimensions_level0": payload.get("slide_dimensions_level0", []),
                "level0_crop_bbox": payload.get("level0_crop_bbox", []),
                "counts_by_crop_id": dict(counts_by_crop_id),
                "total_patches": int(sum(counts_by_crop_id.values())),
            }
        )
        for cell in cells:
            crop_id = str(cell.get("crop_id", ""))
            patch_id = list(cell.get("patch_id", [cell.get("row_id", 0), cell.get("col_id", 0)]))
            row_id = int(cell.get("row_id", patch_id[0]))
            col_id = int(cell.get("col_id", patch_id[1]))
            crop_size = int(cell.get("crop_size_level0", cell.get("level0_width", 0) or 0))
            level0_x = int(round(float(cell.get("level0_top_left_x", 0) or 0)))
            level0_y = int(round(float(cell.get("level0_top_left_y", 0) or 0)))
            level0_w = int(round(float(cell.get("level0_width", crop_size) or crop_size)))
            level0_h = int(round(float(cell.get("level0_height", crop_size) or crop_size)))
            uid = patch_uid(slide_id, crop_id, row_id, col_id)
            rows.append(
                {
                    "patch_uid": uid,
                    "slide_id": slide_id,
                    "crop_id": crop_id,
                    "simulated_magnification": float(cell.get("simulated_magnification", 0.0) or 0.0),
                    "crop_size_level0": crop_size,
                    "target_patch_size": int(cell.get("target_patch_size", 256) or 256),
                    "patch_id": patch_id,
                    "row_id": row_id,
                    "col_id": col_id,
                    "level0_bbox": [level0_x, level0_y, level0_x + level0_w, level0_y + level0_h],
                    "level0_top_left_x": level0_x,
                    "level0_top_left_y": level0_y,
                    "level0_width": level0_w,
                    "level0_height": level0_h,
                    "level0_anchor": [cell.get("level0_anchor_x"), cell.get("level0_anchor_y")],
                    "tissue_coverage_ratio": float(cell.get("tissue_coverage_ratio", 0.0) or 0.0),
                    "selection_reason": str(cell.get("selection_reason", "")),
                    "anchor_source": str(cell.get("anchor_source", "")),
                    "is_selected": bool(cell.get("is_selected")),
                    "grid_json": str(grid_path),
                    "grid_jpg": str(grid_jpg or ""),
                    "wsi_path": str(wsi_path),
                }
            )
    return rows, planned, skipped


def crop_manifest_rows(manifest_rows, output_dir, allow_pil_fallback=False, resume=False):
    by_wsi = defaultdict(list)
    for row in manifest_rows:
        by_wsi[row["wsi_path"]].append(row)
    output_rows = []
    total = len(manifest_rows)
    processed = 0
    reused = 0
    started_all = time.time()
    for wsi_path, rows in by_wsi.items():
        print(json.dumps({"event": "crop_wsi_start", "wsi_path": wsi_path, "n_patches": len(rows)}, ensure_ascii=False), flush=True)
        reader = WSIReader(wsi_path, allow_pil_fallback=allow_pil_fallback)
        try:
            slide_dimensions = tuple(int(v) for v in getattr(reader, "dimensions", []))
            for row in rows:
                crop_id = row["crop_id"]
                slide_dir = Path(output_dir) / "crops" / crop_id / safe_name(row["slide_id"])
                slide_dir.mkdir(parents=True, exist_ok=True)
                crop_path = slide_dir / "{0}.png".format(row["patch_uid"])
                row = dict(row)
                row["image_path"] = str(crop_path.relative_to(output_dir))
                row["absolute_image_path"] = str(crop_path)
                if resume and crop_path.exists():
                    reused += 1
                    processed += 1
                    output_rows.append(row)
                    if processed % 1000 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "crop_progress",
                                    "processed": processed,
                                    "total": total,
                                    "reused": reused,
                                    "elapsed_seconds": int(round(time.time() - started_all)),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    continue
                bbox = list(row["level0_bbox"])
                size = int(row["crop_size_level0"])
                crop, requested_bbox, clipped_bbox = read_padded_bbox(reader, bbox, size=size, slide_dimensions=slide_dimensions)
                crop.save(crop_path)
                row["requested_level0_bbox"] = requested_bbox
                row["clipped_level0_bbox"] = clipped_bbox
                processed += 1
                output_rows.append(row)
                if processed % 1000 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "crop_progress",
                                "processed": processed,
                                "total": total,
                                "reused": reused,
                                "elapsed_seconds": int(round(time.time() - started_all)),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        finally:
            reader.close()
        print(
            json.dumps(
                {"event": "crop_wsi_done", "wsi_path": wsi_path, "processed_total": processed, "reused_total": reused},
                ensure_ascii=False,
            ),
            flush=True,
        )
    return output_rows


def five_class_from_prediction(model_name, prediction):
    if prediction.get("five_class") in FIVE_CLASS_LABELS:
        return prediction["five_class"]
    if model_name == "digepath":
        raw = str(prediction.get("class_name") or prediction.get("label") or "").strip().lower()
        raw = raw.replace(" ", "_")
        return DIGEPATH_TO_FIVE_CLASS.get(raw, "")
    raw = str(prediction.get("crc_label") or prediction.get("label") or prediction.get("class_name") or "").strip().upper()
    return CRC100K_TO_FIVE_CLASS.get(raw, "")


def raw_class_from_prediction(model_name, prediction):
    if model_name == "digepath":
        return str(prediction.get("class_name") or prediction.get("label") or "")
    return str(prediction.get("crc_label") or prediction.get("label") or prediction.get("class_name") or "")


def post_model_predictions(model_name, server_url, manifest_rows, output_dir, timeout_seconds, batch_size, progress_every_batches, skip_done):
    import requests

    pending = [row for row in manifest_rows if (row["patch_uid"], model_name) not in skip_done]
    predictions = []
    errors = []
    started_all = time.time()
    for batch_index, start in enumerate(range(0, len(pending), max(1, int(batch_size))), start=1):
        batch = pending[start : start + max(1, int(batch_size))]
        payload = {
            "image_paths": [str(Path(output_dir) / row["image_path"]) for row in batch],
            "patch_ids": [row["patch_id"] for row in batch],
            "task": "{0}_wsi10sample_grid5class".format(model_name),
        }
        started = time.time()
        try:
            response = requests.post(server_url, json=payload, timeout=timeout_seconds)
            latency_ms = int(round((time.time() - started) * 1000.0))
            if response.status_code != 200:
                errors.append(
                    {
                        "model": model_name,
                        "status": response.status_code,
                        "latency_ms": latency_ms,
                        "text": response.text[:1000],
                        "batch_first_patch_uid": batch[0]["patch_uid"],
                        "batch_size": len(batch),
                    }
                )
                continue
            raw_items = response.json().get("predictions", [])
            raw_by_path = {str(item.get("image_path", "")): item for item in raw_items}
            for row in batch:
                absolute_image_path = str(Path(output_dir) / row["image_path"])
                raw = raw_by_path.get(absolute_image_path)
                if raw is None:
                    raw = raw_by_path.get(str(row.get("absolute_image_path", "")))
                if raw is None:
                    errors.append({"model": model_name, "status": "missing_prediction", "patch_uid": row["patch_uid"]})
                    continue
                five_class = five_class_from_prediction(model_name, raw)
                predictions.append(
                    {
                        "patch_uid": row["patch_uid"],
                        "slide_id": row["slide_id"],
                        "crop_id": row["crop_id"],
                        "model": model_name,
                        "raw_class": raw_class_from_prediction(model_name, raw),
                        "five_class": five_class,
                        "confidence": float(raw.get("confidence", 0.0) or 0.0),
                        "probabilities": raw.get("probabilities", {}) or raw.get("probs", {}),
                        "tissue_context": raw.get("tissue_context", {})
                        or tissue_context_from_probabilities(
                            raw.get("probabilities", {}) or raw.get("probs", {}), model_name=model_name
                        ),
                        "tissue_context_label": raw.get("tissue_context_label", ""),
                        "uncertainty": raw.get("uncertainty"),
                        "latency_ms_for_batch": latency_ms,
                        "raw_prediction": raw,
                    }
                )
                context = predictions[-1]["tissue_context"]
                if not predictions[-1]["tissue_context_label"]:
                    predictions[-1]["tissue_context_label"] = tissue_context_label(context)
                if predictions[-1]["uncertainty"] is None:
                    predictions[-1]["uncertainty"] = normalized_entropy(context)
        except requests.Timeout:
            errors.append(
                {
                    "model": model_name,
                    "status": "timeout",
                    "timeout_seconds": timeout_seconds,
                    "batch_first_patch_uid": batch[0]["patch_uid"],
                    "batch_size": len(batch),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "model": model_name,
                    "status": "request_failed",
                    "error": str(exc),
                    "batch_first_patch_uid": batch[0]["patch_uid"],
                    "batch_size": len(batch),
                }
            )
        if progress_every_batches and batch_index % int(progress_every_batches) == 0:
            print(
                json.dumps(
                    {
                        "event": "{0}_progress".format(model_name),
                        "batches_done": batch_index,
                        "predictions_done": len(predictions),
                        "errors": len(errors),
                        "elapsed_seconds": int(round(time.time() - started_all)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return predictions, errors


def fuse_predictions(manifest_rows, predictions):
    by_patch = defaultdict(dict)
    for pred in predictions:
        by_patch[pred["patch_uid"]][pred["model"]] = pred
    rows = []
    for row in manifest_rows:
        models = by_patch.get(row["patch_uid"], {})
        model_labels = {model: models.get(model, {}).get("five_class", "") for model in MODEL_NAMES}
        available = [label for label in model_labels.values() if label]
        if available:
            fused = max(available, key=lambda label: FIVE_CLASS_PRIORITY.get(label, -1))
            counts = Counter(available)
            max_count = max(counts.values())
            if len(counts) == 1 and len(available) == 3:
                agreement = "all_three_agree"
            elif counts[fused] >= 2:
                agreement = "two_model_agree"
            elif max_count >= 2:
                agreement = "two_model_agree_risk_priority_override"
            elif len(available) < 3:
                agreement = "partial_model_predictions"
            else:
                agreement = "three_model_disagree_risk_priority"
        else:
            fused = ""
            agreement = "no_model_predictions"
        rows.append(
            {
                "patch_uid": row["patch_uid"],
                "slide_id": row["slide_id"],
                "crop_id": row["crop_id"],
                "patch_id": row["patch_id"],
                "level0_bbox": row["level0_bbox"],
                "tissue_coverage_ratio": row["tissue_coverage_ratio"],
                "uni_5class": model_labels.get("uni_prismnet", ""),
                "conch_5class": model_labels.get("conch_zeroshot", ""),
                "digepath_5class": model_labels.get("digepath", ""),
                "fused_5class": fused,
                "agreement_status": agreement,
                "priority": FIVE_CLASS_PRIORITY.get(fused, -1),
            }
        )
    return rows


def summarize(manifest_rows, predictions, fused_rows, errors, planned, skipped_wsi=None):
    summary = {
        "n_manifest": len(manifest_rows),
        "n_predictions": len(predictions),
        "n_fused": len(fused_rows),
        "n_errors": len(errors),
        "n_skipped_wsi": len(skipped_wsi or []),
        "manifest_counts_by_crop_id": dict(Counter(row["crop_id"] for row in manifest_rows)),
        "manifest_counts_by_slide": dict(Counter(row["slide_id"] for row in manifest_rows)),
        "prediction_counts_by_model": dict(Counter(row["model"] for row in predictions)),
        "five_class_counts_by_model": {},
        "five_class_counts_by_crop_id_and_model": {},
        "fused_counts_by_crop_id": {},
        "agreement_counts": dict(Counter(row["agreement_status"] for row in fused_rows)),
        "planned": planned,
        "skipped_wsi": skipped_wsi or [],
    }
    for model in MODEL_NAMES:
        model_rows = [row for row in predictions if row["model"] == model]
        summary["five_class_counts_by_model"][model] = dict(Counter(row["five_class"] for row in model_rows))
        for crop_id in sorted(set(row["crop_id"] for row in manifest_rows)):
            key = "{0}|{1}".format(crop_id, model)
            summary["five_class_counts_by_crop_id_and_model"][key] = dict(
                Counter(row["five_class"] for row in model_rows if row["crop_id"] == crop_id)
            )
    for crop_id in sorted(set(row["crop_id"] for row in manifest_rows)):
        summary["fused_counts_by_crop_id"][crop_id] = dict(Counter(row["fused_5class"] for row in fused_rows if row["crop_id"] == crop_id))
    return summary


def write_tables(output_dir, manifest_rows, predictions, fused_rows):
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pred_by_patch_model = defaultdict(dict)
    for pred in predictions:
        pred_by_patch_model[pred["patch_uid"]][pred["model"]] = pred
    fused_by_uid = {row["patch_uid"]: row for row in fused_rows}

    with (tables_dir / "patch_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "patch_uid",
            "slide_id",
            "crop_id",
            "simulated_magnification",
            "row_id",
            "col_id",
            "level0_bbox",
            "tissue_coverage_ratio",
            "image_path",
            "uni_raw_class",
            "uni_5class",
            "uni_confidence",
            "conch_raw_class",
            "conch_5class",
            "conch_confidence",
            "digepath_raw_class",
            "digepath_5class",
            "digepath_confidence",
            "fused_5class",
            "priority",
            "agreement_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            uid = row["patch_uid"]
            models = pred_by_patch_model.get(uid, {})
            fused = fused_by_uid.get(uid, {})
            out = {
                "patch_uid": uid,
                "slide_id": row.get("slide_id", ""),
                "crop_id": row.get("crop_id", ""),
                "simulated_magnification": row.get("simulated_magnification", ""),
                "row_id": row.get("row_id", ""),
                "col_id": row.get("col_id", ""),
                "level0_bbox": json.dumps(row.get("level0_bbox", [])),
                "tissue_coverage_ratio": row.get("tissue_coverage_ratio", ""),
                "image_path": row.get("image_path", ""),
                "fused_5class": fused.get("fused_5class", ""),
                "priority": fused.get("priority", ""),
                "agreement_status": fused.get("agreement_status", ""),
            }
            for model, prefix in (
                ("uni_prismnet", "uni"),
                ("conch_zeroshot", "conch"),
                ("digepath", "digepath"),
            ):
                pred = models.get(model, {})
                out["{0}_raw_class".format(prefix)] = pred.get("raw_class", "")
                out["{0}_5class".format(prefix)] = pred.get("five_class", "")
                out["{0}_confidence".format(prefix)] = pred.get("confidence", "")
            writer.writerow(out)

    with (tables_dir / "model_predictions_long.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["patch_uid", "slide_id", "crop_id", "model", "raw_class", "five_class", "confidence"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with (tables_dir / "five_class_legend.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["five_class", "priority", "color_hex"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label in FIVE_CLASS_LABELS:
            writer.writerow(
                {
                    "five_class": label,
                    "priority": FIVE_CLASS_PRIORITY[label],
                    "color_hex": rgb_to_hex(FIVE_CLASS_COLORS[label]),
                }
            )

    with (tables_dir / "per_slide_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slide_id", "crop_id", "model", "five_class", "count"])
        writer.writeheader()
        counter = Counter((row["slide_id"], row["crop_id"], row["model"], row["five_class"]) for row in predictions)
        for (slide_id, crop_id, model, five_class), count in sorted(counter.items()):
            writer.writerow({"slide_id": slide_id, "crop_id": crop_id, "model": model, "five_class": five_class, "count": count})
    with (tables_dir / "per_magnification_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["crop_id", "model", "five_class", "count"])
        writer.writeheader()
        counter = Counter((row["crop_id"], row["model"], row["five_class"]) for row in predictions)
        for (crop_id, model, five_class), count in sorted(counter.items()):
            writer.writerow({"crop_id": crop_id, "model": model, "five_class": five_class, "count": count})
    with (tables_dir / "model_disagreement.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["patch_uid", "slide_id", "crop_id", "uni_5class", "conch_5class", "digepath_5class", "fused_5class", "agreement_status"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in fused_rows:
            if row["agreement_status"] != "all_three_agree":
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    with (tables_dir / "high_value_patches.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["patch_uid", "slide_id", "crop_id", "fused_5class", "priority", "level0_bbox", "agreement_status"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in fused_rows:
            if row["priority"] >= 3:
                out = dict(row)
                out["level0_bbox"] = json.dumps(row["level0_bbox"])
                writer.writerow({key: out.get(key, "") for key in fieldnames})


def select_overview_level(level_downsamples, target_downsample):
    candidates = [(index, float(value)) for index, value in enumerate(level_downsamples or [1.0]) if float(value) > 0]
    if not candidates:
        return 0, 1.0
    preferred = [item for item in candidates if item[1] <= target_downsample]
    if preferred:
        return max(preferred, key=lambda item: item[1])
    return min(candidates, key=lambda item: item[1])


def read_clean_wsi_overview(wsi_path, level0_crop_bbox, target_size):
    x1, y1, x2, y2 = [int(round(float(value))) for value in level0_crop_bbox]
    target_width, target_height = [int(value) for value in target_size]
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Invalid overlay target size: {0}".format(target_size))
    level0_width = max(1, x2 - x1)
    level0_height = max(1, y2 - y1)
    target_downsample = max(level0_width / float(target_width), level0_height / float(target_height), 1.0)
    reader = WSIReader(wsi_path, allow_pil_fallback=False)
    try:
        level, level_downsample = select_overview_level(getattr(reader, "level_downsamples", [1.0]), target_downsample)
        level_size = (
            max(1, int(math.ceil(level0_width / level_downsample))),
            max(1, int(math.ceil(level0_height / level_downsample))),
        )
        image = reader.read_region((x1, y1), level, level_size).convert("RGB")
    finally:
        reader.close()
    if image.size != (target_width, target_height):
        resample = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((target_width, target_height), resample=resample)
    return image


def overlay_target_size(info, level0_crop_bbox):
    grid_jpg = info.get("grid_jpg") or ""
    if grid_jpg and Path(grid_jpg).exists():
        with Image.open(grid_jpg) as image:
            return image.size
    x1, y1, x2, y2 = [float(value) for value in level0_crop_bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    max_side = 1200.0
    scale = min(max_side / width, max_side / height, 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def build_clean_overlay_base(info):
    level0_crop_bbox = info.get("level0_crop_bbox") or []
    wsi_path = info.get("wsi_path") or ""
    if not wsi_path or not Path(wsi_path).exists() or len(level0_crop_bbox) != 4:
        raise FileNotFoundError("Missing WSI path or level0_crop_bbox for clean overlay base.")
    target_size = overlay_target_size(info, level0_crop_bbox)
    return read_clean_wsi_overview(wsi_path, level0_crop_bbox, target_size), target_size


def render_overlays(output_dir, planned, manifest_rows, predictions, fused_rows):
    overlay_dir = Path(output_dir) / "overlays"
    pred_by_model = defaultdict(dict)
    for pred in predictions:
        pred_by_model[pred["model"]][pred["patch_uid"]] = pred["five_class"]
    fused_by_uid = {row["patch_uid"]: row["fused_5class"] for row in fused_rows}
    rows_by_slide_crop = defaultdict(list)
    for row in manifest_rows:
        rows_by_slide_crop[(row["slide_id"], row["crop_id"])].append(row)

    planned_by_slide = {row["slide_id"]: row for row in planned}
    base_cache = {}
    report = {
        "overlay_base": "wsi_clean_overview",
        "overlay_images_written": 0,
        "overlay_failures": [],
    }
    for (slide_id, crop_id), rows in rows_by_slide_crop.items():
        info = planned_by_slide.get(slide_id, {})
        level0_crop_bbox = info.get("level0_crop_bbox") or []
        if len(level0_crop_bbox) != 4:
            continue
        if slide_id not in base_cache:
            try:
                base_cache[slide_id] = build_clean_overlay_base(info)
            except Exception as exc:
                report["overlay_failures"].append(
                    {"slide_id": slide_id, "wsi_path": info.get("wsi_path", ""), "error": str(exc)}
                )
                continue
        base, _target_size = base_cache[slide_id]
        for model in MODEL_NAMES + ["fused"]:
            image = base.copy()
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for row in rows:
                uid = row["patch_uid"]
                label = fused_by_uid.get(uid, "") if model == "fused" else pred_by_model.get(model, {}).get(uid, "")
                if label not in FIVE_CLASS_COLORS:
                    continue
                rect = project_level0_bbox(row["level0_bbox"], level0_crop_bbox, image.size)
                color = FIVE_CLASS_COLORS[label] + (92,)
                draw.rectangle(rect, fill=color)
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
            out_dir = overlay_dir / safe_name(slide_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            image.save(out_dir / "{0}_{1}_5class.jpg".format(crop_id, model), quality=92)
            report["overlay_images_written"] += 1
    render_legend(output_dir)
    report["legend_path"] = str(overlay_dir / "_legend_5class.png")
    return report


def rgb_to_hex(rgb):
    return "#{0:02X}{1:02X}{2:02X}".format(*rgb)


def render_legend(output_dir):
    overlay_dir = Path(output_dir) / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    width = 560
    row_height = 34
    margin = 18
    image = Image.new("RGB", (width, margin * 2 + row_height * len(FIVE_CLASS_LABELS)), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for index, label in enumerate(FIVE_CLASS_LABELS):
        y = margin + index * row_height
        color = FIVE_CLASS_COLORS[label]
        draw.rectangle([margin, y + 5, margin + 22, y + 27], fill=color, outline=(80, 80, 80))
        text = "{0}  priority={1}  {2}".format(label, FIVE_CLASS_PRIORITY[label], rgb_to_hex(color))
        draw.text((margin + 34, y + 9), text, fill=(30, 30, 30))
    image.save(overlay_dir / "_legend_5class.png")


def output_schema():
    return {
        "schema_version": "wsi_grid_5class_v1",
        "primary_outputs": {
            "manifest.jsonl": "One row per cropped patch/FOV with slide, crop geometry, grid metadata, and crop image path.",
            "predictions.jsonl": "One row per patch per model with raw class, mapped five_class, confidence, probabilities, and raw model payload.",
            "fused_5class_assignments.jsonl": "One row per patch with UNI/CONCH/DIgePath five_class labels, risk-priority fused_5class, priority, and agreement_status.",
            "summary.json": "Run-level counts, label distributions, input plan, and paths to the result files.",
            "run_config.json": "Input directory, selected crop_ids, server URLs, batch settings, and run limits.",
            "errors.jsonl": "One row per failed model request or missing prediction; empty means no recorded request errors.",
            "skipped_wsi.jsonl": "One row per unreadable WSI skipped when --skip-unreadable-wsi is enabled.",
        },
        "table_outputs": {
            "tables/patch_assignments.csv": "Analysis-ready wide table: one patch per row with crop geometry, per-model labels/confidence, and final fused label.",
            "tables/model_predictions_long.csv": "Long-form model prediction table: one patch-model pair per row.",
            "tables/per_slide_counts.csv": "Counts by slide, crop_id, model, and five_class.",
            "tables/per_magnification_counts.csv": "Counts by crop_id, model, and five_class.",
            "tables/model_disagreement.csv": "Patches whose available model labels are not all identical.",
            "tables/high_value_patches.csv": "Fused patches with priority >= 3, i.e. mucus-rich or epithelial-neoplasia-suspicious contexts.",
            "tables/five_class_legend.csv": "Five-class label, risk priority, and overlay color.",
        },
        "image_outputs": {
            "crops/<crop_id>/<slide_id>/<patch_uid>.png": "WSI level-0 crop used as model input.",
            "overlays/<slide_id>/<crop_id>_<model>_5class.jpg": "Clean WSI overview overlaid with model five-class labels.",
            "overlays/<slide_id>/<crop_id>_fused_5class.jpg": "Clean WSI overview overlaid with final fused five-class labels.",
            "overlays/_legend_5class.png": "Color legend for overlay images.",
        },
        "five_class_labels": [
            {
                "label": label,
                "priority": FIVE_CLASS_PRIORITY[label],
                "color_rgb": list(FIVE_CLASS_COLORS[label]),
                "color_hex": rgb_to_hex(FIVE_CLASS_COLORS[label]),
            }
            for label in FIVE_CLASS_LABELS
        ],
        "model_names": MODEL_NAMES,
        "agreement_status_values": [
            "all_three_agree",
            "two_model_agree",
            "two_model_agree_risk_priority_override",
            "partial_model_predictions",
            "three_model_disagree_risk_priority",
            "no_model_predictions",
        ],
    }


def write_output_contract(output_dir):
    output_dir = Path(output_dir)
    write_json(output_dir / "output_schema.json", output_schema())
    readme = """# WSI Grid Five-Class Experiment Outputs

This directory is a complete result bundle for the UNI PrismNet, CONCH zero-shot,
and DIgePath five-class WSI grid experiment.

## Primary Files

- `manifest.jsonl`: one cropped patch/FOV per line. Use this to trace each result
  back to the slide, grid cell, level-0 bbox, tissue ratio, and crop PNG.
- `predictions.jsonl`: one model prediction per line. This keeps the raw model
  class, mapped five-class label, confidence, probabilities, and raw payload.
- `fused_5class_assignments.jsonl`: one final assignment per patch. This is the
  main machine-readable output for downstream Trace/Navigation experiments.
- `tables/patch_assignments.csv`: the main spreadsheet-friendly output. It has
  one patch per row, three model labels/confidences, and the final fused label.
- `summary.json`: run-level counts and paths to the output files.
- `errors.jsonl`: request errors and missing predictions. An empty file means no
  request-level errors were recorded.
- `skipped_wsi.jsonl`: unreadable WSI records when `--skip-unreadable-wsi` is
  enabled. An empty file means no WSI was skipped.

## Visual Checks

- `overlays/<slide_id>/<crop_id>_fused_5class.jpg`: final fused labels painted on
  a clean overview image cropped from the original WSI.
- `overlays/<slide_id>/<crop_id>_<model>_5class.jpg`: per-model clean-WSI
  overlay images.
- `overlays/_legend_5class.png`: color legend.

## Five-Class Labels

| Label | Priority | Color |
| --- | ---: | --- |
"""
    for label in FIVE_CLASS_LABELS:
        readme += "| `{0}` | {1} | `{2}` |\n".format(label, FIVE_CLASS_PRIORITY[label], rgb_to_hex(FIVE_CLASS_COLORS[label]))
    readme += """
Priority is risk-oriented: higher values win during fusion when models disagree.
"""
    (output_dir / "README_outputs.md").write_text(readme, encoding="utf-8")


def project_level0_bbox(level0_bbox, level0_crop_bbox, image_size):
    x1, y1, x2, y2 = [float(v) for v in level0_bbox]
    bx1, by1, bx2, by2 = [float(v) for v in level0_crop_bbox]
    width, height = image_size
    sx = width / max(1.0, bx2 - bx1)
    sy = height / max(1.0, by2 - by1)
    return [
        int(round((x1 - bx1) * sx)),
        int(round((y1 - by1) * sy)),
        int(round((x2 - bx1) * sx)),
        int(round((y2 - by1) * sy)),
    ]


def main():
    args = parse_args()
    crop_ids = parse_crop_ids(args.crop_ids)
    if not args.output_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / "artifacts" / "wsi10sample_grid5class_uni_conch_digepath_{0}".format(stamp))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "crop_ids": crop_ids,
        "include_all_cells": bool(args.include_all_cells),
        "limit_slides": int(args.limit_slides),
        "limit_patches_per_crop_id": int(args.limit_patches_per_crop_id),
        "batch_size": int(args.batch_size),
        "timeout_seconds": int(args.timeout_seconds),
        "allow_pil_fallback": bool(args.allow_pil_fallback),
        "skip_unreadable_wsi": bool(args.skip_unreadable_wsi),
        "uni_url": args.uni_url,
        "conch_url": args.conch_url,
        "digepath_url": args.digepath_url,
    }
    write_json(output_dir / "run_config.json", run_config)
    write_output_contract(output_dir)

    manifest_path = output_dir / "manifest.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    fused_path = output_dir / "fused_5class_assignments.jsonl"
    errors_path = output_dir / "errors.jsonl"
    skipped_wsi_path = output_dir / "skipped_wsi.jsonl"
    if args.resume:
        for path in (manifest_path, fused_path, skipped_wsi_path):
            if path.exists():
                path.unlink()
    else:
        for path in (manifest_path, predictions_path, fused_path, errors_path, skipped_wsi_path):
            if path.exists():
                path.unlink()
    if not args.dry_run:
        for path in (manifest_path, predictions_path, fused_path, errors_path, skipped_wsi_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

    grid_list = grid_paths(args.input_dir, limit_slides=args.limit_slides)
    manifest_rows, planned, skipped_wsi = collect_manifest_rows(
        args.input_dir,
        grid_list,
        crop_ids,
        include_all_cells=args.include_all_cells,
        limit_per_crop_id=args.limit_patches_per_crop_id,
        allow_pil_fallback=args.allow_pil_fallback,
        skip_unreadable_wsi=args.skip_unreadable_wsi,
    )
    append_jsonl(skipped_wsi_path, skipped_wsi)
    if args.dry_run:
        summary = summarize(manifest_rows, [], [], [], planned, skipped_wsi=skipped_wsi)
        summary["dry_run"] = True
        write_json(output_dir / "summary.json", summary)
        print(json.dumps({"event": "dry_run_done", "summary_json": str(output_dir / "summary.json")}, ensure_ascii=False), flush=True)
        return

    print(json.dumps({"event": "crop_start", "n_patches": len(manifest_rows)}, ensure_ascii=False), flush=True)
    manifest_rows = crop_manifest_rows(manifest_rows, output_dir, allow_pil_fallback=args.allow_pil_fallback, resume=args.resume)
    append_jsonl(manifest_path, manifest_rows)
    print(json.dumps({"event": "crop_done", "n_patches": len(manifest_rows)}, ensure_ascii=False), flush=True)

    existing_done = set()
    existing_predictions = []
    errors = []
    if args.resume:
        for row in iter_jsonl(predictions_path) or []:
            existing_predictions.append(row)
            existing_done.add((row.get("patch_uid"), row.get("model")))
        errors.extend(list(iter_jsonl(errors_path) or []))

    predictions = list(existing_predictions)
    model_jobs = []
    if not args.skip_uni:
        model_jobs.append(("uni_prismnet", args.uni_url))
    if not args.skip_conch:
        model_jobs.append(("conch_zeroshot", args.conch_url))
    if not args.skip_digepath:
        model_jobs.append(("digepath", args.digepath_url))
    for model_name, server_url in model_jobs:
        print(json.dumps({"event": "{0}_start".format(model_name), "n_patches": len(manifest_rows)}, ensure_ascii=False), flush=True)
        rows, model_errors = post_model_predictions(
            model_name,
            server_url,
            manifest_rows,
            output_dir,
            timeout_seconds=args.timeout_seconds,
            batch_size=args.batch_size,
            progress_every_batches=args.progress_every_batches,
            skip_done=existing_done,
        )
        predictions.extend(rows)
        errors.extend(model_errors)
        append_jsonl(predictions_path, rows)
        append_jsonl(errors_path, model_errors)
        for row in rows:
            existing_done.add((row.get("patch_uid"), row.get("model")))
        print(
            json.dumps(
                {"event": "{0}_done".format(model_name), "n_predictions": len(rows), "n_errors": len(model_errors)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    fused_rows = fuse_predictions(manifest_rows, predictions)
    append_jsonl(fused_path, fused_rows)
    write_tables(output_dir, manifest_rows, predictions, fused_rows)
    overlay_report = {}
    if not args.skip_overlays:
        overlay_report = render_overlays(output_dir, planned, manifest_rows, predictions, fused_rows)
    summary = summarize(manifest_rows, predictions, fused_rows, errors, planned, skipped_wsi=skipped_wsi)
    summary.update(
        {
            "run_config_json": str(output_dir / "run_config.json"),
            "output_schema_json": str(output_dir / "output_schema.json"),
            "readme_outputs_md": str(output_dir / "README_outputs.md"),
            "manifest_jsonl": str(manifest_path),
            "predictions_jsonl": str(predictions_path),
            "fused_5class_assignments_jsonl": str(fused_path),
            "errors_jsonl": str(errors_path),
            "skipped_wsi_jsonl": str(skipped_wsi_path),
            "tables_dir": str(output_dir / "tables"),
            "overlays_dir": str(output_dir / "overlays"),
            "patch_assignments_csv": str(output_dir / "tables" / "patch_assignments.csv"),
            "model_predictions_long_csv": str(output_dir / "tables" / "model_predictions_long.csv"),
            "five_class_legend_csv": str(output_dir / "tables" / "five_class_legend.csv"),
            "overlay_report": overlay_report,
        }
    )
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"event": "done", "summary_json": str(output_dir / "summary.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
