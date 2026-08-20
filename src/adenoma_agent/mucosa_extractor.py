import json
import math
import re
import shutil
import tempfile
from collections import OrderedDict, defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from adenoma_agent.agents.junior import generate_anatomical_mask
from adenoma_agent.utils import ensure_dir, read_json, read_jsonl, write_json
from adenoma_agent.wsi import SUPPORTED_IMAGE_SUFFIXES, WSIReader


CRC100K_LABELS = ("DEB", "ADI", "BACK", "MUS", "NORM", "LYM", "MUC", "STR", "TUM")
TISSUE_CONTEXT_LABELS = (
    "background_or_artifact",
    "smooth_muscle_or_deep_tissue",
    "normal_epi_context",
    "inflammatory_context",
    "mucus_rich_context",
    "stromal_context",
    "abnormal_epithelial_candidate",
)
CRC100K_CONTEXT_MAP = OrderedDict(
    [
        ("background_or_artifact", ("DEB", "ADI", "BACK")),
        ("smooth_muscle_or_deep_tissue", ("MUS",)),
        ("normal_epi_context", ("NORM",)),
        ("inflammatory_context", ("LYM",)),
        ("mucus_rich_context", ("MUC",)),
        ("stromal_context", ("STR",)),
        ("abnormal_epithelial_candidate", ("TUM",)),
    ]
)
TISSUE_CONTEXT_COLORS = OrderedDict(
    [
        ("background_or_artifact", (156, 163, 175)),
        ("smooth_muscle_or_deep_tissue", (120, 86, 70)),
        ("normal_epi_context", (63, 174, 90)),
        ("inflammatory_context", (245, 158, 11)),
        ("mucus_rich_context", (6, 182, 212)),
        ("stromal_context", (139, 92, 246)),
        ("abnormal_epithelial_candidate", (220, 38, 38)),
    ]
)
SUPPORTED_WSI_SUFFIXES = set(SUPPORTED_IMAGE_SUFFIXES)

# Backward-compatible test/import name; all runtime call sites use WSIReader.
_WSIReader = WSIReader


class MucosaExtractorConfig:
    def __init__(
        self,
        source_model="uni_prismnet",
        pathprism_url="http://127.0.0.1:8400/predict",
        batch_size=64,
        timeout_seconds=240,
        classifier_magnification=20.0,
        classifier_patch_size=256,
        min_tissue_coverage=0.05,
        overview_downsample=32.0,
        mask_downsample=32.0,
        mucosa_threshold=0.30,
        presence_area_threshold=0.005,
        presence_peak_threshold=0.50,
        five_x_magnification=5.0,
        five_x_patch_size=256,
        base_magnification=0.0,
        mpp=0.0,
        probability_sum_tolerance=1e-4,
        write_crops=True,
        write_overlays=True,
        resume=False,
    ):
        self.source_model = source_model
        self.pathprism_url = pathprism_url
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.classifier_magnification = classifier_magnification
        self.classifier_patch_size = classifier_patch_size
        self.min_tissue_coverage = min_tissue_coverage
        self.overview_downsample = overview_downsample
        self.mask_downsample = mask_downsample
        self.mucosa_threshold = mucosa_threshold
        self.presence_area_threshold = presence_area_threshold
        self.presence_peak_threshold = presence_peak_threshold
        self.five_x_magnification = five_x_magnification
        self.five_x_patch_size = five_x_patch_size
        self.base_magnification = base_magnification
        self.mpp = mpp
        self.probability_sum_tolerance = probability_sum_tolerance
        self.write_crops = write_crops
        self.write_overlays = write_overlays
        self.resume = resume

    def to_dict(self):
        return dict(self.__dict__)


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return value.strip("_") or "unnamed"


def parse_level0_bbox(value):
    if isinstance(value, dict):
        x1 = int(value.get("x1", value.get("xmin", 0)))
        y1 = int(value.get("y1", value.get("ymin", 0)))
        return [
            x1,
            y1,
            int(value.get("x2", x1 + int(value.get("width", 0)))),
            int(value.get("y2", y1 + int(value.get("height", 0)))),
        ]
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [int(round(float(item))) for item in value[:4]]
    return []


def validate_crc100k_probabilities(probabilities, tolerance=1e-4):
    probabilities = probabilities or {}
    missing = [label for label in CRC100K_LABELS if label not in probabilities]
    if missing:
        raise ValueError("Missing CRC100K probabilities: {0}".format(", ".join(missing)))
    values = OrderedDict()
    for label in CRC100K_LABELS:
        value = float(probabilities[label])
        if not math.isfinite(value):
            raise ValueError("Non-finite CRC100K probability for {0}".format(label))
        if value < 0.0:
            raise ValueError("Negative CRC100K probability for {0}".format(label))
        values[label] = value
    original_sum = float(sum(values.values()))
    if original_sum <= 0.0:
        raise ValueError("CRC100K probability sum must be positive")
    renormalized = abs(original_sum - 1.0) > float(tolerance)
    if renormalized:
        values = OrderedDict((label, value / original_sum) for label, value in values.items())
    return values, {"input_probability_sum": original_sum, "renormalized": renormalized}


def tissue_context_from_probabilities(probabilities, model_name="uni_prismnet", strict=True):
    del model_name
    if strict:
        raw, _provenance = validate_crc100k_probabilities(probabilities)
    else:
        raw = OrderedDict((label, float((probabilities or {}).get(label, 0.0) or 0.0)) for label in CRC100K_LABELS)
        total = float(sum(raw.values()))
        if total <= 0.0:
            return OrderedDict((label, 1.0 if label == "background_or_artifact" else 0.0) for label in TISSUE_CONTEXT_LABELS)
        raw = OrderedDict((label, value / total) for label, value in raw.items())
    return OrderedDict(
        (context_label, float(sum(raw[source] for source in source_labels)))
        for context_label, source_labels in CRC100K_CONTEXT_MAP.items()
    )


def normalized_entropy(probabilities, labels=TISSUE_CONTEXT_LABELS):
    values = np.asarray([float((probabilities or {}).get(label, 0.0) or 0.0) for label in labels], dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return 1.0
    values /= total
    positive = values[values > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return max(0.0, min(1.0, entropy / math.log(float(len(labels)))))


def tissue_context_label(probabilities):
    return max(TISSUE_CONTEXT_LABELS, key=lambda label: float((probabilities or {}).get(label, 0.0) or 0.0))


def mucosa_score(probabilities):
    return float((probabilities or {}).get("normal_epi_context", 0.0) or 0.0) + float(
        (probabilities or {}).get("abnormal_epithelial_candidate", 0.0) or 0.0
    )


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _select_level(level_downsamples, target_downsample):
    candidates = [(index, float(value)) for index, value in enumerate(level_downsamples) if float(value) > 0.0]
    preferred = [item for item in candidates if item[1] <= float(target_downsample)]
    return max(preferred, key=lambda item: item[1]) if preferred else min(candidates, key=lambda item: item[1])


def _read_level0_bbox(reader, bbox, target_size):
    x1, y1, x2, y2 = [int(value) for value in bbox]
    target_width, target_height = [int(value) for value in target_size]
    source_width = max(1, x2 - x1)
    source_height = max(1, y2 - y1)
    target_downsample = max(source_width / float(target_width), source_height / float(target_height), 1.0)
    level, downsample = _select_level(reader.level_downsamples, target_downsample)
    level_size = (max(1, int(math.ceil(source_width / downsample))), max(1, int(math.ceil(source_height / downsample))))
    image = reader.read_region((x1, y1), level, level_size)
    if image.mode == "RGBA":
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image).convert("RGB")
    else:
        image = image.convert("RGB")
    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), resample=getattr(Image, "Resampling", Image).LANCZOS)
    return image


def _overview_and_tissue_mask(reader, downsample):
    width = max(1, int(math.ceil(reader.dimensions[0] / float(downsample))))
    height = max(1, int(math.ceil(reader.dimensions[1] / float(downsample))))
    overview = _read_level0_bbox(reader, [0, 0, reader.dimensions[0], reader.dimensions[1]], (width, height))
    rgb = np.asarray(overview, dtype=np.uint8)
    mean = rgb.astype(np.float32).mean(axis=2)
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    tissue = (mean < 247.0) & ((chroma > 8) | (mean < 225.0))
    return overview, tissue


def _infer_base_magnification(row, configured=0.0):
    if float(configured or 0.0) > 0.0:
        return float(configured)
    if float(row.get("base_magnification", 0.0) or 0.0) > 0.0:
        return float(row["base_magnification"])
    simulated = float(row.get("simulated_magnification", 0.0) or 0.0)
    target_size = float(row.get("target_patch_size", 0.0) or 0.0)
    bbox = parse_level0_bbox(row.get("level0_bbox", []))
    if simulated > 0.0 and target_size > 0.0 and len(bbox) == 4:
        return simulated * float(max(1, bbox[2] - bbox[0])) / target_size
    return 0.0


def _build_tiles_from_wsi(wsi_path, output_dir, config):
    # Preserve a safe caller-provided alias in inference-facing manifests.
    # The reader itself resolves the real source only inside local provenance.
    wsi_path = Path(wsi_path).absolute()
    reader = WSIReader(
        wsi_path,
        base_magnification=(float(config.base_magnification) if float(config.base_magnification or 0.0) > 0.0 else None),
        mpp=(float(config.mpp) if float(config.mpp or 0.0) > 0.0 else None),
    )
    try:
        base_magnification = float(config.base_magnification or reader.base_magnification or 0.0)
        if base_magnification <= 0.0:
            raise ValueError("Base magnification is unavailable for {0}; provide base_magnification".format(wsi_path))
        reader.require_physical_metadata()
        tile_extent = max(1, int(round(config.classifier_patch_size * base_magnification / config.classifier_magnification)))
        _overview, tissue_mask = _overview_and_tissue_mask(reader, config.overview_downsample)
        slide_id = wsi_path.stem
        rows = []
        crop_dir = ensure_dir(Path(output_dir) / safe_name(slide_id))
        height, width = tissue_mask.shape
        row_id = 0
        for y in range(0, reader.dimensions[1], tile_extent):
            col_id = 0
            for x in range(0, reader.dimensions[0], tile_extent):
                requested = [x, y, x + tile_extent, y + tile_extent]
                clipped = [x, y, min(reader.dimensions[0], x + tile_extent), min(reader.dimensions[1], y + tile_extent)]
                tx1 = max(0, min(width, int(math.floor(x / config.overview_downsample))))
                ty1 = max(0, min(height, int(math.floor(y / config.overview_downsample))))
                tx2 = max(tx1 + 1, min(width, int(math.ceil((x + tile_extent) / config.overview_downsample))))
                ty2 = max(ty1 + 1, min(height, int(math.ceil((y + tile_extent) / config.overview_downsample))))
                coverage = float(tissue_mask[ty1:ty2, tx1:tx2].mean()) if ty2 > ty1 and tx2 > tx1 else 0.0
                if coverage >= float(config.min_tissue_coverage):
                    uid = "{0}__20x__r{1:05d}_c{2:05d}".format(safe_name(slide_id), row_id, col_id)
                    crop_path = Path(crop_dir) / "{0}.png".format(uid)
                    crop = reader.crop_level0_bbox(
                        requested,
                        requested_magnification=float(config.classifier_magnification),
                        output_pixels=(int(config.classifier_patch_size), int(config.classifier_patch_size)),
                    )
                    if not crop_path.exists():
                        crop.save(crop_path)
                    rows.append(
                        {
                            "patch_uid": uid,
                            "patch_id": [row_id, col_id],
                            "row_id": row_id,
                            "col_id": col_id,
                            "slide_id": slide_id,
                            "crop_id": "20x_{0}".format(config.classifier_patch_size),
                            "simulated_magnification": float(config.classifier_magnification),
                            "target_patch_size": int(config.classifier_patch_size),
                            "base_magnification": base_magnification,
                            "level0_bbox": requested,
                            "requested_level0_bbox": requested,
                            "clipped_level0_bbox": clipped,
                            "tissue_coverage_ratio": coverage,
                            "image_path": str(crop_path),
                            "wsi_path": str(wsi_path),
                            "slide_dimensions_level0": list(reader.dimensions),
                            "wsi_backend": reader.backend_name,
                            "mpp_x": float(reader.mpp_x),
                            "mpp_y": float(reader.mpp_y),
                            "crop_provenance": dict(crop.provenance),
                        }
                    )
                col_id += 1
            row_id += 1
        planned = {
            "slide_id": slide_id,
            "wsi_path": str(wsi_path),
            "slide_dimensions_level0": list(reader.dimensions),
            "level0_crop_bbox": [0, 0, reader.dimensions[0], reader.dimensions[1]],
            "base_magnification": base_magnification,
            "base_magnification_source": reader.base_magnification_source,
            "mpp_x": float(reader.mpp_x),
            "mpp_y": float(reader.mpp_y),
            "mpp_source": reader.mpp_source,
            "wsi_backend": reader.backend_name,
            "classifier_tile_extent_level0": tile_extent,
        }
        return rows, planned
    finally:
        reader.close()


def _post_pathprism_predictions(rows, output_dir, config, already_done):
    import requests

    pending = [row for row in rows if row["patch_uid"] not in already_done]
    predictions = []
    errors = []
    for start in range(0, len(pending), max(1, int(config.batch_size))):
        batch = pending[start : start + max(1, int(config.batch_size))]
        payload = {
            "image_paths": [str(row["image_path"]) for row in batch],
            "patch_ids": [row["patch_id"] for row in batch],
            "task": "mucosa_extractor_crc100k",
        }
        try:
            response = requests.post(config.pathprism_url, json=payload, timeout=int(config.timeout_seconds))
            if response.status_code != 200:
                for row in batch:
                    errors.append(
                        {
                            "slide_id": row.get("slide_id"),
                            "tile_id": row.get("patch_uid"),
                            "stage": "pathprism_inference",
                            "error_type": "HTTPError",
                            "message": "HTTP {0}: {1}".format(response.status_code, response.text[:1000]),
                        }
                    )
                continue
            items = response.json().get("predictions", [])
            by_path = {str(item.get("image_path", "")): item for item in items}
            for row in batch:
                raw = by_path.get(str(row["image_path"]))
                if raw is None:
                    errors.append(
                        {
                            "slide_id": row.get("slide_id"),
                            "tile_id": row.get("patch_uid"),
                            "stage": "pathprism_inference",
                            "error_type": "MissingPrediction",
                            "message": "PathPrism response did not contain this tile",
                        }
                    )
                    continue
                predictions.append(
                    {
                        "patch_uid": row["patch_uid"],
                        "slide_id": row["slide_id"],
                        "crop_id": row["crop_id"],
                        "model": config.source_model,
                        "raw_class": str(raw.get("crc_label") or raw.get("label") or raw.get("class_name") or ""),
                        "confidence": float(raw.get("confidence", 0.0) or 0.0),
                        "probabilities": raw.get("probabilities", {}) or raw.get("probs", {}),
                        "raw_prediction": raw,
                    }
                )
        except Exception as exc:
            for row in batch:
                errors.append(
                    {
                        "slide_id": row.get("slide_id"),
                        "tile_id": row.get("patch_uid"),
                        "stage": "pathprism_inference",
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    }
                )
    if errors:
        _append_jsonl(Path(output_dir) / "errors.jsonl", errors)
    if pending and not predictions:
        raise RuntimeError("PathPrism inference failed for all pending tiles; see errors.jsonl")
    return predictions, errors


def _planned_from_artifact(artifact_dir):
    path = Path(artifact_dir) / "summary.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    planned = payload.get("planned", [])
    if isinstance(planned, dict):
        planned = planned.get("planned_rows", planned.get("planned", []))
    return {str(row.get("slide_id", "")): row for row in planned if row.get("slide_id")}


def _load_artifact_inputs(artifact_dir, config):
    artifact_dir = Path(artifact_dir).resolve()
    manifest_path = artifact_dir / "manifest.jsonl"
    predictions_path = artifact_dir / "predictions.jsonl"
    if not manifest_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("artifact_dir must contain manifest.jsonl and predictions.jsonl")
    all_manifest = read_jsonl(manifest_path)
    manifest = []
    for row in all_manifest:
        magnification = float(row.get("simulated_magnification", 0.0) or 0.0)
        crop_id = str(row.get("crop_id", "") or "").lower()
        if magnification > 0.0:
            eligible = abs(magnification - float(config.classifier_magnification)) < 1e-6
        else:
            eligible = crop_id.startswith("20x")
        if eligible:
            manifest.append(row)
    eligible_uids = {row.get("patch_uid") for row in manifest}
    source_by_uid = OrderedDict()
    for row in read_jsonl(predictions_path):
        if str(row.get("model", "")) != config.source_model or row.get("patch_uid") not in eligible_uids:
            continue
        source_by_uid[row.get("patch_uid")] = row
    source = list(source_by_uid.values())
    wanted = set(source_by_uid)
    manifest = [row for row in manifest if row.get("patch_uid") in wanted]
    planned = _planned_from_artifact(artifact_dir)
    return manifest, source, planned


def _build_context_records(manifest_rows, prediction_rows, config):
    manifest_by_uid = {row.get("patch_uid"): row for row in manifest_rows}
    records = []
    normalized_predictions = []
    errors = []
    for prediction in prediction_rows:
        uid = prediction.get("patch_uid")
        manifest = manifest_by_uid.get(uid)
        if manifest is None:
            errors.append(
                {
                    "slide_id": prediction.get("slide_id"),
                    "tile_id": uid,
                    "stage": "evidence_construction",
                    "error_type": "MissingManifest",
                    "message": "Prediction has no matching tile manifest row",
                }
            )
            continue
        try:
            raw, probability_provenance = validate_crc100k_probabilities(
                prediction.get("probabilities", {}), tolerance=config.probability_sum_tolerance
            )
        except Exception as exc:
            errors.append(
                {
                    "slide_id": manifest.get("slide_id"),
                    "tile_id": uid,
                    "stage": "probability_validation",
                    "error_type": "InvalidProbabilities",
                    "message": str(exc),
                }
            )
            continue
        context = tissue_context_from_probabilities(raw, strict=True)
        score = mucosa_score(context)
        normalized_prediction = dict(prediction)
        normalized_prediction["probabilities"] = raw
        normalized_prediction["probability_validation"] = probability_provenance
        normalized_predictions.append(normalized_prediction)
        records.append(
            {
                "patch_uid": uid,
                "patch_id": manifest.get("patch_id"),
                "slide_id": manifest.get("slide_id"),
                "crop_id": manifest.get("crop_id"),
                "simulated_magnification": manifest.get("simulated_magnification"),
                "target_patch_size": manifest.get("target_patch_size"),
                "base_magnification": manifest.get("base_magnification"),
                "mpp_x": manifest.get("mpp_x"),
                "mpp_y": manifest.get("mpp_y"),
                "level0_bbox": manifest.get("level0_bbox"),
                "requested_level0_bbox": manifest.get("requested_level0_bbox", manifest.get("level0_bbox")),
                "clipped_level0_bbox": manifest.get("clipped_level0_bbox", manifest.get("level0_bbox")),
                "tissue_coverage_ratio": float(manifest.get("tissue_coverage_ratio", 0.0) or 0.0),
                "raw_probabilities": raw,
                "tissue_context": context,
                "tissue_context_label": tissue_context_label(context),
                "mucosa_score": score,
                "mucosa_candidate": bool(score >= config.mucosa_threshold),
                "uncertainty": normalized_entropy(raw, labels=CRC100K_LABELS),
                "source_model": config.source_model,
                "provenance": {
                    "mapping": "crc100k_to_seven_tissue_context_v1",
                    "probability_validation": probability_provenance,
                    "mucosa_formula": "normal_epi_context + abnormal_epithelial_candidate",
                    "mucosa_threshold": float(config.mucosa_threshold),
                    "wsi_backend": manifest.get("wsi_backend"),
                    "crop_provenance": manifest.get("crop_provenance", {}),
                },
            }
        )
    return records, normalized_predictions, errors


def _canvas_shape(window, downsample):
    return (
        max(1, int(math.ceil((window[3] - window[1]) / float(downsample)))),
        max(1, int(math.ceil((window[2] - window[0]) / float(downsample)))),
    )


def _project_bbox(bbox, window, downsample, shape):
    x1, y1, x2, y2 = bbox
    ox, oy, _x2, _y2 = window
    height, width = shape
    return (
        max(0, min(width, int(math.floor((x1 - ox) / float(downsample))))),
        max(0, min(height, int(math.floor((y1 - oy) / float(downsample))))),
        max(0, min(width, int(math.ceil((x2 - ox) / float(downsample))))),
        max(0, min(height, int(math.ceil((y2 - oy) / float(downsample))))),
    )


def _slide_window(slide_id, rows, planned):
    info = planned.get(slide_id, {})
    bbox = parse_level0_bbox(info.get("level0_crop_bbox", []))
    if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        return bbox
    dimensions = info.get("slide_dimensions_level0", [])
    if isinstance(dimensions, (list, tuple)) and len(dimensions) >= 2:
        return [0, 0, int(dimensions[0]), int(dimensions[1])]
    boxes = [parse_level0_bbox(row.get("level0_bbox", [])) for row in rows]
    boxes = [box for box in boxes if len(box) == 4]
    return [min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _build_slide_arrays(rows, window, downsample):
    shape = _canvas_shape(window, downsample)
    raw_sums = np.zeros((len(CRC100K_LABELS), shape[0], shape[1]), dtype=np.float32)
    context_sums = np.zeros((len(TISSUE_CONTEXT_LABELS), shape[0], shape[1]), dtype=np.float32)
    uncertainty_sum = np.zeros(shape, dtype=np.float32)
    count = np.zeros(shape, dtype=np.float32)
    for row in rows:
        bbox = parse_level0_bbox(row.get("clipped_level0_bbox", row.get("level0_bbox", [])))
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = _project_bbox(bbox, window, downsample, shape)
        if x2 <= x1 or y2 <= y1:
            continue
        count[y1:y2, x1:x2] += 1.0
        for index, label in enumerate(CRC100K_LABELS):
            raw_sums[index, y1:y2, x1:x2] += float(row["raw_probabilities"][label])
        for index, label in enumerate(TISSUE_CONTEXT_LABELS):
            context_sums[index, y1:y2, x1:x2] += float(row["tissue_context"][label])
        uncertainty_sum[y1:y2, x1:x2] += float(row["uncertainty"])
    safe = np.where(count > 0.0, count, 1.0)
    return raw_sums / safe[None, :, :], context_sums / safe[None, :, :], uncertainty_sum / safe, count > 0.0, count


def _binary_dilate(mask, radius):
    current = np.asarray(mask, dtype=bool)
    for _index in range(max(0, int(radius))):
        padded = np.pad(current, 1, mode="constant", constant_values=False)
        neighbors = [padded[dy : dy + current.shape[0], dx : dx + current.shape[1]] for dy in range(3) for dx in range(3)]
        current = np.logical_or.reduce(neighbors)
    return current


def _connected_component_labels(mask):
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    component = 0
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            component += 1
            labels[y, x] = component
            queue = deque([(x, y)])
            while queue:
                cx, cy = queue.popleft()
                for ny in range(max(0, cy - 1), min(mask.shape[0], cy + 2)):
                    for nx in range(max(0, cx - 1), min(mask.shape[1], cx + 2)):
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = component
                            queue.append((nx, ny))
    return labels, component


def _fraction(mask, denominator):
    denominator = np.asarray(denominator, dtype=bool)
    if not np.any(denominator):
        return None
    return float(np.asarray(mask, dtype=bool)[denominator].mean())


def _context_statistics(context_maps, valid_mask, argmax_map, mucosa_mask, near_ring, config):
    results = OrderedDict()
    mucosa_bool = np.asarray(mucosa_mask, dtype=np.uint8) > 0
    for index, label in enumerate(TISSUE_CONTEXT_LABELS):
        probabilities = context_maps[index]
        hard = argmax_map == index
        valid_values = probabilities[valid_mask]
        soft_burden = float(valid_values.mean()) if valid_values.size else None
        hard_fraction = _fraction(hard, valid_mask)
        peak = float(valid_values.max()) if valid_values.size else None
        presence = bool(
            hard_fraction is not None
            and peak is not None
            and hard_fraction >= float(config.presence_area_threshold)
            and peak >= float(config.presence_peak_threshold)
        )
        results[label] = {
            "soft_burden": soft_burden,
            "hard_area_fraction": hard_fraction,
            "peak_probability": peak,
            "presence": presence,
            "fraction_of_valid_tissue": hard_fraction,
            "fraction_inside_mucosa_mask": _fraction(hard, valid_mask & mucosa_bool),
            "fraction_near_mucosa_mask": _fraction(hard, valid_mask & near_ring),
            "evaluable": {
                "valid_tissue": bool(np.any(valid_mask)),
                "inside_mucosa_mask": bool(np.any(valid_mask & mucosa_bool)),
                "near_mucosa_mask": bool(np.any(valid_mask & near_ring)),
            },
        }
    return results


def _probability_image(array):
    return Image.fromarray(np.clip(np.asarray(array) * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")


def _hard_context_image(argmax_map, valid_mask):
    rgb = np.full((argmax_map.shape[0], argmax_map.shape[1], 3), 255, dtype=np.uint8)
    for index, label in enumerate(TISSUE_CONTEXT_LABELS):
        rgb[(argmax_map == index) & valid_mask] = TISSUE_CONTEXT_COLORS[label]
    return Image.fromarray(rgb, mode="RGB")


def _overlay_base(planned_info, window, size):
    wsi_path = str(planned_info.get("wsi_path", "") or "")
    if wsi_path and Path(wsi_path).exists():
        reader = WSIReader(wsi_path)
        try:
            return _read_level0_bbox(reader, window, (size[1], size[0]))
        finally:
            reader.close()
    for key in ("overview_png", "grid_jpg"):
        candidate = str(planned_info.get(key, "") or "")
        if candidate and Path(candidate).exists():
            return Image.open(candidate).convert("RGB").resize((size[1], size[0]))
    return Image.new("RGB", (size[1], size[0]), (245, 245, 245))


def _context_overlay(base, context_maps, valid_mask):
    base = base.convert("RGBA")
    context_rgb = np.zeros((valid_mask.shape[0], valid_mask.shape[1], 3), dtype=np.float32)
    for index, label in enumerate(TISSUE_CONTEXT_LABELS):
        context_rgb += context_maps[index, :, :, None] * np.asarray(TISSUE_CONTEXT_COLORS[label], dtype=np.float32)
    context_image = Image.fromarray(np.clip(context_rgb, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")
    alpha = Image.fromarray(np.where(valid_mask, 125, 0).astype(np.uint8), mode="L")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(context_image, mask=alpha)
    return Image.alpha_composite(base, layer).convert("RGB")


def _mask_overlay(base, mucosa_mask):
    base = base.convert("RGBA")
    mask_alpha = Image.fromarray(np.where(np.asarray(mucosa_mask) > 0, 95, 0).astype(np.uint8), mode="L")
    green = Image.new("RGBA", base.size, (30, 190, 110, 255))
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_layer.paste(green, mask=mask_alpha)
    return Image.alpha_composite(base, mask_layer).convert("RGB")


def _panel_cell(image, title, size=(384, 320)):
    width, height = size
    title_height = 28
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    image = image.convert("RGB")
    scale = min(width / float(max(1, image.width)), (height - title_height) / float(max(1, image.height)))
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        resample=getattr(Image, "Resampling", Image).BILINEAR,
    )
    canvas.paste(resized, ((width - resized.width) // 2, title_height + (height - title_height - resized.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(20, 20, 20))
    return canvas


def _save_qc_panel(path, planned_info, window, context_maps, valid_mask, argmax_map, score_map, uncertainty, mucosa_mask):
    base = _overlay_base(planned_info, window, valid_mask.shape).convert("RGB")
    cells = [
        _panel_cell(base, "WSI Thumbnail"),
        _panel_cell(_hard_context_image(argmax_map, valid_mask), "Hard Context Map"),
        _panel_cell(_context_overlay(base, context_maps, valid_mask), "Context Overlay"),
        _panel_cell(_probability_image(score_map), "Mucosa Score Map"),
        _panel_cell(_probability_image(uncertainty), "Uncertainty Map"),
        _panel_cell(_mask_overlay(base, mucosa_mask), "Final Mask Overlay"),
    ]
    panel = Image.new("RGB", (cells[0].width * 3, cells[0].height * 2), (255, 255, 255))
    for index, cell in enumerate(cells):
        panel.paste(cell, ((index % 3) * cell.width, (index // 3) * cell.height))
    panel.save(path)


def _five_x_manifest(slide_id, window, planned_info, uncertainty, valid_mask, mucosa_mask, component_labels, config, rows):
    base_magnification = float(planned_info.get("base_magnification", 0.0) or 0.0)
    if base_magnification <= 0.0 and rows:
        base_magnification = _infer_base_magnification(rows[0], configured=config.base_magnification)
    if base_magnification <= 0.0:
        raise ValueError("Cannot construct 5x manifest without base magnification for {0}".format(slide_id))
    mpp_x = float(planned_info.get("mpp_x", 0.0) or 0.0)
    mpp_y = float(planned_info.get("mpp_y", 0.0) or 0.0)
    mpp_source = planned_info.get("mpp_source")
    if (mpp_x <= 0.0 or mpp_y <= 0.0) and float(config.mpp or 0.0) > 0.0:
        mpp_x = mpp_y = float(config.mpp)
        mpp_source = "explicit_override"
    if (mpp_x <= 0.0 or mpp_y <= 0.0) and base_magnification > 0.0:
        mpp_x = mpp_y = 10.0 / base_magnification
        mpp_source = "derived_from_base_magnification_contract"
    if mpp_x <= 0.0 or mpp_y <= 0.0:
        raise ValueError("Cannot construct 5x manifest without reliable MPP for {0}".format(slide_id))
    extent = max(1, int(round(config.five_x_patch_size * base_magnification / config.five_x_magnification)))
    start_x = int(math.floor(window[0] / float(extent))) * extent
    start_y = int(math.floor(window[1] / float(extent))) * extent
    output = []
    for y in range(start_y, window[3], extent):
        for x in range(start_x, window[2], extent):
            row_index = int(math.floor(y / float(extent)))
            col_index = int(math.floor(x / float(extent)))
            requested = [x, y, x + extent, y + extent]
            clipped = [max(window[0], x), max(window[1], y), min(window[2], x + extent), min(window[3], y + extent)]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            x1, y1, x2, y2 = _project_bbox(clipped, window, config.mask_downsample, valid_mask.shape)
            local_mask = np.asarray(mucosa_mask[y1:y2, x1:x2], dtype=np.uint8) > 0
            if local_mask.size == 0:
                continue
            coverage = float(local_mask.mean())
            if coverage <= 0.0:
                continue
            local_valid = valid_mask[y1:y2, x1:x2]
            uncertainty_values = uncertainty[y1:y2, x1:x2][local_valid]
            component_ids = sorted(int(value) for value in np.unique(component_labels[y1:y2, x1:x2][local_mask]) if int(value) > 0)
            patch_id = "{0}__5x__r{1:05d}_c{2:05d}".format(safe_name(slide_id), row_index, col_index)
            output.append(
                {
                    "slide_id": slide_id,
                    "patch_id": patch_id,
                    "grid_index": [row_index, col_index],
                    "level0_bbox": requested,
                    "clipped_level0_bbox": clipped,
                    "target_magnification": "{0:g}x".format(float(config.five_x_magnification)),
                    "mucosa_coverage": coverage,
                    "mean_uncertainty": float(uncertainty_values.mean()) if uncertainty_values.size else None,
                    "status": "retained",
                    "retention_reason": "positive_mucosa_mask_coverage",
                    "source_component_ids": component_ids,
                    "physical_provenance": {
                        "wsi_backend": planned_info.get("wsi_backend"),
                        "base_magnification": base_magnification,
                        "base_magnification_source": planned_info.get("base_magnification_source"),
                        "mpp_x": mpp_x,
                        "mpp_y": mpp_y,
                        "mpp_source": mpp_source,
                        "requested_magnification": float(config.five_x_magnification),
                        "fov_microns": [float(extent) * mpp_x, float(extent) * mpp_y],
                        "output_pixel_dimensions": [
                            int(config.five_x_patch_size),
                            int(config.five_x_patch_size),
                        ],
                    },
                }
            )
    return output


def _process_slide(slide_id, rows, planned_info, output_dir, config):
    window = _slide_window(slide_id, rows, {slide_id: planned_info})
    raw_maps, context_maps, uncertainty, valid_mask, count_map = _build_slide_arrays(rows, window, config.mask_downsample)
    normal_index = TISSUE_CONTEXT_LABELS.index("normal_epi_context")
    abnormal_index = TISSUE_CONTEXT_LABELS.index("abnormal_epithelial_candidate")
    score_map = context_maps[normal_index] + context_maps[abnormal_index]
    tile_extents = [parse_level0_bbox(row.get("level0_bbox", [])) for row in rows]
    tile_extents = [box[2] - box[0] for box in tile_extents if len(box) == 4]
    tile_extent = int(np.median(tile_extents)) if tile_extents else int(round(config.classifier_patch_size * 2.0))
    patch_size = max(3, int(round(tile_extent / float(config.mask_downsample))))
    final_mask, debug = generate_anatomical_mask(
        score_map * count_map,
        count_map,
        threshold=float(config.mucosa_threshold),
        patch_size=patch_size,
        stride=patch_size,
        region_constraint_mask=valid_mask,
        return_debug=True,
    )
    fallback_used = False
    if not np.any(np.asarray(final_mask) > 0) and np.any(np.asarray(debug["binary_mask"]) > 0):
        final_mask = np.asarray(debug["binary_mask"], dtype=np.uint8)
        fallback_used = True
    argmax_map = np.full(valid_mask.shape, 255, dtype=np.uint8)
    argmax_map[valid_mask] = np.argmax(context_maps, axis=0)[valid_mask].astype(np.uint8)
    radius = max(1, int(math.ceil(tile_extent / float(config.mask_downsample))))
    mucosa_bool = np.asarray(final_mask, dtype=np.uint8) > 0
    near_ring = _binary_dilate(mucosa_bool, radius) & ~mucosa_bool
    statistics = _context_statistics(context_maps, valid_mask, argmax_map, final_mask, near_ring, config)
    component_labels, component_count = _connected_component_labels(mucosa_bool)

    slide_dir = ensure_dir(Path(output_dir) / "slides" / safe_name(slide_id))
    np.savez_compressed(
        str(Path(slide_dir) / "maps.npz"),
        raw_tissue_probabilities=raw_maps.astype(np.float16),
        task_context_scores=context_maps.astype(np.float16),
        uncertainty=uncertainty.astype(np.float16),
        mucosa_score=score_map.astype(np.float16),
        mucosa_candidate_mask=mucosa_bool.astype(np.uint8),
        valid_mask=valid_mask.astype(np.uint8),
    )
    if config.write_overlays:
        _save_qc_panel(
            Path(slide_dir) / "qc_panel.png",
            planned_info,
            window,
            context_maps,
            valid_mask,
            argmax_map,
            score_map,
            uncertainty,
            final_mask,
        )

    five_x_rows = _five_x_manifest(
        slide_id, window, planned_info, uncertainty, valid_mask, final_mask, component_labels, config, rows
    )
    tile_mask_coverage = {}
    for row in rows:
        bbox = parse_level0_bbox(row.get("clipped_level0_bbox", row.get("level0_bbox", [])))
        x1, y1, x2, y2 = _project_bbox(bbox, window, config.mask_downsample, valid_mask.shape)
        local = mucosa_bool[y1:y2, x1:x2]
        tile_mask_coverage[row["patch_uid"]] = float(local.mean()) if local.size else 0.0
    retained_tile_count = sum(value > 0.0 for value in tile_mask_coverage.values())
    grid_rows = [int((row.get("patch_id") or [0, 0])[0]) for row in rows]
    grid_cols = [int((row.get("patch_id") or [0, 0])[1]) for row in rows]
    payload = {
        "slide_id": slide_id,
        "wsi_path": str(planned_info.get("wsi_path", "") or ""),
        "grid_shape": [max(grid_rows) + 1 if grid_rows else 0, max(grid_cols) + 1 if grid_cols else 0],
        "map_shape": list(valid_mask.shape),
        "level0_window": window,
        "mask_downsample": float(config.mask_downsample),
        "base_magnification": float(planned_info.get("base_magnification", 0.0) or _infer_base_magnification(rows[0], config.base_magnification)),
        "valid_tile_count": len(rows),
        "retained_tile_count": retained_tile_count,
        "mucosa_retention_ratio": float(retained_tile_count) / float(max(1, len(rows))),
        "mask_area_fraction_of_valid_tissue": _fraction(mucosa_bool, valid_mask),
        "mucosa_component_count": component_count,
        "five_x_candidate_count": len(five_x_rows),
        "postprocess_fallback_used": fallback_used,
        "context_statistics": statistics,
        "maps": str(Path(slide_dir) / "maps.npz"),
        "qc_panel": str(Path(slide_dir) / "qc_panel.png") if config.write_overlays else "",
    }
    return payload, five_x_rows, tile_mask_coverage


def _existing_predictions_from_tile_index(path, source_model):
    path = Path(path)
    if not path.exists():
        return []
    output = []
    for row in read_jsonl(path):
        probabilities = row.get("raw_prediction", {})
        if not probabilities:
            continue
        output.append(
            {
                "patch_uid": row.get("tile_id"),
                "slide_id": row.get("slide_id"),
                "model": source_model,
                "raw_class": row.get("hard_raw_class", ""),
                "confidence": max(float(value) for value in probabilities.values()),
                "probabilities": probabilities,
            }
        )
    return output


def _tile_index_rows(records, tile_mask_coverage):
    output = []
    for row in records:
        coverage = float(tile_mask_coverage.get(row["patch_uid"], 0.0) or 0.0)
        item = {
            "slide_id": row["slide_id"],
            "tile_id": row["patch_uid"],
            "grid_index": row.get("patch_id"),
            "level0_bbox": row.get("level0_bbox"),
            "clipped_level0_bbox": row.get("clipped_level0_bbox"),
            "tissue_coverage": row["tissue_coverage_ratio"],
            "raw_prediction": row["raw_probabilities"],
            "task_context": row["tissue_context"],
            "hard_context": row["tissue_context_label"],
            "uncertainty": row["uncertainty"],
            "mucosa_score": row["mucosa_score"],
            "mucosa_mask_coverage": coverage,
            "included_in_mucosa_mask": bool(coverage > 0.0),
        }
        probability_validation = row.get("provenance", {}).get("probability_validation", {})
        if probability_validation.get("renormalized"):
            item["input_probability_sum"] = probability_validation.get("input_probability_sum")
            item["probabilities_renormalized"] = True
        output.append(item)
    return output


def _clean_compact_output(output_dir, keep_tile_index=False):
    obsolete_files = (
        "manifest.json",
        "five_x_patch_manifest.jsonl",
        "run_manifest.json",
        "config_snapshot.json",
        "summary.json",
        "tile_manifest.jsonl",
        "raw_pathprism_predictions.jsonl",
        "tissue_context_evidence.jsonl",
    )
    for name in obsolete_files:
        path = Path(output_dir) / name
        if path.exists():
            path.unlink()
    if not keep_tile_index:
        path = Path(output_dir) / "tile_index.jsonl"
        if path.exists():
            path.unlink()
    for name in ("preprocess", "crops"):
        path = Path(output_dir) / name
        if path.exists():
            shutil.rmtree(str(path))
    slides_dir = Path(output_dir) / "slides"
    if slides_dir.exists():
        shutil.rmtree(str(slides_dir))


def run_mucosa_extractor(wsi_paths=None, artifact_dir=None, output_dir=None, config=None):
    config = config or MucosaExtractorConfig()
    if bool(wsi_paths) == bool(artifact_dir):
        raise ValueError("Provide exactly one of wsi_paths or artifact_dir")
    if output_dir is None:
        if artifact_dir:
            output_dir = Path(artifact_dir) / "mucosa_extractor"
        else:
            raise ValueError("output_dir is required for WSI input")
    output_dir = ensure_dir(Path(output_dir).resolve())
    tile_index_path = Path(output_dir) / "tile_index.jsonl"
    existing_predictions = _existing_predictions_from_tile_index(tile_index_path, config.source_model) if config.resume else []
    _clean_compact_output(output_dir, keep_tile_index=bool(config.resume))
    errors_path = Path(output_dir) / "errors.jsonl"
    if errors_path.exists():
        errors_path.unlink()

    normalized_wsi_paths = []
    if artifact_dir:
        manifest_rows, prediction_rows, planned_by_slide = _load_artifact_inputs(artifact_dir, config)
        input_mode = "artifact"
        request_errors = []
        wsi_errors = []
    else:
        if isinstance(wsi_paths, (str, Path)):
            wsi_paths = [wsi_paths]
        normalized_wsi_paths = [str(Path(path).resolve()) for path in (wsi_paths or [])]
        manifest_rows = []
        planned_by_slide = {}
        wsi_errors = []
        with tempfile.TemporaryDirectory(prefix="mucosa_extractor_") as work_dir:
            for path in normalized_wsi_paths:
                try:
                    rows, planned = _build_tiles_from_wsi(path, work_dir, config)
                except Exception as exc:
                    wsi_errors.append(
                        {
                            "slide_id": Path(path).stem,
                            "stage": "wsi_reading",
                            "error_type": exc.__class__.__name__,
                            "message": str(exc),
                        }
                    )
                    continue
                manifest_rows.extend(rows)
                planned_by_slide[planned["slide_id"]] = planned
            if wsi_errors:
                _append_jsonl(errors_path, wsi_errors)
            if not manifest_rows:
                raise RuntimeError("No readable WSI produced eligible tissue tiles; see errors.jsonl")
            done = set()
            for row in existing_predictions:
                try:
                    validate_crc100k_probabilities(row.get("probabilities", {}), tolerance=config.probability_sum_tolerance)
                except Exception:
                    continue
                done.add(row.get("patch_uid"))
            new_predictions, request_errors = _post_pathprism_predictions(manifest_rows, output_dir, config, done)
            prediction_rows = existing_predictions + new_predictions
        input_mode = "raw_wsi"

    records, _normalized_predictions, validation_errors = _build_context_records(manifest_rows, prediction_rows, config)
    if validation_errors:
        _append_jsonl(errors_path, validation_errors)

    records_by_slide = defaultdict(list)
    for row in records:
        records_by_slide[str(row["slide_id"])].append(row)
    slide_payloads = []
    all_five_x = []
    tile_mask_coverage = {}
    for slide_id, rows in sorted(records_by_slide.items()):
        payload, five_x_rows, slide_tile_coverage = _process_slide(
            slide_id, rows, planned_by_slide.get(slide_id, {}), output_dir, config
        )
        slide_payloads.append(payload)
        all_five_x.extend(five_x_rows)
        tile_mask_coverage.update(slide_tile_coverage)
    tile_rows = _tile_index_rows(records, tile_mask_coverage)
    _write_jsonl(tile_index_path, tile_rows)
    _write_jsonl(Path(output_dir) / "five_x_patch_manifest.jsonl", all_five_x)

    error_count = len(validation_errors) + len(request_errors) + len(wsi_errors)
    status = "complete" if error_count == 0 else "complete_with_tile_errors"
    manifest = {
        "schema_version": "mucosa_extractor_v1_compact",
        "run_id": "mucosa_extractor_{0}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        "status": status,
        "input": {
            "mode": input_mode,
            "artifact_dir": str(Path(artifact_dir).resolve()) if artifact_dir else "",
            "wsi_paths": normalized_wsi_paths,
        },
        "model": {
            "encoder": "UNI",
            "classifier": "PathPrism_CRC100K",
            "source_model": config.source_model,
            "http_endpoint": config.pathprism_url if input_mode == "raw_wsi" else "",
            "model_input_size_px": [224, 224],
        },
        "config": {
            "source_magnification": "{0:g}x".format(float(config.classifier_magnification)),
            "source_tile_size_px": [int(config.classifier_patch_size), int(config.classifier_patch_size)],
            "source_stride_px": [int(config.classifier_patch_size), int(config.classifier_patch_size)],
            "mask_downsample": float(config.mask_downsample),
            "min_tissue_coverage": float(config.min_tissue_coverage),
            "mucosa_threshold": float(config.mucosa_threshold),
            "mucosa_formula": "normal_epi_context + abnormal_epithelial_candidate",
            "mapping_version": "context_mapping_v1.0",
            "presence_calibration_status": "unvalidated",
            "presence_area_threshold": float(config.presence_area_threshold),
            "presence_peak_threshold": float(config.presence_peak_threshold),
            "five_x_patch_size_px": [int(config.five_x_patch_size), int(config.five_x_patch_size)],
            "five_x_grid_origin_level0": [0, 0],
        },
        "channel_order": {
            "raw_tissue_probabilities": list(CRC100K_LABELS),
            "task_context_scores": list(TISSUE_CONTEXT_LABELS),
        },
        "context_mapping": {label: list(sources) for label, sources in CRC100K_CONTEXT_MAP.items()},
        "counts": {
            "slides": len(slide_payloads),
            "valid_tiles": len(tile_rows),
            "retained_tiles": sum(bool(row["included_in_mucosa_mask"]) for row in tile_rows),
            "five_x_patches": len(all_five_x),
            "errors": error_count,
        },
        "files": {
            "tile_index": str(tile_index_path),
            "five_x_patch_manifest": str(Path(output_dir) / "five_x_patch_manifest.jsonl"),
            "errors": str(errors_path) if errors_path.exists() else "",
        },
        "slides": slide_payloads,
    }
    write_json(Path(output_dir) / "manifest.json", manifest)
    return manifest
