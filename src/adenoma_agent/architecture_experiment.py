import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ARCHITECTURE_LABELS = ("serrated", "tubular", "villous")
CONTEXT_LABELS = (
    "normal_mucosa_present",
    "reactive_inflammatory_present",
    "other_pattern_present",
)
ALL_OUTPUT_LABELS = ("evaluable",) + ARCHITECTURE_LABELS + CONTEXT_LABELS
ANNOTATION_STATES = ("present", "absent", "uncertain", "unlabeled")


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return value.strip("_") or "unnamed"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _bbox_key(row):
    bbox = row.get("level0_bbox", [])
    return int(bbox[0]), int(bbox[1])


def _child_positions(parent_bbox, child_size=512):
    x1, y1, _x2, _y2 = [int(v) for v in parent_bbox]
    return [
        (x1, y1, 0, 0),
        (x1 + child_size, y1, 0, 1),
        (x1, y1 + child_size, 1, 0),
        (x1 + child_size, y1 + child_size, 1, 1),
    ]


def validate_parent_children(parent_bbox, children, child_size=512):
    if len(children) != 4:
        return False, "expected_four_children"
    x1, y1, x2, y2 = [int(v) for v in parent_bbox]
    if x2 - x1 != 2 * child_size or y2 - y1 != 2 * child_size:
        return False, "unexpected_parent_extent"
    expected = set((x, y) for x, y, _r, _c in _child_positions(parent_bbox, child_size=child_size))
    observed = set(_bbox_key(child) for child in children)
    if observed != expected:
        return False, "child_grid_does_not_cover_parent"
    for child in children:
        cx1, cy1, cx2, cy2 = [int(v) for v in child.get("level0_bbox", [])]
        if cx2 - cx1 != child_size or cy2 - cy1 != child_size:
            return False, "unexpected_child_extent"
    return True, ""


def _mask_coverage(mask, level0_bbox, level0_window, downsample):
    x1, y1, x2, y2 = [float(v) for v in level0_bbox]
    ox, oy, _wx2, _wy2 = [float(v) for v in level0_window]
    height, width = mask.shape
    cx1 = max(0, min(width, int(math.floor((x1 - ox) / downsample))))
    cy1 = max(0, min(height, int(math.floor((y1 - oy) / downsample))))
    cx2 = max(0, min(width, int(math.ceil((x2 - ox) / downsample))))
    cy2 = max(0, min(height, int(math.ceil((y2 - oy) / downsample))))
    if cx2 <= cx1 or cy2 <= cy1:
        return 0.0
    return float(np.asarray(mask[cy1:cy2, cx1:cx2], dtype=bool).mean())


def _context_children(parent_bbox, child_by_origin, child_size=512):
    x1, y1, _x2, _y2 = [int(v) for v in parent_bbox]
    rows = []
    for relative_row in range(4):
        for relative_col in range(4):
            x = x1 + (relative_col - 1) * child_size
            y = y1 + (relative_row - 1) * child_size
            child = child_by_origin.get((x, y))
            if child is None:
                return []
            row = dict(child)
            row["relative_row"] = relative_row
            row["relative_col"] = relative_col
            row["is_target_region"] = relative_row in (1, 2) and relative_col in (1, 2)
            rows.append(row)
    return rows


def _annotation_template(roi_id, slide_id):
    return {
        "roi_id": roi_id,
        "slide_id": slide_id,
        "quality": {"state": "unlabeled", "limitations": []},
        "architecture": {label: "unlabeled" for label in ARCHITECTURE_LABELS},
        "context": {label: "unlabeled" for label in CONTEXT_LABELS},
        "annotation_confidence": None,
        "annotator_id": "",
        "notes": "",
        "label_source": "human_roi_annotation",
        "synthetic_smoke_only": False,
    }


def validate_annotation(annotation):
    errors = []
    quality_state = str(annotation.get("quality", {}).get("state", ""))
    if quality_state not in ("evaluable", "non_evaluable", "unlabeled"):
        errors.append("invalid_quality_state")
    for section, labels in (("architecture", ARCHITECTURE_LABELS), ("context", CONTEXT_LABELS)):
        values = annotation.get(section, {})
        for label in labels:
            if str(values.get(label, "")) not in ANNOTATION_STATES:
                errors.append("invalid_{0}_{1}".format(section, label))
    return errors


def annotation_to_targets(annotation):
    errors = validate_annotation(annotation)
    if errors:
        raise ValueError("Invalid annotation for {0}: {1}".format(annotation.get("roi_id", ""), errors))
    targets = np.zeros(len(ALL_OUTPUT_LABELS), dtype=np.float32)
    masks = np.zeros(len(ALL_OUTPUT_LABELS), dtype=np.float32)
    quality_state = annotation.get("quality", {}).get("state")
    if quality_state in ("evaluable", "non_evaluable"):
        targets[0] = 1.0 if quality_state == "evaluable" else 0.0
        masks[0] = 1.0
    if quality_state != "evaluable":
        return targets, masks
    offset = 1
    for section, labels in (("architecture", ARCHITECTURE_LABELS), ("context", CONTEXT_LABELS)):
        values = annotation.get(section, {})
        for label in labels:
            state = values.get(label)
            if state in ("present", "absent"):
                targets[offset] = 1.0 if state == "present" else 0.0
                masks[offset] = 1.0
            offset += 1
    return targets, masks


def _coverage_stratum(value):
    if value >= 0.60:
        return "high"
    if value >= 0.30:
        return "borderline"
    return "low"


def _sample_slide_rows(rows, max_per_slide, seed):
    if not max_per_slide or len(rows) <= max_per_slide:
        return sorted(rows, key=lambda row: row["roi_id"])
    rng = random.Random(int(seed))
    by_stratum = defaultdict(list)
    for row in rows:
        by_stratum[_coverage_stratum(float(row["mucosa_coverage"]))].append(row)
    targets = {"high": 6, "borderline": 2, "low": 2}
    selected = []
    remaining = []
    for stratum in ("high", "borderline", "low"):
        candidates = list(by_stratum.get(stratum, []))
        rng.shuffle(candidates)
        take = min(len(candidates), targets[stratum], max_per_slide - len(selected))
        selected.extend(candidates[:take])
        remaining.extend(candidates[take:])
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, max_per_slide - len(selected))])
    return sorted(selected, key=lambda row: row["roi_id"])


def assign_slide_splits(rows, seed=17):
    slide_ids = sorted(set(row["slide_id"] for row in rows))
    rng = random.Random(int(seed))
    rng.shuffle(slide_ids)
    if len(slide_ids) >= 3:
        n_test = max(1, int(round(len(slide_ids) * 0.2)))
        n_val = max(1, int(round(len(slide_ids) * 0.2)))
    else:
        n_test = 0
        n_val = 0
    test_ids = set(slide_ids[:n_test])
    val_ids = set(slide_ids[n_test : n_test + n_val])
    split_by_slide = {}
    for slide_id in slide_ids:
        split_by_slide[slide_id] = "test" if slide_id in test_ids else ("val" if slide_id in val_ids else "train")
    output = []
    for row in rows:
        item = dict(row)
        item["split"] = split_by_slide[item["slide_id"]]
        output.append(item)
    return output, split_by_slide


def build_architecture_roi_manifest(artifact_dir, output_dir=None, max_per_slide=0, seed=17, include_context=True):
    artifact_dir = Path(artifact_dir).resolve()
    output_dir = Path(output_dir or artifact_dir / "architecture_experiment").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_jsonl(artifact_dir / "manifest.jsonl")
    rows_by_slide_crop = defaultdict(lambda: defaultdict(list))
    for row in manifest_rows:
        rows_by_slide_crop[str(row.get("slide_id", ""))][str(row.get("crop_id", ""))].append(row)
    extractor_manifest_path = artifact_dir / "mucosa_extractor" / "manifest.json"
    tissue_context_summary_path = artifact_dir / "tissue_context_maps" / "tissue_context_summary.json"
    legacy_mask_summary_path = artifact_dir / "mucosa_masks" / "mucosa_mask_summary.json"
    if extractor_manifest_path.exists():
        mucosa_summary = read_json(extractor_manifest_path)
        mask_json_by_slide = {row["slide_id"]: row for row in mucosa_summary.get("slides", [])}
        mask_source = "mucosa_extractor_v1"
    elif tissue_context_summary_path.exists():
        mucosa_summary = read_json(tissue_context_summary_path)
        mask_json_by_slide = {
            row["slide_id"]: row["tissue_context_map_json"] for row in mucosa_summary.get("slides", [])
        }
        mask_source = "six_channel_tissue_context"
    elif legacy_mask_summary_path.exists():
        mucosa_summary = read_json(legacy_mask_summary_path)
        mask_json_by_slide = {row["slide_id"]: row["mucosa_mask_json"] for row in mucosa_summary.get("slides", [])}
        mask_source = "legacy_five_class_mask"
    else:
        raise FileNotFoundError(
            "Missing mucosa evidence: expected mucosa_extractor/manifest.json, "
            "tissue_context_maps/tissue_context_summary.json, or mucosa_masks/mucosa_mask_summary.json"
        )
    accepted = []
    rejected = []
    for slide_id, crops in sorted(rows_by_slide_crop.items()):
        mask_descriptor = mask_json_by_slide.get(slide_id)
        if not mask_descriptor:
            continue
        if mask_source == "mucosa_extractor_v1":
            maps_path = mask_descriptor["maps"]
            maps = np.load(maps_path)
            mask = np.asarray(maps["mucosa_candidate_mask"], dtype=np.uint8) > 0
            level0_window = mask_descriptor["level0_window"]
            downsample = float(mask_descriptor["mask_downsample"])
            mask_json_path = maps_path
            mask_pipeline = "mucosa_extractor_v1_compact"
            mask_model_policy = {}
        else:
            mask_json_path = mask_descriptor
            mask_payload = read_json(mask_json_path)
            mask_artifacts = mask_payload["artifacts"]
            mask_path = mask_artifacts.get("high_recall_mucosa_search_mask") or mask_artifacts.get("mucosa_candidate_mask")
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            level0_window = mask_payload["processing"]["level0_window"]
            downsample = float(mask_payload["processing"]["mask_downsample"])
            mask_pipeline = mask_payload.get("pipeline", "")
            mask_model_policy = mask_payload.get("model_policy", {})
        child_by_origin = {_bbox_key(row): row for row in crops.get("20x_512", [])}
        slide_rows = []
        for parent in crops.get("10x_1024", []):
            children = []
            for x, y, relative_row, relative_col in _child_positions(parent.get("level0_bbox", [])):
                child = child_by_origin.get((x, y))
                if child is None:
                    continue
                item = dict(child)
                item["relative_row"] = relative_row
                item["relative_col"] = relative_col
                children.append(item)
            valid, reason = validate_parent_children(parent.get("level0_bbox", []), children)
            if not valid:
                rejected.append({"patch_uid": parent.get("patch_uid"), "slide_id": slide_id, "reason": reason})
                continue
            children = sorted(children, key=lambda row: (row["relative_row"], row["relative_col"]))
            roi_id = "{0}__arch__r{1:04d}_c{2:04d}".format(
                safe_name(slide_id), int(parent.get("row_id", 0)), int(parent.get("col_id", 0))
            )
            coverage = _mask_coverage(mask, parent["level0_bbox"], level0_window, downsample)
            row = {
                "roi_id": roi_id,
                "slide_id": slide_id,
                "level0_bbox": parent["level0_bbox"],
                "parent_patch_uid": parent["patch_uid"],
                "parent_image_path": parent.get("absolute_image_path") or str(artifact_dir / parent.get("image_path", "")),
                "parent_crop_id": parent.get("crop_id"),
                "mucosa_coverage": round(coverage, 6),
                "coverage_stratum": _coverage_stratum(coverage),
                "mucosa_mask_json": str(mask_json_path),
                "children": children,
                "context_4x4_children": _context_children(parent["level0_bbox"], child_by_origin) if include_context else [],
                "provenance": {
                    "artifact_dir": str(artifact_dir),
                    "mask_pipeline": mask_pipeline,
                    "mask_model_policy": mask_model_policy,
                    "mask_source": mask_source,
                },
            }
            slide_rows.append(row)
        accepted.extend(_sample_slide_rows(slide_rows, int(max_per_slide), int(seed) + len(accepted)))
    accepted, split_by_slide = assign_slide_splits(accepted, seed=seed)
    annotations = [_annotation_template(row["roi_id"], row["slide_id"]) for row in accepted]
    write_jsonl(output_dir / "roi_manifest.jsonl", accepted)
    write_jsonl(output_dir / "annotation_template.jsonl", annotations)
    write_jsonl(output_dir / "rejected_rois.jsonl", rejected)
    write_json(output_dir / "splits.json", {"seed": int(seed), "split_by_slide": split_by_slide})
    summary = {
        "pipeline": "mucosal_architecture_roi_manifest_v1",
        "artifact_dir": str(artifact_dir),
        "output_dir": str(output_dir),
        "counts": {
            "slides": len(set(row["slide_id"] for row in accepted)),
            "rois": len(accepted),
            "rejected_rois": len(rejected),
            "context_4x4_complete": sum(bool(row["context_4x4_children"]) for row in accepted),
        },
        "coverage_strata": {
            key: sum(row["coverage_stratum"] == key for row in accepted) for key in ("high", "borderline", "low")
        },
        "mask_source": mask_source,
        "split_counts": {key: sum(row["split"] == key for row in accepted) for key in ("train", "val", "test")},
    }
    write_json(output_dir / "prepare_summary.json", summary)
    return summary


def build_synthetic_smoke_annotations(roi_rows):
    annotations = []
    for index, row in enumerate(sorted(roi_rows, key=lambda item: item["roi_id"])):
        digest = hashlib.sha256(row["roi_id"].encode("utf-8")).digest()
        evaluable = index % 7 != 0
        annotation = _annotation_template(row["roi_id"], row["slide_id"])
        annotation["quality"]["state"] = "evaluable" if evaluable else "non_evaluable"
        if evaluable:
            values = list(ARCHITECTURE_LABELS) + list(CONTEXT_LABELS)
            for label_index, label in enumerate(values):
                state = "present" if digest[label_index] % 3 == 0 else "absent"
                section = "architecture" if label in ARCHITECTURE_LABELS else "context"
                annotation[section][label] = state
        else:
            for label in ARCHITECTURE_LABELS:
                annotation["architecture"][label] = "uncertain"
            for label in CONTEXT_LABELS:
                annotation["context"][label] = "uncertain"
            annotation["quality"]["limitations"] = ["synthetic_smoke_quality_negative"]
        annotation["annotation_confidence"] = 1.0
        annotation["annotator_id"] = "synthetic_smoke_generator"
        annotation["label_source"] = "synthetic_smoke_only"
        annotation["synthetic_smoke_only"] = True
        annotations.append(annotation)
    return annotations
