#!/usr/bin/env python3
import argparse
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adenoma_agent.agents.audit import AuditAgent
from adenoma_agent.agents.navigate import NavigateAgent
from adenoma_agent.agents.observe_reason import ObserveReasonAgent
from adenoma_agent.multimodal import HeuristicStageBackend
from adenoma_agent.schemas import CaseSpec, GlobalReviewRecord, TraceCluster
from adenoma_agent.slide_memory import SlideMemoryTracker
from adenoma_agent.utils import ensure_dir, write_json


class _Logger(object):
    def log(self, *args, **kwargs):
        return None


class _HeuristicBackendChain(object):
    def __init__(self):
        self.backend = HeuristicStageBackend()

    def invoke(self, stage, chain_names, request):
        response = self.backend.invoke({"stage": stage, **request}, _minimal_bundle())
        return {
            "backend": response["backend"],
            "attempts": [{"backend": response["backend"], "status": "ok", "latency_ms": 0}],
            "output": response["output"],
            "raw_texts": [],
        }


class _SyntheticCropper(object):
    def export_crops(self, case_spec, bundle_steps, output_dir):
        output_dir = ensure_dir(output_dir)
        crops = []
        for step in bundle_steps:
            role = step.metadata.get("image_role", "detail")
            image_path = output_dir / "{0}_{1}.png".format(step.step_id, role)
            color = _color_for_review_goal(step.review_goal)
            Image.new("RGB", (96, 96), color=color).save(image_path)
            crops.append({"image_path": str(image_path), "m": step.m, "metadata": {"image_role": role}})
        return {
            "manifest": {"crops": crops},
            "result": {"returncode": 0, "stdout": "", "stderr": "", "latency_ms": 0},
        }


def _minimal_bundle(cache_root=None):
    cache_root = Path(cache_root or tempfile.gettempdir()) / "slide_memory_experiment_cache"
    return {
        "runtime": {
            "data": {"serrated_labels": []},
            "trace": {
                "labels": [
                    "epithelial_neoplasia_suspicious",
                    "mucus_rich_or_pale_context",
                    "uncertain_reviewable_mucosa",
                    "inflammatory_or_stromal_context",
                    "reviewable_normal_mucosa",
                    "background_or_artifact",
                    "ssl_suspicious_mucosa",
                    "conventional_adenoma_like",
                    "inflammatory_polyp_like",
                    "normal_mucosa",
                    "background_artifact_stroma",
                ],
            },
            "navigate": {"backend_chain": ["heuristic"], "overlap_threshold": 0.30},
            "observe": {
                "backend_chain": ["heuristic"],
                "patho_r1_question": "Run offline observation and reasoning for the slide-memory experiment.",
                "serrated_criteria": ["serrated_lesion_context", "mucus_rich_context"],
                "abnormal_crypt_criteria": ["basal_dilatation", "crypt_branching"],
                "ssl_criteria": ["basal_dilatation", "crypt_branching"],
                "hp_criteria": ["surface_limited_serration", "straight_crypt_bases"],
                "tsa_criteria": ["ectopic_crypt_foci", "slit_like_serration"],
                "tsa_cytological_atypia_criteria": ["pencillate_nuclei"],
                "conventional_adenoma_criteria": ["tubular_architecture", "crowded_adenomatous_glands"],
                "conventional_architecture_criteria": ["tubular_architecture", "crowded_adenomatous_glands"],
                "inflammatory_criteria": ["mixed_inflammation", "reactive_regenerative_change"],
                "dysplasia_criteria": ["nuclear_enlargement_stratification", "hyperchromasia"],
            },
            "chief_llm": {
                "server_url": "offline",
                "timeout_seconds": 1,
                "model_name": "offline_chief",
                "require_real_chief": False,
            },
            "cache": {"description_cache_root": str(cache_root / "observe")},
        },
        "budget": {
            "max_navigation_steps": 8,
            "max_trace_candidates": 4,
            "max_intra_cell_zoom_targets": 3,
            "magnification_to_region_size": {"2.5": 256, "5.0": 128, "10.0": 64},
        },
    }


def _color_for_review_goal(review_goal):
    if "ssl" in review_goal or "serrated" in review_goal or "abnormal_crypt" in review_goal:
        return (210, 184, 198)
    if "conventional" in review_goal:
        return (120, 82, 118)
    if "inflammatory" in review_goal or "reactive" in review_goal:
        return (190, 96, 86)
    return (190, 156, 166)


def _write_synthetic_grid(output_dir):
    output_dir = ensure_dir(output_dir)
    image_path = output_dir / "synthetic_slide_memory_grid.jpg"
    image = Image.new("RGB", (192, 96), color=(235, 220, 225))
    image.paste((190, 160, 170), (0, 0, 48, 96))
    image.paste((188, 155, 168), (48, 0, 96, 96))
    image.paste((80, 55, 95), (96, 0, 144, 96))
    image.paste((78, 54, 92), (144, 0, 192, 96))
    image.save(image_path)
    cells = []
    for col in range(4):
        is_abnormal = col >= 2
        cells.append(
            {
                "row_id": 0,
                "col_id": col,
                "patch_id": [0, col],
                "is_selected": True,
                "thumbnail_top_left_x": col * 48,
                "thumbnail_top_left_y": 0,
                "thumbnail_width": 48,
                "thumbnail_height": 96,
                "level0_top_left_x": col * 512,
                "level0_top_left_y": 0,
                "level0_width": 512,
                "level0_height": 512,
                "tissue_coverage_ratio": 0.9,
                "synthetic_label": "abnormal_mucosa" if is_abnormal else "normal_mucosa",
                "synthetic_abnormal": bool(is_abnormal),
            }
        )
    grid_path = image_path.with_suffix(".json")
    write_json(
        grid_path,
        {
            "thumbnail_mode": "tissue_grid32x_svs",
            "grid_rows": 1,
            "grid_cols": 4,
            "n_selected_cells": 4,
            "level0_crop_bbox": [0, 0, 2048, 512],
            "grid_cells": cells,
        },
    )
    return image_path, grid_path


def _synthetic_features(torch, feature_dim):
    base = torch.zeros(feature_dim)
    return [
        base.clone(),
        base.clone() + 0.01,
        torch.ones(feature_dim) * 4.0,
        torch.ones(feature_dim) * 4.2,
    ]


def _image_cell_feature(torch, image, cell, feature_dim):
    crop = image.crop(
        (
            int(cell.get("thumbnail_top_left_x", 0)),
            int(cell.get("thumbnail_top_left_y", 0)),
            int(cell.get("thumbnail_top_left_x", 0)) + int(cell.get("thumbnail_width", 1)),
            int(cell.get("thumbnail_top_left_y", 0)) + int(cell.get("thumbnail_height", 1)),
        )
    ).convert("RGB")
    arr = torch.tensor(list(crop.getdata()), dtype=torch.float32).view(-1, 3) / 255.0
    if arr.numel() == 0:
        base = torch.zeros(16, dtype=torch.float32)
    else:
        mean = arr.mean(dim=0)
        std = arr.std(dim=0, unbiased=False)
        q25 = arr.quantile(0.25, dim=0)
        q75 = arr.quantile(0.75, dim=0)
        brightness = arr.mean(dim=1)
        saturation = arr.max(dim=1).values - arr.min(dim=1).values
        base = torch.cat(
            [
                mean,
                std,
                q25,
                q75,
                torch.tensor(
                    [
                        float(brightness.mean().item()),
                        float(brightness.std(unbiased=False).item()),
                        float(saturation.mean().item()),
                        float(saturation.std(unbiased=False).item()),
                    ],
                    dtype=torch.float32,
                ),
            ]
        )
    repeats = int((feature_dim + int(base.numel()) - 1) / int(base.numel()))
    return base.repeat(repeats)[:feature_dim].clone()


def _features_from_grid_image(torch, image_path, grid_payload, feature_dim):
    image = Image.open(image_path).convert("RGB")
    cells = [cell for cell in grid_payload.get("grid_cells", []) if cell.get("is_selected", True)]
    features = [_image_cell_feature(torch, image, cell, feature_dim) for cell in cells]
    image.close()
    return cells, features


def _tracker_only_detection(grid_payload, memory_records):
    abnormal_ids = {
        tuple(cell.get("patch_id", [cell.get("row_id"), cell.get("col_id")]))
        for cell in grid_payload.get("grid_cells", [])
        if cell.get("synthetic_abnormal")
    }
    triggered_ids = {tuple(item["patch_id"]) for item in memory_records if item["trigger_trace_agent"]}
    if not triggered_ids:
        top = sorted(memory_records, key=lambda item: item["raw_surprise"], reverse=True)[: max(1, len(abnormal_ids) or 3)]
        triggered_ids = {tuple(item["patch_id"]) for item in top}
    has_ground_truth = bool(abnormal_ids)
    true_positive_ids = sorted([list(item) for item in triggered_ids.intersection(abnormal_ids)])
    false_positive_ids = sorted([list(item) for item in triggered_ids.difference(abnormal_ids)]) if has_ground_truth else []
    false_negative_ids = sorted([list(item) for item in abnormal_ids.difference(triggered_ids)]) if has_ground_truth else []
    precision = float(len(true_positive_ids)) / float(max(1, len(triggered_ids))) if has_ground_truth else None
    recall = float(len(true_positive_ids)) / float(max(1, len(abnormal_ids))) if has_ground_truth else None
    ranked = sorted(memory_records, key=lambda item: item["raw_surprise"], reverse=True)
    top_k = ranked[: max(1, len(abnormal_ids))]
    top_k_ids = {tuple(item["patch_id"]) for item in top_k}
    top_k_recall = float(len(top_k_ids.intersection(abnormal_ids))) / float(max(1, len(abnormal_ids))) if has_ground_truth else None
    can_screen_abnormal_mucosa = bool(has_ground_truth and recall >= 0.5 and top_k_recall >= 0.5)
    enough_without_downstream = bool(has_ground_truth and precision >= 0.95 and recall >= 0.95)
    if not has_ground_truth:
        recommendation = (
            "real_wsi_no_patch_ground_truth: highlighted boxes are high-surprise candidate mucosa regions, "
            "not confirmed diagnostic abnormal mucosa. Use pathology annotation or downstream morphology review for validation."
        )
    elif enough_without_downstream:
        recommendation = (
            "tracker_only_may_be_sufficient_for_this_synthetic_case: sustained surprise localized all abnormal patches. "
            "Navigation/reasoning would still be needed for subtype and final pathology class."
        )
    elif can_screen_abnormal_mucosa:
        recommendation = (
            "tracker_can_screen_candidates_but_should_not_replace_downstream: surprise identifies abnormal mucosa candidates, "
            "but morphology subtype and diagnostic class remain unresolved."
        )
    else:
        recommendation = (
            "tracker_not_sufficient_even_for_screening_in_this_run: use navigation/reasoning and tune warm-up/threshold settings."
        )
    return {
        "synthetic_abnormal_patch_ids": sorted([list(item) for item in abnormal_ids]),
        "triggered_patch_ids": sorted([list(item) for item in triggered_ids]),
        "true_positive_patch_ids": true_positive_ids,
        "false_positive_patch_ids": false_positive_ids,
        "false_negative_patch_ids": false_negative_ids,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "top_k_recall": round(top_k_recall, 4) if top_k_recall is not None else None,
        "surprise_ranked_patch_ids": [item["patch_id"] for item in ranked],
        "annotation_patch_ids": sorted([list(item) for item in triggered_ids]),
        "has_patch_ground_truth": has_ground_truth,
        "can_screen_abnormal_mucosa": bool(can_screen_abnormal_mucosa),
        "enough_without_downstream": bool(enough_without_downstream),
        "downstream_recommendation": recommendation,
    }


def _record_by_patch_id(memory_records):
    lookup = {}
    for record in memory_records:
        lookup[tuple(record.get("patch_id", []))] = record
    return lookup


def _draw_label(draw, xy, text, fill, outline=(255, 255, 255)):
    x, y = xy
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        text_w = int(bbox[2] - bbox[0])
        text_h = int(bbox[3] - bbox[1])
    except Exception:
        text_w = max(8, len(text) * 6)
        text_h = 10
    pad = 2
    draw.rectangle([x, y, x + text_w + 2 * pad, y + text_h + 2 * pad], fill=fill)
    draw.text((x + pad, y + pad), text, fill=outline, font=font)


def _annotate_surprise_on_original(image_path, grid_payload, memory_records, output_dir, annotation_patch_ids=None):
    output_dir = ensure_dir(output_dir)
    image = Image.open(image_path).convert("RGB")
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    lookup = _record_by_patch_id(memory_records)
    annotation_patch_ids = {tuple(item) for item in (annotation_patch_ids or [])}
    for cell in grid_payload.get("grid_cells", []):
        patch_id = list(cell.get("patch_id", [cell.get("row_id"), cell.get("col_id")]))
        record = lookup.get(tuple(patch_id), {})
        bbox = _thumb_bbox_from_cell(cell)
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        triggered = bool(record.get("trigger_trace_agent")) or tuple(patch_id) in annotation_patch_ids
        abnormal = bool(cell.get("synthetic_abnormal"))
        if triggered and abnormal:
            color = (255, 32, 32)
            width = 4
            label_prefix = "HIGH SURPRISE ABNORMAL"
        elif triggered:
            color = (255, 32, 32)
            width = 4
            label_prefix = "HIGH SURPRISE CANDIDATE"
        elif abnormal:
            color = (80, 160, 255)
            width = 2
            label_prefix = "ABNORMAL GT"
        else:
            color = (150, 150, 150)
            width = 1
            label_prefix = "normal"
        for offset in range(width):
            draw.rectangle([x1 + offset, y1 + offset, x2 - offset - 1, y2 - offset - 1], outline=color)
        if triggered or abnormal:
            surprise = float(record.get("raw_surprise", 0.0) or 0.0)
            label = "{0} p={1} s={2:.3g}".format(label_prefix, patch_id, surprise)
            label_y = max(0, y1 + 3)
            _draw_label(draw, (x1 + 3, label_y), label, fill=color)
    annotated_path = output_dir / "slide_memory_high_surprise_overlay.png"
    annotated.save(annotated_path)
    image.close()
    return annotated_path


def _bbox_from_cell(cell):
    return {
        "x1": int(cell["level0_top_left_x"]),
        "y1": int(cell["level0_top_left_y"]),
        "x2": int(cell["level0_top_left_x"]) + int(cell["level0_width"]),
        "y2": int(cell["level0_top_left_y"]) + int(cell["level0_height"]),
    }


def _thumb_bbox_from_cell(cell):
    return {
        "x1": int(cell["thumbnail_top_left_x"]),
        "y1": int(cell["thumbnail_top_left_y"]),
        "x2": int(cell["thumbnail_top_left_x"]) + int(cell["thumbnail_width"]),
        "y2": int(cell["thumbnail_top_left_y"]) + int(cell["thumbnail_height"]),
    }


def _build_trace_result(case_spec, grid_payload, memory_records, output_dir):
    triggered = [item for item in memory_records if item["trigger_trace_agent"]]
    if not triggered:
        triggered = sorted(memory_records, key=lambda item: item["raw_surprise"], reverse=True)[:1]
    selected_patch_ids = {tuple(item["patch_id"]) for item in triggered}
    cells = [
        cell
        for cell in grid_payload.get("grid_cells", [])
        if tuple(cell.get("patch_id", [cell.get("row_id"), cell.get("col_id")])) in selected_patch_ids
    ]
    level0_boxes = [_bbox_from_cell(cell) for cell in cells]
    thumb_boxes = [_thumb_bbox_from_cell(cell) for cell in cells]
    x1 = min(box["x1"] for box in level0_boxes)
    y1 = min(box["y1"] for box in level0_boxes)
    x2 = max(box["x2"] for box in level0_boxes)
    y2 = max(box["y2"] for box in level0_boxes)
    tx1 = min(box["x1"] for box in thumb_boxes)
    ty1 = min(box["y1"] for box in thumb_boxes)
    tx2 = max(box["x2"] for box in thumb_boxes)
    ty2 = max(box["y2"] for box in thumb_boxes)
    patch_ids = [list(cell["patch_id"]) for cell in cells]
    cluster = TraceCluster(
        cluster_id="slide_memory_surprise_00",
        cluster_bbox_thumb={"x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2},
        cluster_bbox_level0={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        regions_thumb=thumb_boxes,
        regions_level0=level0_boxes,
        l="epithelial_neoplasia_suspicious",
        s=5,
        d=True,
        review_stage="morphology_resolution",
        crypt_disorder_risk=3,
        dysplasia_review_needed=False,
        desc="Sustained slide-memory surprise from CONCH feature trajectory.",
        evidence=["SlideMemoryTracker raw surprise exceeded adaptive threshold without rapid decay suppression."],
        metadata={
            "source": "slide_memory_tracker",
            "workflow_branch": "unresolved",
            "routing_hint": "needs_morphology_resolution",
            "candidate_branches": ["serrated", "conventional"],
            "slide_memory_records": triggered,
        },
        patch_ids_ordered=patch_ids,
        patches_thumb=[
            {**box, "patch_id": patch_id}
            for box, patch_id in zip(thumb_boxes, patch_ids)
        ],
        patches_level0=[
            {**box, "patch_id": patch_id}
            for box, patch_id in zip(level0_boxes, patch_ids)
        ],
        group_bbox_thumb={"x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2},
        group_bbox_level0={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    )
    trace_dir = ensure_dir(output_dir / "trace")
    trace_json = write_json(
        trace_dir / "slide_memory_trace_clusters.json",
        {"clusters": [cluster.to_dict()], "slide_memory_records": memory_records},
    )
    return {
        "selection": {
            "paths": {
                "thumbnail": Path(case_spec.grid_thumbnail_path),
                "boxes_json": trace_json,
                "visualization": Path(case_spec.grid_thumbnail_path),
            },
            "attempts": [],
            "cache_hit": False,
        },
        "payload": {
            "mode": "slide_memory_experiment",
            "thumbnail_meta": {"thumbnail_size": [192, 96], "slide_dimensions_level0": [2048, 512]},
        },
        "clusters": [cluster],
        "all_clusters": [cluster],
        "trace_clusters_json": trace_json,
        "trace_dir": trace_dir,
    }


def _patch_offline_chief(agent):
    def _offline_chief(case_spec, step, record, records, global_reviews, trace_result, pending_steps, observe_dir):
        return GlobalReviewRecord(
            review_id="global_review_offline_0000",
            source_step_id=record.step_id,
            decision="early_stop",
            continue_reason="",
            chief_confidence=0.9,
            resolved_branch_state={
                "serrated": "supported" if step.metadata.get("workflow_branch") == "serrated" else "unresolved",
                "abnormal_crypt": "supported" if "ssl" in record.stage_decision or "abnormal_crypt" in record.stage_decision else "unresolved",
                "conventional": "supported" if step.metadata.get("workflow_branch") == "conventional" else "unresolved",
                "dysplasia": "unresolved",
            },
            sufficient_evidence=[record.reasoning or record.observation],
            unresolved_questions=[],
            next_visual_target=None,
            metadata={"review_source": "chief_model", "mode": "offline_slide_memory_experiment"},
        )

    agent._call_chief_global_review = _offline_chief


def run_experiment(
    output_dir,
    feature_dim=768,
    with_downstream=False,
    image_path=None,
    grid_json=None,
    warm_up_steps=None,
    top_k=3,
):
    import torch

    output_dir = ensure_dir(output_dir)
    real_input_mode = bool(image_path and grid_json)
    if real_input_mode:
        image_path = Path(image_path)
        grid_path = Path(grid_json)
        grid_payload = json.loads(grid_path.read_text(encoding="utf-8"))
        input_dir = ensure_dir(output_dir / "input")
        local_image_path = input_dir / image_path.name
        local_grid_path = input_dir / grid_path.name
        if local_image_path.resolve() != image_path.resolve():
            local_image_path.write_bytes(image_path.read_bytes())
        if local_grid_path.resolve() != grid_path.resolve():
            local_grid_path.write_text(json.dumps(grid_payload, indent=2), encoding="utf-8")
        image_path = local_image_path
        grid_path = local_grid_path
    else:
        image_path, grid_path = _write_synthetic_grid(output_dir / "input")
        grid_payload = json.loads(Path(grid_path).read_text(encoding="utf-8"))
    case_spec = CaseSpec(
        case_id=str(grid_payload.get("slide_id") or "slide_memory_synthetic_case"),
        slide_path=str(image_path),
        task_type="slide_memory_navigation_reasoning_experiment",
        question="Classify the synthetic slide-memory-selected region through navigation and reasoning.",
        input_mode="grid_thumbnail",
        grid_thumbnail_path=str(image_path),
        grid_metadata_path=str(grid_path),
    )
    if real_input_mode:
        cells, features = _features_from_grid_image(torch, image_path, grid_payload, feature_dim)
    else:
        cells = grid_payload["grid_cells"]
        features = _synthetic_features(torch, feature_dim)

    tracker = SlideMemoryTracker(
        feature_dim=feature_dim,
        hidden_dim=min(feature_dim, 768),
        lr=0.01,
        warm_up_steps=int(warm_up_steps if warm_up_steps is not None else (min(5, max(1, len(features) // 4)) if real_input_mode else 2)),
        threshold_lambda=0.5,
        window_size=2,
        rapid_decay_slope=-0.05,
    )
    memory_records = []
    for cell, feature in zip(cells, features):
        result = tracker.process_patch(feature)
        memory_records.append(
            {
                "patch_id": list(cell.get("patch_id", [cell.get("row_id"), cell.get("col_id")])),
                "synthetic_label": cell.get("synthetic_label"),
                "synthetic_abnormal": bool(cell.get("synthetic_abnormal")),
                "raw_surprise": result["raw_surprise"],
                "calibrated_alarm_level": result["calibrated_alarm_level"],
                "trigger_trace_agent": result["trigger_trace_agent"],
            }
        )

    tracker_only_detection = _tracker_only_detection(grid_payload, memory_records)
    if real_input_mode:
        top = sorted(memory_records, key=lambda item: item["raw_surprise"], reverse=True)[: int(top_k)]
        tracker_only_detection["annotation_patch_ids"] = [item["patch_id"] for item in top]
        tracker_only_detection["triggered_patch_ids"] = [item["patch_id"] for item in top]
        tracker_only_detection["real_wsi_candidate_mode"] = True
    annotated_image_path = _annotate_surprise_on_original(
        image_path,
        grid_payload,
        memory_records,
        output_dir / "visualizations",
        annotation_patch_ids=tracker_only_detection.get("annotation_patch_ids"),
    )
    summary = {
        "experiment_question": (
            "Can SlideMemoryTracker identify abnormal mucosa candidates by surprise alone, "
            "and is downstream navigation/reasoning necessary?"
        ),
        "input_mode": "real_wsi_grid_image" if real_input_mode else "synthetic_grid",
        "original_image_path": str(image_path),
        "annotated_image_path": str(annotated_image_path),
        "tracker_threshold": tracker.surprise_threshold,
        "tracker_last_slope": tracker.last_slope,
        "slide_memory_records": memory_records,
        "tracker_only_detection": tracker_only_detection,
        "downstream_ran": False,
    }
    if with_downstream:
        bundle = _minimal_bundle(output_dir / "cache")
        backend_chain = _HeuristicBackendChain()
        trace_result = _build_trace_result(case_spec, grid_payload, memory_records, output_dir)
        logger = _Logger()

        navigate_agent = NavigateAgent(bundle, backend_chain)
        navigation_result = navigate_agent.run(case_spec, trace_result, output_dir, logger)

        observe_agent = ObserveReasonAgent(bundle, _SyntheticCropper(), backend_chain)
        _patch_offline_chief(observe_agent)
        observe_result = observe_agent.run(case_spec, trace_result, navigation_result, output_dir, logger)
        if observe_result.get("trajectory_steps"):
            navigation_result["steps"] = observe_result["trajectory_steps"]

        segmentation_artifact = {
            "thumbnail_path": str(image_path),
            "boxes_json": str(trace_result["trace_clusters_json"]),
            "boxes_visualization": str(image_path),
            "coords_h5": None,
            "patch_manifest": None,
            "trace_clusters_json": str(trace_result["trace_clusters_json"]),
            "navigation_json": str(navigation_result["navigation_json"]),
            "report_json": str(observe_result["report_json"]),
            "coords_returncode": None,
            "patch_export_returncode": None,
            "input_mode": case_spec.input_mode,
            "grid_first_bypassed_coords_export": True,
        }
        case_result = AuditAgent(bundle).run(
            case_spec,
            trace_result,
            navigation_result,
            observe_result,
            segmentation_artifact,
            {"total_runtime_ms": 0},
        )
        case_result_path = write_json(output_dir / "case_result.json", case_result.to_dict())
        summary.update(
            {
                "downstream_ran": True,
                "case_result_path": str(case_result_path),
                "final_11_class": case_result.hierarchical_prediction.get("final_11_class"),
                "classification_status": case_result.hierarchical_prediction.get("classification_status"),
                "final_case_assessment": case_result.final_case_assessment,
                "audit_status": case_result.status,
                "audit_errors": case_result.audit.get("errors", []),
                "audit_warnings": case_result.audit.get("warnings", []),
            }
        )
    write_json(output_dir / "slide_memory_experiment_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run a SlideMemoryTracker abnormal-mucosa screening experiment.")
    parser.add_argument("--output-dir", default="artifacts/slide_memory_experiment")
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--with-downstream", action="store_true", help="Also run navigation/reasoning/audit as a downstream comparison.")
    parser.add_argument("--image-path", default="", help="Optional real WSI-derived grid/thumbnail image to score.")
    parser.add_argument("--grid-json", default="", help="Grid metadata JSON paired with --image-path.")
    parser.add_argument("--warm-up-steps", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3, help="Number of high-surprise real WSI candidates to annotate when no ground truth is available.")
    args = parser.parse_args()
    try:
        summary = run_experiment(
            Path(args.output_dir),
            feature_dim=int(args.feature_dim),
            with_downstream=bool(args.with_downstream),
            image_path=args.image_path or None,
            grid_json=args.grid_json or None,
            warm_up_steps=args.warm_up_steps,
            top_k=int(args.top_k),
        )
    except ImportError as exc:
        print("Experiment cannot run because an optional dependency is missing: {0}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
