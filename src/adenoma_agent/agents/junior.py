import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from adenoma_agent.utils import bbox_to_dict, bbox_xyxy_to_xywh, ensure_dir, map_bbox_thumb_to_level0, read_json, write_json

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openslide = None


def _clip01(value):
    return max(0.0, min(1.0, float(value)))


def _normalize_to_uint8(array):
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    max_value = float(array.max()) if array.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(array, dtype=np.uint8)
    scaled = np.clip(array / max_value, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _mask_to_uint8(mask):
    return np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)


def _mask_to_base64_png(mask):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _ellipse_kernel(kernel_size):
    radius = int(max(1, kernel_size // 2))
    y_coords, x_coords = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    norm = (x_coords.astype(np.float32) ** 2 + y_coords.astype(np.float32) ** 2) / float(max(1, radius * radius))
    return norm <= 1.0


def _binary_dilate_numpy(mask, kernel):
    mask = np.asarray(mask, dtype=bool)
    kernel = np.asarray(kernel, dtype=bool)
    height, width = mask.shape
    kernel_h, kernel_w = kernel.shape
    pad_y = kernel_h // 2
    pad_x = kernel_w // 2
    padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    positions = np.argwhere(kernel)
    for ky, kx in positions:
        output |= padded[ky : ky + height, kx : kx + width]
    return output


def _binary_erode_numpy(mask, kernel):
    mask = np.asarray(mask, dtype=bool)
    kernel = np.asarray(kernel, dtype=bool)
    height, width = mask.shape
    kernel_h, kernel_w = kernel.shape
    pad_y = kernel_h // 2
    pad_x = kernel_w // 2
    padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=False)
    output = np.ones_like(mask, dtype=bool)
    positions = np.argwhere(kernel)
    for ky, kx in positions:
        output &= padded[ky : ky + height, kx : kx + width]
    return output


def _binary_close_numpy(mask, kernel):
    return _binary_erode_numpy(_binary_dilate_numpy(mask, kernel), kernel)


def _connected_components(mask):
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            pixels = []
            while stack:
                cx, cy = stack.pop()
                pixels.append((cx, cy))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            components.append(pixels)
    return components


def generate_anatomical_mask(
    heatmap,
    count_matrix,
    threshold=0.5,
    patch_size=256,
    stride=128,
    region_constraint_mask=None,
    return_debug=False,
):
    """
    Convert accumulated low-magnification mucosa votes into a continuous anatomical mask.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    count_matrix = np.asarray(count_matrix, dtype=np.float32)
    constraint_mask = None
    if region_constraint_mask is not None:
        constraint_mask = np.asarray(region_constraint_mask, dtype=bool)

    safe_count = np.where(count_matrix == 0, 1.0, count_matrix)
    avg_prob_map = heatmap / safe_count
    binary_mask = avg_prob_map >= float(threshold)
    if constraint_mask is not None:
        binary_mask &= constraint_mask

    kernel_size = int(stride * 0.75)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = _ellipse_kernel(max(3, kernel_size))

    if cv2 is not None:  # pragma: no branch - exercised only when available
        cv2_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel.shape[1], kernel.shape[0]))
        binary_u8 = _mask_to_uint8(binary_mask)
        dilated_u8 = cv2.dilate(binary_u8, cv2_kernel, iterations=1)
        closed_u8 = cv2.morphologyEx(dilated_u8, cv2.MORPH_CLOSE, cv2_kernel)
        dilated_mask = dilated_u8 > 0
        closed_mask = closed_u8 > 0
    else:
        dilated_mask = _binary_dilate_numpy(binary_mask, kernel)
        closed_mask = _binary_close_numpy(dilated_mask, kernel)

    if constraint_mask is not None:
        dilated_mask &= constraint_mask
        closed_mask &= constraint_mask

    min_area_threshold = int((patch_size * patch_size) * 2)
    min_area_threshold = min(min_area_threshold, max(256, int(binary_mask.size * 0.02)))
    final_mask_bool = np.zeros_like(closed_mask, dtype=bool)
    for component in _connected_components(closed_mask):
        if len(component) < min_area_threshold:
            continue
        for x, y in component:
            final_mask_bool[y, x] = True

    final_mask = _mask_to_uint8(final_mask_bool)
    if not return_debug:
        return final_mask
    return final_mask, {
        "avg_prob_map": avg_prob_map,
        "binary_mask": _mask_to_uint8(binary_mask),
        "dilated_mask": _mask_to_uint8(dilated_mask),
        "closed_mask": _mask_to_uint8(closed_mask),
    }


class JuniorAgent(object):
    def __init__(self, bundle):
        self.bundle = bundle

    def run(self, case_spec, case_dir, logger):
        junior_dir = ensure_dir(Path(case_dir) / "junior")
        source = self._resolve_source(case_spec)
        if source is None:
            payload = self._write_skipped_payload(
                junior_dir,
                case_spec.case_id,
                "no_readable_overview_source",
            )
            logger.log(
                state="JUNIOR",
                agent="JuniorAgent",
                input_ref=case_spec.slide_path,
                output_ref=str(payload["junior_json"]),
                status="skipped",
                payload={"status": "skipped", "reason": payload["skip_reason"], "roi_count": 0},
            )
            return payload

        try:
            overview, source_info = self._load_overview_image(case_spec, source)
        except Exception as exc:
            payload = self._write_skipped_payload(junior_dir, case_spec.case_id, "overview_load_failed: {0}".format(exc))
            logger.log(
                state="JUNIOR",
                agent="JuniorAgent",
                input_ref=source.get("path"),
                output_ref=str(payload["junior_json"]),
                status="skipped",
                payload={"status": "skipped", "reason": payload["skip_reason"], "roi_count": 0},
            )
            return payload

        cfg = self._cfg()
        tissue_mask, tissue_stats = self._compute_tissue_mask(overview, cfg)
        invalid_mask, filter_stats = self._compute_invalid_mask(overview, tissue_mask)
        valid_region_mask = tissue_mask & (~invalid_mask)
        heatmap, count_matrix, patch_stats = self._accumulate_mucosa_votes(overview, tissue_mask, invalid_mask, cfg)
        final_mask, debug = generate_anatomical_mask(
            heatmap,
            count_matrix,
            threshold=cfg["threshold"],
            patch_size=cfg["patch_size"],
            stride=cfg["stride"],
            region_constraint_mask=valid_region_mask,
            return_debug=True,
        )
        rois = self._extract_rois(final_mask, source_info, cfg)

        artifacts = self._write_debug_artifacts(
            junior_dir,
            overview,
            tissue_mask,
            invalid_mask,
            heatmap,
            count_matrix,
            debug,
            final_mask,
        )
        result_payload = {
            "slide_id": case_spec.case_id,
            "status": "ok",
            "source_image": source_info,
            "pipeline_metadata": {
                "junior_backbone": cfg["backbone_name"],
                "processing_level_downsample": source_info["processing_level_downsample"],
                "patch_size": cfg["patch_size"],
                "stride": cfg["stride"],
                "threshold": cfg["threshold"],
            },
            "statistics": {
                "tissue_fraction": tissue_stats["tissue_fraction"],
                "invalid_fraction": filter_stats["invalid_fraction"],
                "patch_count": patch_stats["patch_count"],
                "accepted_patch_count": patch_stats["accepted_patch_count"],
            },
            "debug_artifacts": artifacts,
            "mucosa_rois": rois,
        }
        junior_json = write_json(junior_dir / "junior_mucosa_rois.json", result_payload)
        logger.log(
            state="JUNIOR",
            agent="JuniorAgent",
            input_ref=source_info["source_path"],
            output_ref=str(junior_json),
            payload={
                "status": "ok",
                "roi_count": len(rois),
                "source_kind": source_info["source_kind"],
                "accepted_patch_count": patch_stats["accepted_patch_count"],
            },
        )
        return {
            "status": "ok",
            "junior_json": junior_json,
            "junior_dir": junior_dir,
            "roi_count": len(rois),
            "skip_reason": "",
        }

    def _cfg(self):
        runtime_cfg = self.bundle.get("runtime", {}).get("junior", {})
        return {
            "enabled": bool(runtime_cfg.get("enabled", True)),
            "backbone_name": str(runtime_cfg.get("backbone_name", "heuristic_low_mag_proxy")),
            "processing_level_downsample": float(runtime_cfg.get("processing_level_downsample", 32.0)),
            "patch_size": int(runtime_cfg.get("patch_size", 256)),
            "stride": int(runtime_cfg.get("stride", 128)),
            "threshold": float(runtime_cfg.get("threshold", 0.5)),
            "min_tissue_fraction": float(runtime_cfg.get("min_tissue_fraction", 0.05)),
            "mucosa_probability_threshold": float(runtime_cfg.get("mucosa_probability_threshold", 0.55)),
            "necrosis_veto_threshold": float(runtime_cfg.get("necrosis_veto_threshold", 0.35)),
            "tissue_brightness_threshold": float(runtime_cfg.get("tissue_brightness_threshold", 242.0)),
            "tissue_saturation_threshold": float(runtime_cfg.get("tissue_saturation_threshold", 10.0)),
        }

    def _resolve_source(self, case_spec):
        cfg = self._cfg()
        if not cfg["enabled"]:
            return None
        if case_spec.overview_thumbnail_path and Path(case_spec.overview_thumbnail_path).exists():
            return {"source_kind": "overview_thumbnail", "path": str(case_spec.overview_thumbnail_path)}
        if case_spec.grid_thumbnail_path and Path(case_spec.grid_thumbnail_path).exists():
            return {"source_kind": "grid_thumbnail", "path": str(case_spec.grid_thumbnail_path)}
        if case_spec.slide_path and Path(case_spec.slide_path).exists():
            return {"source_kind": "slide_path", "path": str(case_spec.slide_path)}
        return None

    def _load_overview_image(self, case_spec, source):
        source_path = Path(source["path"])
        source_kind = source["source_kind"]
        processing_level_downsample = self._cfg()["processing_level_downsample"]
        slide_dimensions_level0 = None
        if case_spec.grid_metadata_path and Path(case_spec.grid_metadata_path).exists():
            grid_meta = read_json(case_spec.grid_metadata_path)
            crop_bbox = list(grid_meta.get("level0_crop_bbox") or [])
            if len(crop_bbox) >= 4:
                slide_dimensions_level0 = [int(crop_bbox[2]), int(crop_bbox[3])]

        if source_kind in {"overview_thumbnail", "grid_thumbnail"}:
            image = Image.open(source_path).convert("RGB")
            overview = np.array(image, dtype=np.uint8)
            if slide_dimensions_level0 is None:
                slide_dimensions_level0 = [int(image.size[0]), int(image.size[1])]
            processing_level_downsample = round(
                (
                    float(slide_dimensions_level0[0]) / float(max(1, image.size[0]))
                    + float(slide_dimensions_level0[1]) / float(max(1, image.size[1]))
                )
                / 2.0,
                4,
            )
        else:
            try:
                image = Image.open(source_path).convert("RGB")
                overview = np.array(image, dtype=np.uint8)
                slide_dimensions_level0 = [int(image.size[0]), int(image.size[1])]
                processing_level_downsample = 1.0
            except Exception:
                if openslide is None:
                    raise
                slide = openslide.open_slide(str(source_path))
                slide_dimensions_level0 = [int(slide.dimensions[0]), int(slide.dimensions[1])]
                thumb_size = (
                    max(1, int(round(float(slide_dimensions_level0[0]) / float(processing_level_downsample)))),
                    max(1, int(round(float(slide_dimensions_level0[1]) / float(processing_level_downsample)))),
                )
                image = slide.get_thumbnail(thumb_size).convert("RGB")
                overview = np.array(image, dtype=np.uint8)
                processing_level_downsample = round(
                    (
                        float(slide_dimensions_level0[0]) / float(max(1, image.size[0]))
                        + float(slide_dimensions_level0[1]) / float(max(1, image.size[1]))
                    )
                    / 2.0,
                    4,
                )
        return overview, {
            "source_kind": source_kind,
            "source_path": str(source_path),
            "width_at_processing_level": int(overview.shape[1]),
            "height_at_processing_level": int(overview.shape[0]),
            "slide_dimensions_level0": [int(slide_dimensions_level0[0]), int(slide_dimensions_level0[1])],
            "processing_level_downsample": float(processing_level_downsample),
        }

    def _compute_tissue_mask(self, overview, cfg):
        overview = overview.astype(np.float32)
        mean_rgb = overview.mean(axis=2)
        sat = overview.max(axis=2) - overview.min(axis=2)
        r = overview[:, :, 0]
        g = overview[:, :, 1]
        b = overview[:, :, 2]
        pink_like = (r > 150.0) & (g > 90.0) & (b > 105.0)
        purple_like = (r > 95.0) & (b > 105.0) & (sat > 8.0)
        tissue_mask = (mean_rgb < cfg["tissue_brightness_threshold"]) & (
            (sat > cfg["tissue_saturation_threshold"]) | pink_like | purple_like
        )
        return tissue_mask, {"tissue_fraction": round(float(tissue_mask.mean()) if tissue_mask.size else 0.0, 6)}

    def _compute_invalid_mask(self, overview, tissue_mask):
        overview = overview.astype(np.float32)
        mean_rgb = overview.mean(axis=2)
        sat = overview.max(axis=2) - overview.min(axis=2)
        r = overview[:, :, 0]
        g = overview[:, :, 1]
        b = overview[:, :, 2]
        necrosis_like = tissue_mask & (mean_rgb > 170.0) & (sat < 22.0)
        glare_fold_like = tissue_mask & (mean_rgb > 220.0) & (sat > 18.0)
        oversaturated_artifact = tissue_mask & (
            ((r > g * 1.28) | (b > g * 1.28) | (g > r * 1.28)) & (sat > 75.0)
        )
        invalid_mask = necrosis_like | glare_fold_like | oversaturated_artifact
        return invalid_mask, {"invalid_fraction": round(float(invalid_mask.mean()) if invalid_mask.size else 0.0, 6)}

    def _score_patch(self, patch, patch_tissue_mask, patch_invalid_mask):
        if patch.size == 0:
            return {
                "contains_mucosa": 0.0,
                "contains_submucosa": 0.0,
                "contains_muscularis": 0.0,
                "contains_necrosis": 0.0,
                "tissue_fraction": 0.0,
            }
        tissue_fraction = float(patch_tissue_mask.mean()) if patch_tissue_mask.size else 0.0
        necrosis_fraction = float(patch_invalid_mask.mean()) if patch_invalid_mask.size else 0.0
        valid_mask = patch_tissue_mask & (~patch_invalid_mask)
        pixels = patch[valid_mask] if valid_mask.any() else patch.reshape(-1, 3)
        if pixels.size == 0:
            pixels = patch.reshape(-1, 3)
        pixels = pixels.astype(np.float32)
        r = pixels[:, 0]
        g = pixels[:, 1]
        b = pixels[:, 2]
        brightness = (r + g + b) / 3.0
        sat = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
        pink_fraction = float(np.mean((r > 150.0) & (g > 90.0) & (b > 105.0) & (sat > 8.0))) if pixels.size else 0.0
        purple_fraction = float(np.mean((r > 95.0) & (b > 105.0) & (b > g * 0.95))) if pixels.size else 0.0
        pale_fraction = float(np.mean((brightness > 175.0) & (sat < 28.0))) if pixels.size else 0.0
        muscular_fraction = float(np.mean((r > 150.0) & (g > 110.0) & (b < 150.0) & (r > b * 1.08))) if pixels.size else 0.0
        mucosa = _clip01(
            0.40 * tissue_fraction
            + 0.30 * pink_fraction
            + 0.20 * purple_fraction
            + 0.15 * float(np.mean((r > g) & (b > g * 0.85)))
            - 0.35 * necrosis_fraction
            - 0.15 * muscular_fraction
        )
        submucosa = _clip01(0.30 * tissue_fraction + 0.45 * pale_fraction + 0.15 * pink_fraction)
        muscularis = _clip01(0.20 * tissue_fraction + 0.70 * muscular_fraction - 0.10 * purple_fraction)
        necrosis = _clip01(0.80 * necrosis_fraction + 0.20 * float(np.mean((brightness > 180.0) & (sat < 15.0))))
        return {
            "contains_mucosa": mucosa,
            "contains_submucosa": submucosa,
            "contains_muscularis": muscularis,
            "contains_necrosis": necrosis,
            "tissue_fraction": tissue_fraction,
        }

    def _accumulate_mucosa_votes(self, overview, tissue_mask, invalid_mask, cfg):
        patch_size = int(cfg["patch_size"])
        stride = int(cfg["stride"])
        height, width = overview.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        count_matrix = np.zeros((height, width), dtype=np.float32)
        patch_count = 0
        accepted_patch_count = 0
        for y1 in range(0, height, stride):
            for x1 in range(0, width, stride):
                y2 = min(height, y1 + patch_size)
                x2 = min(width, x1 + patch_size)
                patch = overview[y1:y2, x1:x2]
                patch_tissue_mask = tissue_mask[y1:y2, x1:x2]
                patch_invalid_mask = invalid_mask[y1:y2, x1:x2]
                scores = self._score_patch(patch, patch_tissue_mask, patch_invalid_mask)
                patch_count += 1
                if scores["tissue_fraction"] < cfg["min_tissue_fraction"]:
                    continue
                if scores["contains_mucosa"] < cfg["mucosa_probability_threshold"]:
                    continue
                if scores["contains_necrosis"] >= cfg["necrosis_veto_threshold"]:
                    continue
                valid_patch = patch_tissue_mask & (~patch_invalid_mask)
                if not valid_patch.any():
                    continue
                contribution = valid_patch.astype(np.float32) * float(scores["contains_mucosa"])
                heatmap[y1:y2, x1:x2] += contribution
                count_matrix[y1:y2, x1:x2] += valid_patch.astype(np.float32)
                accepted_patch_count += 1
        return heatmap, count_matrix, {
            "patch_count": int(patch_count),
            "accepted_patch_count": int(accepted_patch_count),
        }

    def _extract_rois(self, final_mask, source_info, cfg):
        mask_bool = np.asarray(final_mask, dtype=np.uint8) > 0
        components = _connected_components(mask_bool)
        total_area = max(1, int(mask_bool.shape[0]) * int(mask_bool.shape[1]))
        min_roi_area = min(int((cfg["patch_size"] * cfg["patch_size"]) * 2), max(256, int(total_area * 0.02)))
        rois = []
        for index, component in enumerate(sorted(components, key=len, reverse=True)):
            if len(component) < min_roi_area:
                continue
            xs = [point[0] for point in component]
            ys = [point[1] for point in component]
            bbox_thumb = bbox_to_dict(min(xs), min(ys), max(xs) + 1, max(ys) + 1)
            bbox_level0 = map_bbox_thumb_to_level0(
                bbox_thumb,
                (source_info["width_at_processing_level"], source_info["height_at_processing_level"]),
                source_info["slide_dimensions_level0"],
            )
            local_mask = np.zeros(
                (bbox_thumb["y2"] - bbox_thumb["y1"], bbox_thumb["x2"] - bbox_thumb["x1"]),
                dtype=np.uint8,
            )
            for x, y in component:
                local_mask[y - bbox_thumb["y1"], x - bbox_thumb["x1"]] = 255
            area_fraction = float(len(component)) / float(total_area)
            diagnostic_priority = 1
            if area_fraction >= 0.18:
                diagnostic_priority = 4
            elif area_fraction >= 0.08:
                diagnostic_priority = 3
            elif area_fraction >= 0.03:
                diagnostic_priority = 2
            rois.append(
                {
                    "roi_id": "mucosa_zone_{0:02d}".format(index + 1),
                    "diagnostic_priority": diagnostic_priority,
                    "level_0_bounding_box": bbox_xyxy_to_xywh(bbox_level0),
                    "comment": "Low-mag connected mucosal region retained after tissue and artifact filtering",
                    "mask_spec": {
                        "width_at_processing_level": int(local_mask.shape[1]),
                        "height_at_processing_level": int(local_mask.shape[0]),
                        "encoding": "base64_png",
                        "raw_mask_string": _mask_to_base64_png(local_mask),
                    },
                    "metadata": {
                        "bbox_thumb": bbox_thumb,
                        "area_fraction": round(area_fraction, 6),
                        "pixel_area": int(len(component)),
                    },
                }
            )
        return rois

    def _write_debug_artifacts(
        self,
        junior_dir,
        overview,
        tissue_mask,
        invalid_mask,
        heatmap,
        count_matrix,
        debug,
        final_mask,
    ):
        processing_image_path = Path(junior_dir) / "processing_overview.png"
        Image.fromarray(np.asarray(overview, dtype=np.uint8), mode="RGB").save(processing_image_path)
        artifact_paths = {
            "processing_overview": str(processing_image_path),
            "tissue_mask": str(self._save_mask(junior_dir, "tissue_mask.png", _mask_to_uint8(tissue_mask))),
            "artifact_necrosis_filter": str(self._save_mask(junior_dir, "artifact_necrosis_filter.png", _mask_to_uint8(invalid_mask))),
            "mucosa_probability_map": str(self._save_mask(junior_dir, "mucosa_probability_map.png", _normalize_to_uint8(heatmap))),
            "count_matrix": str(self._save_mask(junior_dir, "count_matrix.png", _normalize_to_uint8(count_matrix))),
            "binary_mask": str(self._save_mask(junior_dir, "binary_mask.png", debug["binary_mask"])),
            "post_dilation_mask": str(self._save_mask(junior_dir, "post_dilation_mask.png", debug["dilated_mask"])),
            "post_closing_mask": str(self._save_mask(junior_dir, "post_closing_mask.png", debug["closed_mask"])),
            "anatomical_mask": str(self._save_mask(junior_dir, "anatomical_mask.png", final_mask)),
            "final_roi_boxes": str(self._save_roi_overlay(junior_dir, overview, final_mask)),
        }
        return artifact_paths

    def _save_mask(self, junior_dir, filename, array):
        output_path = Path(junior_dir) / filename
        Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(output_path)
        return output_path

    def _save_roi_overlay(self, junior_dir, overview, final_mask):
        output_path = Path(junior_dir) / "final_roi_boxes.png"
        roi_canvas = np.array(overview, dtype=np.uint8).copy()
        mask_bool = np.asarray(final_mask, dtype=np.uint8) > 0
        for component in _connected_components(mask_bool):
            xs = [point[0] for point in component]
            ys = [point[1] for point in component]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            roi_canvas[max(0, y1 - 1) : min(roi_canvas.shape[0], y1 + 2), x1 : x2 + 1] = [255, 80, 80]
            roi_canvas[max(0, y2 - 1) : min(roi_canvas.shape[0], y2 + 2), x1 : x2 + 1] = [255, 80, 80]
            roi_canvas[y1 : y2 + 1, max(0, x1 - 1) : min(roi_canvas.shape[1], x1 + 2)] = [255, 80, 80]
            roi_canvas[y1 : y2 + 1, max(0, x2 - 1) : min(roi_canvas.shape[1], x2 + 2)] = [255, 80, 80]
        Image.fromarray(roi_canvas, mode="RGB").save(output_path)
        return output_path

    def _write_skipped_payload(self, junior_dir, slide_id, reason):
        payload = {
            "slide_id": slide_id,
            "status": "skipped",
            "skip_reason": str(reason),
            "mucosa_rois": [],
        }
        junior_json = write_json(Path(junior_dir) / "junior_mucosa_rois.json", payload)
        return {
            "status": "skipped",
            "junior_json": junior_json,
            "junior_dir": Path(junior_dir),
            "roi_count": 0,
            "skip_reason": str(reason),
        }
