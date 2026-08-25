import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from adenoma_agent.multimodal import (
    build_conch_digepath_trace_output,
    build_conch_only_trace_output,
    fuse_trace_patch_assignments_with_conch,
    request_conch_patch_predictions,
    request_digepath_patch_predictions,
)
from adenoma_agent.schemas import TraceCluster
from adenoma_agent.utils import (
    bbox_area,
    bbox_overlap_ratio,
    bbox_to_dict,
    bbox_xywh_to_xyxy,
    connected_components_from_grid,
    map_bbox_level0_to_thumb,
    map_bbox_thumb_to_level0,
    read_json,
    write_json,
)


class TraceAgent(object):
    def __init__(self, bundle, selector_adapter, backend_chain):
        self.bundle = bundle
        self.selector_adapter = selector_adapter
        self.backend_chain = backend_chain

    def run(self, case_spec, case_dir, logger, interventions=None, preferred_mode=None):
        trace_dir = Path(case_dir) / "trace"
        if case_spec.input_mode == "grid_thumbnail":
            selection = self._select_grid_input(case_spec, trace_dir)
        else:
            intervention_spec = interventions.override_roi if interventions else None
            selection = self.selector_adapter.select(
                case_spec,
                trace_dir,
                manual_boxes=intervention_spec,
                preferred_mode=preferred_mode,
            )
        payload = selection["payload"]
        thumbnail_meta = payload["thumbnail_meta"]
        thumbnail_path = selection["paths"]["thumbnail"]
        proposals, junior_guidance_used = self._build_proposals_with_source(
            thumbnail_path,
            thumbnail_meta,
            payload.get("boxes", []),
            junior_guidance=None,
        )
        proposal_json = trace_dir / "trace_proposals.json"
        write_json(
            proposal_json,
            {
                "proposals": proposals,
                "junior_guidance_used": junior_guidance_used,
                "junior_roi_count": 0,
            },
        )

        trace_mode = str(self.bundle["runtime"].get("trace", {}).get("mode", "")).strip()
        conch_runtime_metadata = None
        digepath_runtime_metadata = None
        if case_spec.input_mode == "grid_thumbnail" and trace_mode == "conch_only":
            conch_requests = self._collect_conch_patch_requests(
                thumbnail_path,
                payload.get("grid_metadata_path"),
                trace_dir / "conch_patch_crops",
                slide_path=case_spec.slide_path,
                overview_thumbnail_path=case_spec.overview_thumbnail_path
                or case_spec.metadata.get("overview_thumbnail_path")
                or case_spec.metadata.get("thumbnail_path"),
            )
            conch_runtime_metadata = request_conch_patch_predictions(conch_requests, self.bundle)
            if bool(self.bundle["runtime"].get("trace", {}).get("enable_digepath_fusion", False)):
                digepath_runtime_metadata = request_digepath_patch_predictions(conch_requests, self.bundle)
            trace_request = {
                "images": [str(thumbnail_path)],
                "metadata": {
                    "case_id": case_spec.case_id,
                    "thumbnail_meta": thumbnail_meta,
                    "proposals": proposals,
                    "selector_mode": payload.get("mode"),
                },
            }
            if digepath_runtime_metadata and digepath_runtime_metadata.get("enabled"):
                trace_output = build_conch_digepath_trace_output(
                    trace_request,
                    conch_runtime_metadata,
                    digepath_runtime_metadata,
                )
                backend_name = "local_conch_digepath"
            else:
                trace_output = build_conch_only_trace_output(trace_request, conch_runtime_metadata)
                backend_name = "local_conch"
            backend_response = {
                "backend": backend_name,
                "attempts": list(conch_runtime_metadata.get("attempts", []))
                + list((digepath_runtime_metadata or {}).get("attempts", [])),
                "trace_attempts": [],
                "output": trace_output,
                "raw_texts": [],
            }
        else:
            backend_response = self.backend_chain.invoke(
                "trace",
                self.bundle["runtime"]["trace"]["backend_chain"],
                {
                    "images": [str(thumbnail_path)],
                    "prompt": {
                        "question": self.bundle["runtime"]["trace"].get("patho_r1_question", case_spec.question),
                        "task": "ssl_others_dual_branch_trace_annotation",
                    },
                    "metadata": {
                        "case_id": case_spec.case_id,
                        "thumbnail_meta": thumbnail_meta,
                        "proposals": proposals,
                        "selector_mode": payload.get("mode"),
                        "junior_guidance": {
                            "status": "disabled_in_pipeline",
                            "used": False,
                            "roi_count": 0,
                        },
                    },
                },
            )
        proposal_lookup = {proposal["cluster_id"]: proposal for proposal in proposals}
        all_cluster_payloads = list(backend_response["output"].get("all_clusters") or backend_response["output"]["clusters"])
        if (
            trace_mode != "conch_only"
            and case_spec.input_mode == "grid_thumbnail"
            and bool(self.bundle["runtime"]["trace"].get("enable_conch_fusion", False))
        ):
            conch_requests = self._collect_conch_patch_requests(
                thumbnail_path,
                payload.get("grid_metadata_path"),
                trace_dir / "conch_patch_crops",
                slide_path=case_spec.slide_path,
                overview_thumbnail_path=case_spec.overview_thumbnail_path
                or case_spec.metadata.get("overview_thumbnail_path")
                or case_spec.metadata.get("thumbnail_path"),
            )
            conch_runtime_metadata = request_conch_patch_predictions(conch_requests, self.bundle)
            patch_assignments = backend_response["output"].get("patch_assignments", {"patches": []})
            backend_response["output"]["patch_assignments"] = fuse_trace_patch_assignments_with_conch(
                patch_assignments,
                conch_runtime_metadata.get("predictions", {}),
            )
        all_clusters = []
        for cluster_payload in all_cluster_payloads:
            proposal = proposal_lookup.get(cluster_payload["cluster_id"])
            if proposal is None:
                fallback_thumb = cluster_payload.get("group_bbox_thumb", cluster_payload["cluster_bbox_thumb"])
                fallback_level0 = cluster_payload.get("group_bbox_level0", cluster_payload["cluster_bbox_level0"])
                proposal = {
                    "cluster_id": cluster_payload["cluster_id"],
                    "cluster_bbox_thumb": fallback_thumb,
                    "cluster_bbox_level0": fallback_level0,
                    "regions_thumb": cluster_payload.get("regions_thumb", [fallback_thumb]),
                    "regions_level0": cluster_payload.get("regions_level0", [fallback_level0]),
                    "metadata": {},
                }
            all_clusters.append(
                TraceCluster(
                    cluster_id=proposal["cluster_id"],
                    cluster_bbox_thumb=proposal["cluster_bbox_thumb"],
                    cluster_bbox_level0=proposal["cluster_bbox_level0"],
                    regions_thumb=proposal["regions_thumb"],
                    regions_level0=proposal["regions_level0"],
                    l=cluster_payload["l"],
                    s=int(cluster_payload["s"]),
                    d=bool(cluster_payload["d"]),
                    review_stage=cluster_payload["review_stage"],
                    crypt_disorder_risk=int(cluster_payload["crypt_disorder_risk"]),
                    dysplasia_review_needed=bool(cluster_payload["dysplasia_review_needed"]),
                    desc=cluster_payload["desc"],
                    evidence=list(cluster_payload.get("evidence", [])),
                    metadata={
                        **proposal["metadata"],
                        **cluster_payload.get("metadata", {}),
                    },
                    patch_ids_ordered=list(cluster_payload.get("patch_ids_ordered", [])),
                    patches_thumb=list(cluster_payload.get("patches_thumb", [])),
                    patches_level0=list(cluster_payload.get("patches_level0", [])),
                    group_bbox_thumb=dict(cluster_payload.get("group_bbox_thumb", proposal["cluster_bbox_thumb"])),
                    group_bbox_level0=dict(cluster_payload.get("group_bbox_level0", proposal["cluster_bbox_level0"])),
                )
            )

        all_clusters = sorted(all_clusters, key=lambda item: int(item.s), reverse=True)
        clusters = list(all_clusters)
        clusters = clusters[: int(self.bundle["budget"].get("max_trace_candidates", 4))]
        clusters_json = trace_dir / "trace_clusters.json"
        write_json(
            clusters_json,
            {
                "clusters": [cluster.to_dict() for cluster in clusters],
                "all_clusters": [cluster.to_dict() for cluster in all_clusters],
                "patch_assignments": backend_response["output"].get("patch_assignments", {"patches": []}),
                "top_k_cluster_count": int(self.bundle["budget"].get("max_trace_candidates", 4)),
                "coverage_summary": backend_response["output"].get("coverage_summary", {}),
                "conch_runtime_metadata": conch_runtime_metadata,
                "digepath_runtime_metadata": digepath_runtime_metadata,
            },
        )
        backend_json = trace_dir / "trace_backend_attempts.json"
        write_json(
            backend_json,
            {
                "attempts": backend_response["attempts"],
                "trace_attempts": backend_response.get("trace_attempts", []),
                "conch_runtime_metadata": conch_runtime_metadata,
                "digepath_runtime_metadata": digepath_runtime_metadata,
            },
        )
        raw_response_path = selection["paths"].get("raw_response")
        if raw_response_path:
            raw_texts = backend_response.get("raw_texts", [])
            if raw_texts:
                lines = []
                for item in raw_texts:
                    lines.append(
                        "[attempt {0} | {1}]".format(
                            int(item.get("attempt_index", 0)),
                            item.get("attempt_type", "unknown"),
                        )
                    )
                    lines.append(str(item.get("text", "")))
                    lines.append("")
                Path(raw_response_path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
            elif backend_response.get("raw_text"):
                Path(raw_response_path).write_text(str(backend_response.get("raw_text", "")) + "\n", encoding="utf-8")
        logger.log(
            state="TRACE",
            agent="TraceAgent",
            input_ref=case_spec.grid_thumbnail_path if case_spec.input_mode == "grid_thumbnail" else case_spec.slide_path,
            output_ref=str(clusters_json),
            payload={
                "cluster_count": len(clusters),
                "selector_mode": payload.get("mode"),
                "backend": backend_response["backend"],
                "cache_hit": selection["cache_hit"],
                "junior_guidance_used": junior_guidance_used,
                "conch_enabled": bool(conch_runtime_metadata and conch_runtime_metadata.get("enabled")),
                "digepath_enabled": bool(digepath_runtime_metadata and digepath_runtime_metadata.get("enabled")),
            },
            latency_ms=sum(attempt.get("latency_ms", 0) for attempt in selection.get("attempts", [])),
        )
        return {
            "selection": selection,
            "payload": payload,
            "clusters": clusters,
            "all_clusters": all_clusters,
            "trace_clusters_json": clusters_json,
            "trace_backend_json": backend_json,
            "trace_dir": trace_dir,
        }

    def _resolve_classifier_source_image(self, thumbnail_path, grid_metadata_path, grid_payload, overview_thumbnail_path=None):
        candidates = []
        if overview_thumbnail_path:
            candidates.append(Path(overview_thumbnail_path))
        for key in ("overview_thumbnail_path", "thumbnail_path", "clean_thumbnail_path"):
            value = str((grid_payload or {}).get(key) or "").strip()
            if value:
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = Path(grid_metadata_path).parent / candidate
                candidates.append(candidate)
        thumbnail_path = Path(thumbnail_path)
        if "_grid" in thumbnail_path.stem:
            candidates.append(thumbnail_path.with_name(thumbnail_path.name.replace("_grid", "")))
            candidates.append(thumbnail_path.with_name(thumbnail_path.stem.replace("_grid", "") + thumbnail_path.suffix))
        for candidate in candidates:
            if candidate.exists():
                return candidate, "clean_overview"
        return thumbnail_path, "grid_thumbnail_fallback"

    def _collect_conch_patch_requests(self, thumbnail_path, grid_metadata_path, crop_dir, slide_path=None, overview_thumbnail_path=None):
        grid_payload = read_json(grid_metadata_path)
        crop_source = str(self.bundle.get("runtime", {}).get("trace", {}).get("classifier_crop_source", "wsi_first")).strip()
        if crop_source in {"wsi", "wsi_first"}:
            wsi_requests = self._collect_wsi_classifier_patch_requests(
                slide_path=slide_path or grid_payload.get("slide_path"),
                grid_payload=grid_payload,
                crop_dir=crop_dir,
            )
            if wsi_requests or crop_source == "wsi":
                return wsi_requests

        classifier_source_path, classifier_source_mode = self._resolve_classifier_source_image(
            thumbnail_path,
            grid_metadata_path,
            grid_payload,
            overview_thumbnail_path=overview_thumbnail_path,
        )
        thumbnail = Image.open(classifier_source_path).convert("RGB")
        grid_image = Image.open(thumbnail_path)
        grid_w, grid_h = grid_image.size
        grid_image.close()
        source_w, source_h = thumbnail.size
        scale_x = float(source_w) / float(max(1, grid_w))
        scale_y = float(source_h) / float(max(1, grid_h))
        crop_dir = Path(crop_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)
        patch_requests = []
        for cell in grid_payload.get("grid_cells", []):
            if not cell.get("is_selected"):
                continue
            patch_id = list(cell.get("patch_id", [cell.get("row_id"), cell.get("col_id")]))
            raw_x = float(cell.get("thumbnail_top_left_x", 0) or 0)
            raw_y = float(cell.get("thumbnail_top_left_y", 0) or 0)
            raw_w = float(cell.get("thumbnail_width", 0) or 0)
            raw_h = float(cell.get("thumbnail_height", 0) or 0)
            x1 = int(round(raw_x * scale_x))
            y1 = int(round(raw_y * scale_y))
            x2 = int(round((raw_x + raw_w) * scale_x))
            y2 = int(round((raw_y + raw_h) * scale_y))
            x1 = min(max(0, x1), source_w)
            y1 = min(max(0, y1), source_h)
            x2 = min(max(x1 + 1, x2), source_w)
            y2 = min(max(y1 + 1, y2), source_h)
            crop = thumbnail.crop((x1, y1, x2, y2))
            crop_path = crop_dir / "patch_{0}_{1}.png".format(int(patch_id[0]), int(patch_id[1]))
            crop.save(crop_path)
            patch_requests.append(
                {
                    "patch_id": patch_id,
                    "image_path": str(crop_path),
                    "classifier_source_image": str(classifier_source_path),
                    "classifier_source_mode": classifier_source_mode,
                    "classifier_crop_output_size": [int(crop.size[0]), int(crop.size[1])],
                }
            )
        thumbnail.close()
        return patch_requests

    def _collect_wsi_classifier_patch_requests(self, slide_path, grid_payload, crop_dir):
        if not str(slide_path or "").strip():
            return []
        slide_path = Path(str(slide_path or ""))
        if not slide_path.exists() or self._looks_like_thumbnail_path_for_grid(slide_path, grid_payload):
            return []
        try:
            from adenoma_agent.helpers.export_navigation_crops import SlideReader, clean_rgb
        except Exception:
            return []

        trace_cfg = self.bundle.get("runtime", {}).get("trace", {})
        output_size = int(trace_cfg.get("classifier_wsi_crop_output_size", 512) or 0)
        crop_dir = Path(crop_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)
        reader = None
        patch_requests = []
        try:
            reader = SlideReader(slide_path)
            for cell in grid_payload.get("grid_cells", []):
                if not cell.get("is_selected"):
                    continue
                patch_id = list(cell.get("patch_id", [cell.get("row_id"), cell.get("col_id")]))
                level0_x = int(round(float(cell.get("level0_top_left_x", 0) or 0)))
                level0_y = int(round(float(cell.get("level0_top_left_y", 0) or 0)))
                level0_w = int(round(float(cell.get("level0_width", grid_payload.get("grid_cell_size_level0", 2048)) or 2048)))
                level0_h = int(round(float(cell.get("level0_height", grid_payload.get("grid_cell_size_level0", 2048)) or 2048)))
                level0_w = max(1, level0_w)
                level0_h = max(1, level0_h)
                crop = clean_rgb(reader.read_region((level0_x, level0_y), 0, (level0_w, level0_h)))
                if output_size > 0 and crop.size != (output_size, output_size):
                    crop = crop.resize((output_size, output_size), Image.BILINEAR)
                crop_path = crop_dir / "patch_{0}_{1}.png".format(int(patch_id[0]), int(patch_id[1]))
                crop.save(crop_path)
                patch_requests.append(
                    {
                        "patch_id": patch_id,
                        "image_path": str(crop_path),
                        "classifier_source_image": str(slide_path),
                        "classifier_source_mode": "wsi_level0_5x",
                        "classifier_crop_level0_bbox": [level0_x, level0_y, level0_x + level0_w, level0_y + level0_h],
                        "classifier_crop_level0_size": [level0_w, level0_h],
                        "classifier_crop_output_size": [int(crop.size[0]), int(crop.size[1])],
                        "classifier_crop_view": "5x_fov_matched_to_grid_cell",
                    }
                )
        except Exception:
            patch_requests = []
        finally:
            if reader is not None:
                reader.close()
        return patch_requests

    def _looks_like_thumbnail_path_for_grid(self, slide_path, grid_payload):
        if slide_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            return False
        try:
            with Image.open(slide_path) as image:
                width, height = image.size
        except Exception:
            return False
        max_x = 0
        max_y = 0
        for cell in grid_payload.get("grid_cells", []):
            if not cell.get("is_selected"):
                continue
            x2 = int(round(float(cell.get("level0_top_left_x", 0) or 0) + float(cell.get("level0_width", 0) or 0)))
            y2 = int(round(float(cell.get("level0_top_left_y", 0) or 0) + float(cell.get("level0_height", 0) or 0)))
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
        return max_x > width or max_y > height

    def _load_junior_guidance(self, case_spec):
        junior_path = str(case_spec.metadata.get("junior_result_json", "")).strip()
        if not junior_path:
            return None
        candidate = Path(junior_path)
        if not candidate.exists():
            return None
        try:
            payload = read_json(candidate)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _boxes_from_proposals(self, proposals):
        boxes = []
        for proposal in proposals:
            bbox = proposal["cluster_bbox_level0"]
            boxes.append(
                {
                    "x1": int(bbox["x1"]),
                    "y1": int(bbox["y1"]),
                    "x2": int(bbox["x2"]),
                    "y2": int(bbox["y2"]),
                    "score": round(float(proposal.get("metadata", {}).get("diagnostic_priority", 1.0)), 4),
                    "label": "junior_mucosa_roi",
                    "roi_id": proposal["cluster_id"],
                }
            )
        return boxes

    def _select_grid_input(self, case_spec, trace_dir):
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        input_dir = trace_dir / "grid_input"
        input_dir.mkdir(parents=True, exist_ok=True)

        grid_thumbnail_path = Path(case_spec.grid_thumbnail_path or "")
        grid_metadata_path = Path(case_spec.grid_metadata_path or "")
        if not grid_thumbnail_path.exists():
            raise RuntimeError("grid_thumbnail_path is missing or does not exist for case {0}".format(case_spec.case_id))
        if not grid_metadata_path.exists():
            raise RuntimeError("grid_metadata_path is missing or does not exist for case {0}".format(case_spec.case_id))

        local_thumbnail_path = input_dir / grid_thumbnail_path.name
        local_metadata_path = input_dir / (grid_thumbnail_path.stem + ".json")
        if local_thumbnail_path.resolve() != grid_thumbnail_path.resolve():
            shutil.copy2(str(grid_thumbnail_path), str(local_thumbnail_path))
        if local_metadata_path.resolve() != grid_metadata_path.resolve():
            shutil.copy2(str(grid_metadata_path), str(local_metadata_path))

        grid_payload = read_json(local_metadata_path)
        image = Image.open(local_thumbnail_path)
        thumb_w, thumb_h = image.size
        image.close()
        crop_bbox_level0 = list(grid_payload.get("level0_crop_bbox") or [0, 0, thumb_w, thumb_h])
        selected_cells = [cell for cell in grid_payload.get("grid_cells", []) if cell.get("is_selected")]
        boxes = []
        for cell in selected_cells:
            boxes.append(
                {
                    "x1": int(cell["level0_top_left_x"]),
                    "y1": int(cell["level0_top_left_y"]),
                    "x2": int(cell["level0_top_left_x"]) + int(cell["level0_width"]),
                    "y2": int(cell["level0_top_left_y"]) + int(cell["level0_height"]),
                    "score": round(float(cell.get("tissue_coverage_ratio", 0.0)), 4),
                    "label": "selected_grid_cell",
                    "patch_id": list(cell.get("patch_id", [cell.get("row_id"), cell.get("col_id")])),
                }
            )
        if not boxes:
            boxes.append(
                {
                    "x1": int(crop_bbox_level0[0]),
                    "y1": int(crop_bbox_level0[1]),
                    "x2": int(crop_bbox_level0[2]),
                    "y2": int(crop_bbox_level0[3]),
                    "score": 1.0,
                    "label": "grid_crop_bbox",
                }
            )
        payload = {
            "mode": "grid_input",
            "thumbnail_meta": {
                "thumbnail_size": [int(thumb_w), int(thumb_h)],
                "slide_dimensions_level0": [int(crop_bbox_level0[2]), int(crop_bbox_level0[3])],
                "input_mode": "grid_thumbnail",
                "grid_metadata_path": str(local_metadata_path),
            },
            "boxes": boxes,
            "grid_metadata_path": str(local_metadata_path),
            "grid_thumbnail_path": str(local_thumbnail_path),
        }
        boxes_json = trace_dir / "{0}_grid_input_boxes.json".format(case_spec.case_id)
        write_json(boxes_json, payload)
        return {
            "payload": payload,
            "mode": "grid_input",
            "cache_hit": False,
            "paths": {
                "thumbnail": local_thumbnail_path,
                "raw_response": trace_dir / "{0}_grid_input_raw_response.txt".format(case_spec.case_id),
                "boxes_json": boxes_json,
                "visualization": local_thumbnail_path,
            },
            "attempts": [],
        }

    def _build_proposals(self, thumbnail_path, thumbnail_meta, route_c_boxes):
        proposals, _ = self._build_proposals_with_source(thumbnail_path, thumbnail_meta, route_c_boxes, junior_guidance=None)
        return proposals

    def _build_proposals_with_source(self, thumbnail_path, thumbnail_meta, route_c_boxes, junior_guidance=None):
        junior_proposals = self._proposals_from_junior_guidance(junior_guidance, thumbnail_meta)
        if junior_proposals:
            return junior_proposals, True
        image = Image.open(thumbnail_path).convert("RGB")
        arr = np.array(image, dtype=np.float32)
        thumb_w, thumb_h = thumbnail_meta["thumbnail_size"]
        slide_dims = thumbnail_meta["slide_dimensions_level0"]
        grid_size = int(self.bundle["runtime"]["trace"].get("cluster_grid_size", 16))
        min_cell_tissue_fraction = float(self.bundle["runtime"]["trace"].get("min_cell_tissue_fraction", 0.08))
        cell_w = max(1, int(np.ceil(float(thumb_w) / float(grid_size))))
        cell_h = max(1, int(np.ceil(float(thumb_h) / float(grid_size))))

        active_grid = []
        grid_stats = {}
        for gy in range(grid_size):
            row = []
            for gx in range(grid_size):
                x1 = gx * cell_w
                y1 = gy * cell_h
                x2 = min(thumb_w, (gx + 1) * cell_w)
                y2 = min(thumb_h, (gy + 1) * cell_h)
                cell = arr[y1:y2, x1:x2]
                mean_rgb = cell.mean(axis=2)
                sat = cell.max(axis=2) - cell.min(axis=2)
                tissue_mask = (mean_rgb < 235.0) & (sat > 8.0)
                tissue_fraction = float(tissue_mask.mean()) if cell.size else 0.0
                pale_fraction = float(((mean_rgb > 165.0) & (mean_rgb < 235.0) & (sat < 28.0) & tissue_mask).mean()) if cell.size else 0.0
                r = cell[:, :, 0]
                g = cell[:, :, 1]
                b = cell[:, :, 2]
                artifact_fraction = float((((sat > 75.0) & ((r > g * 1.25) | (b > g * 1.25) | (g > r * 1.25))).mean())) if cell.size else 0.0
                row.append(tissue_fraction >= min_cell_tissue_fraction)
                grid_stats[(gx, gy)] = {
                    "bbox_thumb": bbox_to_dict(x1, y1, x2, y2),
                    "tissue_fraction": tissue_fraction,
                    "pale_fraction": pale_fraction,
                    "artifact_fraction": artifact_fraction,
                }
            active_grid.append(row)

        components = connected_components_from_grid(active_grid)
        proposals = []
        total_thumb_area = max(1, thumb_w * thumb_h)
        for index, cells in enumerate(components):
            xs = [grid_stats[(gx, gy)]["bbox_thumb"]["x1"] for gx, gy in cells]
            ys = [grid_stats[(gx, gy)]["bbox_thumb"]["y1"] for gx, gy in cells]
            xe = [grid_stats[(gx, gy)]["bbox_thumb"]["x2"] for gx, gy in cells]
            ye = [grid_stats[(gx, gy)]["bbox_thumb"]["y2"] for gx, gy in cells]
            bbox_thumb = bbox_to_dict(min(xs), min(ys), max(xe), max(ye))
            area_fraction = float(bbox_area(bbox_thumb)) / float(total_thumb_area)
            if area_fraction < float(self.bundle["runtime"]["trace"].get("min_cluster_area_fraction", 0.01)):
                continue
            tissue_fraction = float(np.mean([grid_stats[(gx, gy)]["tissue_fraction"] for gx, gy in cells]))
            pale_fraction = float(np.mean([grid_stats[(gx, gy)]["pale_fraction"] for gx, gy in cells]))
            artifact_fraction = float(np.mean([grid_stats[(gx, gy)]["artifact_fraction"] for gx, gy in cells]))
            route_c_hint_overlap = 0.0
            route_c_hint_score = 0.0
            for box in route_c_boxes or []:
                hint_bbox = bbox_to_dict(box["x1"], box["y1"], box["x2"], box["y2"])
                route_c_hint_overlap = max(route_c_hint_overlap, bbox_overlap_ratio(bbox_thumb, hint_bbox))
                route_c_hint_score = max(route_c_hint_score, float(box.get("score", 0.0)))
            proposals.append(
                {
                    "cluster_id": "cluster_{0:02d}".format(index),
                    "cluster_bbox_thumb": bbox_thumb,
                    "cluster_bbox_level0": map_bbox_thumb_to_level0(bbox_thumb, (thumb_w, thumb_h), slide_dims),
                    "regions_thumb": [bbox_thumb],
                    "regions_level0": [map_bbox_thumb_to_level0(bbox_thumb, (thumb_w, thumb_h), slide_dims)],
                    "metadata": {
                        "cell_count": len(cells),
                        "tissue_fraction": round(tissue_fraction, 4),
                        "pale_fraction": round(pale_fraction, 4),
                        "artifact_fraction": round(artifact_fraction, 4),
                        "area_fraction": round(area_fraction, 4),
                        "route_c_hint_overlap": round(route_c_hint_overlap, 4),
                        "route_c_hint_score": round(route_c_hint_score, 4),
                    },
                }
            )

        if not proposals:
            fallback_bbox = bbox_to_dict(
                int(round(thumb_w * 0.2)),
                int(round(thumb_h * 0.2)),
                int(round(thumb_w * 0.8)),
                int(round(thumb_h * 0.8)),
            )
            proposals.append(
                {
                    "cluster_id": "cluster_00",
                    "cluster_bbox_thumb": fallback_bbox,
                    "cluster_bbox_level0": map_bbox_thumb_to_level0(fallback_bbox, (thumb_w, thumb_h), slide_dims),
                    "regions_thumb": [fallback_bbox],
                    "regions_level0": [map_bbox_thumb_to_level0(fallback_bbox, (thumb_w, thumb_h), slide_dims)],
                    "metadata": {
                        "cell_count": 0,
                        "tissue_fraction": 0.0,
                        "pale_fraction": 0.0,
                        "artifact_fraction": 0.0,
                        "area_fraction": round(float(bbox_area(fallback_bbox)) / float(total_thumb_area), 4),
                        "route_c_hint_overlap": 0.0,
                        "route_c_hint_score": 0.0,
                    },
                }
            )

        return proposals, False

    def _proposals_from_junior_guidance(self, junior_guidance, thumbnail_meta):
        if not junior_guidance or junior_guidance.get("status") != "ok":
            return []
        mucosa_rois = list(junior_guidance.get("mucosa_rois", []))
        if not mucosa_rois:
            return []
        thumb_size = tuple(thumbnail_meta["thumbnail_size"])
        slide_dims = tuple(thumbnail_meta["slide_dimensions_level0"])
        total_thumb_area = max(1, int(thumb_size[0]) * int(thumb_size[1]))
        proposals = []
        for index, roi in enumerate(mucosa_rois):
            level0_box = roi.get("level_0_bounding_box")
            if not isinstance(level0_box, dict):
                continue
            bbox_level0 = bbox_xywh_to_xyxy(level0_box)
            bbox_thumb = map_bbox_level0_to_thumb(bbox_level0, thumb_size, slide_dims)
            if bbox_area(bbox_thumb) <= 0:
                continue
            area_fraction = float(bbox_area(bbox_thumb)) / float(total_thumb_area)
            cluster_id = str(roi.get("roi_id") or "junior_cluster_{0:02d}".format(index + 1))
            proposals.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_bbox_thumb": bbox_thumb,
                    "cluster_bbox_level0": bbox_level0,
                    "regions_thumb": [bbox_thumb],
                    "regions_level0": [bbox_level0],
                    "metadata": {
                        "cell_count": 1,
                        "tissue_fraction": 1.0,
                        "pale_fraction": 0.0,
                        "artifact_fraction": 0.0,
                        "area_fraction": round(area_fraction, 4),
                        "route_c_hint_overlap": 0.0,
                        "route_c_hint_score": 0.0,
                        "diagnostic_priority": int(roi.get("diagnostic_priority", 1)),
                        "source_stage": "junior_agent",
                        "comment": str(roi.get("comment", "")),
                    },
                }
            )
        return proposals
