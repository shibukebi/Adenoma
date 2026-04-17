from pathlib import Path

import numpy as np
from PIL import Image

from adenoma_agent.schemas import TraceCluster
from adenoma_agent.utils import (
    bbox_area,
    bbox_overlap_ratio,
    bbox_to_dict,
    connected_components_from_grid,
    map_bbox_thumb_to_level0,
    write_json,
)


class TraceAgent(object):
    def __init__(self, bundle, selector_adapter, backend_chain):
        self.bundle = bundle
        self.selector_adapter = selector_adapter
        self.backend_chain = backend_chain

    def run(self, case_spec, case_dir, logger, interventions=None, preferred_mode=None):
        trace_dir = Path(case_dir) / "trace"
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
        proposals = self._build_proposals(thumbnail_path, thumbnail_meta, payload.get("boxes", []))
        proposal_json = trace_dir / "trace_proposals.json"
        write_json(proposal_json, {"proposals": proposals})

        backend_response = self.backend_chain.invoke(
            "trace",
            self.bundle["runtime"]["trace"]["backend_chain"],
            {
                "images": [str(thumbnail_path)],
                "prompt": {
                    "question": case_spec.question,
                    "task": "ssa_vs_others_trace_annotation",
                },
                "metadata": {
                    "case_id": case_spec.case_id,
                    "thumbnail_meta": thumbnail_meta,
                    "proposals": proposals,
                    "selector_mode": payload.get("mode"),
                },
            },
        )
        proposal_lookup = {proposal["cluster_id"]: proposal for proposal in proposals}
        clusters = []
        for cluster_payload in backend_response["output"]["clusters"]:
            proposal = proposal_lookup[cluster_payload["cluster_id"]]
            clusters.append(
                TraceCluster(
                    cluster_id=proposal["cluster_id"],
                    cluster_bbox_thumb=proposal["cluster_bbox_thumb"],
                    cluster_bbox_level0=proposal["cluster_bbox_level0"],
                    regions_thumb=proposal["regions_thumb"],
                    regions_level0=proposal["regions_level0"],
                    l=cluster_payload["l"],
                    s=int(cluster_payload["s"]),
                    d=bool(cluster_payload["d"]),
                    desc=cluster_payload["desc"],
                    evidence=list(cluster_payload.get("evidence", [])),
                    metadata={
                        **proposal["metadata"],
                        **cluster_payload.get("metadata", {}),
                    },
                )
            )

        clusters = sorted(clusters, key=lambda item: (item.s, bbox_area(item.cluster_bbox_thumb)), reverse=True)
        clusters = clusters[: int(self.bundle["budget"].get("max_trace_candidates", 4))]
        clusters_json = trace_dir / "trace_clusters.json"
        write_json(clusters_json, {"clusters": [cluster.to_dict() for cluster in clusters]})
        backend_json = trace_dir / "trace_backend_attempts.json"
        write_json(backend_json, {"attempts": backend_response["attempts"]})
        logger.log(
            state="TRACE",
            agent="TraceAgent",
            input_ref=case_spec.slide_path,
            output_ref=str(clusters_json),
            payload={
                "cluster_count": len(clusters),
                "selector_mode": payload.get("mode"),
                "backend": backend_response["backend"],
                "cache_hit": selection["cache_hit"],
            },
            latency_ms=sum(attempt.get("latency_ms", 0) for attempt in selection.get("attempts", [])),
        )
        return {
            "selection": selection,
            "payload": payload,
            "clusters": clusters,
            "trace_clusters_json": clusters_json,
            "trace_backend_json": backend_json,
            "trace_dir": trace_dir,
        }

    def _build_proposals(self, thumbnail_path, thumbnail_meta, route_c_boxes):
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

        return proposals
