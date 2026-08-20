import json
from pathlib import Path

from adenoma_agent.agentflow.contracts import EvidenceRecord


def _read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class MucosaEvidenceBridge(object):
    """Adapts the implemented compact Mucosa v1 output into AgentFlow records."""

    def load(self, case_id, mucosa_output_dir, slide_id=None):
        root = Path(mucosa_output_dir)
        manifest_path = root / "manifest.json"
        tile_index_path = root / "tile_index.jsonl"
        five_x_manifest_path = root / "five_x_patch_manifest.jsonl"
        for path in (manifest_path, tile_index_path, five_x_manifest_path):
            if not path.exists():
                raise FileNotFoundError("Missing compact Mucosa artifact: {0}".format(path))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_model = str(manifest.get("config", {}).get("source_model", "uni_prismnet"))
        schema_version = str(manifest.get("schema_version", "mucosa_extractor_v1_compact"))
        records = []
        for row in _read_jsonl(tile_index_path):
            if slide_id is not None and str(row.get("slide_id")) != str(slide_id):
                continue
            bbox = tuple(int(value) for value in row.get("level0_bbox", []))
            tile_id = str(row.get("tile_id", ""))
            uncertainty = float(row.get("uncertainty", 1.0) or 0.0)
            quality = max(0.0, min(1.0, float(row.get("tissue_coverage", 0.0) or 0.0)))
            for feature, value in (row.get("task_context") or {}).items():
                evaluable = quality > 0.0
                records.append(
                    EvidenceRecord(
                        evidence_id="TISSUE_{0}_{1}".format(tile_id, feature),
                        case_id=case_id,
                        evidence_type="tissue_context_evidence",
                        feature=str(feature),
                        status="measurement" if evaluable else "not_evaluable",
                        confidence=max(0.0, min(1.0, 1.0 - uncertainty)),
                        value=float(value) if evaluable else None,
                        source=source_model,
                        source_version=schema_version,
                        quality=quality,
                        feature_evaluability="adequate" if evaluable else "not_evaluable",
                        scale=20.0,
                        level0_bbox=bbox,
                        patch_id=tile_id,
                        limitations=(
                            "Tissue context is a routing/source signal and is not morphology or dysplasia evidence.",
                        ),
                        metadata={
                            "hard_context": row.get("hard_context"),
                            "uncertainty": uncertainty,
                            "mucosa_score": row.get("mucosa_score"),
                            "mucosa_mask_coverage": row.get("mucosa_mask_coverage"),
                            "included_in_mucosa_mask": row.get("included_in_mucosa_mask"),
                        },
                    )
                )
        return {
            "manifest": manifest,
            "five_x_manifest_path": str(five_x_manifest_path),
            "evidence_records": tuple(records),
        }
