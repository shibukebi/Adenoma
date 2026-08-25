"""Villous source audit and isolated selective Mucosa backfill.

The backfill uses the already validated UNI + PathPrism Mucosa pipeline, but
its outputs live only under the architecture POC artifact root.  It never
writes to the full baseline artifact tree and never imports CONCH.
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .io import iter_jsonl, read_json, sha256_file, sha256_payload, write_csv, write_json, write_jsonl, write_text


VILLOUS_DIAGNOSES = frozenset({"Tubulovillous adenoma", "Villous adenoma", "Villous adenoma (VA)"})
MUCOSA_SCHEMA_VERSION = "architecture_poc_villous_mucosa_backfill_v1"


class BackfillDependencyBlocked(RuntimeError):
    gate = "MUCOSA_BACKFILL_DEPENDENCY_BLOCKED"

    def __init__(self, message: str, issues: Optional[Sequence[Mapping[str, Any]]] = None):
        super().__init__(message)
        self.issues = [dict(item) for item in (issues or ())]


def _positive(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError("not positive")
    return parsed


def _valid_manifest_rows(manifest_path: Path, minimum_coverage: float = 0.60) -> List[Dict[str, Any]]:
    rows = []
    for index, row in enumerate(iter_jsonl(manifest_path), 1):
        try:
            bbox = [int(value) for value in row.get("level0_bbox", [])]
            clipped = [int(value) for value in row.get("clipped_level0_bbox", bbox)]
            grid = [int(value) for value in row.get("grid_index", [])]
            physical = dict(row.get("physical_provenance") or {})
            coverage = float(row.get("mucosa_coverage", -1.0))
            requested = _positive(physical.get("requested_magnification"))
            mpp_x = _positive(physical.get("mpp_x"))
            mpp_y = _positive(physical.get("mpp_y"))
            base = _positive(physical.get("base_magnification"))
            fov = [_positive(value) for value in physical.get("fov_microns", [])]
            dimensions = list(physical.get("output_pixel_dimensions") or [])
            if len(bbox) != 4 or len(clipped) != 4 or len(grid) != 2 or len(fov) != 2:
                continue
            if bbox != clipped or coverage < float(minimum_coverage) or coverage > 1.0:
                continue
            if abs(requested - 5.0) > 1e-6 or dimensions != [256, 256]:
                continue
            rows.append(
                {
                    "source_order": index,
                    "source_patch_id": str(row.get("patch_id", "")),
                    "bbox": bbox,
                    "grid_index": grid,
                    "mucosa_coverage": coverage,
                    "magnification": requested,
                    "mpp_x": mpp_x,
                    "mpp_y": mpp_y,
                    "base_magnification": base,
                    "physical_fov_um": fov,
                }
            )
        except (TypeError, ValueError, KeyError):
            continue
    rows.sort(key=lambda row: (-row["mucosa_coverage"], row["grid_index"], row["source_patch_id"]))
    return rows


def _mucosa_artifact(root: Path, case_alias: str, minimum_coverage: float) -> Dict[str, Any]:
    case_root = Path(root) / case_alias
    boundary_path = case_root / "baseline_boundary.json"
    manifest_path = case_root / "five_x_patch_manifest.jsonl"
    result = {
        "mucosa_root": str(Path(root).resolve()),
        "case_root": str(case_root.resolve()),
        "boundary_path": str(boundary_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "boundary_exists": boundary_path.is_file(),
        "manifest_exists": manifest_path.is_file(),
        "status": "missing",
        "errors": None,
        "boundary_sha256": None,
        "manifest_sha256": None,
        "artifact_sha256": None,
        "eligible_rows": 0,
        "best_row": None,
    }
    if not boundary_path.is_file():
        return result
    try:
        boundary = read_json(boundary_path)
    except Exception as exc:
        result["status"] = "invalid_boundary"
        result["errors"] = str(exc)
        return result
    result["status"] = str(boundary.get("status", "unknown"))
    result["errors"] = int(boundary.get("counts", {}).get("errors", 0) or 0)
    result["boundary_sha256"] = sha256_file(boundary_path)
    if not manifest_path.is_file() or result["status"] != "complete" or result["errors"] != 0:
        return result
    result["manifest_sha256"] = sha256_file(manifest_path)
    eligible = _valid_manifest_rows(manifest_path, minimum_coverage=minimum_coverage)
    result["eligible_rows"] = len(eligible)
    result["best_row"] = eligible[0] if eligible else None
    result["artifact_sha256"] = sha256_payload(
        {
            "boundary_sha256": result["boundary_sha256"],
            "manifest_sha256": result["manifest_sha256"],
            "minimum_mucosa_coverage": float(minimum_coverage),
        }
    )
    return result


def _source_identity(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "source_wsi_size_bytes": int(stat.st_size),
        "source_wsi_mtime_ns": int(stat.st_mtime_ns),
        "source_wsi_identity_sha256": sha256_payload(
            {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        ),
    }


def audit_villous_recruitment(config: Mapping[str, Any], paths: Mapping[str, Path]) -> Dict[str, Any]:
    """Perform the required read-only source/Mucosa audit before backfill."""

    source_code = str(config.get("recruitment", {}).get("source_code", "YX")).upper()
    minimum_coverage = float(config.get("recruitment", {}).get("minimum_mucosa_coverage", 0.60))
    case_rows = {str(row.get("case_alias")): dict(row) for row in iter_jsonl(paths["case_paths"]) if str(row.get("source_code", "")).upper() == source_code}
    label_rows = {
        str(row.get("case_alias")): dict(row)
        for row in iter_jsonl(paths["canonical_labels"])
        if str(row.get("source_code", "")).upper() == source_code and str(row.get("label", "")) in VILLOUS_DIAGNOSES
    }
    baseline_root = paths["per_slide_mucosa_root"]
    rows = []
    for case_alias in sorted(label_rows):
        case = case_rows.get(case_alias, {})
        source_path = Path(str(case.get("source_path", "")))
        row: Dict[str, Any] = {
            "case_alias": case_alias,
            "diagnosis": label_rows[case_alias].get("label", ""),
            "family_id": case.get("family_id") or label_rows[case_alias].get("family_id", ""),
            "source_path": str(source_path),
            "source_exists": source_path.is_file(),
            "invalid_wsi": False,
            "invalid_physical_metadata": False,
            "physical_metadata_error": "",
            "mpp_x": "",
            "mpp_y": "",
            "base_magnification": "",
            "source_backend": "",
            "mucosa_status": "missing",
            "mucosa_boundary_exists": False,
            "valid_5x_manifest": False,
            "eligible_candidate_roi": False,
            "eligible_roi_count": 0,
            "mucosa_boundary_sha256": "",
            "mucosa_manifest_sha256": "",
            "mucosa_artifact_sha256": "",
        }
        if not source_path.is_file():
            row["invalid_wsi"] = True
        else:
            try:
                from adenoma_agent.wsi import WSIReader

                with WSIReader(source_path) as reader:
                    reader.require_physical_metadata()
                    row["mpp_x"] = float(reader.mpp_x)
                    row["mpp_y"] = float(reader.mpp_y)
                    row["base_magnification"] = float(reader.base_magnification)
                    row["source_backend"] = reader.backend_name
                    row.update(_source_identity(source_path))
            except Exception as exc:
                row["invalid_physical_metadata"] = True
                row["physical_metadata_error"] = str(exc)
        artifact = _mucosa_artifact(baseline_root, case_alias, minimum_coverage)
        row["mucosa_status"] = artifact["status"]
        row["mucosa_boundary_exists"] = artifact["boundary_exists"]
        row["valid_5x_manifest"] = bool(artifact["manifest_exists"] and artifact["status"] == "complete" and artifact["errors"] == 0)
        row["eligible_candidate_roi"] = bool(artifact["eligible_rows"])
        row["eligible_roi_count"] = int(artifact["eligible_rows"])
        row["mucosa_boundary_sha256"] = artifact["boundary_sha256"] or ""
        row["mucosa_manifest_sha256"] = artifact["manifest_sha256"] or ""
        row["mucosa_artifact_sha256"] = artifact["artifact_sha256"] or ""
        rows.append(row)

    audit_dir = paths["audit_dir"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_alias",
        "diagnosis",
        "family_id",
        "source_path",
        "source_exists",
        "invalid_wsi",
        "invalid_physical_metadata",
        "physical_metadata_error",
        "mpp_x",
        "mpp_y",
        "base_magnification",
        "source_backend",
        "source_wsi_size_bytes",
        "source_wsi_mtime_ns",
        "source_wsi_identity_sha256",
        "mucosa_status",
        "mucosa_boundary_exists",
        "valid_5x_manifest",
        "eligible_candidate_roi",
        "eligible_roi_count",
        "mucosa_boundary_sha256",
        "mucosa_manifest_sha256",
        "mucosa_artifact_sha256",
    ]
    write_csv(audit_dir / "villous_recruitment_audit.csv", rows, fields)
    summary = {
        "N_SOURCE_VILLOUS_WSI": len(rows),
        "N_LABEL_VALID_WSI": sum(bool(row["diagnosis"]) for row in rows),
        "N_WSI_FILE_EXISTS": sum(bool(row["source_exists"]) for row in rows),
        "N_EXISTING_MUCOSA": sum(row["mucosa_status"] == "complete" and row["mucosa_boundary_exists"] for row in rows),
        "N_MISSING_MUCOSA": sum(not row["mucosa_boundary_exists"] for row in rows),
        "N_INVALID_WSI": sum(bool(row["invalid_wsi"]) for row in rows),
        "N_INVALID_PHYSICAL_METADATA": sum(bool(row["invalid_physical_metadata"]) for row in rows),
        "N_WITH_VALID_5X": sum(bool(row["valid_5x_manifest"]) for row in rows),
        "N_ELIGIBLE_AFTER_CURRENT_GATES": sum(bool(row["eligible_candidate_roi"]) for row in rows),
        "diagnosis_counts": dict(sorted(Counter(str(row["diagnosis"]) for row in rows).items())),
        "family_count": len({str(row["family_id"]) for row in rows if row["family_id"]}),
        "source_code": source_code,
        "minimum_mucosa_coverage": minimum_coverage,
        "audit_csv_sha256": sha256_file(audit_dir / "villous_recruitment_audit.csv"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    missing = summary["N_MISSING_MUCOSA"]
    summary["conclusion"] = (
        "villous source support is sufficient for selective POC backfill"
        if summary["N_SOURCE_VILLOUS_WSI"] >= int(config.get("mucosa_backfill", {}).get("target_villous_wsi", 70))
        else "insufficient villous source WSI"
    )
    report = """# Villous Recruitment Audit

This audit was performed before any new Mucosa processing. Diagnosis is used
only to define the recruitment pool; it is not ROI architecture ground truth.

| Metric | Count |
|---|---:|
| N_SOURCE_VILLOUS_WSI | {N_SOURCE_VILLOUS_WSI} |
| N_LABEL_VALID_WSI | {N_LABEL_VALID_WSI} |
| N_WSI_FILE_EXISTS | {N_WSI_FILE_EXISTS} |
| N_EXISTING_MUCOSA | {N_EXISTING_MUCOSA} |
| N_MISSING_MUCOSA | {N_MISSING_MUCOSA} |
| N_INVALID_WSI | {N_INVALID_WSI} |
| N_INVALID_PHYSICAL_METADATA | {N_INVALID_PHYSICAL_METADATA} |
| N_WITH_VALID_5X | {N_WITH_VALID_5X} |
| N_ELIGIBLE_AFTER_CURRENT_GATES | {N_ELIGIBLE_AFTER_CURRENT_GATES} |

Diagnosis counts: `{diagnosis_counts}`  
Unique specimen families: `{family_count}`  
Conclusion: **{conclusion}**

Existing Mucosa artifacts were inspected read-only. Any missing cases are
eligible only for isolated `mucosa_backfill/` processing under the POC root.
""".format(**summary)
    write_text(audit_dir / "villous_recruitment_audit.md", report)
    summary["audit_report"] = str((audit_dir / "villous_recruitment_audit.md").resolve())
    write_json(audit_dir / "villous_recruitment_audit_summary.json", summary)
    return {"rows": rows, "summary": summary}


def _selected_villous_cases(config: Mapping[str, Any], audit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    target = int(config.get("mucosa_backfill", {}).get("target_villous_wsi", 70))
    seed = int(config.get("recruitment", {}).get("recruitment_seed", 20260823))
    rows = list(audit["rows"])
    valid = [row for row in rows if row["source_exists"] and not row["invalid_wsi"] and not row["invalid_physical_metadata"]]
    valid.sort(key=lambda row: sha256_payload({"seed": seed, "family_id": row["family_id"], "case_alias": row["case_alias"]}))
    selected = []
    families = set()
    for row in valid:
        if row["family_id"] in families:
            continue
        selected.append(row)
        families.add(row["family_id"])
        if len(selected) >= target:
            break
    if len(selected) < target:
        selected_ids = {row["case_alias"] for row in selected}
        for row in valid:
            if row["case_alias"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["case_alias"])
            if len(selected) >= target:
                break
    return selected


def _endpoint_ready(url: str) -> bool:
    try:
        import requests

        endpoint = str(url).rstrip("/")
        if endpoint.endswith("/predict"):
            endpoint = endpoint[: -len("/predict")]
        response = requests.get(endpoint + "/openapi.json", timeout=3)
        return bool(response.ok)
    except Exception:
        return False


def _run_mucosa_shard(args: Mapping[str, Any]) -> Mapping[str, Any]:
    """Worker entry point; each shard owns a disjoint case index subset."""

    from adenoma_agent.architecture_baselines.mucosa import run_mucosa_per_slide

    log_path = Path(str(args["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle, redirect_stdout(log_handle):
        return run_mucosa_per_slide(
            case_paths_jsonl=Path(str(args["case_paths"])),
            output_root=Path(str(args["output_root"])),
            pathprism_url=str(args["pathprism_url"]),
            batch_size=int(args["batch_size"]),
            timeout_seconds=int(args["timeout_seconds"]),
            base_magnification=float(args["base_magnification"]),
            mpp=float(args["mpp"]),
            min_tissue_coverage=float(args["min_tissue_coverage"]),
            mask_downsample=float(args["mask_downsample"]),
            mucosa_threshold=float(args["mucosa_threshold"]),
            resume=True,
            limit_cases=0,
            skip_qc_panel=True,
            source_code="YX",
            shard_count=int(args["shard_count"]),
            shard_index=int(args["shard_index"]),
        )


def selective_mucosa_backfill(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    audit: Mapping[str, Any],
    pathprism_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Reuse valid artifacts and selectively process missing villous cases."""

    backfill = dict(config.get("mucosa_backfill") or {})
    output_root = paths["mucosa_backfill_root"]
    provenance_root = output_root / "provenance"
    logs_root = output_root / "logs"
    failures_root = output_root / "failures"
    output_root.mkdir(parents=True, exist_ok=True)
    provenance_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    failures_root.mkdir(parents=True, exist_ok=True)
    selected = _selected_villous_cases(config, audit)
    if len(selected) < int(backfill.get("target_villous_wsi", 70)):
        summary = {
            "schema_version": MUCOSA_SCHEMA_VERSION,
            "status": "insufficient_source_cases",
            "requested": int(backfill.get("target_villous_wsi", 70)),
            "selected": len(selected),
            "reused": 0,
            "newly_processed": 0,
            "failed": [],
        }
        write_json(output_root / "summary.json", summary)
        error = BackfillDependencyBlocked("INSUFFICIENT_SOURCE_CASES: fewer than target villous WSI are valid", [summary])
        error.gate = "INSUFFICIENT_SOURCE_CASES"
        raise error

    case_index = {str(row["case_alias"]): dict(row) for row in iter_jsonl(paths["case_paths"])}
    selected_case_rows = [case_index[str(row["case_alias"])] for row in selected]
    write_jsonl(provenance_root / "selected_case_paths.jsonl", selected_case_rows)
    write_json(
        provenance_root / "selected_case_list.json",
        {
            "schema_version": MUCOSA_SCHEMA_VERSION,
            "target_villous_wsi": int(backfill.get("target_villous_wsi", 70)),
            "selected_case_aliases": [row["case_alias"] for row in selected],
            "selected_case_list_sha256": sha256_payload([row["case_alias"] for row in selected]),
        },
    )
    existing_root = paths["per_slide_mucosa_root"]
    reused = []
    missing = []
    minimum = float(config.get("recruitment", {}).get("minimum_mucosa_coverage", 0.60))
    for row in selected:
        artifact = _mucosa_artifact(existing_root, str(row["case_alias"]), minimum)
        if not artifact["eligible_rows"]:
            artifact = _mucosa_artifact(output_root / "mucosa_by_slide", str(row["case_alias"]), minimum)
        if artifact["eligible_rows"]:
            reused.append({"case_alias": row["case_alias"], "artifact": artifact})
        else:
            missing.append(row)
    write_jsonl(provenance_root / "reused_cases.jsonl", reused)

    checkpoint_records = {}
    for key in ("uni_weights_path", "prismnet_path"):
        checkpoint_path = Path(str(backfill.get(key, ""))).resolve()
        if not checkpoint_path.is_file():
            summary = {
                "schema_version": MUCOSA_SCHEMA_VERSION,
                "status": "dependency_blocked",
                "blocking_reason": "MISSING_CHECKPOINT",
                "missing_checkpoint": str(checkpoint_path),
                "requested": len(selected),
                "reused": len(reused),
                "newly_processed": 0,
                "failed": [],
            }
            write_json(output_root / "summary.json", summary)
            raise BackfillDependencyBlocked("MUCOSA_BACKFILL_DEPENDENCY_BLOCKED: checkpoint missing", [summary])
        checkpoint_records[key] = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        }
    url = str(pathprism_url or backfill.get("pathprism_url", "http://127.0.0.1:8400/predict"))
    if missing and not _endpoint_ready(url):
        summary = {
            "schema_version": MUCOSA_SCHEMA_VERSION,
            "status": "dependency_blocked",
            "blocking_reason": "PATHPRISM_ENDPOINT_UNAVAILABLE",
            "pathprism_url": url,
            "requested": len(selected),
            "reused": len(reused),
            "missing_mucosa": len(missing),
            "newly_processed": 0,
            "failed": [],
            "checkpoints": checkpoint_records,
        }
        write_json(output_root / "summary.json", summary)
        raise BackfillDependencyBlocked("MUCOSA_BACKFILL_DEPENDENCY_BLOCKED: UNI/PathPrism endpoint unavailable", [summary])

    new_summary = {
        "status": "not_needed" if not missing else "running",
        "requested": len(selected),
        "reused": len(reused),
        "missing_mucosa": len(missing),
        "newly_processed": 0,
        "failed": [],
    }
    reused_baseline_count = sum(
        str(item.get("artifact", {}).get("mucosa_root", "")) == str(existing_root.resolve())
        for item in reused
    )
    reused_poc_count = len(reused) - reused_baseline_count
    historical_poc_complete = sum(
        bool(_mucosa_artifact(output_root / "mucosa_by_slide", str(row["case_alias"]), minimum)["eligible_rows"])
        for row in selected
    )
    if missing:
        from adenoma_agent.architecture_baselines.mucosa import run_mucosa_per_slide

        selective_case_path = provenance_root / "missing_case_paths.jsonl"
        write_jsonl(selective_case_path, [case_index[str(row["case_alias"])] for row in missing])
        workers = max(1, min(int(backfill.get("workers", 3)), 4))
        worker_args = [
            {
                "case_paths": str(selective_case_path),
                "output_root": str(output_root / "mucosa_by_slide"),
                "pathprism_url": url,
                "batch_size": int(backfill.get("batch_size", 16)),
                "timeout_seconds": int(backfill.get("timeout_seconds", 240)),
                "base_magnification": float(backfill.get("base_magnification", 0.0)),
                "mpp": float(backfill.get("mpp", 0.0)),
                "min_tissue_coverage": float(backfill.get("min_tissue_coverage", 0.05)),
                "mask_downsample": float(backfill.get("mask_downsample", 32.0)),
                "mucosa_threshold": float(backfill.get("mucosa_threshold", 0.30)),
                "shard_count": workers,
                "shard_index": index,
                "log_path": str(logs_root / "selective_mucosa_shard_{0:03d}.log".format(index)),
            }
            for index in range(workers)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            shard_results = list(executor.map(_run_mucosa_shard, worker_args))
        completed = [row for result in shard_results for row in result.get("completed", [])]
        skipped = [row for result in shard_results for row in result.get("skipped_complete", [])]
        failed = [row for result in shard_results for row in result.get("failed", [])]
        new_summary["newly_processed"] = len(completed) + len(skipped)
        new_summary["failed"] = failed
        write_json(
            output_root / "batch_summary.json",
            {
                "schema_version": MUCOSA_SCHEMA_VERSION,
                "status": "complete" if not failed else "complete_with_failures",
                "requested_cases": len(missing),
                "completed": completed,
                "skipped_complete": skipped,
                "failed": failed,
                "shard_count": workers,
            },
        )
    else:
        write_json(output_root / "batch_summary.json", {"status": "not_needed", "requested_cases": 0, "completed": [], "failed": []})
        new_summary["newly_processed"] = historical_poc_complete

    provenance = {
        "schema_version": MUCOSA_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pipeline": "adenoma_agent.architecture_baselines.mucosa.run_mucosa_per_slide",
        "config_sha256": str(config.get("_config_sha256", "")),
        "pathprism_url": url,
        "device": str(backfill.get("device", "cuda:0")),
        "log_path": str((logs_root / "selective_mucosa.log").resolve()),
        "checkpoint_records": checkpoint_records,
        "mucosa_parameters": {
            key: backfill.get(key)
            for key in (
                "batch_size",
                "timeout_seconds",
                "base_magnification",
                "mpp",
                "min_tissue_coverage",
                "mask_downsample",
                "mucosa_threshold",
            )
        },
        "selected_case_list_sha256": sha256_file(provenance_root / "selected_case_paths.jsonl"),
        "reused_case_count": len(reused),
        "reused_baseline_case_count": reused_baseline_count,
        "reused_poc_case_count": reused_poc_count,
        "missing_case_count": len(missing),
        "newly_processed": new_summary["newly_processed"],
        "failures": new_summary["failed"],
    }
    eligible_after = []
    for row in selected:
        for root in (existing_root, output_root / "mucosa_by_slide"):
            artifact = _mucosa_artifact(root, str(row["case_alias"]), minimum)
            if artifact["eligible_rows"]:
                eligible_after.append(str(row["case_alias"]))
                break
    provenance["eligible_after_backfill"] = len(eligible_after)
    provenance["eligible_case_aliases_sha256"] = sha256_payload(sorted(eligible_after))
    write_json(provenance_root / "backfill_provenance.json", provenance)
    summary = {
        "schema_version": MUCOSA_SCHEMA_VERSION,
        "status": "complete_with_failures" if new_summary["failed"] else "complete",
        "requested": len(selected),
        "reused": len(reused),
        "reused_baseline": reused_baseline_count,
        "reused_poc_backfill": reused_poc_count,
        "newly_processed": int(new_summary["newly_processed"]),
        "failed": list(new_summary["failed"]),
        "eligible_after_backfill": len(eligible_after),
        "eligible_after_backfill_target_met": len(eligible_after) >= len(selected),
        "selected_case_list_sha256": provenance["selected_case_list_sha256"],
        "provenance_sha256": sha256_file(provenance_root / "backfill_provenance.json"),
    }
    write_json(output_root / "summary.json", summary)
    return {"selected": selected, "reused": reused, "missing": missing, "summary": summary}


def merge_mucosa_roots(paths: Mapping[str, Path]) -> List[Path]:
    roots = [paths["per_slide_mucosa_root"], paths["mucosa_backfill_root"] / "mucosa_by_slide"]
    return [root for root in roots if root.is_dir()]
