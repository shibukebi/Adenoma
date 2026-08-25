"""Sanitized 5x manifest construction and validation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Set

from .io import iter_jsonl, read_json, sha256_file, sha256_payload, write_json, write_jsonl


MANIFEST_SCHEMA_VERSION = "architecture_5x_manifest_v1"


def _bbox(value: Any) -> List[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("level0_bbox must contain four coordinates")
    result = [int(item) for item in value]
    if result[0] < 0 or result[1] < 0 or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("level0_bbox must be ordered and non-negative")
    return result


def _finite_positive(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError("{0} must be finite and positive".format(name))
    return parsed


def validate_patch_row(row: Mapping[str, Any], require_physical: bool = True) -> Mapping[str, Any]:
    case_alias = str(row.get("case_alias", "")).strip()
    patch_id = str(row.get("patch_id", "")).strip()
    if not case_alias or not patch_id:
        raise ValueError("case_alias and patch_id are required")
    _bbox(row.get("level0_bbox"))
    grid_index = row.get("grid_index")
    if not isinstance(grid_index, (list, tuple)) or len(grid_index) != 2:
        raise ValueError("grid_index must contain row and column")
    coverage = float(row.get("mucosa_coverage", -1.0))
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError("mucosa_coverage must be in [0,1]")
    target = str(row.get("target_magnification", "")).lower().rstrip("x")
    if abs(float(target) - 5.0) > 1e-6:
        raise ValueError("Only canonical 5x manifest rows are accepted")
    if require_physical:
        physical = row.get("physical_provenance")
        if not isinstance(physical, Mapping):
            raise ValueError("physical_provenance is required")
        requested = _finite_positive(physical.get("requested_magnification"), "requested_magnification")
        if abs(requested - 5.0) > 1e-6:
            raise ValueError("physical requested_magnification must be 5x")
        _finite_positive(physical.get("mpp_x"), "mpp_x")
        _finite_positive(physical.get("mpp_y"), "mpp_y")
        _finite_positive(physical.get("base_magnification"), "base_magnification")
        output = physical.get("output_pixel_dimensions")
        if list(output or []) != [256, 256]:
            raise ValueError("canonical 5x output_pixel_dimensions must be [256,256]")
        fov = physical.get("fov_microns")
        if not isinstance(fov, (list, tuple)) or len(fov) != 2:
            raise ValueError("fov_microns is required")
        _finite_positive(fov[0], "fov_microns[0]")
        _finite_positive(fov[1], "fov_microns[1]")
    return row


def validate_manifest(path: Path, require_physical: bool = True) -> Mapping[str, Any]:
    seen = set()
    per_case = Counter()
    coverages = []
    errors = []
    for index, row in enumerate(iter_jsonl(path), 1):
        try:
            validate_patch_row(row, require_physical=require_physical)
            key = (str(row["case_alias"]), str(row["patch_id"]))
            if key in seen:
                raise ValueError("duplicate case_alias/patch_id")
            seen.add(key)
            per_case[key[0]] += 1
            coverages.append(float(row["mucosa_coverage"]))
        except Exception as exc:
            errors.append({"row_number": index, "error": str(exc)})
    if errors:
        raise ValueError("Manifest validation failed: {0}".format(errors[:10]))
    if not seen:
        raise ValueError("Manifest contains no rows")
    ordered = sorted(coverages)
    return {
        "schema_version": "architecture_5x_manifest_validation_v1",
        "manifest": str(Path(path).resolve()),
        "manifest_sha256": sha256_file(path),
        "patches": len(seen),
        "cases": len(per_case),
        "patches_per_case": dict(sorted(per_case.items())),
        "coverage_min": ordered[0],
        "coverage_median": ordered[len(ordered) // 2],
        "coverage_max": ordered[-1],
        "coverage_ge_0_30": sum(value >= 0.30 for value in coverages),
        "coverage_ge_0_60": sum(value >= 0.60 for value in coverages),
    }


def _sanitized_patch(case_alias: str, source_row: Mapping[str, Any], source_manifest_hash: str) -> Mapping[str, Any]:
    grid = [int(value) for value in source_row.get("grid_index", [])]
    if len(grid) != 2:
        raise ValueError("Source 5x row has no valid grid_index")
    patch_id = "{0}__5x__r{1:05d}_c{2:05d}".format(case_alias, grid[0], grid[1])
    physical = dict(source_row.get("physical_provenance") or {})
    physical.pop("local_source_path", None)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case_alias": case_alias,
        "patch_id": patch_id,
        "grid_index": grid,
        "level0_bbox": [int(value) for value in source_row.get("level0_bbox", [])],
        "clipped_level0_bbox": [int(value) for value in source_row.get("clipped_level0_bbox", [])],
        "target_magnification": "5x",
        "mucosa_coverage": float(source_row.get("mucosa_coverage", 0.0)),
        "mean_uncertainty": source_row.get("mean_uncertainty"),
        "source_component_ids": [int(value) for value in source_row.get("source_component_ids", [])],
        "physical_provenance": physical,
        "source_manifest_sha256": source_manifest_hash,
        "source_row_sha256": sha256_payload(dict(source_row)),
    }


def merge_sanitized_manifests(
    case_paths_jsonl: Path,
    mucosa_root: Path,
    output_manifest: Path,
    minimum_mucosa_coverage: float = 0.30,
) -> Mapping[str, Any]:
    rows = []
    source_summaries = []
    for case in sorted(iter_jsonl(case_paths_jsonl), key=lambda item: str(item.get("case_alias", ""))):
        case_alias = str(case.get("case_alias", "")).strip()
        if not case_alias:
            raise ValueError("case_paths row is missing case_alias")
        source_path = Path(mucosa_root) / case_alias / "five_x_patch_manifest.jsonl"
        run_manifest_path = Path(mucosa_root) / case_alias / "manifest.json"
        if not source_path.is_file() or not run_manifest_path.is_file():
            continue
        run_manifest = read_json(run_manifest_path)
        if str(run_manifest.get("status", "")) != "complete":
            continue
        if int(run_manifest.get("counts", {}).get("errors", 0) or 0) != 0:
            continue
        source_hash = sha256_file(source_path)
        count = 0
        excluded = 0
        for source_row in iter_jsonl(source_path):
            if float(source_row.get("mucosa_coverage", 0.0)) < float(minimum_mucosa_coverage):
                excluded += 1
                continue
            row = _sanitized_patch(case_alias, source_row, source_hash)
            validate_patch_row(row, require_physical=True)
            rows.append(row)
            count += 1
        source_summaries.append(
            {
                "case_alias": case_alias,
                "patches": count,
                "excluded_below_minimum_coverage": excluded,
                "minimum_mucosa_coverage": float(minimum_mucosa_coverage),
                "source_manifest_sha256": source_hash,
            }
        )
    if not rows:
        raise RuntimeError("No completed per-slide Mucosa manifests were found")
    rows.sort(key=lambda item: (item["case_alias"], item["grid_index"][0], item["grid_index"][1]))
    write_jsonl(output_manifest, rows)
    validation = validate_manifest(output_manifest, require_physical=True)
    summary = {
        **validation,
        "minimum_mucosa_coverage": float(minimum_mucosa_coverage),
        "source_manifests": source_summaries,
        "source_case_index_sha256": sha256_file(case_paths_jsonl),
    }
    write_json(Path(output_manifest).with_suffix(".summary.json"), summary)
    return summary


def eligible_case_aliases(manifest_path: Path, minimum_coverage: float = 0.60) -> Set[str]:
    eligible = set()
    for row in iter_jsonl(manifest_path):
        if float(row.get("mucosa_coverage", 0.0)) >= float(minimum_coverage):
            eligible.add(str(row["case_alias"]))
    return eligible


def manifest_rows_by_case(manifest_path: Path, minimum_coverage: float = 0.0) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    grouped = defaultdict(list)
    for row in iter_jsonl(manifest_path):
        if float(row.get("mucosa_coverage", 0.0)) >= float(minimum_coverage):
            grouped[str(row["case_alias"])].append(row)
    return {key: tuple(value) for key, value in grouped.items()}
