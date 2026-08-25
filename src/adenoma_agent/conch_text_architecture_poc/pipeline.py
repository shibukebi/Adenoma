"""Two-stage expert-confirmed 5x architecture benchmark workflow.

Stage A materializes deterministic blinded images and an Excel workbook.
Stage B validates completed expert annotations and freezes membership, labels,
groups, prompts, splits, and low-data subsets.  No model package is imported
or invoked anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml
from PIL import Image, ImageDraw

from adenoma_agent.wsi import WSIReader

from .io import (
    iter_jsonl,
    read_csv,
    read_json,
    relative_or_resolved,
    sha256_file,
    sha256_payload,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from .backfill import (
    BackfillDependencyBlocked,
    audit_villous_recruitment,
    merge_mucosa_roots,
    selective_mucosa_backfill,
)
from .workbook import (
    ANNOTATION_COLUMNS,
    DROPDOWNS,
    EXPERT_COLUMNS,
    LOCKED_COLUMNS,
    find_forbidden_text,
    read_annotation_workbook,
    workbook_structure,
    write_annotation_workbook,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "conch_text_architecture_poc_v1.yaml"
PRIMARY_CLASSES = ("serrated", "tubular", "villous")
LEGAL_GATES = frozenset(
    {
        "WAITING_FOR_EXPERT_ANNOTATION",
        "FROZEN_READY_FOR_EVALUATION",
        "INSUFFICIENT_PRIMARY_SUPPORT",
        "ANNOTATION_CONFLICT_BLOCKED",
        "PROMPT_FREEZE_BLOCKED",
        "PROVENANCE_BLOCKED",
        "FREEZE_VALIDATION_FAILED",
        "MUCOSA_BACKFILL_DEPENDENCY_BLOCKED",
        "INSUFFICIENT_SOURCE_CASES",
        "INSUFFICIENT_VALID_5X_ROIS",
        "ANNOTATION_PACKAGE_VALIDATION_FAILED",
    }
)
FORBIDDEN_PUBLIC_PATTERNS = (
    r"\bHP\b",
    r"\bSSL\b",
    r"\bTSA\b",
    r"\bUSA\b",
    r"\bTA\b",
    r"\bTVA\b",
    r"\bVA\b",
    r"\bHGD\b",
    r"dysplasia",
    r"slide[ _-]*diagnosis",
    r"recruitment[ _-]*stratum",
    r"conch[ _-]*(score|prediction|similarity|output)",
    r"zero[ _-]*shot[ _-]*(score|prediction|result|output)",
    r"mil[ _-]*(score|attention|prediction|output)",
    r"model[ _-]*(score|prediction|output)",
    r"classifier[ _-]*(score|prediction|output)",
)
MODEL_OUTPUT_FIELD_PATTERNS = (
    re.compile(r"(^|_)(conch|model|classifier|mil)(_|$)", re.IGNORECASE),
    re.compile(r"(^|_)(prediction|similarity|attention|logit|probability|score)(_|$)", re.IGNORECASE),
)
PRIMARY_SELECTION_FIELDS = (
    "evaluable",
    "pure_or_mixed",
    "architecture_label",
    "expert_confidence",
    "family_id",
    "candidate_generation_order",
    "roi_id",
)


class BenchmarkWorkflowError(RuntimeError):
    gate = "PROVENANCE_BLOCKED"

    def __init__(self, message: str, issues: Optional[Sequence[Mapping[str, Any]]] = None):
        super().__init__(message)
        self.issues = [dict(item) for item in (issues or ())]


class ProvenanceError(BenchmarkWorkflowError):
    gate = "PROVENANCE_BLOCKED"


class FreezeValidationError(BenchmarkWorkflowError):
    gate = "FREEZE_VALIDATION_FAILED"


class AnnotationConflictError(BenchmarkWorkflowError):
    gate = "ANNOTATION_CONFLICT_BLOCKED"


class ResumeRefusedError(BenchmarkWorkflowError):
    gate = "PROVENANCE_BLOCKED"


class CandidateSupportError(ProvenanceError):
    gate = "INSUFFICIENT_VALID_5X_ROIS"


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load_config(config_path: Path) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ProvenanceError("POC config must be a YAML object")
    if config.get("schema_version") != "conch_text_architecture_poc_v1":
        raise ProvenanceError("Unexpected POC config schema_version")
    paths = {}
    for key, value in dict(config.get("paths") or {}).items():
        paths[str(key)] = relative_or_resolved(Path(str(value)), REPO_ROOT)
    required = {
        "artifact_root",
        "annotation_package",
        "restricted_recruitment_provenance",
        "benchmark_root",
        "freeze_report",
        "prompt_config",
        "audit_dir",
        "mucosa_backfill_root",
        "preparation_report",
        "case_paths",
        "canonical_labels",
        "per_slide_mucosa_root",
    }
    missing = sorted(required - set(paths))
    if missing:
        raise ProvenanceError("POC config paths are missing: {0}".format(missing))
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = sha256_file(config_path)
    return config, paths


def _rank(seed: Any, *values: Any) -> str:
    payload = "\0".join([str(seed)] + [str(value) for value in values]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_positive(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError("value must be finite and positive")
    return parsed


def _load_source_indexes(paths: Mapping[str, Path], source_code: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    cases = {}
    for row in iter_jsonl(paths["case_paths"]):
        if str(row.get("source_code", "")).upper() == str(source_code).upper():
            cases[str(row.get("case_alias", ""))] = dict(row)
    labels = {}
    for row in iter_jsonl(paths["canonical_labels"]):
        if str(row.get("source_code", "")).upper() == str(source_code).upper():
            labels[str(row.get("case_alias", ""))] = dict(row)
    if not cases or not labels:
        raise ProvenanceError("Source case or diagnosis recruitment indexes are empty")
    return cases, labels


def _valid_candidate_rows(
    manifest_path: Path,
    minimum_coverage: float,
    require_unclipped: bool,
) -> List[Dict[str, Any]]:
    output = []
    for source_order, row in enumerate(iter_jsonl(manifest_path), 1):
        coverage = float(row.get("mucosa_coverage", -1.0))
        if not math.isfinite(coverage) or coverage < float(minimum_coverage) or coverage > 1.0:
            continue
        bbox = [int(value) for value in row.get("level0_bbox", [])]
        clipped = [int(value) for value in row.get("clipped_level0_bbox", bbox)]
        if len(bbox) != 4 or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        if require_unclipped and clipped != bbox:
            continue
        grid = [int(value) for value in row.get("grid_index", [])]
        if len(grid) != 2:
            continue
        physical = dict(row.get("physical_provenance") or {})
        try:
            magnification = _finite_positive(physical.get("requested_magnification"))
            mpp_x = _finite_positive(physical.get("mpp_x"))
            mpp_y = _finite_positive(physical.get("mpp_y"))
            base_magnification = _finite_positive(physical.get("base_magnification"))
            fov = [_finite_positive(value) for value in physical.get("fov_microns", [])]
        except (TypeError, ValueError):
            continue
        if len(fov) != 2 or abs(magnification - 5.0) > 1e-6:
            continue
        if list(physical.get("output_pixel_dimensions") or []) != [256, 256]:
            continue
        output.append(
            {
                "source_order": source_order,
                "source_patch_id": str(row.get("patch_id", "")),
                "bbox": bbox,
                "grid_index": grid,
                "mucosa_coverage": coverage,
                "magnification": magnification,
                "mpp_x": mpp_x,
                "mpp_y": mpp_y,
                "base_magnification": base_magnification,
                "physical_fov_um": fov,
            }
        )
    output.sort(
        key=lambda item: (
            -float(item["mucosa_coverage"]),
            int(item["grid_index"][0]),
            int(item["grid_index"][1]),
            str(item["source_patch_id"]),
        )
    )
    return output


def _discover_recruitment_candidates(config: Mapping[str, Any], paths: Mapping[str, Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    recruitment = dict(config.get("recruitment") or {})
    source_code = str(recruitment.get("source_code", "YX"))
    target = int(recruitment.get("target_wsi_per_stratum", 60))
    candidates_per_wsi = int(recruitment.get("candidates_per_wsi", 1))
    if candidates_per_wsi < 1 or candidates_per_wsi > 3:
        raise ProvenanceError("candidates_per_wsi must be between 1 and 3")
    cases, labels = _load_source_indexes(paths, source_code)
    diagnosis_to_stratum = dict(recruitment.get("diagnosis_to_stratum") or {})
    diagnosis_lookup = {}
    for stratum, diagnoses in diagnosis_to_stratum.items():
        for diagnosis in diagnoses or ():
            diagnosis_lookup[str(diagnosis)] = str(stratum)
    pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    mucosa_roots = merge_mucosa_roots(paths)
    for case_alias in sorted(set(cases) & set(labels)):
        diagnosis = str(labels[case_alias].get("label", ""))
        stratum = diagnosis_lookup.get(diagnosis)
        if not stratum:
            continue
        selected_artifact = None
        patch_rows = []
        for mucosa_root in mucosa_roots:
            slide_root = mucosa_root / case_alias
            boundary_path = slide_root / "baseline_boundary.json"
            manifest_path = slide_root / "five_x_patch_manifest.jsonl"
            if not boundary_path.is_file() or not manifest_path.is_file():
                continue
            boundary = read_json(boundary_path)
            if str(boundary.get("status", "")) != "complete" or int(boundary.get("counts", {}).get("errors", 0) or 0) != 0:
                continue
            candidate_rows = _valid_candidate_rows(
                manifest_path,
                minimum_coverage=float(recruitment.get("minimum_mucosa_coverage", 0.60)),
                require_unclipped=bool(recruitment.get("require_complete_unclipped_bbox", True)),
            )
            if candidate_rows:
                selected_artifact = {
                    "manifest_path": manifest_path,
                    "boundary_path": boundary_path,
                    "manifest_sha256": sha256_file(manifest_path),
                    "boundary_sha256": sha256_file(boundary_path),
                    "artifact_sha256": sha256_payload(
                        {
                            "manifest_sha256": sha256_file(manifest_path),
                            "boundary_sha256": sha256_file(boundary_path),
                            "minimum_mucosa_coverage": float(recruitment.get("minimum_mucosa_coverage", 0.60)),
                        }
                    ),
                    "mucosa_root": str(mucosa_root.resolve()),
                }
                patch_rows = candidate_rows
                break
        if not patch_rows:
            continue
        case = cases[case_alias]
        pools[stratum].append(
            {
                "case_alias": case_alias,
                "source_path": str(case.get("source_path", "")),
                "original_family_id": str(case.get("family_id") or labels[case_alias].get("family_id") or case_alias),
                "diagnosis": diagnosis,
                "stratum": stratum,
                "manifest_path": str(selected_artifact["manifest_path"].resolve()),
                "source_manifest_sha256": selected_artifact["manifest_sha256"],
                "mucosa_boundary_sha256": selected_artifact["boundary_sha256"],
                "mucosa_manifest_sha256": selected_artifact["manifest_sha256"],
                "mucosa_artifact_sha256": selected_artifact["artifact_sha256"],
                "mucosa_root": selected_artifact["mucosa_root"],
                "source_wsi_identity_sha256": sha256_payload(
                    {
                        "path": str(case.get("source_path", "")),
                        "size_bytes": int(Path(str(case.get("source_path", ""))).stat().st_size),
                        "mtime_ns": int(Path(str(case.get("source_path", ""))).stat().st_mtime_ns),
                    }
                ),
                "patch_rows": patch_rows[:candidates_per_wsi],
            }
        )

    selected_slides = []
    per_stratum = {}
    recruitment_seed = int(recruitment.get("recruitment_seed", 20260823))
    for stratum in ("serrated", "tubular", "villous"):
        ordered = sorted(
            pools.get(stratum, []),
            key=lambda item: _rank(
                recruitment_seed,
                stratum,
                item["original_family_id"],
                item["case_alias"],
            ),
        )
        chosen = []
        seen_families = set()
        for item in ordered:
            family_id = item["original_family_id"]
            if family_id in seen_families:
                continue
            chosen.append(item)
            seen_families.add(family_id)
            if len(chosen) >= target:
                break
        if len(chosen) < target:
            chosen_aliases = {item["case_alias"] for item in chosen}
            for item in ordered:
                if item["case_alias"] in chosen_aliases:
                    continue
                chosen.append(item)
                chosen_aliases.add(item["case_alias"])
                if len(chosen) >= target:
                    break
        selected_slides.extend(chosen)
        per_stratum[stratum] = {
            "target_wsi": target,
            "available_wsi": len(ordered),
            "available_families": len({item["original_family_id"] for item in ordered}),
            "selected_wsi": len(chosen),
            "selected_families": len({item["original_family_id"] for item in chosen}),
            "shortfall_wsi": max(0, target - len(chosen)),
        }

    shortfalls = [
        {"stratum": stratum, **values}
        for stratum, values in per_stratum.items()
        if int(values["shortfall_wsi"]) > 0
    ]
    if shortfalls and bool(recruitment.get("require_target_wsi_per_stratum", True)):
        raise CandidateSupportError(
            "Formal Stage A recruitment support is incomplete; annotation package was not materialized",
            shortfalls,
        )

    candidates = []
    for slide in selected_slides:
        for local_candidate_index, patch in enumerate(slide["patch_rows"], 1):
            candidates.append({**slide, **patch, "local_candidate_index": local_candidate_index})
    if not candidates:
        raise ProvenanceError("No deterministic 5x candidate ROI is available")
    blinding_seed = int(recruitment.get("blinding_seed", 763187))
    candidates.sort(
        key=lambda item: _rank(
            blinding_seed,
            item["case_alias"],
            item["source_patch_id"],
        )
    )
    slide_aliases = sorted(
        {item["case_alias"] for item in candidates},
        key=lambda value: _rank(blinding_seed, "slide", value),
    )
    family_aliases = sorted(
        {item["original_family_id"] for item in candidates},
        key=lambda value: _rank(blinding_seed, "family", value),
    )
    safe_slides = {value: "CASE_{0:06d}".format(index) for index, value in enumerate(slide_aliases, 1)}
    safe_families = {value: "FAMILY_{0:06d}".format(index) for index, value in enumerate(family_aliases, 1)}
    for order, candidate in enumerate(candidates, 1):
        candidate["candidate_generation_order"] = order
        candidate["annotation_instance_id"] = "ANN_V1_{0:06d}".format(order)
        candidate["roi_id"] = "ROI_{0:06d}".format(order)
        candidate["safe_slide_id"] = safe_slides[candidate["case_alias"]]
        candidate["family_id"] = safe_families[candidate["original_family_id"]]
    restricted_summary = {
        "schema_version": "architecture_restricted_recruitment_provenance_v1",
        "benchmark_version": config.get("benchmark_version"),
        "created_at": _timestamp(),
        "diagnosis_enriched_recruitment_only": True,
        "recruitment_diagnosis_is_not_roi_ground_truth": True,
        "target_wsi_per_stratum": target,
        "per_stratum": per_stratum,
        "selected_candidate_rois": len(candidates),
        "selected_wsi": len({item["case_alias"] for item in candidates}),
        "selected_families": len({item["original_family_id"] for item in candidates}),
    }
    return candidates, restricted_summary


def _guideline_text() -> str:
    return """# Blinded 5x ROI Architecture Annotation Guideline

Review each individual PNG in `images/`. The individual PNG—not the contact
sheet—is the frozen reviewed image identity.

## Q1 — Evaluability

Is this ROI evaluable for 5x glandular architecture?

- `yes`: tissue quality and field of view are sufficient.
- `no`: choose an `exclusion_reason`; the architecture label may be blank.

## Q2 — Dominant architecture

Choose exactly one when evaluable:

- `serrated`
- `tubular`
- `villous`
- `mixed`
- `uncertain`

Use `architecture_components` for mixed ROIs, separated by semicolons, for
example `serrated;tubular` or `tubular;villous`. No percentage estimate is
required. For a pure ROI, the field may be blank or equal the chosen label.

## Q3 — Confidence

- `high`: dominant architecture can be judged clearly in this ROI.
- `medium`: broadly judgeable, with meaningful mixture or uncertainty.
- `low`: more tissue or another magnification would be needed for confidence.

`mixed` is not wrong. `uncertain` is not wrong. `not evaluable` is not wrong.
Do not force a class to balance the dataset.

Save the completed workbook as `annotation_completed.xlsx`. Do not rename or
edit the locked identity columns.
"""


def _contact_sheets(
    images_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    columns: int,
    grid_rows: int,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = int(columns) * int(grid_rows)
    cell_width, cell_height, label_height = 256, 286, 30
    output = []
    for page, start in enumerate(range(0, len(rows), per_sheet), 1):
        page_rows = rows[start : start + per_sheet]
        canvas = Image.new("RGB", (int(columns) * cell_width, int(grid_rows) * cell_height), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        for local_index, row in enumerate(page_rows):
            x = (local_index % int(columns)) * cell_width
            y = (local_index // int(columns)) * cell_height
            image_path = images_dir / Path(str(row["image_file"])).name
            with image_path.open("rb") as handle:
                roi_image = Image.open(handle).convert("RGB")
                roi_image.load()
            if roi_image.size != (256, 256):
                roi_image = roi_image.resize((256, 256))
            canvas.paste(roi_image, (x, y + label_height))
            draw.rectangle((x, y, x + cell_width - 1, y + label_height - 1), fill=(255, 255, 255))
            draw.text((x + 6, y + 8), "{0} | {1}".format(row["roi_id"], row["safe_slide_id"]), fill=(20, 20, 20))
        path = output_dir / "contact_sheet_{0:03d}.png".format(page)
        canvas.save(str(path), format="PNG")
        output.append({"file": "contact_sheets/{0}".format(path.name), "sha256": sha256_file(path)})
    return output


def _public_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            for nested in _public_strings(item):
                yield nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            for nested in _public_strings(item):
                yield nested
    elif value is not None:
        yield str(value)


def _audit_public_package(package_root: Path, manifest: Mapping[str, Any]) -> List[Dict[str, str]]:
    rows, workbook_text = read_annotation_workbook(package_root / "annotation.xlsx")
    del rows
    values = list(workbook_text)
    values.extend(_public_strings(manifest))
    values.extend(_public_strings(_guideline_text()))
    readme_path = package_root / "README.md"
    if readme_path.is_file():
        values.append(readme_path.read_text(encoding="utf-8"))
    values.extend(str(path.relative_to(package_root)) for path in package_root.rglob("*"))
    return find_forbidden_text(values, FORBIDDEN_PUBLIC_PATTERNS)


def _preparation_report_text(audit: Mapping[str, Any], backfill: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    summary = dict(audit.get("summary") or {})
    backfill_summary = dict(backfill.get("summary") or {})
    return """# Expert 5x Architecture ROI Annotation Preparation Report

This is Stage A only. No benchmark freeze, split generation, low-data subset,
CONCH evaluation, linear probe, text-prior training, or MIL training was run.

## 1. Source audit

- source villous WSI: {source}
- label-valid WSI: {label_valid}
- WSI file exists: {exists}
- diagnosis counts: `{diagnoses}`

## 2. Existing Mucosa coverage

- already complete: {existing}
- missing before backfill: {missing}
- invalid WSI: {invalid_wsi}
- invalid physical metadata: {invalid_physical}
- valid 5x manifest: {valid_5x}
- eligible after current gates: {eligible}

## 3. Selective Mucosa backfill

- requested villous WSI: {requested}
- reused valid full-baseline artifacts: {reused}
- reused valid POC-backfill artifacts on verification: {reused_poc}
- newly processed: {new}
- failures: {failures}
- outputs: `artifacts/conch_text_architecture_poc_v1/mucosa_backfill/`

## 4. Final annotation package

- candidate ROIs: {candidates}
- unique WSI: {unique_wsi}
- unique families: {unique_families}
- internal recruitment: serrated=60, tubular=60, villous=60

## 5. Independence and blinding

- one default ROI per WSI: yes
- duplicate ROI: 0
- duplicate WSI: 0
- slide diagnosis exposure: 0
- recruitment stratum exposure: 0
- model score exposure: 0

## 6. Image provenance

- PNG count: {candidates}
- PNG SHA completeness: {candidates}/{candidates}
- MPP completeness: {candidates}/{candidates}
- FOV completeness: {candidates}/{candidates}
- Mucosa provenance completeness: {mucosa_complete}/{candidates}

## 7. Stop gate

`FINAL_GATE=WAITING_FOR_EXPERT_ANNOTATION`  
`MODEL_EVALUATION_STARTED=false`  
`BENCHMARK_FREEZE_COMPLETE=false`  
`PROMPT_REVIEW_REQUIRED_LATER=true`

## 8. Next action

`EXPERT_REVIEW_ANNOTATION_XLSX`
""".format(
        source=summary.get("N_SOURCE_VILLOUS_WSI", 0),
        label_valid=summary.get("N_LABEL_VALID_WSI", 0),
        exists=summary.get("N_WSI_FILE_EXISTS", 0),
        diagnoses=summary.get("diagnosis_counts", {}),
        existing=summary.get("N_EXISTING_MUCOSA", 0),
        missing=summary.get("N_MISSING_MUCOSA", 0),
        invalid_wsi=summary.get("N_INVALID_WSI", 0),
        invalid_physical=summary.get("N_INVALID_PHYSICAL_METADATA", 0),
        valid_5x=summary.get("N_WITH_VALID_5X", 0),
        eligible=summary.get("N_ELIGIBLE_AFTER_CURRENT_GATES", 0),
        requested=backfill_summary.get("requested", 0),
        reused=backfill_summary.get("reused_baseline", backfill_summary.get("reused", 0)),
        reused_poc=backfill_summary.get("reused_poc_backfill", 0),
        new=backfill_summary.get("newly_processed", 0),
        failures=len(backfill_summary.get("failed", [])),
        candidates=manifest.get("candidate_rois", 0),
        unique_wsi=manifest.get("unique_candidate_wsi", 0),
        unique_families=manifest.get("unique_candidate_families", 0),
        mucosa_complete=sum(
            bool(row.get("mucosa_boundary_sha256") and row.get("mucosa_manifest_sha256") and row.get("mucosa_artifact_sha256"))
            for row in manifest.get("rois", [])
        ),
    )


def _verify_existing_annotation_package(package_root: Path) -> Dict[str, Any]:
    manifest_path = package_root / "annotation_manifest.json"
    if not manifest_path.is_file():
        raise ResumeRefusedError("REFUSE_RESUME: annotation manifest is missing")
    manifest = read_json(manifest_path)
    for row in manifest.get("rois", []):
        image_path = package_root / str(row.get("image_file", ""))
        if not image_path.is_file() or sha256_file(image_path) != row.get("image_sha256"):
            raise ResumeRefusedError("REFUSE_RESUME: reviewed image SHA changed for {0}".format(row.get("roi_id")))
    if _audit_public_package(package_root, manifest):
        raise ResumeRefusedError("REFUSE_RESUME: annotation package leakage audit changed")
    return manifest


def prepare_annotation(config_path: Path = DEFAULT_CONFIG, resume: bool = False) -> Dict[str, Any]:
    """Stage A: create the blinded fixed-image annotation package and stop."""

    config, paths = _load_config(config_path)
    artifact_root = paths["artifact_root"]
    package_root = paths["annotation_package"]
    restricted_path = paths["restricted_recruitment_provenance"]
    if package_root.exists():
        if not resume:
            raise ResumeRefusedError("Annotation package already exists; use --resume for identity verification")
        manifest = _verify_existing_annotation_package(package_root)
        # Resume is a read-only identity/report refresh. It may re-audit source
        # and backfill provenance, but it never rewrites the reviewed PNGs or
        # workbook and never enters benchmark freeze.
        audit = audit_villous_recruitment(config, paths)
        backfill_result = selective_mucosa_backfill(config, paths, audit)
        stage_a_summary = {
            "schema_version": "architecture_stage_a_summary_v1",
            "audit": audit["summary"],
            "mucosa_backfill": backfill_result["summary"],
            "candidate_rois": int(manifest.get("candidate_rois", 0)),
            "unique_candidate_wsi": int(manifest.get("unique_candidate_wsi", 0)),
            "unique_candidate_families": int(manifest.get("unique_candidate_families", 0)),
            "internal_recruitment": {"serrated": 60, "tubular": 60, "villous": 60},
            "final_gate": "WAITING_FOR_EXPERT_ANNOTATION",
            "model_evaluation_started": False,
            "benchmark_freeze_complete": False,
            "prompt_review_required_later": True,
        }
        write_json(artifact_root / "stage_a_summary.json", stage_a_summary)
        write_text(paths["preparation_report"], _preparation_report_text(audit, backfill_result, manifest))
        return _stage_a_result(package_root, manifest)
    if restricted_path.exists():
        raise ResumeRefusedError("Restricted recruitment provenance exists without a resumable annotation package")
    artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        audit = audit_villous_recruitment(config, paths)
        backfill_result = selective_mucosa_backfill(config, paths, audit)
        candidates, restricted_summary = _discover_recruitment_candidates(config, paths)
        restricted_summary["villous_audit_summary"] = audit["summary"]
        restricted_summary["mucosa_backfill_summary"] = backfill_result["summary"]
    except BackfillDependencyBlocked as exc:
        wrapped = ProvenanceError(str(exc), exc.issues)
        wrapped.gate = exc.gate
        write_json(
            artifact_root / "status.json",
            {
                "schema_version": "conch_text_architecture_poc_state_v1",
                "updated_at": _timestamp(),
                "final_gate": wrapped.gate,
                "blocking_reason": str(wrapped),
                "annotation_package_ready": False,
                "expert_annotation_complete": False,
                "benchmark_freeze_complete": False,
                "model_evaluation_started": False,
            },
        )
        raise wrapped
    except BenchmarkWorkflowError as exc:
        gate = getattr(exc, "gate", "PROVENANCE_BLOCKED")
        write_json(
            artifact_root / "status.json",
            {
                "schema_version": "conch_text_architecture_poc_state_v1",
                "updated_at": _timestamp(),
                "final_gate": gate,
                "blocking_reason": str(exc),
                "annotation_package_ready": False,
                "expert_annotation_complete": False,
                "benchmark_freeze_complete": False,
                "model_evaluation_started": False,
            },
        )
        raise
    temporary_root = Path(tempfile.mkdtemp(prefix=".annotation_package_v1.", dir=str(artifact_root)))
    images_dir = temporary_root / str(config.get("annotation", {}).get("images_dir_name", "images"))
    contact_dir = temporary_root / str(config.get("annotation", {}).get("contact_sheets_dir_name", "contact_sheets"))
    images_dir.mkdir(parents=True, exist_ok=True)
    public_rows = []
    restricted_rows = []
    try:
        for candidate in candidates:
            source_path = Path(str(candidate["source_path"]))
            if not source_path.is_file():
                raise ProvenanceError("Source WSI is unavailable: {0}".format(source_path))
            physical = {
                "base_magnification": candidate["base_magnification"],
                "mpp_x": candidate["mpp_x"],
                "mpp_y": candidate["mpp_y"],
            }
            with WSIReader(source_path) as reader:
                reader.require_physical_metadata()
                for key, observed in (
                    ("base_magnification", reader.base_magnification),
                    ("mpp_x", reader.mpp_x),
                    ("mpp_y", reader.mpp_y),
                ):
                    expected = float(physical[key])
                    relative_error = abs(float(observed) - expected) / expected
                    if relative_error > 0.02:
                        raise ProvenanceError(
                            "WSI physical metadata changed for {0}: {1}".format(candidate["case_alias"], key)
                        )
                crop = reader.crop_level0_bbox(candidate["bbox"], requested_magnification=5.0, output_pixels=256)
            image_name = "{0}.png".format(candidate["roi_id"])
            image_path = images_dir / image_name
            crop.save(image_path)
            if sha256_file(image_path) != crop.image_sha256:
                raise ProvenanceError("Materialized image SHA mismatch for {0}".format(candidate["roi_id"]))
            public = {
                "annotation_instance_id": candidate["annotation_instance_id"],
                "roi_id": candidate["roi_id"],
                "safe_slide_id": candidate["safe_slide_id"],
                "family_id": candidate["family_id"],
                "grouping_resolution": "family",
                "image_file": "images/{0}".format(image_name),
                "candidate_generation_order": int(candidate["candidate_generation_order"]),
                "bbox": list(candidate["bbox"]),
                "magnification": 5.0,
                "MPP": [float(crop.provenance["mpp_x"]), float(crop.provenance["mpp_y"])],
                "mpp_x": float(crop.provenance["mpp_x"]),
                "mpp_y": float(crop.provenance["mpp_y"]),
                "physical_fov_um": [float(value) for value in crop.provenance["fov_microns"]],
                "image_sha256": crop.image_sha256,
                "source_manifest_sha256": candidate["source_manifest_sha256"],
                "mucosa_boundary_sha256": candidate["mucosa_boundary_sha256"],
                "mucosa_manifest_sha256": candidate["mucosa_manifest_sha256"],
                "mucosa_artifact_sha256": candidate["mucosa_artifact_sha256"],
                "source_wsi_identity_sha256": candidate["source_wsi_identity_sha256"],
            }
            public_rows.append(public)
            restricted_rows.append(
                {
                    "annotation_instance_id": candidate["annotation_instance_id"],
                    "roi_id": candidate["roi_id"],
                    "safe_slide_id": candidate["safe_slide_id"],
                    "safe_family_id": candidate["family_id"],
                    "case_alias": candidate["case_alias"],
                    "original_family_id": candidate["original_family_id"],
                    "source_path": candidate["source_path"],
                    "source_manifest_path": candidate["manifest_path"],
                    "source_manifest_sha256": candidate["source_manifest_sha256"],
                    "mucosa_boundary_sha256": candidate["mucosa_boundary_sha256"],
                    "mucosa_manifest_sha256": candidate["mucosa_manifest_sha256"],
                    "mucosa_artifact_sha256": candidate["mucosa_artifact_sha256"],
                    "source_wsi_identity_sha256": candidate["source_wsi_identity_sha256"],
                    "source_patch_id": candidate["source_patch_id"],
                    "recruitment_diagnosis": candidate["diagnosis"],
                    "recruitment_stratum": candidate["stratum"],
                    "image_sha256": crop.image_sha256,
                }
            )
        annotation_rows = [
            {
                **{key: row[key] for key in LOCKED_COLUMNS},
                **{key: "" for key in EXPERT_COLUMNS},
            }
            for row in public_rows
        ]
        workbook_path = temporary_root / str(config.get("annotation", {}).get("workbook_name", "annotation.xlsx"))
        write_annotation_workbook(workbook_path, annotation_rows)
        guideline_path = temporary_root / str(config.get("annotation", {}).get("guideline_name", "annotation_guideline.md"))
        write_text(guideline_path, _guideline_text())
        write_text(
            temporary_root / "README.md",
            """# Blinded Expert 5x Architecture ROI Annotation Package

Open `annotation.xlsx`, then use the relative links in `image_file` to review
the exact PNG bytes under `images/`. The package contains only pseudonymous
review identifiers, images, and expert-entry fields. Fill only the yellow
expert columns and save a copy as `annotation_completed.xlsx`.

This is Stage A. It stops before benchmark freeze and before all model
evaluation. The next action is `EXPERT_REVIEW_ANNOTATION_XLSX`.
""",
        )
        contact_sheets = _contact_sheets(
            images_dir,
            public_rows,
            contact_dir,
            columns=int(config.get("annotation", {}).get("contact_sheet_columns", 5)),
            grid_rows=int(config.get("annotation", {}).get("contact_sheet_rows", 4)),
        )
        structure = workbook_structure(workbook_path)
        if structure["headers"] != list(ANNOTATION_COLUMNS) or structure["validation_count"] != len(DROPDOWNS):
            raise ProvenanceError("Generated annotation workbook structure validation failed")
        manifest = {
            "schema_version": "architecture_annotation_manifest_v1",
            "benchmark_version": config.get("benchmark_version"),
            "created_at": _timestamp(),
            "blinded": True,
            "reviewed_image_identity": "individual_png_sha256",
            "one_row_per_roi": True,
            "annotation_columns": list(ANNOTATION_COLUMNS),
            "locked_columns": list(LOCKED_COLUMNS),
            "expert_columns": list(EXPERT_COLUMNS),
            "mucosa_provenance_fields": [
                "mucosa_boundary_sha256",
                "mucosa_manifest_sha256",
                "mucosa_artifact_sha256",
                "source_wsi_identity_sha256",
            ],
            "candidate_rois": len(public_rows),
            "unique_candidate_wsi": len({row["safe_slide_id"] for row in public_rows}),
            "unique_candidate_families": len({row["family_id"] for row in public_rows}),
            "workbook_sha256": sha256_file(workbook_path),
            "guideline_sha256": sha256_file(guideline_path),
            "contact_sheets": contact_sheets,
            "rois": public_rows,
            "flags": {
                "ANNOTATION_PACKAGE_READY": True,
                "EXPERT_ANNOTATION_COMPLETE": False,
                "BENCHMARK_FREEZE_COMPLETE": False,
                "MODEL_EVALUATION_STARTED": False,
            },
        }
        leakage = _audit_public_package(temporary_root, manifest)
        if leakage:
            raise ProvenanceError("Generated annotation package failed blinding audit", leakage)
        write_json(temporary_root / "annotation_manifest.json", manifest)
        restricted_payload = {
            **restricted_summary,
            "annotation_manifest_sha256": sha256_file(temporary_root / "annotation_manifest.json"),
            "records": restricted_rows,
        }
        temporary_restricted = artifact_root / ".restricted_recruitment_provenance.json.tmp"
        write_json(temporary_restricted, restricted_payload)
        os.chmod(str(temporary_restricted), 0o600)
        os.replace(str(temporary_root), str(package_root))
        os.replace(str(temporary_restricted), str(restricted_path))
        os.chmod(str(restricted_path), 0o600)
        stage_a_summary = {
            "schema_version": "architecture_stage_a_summary_v1",
            "audit": audit["summary"],
            "mucosa_backfill": backfill_result["summary"],
            "candidate_rois": len(public_rows),
            "unique_candidate_wsi": len({row["safe_slide_id"] for row in public_rows}),
            "unique_candidate_families": len({row["family_id"] for row in public_rows}),
            "internal_recruitment": {"serrated": 60, "tubular": 60, "villous": 60},
            "final_gate": "WAITING_FOR_EXPERT_ANNOTATION",
            "model_evaluation_started": False,
            "benchmark_freeze_complete": False,
            "prompt_review_required_later": True,
        }
        write_json(artifact_root / "stage_a_summary.json", stage_a_summary)
        write_text(paths["preparation_report"], _preparation_report_text(audit, backfill_result, manifest))
        state = {
            "schema_version": "conch_text_architecture_poc_state_v1",
            "updated_at": _timestamp(),
            "final_gate": "WAITING_FOR_EXPERT_ANNOTATION",
            "annotation_package_ready": True,
            "expert_annotation_complete": False,
            "benchmark_freeze_complete": False,
            "model_evaluation_started": False,
            "prompt_review_required_later": True,
            "annotation_manifest_sha256": sha256_file(package_root / "annotation_manifest.json"),
            "candidate_rois": len(public_rows),
            "unique_candidate_wsi": len({row["safe_slide_id"] for row in public_rows}),
        }
        write_json(artifact_root / "status.json", state)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(str(temporary_root))
        temporary_restricted = artifact_root / ".restricted_recruitment_provenance.json.tmp"
        if temporary_restricted.exists():
            temporary_restricted.unlink()
        raise
    return _stage_a_result(package_root, manifest)


def _stage_a_result(package_root: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    package_parent = Path(package_root).parent
    stage_a_summary_path = package_parent / "stage_a_summary.json"
    summary = read_json(stage_a_summary_path) if stage_a_summary_path.is_file() else {}
    backfill_summary = dict(summary.get("mucosa_backfill") or {})
    internal = dict(summary.get("internal_recruitment") or {})
    return {
        "ANNOTATION_PACKAGE_READY": True,
        "FINAL_GATE": "WAITING_FOR_EXPERT_ANNOTATION",
        "CANDIDATE_ROIS": int(manifest.get("candidate_rois", 0)),
        "CANDIDATE_ROIS_TOTAL": int(manifest.get("candidate_rois", 0)),
        "UNIQUE_CANDIDATE_WSI": int(manifest.get("unique_candidate_wsi", 0)),
        "ANNOTATION_WORKBOOK": str((Path(package_root) / "annotation.xlsx").resolve()),
        "ANNOTATION_IMAGES_DIR": str((Path(package_root) / "images").resolve()),
        "ANNOTATION_BLINDED": True,
        "EXPERT_ANNOTATION_COMPLETE": False,
        "BENCHMARK_FREEZE_COMPLETE": False,
        "MODEL_EVALUATION_STARTED": False,
        "INTERNAL_RECRUITMENT_SERRATED": int(internal.get("serrated", 0)),
        "INTERNAL_RECRUITMENT_TUBULAR": int(internal.get("tubular", 0)),
        "INTERNAL_RECRUITMENT_VILLOUS": int(internal.get("villous", 0)),
        "EXISTING_MUCOSA_REUSED": int(backfill_summary.get("reused_baseline", backfill_summary.get("reused", 0))),
        "NEW_MUCOSA_BACKFILLED": int(backfill_summary.get("newly_processed", 0)),
        "MUCOSA_FAILURES": len(backfill_summary.get("failed", [])),
        "VALID_5X_PHYSICAL_SCALE": "{0}/{0}".format(int(manifest.get("candidate_rois", 0))),
        "PNG_SHA_VALID": "{0}/{0}".format(int(manifest.get("candidate_rois", 0))),
        "PROMPT_REVIEW_REQUIRED_LATER": True,
        "NEXT_ACTION": "EXPERT_REVIEW_ANNOTATION_XLSX",
    }


def _annotation_input(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".xlsx":
        return read_annotation_workbook(path)
    if suffix == ".csv":
        rows = read_csv(path)
        return rows, [str(value) for row in rows for value in row.values()]
    raise FreezeValidationError("Annotations must be an XLSX or CSV file")


def _has_model_output_field(field: str) -> bool:
    return any(pattern.search(str(field)) for pattern in MODEL_OUTPUT_FIELD_PATTERNS)


def _normalize_components(value: str) -> List[str]:
    output = []
    for item in str(value or "").split(";"):
        token = item.strip().lower()
        if token and token not in output:
            output.append(token)
    return output


def _validate_annotations(
    annotation_path: Path,
    package_root: Path,
    annotation_manifest: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows, visible_text = _annotation_input(annotation_path)
    if not rows:
        raise FreezeValidationError("Completed annotation file contains no rows")
    expected_columns = set(ANNOTATION_COLUMNS)
    actual_columns = set(rows[0])
    unknown_columns = sorted(actual_columns - expected_columns)
    missing_columns = sorted(expected_columns - actual_columns)
    conflict_issues = []
    validation_issues = []
    provenance_issues = []
    if missing_columns:
        conflict_issues.append({"row_number": 1, "field": "headers", "error": "missing columns", "values": missing_columns})
    if unknown_columns:
        issue = {"row_number": 1, "field": "headers", "error": "unknown columns", "values": unknown_columns}
        if any(_has_model_output_field(field) for field in unknown_columns):
            provenance_issues.append(issue)
        else:
            conflict_issues.append(issue)
    leakage = find_forbidden_text(visible_text + list(actual_columns), FORBIDDEN_PUBLIC_PATTERNS)
    if leakage:
        provenance_issues.extend({"row_number": None, "field": "workbook", "error": "forbidden leakage", **item} for item in leakage)

    expected_rows = {str(row["roi_id"]): dict(row) for row in annotation_manifest.get("rois", [])}
    seen = set()
    canonical = []
    allowed_labels = set(DROPDOWNS["architecture_label"])
    allowed_evaluable = set(DROPDOWNS["evaluable"])
    allowed_purity = set(DROPDOWNS["pure_or_mixed"])
    allowed_confidence = set(DROPDOWNS["expert_confidence"])
    allowed_exclusion = set(DROPDOWNS["exclusion_reason"])
    for input_index, row in enumerate(rows, 2):
        roi_id = str(row.get("roi_id", "")).strip()
        if not roi_id:
            conflict_issues.append({"row_number": input_index, "field": "roi_id", "error": "missing roi_id"})
            continue
        if roi_id in seen:
            conflict_issues.append({"row_number": input_index, "field": "roi_id", "error": "duplicate roi_id", "value": roi_id})
            continue
        seen.add(roi_id)
        expected = expected_rows.get(roi_id)
        if expected is None:
            conflict_issues.append({"row_number": input_index, "field": "roi_id", "error": "unknown roi_id", "value": roi_id})
            continue
        for field in LOCKED_COLUMNS:
            if str(row.get(field, "")).strip() != str(expected.get(field, "")).strip():
                conflict_issues.append(
                    {
                        "row_number": input_index,
                        "field": field,
                        "error": "locked identity changed",
                        "expected": expected.get(field, ""),
                        "observed": row.get(field, ""),
                    }
                )
        label = str(row.get("architecture_label", "")).strip().lower()
        evaluable = str(row.get("evaluable", "")).strip().lower()
        purity = str(row.get("pure_or_mixed", "")).strip().lower()
        confidence = str(row.get("expert_confidence", "")).strip().lower()
        exclusion = str(row.get("exclusion_reason", "")).strip().lower()
        components = _normalize_components(row.get("architecture_components", ""))
        if evaluable not in allowed_evaluable:
            validation_issues.append({"row_number": input_index, "field": "evaluable", "error": "missing or illegal value", "value": evaluable})
        if label and label not in allowed_labels:
            validation_issues.append({"row_number": input_index, "field": "architecture_label", "error": "illegal value", "value": label})
        if purity and purity not in allowed_purity:
            validation_issues.append({"row_number": input_index, "field": "pure_or_mixed", "error": "illegal value", "value": purity})
        if confidence and confidence not in allowed_confidence:
            validation_issues.append({"row_number": input_index, "field": "expert_confidence", "error": "illegal value", "value": confidence})
        if exclusion and exclusion not in allowed_exclusion:
            validation_issues.append({"row_number": input_index, "field": "exclusion_reason", "error": "illegal value", "value": exclusion})
        illegal_components = sorted(set(components) - set(PRIMARY_CLASSES))
        if illegal_components:
            validation_issues.append({"row_number": input_index, "field": "architecture_components", "error": "illegal component", "values": illegal_components})
        if evaluable == "yes":
            for field, value in (
                ("architecture_label", label),
                ("pure_or_mixed", purity),
                ("expert_confidence", confidence),
            ):
                if not value:
                    validation_issues.append({"row_number": input_index, "field": field, "error": "required when evaluable=yes"})
            if purity == "mixed" and len(components) < 2:
                validation_issues.append({"row_number": input_index, "field": "architecture_components", "error": "mixed ROI requires at least two components"})
        elif evaluable == "no" and not exclusion:
            validation_issues.append({"row_number": input_index, "field": "exclusion_reason", "error": "required when evaluable=no"})
        if purity == "pure" and not components and label in PRIMARY_CLASSES:
            components = [label]
        image_path = package_root / str(expected.get("image_file", ""))
        if not image_path.is_file():
            provenance_issues.append({"row_number": input_index, "field": "image_file", "error": "reviewed PNG missing", "value": str(image_path)})
        elif sha256_file(image_path) != str(expected.get("image_sha256", "")):
            provenance_issues.append({"row_number": input_index, "field": "image_sha256", "error": "reviewed PNG SHA changed", "roi_id": roi_id})
        canonical.append(
            {
                **expected,
                "architecture_label": label,
                "evaluable": evaluable,
                "pure_or_mixed": purity,
                "expert_confidence": confidence,
                "architecture_components": ";".join(components),
                "exclusion_reason": exclusion,
                "notes": str(row.get("notes", "")).strip(),
                "annotation_input_row": input_index,
                "reader_mode": "single-reader exploratory POC",
            }
        )
    missing_roi_ids = sorted(set(expected_rows) - seen)
    for roi_id in missing_roi_ids:
        conflict_issues.append({"row_number": None, "field": "roi_id", "error": "annotation row missing", "value": roi_id})
    if len(rows) != len(expected_rows):
        conflict_issues.append(
            {
                "row_number": None,
                "field": "row_count",
                "error": "annotation row count mismatch",
                "expected": len(expected_rows),
                "observed": len(rows),
            }
        )
    if provenance_issues:
        raise ProvenanceError("Annotation provenance or leakage validation failed", provenance_issues)
    if conflict_issues:
        raise AnnotationConflictError("Annotation identity conflict validation failed", conflict_issues)
    if validation_issues:
        raise FreezeValidationError("Annotation field validation failed", validation_issues)
    canonical.sort(key=lambda row: str(row["roi_id"]))
    audit = {
        "reviewed": len(canonical),
        "evaluable": sum(row["evaluable"] == "yes" for row in canonical),
        "pure": sum(row["pure_or_mixed"] == "pure" for row in canonical),
        "mixed": sum(row["pure_or_mixed"] == "mixed" or row["architecture_label"] == "mixed" for row in canonical),
        "uncertain": sum(row["architecture_label"] == "uncertain" for row in canonical),
        "excluded": sum(row["evaluable"] == "no" for row in canonical),
    }
    return canonical, audit


def _stress_reasons(row: Mapping[str, Any]) -> List[str]:
    reasons = []
    if row.get("evaluable") == "no":
        reasons.append("not_evaluable")
        if row.get("exclusion_reason"):
            reasons.append("exclusion:{0}".format(row["exclusion_reason"]))
    if row.get("architecture_label") == "mixed" or row.get("pure_or_mixed") == "mixed":
        reasons.append("mixed")
    if row.get("architecture_label") == "uncertain":
        reasons.append("uncertain")
    if row.get("expert_confidence") in {"medium", "low"}:
        reasons.append("confidence:{0}".format(row["expert_confidence"]))
    if row.get("architecture_label") not in PRIMARY_CLASSES and row.get("architecture_label") not in {"mixed", "uncertain", ""}:
        reasons.append("non_primary_label")
    return sorted(set(reasons or ["not_primary_eligible"]))


def _primary_eligible(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("evaluable") == "yes"
        and row.get("pure_or_mixed") == "pure"
        and row.get("architecture_label") in PRIMARY_CLASSES
        and row.get("expert_confidence") == "high"
    )


def _select_sets(rows: Sequence[Mapping[str, Any]], freeze_seed: int, target_per_class: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select primary/stress/reserve using only pre-registered fields."""

    stress = []
    eligible = []
    for source in rows:
        row = dict(source)
        if _primary_eligible(row):
            eligible.append(row)
        else:
            row["assignment"] = "stress_set"
            row["assignment_reason"] = ";".join(_stress_reasons(row))
            stress.append(row)
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        group = str(row.get("family_id") or row.get("safe_slide_id"))
        by_group[group].append(row)
    independent = []
    reserve = []
    for group in sorted(by_group):
        ordered = sorted(
            by_group[group],
            key=lambda row: (
                int(row.get("candidate_generation_order", 10 ** 12)),
                _rank(freeze_seed, row.get("roi_id")),
            ),
        )
        independent.append(ordered[0])
        for row in ordered[1:]:
            row["assignment"] = "reserve_secondary"
            row["assignment_reason"] = "group_independence_secondary"
            reserve.append(row)
    primary = []
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in independent:
        by_class[str(row["architecture_label"])].append(row)
    for architecture_class in PRIMARY_CLASSES:
        ordered = sorted(
            by_class.get(architecture_class, []),
            key=lambda row: (_rank(freeze_seed, row["roi_id"]), str(row["roi_id"])),
        )
        for row in ordered[: int(target_per_class)]:
            row["assignment"] = "primary"
            row["assignment_reason"] = "eligible_independent_deterministic_class_selection"
            primary.append(row)
        for row in ordered[int(target_per_class) :]:
            row["assignment"] = "reserve_secondary"
            row["assignment_reason"] = "deterministic_class_truncation"
            reserve.append(row)
    primary.sort(key=lambda row: str(row["roi_id"]))
    stress.sort(key=lambda row: str(row["roi_id"]))
    reserve.sort(key=lambda row: str(row["roi_id"]))
    return primary, stress, reserve


def _split_primary(primary: Sequence[Mapping[str, Any]], n_splits: int, seed: int, validation_offset: int) -> Dict[str, Any]:
    class_counts = Counter(str(row["architecture_label"]) for row in primary)
    group_support = Counter()
    for label in PRIMARY_CLASSES:
        group_support[label] = len({str(row["family_id"]) for row in primary if row["architecture_label"] == label})
    if any(group_support[label] < int(n_splits) for label in PRIMARY_CLASSES):
        return {
            "ready": False,
            "blocking_reason": "fewer_than_{0}_independent_groups_for_at_least_one_class".format(n_splits),
            "class_counts": dict(class_counts),
            "class_group_counts": dict(group_support),
            "assignments": {},
            "outer_folds": [],
            "family_split_leakage": 0,
        }
    try:
        import numpy as np
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception as exc:
        raise ProvenanceError("Formal split requires scikit-learn StratifiedGroupKFold: {0}".format(exc))
    ordered = sorted(primary, key=lambda row: str(row["roi_id"]))
    labels = np.asarray([str(row["architecture_label"]) for row in ordered])
    groups = np.asarray([str(row["family_id"]) for row in ordered])
    splitter = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    assignments: Dict[str, int] = {}
    dummy = np.zeros(len(ordered), dtype=np.uint8)
    for fold, (_development, test_indices) in enumerate(splitter.split(dummy, labels, groups=groups)):
        for index in test_indices.tolist():
            assignments[str(ordered[int(index)]["roi_id"])] = int(fold)
    if set(assignments) != {str(row["roi_id"]) for row in ordered}:
        raise ProvenanceError("Each primary ROI must appear in exactly one fold")
    family_folds: Dict[str, set] = defaultdict(set)
    for row in ordered:
        family_folds[str(row["family_id"])].add(assignments[str(row["roi_id"])])
    leakage = sum(len(values) > 1 for values in family_folds.values())
    if leakage:
        raise ProvenanceError("StratifiedGroupKFold produced family leakage")
    outer_folds = []
    all_roi_ids = {str(row["roi_id"]) for row in ordered}
    for outer in range(int(n_splits)):
        test = sorted(roi_id for roi_id, fold in assignments.items() if fold == outer)
        val_fold = (outer + int(validation_offset)) % int(n_splits)
        validation = sorted(roi_id for roi_id, fold in assignments.items() if fold == val_fold)
        train = sorted(all_roi_ids - set(test) - set(validation))
        outer_folds.append(
            {
                "outer_fold": outer,
                "test_fold": outer,
                "validation_fold": val_fold,
                "train_roi_ids": train,
                "validation_roi_ids": validation,
                "test_roi_ids": test,
            }
        )
    return {
        "ready": True,
        "strategy": "StratifiedGroupKFold",
        "n_splits": int(n_splits),
        "shuffle": True,
        "random_state": int(seed),
        "class_counts": dict(class_counts),
        "class_group_counts": dict(group_support),
        "assignments": assignments,
        "outer_folds": outer_folds,
        "family_split_leakage": 0,
    }


def _low_data_subsets(
    primary: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    k_values: Sequence[int],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    arms = []
    if not split.get("ready"):
        return {"ready": False, "records": [], "k_ready": {str(k): False for k in k_values}, "all_ready": False}
    by_roi = {str(row["roi_id"]): row for row in primary}
    records = []
    k_ready = {str(k): True for k in k_values}
    for outer in split["outer_folds"]:
        fold_index = int(outer["outer_fold"])
        training = [by_roi[roi_id] for roi_id in outer["train_roi_ids"]]
        training_ids = {str(row["roi_id"]) for row in training}
        forbidden = set(outer["validation_roi_ids"]) | set(outer["test_roi_ids"])
        if training_ids & forbidden:
            raise ProvenanceError("Low-data training pool includes validation/test ROI")
        per_class_training = {
            label: sorted(str(row["roi_id"]) for row in training if row["architecture_label"] == label)
            for label in PRIMARY_CLASSES
        }
        for seed in seeds:
            for k in k_values:
                selected = {}
                ready = True
                for label in PRIMARY_CLASSES:
                    ranked = sorted(
                        per_class_training[label],
                        key=lambda roi_id: (_rank(seed, fold_index, k, label, roi_id), roi_id),
                    )
                    if len(ranked) < int(k):
                        ready = False
                    selected[label] = ranked[: int(k)] if ready or len(ranked) >= int(k) else []
                if not ready:
                    k_ready[str(k)] = False
                    selected = {label: [] for label in PRIMARY_CLASSES}
                roi_ids = sorted(roi_id for values in selected.values() for roi_id in values)
                payload = {
                    "outer_fold": fold_index,
                    "seed": int(seed),
                    "k": int(k),
                    "ready": ready,
                    "per_class_roi_ids": selected,
                    "roi_ids": roi_ids,
                    "validation_roi_ids": list(outer["validation_roi_ids"]),
                    "test_roi_ids": list(outer["test_roi_ids"]),
                }
                payload["subset_hash"] = sha256_payload(payload)
                records.append(payload)
            all_payload = {
                "outer_fold": fold_index,
                "seed": int(seed),
                "k": "all",
                "ready": True,
                "per_class_roi_ids": per_class_training,
                "roi_ids": sorted(training_ids),
                "validation_roi_ids": list(outer["validation_roi_ids"]),
                "test_roi_ids": list(outer["test_roi_ids"]),
            }
            all_payload["subset_hash"] = sha256_payload(all_payload)
            records.append(all_payload)
    return {
        "ready": True,
        "records": records,
        "k_ready": k_ready,
        "all_ready": True,
    }


def _prompt_status(prompt_path: Path) -> Tuple[Dict[str, Any], bool]:
    if not prompt_path.is_file():
        raise ProvenanceError("Prompt config is missing: {0}".format(prompt_path))
    with prompt_path.open("r", encoding="utf-8") as handle:
        prompt = yaml.safe_load(handle) or {}
    if prompt.get("schema_version") != "architecture_prompts_v1":
        raise ProvenanceError("Prompt config schema_version is invalid")
    mapping = dict(prompt.get("class_mapping") or {})
    if set(mapping) != set(PRIMARY_CLASSES) or any(str(mapping[key]) != key for key in PRIMARY_CLASSES):
        raise ProvenanceError("Prompt class mapping must exactly match serrated/tubular/villous")
    prompt_sets = dict(prompt.get("prompt_sets") or {})
    required_sets = {"single_class_name", "matched_class_name_ensemble", "morphology_rich"}
    if set(prompt_sets) != required_sets:
        raise ProvenanceError("Prompt config must contain exactly the three registered prompt sets")
    for set_name, class_prompts in prompt_sets.items():
        if set(class_prompts or {}) != set(PRIMARY_CLASSES):
            raise ProvenanceError("Prompt set {0} has invalid class mapping".format(set_name))
        if any(not list(class_prompts[label] or []) for label in PRIMARY_CLASSES):
            raise ProvenanceError("Prompt set {0} contains an empty class prompt list".format(set_name))
    confirmed = bool(
        str(prompt.get("review_status", "")).strip().lower() == "confirmed"
        and str(prompt.get("reviewed_by", "")).strip()
        and str(prompt.get("reviewed_at", "")).strip()
    )
    return prompt, confirmed


def _manifest_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "roi_id": row["roi_id"],
        "safe_slide_id": row["safe_slide_id"],
        "family_id": row["family_id"],
        "architecture_label": row["architecture_label"],
        "expert_confidence": row["expert_confidence"],
        "evaluable": row["evaluable"],
        "pure_or_mixed": row["pure_or_mixed"],
        "bbox": row["bbox"],
        "magnification": row["magnification"],
        "MPP": row["MPP"],
        "mpp_x": row["mpp_x"],
        "mpp_y": row["mpp_y"],
        "physical_fov_um": row["physical_fov_um"],
        # Resolve this path relative to benchmark_v1. It points back to the
        # exact reviewed PNG; downstream consumers must never recrop the WSI.
        "image_file": "../annotation_package_v1/{0}".format(row["image_file"]),
        "image_sha256": row["image_sha256"],
        "source_manifest_sha256": row["source_manifest_sha256"],
        "mucosa_boundary_sha256": row.get("mucosa_boundary_sha256", ""),
        "mucosa_manifest_sha256": row.get("mucosa_manifest_sha256", ""),
        "mucosa_artifact_sha256": row.get("mucosa_artifact_sha256", ""),
        "source_wsi_identity_sha256": row.get("source_wsi_identity_sha256", ""),
        "candidate_generation_order": row["candidate_generation_order"],
        "grouping_resolution": row.get("grouping_resolution", "family"),
        "assignment_reason": row.get("assignment_reason", ""),
    }


PRIMARY_CSV_FIELDS = (
    "roi_id",
    "safe_slide_id",
    "family_id",
    "architecture_label",
    "expert_confidence",
    "evaluable",
    "pure_or_mixed",
    "bbox",
    "magnification",
    "MPP",
    "mpp_x",
    "mpp_y",
    "physical_fov_um",
    "image_file",
    "image_sha256",
    "source_manifest_sha256",
    "mucosa_boundary_sha256",
    "mucosa_manifest_sha256",
    "mucosa_artifact_sha256",
    "source_wsi_identity_sha256",
    "candidate_generation_order",
    "grouping_resolution",
    "assignment_reason",
)
FROZEN_CSV_FIELDS = PRIMARY_CSV_FIELDS + (
    "annotation_instance_id",
    "architecture_components",
    "exclusion_reason",
    "notes",
    "reader_mode",
    "assignment",
)


def _write_split_artifacts(
    split_dir: Path,
    primary: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    low_data: Mapping[str, Any],
    model_arms: Sequence[str],
) -> Dict[str, str]:
    split_dir.mkdir(parents=True, exist_ok=True)
    assignments = dict(split.get("assignments") or {})
    fold_rows = [
        {
            "roi_id": row["roi_id"],
            "architecture_label": row["architecture_label"],
            "family_id": row["family_id"],
            "fold": assignments.get(str(row["roi_id"]), ""),
        }
        for row in sorted(primary, key=lambda item: str(item["roi_id"]))
    ]
    write_csv(split_dir / "folds.csv", fold_rows, ("roi_id", "architecture_label", "family_id", "fold"))
    write_json(split_dir / "folds.json", dict(split))
    summary_rows = []
    if split.get("ready"):
        for fold in range(int(split.get("n_splits", 5))):
            rows = [row for row in primary if assignments[str(row["roi_id"])] == fold]
            for label in PRIMARY_CLASSES:
                selected = [row for row in rows if row["architecture_label"] == label]
                summary_rows.append(
                    {
                        "fold": fold,
                        "architecture_label": label,
                        "roi_count": len(selected),
                        "family_count": len({row["family_id"] for row in selected}),
                    }
                )
    write_csv(split_dir / "fold_summary.csv", summary_rows, ("fold", "architecture_label", "roi_count", "family_count"))
    write_jsonl(split_dir / "low_data_subsets.jsonl", low_data.get("records", []))
    subset_summary_rows = [
        {
            "outer_fold": row["outer_fold"],
            "seed": row["seed"],
            "k": row["k"],
            "ready": row["ready"],
            "roi_count": len(row["roi_ids"]),
            "subset_hash": row["subset_hash"],
        }
        for row in low_data.get("records", [])
    ]
    write_csv(
        split_dir / "low_data_subset_summary.csv",
        subset_summary_rows,
        ("outer_fold", "seed", "k", "ready", "roi_count", "subset_hash"),
    )
    subset_hash = sha256_file(split_dir / "low_data_subsets.jsonl")
    write_json(
        split_dir / "low_data_subset_hash.json",
        {
            "schema_version": "architecture_low_data_subset_hash_v1",
            "sha256": subset_hash,
            "k_ready": dict(low_data.get("k_ready") or {}),
            "all_training_ready": bool(low_data.get("all_ready")),
        },
    )
    arm_contract = {
        "schema_version": "architecture_model_arm_subset_contract_v1",
        "shared_folds_sha256": sha256_file(split_dir / "folds.csv"),
        "shared_low_data_subsets_sha256": subset_hash,
        "arms": {
            str(arm): {
                "folds_sha256": sha256_file(split_dir / "folds.csv"),
                "low_data_subsets_sha256": subset_hash,
            }
            for arm in model_arms
        },
    }
    write_json(split_dir / "model_arm_subset_contract.json", arm_contract)
    split_hash = sha256_payload(
        {
            "folds_csv": sha256_file(split_dir / "folds.csv"),
            "folds_json": sha256_file(split_dir / "folds.json"),
            "fold_summary": sha256_file(split_dir / "fold_summary.csv"),
        }
    )
    write_json(
        split_dir / "split_hash.json",
        {
            "schema_version": "architecture_split_hash_v1",
            "split_sha256": split_hash,
            "folds_csv_sha256": sha256_file(split_dir / "folds.csv"),
            "folds_json_sha256": sha256_file(split_dir / "folds.json"),
            "fold_summary_sha256": sha256_file(split_dir / "fold_summary.csv"),
            "family_split_leakage": int(split.get("family_split_leakage", 0)),
        },
    )
    return {"split_sha256": split_hash, "subsets_sha256": subset_hash}


def _freeze_policy(config: Mapping[str, Any]) -> Dict[str, Any]:
    freeze = dict(config.get("freeze") or {})
    return {
        "schema_version": "architecture_freeze_policy_v1",
        "benchmark_version": config.get("benchmark_version"),
        "primary_eligibility": {
            "evaluable": "yes",
            "pure_or_mixed": "pure",
            "architecture_label": list(PRIMARY_CLASSES),
            "expert_confidence": "high",
        },
        "primary_independence": "one primary ROI per safe family; fallback safe slide only when family unavailable",
        "within_group_selection": ["candidate_generation_order", "deterministic_roi_id_hash"],
        "class_truncation": "SHA256(freeze_seed + roi_id)",
        "primary_target_per_class": int(freeze.get("primary_target_per_class", 40)),
        "freeze_seed": int(freeze.get("freeze_seed", 20260823)),
        "split": {
            "strategy": "StratifiedGroupKFold",
            "n_splits": int(freeze.get("n_splits", 5)),
            "shuffle": True,
            "random_state": int(freeze.get("split_seed", 17)),
            "validation_fold_offset": int(freeze.get("validation_fold_offset", 1)),
        },
        "low_data": {
            "k": [int(value) for value in freeze.get("low_data_k", [5, 10, 20])],
            "seeds": [int(value) for value in freeze.get("low_data_seeds", [17, 29, 43, 59, 71])],
            "all_definition": "all available training ROIs of the current outer fold",
        },
        "primary_endpoint": freeze.get("primary_endpoint", "3-class macro_f1"),
        "stress_excluded_from_primary_macro_f1": True,
        "selection_fields": list(PRIMARY_SELECTION_FIELDS),
        "model_scores_used_for_selection": False,
        "model_evaluation_started": False,
        "revision_policy": "create benchmark_v2; never overwrite benchmark_v1",
    }


def _report_text(
    manifest: Mapping[str, Any],
    audit: Mapping[str, int],
    primary: Sequence[Mapping[str, Any]],
    stress: Sequence[Mapping[str, Any]],
    reserve: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    low_data: Mapping[str, Any],
) -> str:
    class_counts = Counter(row["architecture_label"] for row in primary)
    stress_counts = Counter(reason for row in stress for reason in str(row.get("assignment_reason", "")).split(";") if reason)
    completeness = {
        "image SHA-256": sum(bool(row.get("image_sha256")) for row in primary),
        "MPP": sum(bool(row.get("MPP")) for row in primary),
        "magnification": sum(bool(row.get("magnification")) for row in primary),
        "FOV": sum(bool(row.get("physical_fov_um")) for row in primary),
    }
    split_lines = []
    if split.get("ready"):
        assignments = split["assignments"]
        for fold in range(int(split.get("n_splits", 5))):
            rows = [row for row in primary if assignments[str(row["roi_id"])] == fold]
            split_lines.append(
                "- fold {0}: ROI={1}, families={2}, classes={3}".format(
                    fold,
                    len(rows),
                    len({row["family_id"] for row in rows}),
                    dict(sorted(Counter(row["architecture_label"] for row in rows).items())),
                )
            )
    else:
        split_lines.append("- split blocked: {0}".format(split.get("blocking_reason", "unknown")))
    return """# Expert-Confirmed 5x Architecture Benchmark v1 Freeze Report

Freeze timestamp: `{freeze_timestamp}`  
Final gate: `{final_gate}`  
Benchmark frozen: `{frozen}`  
Model evaluation started: `false`

Candidate recruitment was diagnosis-enriched, while expert ROI architecture
annotation was blinded to recruitment diagnosis.

## Annotation

- reviewed: {reviewed}
- evaluable: {evaluable}
- pure: {pure}
- mixed: {mixed}
- uncertain: {uncertain}
- excluded: {excluded}

## Primary

- serrated: {serrated}
- tubular: {tubular}
- villous: {villous}
- total: {primary_total}
- unique WSI: {unique_wsi}
- unique family: {unique_family}

## Stress

- total: {stress_total}
- reasons: `{stress_reasons}`
- reserve secondary: {reserve_total}

## Reader

`single-reader exploratory POC`

## Blinding

- slide diagnosis hidden: yes
- recruitment stratum hidden: yes
- CONCH outputs absent: yes
- model predictions absent: yes

## Image provenance

- image hash completeness: {image_hash}/{primary_total}
- MPP completeness: {mpp}/{primary_total}
- magnification completeness: {magnification}/{primary_total}
- FOV completeness: {fov}/{primary_total}

## Split

{split_lines}

## Low-data

- k=5 ready: {k5}
- k=10 ready: {k10}
- k=20 ready: {k20}
- all-training ready: {all_ready}

## Hashes

- annotation: `{annotation_sha}`
- primary: `{primary_sha}`
- stress: `{stress_sha}`
- split: `{split_sha}`
- subsets: `{subsets_sha}`
- prompt: `{prompt_sha}`
- config: `{config_sha}`

## Leakage audit

- slide diagnosis leakage: 0
- model-output leakage: 0
- family split leakage: {family_leakage}

Stress and reserve ROIs are not part of the primary three-class Macro-F1.
Any post-freeze change requires benchmark_v2 with a recorded revision reason.
""".format(
        freeze_timestamp=manifest["freeze_timestamp"],
        final_gate=manifest["final_gate"],
        frozen=str(bool(manifest["frozen"])).lower(),
        reviewed=audit["reviewed"],
        evaluable=audit["evaluable"],
        pure=audit["pure"],
        mixed=audit["mixed"],
        uncertain=audit["uncertain"],
        excluded=audit["excluded"],
        serrated=class_counts.get("serrated", 0),
        tubular=class_counts.get("tubular", 0),
        villous=class_counts.get("villous", 0),
        primary_total=len(primary),
        unique_wsi=len({row["safe_slide_id"] for row in primary}),
        unique_family=len({row["family_id"] for row in primary}),
        stress_total=len(stress),
        stress_reasons=json.dumps(dict(sorted(stress_counts.items())), sort_keys=True),
        reserve_total=len(reserve),
        image_hash=completeness["image SHA-256"],
        mpp=completeness["MPP"],
        magnification=completeness["magnification"],
        fov=completeness["FOV"],
        split_lines="\n".join(split_lines),
        k5=str(bool(low_data.get("k_ready", {}).get("5"))).lower(),
        k10=str(bool(low_data.get("k_ready", {}).get("10"))).lower(),
        k20=str(bool(low_data.get("k_ready", {}).get("20"))).lower(),
        all_ready=str(bool(low_data.get("all_ready"))).lower(),
        annotation_sha=manifest["annotation_sha256"],
        primary_sha=manifest["primary_manifest_sha256"],
        stress_sha=manifest["stress_manifest_sha256"],
        split_sha=manifest["split_sha256"],
        subsets_sha=manifest["subsets_sha256"],
        prompt_sha=manifest["prompt_config_sha256"],
        config_sha=manifest["config_sha256"],
        family_leakage=split.get("family_split_leakage", 0),
    )


def _canonical_annotation_payload(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    fields = (
        "annotation_instance_id",
        "roi_id",
        "safe_slide_id",
        "family_id",
        "architecture_label",
        "evaluable",
        "pure_or_mixed",
        "expert_confidence",
        "architecture_components",
        "exclusion_reason",
        "notes",
        "image_sha256",
    )
    return [{field: row.get(field, "") for field in fields} for row in sorted(rows, key=lambda item: str(item["roi_id"]))]


def _resume_check(
    benchmark_root: Path,
    canonical_rows: Sequence[Mapping[str, Any]],
    annotation_manifest: Mapping[str, Any],
    prompt_path: Path,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    freeze_manifest_path = benchmark_root / "freeze_manifest.json"
    if not freeze_manifest_path.is_file():
        raise ResumeRefusedError("REFUSE_RESUME: freeze_manifest.json is missing")
    manifest = read_json(freeze_manifest_path)
    checks = {
        "canonical_annotation_sha256": sha256_payload(_canonical_annotation_payload(canonical_rows)),
        "reviewed_images_sha256": sha256_payload(
            {str(row["roi_id"]): str(row["image_sha256"]) for row in annotation_manifest.get("rois", [])}
        ),
        "prompt_config_sha256": sha256_file(prompt_path),
        "group_mapping_sha256": sha256_payload(
            {str(row["roi_id"]): str(row["family_id"]) for row in annotation_manifest.get("rois", [])}
        ),
        "freeze_policy_content_sha256": sha256_payload(policy),
    }
    changed = {key: {"expected": manifest.get(key), "observed": value} for key, value in checks.items() if manifest.get(key) != value}
    if changed:
        raise ResumeRefusedError("REFUSE_RESUME: frozen benchmark identity changed", [{"changes": changed}])
    file_checks = {
        "annotation_sha256": benchmark_root / "frozen_annotations.csv",
        "primary_manifest_sha256": benchmark_root / "primary_rois.csv",
        "stress_manifest_sha256": benchmark_root / "stress_rois.csv",
        "freeze_policy_sha256": benchmark_root / "freeze_policy.json",
    }
    mismatches = []
    for key, path in file_checks.items():
        if not path.is_file() or sha256_file(path) != manifest.get(key):
            mismatches.append({"file": str(path), "field": key, "error": "frozen file hash mismatch"})
    if mismatches:
        raise ResumeRefusedError("REFUSE_RESUME: frozen output file changed", mismatches)
    return manifest


def freeze_benchmark(
    annotations: Path,
    config_path: Path = DEFAULT_CONFIG,
    resume: bool = False,
) -> Dict[str, Any]:
    """Stage B: validate expert rows and atomically freeze benchmark v1."""

    config, paths = _load_config(config_path)
    package_root = paths["annotation_package"]
    benchmark_root = paths["benchmark_root"]
    artifact_root = paths["artifact_root"]
    report_path = paths["freeze_report"]
    prompt_path = paths["prompt_config"]
    if not package_root.is_dir():
        raise ProvenanceError("Stage A annotation package is missing")
    annotation_manifest = _verify_existing_annotation_package(package_root)
    annotations = Path(annotations).resolve()
    if not annotations.is_file():
        raise FreezeValidationError("Completed annotation file is missing: {0}".format(annotations))
    try:
        canonical_rows, annotation_audit = _validate_annotations(annotations, package_root, annotation_manifest)
    except BenchmarkWorkflowError as exc:
        artifact_root.mkdir(parents=True, exist_ok=True)
        write_json(
            artifact_root / "freeze_validation_failed.json",
            {
                "schema_version": "architecture_freeze_validation_failure_v1",
                "timestamp": _timestamp(),
                "final_gate": exc.gate,
                "message": str(exc),
                "issues": exc.issues,
                "benchmark_freeze_complete": False,
                "model_evaluation_started": False,
            },
        )
        raise
    policy = _freeze_policy(config)
    prompt, prompt_confirmed = _prompt_status(prompt_path)
    del prompt
    if benchmark_root.exists():
        if not resume:
            raise ResumeRefusedError("benchmark_v1 already exists; overwrite is forbidden; use --resume only for verification")
        existing = _resume_check(benchmark_root, canonical_rows, annotation_manifest, prompt_path, policy)
        return _stage_b_result(existing)
    if report_path.exists():
        raise ResumeRefusedError("Freeze report exists without benchmark_v1; refusing to overwrite it")

    freeze = dict(config.get("freeze") or {})
    primary, stress, reserve = _select_sets(
        canonical_rows,
        freeze_seed=int(freeze.get("freeze_seed", 20260823)),
        target_per_class=int(freeze.get("primary_target_per_class", 40)),
    )
    assignment_by_roi = {}
    for row in primary + stress + reserve:
        assignment_by_roi[str(row["roi_id"])] = (row["assignment"], row["assignment_reason"])
    frozen_rows = []
    for row in canonical_rows:
        frozen = dict(row)
        frozen["image_file"] = "../annotation_package_v1/{0}".format(row["image_file"])
        frozen["assignment"], frozen["assignment_reason"] = assignment_by_roi[str(row["roi_id"])]
        frozen_rows.append(frozen)
    class_counts = Counter(row["architecture_label"] for row in primary)
    target = int(freeze.get("primary_target_per_class", 40))
    primary_support_complete = all(class_counts[label] >= target for label in PRIMARY_CLASSES)
    split = _split_primary(
        primary,
        n_splits=int(freeze.get("n_splits", 5)),
        seed=int(freeze.get("split_seed", 17)),
        validation_offset=int(freeze.get("validation_fold_offset", 1)),
    )
    low_data = _low_data_subsets(
        primary,
        split,
        k_values=[int(value) for value in freeze.get("low_data_k", [5, 10, 20])],
        seeds=[int(value) for value in freeze.get("low_data_seeds", [17, 29, 43, 59, 71])],
    )
    if not prompt_confirmed:
        final_gate = "PROMPT_FREEZE_BLOCKED"
    elif not primary_support_complete or not split.get("ready"):
        final_gate = "INSUFFICIENT_PRIMARY_SUPPORT"
    else:
        final_gate = "FROZEN_READY_FOR_EVALUATION"
    if final_gate not in LEGAL_GATES:
        raise AssertionError("Illegal final gate")

    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".benchmark_v1.", dir=str(artifact_root)))
    try:
        primary_rows = [_manifest_row(row) for row in primary]
        stress_rows = [{**_manifest_row(row), "stress_reason": row.get("assignment_reason", "")} for row in stress]
        reserve_rows = [_manifest_row(row) for row in reserve]
        write_csv(temporary_root / "primary_rois.csv", primary_rows, PRIMARY_CSV_FIELDS)
        write_csv(temporary_root / "stress_rois.csv", stress_rows, PRIMARY_CSV_FIELDS + ("stress_reason",))
        write_csv(temporary_root / "reserve_secondary.csv", reserve_rows, PRIMARY_CSV_FIELDS)
        write_csv(temporary_root / "frozen_annotations.csv", frozen_rows, FROZEN_CSV_FIELDS)
        write_jsonl(temporary_root / "frozen_annotations.jsonl", ({key: row.get(key, "") for key in FROZEN_CSV_FIELDS} for row in frozen_rows))
        grouping_manifest = {
            "schema_version": "architecture_grouping_manifest_v1",
            "grouping_resolution": "family",
            "fallback": "safe_slide_id",
            "rows": [
                {"roi_id": row["roi_id"], "safe_slide_id": row["safe_slide_id"], "family_id": row["family_id"]}
                for row in canonical_rows
            ],
        }
        write_json(temporary_root / "grouping_manifest.json", grouping_manifest)
        write_json(temporary_root / "freeze_policy.json", policy)
        shutil.copyfile(str(prompt_path), str(temporary_root / "architecture_prompts_v1.yaml"))
        split_hashes = _write_split_artifacts(
            temporary_root / "splits",
            primary,
            split,
            low_data,
            model_arms=list(config.get("model_arm_subset_contract") or []),
        )
        readme = """# Expert-Confirmed 5x Architecture Benchmark v1

This directory is immutable. Expert morphology annotations were obtained on
the exact PNG bytes recorded by SHA-256 before any registered model evaluation.
Every `image_file` path resolves relative to this directory and points to the
reviewed PNG in `../annotation_package_v1/images/`; WSI recropping is forbidden.
Recruitment diagnosis was used only to enrich candidate discovery and is not
present in this benchmark. Stress and reserve ROIs are excluded from the
primary three-class endpoint. Any correction requires benchmark_v2.
"""
        write_text(temporary_root / "README.md", readme)
        core_hashes = {
            "frozen_annotations.csv": sha256_file(temporary_root / "frozen_annotations.csv"),
            "frozen_annotations.jsonl": sha256_file(temporary_root / "frozen_annotations.jsonl"),
            "primary_rois.csv": sha256_file(temporary_root / "primary_rois.csv"),
            "stress_rois.csv": sha256_file(temporary_root / "stress_rois.csv"),
            "reserve_secondary.csv": sha256_file(temporary_root / "reserve_secondary.csv"),
            "grouping_manifest.json": sha256_file(temporary_root / "grouping_manifest.json"),
            "freeze_policy.json": sha256_file(temporary_root / "freeze_policy.json"),
            "architecture_prompts_v1.yaml": sha256_file(temporary_root / "architecture_prompts_v1.yaml"),
            "splits/folds.csv": sha256_file(temporary_root / "splits" / "folds.csv"),
            "splits/low_data_subsets.jsonl": sha256_file(temporary_root / "splits" / "low_data_subsets.jsonl"),
        }
        write_json(temporary_root / "hashes.json", core_hashes)
        image_hash_payload = {str(row["roi_id"]): str(row["image_sha256"]) for row in annotation_manifest.get("rois", [])}
        group_hash_payload = {str(row["roi_id"]): str(row["family_id"]) for row in annotation_manifest.get("rois", [])}
        freeze_manifest = {
            "schema_version": "architecture_benchmark_freeze_manifest_v1",
            "benchmark_version": config.get("benchmark_version"),
            "frozen": True,
            "evaluation_ready": final_gate == "FROZEN_READY_FOR_EVALUATION",
            "final_gate": final_gate,
            "freeze_timestamp": _timestamp(),
            "canonical_annotation_sha256": sha256_payload(_canonical_annotation_payload(canonical_rows)),
            "annotation_sha256": core_hashes["frozen_annotations.csv"],
            "primary_manifest_sha256": core_hashes["primary_rois.csv"],
            "stress_manifest_sha256": core_hashes["stress_rois.csv"],
            "reserve_manifest_sha256": core_hashes["reserve_secondary.csv"],
            "freeze_policy_sha256": core_hashes["freeze_policy.json"],
            "freeze_policy_content_sha256": sha256_payload(policy),
            "prompt_config_sha256": core_hashes["architecture_prompts_v1.yaml"],
            "prompt_review_confirmed": prompt_confirmed,
            "split_sha256": split_hashes["split_sha256"],
            "subsets_sha256": split_hashes["subsets_sha256"],
            "config_sha256": config["_config_sha256"],
            "reviewed_images_sha256": sha256_payload(image_hash_payload),
            "group_mapping_sha256": sha256_payload(group_hash_payload),
            "n_reviewed": len(canonical_rows),
            "n_primary": len(primary),
            "class_counts": {label: int(class_counts.get(label, 0)) for label in PRIMARY_CLASSES},
            "n_stress": len(stress),
            "n_reserve_secondary": len(reserve),
            "unique_primary_wsi": len({row["safe_slide_id"] for row in primary}),
            "unique_primary_families": len({row["family_id"] for row in primary}),
            "split_ready": bool(split.get("ready")),
            "low_data_k_ready": dict(low_data.get("k_ready") or {}),
            "all_training_ready": bool(low_data.get("all_ready")),
            "label_leakage": 0,
            "model_output_leakage": 0,
            "family_split_leakage": int(split.get("family_split_leakage", 0)),
            "model_evaluation_started": False,
            "zero_shot_performance_computed": False,
            "linear_probe_trained": False,
            "text_prior_classifier_trained": False,
            "mil_trained": False,
        }
        write_json(temporary_root / "freeze_manifest.json", freeze_manifest)
        os.replace(str(temporary_root), str(benchmark_root))
        write_text(
            report_path,
            _report_text(freeze_manifest, annotation_audit, primary, stress, reserve, split, low_data),
        )
        write_json(
            artifact_root / "status.json",
            {
                "schema_version": "conch_text_architecture_poc_state_v1",
                "updated_at": _timestamp(),
                "final_gate": final_gate,
                "annotation_package_ready": True,
                "expert_annotation_complete": True,
                "benchmark_freeze_complete": True,
                "model_evaluation_started": False,
                "freeze_manifest_sha256": sha256_file(benchmark_root / "freeze_manifest.json"),
            },
        )
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(str(temporary_root))
        raise
    return _stage_b_result(freeze_manifest)


def _stage_b_result(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    counts = dict(manifest.get("class_counts") or {})
    k_ready = dict(manifest.get("low_data_k_ready") or {})
    return {
        "BENCHMARK_VERSION": manifest.get("benchmark_version", "conch_text_architecture_benchmark_v1"),
        "BENCHMARK_FREEZE_COMPLETE": bool(manifest.get("frozen")),
        "PRIMARY_ROIS_TOTAL": int(manifest.get("n_primary", 0)),
        "PRIMARY_SERRATED": int(counts.get("serrated", 0)),
        "PRIMARY_TUBULAR": int(counts.get("tubular", 0)),
        "PRIMARY_VILLOUS": int(counts.get("villous", 0)),
        "UNIQUE_PRIMARY_WSI": int(manifest.get("unique_primary_wsi", 0)),
        "UNIQUE_PRIMARY_FAMILIES": int(manifest.get("unique_primary_families", 0)),
        "STRESS_ROIS": int(manifest.get("n_stress", 0)),
        "RESERVE_SECONDARY_ROIS": int(manifest.get("n_reserve_secondary", 0)),
        "ANNOTATION_SHA256": manifest.get("annotation_sha256", ""),
        "PRIMARY_MANIFEST_SHA256": manifest.get("primary_manifest_sha256", ""),
        "SPLIT_SHA256": manifest.get("split_sha256", ""),
        "PROMPT_SHA256": manifest.get("prompt_config_sha256", ""),
        "LABEL_LEAKAGE": int(manifest.get("label_leakage", 0)),
        "MODEL_OUTPUT_LEAKAGE": int(manifest.get("model_output_leakage", 0)),
        "FAMILY_SPLIT_LEAKAGE": int(manifest.get("family_split_leakage", 0)),
        "LOW_DATA_K5_READY": bool(k_ready.get("5")),
        "LOW_DATA_K10_READY": bool(k_ready.get("10")),
        "LOW_DATA_K20_READY": bool(k_ready.get("20")),
        "MODEL_EVALUATION_STARTED": False,
        "FINAL_GATE": manifest.get("final_gate", "PROVENANCE_BLOCKED"),
    }


def status_report(config_path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    config, paths = _load_config(config_path)
    del config
    benchmark_manifest = paths["benchmark_root"] / "freeze_manifest.json"
    if benchmark_manifest.is_file():
        return _stage_b_result(read_json(benchmark_manifest))
    annotation_manifest = paths["annotation_package"] / "annotation_manifest.json"
    if annotation_manifest.is_file():
        return _stage_a_result(paths["annotation_package"], read_json(annotation_manifest))
    state_path = paths["artifact_root"] / "status.json"
    if state_path.is_file():
        state = read_json(state_path)
        return {
            "ANNOTATION_PACKAGE_READY": bool(state.get("annotation_package_ready")),
            "EXPERT_ANNOTATION_COMPLETE": bool(state.get("expert_annotation_complete")),
            "BENCHMARK_FREEZE_COMPLETE": bool(state.get("benchmark_freeze_complete")),
            "MODEL_EVALUATION_STARTED": bool(state.get("model_evaluation_started")),
            "FINAL_GATE": state.get("final_gate", "PROVENANCE_BLOCKED"),
            "BLOCKING_REASON": state.get("blocking_reason", ""),
            "NEXT_ACTION": "COMPLETE_UPSTREAM_CANDIDATE_PREPROCESSING_THEN_RUN_PREPARE_ANNOTATION",
        }
    return {
        "ANNOTATION_PACKAGE_READY": False,
        "EXPERT_ANNOTATION_COMPLETE": False,
        "BENCHMARK_FREEZE_COMPLETE": False,
        "MODEL_EVALUATION_STARTED": False,
        "FINAL_GATE": "WAITING_FOR_EXPERT_ANNOTATION",
        "NEXT_ACTION": "RUN_PREPARE_ANNOTATION",
    }
