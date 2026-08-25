"""Per-slide Mucosa extraction driver for resumable baseline preprocessing."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

from adenoma_agent.mucosa_extractor import MucosaExtractorConfig, run_mucosa_extractor

from .io import iter_jsonl, read_json, sha256_payload, write_json, write_jsonl


def _progress(event: str, payload: Mapping[str, Any]) -> None:
    print(json.dumps({"event": event, **dict(payload)}, ensure_ascii=False), flush=True)


def _source_stat(path: Path) -> Mapping[str, Any]:
    stat = Path(path).stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _completed_boundary_matches(boundary_path: Path, expected_fingerprint: str) -> bool:
    if not Path(boundary_path).is_file():
        return False
    try:
        boundary = read_json(boundary_path)
    except Exception:
        return False
    return bool(
        boundary.get("status") == "complete"
        and int(boundary.get("counts", {}).get("errors", 0)) == 0
        and boundary.get("run_fingerprint") == expected_fingerprint
        and Path(boundary.get("five_x_patch_manifest", "")).is_file()
    )


def run_mucosa_per_slide(
    case_paths_jsonl: Path,
    output_root: Path,
    pathprism_url: str,
    batch_size: int = 16,
    timeout_seconds: int = 240,
    base_magnification: float = 0.0,
    mpp: float = 0.0,
    min_tissue_coverage: float = 0.05,
    mask_downsample: float = 32.0,
    mucosa_threshold: float = 0.30,
    resume: bool = False,
    limit_cases: int = 0,
    skip_qc_panel: bool = True,
    source_code: str = "YX",
    shard_count: int = 1,
    shard_index: int = 0,
) -> Mapping[str, Any]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = sorted(iter_jsonl(case_paths_jsonl), key=lambda row: str(row.get("case_alias", "")))
    if str(source_code).strip():
        wanted = str(source_code).strip().upper()
        cases = [row for row in cases if str(row.get("source_code", "")).strip().upper() == wanted]
    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    cases = cases[shard_index::shard_count]
    if int(limit_cases) > 0:
        cases = cases[: int(limit_cases)]
    completed = []
    failed = []
    skipped = []
    for case in cases:
        case_alias = str(case.get("case_alias", "")).strip()
        source_path = Path(str(case.get("source_path", ""))).resolve()
        if not case_alias or not source_path.is_file():
            failed.append({"case_alias": case_alias, "error": "missing case_alias or source_path"})
            continue
        case_output = output_root / case_alias
        run_config = {
            "schema_version": "architecture_baseline_mucosa_run_v1",
            "case_alias": case_alias,
            "source_stat": _source_stat(source_path),
            "pathprism_url": str(pathprism_url),
            "batch_size": int(batch_size),
            "timeout_seconds": int(timeout_seconds),
            "base_magnification": float(base_magnification),
            "mpp": float(mpp),
            "min_tissue_coverage": float(min_tissue_coverage),
            "mask_downsample": float(mask_downsample),
            "mucosa_threshold": float(mucosa_threshold),
            "five_x_patch_size": 256,
        }
        fingerprint = sha256_payload(run_config)
        boundary_path = case_output / "baseline_boundary.json"
        if resume and _completed_boundary_matches(boundary_path, fingerprint):
            skipped.append(case_alias)
            _progress("architecture_mucosa_case_skipped", {"case_alias": case_alias})
            continue
        started = time.time()
        try:
            manifest = run_mucosa_extractor(
                wsi_paths=[source_path],
                output_dir=case_output,
                config=MucosaExtractorConfig(
                    pathprism_url=str(pathprism_url),
                    batch_size=int(batch_size),
                    timeout_seconds=int(timeout_seconds),
                    base_magnification=float(base_magnification),
                    mpp=float(mpp),
                    min_tissue_coverage=float(min_tissue_coverage),
                    mask_downsample=float(mask_downsample),
                    mucosa_threshold=float(mucosa_threshold),
                    five_x_patch_size=256,
                    write_overlays=not bool(skip_qc_panel),
                    resume=bool(resume),
                ),
            )
            boundary = {
                **run_config,
                "status": manifest.get("status", "complete"),
                "run_fingerprint": fingerprint,
                "elapsed_seconds": round(time.time() - started, 3),
                "five_x_patch_manifest": str((case_output / "five_x_patch_manifest.jsonl").resolve()),
                "counts": dict(manifest.get("counts", {})),
            }
            write_json(boundary_path, boundary)
            if boundary["status"] == "complete" and int(boundary["counts"].get("errors", 0)) == 0:
                completed.append(
                    {"case_alias": case_alias, "status": boundary["status"], "counts": boundary["counts"]}
                )
                _progress(
                    "architecture_mucosa_case_complete",
                    {
                        "case_alias": case_alias,
                        "status": boundary["status"],
                        "elapsed_seconds": boundary["elapsed_seconds"],
                        "five_x_patches": boundary["counts"].get("five_x_patches", 0),
                    },
                )
            else:
                failure_row = {
                    "case_alias": case_alias,
                    "error_type": "MucosaTileErrors",
                    "error": "status={0}, errors={1}".format(
                        boundary["status"], boundary["counts"].get("errors", 0)
                    ),
                }
                failed.append(failure_row)
                _progress("architecture_mucosa_case_failed", failure_row)
        except Exception as exc:
            failure = {
                **run_config,
                "status": "failed",
                "run_fingerprint": fingerprint,
                "elapsed_seconds": round(time.time() - started, 3),
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
            write_json(boundary_path, failure)
            failed.append({"case_alias": case_alias, "error_type": exc.__class__.__name__, "error": str(exc)})
            _progress(
                "architecture_mucosa_case_failed",
                {
                    "case_alias": case_alias,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
    summary = {
        "schema_version": "architecture_baseline_mucosa_batch_v1",
        "requested_cases": len(cases),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "completed": completed,
        "skipped_complete": skipped,
        "failed": failed,
        "status": "complete" if not failed else "complete_with_failures",
    }
    summary_name = (
        "batch_summary.json"
        if shard_count == 1
        else "batch_summary.shard_{0:03d}_of_{1:03d}.json".format(shard_index, shard_count)
    )
    write_json(output_root / summary_name, summary)
    summary["summary_path"] = str((output_root / summary_name).resolve())
    return summary


def aggregate_mucosa_shards(output_root: Path, shard_count: int) -> Mapping[str, Any]:
    """Validate non-overlapping shard summaries and write the formal batch summary."""

    output_root = Path(output_root).resolve()
    shard_count = int(shard_count)
    summaries = []
    for shard_index in range(shard_count):
        path = output_root / "batch_summary.shard_{0:03d}_of_{1:03d}.json".format(
            shard_index, shard_count
        )
        payload = read_json(path)
        if int(payload.get("shard_count", -1)) != shard_count:
            raise ValueError("Mucosa shard_count mismatch: {0}".format(path))
        if int(payload.get("shard_index", -1)) != shard_index:
            raise ValueError("Mucosa shard_index mismatch: {0}".format(path))
        summaries.append(payload)
    completed = [row for payload in summaries for row in payload.get("completed", [])]
    skipped = [row for payload in summaries for row in payload.get("skipped_complete", [])]
    failed = [row for payload in summaries for row in payload.get("failed", [])]
    aliases = [str(row.get("case_alias", "")) for row in completed]
    aliases.extend(str(value) for value in skipped)
    aliases.extend(str(row.get("case_alias", "")) for row in failed)
    if len(aliases) != len(set(aliases)):
        raise ValueError("Mucosa shard aggregation found duplicate case aliases")
    aggregate = {
        "schema_version": "architecture_baseline_mucosa_batch_v1",
        "requested_cases": sum(int(payload.get("requested_cases", 0)) for payload in summaries),
        "shard_count": shard_count,
        "shard_index": None,
        "completed": completed,
        "skipped_complete": skipped,
        "failed": failed,
        "status": "complete" if not failed else "complete_with_failures",
        "source_shard_summaries": [
            str(
                (
                    output_root
                    / "batch_summary.shard_{0:03d}_of_{1:03d}.json".format(index, shard_count)
                ).resolve()
            )
            for index in range(shard_count)
        ],
    }
    if aggregate["requested_cases"] != len(aliases):
        raise ValueError("Mucosa shard aggregate requested/result counts disagree")
    write_json(output_root / "batch_summary.json", aggregate)
    experiment_root = output_root.parent
    case_paths = experiment_root / "data_audit" / "local_provenance" / "case_paths.jsonl"
    canonical_labels = experiment_root / "data_audit" / "labels" / "canonical_labels.jsonl"
    if case_paths.is_file() and canonical_labels.is_file():
        eligibility = build_mucosa_eligibility_ledger(
            case_paths,
            output_root,
            experiment_root / "manifests",
            canonical_labels_jsonl=canonical_labels,
        )
        aggregate["mucosa_eligibility_summary"] = str(
            (experiment_root / "manifests" / "mucosa_eligibility_summary.json").resolve()
        )
        aggregate["eligible_cases"] = int(eligibility["eligible_cases"])
        write_json(output_root / "batch_summary.json", aggregate)
    return aggregate


def build_mucosa_eligibility_ledger(
    case_paths_jsonl: Path,
    mucosa_root: Path,
    output_dir: Path,
    canonical_labels_jsonl: Optional[Path] = None,
    source_code: str = "YX",
    manifest_threshold: float = 0.30,
    training_threshold: float = 0.60,
) -> Mapping[str, Any]:
    """Write the full-cohort technical and training eligibility ledger.

    Technical failures are recorded before this function raises. Clean slides
    with no patch at the training threshold are retained as explicit,
    pre-registered attrition rather than being treated as runtime failures.
    """

    mucosa_root = Path(mucosa_root).resolve()
    output_dir = Path(output_dir).resolve()
    wanted = str(source_code).strip().upper()
    cases = [
        row
        for row in iter_jsonl(case_paths_jsonl)
        if str(row.get("source_code", "")).strip().upper() == wanted
    ]
    cases.sort(key=lambda row: str(row.get("case_alias", "")))
    labels = {}
    if canonical_labels_jsonl is not None:
        for row in iter_jsonl(canonical_labels_jsonl):
            if str(row.get("source_code", "")).strip().upper() != wanted:
                continue
            alias = str(row.get("case_alias", ""))
            if alias in labels:
                raise ValueError("Duplicate canonical label for {0}".format(alias))
            labels[alias] = row

    ledger_rows = []
    for case in cases:
        alias = str(case.get("case_alias", "")).strip()
        case_dir = mucosa_root / alias
        boundary_path = case_dir / "baseline_boundary.json"
        manifest_path = case_dir / "five_x_patch_manifest.jsonl"
        errors_path = case_dir / "errors.jsonl"
        boundary = read_json(boundary_path) if boundary_path.is_file() else {}
        boundary_status = str(boundary.get("status", "missing"))
        counts = dict(boundary.get("counts", {}) or {})
        declared_errors = int(counts.get("errors", 0) or 0)
        actual_errors = sum(1 for _row in iter_jsonl(errors_path)) if errors_path.is_file() else 0
        rows = list(iter_jsonl(manifest_path)) if manifest_path.is_file() else []
        coverages = [float(row.get("mucosa_coverage", 0.0)) for row in rows]
        n_total = len(rows)
        n_ge_manifest = sum(value >= float(manifest_threshold) for value in coverages)
        n_ge_training = sum(value >= float(training_threshold) for value in coverages)
        technical_reasons = []
        if not boundary_path.is_file():
            technical_reasons.append("missing_boundary")
        if boundary_status != "complete":
            technical_reasons.append("boundary_status_{0}".format(boundary_status))
        if declared_errors or actual_errors:
            technical_reasons.append("tile_errors")
        if not manifest_path.is_file():
            technical_reasons.append("missing_five_x_manifest")
        if manifest_path.is_file() and int(counts.get("five_x_patches", n_total)) != n_total:
            technical_reasons.append("manifest_count_mismatch")
        technical_outcome = "complete" if not technical_reasons else "technical_failure"
        eligible = bool(technical_outcome == "complete" and n_ge_training > 0)
        if technical_reasons:
            exclusion_reason = ";".join(sorted(set(technical_reasons)))
        elif n_total == 0:
            exclusion_reason = "no_five_x_patch"
        elif n_ge_training == 0:
            exclusion_reason = "no_patch_at_training_coverage"
        else:
            exclusion_reason = ""
        label = labels.get(alias, {})
        ledger_rows.append(
            {
                "schema_version": "architecture_mucosa_eligibility_v1",
                "case_alias": alias,
                "boundary_status": boundary_status,
                "declared_errors": declared_errors,
                "actual_error_records": actual_errors,
                "valid_tiles": int(counts.get("valid_tiles", 0) or 0),
                "retained_tiles": int(counts.get("retained_tiles", 0) or 0),
                "n_5x_total": n_total,
                "n_ge_0_30": n_ge_manifest,
                "n_ge_0_60": n_ge_training,
                "technical_outcome": technical_outcome,
                "eligible_for_training": eligible,
                "exclusion_reason": exclusion_reason,
                "label": label.get("label", ""),
                "grade": label.get("grade", ""),
                "family_id": label.get("family_id", case.get("family_id", "")),
            }
        )

    write_jsonl(output_dir / "mucosa_eligibility_ledger.jsonl", ledger_rows)
    technical_failures = [row for row in ledger_rows if row["technical_outcome"] != "complete"]
    eligible_rows = [row for row in ledger_rows if row["eligible_for_training"]]
    attrition_by_reason = Counter(
        row["exclusion_reason"] for row in ledger_rows if not row["eligible_for_training"]
    )
    class_total = Counter(str(row["label"]) for row in ledger_rows)
    class_eligible = Counter(str(row["label"]) for row in eligible_rows)
    summary = {
        "schema_version": "architecture_mucosa_eligibility_summary_v1",
        "source_code": wanted,
        "requested_cases": len(cases),
        "technical_complete_cases": len(ledger_rows) - len(technical_failures),
        "technical_failure_cases": len(technical_failures),
        "eligible_cases": len(eligible_rows),
        "ineligible_cases": len(ledger_rows) - len(eligible_rows),
        "manifest_threshold": float(manifest_threshold),
        "training_threshold": float(training_threshold),
        "attrition_by_reason": dict(sorted(attrition_by_reason.items())),
        "class_total": dict(sorted(class_total.items())),
        "class_eligible": dict(sorted(class_eligible.items())),
        "ledger": str((output_dir / "mucosa_eligibility_ledger.jsonl").resolve()),
    }
    write_json(output_dir / "mucosa_eligibility_summary.json", summary)
    if technical_failures:
        raise RuntimeError(
            "Formal Mucosa technical gate failed for {0} cases; see {1}".format(
                len(technical_failures), summary["ledger"]
            )
        )
    return summary
