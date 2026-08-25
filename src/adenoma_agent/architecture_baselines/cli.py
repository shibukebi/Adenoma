"""Command line orchestration for the Frozen-CONCH 5x baseline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import yaml

from .conch import (
    CONCH_EMBEDDING_DIM,
    conch_gpu_preflight,
    encode_manifest_cache,
    validate_cache_root,
)
from .data import annotation_guard, audit_architecture_data
from .evaluation import (
    classification_metrics,
    family_cluster_bootstrap,
    holm_adjust,
    paired_family_bootstrap,
    patch_ranking_summary,
)
from .io import (
    iter_jsonl,
    read_json,
    require_output_outside_sources,
    sha256_file,
    sha256_payload,
    write_json,
    write_jsonl,
)
from .manifest import eligible_case_aliases, merge_sanitized_manifests, validate_manifest
from .models import build_model
from .mucosa import run_mucosa_per_slide
from .training import load_cached_bags, predict_bags, train_baseline
from adenoma_agent.wsi import WSIReader


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "architecture_baselines" / "frozen_conch_5x_mil_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "architecture_baselines" / "frozen_conch_5x_mil_v1"
DEFAULT_YX_ROOT = Path("/mnt/zhengke_usb2/yuexin_data/Adenoma_yx")
DEFAULT_HP_ROOT = Path("/mnt/zhengke_usb2/yuexin_data/Adenoma_hp")
DEFAULT_LABELS = REPO_ROOT / "data" / "label" / "Adenoma_filtered.xlsx"
DEFAULT_CONCH = REPO_ROOT / "models" / "CONCH" / "pytorch_model.bin"


def _load_config(path: Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Baseline config must be a YAML object")
    return payload


def _git_state() -> Mapping[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty_status": run("status", "--short").splitlines(),
    }


def _common_provenance(config_path: Path) -> Mapping[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "config": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "git": _git_state(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _require_explicit_cuda(device: str) -> None:
    if str(device) != "cuda:0":
        raise ValueError(
            "This baseline requires logical --device cuda:0; select the physical GPU with CUDA_VISIBLE_DEVICES"
        )
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if not visible.isdigit() or "," in visible:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must explicitly select exactly one physical GPU index, got {0!r}".format(
                visible
            )
        )


def command_preflight(args) -> Mapping[str, Any]:
    _require_explicit_cuda(args.device)
    output_root = Path(args.output_root).resolve()
    require_output_outside_sources(output_root, [args.yx_root, args.hp_root])
    output_root.mkdir(parents=True, exist_ok=True)
    dependencies = {}
    for module_name in (
        "torch",
        "torchvision",
        "sklearn",
        "scipy",
        "conch",
        "timm",
        "fastapi",
        "uvicorn",
    ):
        try:
            module = __import__(module_name)
            dependencies[module_name] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:
            raise RuntimeError("Required dependency {0} is unavailable: {1}".format(module_name, exc))
    disk = shutil.disk_usage(str(output_root))
    if int(disk.free) < 20 * 1024 ** 3:
        raise RuntimeError("Artifact filesystem has less than 20 GiB free")
    yx_candidates = sorted(Path(args.yx_root).glob("*.svs"))
    if not yx_candidates:
        raise RuntimeError("No YX SVS is available for the WSI preflight")
    with WSIReader(yx_candidates[0]) as reader:
        reader.require_physical_metadata()
        wsi_preflight = {
            "source_code": "YX",
            "source_format": "svs",
            "backend": reader.backend_name,
            "dimensions": list(reader.dimensions),
            "base_magnification": reader.base_magnification,
            "mpp_x": reader.mpp_x,
            "mpp_y": reader.mpp_y,
        }
    physical_gpu_index = str(os.environ["CUDA_VISIBLE_DEVICES"]).strip()
    nvidia = subprocess.run(
        [
            "nvidia-smi",
            "--id={0}".format(physical_gpu_index),
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if nvidia.returncode != 0:
        raise RuntimeError("nvidia-smi failed; CPU fallback is forbidden: {0}".format(nvidia.stderr.strip()))
    result = dict(
        conch_gpu_preflight(
            checkpoint_path=args.checkpoint,
            device=args.device,
            candidate_batch_sizes=[int(value) for value in args.batch_candidates.split(",")],
        )
    )
    result.update(_common_provenance(args.config))
    expected_hash = str(_load_config(args.config).get("conch", {}).get("checkpoint_sha256", ""))
    if expected_hash and result["checkpoint_sha256"] != expected_hash:
        raise RuntimeError("CONCH checkpoint SHA-256 does not match the registered config")
    result["nvidia_smi"] = nvidia.stdout.strip().splitlines()
    result["physical_gpu_index"] = int(physical_gpu_index)
    result["logical_torch_device"] = args.device
    result["dependencies"] = dependencies
    result["artifact_disk_free_bytes"] = int(disk.free)
    result["wsi_preflight"] = wsi_preflight
    result["status"] = "passed"
    write_json(output_root / "resolved_runtime.json", result)
    return result


def command_audit(args) -> Mapping[str, Any]:
    output_root = Path(args.output_root).resolve()
    require_output_outside_sources(output_root, [args.yx_root, args.hp_root])
    audit = audit_architecture_data(
        label_workbook=args.label_workbook,
        yx_root=args.yx_root,
        hp_root=args.hp_root,
        output_dir=output_root / "data_audit",
        seed=int(args.seed),
    )
    return {
        "summary": dict(audit.summary),
        "artifact_paths": dict(audit.artifact_paths),
        "annotation": audit.annotation.to_json(),
    }


def command_run_mucosa(args) -> Mapping[str, Any]:
    output_root = Path(args.output_root).resolve()
    require_output_outside_sources(output_root, [args.yx_root, args.hp_root])
    result = run_mucosa_per_slide(
        case_paths_jsonl=args.case_paths,
        output_root=output_root / "mucosa_by_slide",
        pathprism_url=args.pathprism_url,
        batch_size=int(args.batch_size),
        timeout_seconds=int(args.timeout_seconds),
        base_magnification=float(args.base_magnification),
        mpp=float(args.mpp),
        min_tissue_coverage=float(args.min_tissue_coverage),
        mask_downsample=float(args.mask_downsample),
        mucosa_threshold=float(args.mucosa_threshold),
        resume=bool(args.resume),
        limit_cases=int(args.limit_cases),
        skip_qc_panel=bool(args.skip_qc_panel),
        source_code="YX",
        shard_count=int(args.shard_count),
        shard_index=int(args.shard_index),
    )
    return {
        "status": result["status"],
        "requested_cases": result["requested_cases"],
        "completed_count": len(result["completed"]),
        "skipped_complete_count": len(result["skipped_complete"]),
        "failed_count": len(result["failed"]),
        "summary": result["summary_path"],
    }


def command_validate_manifest(args) -> Mapping[str, Any]:
    output_root = Path(args.output_root).resolve()
    manifest = output_root / "manifests" / "five_x_patch_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.merge:
        summary = merge_sanitized_manifests(
            case_paths_jsonl=args.case_paths,
            mucosa_root=output_root / "mucosa_by_slide",
            output_manifest=manifest,
        )
        return {
            "status": "complete",
            "manifest": str(manifest.resolve()),
            "manifest_sha256": summary["manifest_sha256"],
            "cases": summary["cases"],
            "patches": summary["patches"],
            "coverage_ge_0_30": summary["coverage_ge_0_30"],
            "coverage_ge_0_60": summary["coverage_ge_0_60"],
        }
    return validate_manifest(args.manifest or manifest, require_physical=True)


def command_embed_conch(args) -> Mapping[str, Any]:
    _require_explicit_cuda(args.device)
    output_root = Path(args.output_root).resolve()
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        resolved = read_json(output_root / "resolved_runtime.json")
        batch_size = int(resolved["selected_conch_batch_size"])
    summary = encode_manifest_cache(
        manifest_path=args.manifest,
        case_paths_jsonl=args.case_paths,
        cache_root=output_root / "conch_embeddings",
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=batch_size,
        minimum_coverage=float(args.minimum_coverage),
        resume=bool(args.resume),
        use_amp=not bool(args.no_amp),
        limit_cases=int(args.limit_cases),
    )
    validation = validate_cache_root(
        output_root / "conch_embeddings",
        expected_case_aliases=summary["requested_case_aliases"],
    )
    validation_path = output_root / "conch_embeddings" / "cache_validation.json"
    write_json(validation_path, validation)
    summary["validation"] = {
        "cases": validation["cases"],
        "patches": validation["patches"],
        "embedding_dim": validation["embedding_dim"],
        "artifact": str(validation_path.resolve()),
    }
    write_json(output_root / "conch_embeddings" / "cache_summary.json", summary)
    return {
        "status": "complete",
        "requested_cases": summary["requested_cases"],
        "completed_count": len(summary["completed_cases"]),
        "skipped_count": len(summary["skipped_cases"]),
        "patches": validation["patches"],
        "embedding_dim": validation["embedding_dim"],
        "summary": str((output_root / "conch_embeddings" / "cache_summary.json").resolve()),
    }


def _read_case_aliases(path: Path) -> List[str]:
    payload = read_json(path)
    if isinstance(payload, list):
        return [str(value.get("case_alias", "") if isinstance(value, dict) else value) for value in payload]
    return [str(value) for value in payload.get("case_aliases", [])]


def command_build_splits(args) -> Mapping[str, Any]:
    from .data import CanonicalLabel, write_five_fold_split_artifacts

    eligible = eligible_case_aliases(args.manifest, minimum_coverage=float(args.minimum_coverage))
    labels = [
        CanonicalLabel(
            case_alias=str(row["case_alias"]),
            source_code=str(row["source_code"]),
            label=str(row["label"]),
            grade=str(row.get("grade", "")),
            family_id=str(row["family_id"]),
            label_source_sha256=str(row["label_source_sha256"]),
        )
        for row in iter_jsonl(args.labels)
    ]
    folds = write_five_fold_split_artifacts(
        labels,
        output_dir=Path(args.output_root) / "splits",
        eligible_aliases=eligible,
        seed=int(args.seed),
    )
    return read_json(Path(args.output_root) / "splits" / "split_summary.json")


def command_baseline1_status(args) -> Mapping[str, Any]:
    manifest_rows = list(iter_jsonl(args.manifest))
    annotation_rows = list(iter_jsonl(args.annotations)) if args.annotations else []
    result = annotation_guard(manifest_rows, annotation_rows=annotation_rows, slide_labels=None)
    result["schema_version"] = "architecture_baseline1_status_v1"
    result["annotations"] = str(Path(args.annotations).resolve()) if args.annotations else ""
    result["slide_labels_used"] = False
    write_json(Path(args.output_root) / "baseline1" / "annotation_status.json", result)
    return result


def _class_map(labels_path: Path) -> Mapping[str, int]:
    labels = sorted({str(row["label"]) for row in iter_jsonl(labels_path)})
    if len(labels) != 7:
        raise ValueError("Expected exactly seven training labels, got {0}".format(labels))
    return {label: index for index, label in enumerate(labels)}


def _split_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, List[Mapping[str, Any]]]:
    output = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row.get("split", ""))
        if split in output:
            output[split].append(row)
    if any(not output[name] for name in output):
        raise ValueError("Each fold requires non-empty train/val/test bags")
    return output


def _write_run_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_jsonl(path, rows)


def command_train_mil(args) -> Mapping[str, Any]:
    _require_explicit_cuda(args.device)
    output_root = Path(args.output_root).resolve()
    class_to_index = _class_map(args.labels)
    write_json(output_root / "splits" / "class_map.json", class_to_index)
    cache_summary = read_json(output_root / "conch_embeddings" / "cache_summary.json")
    validate_cache_root(
        output_root / "conch_embeddings",
        expected_case_aliases=cache_summary.get("requested_case_aliases"),
    )
    common_provenance = _common_provenance(args.config)
    model_names = [value.strip() for value in args.models.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    folds = range(5) if args.folds == "all" else [int(value) for value in args.folds.split(",")]
    summaries = []
    for fold in folds:
        fold_dir = output_root / "splits" / "fold_{0}".format(fold)
        bags = load_cached_bags(
            cache_root=output_root / "conch_embeddings",
            labels_jsonl=args.labels,
            split_dir=fold_dir,
            min_mucosa_coverage=float(args.minimum_coverage),
            class_to_index=class_to_index,
            feature_dim=CONCH_EMBEDDING_DIM,
        )
        split = _split_rows(bags)
        for model_name in model_names:
            for seed in seeds:
                run_dir = output_root / "runs" / model_name / "fold_{0}".format(fold) / "seed_{0}".format(seed)
                test_path = run_dir / "test_predictions.jsonl"
                run_manifest_path = run_dir / "run_manifest.json"
                checkpoint_path = run_dir / "best_checkpoint.pt"
                training_summary_path = run_dir / "training_summary.json"
                split_summary_sha256 = sha256_file(fold_dir / "split_summary.json")
                cache_summary_sha256 = sha256_file(output_root / "conch_embeddings" / "cache_summary.json")
                training_protocol = {
                    "model": model_name,
                    "fold": int(fold),
                    "seed": int(seed),
                    "feature_dim": CONCH_EMBEDDING_DIM,
                    "num_classes": 7,
                    "minimum_mucosa_coverage": float(args.minimum_coverage),
                    "hidden_dim": int(args.hidden_dim),
                    "dropout": float(args.dropout),
                    "max_epochs": int(args.max_epochs),
                    "patience": int(args.patience),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "class_map": class_to_index,
                    "split_summary_sha256": split_summary_sha256,
                    "cache_summary_sha256": cache_summary_sha256,
                    "conch_checkpoint_sha256": cache_summary.get("checkpoint_sha256"),
                }
                run_fingerprint = sha256_payload(training_protocol)
                # A test-prediction file alone is not sufficient evidence that
                # a run is complete: the downstream real-smoke restores the
                # selected checkpoint and training summary.  If a long job was
                # interrupted after predictions were written, force this run
                # to retrain instead of deferring the failure to smoke.
                if args.resume and test_path.is_file() and checkpoint_path.is_file() and training_summary_path.is_file():
                    if run_manifest_path.is_file():
                        previous = read_json(run_manifest_path)
                        if previous.get("run_fingerprint") != run_fingerprint:
                            raise RuntimeError(
                                "Resume fingerprint mismatch for {0}; use a new output root for a changed protocol".format(
                                    run_dir
                                )
                            )
                        prediction_aliases = {str(row["case_alias"]) for row in iter_jsonl(test_path)}
                        expected_aliases = {str(row["case_alias"]) for row in split["test"]}
                        if prediction_aliases != expected_aliases:
                            raise RuntimeError("Resume test prediction cohort mismatch for {0}".format(run_dir))
                        summaries.append(
                            {"model": model_name, "fold": fold, "seed": seed, "status": "skipped_complete"}
                        )
                        continue
                print(
                    json.dumps(
                        {
                            "event": "architecture_training_run_started",
                            "model": model_name,
                            "fold": int(fold),
                            "seed": int(seed),
                            "train_cases": len(split["train"]),
                            "val_cases": len(split["val"]),
                            "test_cases": len(split["test"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                model = build_model(
                    model_name,
                    feature_dim=CONCH_EMBEDDING_DIM,
                    num_classes=7,
                    hidden_dim=int(args.hidden_dim),
                    dropout=float(args.dropout),
                )
                training_started = time.time()
                train_summary = train_baseline(
                    model=model,
                    train_rows=split["train"],
                    val_rows=split["val"],
                    output_dir=run_dir,
                    model_name=model_name,
                    seed=seed,
                    max_epochs=int(args.max_epochs),
                    patience=int(args.patience),
                    learning_rate=float(args.learning_rate),
                    weight_decay=float(args.weight_decay),
                    num_classes=7,
                    feature_dim=CONCH_EMBEDDING_DIM,
                    device=args.device,
                )
                training_seconds = time.time() - training_started
                inference_started = time.time()
                test_predictions = predict_bags(model, split["test"], device=args.device, feature_dim=CONCH_EMBEDDING_DIM)
                inference_seconds = time.time() - inference_started
                _write_run_predictions(test_path, test_predictions)
                run_manifest = {
                    "schema_version": "architecture_baseline_training_run_v1",
                    "model": model_name,
                    "fold": fold,
                    "seed": seed,
                    "train_cases": len(split["train"]),
                    "val_cases": len(split["val"]),
                    "test_cases": len(split["test"]),
                    "test_predictions": str(test_path.resolve()),
                    "training": train_summary,
                    "training_seconds": round(training_seconds, 3),
                    "test_inference_seconds": round(inference_seconds, 3),
                    "test_inference_seconds_per_case": round(inference_seconds / max(1, len(split["test"])), 6),
                    "test_used_for_selection": False,
                    "encoder_trainable": False,
                    "training_protocol": training_protocol,
                    "run_fingerprint": run_fingerprint,
                    "class_map": class_to_index,
                    "split_summary_sha256": split_summary_sha256,
                    "cache_summary_sha256": cache_summary_sha256,
                    "conch_checkpoint_sha256": cache_summary.get("checkpoint_sha256"),
                    "config": common_provenance,
                }
                write_json(run_manifest_path, run_manifest)
                summaries.append(run_manifest)
                print(
                    json.dumps(
                        {
                            "event": "architecture_training_run_complete",
                            "model": model_name,
                            "fold": int(fold),
                            "seed": int(seed),
                            "best_epoch": train_summary["best_epoch"],
                            "best_val_macro_f1": train_summary["best_val_macro_f1"],
                            "training_seconds": round(training_seconds, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    result = {"schema_version": "architecture_baseline_training_batch_v1", "runs": summaries}
    summary_path = output_root / "runs" / "training_batch_summary.json"
    write_json(summary_path, result)
    return {
        "status": "complete",
        "run_count": len(summaries),
        "completed_count": sum(row.get("status") != "skipped_complete" for row in summaries),
        "skipped_count": sum(row.get("status") == "skipped_complete" for row in summaries),
        "summary": str(summary_path.resolve()),
    }


def _ensemble_predictions(paths: Sequence[Path]) -> List[Mapping[str, Any]]:
    by_case = defaultdict(list)
    for path in paths:
        for row in iter_jsonl(path):
            by_case[str(row["case_alias"])].append(row)
    output = []
    for case_alias in sorted(by_case):
        rows = by_case[case_alias]
        identity = {(int(row["label"]), str(row["family_id"]), str(row.get("split", ""))) for row in rows}
        if len(identity) != 1:
            raise ValueError("Seed predictions disagree on identity for {0}".format(case_alias))
        probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64).mean(axis=0)
        entropy = float(
            -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
            / np.log(len(probabilities))
        )
        ensemble = {
            "case_alias": case_alias,
            "label": rows[0]["label"],
            "family_id": rows[0]["family_id"],
            "split": rows[0].get("split", "test"),
            "probabilities": probabilities.tolist(),
            "prediction": int(probabilities.argmax()),
            "confidence": float(probabilities.max()),
            "normalized_entropy": entropy,
            "seed_count": len(rows),
        }
        for key in ("attention", "instance_scores", "class_attention"):
            if all(key in row for row in rows):
                arrays = [np.asarray(row[key], dtype=np.float64) for row in rows]
                if len({array.shape for array in arrays}) == 1:
                    ensemble[key] = np.mean(arrays, axis=0).tolist()
        output.append(ensemble)
    return output


def _filtered_patch_index(cache_root: Path, case_alias: str, minimum_coverage: float) -> List[Mapping[str, Any]]:
    return [
        row for row in iter_jsonl(Path(cache_root) / case_alias / "index.jsonl")
        if float(row.get("mucosa_coverage", 0.0)) >= float(minimum_coverage)
    ]


def command_evaluate(args) -> Mapping[str, Any]:
    output_root = Path(args.output_root).resolve()
    class_map = read_json(output_root / "splits" / "class_map.json")
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    evaluation_root = output_root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    oof_by_model = {}
    metrics_by_model = {}
    for model_name in models:
        oof = []
        rankings = []
        fold_metrics = []
        for fold in range(5):
            paths = [
                output_root / "runs" / model_name / "fold_{0}".format(fold) / "seed_{0}".format(seed) / "test_predictions.jsonl"
                for seed in seeds
            ]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing seed predictions: {0}".format(missing))
            fold_predictions = _ensemble_predictions(paths)
            oof.extend(fold_predictions)
            fold_metrics.append({"fold": fold, **classification_metrics(fold_predictions, num_classes=7)})
            if model_name in ("abmil", "dsmil"):
                for row in fold_predictions:
                    patches = _filtered_patch_index(output_root / "conch_embeddings", row["case_alias"], args.minimum_coverage)
                    if model_name == "abmil":
                        scores = row.get("attention", [])
                        if len(scores) != len(patches) or not scores:
                            continue
                        rankings.append(
                            {
                                "case_alias": row["case_alias"],
                                "model": model_name,
                                "semantics": "candidate_diagnostic_relevance",
                                **patch_ranking_summary(patches, scores, top_k=int(args.top_k)),
                            }
                        )
                    else:
                        instance = np.asarray(row.get("instance_scores", []), dtype=np.float64)
                        if instance.ndim != 2 or instance.shape[0] != len(patches):
                            continue
                        predicted_class = int(row["prediction"])
                        true_class = int(row["label"])
                        rankings.append(
                            {
                                "case_alias": row["case_alias"],
                                "model": model_name,
                                "semantics": "candidate_diagnostic_relevance",
                                "predicted_class": predicted_class,
                                "true_class": true_class,
                                "predicted_class_ranking": patch_ranking_summary(
                                    patches,
                                    instance[:, predicted_class].tolist(),
                                    top_k=int(args.top_k),
                                ),
                                "true_class_ranking": patch_ranking_summary(
                                    patches,
                                    instance[:, true_class].tolist(),
                                    top_k=int(args.top_k),
                                ),
                            }
                        )
        if len({row["case_alias"] for row in oof}) != len(oof):
            raise ValueError("OOF predictions contain duplicate cases for {0}".format(model_name))
        oof_by_model[model_name] = oof
        metrics = classification_metrics(oof, num_classes=7)
        seed_metrics = []
        for seed in seeds:
            seed_oof = []
            for fold in range(5):
                seed_oof.extend(
                    iter_jsonl(
                        output_root / "runs" / model_name / "fold_{0}".format(fold)
                        / "seed_{0}".format(seed) / "test_predictions.jsonl"
                    )
                )
            seed_metrics.append({"seed": seed, **classification_metrics(seed_oof, num_classes=7)})
        seed_macro_f1 = np.asarray([row["macro_f1"] for row in seed_metrics], dtype=np.float64)
        metrics["fold_metrics"] = fold_metrics
        metrics["seed_metrics"] = seed_metrics
        metrics["seed_macro_f1_mean"] = float(seed_macro_f1.mean())
        metrics["seed_macro_f1_std"] = float(seed_macro_f1.std(ddof=1)) if len(seed_macro_f1) > 1 else 0.0
        metrics["high_confidence_errors"] = sorted(
            (
                {
                    "case_alias": row["case_alias"],
                    "label": row["label"],
                    "prediction": row["prediction"],
                    "confidence": row["confidence"],
                    "normalized_entropy": row["normalized_entropy"],
                }
                for row in oof
                if int(row["label"]) != int(row["prediction"])
            ),
            key=lambda item: (-float(item["confidence"]), str(item["case_alias"])),
        )[:50]
        metrics["macro_f1_family_bootstrap"] = family_cluster_bootstrap(
            oof, metric_name="macro_f1", num_classes=7, n_bootstrap=int(args.bootstrap), seed=int(args.seed)
        )
        if metrics.get("macro_auc") is not None:
            metrics["macro_auc_family_bootstrap"] = family_cluster_bootstrap(
                oof, metric_name="macro_auc", num_classes=7, n_bootstrap=int(args.bootstrap), seed=int(args.seed)
            )
        metrics_by_model[model_name] = metrics
        write_jsonl(evaluation_root / "{0}_oof_predictions.jsonl".format(model_name), oof)
        write_json(evaluation_root / "{0}_metrics.json".format(model_name), metrics)
        if rankings:
            write_jsonl(evaluation_root / "{0}_patch_rankings.jsonl".format(model_name), rankings)
    training_labels = {
        str(row["case_alias"]): row
        for row in iter_jsonl(output_root / "splits" / "training_labels.jsonl")
    }
    majority_oof = []
    for fold in range(5):
        fold_dir = output_root / "splits" / "fold_{0}".format(fold)
        train_aliases = _read_case_aliases(fold_dir / "train_cases.json")
        test_aliases = _read_case_aliases(fold_dir / "test_cases.json")
        majority = Counter(
            int(training_labels[alias]["label_index"]) for alias in train_aliases
        ).most_common(1)[0][0]
        for alias in test_aliases:
            probabilities = [0.0] * 7
            probabilities[int(majority)] = 1.0
            majority_oof.append(
                {
                    "case_alias": alias,
                    "family_id": training_labels[alias]["family_id"],
                    "label": int(training_labels[alias]["label_index"]),
                    "prediction": int(majority),
                    "probabilities": probabilities,
                }
            )
    majority_metrics = classification_metrics(majority_oof, num_classes=7)
    majority_metrics["macro_f1_family_bootstrap"] = family_cluster_bootstrap(
        majority_oof,
        metric_name="macro_f1",
        num_classes=7,
        n_bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    metrics_by_model["majority"] = majority_metrics
    write_jsonl(evaluation_root / "majority_oof_predictions.jsonl", majority_oof)
    grade_strata = sorted({str(row.get("grade", "")).strip() for row in training_labels.values()} - {""})
    predictions_with_majority = {**oof_by_model, "majority": majority_oof}
    for model_name, predictions in predictions_with_majority.items():
        by_grade = {}
        for grade in grade_strata:
            selected = [
                row
                for row in predictions
                if str(training_labels[str(row["case_alias"])].get("grade", "")).strip() == grade
            ]
            if selected:
                by_grade[grade] = classification_metrics(selected, num_classes=7)
        metrics_by_model[model_name]["post_hoc_grade_strata"] = by_grade
        write_json(evaluation_root / "{0}_metrics.json".format(model_name), metrics_by_model[model_name])
    comparisons = {}
    auc_comparisons = {}
    pairs = [("mean_pool", "abmil"), ("mean_pool", "dsmil"), ("abmil", "dsmil")]
    for first, second in pairs:
        if first not in oof_by_model or second not in oof_by_model:
            continue
        key = "{0}_minus_{1}".format(first, second)
        comparisons[key] = paired_family_bootstrap(
            oof_by_model[first], oof_by_model[second], metric_name="macro_f1",
            num_classes=7, n_bootstrap=int(args.bootstrap), seed=int(args.seed),
        )
        auc_comparisons[key] = paired_family_bootstrap(
            oof_by_model[first], oof_by_model[second], metric_name="macro_auc",
            num_classes=7, n_bootstrap=int(args.bootstrap), seed=int(args.seed),
        )
    correction = holm_adjust({key: value["p_value"] for key, value in comparisons.items()}) if comparisons else {}
    auc_correction = (
        holm_adjust({key: value["p_value"] for key, value in auc_comparisons.items()})
        if auc_comparisons else {}
    )
    result = {
        "schema_version": "architecture_baseline_evaluation_v1",
        "class_map": class_map,
        "models": metrics_by_model,
        "paired_comparisons": comparisons,
        "holm_correction": correction,
        "paired_macro_auc_comparisons": auc_comparisons,
        "macro_auc_holm_correction": auc_correction,
        "attention_semantics": "candidate_diagnostic_relevance",
    }
    write_json(evaluation_root / "evaluation_summary.json", result)
    return {
        "status": "complete",
        "models": {
            name: {
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_auc": metrics.get("macro_auc"),
            }
            for name, metrics in metrics_by_model.items()
        },
        "summary": str((evaluation_root / "evaluation_summary.json").resolve()),
    }


def _best_fold_zero_mil(output_root: Path, seeds: Sequence[int]) -> str:
    scores = {}
    for model in ("abmil", "dsmil"):
        values = []
        losses = []
        for seed in seeds:
            summary = read_json(output_root / "runs" / model / "fold_0" / "seed_{0}".format(seed) / "training_summary.json")
            values.append(float(summary["best_val_macro_f1"]))
            losses.append(float(summary["best_val_loss"]))
        scores[model] = (float(np.mean(values)), -float(np.mean(losses)))
    return max(sorted(scores), key=lambda name: scores[name])


def command_real_smoke(args) -> Mapping[str, Any]:
    """Run a fresh one-slide WSI→Mucosa→CONCH→MIL closure without labels."""

    import torch

    _require_explicit_cuda(args.device)
    output_root = Path(args.output_root).resolve()
    seeds = [int(value) for value in args.seeds.split(",")]
    test_aliases = _read_case_aliases(output_root / "splits" / "fold_0" / "test_cases.json")
    alias = sorted(test_aliases)[0]
    resolver = {str(row["case_alias"]): row for row in iter_jsonl(args.case_paths)}
    if alias not in resolver:
        raise ValueError("Smoke case is missing from local provenance")
    smoke_root = output_root / "real_smoke"
    local = smoke_root / "local_provenance" / "case_paths.jsonl"
    write_jsonl(local, [resolver[alias]])
    mucosa_summary = run_mucosa_per_slide(
        case_paths_jsonl=local,
        output_root=smoke_root / "mucosa_by_slide",
        pathprism_url=args.pathprism_url,
        batch_size=int(args.mucosa_batch_size),
        resume=bool(args.resume),
        limit_cases=1,
        source_code="YX",
    )
    manifest = smoke_root / "manifests" / "five_x_patch_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    merge_sanitized_manifests(local, smoke_root / "mucosa_by_slide", manifest)
    encode_manifest_cache(
        manifest_path=manifest,
        case_paths_jsonl=local,
        cache_root=smoke_root / "conch_embeddings",
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=int(args.conch_batch_size),
        minimum_coverage=0.30,
        resume=bool(args.resume),
        limit_cases=1,
    )
    model_name = _best_fold_zero_mil(output_root, seeds)
    features = np.load(smoke_root / "conch_embeddings" / alias / "features.npy", allow_pickle=False)
    patches = _filtered_patch_index(smoke_root / "conch_embeddings", alias, 0.60)
    full_index = list(iter_jsonl(smoke_root / "conch_embeddings" / alias / "index.jsonl"))
    selected = [index for index, row in enumerate(full_index) if float(row["mucosa_coverage"]) >= 0.60]
    if not selected:
        raise RuntimeError("Smoke case has no patch at training coverage threshold")
    tensor = torch.as_tensor(np.asarray(features[selected], dtype=np.float32), device=args.device).unsqueeze(0)
    probabilities = []
    attention = []
    instance_scores = []
    for seed in seeds:
        checkpoint = torch.load(
            output_root / "runs" / model_name / "fold_0" / "seed_{0}".format(seed) / "best_checkpoint.pt",
            map_location="cpu",
        )
        model_config = dict(checkpoint.get("model_config", {}))
        model = build_model(
            model_name,
            feature_dim=int(model_config.get("feature_dim", checkpoint.get("feature_dim", 512))),
            num_classes=int(model_config.get("num_classes", checkpoint.get("num_classes", 7))),
            hidden_dim=int(model_config.get("hidden_dim", 256)),
            dropout=float(model_config.get("dropout", 0.25)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(args.device).eval()
        with torch.inference_mode():
            output = model(tensor)
        probabilities.append(torch.softmax(output["logits"], dim=1)[0].cpu().numpy())
        if output.get("attention") is not None:
            attention.append(output["attention"][0].cpu().numpy())
        if output.get("instance_scores") is not None:
            instance_scores.append(output["instance_scores"][0].cpu().numpy())
    probability = np.mean(probabilities, axis=0)
    predicted = int(probability.argmax())
    scores = np.mean(attention, axis=0) if model_name == "abmil" else np.mean(instance_scores, axis=0)[:, predicted]
    result = {
        "schema_version": "architecture_baseline_real_yx_smoke_v1",
        "case_alias": alias,
        "ground_truth_used_before_prediction": False,
        "model": model_name,
        "fold": 0,
        "seeds": seeds,
        "prediction": predicted,
        "probabilities": probability.tolist(),
        "patch_count": len(patches),
        "patch_ranking": patch_ranking_summary(patches, scores.tolist(), top_k=int(args.top_k)),
        "mucosa": mucosa_summary,
        "manifest_sha256": sha256_file(manifest),
        "checkpoint_paths": [
            str((output_root / "runs" / model_name / "fold_0" / "seed_{0}".format(seed) / "best_checkpoint.pt").resolve())
            for seed in seeds
        ],
        "conch_checkpoint_sha256": sha256_file(args.checkpoint),
    }
    write_json(smoke_root / "inference_result.json", result)
    return result


def command_report(args) -> Mapping[str, Any]:
    output_root = Path(args.output_root).resolve()
    audit = read_json(output_root / "data_audit" / "audit_summary.json")
    baseline1 = read_json(output_root / "baseline1" / "annotation_status.json")
    evaluation = read_json(output_root / "evaluation" / "evaluation_summary.json")
    cache = read_json(output_root / "conch_embeddings" / "cache_summary.json")
    split_summary = read_json(output_root / "splits" / "split_summary.json")
    smoke_path = output_root / "real_smoke" / "inference_result.json"
    smoke = read_json(smoke_path) if smoke_path.is_file() else {"status": "not_run"}
    lines = [
        "# Frozen CONCH 5x Architecture Baseline — Executed Report",
        "",
        "This is a research baseline using weak slide supervision; relative scores are not clinical probabilities.",
        "",
        "## Data audit",
        "",
        "- Formal YX labels: {0}".format(
            audit.get("sources", {}).get("YX", {}).get("matched_clean_labels")
        ),
        "- Class vocabulary: {0}".format(", ".join(audit.get("class_vocabulary", []))),
        "- Baseline 1: `{0}`".format(baseline1.get("status")),
        "- Eligible YX cases after Mucosa: {0}".format(split_summary.get("eligible_alias_count")),
        "",
        "## Embedding pipeline",
        "",
        "- Encoder: {0}".format(cache.get("encoder_id", "conch_ViT-B-16")),
        "- Representation: pre-projection, unnormalized, 512 dimensions",
        "- CONCH checkpoint SHA-256: `{0}`".format(cache.get("checkpoint_sha256", "NA")),
        "- Cached cases: {0}".format(cache.get("requested_cases", "NA")),
        "- Diagnosis labels read by encoder: `{0}`".format(cache.get("labels_read_by_encoder", False)),
        "",
        "## OOF results",
        "",
        "| Model | Macro-F1 | Family-bootstrap 95% CI | Balanced accuracy | Macro AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name, metrics in sorted(evaluation.get("models", {}).items()):
        lines.append(
            "| {0} | {1:.4f} | {2} | {3:.4f} | {4} |".format(
                model_name,
                float(metrics["macro_f1"]),
                (
                    "[{0:.4f}, {1:.4f}]".format(
                        *metrics.get("macro_f1_family_bootstrap", {}).get("ci95", [float("nan"), float("nan")])
                    )
                    if metrics.get("macro_f1_family_bootstrap") else "NA"
                ),
                float(metrics["balanced_accuracy"]),
                "{0:.4f}".format(float(metrics["macro_auc"])) if metrics.get("macro_auc") is not None else "NA",
            )
        )
    lines.extend(["", "## Paired family-bootstrap comparisons", ""])
    for metric_label, comparison_key, correction_key in (
        ("Macro-F1", "paired_comparisons", "holm_correction"),
        ("Macro AUC", "paired_macro_auc_comparisons", "macro_auc_holm_correction"),
    ):
        comparisons = evaluation.get(comparison_key, {})
        corrections = evaluation.get(correction_key, {})
        for name, comparison in sorted(comparisons.items()):
            interval = comparison.get("ci95", [float("nan"), float("nan")])
            adjusted = corrections.get(name, {}).get("holm_adjusted_p_value")
            lines.append(
                "- {0} `{1}`: delta={2:.4f}, 95% CI [{3:.4f}, {4:.4f}], Holm p={5}.".format(
                    metric_label,
                    name,
                    float(comparison["point_delta"]),
                    float(interval[0]),
                    float(interval[1]),
                    "{0:.4g}".format(float(adjusted)) if adjusted is not None else "NA",
                )
            )
    lines.extend(["", "## Per-class and error analysis", ""])
    class_by_index = {int(index): label for label, index in evaluation.get("class_map", {}).items()}
    for model_name, metrics in sorted(evaluation.get("models", {}).items()):
        per_class = metrics.get("per_class", [])
        if per_class:
            weakest = min(per_class, key=lambda row: (float(row["f1"]), int(row["class_index"])))
            lines.append(
                "- {0}: weakest class `{1}` F1={2:.4f}; high-confidence errors={3}.".format(
                    model_name,
                    class_by_index.get(int(weakest["class_index"]), weakest["class_index"]),
                    float(weakest["f1"]),
                    len(metrics.get("high_confidence_errors", [])),
                )
            )
        grade_strata = metrics.get("post_hoc_grade_strata", {})
        if grade_strata:
            lines.append(
                "- {0}: post-hoc grade strata {1}.".format(
                    model_name,
                    ", ".join(
                        "{0} n={1}, macro-F1={2:.4f}".format(
                            grade,
                            int(values["n_cases"]),
                            float(values["macro_f1"]),
                        )
                        for grade, values in sorted(grade_strata.items())
                    ),
                )
            )
    lines.extend(
        [
            "",
            "Confusion matrices, OOF predictions and the complete per-class tables are stored under `evaluation/`.",
            "",
            "## Instance-level analysis",
            "",
            "ABMIL and DSMIL patch-ranking JSONL files include top/bottom patches, top 1/5/10% signal concentration and spatial component summaries.",
        ]
    )
    lines.extend(
        [
            "",
            "## Real YX smoke",
            "",
            "- Status: {0}".format("complete" if smoke.get("prediction") is not None else smoke.get("status", "not_run")),
            "- Model: {0}".format(smoke.get("model", "NA")),
            "- Case alias: {0}".format(smoke.get("case_alias", "NA")),
            "- Patch count: {0}".format(smoke.get("patch_count", "NA")),
            "",
            "## Interpretation boundary",
            "",
            "MIL attention and instance scores are candidate diagnostic relevance, not pathologist-confirmed morphology truth.",
            "",
            "## Remaining gaps",
            "",
            "- Expert patch/ROI morphology validation remains unavailable.",
            "- Baseline outputs are not connected to Agent EvidenceLedger or Reviewer runtime.",
            "- No semantic-guided, multiscale or CONCH fine-tuning experiment was performed.",
            "",
            "## Recommendation",
            "",
            "- Decision: **{0}**".format(args.recommendation),
            "- Rationale: {0}".format(args.recommendation_rationale.strip()),
        ]
    )
    report_path = Path(
        args.report_path or (output_root / "report" / "architecture_baseline_report.md")
    ).resolve()
    require_output_outside_sources(report_path, [DEFAULT_YX_ROOT, DEFAULT_HP_ROOT])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"status": "complete", "report": str(report_path), "report_sha256": sha256_file(report_path)}
    write_json(output_root / "report" / "report_manifest.json", result)
    return result


def _add_common(parser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen CONCH 5x architecture baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    _add_common(preflight)
    preflight.add_argument("--device", default="cuda:0")
    preflight.add_argument("--checkpoint", type=Path, default=DEFAULT_CONCH)
    preflight.add_argument("--batch-candidates", default="8,16,32")
    preflight.add_argument("--yx-root", type=Path, default=DEFAULT_YX_ROOT)
    preflight.add_argument("--hp-root", type=Path, default=DEFAULT_HP_ROOT)
    preflight.set_defaults(handler=command_preflight)

    audit = subparsers.add_parser("audit")
    _add_common(audit)
    audit.add_argument("--label-workbook", type=Path, default=DEFAULT_LABELS)
    audit.add_argument("--yx-root", type=Path, default=DEFAULT_YX_ROOT)
    audit.add_argument("--hp-root", type=Path, default=DEFAULT_HP_ROOT)
    audit.add_argument("--seed", type=int, default=17)
    audit.set_defaults(handler=command_audit)

    mucosa = subparsers.add_parser("run-mucosa")
    _add_common(mucosa)
    mucosa.add_argument("--case-paths", type=Path, required=True)
    mucosa.add_argument("--yx-root", type=Path, default=DEFAULT_YX_ROOT)
    mucosa.add_argument("--hp-root", type=Path, default=DEFAULT_HP_ROOT)
    mucosa.add_argument("--pathprism-url", default="http://127.0.0.1:8400/predict")
    mucosa.add_argument("--batch-size", type=int, default=16)
    mucosa.add_argument("--timeout-seconds", type=int, default=240)
    mucosa.add_argument("--base-magnification", type=float, default=0.0)
    mucosa.add_argument("--mpp", type=float, default=0.0)
    mucosa.add_argument("--min-tissue-coverage", type=float, default=0.05)
    mucosa.add_argument("--mask-downsample", type=float, default=32.0)
    mucosa.add_argument("--mucosa-threshold", type=float, default=0.30)
    mucosa.add_argument("--limit-cases", type=int, default=0)
    mucosa.add_argument("--shard-count", type=int, default=1)
    mucosa.add_argument("--shard-index", type=int, default=0)
    mucosa.add_argument("--resume", action="store_true")
    mucosa.add_argument("--skip-qc-panel", action="store_true")
    mucosa.set_defaults(handler=command_run_mucosa)

    validate = subparsers.add_parser("validate-manifest")
    _add_common(validate)
    validate.add_argument("--case-paths", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, default=None)
    validate.add_argument("--merge", action="store_true")
    validate.set_defaults(handler=command_validate_manifest)

    embed = subparsers.add_parser("embed-conch")
    _add_common(embed)
    embed.add_argument("--manifest", type=Path, required=True)
    embed.add_argument("--case-paths", type=Path, required=True)
    embed.add_argument("--checkpoint", type=Path, default=DEFAULT_CONCH)
    embed.add_argument("--device", default="cuda:0")
    embed.add_argument("--batch-size", type=int, default=0, help="0 reads resolved_runtime.json")
    embed.add_argument("--minimum-coverage", type=float, default=0.30)
    embed.add_argument("--limit-cases", type=int, default=0)
    embed.add_argument("--resume", action="store_true")
    embed.add_argument("--no-amp", action="store_true")
    embed.set_defaults(handler=command_embed_conch)

    splits = subparsers.add_parser("build-splits")
    _add_common(splits)
    splits.add_argument("--manifest", type=Path, required=True)
    splits.add_argument("--labels", type=Path, required=True)
    splits.add_argument("--minimum-coverage", type=float, default=0.60)
    splits.add_argument("--seed", type=int, default=17)
    splits.set_defaults(handler=command_build_splits)

    baseline1 = subparsers.add_parser("baseline1-status")
    _add_common(baseline1)
    baseline1.add_argument("--manifest", type=Path, required=True)
    baseline1.add_argument("--annotations", type=Path, default=None)
    baseline1.set_defaults(handler=command_baseline1_status)

    train = subparsers.add_parser("train-mil")
    _add_common(train)
    train.add_argument("--labels", type=Path, required=True)
    train.add_argument("--models", default="mean_pool,abmil,dsmil")
    train.add_argument("--folds", default="all")
    train.add_argument("--seeds", default="17,29,43")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--minimum-coverage", type=float, default=0.60)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--dropout", type=float, default=0.25)
    train.add_argument("--max-epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--resume", action="store_true")
    train.set_defaults(handler=command_train_mil)

    evaluate = subparsers.add_parser("evaluate")
    _add_common(evaluate)
    evaluate.add_argument("--models", default="mean_pool,abmil,dsmil")
    evaluate.add_argument("--seeds", default="17,29,43")
    evaluate.add_argument("--minimum-coverage", type=float, default=0.60)
    evaluate.add_argument("--bootstrap", type=int, default=2000)
    evaluate.add_argument("--seed", type=int, default=17)
    evaluate.add_argument("--top-k", type=int, default=10)
    evaluate.set_defaults(handler=command_evaluate)

    smoke = subparsers.add_parser("real-smoke")
    _add_common(smoke)
    smoke.add_argument("--case-paths", type=Path, required=True)
    smoke.add_argument("--checkpoint", type=Path, default=DEFAULT_CONCH)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--pathprism-url", default="http://127.0.0.1:8400/predict")
    smoke.add_argument("--mucosa-batch-size", type=int, default=16)
    smoke.add_argument("--conch-batch-size", type=int, default=16)
    smoke.add_argument("--seeds", default="17,29,43")
    smoke.add_argument("--top-k", type=int, default=10)
    smoke.add_argument("--resume", action="store_true")
    smoke.set_defaults(handler=command_real_smoke)

    report = subparsers.add_parser("report")
    _add_common(report)
    report.add_argument("--report-path", type=Path, default=None)
    report.add_argument("--recommendation", choices=["A", "B", "C", "D"], required=True)
    report.add_argument("--recommendation-rationale", required=True)
    report.set_defaults(handler=command_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.time()
    try:
        if hasattr(args, "output_root"):
            require_output_outside_sources(
                Path(args.output_root).resolve(),
                [DEFAULT_YX_ROOT, DEFAULT_HP_ROOT],
            )
        result = args.handler(args)
    except Exception as exc:
        payload = {
            "event": "architecture_baseline_failed",
            "command": args.command,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    payload = {
        "event": "architecture_baseline_complete",
        "command": args.command,
        "elapsed_seconds": round(time.time() - started, 3),
        "result": result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
