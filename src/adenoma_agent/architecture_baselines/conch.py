"""Frozen CONCH embedding extraction with a label-free cache boundary."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from adenoma_agent.wsi import WSIReader

from .io import (
    canonical_json_bytes,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_payload,
    write_json,
    write_jsonl,
)
from .manifest import validate_patch_row


CONCH_MODEL_ID = "conch_ViT-B-16"
CONCH_EMBEDDING_DIM = 512
CONCH_PREPROCESS_VERSION = "official_conch_v1_resize_center_crop_448"


def cuda_library_paths(python_prefix: Optional[Path] = None) -> List[str]:
    prefix = Path(python_prefix or os.sys.prefix)
    version = "python{0}.{1}".format(os.sys.version_info.major, os.sys.version_info.minor)
    candidates = [
        prefix / "lib" / version / "site-packages" / "nvidia" / "nvjitlink" / "lib",
        prefix / "lib" / version / "site-packages" / "nvidia" / "cusparse" / "lib",
        prefix / "lib",
    ]
    return [str(path) for path in candidates if path.exists()]


def ensure_cuda_linker_environment(marker: str = "_ADENOMA_BASELINE_LD_READY") -> None:
    """Re-exec before importing torch so packaged CUDA libraries resolve consistently."""

    if os.environ.get(marker) == "1":
        return
    preferred = cuda_library_paths()
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    for part in current:
        if part not in preferred:
            preferred.append(part)
    if not preferred:
        os.environ[marker] = "1"
        return
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(preferred)
    environment[marker] = "1"
    os.execvpe(os.sys.executable, [os.sys.executable, *os.sys.argv], environment)


def _require_cuda_device(device: str) -> None:
    import torch

    if not str(device).startswith("cuda"):
        raise ValueError("Formal CONCH embedding requires an explicit CUDA device")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is not available; CPU fallback is forbidden")
    try:
        torch.empty(1, device=torch.device(device))
    except Exception as exc:
        raise RuntimeError("CUDA allocation failed on {0}: {1}".format(device, exc))


class FrozenConchEncoder:
    def __init__(self, checkpoint_path: Path, device: str = "cuda:0", use_amp: bool = True):
        import torch
        from conch.open_clip_custom import create_model_from_pretrained

        _require_cuda_device(device)
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError("CONCH checkpoint is missing: {0}".format(checkpoint_path))
        model, preprocess = create_model_from_pretrained(
            CONCH_MODEL_ID,
            checkpoint_path=str(checkpoint_path),
            device=str(device),
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.torch = torch
        self.model = model
        self.preprocess = preprocess
        self.device = torch.device(device)
        self.use_amp = bool(use_amp)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "encoder_id": CONCH_MODEL_ID,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocessing_version": CONCH_PREPROCESS_VERSION,
            "source_crop_pixel_dimensions": [256, 256],
            "model_input_pixel_dimensions": [448, 448],
            "embedding_dim": CONCH_EMBEDDING_DIM,
            "embedding_semantics": "pre_projection_unnormalized",
            "proj_contrast": False,
            "normalize": False,
            "inference_amp": self.use_amp,
            "device": str(self.device),
        }

    def encode_tensors(self, tensors) -> np.ndarray:
        torch = self.torch
        if not tensors:
            return np.empty((0, CONCH_EMBEDDING_DIM), dtype=np.float32)
        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=False)
        with torch.inference_mode():
            amp_enabled = bool(self.use_amp and self.device.type == "cuda")
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp_enabled):
                features = self.model.encode_image(batch, proj_contrast=False, normalize=False)
        features = features.detach().to(dtype=torch.float32).cpu().numpy()
        if features.ndim != 2 or features.shape[1] != CONCH_EMBEDDING_DIM:
            raise RuntimeError("Unexpected CONCH embedding shape: {0}".format(features.shape))
        if not np.isfinite(features).all():
            raise RuntimeError("CONCH returned NaN or Inf embeddings")
        return features

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        return self.encode_tensors([self.preprocess(image.convert("RGB")) for image in images])


def _atomic_numpy(path: Path, value: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".{0}.".format(path.name), suffix=".npy", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _resolver(case_paths_jsonl: Path) -> Mapping[str, Mapping[str, Any]]:
    output = {}
    for row in iter_jsonl(case_paths_jsonl):
        alias = str(row.get("case_alias", "")).strip()
        source = str(row.get("source_path", "")).strip()
        if not alias or not source:
            raise ValueError("case_paths rows require case_alias and source_path")
        if alias in output:
            raise ValueError("Duplicate case_alias in case_paths: {0}".format(alias))
        output[alias] = row
    return output


def _case_manifest_rows(manifest_path: Path, minimum_coverage: float) -> Mapping[str, List[Mapping[str, Any]]]:
    grouped = defaultdict(list)
    seen = set()
    for row in iter_jsonl(manifest_path):
        validate_patch_row(row, require_physical=True)
        if float(row.get("mucosa_coverage", 0.0)) < float(minimum_coverage):
            continue
        key = (str(row["case_alias"]), str(row["patch_id"]))
        if key in seen:
            raise ValueError("Duplicate manifest patch: {0}".format(key))
        seen.add(key)
        grouped[key[0]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: (item["grid_index"][0], item["grid_index"][1], item["patch_id"]))
    return grouped


def _cache_is_valid(case_dir: Path, expected_fingerprint: str) -> bool:
    try:
        metadata = read_json(case_dir / "metadata.json")
        if metadata.get("cache_fingerprint") != expected_fingerprint:
            return False
        validate_case_cache(case_dir, expected_dim=int(metadata.get("embedding_dim", 0)))
        return True
    except Exception:
        return False


def encode_manifest_cache(
    manifest_path: Path,
    case_paths_jsonl: Path,
    cache_root: Path,
    checkpoint_path: Path,
    device: str = "cuda:0",
    batch_size: int = 16,
    minimum_coverage: float = 0.30,
    resume: bool = False,
    use_amp: bool = True,
    limit_cases: int = 0,
) -> Mapping[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    resolver = _resolver(case_paths_jsonl)
    grouped = _case_manifest_rows(manifest_path, minimum_coverage=float(minimum_coverage))
    aliases = sorted(grouped)
    if int(limit_cases) > 0:
        aliases = aliases[: int(limit_cases)]
    missing = [alias for alias in aliases if alias not in resolver]
    if missing:
        raise ValueError("Manifest cases missing from local resolver: {0}".format(missing[:10]))
    encoder = FrozenConchEncoder(checkpoint_path=checkpoint_path, device=device, use_amp=use_amp)
    run_basis = {
        **dict(encoder.metadata),
        "source_manifest_sha256": sha256_file(manifest_path),
        "minimum_mucosa_coverage": float(minimum_coverage),
        "storage_dtype": "float16",
    }
    completed = []
    skipped = []
    started = time.time()
    for alias in aliases:
        case_rows = grouped[alias]
        case_dir = cache_root / alias
        case_fingerprint = sha256_payload(
            {
                **run_basis,
                "case_alias": alias,
                "patch_rows": [row.get("source_row_sha256") or sha256_payload(row) for row in case_rows],
            }
        )
        if resume and _cache_is_valid(case_dir, case_fingerprint):
            skipped.append(alias)
            continue
        source_path = Path(str(resolver[alias]["source_path"])).resolve()
        features = []
        index_rows = []
        with WSIReader(source_path) as reader:
            pending_tensors = []
            pending_rows = []
            pending_crops = []

            def flush() -> None:
                if not pending_tensors:
                    return
                encoded = encoder.encode_tensors(pending_tensors)
                base_index = len(features)
                for local_index, (row, crop) in enumerate(zip(pending_rows, pending_crops)):
                    crop_provenance = dict(crop.provenance)
                    crop_provenance.pop("local_source_path", None)
                    features.append(encoded[local_index])
                    index_rows.append(
                        {
                            "schema_version": "conch_patch_embedding_index_v1",
                            "case_alias": alias,
                            "patch_id": str(row["patch_id"]),
                            "feature_row": base_index + local_index,
                            "grid_index": [int(value) for value in row["grid_index"]],
                            "level0_bbox": [int(value) for value in row["level0_bbox"]],
                            "mucosa_coverage": float(row["mucosa_coverage"]),
                            "target_magnification": "5x",
                            "manifest_row_sha256": row.get("source_row_sha256") or sha256_payload(row),
                            "image_sha256": crop.image_sha256,
                            "crop_provenance": crop_provenance,
                        }
                    )
                pending_tensors.clear()
                pending_rows.clear()
                pending_crops.clear()

            for row in case_rows:
                crop = reader.crop_level0_bbox(
                    row["level0_bbox"],
                    requested_magnification=5.0,
                    output_pixels=(256, 256),
                )
                pending_tensors.append(encoder.preprocess(crop.image.convert("RGB")))
                pending_rows.append(row)
                pending_crops.append(crop)
                if len(pending_tensors) >= int(batch_size):
                    flush()
            flush()
        matrix = np.asarray(features, dtype=np.float16)
        if matrix.shape != (len(case_rows), CONCH_EMBEDDING_DIM):
            raise RuntimeError("Cache shape mismatch for {0}: {1}".format(alias, matrix.shape))
        case_dir.mkdir(parents=True, exist_ok=True)
        _atomic_numpy(case_dir / "features.npy", matrix)
        write_jsonl(case_dir / "index.jsonl", index_rows)
        metadata = {
            "schema_version": "conch_embedding_cache_v1",
            **run_basis,
            "cache_fingerprint": case_fingerprint,
            "case_alias": alias,
            "item_count": len(index_rows),
            "embedding_dim": CONCH_EMBEDDING_DIM,
            "features_sha256": sha256_file(case_dir / "features.npy"),
            "index_sha256": sha256_file(case_dir / "index.jsonl"),
            "labels_in_cache": False,
        }
        write_json(case_dir / "metadata.json", metadata)
        validate_case_cache(case_dir, expected_dim=CONCH_EMBEDDING_DIM)
        completed.append(alias)
    summary = {
        "schema_version": "conch_embedding_cache_run_v1",
        **run_basis,
        "requested_cases": len(aliases),
        "requested_case_aliases": aliases,
        "completed_cases": completed,
        "skipped_cases": skipped,
        "elapsed_seconds": round(time.time() - started, 3),
        "labels_read_by_encoder": False,
    }
    write_json(cache_root / "cache_summary.json", summary)
    return summary


def validate_case_cache(case_dir: Path, expected_dim: int = CONCH_EMBEDDING_DIM) -> Mapping[str, Any]:
    case_dir = Path(case_dir)
    features_path = case_dir / "features.npy"
    index_path = case_dir / "index.jsonl"
    metadata_path = case_dir / "metadata.json"
    if not features_path.is_file() or not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Incomplete embedding cache: {0}".format(case_dir))
    features = np.load(features_path, allow_pickle=False, mmap_mode="r")
    rows = list(iter_jsonl(index_path))
    metadata = read_json(metadata_path)
    if features.ndim != 2 or features.shape[1] != int(expected_dim):
        raise ValueError("Wrong embedding shape: {0}".format(features.shape))
    if len(rows) != features.shape[0] or int(metadata.get("item_count", -1)) != features.shape[0]:
        raise ValueError("Embedding/index/metadata item counts disagree")
    if not np.isfinite(features).all():
        raise ValueError("Embedding cache contains NaN or Inf")
    seen = set()
    expected_alias = str(metadata.get("case_alias", ""))
    for expected_row, row in enumerate(rows):
        if str(row.get("case_alias", "")) != expected_alias:
            raise ValueError("Cross-slide patch contamination in cache")
        patch_id = str(row.get("patch_id", ""))
        if not patch_id or patch_id in seen:
            raise ValueError("Duplicate or empty patch_id in cache")
        seen.add(patch_id)
        if int(row.get("feature_row", -1)) != expected_row:
            raise ValueError("feature_row is not contiguous")
        if not str(row.get("image_sha256", "")):
            raise ValueError("image_sha256 is required")
    if metadata.get("labels_in_cache") is not False:
        raise ValueError("Embedding metadata must explicitly exclude labels")
    if metadata.get("features_sha256") != sha256_file(features_path):
        raise ValueError("features.npy hash mismatch")
    if metadata.get("index_sha256") != sha256_file(index_path):
        raise ValueError("index.jsonl hash mismatch")
    return {
        "case_alias": expected_alias,
        "items": features.shape[0],
        "embedding_dim": features.shape[1],
        "dtype": str(features.dtype),
    }


def validate_cache_root(
    cache_root: Path,
    expected_case_aliases: Optional[Iterable[str]] = None,
) -> Mapping[str, Any]:
    expected = None if expected_case_aliases is None else {str(value) for value in expected_case_aliases}
    rows = []
    for metadata_path in sorted(Path(cache_root).glob("*/metadata.json")):
        if expected is None or metadata_path.parent.name in expected:
            rows.append(validate_case_cache(metadata_path.parent))
    if not rows:
        raise ValueError("No per-slide caches found in {0}".format(cache_root))
    actual = {str(row["case_alias"]) for row in rows}
    if expected is not None and actual != expected:
        raise ValueError(
            "Embedding cache cohort mismatch: missing={0}, unexpected={1}".format(
                sorted(expected - actual)[:10],
                sorted(actual - expected)[:10],
            )
        )
    all_cache_aliases = {path.parent.name for path in Path(cache_root).glob("*/metadata.json")}
    stale = sorted(all_cache_aliases - actual)
    if stale:
        raise ValueError(
            "Embedding cache contains stale case directories outside the active manifest: {0}".format(
                stale[:10]
            )
        )
    return {
        "schema_version": "conch_embedding_cache_validation_v1",
        "cases": len(rows),
        "patches": sum(int(row["items"]) for row in rows),
        "embedding_dim": CONCH_EMBEDDING_DIM,
        "case_summaries": rows,
    }


def conch_gpu_preflight(
    checkpoint_path: Path,
    device: str = "cuda:0",
    candidate_batch_sizes: Iterable[int] = (8, 16, 32),
) -> Mapping[str, Any]:
    import torch
    from PIL import Image

    _require_cuda_device(device)
    encoder = FrozenConchEncoder(checkpoint_path=checkpoint_path, device=device, use_amp=True)
    passed = []
    failed = []
    dummy = Image.new("RGB", (256, 256), (255, 255, 255))
    tensor = encoder.preprocess(dummy)
    total_memory = int(torch.cuda.get_device_properties(torch.device(device)).total_memory)
    for batch_size in sorted(set(int(value) for value in candidate_batch_sizes if int(value) > 0)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        try:
            output = encoder.encode_tensors([tensor] * batch_size)
            peak = int(torch.cuda.max_memory_allocated(torch.device(device)))
            if output.shape != (batch_size, CONCH_EMBEDDING_DIM):
                raise RuntimeError("unexpected dummy output shape")
            passed.append({"batch_size": batch_size, "peak_memory_bytes": peak})
        except torch.cuda.OutOfMemoryError as exc:
            failed.append({"batch_size": batch_size, "error": str(exc)})
            torch.cuda.empty_cache()
    safe = [row for row in passed if int(row["peak_memory_bytes"]) <= int(total_memory * 0.80)]
    selected = max((row["batch_size"] for row in safe), default=0)
    if selected <= 0:
        raise RuntimeError("No candidate CONCH batch size passed the 80% memory safety gate")
    return {
        "schema_version": "architecture_baseline_gpu_preflight_v1",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        "gpu_total_memory_bytes": total_memory,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "checkpoint_sha256": encoder.checkpoint_sha256,
        "embedding_dim": CONCH_EMBEDDING_DIM,
        "candidate_results": passed + failed,
        "selected_conch_batch_size": selected,
        "cpu_fallback_allowed": False,
    }
