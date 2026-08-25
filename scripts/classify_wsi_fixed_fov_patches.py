#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adenoma_agent.multimodal import (  # noqa: E402
    DIGEPATH_ROI9_CLASS_NAMES,
    _normalize_conch_label,
    _trace_label_from_digepath_class,
)


CONCH_CRC100K_LABELS = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def clean_rgb(image):
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        return background.convert("RGB")
    return image.convert("RGB")


class CtypesOpenSlideReader(object):
    _lib = None

    @classmethod
    def _load_lib(cls):
        if cls._lib is not None:
            return cls._lib
        last_error = None
        for name in ("libopenslide.so.0", "libopenslide.so"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError as exc:
                last_error = exc
        else:
            raise RuntimeError("libopenslide is unavailable: {0}".format(last_error))

        lib.openslide_open.argtypes = [ctypes.c_char_p]
        lib.openslide_open.restype = ctypes.c_void_p
        lib.openslide_close.argtypes = [ctypes.c_void_p]
        lib.openslide_close.restype = None
        lib.openslide_get_error.argtypes = [ctypes.c_void_p]
        lib.openslide_get_error.restype = ctypes.c_char_p
        lib.openslide_get_level0_dimensions.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.openslide_get_level0_dimensions.restype = None
        lib.openslide_get_level_count.argtypes = [ctypes.c_void_p]
        lib.openslide_get_level_count.restype = ctypes.c_int32
        lib.openslide_get_level_downsample.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.openslide_get_level_downsample.restype = ctypes.c_double
        lib.openslide_get_level_dimensions.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.openslide_get_level_dimensions.restype = None
        lib.openslide_read_region.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        lib.openslide_read_region.restype = None
        cls._lib = lib
        return lib

    def __init__(self, slide_path):
        self.slide_path = str(slide_path)
        self.lib = self._load_lib()
        self.handle = self.lib.openslide_open(self.slide_path.encode("utf-8"))
        if not self.handle:
            raise RuntimeError("OpenSlide could not open {0}".format(self.slide_path))
        width = ctypes.c_int64()
        height = ctypes.c_int64()
        self.lib.openslide_get_level0_dimensions(self.handle, ctypes.byref(width), ctypes.byref(height))
        self.dimensions = (int(width.value), int(height.value))
        self.level_count = int(self.lib.openslide_get_level_count(self.handle))
        self.level_downsamples = []
        self.level_dimensions = []
        for level in range(max(1, self.level_count)):
            level_width = ctypes.c_int64()
            level_height = ctypes.c_int64()
            self.lib.openslide_get_level_dimensions(
                self.handle,
                int(level),
                ctypes.byref(level_width),
                ctypes.byref(level_height),
            )
            self.level_dimensions.append((int(level_width.value), int(level_height.value)))
            self.level_downsamples.append(float(self.lib.openslide_get_level_downsample(self.handle, int(level))))
        self._raise_if_error()

    def _raise_if_error(self):
        error = self.lib.openslide_get_error(self.handle)
        if error:
            raise RuntimeError(error.decode("utf-8", errors="replace"))

    def read_region(self, top_left, level, size):
        x, y = [int(v) for v in top_left]
        width, height = [int(v) for v in size]
        buffer = (ctypes.c_uint32 * (width * height))()
        self.lib.openslide_read_region(self.handle, buffer, x, y, int(level), width, height)
        self._raise_if_error()
        data = ctypes.string_at(buffer, width * height * 4)
        return Image.frombuffer("RGBA", (width, height), data, "raw", "BGRA", 0, 1)

    def close(self):
        if self.handle:
            self.lib.openslide_close(self.handle)
            self.handle = None


class WSIReader(object):
    def __init__(self, slide_path, allow_pil_fallback=False):
        self.slide_path = str(slide_path)
        self.reader = None
        self.mode = ""
        try:
            import openslide

            self.reader = openslide.open_slide(self.slide_path)
            self.mode = "openslide_python"
            self.dimensions = tuple(int(v) for v in self.reader.dimensions)
            self.level_dimensions = [tuple(int(v) for v in dims) for dims in self.reader.level_dimensions]
            self.level_downsamples = [float(v) for v in self.reader.level_downsamples]
            return
        except Exception:
            self.reader = None
        try:
            self.reader = CtypesOpenSlideReader(self.slide_path)
            self.mode = "openslide_ctypes"
            self.dimensions = tuple(int(v) for v in self.reader.dimensions)
            self.level_dimensions = list(self.reader.level_dimensions)
            self.level_downsamples = list(self.reader.level_downsamples)
            return
        except Exception as exc:
            if not allow_pil_fallback:
                raise RuntimeError(
                    "OpenSlide Python and ctypes readers failed for {0}; refusing PIL fallback for WSI-scale tiling: {1}".format(
                        self.slide_path, exc
                    )
                )
        self.reader = Image.open(self.slide_path)
        self.mode = "pil"
        self.dimensions = tuple(int(v) for v in self.reader.size)
        self.level_dimensions = [self.dimensions]
        self.level_downsamples = [1.0]

    def read_region(self, top_left, level, size):
        if self.mode == "pil":
            x, y = [int(v) for v in top_left]
            width, height = [int(v) for v in size]
            crop = Image.new("RGB", (width, height), (255, 255, 255))
            source_box = (max(0, x), max(0, y), min(self.reader.width, x + width), min(self.reader.height, y + height))
            if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
                crop.paste(self.reader.crop(source_box).convert("RGB"), (max(0, -x), max(0, -y)))
            return crop
        return self.reader.read_region(top_left, int(level), size)

    def close(self):
        if self.reader is not None and hasattr(self.reader, "close"):
            self.reader.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Crop fixed-size level-0 WSI fields of view and classify each crop "
            "with local CONCH and DIgePath HTTP servers."
        )
    )
    parser.add_argument("--grid-dir", default=str(REPO_ROOT / "data" / "10_sample" / "5x_sample_10"))
    parser.add_argument("--wsi-dir", default=str(REPO_ROOT / "data" / "10_sample" / "WSI"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--crop-sizes", default="512,256", help="Comma-separated level-0 crop sizes in pixels.")
    parser.add_argument(
        "--tiling-mode",
        choices=["full_slide", "selected_cells"],
        default="full_slide",
        help="full_slide covers the whole level-0 WSI; selected_cells crops around tissue-grid anchors.",
    )
    parser.add_argument("--stride", type=int, default=0, help="Full-slide tiling stride in pixels. Defaults to crop size.")
    parser.add_argument("--conch-url", default="http://127.0.0.1:8200/predict")
    parser.add_argument("--digepath-url", default="http://127.0.0.1:8300/predict")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-every-batches", type=int, default=20)
    parser.add_argument("--limit-slides", type=int, default=0)
    parser.add_argument("--limit-patches-per-slide", type=int, default=0)
    parser.add_argument("--dry-run-counts", action="store_true")
    parser.add_argument("--allow-pil-fallback", action="store_true")
    parser.add_argument("--skip-conch", action="store_true")
    parser.add_argument("--skip-digepath", action="store_true")
    return parser.parse_args()


def load_runtime_defaults(args):
    runtime_path = REPO_ROOT / "configs" / "runtime.yaml"
    if not runtime_path.exists():
        return
    try:
        import yaml
    except Exception:
        return
    try:
        payload = yaml.safe_load(runtime_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return
    backends = payload.get("backends", {})
    conch = backends.get("local_conch", {})
    digepath = backends.get("local_digepath", {})
    if conch.get("server_url"):
        args.conch_url = str(conch["server_url"])
    if digepath.get("server_url"):
        args.digepath_url = str(digepath["server_url"])
    if conch.get("timeout_seconds"):
        args.timeout_seconds = int(conch["timeout_seconds"])


def parse_crop_sizes(value):
    sizes = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        size = int(part)
        if size <= 0:
            raise ValueError("crop sizes must be positive integers")
        sizes.append(size)
    if not sizes:
        raise ValueError("at least one crop size is required")
    return sizes


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path, rows):
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_wsi_path(slide_id, grid_payload, wsi_dir):
    wsi_dir = Path(wsi_dir)
    for suffix in (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".jpg", ".png"):
        candidate = wsi_dir / "{0}{1}".format(slide_id, suffix)
        if candidate.exists():
            return candidate
    source_path = Path(str(grid_payload.get("slide_path", "")))
    if source_path.exists():
        return source_path
    matches = sorted(wsi_dir.glob("{0}.*".format(slide_id)))
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not find WSI for slide_id={0} in {1}".format(slide_id, wsi_dir))


def iter_grid_payloads(grid_dir, limit_slides=0):
    paths = sorted(Path(grid_dir).glob("*_grid.json"))
    if limit_slides:
        paths = paths[: int(limit_slides)]
    for path in paths:
        payload = read_json(path)
        yield path, payload


def selected_cells(grid_payload, limit=0):
    cells = [cell for cell in grid_payload.get("grid_cells", []) if cell.get("is_selected")]
    if limit:
        cells = cells[: int(limit)]
    return cells


def cell_center(cell):
    if cell.get("level0_anchor_x") is not None and cell.get("level0_anchor_y") is not None:
        return int(round(float(cell.get("level0_anchor_x")))), int(round(float(cell.get("level0_anchor_y"))))
    x = float(cell.get("level0_top_left_x", 0) or 0) + float(cell.get("level0_width", 0) or 0) / 2.0
    y = float(cell.get("level0_top_left_y", 0) or 0) + float(cell.get("level0_height", 0) or 0) / 2.0
    return int(round(x)), int(round(y))


def read_padded_bbox(reader, requested_bbox, size, slide_dimensions):
    slide_w, slide_h = [int(v) for v in slide_dimensions]
    clipped_bbox = [
        max(0, requested_bbox[0]),
        max(0, requested_bbox[1]),
        min(slide_w, requested_bbox[2]),
        min(slide_h, requested_bbox[3]),
    ]
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    if clipped_bbox[2] <= clipped_bbox[0] or clipped_bbox[3] <= clipped_bbox[1]:
        return canvas, requested_bbox, clipped_bbox
    region = clean_rgb(
        reader.read_region(
            (clipped_bbox[0], clipped_bbox[1]),
            0,
            (clipped_bbox[2] - clipped_bbox[0], clipped_bbox[3] - clipped_bbox[1]),
        )
    )
    paste_at = (clipped_bbox[0] - requested_bbox[0], clipped_bbox[1] - requested_bbox[1])
    canvas.paste(region, paste_at)
    return canvas, requested_bbox, clipped_bbox


def read_padded_square(reader, center_x, center_y, size, slide_dimensions):
    half = int(size // 2)
    requested_bbox = [int(center_x - half), int(center_y - half), int(center_x - half + size), int(center_y - half + size)]
    return read_padded_bbox(reader, requested_bbox, size, slide_dimensions)


def tile_count_for_size(slide_dimensions, crop_size, stride=0):
    width, height = [int(v) for v in slide_dimensions]
    step = int(stride or crop_size)
    return int(math.ceil(width / float(step))) * int(math.ceil(height / float(step)))


def estimate_crop_counts(grid_payload, crop_sizes, tiling_mode, stride=0, limit_patches_per_slide=0):
    if tiling_mode == "selected_cells":
        count_per_size = len(selected_cells(grid_payload, limit=limit_patches_per_slide))
        return {str(size): count_per_size for size in crop_sizes}
    counts = {}
    for size in crop_sizes:
        count = tile_count_for_size(grid_payload["slide_dimensions_level0"], int(size), stride=stride or int(size))
        if limit_patches_per_slide:
            count = min(count, int(limit_patches_per_slide))
        counts[str(size)] = count
    return counts


def iter_full_slide_tiles(slide_dimensions, crop_size, stride=0, limit=0):
    width, height = [int(v) for v in slide_dimensions]
    step = int(stride or crop_size)
    emitted = 0
    tile_row = 0
    for y in range(0, height, step):
        tile_col = 0
        for x in range(0, width, step):
            yield tile_row, tile_col, x, y
            emitted += 1
            if limit and emitted >= int(limit):
                return
            tile_col += 1
        tile_row += 1


def crop_slide(
    grid_path,
    grid_payload,
    wsi_path,
    crop_sizes,
    output_dir,
    tiling_mode="full_slide",
    stride=0,
    limit_patches_per_slide=0,
    allow_pil_fallback=False,
):
    slide_id = str(grid_payload.get("slide_id") or grid_path.name.split("_tissuegrid", 1)[0])
    slide_dimensions = grid_payload.get("slide_dimensions_level0")
    if not slide_dimensions:
        raise ValueError("grid payload lacks slide_dimensions_level0: {0}".format(grid_path))
    reader = WSIReader(wsi_path, allow_pil_fallback=allow_pil_fallback)
    reader_dimensions = tuple(int(v) for v in getattr(reader, "dimensions", slide_dimensions))
    if reader_dimensions != tuple(int(v) for v in slide_dimensions):
        slide_dimensions = reader_dimensions
    rows = []
    try:
        if tiling_mode == "full_slide":
            for crop_size in crop_sizes:
                crop_size = int(crop_size)
                step = int(stride or crop_size)
                crop_dir = Path(output_dir) / "crops" / "fov{0}".format(crop_size) / slide_id
                crop_dir.mkdir(parents=True, exist_ok=True)
                for tile_row, tile_col, x, y in iter_full_slide_tiles(
                    slide_dimensions,
                    crop_size=crop_size,
                    stride=step,
                    limit=limit_patches_per_slide,
                ):
                    requested_bbox = [int(x), int(y), int(x + crop_size), int(y + crop_size)]
                    crop, requested_bbox, clipped_bbox = read_padded_bbox(
                        reader,
                        requested_bbox=requested_bbox,
                        size=crop_size,
                        slide_dimensions=slide_dimensions,
                    )
                    patch_id = [int(tile_row), int(tile_col)]
                    crop_path = crop_dir / "tile_r{0:04d}_c{1:04d}_{2}.png".format(tile_row, tile_col, crop_size)
                    crop.save(crop_path)
                    rows.append(
                        {
                            "slide_id": slide_id,
                            "grid_json": str(grid_path),
                            "wsi_path": str(wsi_path),
                            "patch_id": patch_id,
                            "row_id": int(tile_row),
                            "col_id": int(tile_col),
                            "crop_size": int(crop_size),
                            "stride": int(step),
                            "tiling_mode": "full_slide",
                            "simulated_view": "20x_fov_512px" if crop_size == 512 else (
                                "40x_fov_256px" if crop_size == 256 else "fixed_level0_fov_{0}px".format(crop_size)
                            ),
                            "image_path": str(crop_path),
                            "center_level0": [int(x + crop_size / 2), int(y + crop_size / 2)],
                            "requested_level0_bbox": requested_bbox,
                            "clipped_level0_bbox": clipped_bbox,
                            "classifier_source_image": str(wsi_path),
                            "classifier_source_mode": "wsi_level0_full_slide_tiling",
                            "classifier_crop_level0_bbox": requested_bbox,
                            "classifier_crop_level0_size": [int(crop_size), int(crop_size)],
                            "classifier_crop_output_size": [int(crop_size), int(crop_size)],
                            "classifier_crop_view": "level0_{0}x{0}_full_slide_stride_{1}".format(crop_size, step),
                        }
                    )
        else:
            for cell in selected_cells(grid_payload, limit=limit_patches_per_slide):
                patch_id = list(cell.get("patch_id", [cell.get("row_id", 0), cell.get("col_id", 0)]))
                center_x, center_y = cell_center(cell)
                for crop_size in crop_sizes:
                    crop, requested_bbox, clipped_bbox = read_padded_square(
                        reader,
                        center_x=center_x,
                        center_y=center_y,
                        size=int(crop_size),
                        slide_dimensions=slide_dimensions,
                    )
                    crop_dir = Path(output_dir) / "crops" / "fov{0}".format(crop_size) / slide_id
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = crop_dir / "patch_{0}_{1}_{2}.png".format(int(patch_id[0]), int(patch_id[1]), crop_size)
                    crop.save(crop_path)
                    rows.append(
                        {
                            "slide_id": slide_id,
                            "grid_json": str(grid_path),
                            "wsi_path": str(wsi_path),
                            "patch_id": patch_id,
                            "row_id": int(patch_id[0]),
                            "col_id": int(patch_id[1]),
                            "crop_size": int(crop_size),
                            "stride": 0,
                            "tiling_mode": "selected_cells",
                            "simulated_view": "20x_fov_512px" if int(crop_size) == 512 else (
                                "40x_fov_256px" if int(crop_size) == 256 else "fixed_level0_fov_{0}px".format(crop_size)
                            ),
                            "image_path": str(crop_path),
                            "center_level0": [int(center_x), int(center_y)],
                            "requested_level0_bbox": requested_bbox,
                            "clipped_level0_bbox": clipped_bbox,
                            "selection_reason": cell.get("selection_reason", ""),
                            "anchor_source": cell.get("anchor_source", ""),
                            "tissue_coverage_ratio": float(cell.get("tissue_coverage_ratio", 0.0) or 0.0),
                            "classifier_source_image": str(wsi_path),
                            "classifier_source_mode": "wsi_level0_fixed_fov",
                            "classifier_crop_level0_bbox": requested_bbox,
                            "classifier_crop_level0_size": [int(crop_size), int(crop_size)],
                            "classifier_crop_output_size": [int(crop_size), int(crop_size)],
                            "classifier_crop_view": "level0_{0}x{0}_centered_on_selected_grid_anchor".format(crop_size),
                        }
                    )
    finally:
        reader.close()
    return rows


def batched(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def post_predictions(server_url, rows, model_name, timeout_seconds, batch_size, progress_every_batches=20):
    import requests

    predictions = []
    errors = []
    started_all = time.time()
    for batch_index, batch in enumerate(batched(rows, max(1, int(batch_size))), start=1):
        payload = {
            "image_paths": [row["image_path"] for row in batch],
            "patch_ids": [row["patch_id"] for row in batch],
            "task": (
                "digepath_roi9_patch_classification"
                if model_name == "digepath"
                else "global_screening_patch_classification"
            ),
        }
        if model_name == "digepath":
            payload["class_names"] = list(DIGEPATH_ROI9_CLASS_NAMES)
        else:
            payload["crc100k_labels"] = list(CONCH_CRC100K_LABELS)
        started = time.time()
        try:
            response = requests.post(server_url, json=payload, timeout=timeout_seconds)
            latency_ms = int(round((time.time() - started) * 1000.0))
            if response.status_code != 200:
                errors.append(
                    {
                        "model": model_name,
                        "status": response.status_code,
                        "latency_ms": latency_ms,
                        "text": response.text[:1000],
                        "batch_first_image": batch[0]["image_path"],
                        "batch_size": len(batch),
                    }
                )
                continue
            response_payload = response.json()
            raw_predictions = response_payload.get("predictions", [])
            raw_by_path = {str(item.get("image_path", "")): item for item in raw_predictions if item.get("image_path")}
            for row in batch:
                item = raw_by_path.get(row["image_path"])
                if item is None:
                    errors.append(
                        {
                            "model": model_name,
                            "status": "missing_prediction",
                            "image_path": row["image_path"],
                        }
                    )
                    continue
                predictions.append(format_prediction(row, item, model_name, latency_ms))
        except requests.Timeout:
            errors.append(
                {
                    "model": model_name,
                    "status": "timeout",
                    "timeout_seconds": timeout_seconds,
                    "batch_first_image": batch[0]["image_path"],
                    "batch_size": len(batch),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "model": model_name,
                    "status": "request_failed",
                    "error": str(exc),
                    "batch_first_image": batch[0]["image_path"],
                    "batch_size": len(batch),
                }
            )
        if progress_every_batches and batch_index % int(progress_every_batches) == 0:
            print(
                json.dumps(
                    {
                        "event": "{0}_classification_progress".format(model_name),
                        "batches_done": batch_index,
                        "predictions_done": len(predictions),
                        "errors": len(errors),
                        "elapsed_seconds": int(round(time.time() - started_all)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return predictions, errors


def format_prediction(row, raw_prediction, model_name, latency_ms):
    if model_name == "conch":
        class_label = str(
            raw_prediction.get("crc_label")
            or raw_prediction.get("label")
            or raw_prediction.get("class_name")
            or raw_prediction.get("prediction")
            or ""
        )
        trace_semantic = _normalize_conch_label(class_label)
        probs = raw_prediction.get("probabilities", {}) or raw_prediction.get("probs", {}) or {}
    else:
        class_label = str(
            raw_prediction.get("class_name")
            or raw_prediction.get("label")
            or raw_prediction.get("prediction")
            or ""
        )
        trace_semantic = _trace_label_from_digepath_class(class_label)
        probs = raw_prediction.get("probabilities", {}) or raw_prediction.get("probs", {}) or {}
    confidence = float(raw_prediction.get("confidence", 0.0) or 0.0)
    result = dict(row)
    result.update(
        {
            "model": model_name,
            "predicted_class": class_label,
            "trace_semantic": trace_semantic,
            "confidence": confidence,
            "probabilities": probs,
            "raw_prediction": raw_prediction,
            "latency_ms_for_batch": latency_ms,
        }
    )
    return result


def summarize(manifest_rows, prediction_rows, errors):
    summary = {
        "n_crops": len(manifest_rows),
        "n_predictions": len(prediction_rows),
        "n_errors": len(errors),
        "crop_counts_by_size": dict(Counter(str(row["crop_size"]) for row in manifest_rows)),
        "prediction_counts_by_model": dict(Counter(row["model"] for row in prediction_rows)),
        "prediction_counts_by_size_and_model": {},
        "class_counts_by_size_and_model": {},
        "trace_semantic_counts_by_size_and_model": {},
        "mean_confidence_by_size_and_model": {},
        "slide_counts": dict(Counter(row["slide_id"] for row in manifest_rows)),
    }
    by_size_model = defaultdict(list)
    for row in prediction_rows:
        key = "{0}|{1}".format(row["crop_size"], row["model"])
        by_size_model[key].append(row)
    for key, rows in sorted(by_size_model.items()):
        summary["prediction_counts_by_size_and_model"][key] = len(rows)
        summary["class_counts_by_size_and_model"][key] = dict(Counter(row["predicted_class"] for row in rows))
        summary["trace_semantic_counts_by_size_and_model"][key] = dict(Counter(row["trace_semantic"] for row in rows))
        summary["mean_confidence_by_size_and_model"][key] = round(
            sum(float(row["confidence"]) for row in rows) / max(1, len(rows)),
            6,
        )
    return summary


def summarize_planned_counts(planned_rows):
    total_by_size = Counter()
    total_by_slide = Counter()
    for row in planned_rows:
        slide_id = row["slide_id"]
        slide_total = 0
        for size, count in row["counts_by_size"].items():
            total_by_size[str(size)] += int(count)
            slide_total += int(count)
        total_by_slide[slide_id] += slide_total
    return {
        "planned_n_crops": sum(total_by_size.values()),
        "planned_crop_counts_by_size": dict(total_by_size),
        "planned_crop_counts_by_slide": dict(total_by_slide),
        "planned_rows": planned_rows,
    }


def main():
    args = parse_args()
    load_runtime_defaults(args)
    crop_sizes = parse_crop_sizes(args.crop_sizes)
    if not args.output_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / "artifacts" / "wsi_fixed_fov_conch_digepath_{0}".format(stamp))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"
    for path in (manifest_path, predictions_path, errors_path):
        if path.exists():
            path.unlink()

    manifest_rows = []
    planned_rows = []
    for grid_path, grid_payload in iter_grid_payloads(args.grid_dir, limit_slides=args.limit_slides):
        slide_id = str(grid_payload.get("slide_id") or grid_path.name.split("_tissuegrid", 1)[0])
        wsi_path = resolve_wsi_path(slide_id, grid_payload, args.wsi_dir)
        counts_by_size = estimate_crop_counts(
            grid_payload,
            crop_sizes,
            tiling_mode=args.tiling_mode,
            stride=args.stride,
            limit_patches_per_slide=args.limit_patches_per_slide,
        )
        planned_rows.append(
            {
                "slide_id": slide_id,
                "wsi_path": str(wsi_path),
                "slide_dimensions_level0": grid_payload.get("slide_dimensions_level0", []),
                "counts_by_size": counts_by_size,
                "total_crops": int(sum(int(v) for v in counts_by_size.values())),
            }
        )
        print(
            json.dumps(
                {
                    "event": "crop_slide_start",
                    "slide_id": slide_id,
                    "wsi_path": str(wsi_path),
                    "crop_sizes": crop_sizes,
                    "tiling_mode": args.tiling_mode,
                    "planned_counts_by_size": counts_by_size,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.dry_run_counts:
            continue
        rows = crop_slide(
            grid_path=grid_path,
            grid_payload=grid_payload,
            wsi_path=wsi_path,
            crop_sizes=crop_sizes,
            output_dir=output_dir,
            tiling_mode=args.tiling_mode,
            stride=args.stride,
            limit_patches_per_slide=args.limit_patches_per_slide,
            allow_pil_fallback=args.allow_pil_fallback,
        )
        manifest_rows.extend(rows)
        append_jsonl(manifest_path, rows)
        print(
            json.dumps(
                {"event": "crop_slide_done", "slide_id": slide_id, "n_crops": len(rows)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    if args.dry_run_counts:
        summary = summarize_planned_counts(planned_rows)
        summary.update(
            {
                "output_dir": str(output_dir),
                "conch_url": args.conch_url,
                "digepath_url": args.digepath_url,
                "crop_sizes": crop_sizes,
                "tiling_mode": args.tiling_mode,
                "stride": int(args.stride),
                "grid_dir": str(args.grid_dir),
                "wsi_dir": str(args.wsi_dir),
                "dry_run_counts": True,
            }
        )
        write_json(output_dir / "summary.json", summary)
        print(json.dumps({"event": "done", "summary_json": str(output_dir / "summary.json")}, ensure_ascii=False), flush=True)
        return

    prediction_rows = []
    errors = []
    if not args.skip_conch:
        print(json.dumps({"event": "conch_classification_start", "n_crops": len(manifest_rows)}, ensure_ascii=False), flush=True)
        conch_rows, conch_errors = post_predictions(
            args.conch_url,
            manifest_rows,
            model_name="conch",
            timeout_seconds=args.timeout_seconds,
            batch_size=args.batch_size,
            progress_every_batches=args.progress_every_batches,
        )
        prediction_rows.extend(conch_rows)
        errors.extend(conch_errors)
        append_jsonl(predictions_path, conch_rows)
        append_jsonl(errors_path, conch_errors)
        print(
            json.dumps(
                {"event": "conch_classification_done", "n_predictions": len(conch_rows), "n_errors": len(conch_errors)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    if not args.skip_digepath:
        print(json.dumps({"event": "digepath_classification_start", "n_crops": len(manifest_rows)}, ensure_ascii=False), flush=True)
        digepath_rows, digepath_errors = post_predictions(
            args.digepath_url,
            manifest_rows,
            model_name="digepath",
            timeout_seconds=args.timeout_seconds,
            batch_size=args.batch_size,
            progress_every_batches=args.progress_every_batches,
        )
        prediction_rows.extend(digepath_rows)
        errors.extend(digepath_errors)
        append_jsonl(predictions_path, digepath_rows)
        append_jsonl(errors_path, digepath_errors)
        print(
            json.dumps(
                {"event": "digepath_classification_done", "n_predictions": len(digepath_rows), "n_errors": len(digepath_errors)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = summarize(manifest_rows, prediction_rows, errors)
    summary.update(
        {
            "output_dir": str(output_dir),
            "manifest_jsonl": str(manifest_path),
            "predictions_jsonl": str(predictions_path),
            "errors_jsonl": str(errors_path),
            "conch_url": args.conch_url,
            "digepath_url": args.digepath_url,
            "crop_sizes": crop_sizes,
            "tiling_mode": args.tiling_mode,
            "stride": int(args.stride),
            "grid_dir": str(args.grid_dir),
            "wsi_dir": str(args.wsi_dir),
            "planned": summarize_planned_counts(planned_rows),
        }
    )
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"event": "done", "summary_json": str(output_dir / "summary.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
