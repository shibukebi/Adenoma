from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import fcntl
import math
import multiprocessing
from pathlib import Path
import queue
import threading

from isyntax import ISyntax
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image

from .config import (
    ISYNTAX_WORKER_TIMEOUT_SECONDS,
    TILE_JPEG_QUALITY,
    WSI_CACHE_SIZE,
    WSI_DISK_CACHE_ROOT,
    WSI_OVERVIEW_MAX_DIMENSION,
)
from .wsi_cache import PersistentWsiCache


Image.MAX_IMAGE_PIXELS = None
DEEPZOOM_TILE_SIZE = 254
DEEPZOOM_OVERLAP = 1
INTERACTIVE_PRIORITY = 0
BACKGROUND_PRIORITY = 10


class ISyntaxOpenSlideAdapter:
    def __init__(self, path):
        self._slide = ISyntax.open(str(path))
        self.level_count = int(self._slide.level_count)
        self.level_dimensions = [tuple(map(int, dims)) for dims in self._slide.level_dimensions]
        self.level_downsamples = [float(value) for value in self._slide.level_downsamples]
        self.dimensions = self.level_dimensions[0]
        self.properties = {}
        mpp_x = getattr(self._slide, "mpp_x", None)
        if mpp_x:
            self.properties[openslide.PROPERTY_NAME_MPP_X] = str(mpp_x)
            self.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER] = str(10.0 / float(mpp_x))

    def read_region(self, location, level, size):
        downsample = self.level_downsamples[level]
        x = int(round(location[0] / downsample))
        y = int(round(location[1] / downsample))
        rgba = self._slide.read_region(x, y, int(size[0]), int(size[1]), level=level)
        return Image.fromarray(rgba, mode="RGBA")

    def get_best_level_for_downsample(self, downsample):
        best = 0
        for level, value in enumerate(self.level_downsamples):
            if value > downsample:
                break
            best = level
        return best

    def close(self):
        self._slide.close()


@dataclass
class SlideEntry:
    path: str
    slide: object
    deepzoom: DeepZoomGenerator
    lock: threading.RLock
    users: int = 0
    pending_close: bool = False

    def close(self):
        if hasattr(self.slide, "close"):
            self.slide.close()


class SlideCache:
    def __init__(self, max_size=WSI_CACHE_SIZE):
        self.max_size = max_size
        self.entries = OrderedDict()
        self.lock = threading.RLock()
        self.opening = {}

    def _open(self, path):
        slide = ISyntaxOpenSlideAdapter(path) if _is_isyntax(path) else openslide.OpenSlide(path)
        return SlideEntry(
            path=str(path),
            slide=slide,
            deepzoom=DeepZoomGenerator(
                slide,
                tile_size=DEEPZOOM_TILE_SIZE,
                overlap=DEEPZOOM_OVERLAP,
                limit_bounds=False,
            ),
            lock=threading.RLock(),
        )

    def _acquire(self, path):
        path = str(Path(path))
        while True:
            with self.lock:
                entry = self.entries.pop(path, None)
                if entry is not None:
                    entry.users += 1
                    self.entries[path] = entry
                    return entry
                opened = self.opening.get(path)
                if opened is None:
                    opened = threading.Event()
                    self.opening[path] = opened
                    break
            opened.wait()

        try:
            entry = self._open(path)
        except Exception:
            with self.lock:
                self.opening.pop(path).set()
            raise

        to_close = []
        with self.lock:
            entry.users = 1
            self.entries[path] = entry
            while len(self.entries) > self.max_size:
                _, old_entry = self.entries.popitem(last=False)
                if old_entry.users:
                    old_entry.pending_close = True
                else:
                    to_close.append(old_entry)
            self.opening.pop(path).set()
        for old_entry in to_close:
            old_entry.close()
        return entry

    def _release(self, entry):
        should_close = False
        with self.lock:
            entry.users -= 1
            if entry.users == 0 and entry.pending_close:
                should_close = True
        if should_close:
            entry.close()

    @contextmanager
    def use(self, path):
        entry = self._acquire(path)
        try:
            yield entry
        finally:
            self._release(entry)

    def close_all(self):
        to_close = []
        with self.lock:
            for entry in self.entries.values():
                if entry.users:
                    entry.pending_close = True
                else:
                    to_close.append(entry)
            self.entries.clear()
        for entry in to_close:
            entry.close()


slide_cache = SlideCache()
disk_cache = PersistentWsiCache()


def _is_isyntax(path) -> bool:
    return str(path).lower().endswith(".isyntax")


def _jpeg_bytes(image, quality=TILE_JPEG_QUALITY):
    if image.mode != "RGB":
        background = Image.new("RGB", image.size, "white")
        if "A" in image.getbands():
            background.paste(image, mask=image.getchannel("A"))
        else:
            background.paste(image)
        image = background
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def _local_get_tile(path, level, address):
    with slide_cache.use(path) as entry:
        with entry.lock:
            tile = entry.deepzoom.get_tile(level, address)
    return _jpeg_bytes(tile)


def _local_get_dzi(path):
    with slide_cache.use(path) as entry:
        return entry.deepzoom.get_dzi("jpeg")


def _local_get_metadata(path):
    with slide_cache.use(path) as entry:
        with entry.lock:
            objective = entry.slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
            mpp_x = entry.slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
            width, height = entry.slide.dimensions
            return {
                "width": width,
                "height": height,
                "level_count": entry.deepzoom.level_count,
                "tile_size": DEEPZOOM_TILE_SIZE,
                "tile_overlap": DEEPZOOM_OVERLAP,
                "objective_power": float(objective) if objective else None,
                "mpp_x": float(mpp_x) if mpp_x else None,
            }


def _local_get_thumbnail(path, max_size=1200):
    with slide_cache.use(path) as entry:
        with entry.lock:
            dimensions = entry.slide.level_dimensions
            eligible = [(level, dims) for level, dims in enumerate(dimensions) if max(dims) <= max_size * 2]
            level, dims = eligible[0] if eligible else (len(dimensions) - 1, dimensions[-1])
            image = entry.slide.read_region((0, 0), level, dims)
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return _jpeg_bytes(image, quality=88)


def _isyntax_worker_dispatch(operation: str, path: str, args: tuple):
    if operation == "_test_crash":
        import os

        os._exit(17)
    if operation == "_test_echo":
        return args[0]
    lock_path = WSI_DISK_CACHE_ROOT / "isyntax-reader.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if operation == "metadata":
                return _local_get_metadata(path)
            if operation == "dzi":
                return _local_get_dzi(path)
            if operation == "thumbnail":
                return _local_get_thumbnail(path, *args)
            if operation == "tile":
                return _local_get_tile(path, *args)
            raise ValueError(f"Unknown iSyntax operation: {operation}")
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


@dataclass
class WorkerJob:
    operation: str
    path: str
    args: tuple
    event: threading.Event
    result: object = None
    error: Exception | None = None


class ISyntaxWorkerManager:
    def __init__(self, timeout=ISYNTAX_WORKER_TIMEOUT_SECONDS):
        self.timeout = timeout
        self.jobs = queue.PriorityQueue()
        self.sequence = 0
        self.sequence_lock = threading.Lock()
        self.lifecycle_lock = threading.Lock()
        self.executor = None
        self.thread = None
        self.stop_event = threading.Event()
        self.restarts = 0

    def _new_executor(self):
        return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))

    def start(self):
        with self.lifecycle_lock:
            if self.thread and self.thread.is_alive():
                return
            self.stop_event.clear()
            self.executor = self._new_executor()
            self.thread = threading.Thread(target=self._run, name="isyntax-worker-manager", daemon=True)
            self.thread.start()

    def _reset_executor(self):
        executor = self.executor
        if executor is not None:
            for process in list(getattr(executor, "_processes", {}).values()):
                if process.is_alive():
                    process.terminate()
            executor.shutdown(wait=False, cancel_futures=True)
        self.executor = self._new_executor()
        self.restarts += 1

    def _run(self):
        while not self.stop_event.is_set():
            try:
                _, _, job = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                future = self.executor.submit(_isyntax_worker_dispatch, job.operation, job.path, job.args)
                job.result = future.result(timeout=self.timeout)
            except (FutureTimeoutError, BrokenProcessPool) as exc:
                self._reset_executor()
                job.error = RuntimeError(f"iSyntax worker restarted after {type(exc).__name__}")
            except Exception as exc:
                job.error = exc
            finally:
                job.event.set()

    def call(self, operation: str, path: str, *args, priority=INTERACTIVE_PRIORITY):
        self.start()
        job = WorkerJob(operation=operation, path=str(path), args=args, event=threading.Event())
        with self.sequence_lock:
            self.sequence += 1
            sequence = self.sequence
        self.jobs.put((priority, sequence, job))
        if not job.event.wait(self.timeout * 2):
            raise TimeoutError(f"Timed out waiting for iSyntax {operation}")
        if job.error:
            raise job.error
        return job.result

    def status(self):
        return {
            "running": bool(self.thread and self.thread.is_alive()),
            "queued_operations": self.jobs.qsize(),
            "restarts": self.restarts,
        }

    def close(self):
        with self.lifecycle_lock:
            self.stop_event.set()
            with self.sequence_lock:
                self.sequence += 1
                sequence = self.sequence
            self.jobs.put((-100, sequence, None))
            executor = self.executor
            if executor is not None:
                for process in list(getattr(executor, "_processes", {}).values()):
                    if process.is_alive():
                        process.terminate()
                executor.shutdown(wait=False, cancel_futures=True)
            if self.thread:
                self.thread.join(timeout=3)
            self.executor = None
            self.thread = None


isyntax_worker = ISyntaxWorkerManager()


def _cache_identity(path, slide_id=None):
    path = str(path)
    return slide_id or Path(path).stem, path


def get_metadata(path, slide_id=None, pinned=False, priority=INTERACTIVE_PRIORITY):
    if not _is_isyntax(path):
        return _local_get_metadata(path)
    slide_id, source_path = _cache_identity(path, slide_id)
    cache_path = disk_cache.metadata_path(slide_id, source_path)
    cached = disk_cache.read_json(cache_path)
    if cached is not None:
        return cached
    with disk_cache.key_lock(str(cache_path)):
        cached = disk_cache.read_json(cache_path)
        if cached is not None:
            return cached
        metadata = isyntax_worker.call("metadata", source_path, priority=priority)
        disk_cache.write_json(cache_path, metadata, slide_id, source_path, "metadata", True)
        return metadata


def get_dzi(path, slide_id=None, pinned=False, priority=INTERACTIVE_PRIORITY):
    if not _is_isyntax(path):
        return _local_get_dzi(path)
    slide_id, source_path = _cache_identity(path, slide_id)
    cache_path = disk_cache.dzi_path(slide_id, source_path)
    cached = disk_cache.read_bytes(cache_path)
    if cached is not None:
        return cached.decode("utf-8")
    with disk_cache.key_lock(str(cache_path)):
        cached = disk_cache.read_bytes(cache_path)
        if cached is not None:
            return cached.decode("utf-8")
        dzi = isyntax_worker.call("dzi", source_path, priority=priority)
        disk_cache.write_bytes(cache_path, dzi.encode("utf-8"), slide_id, source_path, "dzi", True)
        return dzi


def get_thumbnail(path, slide_id=None, max_size=1200, pinned=False, priority=INTERACTIVE_PRIORITY):
    if not _is_isyntax(path):
        return _local_get_thumbnail(path, max_size)
    slide_id, source_path = _cache_identity(path, slide_id)
    cache_path = disk_cache.thumbnail_path(slide_id, source_path)
    cached = disk_cache.read_bytes(cache_path)
    if cached is not None:
        return cached
    with disk_cache.key_lock(str(cache_path)):
        cached = disk_cache.read_bytes(cache_path)
        if cached is not None:
            return cached
        content = isyntax_worker.call("thumbnail", source_path, max_size, priority=priority)
        disk_cache.write_bytes(cache_path, content, slide_id, source_path, "thumbnail", pinned)
        return content


def get_tile(
    path,
    level,
    address,
    slide_id=None,
    pinned=False,
    priority=INTERACTIVE_PRIORITY,
):
    if not _is_isyntax(path):
        return _local_get_tile(path, level, address)
    slide_id, source_path = _cache_identity(path, slide_id)
    column, row = address
    cache_path = disk_cache.tile_path(slide_id, source_path, level, column, row)
    cached = disk_cache.read_bytes(cache_path)
    if cached is not None:
        return cached
    with disk_cache.key_lock(str(cache_path)):
        cached = disk_cache.read_bytes(cache_path)
        if cached is not None:
            return cached
        content = isyntax_worker.call("tile", source_path, level, address, priority=priority)
        disk_cache.write_bytes(cache_path, content, slide_id, source_path, "tile", pinned)
        return content


def _overview_target_level(metadata, max_dimension=WSI_OVERVIEW_MAX_DIMENSION):
    max_level = metadata["level_count"] - 1
    largest = max(metadata["width"], metadata["height"])
    if largest <= max_dimension:
        return max_level
    return max(0, max_level - math.ceil(math.log2(largest / max_dimension)))


def prewarm_slide(slide_id: str, source_path: str, pinned: bool = False):
    if disk_cache.overview_ready(slide_id, source_path):
        disk_cache.set_warm_status(slide_id, "ready")
        return {"slide_id": slide_id, "status": "ready", "skipped": True}
    disk_cache.set_warm_status(slide_id, "warming")
    try:
        metadata = get_metadata(
            source_path, slide_id, pinned=True, priority=BACKGROUND_PRIORITY
        )
        get_dzi(source_path, slide_id, pinned=True, priority=BACKGROUND_PRIORITY)
        get_thumbnail(
            source_path,
            slide_id,
            max_size=1200,
            pinned=pinned,
            priority=BACKGROUND_PRIORITY,
        )
        max_level = metadata["level_count"] - 1
        target_level = _overview_target_level(metadata)
        tile_size = metadata["tile_size"]
        generated = 0
        for level in range(target_level + 1):
            scale = 2 ** (max_level - level)
            level_width = math.ceil(metadata["width"] / scale)
            level_height = math.ceil(metadata["height"] / scale)
            columns = math.ceil(level_width / tile_size)
            rows = math.ceil(level_height / tile_size)
            for row in range(rows):
                for column in range(columns):
                    get_tile(
                        source_path,
                        level,
                        (column, row),
                        slide_id=slide_id,
                        pinned=pinned,
                        priority=BACKGROUND_PRIORITY,
                    )
                    generated += 1
        disk_cache.mark_overview_ready(slide_id, source_path, pinned)
        disk_cache.set_warm_status(slide_id, "ready")
        return {
            "slide_id": slide_id,
            "status": "ready",
            "target_level": target_level,
            "tiles": generated,
        }
    except Exception as exc:
        disk_cache.set_warm_status(slide_id, "failed", str(exc)[:1000])
        raise


class WarmManager:
    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        disk_cache.reset_interrupted_warms()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="wsi-prewarm", daemon=True)
        self.thread.start()

    def submit(self, slide_id: str, source_path: str, priority=20, pinned=False):
        status = disk_cache.enqueue_warm(slide_id, source_path, priority, pinned)
        self.wake_event.set()
        return status

    def _run(self):
        while not self.stop_event.is_set():
            job = disk_cache.claim_next_warm()
            if not job:
                self.wake_event.wait(1.0)
                self.wake_event.clear()
                continue
            try:
                prewarm_slide(job["slide_id"], job["source_path"], bool(job["pinned"]))
            except Exception:
                pass

    def close(self):
        self.stop_event.set()
        self.wake_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.thread = None


warm_manager = WarmManager()


class WsiService:
    def start(self):
        warm_manager.start()

    def close(self):
        warm_manager.close()
        isyntax_worker.close()
        slide_cache.close_all()

    def cache_status(self, slide_id: str, source_path: str):
        return disk_cache.warm_status(slide_id, source_path)

    def stats(self):
        return {**disk_cache.stats(), "worker": isyntax_worker.status()}


wsi_service = WsiService()
