"""Shared, provenance-first whole-slide image access.

The production adapters in this module never fall back to Pillow for an
unreadable WSI.  Small raster images remain supported as an explicit fixture
format for tests and local contract exercises.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from PIL import Image


WSI_SUFFIXES = frozenset(
    {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".bif", ".isyntax"}
)
RASTER_FIXTURE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
SUPPORTED_IMAGE_SUFFIXES = WSI_SUFFIXES | RASTER_FIXTURE_SUFFIXES


class WSIError(RuntimeError):
    """Base class for explicit WSI integration failures."""


class WSIReaderUnavailableError(WSIError):
    """No valid reader backend is available for the requested slide."""


class WSIPhysicalMetadataError(WSIError):
    """Physical-scale metadata required for a trustworthy crop is missing."""


def _positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _normalize_mpp(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return _positive_float(value[0]), _positive_float(value[1])
    scalar = _positive_float(value)
    return scalar, scalar


def _white_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, image).convert("RGB")
    return image.convert("RGB")


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass(frozen=True)
class WSICropResult:
    image: Image.Image
    provenance: Mapping[str, Any]
    image_sha256: str
    _encoded_png: bytes = field(repr=False)

    @property
    def pixel_dimensions(self) -> Tuple[int, int]:
        return tuple(int(value) for value in self.image.size)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._encoded_png)
        return path


class _CtypesOpenSlideAdapter:
    backend_name = "openslide_ctypes"
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
            raise WSIReaderUnavailableError("libopenslide is unavailable: {0}".format(last_error))
        lib.openslide_open.argtypes = [ctypes.c_char_p]
        lib.openslide_open.restype = ctypes.c_void_p
        lib.openslide_close.argtypes = [ctypes.c_void_p]
        lib.openslide_close.restype = None
        lib.openslide_get_error.argtypes = [ctypes.c_void_p]
        lib.openslide_get_error.restype = ctypes.c_char_p
        lib.openslide_get_level_count.argtypes = [ctypes.c_void_p]
        lib.openslide_get_level_count.restype = ctypes.c_int32
        lib.openslide_get_level_downsample.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.openslide_get_level_downsample.restype = ctypes.c_double
        lib.openslide_get_level0_dimensions.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.openslide_get_level_dimensions.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.openslide_read_region.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        lib.openslide_get_property_value.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.openslide_get_property_value.restype = ctypes.c_char_p
        cls._lib = lib
        return lib

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lib = self._load_lib()
        self.handle = self.lib.openslide_open(str(self.path).encode("utf-8"))
        if not self.handle:
            raise WSIReaderUnavailableError("OpenSlide could not open {0}".format(self.path))
        error = self.lib.openslide_get_error(self.handle)
        if error:
            message = error.decode("utf-8", errors="replace")
            self.close()
            raise WSIReaderUnavailableError(message)
        width = ctypes.c_int64()
        height = ctypes.c_int64()
        self.lib.openslide_get_level0_dimensions(self.handle, ctypes.byref(width), ctypes.byref(height))
        self.dimensions = (int(width.value), int(height.value))
        self.level_count = int(self.lib.openslide_get_level_count(self.handle))
        self.level_dimensions = []
        self.level_downsamples = []
        for level in range(self.level_count):
            level_width = ctypes.c_int64()
            level_height = ctypes.c_int64()
            self.lib.openslide_get_level_dimensions(
                self.handle,
                int(level),
                ctypes.byref(level_width),
                ctypes.byref(level_height),
            )
            self.level_dimensions.append((int(level_width.value), int(level_height.value)))
            self.level_downsamples.append(
                float(self.lib.openslide_get_level_downsample(self.handle, int(level)))
            )
        self.properties = {}
        for key in (
            "openslide.objective-power",
            "aperio.AppMag",
            "hamamatsu.SourceLens",
            "openslide.mpp-x",
            "openslide.mpp-y",
            "aperio.MPP",
        ):
            raw = self.lib.openslide_get_property_value(self.handle, key.encode("utf-8"))
            if raw:
                self.properties[key] = raw.decode("utf-8", errors="replace")

    def read_region(self, top_left: Sequence[int], level: int, size: Sequence[int]) -> Image.Image:
        x, y = [int(value) for value in top_left]
        width, height = [int(value) for value in size]
        buffer = (ctypes.c_uint32 * (width * height))()
        self.lib.openslide_read_region(self.handle, buffer, x, y, int(level), width, height)
        error = self.lib.openslide_get_error(self.handle)
        if error:
            raise WSIError(error.decode("utf-8", errors="replace"))
        data = ctypes.string_at(buffer, width * height * 4)
        return Image.frombuffer("RGBA", (width, height), data, "raw", "BGRA", 0, 1)

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.openslide_close(self.handle)
            self.handle = None


class _PythonOpenSlideAdapter:
    backend_name = "openslide_python"

    def __init__(self, path: Path):
        try:
            import openslide
        except Exception as exc:
            raise WSIReaderUnavailableError("openslide-python is unavailable: {0}".format(exc))
        try:
            self.slide = openslide.open_slide(str(path))
        except Exception as exc:
            raise WSIReaderUnavailableError("OpenSlide could not open {0}: {1}".format(path, exc))
        self.dimensions = tuple(int(value) for value in self.slide.dimensions)
        self.level_dimensions = [tuple(int(value) for value in row) for row in self.slide.level_dimensions]
        self.level_downsamples = [float(value) for value in self.slide.level_downsamples]
        self.level_count = len(self.level_dimensions)
        self.properties = dict(self.slide.properties)

    def read_region(self, top_left: Sequence[int], level: int, size: Sequence[int]) -> Image.Image:
        return self.slide.read_region(tuple(int(value) for value in top_left), int(level), tuple(int(value) for value in size))

    def close(self) -> None:
        self.slide.close()


class _ISyntaxAdapter:
    backend_name = "pyisyntax"

    def __init__(self, path: Path):
        try:
            from isyntax import ISyntax
        except Exception as exc:
            raise WSIReaderUnavailableError(
                "pyisyntax is required for .isyntax slides: {0}".format(exc)
            )
        try:
            self.slide = ISyntax.open(str(path))
        except Exception as exc:
            raise WSIReaderUnavailableError("pyisyntax could not open {0}: {1}".format(path, exc))
        self.dimensions = tuple(int(value) for value in self.slide.level_dimensions[0])
        self.level_dimensions = [
            tuple(int(value) for value in self.slide.level_dimensions[index])
            for index in range(int(self.slide.level_count))
        ]
        self.level_downsamples = [
            float(self.slide.level_downsamples[index])
            for index in range(int(self.slide.level_count))
        ]
        self.level_count = len(self.level_dimensions)
        self.properties = {
            "isyntax.mpp-x": getattr(self.slide, "mpp_x", None),
            "isyntax.mpp-y": getattr(self.slide, "mpp_y", None),
        }

    def read_region(self, top_left: Sequence[int], level: int, size: Sequence[int]) -> Image.Image:
        x, y = [int(value) for value in top_left]
        width, height = [int(value) for value in size]
        array = self.slide.read_region(x, y, width, height, level=int(level))
        return Image.fromarray(array, mode="RGBA" if getattr(array, "shape", (0, 0, 0))[-1:] == (4,) else "RGB")

    def close(self) -> None:
        self.slide.close()


class _RasterFixtureAdapter:
    backend_name = "pillow_fixture"

    def __init__(self, path: Path):
        try:
            self.image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise WSIReaderUnavailableError("Fixture image could not be opened: {0}".format(exc))
        self.dimensions = tuple(int(value) for value in self.image.size)
        self.level_dimensions = [self.dimensions]
        self.level_downsamples = [1.0]
        self.level_count = 1
        self.properties = {}

    def read_region(self, top_left: Sequence[int], level: int, size: Sequence[int]) -> Image.Image:
        if int(level) != 0:
            raise WSIError("Raster fixture has only level 0")
        x, y = [int(value) for value in top_left]
        width, height = [int(value) for value in size]
        output = Image.new("RGB", (width, height), (255, 255, 255))
        source_box = (
            max(0, x),
            max(0, y),
            min(self.image.width, x + width),
            min(self.image.height, y + height),
        )
        if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
            output.paste(self.image.crop(source_box), (max(0, -x), max(0, -y)))
        return output

    def close(self) -> None:
        self.image.close()


class WSIReader:
    """Unified WSI reader with explicit backend and physical provenance."""

    def __init__(
        self,
        path: Path,
        base_magnification: Optional[float] = None,
        mpp: Any = None,
        allow_raster_fixture: bool = True,
    ):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise WSIReaderUnavailableError("Slide does not exist: {0}".format(self.path))
        suffix = self.path.suffix.lower()
        if suffix == ".isyntax":
            self._adapter = _ISyntaxAdapter(self.path)
        elif suffix in RASTER_FIXTURE_SUFFIXES:
            if not allow_raster_fixture:
                raise WSIReaderUnavailableError("Raster fixture input is disabled")
            self._adapter = _RasterFixtureAdapter(self.path)
        elif suffix in WSI_SUFFIXES:
            errors = []
            for adapter_class in (_PythonOpenSlideAdapter, _CtypesOpenSlideAdapter):
                try:
                    self._adapter = adapter_class(self.path)
                    break
                except Exception as exc:
                    errors.append("{0}: {1}".format(adapter_class.backend_name, exc))
            else:
                raise WSIReaderUnavailableError(
                    "No WSI backend could open {0}; {1}".format(self.path, "; ".join(errors))
                )
        else:
            raise WSIReaderUnavailableError("Unsupported slide suffix: {0}".format(suffix))
        self.backend_name = self._adapter.backend_name
        self.mode = self.backend_name
        self.dimensions = tuple(self._adapter.dimensions)
        self.level_dimensions = tuple(tuple(row) for row in self._adapter.level_dimensions)
        self.level_downsamples = tuple(float(value) for value in self._adapter.level_downsamples)
        self.level_count = len(self.level_dimensions)
        self.properties = dict(getattr(self._adapter, "properties", {}) or {})

        explicit_mpp = _normalize_mpp(mpp)
        property_mpp_x = _positive_float(
            self.properties.get("openslide.mpp-x")
            or self.properties.get("aperio.MPP")
            or self.properties.get("isyntax.mpp-x")
        )
        property_mpp_y = _positive_float(
            self.properties.get("openslide.mpp-y")
            or self.properties.get("aperio.MPP")
            or self.properties.get("isyntax.mpp-y")
        )
        self.mpp_x = explicit_mpp[0] or property_mpp_x
        self.mpp_y = explicit_mpp[1] or property_mpp_y
        if mpp is not None:
            self.mpp_source = "explicit_override"
        elif self.mpp_x and self.mpp_y:
            self.mpp_source = "slide_metadata"
        else:
            self.mpp_source = None

        property_base = None
        for key in ("openslide.objective-power", "aperio.AppMag", "hamamatsu.SourceLens"):
            property_base = _positive_float(self.properties.get(key))
            if property_base:
                break
        explicit_base = _positive_float(base_magnification)
        if explicit_base:
            self.base_magnification = explicit_base
            self.base_magnification_source = "explicit_override"
        elif property_base:
            self.base_magnification = property_base
            self.base_magnification_source = "slide_metadata"
        elif self.mpp_x and self.mpp_y:
            self.base_magnification = 10.0 / ((self.mpp_x + self.mpp_y) / 2.0)
            self.base_magnification_source = "derived_from_slide_mpp"
        else:
            self.base_magnification = None
            self.base_magnification_source = None
        if (self.mpp_x is None or self.mpp_y is None) and explicit_base:
            derived_mpp = 10.0 / explicit_base
            self.mpp_x = self.mpp_x or derived_mpp
            self.mpp_y = self.mpp_y or derived_mpp
            self.mpp_source = "derived_from_explicit_base_magnification"

    def read_region(self, top_left: Sequence[int], level: int, size: Sequence[int]) -> Image.Image:
        return self._adapter.read_region(top_left, level, size)

    def require_physical_metadata(self) -> None:
        missing = []
        if not _positive_float(self.base_magnification):
            missing.append("base_magnification")
        if not _positive_float(self.mpp_x) or not _positive_float(self.mpp_y):
            missing.append("mpp")
        if missing:
            raise WSIPhysicalMetadataError(
                "Reliable physical metadata is unavailable for {0}: missing {1}".format(
                    self.path,
                    ", ".join(missing),
                )
            )

    def select_level(self, requested_magnification: float) -> Tuple[int, float]:
        self.require_physical_metadata()
        requested = _positive_float(requested_magnification)
        if requested is None:
            raise ValueError("requested_magnification must be positive")
        target_downsample = float(self.base_magnification) / requested
        candidates = [(index, value) for index, value in enumerate(self.level_downsamples) if value > 0.0]
        preferred = [item for item in candidates if item[1] <= target_downsample + 1e-9]
        if preferred:
            return max(preferred, key=lambda item: item[1])
        return min(candidates, key=lambda item: item[1])

    def crop_level0_bbox(
        self,
        level0_bbox: Sequence[int],
        requested_magnification: float,
        output_pixels: Optional[Any] = None,
    ) -> WSICropResult:
        self.require_physical_metadata()
        bbox = tuple(int(value) for value in level0_bbox)
        if len(bbox) != 4 or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("level0_bbox must be ordered non-negative [x1, y1, x2, y2]")
        source_width = bbox[2] - bbox[0]
        source_height = bbox[3] - bbox[1]
        target_downsample = float(self.base_magnification) / float(requested_magnification)
        if target_downsample <= 0.0:
            raise ValueError("requested_magnification must be positive")
        if output_pixels is None:
            output_size = (
                max(1, int(round(source_width / target_downsample))),
                max(1, int(round(source_height / target_downsample))),
            )
        elif isinstance(output_pixels, int):
            output_size = (int(output_pixels), int(output_pixels))
        else:
            output_size = tuple(int(value) for value in output_pixels)
            if len(output_size) != 2:
                raise ValueError("output_pixels must be an int or [width, height]")
        if any(value <= 0 for value in output_size):
            raise ValueError("output pixel dimensions must be positive")

        fov_microns = (source_width * float(self.mpp_x), source_height * float(self.mpp_y))
        output_mpp = (
            float(fov_microns[0]) / float(output_size[0]),
            float(fov_microns[1]) / float(output_size[1]),
        )
        expected_output_mpp = 10.0 / float(requested_magnification)
        relative_errors = tuple(
            abs(value - expected_output_mpp) / expected_output_mpp for value in output_mpp
        )
        if any(error > 0.10 for error in relative_errors):
            raise WSIPhysicalMetadataError(
                "ROI bbox/output pixels are inconsistent with requested magnification {0:g}x: "
                "output_mpp={1}, expected_mpp={2:.6f}".format(
                    float(requested_magnification),
                    output_mpp,
                    expected_output_mpp,
                )
            )

        level, source_downsample = self.select_level(requested_magnification)
        source_level_size = (
            max(1, int(math.ceil(source_width / source_downsample))),
            max(1, int(math.ceil(source_height / source_downsample))),
        )
        image = _white_rgb(self.read_region((bbox[0], bbox[1]), level, source_level_size))
        if image.size != output_size:
            image = image.resize(output_size, resample=getattr(Image, "Resampling", Image).LANCZOS)
        encoded = _png_bytes(image)
        digest = hashlib.sha256(encoded).hexdigest()
        provenance: Dict[str, Any] = {
            "schema_version": "wsi_crop_provenance_v1",
            "local_source_path": str(self.path),
            "source_backend": self.backend_name,
            "source_level": int(level),
            "source_downsample": float(source_downsample),
            "mpp_x": float(self.mpp_x),
            "mpp_y": float(self.mpp_y),
            "mpp_source": self.mpp_source,
            "base_magnification": float(self.base_magnification),
            "base_magnification_source": self.base_magnification_source,
            "requested_magnification": float(requested_magnification),
            "target_downsample": float(target_downsample),
            "level0_bbox": list(bbox),
            "fov_microns": [float(fov_microns[0]), float(fov_microns[1])],
            "output_pixel_dimensions": list(output_size),
            "output_mpp": [float(output_mpp[0]), float(output_mpp[1])],
            "expected_output_mpp": float(expected_output_mpp),
            "magnification_mpp_relative_error": [
                float(relative_errors[0]),
                float(relative_errors[1]),
            ],
            "image_sha256": digest,
        }
        return WSICropResult(image=image, provenance=provenance, image_sha256=digest, _encoded_png=encoded)

    def close(self) -> None:
        self._adapter.close()

    def __enter__(self) -> "WSIReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def open_wsi(
    path: Path,
    base_magnification: Optional[float] = None,
    mpp: Any = None,
    allow_raster_fixture: bool = True,
) -> WSIReader:
    return WSIReader(
        path,
        base_magnification=base_magnification,
        mpp=mpp,
        allow_raster_fixture=allow_raster_fixture,
    )
