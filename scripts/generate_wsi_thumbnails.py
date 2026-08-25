#!/usr/bin/env python3
"""
批量读取全切片图像（WSI），生成适用于 Patho-R1 推理的高质量缩略图。

设计目标：
1. 支持常见病理格式：.svs / .tif / .tiff / .ndpi 等
2. 优先使用 OpenSlide 读取，多数病理金字塔格式兼容性更好
3. 当 OpenSlide 无法打开时，自动回退到 tifffile
4. 自动选择合适层级，避免直接读取 level 0 导致内存溢出
5. 在组织裁剪与网格筛选时使用 tissue mask，缩略图本身保留原始背景像素

示例：
python tools/generate_wsi_thumbnails.py \
    --input-dir /path/to/wsi_dir \
    --output-dir /path/to/output_dir \
    --recursive
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - 属于环境依赖问题
    raise SystemExit("缺少依赖 numpy，请先安装后再运行脚本。") from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - 属于环境依赖问题
    raise SystemExit("缺少依赖 Pillow，请先安装后再运行脚本。") from exc
try:
    from PIL import ImageDraw
except ImportError as exc:  # pragma: no cover - 属于环境依赖问题
    raise SystemExit("缺少依赖 Pillow.ImageDraw，请先安装后再运行脚本。") from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - 属于环境依赖问题
    raise SystemExit("缺少依赖 tqdm，请先安装后再运行脚本。") from exc

try:
    import openslide
except Exception:  # pragma: no cover - 运行环境缺少 openslide 时走回退逻辑
    openslide = None

try:
    from isyntax import ISyntax
except Exception:  # pragma: no cover - 运行环境缺少 pyisyntax 时走其他后端
    ISyntax = None

try:
    import tifffile
except Exception:  # pragma: no cover - 运行环境缺少 tifffile 时给出清晰报错
    tifffile = None

try:
    import h5py
except Exception:  # pragma: no cover - 某些模式可能不需要 h5py
    h5py = None

# Pillow 9/10/11 的重采样接口略有差异，这里做兼容处理。
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    RESAMPLE_LANCZOS = Image.LANCZOS


SUPPORTED_EXTENSIONS = {
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".isyntax",
    ".mrxs",
    ".scn",
    ".vms",
    ".vmu",
    ".bif",
}


@dataclass
class LevelInfo:
    """记录某个金字塔层级的核心信息，便于后续做统一打分。"""

    index: int
    width: int
    height: int
    long_side: int
    downsample: float
    estimated_magnification: Optional[float]


@dataclass
class OutputArtifacts:
    """统一描述单张切片输出的几个文件路径。"""

    thumbnail_path: Path
    grid_overlay_path: Optional[Path]
    metadata_path: Optional[Path]


@dataclass
class TissueCropResult:
    """保存裁剪后的 thumbnail / mask 及其相对原 thumbnail 的偏移。"""

    cropped_thumbnail: Image.Image
    cropped_mask: np.ndarray
    crop_x: int
    crop_y: int
    full_thumbnail_width: int
    full_thumbnail_height: int


class BaseWSIReader:
    """统一的 WSI 读取接口，屏蔽 OpenSlide / tifffile 的差异。"""

    backend_name: str = "base"

    def get_level_infos(self) -> List[LevelInfo]:
        raise NotImplementedError

    def read_level_image(self, level_index: int) -> Image.Image:
        raise NotImplementedError

    def close(self) -> None:
        """释放底层资源。"""

    def __enter__(self) -> "BaseWSIReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class OpenSlideWSIReader(BaseWSIReader):
    """优先使用 OpenSlide 读取 WSI。"""

    backend_name = "openslide"

    def __init__(self, slide_path: Path) -> None:
        if openslide is None:
            raise RuntimeError("当前环境未安装 openslide-python。")

        self.slide_path = slide_path
        self.slide = openslide.OpenSlide(str(slide_path))
        self.base_magnification = self._extract_base_magnification()

    def _extract_base_magnification(self) -> Optional[float]:
        """
        尝试从不同厂商常见属性中解析扫描物镜倍率。
        常见情况：
        - openslide.objective-power
        - aperio.AppMag
        - hamamatsu.SourceLens
        """
        property_keys: Sequence[str] = (
            getattr(openslide, "PROPERTY_NAME_OBJECTIVE_POWER", "openslide.objective-power"),
            "openslide.objective-power",
            "aperio.AppMag",
            "hamamatsu.SourceLens",
            "leica.objective",
        )
        for key in property_keys:
            value = self.slide.properties.get(key)
            parsed = parse_float(value)
            if parsed is not None and parsed > 0:
                return parsed
        return None

    def get_level_infos(self) -> List[LevelInfo]:
        level_infos: List[LevelInfo] = []
        for level_index in range(self.slide.level_count):
            width, height = self.slide.level_dimensions[level_index]
            downsample = float(self.slide.level_downsamples[level_index])
            estimated_mag = None
            if self.base_magnification is not None and downsample > 0:
                estimated_mag = self.base_magnification / downsample

            level_infos.append(
                LevelInfo(
                    index=level_index,
                    width=int(width),
                    height=int(height),
                    long_side=int(max(width, height)),
                    downsample=downsample,
                    estimated_magnification=estimated_mag,
                )
            )
        return level_infos

    def read_level_image(self, level_index: int) -> Image.Image:
        width, height = self.slide.level_dimensions[level_index]

        # OpenSlide 返回 RGBA，这里统一铺到白底后转成 RGB，
        # 保持 JPEG 输出背景稳定，同时不改写真实组织区域像素。
        rgba_image = self.slide.read_region((0, 0), level_index, (width, height))
        white_canvas = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
        return Image.alpha_composite(white_canvas, rgba_image).convert("RGB")

    def close(self) -> None:
        self.slide.close()


class TiffFileWSIReader(BaseWSIReader):
    """当 OpenSlide 无法读取时，回退到 tifffile。"""

    backend_name = "tifffile"

    def __init__(self, slide_path: Path) -> None:
        if tifffile is None:
            raise RuntimeError("当前环境未安装 tifffile。")

        self.slide_path = slide_path
        self.tif = tifffile.TiffFile(str(slide_path))
        if not self.tif.series:
            raise RuntimeError("tifffile 未找到可读取的图像序列。")

        # 对 WSI 而言，通常第一个 series 就是完整金字塔。
        self.series = self.tif.series[0]
        raw_levels = list(getattr(self.series, "levels", []) or [self.series])
        if not raw_levels:
            raw_levels = [self.series]
        self.levels = raw_levels
        self.base_magnification = self._extract_base_magnification()

    def _extract_base_magnification(self) -> Optional[float]:
        """
        尝试从 TIFF 描述信息中解析倍率。
        Aperio 等格式常把倍率写在 ImageDescription 中，例如 AppMag = 20。
        """
        description_candidates: List[str] = []

        try:
            first_page = self.tif.pages[0]
            if getattr(first_page, "description", None):
                description_candidates.append(str(first_page.description))
        except Exception:
            pass

        try:
            if getattr(self.series, "pages", None):
                page0 = self.series.pages[0]
                if getattr(page0, "description", None):
                    description_candidates.append(str(page0.description))
        except Exception:
            pass

        patterns = (
            r"AppMag\s*=?\s*([0-9.]+)",
            r"Magnification\s*=?\s*([0-9.]+)",
            r"Objective\s*=?\s*([0-9.]+)",
        )
        for description in description_candidates:
            for pattern in patterns:
                match = re.search(pattern, description, flags=re.IGNORECASE)
                if match:
                    parsed = parse_float(match.group(1))
                    if parsed is not None and parsed > 0:
                        return parsed
        return None

    def get_level_infos(self) -> List[LevelInfo]:
        base_width, base_height = level_shape_to_wh(self.levels[0].shape)
        base_long_side = float(max(base_width, base_height))

        level_infos: List[LevelInfo] = []
        for level_index, level in enumerate(self.levels):
            width, height = level_shape_to_wh(level.shape)
            long_side = int(max(width, height))
            downsample = base_long_side / max(long_side, 1)

            estimated_mag = None
            if self.base_magnification is not None and downsample > 0:
                estimated_mag = self.base_magnification / downsample

            level_infos.append(
                LevelInfo(
                    index=level_index,
                    width=int(width),
                    height=int(height),
                    long_side=long_side,
                    downsample=float(downsample),
                    estimated_magnification=estimated_mag,
                )
            )
        return level_infos

    def read_level_image(self, level_index: int) -> Image.Image:
        array = self.levels[level_index].asarray()
        rgb_array = normalize_array_to_rgb(array)
        return Image.fromarray(rgb_array, mode="RGB")

    def close(self) -> None:
        self.tif.close()


class ISyntaxWSIReader(BaseWSIReader):
    """使用 pyisyntax 读取 Philips iSyntax。"""

    backend_name = "pyisyntax"

    def __init__(self, slide_path: Path) -> None:
        if ISyntax is None:
            raise RuntimeError("当前环境未安装 pyisyntax。")

        self.slide_path = slide_path
        self.slide = ISyntax.open(str(slide_path))
        self.base_magnification = self._estimate_base_magnification()

    def _estimate_base_magnification(self) -> Optional[float]:
        """
        Philips iSyntax 通常提供 mpp（um/pixel）。
        病理扫描里常见近似关系：
        - 40x ≈ 0.25 um/pixel
        - 20x ≈ 0.50 um/pixel
        因此可用 10 / mpp 估计基准倍率。
        """
        mpp_x = parse_float(getattr(self.slide, "mpp_x", None))
        mpp_y = parse_float(getattr(self.slide, "mpp_y", None))
        candidate_mpp = None
        for value in (mpp_x, mpp_y):
            if value is not None and value > 0:
                candidate_mpp = value
                break
        if candidate_mpp is None:
            return None
        return 10.0 / candidate_mpp

    def get_level_infos(self) -> List[LevelInfo]:
        level_infos: List[LevelInfo] = []
        for level_index in range(int(self.slide.level_count)):
            width, height = self.slide.level_dimensions[level_index]
            downsample = float(self.slide.level_downsamples[level_index])
            estimated_mag = None
            if self.base_magnification is not None and downsample > 0:
                estimated_mag = self.base_magnification / downsample

            level_infos.append(
                LevelInfo(
                    index=level_index,
                    width=int(width),
                    height=int(height),
                    long_side=int(max(width, height)),
                    downsample=downsample,
                    estimated_magnification=estimated_mag,
                )
            )
        return level_infos

    def read_level_image(self, level_index: int) -> Image.Image:
        width, height = self.slide.level_dimensions[level_index]
        rgba_array = self.slide.read_region(0, 0, width, height, level=level_index)
        rgba_image = Image.fromarray(rgba_array, mode="RGBA")
        white_canvas = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
        return Image.alpha_composite(white_canvas, rgba_image).convert("RGB")

    def close(self) -> None:
        self.slide.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量生成适用于 Patho-R1 推理的 WSI 缩略图。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="输入目录，脚本会在该目录中查找 WSI 文件。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录。默认为输入目录下的 thumbnails 子目录。",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="是否递归遍历子目录。",
    )
    parser.add_argument(
        "--mode",
        choices=("grid32x", "auto", "tissue_grid32x_svs", "tissue_grid32x_isyntax"),
        default="grid32x",
        help=(
            "缩略图模式。"
            "grid32x=按固定 32x 下采样并生成网格元数据；"
            "auto=按旧逻辑自动选层并控制长边到 2048-4096；"
            "tissue_grid32x_svs=SVS 专用，复用 CLAM segmentation，先裁组织再切默认 64x64 thumbnail 网格"
            "（等效 2048x2048 level0 5x 视野）；"
            "tissue_grid32x_isyntax=iSyntax 专用，优先复用 CLAM segmentation 裁组织后再切网格。"
        ),
    )
    parser.add_argument(
        "--min-long-side",
        type=int,
        default=2048,
        help="auto 模式下期望缩略图长边的最小像素，默认 2048。",
    )
    parser.add_argument(
        "--max-long-side",
        type=int,
        default=4096,
        help="auto 模式下期望缩略图长边的最大像素，默认 4096。",
    )
    parser.add_argument(
        "--preferred-magnification",
        type=float,
        default=20.0,
        help="auto 模式下优先靠近的扫描倍率，默认 20x。",
    )
    parser.add_argument(
        "--target-downsample",
        type=float,
        default=32.0,
        help="grid32x 模式下的目标下采样倍数，默认 32。",
    )
    parser.add_argument(
        "--grid-cell-size-level0",
        type=int,
        default=16000,
        help="grid32x 模式下，level 0（40x）坐标系中的标准网格边长，默认 16000 像素。",
    )
    parser.add_argument(
        "--grid-overlap-ratio",
        type=float,
        default=0.05,
        help="grid32x / tissue_grid32x_svs 模式下的网格重叠比例，默认 0.05（即 5%% overlap）。",
    )
    parser.add_argument(
        "--grid-cell-size-thumb",
        type=int,
        default=64,
        help="tissue_grid32x_* 模式下，裁剪后 thumbnail 上的网格边长，默认 64 像素（64*32=2048，对齐本项目 5x 视野）。",
    )
    parser.add_argument(
        "--grid-physical-level0-extent",
        type=int,
        default=0,
        help="若大于 0，则作为 grid cell 在 level 0 的物理边长；用于与 UNI 物理视野对齐。本项目 5x 标准为 2048。",
    )
    parser.add_argument(
        "--grid-align-to-patch-h5",
        action="store_true",
        help="使用 --patch-h5-dir 中的 level0 coords 作为 grid 起点，使切图与 UNI patch 坐标对齐。",
    )
    parser.add_argument(
        "--tissue-coverage-threshold",
        type=float,
        default=0.05,
        help="tissue_grid32x_svs 模式下的最小组织覆盖率阈值，默认 0.05。",
    )
    parser.add_argument(
        "--segmentations-dir",
        type=Path,
        default=None,
        help="tissue_grid32x_svs 模式下 CLAM segmentation .pkl 所在目录。",
    )
    parser.add_argument(
        "--patch-h5-dir",
        type=Path,
        default=None,
        help="tissue_grid32x_isyntax 模式下 patch 坐标 .h5 所在目录。",
    )
    parser.add_argument(
        "--mask-jpg-dir",
        type=Path,
        default=None,
        help="tissue_grid32x_isyntax 模式下可选的 mask jpg 目录；当 h5 缺失时用于兜底。",
    )
    parser.add_argument(
        "--no-background-filter",
        action="store_true",
        help="兼容旧参数：跳过背景统计。当前不会再把背景替换为黑色。",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=240,
        help="纯白背景判定阈值，默认 RGB > 240 即视为背景候选。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出文件已存在，是否覆盖。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并行处理的进程数，默认 1。",
    )
    return parser.parse_args()


def discover_wsi_files(input_dir: Path, recursive: bool) -> List[Path]:
    """扫描目录中的 WSI 文件。"""
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    wsi_files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(wsi_files)


def open_wsi_reader(slide_path: Path) -> BaseWSIReader:
    """
    先尝试 OpenSlide，若失败再回退到 tifffile。
    这样既兼容 SVS/NDPI，也兼容部分仅能通过 TIFF 方式访问的金字塔 TIFF。
    """
    open_slide_error: Optional[Exception] = None
    isyntax_error: Optional[Exception] = None
    tiff_error: Optional[Exception] = None

    if slide_path.suffix.lower() == ".isyntax":
        if ISyntax is not None:
            try:
                return ISyntaxWSIReader(slide_path)
            except Exception as exc:
                isyntax_error = exc
        raise RuntimeError(
            f"无法读取 iSyntax 文件：{slide_path}\n"
            f"pyisyntax 错误：{isyntax_error}\n"
            "请确认已安装 pyisyntax，且文件未损坏。"
        )

    if openslide is not None:
        try:
            return OpenSlideWSIReader(slide_path)
        except Exception as exc:
            open_slide_error = exc

    if tifffile is not None:
        try:
            return TiffFileWSIReader(slide_path)
        except Exception as exc:
            tiff_error = exc

    raise RuntimeError(
        f"无法读取 WSI 文件：{slide_path}\n"
        f"OpenSlide 错误：{open_slide_error}\n"
        f"pyisyntax 错误：{isyntax_error}\n"
        f"tifffile 错误：{tiff_error}\n"
        "请确认文件未损坏，并已正确安装 openslide-python / pyisyntax / tifffile。"
    )


def choose_best_level(
    level_infos: Sequence[LevelInfo],
    min_long_side: int,
    max_long_side: int,
    preferred_magnification: float,
) -> LevelInfo:
    """
    自动选择最佳层级。

    选择原则：
    1. 强烈避免直接读取 level 0（除非根本没有其他层级）
    2. 优先让长边接近 [2048, 4096] 的目标范围
    3. 如果能解析出扫描倍率，则额外偏向接近 20x 的层级
    """
    if not level_infos:
        raise RuntimeError("WSI 中未找到任何可用层级。")

    target_long_side = int((min_long_side + max_long_side) / 2)
    multi_level = len(level_infos) > 1

    best_level: Optional[LevelInfo] = None
    best_score = float("inf")

    for level in level_infos:
        if min_long_side <= level.long_side <= max_long_side:
            size_penalty = 0.0
        elif level.long_side > max_long_side:
            # 偏大仍可读取后再缩小，因此罚分相对温和。
            size_penalty = (level.long_side - max_long_side) / float(max_long_side)
        else:
            # 偏小意味着细节不足，需要更强的惩罚。
            size_penalty = 1.5 * (min_long_side - level.long_side) / float(min_long_side)

        center_penalty = abs(math.log2(max(level.long_side, 1) / float(target_long_side)))

        if level.estimated_magnification is None or level.estimated_magnification <= 0:
            magnification_penalty = 0.5
        else:
            magnification_penalty = abs(
                math.log2(level.estimated_magnification / float(preferred_magnification))
            )

        # 对特别大的层级追加罚分，避免读出过大的整幅图像。
        oversize_penalty = 0.0
        if level.long_side > max_long_side * 2:
            oversize_penalty = (level.long_side - max_long_side * 2) / float(max_long_side)

        # 有多层时尽量不碰 level 0；若只有单层，则必须允许读取。
        level0_penalty = 2.0 if multi_level and level.index == 0 else 0.0

        score = (
            2.2 * size_penalty
            + 0.4 * center_penalty
            + 0.8 * magnification_penalty
            + 1.2 * oversize_penalty
            + level0_penalty
        )

        if score < best_score:
            best_score = score
            best_level = level

    if best_level is None:
        raise RuntimeError("层级选择失败。")
    return best_level


def choose_level_for_exact_downsample(
    level_infos: Sequence[LevelInfo],
    target_downsample: float,
) -> LevelInfo:
    """
    为固定倍数下采样选择最合适的原生金字塔层级。

    策略：
    1. 优先选择 downsample <= target_downsample 且最接近 target 的层，
       这样后续只需要继续缩小，不会出现先过度下采样再放大的情况。
    2. 如果所有原生层级都比 target 更粗，则退而求其次选择最接近的更粗层级。
    """
    if not level_infos:
        raise RuntimeError("WSI 中未找到任何可用层级。")

    candidate_levels = [level for level in level_infos if level.downsample <= target_downsample]
    if candidate_levels:
        return max(candidate_levels, key=lambda level: level.downsample)

    return min(level_infos, key=lambda level: abs(level.downsample - target_downsample))


def build_exact_downsample_thumbnail(
    reader: BaseWSIReader,
    level_infos: Sequence[LevelInfo],
    target_downsample: float,
) -> Tuple[Image.Image, LevelInfo]:
    """
    按固定下采样倍数构建缩略图。

    最终目标尺寸用 ceil(level0 / downsample) 计算，
    这样 thumbnail 像素到 level 0 的映射可以稳定使用 target_downsample。
    """
    if target_downsample <= 0:
        raise ValueError("target_downsample 必须大于 0。")

    base_level = level_infos[0]
    target_width = max(1, int(math.ceil(base_level.width / float(target_downsample))))
    target_height = max(1, int(math.ceil(base_level.height / float(target_downsample))))
    source_level = choose_level_for_exact_downsample(
        level_infos=level_infos,
        target_downsample=target_downsample,
    )

    thumbnail = reader.read_level_image(source_level.index)
    if thumbnail.size != (target_width, target_height):
        thumbnail = thumbnail.resize((target_width, target_height), RESAMPLE_LANCZOS)
    return thumbnail, source_level


def build_grid_axis_positions(length: int, cell_size: int, stride: int) -> List[int]:
    """
    生成一维方向上的网格起点列表，并确保尾部能被覆盖。
    """
    if length <= 0:
        raise ValueError("网格轴长度必须大于 0。")
    if cell_size <= 0:
        raise ValueError("网格单元边长必须大于 0。")
    if stride <= 0:
        raise ValueError("网格步长必须大于 0。")

    max_start = max(length - cell_size, 0)
    positions = [0]
    current = 0
    while current + stride <= max_start:
        current += stride
        positions.append(current)
    if positions[-1] != max_start:
        positions.append(max_start)
    return positions


def thumbnail_coord_to_level0(value: int | float, target_downsample: float) -> int:
    """
    Map a coordinate from the final thumbnail coordinate system to level 0.

    The final thumbnail coordinate system is defined only by target_downsample.
    source_level.downsample is just an intermediate read level and must not be
    used for coordinate mapping.
    """
    return int(round(float(value) * float(target_downsample)))


def build_grid_metadata(
    slide_path: Path,
    level0_width: int,
    level0_height: int,
    thumbnail_width: int,
    thumbnail_height: int,
    target_downsample: float,
    grid_cell_size_level0: int,
    grid_overlap_ratio: float,
) -> Dict[str, Any]:
    """
    根据固定 32x 缩略图生成网格元数据。
    """
    if not (0.0 <= grid_overlap_ratio < 1.0):
        raise ValueError("grid_overlap_ratio 必须落在 [0, 1) 区间。")

    grid_cell_size_thumb = max(1, int(round(grid_cell_size_level0 / float(target_downsample))))
    grid_stride_thumb = max(1, int(round(grid_cell_size_thumb * (1.0 - grid_overlap_ratio))))
    grid_stride_level0 = max(1, int(round(grid_cell_size_level0 * (1.0 - grid_overlap_ratio))))

    row_positions = build_grid_axis_positions(thumbnail_height, grid_cell_size_thumb, grid_stride_thumb)
    col_positions = build_grid_axis_positions(thumbnail_width, grid_cell_size_thumb, grid_stride_thumb)

    grid_cells: List[Dict[str, Any]] = []
    for row_id, thumb_y in enumerate(row_positions):
        for col_id, thumb_x in enumerate(col_positions):
            thumb_w = min(grid_cell_size_thumb, thumbnail_width - thumb_x)
            thumb_h = min(grid_cell_size_thumb, thumbnail_height - thumb_y)

            level0_x = thumbnail_coord_to_level0(thumb_x, target_downsample)
            level0_y = thumbnail_coord_to_level0(thumb_y, target_downsample)
            level0_w = thumbnail_coord_to_level0(thumb_w, target_downsample)
            level0_h = thumbnail_coord_to_level0(thumb_h, target_downsample)

            grid_cells.append(
                {
                    "patch_id": [row_id, col_id],
                    "row_id": row_id,
                    "col_id": col_id,
                    "thumbnail_top_left_x": int(thumb_x),
                    "thumbnail_top_left_y": int(thumb_y),
                    "thumbnail_width": int(thumb_w),
                    "thumbnail_height": int(thumb_h),
                    "level0_top_left_x": int(level0_x),
                    "level0_top_left_y": int(level0_y),
                    "level0_width": int(level0_w),
                    "level0_height": int(level0_h),
                }
            )

    return {
        "slide_id": slide_path.stem,
        "slide_path": str(slide_path),
        "thumbnail_mode": "grid32x",
        "target_downsample": target_downsample,
        "coordinate_downsample": target_downsample,
        "physical_level_0_extent": grid_cell_size_level0,
        "grid_cell_size_level0": grid_cell_size_level0,
        "grid_cell_size_thumbnail": grid_cell_size_thumb,
        "grid_overlap_ratio": grid_overlap_ratio,
        "grid_stride_level0": grid_stride_level0,
        "grid_stride_thumbnail": grid_stride_thumb,
        "slide_dimensions_level0": [level0_width, level0_height],
        "thumbnail_size": [thumbnail_width, thumbnail_height],
        "grid_rows": len(row_positions),
        "grid_cols": len(col_positions),
        "grid_cells": grid_cells,
    }


def draw_grid_overlay(
    thumbnail: Image.Image,
    grid_metadata: Dict[str, Any],
) -> Image.Image:
    """
    在缩略图上绘制网格矩形和 (row_id, col_id) 标签。
    """
    overlay = thumbnail.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)

    for cell in grid_metadata["grid_cells"]:
        x1 = int(cell["thumbnail_top_left_x"])
        y1 = int(cell["thumbnail_top_left_y"])
        x2 = x1 + int(cell["thumbnail_width"]) - 1
        y2 = y1 + int(cell["thumbnail_height"]) - 1

        if "is_selected" in cell:
            outline_color = (48, 214, 93) if bool(cell["is_selected"]) else (255, 140, 0)
            label_color = (255, 255, 255) if bool(cell["is_selected"]) else (255, 220, 180)
        else:
            outline_color = (255, 64, 64)
            label_color = (0, 255, 0)

        draw.rectangle((x1, y1, x2, y2), outline=outline_color, width=2)

        if cell.get("center_in_tissue") is not None:
            center_x = int(cell["thumbnail_center_x"])
            center_y = int(cell["thumbnail_center_y"])
            center_color = (0, 255, 255) if bool(cell["center_in_tissue"]) else (255, 0, 255)
            draw.ellipse((center_x - 2, center_y - 2, center_x + 2, center_y + 2), fill=center_color)

        draw.text((x1 + 4, y1 + 4), f"({cell['row_id']},{cell['col_id']})", fill=label_color)

    return overlay


def load_clam_segmentation(segmentation_path: Path) -> Dict[str, Any]:
    """加载 CLAM 保存的 segmentation pickle。"""
    with segmentation_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"segmentation 文件格式异常：{segmentation_path}")
    if "tissue" not in payload or "holes" not in payload:
        raise ValueError(f"segmentation 文件缺少 tissue/holes 字段：{segmentation_path}")
    return payload


def build_tissue_mask_from_patch_h5(
    patch_h5_path: Path,
    thumbnail_width: int,
    thumbnail_height: int,
    target_downsample: float,
) -> np.ndarray:
    """
    从 CLAM patch 坐标 h5 重建 thumbnail 空间中的组织 mask。

    思路：
    - 每个 patch 坐标是 level 0 左上角
    - patch_size 默认 256（level 0）
    - 将 patch 投影到 thumbnail 空间后，在对应矩形区域置为 1
    """
    if h5py is None:
        raise RuntimeError("当前环境未安装 h5py，无法读取 patch h5。")

    with h5py.File(str(patch_h5_path), "r") as handle:
        if "coords" not in handle:
            raise ValueError(f"h5 文件缺少 coords 数据集：{patch_h5_path}")
        coords = handle["coords"]
        patch_size_level0 = int(coords.attrs.get("patch_size", 256))
        coords_array = np.asarray(coords)

    mask = np.zeros((thumbnail_height, thumbnail_width), dtype=np.uint8)

    for x_level0, y_level0 in coords_array:
        x0 = int(round(float(x_level0) / float(target_downsample)))
        y0 = int(round(float(y_level0) / float(target_downsample)))
        x1 = int(round(float(x_level0 + patch_size_level0) / float(target_downsample)))
        y1 = int(round(float(y_level0 + patch_size_level0) / float(target_downsample)))

        x0 = min(max(x0, 0), thumbnail_width)
        y0 = min(max(y0, 0), thumbnail_height)
        x1 = min(max(x1, 0), thumbnail_width)
        y1 = min(max(y1, 0), thumbnail_height)

        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1

    return mask


def load_patch_h5_level0_coords(patch_h5_path: Path) -> np.ndarray:
    """读取 patch h5 中的 level 0 左上角坐标。"""
    if h5py is None:
        raise RuntimeError("当前环境未安装 h5py，无法读取 patch h5。")
    with h5py.File(str(patch_h5_path), "r") as handle:
        if "coords" not in handle:
            raise ValueError(f"h5 文件缺少 coords 数据集：{patch_h5_path}")
        coords = np.asarray(handle["coords"][:], dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords 应为 [N, 2]，实际为 {coords.shape}：{patch_h5_path}")
    return coords


def infer_positive_stride(values: np.ndarray, fallback: int) -> int:
    """从一维坐标中推断最小正步长，无法推断时使用 fallback。"""
    unique_values = np.unique(values.astype(np.int64))
    if unique_values.size < 2:
        return int(fallback)
    diffs = np.diff(np.sort(unique_values))
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return int(fallback)
    return int(np.median(positive_diffs))


def build_tissue_mask_from_mask_jpg(
    mask_jpg_path: Path,
    thumbnail_width: int,
    thumbnail_height: int,
) -> np.ndarray:
    """
    从已有的 mask jpg 构建 thumbnail 空间下的组织 mask。

    经验规则：
    - CLAM 导出的 mask 可视化图背景接近白色
    - 组织及轮廓区域明显更暗或更有颜色
    """
    mask_image = Image.open(mask_jpg_path).convert("RGB")
    if mask_image.size != (thumbnail_width, thumbnail_height):
        mask_image = mask_image.resize((thumbnail_width, thumbnail_height), RESAMPLE_LANCZOS)

    rgb = np.asarray(mask_image, dtype=np.uint8)
    gray = np.round(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
    sat = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)

    tissue_mask = ((gray < 235) | (sat > 10)).astype(np.uint8)
    return tissue_mask


def contour_to_thumbnail_points(
    contour: np.ndarray,
    target_downsample: float,
    thumbnail_width: int,
    thumbnail_height: int,
) -> List[Tuple[int, int]]:
    """将 level 0 坐标系中的 contour 映射到 thumbnail 坐标系。"""
    flat_contour = np.asarray(contour).reshape(-1, 2)
    points: List[Tuple[int, int]] = []
    for x_value, y_value in flat_contour:
        thumb_x = int(round(float(x_value) / float(target_downsample)))
        thumb_y = int(round(float(y_value) / float(target_downsample)))
        thumb_x = min(max(thumb_x, 0), max(thumbnail_width - 1, 0))
        thumb_y = min(max(thumb_y, 0), max(thumbnail_height - 1, 0))
        points.append((thumb_x, thumb_y))
    return points


def rasterize_clam_segmentation_to_thumbnail(
    segmentation_payload: Dict[str, Any],
    thumbnail_width: int,
    thumbnail_height: int,
    target_downsample: float,
) -> np.ndarray:
    """
    将 CLAM segmentation contour rasterize 到 32x thumbnail 空间。
    返回 uint8 mask，前景组织为 1，背景为 0。
    """
    mask_image = Image.new("L", (thumbnail_width, thumbnail_height), 0)
    draw = ImageDraw.Draw(mask_image)

    tissue_contours = segmentation_payload.get("tissue", [])
    hole_contours_grouped = segmentation_payload.get("holes", [])

    for contour_index, tissue_contour in enumerate(tissue_contours):
        points = contour_to_thumbnail_points(
            contour=tissue_contour,
            target_downsample=target_downsample,
            thumbnail_width=thumbnail_width,
            thumbnail_height=thumbnail_height,
        )
        if len(points) >= 3:
            draw.polygon(points, fill=1)

        if contour_index < len(hole_contours_grouped):
            for hole_contour in hole_contours_grouped[contour_index]:
                hole_points = contour_to_thumbnail_points(
                    contour=hole_contour,
                    target_downsample=target_downsample,
                    thumbnail_width=thumbnail_width,
                    thumbnail_height=thumbnail_height,
                )
                if len(hole_points) >= 3:
                    draw.polygon(hole_points, fill=0)

    return np.asarray(mask_image, dtype=np.uint8)


def compute_mask_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    从二值 tissue mask 计算最小外接框，返回 (x1, y1, x2, y2)，其中 x2/y2 为开区间。
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("组织 mask 为空，无法计算 tissue bounding box。")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_thumbnail_and_mask_to_tissue(
    thumbnail: Image.Image,
    tissue_mask: np.ndarray,
) -> TissueCropResult:
    """按 tissue bounding box 裁剪 thumbnail 和对应的 tissue mask。"""
    crop_x1, crop_y1, crop_x2, crop_y2 = compute_mask_bounding_box(tissue_mask)
    cropped_thumbnail = thumbnail.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    cropped_mask = tissue_mask[crop_y1:crop_y2, crop_x1:crop_x2]
    return TissueCropResult(
        cropped_thumbnail=cropped_thumbnail,
        cropped_mask=cropped_mask,
        crop_x=crop_x1,
        crop_y=crop_y1,
        full_thumbnail_width=thumbnail.size[0],
        full_thumbnail_height=thumbnail.size[1],
    )


def build_tissue_grid_metadata(
    slide_path: Path,
    crop_result: TissueCropResult,
    source_level: LevelInfo,
    target_downsample: float,
    grid_cell_size_thumb: int,
    grid_overlap_ratio: float,
    tissue_coverage_threshold: float,
) -> Dict[str, Any]:
    """
    基于裁剪后的 tissue thumbnail 生成组织感知网格元数据。
    """
    if grid_cell_size_thumb <= 0:
        raise ValueError("grid_cell_size_thumb 必须大于 0。")
    if not (0.0 <= grid_overlap_ratio < 1.0):
        raise ValueError("grid_overlap_ratio 必须位于 [0, 1) 区间。")
    if not (0.0 <= tissue_coverage_threshold <= 1.0):
        raise ValueError("tissue_coverage_threshold 必须位于 [0, 1] 区间。")

    cropped_width, cropped_height = crop_result.cropped_thumbnail.size
    stride_thumb = max(1, int(round(grid_cell_size_thumb * (1.0 - grid_overlap_ratio))))
    cell_size_level0 = max(1, thumbnail_coord_to_level0(grid_cell_size_thumb, target_downsample))
    stride_level0 = max(1, thumbnail_coord_to_level0(stride_thumb, target_downsample))

    row_positions = build_grid_axis_positions(
        length=cropped_height,
        cell_size=grid_cell_size_thumb,
        stride=stride_thumb,
    )
    col_positions = build_grid_axis_positions(
        length=cropped_width,
        cell_size=grid_cell_size_thumb,
        stride=stride_thumb,
    )

    degenerate_width = cropped_width < grid_cell_size_thumb
    degenerate_height = cropped_height < grid_cell_size_thumb

    low_tissue_drop_threshold = 0.05
    high_tissue_keep_threshold = 0.50

    grid_cells: List[Dict[str, Any]] = []
    n_selected = 0

    for row_id, local_y in enumerate(row_positions):
        for col_id, local_x in enumerate(col_positions):
            ideal_x = col_id * stride_thumb
            ideal_y = row_id * stride_thumb
            is_boundary_shifted = (local_x != ideal_x) or (local_y != ideal_y)

            cell_width = grid_cell_size_thumb if not degenerate_width else cropped_width
            cell_height = grid_cell_size_thumb if not degenerate_height else cropped_height

            mask_patch = crop_result.cropped_mask[local_y : local_y + cell_height, local_x : local_x + cell_width]
            tissue_coverage_ratio = float(mask_patch.mean()) if mask_patch.size > 0 else 0.0

            local_center_x = min(local_x + max(cell_width // 2, 0), max(cropped_width - 1, 0))
            local_center_y = min(local_y + max(cell_height // 2, 0), max(cropped_height - 1, 0))
            center_in_tissue = bool(crop_result.cropped_mask[local_center_y, local_center_x] > 0)

            centroid_local_x: Optional[int] = None
            centroid_local_y: Optional[int] = None
            centroid_in_tissue = False

            tissue_points = np.argwhere(mask_patch > 0)
            if tissue_points.size > 0:
                centroid_y_float = float(tissue_points[:, 0].mean())
                centroid_x_float = float(tissue_points[:, 1].mean())
                centroid_local_y = int(round(local_y + centroid_y_float))
                centroid_local_x = int(round(local_x + centroid_x_float))
                centroid_local_x = min(max(centroid_local_x, local_x), local_x + cell_width - 1)
                centroid_local_y = min(max(centroid_local_y, local_y), local_y + cell_height - 1)
                centroid_in_tissue = bool(crop_result.cropped_mask[centroid_local_y, centroid_local_x] > 0)

            anchor_local_x = local_center_x
            anchor_local_y = local_center_y
            anchor_source = "center_pt"
            selection_reason = "drop_low_tissue"
            is_selected = False

            if tissue_coverage_ratio <= low_tissue_drop_threshold:
                is_selected = False
                selection_reason = "drop_tissue_ratio_le_0.05"
            elif tissue_coverage_ratio > high_tissue_keep_threshold:
                is_selected = True
                if center_in_tissue:
                    anchor_source = "center_pt"
                    selection_reason = "keep_high_tissue_center"
                elif centroid_in_tissue and centroid_local_x is not None and centroid_local_y is not None:
                    anchor_local_x = centroid_local_x
                    anchor_local_y = centroid_local_y
                    anchor_source = "dynamic_centroid"
                    selection_reason = "keep_high_tissue_centroid_fallback"
                else:
                    # 理论上高占比区域极少出现质心也踩空，这里仍保留并退回几何中心。
                    anchor_source = "center_pt_fallback"
                    selection_reason = "keep_high_tissue_center_fallback"
            else:
                if center_in_tissue:
                    is_selected = True
                    anchor_source = "center_pt"
                    selection_reason = "keep_mid_tissue_center"
                elif centroid_in_tissue and centroid_local_x is not None and centroid_local_y is not None:
                    is_selected = True
                    anchor_local_x = centroid_local_x
                    anchor_local_y = centroid_local_y
                    anchor_source = "dynamic_centroid"
                    selection_reason = "keep_mid_tissue_centroid"
                else:
                    is_selected = False
                    selection_reason = "drop_mid_tissue_no_valid_anchor"

            if is_selected:
                n_selected += 1

            full_thumb_x = crop_result.crop_x + local_x
            full_thumb_y = crop_result.crop_y + local_y
            full_thumb_anchor_x = crop_result.crop_x + anchor_local_x
            full_thumb_anchor_y = crop_result.crop_y + anchor_local_y
            level0_x = thumbnail_coord_to_level0(full_thumb_x, target_downsample)
            level0_y = thumbnail_coord_to_level0(full_thumb_y, target_downsample)
            level0_width = thumbnail_coord_to_level0(cell_width, target_downsample)
            level0_height = thumbnail_coord_to_level0(cell_height, target_downsample)
            level0_anchor_x = thumbnail_coord_to_level0(full_thumb_anchor_x, target_downsample)
            level0_anchor_y = thumbnail_coord_to_level0(full_thumb_anchor_y, target_downsample)

            grid_cells.append(
                {
                    "patch_id": [row_id, col_id],
                    "row_id": row_id,
                    "col_id": col_id,
                    "thumbnail_top_left_x": int(local_x),
                    "thumbnail_top_left_y": int(local_y),
                    "thumbnail_center_x": int(local_center_x),
                    "thumbnail_center_y": int(local_center_y),
                    "thumbnail_centroid_x": None if centroid_local_x is None else int(centroid_local_x),
                    "thumbnail_centroid_y": None if centroid_local_y is None else int(centroid_local_y),
                    "thumbnail_anchor_x": int(anchor_local_x),
                    "thumbnail_anchor_y": int(anchor_local_y),
                    "thumbnail_width": int(cell_width),
                    "thumbnail_height": int(cell_height),
                    "full_thumbnail_top_left_x": int(full_thumb_x),
                    "full_thumbnail_top_left_y": int(full_thumb_y),
                    "full_thumbnail_anchor_x": int(full_thumb_anchor_x),
                    "full_thumbnail_anchor_y": int(full_thumb_anchor_y),
                    "level0_top_left_x": int(level0_x),
                    "level0_top_left_y": int(level0_y),
                    "level0_anchor_x": int(level0_anchor_x),
                    "level0_anchor_y": int(level0_anchor_y),
                    "level0_width": int(level0_width),
                    "level0_height": int(level0_height),
                    "center_in_tissue": center_in_tissue,
                    "centroid_in_tissue": centroid_in_tissue,
                    "tissue_coverage_ratio": tissue_coverage_ratio,
                    "is_selected": is_selected,
                    "selection_reason": selection_reason,
                    "anchor_source": anchor_source,
                    "is_boundary_shifted": is_boundary_shifted,
                    "is_degenerate_singleton": bool(degenerate_width or degenerate_height),
                }
            )

    return {
        "slide_id": slide_path.stem,
        "slide_path": str(slide_path),
        "thumbnail_mode": "tissue_grid32x_svs",
        "target_downsample": target_downsample,
        "coordinate_downsample": target_downsample,
        "physical_level_0_extent": int(cell_size_level0),
        "source_level": source_level.index,
        "source_level_downsample": source_level.downsample,
        "estimated_magnification": source_level.estimated_magnification,
        "full_thumbnail_size": [crop_result.full_thumbnail_width, crop_result.full_thumbnail_height],
        "cropped_thumbnail_size": [cropped_width, cropped_height],
        "full_thumbnail_crop_x": int(crop_result.crop_x),
        "full_thumbnail_crop_y": int(crop_result.crop_y),
        "level0_crop_x": thumbnail_coord_to_level0(crop_result.crop_x, target_downsample),
        "level0_crop_y": thumbnail_coord_to_level0(crop_result.crop_y, target_downsample),
        "full_thumbnail_crop_bbox": [
            int(crop_result.crop_x),
            int(crop_result.crop_y),
            int(crop_result.crop_x + cropped_width),
            int(crop_result.crop_y + cropped_height),
        ],
        "level0_crop_bbox": [
            thumbnail_coord_to_level0(crop_result.crop_x, target_downsample),
            thumbnail_coord_to_level0(crop_result.crop_y, target_downsample),
            thumbnail_coord_to_level0(crop_result.crop_x + cropped_width, target_downsample),
            thumbnail_coord_to_level0(crop_result.crop_y + cropped_height, target_downsample),
        ],
        "grid_cell_size_thumbnail": int(grid_cell_size_thumb),
        "grid_stride_thumbnail": int(stride_thumb),
        "grid_cell_size_level0": int(cell_size_level0),
        "grid_stride_level0": int(stride_level0),
        "grid_overlap_ratio": float(grid_overlap_ratio),
        "tissue_coverage_threshold": float(tissue_coverage_threshold),
        "low_tissue_drop_threshold": low_tissue_drop_threshold,
        "high_tissue_keep_threshold": high_tissue_keep_threshold,
        "grid_rows": len(row_positions),
        "grid_cols": len(col_positions),
        "n_grid_cells": len(grid_cells),
        "n_selected_cells": n_selected,
        "grid_cells": grid_cells,
    }


def build_patch_aligned_tissue_grid_metadata(
    slide_path: Path,
    crop_result: TissueCropResult,
    source_level: LevelInfo,
    target_downsample: float,
    patch_h5_path: Path,
    physical_level0_extent: int,
    tissue_coverage_threshold: float,
) -> Dict[str, Any]:
    """
    基于 UNI/CLAM patch h5 的 level 0 坐标生成组织感知网格元数据。

    该模式用于让后续切图和 UNI patch 对齐：level0_top_left_x/y 直接来自 h5
    coords，不能从 thumbnail 反推覆盖这些原始坐标。
    """
    if physical_level0_extent <= 0:
        raise ValueError("physical_level0_extent 必须大于 0。")
    if not (0.0 <= tissue_coverage_threshold <= 1.0):
        raise ValueError("tissue_coverage_threshold 必须位于 [0, 1] 区间。")

    coords = load_patch_h5_level0_coords(patch_h5_path)
    if coords.shape[0] == 0:
        raise ValueError(f"patch h5 中没有坐标：{patch_h5_path}")

    cropped_width, cropped_height = crop_result.cropped_thumbnail.size
    cell_size_thumb = max(1, int(round(float(physical_level0_extent) / float(target_downsample))))
    stride_level0_x = infer_positive_stride(coords[:, 0], physical_level0_extent)
    stride_level0_y = infer_positive_stride(coords[:, 1], physical_level0_extent)
    stride_level0 = int(round(float(stride_level0_x + stride_level0_y) / 2.0))
    stride_thumb = max(1, int(round(float(stride_level0) / float(target_downsample))))

    row_lookup = {value: idx for idx, value in enumerate(np.unique(coords[:, 1]))}
    col_lookup = {value: idx for idx, value in enumerate(np.unique(coords[:, 0]))}

    low_tissue_drop_threshold = 0.05
    high_tissue_keep_threshold = 0.50

    grid_cells: List[Dict[str, Any]] = []
    n_selected = 0
    for coord_index, (level0_x_raw, level0_y_raw) in enumerate(coords):
        level0_x = int(level0_x_raw)
        level0_y = int(level0_y_raw)
        full_thumb_x_float = float(level0_x) / float(target_downsample)
        full_thumb_y_float = float(level0_y) / float(target_downsample)
        full_thumb_x = int(round(full_thumb_x_float))
        full_thumb_y = int(round(full_thumb_y_float))
        local_x = full_thumb_x - int(crop_result.crop_x)
        local_y = full_thumb_y - int(crop_result.crop_y)

        mask_x0 = max(local_x, 0)
        mask_y0 = max(local_y, 0)
        mask_x1 = min(local_x + cell_size_thumb, cropped_width)
        mask_y1 = min(local_y + cell_size_thumb, cropped_height)
        mask_patch = crop_result.cropped_mask[mask_y0:mask_y1, mask_x0:mask_x1]
        tissue_pixels = float(mask_patch.sum()) if mask_patch.size > 0 else 0.0
        tissue_coverage_ratio = tissue_pixels / float(cell_size_thumb * cell_size_thumb)

        local_center_x = local_x + cell_size_thumb // 2
        local_center_y = local_y + cell_size_thumb // 2
        center_in_bounds = 0 <= local_center_x < cropped_width and 0 <= local_center_y < cropped_height
        center_in_tissue = bool(
            center_in_bounds and crop_result.cropped_mask[local_center_y, local_center_x] > 0
        )

        centroid_local_x: Optional[int] = None
        centroid_local_y: Optional[int] = None
        centroid_in_tissue = False
        if mask_patch.size > 0:
            tissue_points = np.argwhere(mask_patch > 0)
            if tissue_points.size > 0:
                centroid_y_float = float(tissue_points[:, 0].mean())
                centroid_x_float = float(tissue_points[:, 1].mean())
                centroid_local_y = int(round(mask_y0 + centroid_y_float))
                centroid_local_x = int(round(mask_x0 + centroid_x_float))
                centroid_in_tissue = bool(crop_result.cropped_mask[centroid_local_y, centroid_local_x] > 0)

        anchor_local_x = local_center_x
        anchor_local_y = local_center_y
        anchor_source = "center_pt"
        selection_reason = "drop_low_tissue"
        is_selected = False

        if tissue_coverage_ratio <= low_tissue_drop_threshold:
            is_selected = False
            selection_reason = "drop_tissue_ratio_le_0.05"
        elif tissue_coverage_ratio > high_tissue_keep_threshold:
            is_selected = True
            if center_in_tissue:
                anchor_source = "center_pt"
                selection_reason = "keep_high_tissue_center"
            elif centroid_in_tissue and centroid_local_x is not None and centroid_local_y is not None:
                anchor_local_x = centroid_local_x
                anchor_local_y = centroid_local_y
                anchor_source = "dynamic_centroid"
                selection_reason = "keep_high_tissue_centroid_fallback"
            else:
                anchor_source = "center_pt_fallback"
                selection_reason = "keep_high_tissue_center_fallback"
        else:
            if center_in_tissue:
                is_selected = True
                anchor_source = "center_pt"
                selection_reason = "keep_mid_tissue_center"
            elif centroid_in_tissue and centroid_local_x is not None and centroid_local_y is not None:
                is_selected = True
                anchor_local_x = centroid_local_x
                anchor_local_y = centroid_local_y
                anchor_source = "dynamic_centroid"
                selection_reason = "keep_mid_tissue_centroid"
            else:
                is_selected = False
                selection_reason = "drop_mid_tissue_no_valid_anchor"

        if is_selected:
            n_selected += 1

        full_thumb_anchor_x = int(round(float(level0_x + physical_level0_extent / 2.0) / float(target_downsample)))
        full_thumb_anchor_y = int(round(float(level0_y + physical_level0_extent / 2.0) / float(target_downsample)))
        level0_anchor_x = level0_x + physical_level0_extent // 2
        level0_anchor_y = level0_y + physical_level0_extent // 2
        row_id = int(row_lookup[int(level0_y_raw)])
        col_id = int(col_lookup[int(level0_x_raw)])

        grid_cells.append(
            {
                "patch_id": [row_id, col_id],
                "uni_coord_index": int(coord_index),
                "row_id": row_id,
                "col_id": col_id,
                "thumbnail_top_left_x": int(local_x),
                "thumbnail_top_left_y": int(local_y),
                "thumbnail_center_x": int(local_center_x),
                "thumbnail_center_y": int(local_center_y),
                "thumbnail_centroid_x": None if centroid_local_x is None else int(centroid_local_x),
                "thumbnail_centroid_y": None if centroid_local_y is None else int(centroid_local_y),
                "thumbnail_anchor_x": int(anchor_local_x),
                "thumbnail_anchor_y": int(anchor_local_y),
                "thumbnail_width": int(cell_size_thumb),
                "thumbnail_height": int(cell_size_thumb),
                "full_thumbnail_top_left_x": int(full_thumb_x),
                "full_thumbnail_top_left_y": int(full_thumb_y),
                "full_thumbnail_top_left_x_float": float(full_thumb_x_float),
                "full_thumbnail_top_left_y_float": float(full_thumb_y_float),
                "full_thumbnail_anchor_x": int(full_thumb_anchor_x),
                "full_thumbnail_anchor_y": int(full_thumb_anchor_y),
                "level0_top_left_x": int(level0_x),
                "level0_top_left_y": int(level0_y),
                "level0_anchor_x": int(level0_anchor_x),
                "level0_anchor_y": int(level0_anchor_y),
                "level0_width": int(physical_level0_extent),
                "level0_height": int(physical_level0_extent),
                "center_in_tissue": center_in_tissue,
                "centroid_in_tissue": centroid_in_tissue,
                "tissue_coverage_ratio": tissue_coverage_ratio,
                "is_selected": is_selected,
                "selection_reason": selection_reason,
                "anchor_source": anchor_source,
                "is_boundary_shifted": False,
                "is_degenerate_singleton": False,
                "grid_alignment_source": "patch_h5_level0_coords",
            }
        )

    return {
        "slide_id": slide_path.stem,
        "slide_path": str(slide_path),
        "thumbnail_mode": "tissue_grid32x_svs",
        "grid_alignment": "patch_h5_level0_coords",
        "grid_alignment_patch_h5_path": str(patch_h5_path),
        "target_downsample": target_downsample,
        "coordinate_downsample": target_downsample,
        "physical_level_0_extent": int(physical_level0_extent),
        "source_level": source_level.index,
        "source_level_downsample": source_level.downsample,
        "estimated_magnification": source_level.estimated_magnification,
        "full_thumbnail_size": [crop_result.full_thumbnail_width, crop_result.full_thumbnail_height],
        "cropped_thumbnail_size": [cropped_width, cropped_height],
        "full_thumbnail_crop_x": int(crop_result.crop_x),
        "full_thumbnail_crop_y": int(crop_result.crop_y),
        "level0_crop_x": thumbnail_coord_to_level0(crop_result.crop_x, target_downsample),
        "level0_crop_y": thumbnail_coord_to_level0(crop_result.crop_y, target_downsample),
        "full_thumbnail_crop_bbox": [
            int(crop_result.crop_x),
            int(crop_result.crop_y),
            int(crop_result.crop_x + cropped_width),
            int(crop_result.crop_y + cropped_height),
        ],
        "level0_crop_bbox": [
            thumbnail_coord_to_level0(crop_result.crop_x, target_downsample),
            thumbnail_coord_to_level0(crop_result.crop_y, target_downsample),
            thumbnail_coord_to_level0(crop_result.crop_x + cropped_width, target_downsample),
            thumbnail_coord_to_level0(crop_result.crop_y + cropped_height, target_downsample),
        ],
        "grid_cell_size_thumbnail": int(cell_size_thumb),
        "grid_stride_thumbnail": int(stride_thumb),
        "grid_cell_size_level0": int(physical_level0_extent),
        "grid_stride_level0": int(stride_level0),
        "grid_overlap_ratio": None,
        "tissue_coverage_threshold": float(tissue_coverage_threshold),
        "low_tissue_drop_threshold": low_tissue_drop_threshold,
        "high_tissue_keep_threshold": high_tissue_keep_threshold,
        "grid_rows": len(row_lookup),
        "grid_cols": len(col_lookup),
        "n_grid_cells": len(grid_cells),
        "n_selected_cells": n_selected,
        "grid_cells": grid_cells,
    }


def resize_to_valid_long_side(
    image: Image.Image,
    min_long_side: int,
    max_long_side: int,
) -> Image.Image:
    """
    将最终缩略图长边控制在指定范围内。
    如果当前长边已经在范围内，则保持原尺寸，避免无意义重采样。
    """
    width, height = image.size
    current_long_side = max(width, height)

    if min_long_side <= current_long_side <= max_long_side:
        return image

    target_long_side = min_long_side if current_long_side < min_long_side else max_long_side
    scale = target_long_side / float(current_long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), RESAMPLE_LANCZOS)


def compute_otsu_threshold(gray_image: np.ndarray) -> int:
    """
    使用 NumPy 实现 Otsu 阈值，避免额外依赖 skimage/opencv。
    gray_image 必须是 uint8 的二维灰度图。
    """
    if gray_image.dtype != np.uint8:
        raise ValueError("Otsu 输入必须是 uint8 灰度图。")

    histogram = np.bincount(gray_image.reshape(-1), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 255

    probability = histogram / total
    cumulative_probability = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * np.arange(256))
    global_mean = cumulative_mean[-1]

    denominator = cumulative_probability * (1.0 - cumulative_probability)
    valid = denominator > 0

    between_class_variance = np.zeros(256, dtype=np.float64)
    between_class_variance[valid] = (
        (global_mean * cumulative_probability[valid] - cumulative_mean[valid]) ** 2
        / denominator[valid]
    )
    return int(np.argmax(between_class_variance))


def analyze_white_background(
    image: Image.Image,
    white_threshold: int,
) -> Tuple[int, Tuple[int, int, int]]:
    """
    背景统计逻辑：
    1. 先找出明显接近纯白的像素（RGB > 240）
    2. 计算候选背景像素的中值颜色，增强对白底的鲁棒性
    3. 基于灰度图计算 Otsu 阈值

    说明：
    - 当前 TraceAgent thumbnail grid 产物需要保留真实缩略图背景
    - 返回值只用于调试或日志记录，不再改写任何图像像素
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("输入图像无法转换为 RGB。")

    gray = np.round(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(
        np.uint8
    )
    otsu_threshold = compute_otsu_threshold(gray)

    white_mask = np.all(rgb > white_threshold, axis=2)

    # 候选背景不足时，退回到全图中值，避免空集合报错。
    if np.any(white_mask):
        background_pixels = rgb[white_mask]
    else:
        bright_mask = gray >= max(white_threshold, 245)
        background_pixels = rgb[bright_mask] if np.any(bright_mask) else rgb.reshape(-1, 3)

    median_background_rgb = tuple(
        int(v) for v in np.median(background_pixels, axis=0).round().astype(np.uint8).tolist()
    )

    return otsu_threshold, median_background_rgb


def normalize_array_to_rgb(array: np.ndarray) -> np.ndarray:
    """
    将 tifffile 读出的数组规范化为 uint8 RGB。
    兼容灰度图、RGBA、channels-first 等常见情况。
    """
    array = np.asarray(array)

    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            pass
        elif array.shape[0] in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        else:
            raise ValueError(f"不支持的 TIFF 图像形状：{array.shape}")
    else:
        raise ValueError(f"不支持的 TIFF 图像维度：{array.shape}")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] >= 4:
        array = array[:, :, :3]

    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.integer):
        dtype_info = np.iinfo(array.dtype)
        if dtype_info.max == 0:
            return np.zeros_like(array, dtype=np.uint8)
        array = array.astype(np.float32) / float(dtype_info.max)
        array = np.clip(array * 255.0, 0, 255)
        return array.astype(np.uint8)

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size > 0 else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255)
        return array.astype(np.uint8)

    raise ValueError(f"不支持的 TIFF 数据类型：{array.dtype}")


def level_shape_to_wh(shape: Sequence[int]) -> Tuple[int, int]:
    """从 TIFF level 的 shape 中解析宽高。"""
    if len(shape) == 2:
        height, width = shape
        return int(width), int(height)

    if len(shape) == 3:
        if shape[-1] in (1, 3, 4):
            height, width = shape[0], shape[1]
            return int(width), int(height)
        if shape[0] in (1, 3, 4):
            height, width = shape[1], shape[2]
            return int(width), int(height)

    raise ValueError(f"无法从 shape={shape} 解析图像宽高。")


def parse_float(value: object) -> Optional[float]:
    """尽量把任意倍率字段转换为 float。"""
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def build_output_dir(
    input_root: Path,
    output_root: Path,
    slide_path: Path,
) -> Path:
    """
    如果开启递归处理，则在输出目录中保留源目录层次，避免重名覆盖。
    """
    relative_parent = slide_path.parent.relative_to(input_root)
    output_dir = output_root / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_auto_output_artifacts(
    input_root: Path,
    output_root: Path,
    slide_path: Path,
    selected_level: int,
) -> OutputArtifacts:
    """auto 模式下只输出一张缩略图。"""
    output_dir = build_output_dir(input_root=input_root, output_root=output_root, slide_path=slide_path)
    return OutputArtifacts(
        thumbnail_path=output_dir / f"{slide_path.stem}_thumbnail_level{selected_level}.jpg",
        grid_overlay_path=None,
        metadata_path=None,
    )


def build_grid32x_output_artifacts(
    input_root: Path,
    output_root: Path,
    slide_path: Path,
    source_level: int,
    target_downsample: float,
) -> OutputArtifacts:
    """
    grid32x 模式下输出：
    1. 原始缩略图
    2. 带网格线的可视化图
    3. 网格元数据 JSON
    """
    output_dir = build_output_dir(input_root=input_root, output_root=output_root, slide_path=slide_path)
    downsample_label = str(int(target_downsample)) if float(target_downsample).is_integer() else str(target_downsample)
    prefix = f"{slide_path.stem}_thumbnail_ds{downsample_label}_level{source_level}"
    return OutputArtifacts(
        thumbnail_path=output_dir / f"{prefix}.jpg",
        grid_overlay_path=output_dir / f"{prefix}_grid.jpg",
        metadata_path=output_dir / f"{prefix}_grid.json",
    )


def build_tissue_grid_output_artifacts(
    input_root: Path,
    output_root: Path,
    slide_path: Path,
    source_level: int,
    target_downsample: float,
    grid_cell_size_thumb: int,
) -> OutputArtifacts:
    """tissue_grid32x_svs 模式下的输出命名。"""
    output_dir = build_output_dir(input_root=input_root, output_root=output_root, slide_path=slide_path)
    downsample_label = str(int(target_downsample)) if float(target_downsample).is_integer() else str(target_downsample)
    prefix = (
        f"{slide_path.stem}_tissuegrid{grid_cell_size_thumb}_"
        f"ds{downsample_label}_level{source_level}"
    )
    return OutputArtifacts(
        thumbnail_path=output_dir / f"{prefix}.jpg",
        grid_overlay_path=output_dir / f"{prefix}_grid.jpg",
        metadata_path=output_dir / f"{prefix}_grid.json",
    )


def process_single_wsi(
    slide_path: Path,
    input_root: Path,
    output_root: Path,
    mode: str,
    segmentations_dir: Optional[Path],
    patch_h5_dir: Optional[Path],
    mask_jpg_dir: Optional[Path],
    min_long_side: int,
    max_long_side: int,
    preferred_magnification: float,
    target_downsample: float,
    grid_cell_size_level0: int,
    grid_cell_size_thumb: int,
    grid_physical_level0_extent: int,
    grid_align_to_patch_h5: bool,
    grid_overlap_ratio: float,
    tissue_coverage_threshold: float,
    white_threshold: int,
    no_background_filter: bool,
    overwrite: bool,
    verbose: bool = True,
) -> Optional[Path]:
    """处理单张 WSI，成功时返回输出路径。"""
    with open_wsi_reader(slide_path) as reader:
        level_infos = reader.get_level_infos()
        if mode == "auto":
            selected_level = choose_best_level(
                level_infos=level_infos,
                min_long_side=min_long_side,
                max_long_side=max_long_side,
                preferred_magnification=preferred_magnification,
            )

            artifacts = build_auto_output_artifacts(
                input_root=input_root,
                output_root=output_root,
                slide_path=slide_path,
                selected_level=selected_level.index,
            )
            if artifacts.thumbnail_path.exists() and not overwrite:
                return artifacts.thumbnail_path

            thumbnail = reader.read_level_image(selected_level.index)
            thumbnail = resize_to_valid_long_side(
                thumbnail,
                min_long_side=min_long_side,
                max_long_side=max_long_side,
            )

            otsu_threshold: Optional[int] = None
            median_background_rgb: Optional[Tuple[int, int, int]] = None
            if not no_background_filter:
                otsu_threshold, median_background_rgb = analyze_white_background(
                    thumbnail,
                    white_threshold=white_threshold,
                )

            thumbnail.save(
                artifacts.thumbnail_path,
                format="JPEG",
                quality=90,
                optimize=True,
                subsampling=0,
            )

            if verbose:
                tqdm.write(
                    f"[OK] {slide_path.name} | mode=auto | backend={reader.backend_name} | "
                    f"level={selected_level.index} | downsample={selected_level.downsample} | "
                    f"size={thumbnail.size} | est_mag={selected_level.estimated_magnification} | "
                    f"otsu={otsu_threshold} | median_bg={median_background_rgb} -> {artifacts.thumbnail_path}"
                )
            return artifacts.thumbnail_path

        if mode not in {"grid32x", "tissue_grid32x_svs", "tissue_grid32x_isyntax"}:
            raise ValueError(f"不支持的 mode: {mode}")

        thumbnail, source_level = build_exact_downsample_thumbnail(
            reader=reader,
            level_infos=level_infos,
            target_downsample=target_downsample,
        )

        if mode == "tissue_grid32x_svs":
            if slide_path.suffix.lower() != ".svs":
                raise ValueError("tissue_grid32x_svs 模式当前仅支持 .svs 文件。")
            if segmentations_dir is None:
                raise ValueError("tissue_grid32x_svs 模式必须提供 --segmentations-dir。")

            segmentation_path = segmentations_dir / f"{slide_path.stem}.pkl"
            if not segmentation_path.exists():
                raise FileNotFoundError(f"未找到 segmentation 文件：{segmentation_path}")

            segmentation_payload = load_clam_segmentation(segmentation_path)
            tissue_mask_full = rasterize_clam_segmentation_to_thumbnail(
                segmentation_payload=segmentation_payload,
                thumbnail_width=thumbnail.size[0],
                thumbnail_height=thumbnail.size[1],
                target_downsample=target_downsample,
            )
            crop_result = crop_thumbnail_and_mask_to_tissue(
                thumbnail=thumbnail,
                tissue_mask=tissue_mask_full,
            )

            cropped_thumbnail = crop_result.cropped_thumbnail
            otsu_threshold: Optional[int] = None
            median_background_rgb: Optional[Tuple[int, int, int]] = None
            if not no_background_filter:
                otsu_threshold, median_background_rgb = analyze_white_background(
                    cropped_thumbnail,
                    white_threshold=white_threshold,
                )

            effective_grid_cell_size_thumb = (
                max(1, int(round(float(grid_physical_level0_extent) / float(target_downsample))))
                if grid_align_to_patch_h5 and int(grid_physical_level0_extent) > 0
                else grid_cell_size_thumb
            )
            artifacts = build_tissue_grid_output_artifacts(
                input_root=input_root,
                output_root=output_root,
                slide_path=slide_path,
                source_level=source_level.index,
                target_downsample=target_downsample,
                grid_cell_size_thumb=effective_grid_cell_size_thumb,
            )
            expected_paths = [
                artifacts.thumbnail_path,
                artifacts.grid_overlay_path,
                artifacts.metadata_path,
            ]
            if not overwrite and all(path is not None and path.exists() for path in expected_paths):
                return artifacts.thumbnail_path

            cropped_thumbnail.save(
                artifacts.thumbnail_path,
                format="JPEG",
                quality=90,
                optimize=True,
                subsampling=0,
            )

            align_patch_h5_path = None if patch_h5_dir is None else patch_h5_dir / f"{slide_path.stem}.h5"
            if grid_align_to_patch_h5:
                if align_patch_h5_path is None or not align_patch_h5_path.exists():
                    raise FileNotFoundError(f"启用 h5 对齐但未找到 patch h5：{align_patch_h5_path}")
                physical_extent = (
                    int(grid_physical_level0_extent)
                    if int(grid_physical_level0_extent) > 0
                    else thumbnail_coord_to_level0(grid_cell_size_thumb, target_downsample)
                )
                grid_metadata = build_patch_aligned_tissue_grid_metadata(
                    slide_path=slide_path,
                    crop_result=crop_result,
                    source_level=source_level,
                    target_downsample=target_downsample,
                    patch_h5_path=align_patch_h5_path,
                    physical_level0_extent=physical_extent,
                    tissue_coverage_threshold=tissue_coverage_threshold,
                )
            else:
                grid_metadata = build_tissue_grid_metadata(
                    slide_path=slide_path,
                    crop_result=crop_result,
                    source_level=source_level,
                    target_downsample=target_downsample,
                    grid_cell_size_thumb=grid_cell_size_thumb,
                    grid_overlap_ratio=grid_overlap_ratio,
                    tissue_coverage_threshold=tissue_coverage_threshold,
                )
            grid_metadata["backend"] = reader.backend_name
            grid_metadata["background_filter_applied"] = False
            grid_metadata["otsu_threshold"] = otsu_threshold
            grid_metadata["median_background_rgb"] = median_background_rgb
            grid_metadata["segmentation_path"] = str(segmentation_path)
            grid_metadata["slide_dimensions_level0"] = [level_infos[0].width, level_infos[0].height]

            overlay = draw_grid_overlay(thumbnail=cropped_thumbnail, grid_metadata=grid_metadata)
            if artifacts.grid_overlay_path is None or artifacts.metadata_path is None:
                raise RuntimeError("tissue_grid32x_svs 模式缺少 overlay 或 metadata 输出路径。")

            overlay.save(
                artifacts.grid_overlay_path,
                format="JPEG",
                quality=90,
                optimize=True,
                subsampling=0,
            )
            artifacts.metadata_path.write_text(
                json.dumps(grid_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if verbose:
                tqdm.write(
                    f"[OK] {slide_path.name} | mode=tissue_grid32x_svs | backend={reader.backend_name} | "
                    f"source_level={source_level.index} | crop_size={cropped_thumbnail.size} | "
                    f"grid={grid_metadata['grid_rows']}x{grid_metadata['grid_cols']} | "
                    f"selected={grid_metadata['n_selected_cells']}/{grid_metadata['n_grid_cells']} -> {artifacts.thumbnail_path}"
                )
            return artifacts.thumbnail_path

        if mode == "tissue_grid32x_isyntax":
            if slide_path.suffix.lower() != ".isyntax":
                raise ValueError("tissue_grid32x_isyntax 模式当前仅支持 .isyntax 文件。")
            segmentation_path = None if segmentations_dir is None else segmentations_dir / f"{slide_path.stem}.pkl"
            patch_h5_path = None if patch_h5_dir is None else patch_h5_dir / f"{slide_path.stem}.h5"
            mask_jpg_path = None if mask_jpg_dir is None else mask_jpg_dir / f"{slide_path.stem}.jpg"

            # iSyntax 场景下优先级：
            # 1. CLAM segmentation pkl
            # 2. 现成的 mask jpg
            # 3. patch h5 兜底
            if segmentation_path is not None and segmentation_path.exists():
                tissue_mask_source = "clam_segmentation_pkl"
                segmentation_payload = load_clam_segmentation(segmentation_path)
                tissue_mask_full = rasterize_clam_segmentation_to_thumbnail(
                    segmentation_payload=segmentation_payload,
                    thumbnail_width=thumbnail.size[0],
                    thumbnail_height=thumbnail.size[1],
                    target_downsample=target_downsample,
                )
            elif mask_jpg_path is not None and mask_jpg_path.exists():
                tissue_mask_source = "mask_jpg"
                tissue_mask_full = build_tissue_mask_from_mask_jpg(
                    mask_jpg_path=mask_jpg_path,
                    thumbnail_width=thumbnail.size[0],
                    thumbnail_height=thumbnail.size[1],
                )
            elif patch_h5_path is not None and patch_h5_path.exists():
                tissue_mask_source = "patch_h5"
                tissue_mask_full = build_tissue_mask_from_patch_h5(
                    patch_h5_path=patch_h5_path,
                    thumbnail_width=thumbnail.size[0],
                    thumbnail_height=thumbnail.size[1],
                    target_downsample=target_downsample,
                )
            else:
                raise FileNotFoundError(
                    f"未找到可用的组织来源：segmentation={segmentation_path}, mask_jpg={mask_jpg_path}, patch_h5={patch_h5_path}"
                )

            crop_result = crop_thumbnail_and_mask_to_tissue(
                thumbnail=thumbnail,
                tissue_mask=tissue_mask_full,
            )

            cropped_thumbnail = crop_result.cropped_thumbnail
            otsu_threshold: Optional[int] = None
            median_background_rgb: Optional[Tuple[int, int, int]] = None
            if not no_background_filter:
                otsu_threshold, median_background_rgb = analyze_white_background(
                    cropped_thumbnail,
                    white_threshold=white_threshold,
                )

            effective_grid_cell_size_thumb = (
                max(1, int(round(float(grid_physical_level0_extent) / float(target_downsample))))
                if grid_align_to_patch_h5 and int(grid_physical_level0_extent) > 0
                else grid_cell_size_thumb
            )
            artifacts = build_tissue_grid_output_artifacts(
                input_root=input_root,
                output_root=output_root,
                slide_path=slide_path,
                source_level=source_level.index,
                target_downsample=target_downsample,
                grid_cell_size_thumb=effective_grid_cell_size_thumb,
            )
            expected_paths = [
                artifacts.thumbnail_path,
                artifacts.grid_overlay_path,
                artifacts.metadata_path,
            ]
            if not overwrite and all(path is not None and path.exists() for path in expected_paths):
                return artifacts.thumbnail_path

            cropped_thumbnail.save(
                artifacts.thumbnail_path,
                format="JPEG",
                quality=90,
                optimize=True,
                subsampling=0,
            )

            if grid_align_to_patch_h5:
                if patch_h5_path is None or not patch_h5_path.exists():
                    raise FileNotFoundError(f"启用 h5 对齐但未找到 patch h5：{patch_h5_path}")
                physical_extent = (
                    int(grid_physical_level0_extent)
                    if int(grid_physical_level0_extent) > 0
                    else thumbnail_coord_to_level0(grid_cell_size_thumb, target_downsample)
                )
                grid_metadata = build_patch_aligned_tissue_grid_metadata(
                    slide_path=slide_path,
                    crop_result=crop_result,
                    source_level=source_level,
                    target_downsample=target_downsample,
                    patch_h5_path=patch_h5_path,
                    physical_level0_extent=physical_extent,
                    tissue_coverage_threshold=tissue_coverage_threshold,
                )
            else:
                grid_metadata = build_tissue_grid_metadata(
                    slide_path=slide_path,
                    crop_result=crop_result,
                    source_level=source_level,
                    target_downsample=target_downsample,
                    grid_cell_size_thumb=grid_cell_size_thumb,
                    grid_overlap_ratio=grid_overlap_ratio,
                    tissue_coverage_threshold=tissue_coverage_threshold,
                )
            grid_metadata["thumbnail_mode"] = "tissue_grid32x_isyntax"
            grid_metadata["backend"] = reader.backend_name
            grid_metadata["background_filter_applied"] = False
            grid_metadata["otsu_threshold"] = otsu_threshold
            grid_metadata["median_background_rgb"] = median_background_rgb
            grid_metadata["segmentation_path"] = None if segmentation_path is None else str(segmentation_path)
            grid_metadata["patch_h5_path"] = None if patch_h5_path is None else str(patch_h5_path)
            grid_metadata["mask_jpg_path"] = None if mask_jpg_path is None else str(mask_jpg_path)
            grid_metadata["tissue_mask_source"] = tissue_mask_source
            grid_metadata["stain_filter_applied"] = False
            grid_metadata["slide_dimensions_level0"] = [level_infos[0].width, level_infos[0].height]

            overlay = draw_grid_overlay(thumbnail=cropped_thumbnail, grid_metadata=grid_metadata)
            if artifacts.grid_overlay_path is None or artifacts.metadata_path is None:
                raise RuntimeError("tissue_grid32x_isyntax 模式缺少 overlay 或 metadata 输出路径。")

            overlay.save(
                artifacts.grid_overlay_path,
                format="JPEG",
                quality=90,
                optimize=True,
                subsampling=0,
            )
            artifacts.metadata_path.write_text(
                json.dumps(grid_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if verbose:
                tqdm.write(
                    f"[OK] {slide_path.name} | mode=tissue_grid32x_isyntax | backend={reader.backend_name} | "
                    f"source_level={source_level.index} | crop_size={cropped_thumbnail.size} | "
                    f"grid={grid_metadata['grid_rows']}x{grid_metadata['grid_cols']} | "
                    f"selected={grid_metadata['n_selected_cells']}/{grid_metadata['n_grid_cells']} -> {artifacts.thumbnail_path}"
                )
            return artifacts.thumbnail_path

        otsu_threshold: Optional[int] = None
        median_background_rgb: Optional[Tuple[int, int, int]] = None
        if not no_background_filter:
            otsu_threshold, median_background_rgb = analyze_white_background(
                thumbnail,
                white_threshold=white_threshold,
            )

        artifacts = build_grid32x_output_artifacts(
            input_root=input_root,
            output_root=output_root,
            slide_path=slide_path,
            source_level=source_level.index,
            target_downsample=target_downsample,
        )
        expected_paths = [
            artifacts.thumbnail_path,
            artifacts.grid_overlay_path,
            artifacts.metadata_path,
        ]
        if not overwrite and all(path is not None and path.exists() for path in expected_paths):
            return artifacts.thumbnail_path

        thumbnail.save(
            artifacts.thumbnail_path,
            format="JPEG",
            quality=90,
            optimize=True,
            subsampling=0,
        )

        grid_metadata = build_grid_metadata(
            slide_path=slide_path,
            level0_width=level_infos[0].width,
            level0_height=level_infos[0].height,
            thumbnail_width=thumbnail.size[0],
            thumbnail_height=thumbnail.size[1],
            target_downsample=target_downsample,
            grid_cell_size_level0=grid_cell_size_level0,
            grid_overlap_ratio=grid_overlap_ratio,
        )
        grid_metadata["source_level"] = source_level.index
        grid_metadata["source_level_downsample"] = source_level.downsample
        grid_metadata["estimated_magnification"] = source_level.estimated_magnification
        grid_metadata["backend"] = reader.backend_name
        grid_metadata["background_filter_applied"] = False
        grid_metadata["otsu_threshold"] = otsu_threshold
        grid_metadata["median_background_rgb"] = median_background_rgb

        overlay = draw_grid_overlay(thumbnail=thumbnail, grid_metadata=grid_metadata)
        if artifacts.grid_overlay_path is None or artifacts.metadata_path is None:
            raise RuntimeError("grid32x 模式缺少 overlay 或 metadata 输出路径。")

        overlay.save(
            artifacts.grid_overlay_path,
            format="JPEG",
            quality=90,
            optimize=True,
            subsampling=0,
        )
        artifacts.metadata_path.write_text(
            json.dumps(grid_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if verbose:
            tqdm.write(
                f"[OK] {slide_path.name} | mode=grid32x | backend={reader.backend_name} | "
                f"source_level={source_level.index} | source_downsample={source_level.downsample} | "
                f"target_downsample={target_downsample} | thumb_size={thumbnail.size} | "
                f"grid={grid_metadata['grid_rows']}x{grid_metadata['grid_cols']} -> {artifacts.thumbnail_path}"
            )
        return artifacts.thumbnail_path


def process_single_wsi_safe(task: Dict[str, object]) -> Dict[str, object]:
    """
    multiprocessing 包装函数。
    统一捕获异常，避免单张坏片导致整个批处理进程池中断。
    """
    slide_path = Path(str(task["slide_path"]))
    try:
        result = process_single_wsi(
            slide_path=slide_path,
            input_root=Path(str(task["input_root"])),
            output_root=Path(str(task["output_root"])),
            mode=str(task["mode"]),
            segmentations_dir=None
            if task.get("segmentations_dir") in (None, "")
            else Path(str(task["segmentations_dir"])),
            patch_h5_dir=None
            if task.get("patch_h5_dir") in (None, "")
            else Path(str(task["patch_h5_dir"])),
            mask_jpg_dir=None
            if task.get("mask_jpg_dir") in (None, "")
            else Path(str(task["mask_jpg_dir"])),
            min_long_side=int(task["min_long_side"]),
            max_long_side=int(task["max_long_side"]),
            preferred_magnification=float(task["preferred_magnification"]),
            target_downsample=float(task["target_downsample"]),
            grid_cell_size_level0=int(task["grid_cell_size_level0"]),
            grid_cell_size_thumb=int(task["grid_cell_size_thumb"]),
            grid_physical_level0_extent=int(task.get("grid_physical_level0_extent", 0)),
            grid_align_to_patch_h5=bool(task.get("grid_align_to_patch_h5", False)),
            grid_overlap_ratio=float(task["grid_overlap_ratio"]),
            tissue_coverage_threshold=float(task["tissue_coverage_threshold"]),
            white_threshold=int(task["white_threshold"]),
            no_background_filter=bool(task["no_background_filter"]),
            overwrite=bool(task["overwrite"]),
            verbose=False,
        )
        return {
            "ok": True,
            "slide_path": str(slide_path),
            "output_path": None if result is None else str(result),
        }
    except Exception as exc:
        return {
            "ok": False,
            "slide_path": str(slide_path),
            "error": str(exc),
        }


def main() -> int:
    args = parse_arguments()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (input_dir / "thumbnails").resolve()
    )

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在或不是文件夹：{input_dir}", file=sys.stderr)
        return 1

    segmentations_dir = None if args.segmentations_dir is None else args.segmentations_dir.expanduser().resolve()
    patch_h5_dir = None if args.patch_h5_dir is None else args.patch_h5_dir.expanduser().resolve()
    mask_jpg_dir = None if args.mask_jpg_dir is None else args.mask_jpg_dir.expanduser().resolve()

    if args.mode == "auto":
        if args.min_long_side <= 0 or args.max_long_side <= 0:
            print("[ERROR] 缩略图长边范围必须是正整数。", file=sys.stderr)
            return 1
        if args.min_long_side > args.max_long_side:
            print("[ERROR] --min-long-side 不能大于 --max-long-side。", file=sys.stderr)
            return 1
    elif args.mode == "grid32x":
        if args.target_downsample <= 0:
            print("[ERROR] --target-downsample 必须大于 0。", file=sys.stderr)
            return 1
        if args.grid_cell_size_level0 <= 0:
            print("[ERROR] --grid-cell-size-level0 必须大于 0。", file=sys.stderr)
            return 1
        if not (0.0 <= args.grid_overlap_ratio < 1.0):
            print("[ERROR] --grid-overlap-ratio 必须位于 [0, 1) 区间。", file=sys.stderr)
            return 1
    elif args.mode == "tissue_grid32x_svs":
        if args.target_downsample <= 0:
            print("[ERROR] --target-downsample 必须大于 0。", file=sys.stderr)
            return 1
        if args.grid_cell_size_thumb <= 0:
            print("[ERROR] --grid-cell-size-thumb 必须大于 0。", file=sys.stderr)
            return 1
        if not (0.0 <= args.grid_overlap_ratio < 1.0):
            print("[ERROR] --grid-overlap-ratio 必须位于 [0, 1) 区间。", file=sys.stderr)
            return 1
        if not (0.0 <= args.tissue_coverage_threshold <= 1.0):
            print("[ERROR] --tissue-coverage-threshold 必须位于 [0, 1] 区间。", file=sys.stderr)
            return 1
        if segmentations_dir is None:
            print("[ERROR] tissue_grid32x_svs 模式必须提供 --segmentations-dir。", file=sys.stderr)
            return 1
        if not segmentations_dir.exists() or not segmentations_dir.is_dir():
            print(f"[ERROR] segmentation 目录不存在或不可用：{segmentations_dir}", file=sys.stderr)
            return 1
        if args.grid_physical_level0_extent < 0:
            print("[ERROR] --grid-physical-level0-extent 不能为负数。", file=sys.stderr)
            return 1
        if args.grid_align_to_patch_h5:
            if patch_h5_dir is None:
                print("[ERROR] --grid-align-to-patch-h5 需要同时提供 --patch-h5-dir。", file=sys.stderr)
                return 1
            if not patch_h5_dir.exists() or not patch_h5_dir.is_dir():
                print(f"[ERROR] patch h5 目录不存在或不可用：{patch_h5_dir}", file=sys.stderr)
                return 1
    else:
        if args.target_downsample <= 0:
            print("[ERROR] --target-downsample 必须大于 0。", file=sys.stderr)
            return 1
        if args.grid_cell_size_thumb <= 0:
            print("[ERROR] --grid-cell-size-thumb 必须大于 0。", file=sys.stderr)
            return 1
        if not (0.0 <= args.grid_overlap_ratio < 1.0):
            print("[ERROR] --grid-overlap-ratio 必须位于 [0, 1) 区间。", file=sys.stderr)
            return 1
        if not (0.0 <= args.tissue_coverage_threshold <= 1.0):
            print("[ERROR] --tissue-coverage-threshold 必须位于 [0, 1] 区间。", file=sys.stderr)
            return 1
        if segmentations_dir is None and mask_jpg_dir is None and patch_h5_dir is None:
            print(
                "[ERROR] tissue_grid32x_isyntax 模式需要至少提供 --segmentations-dir、--mask-jpg-dir 或 --patch-h5-dir 之一。",
                file=sys.stderr,
            )
            return 1
        if segmentations_dir is not None and (not segmentations_dir.exists() or not segmentations_dir.is_dir()):
            print(f"[ERROR] segmentation 目录不存在或不可用：{segmentations_dir}", file=sys.stderr)
            return 1
        if mask_jpg_dir is not None and (not mask_jpg_dir.exists() or not mask_jpg_dir.is_dir()):
            print(f"[ERROR] mask jpg 目录不存在或不可用：{mask_jpg_dir}", file=sys.stderr)
            return 1
        if patch_h5_dir is not None and (not patch_h5_dir.exists() or not patch_h5_dir.is_dir()):
            print(f"[ERROR] patch h5 目录不存在或不可用：{patch_h5_dir}", file=sys.stderr)
            return 1
        if args.grid_align_to_patch_h5:
            if patch_h5_dir is None:
                print("[ERROR] --grid-align-to-patch-h5 需要同时提供 --patch-h5-dir。", file=sys.stderr)
                return 1
        if args.grid_physical_level0_extent < 0:
            print("[ERROR] --grid-physical-level0-extent 不能为负数。", file=sys.stderr)
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    wsi_files = discover_wsi_files(input_dir=input_dir, recursive=args.recursive)

    if not wsi_files:
        print(f"[WARNING] 在目录中未找到支持的 WSI 文件：{input_dir}")
        return 0

    success_count = 0
    failed_cases: List[Tuple[Path, str]] = []
    num_workers = max(1, int(args.num_workers))

    if num_workers == 1:
        for slide_path in tqdm(wsi_files, desc="处理 WSI", unit="slide"):
            try:
                result = process_single_wsi(
                    slide_path=slide_path,
                    input_root=input_dir,
                    output_root=output_dir,
                    mode=args.mode,
                    segmentations_dir=segmentations_dir,
                    patch_h5_dir=patch_h5_dir,
                    mask_jpg_dir=mask_jpg_dir,
                    min_long_side=args.min_long_side,
                    max_long_side=args.max_long_side,
                    preferred_magnification=args.preferred_magnification,
                    target_downsample=args.target_downsample,
                    grid_cell_size_level0=args.grid_cell_size_level0,
                    grid_cell_size_thumb=args.grid_cell_size_thumb,
                    grid_physical_level0_extent=args.grid_physical_level0_extent,
                    grid_align_to_patch_h5=args.grid_align_to_patch_h5,
                    grid_overlap_ratio=args.grid_overlap_ratio,
                    tissue_coverage_threshold=args.tissue_coverage_threshold,
                    white_threshold=args.white_threshold,
                    no_background_filter=args.no_background_filter,
                    overwrite=args.overwrite,
                )
                if result is not None:
                    success_count += 1
            except Exception as exc:
                failed_cases.append((slide_path, str(exc)))
                tqdm.write(f"[ERROR] {slide_path}: {exc}")
    else:
        tasks: List[Dict[str, object]] = [
            {
                "slide_path": str(slide_path),
                "input_root": str(input_dir),
                "output_root": str(output_dir),
                "mode": args.mode,
                "segmentations_dir": "" if segmentations_dir is None else str(segmentations_dir),
                "patch_h5_dir": "" if patch_h5_dir is None else str(patch_h5_dir),
                "mask_jpg_dir": "" if mask_jpg_dir is None else str(mask_jpg_dir),
                "min_long_side": args.min_long_side,
                "max_long_side": args.max_long_side,
                "preferred_magnification": args.preferred_magnification,
                "target_downsample": args.target_downsample,
                "grid_cell_size_level0": args.grid_cell_size_level0,
                "grid_cell_size_thumb": args.grid_cell_size_thumb,
                "grid_physical_level0_extent": args.grid_physical_level0_extent,
                "grid_align_to_patch_h5": args.grid_align_to_patch_h5,
                "grid_overlap_ratio": args.grid_overlap_ratio,
                "tissue_coverage_threshold": args.tissue_coverage_threshold,
                "white_threshold": args.white_threshold,
                "no_background_filter": args.no_background_filter,
                "overwrite": args.overwrite,
            }
            for slide_path in wsi_files
        ]

        # 使用 spawn 以减少某些底层库在 fork 后状态不一致的问题。
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_wsi_safe, tasks),
                total=len(tasks),
                desc="处理 WSI",
                unit="slide",
            ):
                slide_path = Path(str(result["slide_path"]))
                if bool(result["ok"]):
                    success_count += 1
                else:
                    error_message = str(result.get("error", "未知错误"))
                    failed_cases.append((slide_path, error_message))
                    tqdm.write(f"[ERROR] {slide_path}: {error_message}")

    print("\n处理完成")
    print(f"成功：{success_count}")
    print(f"失败：{len(failed_cases)}")

    if failed_cases:
        print("\n失败明细：")
        for slide_path, message in failed_cases:
            print(f"- {slide_path}: {message}")

    # 即使部分失败，也不直接返回 1，这样批量任务可以尽可能完成。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
