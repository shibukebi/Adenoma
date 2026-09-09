from pathlib import Path
from io import BytesIO

import pytest
from PIL import Image

from challenge_review.wsi import get_metadata, get_tile, wsi_service


SVS = Path("/data15/zhengke_usb/Adenoma_yx/640659 1.svs")
ISYNTAX = Path("/data15/zhengke_usb2/yuexin_data/Adenoma_hp/2929670e-28be-489e-825e-9f0e6e386598.isyntax")


@pytest.fixture(scope="module", autouse=True)
def close_wsi_service():
    yield
    wsi_service.close()


@pytest.mark.parametrize("path", [SVS, ISYNTAX])
def test_deepzoom_metadata_and_center_tile(path):
    if not path.exists():
        pytest.skip(f"Missing integration WSI: {path}")
    metadata = get_metadata(path)
    assert metadata["width"] > 1000
    assert metadata["height"] > 1000
    assert metadata["level_count"] > 5
    level = metadata["level_count"] - 1
    columns = (metadata["width"] + metadata["tile_size"] - 1) // metadata["tile_size"]
    rows = (metadata["height"] + metadata["tile_size"] - 1) // metadata["tile_size"]
    tile = get_tile(path, level, (columns // 2, rows // 2))
    assert tile.startswith(b"\xff\xd8")
    decoded = Image.open(BytesIO(tile))
    assert 1 <= decoded.width <= 256
    assert 1 <= decoded.height <= 256
