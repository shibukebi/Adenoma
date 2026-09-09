#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAM_ROOT = PROJECT_ROOT / "CLAM"
if not CLAM_ROOT.exists():
    CLAM_ROOT = PROJECT_ROOT.parent / "CLAM"
sys.path.insert(0, str(CLAM_ROOT))

extra_site_packages = os.environ.get("PYISYNTAX_SITE_PACKAGES", "").strip()
if extra_site_packages:
    for site_path in extra_site_packages.split(":"):
        site_path = site_path.strip()
        if site_path:
            sys.path.append(site_path)

from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP  # noqa: E402
from models import get_encoder  # noqa: E402
from utils.file_utils import save_hdf5  # noqa: E402
from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route A feature extraction with controllable encoder init.")
    parser.add_argument("--data_h5_dir", required=True)
    parser.add_argument("--data_slide_dir", required=True)
    parser.add_argument("--slide_ext", default=".svs")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--feat_dir", required=True)
    parser.add_argument(
        "--model-name",
        default="resnet50_trunc",
        choices=["resnet50_trunc", "uni_v1", "conch_v1", "conch_v1_5"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_patch_size", type=int, default=224)
    parser.add_argument("--base-magnification", type=float, default=40.0)
    parser.add_argument("--target-magnification", type=float, default=0.0)
    parser.add_argument(
        "--physical-extent-patch-size",
        type=int,
        default=0,
        help=(
            "Patch size used to convert target magnification to a level-0 physical "
            "extent. Defaults to the input h5 coords patch_size."
        ),
    )
    parser.add_argument(
        "--expected-physical-level0-extent",
        type=int,
        default=0,
        help=(
            "If > 0, ignore h5 patch_level for image extraction and read this many "
            "level-0 pixels per side for every patch before model transforms."
        ),
    )
    parser.add_argument("--encoder-init", choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--no_auto_skip", action="store_true", default=False)
    return parser.parse_args()


def compute_physical_level0_extent(
    *,
    base_magnification: float,
    target_magnification: float,
    patch_size: int,
) -> int:
    if base_magnification <= 0:
        raise ValueError("--base-magnification must be positive.")
    if target_magnification <= 0:
        raise ValueError("--target-magnification must be positive when deriving physical extent.")
    if patch_size <= 0:
        raise ValueError("physical extent patch size must be positive.")
    return int(round(float(patch_size) * float(base_magnification) / float(target_magnification)))


def as_float_pair(value) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 1.0, 1.0
    if array.size == 1:
        return float(array[0]), float(array[0])
    return float(array[0]), float(array[1])


def read_patch_metadata(h5_file_path: str) -> dict:
    with h5py.File(h5_file_path, "r") as handle:
        coords = handle["coords"]
        patch_size = int(coords.attrs.get("patch_size", 0))
        patch_level = int(coords.attrs.get("patch_level", -1))
        downsample = as_float_pair(coords.attrs.get("downsample", (1.0, 1.0)))
        physical_extent_x = int(round(patch_size * downsample[0]))
        physical_extent_y = int(round(patch_size * downsample[1]))
        if physical_extent_x != physical_extent_y:
            print(
                "warning: anisotropic legacy physical extent "
                f"{physical_extent_x}x{physical_extent_y} in {h5_file_path}; "
                "using the rounded maximum as scalar legacy extent."
            )
        return {
            "patch_size": patch_size,
            "patch_level": patch_level,
            "legacy_downsample": downsample,
            "legacy_physical_level_0_extent": [physical_extent_x, physical_extent_y],
            "legacy_physical_level_0_extent_scalar": max(physical_extent_x, physical_extent_y),
            "num_patches": int(coords.shape[0]),
        }


class PhysicalExtentWholeSlideBag(Dataset):
    """
    Read patches by level-0 physical extent, not by a pyramid level index.

    The h5 coords are already level-0 coordinates. patch_level in the h5 is kept
    only as legacy provenance and is not used for coordinate or image extraction.
    """

    def __init__(
        self,
        file_path: str,
        wsi,
        img_transforms,
        physical_level0_extent: int,
    ) -> None:
        if physical_level0_extent <= 0:
            raise ValueError("physical_level0_extent must be positive.")
        self.file_path = file_path
        self.wsi = wsi
        self.roi_transforms = img_transforms
        self.physical_level0_extent = int(physical_level0_extent)

        with h5py.File(self.file_path, "r") as handle:
            self.length = int(handle["coords"].shape[0])
            self.legacy_metadata = read_patch_metadata(self.file_path)

        self.summary()

    def __len__(self) -> int:
        return self.length

    def summary(self) -> None:
        print("\nphysical-scale feature extraction settings")
        print(f"physical_level_0_extent: {self.physical_level0_extent}")
        print(f"legacy_patch_level: {self.legacy_metadata['patch_level']}")
        print(f"legacy_patch_size: {self.legacy_metadata['patch_size']}")
        print(f"legacy_downsample: {self.legacy_metadata['legacy_downsample']}")
        print(f"legacy_physical_level_0_extent: {self.legacy_metadata['legacy_physical_level_0_extent']}")
        print("transformations: ", self.roi_transforms)

    def __getitem__(self, idx: int) -> dict:
        with h5py.File(self.file_path, "r") as handle:
            coord = handle["coords"][idx].astype(np.int64)
        x = int(coord[0])
        y = int(coord[1])
        extent = self.physical_level0_extent
        img = self.wsi.read_region((x, y), 0, (extent, extent)).convert("RGB")
        img = self.roi_transforms(img)
        return {"img": img, "coord": coord.astype(np.int32)}


def configure_encoder_env(model_name: str, weights_path: str | None) -> None:
    if not weights_path:
        return
    if model_name == "resnet50_trunc":
        os.environ["TIMM_RESNET50_WEIGHTS_PATH"] = weights_path
    elif model_name == "uni_v1":
        os.environ["UNI_CKPT_PATH"] = weights_path
    elif model_name in {"conch_v1", "conch_v1_5"}:
        os.environ["CONCH_CKPT_PATH"] = weights_path


def compute_w_loader(
    output_path: str,
    loader: DataLoader,
    model: torch.nn.Module,
    attr_dict: dict | None = None,
    verbose: int = 0,
) -> str:
    if verbose > 0:
        print(f"processing a total of {len(loader)} batches")

    mode = "w"
    for data in tqdm(loader):
        with torch.inference_mode():
            batch = data["img"].to(device, non_blocking=True)
            coords = data["coord"].numpy().astype(np.int32)
            features = model(batch).cpu().numpy().astype(np.float32)

            asset_dict = {"features": features, "coords": coords}
            save_hdf5(output_path, asset_dict, attr_dict=attr_dict if mode == "w" else None, mode=mode)
            mode = "a"

    if attr_dict is not None and "features" in attr_dict:
        root_attr_keys = (
            "extraction_mode",
            "physical_level_0_extent",
            "expected_physical_level_0_extent",
            "physical_extent_standardized",
            "base_magnification",
            "target_magnification",
            "scale_to_level0",
            "physical_extent_patch_size",
            "legacy_patch_level",
            "legacy_patch_size",
            "legacy_downsample",
            "legacy_physical_level_0_extent",
            "target_patch_size",
            "model_name",
        )
        with h5py.File(output_path, "a") as handle:
            for key in root_attr_keys:
                if key in attr_dict["features"]:
                    handle.attrs[key] = attr_dict["features"][key]
    return output_path


def main() -> None:
    args = parse_args()

    print("initializing dataset")
    bags_dataset = Dataset_All_Bags(args.csv_path)

    feat_dir = Path(args.feat_dir)
    pt_dir = feat_dir / "pt_files"
    h5_dir = feat_dir / "h5_files"
    pt_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)
    dest_files = set(os.listdir(pt_dir))

    print(f"loading model checkpoint (encoder_init={args.encoder_init}, model_name={args.model_name})")
    if args.weights_path:
        print(f"weights_path={args.weights_path}")
    configure_encoder_env(args.model_name, args.weights_path)
    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)

    model = model.eval().to(device)
    total = len(bags_dataset)
    if device.type == "cuda":
        loader_kwargs = {"num_workers": 8, "pin_memory": True}
    else:
        loader_kwargs = {}

    if args.slide_ext.lower() == ".isyntax":
        loader_kwargs["num_workers"] = 0
        if device.type == "cuda":
            loader_kwargs["pin_memory"] = True

    for bag_candidate_idx in tqdm(range(total)):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = f"{slide_id}.h5"
        h5_file_path = os.path.join(args.data_h5_dir, bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, f"{slide_id}{args.slide_ext}")
        print(f"\nprogress: {bag_candidate_idx}/{total}")
        print(slide_id)

        if not args.no_auto_skip and f"{slide_id}.pt" in dest_files:
            print(f"skipped {slide_id}")
            continue

        output_path = str(h5_dir / bag_name)
        time_start = time.time()
        wsi_object = WholeSlideImage(slide_file_path)
        wsi = wsi_object.getOpenSlide()
        patch_meta = read_patch_metadata(h5_file_path)
        physical_extent_patch_size = int(args.physical_extent_patch_size or patch_meta["patch_size"])
        if args.expected_physical_level0_extent > 0:
            expected_extent = int(args.expected_physical_level0_extent)
        elif args.target_magnification > 0:
            expected_extent = compute_physical_level0_extent(
                base_magnification=float(args.base_magnification),
                target_magnification=float(args.target_magnification),
                patch_size=physical_extent_patch_size,
            )
        else:
            expected_extent = 0
        if args.model_name == "uni_v1" and expected_extent <= 0:
            raise ValueError(
                "UNI feature extraction requires --target-magnification or "
                "--expected-physical-level0-extent. "
                "Do not rely on legacy h5 patch_level for UNI image extraction."
            )
        if expected_extent > 0:
            dataset = PhysicalExtentWholeSlideBag(
                file_path=h5_file_path,
                wsi=wsi,
                img_transforms=img_transforms,
                physical_level0_extent=expected_extent,
            )
            physical_level0_extent = expected_extent
            physical_extent_standardized = patch_meta["legacy_physical_level_0_extent"] != [
                expected_extent,
                expected_extent,
            ]
            extraction_mode = "physical_level0_extent"
        else:
            dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, img_transforms=img_transforms)
            physical_level0_extent = patch_meta["legacy_physical_level_0_extent_scalar"]
            physical_extent_standardized = False
            extraction_mode = "legacy_h5_patch_level"

        attr_dict = {
            "features": {
                "extraction_mode": extraction_mode,
                "physical_level_0_extent": physical_level0_extent,
                "expected_physical_level_0_extent": expected_extent,
                "physical_extent_standardized": physical_extent_standardized,
                "base_magnification": float(args.base_magnification),
                "target_magnification": float(args.target_magnification),
                "scale_to_level0": (
                    float(args.base_magnification) / float(args.target_magnification)
                    if float(args.target_magnification) > 0
                    else 0.0
                ),
                "physical_extent_patch_size": physical_extent_patch_size,
                "legacy_patch_level": patch_meta["patch_level"],
                "legacy_patch_size": patch_meta["patch_size"],
                "legacy_downsample": patch_meta["legacy_downsample"],
                "legacy_physical_level_0_extent": patch_meta["legacy_physical_level_0_extent"],
                "target_patch_size": int(args.target_patch_size),
                "model_name": args.model_name,
            },
            "coords": {
                "coordinate_space": "level0",
                "level_index_for_coordinate_mapping": "forbidden",
                "source_level_downsample_for_coordinate_mapping": "forbidden",
                "physical_level_0_extent": physical_level0_extent,
                "base_magnification": float(args.base_magnification),
                "target_magnification": float(args.target_magnification),
                "physical_extent_patch_size": physical_extent_patch_size,
            },
        }
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
        output_file_path = compute_w_loader(output_path, loader=loader, model=model, attr_dict=attr_dict, verbose=1)
        time_elapsed = time.time() - time_start
        print(f"\ncomputing features for {output_file_path} took {time_elapsed} s")

        with h5py.File(output_file_path, "r") as file:
            features = file["features"][:]
            print("features size: ", features.shape)
            print("coordinates size: ", file["coords"].shape)

        torch.save(torch.from_numpy(features), pt_dir / f"{slide_id}.pt")
        wsi.close()


if __name__ == "__main__":
    main()
