from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from patch_gcn.graph_utils import load_torch_payload


def load_feature_tensor(path: Path) -> torch.Tensor:
    payload = load_torch_payload(path)
    if isinstance(payload, torch.Tensor):
        features = payload
    elif isinstance(payload, dict):
        features = None
        for key in ("features", "feature", "x", "data"):
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                features = value
                break
        if features is None:
            raise TypeError(f"Unsupported feature payload keys in {path}")
    else:
        raise TypeError(f"Unsupported feature payload type {type(payload).__name__} in {path}")

    features = features.detach().cpu().float()
    if features.dim() == 1:
        features = features.unsqueeze(0)
    if features.dim() != 2:
        raise ValueError(f"Expected 2D features in {path}, got shape {tuple(features.shape)}")
    return features


class DualStreamBagDataset(Dataset):
    def __init__(
        self,
        slide_data: pd.DataFrame,
        feature_dir_a: str | Path,
        feature_dir_b: str | Path,
        num_classes: int = 2,
    ) -> None:
        self.slide_data = slide_data.reset_index(drop=True).copy()
        self.feature_dir_a = Path(feature_dir_a)
        self.feature_dir_b = Path(feature_dir_b)
        self.num_classes = int(num_classes)
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for class_idx in range(self.num_classes):
            self.slide_cls_ids[class_idx] = self.slide_data.index[self.slide_data["label"] == class_idx].to_numpy()

    def __len__(self) -> int:
        return len(self.slide_data)

    def getlabel(self, idx: int) -> int:
        return int(self.slide_data.iloc[int(idx)]["label"])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.slide_data.iloc[int(idx)]
        slide_id = str(row["slide_id"])
        label = int(row["label"])

        feature_path_a = self.feature_dir_a / "pt_files" / f"{slide_id}.pt"
        feature_path_b = self.feature_dir_b / "pt_files" / f"{slide_id}.pt"

        features_a = load_feature_tensor(feature_path_a)
        features_b = load_feature_tensor(feature_path_b)
        return features_a, features_b, label


def build_dual_stream_split(
    slide_data: pd.DataFrame,
    ids: list[str],
    feature_dir_a: str | Path,
    feature_dir_b: str | Path,
    num_classes: int = 2,
) -> DualStreamBagDataset:
    subset = slide_data[slide_data["slide_id"].isin(ids)].reset_index(drop=True)
    return DualStreamBagDataset(
        slide_data=subset,
        feature_dir_a=feature_dir_a,
        feature_dir_b=feature_dir_b,
        num_classes=num_classes,
    )


class SingleStreamBagDataset(Dataset):
    def __init__(
        self,
        slide_data: pd.DataFrame,
        feature_dir: str | Path,
        num_classes: int = 2,
    ) -> None:
        self.slide_data = slide_data.reset_index(drop=True).copy()
        self.feature_dir = Path(feature_dir)
        self.num_classes = int(num_classes)
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for class_idx in range(self.num_classes):
            self.slide_cls_ids[class_idx] = self.slide_data.index[self.slide_data["label"] == class_idx].to_numpy()

    def __len__(self) -> int:
        return len(self.slide_data)

    def getlabel(self, idx: int) -> int:
        return int(self.slide_data.iloc[int(idx)]["label"])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.slide_data.iloc[int(idx)]
        slide_id = str(row["slide_id"])
        label = int(row["label"])
        feature_path = self.feature_dir / "pt_files" / f"{slide_id}.pt"
        features = load_feature_tensor(feature_path)
        return features, label


def build_single_stream_split(
    slide_data: pd.DataFrame,
    ids: list[str],
    feature_dir: str | Path,
    num_classes: int = 2,
) -> SingleStreamBagDataset:
    subset = slide_data[slide_data["slide_id"].isin(ids)].reset_index(drop=True)
    return SingleStreamBagDataset(
        slide_data=subset,
        feature_dir=feature_dir,
        num_classes=num_classes,
    )


def collate_single_stream(batch: list[tuple[torch.Tensor, int]]):
    if len(batch) != 1:
        raise ValueError("Single-stream MIL loaders currently expect batch_size=1")
    features, label = batch[0]
    return features, torch.LongTensor([int(label)])


def collate_dual_stream(batch: list[tuple[torch.Tensor, torch.Tensor, int]]):
    if len(batch) != 1:
        raise ValueError("Dual-stream MIL loaders currently expect batch_size=1")
    features_a, features_b, label = batch[0]
    return features_a, features_b, torch.LongTensor([int(label)])
