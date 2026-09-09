from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .graph_utils import ensure_graph_cache, load_torch_payload


class PatchGCNDataset(Dataset):
    def __init__(
        self,
        slide_data: pd.DataFrame,
        feature_dir: str | Path,
        patch_dir: str | Path,
        graph_dir: str | Path,
        k_neighbors: int = 8,
        symmetrize: bool = True,
        lazy_graph_cache: bool = True,
    ):
        self.slide_data = slide_data.reset_index(drop=True).copy()
        self.feature_dir = Path(feature_dir)
        self.patch_dir = Path(patch_dir)
        self.graph_dir = Path(graph_dir)
        self.k_neighbors = int(k_neighbors)
        self.symmetrize = bool(symmetrize)
        self.lazy_graph_cache = bool(lazy_graph_cache)
        self.num_classes = int(self.slide_data["label"].nunique())

    def __len__(self) -> int:
        return len(self.slide_data)

    def __getitem__(self, idx: int) -> dict:
        row = self.slide_data.iloc[idx]
        slide_id = str(row["slide_id"])
        label = int(row["label"])

        feature_path = self.feature_dir / "pt_files" / f"{slide_id}.pt"
        features = load_torch_payload(feature_path)
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Expected a tensor in {feature_path}, got {type(features).__name__}")
        if features.ndim != 2:
            raise ValueError(f"Expected features with shape [N, C], got {tuple(features.shape)} in {feature_path}")
        features = features.float()

        graph_payload = ensure_graph_cache(
            slide_id=slide_id,
            patch_dir=self.patch_dir,
            graph_dir=self.graph_dir,
            k=self.k_neighbors,
            expected_num_nodes=int(features.shape[0]),
            symmetrize=self.symmetrize,
            rebuild=False,
        )

        return {
            "slide_id": slide_id,
            "label": torch.tensor(label, dtype=torch.long),
            "features": features,
            "edge_index": graph_payload["edge_index"].long(),
            "edge_weight": graph_payload["edge_weight"].float(),
        }


def build_patch_gcn_split(
    slide_data: pd.DataFrame,
    ids: list[str],
    feature_dir: str | Path,
    patch_dir: str | Path,
    graph_dir: str | Path,
    k_neighbors: int = 8,
    symmetrize: bool = True,
    lazy_graph_cache: bool = True,
) -> PatchGCNDataset:
    subset = slide_data[slide_data["slide_id"].isin(ids)].reset_index(drop=True)
    return PatchGCNDataset(
        slide_data=subset,
        feature_dir=feature_dir,
        patch_dir=patch_dir,
        graph_dir=graph_dir,
        k_neighbors=k_neighbors,
        symmetrize=symmetrize,
        lazy_graph_cache=lazy_graph_cache,
    )
