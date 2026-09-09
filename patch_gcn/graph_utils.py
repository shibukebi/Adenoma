from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def load_torch_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_patch_coords(patch_h5_path: Path) -> np.ndarray:
    with h5py.File(patch_h5_path, "r") as handle:
        if "coords" not in handle:
            raise KeyError(f"'coords' dataset not found in {patch_h5_path}")
        coords = np.asarray(handle["coords"][:], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords with shape [N, 2], got {coords.shape} from {patch_h5_path}")
    return coords


def build_knn_graph(
    coords: np.ndarray,
    k: int = 8,
    symmetrize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords with shape [N, 2], got {coords.shape}")

    num_nodes = int(coords.shape[0])
    if num_nodes < 1:
        raise ValueError("Cannot build a graph with zero nodes")

    if num_nodes == 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0], dtype=torch.float32)
        return edge_index, edge_weight

    n_neighbors = min(num_nodes, int(k) + 1)
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(coords)
    indices = nn_model.kneighbors(coords, return_distance=False)

    src = np.repeat(np.arange(num_nodes, dtype=np.int64), n_neighbors - 1)
    dst = indices[:, 1:].reshape(-1).astype(np.int64, copy=False)
    edges = np.stack([dst, src], axis=1)

    if symmetrize:
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)

    self_loops = np.arange(num_nodes, dtype=np.int64)
    self_edges = np.stack([self_loops, self_loops], axis=1)
    edges = np.concatenate([edges, self_edges], axis=0)
    edges = np.unique(edges, axis=0)

    dst_nodes = edges[:, 0]
    degrees = np.bincount(dst_nodes, minlength=num_nodes).astype(np.float32)
    weights = 1.0 / np.clip(degrees[dst_nodes], a_min=1.0, a_max=None)

    edge_index = torch.from_numpy(edges.T.copy()).long()
    edge_weight = torch.from_numpy(weights.copy()).float()
    return edge_index, edge_weight


def graph_cache_path(graph_dir: Path, slide_id: str) -> Path:
    return graph_dir / f"{slide_id}.pt"


def load_graph_cache(graph_path: Path) -> dict:
    payload = load_torch_payload(graph_path)
    if "edge_index" not in payload or "edge_weight" not in payload or "num_nodes" not in payload:
        raise KeyError(f"Invalid graph cache payload in {graph_path}")
    return payload


def save_graph_cache(graph_path: Path, payload: dict) -> None:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = graph_path.with_suffix(f"{graph_path.suffix}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(graph_path)


def ensure_graph_cache(
    slide_id: str,
    patch_dir: Path,
    graph_dir: Path,
    k: int,
    expected_num_nodes: int | None = None,
    symmetrize: bool = True,
    rebuild: bool = False,
) -> dict:
    cache_path = graph_cache_path(graph_dir, slide_id)
    if cache_path.exists() and not rebuild:
        payload = load_graph_cache(cache_path)
        cached_num_nodes = int(payload["num_nodes"])
        if int(payload.get("k_neighbors", k)) == int(k) and (
            expected_num_nodes is None or cached_num_nodes == int(expected_num_nodes)
        ):
            return payload

    patch_h5_path = patch_dir / f"{slide_id}.h5"
    coords = load_patch_coords(patch_h5_path)
    if expected_num_nodes is not None and int(coords.shape[0]) != int(expected_num_nodes):
        raise ValueError(
            f"Graph node count mismatch for {slide_id}: coords={coords.shape[0]}, expected={expected_num_nodes}"
        )

    edge_index, edge_weight = build_knn_graph(coords, k=k, symmetrize=symmetrize)
    payload = {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "num_nodes": int(coords.shape[0]),
        "k_neighbors": int(k),
        "symmetrized": bool(symmetrize),
    }
    save_graph_cache(cache_path, payload)
    return payload
