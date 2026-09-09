#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transmil.model import TransMIL  # noqa: E402
from transmil.training import TrainConfig, run_transmil_training  # noqa: E402
from transmil_official.datasets import AdenomaSslData  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-style TransMIL training entrypoint for adenoma.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.gpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_dataset = AdenomaSslData(cfg["data"]["manifest_csv"], cfg["data"]["feature_dir"], "train")
    val_dataset = AdenomaSslData(cfg["data"]["manifest_csv"], cfg["data"]["feature_dir"], "val")
    test_dataset = AdenomaSslData(cfg["data"]["manifest_csv"], cfg["data"]["feature_dir"], "test")

    model = TransMIL(
        embed_dim=cfg["model"]["embed_dim"],
        model_dim=cfg["model"]["model_dim"],
        n_classes=cfg["model"]["n_classes"],
        dropout=cfg["model"]["dropout"],
    )
    train_cfg = TrainConfig(
        results_dir=Path(cfg["results_dir"]),
        max_epochs=cfg["training"]["max_epochs"],
        seed=cfg["training"]["seed"],
        lr=cfg["training"]["lr"],
        reg=cfg["training"]["weight_decay"],
        weighted_sample=cfg["training"]["weighted_sample"],
        early_stopping=cfg["training"]["early_stopping"],
        smoke=bool(cfg["training"].get("smoke", False)),
        task_name=cfg["task_name"],
        model_name="transmil_official",
        fold=cfg["fold"],
        positive_name="SSL",
        negative_name="others",
    )
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["results_dir"]) / "official_config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    run_transmil_training(model, train_dataset, val_dataset, test_dataset, train_cfg)


if __name__ == "__main__":
    main()
