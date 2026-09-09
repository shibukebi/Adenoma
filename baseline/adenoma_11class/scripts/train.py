#!/usr/bin/env python3
"""Unified launcher for one adenoma 11-class benchmark experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parents[1]
MODEL_DIRS = {
    "clam-sb": "CLAM-SB",
    "transmil": "TransMIL",
    "dsmil": "DSMIL",
    "mist": "MIST",
}
SINGLE_SCALES = {"2p5x", "5x", "10x", "20x"}
MIST_SCALES = {"2p5x_5x", "5x_10x"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_DIRS), required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--ready-csv", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--mist-manifest-root", type=Path)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_path(value: Path | None, option: str) -> Path:
    if value is None:
        raise SystemExit(f"{option} is required for this model")
    return value.resolve()


def build_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    model_dir = MODEL_DIRS[args.model]
    output_dir = args.results_root.resolve() / model_dir / args.feature / f"fold-{args.fold}"
    epochs = args.max_epochs or (200 if args.model == "mist" else 100)

    if args.model == "mist":
        if args.feature not in MIST_SCALES:
            raise SystemExit(f"MIST feature must be one of {sorted(MIST_SCALES)}")
        manifest_root = require_path(args.mist_manifest_root, "--mist-manifest-root")
        fold_root = manifest_root / args.feature / f"fold-{args.fold}"
        train_csv = fold_root / "train" / f"mist_11class_{args.feature}_fold{args.fold}_train.csv"
        val_csv = fold_root / "val" / f"mist_11class_{args.feature}_fold{args.fold}_val.csv"
        command = [
            args.python_bin,
            str(PROJECT_ROOT / "third_party/mist/train_ade.py"),
            "--dataset_train", str(train_csv),
            "--dataset_val", str(val_csv),
            "--num_classes", "11",
            "--feats_size", "5120",
            "--num_epochs", str(epochs),
            "--gpu_index", "0",
            "--save_dir", str(output_dir),
        ]
        return command, output_dir

    if args.feature not in SINGLE_SCALES:
        raise SystemExit(f"{args.model} feature must be one of {sorted(SINGLE_SCALES)}")
    ready_csv = require_path(args.ready_csv, "--ready-csv")
    split_dir = require_path(args.split_dir, "--split-dir")
    feature_dir = require_path(args.feature_root, "--feature-root") / args.feature
    common = [
        "--ready-csv", str(ready_csv),
        "--split-dir", str(split_dir),
        "--feature-dir", str(feature_dir),
        "--results-dir", str(output_dir),
        "--fold", str(args.fold),
        "--max-epochs", str(epochs),
        "--seed", str(args.seed),
        "--lr", "0.0001",
        "--reg", "0.00001",
        "--drop-out", "0.25",
        "--embed-dim", "1024",
        "--weighted-sample",
        "--early-stopping",
        "--task-mode", "adenoma_11class",
        "--task-name", f"adenoma_11class_{args.feature}_{args.model}_fold{args.fold}",
    ]
    if args.model == "clam-sb":
        command = [
            args.python_bin,
            str(PROJECT_ROOT / "scripts/train_clam_ssl_20x.py"),
            "--dataset-csv", str(ready_csv),
            *common,
            "--allow-small-splits",
            "--model-type", "clam_sb",
            "--bag-loss", "ce",
            "--inst-loss", "ce",
            "--model-size", "small",
            "--k-sample", "8",
            "--bag-weight", "0.7",
        ]
    elif args.model == "transmil":
        command = [
            args.python_bin,
            str(PROJECT_ROOT / "scripts/train_transmil_ssl.py"),
            *common,
            "--model-dim", "512",
        ]
    else:
        command = [
            args.python_bin,
            str(PROJECT_ROOT / "scripts/train_dsmil_single_11class.py"),
            *common,
            "--hidden-dim", "512",
            "--attn-dim", "128",
        ]
    return command, output_dir


def main() -> None:
    args = parse_args()
    command, output_dir = build_command(args)
    print(shlex.join(command))
    if args.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if args.gpu_index is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
