#!/usr/bin/env python3
"""Inventory and optionally copy verified experiment checkpoints for GitHub.

The script is intentionally conservative: feature tensors and raw data are
never copied based only on their extension. A checkpoint must have a model
checkpoint-style filename and live under an explicitly supplied result root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
CHECKPOINT_TOKENS = (
    "checkpoint",
    "best",
    "model",
    "state_dict",
    "weights",
)
EXCLUDED_TOKENS = (
    "feature",
    "embedding",
    "latent",
    "patch",
)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def is_checkpoint(path: Path) -> bool:
    if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        return False
    name = path.name.lower()
    if any(token in name for token in EXCLUDED_TOKENS):
        return False
    if any(token in name for token in CHECKPOINT_TOKENS) or name.startswith("s_"):
        return True
    # MIST names its epoch checkpoints numerically (for example, 1.pth).
    return bool(re.fullmatch(r"\d+\.(pt|pth|ckpt)", name)) and "mist" in path.as_posix().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path, default=Path("release/experiment_artifacts.json"))
    parser.add_argument("--copy-weights", action="store_true")
    parser.add_argument("--release-dir", type=Path, default=Path("release"))
    parser.add_argument(
        "--include-source-paths",
        action="store_true",
        help="include absolute source paths in the inventory; keep disabled for GitHub publication",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_path = (repo_root / args.output).resolve()
    release_dir = (repo_root / args.release_dir).resolve()
    weight_dir = release_dir / "weights"
    if args.copy_weights:
        weight_dir.mkdir(parents=True, exist_ok=True)

    inventory = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": repo_root.name,
        "roots": [],
        "files": [],
        "notes": [
            "WSI files and feature tensors are intentionally excluded.",
            "A checkpoint is copied only when its filename passes the conservative checkpoint filter.",
        ],
    }
    seen_hashes: set[str] = set()
    for root_arg in args.weights_root:
        root = root_arg.expanduser().resolve()
        root_entry = {"root_name": root.name, "status": "present" if root.is_dir() else "missing_root"}
        if args.include_source_paths:
            root_entry["path"] = str(root)
        inventory["roots"].append(root_entry)
        if not root.is_dir():
            continue
        for source in sorted(path for path in root.rglob("*") if path.is_file() and is_checkpoint(path)):
            digest = sha256(source)
            relative = source.relative_to(root)
            destination = weight_dir / root.name / relative
            try:
                destination_label = str(destination.relative_to(repo_root))
            except ValueError:
                destination_label = str(Path("<release-dir>") / "weights" / root.name / relative)
            record = {
                "source_root": root.name,
                "source_relative_path": str(relative),
                "size_bytes": source.stat().st_size,
                "sha256": digest,
                "copied": False,
                "destination": destination_label,
            }
            if args.include_source_paths:
                record["source_path"] = str(source)
            if digest in seen_hashes:
                record["status"] = "duplicate_sha256"
            else:
                seen_hashes.add(digest)
                record["status"] = "checkpoint_candidate"
                if args.copy_weights:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    record["copied"] = True
            inventory["files"].append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checksum_path = release_dir / "sha256sums.txt"
    copied = [item for item in inventory["files"] if item["copied"]]
    checksum_path.write_text(
        "".join(f"{item['sha256']}  {item['destination']}\n" for item in copied),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output_path),
        "roots": len(inventory["roots"]),
        "checkpoint_candidates": len(inventory["files"]),
        "copied": len(copied),
        "missing_roots": sum(item["status"] == "missing_root" for item in inventory["roots"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
