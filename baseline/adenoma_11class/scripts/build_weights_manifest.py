#!/usr/bin/env python3
"""Build the canonical 70-checkpoint benchmark manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "CLAM-SB": ["2p5x", "5x", "10x", "20x"],
    "TransMIL": ["2p5x", "5x", "10x", "20x"],
    "DSMIL": ["2p5x", "5x", "10x", "20x"],
    "MIST": ["2p5x_5x", "5x_10x"],
}
LEGACY_NAMES = {
    "CLAM-SB": ["CLAM-SB"],
    "TransMIL": ["transmil", "TransMIL"],
    "DSMIL": ["dsmil", "DSMIL"],
    "MIST": ["MIST"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=BENCHMARK_ROOT / "weights/weights_manifest.json")
    parser.add_argument("--copy-weights", action="store_true")
    parser.add_argument("--include-source-paths", action="store_true")
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checkpoint_name(model: str, fold: int, legacy: bool = False) -> str:
    if model == "MIST":
        return "1.pth"
    return f"s_{5 if legacy else fold}_checkpoint.pt"


def candidate_paths(args: argparse.Namespace, model: str, feature: str, fold: int) -> list[tuple[Path, str]]:
    paths = [
        (
            args.new_root / model / "11class" / feature / f"fold-{fold}" / checkpoint_name(model, fold),
            "fivefold_root",
        )
    ]
    if fold != 4:
        return paths
    for legacy_model in LEGACY_NAMES[model]:
        if model == "MIST":
            suffix = Path("MIST/1.pth") if feature == "2p5x_5x" else Path("MIST/5x_10x/1.pth")
            paths.append((args.legacy_root / suffix, "legacy_fold5_root"))
            break
        paths.append(
            (
                args.legacy_root / legacy_model / "11class" / feature / checkpoint_name(model, fold, legacy=True),
                "legacy_fold5_root",
            )
        )
    return paths


def main() -> None:
    args = parse_args()
    artifacts = []
    checksum_lines = []
    for model, features in EXPERIMENTS.items():
        for feature in features:
            for fold in range(5):
                candidates = candidate_paths(args, model, feature, fold)
                located = next(((path, source_kind) for path, source_kind in candidates if path.is_file()), None)
                suffix = ".pth" if model == "MIST" else ".pt"
                destination = BENCHMARK_ROOT / "weights" / model / feature / f"fold-{fold}" / f"checkpoint{suffix}"
                record = {
                    "model": model,
                    "feature": feature,
                    "fold": fold,
                    "historical_fold": 5 if fold == 4 else fold,
                    "status": "missing" if located is None else "available",
                    "destination": str(destination.relative_to(BENCHMARK_ROOT)),
                    "original_filename": None,
                    "source_kind": None,
                    "size_bytes": None,
                    "sha256": None,
                    "download_url": None,
                }
                if located:
                    source, source_kind = located
                    record.update({
                        "original_filename": source.name,
                        "source_kind": source_kind,
                        "size_bytes": source.stat().st_size,
                        "sha256": digest(source),
                    })
                    if args.include_source_paths:
                        record["source_path"] = str(source.resolve())
                    if args.copy_weights:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
                        record["status"] = "copied"
                        checksum_lines.append(f"{record['sha256']}  {record['destination']}\n")
                artifacts.append(record)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "adenoma_11class_5fold",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_artifacts": len(artifacts),
        "available_artifacts": sum(item["status"] != "missing" for item in artifacts),
        "artifacts": artifacts,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (BENCHMARK_ROOT / "weights/SHA256SUMS").write_text("".join(checksum_lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "expected": len(artifacts), "available": payload["available_artifacts"]}))


if __name__ == "__main__":
    main()
