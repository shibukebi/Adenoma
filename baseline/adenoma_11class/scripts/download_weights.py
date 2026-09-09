#!/usr/bin/env python3
"""Download benchmark checkpoints listed in weights_manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=BENCHMARK_ROOT / "weights/weights_manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    downloaded = 0
    skipped = 0
    unavailable = 0
    for artifact in payload.get("artifacts", []):
        url = artifact.get("download_url")
        expected = artifact.get("sha256")
        if not url or not expected:
            unavailable += 1
            continue
        destination = BENCHMARK_ROOT / artifact["destination"]
        if destination.exists() and not args.overwrite:
            if sha256(destination) != expected:
                raise SystemExit(f"Checksum mismatch for existing file: {destination}")
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        with urlopen(url) as response, temporary.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        if sha256(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"Checksum mismatch after download: {destination}")
        temporary.replace(destination)
        downloaded += 1
    print(json.dumps({"downloaded": downloaded, "skipped": skipped, "unavailable": unavailable}))


if __name__ == "__main__":
    main()
