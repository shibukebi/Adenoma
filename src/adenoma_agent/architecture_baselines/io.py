"""Small deterministic I/O helpers for architecture baseline artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSONL at {0}:{1}: {2}".format(path, line_number, exc))
            if not isinstance(value, dict):
                raise ValueError("JSONL row must be an object at {0}:{1}".format(path, line_number))
            yield value


def read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    return list(iter_jsonl(path))


def _atomic_replace(path: Path, writer) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".{0}.".format(path.name), dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_json(path: Path, payload: Any) -> Path:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    return _atomic_replace(Path(path), lambda handle: handle.write(encoded))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    def _write(handle) -> None:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")

    return _atomic_replace(Path(path), _write)


def path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def require_output_outside_sources(output_path: Path, source_roots: Iterable[Path]) -> None:
    output_path = Path(output_path).resolve()
    for source_root in source_roots:
        if path_is_within(output_path, Path(source_root)):
            raise ValueError(
                "Artifact output must not be inside read-only source data: {0}".format(output_path)
            )
