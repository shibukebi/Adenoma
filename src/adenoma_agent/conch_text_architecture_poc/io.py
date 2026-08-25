"""Deterministic, dependency-light I/O for the architecture POC."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence


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
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
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
            value = line.strip()
            if not value:
                continue
            try:
                row = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSONL at {0}:{1}: {2}".format(path, line_number, exc))
            if not isinstance(row, dict):
                raise ValueError("JSONL row must be an object at {0}:{1}".format(path, line_number))
            yield row


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".{0}.".format(path.name), dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def write_json(path: Path, payload: Any) -> Path:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    return _atomic_bytes(path, encoded)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    return _atomic_bytes(path, payload)


def write_text(path: Path, value: str) -> Path:
    return _atomic_bytes(path, value.encode("utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".{0}.".format(path.name), dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [{str(key).strip(): str(value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def relative_or_resolved(path: Path, root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (Path(root) / value).resolve()
