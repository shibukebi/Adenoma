from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading

from .config import (
    WSI_DISK_CACHE_MAX_BYTES,
    WSI_DISK_CACHE_ROOT,
    WSI_DISK_CACHE_TRIM_BYTES,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PersistentWsiCache:
    def __init__(
        self,
        root: Path = WSI_DISK_CACHE_ROOT,
        max_bytes: int = WSI_DISK_CACHE_MAX_BYTES,
        trim_bytes: int = WSI_DISK_CACHE_TRIM_BYTES,
    ):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.trim_bytes = min(trim_bytes, max_bytes)
        self.index_path = self.root / "cache_index.sqlite3"
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.index_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    relative_path TEXT PRIMARY KEY,
                    slide_key TEXT NOT NULL,
                    slide_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cache_lru
                ON entries(pinned, last_accessed_at);
                CREATE INDEX IF NOT EXISTS idx_cache_slide
                ON entries(slide_id, kind);

                CREATE TABLE IF NOT EXISTS warm_status (
                    slide_id TEXT PRIMARY KEY,
                    source_path TEXT,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 20,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(warm_status)")
            }
            if "source_path" not in columns:
                connection.execute("ALTER TABLE warm_status ADD COLUMN source_path TEXT")
            if "priority" not in columns:
                connection.execute(
                    "ALTER TABLE warm_status ADD COLUMN priority INTEGER NOT NULL DEFAULT 20"
                )
            if "pinned" not in columns:
                connection.execute(
                    "ALTER TABLE warm_status ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
                )

    @staticmethod
    def slide_key(slide_id: str, source_path: str) -> str:
        digest = hashlib.sha256(f"{slide_id}\0{source_path}".encode("utf-8")).hexdigest()
        return digest[:24]

    def slide_dir(self, slide_id: str, source_path: str) -> Path:
        return self.root / self.slide_key(slide_id, source_path)

    def metadata_path(self, slide_id: str, source_path: str) -> Path:
        return self.slide_dir(slide_id, source_path) / "metadata.json"

    def dzi_path(self, slide_id: str, source_path: str) -> Path:
        return self.slide_dir(slide_id, source_path) / "slide.dzi"

    def thumbnail_path(self, slide_id: str, source_path: str) -> Path:
        return self.slide_dir(slide_id, source_path) / "thumbnail.jpeg"

    def tile_path(self, slide_id: str, source_path: str, level: int, column: int, row: int) -> Path:
        return self.slide_dir(slide_id, source_path) / "tiles" / str(level) / f"{column}_{row}.jpeg"

    def overview_marker_path(self, slide_id: str, source_path: str) -> Path:
        return self.slide_dir(slide_id, source_path) / "overview.ready"

    @contextmanager
    def key_lock(self, key: str):
        with self._key_locks_guard:
            lock = self._key_locks.setdefault(key, threading.Lock())
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def read_bytes(self, path: Path) -> bytes | None:
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return None
        self.touch(path)
        return content

    def read_json(self, path: Path) -> dict | None:
        content = self.read_bytes(path)
        if content is None:
            return None
        try:
            return json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.remove(path)
            return None

    def write_json(self, path: Path, value: dict, slide_id: str, source_path: str, kind: str, pinned=False):
        self.write_bytes(
            path,
            json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8"),
            slide_id,
            source_path,
            kind,
            pinned,
        )

    def write_bytes(
        self,
        path: Path,
        content: bytes,
        slide_id: str,
        source_path: str,
        kind: str,
        pinned: bool = False,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".cache-", delete=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            os.replace(temporary_path, path)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

        now = utc_now()
        relative_path = self._relative(path)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO entries(
                    relative_path, slide_key, slide_id, kind, size_bytes, pinned,
                    created_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET
                    size_bytes=excluded.size_bytes,
                    pinned=MAX(entries.pinned, excluded.pinned),
                    last_accessed_at=excluded.last_accessed_at
                """,
                (
                    relative_path,
                    self.slide_key(slide_id, source_path),
                    slide_id,
                    kind,
                    len(content),
                    int(pinned),
                    now,
                    now,
                ),
            )
        self.evict_if_needed()

    def touch(self, path: Path):
        relative_path = self._relative(path)
        try:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE entries SET last_accessed_at=? WHERE relative_path=?",
                    (utc_now(), relative_path),
                )
        except sqlite3.Error:
            pass

    def remove(self, path: Path):
        path.unlink(missing_ok=True)
        with self._connect() as connection:
            connection.execute("DELETE FROM entries WHERE relative_path=?", (self._relative(path),))

    def mark_overview_ready(self, slide_id: str, source_path: str, pinned: bool):
        self.write_bytes(
            self.overview_marker_path(slide_id, source_path),
            utc_now().encode("ascii"),
            slide_id,
            source_path,
            "overview_marker",
            pinned,
        )

    def overview_ready(self, slide_id: str, source_path: str) -> bool:
        return self.overview_marker_path(slide_id, source_path).is_file()

    def set_warm_status(self, slide_id: str, status: str, error: str | None = None):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO warm_status(slide_id,status,error,updated_at) VALUES (?,?,?,?)
                ON CONFLICT(slide_id) DO UPDATE SET
                    status=excluded.status,error=excluded.error,updated_at=excluded.updated_at
                """,
                (slide_id, status, error, utc_now()),
            )

    def enqueue_warm(
        self,
        slide_id: str,
        source_path: str,
        priority: int = 20,
        pinned: bool = False,
    ) -> str:
        if self.overview_ready(slide_id, source_path):
            self.set_warm_status(slide_id, "ready")
            return "ready"
        now = utc_now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status,priority,pinned FROM warm_status WHERE slide_id=?",
                (slide_id,),
            ).fetchone()
            if row and row["status"] in {"queued", "warming"}:
                connection.execute(
                    """
                    UPDATE warm_status SET source_path=?,priority=MIN(priority,?),
                        pinned=MAX(pinned,?),updated_at=? WHERE slide_id=?
                    """,
                    (source_path, priority, int(pinned), now, slide_id),
                )
                return row["status"]
            connection.execute(
                """
                INSERT INTO warm_status(
                    slide_id,source_path,status,priority,pinned,error,updated_at
                ) VALUES (?,?,?,?,?,NULL,?)
                ON CONFLICT(slide_id) DO UPDATE SET
                    source_path=excluded.source_path,status='queued',
                    priority=excluded.priority,pinned=excluded.pinned,
                    error=NULL,updated_at=excluded.updated_at
                """,
                (slide_id, source_path, "queued", priority, int(pinned), now),
            )
        return "queued"

    def reset_interrupted_warms(self):
        with self._connect() as connection:
            connection.execute(
                "UPDATE warm_status SET status='queued',updated_at=? WHERE status='warming'",
                (utc_now(),),
            )

    def claim_next_warm(self) -> dict | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT slide_id,source_path,priority,pinned FROM warm_status
                WHERE status='queued' AND source_path IS NOT NULL
                ORDER BY priority ASC,updated_at ASC LIMIT 1
                """
            ).fetchone()
            if not row:
                connection.commit()
                return None
            connection.execute(
                "UPDATE warm_status SET status='warming',updated_at=? WHERE slide_id=?",
                (utc_now(), row["slide_id"]),
            )
            connection.commit()
            return dict(row)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def warm_status(self, slide_id: str, source_path: str) -> str:
        if self.overview_ready(slide_id, source_path):
            return "ready"
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM warm_status WHERE slide_id=?", (slide_id,)
            ).fetchone()
        return row["status"] if row and row["status"] in {"queued", "warming"} else "cold"

    def warm_counts(self, slide_ids: list[str]) -> dict:
        if not slide_ids:
            return {}
        placeholders = ",".join("?" for _ in slide_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT status,COUNT(*) AS count FROM warm_status
                WHERE slide_id IN ({placeholders}) GROUP BY status
                """,
                slide_ids,
            ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    def evict_if_needed(self):
        with self._connect() as connection:
            total = connection.execute("SELECT COALESCE(SUM(size_bytes),0) FROM entries").fetchone()[0]
            if total <= self.max_bytes:
                return
            rows = connection.execute(
                """
                SELECT relative_path,size_bytes FROM entries
                WHERE pinned=0 ORDER BY last_accessed_at ASC
                """
            ).fetchall()
            for row in rows:
                if total <= self.trim_bytes:
                    break
                (self.root / row["relative_path"]).unlink(missing_ok=True)
                connection.execute("DELETE FROM entries WHERE relative_path=?", (row["relative_path"],))
                total -= row["size_bytes"]

    def stats(self) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS entries, COALESCE(SUM(size_bytes),0) AS bytes,
                       COALESCE(SUM(CASE WHEN pinned=1 THEN size_bytes ELSE 0 END),0) AS pinned_bytes
                FROM entries
                """
            ).fetchone()
            ready = connection.execute(
                "SELECT COUNT(*) FROM warm_status WHERE status='ready'"
            ).fetchone()[0]
            failed = connection.execute(
                "SELECT COUNT(*) FROM warm_status WHERE status='failed'"
            ).fetchone()[0]
            recent_failures = [
                dict(item)
                for item in connection.execute(
                    """
                    SELECT slide_id,error,updated_at FROM warm_status
                    WHERE status='failed' ORDER BY updated_at DESC LIMIT 20
                    """
                )
            ]
            queue_counts = {
                item["status"]: item["count"]
                for item in connection.execute(
                    "SELECT status,COUNT(*) AS count FROM warm_status GROUP BY status"
                )
            }
        return {
            "root": str(self.root),
            "entries": row["entries"],
            "bytes": row["bytes"],
            "pinned_bytes": row["pinned_bytes"],
            "max_bytes": self.max_bytes,
            "trim_bytes": self.trim_bytes,
            "ready_slides": ready,
            "failed_slides": failed,
            "recent_failures": recent_failures,
            "queue": queue_counts,
        }
