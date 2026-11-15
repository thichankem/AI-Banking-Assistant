"""Thread-safe memory cache backed by a shared SQLite file."""
from collections import OrderedDict
import json
from pathlib import Path
import sqlite3
from threading import Lock
import time

_SCHEMA_LOCK = Lock()
_TABLE = """
CREATE TABLE IF NOT EXISTS page_vision_cache (
    page_sha256 TEXT PRIMARY KEY,
    layout_json TEXT,
    width INTEGER,
    height INTEGER,
    created_at REAL
) WITHOUT ROWID
"""
_INSERT = """
INSERT OR REPLACE INTO page_vision_cache (page_sha256, layout_json, width, height, created_at)
VALUES (?, ?, ?, ?, ?)
"""


def _enable_wal(connection):
    deadline = time.monotonic() + 30
    while True:
        try:
            connection.execute("PRAGMA journal_mode=WAL").fetchone()
            return
        except sqlite3.OperationalError as error:
            if (getattr(error, "sqlite_errorcode", None) not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                    or time.monotonic() >= deadline):
                raise
            time.sleep(0.05)


def _key_is_valid(key):
    if not isinstance(key, str) or not key:
        return False
    try:
        key.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _encode_entry(key, value, width=0, height=0):
    if not _key_is_valid(key) or value is None:
        return None
    try:
        dimensions = int(width), int(height)
        if any(not 0 <= size <= 2**63 - 1 for size in dimensions):
            return None
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        # Verify the string can be bound by SQLite before starting a write.
        encoded.encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None
    return key, encoded, *dimensions


class PageVisionCache:
    """Store JSON results under their content and configuration fingerprints."""

    def __init__(self, db_path="cache/page_vision_cache.db", max_l1_entries=10000):
        self.db_path = str(db_path)
        self.max_l1_entries = max_l1_entries
        self._l1_cache = OrderedDict()
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0
        self._puts = 0
        self._lock = Lock()
        self._conn = None
        self._data_version = None
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection()

    def _connection(self):
        if self._conn is None:
            connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            try:
                with _SCHEMA_LOCK, connection:
                    mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                    if mode not in ("wal", "memory"):
                        _enable_wal(connection)
                    connection.execute("PRAGMA synchronous=NORMAL")
                    connection.execute("PRAGMA cache_size=-64000")
                    connection.execute("PRAGMA mmap_size=268435456")
                    connection.execute(_TABLE)
            except BaseException:
                connection.close()
                raise
            self._conn = connection
        if self.max_l1_entries > 0:
            data_version = self._conn.execute("PRAGMA data_version").fetchone()[0]
            if data_version != self._data_version:
                self._l1_cache.clear()
                self._data_version = data_version
        return self._conn

    def _remember(self, key, encoded):
        if self.max_l1_entries > 0:
            self._l1_cache[key] = encoded
            self._l1_cache.move_to_end(key)
            if len(self._l1_cache) > self.max_l1_entries:
                self._l1_cache.popitem(last=False)

    def get(self, page_sha256):
        if not _key_is_valid(page_sha256):
            return None
        with self._lock:
            connection = self._connection()
            encoded = self._l1_cache.get(page_sha256)
            if encoded is not None:
                self._l1_cache.move_to_end(page_sha256)
                self._l1_hits += 1
                return json.loads(encoded)
            row = connection.execute(
                "SELECT layout_json FROM page_vision_cache WHERE page_sha256 = ?", (page_sha256,)
            ).fetchone()
            if row is not None:
                try:
                    value = json.loads(row[0])
                except (ValueError, TypeError, RecursionError):
                    value = None
                if value is not None:
                    self._remember(page_sha256, row[0])
                    self._l2_hits += 1
                    return value
            self._misses += 1
            return None

    def put(self, page_sha256, layout_dict, width=0, height=0):
        """Store a single result. Return False for an invalid JSON entry."""
        row = _encode_entry(page_sha256, layout_dict, width, height)
        if row is None:
            return False
        self._put_rows([row])
        return True

    def put_batch(self, items):
        """Store valid (key, layout[, width[, height]]) entries in one transaction."""
        rows = []
        for item in items or ():
            if isinstance(item, (tuple, list)) and 2 <= len(item) <= 4:
                row = _encode_entry(*item)
                if row is not None:
                    rows.append(row)
        return self._put_rows(rows)

    def _put_rows(self, rows):
        if not rows:
            return 0
        created_at = time.time()
        with self._lock:
            connection = self._connection()
            with connection:
                connection.executemany(_INSERT, ((*row, created_at) for row in rows))
            for key, encoded, _, _ in rows:
                self._remember(key, encoded)
            self._puts += len(rows)
        return len(rows)

    def contains(self, page_sha256):
        if not _key_is_valid(page_sha256):
            return False
        with self._lock:
            connection = self._connection()
            return page_sha256 in self._l1_cache or connection.execute(
                "SELECT 1 FROM page_vision_cache WHERE page_sha256 = ?", (page_sha256,)
            ).fetchone() is not None

    def delete(self, page_sha256):
        if not _key_is_valid(page_sha256):
            return False
        with self._lock:
            connection = self._connection()
            with connection:
                connection.execute("DELETE FROM page_vision_cache WHERE page_sha256 = ?", (page_sha256,))
            self._l1_cache.pop(page_sha256, None)
        return True

    def clear(self):
        with self._lock:
            connection = self._connection()
            with connection:
                connection.execute("DELETE FROM page_vision_cache")
            self._l1_cache.clear()

    def stats(self):
        with self._lock:
            connection = self._connection()
            hits = self._l1_hits + self._l2_hits
            total_queries = hits + self._misses
            return {
                "l1_entries": len(self._l1_cache),
                "l2_entries": connection.execute("SELECT COUNT(*) FROM page_vision_cache").fetchone()[0],
                "l1_hits": self._l1_hits,
                "l2_hits": self._l2_hits,
                "misses": self._misses,
                "puts": self._puts,
                "total_queries": total_queries,
                "hit_ratio": round(hits / total_queries, 4) if total_queries else 0.0,
            }

    def close(self):
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._l1_cache.clear()
            self._data_version = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
