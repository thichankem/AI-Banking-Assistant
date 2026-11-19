"""Coordinate cloud request pacing across chat, OCR and worker processes."""

from hashlib import sha256
import math
import os
from pathlib import Path
import sqlite3
import time

from agent_banking.core.config.settings import CoreConfig, get_core_messages
from agent_banking.core.environment import ROOT_DIR

_RATE_CFG = CoreConfig.CORE_CLOUD_RATE_LIMIT
_RATE_ENV = _RATE_CFG["env_var"]
_DEFAULT_RPM = float(_RATE_CFG["default_requests_per_minute"])
_DB_RELPATH = _RATE_CFG["database_relpath"]
_SQLITE_TIMEOUT = float(_RATE_CFG["sqlite_timeout_sec"])
_POLL_MIN_SLEEP = float(_RATE_CFG["poll_min_sleep_sec"])
_POLL_MAX_SLEEP = float(_RATE_CFG["poll_max_sleep_sec"])
_MAX_RETRY_DELAY = float(_RATE_CFG["max_retry_delay_sec"])
_ENDPOINT_SUFFIXES = tuple(_RATE_CFG["endpoint_suffixes"])
_MESSAGES = get_core_messages("cloud_rate_limit", "core_messages.yaml", "CORE_RATE_LIMIT_MESSAGES")


def _default_database_path():
    target = ROOT_DIR
    for part in str(_DB_RELPATH).replace("\\", "/").split("/"):
        if part:
            target = target / part
    return target


def acquire_cloud_slot(model, endpoint, *, database=None):
    rate = float(os.getenv(_RATE_ENV, str(_DEFAULT_RPM)))
    if not math.isfinite(rate) or rate < 0:
        raise ValueError(_MESSAGES["invalid_rate_error"])
    if rate == 0:
        return
    if database is None:
        database = _default_database_path()
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    # Only a hash of provider identity is persisted; never store credentials.
    provider = endpoint
    for suffix in _ENDPOINT_SUFFIXES:
        provider = provider.removesuffix(suffix)
    provider = provider.rstrip("/")
    identity = sha256((provider + "\n" + model).encode("utf-8")).hexdigest()
    interval = 60 / rate
    while True:
        connection = sqlite3.connect(database, timeout=_SQLITE_TIMEOUT)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS cloud_slots (identity TEXT PRIMARY KEY, next_at REAL NOT NULL)")
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT next_at FROM cloud_slots WHERE identity=?", (identity,)).fetchone()
            now = time.time()
            due = now
            if row:
                due = min(row[0], now + interval)
            if due <= now:
                connection.execute("INSERT OR REPLACE INTO cloud_slots VALUES (?, ?)", (identity, now + interval))
                connection.commit()
                return
            connection.rollback()
        finally:
            connection.close()
        time.sleep(min(_POLL_MAX_SLEEP, max(_POLL_MIN_SLEEP, due - now)))


def retry_delay(error, default):
    cause = getattr(error, "last_error", error)
    headers = getattr(getattr(cause, "response", None), "headers", {})
    value = headers.get("retry-after", getattr(cause, "retry_after", default))
    try:
        delay = float(value)
    except (ValueError, TypeError):
        return default
    if not math.isfinite(delay) or delay < 0:
        return default
    return min(_MAX_RETRY_DELAY, max(default, delay))
