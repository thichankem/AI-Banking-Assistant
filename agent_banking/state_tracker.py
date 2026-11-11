"""
Incremental Data State and Deduplication Tracker.
================================================================================
Controls input and output state for each module in the AI Banking Assistant system:
1. Computes SHA-256 hash of input file, text, or dictionary object.
2. Compares against the state ledger stored in the output directory.
3. Eliminates duplicate computation and redundant API calls.
4. Executes processing only for new or modified data.

MANDATORY RULES:
1. NO __INIT__.PY: Import module files directly.
2. NO TYPING LIBRARY: Use Python native types only.
3. ZERO TERNARY: Explicit if/else blocks only (no ?: or conditional expressions).
4. 0% REGEX LAW: Strictly string methods only (in, split, startswith, endswith, strip, lower, replace, find).
"""

from __future__ import annotations
import os
import json
import time
import hashlib
from pathlib import Path


class StateTracker:
    """
    Incremental state tracking and deduplication engine (0% Regex, Zero Ternary).
    """

    def __init__(self, state_file_path: str | Path):
        self.state_file_path = Path(state_file_path)
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.load()

    def load(self) -> None:
        """Load state records from JSON file."""
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = {}
        else:
            self.records = {}

    def save(self) -> None:
        """Save state records to JSON file."""
        try:
            with open(self.state_file_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[StateTracker] Warning: Failed to save state file {self.state_file_path}: {e}")

    @staticmethod
    def compute_file_hash(file_path: str | Path) -> str:
        """Compute SHA-256 hash of a file on disk."""
        p = Path(file_path)
        if not p.exists():
            return ""
        if not p.is_file():
            return ""

        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute SHA-256 hash of a text string."""
        if text is not None:
            clean = str(text).strip()
        else:
            clean = ""
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_dict_hash(data: dict) -> str:
        """Compute SHA-256 hash of a dictionary / JSON object."""
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def is_processed(self, item_id: str, content_hash: str) -> bool:
        """Check whether item_id has already been processed with the given content_hash."""
        if item_id in self.records:
            recorded = self.records[item_id]
            rec_hash = recorded.get("hash")
            if rec_hash == content_hash:
                return True
            else:
                return False
        else:
            return False

    def are_all_items_processed(self, items_dict: dict) -> bool:
        """
        Check whether all items in {item_id: content_hash} have already been processed.
        Returns True if every item is recorded and its hash matches, False otherwise.
        """
        if not items_dict:
            return False
        for item_id, content_hash in items_dict.items():
            if not self.is_processed(item_id, content_hash):
                return False
        return True

    def record_processed(self, item_id: str, content_hash: str, metadata: dict | None = None) -> None:
        """Record successful processing of item_id with metadata."""
        if metadata is not None:
            meta = dict(metadata)
        else:
            meta = {}
        meta["hash"] = content_hash
        meta["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.records[item_id] = meta

    def get_record(self, item_id: str) -> dict | None:
        """Retrieve recorded metadata for item_id."""
        if item_id in self.records:
            return self.records[item_id]
        else:
            return None

    def clear(self) -> None:
        """Clear all records (used when force re-running)."""
        self.records = {}
        self.save()
