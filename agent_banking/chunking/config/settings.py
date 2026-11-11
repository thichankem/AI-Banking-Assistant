# -*- coding: utf-8 -*-
"""Chunking configuration loading from YAML settings."""
from pathlib import Path
import yaml


def _load_yaml_settings():
    yaml_path = Path(__file__).resolve().parent / "settings" / "config.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


_cfg = _load_yaml_settings()


class ChunkingConfig:
    CHUNK_MAX_SIZE = _cfg.get("chunk_max_size", 800)
    KEEP_TABLE_ATOMIC = _cfg.get("keep_table_atomic", True)
    CHUNK_SIZE_TOKENS = _cfg.get("chunk_size_tokens", 512)
    OVERLAP_TOKENS = _cfg.get("overlap_tokens", 64)
