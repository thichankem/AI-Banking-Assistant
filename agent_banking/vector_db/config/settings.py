# -*- coding: utf-8 -*-
"""Vector DB configuration loading from YAML settings."""
import os
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


class VectorDBConfig:
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", str(_cfg.get("collection_name", "banking_dual_index_1024")))
    QDRANT_STORAGE_PATH = str(_cfg.get("qdrant_storage_path", "agent_banking/vector_db/qdrant_db"))
