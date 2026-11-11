# -*- coding: utf-8 -*-
"""Embedding configuration loading from YAML settings."""
import os
from pathlib import Path
import yaml
from agent_banking.core.models import model_primary


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


class EmbeddingConfig:
    EMBEDDING_MODEL_1024 = model_primary("embedding").model
    EMBEDDING_URL = model_primary("embedding").url
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "")
    EMBEDDING_DIM = _cfg.get("embedding_dim", model_primary("embedding").dimension)
    BATCH_SIZE = _cfg.get("batch_size", 32)
