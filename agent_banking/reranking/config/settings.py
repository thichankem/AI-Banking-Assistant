# -*- coding: utf-8 -*-
"""Reranking configuration loading from YAML settings."""
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


class RerankingConfig:
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", str(_cfg.get("rerank_top_k", 3))))
    RERANKER_MODEL = model_primary("reranking").model
