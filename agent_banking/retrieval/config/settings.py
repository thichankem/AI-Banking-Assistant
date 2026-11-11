# -*- coding: utf-8 -*-
"""Retrieval configuration loading from YAML settings."""
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


class RetrievalConfig:
    BM25_K1 = float(_cfg.get("bm25_k1", 1.5))
    BM25_B = float(_cfg.get("bm25_b", 0.75))
    BREADCRUMB_BOOST = float(_cfg.get("breadcrumb_boost", 1.5))
    HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", str(_cfg.get("hybrid_top_k", 5))))
    HYBRID_CANDIDATE_K = int(_cfg.get("hybrid_candidate_k", 15))
    RRF_K = int(_cfg.get("rrf_k", 60))
    DENSE_WEIGHT = float(_cfg.get("dense_weight", 0.5))
