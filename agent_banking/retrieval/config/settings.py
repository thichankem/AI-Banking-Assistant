# -*- coding: utf-8 -*-
"""Retrieval configuration loading from YAML settings."""
import os
from pathlib import Path
import yaml


def _load_yaml_settings():
    yaml_path = Path(__file__).resolve().parent / "settings" / "config.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing retrieval settings file: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Retrieval settings file must contain a mapping: {yaml_path}")
    return data


_cfg = _load_yaml_settings()

_env_top_k = os.getenv("HYBRID_TOP_K")
if _env_top_k is not None and _env_top_k.strip() != "":
    _hybrid_top_k = int(_env_top_k.strip())
else:
    _hybrid_top_k = int(_cfg["hybrid_top_k"])

_env_candidate_k = os.getenv("HYBRID_CANDIDATE_K")
if _env_candidate_k is not None and _env_candidate_k.strip() != "":
    _hybrid_candidate_k = int(_env_candidate_k.strip())
else:
    _hybrid_candidate_k = int(_cfg["hybrid_candidate_k"])

_raw_bm25_weight = _cfg.get("bm25_weight")
if _raw_bm25_weight is not None:
    _bm25_weight = float(_raw_bm25_weight)
else:
    _bm25_weight = None


class RetrievalConfig:
    BM25_K1 = float(_cfg["bm25_k1"])
    BM25_B = float(_cfg["bm25_b"])
    BREADCRUMB_BOOST = float(_cfg["breadcrumb_boost"])
    HYBRID_TOP_K = _hybrid_top_k
    HYBRID_CANDIDATE_K = _hybrid_candidate_k
    RRF_K = int(_cfg["rrf_k"])
    DENSE_WEIGHT = float(_cfg["dense_weight"])
    RETRIEVAL_BM25_WEIGHT = _bm25_weight
    RETRIEVAL_CANDIDATE_MULTIPLIER = int(_cfg["candidate_multiplier"])
    RETRIEVAL_MIN_CANDIDATE_K = int(_cfg["min_candidate_k"])
    RETRIEVAL_MIN_UNFILTERED_LEG_K = int(_cfg["min_unfiltered_leg_k"])
    RETRIEVAL_DIVERSIFY_MIN_TOP_K = int(_cfg["diversify_min_top_k"])
    RETRIEVAL_DIVERSIFY_MAX_PER_SOURCE = int(_cfg["diversify_max_per_source"])
    RETRIEVAL_SCORE_ROUND_DECIMALS = int(_cfg["score_round_decimals"])
    RETRIEVAL_RRF_ROUND_DECIMALS = int(_cfg["rrf_round_decimals"])
    RETRIEVAL_LATENCY_ROUND_DECIMALS = int(_cfg["latency_round_decimals"])
    RETRIEVAL_FILTER_ALIASES = dict(_cfg["filter_aliases"])
    RETRIEVAL_PUBLICATION_SCOPE_KEY = str(_cfg["publication_scope_key"])
    RETRIEVAL_DEFAULT_INTENT = str(_cfg["default_intent"])
    RETRIEVAL_DEFAULT_SUB_INTENT = str(_cfg["default_sub_intent"])
    RETRIEVAL_DEFAULT_DOMAIN = str(_cfg["default_domain"])
    RETRIEVAL_INPUT_DIR = str(_cfg["input_dir"])
    RETRIEVAL_OUTPUT_DIR = str(_cfg["output_dir"])
    RETRIEVAL_DOWNSTREAM_RERANKING_INPUT_DIR = str(_cfg["downstream_reranking_input_dir"])
    RETRIEVAL_DOWNSTREAM_CANDIDATES_FILENAME = str(_cfg["downstream_candidates_filename"])
    RETRIEVAL_CANDIDATE_INPUT_FILENAMES = tuple(_cfg["candidate_input_filenames"])
    RETRIEVAL_OUTPUT_CANDIDATES_FILENAME = str(_cfg["output_candidates_filename"])
    RETRIEVAL_LATEST_CANDIDATES_FILENAME = str(_cfg["latest_candidates_filename"])
    RETRIEVAL_MANIFEST_FILENAME = str(_cfg["manifest_filename"])
    RETRIEVAL_STATE_TRACKER_FILENAME = str(_cfg["state_tracker_filename"])
    RETRIEVAL_PREVIEW_SNIPPET_CHARS = int(_cfg["preview_snippet_chars"])
    RETRIEVAL_QUERY_PREVIEW_CHARS = int(_cfg["query_preview_chars"])
