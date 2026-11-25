# -*- coding: utf-8 -*-
"""Reranking configuration loading from YAML settings."""
import os
from pathlib import Path
import yaml
from agent_banking.core.models import model_primary


def _load_yaml_settings():
    yaml_path = Path(__file__).resolve().parent / "settings" / "config.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing reranking settings file: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Reranking settings file must contain a mapping: {yaml_path}")
    return data


_cfg = _load_yaml_settings()
_primary = model_primary("reranking")

_env_rerank_top_k = os.getenv("RERANK_TOP_K")
if _env_rerank_top_k is not None and _env_rerank_top_k.strip() != "":
    _rerank_top_k = int(_env_rerank_top_k.strip())
else:
    _rerank_top_k = int(_cfg["rerank_top_k"])


class RerankingConfig:
    RERANK_TOP_K = _rerank_top_k
    RERANKER_MODEL = _primary.model
    RERANK_SCORE_THRESHOLD = float(_cfg["score_threshold"])
    RERANK_BATCH_SIZE = int(_cfg["batch_size"])
    RERANK_MAX_DOC_CHARS = int(_cfg["max_doc_chars"])
    RERANK_SCORE_MIN = float(_cfg["score_min"])
    RERANK_SCORE_MAX = float(_cfg["score_max"])
    RERANK_SCORE_ROUND_DECIMALS = int(_cfg["score_round_decimals"])
    RERANK_DEDUP_SNIPPET_CHARS = int(_cfg["dedup_snippet_chars"])
    RERANK_DIVERSITY_MIN_TOP_K = int(_cfg["diversity_min_top_k"])
    RERANK_DIVERSITY_MAX_PER_SOURCE = int(_cfg["diversity_max_per_source"])
    RERANK_DOC_HEADER_PREFIX = str(_cfg["doc_header_prefix"])
    RERANK_CONTEXT_PREFIX_MARKER = str(_cfg["context_prefix_marker"])
    RERANK_CACHE_ENABLED = bool(_cfg.get("cache_enabled", True))
    RERANK_CACHE_DB_PATH = str(_cfg.get("cache_db_path", "agent_banking/vector_db/rerank_cache.db"))
    RERANK_CACHE_KEY_VERSION = str(_cfg.get("cache_key_version", "v1"))
    RERANK_SQLITE_TIMEOUT_SEC = float(_cfg.get("sqlite_timeout_sec", 5.0))
    RERANK_SQLITE_BUSY_TIMEOUT_MS = int(_cfg.get("sqlite_busy_timeout_ms", 5000))
    RERANK_LEXICAL_BREADCRUMB_MULTIPLIER = float(_cfg["lexical_breadcrumb_multiplier"])
    RERANK_LEXICAL_BREADCRUMB_OVERLAP_WEIGHT = float(_cfg["lexical_breadcrumb_overlap_weight"])
    RERANK_METADATA_BONUS = dict(_cfg["metadata_bonus"])
    RERANKING_INPUT_DIR = str(_cfg["input_dir"])
    RERANKING_OUTPUT_DIR = str(_cfg["output_dir"])
    RERANKING_CANDIDATES_FILENAME = str(_cfg["candidates_filename"])
    RERANKING_RETRIEVAL_OUTPUT_CANDIDATES_PATH = str(_cfg["retrieval_output_candidates_path"])
    RERANKING_DOWNSTREAM_GENERATOR_INPUT_DIR = str(_cfg["downstream_generator_input_dir"])
    RERANKING_DOWNSTREAM_OUTPUT_FILENAME = str(_cfg["downstream_output_filename"])
    RERANKING_OUTPUT_CHUNKS_FILENAME = str(_cfg["output_chunks_filename"])
    RERANKING_LATEST_OUTPUT_FILENAME = str(_cfg["latest_output_filename"])
    RERANKING_MANIFEST_FILENAME = str(_cfg["manifest_filename"])
    RERANKING_STATE_TRACKER_FILENAME = str(_cfg["state_tracker_filename"])
    RERANKING_CANDIDATE_TEXT_HASH_LEN = int(_cfg["candidate_text_hash_len"])
    RERANKING_QUERY_PREVIEW_CHARS = int(_cfg["query_preview_chars"])
