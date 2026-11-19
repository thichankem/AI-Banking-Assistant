# -*- coding: utf-8 -*-
"""Vector DB configuration loading from YAML settings."""
import os
from pathlib import Path
import yaml


def _load_yaml_settings():
    yaml_path = Path(__file__).resolve().parent / "settings" / "config.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing vector_db settings file: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Vector DB settings file must contain a mapping: {yaml_path}")
    return data


_cfg = _load_yaml_settings()

_env_collection = os.getenv("QDRANT_COLLECTION")
if _env_collection is not None and _env_collection.strip() != "":
    _qdrant_collection = _env_collection.strip()
else:
    _qdrant_collection = str(_cfg["collection_name"])


class VectorDBConfig:
    QDRANT_COLLECTION = _qdrant_collection
    QDRANT_STORAGE_PATH = str(_cfg["qdrant_storage_path"])
    QDRANT_DISTANCE_METRIC = str(_cfg["distance_metric"])
    QDRANT_VECTOR_SIZE = int(_cfg["vector_size"])
    QDRANT_HNSW_M = int(_cfg["hnsw_m"])
    QDRANT_HNSW_EF_CONSTRUCT = int(_cfg["hnsw_ef_construct"])
    VECTOR_DB_PAYLOAD_INDEX_FIELDS = list(_cfg["payload_index_fields"])
    VECTOR_DB_UPSERT_BATCH_SIZE = int(_cfg["upsert_batch_size"])
    VECTOR_DB_SCROLL_BATCH_SIZE = int(_cfg["scroll_batch_size"])
    VECTOR_DB_MUTATE_BATCH_SIZE = int(_cfg["mutate_batch_size"])
    VECTOR_DB_SCROLL_LIMIT_BUFFER = int(_cfg["scroll_limit_buffer"])
    VECTOR_DB_MIN_SCROLL_LIMIT = int(_cfg["min_scroll_limit"])
    VECTOR_DB_BM25_K1 = float(_cfg["bm25_k1"])
    VECTOR_DB_BM25_B = float(_cfg["bm25_b"])
    VECTOR_DB_BREADCRUMB_BOOST = float(_cfg["breadcrumb_boost"])
    VECTOR_DB_BM25_FIELD_WEIGHTS = dict(_cfg["bm25_field_weights"])
    VECTOR_DB_DEFAULT_TOP_K = int(_cfg["default_top_k"])
    VECTOR_DB_DEFAULT_CANDIDATE_K = int(_cfg["default_candidate_k"])
    VECTOR_DB_DEFAULT_ALPHA = float(_cfg["default_alpha"])
    VECTOR_DB_DEFAULT_RRF_K = int(_cfg["default_rrf_k"])
    VECTOR_DB_SCORE_ROUND_DECIMALS = int(_cfg["score_round_decimals"])
    VECTOR_DB_RRF_ROUND_DECIMALS = int(_cfg["rrf_round_decimals"])
    VECTOR_DB_TABLE_DICT_VALUE_KEYS = tuple(_cfg["table_dict_value_keys"])
    VECTOR_DB_TABLE_META_KEYS = tuple(_cfg["table_meta_keys"])
    VECTOR_DB_TABLE_SKIP_KEYS = tuple(_cfg["table_skip_keys"])
    VECTOR_DB_IMAGE_TEXT_FIELDS = tuple(_cfg["image_text_fields"])
    VECTOR_DB_INPUT_DIR = str(_cfg["input_dir"])
    VECTOR_DB_OUTPUT_DIR = str(_cfg["output_dir"])
    VECTOR_DB_EMBEDDING_OUTPUT_CHUNKS_PATH = str(_cfg["embedding_output_chunks_path"])
    VECTOR_DB_INPUT_CHUNKS_FILENAME = str(_cfg["input_chunks_filename"])
    VECTOR_DB_STATE_TRACKER_FILENAME = str(_cfg["state_tracker_filename"])
    VECTOR_DB_MANIFEST_FILENAME = str(_cfg["manifest_filename"])
