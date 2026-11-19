# -*- coding: utf-8 -*-
"""Embedding configuration loading from YAML settings."""
import os
from pathlib import Path
import yaml
from agent_banking.core.models import model_primary


def _load_yaml_settings():
    yaml_path = Path(__file__).resolve().parent / "settings" / "config.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing embedding settings file: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Embedding settings file must contain a mapping: {yaml_path}")
    return data


_cfg = _load_yaml_settings()
_primary = model_primary("embedding")

_env_dim = os.getenv("EMBEDDING_DIM")
if _env_dim is not None and _env_dim.strip() != "":
    _embedding_dim = int(_env_dim.strip())
elif "embedding_dim" in _cfg:
    _embedding_dim = int(_cfg["embedding_dim"])
else:
    _embedding_dim = int(_primary.dimension)

_env_local_model = os.getenv("LOCAL_EMBEDDING_MODEL")
if _env_local_model is not None and _env_local_model.strip() != "":
    _local_embedding_model = _env_local_model.strip()
else:
    _local_embedding_model = str(_cfg["local_embedding_model"])


class EmbeddingConfig:
    EMBEDDING_MODEL_1024 = _primary.model
    EMBEDDING_URL = _primary.url
    LOCAL_EMBEDDING_MODEL = _local_embedding_model
    EMBEDDING_DIM = _embedding_dim
    BATCH_SIZE = int(_cfg["batch_size"])
    EMBEDDING_NORMALIZE = bool(_cfg["normalize_embeddings"])
    EMBEDDING_DEVICE = str(_cfg["device"])
    EMBEDDING_CACHE_ENABLED = bool(_cfg["cache_enabled"])
    EMBEDDING_CACHE_DB_PATH = str(_cfg["cache_db_path"])
    EMBEDDING_SQLITE_TIMEOUT_SEC = float(_cfg["sqlite_timeout_sec"])
    EMBEDDING_SQLITE_JOURNAL_MODE = str(_cfg["sqlite_journal_mode"])
    EMBEDDING_SQLITE_SYNCHRONOUS = str(_cfg["sqlite_synchronous"])
    EMBEDDING_SQLITE_BUSY_TIMEOUT_MS = int(_cfg["sqlite_busy_timeout_ms"])
    EMBEDDING_SQLITE_CACHE_SIZE_KB = int(_cfg["sqlite_cache_size_kb"])
    EMBEDDING_SQLITE_MMAP_SIZE_BYTES = int(_cfg["sqlite_mmap_size_bytes"])
    EMBEDDING_CACHE_KEY_VERSION = str(_cfg["cache_key_version"])
    EMBEDDING_CACHE_QUERY_BATCH_SIZE = int(_cfg["cache_query_batch_size"])
    EMBEDDING_MAX_PASSAGE_TOKENS = int(_cfg["max_passage_tokens"])
    EMBEDDING_MIN_PASSAGE_TOKENS_LIMIT = int(_cfg["min_passage_tokens_limit"])
    EMBEDDING_MAX_PASSAGE_TOKENS_LIMIT = int(_cfg["max_passage_tokens_limit"])
    EMBEDDING_ENCODING_FORMAT = str(_cfg["encoding_format"])
    EMBEDDING_INPUT_TEXT_TRUNCATE = str(_cfg["input_text_truncate"])
    EMBEDDING_L2_NORM_EPSILON = float(_cfg["l2_norm_epsilon"])
    EMBEDDING_MAX_PARENT_SUMMARY_CHARS = int(_cfg["max_parent_summary_chars"])
    EMBEDDING_QUERY_PREFIX = str(_cfg["query_prefix"])
    EMBEDDING_PASSAGE_PREFIX = str(_cfg["passage_prefix"])
    EMBEDDING_LEGACY_PREFIX_MARKERS = tuple(_cfg["legacy_prefix_markers"])
    EMBEDDING_IMAGE_CAPTION_FIELDS = tuple(_cfg["image_caption_fields"])
    EMBEDDING_PASSAGE_LABELS = dict(_cfg["labels"])
    EMBEDDING_INPUT_DIR = str(_cfg["input_dir"])
    EMBEDDING_OUTPUT_DIR = str(_cfg["output_dir"])
    EMBEDDING_CHUNKING_OUTPUT_DIR = str(_cfg["chunking_output_dir"])
    EMBEDDING_CHUNKING_MANIFEST_FILENAME = str(_cfg["chunking_manifest_filename"])
    EMBEDDING_ALL_CHUNKS_FILENAME = str(_cfg["all_chunks_filename"])
    EMBEDDING_INPUT_CHUNKS_FILENAME = str(_cfg["input_chunks_filename"])
    EMBEDDING_OUTPUT_CHUNKS_FILENAME = str(_cfg["output_chunks_filename"])
    EMBEDDING_SAMPLE_OUTPUT_FILENAME = str(_cfg["sample_output_filename"])
    EMBEDDING_OUTPUT_MANIFEST_FILENAME = str(_cfg["output_manifest_filename"])
    EMBEDDING_STATE_TRACKER_FILENAME = str(_cfg["state_tracker_filename"])
    EMBEDDING_DEFAULT_SOURCE_NAME = str(_cfg["default_source_name"])
    EMBEDDING_SAMPLE_COUNT = int(_cfg["sample_count"])
    EMBEDDING_SAMPLE_VECTOR_PREVIEW_DIMS = int(_cfg["sample_vector_preview_dims"])
