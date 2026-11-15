"""Chunking settings used by the shared pipeline configuration."""
from pathlib import Path


def _load_settings():
    path = Path(__file__).resolve().parent / "settings" / "config.yaml"
    if not path.is_file():
        return {}
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Chunking settings must be a mapping")
    return value


_settings = _load_settings()


class ChunkingConfig:
    CHUNK_MAX_SIZE = _settings.get("chunk_max_size", 800)
    KEEP_TABLE_ATOMIC = _settings.get("keep_table_atomic", True)
    CHUNK_SIZE_TOKENS = _settings.get("chunk_size_tokens", 512)
    CHUNK_MAX_TOKENS = _settings.get("chunk_max_tokens", 1024)
    CHUNK_MIN_TOKENS = _settings.get("chunk_min_tokens", 128)
    CHUNK_SEARCH_LANGUAGE = _settings.get("search_language", "vi")
    OVERLAP_TOKENS = _settings.get("overlap_tokens", 64)
