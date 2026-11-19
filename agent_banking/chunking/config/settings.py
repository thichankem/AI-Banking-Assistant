"""Thin loader for chunking settings, rules, and benchmark definitions from config/settings/."""

from copy import deepcopy
from pathlib import Path
import yaml


_SETTINGS_DIR = Path(__file__).resolve().parent / "settings"
_CACHE = {}


def load_yaml_settings(filename):
    """Load and cache a YAML mapping from config/settings/."""
    if filename in _CACHE:
        return deepcopy(_CACHE[filename])
    path = _SETTINGS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError("Chunking settings file not found: " + str(path))
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("Chunking settings must be a mapping: " + filename)
    _CACHE[filename] = value
    return deepcopy(value)


def get_config_section(section, filename="config.yaml"):
    """Return a named dictionary section from a settings YAML file."""
    data = load_yaml_settings(filename)
    if section is None:
        return data
    value = data.get(section)
    if value is None:
        raise KeyError("Missing chunking settings section '" + str(section) + "' in " + filename)
    if not isinstance(value, dict):
        raise ValueError("Chunking settings section must be a mapping: " + str(section))
    return value


def get_rules(section=None):
    return get_config_section(section, "rules.yaml")


def get_golden_suite_settings():
    return load_yaml_settings("golden_suite.yaml")


_settings = load_yaml_settings("config.yaml")


class ChunkingConfig:
    CHUNK_MAX_SIZE = _settings["chunk_max_size"]
    KEEP_TABLE_ATOMIC = _settings["keep_table_atomic"]
    CHUNK_SIZE_TOKENS = _settings["chunk_size_tokens"]
    CHUNK_MAX_TOKENS = _settings["chunk_max_tokens"]
    CHUNK_MIN_TOKENS = _settings["chunk_min_tokens"]
    CHUNK_SEARCH_LANGUAGE = _settings["search_language"]
    OVERLAP_TOKENS = _settings["overlap_tokens"]
    TABLE_ROWS_PER_CHUNK = _settings["table_rows_per_chunk"]
    OCR_WORKERS = _settings["ocr_workers"]
