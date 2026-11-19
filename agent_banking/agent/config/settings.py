"""Cloud model and agent runtime settings loaded from local YAML configuration."""

import os
from pathlib import Path
import yaml

from agent_banking.core.models import get_model_chain


_SETTINGS_DIR = Path(__file__).resolve().parent / "settings"


def _load_yaml_file(filename: str) -> dict:
    yaml_path = _SETTINGS_DIR / filename
    if not yaml_path.is_file():
        return {}
    with open(yaml_path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Agent settings file must be a mapping: " + filename)
    return data


_cfg = _load_yaml_file("config.yaml")
_workers_cfg = _load_yaml_file("workers.yaml")


def get_setting(key: str, default=None):
    if key in _cfg:
        return _cfg[key]
    return default


def get_int_setting(key: str, default: int = 0) -> int:
    value = get_setting(key, default)
    if value is None:
        return default
    return int(value)


def get_float_setting(key: str, default: float = 0.0) -> float:
    value = get_setting(key, default)
    if value is None:
        return default
    return float(value)


def get_workers() -> dict:
    raw = _workers_cfg.get("workers", {})
    if not isinstance(raw, dict):
        return {}
    workers = {}
    for role, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        workers[role] = {
            "description": str(spec.get("description", "")),
            "tool": tuple(spec.get("tool", ())),
            "skill": tuple(spec.get("skill", ())),
        }
    return workers


def get_financial_operations() -> dict:
    raw = _cfg.get("financial_operations", {})
    if not isinstance(raw, dict):
        return {}
    operations = {}
    for op_name, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        tool_name = str(spec.get("tool", ""))
        arg_mapping = spec.get("arg_mapping", {})
        if isinstance(arg_mapping, dict) and tool_name:
            operations[op_name] = (tool_name, dict(arg_mapping))
    return operations


class AgentConfig:
    _chain = get_model_chain("agent", routed=False)
    if _chain:
        _primary = _chain[0]
        AGENT_LLM_MODEL = _primary.model
        AGENT_LLM_BASE_URL = _primary.url
        AGENT_LLM_API_KEY = _primary.api_key
    else:
        AGENT_LLM_MODEL = ""
        AGENT_LLM_BASE_URL = ""
        AGENT_LLM_API_KEY = ""
    KNOWLEDGE_PUBLICATION_SCOPE = os.getenv(
        "ABANK_KNOWLEDGE_SCOPE", str(_cfg.get("knowledge_publication_scope", ""))
    )
    ABANK_QDRANT_STORAGE_PATH = os.getenv(
        "ABANK_QDRANT_STORAGE_PATH", str(_cfg.get("qdrant_storage_path", ""))
    )
    MAX_ACTIONS = get_int_setting("max_actions")
    MAX_DELEGATIONS = get_int_setting("max_delegations")
    MAX_WORKER_ACTIONS = get_int_setting("max_worker_actions")
    MAX_REVISIONS = get_int_setting("max_revisions")
    MAX_INTERACTIONS = get_int_setting("max_interactions")
    MAX_ACTIVE_SKILL_BYTES = get_int_setting("max_active_skill_bytes")
    MAX_EVIDENCE = get_int_setting("max_evidence")
    RECENT_MESSAGES = get_int_setting("recent_messages")
    MAX_HISTORY_MESSAGES = get_int_setting("max_history_messages")
    MAX_HISTORY_CHARS = get_int_setting("max_history_chars")
    MAX_NOTES = get_int_setting("max_notes")
    MAX_COMPACT_PER_TURN = get_int_setting("max_compact_per_turn")
    AGENT_CONTEXT_BYTES = int(
        os.getenv("AGENT_CONTEXT_BYTES", str(get_int_setting("context_bytes_default")))
    )
