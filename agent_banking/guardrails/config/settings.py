# -*- coding: utf-8 -*-
"""Guardrails configuration loading from YAML settings and prompts."""

from importlib import import_module
import os
from pathlib import Path
from types import SimpleNamespace
import yaml


_CONFIG_DIR = Path(__file__).resolve().parent
_SETTINGS_DIR = _CONFIG_DIR / "settings"
_PROMPTS_DIR = _CONFIG_DIR / "prompts"


def _load_yaml_file(directory, filename):
    yaml_path = directory / filename
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8-sig") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def load_guardrail_settings():
    """Load scalar thresholds, limits, and runner settings from config.yaml."""
    return _load_yaml_file(_SETTINGS_DIR, "config.yaml")


def load_guardrail_patterns():
    """Load injection, HTML/XSS, leetspeak, and leak patterns from patterns.yaml."""
    return _load_yaml_file(_SETTINGS_DIR, "patterns.yaml")


def load_guardrail_rules():
    """Load domain, fraud, auth-leak, PII, output, and tool rules from rules.yaml."""
    return _load_yaml_file(_SETTINGS_DIR, "rules.yaml")


def load_guardrail_prompts():
    """Load prompt templates and refusal/error messages from config/prompts."""
    try:
        return import_module("agent_banking.guardrails.config.prompts")
    except ModuleNotFoundError as exc:
        if exc.name != "agent_banking.guardrails.config.prompts":
            raise
    refusal_data = _load_yaml_file(_PROMPTS_DIR, "refusal_messages.yaml")
    semantic_data = _load_yaml_file(_PROMPTS_DIR, "semantic_templates.yaml")
    messages_data = _load_yaml_file(_PROMPTS_DIR, "messages.yaml")
    return SimpleNamespace(
        REFUSAL_MESSAGES=dict(refusal_data.get("refusal_messages") or {}),
        DEFAULT_REFUSAL_MESSAGE=str(refusal_data.get("default_refusal_message") or ""),
        EMPTY_QUERY_REFUSAL_MESSAGE=str(refusal_data.get("empty_query_refusal_message") or ""),
        SEMANTIC_ATTACK_TEMPLATES=dict(semantic_data.get("semantic_attack_templates") or {}),
        BENIGN_BANKING_TEMPLATES=list(semantic_data.get("benign_banking_templates") or []),
        REDACTION_TOKENS=dict(messages_data.get("redaction_tokens") or {}),
        TOOL_ERROR_MESSAGES=dict(messages_data.get("tool_error_messages") or {}),
        SEMANTIC_LOG_MESSAGES=dict(messages_data.get("semantic_log_messages") or {}),
        GUARDRAIL_LOG_MESSAGES=dict(messages_data.get("guardrail_log_messages") or {}),
        RUNNER_MESSAGES=dict(messages_data.get("runner_messages") or {}),
    )


def _env_or_cfg(env_key, cfg, cfg_key):
    env_val = os.getenv(env_key)
    if env_val is not None:
        return env_val
    return cfg.get(cfg_key)


def _as_bool(val):
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() == "true"


def _as_int(val):
    if val is None or val == "":
        return 0
    return int(val)


def _as_float(val):
    if val is None or val == "":
        return 0.0
    return float(val)


_cfg = load_guardrail_settings()


class GuardrailsConfig:
    ENABLE_SEMANTIC_GUARDRAIL = _as_bool(_env_or_cfg("ENABLE_SEMANTIC_GUARDRAIL", _cfg, "enable_semantic_guardrail"))
    SEMANTIC_SIMILARITY_THRESHOLD = _as_float(_env_or_cfg("SEMANTIC_SIMILARITY_THRESHOLD", _cfg, "semantic_similarity_threshold"))
    ENABLE_INDIRECT_INJECTION_SCAN = _as_bool(_env_or_cfg("ENABLE_INDIRECT_INJECTION_SCAN", _cfg, "enable_indirect_injection_scan"))
    REQUIRE_INDEX_READY = _as_bool(_env_or_cfg("REQUIRE_INDEX_READY", _cfg, "require_index_ready"))
    MAX_LOAN_AMOUNT_VND = _as_int(_env_or_cfg("MAX_LOAN_AMOUNT_VND", _cfg, "max_loan_amount_vnd"))
    MAX_LOAN_TENURE_MONTHS = _as_int(_env_or_cfg("MAX_LOAN_TENURE_MONTHS", _cfg, "max_loan_tenure_months"))
    MIN_SAVINGS_AMOUNT_VND = _as_int(_env_or_cfg("MIN_SAVINGS_AMOUNT_VND", _cfg, "min_savings_amount_vnd"))
    DECISION_CACHE_MAX_SIZE = _as_int(_env_or_cfg("DECISION_CACHE_MAX_SIZE", _cfg, "decision_cache_max_size"))

    SEMANTIC_SETTINGS = dict(_cfg.get("semantic") or {})
    INDIRECT_INJECTION_SETTINGS = dict(_cfg.get("indirect_injection") or {})
    TOOL_BOUNDS_SETTINGS = dict(_cfg.get("tool_bounds") or {})
    RUNNER_SETTINGS = dict(_cfg.get("runner") or {})
