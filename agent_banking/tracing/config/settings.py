# -*- coding: utf-8 -*-
"""Tracing configuration loading from YAML settings and prompts."""

from importlib import import_module
import os
from pathlib import Path
from types import SimpleNamespace
import yaml

from agent_banking.core.environment import load_environment


load_environment()

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


def load_tracing_settings():
    """Load LangSmith tracing, span, and telemetry settings from config.yaml."""
    return _load_yaml_file(_SETTINGS_DIR, "config.yaml")


def load_tracing_prompts():
    """Load span labels, LangSmith log templates, and runner messages from config/prompts."""
    try:
        return import_module("agent_banking.tracing.config.prompts")
    except ModuleNotFoundError as exc:
        if exc.name != "agent_banking.tracing.config.prompts":
            raise
    messages_data = _load_yaml_file(_PROMPTS_DIR, "messages.yaml")
    return SimpleNamespace(
        SPAN_LABELS=dict(messages_data.get("span_labels") or {}),
        LANGSMITH_MESSAGES=dict(messages_data.get("langsmith_messages") or {}),
        TERMINAL_SUMMARY_TEMPLATES=dict(messages_data.get("terminal_summary") or {}),
        RUNNER_MESSAGES=dict(messages_data.get("runner_messages") or {}),
    )


def _env_or_cfg(env_keys, cfg, cfg_key):
    if isinstance(env_keys, str):
        keys = (env_keys,)
    else:
        keys = tuple(env_keys)
    for env_key in keys:
        env_val = os.getenv(env_key)
        if env_val is not None and str(env_val).strip() != "":
            return env_val
    return cfg.get(cfg_key)


def _as_bool(val):
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() == "true"


def _as_str(val):
    if val is None:
        return ""
    return str(val).strip()


_cfg = load_tracing_settings()
_trace_dir_name = _as_str(_cfg.get("trace_dir_name"))
if not _trace_dir_name:
    _trace_dir_name = "traces"


class TracingConfig:
    ENABLE_TRACING = _as_bool(_env_or_cfg("ENABLE_TRACING", _cfg, "enable_tracing"))
    LANGSMITH_TRACING = _as_bool(
        _env_or_cfg(("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"), _cfg, "langsmith_tracing")
    )
    LANGSMITH_API_KEY = _as_str(_env_or_cfg(("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"), {}, ""))
    LANGSMITH_ENDPOINT = _as_str(
        _env_or_cfg(("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"), _cfg, "langsmith_endpoint")
    )
    LANGSMITH_PROJECT = _as_str(
        _env_or_cfg(("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"), _cfg, "langsmith_project")
    )
    LANGSMITH_WEB_URL = _as_str(_env_or_cfg("LANGSMITH_WEB_URL", _cfg, "langsmith_web_url"))
    LOCAL_TRACE_EXPORT = _as_bool(_env_or_cfg("LOCAL_TRACE_EXPORT", _cfg, "local_trace_export"))
    TRACE_DIR = Path(__file__).resolve().parent.parent / _trace_dir_name

    LANGSMITH_RUNS_FILENAME = _as_str(_cfg.get("langsmith_runs_filename"))
    LOGGER_NAME = _as_str(_cfg.get("logger_name"))

    LANGSMITH_SETTINGS = dict(_cfg.get("langsmith") or {})
    SPAN_SETTINGS = dict(_cfg.get("spans") or {})
    DEFAULT_TRACE_SETTINGS = dict(_cfg.get("defaults") or {})
    FORMATTING_SETTINGS = dict(_cfg.get("formatting") or {})
    RUN_METADATA_SETTINGS = dict(_cfg.get("run_metadata") or {})
    RUNNER_SETTINGS = dict(_cfg.get("runner") or {})
