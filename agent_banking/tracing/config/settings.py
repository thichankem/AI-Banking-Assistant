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
    """Load tracing, Phoenix, span, and telemetry settings from config.yaml."""
    return _load_yaml_file(_SETTINGS_DIR, "config.yaml")


def load_tracing_prompts():
    """Load span labels, Phoenix log templates, and runner messages from config/prompts."""
    try:
        return import_module("agent_banking.tracing.config.prompts")
    except ModuleNotFoundError as exc:
        if exc.name != "agent_banking.tracing.config.prompts":
            raise
    messages_data = _load_yaml_file(_PROMPTS_DIR, "messages.yaml")
    return SimpleNamespace(
        SPAN_LABELS=dict(messages_data.get("span_labels") or {}),
        PHOENIX_MESSAGES=dict(messages_data.get("phoenix_messages") or {}),
        TERMINAL_SUMMARY_TEMPLATES=dict(messages_data.get("terminal_summary") or {}),
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


def _as_str(val):
    if val is None:
        return ""
    return str(val)


_cfg = load_tracing_settings()
_trace_dir_name = _as_str(_cfg.get("trace_dir_name"))
if not _trace_dir_name:
    _trace_dir_name = "traces"


class TracingConfig:
    ENABLE_TRACING = _as_bool(_env_or_cfg("ENABLE_TRACING", _cfg, "enable_tracing"))
    ENABLE_PHOENIX = _as_bool(_env_or_cfg("ENABLE_PHOENIX", _cfg, "enable_phoenix"))
    PHOENIX_COLLECTOR_ENDPOINT = _as_str(_env_or_cfg("PHOENIX_COLLECTOR_ENDPOINT", _cfg, "phoenix_collector_endpoint"))
    PHOENIX_PROJECT_NAME = _as_str(_env_or_cfg("PHOENIX_PROJECT_NAME", _cfg, "phoenix_project_name"))
    TRACE_DIR = Path(__file__).resolve().parent.parent / _trace_dir_name

    RAG_TRACES_FILENAME = _as_str(_cfg.get("rag_traces_filename"))
    AGENT_RUNS_FILENAME = _as_str(_cfg.get("agent_runs_filename"))
    LOGGER_NAME = _as_str(_cfg.get("logger_name"))

    PHOENIX_SETTINGS = dict(_cfg.get("phoenix") or {})
    SPAN_SETTINGS = dict(_cfg.get("spans") or {})
    DEFAULT_TRACE_SETTINGS = dict(_cfg.get("defaults") or {})
    FORMATTING_SETTINGS = dict(_cfg.get("formatting") or {})
    AGENT_TRACE_SETTINGS = dict(_cfg.get("agent_trace") or {})
    RUNNER_SETTINGS = dict(_cfg.get("runner") or {})
