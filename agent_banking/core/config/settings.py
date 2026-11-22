# -*- coding: utf-8 -*-
"""
Core, Pipeline, and API Settings Loader.
Loads configuration from settings/config.yaml and prompt bundles from prompts/.
Complies with 0% Regex Law, Zero Ternary Law.
"""

from importlib import import_module
from pathlib import Path
import yaml

_SETTINGS_DIR = Path(__file__).resolve().parent / "settings"
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_yaml_file(path):
    if path.is_file():
        with open(path, "r", encoding="utf-8-sig") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    return {}


def load_core_settings():
    yaml_path = _SETTINGS_DIR / "config.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing core settings file: {yaml_path}")
    data = _load_yaml_file(yaml_path)
    if not data:
        raise ValueError(f"Core settings file must contain a mapping: {yaml_path}")
    return data


def load_core_prompts():
    try:
        return import_module("agent_banking.core.config.prompts")
    except ModuleNotFoundError as exc:
        if exc.name != "agent_banking.core.config.prompts":
            raise
        return None


def get_core_messages(section, filename="core_messages.yaml", prompts_attr=None):
    prompts_mod = load_core_prompts()
    if prompts_mod is not None and prompts_attr:
        value = getattr(prompts_mod, prompts_attr, None)
        if isinstance(value, dict):
            return dict(value)
    data = _load_yaml_file(_PROMPTS_DIR / filename)
    if section:
        sub = data.get(section)
        if isinstance(sub, dict):
            return dict(sub)
        return {}
    return dict(data)


_cfg = load_core_settings()


class CoreConfig:
    """Core, pipeline, and API configuration loaded from YAML settings."""

    CORE_PATHS = dict(_cfg["paths"])
    CORE_MODELS = dict(_cfg["models"])
    CORE_MODEL_ROUTING = dict(_cfg["model_routing"])
    CORE_FALLBACK = dict(_cfg["fallback"])
    CORE_CHAT = dict(_cfg["chat"])
    CORE_CLOUD_RATE_LIMIT = dict(_cfg["cloud_rate_limit"])
    CORE_PLUGIN_SPECS = dict(_cfg["plugin_specs"])
    CORE_SEARCH_TEXT = dict(_cfg["search_text"])
    CORE_STATE_TRACKER = dict(_cfg["state_tracker"])
    CORE_RAG_PIPELINE = dict(_cfg["rag_pipeline"])
    CORE_AGENT_PIPELINE = dict(_cfg["agent_pipeline"])
    CORE_NODE_PIPELINE = dict(_cfg["node_pipeline"])
    CORE_API = dict(_cfg["api"])
