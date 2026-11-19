# -*- coding: utf-8 -*-
"""Banking tools configuration loading from YAML settings."""

from pathlib import Path
import yaml

_SETTINGS_DIR = Path(__file__).resolve().parent / "settings"


def load_tools_yaml(filename):
    yaml_path = _SETTINGS_DIR / filename
    if yaml_path.is_file():
        try:
            with open(yaml_path, "r", encoding="utf-8-sig") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def get_tools_config():
    return load_tools_yaml("config.yaml")


def get_banking_data():
    return load_tools_yaml("banking_data.yaml")


def get_query_rules():
    return load_tools_yaml("query_rules.yaml")


def get_tool_runtime_config():
    return load_tools_yaml("tool_runtime.yaml")


def get_tool_runtime_override(tool_name):
    """Return optional runtime binding overrides for a built-in tool from config/settings/."""
    overrides = get_tool_runtime_config().get("tool_overrides", {})
    if isinstance(overrides, dict):
        entry = overrides.get(tool_name)
        if isinstance(entry, dict):
            return dict(entry)
    return {}


_cfg = get_tools_config()


class ToolsConfig:
    API_TIMEOUT = _cfg.get("api_timeout")
    MAX_RESULTS = _cfg.get("max_results")
    ENABLE_BANKING_TOOLS = _cfg.get("enable_banking_tools")
