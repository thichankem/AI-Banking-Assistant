# -*- coding: utf-8 -*-
"""Banking tools configuration loading from YAML settings."""
from pathlib import Path
import yaml


def _load_yaml_settings():
    yaml_path = Path(__file__).resolve().parent / "settings" / "config.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


_cfg = _load_yaml_settings()


class ToolsConfig:
    API_TIMEOUT = _cfg.get("api_timeout", 10.0)
    MAX_RESULTS = _cfg.get("max_results", 5)
