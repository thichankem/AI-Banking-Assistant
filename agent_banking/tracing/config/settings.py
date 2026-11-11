# -*- coding: utf-8 -*-
"""Tracing configuration loading from YAML settings."""
import os
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


class TracingConfig:
    ENABLE_TRACING = os.getenv("ENABLE_TRACING", str(_cfg.get("enable_tracing", True))).lower() == "true"
    ENABLE_PHOENIX = os.getenv("ENABLE_PHOENIX", str(_cfg.get("enable_phoenix", True))).lower() == "true"
    PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", str(_cfg.get("phoenix_collector_endpoint", "")))
    PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", str(_cfg.get("phoenix_project_name", "")))
