# -*- coding: utf-8 -*-
"""Model chain loader for skills from YAML settings."""

from pathlib import Path
import yaml


def _load_models():
    yaml_path = Path(__file__).resolve().parent / "settings" / "models.yaml"
    if yaml_path.is_file():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            models = data.get("models", [])
            if isinstance(models, list):
                return models
    return []


MODELS = _load_models()
