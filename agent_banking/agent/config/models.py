"""Model specifications loaded from local YAML configuration."""

from pathlib import Path
import yaml


def _load_models() -> list:
    yaml_path = Path(__file__).resolve().parent / "settings" / "models.yaml"
    if not yaml_path.is_file():
        return []
    with open(yaml_path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Agent models configuration must be a mapping")
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError("Agent models configuration 'models' must be a list")
    return list(models)


MODELS = _load_models()
