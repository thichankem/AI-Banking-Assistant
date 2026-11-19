"""Only equivalent embedding spaces may fail over against an existing index."""

from pathlib import Path
import yaml


def _load_models():
    yaml_path = Path(__file__).resolve().parent / "settings" / "models.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing embedding models file: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError(f"Embedding models file must define a 'models' list: {yaml_path}")
    return data["models"]


MODELS = _load_models()
