# -*- coding: utf-8 -*-
"""Decision engine configuration loading from YAML settings."""
import os
from pathlib import Path
import yaml
from agent_banking.core.models import model_primary


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


class DecisionConfig:
    DECISION_API_KEY = model_primary("decision").api_key
    DECISION_BASE_URL = model_primary("decision").url
    DECISION_MODEL = model_primary("decision").model
    DECISION_TIMEOUT = float(os.getenv("DECISION_TIMEOUT", str(_cfg.get("timeout", model_primary("decision").timeout))))
    ENABLE_DECISION = _cfg.get("enable_decision", True)
    if os.getenv("ENABLE_DECISION", "true").lower() != "true":
        ENABLE_DECISION = False
