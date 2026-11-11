# -*- coding: utf-8 -*-
"""Skills configuration loading from YAML settings."""
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


class SkillsConfig:
    SKILLS_LLM_MODEL = model_primary("skills").model
    SKILLS_LLM_BASE_URL = model_primary("skills").url
    SKILLS_LLM_API_KEY = model_primary("skills").api_key
    MAX_EXECUTION_TIMEOUT = _cfg.get("max_execution_timeout", 30.0)
