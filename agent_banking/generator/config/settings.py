# -*- coding: utf-8 -*-
"""Generator configuration loading from YAML settings."""
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


class ContextConfig:
    MAX_CONTEXT_TOKENS = _cfg.get("max_context_tokens", 2000)


class GeneratorConfig:
    LLM_API_KEY = model_primary("generator").api_key
    LLM_BASE_URL = model_primary("generator").url
    LLM_MODEL = model_primary("generator").model
    LLM_TEMPERATURE = float(_cfg.get("llm_temperature", 0.2))
    LLM_MAX_TOKENS = int(_cfg.get("llm_max_tokens", 2500))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", str(_cfg.get("max_retries", 2))))
