# -*- coding: utf-8 -*-
"""LangGraph Agent configuration loading from YAML settings."""
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


class AgentConfig:
    AGENT_LLM_MODEL = model_primary("langgraph_agent").model
    AGENT_LLM_BASE_URL = model_primary("langgraph_agent").url
    AGENT_LLM_API_KEY = model_primary("langgraph_agent").api_key
    MAX_RETRIES = _cfg.get("max_retries", 3)
    HALLUCINATION_RETRIES = _cfg.get("hallucination_retries", 2)
