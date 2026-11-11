# -*- coding: utf-8 -*-
"""Tasks configuration loading from YAML settings."""
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


class TasksConfig:
    TASKS_LLM_MODEL = model_primary("tasks").model
    TASKS_LLM_BASE_URL = model_primary("tasks").url
    TASKS_LLM_API_KEY = model_primary("tasks").api_key
    DEFAULT_TASK = _cfg.get("default_task", "TASK_POLICY_INQUIRY")
    MAX_TURNS = _cfg.get("max_turns", 5)
