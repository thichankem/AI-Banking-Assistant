# -*- coding: utf-8 -*-
"""Skills configuration loading from YAML settings."""

from pathlib import Path
import yaml
from agent_banking.core.models import model_primary

_SETTINGS_DIR = Path(__file__).resolve().parent / "settings"


def load_skills_yaml(filename):
    yaml_path = _SETTINGS_DIR / filename
    if yaml_path.is_file():
        try:
            with open(yaml_path, "r", encoding="utf-8-sig") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def get_skills_config():
    return load_skills_yaml("config.yaml")


def get_skill_runtime_config():
    return load_skills_yaml("skill_runtime.yaml")


def get_skill_runtime_override(skill_id):
    """Return optional runtime binding override for a built-in skill from config/settings/."""
    merged = {}
    overrides = load_skills_yaml("skill_overrides.yaml").get("skills", {})
    if isinstance(overrides, dict):
        entry = overrides.get(skill_id)
        if isinstance(entry, dict):
            merged.update(entry)
    per_skill = _SETTINGS_DIR / "skills" / skill_id / "runtime.yaml"
    if per_skill.is_file():
        try:
            with open(per_skill, "r", encoding="utf-8-sig") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    merged.update(data)
        except Exception:
            pass
    return merged


_cfg = get_skills_config()
_primary = model_primary("skills")


class SkillsConfig:
    SKILLS_LLM_MODEL = _primary.model
    SKILLS_LLM_BASE_URL = _primary.url
    SKILLS_LLM_API_KEY = _primary.api_key
    ENABLE_SKILLS = _cfg.get("enable_skills")
    MAX_EXECUTION_TIMEOUT = _cfg.get("max_execution_timeout")
    AUDIT_PASS_THRESHOLD = _cfg.get("audit_pass_threshold")
