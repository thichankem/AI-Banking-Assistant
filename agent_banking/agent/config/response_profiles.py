"""Response profiles and prompts for the ReAct agent, loaded from local prompts config."""

from importlib import import_module

try:
    _profiles = import_module("agent_banking.agent.config.prompts")
except ModuleNotFoundError as exc:
    if exc.name != "agent_banking.agent.config.prompts":
        raise
    from agent_banking.agent.config import default_profiles as _profiles


PERSONA_PROFILES = getattr(_profiles, "PERSONA_PROFILES", {})
TONE_PROFILES = getattr(_profiles, "TONE_PROFILES", {})
FORMAT_PROFILES = getattr(_profiles, "FORMAT_PROFILES", {})
SPECIALIST_PROMPTS = getattr(_profiles, "SPECIALIST_PROMPTS", {})
MESSAGES = getattr(_profiles, "MESSAGES", {})


def get_prompt(name: str, default: str = "") -> str:
    value = getattr(_profiles, name, default)
    if isinstance(value, str):
        return value
    return default


def get_messages(section: str = "") -> dict:
    if not isinstance(MESSAGES, dict):
        return {}
    if not section:
        return MESSAGES
    value = MESSAGES.get(section, {})
    if isinstance(value, dict):
        return value
    return {}
