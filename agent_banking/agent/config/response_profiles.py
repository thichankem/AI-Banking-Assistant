"""Response profiles for the ReAct agent, with optional local overrides."""

from importlib import import_module

try:
    _profiles = import_module("agent_banking.agent.config.prompts")
except ModuleNotFoundError as exc:
    if exc.name != "agent_banking.agent.config.prompts":
        raise
    from agent_banking.agent.config import default_profiles as _profiles


PERSONA_PROFILES = _profiles.PERSONA_PROFILES
TONE_PROFILES = _profiles.TONE_PROFILES
FORMAT_PROFILES = _profiles.FORMAT_PROFILES
