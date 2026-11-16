"""Cloud model settings for the ReAct agent."""

from agent_banking.core.models import model_primary


class AgentConfig:
    _primary = model_primary("agent")
    AGENT_LLM_MODEL = _primary.model
    AGENT_LLM_BASE_URL = _primary.url
    AGENT_LLM_API_KEY = _primary.api_key
