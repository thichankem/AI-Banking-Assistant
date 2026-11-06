from agent_banking.config.models import model_primary


class AgentConfig:
    AGENT_LLM_MODEL = model_primary("langgraph_agent").model
    AGENT_LLM_BASE_URL = model_primary("langgraph_agent").url
    AGENT_LLM_API_KEY = model_primary("langgraph_agent").api_key
