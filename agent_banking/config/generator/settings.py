import os
from agent_banking.config.models import model_primary

class ContextConfig:
    # 7. Context Synthesizer Configuration (Step 4)
    MAX_CONTEXT_TOKENS = 2000     # Token limit for Markdown context


class GeneratorConfig:
    # 8. LLM Generator & Guardrails Configuration (Step 5)
    LLM_API_KEY = model_primary("generator").api_key
    LLM_BASE_URL = model_primary("generator").url
    LLM_MODEL = model_primary("generator").model
    LLM_TEMPERATURE = 0.2         # Low temperature for factual responses, reducing hallucinations
    LLM_MAX_TOKENS = 2500
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # Maximum retry limit for LangGraph Agent cycle
