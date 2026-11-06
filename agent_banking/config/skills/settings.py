from agent_banking.config.models import model_primary


class SkillsConfig:
    SKILLS_LLM_MODEL = model_primary("skills").model
    SKILLS_LLM_BASE_URL = model_primary("skills").url
    SKILLS_LLM_API_KEY = model_primary("skills").api_key
