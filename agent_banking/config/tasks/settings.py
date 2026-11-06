from agent_banking.config.models import model_primary


class TasksConfig:
    TASKS_LLM_MODEL = model_primary("tasks").model
    TASKS_LLM_BASE_URL = model_primary("tasks").url
    TASKS_LLM_API_KEY = model_primary("tasks").api_key
