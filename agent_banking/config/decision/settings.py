import os
from agent_banking.config.models import model_primary

class DecisionConfig:
    DECISION_API_KEY = model_primary("decision").api_key
    DECISION_BASE_URL = model_primary("decision").url
    DECISION_MODEL = model_primary("decision").model
    DECISION_TIMEOUT = float(os.getenv("DECISION_TIMEOUT", str(model_primary("decision").timeout)))
    ENABLE_DECISION = True
    if os.getenv("ENABLE_DECISION", "true").lower() != "true":
        ENABLE_DECISION = False
