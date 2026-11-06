import os
from agent_banking.config.models import model_primary

class DecisionConfig:
    # 11. Cấu hình TypeSafe AI Jev Decision Model (System One - Routing & Verification)
    DECISION_API_KEY = os.getenv("DECISION_API_KEY", os.getenv("DECISION_API_KEY", ""))
    DECISION_API_KEY = model_primary("decision").api_key
    DECISION_BASE_URL = model_primary("decision").url
    DECISION_MODEL = model_primary("decision").model
    JEV_TIMEOUT = float(os.getenv("JEV_TIMEOUT", str(model_primary("decision").timeout)))
    ENABLE_JEV_DECISION = True
    if os.getenv("ENABLE_JEV_DECISION", "true").lower() != "true":
        ENABLE_JEV_DECISION = False


