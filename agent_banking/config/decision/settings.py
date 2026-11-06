import os
from agent_banking.config.models import model_primary

class DecisionConfig:
    # 11. Cấu hình TypeSafe AI Jev Decision Model (System One - Routing & Verification)
    TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", os.getenv("JEV_API_KEY", ""))
    JEV_API_KEY = model_primary("decision").api_key
    JEV_BASE_URL = model_primary("decision").url
    JEV_MODEL = model_primary("decision").model
    JEV_TIMEOUT = float(os.getenv("JEV_TIMEOUT", str(model_primary("decision").timeout)))
    ENABLE_JEV_DECISION = True
    if os.getenv("ENABLE_JEV_DECISION", "true").lower() != "true":
        ENABLE_JEV_DECISION = False


