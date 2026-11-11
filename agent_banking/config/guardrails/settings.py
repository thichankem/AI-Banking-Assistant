import os

class GuardrailsConfig:
    # 10. Enterprise Banking Guardrails Configuration (Multi-Tier Defense)
    ENABLE_SEMANTIC_GUARDRAIL = os.getenv("ENABLE_SEMANTIC_GUARDRAIL", "true").lower() == "true"
    SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.88"))
    ENABLE_INDIRECT_INJECTION_SCAN = os.getenv("ENABLE_INDIRECT_INJECTION_SCAN", "true").lower() == "true"
    MAX_LOAN_AMOUNT_VND = int(os.getenv("MAX_LOAN_AMOUNT_VND", "50000000000"))
    MAX_LOAN_TENURE_MONTHS = int(os.getenv("MAX_LOAN_TENURE_MONTHS", "360"))
    MIN_SAVINGS_AMOUNT_VND = int(os.getenv("MIN_SAVINGS_AMOUNT_VND", "1000000"))
