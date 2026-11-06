import os
from agent_banking.config.models import model_primary

class ContextConfig:
    # 7. Cấu hình Context Synthesizer (Bước 4)
    MAX_CONTEXT_TOKENS = 2000     # Kiểm soát giới hạn token cho ngữ cảnh Markdown



class GeneratorConfig:
    # 8. Cấu hình LLM Generator & Guardrails (Bước 5)
    LLM_API_KEY = model_primary("generator").api_key
    LLM_BASE_URL = model_primary("generator").url
    LLM_MODEL = model_primary("generator").model
    LLM_TEMPERATURE = 0.2         # Nhiệt độ thấp để trả lời chính xác, tránh hallucination
    LLM_MAX_TOKENS = 2500
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # Giới hạn số lần thử lại tối đa cho chu trình LangGraph Agent

