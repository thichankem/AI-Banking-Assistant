import os
from agent_banking.config.models import model_primary

class RerankingConfig:
    # 6. Cấu hình Hybrid Retrieval & Reranking (Tự do tùy chỉnh Top-K)
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))          # Bước 3: Top-K tinh lọc qua Cross-Encoder cho LLM (Mặc định 3)
    RERANKER_MODEL = model_primary("reranking").model

