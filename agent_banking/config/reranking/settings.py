import os
from agent_banking.config.models import model_primary

class RerankingConfig:
    # 6. Hybrid Retrieval & Reranking Configuration (Customizable Top-K)
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))          # Step 3: Top-K candidates refined via Cross-Encoder for LLM (Default: 3)
    RERANKER_MODEL = model_primary("reranking").model
