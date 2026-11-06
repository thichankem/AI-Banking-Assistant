import os
from agent_banking.config.models import model_primary

class EmbeddingConfig:
    # 3. Cấu hình Embedding (1024 chiều)
    EMBEDDING_MODEL_1024 = model_primary("embedding").model
    EMBEDDING_URL = model_primary("embedding").url
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "")
    EMBEDDING_DIM = model_primary("embedding").dimension

