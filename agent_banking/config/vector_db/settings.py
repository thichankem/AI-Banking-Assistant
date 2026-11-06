import os

class VectorDBConfig:
    # 4. Cấu hình Vector Database (Qdrant Localhost 1024 chiều)
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "banking_dual_index_1024")
    QDRANT_STORAGE_PATH = "agent_banking/vector_db/qdrant_db"

