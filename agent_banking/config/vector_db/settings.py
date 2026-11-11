import os

class VectorDBConfig:
    # Vector Database Configuration (1024 dimensions)
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "banking_dual_index_1024")
    QDRANT_STORAGE_PATH = "agent_banking/vector_db/qdrant_db"

