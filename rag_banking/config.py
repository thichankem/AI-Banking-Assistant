"""
Module Cấu hình Hệ thống RAG Banking (AI Banking Assistant)
Chứa các tham số cấu hình đường dẫn, mô hình, Weaviate, và ngưỡng truy xuất.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Tải biến môi trường từ file .env ở thư mục gốc dự án
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

class Config:
    # 1. Đường dẫn dữ liệu
    ROOT_DIR = ROOT_DIR
    RETRIEVAL_DATASET_DIR = ROOT_DIR / "dataset" / "retrieval_study" / "retrieval_dataset"
    CHUNKING_DATASET_DIR = (
        ROOT_DIR / "rag_banking" / "chunking" / "chunking_dataset"
        if (ROOT_DIR / "rag_banking" / "chunking" / "chunking_dataset").exists()
        else ROOT_DIR / "dataset" / "chunking_dataset"
    )
    DATASET_DIR = RETRIEVAL_DATASET_DIR if RETRIEVAL_DATASET_DIR.exists() else CHUNKING_DATASET_DIR
    OUTPUT_DATASET_DIR = CHUNKING_DATASET_DIR / "output_dataset"
    STORAGE_DIR = ROOT_DIR / "rag_banking" / "storage"

    # 2. Cấu hình Chunking (Layout-Aware)
    CHUNK_MAX_SIZE = 800          # Kích thước ký tự tối đa cho 1 text chunk
    KEEP_TABLE_ATOMIC = True      # Giữ nguyên bảng biểu làm 1 đơn vị dữ liệu độc lập

    # 3. Cấu hình Embedding (1024 chiều)
    # Hỗ trợ FPT Cloud multilingual-e5-large / Vietnamese_Embedding (1024-dim) kèm offline 1024-dim projection
    EMBEDDING_MODEL_1024 = os.getenv("EMBEDDING_MODEL", "multilingual-e5-large")
    EMBEDDING_URL = os.getenv("EMBEDDING_URL", "https://mkp-api.fptcloud.com/v1/embeddings")
    LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 1024

    # 4. Cấu hình Vector Database (Cosine HNSW 1024 chiều)
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
    WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
    WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    WEAVIATE_COLLECTION_NAME = "BankingDocumentChunk"
    QDRANT_COLLECTION = "banking_dual_index_1024"

    # 5. Cấu hình Keyword Search (BM25 Okapi)
    BM25_K1 = 1.5                 # Tham số điều chỉnh độ bão hòa tần suất từ
    BM25_B = 0.75                 # Tham số phạt độ dài tài liệu

    # 6. Cấu hình Hybrid Retrieval & Reranking
    HYBRID_TOP_K = 5              # Bước 2: Top-5 ứng viên sau khi hợp nhất RRF
    HYBRID_CANDIDATE_K = 15       # Số lượng ứng viên lấy từ mỗi phương pháp
    RRF_K = 60                    # Hằng số điều hòa trong Reciprocal Rank Fusion (k=60)
    DENSE_WEIGHT = 0.5            # Cân bằng 50/50 giữa Dense Cosine HNSW và Sparse BM25
    RERANK_TOP_K = 3              # Bước 3: Tinh lọc ra Top-3 chunks chính xác nhất
    RERANKER_MODEL = "bge-reranker-v2-m3"

    # 7. Cấu hình Context Synthesizer (Bước 4)
    MAX_CONTEXT_TOKENS = 2000     # Kiểm soát giới hạn token cho ngữ cảnh Markdown

    # 8. Cấu hình LLM Generator & Guardrails (Bước 5)
    LLM_API_KEY = os.getenv("API_KEY", "")
    LLM_BASE_URL = os.getenv("BASE_URL", "https://mkp-api.fptcloud.com/v1")
    LLM_MODEL = os.getenv("MODEL", "DeepSeek-V4-Flash")
    LLM_TEMPERATURE = 0.2         # Nhiệt độ thấp để trả lời chính xác, tránh hallucination
    LLM_MAX_TOKENS = 2500

    # 9. Cấu hình Tracing
    ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ai-banking-rag")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
