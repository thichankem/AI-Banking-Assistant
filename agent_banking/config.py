"""
Module Cấu hình Hệ thống RAG Banking (AI Banking Assistant)
======================================================================
Quản lý tập trung các tham số cấu hình:
- Đường dẫn lưu trữ (vector_db, dataset)
- Cấu hình mô hình Embedding (1024 chiều) & Qdrant HNSW
- Cấu hình BM25 Okapi & Hybrid RRF Search
- Cấu hình Cross-Encoder Reranker & LLM Serving
- Cấu hình Tracing (Arize Phoenix & Local Tracing)

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không dùng regex.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. NO __INIT__.PY: Import trực tiếp file module.
"""

import os
from pathlib import Path

# Tải biến môi trường từ file .env ở thư mục gốc dự án (thuần Python, không cần thư viện dotenv)
ROOT_DIR = Path(__file__).resolve().parent.parent
env_file = ROOT_DIR / ".env"
if env_file.exists():
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except Exception:
        pass

# Xác định đường dẫn thư mục dataset
retrieval_dir = ROOT_DIR / "dataset" / "retrieval_study" / "retrieval_dataset"
chunking_dir_candidate = ROOT_DIR / "agent_banking" / "chunking" / "chunking_dataset"

if chunking_dir_candidate.exists():
    chunking_dir = chunking_dir_candidate
else:
    chunking_dir = ROOT_DIR / "dataset" / "chunking_dataset"

if retrieval_dir.exists():
    active_dataset_dir = retrieval_dir
else:
    active_dataset_dir = chunking_dir


class Config:
    # 1. Đường dẫn dữ liệu
    ROOT_DIR = ROOT_DIR
    RETRIEVAL_DATASET_DIR = retrieval_dir
    CHUNKING_DATASET_DIR = chunking_dir
    DATASET_DIR = active_dataset_dir
    STORAGE_DIR = ROOT_DIR / "agent_banking" / "vector_db"
    TRACE_DIR = ROOT_DIR / "agent_banking" / "tracing" / "traces"

    # 2. Cấu hình Chunking (Layout-Aware)
    CHUNK_MAX_SIZE = 800          # Kích thước ký tự tối đa cho 1 text chunk
    KEEP_TABLE_ATOMIC = True      # Giữ nguyên bảng biểu làm 1 đơn vị dữ liệu độc lập

    # 3. Cấu hình Embedding (1024 chiều)
    EMBEDDING_MODEL_1024 = os.getenv("EMBEDDING_MODEL", "")
    EMBEDDING_URL = os.getenv("EMBEDDING_URL", "")
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "")
    EMBEDDING_DIM = 1024

    # 4. Cấu hình Vector Database (Qdrant Localhost 1024 chiều)
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "banking_dual_index_1024")
    QDRANT_STORAGE_PATH = "agent_banking/vector_db/qdrant_db"

    # 5. Cấu hình Keyword Search (BM25 Okapi)
    BM25_K1 = 1.5                 # Tham số điều chỉnh độ bão hòa tần suất từ
    BM25_B = 0.75                 # Tham số phạt độ dài tài liệu
    BREADCRUMB_BOOST = 1.5        # Hệ số nhân ưu tiên cho từ khóa xuất hiện trong Breadcrumb (BM25F)

    # 6. Cấu hình Hybrid Retrieval & Reranking (Tự do tùy chỉnh Top-K)
    HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", "5"))          # Bước 2: Top-K ứng viên sau khi hợp nhất RRF (Mặc định 5)
    HYBRID_CANDIDATE_K = 15       # Số lượng ứng viên lấy từ mỗi phương pháp
    RRF_K = 60                    # Hằng số điều hòa trong Reciprocal Rank Fusion (k=60)
    DENSE_WEIGHT = 0.5            # Cân bằng 50/50 giữa Dense Cosine HNSW và Sparse BM25
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))          # Bước 3: Top-K tinh lọc qua Cross-Encoder cho LLM (Mặc định 3)
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "bge-reranker-v2-m3")

    # 7. Cấu hình Context Synthesizer (Bước 4)
    MAX_CONTEXT_TOKENS = 2000     # Kiểm soát giới hạn token cho ngữ cảnh Markdown

    # 8. Cấu hình LLM Generator & Guardrails (Bước 5)
    LLM_API_KEY = os.getenv("API_KEY", "")
    LLM_BASE_URL = os.getenv("BASE_URL", "")
    LLM_MODEL = os.getenv("MODEL", "")
    LLM_TEMPERATURE = 0.2         # Nhiệt độ thấp để trả lời chính xác, tránh hallucination
    LLM_MAX_TOKENS = 2500
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # Giới hạn số lần thử lại tối đa cho chu trình LangGraph Agent

    # 9. Cấu hình Tracing & Observability (Arize Phoenix & Local Tracing)
    ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"

    # Cấu hình Arize Phoenix (RAG OpenTelemetry Tracing)
    ENABLE_PHOENIX = os.getenv("ENABLE_PHOENIX", "true").lower() == "true"
    PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "")
    PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "")

    # 10. Cấu hình Enterprise Banking Guardrails 4.0 (Multi-Tier Defense)
    ENABLE_SEMANTIC_GUARDRAIL = os.getenv("ENABLE_SEMANTIC_GUARDRAIL", "true").lower() == "true"
    SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.88"))
    ENABLE_INDIRECT_INJECTION_SCAN = os.getenv("ENABLE_INDIRECT_INJECTION_SCAN", "true").lower() == "true"
    MAX_LOAN_AMOUNT_VND = int(os.getenv("MAX_LOAN_AMOUNT_VND", "50000000000"))
    MAX_LOAN_TENURE_MONTHS = int(os.getenv("MAX_LOAN_TENURE_MONTHS", "360"))
    MIN_SAVINGS_AMOUNT_VND = int(os.getenv("MIN_SAVINGS_AMOUNT_VND", "1000000"))
