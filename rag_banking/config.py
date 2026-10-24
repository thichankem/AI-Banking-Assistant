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
chunking_dir_candidate = ROOT_DIR / "rag_banking" / "chunking" / "chunking_dataset"

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
    STORAGE_DIR = ROOT_DIR / "rag_banking" / "vector_db"
    TRACE_DIR = ROOT_DIR / "rag_banking" / "tracing" / "traces"

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
    QDRANT_STORAGE_PATH = "rag_banking/vector_db/qdrant_db"

    # 5. Cấu hình Keyword Search (BM25 Okapi)
    BM25_K1 = 1.5                 # Tham số điều chỉnh độ bão hòa tần suất từ
    BM25_B = 0.75                 # Tham số phạt độ dài tài liệu
    BREADCRUMB_BOOST = 1.5        # Hệ số nhân ưu tiên cho từ khóa xuất hiện trong Breadcrumb (BM25F)

    # 6. Cấu hình Hybrid Retrieval & Reranking
    HYBRID_TOP_K = 5              # Bước 2: Top-5 ứng viên sau khi hợp nhất RRF
    HYBRID_CANDIDATE_K = 15       # Số lượng ứng viên lấy từ mỗi phương pháp
    RRF_K = 60                    # Hằng số điều hòa trong Reciprocal Rank Fusion (k=60)
    DENSE_WEIGHT = 0.5            # Cân bằng 50/50 giữa Dense Cosine HNSW và Sparse BM25
    RERANK_TOP_K = 3              # Bước 3: Tinh lọc ra Top-3 chunks chính xác nhất
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "")

    # 7. Cấu hình Context Synthesizer (Bước 4)
    MAX_CONTEXT_TOKENS = 2000     # Kiểm soát giới hạn token cho ngữ cảnh Markdown

    # 8. Cấu hình LLM Generator & Guardrails (Bước 5)
    LLM_API_KEY = os.getenv("API_KEY", "")
    LLM_BASE_URL = os.getenv("BASE_URL", "")
    LLM_MODEL = os.getenv("MODEL", "")
    LLM_TEMPERATURE = 0.2         # Nhiệt độ thấp để trả lời chính xác, tránh hallucination
    LLM_MAX_TOKENS = 2500

    # 9. Cấu hình Tracing & Observability (Arize Phoenix & Local Tracing)
    ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"

    # Cấu hình Arize Phoenix (RAG OpenTelemetry Tracing)
    ENABLE_PHOENIX = os.getenv("ENABLE_PHOENIX", "true").lower() == "true"
    PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "")
    PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "")
