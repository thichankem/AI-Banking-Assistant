import os

class RetrievalConfig:
    # 5. Cấu hình Keyword Search (BM25 Okapi)
    BM25_K1 = 1.5                 # Tham số điều chỉnh độ bão hòa tần suất từ
    BM25_B = 0.75                 # Tham số phạt độ dài tài liệu
    BREADCRUMB_BOOST = 1.5        # Hệ số nhân ưu tiên cho từ khóa xuất hiện trong Breadcrumb (BM25F)


    HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", "5"))          # Bước 2: Top-K ứng viên sau khi hợp nhất RRF (Mặc định 5)
    HYBRID_CANDIDATE_K = 15       # Số lượng ứng viên lấy từ mỗi phương pháp
    RRF_K = 60                    # Hằng số điều hòa trong Reciprocal Rank Fusion (k=60)
    DENSE_WEIGHT = 0.5            # Cân bằng 50/50 giữa Dense Cosine HNSW và Sparse BM25
