import os

class RetrievalConfig:
    # 5. Keyword Search Configuration (BM25 Okapi)
    BM25_K1 = 1.5                 # Term frequency saturation parameter
    BM25_B = 0.75                 # Document length penalty parameter
    BREADCRUMB_BOOST = 1.5        # Boost multiplier for keywords appearing in breadcrumb (BM25F)

    HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", "5"))          # Step 2: Top-K candidates after RRF fusion (Default: 5)
    HYBRID_CANDIDATE_K = 15       # Number of candidates retrieved from each method
    RRF_K = 60                    # Smoothing constant in Reciprocal Rank Fusion (k=60)
    DENSE_WEIGHT = 0.5            # 50/50 balance between Dense Cosine HNSW and Sparse BM25
