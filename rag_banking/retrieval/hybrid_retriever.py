"""
Hybrid Retriever (Bước 2: Truy vấn Kết hợp Dense Vector & BM25 + Reciprocal Rank Fusion)
Nhiệm vụ:
1. Truy vấn đồng thời trên Dense Vector Store (Cosine HNSW 1024 chiều) bằng normalized query.
2. Truy vấn đồng thời trên BM25 Okapi bằng từ khóa chuyên ngành (keywords).
3. Hợp nhất danh sách bằng thuật toán Reciprocal Rank Fusion (RRF k=60).
4. Lấy ra chính xác Top-5 ứng viên tiềm năng nhất.
"""

import time
from typing import List, Dict, Any, Union
from ..query_parser.banking_query_parser import ParsedQuery

class HybridRetriever:
    def __init__(
        self,
        vector_store,
        bm25_search,
        embedding_model,
        rrf_k: int = 60,
        dense_weight: float = 0.5
    ):
        self.vector_store = vector_store
        self.bm25_search = bm25_search
        self.embedding_model = embedding_model
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight

    def retrieve(
        self,
        query: Union[str, ParsedQuery],
        top_k: int = 5,
        candidate_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện Hybrid Retrieval và hợp nhất thứ hạng với RRF k=60.
        Trả về đúng Top-5 ứng viên.
        """
        # 1. Xác định chuỗi truy vấn cho từng kênh
        if isinstance(query, ParsedQuery):
            dense_query_text = query.normalized_query
            bm25_query_text = query.keywords_str or query.raw_query
            raw_query_text = query.raw_query
        else:
            dense_query_text = str(query)
            bm25_query_text = str(query)
            raw_query_text = str(query)

        # 2. Dense Vector Search (Cosine HNSW 1024-dim)
        t_dense_start = time.time()
        query_vector = self.embedding_model.embed_query(dense_query_text)
        dense_results = self.vector_store.search_vector(query_vector, top_k=candidate_k)
        t_dense_duration = (time.time() - t_dense_start) * 1000

        # 3. Sparse BM25 Okapi Search
        t_bm25_start = time.time()
        bm25_results = self.bm25_search.search(bm25_query_text, top_k=candidate_k)
        t_bm25_duration = (time.time() - t_bm25_start) * 1000

        # 4. Hợp nhất bằng thuật toán Reciprocal Rank Fusion (RRF k=60)
        candidates = {}

        # Ghi nhận kết quả Dense
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            chunk_id = chunk.get("chunk_id", str(id(chunk)))
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "chunk": chunk,
                    "dense_rank": rank,
                    "dense_score": float(score),
                    "bm25_rank": None,
                    "bm25_score": 0.0,
                    "rrf_score": 0.0
                }
            else:
                candidates[chunk_id]["dense_rank"] = rank
                candidates[chunk_id]["dense_score"] = float(score)

        # Ghi nhận kết quả BM25
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            chunk_id = chunk.get("chunk_id", str(id(chunk)))
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "chunk": chunk,
                    "dense_rank": None,
                    "dense_score": 0.0,
                    "bm25_rank": rank,
                    "bm25_score": float(score),
                    "rrf_score": 0.0
                }
            else:
                candidates[chunk_id]["bm25_rank"] = rank
                candidates[chunk_id]["bm25_score"] = float(score)

        # Tính toán điểm số RRF chuẩn k=60:
        # RRF(d) = (w_dense / (k + rank_dense)) + (w_bm25 / (k + rank_bm25))
        bm25_weight = 1.0 - self.dense_weight
        for item in candidates.values():
            rrf = 0.0
            if item["dense_rank"] is not None:
                rrf += self.dense_weight / (self.rrf_k + item["dense_rank"])
            if item["bm25_rank"] is not None:
                rrf += bm25_weight / (self.rrf_k + item["bm25_rank"])
            item["rrf_score"] = rrf

        # Sắp xếp các ứng viên theo điểm RRF giảm dần
        sorted_candidates = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Trích xuất chính xác Top-5 ứng viên
        top_candidates = []
        for rank, item in enumerate(sorted_candidates[:top_k], start=1):
            top_candidates.append({
                "candidate_rank": rank,
                "chunk": item["chunk"],
                "rrf_score": round(item["rrf_score"], 6),
                "dense_rank": item["dense_rank"],
                "dense_score": round(item["dense_score"], 4),
                "bm25_rank": item["bm25_rank"],
                "bm25_score": round(item["bm25_score"], 4),
                "retrieval_meta": {
                    "dense_latency_ms": round(t_dense_duration, 2),
                    "bm25_latency_ms": round(t_bm25_duration, 2),
                    "rrf_k": self.rrf_k
                }
            })

        return top_candidates
