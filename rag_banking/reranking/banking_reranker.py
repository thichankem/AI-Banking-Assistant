"""
Banking Reranker (Bước 3: Tinh lọc Top-3 Chunks bằng So khớp Câu hỏi chuẩn & Độ trùng lặp từ vựng)
Nhiệm vụ:
1. So khớp câu hỏi người dùng với danh sách câu hỏi chuẩn (Canonical Questions) trong từng chunk.
2. Đo lường độ trùng lặp từ vựng (Lexical Overlap / Keyword Containment) giữa truy vấn và ngữ cảnh.
3. Kết hợp điểm số RRF và Cross-Encoder (bge-reranker-v2-m3 nếu khả dụng).
4. Tinh lọc từ Top-5 ứng viên xuống chính xác Top-3 chunks chất lượng và phù hợp nhất.
"""

import re
import os
import httpx
from typing import List, Dict, Any, Union
from ..query_parser.banking_query_parser import ParsedQuery

class BankingReranker:
    def __init__(
        self,
        model_name: str = "bge-reranker-v2-m3",
        api_key: str = None,
        base_url: str = None,
        canonical_weight: float = 0.40,
        lexical_weight: float = 0.35,
        rrf_weight: float = 0.25
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.base_url = base_url or os.getenv("BASE_URL", "https://mkp-api.fptcloud.com/v1")
        self.canonical_weight = canonical_weight
        self.lexical_weight = lexical_weight
        self.rrf_weight = rrf_weight

    def _tokenize(self, text: str) -> set:
        """Tách từ đơn giản và loại bỏ ký tự đặc biệt"""
        return set(re.findall(r"\w+", text.lower()))

    def _match_canonical_questions(self, query_tokens: set, raw_query: str, chunk: dict) -> float:
        """
        Tính điểm so khớp câu hỏi với các câu hỏi chuẩn (Canonical Questions):
        Kiểm tra danh mục canonical_questions của chunk hoặc tiêu đề phần mục.
        """
        canonical_qs = chunk.get("canonical_questions", [])
        section_title = chunk.get("section_title", "")
        if section_title and section_title not in canonical_qs:
            canonical_qs = list(canonical_qs) + [section_title]

        if not canonical_qs:
            # Thử trích xuất câu hỏi từ content nếu có 'Câu X: ...'
            inline_qs = re.findall(r"(?:Câu\s*\d+|Hỏi)\s*[:\.\-]\s*([^\n\?]+\?)", chunk.get("raw_content", ""))
            canonical_qs.extend(inline_qs)

        if not canonical_qs:
            return 0.1  # Điểm sàn mặc định nếu không có câu hỏi chuẩn

        max_sim = 0.0
        q_clean = raw_query.lower()

        for cq in canonical_qs:
            cq_clean = cq.lower()
            cq_tokens = self._tokenize(cq_clean)
            if not cq_tokens:
                continue

            # Jaccard Token Similarity
            intersection = query_tokens.intersection(cq_tokens)
            union = query_tokens.union(cq_tokens)
            jaccard = len(intersection) / len(union) if union else 0.0

            # Substring / Exact match boost
            sub_boost = 0.2 if any(term in cq_clean for term in query_tokens if len(term) > 3) else 0.0
            
            sim = min(jaccard * 1.5 + sub_boost, 1.0)
            if sim > max_sim:
                max_sim = sim

        return round(max_sim, 4)

    def _compute_lexical_overlap(self, query_tokens: set, keywords: list, chunk_text: str) -> float:
        """
        Tính độ trùng lặp từ vựng (Lexical Overlap):
        Tỷ lệ từ khóa trong câu hỏi xuất hiện trong nội dung chunk.
        """
        if not query_tokens:
            return 0.0

        chunk_tokens = self._tokenize(chunk_text)
        if not chunk_tokens:
            return 0.0

        # 1. Tỷ lệ token trùng lặp
        common_tokens = query_tokens.intersection(chunk_tokens)
        token_ratio = len(common_tokens) / len(query_tokens)

        # 2. Tỷ lệ keyword chuyên ngành khớp chính xác
        kw_matches = 0
        total_kws = max(len(keywords), 1)
        chunk_lower = chunk_text.lower()
        for kw in keywords:
            if kw.lower() in chunk_lower:
                kw_matches += 1
        kw_ratio = kw_matches / total_kws

        # Điểm trùng lặp từ vựng tổng hợp
        lexical_score = 0.5 * token_ratio + 0.5 * kw_ratio
        return round(min(lexical_score, 1.0), 4)

    def _try_api_rerank(self, query: str, candidate_chunks: list) -> list:
        """Thử gọi Cross-Encoder API bge-reranker-v2-m3 trên FPT Cloud nếu khả dụng"""
        if not self.api_key or not self.base_url:
            return []

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            docs = [c["chunk"].get("content", "")[:1000] for c in candidate_chunks]
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": docs
            }
            with httpx.Client(timeout=8.0) as client:
                r = client.post(f"{self.base_url}/rerank", json=payload, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    scores = [0.0] * len(candidate_chunks)
                    for item in data.get("results", []):
                        idx = item.get("index", 0)
                        score = float(item.get("relevance_score", 0.0))
                        scores[idx] = score
                    return scores
        except Exception:
            pass
        return []

    def rerank(
        self,
        query: Union[str, ParsedQuery],
        candidates: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Tinh lọc danh sách ứng viên từ Top-5 xuống Top-3 chunks.
        """
        if not candidates:
            return []

        # Chuẩn bị dữ liệu truy vấn
        if isinstance(query, ParsedQuery):
            raw_query = query.raw_query
            keywords = query.keywords
        else:
            raw_query = str(query)
            keywords = [w for w in re.findall(r"\w+", raw_query.lower()) if len(w) > 2]

        query_tokens = self._tokenize(raw_query)

        # Tính toán điểm RRF chuẩn hóa (0..1)
        max_rrf = max((c.get("rrf_score", 0.0001) for c in candidates), default=1.0)

        # Thử lấy điểm Cross-Encoder nếu có
        api_scores = self._try_api_rerank(raw_query, candidates)

        scored_candidates = []
        for i, item in enumerate(candidates):
            chunk = item["chunk"]
            chunk_content = chunk.get("content", "")

            # 1. So khớp câu hỏi với câu hỏi chuẩn
            canonical_score = self._match_canonical_questions(query_tokens, raw_query, chunk)

            # 2. Độ trùng lặp từ vựng
            lexical_score = self._compute_lexical_overlap(query_tokens, keywords, chunk_content)

            # 3. Chuẩn hóa điểm RRF
            norm_rrf = item.get("rrf_score", 0.0) / max_rrf if max_rrf > 0 else 0.5

            # 4. Điểm API Cross-Encoder (nếu có)
            api_score = api_scores[i] if (api_scores and i < len(api_scores)) else None

            # Tổng hợp Rerank Score
            if api_score is not None:
                # Kết hợp 4 tín hiệu khi có API
                final_score = (
                    0.30 * canonical_score +
                    0.30 * lexical_score +
                    0.20 * norm_rrf +
                    0.20 * max(0.0, min(1.0, api_score))
                )
            else:
                # Chuẩn 3 tín hiệu theo quy định đề tài
                final_score = (
                    self.canonical_weight * canonical_score +
                    self.lexical_weight * lexical_score +
                    self.rrf_weight * norm_rrf
                )

            scored_candidates.append({
                "chunk": chunk,
                "final_rerank_score": round(final_score, 4),
                "canonical_matching_score": canonical_score,
                "lexical_overlap_score": lexical_score,
                "rrf_score": item.get("rrf_score", 0.0),
                "cross_encoder_score": round(api_score, 4) if api_score is not None else None,
                "original_hybrid_rank": item.get("candidate_rank", i + 1)
            })

        # Sắp xếp theo final_rerank_score giảm dần
        scored_candidates.sort(key=lambda x: x["final_rerank_score"], reverse=True)

        # Trích xuất đúng Top-3 chunks
        top_reranked = []
        for rank, item in enumerate(scored_candidates[:top_k], start=1):
            item["rerank_rank"] = rank
            top_reranked.append(item)

        return top_reranked
