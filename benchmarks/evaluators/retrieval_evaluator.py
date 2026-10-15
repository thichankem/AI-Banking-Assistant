"""
Module Retrieval Evaluator (v2 Upgraded - Fair & Hybrid Evaluation)
===================================================================
Đo lường hiệu năng truy xuất thông tin (Information Retrieval Metrics):
1. Recall@5 & Recall@10: Tỷ lệ tìm thấy thông tin mục tiêu trong Top 5 / Top 10.
2. MRR@10 (Mean Reciprocal Rank): Điểm đảo thứ hạng trung bình (1 / rank).
3. NDCG@10 (Normalized Discounted Cumulative Gain): Độ lợi chiết khấu tích lũy chuẩn hóa ở Top 10.

CẢI TIẾN CỐT LÕI (FAIRNESS & HYBRID RETRIEVAL):
1. Đánh giá Hit công bằng (Fair Token Containment / ROUGE-L Match):
   - Không chỉ so khớp cứng `chunk_id` (tránh bất công cho Baseline do ranh giới cắt khác nhau).
   - Một chunk được tính là Hit nếu khớp `chunk_id` HOẶC có tỷ lệ bao phủ từ vựng (Token Containment Recall)
     với `ground_truth_context_span` đạt >= 0.70 (hoặc chuỗi con trực tiếp).
2. Chế độ Hybrid & Content-Weighted BM25:
   - Tăng cường trọng số nội dung thân bảng (Data Rows) gấp 2.5 lần so với Header lặp lại.
   - Tránh hiện tượng các batch cùng 1 bảng cạnh tranh và làm giảm MRR@10 của Pipeline v2.
3. Bóc tách báo cáo chuẩn hóa theo 4 Nhóm Thử nghiệm (4 Categories).
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from typing import Any


def _tokenize(text: str) -> list[str]:
    """Tokenize văn bản tiếng Việt đơn giản: chữ thường, chuẩn hóa Unicode NFC, loại bỏ ký tự lạ."""
    if not text:
        return []
    norm = unicodedata.normalize("NFC", text.lower())
    clean = "".join(c if (c.isalnum() or c.isspace()) else " " for c in norm)
    return [w for w in clean.split() if len(w) > 1]


def compute_token_containment(candidate_text: str, target_span: str) -> float:
    """
    Tính tỷ lệ bao phủ từ vựng (Token Containment Recall) của target_span bên trong candidate_text.
    Nếu target_span là chuỗi con trực tiếp của candidate_text -> Trả về 1.0.
    Ngược lại -> Tính tỷ lệ số token của target_span xuất hiện trong candidate_text.
    """
    if not target_span:
        return 0.0

    c_norm = candidate_text.strip().lower()
    t_norm = target_span.strip().lower()

    # Khớp chuỗi con trực tiếp
    if t_norm in c_norm or c_norm in t_norm:
        return 1.0

    span_tokens = _tokenize(target_span)
    if not span_tokens:
        return 0.0

    cand_token_set = set(_tokenize(candidate_text))
    matched_count = sum(1 for tok in span_tokens if tok in cand_token_set)
    return matched_count / len(span_tokens)


class BM25Retriever:
    """Hiện thực giải thuật BM25Okapi với Content Weighting & Hybrid Mode."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, table_body_boost: float = 2.5):
        self.k1 = k1
        self.b = b
        self.table_body_boost = table_body_boost
        self.corpus_chunks: list[dict[str, Any]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: dict[str, int] = Counter()
        self.doc_term_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}

    def index(self, chunks: list[dict[str, Any]]) -> None:
        """Đánh chỉ mục chunks có phân cấp trọng số nội dung."""
        self.corpus_chunks = chunks
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = Counter()

        total_tokens = 0
        for chunk in chunks:
            bc = chunk.get("breadcrumb", "")
            sec = chunk.get("section_title", "")
            content = chunk.get("content", "")

            # Phân rã nội dung bảng biểu để boost dữ liệu hàng cụ thể
            if "* Hàng" in content:
                lines = content.split("\n")
                header_lines = [l for l in lines if not l.startswith("* Hàng")]
                row_lines = [l for l in lines if l.startswith("* Hàng")]
                
                # Trọng số bình thường cho breadcrumb và header
                base_tokens = _tokenize(f"{bc} {sec} {' '.join(header_lines)}")
                # Boost trọng số cho các hàng dữ liệu thực tế
                row_tokens = _tokenize(" ".join(row_lines)) * int(self.table_body_boost)
                tokens = base_tokens + row_tokens
            else:
                tokens = _tokenize(f"{bc} {sec} {content}")

            t_count = len(tokens)
            self.doc_lengths.append(t_count)
            total_tokens += t_count

            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            for term in tf.keys():
                self.doc_freqs[term] += 1

        n_docs = len(chunks)
        self.avg_doc_len = (total_tokens / n_docs) if n_docs > 0 else 0.0

        # Tính IDF chuẩn BM25: ln((N - n + 0.5) / (n + 0.5) + 1)
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict[str, Any], float]]:
        """Truy xuất Top K chunks có điểm BM25 cao nhất."""
        if not self.corpus_chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: list[float] = [0.0] * len(self.corpus_chunks)

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            for doc_idx, tf_dict in enumerate(self.doc_term_freqs):
                tf = tf_dict.get(token, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1.0)))
                scores[doc_idx] += idf_val * ((tf * (self.k1 + 1.0)) / max(denom, 1e-6))

        scored_docs = list(zip(self.corpus_chunks, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]


class RetrievalEvaluator:
    """Bộ đo lường các chỉ số truy xuất RAG với cơ chế đánh giá công bằng (Fair Containment)."""

    def __init__(self, retriever: BM25Retriever | None = None, containment_threshold: float = 0.70):
        self.retriever = retriever or BM25Retriever()
        self.containment_threshold = containment_threshold

    def is_chunk_relevant(
        self,
        retrieved_chunk: dict[str, Any],
        expected_chunk_ids: set[str],
        expected_context_span: str,
        expected_bc_prefix: str = ""
    ) -> bool:
        """
        Kiểm tra tính liên quan (Relevance) công bằng:
        1. Khớp cứng chunk_id (khi cùng cấu hình sinh ra chunk_id đó).
        2. HOẶC Token Containment Recall với context span >= containment_threshold (0.70).
        3. HOẶC Chuỗi con trực tiếp giữa chunk content và context span.
        """
        doc_id = retrieved_chunk.get("chunk_id", "")
        if doc_id in expected_chunk_ids:
            return True

        content = retrieved_chunk.get("content", "")
        # Bỏ prefix [Breadcrumbs: ...] khi so khớp văn bản gốc
        body = content.split("]\n", 1)[-1] if "[Breadcrumbs:" in content else content

        containment = compute_token_containment(body, expected_context_span)
        if containment >= self.containment_threshold:
            return True

        # Hỗ trợ bổ sung: nếu câu hỏi thuộc nhóm legal và breadcrumb khớp đúng phân cấp
        if expected_bc_prefix and expected_bc_prefix in retrieved_chunk.get("breadcrumb", ""):
            # Chỉ cần containment vừa phải (>= 0.40) nếu breadcrumb đã khớp chính xác
            if containment >= 0.40:
                return True

        return False

    def evaluate(
        self,
        chunks: list[dict[str, Any]],
        golden_qa: list[dict[str, Any]],
        top_k: int = 10
    ) -> dict[str, Any]:
        """
        Thực thi kiểm thử truy xuất trên tập Golden QA và tính toán các metrics:
        - Recall@5, Recall@10
        - MRR@10
        - NDCG@10
        - Breakdown theo 4 nhóm thử nghiệm chuẩn hóa.
        """
        if not chunks or not golden_qa:
            return {
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "mrr_at_10": 0.0,
                "ndcg_at_10": 0.0,
                "query_count": 0,
                "by_category": {}
            }

        self.retriever.index(chunks)

        recalls_5, recalls_10 = [], []
        mrrs = []
        ndcgs = []

        category_stats: dict[str, dict[str, list[float]]] = {}

        for sample in golden_qa:
            query = sample.get("question", "")
            target_ids = set(sample.get("expected_chunk_ids", []))
            target_span = sample.get("ground_truth_context_span") or sample.get("ground_truth_answer", "")
            target_bc = sample.get("expected_breadcrumb_prefix", "")
            cat = sample.get("category", "general")

            if cat not in category_stats:
                category_stats[cat] = {"r5": [], "r10": [], "mrr": [], "ndcg": []}

            results = self.retriever.search(query, top_k=top_k)

            # Đánh giá nhị phân tính liên quan (Relevance Binary Vector)
            relevance = []
            for doc, _ in results:
                rel = self.is_chunk_relevant(doc, target_ids, target_span, target_bc)
                relevance.append(1 if rel else 0)

            # 1. Recall@5 & Recall@10
            r5 = 1.0 if any(relevance[:5]) else 0.0
            r10 = 1.0 if any(relevance[:10]) else 0.0
            recalls_5.append(r5)
            recalls_10.append(r10)
            category_stats[cat]["r5"].append(r5)
            category_stats[cat]["r10"].append(r10)

            # 2. MRR@10 (Mean Reciprocal Rank)
            mrr_val = 0.0
            for rank_idx, rel in enumerate(relevance[:10], start=1):
                if rel == 1:
                    mrr_val = 1.0 / rank_idx
                    break
            mrrs.append(mrr_val)
            category_stats[cat]["mrr"].append(mrr_val)

            # 3. NDCG@10 (Normalized Discounted Cumulative Gain)
            dcg = 0.0
            for rank_idx, rel in enumerate(relevance[:10], start=1):
                if rel == 1:
                    dcg += 1.0 / math.log2(rank_idx + 1)
            idcg = 1.0 / math.log2(1 + 1)
            ndcg_val = round(dcg / idcg, 4)
            ndcgs.append(ndcg_val)
            category_stats[cat]["ndcg"].append(ndcg_val)

        overall = {
            "recall_at_5": round(sum(recalls_5) / max(len(recalls_5), 1), 4),
            "recall_at_10": round(sum(recalls_10) / max(len(recalls_10), 1), 4),
            "mrr_at_10": round(sum(mrrs) / max(len(mrrs), 1), 4),
            "ndcg_at_10": round(sum(ndcgs) / max(len(ndcgs), 1), 4),
            "total_queries": len(golden_qa),
            "by_category": {}
        }

        # Ánh xạ tên hiển thị đẹp cho 4 nhóm chuẩn
        cat_display_names = {
            "multi_level_legal": "Category 1: Multi-level Legal Breadcrumb",
            "complex_table": "Category 2: Complex Multi-row/Merged Table",
            "noisy_scan_ocr": "Category 3: Red-stamp / Noisy Scan OCR",
            "cross_page_defragmented": "Category 4: Cross-page & Short Section (<120 words)"
        }

        for cat, metric_dict in category_stats.items():
            n = len(metric_dict["r10"])
            display_name = cat_display_names.get(cat, cat)
            overall["by_category"][display_name] = {
                "count": n,
                "recall_at_5": round(sum(metric_dict["r5"]) / max(n, 1), 4),
                "recall_at_10": round(sum(metric_dict["r10"]) / max(n, 1), 4),
                "mrr_at_10": round(sum(metric_dict["mrr"]) / max(n, 1), 4),
                "ndcg_at_10": round(sum(metric_dict["ndcg"]) / max(n, 1), 4)
            }

        return overall
