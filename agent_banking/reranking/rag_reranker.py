"""
Module Banking Document Reranker (Tối Giản - Gọn Nhẹ - Chuẩn Xác)
======================================================================
Chuyên trách:
1. Nhận danh sách Top-K ứng viên từ Hybrid Search (Dense + BM25).
2. Gọi Cross-Encoder API trên Cloud Provider (model: reranker-model) để chấm điểm tương quan ngữ nghĩa sâu giữa Query và Context.
3. Tinh lọc danh sách từ Top-5/10 xuống Top-3 chunks chính xác nhất cho LLM.
4. Thử lần lượt model Cloud theo cấu hình; báo lỗi khi toàn bộ danh sách thất bại.

NGUYÊN TẮC THIẾT KẾ (GEMINI.md & AGENTS.md):
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. NO TYPING LIBRARY: Tuyệt đối không dùng typing library hoặc type annotations.
4. NO __INIT__.PY: Import trực tiếp file module.
"""

import os
import json
import urllib.request
import math
from agent_banking.config.models import get_model_chain, model_primary
from agent_banking.config.fallback import run_fallback


class BankingReranker:
    """
    Bộ tái xếp hạng (Reranker) tối giản sử dụng Cross-Encoder reranker-model.
    """

    def __init__(
        self,
        model_name=None,
        api_key=None,
        rerank_url=None,
        base_url=None
    ):
        primary = model_primary("reranking")
        self.model_name = model_name or primary.model
        self.api_key = primary.api_key
        if api_key is not None:
            self.api_key = api_key
        self.rerank_url = primary.url
        if rerank_url:
            self.rerank_url = rerank_url
        elif base_url:
            self.rerank_url = base_url.rstrip("/") + "/rerank"

    @staticmethod
    def _normalize_query(query):
        """Trích xuất và chuẩn hóa chuỗi văn bản câu hỏi từ nhiều định dạng đầu vào"""
        if query is None:
            return ""
        if isinstance(query, str):
            return query.strip()
        if isinstance(query, dict):
            q_val = (
                query.get("clean_query")
                or query.get("query_text")
                or query.get("query")
                or query.get("raw_query")
                or query.get("normalized_query")
                or ""
            )
            if not isinstance(q_val, str):
                q_val = str(q_val)
            return q_val.strip()
        # Đối tượng ParsedQuery hoặc đối tượng tùy biến khác
        for attr in ("clean_query", "query_text", "raw_query", "normalized_query", "query"):
            val = getattr(query, attr, None)
            if val is not None:
                return str(val).strip()
        return str(query).strip()

    @staticmethod
    def _extract_text(item):
        """Trích xuất chuỗi văn bản từ đối tượng ứng viên bất kỳ"""
        if isinstance(item, str):
            return item
        if isinstance(item, tuple) and len(item) > 0:
            item = item[0]
            if isinstance(item, str):
                return item
        if isinstance(item, dict):
            # Nếu là kết quả từ Qdrant search_hybrid: {"chunk": {...}, ...}
            chunk = item.get("chunk", item)
            if not isinstance(chunk, dict):
                chunk = item
            raw_text = chunk.get("content")
            if raw_text is None or raw_text == "":
                raw_text = chunk.get("original_text")
            if raw_text is None or raw_text == "":
                raw_text = chunk.get("text")
            if raw_text is None:
                raw_text = ""
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)
            text = raw_text.strip()
            # Tránh trùng lặp tiền tố nếu text đã được prepend [Ngữ cảnh:
            if text.startswith("[Ngữ cảnh:"):
                return text
            bc = chunk.get("breadcrumb", "")
            if not isinstance(bc, str):
                bc = str(bc)
            title = chunk.get("section_title", "")
            if not isinstance(title, str):
                title = str(title)
            prefix = ""
            if bc and title:
                prefix = f"[{bc} > {title}]\n"
            elif title:
                prefix = f"[{title}]\n"
            elif bc:
                prefix = f"[{bc}]\n"
            return f"{prefix}{text}".strip()
        return str(item)

    def _fallback_local_rerank(self, query, candidates, top_k=3):
        """
        Dự phòng cục bộ bằng so khớp từ vựng Jaccard khi mất mạng hoặc API lỗi (0% Regex Law).
        Tích hợp đánh giá điểm cao hơn cho Breadcrumb (Breadcrumb Boost).
        """
        if not candidates:
            return []
        if top_k is None:
            return []
        if top_k <= 0:
            return []

        def _get_tokens(s):
            if not s:
                return set()
            cleaned = []
            for ch in str(s).lower():
                if ch.isalnum() or ch == "_":
                    cleaned.append(ch)
                else:
                    cleaned.append(" ")
            return set("".join(cleaned).split())

        q_tokens = _get_tokens(query)
        scored = []
        for idx, item in enumerate(candidates):
            text = self._extract_text(item)
            doc_tokens = _get_tokens(text)

            chunk = {}
            if isinstance(item, dict):
                chunk = item.get("chunk", item)
                if not isinstance(chunk, dict):
                    chunk = item
            bc_str = ""
            if isinstance(chunk, dict):
                bc_val = chunk.get("breadcrumb", "")
                if not isinstance(bc_val, str):
                    bc_val = str(bc_val)
                title_val = chunk.get("section_title", "")
                if not isinstance(title_val, str):
                    title_val = str(title_val)
                bc_str = f"{bc_val} {title_val}"
            bc_tokens = _get_tokens(bc_str)
            bc_matches = q_tokens.intersection(bc_tokens)

            intersection = q_tokens.intersection(doc_tokens)
            union = q_tokens.union(doc_tokens)
            if len(union) > 0:
                base_score = len(intersection) / len(union)
            else:
                base_score = 0.0

            # Đánh giá cao hơn cho Breadcrumb: cộng bonus khi query khớp thực thể trong breadcrumb
            if bc_matches:
                score = base_score * 1.3 + (len(bc_matches) / max(len(q_tokens), 1)) * 0.1
            else:
                score = base_score

            scored.append((item, score, idx))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (item, score, orig_idx) in enumerate(scored[:top_k], start=1):
            if isinstance(item, dict):
                item_copy = dict(item)
                item_copy["relevance_score"] = round(float(score), 4)
                item_copy["rerank_score"] = round(float(score), 4)
                item_copy["rerank_rank"] = rank
                results.append(item_copy)
            else:
                results.append({
                    "document": item,
                    "relevance_score": round(float(score), 4),
                    "rerank_score": round(float(score), 4),
                    "rerank_rank": rank
                })
        return results

    def rerank(self, query, candidates, top_k=3):
        """
        Tái xếp hạng danh sách candidates theo query.
        Tham số:
        - query: Câu hỏi người dùng (str, dict, hoặc ParsedQuery)
        - candidates: Danh sách chunks (list dict hoặc list str)
        - top_k: Số lượng tài liệu tinh lọc cần lấy (mặc định 3)
        """
        if not candidates:
            return []
        if top_k is None:
            return []
        if top_k <= 0:
            return []

        if query is None:
            query = ""

        # 1. Trích xuất văn bản câu hỏi hỗ trợ str, dict, ParsedQuery
        query_text = self._normalize_query(query)

        if not query_text:
            return self._fallback_local_rerank("", candidates, top_k=top_k)

        # Trích xuất danh sách văn bản tương ứng (giới hạn 2500 ký tự)
        docs = [self._extract_text(c)[:2500] for c in candidates]

        chain = get_model_chain("reranking", {
            "model": self.model_name, "url": self.rerank_url, "api_key": self.api_key,
        })

        def call(candidate):
            headers = {
                "Authorization": f"Bearer {candidate.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": candidate.model,
                "query": query_text,
                "documents": docs,
                "top_n": min(len(candidates), max(top_k, 5))
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(candidate.url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=candidate.timeout) as resp:
                if resp.status == 200:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    api_results = resp_data.get("results", [])
                    if not isinstance(api_results, list) or not api_results:
                        raise ValueError("Empty reranker response")
                    seen = set()
                    for result in api_results:
                        index = result.get("index")
                        score = result.get("relevance_score")
                        if not isinstance(index, int) or index in seen or not 0 <= index < len(candidates):
                            raise ValueError("Invalid reranker index")
                        if not isinstance(score, (int, float)) or not math.isfinite(score):
                            raise ValueError("Invalid reranker score")
                        seen.add(index)

                    # Sắp xếp và gắn điểm vào candidates (kèm Breadcrumb Entity Alignment)
                    reranked = []
                    for res in api_results:
                        idx = res.get("index", 0)
                        score = float(res.get("relevance_score", 0.0))

                        # Bảo vệ chỉ mục hợp lệ từ phản hồi API
                        if 0 <= idx < len(candidates):
                            orig_item = candidates[idx]

                            # Breadcrumb Entity Alignment: cộng bonus nhẹ (0.02) khi từ khóa chính khớp Breadcrumb
                            chunk = {}
                            if isinstance(orig_item, dict):
                                chunk = orig_item.get("chunk", orig_item)
                            if isinstance(chunk, dict):
                                bc_val = chunk.get("breadcrumb", "")
                                if not isinstance(bc_val, str):
                                    bc_val = str(bc_val)
                                title_val = chunk.get("section_title", "")
                                if not isinstance(title_val, str):
                                    title_val = str(title_val)
                                bc_str = f"{bc_val} {title_val}".lower()
                                for token in query_text.lower().split():
                                    if len(token) >= 3 and token in bc_str:
                                        score = min(1.0, score + 0.02)
                                        break

                            if isinstance(orig_item, dict):
                                item_copy = dict(orig_item)
                                item_copy["relevance_score"] = round(score, 4)
                                item_copy["rerank_score"] = round(score, 4)
                                reranked.append(item_copy)
                            else:
                                reranked.append({
                                    "document": orig_item,
                                    "relevance_score": round(score, 4),
                                    "rerank_score": round(score, 4)
                                })

                    if reranked:
                        # Sắp xếp giảm dần theo điểm và gán thứ hạng rerank_rank từ 1 đến top_k
                        reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
                        final_results = []
                        for rank, item in enumerate(reranked[:top_k], start=1):
                            item["rerank_rank"] = rank
                            final_results.append(item)
                        return final_results
            raise ValueError("No valid reranker results")

        return run_fallback("reranking", chain, call)


# Alias tương thích ngược
Reranker = BankingReranker
RAGReranker = BankingReranker
