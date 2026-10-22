"""
Module Banking Document Reranker (Tối Giản - Gọn Nhẹ - Chuẩn Xác)
======================================================================
Chuyên trách:
1. Nhận danh sách Top-K ứng viên từ Hybrid Search (Dense + BM25).
2. Gọi Cross-Encoder API trên FPT Cloud (model: bge-reranker-v2-m3) để chấm điểm tương quan ngữ nghĩa sâu giữa Query và Context.
3. Tinh lọc danh sách từ Top-5/10 xuống Top-3 chunks chính xác nhất cho LLM.
4. Tự động chuyển sang Local Token Fallback nếu mất mạng hoặc API lỗi, đảm bảo pipeline không bao giờ bị crash.

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. NO __INIT__.PY: Import trực tiếp file module.
"""

import os
import json
import urllib.request


class BankingReranker:
    """
    Bộ tái xếp hạng (Reranker) tối giản sử dụng Cross-Encoder bge-reranker-v2-m3.
    """

    def __init__(
        self,
        model_name=None,
        api_key=None,
        rerank_url=None,
        base_url=None
    ):
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = os.getenv("RERANKER_MODEL", "bge-reranker-v2-m3")

        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("API_KEY", "")

        # Xác định URL endpoint chuẩn của FPT Cloud Rerank
        if rerank_url:
            self.rerank_url = rerank_url
        elif base_url:
            self.rerank_url = f"{base_url.rstrip('/')}/rerank"
        else:
            env_base = os.getenv("BASE_URL", "")
            if env_base:
                self.rerank_url = os.getenv("RERANK_URL", f"{env_base.rstrip('/')}/rerank")
            else:
                self.rerank_url = os.getenv("RERANK_URL", "")

    @staticmethod
    def _extract_text(item):
        """Trích xuất chuỗi văn bản từ đối tượng ứng viên bất kỳ"""
        if isinstance(item, str):
            return item
        if isinstance(item, tuple) and len(item) > 0:
            item = item[0]
        if isinstance(item, dict):
            # Nếu là kết quả từ Qdrant search_hybrid: {"chunk": {...}, ...}
            chunk = item.get("chunk", item)
            text = chunk.get("content") or chunk.get("original_text") or chunk.get("text") or ""
            bc = chunk.get("breadcrumb", "")
            title = chunk.get("section_title", "")
            prefix = ""
            if bc and title:
                prefix = f"[{bc} > {title}]\n"
            elif title:
                prefix = f"[{title}]\n"
            return f"{prefix}{text}".strip()
        return str(item)

    def _fallback_local_rerank(self, query, candidates, top_k=3):
        """
        Dự phòng cục bộ bằng so khớp từ vựng Jaccard khi mất mạng hoặc API lỗi (0% Regex Law).
        """
        def _get_tokens(s: str) -> set[str]:
            if not s:
                return set()
            cleaned = []
            for ch in s.lower():
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
            intersection = q_tokens.intersection(doc_tokens)
            union = q_tokens.union(doc_tokens)
            if len(union) > 0:
                score = len(intersection) / len(union)
            else:
                score = 0.0
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
        - query: Câu hỏi người dùng (str hoặc object có .raw_query / .query)
        - candidates: Danh sách chunks (list dict hoặc list str)
        - top_k: Số lượng tài liệu tinh lọc cần lấy (mặc định 3)
        """
        if not candidates:
            return []

        # Chuẩn hóa câu hỏi dạng text
        query_text = getattr(query, "raw_query", getattr(query, "query", str(query))).strip()
        if not query_text:
            return candidates[:top_k]

        # Trích xuất danh sách văn bản tương ứng
        docs = [self._extract_text(c)[:2500] for c in candidates]

        # 1. Gọi FPT Cloud Rerank API (bge-reranker-v2-m3)
        if self.api_key and self.rerank_url:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_name,
                    "query": query_text,
                    "documents": docs,
                    "top_n": min(len(candidates), max(top_k, 5))
                }

                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(self.rerank_url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=20.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        api_results = data.get("results", [])

                        # Sắp xếp và gắn điểm vào candidates
                        reranked = []
                        for rank, res in enumerate(api_results[:top_k], start=1):
                            idx = res.get("index", 0)
                            score = float(res.get("relevance_score", 0.0))
                            orig_item = candidates[idx]

                            if isinstance(orig_item, dict):
                                item_copy = dict(orig_item)
                                item_copy["relevance_score"] = round(score, 4)
                                item_copy["rerank_score"] = round(score, 4)
                                item_copy["rerank_rank"] = rank
                                reranked.append(item_copy)
                            else:
                                reranked.append({
                                    "document": orig_item,
                                    "relevance_score": round(score, 4),
                                    "rerank_score": round(score, 4),
                                    "rerank_rank": rank
                                })

                        return reranked
            except Exception as e:
                print(f"⚠️ [Reranker] Lỗi gọi FPT Cloud API: {e}. Đang chuyển sang Local Fallback...")

        # 2. Local Fallback an toàn nếu API không khả dụng
        print("⚡ [Reranker] Sử dụng Local Fallback Token Overlap")
        return self._fallback_local_rerank(query_text, candidates, top_k=top_k)


# Alias tiện dụng
Reranker = BankingReranker
