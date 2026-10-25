# -*- coding: utf-8 -*-
"""
Banking RAG Tool for LangGraph / Agentic Workflows
======================================================================
Đóng gói toàn bộ luồng RAG chuyên sâu (Query Parser -> Hybrid Retrieval -> Cross-Encoder Reranker)
thành một Công cụ chuẩn (Standard Tool) cho AI Agent:
- Tái sử dụng 100% các class và logic hiện có trong pipeline.py và node_pipeline.py.
- Không phụ thuộc vào LangChain Runnable hay BaseRetriever.
- Trả về cấu trúc Payload chuẩn hóa phục vụ Context Grounding và Citation.
"""

from __future__ import annotations
import time
from typing import Any
from pathlib import Path


class BankingRAGTool:
    """
    Công cụ tra cứu tri thức nghiệp vụ ngân hàng Vietcombank.
    """
    name: str = "search_banking_knowledge"
    description: str = (
        "Tra cứu tài liệu quy chế, quy trình mở tài khoản eKYC, hạn mức giao dịch, "
        "biểu phí dịch vụ và điều khoản sử dụng thẻ Vietcombank."
    )

    def __init__(self, pipeline_instance=None, config=None, use_memory: bool = False):
        if pipeline_instance is not None:
            self.pipeline = pipeline_instance
        else:
            # Lazy import để tránh circular import
            try:
                from agent_banking.pipeline import BankingRAGPipeline
                self.pipeline = BankingRAGPipeline(config=config, use_memory=use_memory)
            except ImportError:
                from rag_banking.pipeline import BankingRAGPipeline
                self.pipeline = BankingRAGPipeline(config=config, use_memory=use_memory)

    def execute(self, query: str, top_k: int = 3) -> dict[str, Any]:
        """
        Thực thi tra cứu tri thức ngân hàng thông qua Hybrid Retrieval & Reranker.
        """
        start_t = time.perf_counter()
        clean_query = query.strip()
        
        if not clean_query:
            return {
                "status": "empty_query",
                "documents": [],
                "sources": [],
                "citations": [],
                "latency_ms": 0.0
            }

        # 1. Parse Query & Acronym Normalization
        parsed_q = self.pipeline.query_parser.parse(clean_query)
        search_query = parsed_q.normalized_query
        
        # 2. Hybrid Retrieval (Dense Vector + BM25 Okapi với RRF k=60)
        candidates = self.pipeline.retriever.retrieve(
            query=parsed_q,
            top_k=self.pipeline.config.HYBRID_TOP_K,
            candidate_k=self.pipeline.config.HYBRID_CANDIDATE_K
        )
        
        # 3. Cross-Encoder Reranker (bge-reranker-v2-m3)
        reranked = self.pipeline.reranker.rerank(
            query=clean_query,
            candidates=candidates,
            top_k=top_k
        )

        # 4. Trích xuất tài liệu và văn bản nguồn
        docs = []
        sources = set()
        citations = []

        for idx, item in enumerate(reranked):
            chunk = item.get("chunk", {})
            if not isinstance(chunk, dict):
                content = str(chunk)
                source = "Tài liệu Vietcombank"
                breadcrumb = ""
                chunk_id = f"chunk_{idx}"
            else:
                content = chunk.get("content", "")
                source = chunk.get("source", "Tài liệu Vietcombank")
                breadcrumb = chunk.get("breadcrumb", "")
                chunk_id = chunk.get("chunk_id", f"chunk_{idx}")

            if source:
                sources.add(source)

            # Tạo citation tag nếu có breadcrumb hoặc tên điều khoản
            if breadcrumb:
                cite_label = f"[{breadcrumb}]"
            else:
                cite_label = f"[{source}]"

            citations.append({
                "index": idx + 1,
                "label": cite_label,
                "source": source,
                "chunk_id": chunk_id,
                "relevance_score": round(float(item.get("relevance_score", 0.0)), 4)
            })

            docs.append({
                "rank": idx + 1,
                "chunk_id": chunk_id,
                "content": content,
                "source": source,
                "breadcrumb": breadcrumb,
                "relevance_score": item.get("relevance_score", 0.0)
            })

        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

        return {
            "status": "success",
            "query": clean_query,
            "normalized_query": search_query,
            "total_found": len(docs),
            "documents": docs,
            "sources": sorted(list(sources)),
            "citations": citations,
            "latency_ms": latency_ms
        }
