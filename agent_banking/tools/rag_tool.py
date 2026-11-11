# -*- coding: utf-8 -*-
"""
Banking RAG Tool for LangGraph / Agentic Workflows
======================================================================
Encapsulates RAG pipeline (Query Parser -> Hybrid Retrieval -> Cross-Encoder Reranker)
into a standard tool for AI Agents:
- Reuses classes and logic from pipeline.py.
- Independent of LangChain Runnable or BaseRetriever.
- Returns normalized payload for Context Grounding and Citation.
"""

from __future__ import annotations
import time
from pathlib import Path


class BankingRAGTool:
    """
    Tool for retrieving banking knowledge and regulatory policies.
    """
    name: str = "search_banking_knowledge"
    description: str = (
        "Tra cứu tài liệu quy chế, quy trình mở tài khoản eKYC, hạn mức giao dịch, "
        "biểu phí dịch vụ và điều khoản sử dụng thẻ."
    )

    def __init__(self, pipeline_instance=None, config=None, use_memory: bool = False):
        if pipeline_instance is not None:
            self.pipeline = pipeline_instance
        else:
            # Lazy import to avoid circular imports
            try:
                from agent_banking.pipeline import BankingRAGPipeline
                self.pipeline = BankingRAGPipeline(config=config, use_memory=use_memory)
            except ImportError:
                from rag_banking.pipeline import BankingRAGPipeline
                self.pipeline = BankingRAGPipeline(config=config, use_memory=use_memory)

    def execute(self, query: str, top_k: int = 3) -> dict:
        """
        Execute banking knowledge retrieval via Hybrid Retrieval & Reranker.
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

        # 2. Hybrid Retrieval (Dense Vector + BM25 Okapi with RRF k=60)
        candidates = self.pipeline.retriever.retrieve(
            query=parsed_q,
            top_k=self.pipeline.config.HYBRID_TOP_K,
            candidate_k=self.pipeline.config.HYBRID_CANDIDATE_K
        )

        # 3. Cross-Encoder Reranker
        reranked = self.pipeline.reranker.rerank(
            query=clean_query,
            candidates=candidates,
            top_k=top_k
        )

        # 4. Extract documents and source references
        docs = []
        sources = set()
        citations = []

        for idx, item in enumerate(reranked):
            chunk = item.get("chunk", {})
            if not isinstance(chunk, dict):
                content = str(chunk)
                source = "Tài liệu ngân hàng"
                breadcrumb = ""
                chunk_id = f"chunk_{idx}"
            else:
                content = chunk.get("content", "")
                source = chunk.get("source", "Tài liệu ngân hàng")
                breadcrumb = chunk.get("breadcrumb", "")
                chunk_id = chunk.get("chunk_id", f"chunk_{idx}")

            if source:
                sources.add(source)

            # Build citation tag from breadcrumb or source name
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
