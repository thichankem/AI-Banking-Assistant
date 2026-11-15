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
            from agent_banking.pipeline import BankingRAGPipeline
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
            if isinstance(item, dict):
                raw_chunk = item.get("chunk", item)
                score = float(item.get("relevance_score", item.get("rerank_score", item.get("score", 0.0))))
                rerank_score = float(item.get("rerank_score", score))
                rerank_rank = item.get("rerank_rank", idx + 1)
                rrf_score = float(item.get("rrf_score", 0.0))
            else:
                raw_chunk = item
                score = 0.0
                rerank_score = 0.0
                rerank_rank = idx + 1
                rrf_score = 0.0

            if not isinstance(raw_chunk, dict):
                chunk_id = f"chunk_{idx}"
                parent_id = ""
                source = "Tài liệu ngân hàng"
                chunk_type = "text"
                breadcrumb = ""
                section_title = ""
                content = str(raw_chunk)
                parent_content = ""
                business_metadata = {}
                prev_chunk_id = ""
                next_chunk_id = ""
                table_data = None
                images = []
            else:
                chunk_id = str(raw_chunk.get("chunk_id") or f"chunk_{idx}")
                parent_id = str(raw_chunk.get("parent_id") or "")
                source = str(raw_chunk.get("source") or "Tài liệu ngân hàng")
                chunk_type = str(raw_chunk.get("chunk_type") or "text")
                breadcrumb = str(raw_chunk.get("breadcrumb") or "")
                section_title = str(raw_chunk.get("section_title") or "")
                content = str(raw_chunk.get("content") or "")
                parent_content = str(raw_chunk.get("parent_content") or "")
                bm = raw_chunk.get("business_metadata")
                if isinstance(bm, dict):
                    business_metadata = bm
                else:
                    business_metadata = {}
                prev_chunk_id = str(raw_chunk.get("prev_chunk_id") or "")
                next_chunk_id = str(raw_chunk.get("next_chunk_id") or "")
                table_data = raw_chunk.get("table_data")
                raw_images = raw_chunk.get("images")
                if isinstance(raw_images, list):
                    images = raw_images
                else:
                    images = []

            doc_name = str(business_metadata.get("document_name") or source or "")
            if source:
                sources.add(source)

            if breadcrumb:
                cite_label = f"[{breadcrumb}]"
            elif section_title:
                cite_label = f"[{section_title}]"
            else:
                cite_label = f"[{source}]"

            citations.append({
                "index": idx + 1,
                "label": cite_label,
                "source": source,
                "document_name": doc_name,
                "breadcrumb": breadcrumb,
                "section_title": section_title,
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "relevance_score": round(score, 4)
            })

            docs.append({
                "rank": idx + 1,
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "source": source,
                "chunk_type": chunk_type,
                "breadcrumb": breadcrumb,
                "section_title": section_title,
                "content": content,
                "parent_content": parent_content,
                "business_metadata": business_metadata,
                "prev_chunk_id": prev_chunk_id,
                "next_chunk_id": next_chunk_id,
                "table_data": table_data,
                "images": images,
                "relevance_score": score,
                "rerank_score": rerank_score,
                "rerank_rank": rerank_rank,
                "rrf_score": rrf_score
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
