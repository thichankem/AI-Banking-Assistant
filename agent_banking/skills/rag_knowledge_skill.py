"""
Kỹ năng RAG Truy xuất Tri thức Ngân hàng (RAG Knowledge Skill).
Kế thừa toàn diện các tính năng cốt lõi của Banking RAG Pipeline:
- Chuẩn hóa thuật ngữ ngân hàng (eKYC, VCB Digibank, TKTT...).
- Tìm kiếm lai Hybrid: Dense Vector 1024 chiều + Okapi BM25 (RRF k=60).
- Tinh lọc bằng Cross-Encoder bge-reranker-v2-m3.
- Đóng gói Markdown và gắn thẻ trích dẫn [Điều X, Khoản Y].

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations

from .base_skill import BaseSkill, SkillMetadata
from agent_banking.config import Config


class RAGKnowledgeSkill(BaseSkill):
    """Kỹ năng tìm kiếm văn bản quy chế, biểu phí và chính sách Vietcombank."""

    def __init__(self, pipeline_instance=None, config: Config | None = None):
        meta = SkillMetadata(
            skill_id="rag_knowledge_retrieval",
            display_name="Truy xuất Tri thức Quy chế Ngân hàng (Agentic RAG)",
            description="Tìm kiếm văn bản pháp lý, quy chế mở tài khoản, hạn mức giao dịch, biểu phí bằng chỉ mục kép Hybrid và Cross-Encoder.",
            capabilities=[
                "hybrid_vector_bm25_search",
                "rrf_rank_fusion_k60",
                "cross_encoder_rerank",
                "legal_citation_tagging"
            ],
            intent_triggers=[
                "quy chế", "hạn mức", "biểu phí", "phí thường niên",
                "điều khoản", "ekyc", "mở tài khoản", "vcb digibank",
                "thủ tục phát hành", "chính sách thẻ"
            ],
            category="RAG"
        )
        super().__init__(metadata=meta, config=config)
        self.pipeline = pipeline_instance

        if self.pipeline is None:
            from agent_banking.pipeline import BankingRAGPipeline
            if config is not None:
                cfg = config
            else:
                cfg = Config()
            self.pipeline = BankingRAGPipeline(config=cfg)

    def _run(self, params: dict) -> dict:
        raw_query = params.get("query", "").strip()
        if not raw_query:
            return {
                "chunks": [],
                "formatted_context": "",
                "citations": [],
                "sources": [],
                "query_used": ""
            }

        top_k = params.get("top_k", 3)
        candidate_k = params.get("candidate_k", 5)

        # 1. Chuẩn hóa câu hỏi
        parsed_query = self.pipeline.step1_query_parser(raw_query)

        # 2. Truy xuất lai Hybrid (Dense + BM25 qua RRF k=60)
        candidates = self.pipeline.step2_hybrid_retrieval(
            parsed_query=parsed_query,
            top_k=candidate_k
        )

        # 3. Tái xếp hạng bằng Cross-Encoder
        reranked_chunks = self.pipeline.step3_reranking(
            query_or_parsed=parsed_query,
            candidates=candidates,
            top_k=top_k
        )

        # 4. Đóng gói ngữ cảnh Markdown và trích xuất nguồn
        synthesized_context = self.pipeline.step4_context_synthesizer(reranked_chunks)
        context_text = ""
        if hasattr(synthesized_context, "markdown_context"):
            context_text = synthesized_context.markdown_context
        else:
            context_text = str(synthesized_context)

        citations = []
        sources = []
        for ch in reranked_chunks:
            doc_name = ch.get("document_name", "")
            if doc_name and doc_name not in sources:
                sources.append(doc_name)

            score = ch.get("rerank_score", ch.get("score", 0.0))
            citations.append({
                "document_name": doc_name,
                "breadcrumbs": ch.get("breadcrumbs", ""),
                "section": ch.get("section", ""),
                "score": round(score, 4)
            })

        return {
            "chunks": reranked_chunks,
            "formatted_context": context_text,
            "citations": citations,
            "sources": sources,
            "query_used": raw_query
        }
