"""
Banking RAG Pipeline (Hệ thống RAG Chuyên sâu Ngân hàng - Dual Indexing & 5-Step Pipeline)
Kiến trúc 5 bước chính xác theo đặc tả kỹ thuật:
- Pha Ingestion & Dual-Indexing:
    * Tài liệu được phân đoạn bảo toàn theo dạng Breadcrumbs.
    * Lập chỉ mục kép song song: Sparse BM25 (Okapi BM25) và Dense Vector Store (Cosine HNSW 1024 chiều).
- Bước 1 (Query Parser):
    * Chuẩn hóa từ viết tắt chuyên ngành (eKYC, TKTT, Smart OTP, VCB Digibank) và trích xuất từ khóa.
- Bước 2 (Hybrid Retrieval):
    * Truy vấn đồng thời trên Dense Vector và BM25, hợp nhất bằng Reciprocal Rank Fusion (RRF k=60) lấy Top-5 ứng viên.
- Bước 3 (Reranking):
    * So khớp câu hỏi với câu hỏi chuẩn và độ trùng lặp từ vựng để tinh lọc ra Top-3 chunks chính xác nhất.
- Bước 4 (Context Synthesizer):
    * Đóng gói ngữ cảnh Markdown, gắn thẻ nguồn tham chiếu pháp lý [Điều X, Khoản Y] và kiểm soát giới hạn token.
- Bước 5 (LLM Serving):
    * Truyền ngữ cảnh kèm System Guardrails ngân hàng vào LLM (DeepSeek / Gemini) để sinh câu trả lời chính xác, tránh bịa đặt số liệu.
"""

import time
from pathlib import Path
from typing import Dict, Any, List

from .config import Config
from .chunking.breadcrumbs_chunker import BreadcrumbsChunker
from .embeddings.embedding_model import EmbeddingModel
from .vector_db.cosine_hnsw_store import CosineHNSWVectorStore
from .keyword_search.bm25_search import BM25Search
from .query_parser.banking_query_parser import BankingQueryParser, ParsedQuery
from .retrieval.hybrid_retriever import HybridRetriever
from .reranking.banking_reranker import BankingReranker
from .synthesizer.context_synthesizer import ContextSynthesizer, SynthesizedContext
from .generator.rag_generator import RAGGenerator

class BankingRAGPipeline:
    def __init__(self, config=None):
        self.config = config or Config()

        print("=" * 70)
        print("  KHỞI TẠO HỆ THỐNG AI BANKING RAG ASSISTANT (VIETCOMBANK DUAL-INDEX)")
        print("=" * 70)

        # 1. Khởi tạo Breadcrumbs Chunker
        print("[1/6] Khởi tạo Breadcrumbs Chunker (Bảo toàn phân cấp & Thẻ pháp lý)...")
        self.chunker = BreadcrumbsChunker(
            max_chunk_size=self.config.CHUNK_MAX_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )

        # 2. Khởi tạo Dense Embedding Model 1024 chiều
        print(f"[2/6] Khởi tạo Dense Embedding Model 1024 chiều ({self.config.EMBEDDING_MODEL_1024})...")
        self.embedding_model = EmbeddingModel(
            model_name=self.config.EMBEDDING_MODEL_1024,
            dimension=self.config.EMBEDDING_DIM,
            api_key=self.config.LLM_API_KEY,
            embedding_url=self.config.EMBEDDING_URL
        )

        # 3. Khởi tạo Dense Vector Store (Cosine HNSW 1024 chiều)
        print(f"[3/6] Khởi tạo Dense Vector Store (Cosine HNSW 1024 chiều)...")
        self.vector_store = CosineHNSWVectorStore(
            collection_name=self.config.QDRANT_COLLECTION,
            dimension=self.config.EMBEDDING_DIM
        )

        # 4. Khởi tạo Sparse BM25 Search (Okapi BM25)
        print(f"[4/6] Khởi tạo Sparse Keyword Search (Okapi BM25 k1={self.config.BM25_K1}, b={self.config.BM25_B})...")
        self.bm25_search = BM25Search(
            k1=self.config.BM25_K1,
            b=self.config.BM25_B
        )

        # 5. Khởi tạo Query Parser, Hybrid Retriever (RRF k=60) & Reranker
        print(f"[5/6] Khởi tạo Query Parser, Hybrid Retriever (RRF k=60) & Reranker...")
        self.query_parser = BankingQueryParser()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_search=self.bm25_search,
            embedding_model=self.embedding_model,
            rrf_k=self.config.RRF_K,
            dense_weight=self.config.DENSE_WEIGHT
        )
        self.reranker = BankingReranker(
            model_name=self.config.RERANKER_MODEL,
            api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL
        )

        # 6. Khởi tạo Context Synthesizer & LLM Serving với System Guardrails
        print(f"[6/6] Khởi tạo Context Synthesizer & LLM Serving ({self.config.LLM_MODEL}) kèm Banking Guardrails...")
        self.synthesizer = ContextSynthesizer(
            max_context_tokens=self.config.MAX_CONTEXT_TOKENS
        )
        self.generator = RAGGenerator(
            api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL,
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE,
            max_tokens=self.config.LLM_MAX_TOKENS
        )

        self.indexed_chunks = []
        print("[Hệ thống] Toàn bộ 5 phân hệ đã sẵn sàng hoạt động!\n")

    def ingest_documents(self, dataset_dir=None, max_files=None, extensions=(".md", ".txt")):
        """
        Pha Ingestion & Dual-Indexing:
        1. Phân đoạn bảo toàn theo dạng Breadcrumbs
        2. Lập chỉ mục kép song song:
           - Dense Vector Store (Cosine HNSW 1024 chiều)
           - Sparse BM25 (Okapi BM25)
        """
        target_dir = Path(dataset_dir or self.config.DATASET_DIR)
        print(f"[*] Bắt đầu Pha Ingestion từ: {target_dir}")

        if not target_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {target_dir}")

        # Bước A: Phân đoạn Breadcrumbs
        t0 = time.time()
        print(" -> [Ingestion] Đang phân đoạn tài liệu bảo toàn dạng Breadcrumbs & Pháp lý...")
        chunks = self.chunker.chunk_directory(target_dir, max_files=max_files, extensions=extensions)
        print(f"    + Đã trích xuất: {len(chunks)} chunks ({round((time.time() - t0)*1000, 1)}ms)")

        # Thống kê phân loại
        types_count = {}
        legal_count = 0
        faq_count = 0
        for c in chunks:
            t = c.get("chunk_type", "text")
            types_count[t] = types_count.get(t, 0) + 1
            if c.get("legal_ref"):
                legal_count += 1
            if c.get("canonical_questions"):
                faq_count += 1
        print(f"    + Phân loại: {types_count} | Chunks có thẻ pháp lý: {legal_count} | Chunks có câu hỏi chuẩn: {faq_count}")

        # Bước B: Lập chỉ mục kép - 1. Dense Vector Store (Cosine HNSW 1024 chiều)
        t_dense = time.time()
        print(" -> [Dual-Indexing 1/2] Đang tạo vector nhúng 1024 chiều & lập chỉ mục Cosine HNSW...")
        texts_to_embed = [c["content"] for c in chunks]
        vectors = self.embedding_model.embed_documents(texts_to_embed, batch_size=32)
        self.vector_store.add_chunks(chunks, vectors)
        print(f"    + Đã nạp {len(vectors)} vector (1024-d) vào Cosine HNSW Vector Store ({round((time.time() - t_dense)*1000, 1)}ms)")

        # Bước C: Lập chỉ mục kép - 2. Sparse BM25 (Okapi BM25)
        t_bm25 = time.time()
        print(" -> [Dual-Indexing 2/2] Đang lập chỉ mục Sparse Okapi BM25...")
        self.bm25_search.index(chunks)
        print(f"    + Đã lập chỉ mục Okapi BM25 trên {len(chunks)} chunks ({round((time.time() - t_bm25)*1000, 1)}ms)")

        self.indexed_chunks = chunks
        print(f"[Thành công] Hoàn tất Pha Ingestion & Dual-Indexing ({len(chunks)} chunks)!\n")
        return len(chunks)

    def step1_query_parser(self, query: str) -> ParsedQuery:
        """
        Bước 1: Query Parser
        Chuẩn hóa từ viết tắt (eKYC, TKTT, Smart OTP, VCB Digibank...) và trích xuất từ khóa.
        """
        parsed = self.query_parser.parse(query)
        return parsed

    def step2_hybrid_retrieval(self, parsed_query: ParsedQuery, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Bước 2: Hybrid Retrieval
        Truy vấn đồng thời trên Dense Vector và BM25, sau đó hợp nhất bằng RRF k=60 để lấy Top-5 ứng viên.
        """
        k = top_k or self.config.HYBRID_TOP_K
        candidates_top5 = self.retriever.retrieve(
            query=parsed_query,
            top_k=k,
            candidate_k=self.config.HYBRID_CANDIDATE_K
        )
        return candidates_top5

    def step3_reranking(self, parsed_query: ParsedQuery, candidates_top5: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Bước 3: Reranking
        So khớp câu hỏi với câu hỏi chuẩn và độ trùng lặp từ vựng để tinh lọc ra Top-3 chunks chính xác nhất.
        """
        k = top_k or self.config.RERANK_TOP_K
        top3_chunks = self.reranker.rerank(
            query=parsed_query,
            candidates=candidates_top5,
            top_k=k
        )
        return top3_chunks

    def step4_context_synthesizer(self, top3_chunks: List[Dict[str, Any]], max_tokens: int = None) -> SynthesizedContext:
        """
        Bước 4: Context Synthesizer
        Đóng gói ngữ cảnh Markdown, gắn thẻ nguồn tham chiếu pháp lý [Điều X, Khoản Y] và kiểm soát giới hạn token.
        """
        budget = max_tokens or self.config.MAX_CONTEXT_TOKENS
        synthesized = self.synthesizer.synthesize(top_chunks=top3_chunks, max_tokens=budget)
        return synthesized

    def step5_llm_serving(self, query: str, synthesized_context: SynthesizedContext) -> Dict[str, Any]:
        """
        Bước 5: LLM Serving
        Truyền ngữ cảnh kèm System Guardrails ngân hàng vào LLM (DeepSeek / Gemini) sinh câu trả lời chính xác, tránh bịa đặt số liệu.
        """
        serving_result = self.generator.generate_answer(
            query=query,
            synthesized_context=synthesized_context
        )
        return serving_result

    def run_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Thực thi toàn trình 5 bước AI Banking RAG:
        Trả về kết quả chi tiết kèm vết trung gian của từng bước.
        """
        t_start = time.time()
        print(f"\n[QUERY] '{query}'")

        # Bước 1
        t1 = time.time()
        parsed_query = self.step1_query_parser(query)
        t1_ms = round((time.time() - t1) * 1000, 1)

        # Bước 2
        t2 = time.time()
        candidates_top5 = self.step2_hybrid_retrieval(parsed_query, top_k=self.config.HYBRID_TOP_K)
        t2_ms = round((time.time() - t2) * 1000, 1)

        # Bước 3
        t3 = time.time()
        top3_chunks = self.step3_reranking(parsed_query, candidates_top5, top_k=self.config.RERANK_TOP_K)
        t3_ms = round((time.time() - t3) * 1000, 1)

        # Bước 4
        t4 = time.time()
        synthesized_context = self.step4_context_synthesizer(top3_chunks, max_tokens=self.config.MAX_CONTEXT_TOKENS)
        t4_ms = round((time.time() - t4) * 1000, 1)

        # Bước 5
        t5 = time.time()
        serving_result = self.step5_llm_serving(query, synthesized_context)
        t5_ms = round((time.time() - t5) * 1000, 1)

        total_latency_ms = round((time.time() - t_start) * 1000, 1)

        return {
            "query": query,
            "step1_parsed_query": parsed_query,
            "step2_candidates_top5": candidates_top5,
            "step3_top3_chunks": top3_chunks,
            "step4_synthesized_context": synthesized_context,
            "step5_serving_result": serving_result,
            "answer": serving_result["answer"],
            "legal_citations": synthesized_context.legal_citations,
            "sources_used": synthesized_context.sources_used,
            "latency": {
                "step1_query_parser_ms": t1_ms,
                "step2_hybrid_retrieval_ms": t2_ms,
                "step3_reranking_ms": t3_ms,
                "step4_context_synthesizer_ms": t4_ms,
                "step5_llm_serving_ms": t5_ms,
                "total_pipeline_ms": total_latency_ms
            }
        }

    def ask(self, query: str) -> Dict[str, Any]:
        """Giao diện gọi nhanh cho người dùng"""
        return self.run_pipeline(query)

    def close(self):
        """Giải phóng tài nguyên"""
        self.vector_store.close()
