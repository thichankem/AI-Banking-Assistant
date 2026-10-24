"""
Module Banking RAG Pipeline (Hệ thống RAG Chuyên sâu Ngân hàng - Dual Indexing & 5-Step Pipeline)
======================================================================
Kiến trúc 5 bước chính xác theo đặc tả kỹ thuật:
- Pha Ingestion & Dual-Indexing:
    * Phân đoạn bảo toàn theo dạng Breadcrumbs (DLA Chunker).
    * Lập chỉ mục kép song song vào Qdrant: Sparse BM25 và Dense Vector Store 1024 chiều.
- Bước 1 (Query Parser):
    * Chuẩn hóa từ viết tắt chuyên ngành (eKYC, TKTT, Smart OTP, VCB Digibank) và trích xuất từ khóa.
- Bước 2 (Hybrid Retrieval):
    * Truy vấn đồng thời trên Dense Vector và BM25, hợp nhất bằng RRF k=60 lấy Top-5 ứng viên.
- Bước 3 (Reranking):
    * So khớp qua Cross-Encoder bge-reranker-v2-m3 tinh lọc ra Top-3 chunks chính xác nhất.
- Bước 4 (Context Synthesizer):
    * Đóng gói ngữ cảnh Markdown, gắn thẻ tham chiếu pháp lý [Điều X, Khoản Y].
- Bước 5 (LLM Serving):
    * Sinh câu trả lời qua DeepSeek/Gemini kèm Banking Guardrails nghiêm ngặt.

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. NO __INIT__.PY: Import trực tiếp file module.
"""

import sys
import time
import json
from pathlib import Path

# Đảm bảo UTF-8 an toàn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .config import Config
from .chunking.unified_chunk.layout_aware_chunker import DLAChunker
from .embedding.embedder import BankingEmbedder
from .vector_db.qdrant_store import QdrantBankingStore, QdrantVectorStore
from .query_parser.rag_query_parser import BankingQueryParser, ParsedQuery
from .retrieval.hybrid_retriever import HybridRetriever
from .reranking.rag_reranker import BankingReranker
from .generator.rag_generator import RAGGenerator, ContextBundle
from .tracing.tracer import BankingTracer
from .guardrails.rag_guardrails import BankingGuardrailManager


class BankingRAGPipeline:
    """
    Pipeline toàn trình AI Banking Assistant 5 bước chuẩn hóa Vietcombank.
    """

    def __init__(self, config=None, use_memory=False):
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.use_memory = use_memory

        print("=" * 70)
        print("  KHỞI TẠO HỆ THỐNG AI BANKING RAG ASSISTANT (VIETCOMBANK DUAL-INDEX)")
        print("=" * 70)

        # 1. Khởi tạo DLA Chunker (Hỗ trợ DOCX & PDF)
        print("[1/6] Khởi tạo Canonical DOCX Chunker (Chuẩn hóa Hợp đồng dữ liệu DOCX 100%)...")
        self.dla_chunker = DLAChunker(
            max_chunk_size=self.config.CHUNK_MAX_SIZE
        )
        self.chunker = self.dla_chunker

        # 2. Khởi tạo Dense Embedding Model 1024 chiều (tích hợp SQLite Cache)
        print(f"[2/6] Khởi tạo Dense Embedding Model 1024 chiều ({self.config.EMBEDDING_MODEL_1024})...")
        self.embedder = BankingEmbedder(
            dimension=self.config.EMBEDDING_DIM,
            model_name=self.config.EMBEDDING_MODEL_1024,
            embedding_url=self.config.EMBEDDING_URL,
            api_key=self.config.LLM_API_KEY,
            cache_db_path=str(self.config.STORAGE_DIR / "embedding_cache.db")
        )
        self.embedding_model = self.embedder

        # 3. Khởi tạo Dense Vector Store & BM25 Hybrid Store (Qdrant Localhost HNSW 1024 chiều)
        if use_memory:
            store_mode = "In-Memory"
        else:
            store_mode = "Disk"
        print(f"[3/6] Khởi tạo Qdrant Vector & BM25 Store ({store_mode})...")
        self.vector_store = QdrantBankingStore(
            collection_name=self.config.QDRANT_COLLECTION,
            dimension=self.config.EMBEDDING_DIM,
            storage_path=str(self.config.STORAGE_DIR / "qdrant_db"),
            use_memory=use_memory
        )

        # 4. Khởi tạo Query Parser, Hybrid Retriever (RRF k=60) & Reranker
        print(f"[4/6] Khởi tạo Query Parser & Hybrid Retriever (RRF k={self.config.RRF_K})...")
        self.query_parser = BankingQueryParser()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedder,
            rrf_k=self.config.RRF_K,
            dense_weight=self.config.DENSE_WEIGHT
        )

        print(f"[5/6] Khởi tạo Reranker ({self.config.RERANKER_MODEL})...")
        self.reranker = BankingReranker(
            model_name=self.config.RERANKER_MODEL,
            api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL
        )

        # 5. Khởi tạo LLM Serving với System Guardrails
        print(f"[5/5] Khởi tạo LLM Serving ({self.config.LLM_MODEL})...")
        self.generator = RAGGenerator(
            api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL,
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE,
            max_tokens=self.config.LLM_MAX_TOKENS
        )
        self.synthesizer = self.generator
        self.tracer = BankingTracer(config=self.config)
        self.guardrails = BankingGuardrailManager()

        self.indexed_chunks = []
        print("[Hệ thống] Toàn bộ 5 phân hệ, Tracing và Banking Guardrails đã sẵn sàng!\n")



    def ingest_chunks(self, chunks, batch_size=32):
        """
        Nạp trực tiếp danh sách chunks vào hệ thống:
        1. Tạo vector nhúng 1024 chiều (tự động đọc/ghi SQLite cache).
        2. Nạp đồng thời vào Qdrant HNSW và lập chỉ mục Okapi BM25.
        """
        if not chunks:
            return 0

        t0 = time.time()
        print(f" -> [Ingest] Đang tạo vector 1024 chiều cho {len(chunks)} chunks...")
        chunks_with_vec = self.embedder.embed_chunks(chunks, batch_size=batch_size)
        t_embed = (time.time() - t0) * 1000

        self.tracer.trace_embedding(
            num_texts=len(chunks),
            dimension=self.config.EMBEDDING_DIM,
            duration_ms=round(t_embed, 1),
            model_name=self.config.EMBEDDING_MODEL_1024,
            cache_hits=getattr(self.embedder, "last_cache_hits", 0)
        )

        t1 = time.time()
        print(f" -> [Ingest] Đang nạp vào Qdrant Vector & BM25 Store...")
        self.vector_store.add_chunks(chunks_with_vec)
        t_store = (time.time() - t1) * 1000

        print(f"    + Hoàn tất nhúng ({t_embed:.1f}ms) và nạp kho ({t_store:.1f}ms).")
        self.indexed_chunks.extend(chunks)
        return len(chunks)

    def ingest_chunk_files(self, chunks_dir, file_pattern="*.json"):
        """
        Nạp tất cả các file JSON chunk trong thư mục chỉ định vào Vector DB.
        """
        c_dir = Path(chunks_dir)
        if not c_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục chunks: {c_dir}")

        json_files = sorted(list(c_dir.glob(file_pattern)))
        print(f"[*] Tìm thấy {len(json_files)} file chunk JSON trong {c_dir}")

        total_ingested = 0
        for idx, jf in enumerate(json_files, start=1):
            with open(jf, "r", encoding="utf-8") as f:
                file_chunks = json.load(f)
            print(f"  [{idx}/{len(json_files)}] {jf.name}: {len(file_chunks)} chunks")
            self.ingest_chunks(file_chunks)
            total_ingested += len(file_chunks)

        print(f"✅ Đã nạp thành công tổng cộng {total_ingested} chunks vào Vector DB!")
        return total_ingested

    def ingest_documents(
        self,
        dataset_dir=None,
        max_files=None,
        extensions=None,
        use_dla=True,
        filenames=None,
        max_pdf_pages=None
    ):
        """
        Pha Ingestion & Dual-Indexing từ tài liệu chuẩn hóa (.docx):
        1. Phân đoạn bảo toàn theo dạng Breadcrumbs (DLA Chunker).
        2. Lập chỉ mục kép song song vào Dense Vector Store và Okapi BM25.
        """
        if dataset_dir:
            target_dir = Path(dataset_dir)
        else:
            target_dir = Path(self.config.DATASET_DIR)

        print(f"[*] Bắt đầu Pha Ingestion từ: {target_dir}")

        if not target_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {target_dir}")

        if extensions:
            target_exts = extensions
        else:
            target_exts = [".docx"]

        # Phân đoạn DLA Layout-Aware
        t0 = time.time()
        if filenames:
            print(f" -> [Ingestion] Đang phân đoạn {len(filenames)} tệp chỉ định...")
            chunks = []
            for fname in filenames:
                if (target_dir / fname).exists():
                    fpaths = [target_dir / fname]
                else:
                    fpaths = list(target_dir.rglob(fname))
                if fpaths:
                    fpath = fpaths[0]
                    c_list = self.dla_chunker.chunk_file(fpath, max_pdf_pages=max_pdf_pages)
                    chunks.extend(c_list)
        else:
            print(" -> [Ingestion] Đang phân đoạn theo DLA Layout-Aware (DOCX & PDF)...")
            chunks = self.dla_chunker.chunk_directory(
                target_dir,
                extensions=target_exts,
                max_files=max_files,
                max_pdf_pages=max_pdf_pages
            )

        print(f"    + Đã trích xuất: {len(chunks)} chunks ({round((time.time() - t0)*1000, 1)}ms)")
        chunk_duration_ms = round((time.time() - t0) * 1000, 1)
        self.tracer.trace_chunking(
            file_path=str(target_dir),
            num_chunks=len(chunks),
            duration_ms=chunk_duration_ms,
            details={"extensions": target_exts, "max_files": max_files}
        )

        # Nạp vào Vector DB và BM25
        return self.ingest_chunks(chunks)

    def step1_query_parser(self, query):
        """
        Bước 1: Query Parser
        Chuẩn hóa từ viết tắt (eKYC, TKTT, Smart OTP, VCB Digibank...) và trích xuất từ khóa.
        """
        parsed = self.query_parser.parse(query)
        return parsed

    def step2_hybrid_retrieval(
        self,
        parsed_query,
        top_k=None,
        filter_dict=None
    ):
        """
        Bước 2: Hybrid Retrieval
        Truy vấn đồng thời trên Dense Vector và BM25, sau đó hợp nhất bằng RRF k=60 để lấy Top-5 ứng viên.
        """
        if top_k is not None:
            k = top_k
        else:
            k = self.config.HYBRID_TOP_K

        candidates_top5 = self.retriever.retrieve(
            query=parsed_query,
            top_k=k,
            candidate_k=self.config.HYBRID_CANDIDATE_K,
            filter_dict=filter_dict
        )
        return candidates_top5

    def step3_reranking(
        self,
        query_or_parsed,
        candidates,
        top_k=None
    ):
        """
        Bước 3: Reranking
        Sử dụng Cross-Encoder bge-reranker-v2-m3 để tinh lọc ra Top-3 chunks chính xác nhất.
        """
        if top_k is not None:
            k = top_k
        else:
            k = self.config.RERANK_TOP_K

        if isinstance(query_or_parsed, ParsedQuery):
            raw_query = query_or_parsed.raw_query
        else:
            raw_query = str(query_or_parsed)

        top_reranked = self.reranker.rerank(
            query=raw_query,
            candidates=candidates,
            top_k=k
        )
        return top_reranked

    def step4_context_synthesizer(
        self,
        top_chunks,
        max_tokens=None
    ):
        """
        Bước 4: Context Synthesizer
        Đóng gói ngữ cảnh Markdown, gắn thẻ nguồn tham chiếu pháp lý [Điều X, Khoản Y].
        """
        return self.generator.format_chunks(top_chunks)

    def step5_llm_serving(
        self,
        query,
        synthesized_context
    ):
        """
        Bước 5: LLM Serving
        Truyền ngữ cảnh kèm System Guardrails ngân hàng vào LLM (DeepSeek / Gemini) sinh câu trả lời chính xác, tránh bịa đặt số liệu.
        """
        serving_result = self.generator.generate_answer(
            query=query,
            synthesized_context=synthesized_context
        )
        return serving_result

    def run_pipeline(
        self,
        query,
        filter_dict=None
    ):
        """
        Thực thi toàn trình 5 bước AI Banking RAG:
        Tự động bảo vệ qua 2 lớp Banking Guardrails và ghi nhận telemetry lên Arize Phoenix & Local traces.
        """
        t_start = time.time()
        print(f"\n[QUERY] '{query}'")

        # -------------------------------------------------------------
        # TẦNG 1: INPUT GUARDRAIL (Prompt Injection, Fraud, PII/Auth)
        # -------------------------------------------------------------
        input_guard = self.guardrails.guard_input(query)
        if not input_guard.is_safe:
            print(f"🛑 [BANKING GUARDRAIL BLOCKED] Lý do: {input_guard.trigger_reason}")
            if input_guard.refusal_message:
                refusal_text = input_guard.refusal_message
            else:
                refusal_text = "Yêu cầu bị từ chối do không phù hợp với quy định an toàn ngân hàng."

            with self.tracer.trace_query(query=query) as trace:
                trace.record_guardrail_block(
                    reason=input_guard.trigger_reason,
                    details=input_guard.details
                )
                trace.finish(
                    answer=refusal_text,
                    citations=[],
                    sources=[],
                    is_blocked=True,
                    guardrail_trigger=input_guard.trigger_reason
                )

            return {
                "query": query,
                "answer": refusal_text,
                "is_blocked": True,
                "guardrail_trigger": input_guard.trigger_reason,
                "guardrail_details": input_guard.details,
                "step1_parsed_query": None,
                "step2_candidates_top5": [],
                "step3_top3_chunks": [],
                "retrieved_chunks": [],
                "step4_synthesized_context": None,
                "step5_serving_result": {"answer": refusal_text, "blocked": True},
                "legal_citations": [],
                "sources_used": [],
                "latency": {
                    "total_pipeline_ms": round((time.time() - t_start) * 1000, 1)
                }
            }

        sanitized_query = input_guard.sanitized_query

        with self.tracer.trace_query(query=sanitized_query) as trace:
            # Bước 1: Query Parser
            t1 = time.time()
            parsed_query = self.step1_query_parser(sanitized_query)
            t1_ms = round((time.time() - t1) * 1000, 1)
            trace.record_query_parser(input_query=sanitized_query, parsed=parsed_query, duration_ms=t1_ms)

            # Bước 2: Hybrid Retrieval
            t2 = time.time()
            candidates_top5 = self.step2_hybrid_retrieval(
                parsed_query,
                top_k=self.config.HYBRID_TOP_K,
                filter_dict=filter_dict
            )
            t2_ms = round((time.time() - t2) * 1000, 1)

            # Ghi nhận vết Query Embedding
            trace.record_embedding(
                num_texts=1,
                dimension=self.config.EMBEDDING_DIM,
                duration_ms=round(t2_ms * 0.3, 1),
                model_name=self.config.EMBEDDING_MODEL_1024,
                is_query=True
            )
            trace.record_retrieval(query=sanitized_query, candidates=candidates_top5, duration_ms=t2_ms)

            # Bước 3: Reranking
            t3 = time.time()
            top3_chunks = self.step3_reranking(
                parsed_query,
                candidates_top5,
                top_k=self.config.RERANK_TOP_K
            )
            t3_ms = round((time.time() - t3) * 1000, 1)
            trace.record_rerank(query=sanitized_query, candidates=candidates_top5, reranked=top3_chunks, duration_ms=t3_ms)

            # Bước 4: Context Synthesizer
            t4 = time.time()
            synthesized_context = self.step4_context_synthesizer(
                top3_chunks,
                max_tokens=self.config.MAX_CONTEXT_TOKENS
            )
            t4_ms = round((time.time() - t4) * 1000, 1)
            trace.record_synthesis(synthesized=synthesized_context, duration_ms=t4_ms)

            # Bước 5: LLM Serving
            t5 = time.time()
            serving_result = self.step5_llm_serving(sanitized_query, synthesized_context)
            t5_ms = round((time.time() - t5) * 1000, 1)
            trace.record_llm_serving(
                query=sanitized_query,
                result=serving_result,
                model=self.config.LLM_MODEL,
                duration_ms=t5_ms
            )

            # -------------------------------------------------------------
            # TẦNG 2: OUTPUT GUARDRAIL (Prompt Leakage, PII Leak, Thinking tags)
            # -------------------------------------------------------------
            raw_answer = serving_result.get("answer", "")
            output_guard = self.guardrails.guard_output(raw_answer)
            answer = output_guard.sanitized_answer
            serving_result["answer"] = answer
            if output_guard.flags:
                serving_result["guardrail_flags"] = output_guard.flags
                trace.record_guardrail_check(
                    layer="output_guardrail",
                    is_safe=True,
                    details=output_guard.flags
                )

            total_latency_ms = round((time.time() - t_start) * 1000, 1)

            # Kết thúc phiên truy vết đồng bộ Arize Phoenix & Local traces
            trace.finish(
                answer=answer,
                citations=synthesized_context.legal_citations,
                sources=synthesized_context.sources_used
            )

        return {
            "query": query,
            "sanitized_query": sanitized_query,
            "is_blocked": False,
            "step1_parsed_query": parsed_query,
            "step2_candidates_top5": candidates_top5,
            "step3_top3_chunks": top3_chunks,
            "retrieved_chunks": candidates_top5,
            "step4_synthesized_context": synthesized_context,
            "step5_serving_result": serving_result,
            "answer": answer,
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

    def ask(self, query, top_k=None):
        """Giao diện gọi nhanh cho ứng dụng"""
        return self.run_pipeline(query)

    def close(self):
        """Giải phóng tài nguyên kết nối"""
        if hasattr(self.vector_store, "close"):
            self.vector_store.close()
