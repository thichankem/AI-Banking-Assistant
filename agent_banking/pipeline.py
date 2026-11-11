"""
Banking RAG Pipeline Module (Enterprise Banking RAG System - Dual Indexing & 5-Step Pipeline)
======================================================================
5-step architecture conforming strictly to technical specifications:
- Ingestion & Dual-Indexing Phase:
    * Layout-aware preservation via Breadcrumbs (DLA Chunker).
    * Parallel dual indexing into Qdrant: Sparse BM25 and Dense Vector Store (1024-dim).
- Step 1 (Query Parser):
    * Banking acronym normalization (eKYC, TKTT, Smart OTP, Digibank) and keyword extraction.
- Step 2 (Hybrid Retrieval):
    * Concurrent query across Dense Vector and BM25, fused via RRF k=60 to retrieve Top-5 candidates.
- Step 3 (Reranking):
    * Cross-Encoder semantic reranker refining candidates to Top-3 most accurate chunks.
- Step 4 (Context Synthesizer):
    * Markdown context packaging, attaching legal references [Article X, Clause Y].
- Step 5 (LLM Serving):
    * Answer generation via dynamically configured LLM with strict banking guardrails.

DESIGN PRINCIPLES:
1. 0% REGEX LAW: Strictly string methods only (in, split, startswith, endswith, strip, lower, replace, find).
2. ZERO TERNARY: Explicit if/else syntax only.
3. NO __INIT__.PY: Direct module imports.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path

# Ensure UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agent_banking.core.settings import Config
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
    Enterprise Banking AI RAG Pipeline with 5-step dual-indexing architecture.
    """

    def __init__(self, config=None, use_memory=False):
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.use_memory = use_memory

        print("=" * 70)
        print("  INITIALIZING ENTERPRISE BANKING RAG ASSISTANT PIPELINE")
        print("=" * 70)

        # 1. Initialize DLA Chunker (Canonical document chunking)
        print("[1/5] Initializing Layout-Aware Document Chunker...")
        self.dla_chunker = DLAChunker(
            max_chunk_size=self.config.CHUNK_MAX_SIZE
        )
        self.chunker = self.dla_chunker

        # 2. Initialize Dense Embedding Model (integrated SQLite Cache)
        print(f"[2/5] Initializing Dense Embedding Model ({self.config.EMBEDDING_MODEL_1024})...")
        self.embedder = BankingEmbedder(
            dimension=self.config.EMBEDDING_DIM,
            model_name=self.config.EMBEDDING_MODEL_1024,
            embedding_url=self.config.EMBEDDING_URL,
            api_key=self.config.LLM_API_KEY,
            cache_db_path=str(self.config.STORAGE_DIR / "embedding_cache.db")
        )
        self.embedding_model = self.embedder

        # 3. Initialize Dense Vector Store & BM25 Hybrid Store
        if use_memory:
            store_mode = "In-Memory"
        else:
            store_mode = "Disk"
        print(f"[3/5] Initializing Vector & BM25 Store ({store_mode})...")
        self.vector_store = QdrantBankingStore(
            collection_name=self.config.QDRANT_COLLECTION,
            dimension=self.config.EMBEDDING_DIM,
            storage_path=str(self.config.STORAGE_DIR / "qdrant_db"),
            use_memory=use_memory
        )

        # 4. Initialize Query Parser, Hybrid Retriever (RRF) & Reranker
        print(f"[4/5] Initializing Query Parser & Hybrid Retriever (RRF k={self.config.RRF_K})...")
        self.query_parser = BankingQueryParser()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedder,
            rrf_k=self.config.RRF_K,
            dense_weight=self.config.DENSE_WEIGHT
        )

        print(f"[4.1/5] Initializing Reranker ({self.config.RERANKER_MODEL})...")
        self.reranker = BankingReranker(
            model_name=self.config.RERANKER_MODEL,
            api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL
        )

        # 5. Initialize LLM Serving with System Guardrails
        print(f"[5/5] Initializing LLM Serving ({self.config.LLM_MODEL})...")
        self.generator = RAGGenerator(
            api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL,
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE,
            max_tokens=self.config.LLM_MAX_TOKENS
        )
        self.synthesizer = self.generator
        self.tracer = BankingTracer(config=self.config)
        self.guardrails = BankingGuardrailManager(embedder=self.embedder)

        self.indexed_chunks = []
        print("[System] Enterprise RAG Pipeline, Tracing, and Guardrails are ready.\n")

    def ingest_chunks(self, chunks, batch_size=32):
        """
        Directly ingest a list of chunks into the system:
        1. Generate dense vector embeddings (using SQLite cache).
        2. Ingest concurrently into HNSW Vector Store and Okapi BM25 index.
        """
        if not chunks:
            return 0

        t0 = time.time()
        print(f" -> [Ingest] Generating embeddings for {len(chunks)} chunks...")
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
        print(" -> [Ingest] Ingesting into Vector & BM25 Store...")
        self.vector_store.add_chunks(chunks_with_vec)
        t_store = (time.time() - t1) * 1000

        print(f"    + Embedding finished ({t_embed:.1f}ms) and storage finished ({t_store:.1f}ms).")
        self.indexed_chunks.extend(chunks)
        return len(chunks)

    def ingest_chunk_files(self, chunks_dir, file_pattern="*.json"):
        """
        Ingest all chunk JSON files in the specified directory into Vector DB.
        """
        c_dir = Path(chunks_dir)
        if not c_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {c_dir}")

        json_files = sorted(list(c_dir.glob(file_pattern)))
        print(f"[*] Found {len(json_files)} chunk JSON files in {c_dir}")

        total_ingested = 0
        for idx, jf in enumerate(json_files, start=1):
            with open(jf, "r", encoding="utf-8") as f:
                file_chunks = json.load(f)
            print(f"  [{idx}/{len(json_files)}] {jf.name}: {len(file_chunks)} chunks")
            self.ingest_chunks(file_chunks)
            total_ingested += len(file_chunks)

        print(f"Successfully ingested a total of {total_ingested} chunks into Vector DB.")
        return total_ingested

    def ingest_documents(
        self,
        dataset_dir=None,
        max_files=None,
        extensions=None,
        use_dla=True,
        filenames=None
    ):
        """
        Ingestion & Dual-Indexing phase from documents (.docx):
        1. Layout-aware chunking via Breadcrumbs (DLA Chunker).
        2. Parallel dual indexing into Dense Vector Store and Okapi BM25.
        """
        if dataset_dir:
            target_dir = Path(dataset_dir)
        else:
            target_dir = Path(self.config.DATASET_DIR)

        print(f"[*] Starting ingestion phase from: {target_dir}")

        if not target_dir.exists():
            raise FileNotFoundError(f"Directory not found: {target_dir}")

        if extensions:
            target_exts = extensions
        else:
            target_exts = [".docx"]

        # Layout-Aware DLA Chunking
        t0 = time.time()
        if filenames:
            print(f" -> [Ingestion] Chunking {len(filenames)} specified files...")
            chunks = []
            for fname in filenames:
                if (target_dir / fname).exists():
                    fpaths = [target_dir / fname]
                else:
                    fpaths = list(target_dir.rglob(fname))
                if fpaths:
                    fpath = fpaths[0]
                    c_list = self.dla_chunker.chunk_file(fpath)
                    chunks.extend(c_list)
        else:
            print(" -> [Ingestion] Chunking via Layout-Aware Chunker (DOCX)...")
            chunks = self.dla_chunker.chunk_directory(
                target_dir,
                extensions=target_exts,
                max_files=max_files
            )

        print(f"    + Extracted: {len(chunks)} chunks ({round((time.time() - t0)*1000, 1)}ms)")
        chunk_duration_ms = round((time.time() - t0) * 1000, 1)
        self.tracer.trace_chunking(
            file_path=str(target_dir),
            num_chunks=len(chunks),
            duration_ms=chunk_duration_ms,
            details={"extensions": target_exts, "max_files": max_files}
        )

        # Ingest into Vector DB and BM25
        return self.ingest_chunks(chunks)

    def step1_query_parser(self, query):
        """
        Step 1: Query Parser
        Normalize domain acronyms (eKYC, TKTT, Smart OTP, Digibank) and extract keywords.
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
        Step 2: Hybrid Retrieval
        Query concurrently across Dense Vector and BM25, then fuse via RRF k=60 to get top candidates.
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
        Step 3: Reranking
        Use Cross-Encoder reranker to refine candidates to top most accurate chunks.
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
        Step 4: Context Synthesizer
        Package context in Markdown format with legal reference citations [Article X, Clause Y].
        """
        return self.generator.format_chunks(top_chunks)

    def step5_llm_serving(
        self,
        query,
        synthesized_context,
        persona_config=None
    ):
        """
        Step 5: LLM Serving
        Pass context with dynamic persona and banking guardrails to the configured LLM.
        """
        serving_result = self.generator.generate_answer(
            query=query,
            synthesized_context=synthesized_context,
            persona_config=persona_config
        )
        return serving_result

    def run_pipeline(
        self,
        query,
        filter_dict=None,
        persona_config=None
    ):
        """
        Execute full 5-step Enterprise Banking RAG pipeline:
        Secured with dual-layer Banking Guardrails and telemetry to Arize Phoenix & local traces.
        """
        t_start = time.time()
        print(f"\n[QUERY] '{query}'")

        # -------------------------------------------------------------
        # LAYER 1: INPUT GUARDRAIL (Prompt Injection, Fraud, PII/Auth)
        # -------------------------------------------------------------
        input_guard = self.guardrails.guard_input(query)
        if not input_guard.is_safe:
            print(f"[Banking Guardrail Blocked] Reason: {input_guard.trigger_reason}")
            if input_guard.refusal_message:
                refusal_text = input_guard.refusal_message
            else:
                refusal_text = "The request was declined in accordance with banking security policies."

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
            # Step 1: Query Parser
            t1 = time.time()
            parsed_query = self.step1_query_parser(sanitized_query)
            t1_ms = round((time.time() - t1) * 1000, 1)
            trace.record_query_parser(input_query=sanitized_query, parsed=parsed_query, duration_ms=t1_ms)

            # Step 2: Hybrid Retrieval
            t2 = time.time()
            candidates_top5 = self.step2_hybrid_retrieval(
                parsed_query,
                top_k=self.config.HYBRID_TOP_K,
                filter_dict=filter_dict
            )
            t2_ms = round((time.time() - t2) * 1000, 1)

            # Record Query Embedding trace
            trace.record_embedding(
                num_texts=1,
                dimension=self.config.EMBEDDING_DIM,
                duration_ms=round(t2_ms * 0.3, 1),
                model_name=self.config.EMBEDDING_MODEL_1024,
                is_query=True
            )
            trace.record_retrieval(query=sanitized_query, candidates=candidates_top5, duration_ms=t2_ms)

            # Step 3: Reranking
            t3 = time.time()
            top3_chunks = self.step3_reranking(
                parsed_query,
                candidates_top5,
                top_k=self.config.RERANK_TOP_K
            )
            t3_ms = round((time.time() - t3) * 1000, 1)
            trace.record_rerank(query=sanitized_query, candidates=candidates_top5, reranked=top3_chunks, duration_ms=t3_ms)

            # Stage 2: Pre-Generation Relevance Gating (Anti-hallucination prior to generation)
            relevance_threshold = 0.25
            if hasattr(self.config, "RELEVANCE_THRESHOLD"):
                relevance_threshold = self.config.RELEVANCE_THRESHOLD
            is_relevant, no_info_msg, rel_details = self.guardrails.guard_rag_relevance(
                sanitized_query,
                top3_chunks,
                min_score=relevance_threshold
            )
            if is_relevant is False:
                print(f"[Relevance Gating Blocked] RAG context does not contain relevant information: {rel_details}")
                trace.record_guardrail_block(
                    reason="RAG_IRRELEVANT_CHUNKS",
                    details=rel_details
                )
                trace.finish(
                    answer=no_info_msg,
                    citations=[],
                    sources=[],
                    is_blocked=False,
                    guardrail_trigger="RAG_IRRELEVANT_CHUNKS"
                )
                return {
                    "query": query,
                    "sanitized_query": sanitized_query,
                    "answer": no_info_msg,
                    "is_blocked": False,
                    "guardrail_trigger": "RAG_IRRELEVANT_CHUNKS",
                    "guardrail_details": rel_details,
                    "step1_parsed_query": parsed_query,
                    "step2_candidates_top5": candidates_top5,
                    "step3_top3_chunks": top3_chunks,
                    "retrieved_chunks": candidates_top5,
                    "step4_synthesized_context": None,
                    "step5_serving_result": {"answer": no_info_msg, "fail_fast": True},
                    "legal_citations": [],
                    "sources_used": [],
                    "latency": {
                        "step1_query_parser_ms": t1_ms,
                        "step2_hybrid_retrieval_ms": t2_ms,
                        "step3_reranking_ms": t3_ms,
                        "total_pipeline_ms": round((time.time() - t_start) * 1000, 1)
                    }
                }

            # Step 4: Context Synthesizer
            t4 = time.time()
            # Stage 3: Scan for indirect prompt injection in chunks before passing to context
            safe_chunks, blocked_cnt, threats = self.guardrails.guard_retrieved_chunks(top3_chunks)
            if blocked_cnt > 0:
                print(f"[Indirect Injection Guardrail] Removed {blocked_cnt} unsafe chunk(s) from context.")
            top3_chunks = safe_chunks

            synthesized_context = self.step4_context_synthesizer(
                top3_chunks,
                max_tokens=self.config.MAX_CONTEXT_TOKENS
            )
            t4_ms = round((time.time() - t4) * 1000, 1)
            trace.record_synthesis(synthesized=synthesized_context, duration_ms=t4_ms)

            # Step 5: LLM Serving
            t5 = time.time()
            serving_result = self.step5_llm_serving(sanitized_query, synthesized_context, persona_config=persona_config)
            t5_ms = round((time.time() - t5) * 1000, 1)
            trace.record_llm_serving(
                query=sanitized_query,
                result=serving_result,
                model=self.config.LLM_MODEL,
                duration_ms=t5_ms,
                persona=serving_result.get("persona", "advisor"),
                tone=serving_result.get("tone", "formal"),
                format_style=serving_result.get("format_style", "standard")
            )

            # -------------------------------------------------------------
            # STAGE 4: OUTPUT GUARDRAIL & FAITHFULNESS GROUNDING
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

            # Cross-check grounding & faithfulness with actual retrieved chunks
            is_faithful, faith_flags = self.guardrails.guard_rag_faithfulness(answer, top3_chunks)
            if is_faithful is False:
                serving_result["faithfulness_flags"] = faith_flags
                trace.record_guardrail_check(
                    layer="rag_faithfulness",
                    is_safe=False,
                    details=faith_flags
                )

            total_latency_ms = round((time.time() - t_start) * 1000, 1)

            # Finish trace span and export telemetry
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
        """Convenience method for application querying."""
        return self.run_pipeline(query)

    def close(self):
        """Release connection resources."""
        if hasattr(self.vector_store, "close"):
            self.vector_store.close()
