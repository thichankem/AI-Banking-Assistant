"""
Module Tracing & Observability Chuyên Sâu cho AI Banking Assistant
======================================================================
Tích hợp đồng thời 2 nền tảng giám sát hàng đầu:
1. Langfuse:
   - Nền tảng LLM Observability & Evaluation mã nguồn mở.
   - Ghi nhận chi tiết traces, spans, generations, token usage, cost và latency.
   - Đồng bộ lên Langfuse Cloud hoặc self-hosted Langfuse server.
2. Arize Phoenix:
   - Nền tảng AI Observability chuyên sâu cho RAG & AI Agents.
   - Chuẩn hóa theo OpenTelemetry & OpenInference Semantic Conventions.
   - Trực quan hóa đồ thị RAG (Retriever, Reranker, Chain, LLM Generation).
   - Xem dashboard trực quan qua Arize Phoenix Web UI.

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. NO __INIT__.PY: Import trực tiếp file module.
"""

import sys
import os
import time
import json
import socket
import logging
from urllib.parse import urlparse
from pathlib import Path

# Đảm bảo UTF-8 an toàn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger("rag_banking.tracing")

# ---------------------------------------------------------------------------
# Import Langfuse (v5+)
# ---------------------------------------------------------------------------
LANGFUSE_AVAILABLE = False
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Import Arize Phoenix & OpenTelemetry & OpenInference
# ---------------------------------------------------------------------------
PHOENIX_AVAILABLE = False
_PHOENIX_REGISTERED = False

try:
    import phoenix as px
    from phoenix.otel import register as phoenix_register
    from opentelemetry import trace as otel_trace
    from openinference.semconv.trace import (
        SpanAttributes,
        OpenInferenceSpanKindValues,
        RerankerAttributes
    )
    PHOENIX_AVAILABLE = True
except ImportError:
    pass


def _check_endpoint_reachable(endpoint_url, timeout=0.5):
    """
    Kiểm tra nhanh xem endpoint collector (ví dụ Phoenix 6006) có đang hoạt động hay không.
    Tránh tình trạng blocking/retrying làm chậm luồng truy vấn khi chưa bật server Phoenix.
    """
    try:
        parsed = urlparse(endpoint_url)
        if parsed.hostname:
            host = parsed.hostname
        else:
            host = "127.0.0.1"

        if parsed.port is not None:
            port = parsed.port
        elif parsed.scheme == "https":
            port = 443
        else:
            port = 80

        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


class ActiveTrace:
    """
    Quản lý vòng đời và dữ liệu của một phiên truy vết (Trace) cho một câu hỏi khách hàng.
    Đồng bộ dữ liệu đồng thời vào Langfuse và Phoenix.
    """

    def __init__(self, tracer, query):
        self.tracer = tracer
        self.query = query
        self.start_time = time.time()
        self.spans = []
        self.finished = False

        # 1. Langfuse Root Observation
        self.lf_root = None
        if self.tracer.is_langfuse_active and self.tracer.lf_client:
            try:
                self.lf_root = self.tracer.lf_client.start_observation(
                    name="banking_rag_pipeline",
                    input={"query": query},
                    metadata={"project": "ai-banking-assistant", "domain": "vietcombank"}
                )
            except Exception as e:
                logger.debug(f"Langfuse start_observation error: {e}")

        # 2. Phoenix OpenTelemetry Root Span
        self.px_root_span = None
        self.px_context = None
        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                self.px_root_span = self.tracer.px_tracer.start_span("banking_rag_pipeline")
                if PHOENIX_AVAILABLE:
                    self.px_root_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                    self.px_root_span.set_attribute(SpanAttributes.INPUT_VALUE, query)
                self.px_context = otel_trace.set_span_in_context(self.px_root_span)
            except Exception as e:
                logger.debug(f"Phoenix root span error: {e}")

    def record_chunking(self, file_path, num_chunks, duration_ms, details=None):
        """Bước 0: Ghi nhận vết phân đoạn văn bản DLA Layout-Aware Chunker"""
        span_data = {
            "name": "Step 0: DLA Layout-Aware Chunking",
            "duration_ms": duration_ms,
            "file_path": str(file_path),
            "num_chunks": num_chunks,
            "details": details or {}
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("dla_layout_aware_chunking", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, f"Tài liệu đầu vào: {file_path}")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, f"Trích xuất {num_chunks} chunks cấu trúc Breadcrumb bảo toàn bố cục Layout")
                    s_px.set_attribute("chunking.num_chunks", num_chunks)
                    s_px.set_attribute("chunking.file_path", str(file_path))
                    if details:
                        for k, v in details.items():
                            s_px.set_attribute(f"chunking.{k}", str(v))
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix chunking error: {e}")

    def record_embedding(self, num_texts, dimension, duration_ms, model_name=None, cache_hits=0, is_query=False):
        """Ghi nhận vết nhúng vector 1024 chiều (Dense Vector Store)"""
        if is_query:
            span_name = "dense_query_embedding"
            disp_name = "Dense Query Embedding (1024-dim)"
            inp_text = f"Nhúng vector câu hỏi người dùng ({num_texts} query)"
        else:
            span_name = "dense_chunk_embedding"
            disp_name = "Dense Chunk Embedding (1024-dim)"
            inp_text = f"Nhúng vector cho {num_texts} chunks tài liệu"

        span_data = {
            "name": disp_name,
            "duration_ms": duration_ms,
            "num_texts": num_texts,
            "dimension": dimension,
            "model": model_name or "multilingual-e5-large",
            "cache_hits": cache_hits
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span(span_name, context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value)
                    s_px.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, model_name or "multilingual-e5-large")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, inp_text)
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, f"Tạo vector {dimension} chiều chuẩn hóa L2 (SQLite Cache hits: {cache_hits})")
                    s_px.set_attribute("embedding.num_texts", num_texts)
                    s_px.set_attribute("embedding.dimension", dimension)
                    s_px.set_attribute("embedding.cache_hits", cache_hits)
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix embedding error: {e}")

    def record_query_parser(self, input_query, parsed, duration_ms):
        """Bước 1: Ghi nhận vết Query Parser (chuẩn hóa viết tắt, phân tích thực thể)"""
        norm_query = getattr(parsed, "normalized_query", getattr(parsed, "raw_query", str(parsed)))
        keywords = getattr(parsed, "keywords", [])
        acronyms = getattr(parsed, "detected_acronyms", getattr(parsed, "expanded_acronyms", []))
        legal_refs = getattr(parsed, "legal_references", [])

        span_data = {
            "name": "Step 1: Query Parser",
            "duration_ms": duration_ms,
            "normalized_query": norm_query,
            "keywords": keywords,
            "acronyms": acronyms,
            "legal_references": legal_refs
        }
        self.spans.append(span_data)

        # Langfuse span
        if self.lf_root:
            try:
                s1 = self.lf_root.start_observation(
                    name="step1_query_parser",
                    input={"raw_query": input_query},
                    metadata={"step": 1, "duration_ms": duration_ms}
                )
                s1.update(output={
                    "normalized_query": norm_query,
                    "keywords": keywords,
                    "acronyms_detected": acronyms,
                    "legal_references": legal_refs
                })
                s1.end()
            except Exception as e:
                logger.debug(f"Langfuse s1 error: {e}")

        # Phoenix span
        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("step1_query_parser", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, input_query)
                    out_desc = f"Query chuẩn hóa: '{norm_query}' | Từ khóa: {', '.join(keywords[:5])}"
                    if acronyms:
                        out_desc = out_desc + f" | Viết tắt giải nghĩa: {', '.join(acronyms)}"
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, out_desc)
                    s_px.set_attribute("query_parser.normalized_query", norm_query)
                    s_px.set_attribute("query_parser.keywords", json.dumps(keywords, ensure_ascii=False))
                    s_px.set_attribute("query_parser.acronyms", json.dumps(acronyms, ensure_ascii=False))
                    s_px.set_attribute("query_parser.legal_references", json.dumps(legal_refs, ensure_ascii=False))
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix s1 error: {e}")

    def record_retrieval(self, query, candidates, duration_ms):
        """Bước 2: Ghi nhận vết Hybrid Retrieval (Dense Vector + BM25 hợp nhất RRF)"""
        doc_count = len(candidates)
        top_chunks_summary = []
        for i, doc in enumerate(candidates[:5]):
            if isinstance(doc, dict):
                chunk = doc.get("chunk", doc)
                score = doc.get("rrf_score", doc.get("score", 0.0))
            elif isinstance(doc, tuple) and len(doc) >= 2:
                chunk = doc[0]
                score = doc[1]
            else:
                chunk = doc
                score = 0.0

            doc_text = chunk.get("content") or chunk.get("original_text") or chunk.get("text") or ""
            if len(doc_text) > 120:
                doc_preview = doc_text[:120] + "..."
            else:
                doc_preview = doc_text

            title = chunk.get("section_title") or chunk.get("breadcrumb") or ""
            cid = chunk.get("chunk_id", f"c_{i}")

            top_chunks_summary.append({
                "index": i + 1,
                "chunk_id": cid,
                "title": title,
                "score": round(float(score), 4),
                "preview": doc_preview
            })

        span_data = {
            "name": "Step 2: Hybrid Retrieval",
            "duration_ms": duration_ms,
            "retrieved_candidates": doc_count,
            "top_candidates": top_chunks_summary
        }
        self.spans.append(span_data)

        # Langfuse span
        if self.lf_root:
            try:
                s2 = self.lf_root.start_observation(
                    name="step2_hybrid_retrieval",
                    input={"query": query},
                    metadata={"step": 2, "candidate_count": doc_count, "duration_ms": duration_ms}
                )
                s2.update(output={"candidates": top_chunks_summary})
                s2.end()
            except Exception as e:
                logger.debug(f"Langfuse s2 error: {e}")

        # Phoenix span (Chuẩn OpenInference RETRIEVER)
        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("step2_hybrid_retrieval", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, query)
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, f"Truy xuất {doc_count} ứng viên qua Dual-Index Hybrid RRF k=60 (Dense + BM25)")
                    for i, doc in enumerate(candidates[:10]):
                        if isinstance(doc, dict):
                            c_item = doc.get("chunk", doc)
                            c_score = doc.get("rrf_score", doc.get("score", 0.0))
                        elif isinstance(doc, tuple) and len(doc) >= 2:
                            c_item = doc[0]
                            c_score = doc[1]
                        else:
                            c_item = doc
                            c_score = 0.0

                        c_text = c_item.get("content") or c_item.get("original_text") or c_item.get("text") or ""
                        c_id = c_item.get("chunk_id", f"chunk_{i}")

                        prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}"
                        s_px.set_attribute(f"{prefix}.document.id", str(c_id))
                        s_px.set_attribute(f"{prefix}.document.content", c_text[:500])
                        s_px.set_attribute(f"{prefix}.document.score", float(c_score))
                        s_px.set_attribute(f"{prefix}.document.metadata", json.dumps({
                            "source": c_item.get("source", ""),
                            "breadcrumb": c_item.get("breadcrumb", ""),
                            "section_title": c_item.get("section_title", "")
                        }, ensure_ascii=False))
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix s2 error: {e}")

    def record_rerank(self, query, candidates, reranked, duration_ms):
        """Bước 3: Ghi nhận vết Reranking qua Cross-Encoder bge-reranker-v2-m3"""
        rerank_summary = []
        for i, doc in enumerate(reranked):
            if isinstance(doc, dict):
                chunk = doc.get("chunk", doc)
                score = doc.get("rerank_score", doc.get("relevance_score", doc.get("score", 0.0)))
            elif isinstance(doc, tuple) and len(doc) >= 2:
                chunk = doc[0]
                score = doc[1]
            else:
                chunk = doc
                score = 0.0

            doc_text = chunk.get("content") or chunk.get("original_text") or chunk.get("text") or ""
            if len(doc_text) > 120:
                doc_preview = doc_text[:120] + "..."
            else:
                doc_preview = doc_text

            title = chunk.get("breadcrumb") or chunk.get("section_title") or ""
            article = chunk.get("section_title") or ""
            cid = chunk.get("chunk_id", f"c_{i}")

            rerank_summary.append({
                "rank": i + 1,
                "chunk_id": cid,
                "rerank_score": round(float(score), 4),
                "article": article,
                "title": title,
                "preview": doc_preview
            })

        span_data = {
            "name": "Step 3: Reranking",
            "duration_ms": duration_ms,
            "input_count": len(candidates),
            "top_reranked_count": len(reranked),
            "top_reranked": rerank_summary
        }
        self.spans.append(span_data)

        # Langfuse span
        if self.lf_root:
            try:
                s3 = self.lf_root.start_observation(
                    name="step3_reranking",
                    input={"query": query, "input_candidates": len(candidates)},
                    metadata={"step": 3, "rerank_model": "bge-reranker-v2-m3", "duration_ms": duration_ms}
                )
                s3.update(output={"top_reranked": rerank_summary})
                s3.end()
            except Exception as e:
                logger.debug(f"Langfuse s3 error: {e}")

        # Phoenix span (Chuẩn OpenInference RERANKER)
        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("step3_reranking", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RERANKER.value)
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, query)

                    if reranked:
                        first_doc = reranked[0]
                        if isinstance(first_doc, dict):
                            best_score = first_doc.get("rerank_score", first_doc.get("relevance_score", first_doc.get("score", 0.0)))
                        elif isinstance(first_doc, tuple) and len(first_doc) >= 2:
                            best_score = first_doc[1]
                        else:
                            best_score = 0.0
                    else:
                        best_score = 0.0

                    s_px.set_attribute(
                        SpanAttributes.OUTPUT_VALUE,
                        f"Đã xếp hạng lại {len(candidates)} ứng viên -> Top {len(reranked)} chunks chuẩn xác nhất (Điểm cao nhất: {best_score:.4f})"
                    )
                    s_px.set_attribute(RerankerAttributes.RERANKER_QUERY, query)
                    s_px.set_attribute(RerankerAttributes.RERANKER_MODEL_NAME, "bge-reranker-v2-m3")
                    s_px.set_attribute(RerankerAttributes.RERANKER_TOP_K, len(reranked))

                    for i, doc in enumerate(reranked):
                        if isinstance(doc, dict):
                            c_item = doc.get("chunk", doc)
                            c_score = doc.get("rerank_score", doc.get("relevance_score", doc.get("score", 0.0)))
                        elif isinstance(doc, tuple) and len(doc) >= 2:
                            c_item = doc[0]
                            c_score = doc[1]
                        else:
                            c_item = doc
                            c_score = 0.0

                        c_text = c_item.get("content") or c_item.get("original_text") or c_item.get("text") or ""
                        c_id = c_item.get("chunk_id", f"chunk_{i}")
                        c_title = c_item.get("breadcrumb") or c_item.get("section_title") or ""

                        # Chuẩn OpenInference RERANKER output documents
                        r_prefix = f"{RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS}.{i}"
                        s_px.set_attribute(f"{r_prefix}.document.id", str(c_id))
                        s_px.set_attribute(f"{r_prefix}.document.score", float(c_score))
                        s_px.set_attribute(f"{r_prefix}.document.content", c_text[:500])

                        # Chuẩn OpenInference RETRIEVAL_DOCUMENTS để hiển thị bảng tài liệu trên Phoenix
                        ret_prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}"
                        s_px.set_attribute(f"{ret_prefix}.document.id", str(c_id))
                        s_px.set_attribute(f"{ret_prefix}.document.score", float(c_score))
                        s_px.set_attribute(f"{ret_prefix}.document.content", c_text[:500])
                        s_px.set_attribute(f"{ret_prefix}.document.metadata", json.dumps({
                            "title": c_title,
                            "rank": i + 1,
                            "rerank_score": float(c_score)
                        }, ensure_ascii=False))
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix s3 error: {e}")

    def record_synthesis(self, synthesized, duration_ms):
        """Bước 4: Ghi nhận vết Context Synthesizer (Markdown Context, Legal Citations)"""
        citations = getattr(synthesized, "legal_citations", [])
        sources = getattr(synthesized, "sources_used", [])
        tokens_est = getattr(synthesized, "total_tokens", getattr(synthesized, "total_tokens_estimated", 0))
        formatted_context = getattr(synthesized, "markdown_context", getattr(synthesized, "formatted_context", str(synthesized)))
        chunks_used = getattr(synthesized, "chunks_included", len(sources))

        span_data = {
            "name": "Step 4: Context Synthesizer",
            "duration_ms": duration_ms,
            "legal_citations": citations,
            "sources_used": sources,
            "estimated_tokens": tokens_est,
            "chunks_included": chunks_used
        }
        self.spans.append(span_data)

        # Langfuse span
        if self.lf_root:
            try:
                s4 = self.lf_root.start_observation(
                    name="step4_context_synthesizer",
                    input={"sources_count": len(sources)},
                    metadata={"step": 4, "tokens_estimated": tokens_est, "duration_ms": duration_ms, "chunks_included": chunks_used}
                )
                if len(formatted_context) > 200:
                    ctx_preview = formatted_context[:200] + "..."
                else:
                    ctx_preview = formatted_context

                s4.update(output={
                    "legal_citations": citations,
                    "sources_used": sources,
                    "context_preview": ctx_preview,
                    "tokens": tokens_est
                })
                s4.end()
            except Exception as e:
                logger.debug(f"Langfuse s4 error: {e}")

        # Phoenix span
        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("step4_context_synthesizer", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, f"Đầu vào {chunks_used} chunks từ kết quả Reranking")
                    s_px.set_attribute(
                        SpanAttributes.OUTPUT_VALUE,
                        f"Đóng gói Markdown Context ({tokens_est} tokens) kèm {len(citations)} trích dẫn pháp lý [Điều/Khoản]"
                    )
                    s_px.set_attribute("synthesis.citations", json.dumps(citations, ensure_ascii=False))
                    s_px.set_attribute("synthesis.sources", json.dumps(sources, ensure_ascii=False))
                    s_px.set_attribute("synthesis.estimated_tokens", tokens_est)
                    s_px.set_attribute("synthesis.chunks_included", chunks_used)
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix s4 error: {e}")

    def record_llm_serving(self, query, result, model, duration_ms):
        """Bước 5: Ghi nhận vết LLM Serving (Model, Guardrails, Tokens & Generation)"""
        answer = result.get("answer", "")
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", result.get("context_tokens", 0))
        completion_tokens = usage.get("completion_tokens", max(1, len(answer) // 4))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        if len(answer) > 120:
            ans_preview = answer[:120] + "..."
        else:
            ans_preview = answer

        span_data = {
            "name": "Step 5: LLM Serving",
            "duration_ms": duration_ms,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "answer_preview": ans_preview
        }
        self.spans.append(span_data)

        # Langfuse Generation observation
        if self.lf_root:
            try:
                gen = self.lf_root.start_observation(
                    name="step5_llm_serving",
                    as_type="generation",
                    model=model,
                    input=[{"role": "user", "content": query}],
                    metadata={"step": 5, "duration_ms": duration_ms}
                )
                gen.update(
                    output=answer,
                    usage_details={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": total_tokens
                    }
                )
                gen.end()
            except Exception as e:
                logger.debug(f"Langfuse s5 error: {e}")

        # Phoenix span (Chuẩn OpenInference LLM)
        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("step5_llm_serving", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
                    s_px.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, query)
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
                    s_px.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
                    s_px.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens)
                    s_px.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens)
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix s5 error: {e}")

    def record_guardrail_block(self, reason, details=None, duration_ms=0.0):
        """Ghi nhận vết chặn an ninh từ Banking Guardrails (Prompt Injection, Fraud, PII)"""
        if details:
            dt_str = str(details)
        else:
            dt_str = ""
        span_data = {
            "name": "Banking Guardrail Block",
            "duration_ms": duration_ms,
            "blocked": True,
            "reason": reason,
            "details": dt_str
        }
        self.spans.append(span_data)

        # Langfuse span
        if self.lf_root:
            try:
                sg = self.lf_root.start_observation(
                    name="guardrail_block",
                    input={"reason": reason},
                    metadata={"blocked": True, "details": dt_str}
                )
                sg.update(output={"status": "BLOCKED", "reason": reason})
                sg.end()
            except Exception as e:
                logger.debug(f"Langfuse guardrail error: {e}")

        # Phoenix span
        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("guardrail_block", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.GUARDRAIL.value)
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, self.query)
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, f"BLOCKED: {reason}")
                    s_px.set_attribute("guardrail.blocked", True)
                    s_px.set_attribute("guardrail.reason", reason)
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix guardrail error: {e}")

    def record_guardrail_check(self, layer, is_safe, details=None, duration_ms=0.0):
        """Ghi nhận vết kiểm duyệt an toàn nội dung (Output Guardrail audit)"""
        span_data = {
            "name": f"Guardrail Check: {layer}",
            "duration_ms": duration_ms,
            "is_safe": is_safe,
            "details": details or []
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("guardrail_check", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.GUARDRAIL.value)
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, layer)
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, f"Safe: {is_safe} | Details: {details or []}")
                    s_px.set_attribute("guardrail.is_safe", is_safe)
                    s_px.set_attribute("guardrail.layer", layer)
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix guardrail check error: {e}")

    def finish(
        self,
        answer,
        citations=None,
        sources=None,
        is_blocked=False,
        guardrail_trigger=None
    ):
        """Hoàn tất toàn bộ trace, đóng các root spans và lưu log"""
        if self.finished:
            return
        self.finished = True
        total_latency_ms = round((time.time() - self.start_time) * 1000, 1)

        # Đóng Langfuse Root
        if self.lf_root:
            try:
                self.lf_root.update(
                    output={"answer": answer, "citations": citations or [], "sources": sources or []},
                    metadata={
                        "total_latency_ms": total_latency_ms,
                        "is_blocked": is_blocked,
                        "guardrail_trigger": guardrail_trigger or ""
                    }
                )
                self.lf_root.end()
                # Flush bất đồng bộ
                if self.tracer.lf_client:
                    self.tracer.lf_client.flush()
            except Exception as e:
                logger.debug(f"Langfuse flush error: {e}")

        # Đóng Phoenix Root
        if self.px_root_span:
            try:
                if PHOENIX_AVAILABLE:
                    self.px_root_span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
                    self.px_root_span.set_attribute("rag.citations", json.dumps(citations or [], ensure_ascii=False))
                    self.px_root_span.set_attribute("rag.sources", json.dumps(sources or [], ensure_ascii=False))
                    self.px_root_span.set_attribute("rag.total_latency_ms", total_latency_ms)
                    self.px_root_span.set_attribute("rag.is_blocked", is_blocked)
                    if guardrail_trigger:
                        self.px_root_span.set_attribute("rag.guardrail_trigger", guardrail_trigger)
                self.px_root_span.end()
            except Exception as e:
                logger.debug(f"Phoenix end span error: {e}")

        # Lưu file JSONL cục bộ để kiểm toán offline
        if answer:
            ans_rec = answer[:200]
        else:
            ans_rec = ""

        trace_record = {
            "query": self.query,
            "answer_preview": ans_rec,
            "is_blocked": is_blocked,
            "guardrail_trigger": guardrail_trigger or "",
            "legal_citations": citations or [],
            "sources_used": sources or [],
            "total_latency_ms": total_latency_ms,
            "timestamp": time.time(),
            "spans": self.spans
        }
        self.tracer.save_local_log(trace_record)

        # Hiển thị bảng tổng hợp trên terminal
        self.tracer.print_trace_summary(trace_record)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.finished:
            if exc_val is not None:
                err_msg = str(exc_val)
                self.finish(answer=f"Pipeline exception: {err_msg}")
            else:
                self.finish(answer="")


class BankingTracer:
    """
    Trình quản lý Tracing tập trung cho toàn bộ hệ thống RAG Banking.
    Điều phối trực tiếp Langfuse và Arize Phoenix.
    """

    def __init__(self, config=None):
        from ..config import Config
        self.config = config or Config()

        self.enabled = getattr(self.config, "ENABLE_TRACING", True)
        self.enable_langfuse = getattr(self.config, "ENABLE_LANGFUSE", True)
        self.enable_phoenix = getattr(self.config, "ENABLE_PHOENIX", True)

        # Thư mục lưu vết (đặt trực tiếp trong phân hệ tracing)
        if hasattr(self.config, "TRACE_DIR"):
            self.trace_dir = self.config.TRACE_DIR
        else:
            self.trace_dir = Path(__file__).resolve().parent / "traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_log_file = self.trace_dir / "rag_traces.jsonl"

        # Khởi tạo trạng thái Langfuse & Phoenix
        self.lf_client = None
        self.is_langfuse_active = False

        self.px_tracer = None
        self.is_phoenix_active = False

        if self.enabled:
            self._init_langfuse()
            self._init_phoenix()

    def _init_langfuse(self):
        """Khởi tạo kết nối Langfuse nếu có khóa API hoặc cấu hình đầy đủ"""
        if not self.enable_langfuse or not LANGFUSE_AVAILABLE:
            return

        pk = getattr(self.config, "LANGFUSE_PUBLIC_KEY", "") or os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sk = getattr(self.config, "LANGFUSE_SECRET_KEY", "") or os.getenv("LANGFUSE_SECRET_KEY", "")
        host = getattr(self.config, "LANGFUSE_HOST", "") or os.getenv("LANGFUSE_HOST", "")

        if pk and sk and host:
            try:
                self.lf_client = Langfuse(public_key=pk, secret_key=sk, host=host)
                self.is_langfuse_active = True
                print(f"📡 [Langfuse] Đã kết nối Langfuse Observability ({host})")
            except Exception as e:
                logger.warning(f"Không thể khởi tạo Langfuse client: {e}")
                self.is_langfuse_active = False
        else:
            logger.debug("[Langfuse] Thiếu LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST. Chế độ Langfuse tạm thời tắt.")

    def _init_phoenix(self):
        """Khởi tạo kết nối Arize Phoenix qua OpenTelemetry TracerProvider"""
        global _PHOENIX_REGISTERED
        if not self.enable_phoenix or not PHOENIX_AVAILABLE:
            return

        endpoint = getattr(self.config, "PHOENIX_COLLECTOR_ENDPOINT", "") or os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "")
        project_name = getattr(self.config, "PHOENIX_PROJECT_NAME", "") or os.getenv("PHOENIX_PROJECT_NAME", "")

        if not endpoint:
            self.is_phoenix_active = False
            return

        # Kiểm tra nhanh xem collector có sẵn sàng không
        if _check_endpoint_reachable(endpoint, timeout=0.3):
            try:
                if not _PHOENIX_REGISTERED:
                    phoenix_register(
                        project_name=project_name,
                        endpoint=endpoint,
                        batch=False,
                        protocol="http/protobuf",
                        verbose=False
                    )
                    _PHOENIX_REGISTERED = True
                self.px_tracer = otel_trace.get_tracer("rag_banking_tracer")
                self.is_phoenix_active = True
                print(f"🔥 [Phoenix] Đã kết nối Arize Phoenix Observability ({endpoint})")
            except Exception as e:
                logger.warning(f"Không thể kết nối Phoenix OTel: {e}")
                self.is_phoenix_active = False
        else:
            # Phoenix server chưa khởi động, tắt tạm thời để tránh log lỗi kết nối
            self.is_phoenix_active = False
            logger.debug(f"[Phoenix] Server chưa bật tại {endpoint}. Khởi chạy bằng 'python scripts/run_phoenix.py'.")

    def trace_chunking(self, file_path, num_chunks, duration_ms, details=None):
        """
        Ghi vết phân đoạn tài liệu DLA Layout-Aware độc lập (pha Ingestion).
        Hiển thị ngay trên Phoenix dashboard dưới dạng ROOT SPAN.
        """
        fname = Path(file_path).name
        record = {
            "query": f"Ingestion Chunking: {fname}",
            "file_path": str(file_path),
            "num_chunks": num_chunks,
            "total_latency_ms": duration_ms,
            "timestamp": time.time(),
            "spans": [{
                "name": "DLA Layout-Aware Chunking",
                "duration_ms": duration_ms,
                "file_path": str(file_path),
                "num_chunks": num_chunks,
                "details": details or {}
            }]
        }
        self.save_local_log(record)

        if self.is_phoenix_active and self.px_tracer:
            try:
                s_px = self.px_tracer.start_span("dla_layout_aware_chunking")
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, f"Tài liệu ngân hàng đầu vào: {file_path}")
                    s_px.set_attribute(
                        SpanAttributes.OUTPUT_VALUE,
                        f"Đã trích xuất {num_chunks} chunks cấu trúc Breadcrumb bảo toàn bố cục văn bản"
                    )
                    s_px.set_attribute("chunking.file_path", str(file_path))
                    s_px.set_attribute("chunking.num_chunks", num_chunks)
                    s_px.set_attribute("chunking.duration_ms", duration_ms)
                    if details:
                        for k, v in details.items():
                            s_px.set_attribute(f"chunking.{k}", str(v))
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix trace_chunking error: {e}")

    def trace_embedding(self, num_texts, dimension, duration_ms, model_name=None, cache_hits=0):
        """
        Ghi vết nhúng vector độc lập (pha Ingestion indexing).
        Hiển thị ngay trên Phoenix dashboard dưới dạng ROOT SPAN.
        """
        record = {
            "query": f"Ingestion Embedding ({num_texts} chunks)",
            "num_texts": num_texts,
            "dimension": dimension,
            "total_latency_ms": duration_ms,
            "timestamp": time.time(),
            "spans": [{
                "name": "Dense Chunk Embedding (1024-dim)",
                "duration_ms": duration_ms,
                "num_texts": num_texts,
                "dimension": dimension,
                "model": model_name or "multilingual-e5-large",
                "cache_hits": cache_hits
            }]
        }
        self.save_local_log(record)

        if self.is_phoenix_active and self.px_tracer:
            try:
                s_px = self.px_tracer.start_span("dense_chunk_embedding")
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value)
                    s_px.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, model_name or "multilingual-e5-large")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, f"Nhúng vector cho {num_texts} chunks tài liệu vào Qdrant")
                    s_px.set_attribute(
                        SpanAttributes.OUTPUT_VALUE,
                        f"Đã tạo {num_texts} vector {dimension} chiều (SQLite Cache hits: {cache_hits})"
                    )
                    s_px.set_attribute("embedding.num_texts", num_texts)
                    s_px.set_attribute("embedding.dimension", dimension)
                    s_px.set_attribute("embedding.cache_hits", cache_hits)
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix trace_embedding error: {e}")

    def trace_query(self, query):
        """Tạo một phiên truy vết mới cho một câu hỏi ngân hàng"""
        return ActiveTrace(self, query)

    def save_local_log(self, record):
        """Lưu trữ lịch sử truy vết dạng JSON Lines"""
        try:
            with open(self.trace_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Không thể ghi log trace cục bộ: {e}")

    def print_trace_summary(self, record):
        """In bảng thông tin truy vết trực quan dạng terminal chuẩn"""
        total_ms = record.get("total_latency_ms", 0.0)
        query = record.get("query", "")
        spans = record.get("spans", [])

        # Chỉ báo đồng bộ
        sync_tags = []
        if self.is_langfuse_active:
            sync_tags.append("Langfuse: ON")
        if self.is_phoenix_active:
            sync_tags.append("Phoenix: ON")

        if sync_tags:
            sync_str = f" [{', '.join(sync_tags)}]"
        else:
            sync_str = ""

        if len(query) > 50:
            q_disp = query[:50] + "..."
        else:
            q_disp = query

        header = f"🔍 [RAG TRACE OBSERVABILITY]{sync_str} - Tổng: {total_ms:.1f}ms"
        
        print("\n" + "┌" + "─" * 70 + "┐")
        print(f"│ {header:<68} │")
        print("├" + "─" * 70 + "┤")
        print(f"│  Query: \"{q_disp}\"" + " " * max(1, 68 - len(f"│  Query: \"{q_disp}\"")) + "│")
        if record.get("is_blocked"):
            block_line = f"│  🛑 CHẶN BẢO MẬT: {record.get('guardrail_trigger', '')}"
            if len(block_line) < 70:
                print(block_line + " " * (70 - len(block_line)) + "│")
            else:
                print(block_line[:69] + "│")
        print("│  Tiến trình thực thi phân hệ:" + " " * (69 - len("│  Tiến trình thực thi phân hệ:")) + "│")

        for idx, span in enumerate(spans):
            if idx == len(spans) - 1:
                branch = "└─"
            else:
                branch = "├─"
            name = span.get("name", "")
            duration = span.get("duration_ms", 0.0)
            line = f"│   {branch} ⚡ {name}: {duration:.1f}ms"
            if len(line) < 70:
                print(line + " " * (70 - len(line)) + "│")
            else:
                print(line[:69] + "│")

        citations = record.get("legal_citations", [])
        if citations:
            c_str = f"Căn cứ: {', '.join(citations[:3])}"
            c_line = f"│  📜 {c_str}"
            if len(c_line) < 70:
                print(c_line + " " * (70 - len(c_line)) + "│")
            else:
                print(c_line[:69] + "│")

        print("└" + "─" * 70 + "┘\n")


# Khởi tạo singleton tracer
tracer = BankingTracer()
