"""
Module Tracing & Observability Chuyên Sâu cho AI Banking Assistant
======================================================================
Tích hợp nền tảng giám sát AI Observability hàng đầu:
1. Arize Phoenix:
   - Nền tảng AI Observability chuyên sâu cho RAG & AI Agents.
   - Chuẩn hóa theo OpenTelemetry & OpenInference Semantic Conventions.
   - Trực quan hóa đồ thị RAG (Retriever, Reranker, Chain, LLM Generation).
   - Xem dashboard trực quan qua Arize Phoenix Web UI.
2. Local Tracing:
   - Lưu trữ nhật ký vết định dạng JSONL (rag_traces.jsonl).

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
import math
from urllib.parse import urlparse
from pathlib import Path

# Đảm bảo UTF-8 an toàn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger("agent_banking.tracing")

# ---------------------------------------------------------------------------
# Import Arize Phoenix & OpenTelemetry & OpenInference
# ---------------------------------------------------------------------------
PHOENIX_AVAILABLE = False
_PHOENIX_REGISTERED = False
_GLOBAL_PX_TRACER = None

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
    Đồng bộ dữ liệu vào Arize Phoenix và lưu trữ cục bộ.
    """

    def __init__(self, tracer, query):
        self.tracer = tracer
        self.query = query
        self.start_time = time.time()
        self.spans = []
        self.finished = False
        self.lf_root = None

        # Phoenix OpenTelemetry Root Span
        self.px_root_span = None
        self.px_context = None
        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                self.px_root_span = self.tracer.px_tracer.start_span("rag_query_pipeline")
                if PHOENIX_AVAILABLE:
                    self.px_root_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                    root_inp = {
                        "user_query": query,
                        "pipeline": "AI Banking Hybrid RAG Pipeline (Chunking -> Guardrails -> Query Parser -> Dense Embedding -> Top-K Hybrid Retrieval -> Top-K Reranking -> Synthesizer -> LLM Serving -> Output Guardrail)",
                        "tracked_stages": [
                            "1. chunking_layout_aware (DLA Layout-Aware Hierarchy Chunking)",
                            "2. input_guardrail (Prompt Injection & Security)",
                            "3. query_parser (Entities & Keyword extraction)",
                            "4. dense_embedding (1024-dim Vector representation)",
                            "5. top_k_hybrid_retrieval (Top-5 Chunks: Dense HNSW + BM25 + RRF)",
                            "6. top_k_cross_encoder_reranker (Top-3 Chunks: bge-reranker-v2-m3)",
                            "7. context_synthesizer (Token budgeting & Citation map)",
                            "8. llm_generation (DeepSeek-V4-Flash / Persona & Tone)",
                            "9. output_guardrail (PII mask & Safety audit)"
                        ]
                    }
                    self.px_root_span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    self.px_root_span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(root_inp, ensure_ascii=False, indent=2))
                self.px_context = otel_trace.set_span_in_context(self.px_root_span)
            except Exception as e:
                logger.debug(f"Phoenix root span error: {e}")

    def record_chunking_info(self, total_chunks, max_size=800, strategy="DLA Layout-Aware Hierarchy Chunking", duration_ms=0.5):
        """Bước 1: Ghi nhận vết hệ thống phân đoạn Chunking DLA Layout-Aware vào Phoenix"""
        span_data = {
            "name": "Node 1: DLA Layout-Aware Chunking",
            "duration_ms": duration_ms,
            "total_chunks_stored": total_chunks,
            "strategy": strategy,
            "max_size": max_size
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer and self.px_context:
            try:
                s_px = self.tracer.px_tracer.start_span("chunking_layout_aware", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                    s_px.set_attribute("chunking.strategy", str(strategy))
                    s_px.set_attribute("chunking.total_indexed_chunks", int(total_chunks))
                    s_px.set_attribute("chunking.max_characters_per_chunk", int(max_size))
                    s_px.set_attribute("chunking.keep_tables_atomic", True)
                    s_px.set_attribute("chunking.hierarchy_preservation", "Heading (I. -> 1. -> a) -> Điều)")
                    
                    inp_dict = {
                        "chunking_algorithm": strategy,
                        "chunk_size_limit": max_size,
                        "table_policy": "Keep atomic (Preserve markdown tables)"
                    }
                    out_dict = {
                        "status": "READY",
                        "total_indexed_chunks": total_chunks,
                        "chunking_description": "Kho văn bản quy chuẩn đã được phân đoạn theo cấu trúc phân cấp Layout-Aware."
                    }
                    s_px.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(inp_dict, ensure_ascii=False, indent=2))
                    s_px.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(out_dict, ensure_ascii=False, indent=2))
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix chunking_info error: {e}")

    def record_chunking(self, file_path, num_chunks, duration_ms, details=None, chunks=None, sample_chunks=None):
        """Bước 0: Ghi nhận vết phân đoạn văn bản DLA Layout-Aware Chunker toàn diện tất cả chunks"""
        if details:
            dt_dict = details
        else:
            dt_dict = {}

        if chunks is not None:
            raw_chunks = chunks
        elif sample_chunks is not None:
            raw_chunks = sample_chunks
        else:
            raw_chunks = []

        extracted_chunks_list = []
        total_characters = 0
        total_tokens = 0

        for i, c in enumerate(raw_chunks):
            if isinstance(c, dict):
                c_id = c.get("chunk_id", f"chunk_{i + 1}")
                c_bc = c.get("breadcrumb", "")
                c_title = c.get("section_title", "")
                c_tok = c.get("token_count", 0)
                c_raw = c.get("content", "") or c.get("original_text", "") or c.get("text", "")
            else:
                c_id = getattr(c, "chunk_id", f"chunk_{i + 1}")
                c_bc = getattr(c, "breadcrumb", "")
                c_title = getattr(c, "section_title", "")
                c_tok = getattr(c, "token_count", 0)
                c_raw = getattr(c, "content", "") or getattr(c, "original_text", "") or getattr(c, "text", "")

            char_cnt = len(c_raw)
            total_characters += char_cnt
            if isinstance(c_tok, (int, float)):
                total_tokens += int(c_tok)

            extracted_chunks_list.append({
                "chunk_index": i + 1,
                "chunk_id": str(c_id),
                "breadcrumb": c_bc,
                "section_title": c_title,
                "token_count": c_tok,
                "char_count": char_cnt,
                "content": c_raw,
                "full_text_content": c_raw
            })

        if num_chunks:
            final_count = num_chunks
        else:
            final_count = len(extracted_chunks_list)

        if extracted_chunks_list:
            avg_tok = round(total_tokens / len(extracted_chunks_list), 1)
            avg_ch = round(total_characters / len(extracted_chunks_list), 1)
        else:
            avg_tok = 0
            avg_ch = 0

        span_data = {
            "name": "DLA Layout-Aware Chunking",
            "duration_ms": duration_ms,
            "file_path": str(file_path),
            "num_chunks": final_count,
            "total_tokens": total_tokens,
            "details": dt_dict
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("chunking_dla_layout_aware", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)

                    inp_payload = {
                        "action": "document_layout_aware_chunking",
                        "file_path": str(file_path),
                        "file_name": Path(file_path).name,
                        "chunking_strategy": "DLA Layout-Aware Hierarchy Chunking (Breadcrumb preserved)",
                        "hierarchy_rules": [
                            "Preserve document heading hierarchy (I. -> 1. -> a) -> Điều)",
                            "Attach contextual breadcrumb navigation prefix to each chunk",
                            "Preserve table structures and legal clauses"
                        ],
                        "options": dt_dict
                    }

                    out_payload = {
                        "status": "SUCCESS",
                        "total_chunks_extracted": final_count,
                        "chunking_duration_ms": duration_ms,
                        "summary_statistics": {
                            "total_chunks": final_count,
                            "total_characters": total_characters,
                            "total_tokens": total_tokens,
                            "average_tokens_per_chunk": avg_tok,
                            "average_characters_per_chunk": avg_ch
                        },
                        "extracted_chunks": extracted_chunks_list
                    }

                    s_px.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(inp_payload, ensure_ascii=False, indent=2))
                    s_px.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(out_payload, ensure_ascii=False, indent=2))
                    s_px.set_attribute("chunking.num_chunks", final_count)
                    s_px.set_attribute("chunking.file_path", str(file_path))
                    s_px.set_attribute("chunking.duration_ms", duration_ms)
                    s_px.set_attribute("chunking.total_characters", total_characters)
                    s_px.set_attribute("chunking.total_tokens", total_tokens)
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix chunking error: {e}")

    def record_embedding(
        self,
        num_texts,
        dimension,
        duration_ms,
        model_name=None,
        cache_hits=0,
        is_query=False,
        query_text=None,
        query_vector=None,
        chunks=None
    ):
        """Ghi nhận vết nhúng vector 1024 chiều (Dense Vector Store) chi tiết vector"""
        if model_name:
            chosen_model = model_name
        else:
            chosen_model = "multilingual-e5-large"

        if is_query:
            span_name = "dense_embedding"
            disp_name = "Dense Query Embedding (1024-dim)"
            if query_text:
                q_content = query_text
            else:
                q_content = self.query

            formatted_query_input = f"query: {q_content}"

            inp_obj = {
                "action": "query_dense_embedding",
                "input_query": q_content,
                "model": chosen_model,
                "instruction_prefix": "query: ",
                "formatted_embedding_input": formatted_query_input,
                "target_dimension": dimension
            }

            if query_vector is not None and isinstance(query_vector, (list, tuple)):
                q_vec_list = list(query_vector)
                first_8 = [round(float(x), 5) for x in q_vec_list[:8]]
                first_10 = [round(float(x), 6) for x in q_vec_list[:10]]
                norm_val = round(math.sqrt(sum(float(x) * float(x) for x in q_vec_list)), 4)
                vector_preview_str = f"[{', '.join(str(x) for x in first_8)}, ... ({len(q_vec_list)} dimensions)]"
            else:
                first_8 = []
                first_10 = []
                norm_val = 1.0
                vector_preview_str = f"[{dimension}-dim normalized vector, dimension={dimension}]"

            out_obj = {
                "status": "QUERY_EMBEDDED_SUCCESS",
                "output_dimension": dimension,
                "l2_normalized": True,
                "vector_norm": norm_val,
                "cache_hits": cache_hits,
                "vector_preview": vector_preview_str,
                "first_10_floats": first_10,
                "vector_sample_first_10_floats": first_10,
                "duration_ms": duration_ms
            }
        else:
            span_name = "dense_embedding"
            disp_name = "Dense Chunk Embedding (1024-dim)"

            input_chunks_summary = []
            embedded_chunks_details = []

            if chunks:
                for i, c in enumerate(chunks):
                    if isinstance(c, dict):
                        c_id = c.get("chunk_id", f"chunk_{i + 1}")
                        c_bc = c.get("breadcrumb", "")
                        c_raw = c.get("content", "") or c.get("original_text", "") or c.get("text", "")
                        c_vec = c.get("vector")
                    else:
                        c_id = getattr(c, "chunk_id", f"chunk_{i + 1}")
                        c_bc = getattr(c, "breadcrumb", "")
                        c_raw = getattr(c, "content", "") or getattr(c, "original_text", "") or getattr(c, "text", "")
                        c_vec = getattr(c, "vector", None)

                    if c_bc:
                        formatted_passage = f"passage: [Ngữ cảnh: {c_bc}]\n{c_raw}"
                    else:
                        formatted_passage = f"passage: {c_raw}"

                    input_chunks_summary.append({
                        "chunk_index": i + 1,
                        "chunk_id": str(c_id),
                        "breadcrumb": c_bc,
                        "formatted_input_for_model": formatted_passage
                    })

                    if c_vec is not None and isinstance(c_vec, (list, tuple)):
                        v_list = list(c_vec)
                        v_first8 = [round(float(x), 5) for x in v_list[:8]]
                        v_first10 = [round(float(x), 6) for x in v_list[:10]]
                        v_norm = round(math.sqrt(sum(float(x) * float(x) for x in v_list)), 4)
                        v_prev = f"[{', '.join(str(x) for x in v_first8)}, ... ({len(v_list)} dimensions)]"
                        v_dim = len(v_list)
                    else:
                        v_first8 = []
                        v_first10 = []
                        v_norm = 1.0
                        v_prev = f"[{dimension}-dim vector generated]"
                        v_dim = dimension

                    embedded_chunks_details.append({
                        "chunk_index": i + 1,
                        "chunk_id": str(c_id),
                        "breadcrumb": c_bc,
                        "vector_dimension": v_dim,
                        "l2_norm": v_norm,
                        "vector_preview": v_prev,
                        "first_10_floats": v_first10,
                        "vector_sample_first_10_floats": v_first10,
                        "content": c_raw,
                        "full_text_content": c_raw,
                        "raw_content": c_raw
                    })

            inp_obj = {
                "action": "chunk_batch_embedding",
                "num_chunks": num_texts,
                "model": chosen_model,
                "instruction_prefix": "passage: ",
                "target_dimension": dimension,
                "chunks_to_embed": input_chunks_summary
            }
            out_obj = {
                "status": "BATCH_EMBEDDED_SUCCESS",
                "total_vectors_generated": num_texts,
                "dimension": dimension,
                "sqlite_cache_hits": cache_hits,
                "duration_ms": duration_ms,
                "embedded_chunks_details": embedded_chunks_details
            }

        span_data = {
            "name": disp_name,
            "duration_ms": duration_ms,
            "num_texts": num_texts,
            "dimension": dimension,
            "model": chosen_model,
            "cache_hits": cache_hits
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span(span_name, context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value)
                    s_px.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, chosen_model)
                    s_px.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(inp_obj, ensure_ascii=False, indent=2))
                    s_px.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(out_obj, ensure_ascii=False, indent=2))
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

        # Phoenix span
        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("query_parser", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)

                    acronym_list = []
                    if isinstance(acronyms, list):
                        for item in acronyms:
                            if isinstance(item, tuple) and len(item) == 2:
                                acronym_list.append({"acronym": str(item[0]), "expanded": str(item[1])})
                            elif isinstance(item, dict):
                                acronym_list.append(item)
                            else:
                                acronym_list.append({"acronym": str(item)})
                    elif isinstance(acronyms, dict):
                        for k, v in acronyms.items():
                            acronym_list.append({"acronym": k, "expanded": str(v)})

                    inp_obj = {
                        "raw_user_query": input_query
                    }
                    out_obj = {
                        "normalized_query": norm_query,
                        "extracted_keywords": keywords,
                        "detected_banking_acronyms": acronym_list,
                        "legal_references_found": legal_refs,
                        "parsing_duration_ms": duration_ms
                    }
                    s_px.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(inp_obj, ensure_ascii=False, indent=2))
                    s_px.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(out_obj, ensure_ascii=False, indent=2))
                    s_px.set_attribute("query_parser.normalized_query", norm_query)
                    s_px.set_attribute("query_parser.keywords", json.dumps(keywords, ensure_ascii=False))
                    s_px.set_attribute("query_parser.acronyms", json.dumps(acronym_list, ensure_ascii=False))
                    s_px.set_attribute("query_parser.legal_references", json.dumps(legal_refs, ensure_ascii=False))
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix query parser error: {e}")

    def record_retrieval(self, query, candidates, duration_ms, params=None):
        """Bước 2: Ghi nhận vết Hybrid Retrieval (Dense Vector + BM25 hợp nhất RRF) đầy đủ tất cả chunk"""
        doc_count = len(candidates)
        top_chunks_summary = []
        candidate_details = []

        for i, doc in enumerate(candidates):
            if isinstance(doc, dict):
                c_item = doc.get("chunk", doc)
                c_score = doc.get("rrf_score", doc.get("score", 0.0))
                dense_sc = doc.get("dense_score", 0.0)
                bm25_sc = doc.get("bm25_score", 0.0)
            elif isinstance(doc, tuple) and len(doc) >= 2:
                c_item = doc[0]
                c_score = doc[1]
                dense_sc = 0.0
                bm25_sc = 0.0
            else:
                c_item = doc
                c_score = 0.0
                dense_sc = 0.0
                bm25_sc = 0.0

            doc_text = c_item.get("content") or c_item.get("original_text") or c_item.get("text") or ""
            title = c_item.get("section_title") or c_item.get("breadcrumb") or ""
            cid = c_item.get("chunk_id", f"c_{i}")
            c_source = c_item.get("source", "")

            c_info = {
                "rank": i + 1,
                "chunk_id": str(cid),
                "rrf_score": round(float(c_score), 4),
                "dense_score": round(float(dense_sc), 4),
                "bm25_score": round(float(bm25_sc), 2),
                "source_file": c_source,
                "breadcrumb": title,
                "content": doc_text,
                "char_count": len(doc_text)
            }
            candidate_details.append(c_info)

            top_chunks_summary.append({
                "index": i + 1,
                "chunk_id": str(cid),
                "title": title,
                "score": round(float(c_score), 4),
                "content": doc_text
            })

        span_data = {
            "name": "Step 2: Hybrid Retrieval",
            "duration_ms": duration_ms,
            "retrieved_candidates": doc_count,
            "candidates": candidate_details
        }
        self.spans.append(span_data)

        # Phoenix span (Chuẩn OpenInference RETRIEVER)
        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("top_k_hybrid_retrieval", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
                    s_px.set_attribute("retrieval.top_k", int(doc_count))
                    s_px.set_attribute("top_k", int(doc_count))
                    
                    if params:
                        ret_params = params
                    else:
                        ret_params = {
                            "dense_k": 20,
                            "bm25_k": 20,
                            "rrf_k": 60,
                            "returned_candidates": doc_count
                        }

                    inp_obj = {
                        "search_query": query,
                        "retrieval_strategy": "Hybrid RRF (Dense Vector Qdrant HNSW + Okapi BM25)",
                        "fusion_algorithm": "Reciprocal Rank Fusion (k=60)",
                        "top_k_requested": doc_count,
                        "parameters": ret_params
                    }
                    out_obj = {
                        "total_candidates_retrieved": doc_count,
                        "top_k": doc_count,
                        "retrieval_latency_ms": duration_ms,
                        "candidates": candidate_details
                    }
                    s_px.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(inp_obj, ensure_ascii=False, indent=2))
                    s_px.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(out_obj, ensure_ascii=False, indent=2))

                    for i, cand in enumerate(candidate_details):
                        prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}"
                        s_px.set_attribute(f"{prefix}.document.id", f"[Top-{i+1} Retrieval Chunk] {cand['chunk_id']}")
                        s_px.set_attribute(f"{prefix}.document.content", cand["content"])
                        s_px.set_attribute(f"{prefix}.document.score", cand["rrf_score"])
                        s_px.set_attribute(f"{prefix}.document.metadata", json.dumps({
                            "rank": f"Top-{i+1}",
                            "source": cand["source_file"],
                            "breadcrumb": cand["breadcrumb"],
                            "dense_score": cand["dense_score"],
                            "bm25_score": cand["bm25_score"]
                        }, ensure_ascii=False))
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix top_k_hybrid_retrieval error: {e}")

    def record_rerank(self, query, candidates, reranked, duration_ms):
        """Bước 3: Ghi nhận vết Reranking qua Cross-Encoder bge-reranker-v2-m3 đầy đủ tất cả chunk"""
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
            title = chunk.get("breadcrumb") or chunk.get("section_title") or ""
            article = chunk.get("section_title") or ""
            cid = chunk.get("chunk_id", f"c_{i}")

            rerank_summary.append({
                "rank": i + 1,
                "chunk_id": cid,
                "rerank_score": round(float(score), 4),
                "article": article,
                "title": title,
                "content": doc_text
            })

        span_data = {
            "name": "Step 3: Reranking",
            "duration_ms": duration_ms,
            "input_count": len(candidates),
            "top_reranked_count": len(reranked),
            "top_reranked": rerank_summary
        }
        self.spans.append(span_data)

        # Phoenix span (Chuẩn OpenInference RERANKER)
        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                s_px = self.tracer.px_tracer.start_span("top_k_cross_encoder_reranker", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RERANKER.value)
                    s_px.set_attribute(RerankerAttributes.RERANKER_QUERY, query)
                    s_px.set_attribute(RerankerAttributes.RERANKER_MODEL_NAME, "bge-reranker-v2-m3")
                    s_px.set_attribute(RerankerAttributes.RERANKER_TOP_K, len(reranked))
                    s_px.set_attribute("reranker.top_k", len(reranked))
                    s_px.set_attribute("top_k", len(reranked))

                    # 1. Input candidates summary
                    input_candidates_summary = []
                    for i, doc in enumerate(candidates):
                        if isinstance(doc, dict):
                            c_item = doc.get("chunk", doc)
                            c_score = doc.get("rrf_score", doc.get("score", 0.0))
                        elif isinstance(doc, tuple) and len(doc) >= 2:
                            c_item = doc[0]
                            c_score = doc[1]
                        else:
                            c_item = doc
                            c_score = 0.0

                        c_id = c_item.get("chunk_id", f"c_{i}")
                        c_title = c_item.get("breadcrumb") or c_item.get("section_title") or ""
                        c_text = c_item.get("content") or c_item.get("original_text") or c_item.get("text") or ""
                        c_source = c_item.get("source", "")

                        input_candidates_summary.append({
                            "input_rank": i + 1,
                            "chunk_id": str(c_id),
                            "initial_score": round(float(c_score), 4),
                            "source": c_source,
                            "breadcrumb": c_title,
                            "content": c_text,
                            "char_count": len(c_text)
                        })

                        # Ghi nhận toàn bộ Candidate đầu vào vào OpenInference RERANKER_INPUT_DOCUMENTS
                        in_prefix = f"{RerankerAttributes.RERANKER_INPUT_DOCUMENTS}.{i}"
                        s_px.set_attribute(f"{in_prefix}.document.id", f"[Candidate-{i+1}/{len(candidates)}] {c_id}")
                        s_px.set_attribute(f"{in_prefix}.document.content", c_text)
                        s_px.set_attribute(f"{in_prefix}.document.score", float(c_score))

                    # 2. Output reranked summary (Top-K Chunks được chọn)
                    reranked_results = []
                    best_score = 0.0
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
                        c_source = c_item.get("source", "")

                        if i == 0:
                            best_score = round(float(c_score), 4)

                        orig_rank = None
                        for inc in input_candidates_summary:
                            if inc["chunk_id"] == str(c_id):
                                orig_rank = inc["input_rank"]
                                break

                        if orig_rank is not None:
                            rank_shift = orig_rank - (i + 1)
                        else:
                            rank_shift = 0

                        if rank_shift > 0:
                            shift_str = f"+{rank_shift}"
                        else:
                            shift_str = str(rank_shift)

                        reranked_results.append({
                            "final_rank": i + 1,
                            "chunk_id": str(c_id),
                            "cross_encoder_score": round(float(c_score), 4),
                            "original_rank": orig_rank,
                            "rank_shift": shift_str,
                            "source": c_source,
                            "breadcrumb": c_title,
                            "content": c_text,
                            "char_count": len(c_text)
                        })

                        # OpenInference RERANKER_OUTPUT_DOCUMENTS (Top-K sau khi Cross-Encoder chấm điểm)
                        r_prefix = f"{RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS}.{i}"
                        s_px.set_attribute(f"{r_prefix}.document.id", f"[Top-{i+1}/{len(reranked)} Rerank Chunk] {c_id}")
                        s_px.set_attribute(f"{r_prefix}.document.score", float(c_score))
                        s_px.set_attribute(f"{r_prefix}.document.content", c_text)

                        # OpenInference RETRIEVAL_DOCUMENTS
                        ret_prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}"
                        s_px.set_attribute(f"{ret_prefix}.document.id", f"[Top-{i+1}/{len(reranked)} Rerank Chunk] {c_id}")
                        s_px.set_attribute(f"{ret_prefix}.document.score", float(c_score))
                        s_px.set_attribute(f"{ret_prefix}.document.content", c_text)
                        s_px.set_attribute(f"{ret_prefix}.document.metadata", json.dumps({
                            "top_k_rank": f"Top-{i+1} of {len(reranked)}",
                            "initial_rank": orig_rank,
                            "rank_shift": shift_str,
                            "source": c_source,
                            "breadcrumb": c_title,
                            "cross_encoder_score": float(c_score)
                        }, ensure_ascii=False))

                    inp_obj = {
                        "query": query,
                        "model": "bge-reranker-v2-m3",
                        "total_input_candidates": len(candidates),
                        "requested_top_k": len(reranked),
                        "candidates_before_rerank": input_candidates_summary
                    }
                    out_obj = {
                        "model": "bge-reranker-v2-m3",
                        "rerank_latency_ms": duration_ms,
                        "reranked_top_k": len(reranked),
                        "best_relevance_score": best_score,
                        "reranked_results": reranked_results
                    }
                    s_px.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(inp_obj, ensure_ascii=False, indent=2))
                    s_px.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(out_obj, ensure_ascii=False, indent=2))
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix cross_encoder_reranker error: {e}")

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

        if self.tracer.is_phoenix_active and self.tracer.px_tracer and self.px_context:
            try:
                s_px = self.tracer.px_tracer.start_span("context_synthesizer", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                    inp_dict = {
                        "action": "synthesize_context",
                        "chunks_count": chunks_used,
                        "sources_count": len(sources),
                        "estimated_tokens": tokens_est
                    }
                    if len(formatted_context) > 600:
                        ctx_preview = formatted_context[:600] + "..."
                    else:
                        ctx_preview = formatted_context
                    out_dict = {
                        "legal_citations": citations,
                        "sources_used": sources,
                        "context_tokens": tokens_est,
                        "context_preview": ctx_preview
                    }
                    s_px.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(inp_dict, ensure_ascii=False, indent=2))
                    s_px.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(out_dict, ensure_ascii=False, indent=2))
                    s_px.set_attribute("synthesis.duration_ms", duration_ms)
                    s_px.set_attribute("synthesis.tokens_count", tokens_est)
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix context_synthesizer error: {e}")

    def record_llm_serving(self, query, result, model, duration_ms, persona=None, tone=None, format_style=None):
        """Bước 5: Ghi nhận vết LLM Serving (Model, Guardrails, Persona, Tokens & Generation)"""
        answer = result.get("answer", "")
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", result.get("context_tokens", 0))
        completion_tokens = usage.get("completion_tokens", max(1, len(answer) // 4))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        chosen_persona = persona or result.get("persona", "advisor")
        chosen_tone = tone or result.get("tone", "formal")
        chosen_format = format_style or result.get("format_style", "standard")

        if len(answer) > 120:
            ans_preview = answer[:120] + "..."
        else:
            ans_preview = answer

        span_data = {
            "name": "Step 5: LLM Serving",
            "duration_ms": duration_ms,
            "model": model,
            "persona": chosen_persona,
            "tone": chosen_tone,
            "format_style": chosen_format,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "answer_preview": ans_preview
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer and self.px_context:
            try:
                s_px = self.tracer.px_tracer.start_span("llm_generation", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
                    s_px.set_attribute(SpanAttributes.LLM_MODEL_NAME, str(model))
                    s_px.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(prompt_tokens))
                    s_px.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(completion_tokens))
                    s_px.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, int(total_tokens))
                    
                    llm_input_info = {
                        "user_query": query,
                        "persona_configuration": {
                            "persona": chosen_persona,
                            "tone": chosen_tone,
                            "format_style": chosen_format
                        },
                        "estimated_prompt_tokens": prompt_tokens
                    }
                    s_px.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(llm_input_info, ensure_ascii=False, indent=2))
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
                    s_px.set_attribute("llm.latency_ms", duration_ms)
                    s_px.set_attribute("llm.persona", str(chosen_persona))
                    s_px.set_attribute("llm.tone", str(chosen_tone))
                    s_px.set_attribute("llm.format_style", str(chosen_format))
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix llm_generation error: {e}")

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

        if self.tracer.is_phoenix_active and self.tracer.px_tracer and self.px_context:
            try:
                s_px = self.tracer.px_tracer.start_span("input_guardrail_blocked", context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.GUARDRAIL.value)
                    s_px.set_attribute("guardrail.blocked", True)
                    s_px.set_attribute("guardrail.reason", str(reason))
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, f"BLOCKED: {reason}")
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix guardrail block error: {e}")

    def record_guardrail_check(self, layer, is_safe, details=None, duration_ms=0.0):
        """Ghi nhận vết kiểm duyệt an toàn nội dung (Input/Output Guardrail audit)"""
        if details:
            dt_list = details
        else:
            dt_list = []

        span_data = {
            "name": f"Guardrail Check: {layer}",
            "duration_ms": duration_ms,
            "is_safe": is_safe,
            "details": dt_list
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer and self.px_context:
            try:
                span_name = f"guardrail_{str(layer).lower().replace(' ', '_')}"
                s_px = self.tracer.px_tracer.start_span(span_name, context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.GUARDRAIL.value)
                    s_px.set_attribute("guardrail.is_safe", bool(is_safe))
                    s_px.set_attribute("guardrail.layer", str(layer))
                    s_px.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps({
                        "layer": layer,
                        "is_safe": is_safe,
                        "details": dt_list
                    }, ensure_ascii=False, indent=2))
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

        # Đóng Phoenix Root
        if self.px_root_span:
            try:
                if PHOENIX_AVAILABLE:
                    if is_blocked:
                        status_str = "BLOCKED"
                    else:
                        status_str = "COMPLETED"

                    if citations:
                        cit_list = citations
                    else:
                        cit_list = []

                    if sources:
                        src_list = sources
                    else:
                        src_list = []

                    if answer:
                        ans_summary = answer[:200]
                    else:
                        ans_summary = ""

                    root_out = {
                        "status": status_str,
                        "total_latency_ms": total_latency_ms,
                        "selected_sources_count": len(src_list),
                        "top_sources": src_list[:5],
                        "legal_citations_count": len(cit_list),
                        "legal_citations": cit_list,
                        "answer_preview": ans_summary
                    }
                    self.px_root_span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    self.px_root_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(root_out, ensure_ascii=False, indent=2))
                    self.px_root_span.set_attribute("rag.citations", json.dumps(cit_list, ensure_ascii=False))
                    self.px_root_span.set_attribute("rag.sources", json.dumps(src_list, ensure_ascii=False))
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
    Điều phối trực tiếp Arize Phoenix và Local Tracing.
    """

    def __init__(self, config=None):
        from ..config import Config
        self.config = config or Config()

        self.enabled = getattr(self.config, "ENABLE_TRACING", True)
        self.enable_langfuse = False
        self.enable_phoenix = getattr(self.config, "ENABLE_PHOENIX", True)

        # Thư mục lưu vết (đặt trực tiếp trong phân hệ tracing)
        if hasattr(self.config, "TRACE_DIR"):
            self.trace_dir = self.config.TRACE_DIR
        else:
            self.trace_dir = Path(__file__).resolve().parent / "traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_log_file = self.trace_dir / "rag_traces.jsonl"

        # Khởi tạo trạng thái Phoenix & Local Tracing
        self.lf_client = None
        self.is_langfuse_active = False

        self.px_tracer = None
        self.is_phoenix_active = False

        if self.enabled:
            self._init_phoenix()

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
                global _PHOENIX_REGISTERED, _GLOBAL_PX_TRACER
                if not _PHOENIX_REGISTERED or _GLOBAL_PX_TRACER is None:
                    provider = phoenix_register(
                        project_name=project_name,
                        endpoint=endpoint,
                        batch=False,
                        protocol="http/protobuf",
                        verbose=False,
                        set_global_tracer_provider=False,
                        auto_instrument=False
                    )
                    _PHOENIX_REGISTERED = True
                    _GLOBAL_PX_TRACER = provider.get_tracer("agent_banking_tracer")
                self.px_tracer = _GLOBAL_PX_TRACER
                self.is_phoenix_active = True
                print(f"🔥 [Phoenix] Đã kết nối Arize Phoenix Observability ({endpoint})")
            except Exception as e:
                logger.warning(f"Không thể kết nối Phoenix OTel: {e}")
                self.is_phoenix_active = False
        else:
            # Phoenix server chưa khởi động, tắt tạm thời để tránh log lỗi kết nối
            self.is_phoenix_active = False
            logger.debug(f"[Phoenix] Server chưa bật tại {endpoint}. Khởi chạy bằng 'python scripts/run_phoenix.py'.")

    def trace_chunking(self, file_path, num_chunks, duration_ms, details=None, chunks=None, sample_chunks=None):
        """
        Ghi vết phân đoạn tài liệu DLA Layout-Aware độc lập (pha Ingestion).
        Bao gồm toàn bộ tất cả chunks trích xuất được và nội dung đầy đủ.
        Hiển thị ngay trên Phoenix dashboard dưới dạng ROOT SPAN.
        """
        fname = Path(file_path).name
        if details:
            dt_dict = details
        else:
            dt_dict = {}

        if chunks is not None:
            raw_chunks = chunks
        elif sample_chunks is not None:
            raw_chunks = sample_chunks
        else:
            raw_chunks = []

        extracted_chunks_list = []
        total_characters = 0
        total_tokens = 0

        for i, c in enumerate(raw_chunks):
            if isinstance(c, dict):
                c_id = c.get("chunk_id", f"chunk_{i + 1}")
                c_bc = c.get("breadcrumb", "")
                c_title = c.get("section_title", "")
                c_tok = c.get("token_count", 0)
                c_raw = c.get("content", "") or c.get("original_text", "") or c.get("text", "")
            else:
                c_id = getattr(c, "chunk_id", f"chunk_{i + 1}")
                c_bc = getattr(c, "breadcrumb", "")
                c_title = getattr(c, "section_title", "")
                c_tok = getattr(c, "token_count", 0)
                c_raw = getattr(c, "content", "") or getattr(c, "original_text", "") or getattr(c, "text", "")

            char_cnt = len(c_raw)
            total_characters += char_cnt
            if isinstance(c_tok, (int, float)):
                total_tokens += int(c_tok)

            extracted_chunks_list.append({
                "chunk_index": i + 1,
                "chunk_id": str(c_id),
                "breadcrumb": c_bc,
                "section_title": c_title,
                "token_count": c_tok,
                "char_count": char_cnt,
                "content": c_raw,
                "full_text_content": c_raw
            })

        if num_chunks:
            final_count = num_chunks
        else:
            final_count = len(extracted_chunks_list)

        if extracted_chunks_list:
            avg_tok = round(total_tokens / len(extracted_chunks_list), 1)
            avg_ch = round(total_characters / len(extracted_chunks_list), 1)
        else:
            avg_tok = 0
            avg_ch = 0

        record = {
            "query": f"Ingestion Chunking: {fname}",
            "file_path": str(file_path),
            "num_chunks": final_count,
            "total_tokens": total_tokens,
            "total_latency_ms": duration_ms,
            "timestamp": time.time(),
            "spans": [{
                "name": "DLA Layout-Aware Chunking",
                "duration_ms": duration_ms,
                "file_path": str(file_path),
                "num_chunks": final_count,
                "details": dt_dict
            }]
        }
        self.save_local_log(record)

        if self.is_phoenix_active and self.px_tracer:
            try:
                s_px = self.px_tracer.start_span("chunking_dla_layout_aware")
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)

                    inp_payload = {
                        "action": "document_layout_aware_chunking",
                        "file_path": str(file_path),
                        "file_name": fname,
                        "chunking_strategy": "DLA Layout-Aware Hierarchy Chunking (Breadcrumb preserved)",
                        "hierarchy_rules": [
                            "Preserve document heading hierarchy (I. -> 1. -> a) -> Điều)",
                            "Attach contextual breadcrumb navigation prefix to each chunk",
                            "Preserve table structures and legal clauses"
                        ],
                        "options": dt_dict
                    }

                    out_payload = {
                        "status": "SUCCESS",
                        "total_chunks_extracted": final_count,
                        "chunking_duration_ms": duration_ms,
                        "summary_statistics": {
                            "total_chunks": final_count,
                            "total_characters": total_characters,
                            "total_tokens": total_tokens,
                            "average_tokens_per_chunk": avg_tok,
                            "average_characters_per_chunk": avg_ch
                        },
                        "extracted_chunks": extracted_chunks_list
                    }

                    s_px.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(inp_payload, ensure_ascii=False, indent=2))
                    s_px.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(out_payload, ensure_ascii=False, indent=2))
                    s_px.set_attribute("chunking.file_path", str(file_path))
                    s_px.set_attribute("chunking.num_chunks", final_count)
                    s_px.set_attribute("chunking.duration_ms", duration_ms)
                    s_px.set_attribute("chunking.total_characters", total_characters)
                    s_px.set_attribute("chunking.total_tokens", total_tokens)
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix trace_chunking error: {e}")

    def trace_embedding(
        self,
        num_texts,
        dimension,
        duration_ms,
        model_name=None,
        cache_hits=0,
        chunks=None,
        vectors=None
    ):
        """
        Ghi vết nhúng vector độc lập (pha Ingestion indexing).
        Bao gồm thông tin mô hình, tiền tố instruction, vector preview và L2 norm.
        Hiển thị ngay trên Phoenix dashboard dưới dạng ROOT SPAN.
        """
        if model_name:
            chosen_model = model_name
        else:
            chosen_model = "multilingual-e5-large"

        input_chunks_summary = []
        embedded_chunks_details = []

        if chunks:
            for i, c in enumerate(chunks):
                if isinstance(c, dict):
                    c_id = c.get("chunk_id", f"chunk_{i + 1}")
                    c_bc = c.get("breadcrumb", "")
                    c_raw = c.get("content", "") or c.get("original_text", "") or c.get("text", "")
                    c_vec = c.get("vector")
                else:
                    c_id = getattr(c, "chunk_id", f"chunk_{i + 1}")
                    c_bc = getattr(c, "breadcrumb", "")
                    c_raw = getattr(c, "content", "") or getattr(c, "original_text", "") or getattr(c, "text", "")
                    c_vec = getattr(c, "vector", None)

                if c_bc:
                    formatted_passage = f"passage: [Ngữ cảnh: {c_bc}]\n{c_raw}"
                else:
                    formatted_passage = f"passage: {c_raw}"

                input_chunks_summary.append({
                    "chunk_index": i + 1,
                    "chunk_id": str(c_id),
                    "breadcrumb": c_bc,
                    "formatted_input_for_model": formatted_passage
                })

                if c_vec is not None and isinstance(c_vec, (list, tuple)):
                    v_list = list(c_vec)
                    v_first8 = [round(float(x), 5) for x in v_list[:8]]
                    v_first10 = [round(float(x), 6) for x in v_list[:10]]
                    v_norm = round(math.sqrt(sum(float(x) * float(x) for x in v_list)), 4)
                    v_prev = f"[{', '.join(str(x) for x in v_first8)}, ... ({len(v_list)} dimensions)]"
                    v_dim = len(v_list)
                else:
                    v_first8 = []
                    v_first10 = []
                    v_norm = 1.0
                    v_prev = f"[{dimension}-dim vector generated]"
                    v_dim = dimension

                embedded_chunks_details.append({
                    "chunk_index": i + 1,
                    "chunk_id": str(c_id),
                    "breadcrumb": c_bc,
                    "vector_dimension": v_dim,
                    "l2_norm": v_norm,
                    "vector_preview": v_prev,
                    "first_10_floats": v_first10,
                    "vector_sample_first_10_floats": v_first10,
                    "content": c_raw,
                    "full_text_content": c_raw,
                    "raw_content": c_raw
                })

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
                "model": chosen_model,
                "cache_hits": cache_hits
            }]
        }
        self.save_local_log(record)

        if self.is_phoenix_active and self.px_tracer:
            try:
                s_px = self.px_tracer.start_span("dense_embedding")
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value)
                    s_px.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, chosen_model)

                    inp_payload = {
                        "action": "ingestion_batch_embedding",
                        "num_chunks": num_texts,
                        "model": chosen_model,
                        "instruction_prefix": "passage: ",
                        "target_dimension": dimension,
                        "chunks_to_embed": input_chunks_summary
                    }
                    out_payload = {
                        "status": "BATCH_EMBEDDED_SUCCESS",
                        "total_vectors_generated": num_texts,
                        "dimension": dimension,
                        "sqlite_cache_hits": cache_hits,
                        "duration_ms": duration_ms,
                        "embedded_chunks_details": embedded_chunks_details
                    }

                    s_px.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(inp_payload, ensure_ascii=False, indent=2))
                    s_px.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(out_payload, ensure_ascii=False, indent=2))
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
