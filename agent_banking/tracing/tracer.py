"""
Tracing & Observability Module for AI Banking Assistant
=======================================================
Integrated AI Observability platform:
1. Arize Phoenix:
   - AI Observability platform for RAG & AI Agents.
   - Standardized on OpenTelemetry & OpenInference Semantic Conventions.
   - Visualizes RAG execution graphs (Retriever, Reranker, Chain, LLM Generation).
2. Local Tracing:
   - Stores trace logs in JSONL format (rag_traces.jsonl).

DESIGN INVARIANTS:
1. 0% REGEX LAW: No regex methods.
2. ZERO TERNARY: Explicit if/else statements.
3. NO __INIT__.PY: Direct module imports.
"""

import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import time
from urllib.parse import urlparse

from agent_banking.tracing.config.settings import TracingConfig, load_tracing_prompts


# Ensure safe UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_PHOENIX_CFG = dict(TracingConfig.PHOENIX_SETTINGS or {})
_SPAN_CFG = dict(TracingConfig.SPAN_SETTINGS or {})
_DEFAULTS_CFG = dict(TracingConfig.DEFAULT_TRACE_SETTINGS or {})
_FMT_CFG = dict(TracingConfig.FORMATTING_SETTINGS or {})
_RAG_TRACES_FILENAME = str(TracingConfig.RAG_TRACES_FILENAME or "")
_LOGGER_NAME = str(TracingConfig.LOGGER_NAME or "")

_prompts = load_tracing_prompts()
_SPAN_LABELS = dict(getattr(_prompts, "SPAN_LABELS", {}) or {})
_PHOENIX_MSGS = dict(getattr(_prompts, "PHOENIX_MESSAGES", {}) or {})
_SUMMARY_TPLS = dict(getattr(_prompts, "TERMINAL_SUMMARY_TEMPLATES", {}) or {})

logger = logging.getLogger(_LOGGER_NAME)

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


def _lbl(key):
    return str(_SPAN_LABELS.get(key) or "")


def _span_name(key):
    return str(_SPAN_CFG.get(key) or "")


def _fmt_str(key):
    return str(_FMT_CFG.get(key) or "")


def _px_msg(key, **kwargs):
    tpl = str(_PHOENIX_MSGS.get(key) or "")
    if not tpl:
        return ""
    return tpl.format(**kwargs)


def _check_endpoint_reachable(endpoint_url, timeout=None):
    """
    Quickly check whether the collector endpoint (e.g. Phoenix) is reachable
    to avoid blocking the query pipeline when the server is offline.
    """
    if timeout is None:
        timeout = float(_PHOENIX_CFG.get("default_reachability_timeout_sec") or 0.0)
    try:
        parsed = urlparse(endpoint_url)
        if parsed.hostname:
            host = parsed.hostname
        else:
            host = str(_PHOENIX_CFG.get("default_fallback_host") or "")

        if parsed.port is not None:
            port = parsed.port
        elif parsed.scheme == "https":
            port = int(_PHOENIX_CFG.get("default_https_port") or 0)
        else:
            port = int(_PHOENIX_CFG.get("default_http_port") or 0)

        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _to_float_score(value, default=0.0):
    """Safely convert an optional numeric score to float without failing on None."""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pick_float_score(doc, keys, default=0.0):
    """Return the first non-None score across candidate keys in a record dict."""
    for key in keys:
        val = doc.get(key)
        if val is not None:
            return _to_float_score(val, default=default)
    return float(default)


def _extract_chunk_trace_fields(raw_item, index=0):
    """Extract normalized 13-field universal chunk metadata for tracing spans."""
    if isinstance(raw_item, dict):
        if "chunk" in raw_item and isinstance(raw_item.get("chunk"), dict):
            c_obj = raw_item["chunk"]
        else:
            c_obj = raw_item
        c_id = c_obj.get("chunk_id")
        if not c_id:
            c_id = raw_item.get("chunk_id")
        if not c_id:
            c_id = f"chunk_{index + 1}"
        parent_id = str(c_obj.get("parent_id") or "")
        c_source = str(c_obj.get("source") or "")
        c_type = str(c_obj.get("chunk_type") or "text")
        c_bc = str(c_obj.get("breadcrumb") or "")
        c_title = str(c_obj.get("section_title") or "")
        c_raw = ""
        for fk in ["content", "parent_content", "original_text", "text"]:
            val = c_obj.get(fk)
            if val:
                c_raw = str(val)
                break
        bmeta = c_obj.get("business_metadata")
        if not isinstance(bmeta, dict):
            bmeta = {}
        ret_meta = bmeta.get("retrieval")
        if not isinstance(ret_meta, dict):
            ret_meta = {}
        qual_meta = bmeta.get("quality")
        if not isinstance(qual_meta, dict):
            qual_meta = {}
        c_tok = ret_meta.get("token_count")
        if c_tok is None:
            c_tok = c_obj.get("token_count", 0)
        table_data = c_obj.get("table_data")
        images = c_obj.get("images")
        c_vec = c_obj.get("vector")
        if c_vec is None:
            c_vec = raw_item.get("vector")
    else:
        c_id = getattr(raw_item, "chunk_id", f"chunk_{index + 1}")
        parent_id = str(getattr(raw_item, "parent_id", "") or "")
        c_source = str(getattr(raw_item, "source", "") or "")
        c_type = str(getattr(raw_item, "chunk_type", "text") or "text")
        c_bc = str(getattr(raw_item, "breadcrumb", "") or "")
        c_title = str(getattr(raw_item, "section_title", "") or "")
        c_raw = ""
        for fk in ["content", "parent_content", "original_text", "text"]:
            val = getattr(raw_item, fk, None)
            if val:
                c_raw = str(val)
                break
        bmeta = getattr(raw_item, "business_metadata", {})
        if not isinstance(bmeta, dict):
            bmeta = {}
        ret_meta = bmeta.get("retrieval")
        if not isinstance(ret_meta, dict):
            ret_meta = {}
        qual_meta = bmeta.get("quality")
        if not isinstance(qual_meta, dict):
            qual_meta = {}
        c_tok = ret_meta.get("token_count")
        if c_tok is None:
            c_tok = getattr(raw_item, "token_count", 0)
        table_data = getattr(raw_item, "table_data", None)
        images = getattr(raw_item, "images", None)
        c_vec = getattr(raw_item, "vector", None)

    passage_header = str(_FMT_CFG.get("passage_header") or "")
    ctx_label = _lbl("passage_context_label")
    ctx_str = str(ret_meta.get("context") or "").strip()
    sit_str = str(ret_meta.get("situated_context") or "").strip()
    if ctx_str or sit_str:
        p_parts = [passage_header]
        if ctx_str:
            p_parts.append("[" + ctx_str + "]")
        if sit_str:
            p_parts.append(ctx_label + sit_str)
        p_parts.append(c_raw)
        formatted_passage = "\n".join(p_parts)
    elif c_bc:
        formatted_passage = f"{passage_header} [{ctx_label}{c_bc}]\n{c_raw}"
    else:
        formatted_passage = f"{passage_header} {c_raw}"

    has_table = isinstance(table_data, dict) and bool(table_data)
    if isinstance(images, list):
        image_count = len(images)
    else:
        image_count = 0

    return {
        "chunk_id": str(c_id),
        "parent_id": parent_id,
        "source": c_source,
        "chunk_type": c_type,
        "breadcrumb": c_bc,
        "section_title": c_title,
        "content": c_raw,
        "token_count": c_tok,
        "schema_version": str(bmeta.get("schema_version") or ""),
        "domain": str(bmeta.get("domain") or ""),
        "index_ready": qual_meta.get("index_ready"),
        "has_table": has_table,
        "image_count": image_count,
        "passage": formatted_passage,
        "vector": c_vec,
    }


def _summarize_vector(vec_seq, fallback_dim):
    head_floats = int(_FMT_CFG.get("vector_preview_head_floats") or 0)
    sample_floats = int(_FMT_CFG.get("vector_sample_floats") or 0)
    preview_digits = int(_FMT_CFG.get("vector_preview_round_digits") or 0)
    sample_digits = int(_FMT_CFG.get("vector_sample_round_digits") or 0)
    norm_digits = int(_FMT_CFG.get("norm_round_digits") or 0)

    if vec_seq is not None and isinstance(vec_seq, (list, tuple)):
        v_list = list(vec_seq)
        first_head = [round(float(x), preview_digits) for x in v_list[:head_floats]]
        first_sample = [round(float(x), sample_digits) for x in v_list[:sample_floats]]
        norm_val = round(math.sqrt(sum(float(x) * float(x) for x in v_list)), norm_digits)
        preview_str = f"[{', '.join(str(x) for x in first_head)}, ... ({len(v_list)} dimensions)]"
        return first_sample, norm_val, preview_str, len(v_list)
    return [], 1.0, f"[{fallback_dim}-dim vector generated]", fallback_dim


def _build_chunk_summary_list(raw_chunks):
    avg_digits = int(_FMT_CFG.get("avg_round_digits") or 0)
    extracted_chunks_list = []
    total_characters = 0
    total_tokens = 0
    chunk_type_counts = {}

    for i, c in enumerate(raw_chunks):
        c_meta = _extract_chunk_trace_fields(c, index=i)
        c_raw = c_meta["content"]
        c_tok = c_meta["token_count"]
        c_type = c_meta["chunk_type"]
        chunk_type_counts[c_type] = chunk_type_counts.get(c_type, 0) + 1

        char_cnt = len(c_raw)
        total_characters += char_cnt
        if isinstance(c_tok, (int, float)):
            total_tokens += int(c_tok)

        extracted_chunks_list.append({
            "chunk_index": i + 1,
            "chunk_id": c_meta["chunk_id"],
            "parent_id": c_meta["parent_id"],
            "chunk_type": c_type,
            "domain": c_meta["domain"],
            "schema_version": c_meta["schema_version"],
            "index_ready": c_meta["index_ready"],
            "has_table": c_meta["has_table"],
            "image_count": c_meta["image_count"],
            "breadcrumb": c_meta["breadcrumb"],
            "section_title": c_meta["section_title"],
            "token_count": c_tok,
            "char_count": char_cnt,
            "content": c_raw,
            "full_text_content": c_raw
        })

    if extracted_chunks_list:
        avg_tok = round(total_tokens / len(extracted_chunks_list), avg_digits)
        avg_ch = round(total_characters / len(extracted_chunks_list), avg_digits)
    else:
        avg_tok = 0
        avg_ch = 0

    return extracted_chunks_list, total_characters, total_tokens, chunk_type_counts, avg_tok, avg_ch


class ActiveTrace:
    """
    Manages the lifecycle and spans of a single query trace.
    Synchronizes spans to Arize Phoenix and persists local JSONL trace logs.
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
                root_name = _span_name("root_span_name")
                self.px_root_span = self.tracer.px_tracer.start_span(root_name)
                if PHOENIX_AVAILABLE:
                    self.px_root_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                    root_inp = {"user_query": query}
                    self.px_root_span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    self.px_root_span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(root_inp, ensure_ascii=False, indent=2))
                self.px_context = otel_trace.set_span_in_context(self.px_root_span)
            except Exception as e:
                logger.debug(f"Phoenix root span error: {e}")

    def record_chunking_info(self, total_chunks, max_size=None, strategy=None, duration_ms=None):
        """Step 1: Record DLA Layout-Aware Chunking summary into Phoenix."""
        if max_size is None:
            max_size = int(_DEFAULTS_CFG.get("chunking_max_size") or 0)
        if strategy is None:
            strategy = _lbl("chunking_strategy")
        if duration_ms is None:
            duration_ms = float(_DEFAULTS_CFG.get("chunking_info_duration_ms") or 0.0)

        span_data = {
            "name": _lbl("chunking_info_display"),
            "duration_ms": duration_ms,
            "total_chunks_stored": total_chunks,
            "strategy": strategy,
            "max_size": max_size
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer and self.px_context:
            try:
                span_name = _span_name("chunking_info_span_name")
                s_px = self.tracer.px_tracer.start_span(span_name, context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                    s_px.set_attribute("chunking.strategy", str(strategy))
                    s_px.set_attribute("chunking.total_indexed_chunks", int(total_chunks))
                    s_px.set_attribute("chunking.max_characters_per_chunk", int(max_size))
                    s_px.set_attribute("chunking.keep_tables_atomic", True)
                    s_px.set_attribute("chunking.hierarchy_preservation", _lbl("chunking_hierarchy_preservation"))

                    inp_dict = {
                        "chunking_algorithm": strategy,
                        "chunk_size_limit": max_size,
                        "table_policy": _lbl("chunking_table_policy")
                    }
                    out_dict = {
                        "status": _lbl("status_ready"),
                        "total_indexed_chunks": total_chunks,
                        "chunking_description": _lbl("chunking_ready_description")
                    }
                    s_px.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(inp_dict, ensure_ascii=False, indent=2))
                    s_px.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(out_dict, ensure_ascii=False, indent=2))
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix chunking_info error: {e}")

    def record_chunking(self, file_path, num_chunks, duration_ms, details=None, chunks=None, sample_chunks=None):
        """Step 0: Record detailed DLA Layout-Aware Chunking span across extracted chunks."""
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

        extracted_chunks_list, total_characters, total_tokens, chunk_type_counts, avg_tok, avg_ch = _build_chunk_summary_list(raw_chunks)

        if num_chunks:
            final_count = num_chunks
        else:
            final_count = len(extracted_chunks_list)

        span_data = {
            "name": _lbl("chunking_display"),
            "duration_ms": duration_ms,
            "file_path": str(file_path),
            "num_chunks": final_count,
            "total_tokens": total_tokens,
            "chunk_type_counts": chunk_type_counts,
            "details": dt_dict
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                span_name = _span_name("chunking_span_name")
                s_px = self.tracer.px_tracer.start_span(span_name, context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)

                    inp_payload = {
                        "action": "document_layout_aware_chunking",
                        "file_path": str(file_path),
                        "file_name": Path(file_path).name,
                        "chunking_strategy": _lbl("chunking_strategy_breadcrumb"),
                        "hierarchy_rules": list(_SPAN_LABELS.get("chunking_hierarchy_rules") or []),
                        "options": dt_dict
                    }

                    out_payload = {
                        "status": _lbl("status_success"),
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
        """Record dense vector embedding span with vector preview and norm."""
        default_embed_model = str(_DEFAULTS_CFG.get("default_embedding_model") or "")
        if model_name:
            chosen_model = model_name
        else:
            chosen_model = getattr(self.tracer.config, "EMBEDDING_MODEL_1024", getattr(self.tracer.config, "EMBEDDING_MODEL", default_embed_model))

        span_name = _span_name("embedding_span_name")
        query_prefix = _fmt_str("query_prefix")
        passage_prefix = _fmt_str("passage_prefix")

        if is_query:
            disp_name = _lbl("query_embedding_display")
            if query_text:
                q_content = query_text
            else:
                q_content = self.query

            formatted_query_input = query_prefix + q_content

            inp_obj = {
                "action": "query_dense_embedding",
                "input_query": q_content,
                "model": chosen_model,
                "instruction_prefix": query_prefix,
                "formatted_embedding_input": formatted_query_input,
                "target_dimension": dimension
            }

            if query_vector is not None and isinstance(query_vector, (list, tuple)):
                first_sample, norm_val, vector_preview_str, _ = _summarize_vector(query_vector, dimension)
            else:
                first_sample = []
                norm_val = 1.0
                vector_preview_str = f"[{dimension}-dim normalized vector, dimension={dimension}]"

            out_obj = {
                "status": _lbl("status_query_embedded"),
                "output_dimension": dimension,
                "l2_normalized": True,
                "vector_norm": norm_val,
                "cache_hits": cache_hits,
                "vector_preview": vector_preview_str,
                "first_10_floats": first_sample,
                "vector_sample_first_10_floats": first_sample,
                "duration_ms": duration_ms
            }
        else:
            disp_name = _lbl("batch_embedding_display")

            input_chunks_summary = []
            embedded_chunks_details = []

            if chunks:
                for i, c in enumerate(chunks):
                    c_meta = _extract_chunk_trace_fields(c, index=i)
                    c_id = c_meta["chunk_id"]
                    c_bc = c_meta["breadcrumb"]
                    c_raw = c_meta["content"]
                    c_vec = c_meta["vector"]
                    formatted_passage = c_meta["passage"]

                    input_chunks_summary.append({
                        "chunk_index": i + 1,
                        "chunk_id": str(c_id),
                        "chunk_type": c_meta["chunk_type"],
                        "breadcrumb": c_bc,
                        "formatted_input_for_model": formatted_passage
                    })

                    v_first10, v_norm, v_prev, v_dim = _summarize_vector(c_vec, dimension)

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
                "instruction_prefix": passage_prefix,
                "target_dimension": dimension,
                "chunks_to_embed": input_chunks_summary
            }
            out_obj = {
                "status": _lbl("status_batch_embedded"),
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
        """Step 1: Record Query Parser trace span (acronyms, keywords, legal references)."""
        norm_query = getattr(parsed, "normalized_query", getattr(parsed, "raw_query", str(parsed)))
        keywords = getattr(parsed, "keywords", [])
        acronyms = getattr(parsed, "detected_acronyms", getattr(parsed, "expanded_acronyms", []))
        legal_refs = getattr(parsed, "legal_references", [])

        span_data = {
            "name": _lbl("query_parser_display"),
            "duration_ms": duration_ms,
            "normalized_query": norm_query,
            "keywords": keywords,
            "acronyms": acronyms,
            "legal_references": legal_refs
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                span_name = _span_name("query_parser_span_name")
                s_px = self.tracer.px_tracer.start_span(span_name, context=self.px_context)
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
        """Step 2: Record Hybrid Retrieval span across all candidates."""
        score_digits = int(_FMT_CFG.get("score_round_digits") or 0)
        bm25_digits = int(_FMT_CFG.get("bm25_round_digits") or 0)

        doc_count = len(candidates)
        candidate_details = []

        for i, doc in enumerate(candidates):
            if isinstance(doc, dict):
                c_item = doc.get("chunk", doc)
                c_score = _pick_float_score(doc, ("rrf_score", "score"))
                dense_sc = _pick_float_score(doc, ("dense_score",))
                bm25_sc = _pick_float_score(doc, ("bm25_score",))
            elif isinstance(doc, tuple) and len(doc) >= 2:
                c_item = doc[0]
                c_score = _to_float_score(doc[1])
                dense_sc = 0.0
                bm25_sc = 0.0
            else:
                c_item = doc
                c_score = 0.0
                dense_sc = 0.0
                bm25_sc = 0.0

            c_meta = _extract_chunk_trace_fields(c_item, index=i)
            doc_text = c_meta["content"]
            title = c_meta["section_title"] or c_meta["breadcrumb"]
            cid = c_meta["chunk_id"]
            c_source = c_meta["source"]

            c_info = {
                "rank": i + 1,
                "chunk_id": cid,
                "parent_id": c_meta["parent_id"],
                "chunk_type": c_meta["chunk_type"],
                "domain": c_meta["domain"],
                "has_table": c_meta["has_table"],
                "image_count": c_meta["image_count"],
                "rrf_score": round(float(c_score), score_digits),
                "dense_score": round(float(dense_sc), score_digits),
                "bm25_score": round(float(bm25_sc), bm25_digits),
                "source_file": c_source,
                "breadcrumb": title,
                "content": doc_text,
                "char_count": len(doc_text)
            }
            candidate_details.append(c_info)

        span_data = {
            "name": _lbl("retrieval_display"),
            "duration_ms": duration_ms,
            "retrieved_candidates": doc_count,
            "candidates": candidate_details
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                span_name = _span_name("retrieval_span_name")
                s_px = self.tracer.px_tracer.start_span(span_name, context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
                    s_px.set_attribute("retrieval.top_k", int(doc_count))
                    s_px.set_attribute("top_k", int(doc_count))

                    if params:
                        ret_params = params
                    else:
                        ret_params = {
                            "dense_k": int(_DEFAULTS_CFG.get("retrieval_dense_k") or 0),
                            "bm25_k": int(_DEFAULTS_CFG.get("retrieval_bm25_k") or 0),
                            "rrf_k": int(_DEFAULTS_CFG.get("retrieval_rrf_k") or 0),
                            "returned_candidates": doc_count
                        }

                    inp_obj = {
                        "search_query": query,
                        "retrieval_strategy": _lbl("retrieval_strategy"),
                        "fusion_algorithm": _lbl("fusion_algorithm"),
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
        """Record Reranking step across candidates."""
        score_digits = int(_FMT_CFG.get("score_round_digits") or 0)
        default_reranker_model = str(_DEFAULTS_CFG.get("default_reranker_model") or "")

        rerank_summary = []
        for i, doc in enumerate(reranked):
            if isinstance(doc, dict):
                chunk = doc.get("chunk", doc)
                score = _pick_float_score(doc, ("rerank_score", "relevance_score", "score"))
            elif isinstance(doc, tuple) and len(doc) >= 2:
                chunk = doc[0]
                score = _to_float_score(doc[1])
            else:
                chunk = doc
                score = 0.0

            c_meta = _extract_chunk_trace_fields(chunk, index=i)
            doc_text = c_meta["content"]
            title = c_meta["breadcrumb"] or c_meta["section_title"]
            article = c_meta["section_title"]
            cid = c_meta["chunk_id"]

            rerank_summary.append({
                "rank": i + 1,
                "chunk_id": cid,
                "chunk_type": c_meta["chunk_type"],
                "domain": c_meta["domain"],
                "has_table": c_meta["has_table"],
                "image_count": c_meta["image_count"],
                "rerank_score": round(float(score), score_digits),
                "article": article,
                "title": title,
                "content": doc_text
            })

        span_data = {
            "name": _lbl("rerank_display"),
            "duration_ms": duration_ms,
            "input_count": len(candidates),
            "top_reranked_count": len(reranked),
            "top_reranked": rerank_summary
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer:
            try:
                span_name = _span_name("reranker_span_name")
                s_px = self.tracer.px_tracer.start_span(span_name, context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RERANKER.value)
                    s_px.set_attribute(RerankerAttributes.RERANKER_QUERY, query)
                    reranker_model = getattr(self.tracer.config, "RERANKER_MODEL", default_reranker_model)
                    s_px.set_attribute(RerankerAttributes.RERANKER_MODEL_NAME, reranker_model)
                    s_px.set_attribute(RerankerAttributes.RERANKER_TOP_K, len(reranked))
                    s_px.set_attribute("reranker.top_k", len(reranked))
                    s_px.set_attribute("top_k", len(reranked))

                    input_candidates_summary = []
                    for i, doc in enumerate(candidates):
                        if isinstance(doc, dict):
                            c_item = doc.get("chunk", doc)
                            c_score = _pick_float_score(doc, ("rrf_score", "score"))
                        elif isinstance(doc, tuple) and len(doc) >= 2:
                            c_item = doc[0]
                            c_score = _to_float_score(doc[1])
                        else:
                            c_item = doc
                            c_score = 0.0

                        c_meta = _extract_chunk_trace_fields(c_item, index=i)
                        c_id = c_meta["chunk_id"]
                        c_title = c_meta["breadcrumb"] or c_meta["section_title"]
                        c_text = c_meta["content"]
                        c_source = c_meta["source"]

                        input_candidates_summary.append({
                            "input_rank": i + 1,
                            "chunk_id": str(c_id),
                            "initial_score": round(float(c_score), score_digits),
                            "source": c_source,
                            "breadcrumb": c_title,
                            "content": c_text,
                            "char_count": len(c_text)
                        })

                        in_prefix = f"{RerankerAttributes.RERANKER_INPUT_DOCUMENTS}.{i}"
                        s_px.set_attribute(f"{in_prefix}.document.id", f"[Candidate-{i+1}/{len(candidates)}] {c_id}")
                        s_px.set_attribute(f"{in_prefix}.document.content", c_text)
                        s_px.set_attribute(f"{in_prefix}.document.score", float(c_score))

                    reranked_results = []
                    best_score = 0.0
                    for i, doc in enumerate(reranked):
                        if isinstance(doc, dict):
                            c_item = doc.get("chunk", doc)
                            c_score = _pick_float_score(doc, ("rerank_score", "relevance_score", "score"))
                        elif isinstance(doc, tuple) and len(doc) >= 2:
                            c_item = doc[0]
                            c_score = _to_float_score(doc[1])
                        else:
                            c_item = doc
                            c_score = 0.0

                        c_meta = _extract_chunk_trace_fields(c_item, index=i)
                        c_text = c_meta["content"]
                        c_id = c_meta["chunk_id"]
                        c_title = c_meta["breadcrumb"] or c_meta["section_title"]
                        c_source = c_meta["source"]

                        if i == 0:
                            best_score = round(float(c_score), score_digits)

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
                            "cross_encoder_score": round(float(c_score), score_digits),
                            "original_rank": orig_rank,
                            "rank_shift": shift_str,
                            "source": c_source,
                            "breadcrumb": c_title,
                            "content": c_text,
                            "char_count": len(c_text)
                        })

                        r_prefix = f"{RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS}.{i}"
                        s_px.set_attribute(f"{r_prefix}.document.id", f"[Top-{i+1}/{len(reranked)} Rerank Chunk] {c_id}")
                        s_px.set_attribute(f"{r_prefix}.document.score", float(c_score))
                        s_px.set_attribute(f"{r_prefix}.document.content", c_text)

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
                        "model": reranker_model,
                        "total_input_candidates": len(candidates),
                        "requested_top_k": len(reranked),
                        "candidates_before_rerank": input_candidates_summary
                    }
                    out_obj = {
                        "model": reranker_model,
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

    def record_guardrail_block(self, reason, details=None, duration_ms=0.0):
        """Record security block event from Banking Guardrails."""
        if details:
            dt_str = str(details)
        else:
            dt_str = ""
        span_data = {
            "name": _lbl("guardrail_block_display"),
            "duration_ms": duration_ms,
            "blocked": True,
            "reason": reason,
            "details": dt_str
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer and self.px_context:
            try:
                span_name = _span_name("guardrail_block_span_name")
                s_px = self.tracer.px_tracer.start_span(span_name, context=self.px_context)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.GUARDRAIL.value)
                    s_px.set_attribute("guardrail.blocked", True)
                    s_px.set_attribute("guardrail.reason", str(reason))
                    s_px.set_attribute(SpanAttributes.OUTPUT_VALUE, f"{_lbl('guardrail_block_output_prefix')}{reason}")
                s_px.end()
            except Exception as e:
                logger.debug(f"Phoenix guardrail block error: {e}")

    def record_guardrail_check(self, layer, is_safe, details=None, duration_ms=0.0):
        """Record guardrail content inspection event."""
        if details:
            dt_list = details
        else:
            dt_list = []

        span_data = {
            "name": f"{_lbl('guardrail_check_prefix')}{layer}",
            "duration_ms": duration_ms,
            "is_safe": is_safe,
            "details": dt_list
        }
        self.spans.append(span_data)

        if self.tracer.is_phoenix_active and self.tracer.px_tracer and self.px_context:
            try:
                span_prefix = _span_name("guardrail_check_span_prefix")
                span_name = f"{span_prefix}{str(layer).lower().replace(' ', '_')}"
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
        """Finalize trace, close Phoenix root span, and persist local JSONL log."""
        if self.finished:
            return
        self.finished = True
        latency_digits = int(_FMT_CFG.get("latency_round_digits") or 0)
        answer_preview_max = int(_FMT_CFG.get("answer_preview_max_chars") or 0)
        top_sources_limit = int(_FMT_CFG.get("top_sources_limit") or 0)

        total_latency_ms = round((time.time() - self.start_time) * 1000, latency_digits)

        if self.px_root_span:
            try:
                if PHOENIX_AVAILABLE:
                    if is_blocked:
                        status_str = _lbl("status_blocked")
                    else:
                        status_str = _lbl("status_completed")

                    if citations:
                        cit_list = citations
                    else:
                        cit_list = []

                    if sources:
                        src_list = sources
                    else:
                        src_list = []

                    if answer:
                        ans_summary = answer[:answer_preview_max]
                    else:
                        ans_summary = ""

                    root_out = {
                        "status": status_str,
                        "total_latency_ms": total_latency_ms,
                        "selected_sources_count": len(src_list),
                        "top_sources": src_list[:top_sources_limit],
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

        if answer:
            ans_rec = answer[:answer_preview_max]
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
        self.tracer.print_trace_summary(trace_record)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.finished:
            if exc_val is not None:
                err_msg = str(exc_val)
                self.finish(answer=f"{_lbl('pipeline_exception_prefix')}{err_msg}")
            else:
                self.finish(answer="")


class BankingTracer:
    """
    Centralized Tracing Manager for AI Banking Assistant.
    Coordinates Arize Phoenix OpenTelemetry and local JSONL trace logs.
    """

    def __init__(self, config=None):
        from agent_banking.core.settings import Config
        self.config = config or Config()

        self.enabled = getattr(self.config, "ENABLE_TRACING", TracingConfig.ENABLE_TRACING)
        self.enable_langfuse = False
        self.enable_phoenix = getattr(self.config, "ENABLE_PHOENIX", TracingConfig.ENABLE_PHOENIX)

        if hasattr(self.config, "TRACE_DIR"):
            self.trace_dir = Path(self.config.TRACE_DIR)
        else:
            self.trace_dir = Path(TracingConfig.TRACE_DIR)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_log_file = self.trace_dir / _RAG_TRACES_FILENAME

        self.lf_client = None
        self.is_langfuse_active = False

        self.px_tracer = None
        self.is_phoenix_active = False

        if self.enabled:
            self._init_phoenix()

    def _init_phoenix(self):
        """Initialize Arize Phoenix connection via OpenTelemetry TracerProvider."""
        global _PHOENIX_REGISTERED, _GLOBAL_PX_TRACER
        if not self.enable_phoenix or not PHOENIX_AVAILABLE:
            return

        endpoint = getattr(self.config, "PHOENIX_COLLECTOR_ENDPOINT", "") or os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "")
        project_name = getattr(self.config, "PHOENIX_PROJECT_NAME", "") or os.getenv("PHOENIX_PROJECT_NAME", "")

        if not endpoint:
            self.is_phoenix_active = False
            return

        check_timeout = float(_PHOENIX_CFG.get("endpoint_check_timeout_sec") or 0.0)
        if _check_endpoint_reachable(endpoint, timeout=check_timeout):
            try:
                if not _PHOENIX_REGISTERED or _GLOBAL_PX_TRACER is None:
                    provider = phoenix_register(
                        project_name=project_name,
                        endpoint=endpoint,
                        batch=bool(_PHOENIX_CFG.get("batch")),
                        protocol=str(_PHOENIX_CFG.get("protocol") or ""),
                        verbose=bool(_PHOENIX_CFG.get("verbose")),
                        set_global_tracer_provider=bool(_PHOENIX_CFG.get("set_global_tracer_provider")),
                        auto_instrument=bool(_PHOENIX_CFG.get("auto_instrument"))
                    )
                    _PHOENIX_REGISTERED = True
                    _GLOBAL_PX_TRACER = provider.get_tracer(str(_PHOENIX_CFG.get("tracer_name") or ""))
                self.px_tracer = _GLOBAL_PX_TRACER
                self.is_phoenix_active = True
                print(_px_msg("connected", endpoint=endpoint))
            except Exception as e:
                logger.warning(_px_msg("connect_failed", error=e))
                self.is_phoenix_active = False
        else:
            self.is_phoenix_active = False
            logger.debug(_px_msg("server_not_running", endpoint=endpoint))

    def trace_chunking(self, file_path, num_chunks, duration_ms, details=None, chunks=None, sample_chunks=None):
        """Record standalone DLA Layout-Aware chunking span during ingestion."""
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

        extracted_chunks_list, total_characters, total_tokens, chunk_type_counts, avg_tok, avg_ch = _build_chunk_summary_list(raw_chunks)

        if num_chunks:
            final_count = num_chunks
        else:
            final_count = len(extracted_chunks_list)

        record = {
            "query": f"{_lbl('ingestion_chunking_prefix')}{fname}",
            "file_path": str(file_path),
            "num_chunks": final_count,
            "total_tokens": total_tokens,
            "chunk_type_counts": chunk_type_counts,
            "total_latency_ms": duration_ms,
            "timestamp": time.time(),
            "spans": [{
                "name": _lbl("chunking_display"),
                "duration_ms": duration_ms,
                "file_path": str(file_path),
                "num_chunks": final_count,
                "chunk_type_counts": chunk_type_counts,
                "details": dt_dict
            }]
        }
        self.save_local_log(record)

        if self.is_phoenix_active and self.px_tracer:
            try:
                span_name = _span_name("chunking_span_name")
                s_px = self.px_tracer.start_span(span_name)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)

                    inp_payload = {
                        "action": "document_layout_aware_chunking",
                        "file_path": str(file_path),
                        "file_name": fname,
                        "chunking_strategy": _lbl("chunking_strategy_breadcrumb"),
                        "hierarchy_rules": list(_SPAN_LABELS.get("chunking_hierarchy_rules") or []),
                        "options": dt_dict
                    }

                    out_payload = {
                        "status": _lbl("status_success"),
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
        """Record standalone vector embedding span during ingestion indexing."""
        default_embed_model = str(_DEFAULTS_CFG.get("default_embedding_model") or "")
        passage_prefix = _fmt_str("passage_prefix")

        if model_name:
            chosen_model = model_name
        else:
            chosen_model = getattr(self.config, "EMBEDDING_MODEL_1024", getattr(self.config, "EMBEDDING_MODEL", default_embed_model))

        input_chunks_summary = []
        embedded_chunks_details = []

        if chunks:
            for i, c in enumerate(chunks):
                c_meta = _extract_chunk_trace_fields(c, index=i)
                c_id = c_meta["chunk_id"]
                c_bc = c_meta["breadcrumb"]
                c_raw = c_meta["content"]
                c_vec = c_meta["vector"]
                formatted_passage = c_meta["passage"]

                input_chunks_summary.append({
                    "chunk_index": i + 1,
                    "chunk_id": str(c_id),
                    "chunk_type": c_meta["chunk_type"],
                    "breadcrumb": c_bc,
                    "formatted_input_for_model": formatted_passage
                })

                v_first10, v_norm, v_prev, v_dim = _summarize_vector(c_vec, dimension)

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

        query_tpl = _lbl("ingestion_embedding_template")
        if query_tpl:
            query_label = query_tpl.format(num_texts=num_texts)
        else:
            query_label = str(num_texts)

        record = {
            "query": query_label,
            "num_texts": num_texts,
            "dimension": dimension,
            "total_latency_ms": duration_ms,
            "timestamp": time.time(),
            "spans": [{
                "name": _lbl("batch_embedding_display"),
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
                span_name = _span_name("embedding_span_name")
                s_px = self.px_tracer.start_span(span_name)
                if PHOENIX_AVAILABLE:
                    s_px.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value)
                    s_px.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, chosen_model)

                    inp_payload = {
                        "action": "ingestion_batch_embedding",
                        "num_chunks": num_texts,
                        "model": chosen_model,
                        "instruction_prefix": passage_prefix,
                        "target_dimension": dimension,
                        "chunks_to_embed": input_chunks_summary
                    }
                    out_payload = {
                        "status": _lbl("status_batch_embedded"),
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

    def trace_agent(self, entrypoint, session_id):
        from agent_banking.tracing.agent_trace import AgentRun

        trace_dir = None
        if self.enabled:
            trace_dir = self.trace_dir
        return AgentRun(trace_dir, entrypoint, session_id).capture()

    def trace_query(self, query):
        """Create a new trace session for a banking query."""
        return ActiveTrace(self, query)

    def save_local_log(self, record):
        """Persist trace record in JSON Lines format."""
        try:
            with open(self.trace_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(_px_msg("local_log_error", error=e))

    def print_trace_summary(self, record):
        """Render terminal trace summary box."""
        box_width = int(_FMT_CFG.get("terminal_box_width") or 0)
        query_preview_max = int(_FMT_CFG.get("query_preview_max_chars") or 0)
        citations_limit = int(_FMT_CFG.get("citations_preview_limit") or 0)

        total_ms = record.get("total_latency_ms", 0.0)
        query = record.get("query", "")
        spans = record.get("spans", [])

        sync_tags = []
        if self.is_phoenix_active:
            sync_tags.append(str(_SUMMARY_TPLS.get("phoenix_on_tag") or ""))

        if sync_tags:
            sync_str = f" [{', '.join(sync_tags)}]"
        else:
            sync_str = ""

        if len(query) > query_preview_max:
            q_disp = query[:query_preview_max] + "..."
        else:
            q_disp = query

        header_tpl = str(_SUMMARY_TPLS.get("header_template") or "")
        header = header_tpl.format(sync_str=sync_str, total_ms=total_ms)

        print("\n" + "┌" + "─" * box_width + "┐")
        print(f"│ {header:<{box_width - 2}} │")
        print("├" + "─" * box_width + "┤")

        query_line = str(_SUMMARY_TPLS.get("query_line_template") or "").format(q_disp=q_disp)
        print(query_line + " " * max(1, (box_width - 2) - len(query_line)) + "│")

        if record.get("is_blocked"):
            block_line = str(_SUMMARY_TPLS.get("block_line_template") or "").format(trigger=record.get("guardrail_trigger", ""))
            if len(block_line) < box_width:
                print(block_line + " " * (box_width - len(block_line)) + "│")
            else:
                print(block_line[:box_width - 1] + "│")

        stages_hdr = str(_SUMMARY_TPLS.get("stages_header") or "")
        print(stages_hdr + " " * max(0, (box_width - 1) - len(stages_hdr)) + "│")

        stage_tpl = str(_SUMMARY_TPLS.get("stage_line_template") or "")
        for idx, span in enumerate(spans):
            if idx == len(spans) - 1:
                branch = "└─"
            else:
                branch = "├─"
            name = span.get("name", "")
            duration = span.get("duration_ms", 0.0)
            line = stage_tpl.format(branch=branch, name=name, duration=duration)
            if len(line) < box_width:
                print(line + " " * (box_width - len(line)) + "│")
            else:
                print(line[:box_width - 1] + "│")

        citations = record.get("legal_citations", [])
        if citations:
            cit_items = []
            for c_item in citations[:citations_limit]:
                if isinstance(c_item, dict):
                    sec_val = c_item.get("section", "")
                    if not sec_val:
                        sec_val = c_item.get("source", str(c_item))
                    cit_items.append(str(sec_val))
                else:
                    cit_items.append(str(c_item))
            c_str = f"{_SUMMARY_TPLS.get('citations_prefix', '')}{', '.join(cit_items)}"
            c_line = str(_SUMMARY_TPLS.get("citations_line_template") or "").format(c_str=c_str)
            if len(c_line) < box_width:
                print(c_line + " " * (box_width - len(c_line)) + "│")
            else:
                print(c_line[:box_width - 1] + "│")

        print("└" + "─" * box_width + "┘\n")


# Initialize singleton tracer
tracer = BankingTracer()
