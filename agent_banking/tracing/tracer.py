# -*- coding: utf-8 -*-
"""
Unified LangSmith Tracing & Observability Module for AI Banking Assistant.
==========================================================================
Centralizes all execution tracing on LangSmith:
- LangSmith RunTree + tracing_context + LangChainTracer (LangGraph integration)
- OfflineLangSmithClient for local/test/evaluation runs without network calls
- Safe metadata sanitization and session fingerprinting
- Optional local JSONL run export (langsmith_runs.jsonl) for offline evaluation

DESIGN INVARIANTS:
1. 0% REGEX LAW: No regex imports or methods.
2. ZERO TERNARY: Explicit if/else statements only.
4. NO __INIT__.PY: Direct module imports.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
import json
import logging
import math
import os
from pathlib import Path
import sys
from threading import Lock
import time
from time import perf_counter, time_ns
from uuid import uuid4

from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client, RunTree, get_current_run_tree, tracing_context

from agent_banking.tracing.config.settings import TracingConfig, load_tracing_prompts


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_LS_CFG = dict(TracingConfig.LANGSMITH_SETTINGS or {})
_SPAN_CFG = dict(TracingConfig.SPAN_SETTINGS or {})
_DEFAULTS_CFG = dict(TracingConfig.DEFAULT_TRACE_SETTINGS or {})
_FMT_CFG = dict(TracingConfig.FORMATTING_SETTINGS or {})
_META_CFG = dict(TracingConfig.RUN_METADATA_SETTINGS or {})

_LANGSMITH_RUNS_FILENAME = str(TracingConfig.LANGSMITH_RUNS_FILENAME or "langsmith_runs.jsonl")
_LOGGER_NAME = str(TracingConfig.LOGGER_NAME or "agent_banking.tracing")

_prompts = load_tracing_prompts()
_SPAN_LABELS = dict(getattr(_prompts, "SPAN_LABELS", {}) or {})
_LS_MSGS = dict(getattr(_prompts, "LANGSMITH_MESSAGES", {}) or {})
_SUMMARY_TPLS = dict(getattr(_prompts, "TERMINAL_SUMMARY_TEMPLATES", {}) or {})

_CONTEXT_VAR_NAME = str(_META_CFG.get("context_var_name") or "banking_langsmith_run")
_SCHEMA_VERSION = int(_META_CFG.get("schema_version") or 2)
_FINGERPRINT_LENGTH = int(_META_CFG.get("fingerprint_length") or 24)
_DURATION_ROUND_DIGITS = int(_META_CFG.get("duration_round_digits") or 3)
_ROOT_SPAN_PREFIX = str(_META_CFG.get("root_span_prefix") or "agent.")
_DEFAULT_OUTCOME = str(_META_CFG.get("default_outcome") or "completed")
_SUSPENDED_STATUS = str(_META_CFG.get("suspended_status") or "suspended")
_OK_STATUS = str(_META_CFG.get("ok_status") or "ok")
_ERROR_STATUS = str(_META_CFG.get("error_status") or "error")
_BLOCKED_AUDIT_STATUSES = tuple(_META_CFG.get("blocked_audit_statuses") or ("BLOCKED", "ERROR", "FAIL", "NEEDS_INPUT"))
_REJECTED_HTTP_MIN = int(_META_CFG.get("rejected_http_status_min") or 400)
_REJECTED_HTTP_MAX = int(_META_CFG.get("rejected_http_status_max") or 500)
_REJECTED_OUTCOME = str(_META_CFG.get("rejected_outcome") or "rejected")
_SAVE_WARNING_MSG = str(_LS_MSGS.get("run_save_warning") or "LangSmith run %s could not be saved (%s)")

_REDACTED_KEYS = frozenset({"api_key", "authorization", "secret", "password", "access_token", "token"})
_HASHED_SESSION_KEYS = frozenset({"session_id", "thread_id"})

_current_run = ContextVar(_CONTEXT_VAR_NAME, default=None)
_write_lock = Lock()
logger = logging.getLogger(_LOGGER_NAME)


def _lbl(key):
    return str(_SPAN_LABELS.get(key) or "")


def _span_name(key):
    return str(_SPAN_CFG.get(key) or key)


def _fmt_str(key):
    return str(_FMT_CFG.get(key) or "")


def _ls_msg(key, **kwargs):
    tpl = str(_LS_MSGS.get(key) or "")
    if not tpl:
        return ""
    return tpl.format(**kwargs)


def fingerprint(value):
    """Hash sensitive identifiers deterministically before attaching to traces."""
    if value:
        return sha256(str(value).encode("utf-8", errors="replace")).hexdigest()[:_FINGERPRINT_LENGTH]
    return None


def _sanitize_value(value):
    """Strip secret keys, hash session identifiers, and keep JSON-safe primitive structures."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in _REDACTED_KEYS:
                continue
            if key_text in _HASHED_SESSION_KEYS and isinstance(item, str) and item:
                cleaned[key] = fingerprint(item)
                continue
            cleaned[key] = _sanitize_value(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _to_langsmith_usage(usage):
    """Convert OpenAI/internal usage dicts to LangSmith ExtractedUsageMetadata."""
    if not isinstance(usage, dict) or not usage:
        return None
    prompt_tokens = usage.get("input_tokens")
    if prompt_tokens is None:
        prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("output_tokens")
    if completion_tokens is None:
        completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
    return {
        "input_tokens": int(prompt_tokens or 0),
        "output_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _to_float_score(value, default=0.0):
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pick_float_score(doc, keys, default=0.0):
    for key in keys:
        val = doc.get(key)
        if val is not None:
            return _to_float_score(val, default=default)
    return float(default)


def _extract_chunk_trace_fields(raw_item, index=0):
    """Extract normalized universal chunk metadata for LangSmith spans."""
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
        })

    if extracted_chunks_list:
        avg_tok = round(total_tokens / len(extracted_chunks_list), avg_digits)
        avg_ch = round(total_characters / len(extracted_chunks_list), avg_digits)
    else:
        avg_tok = 0
        avg_ch = 0

    return extracted_chunks_list, total_characters, total_tokens, chunk_type_counts, avg_tok, avg_ch


class OfflineLangSmithClient(Client):
    """In-memory LangSmith Client for offline execution and unit tests."""

    def __init__(self, api_url="https://api.smith.langchain.com", web_url="https://smith.langchain.com"):
        self._api_url = api_url
        self._web_url = web_url
        self._api_key = None
        self._custom_headers = {}
        self._headers = {}
        self._manual_cleanup = True
        self._cache = None
        self._atexit_handler = None
        self.session = None
        self._runs = {}
        self._lock = Lock()

    def create_run(self, **kwargs):
        run_id = str(kwargs.get("id") or "")
        if not run_id:
            return
        with self._lock:
            self._runs[run_id] = _sanitize_value(dict(kwargs))

    def update_run(self, run_id, **kwargs):
        key = str(run_id)
        with self._lock:
            existing = self._runs.get(key, {"id": key})
            for field, val in kwargs.items():
                if val is not None:
                    existing[field] = _sanitize_value(val)
            self._runs[key] = existing

    def flush(self, timeout=None):
        return None

    def get_run_url(self, *, run=None, project_name=None, project_id=None):
        run_id = ""
        if run is not None:
            run_id = str(getattr(run, "id", ""))
        base = str(self._web_url or "https://smith.langchain.com").rstrip("/")
        if run_id:
            return f"{base}/o/default/projects/p/{project_name or 'default'}/r/{run_id}"
        return base

    def captured_runs(self, trace_id=None):
        with self._lock:
            items = list(self._runs.values())
        if trace_id is None:
            return items
        target = str(trace_id)
        return [item for item in items if str(item.get("trace_id") or "") == target]


def _safe_post(run_tree):
    if run_tree is None:
        return
    try:
        run_tree.post()
    except Exception as exc:
        logger.debug("LangSmith post error (%s)", type(exc).__name__)


def _safe_patch(run_tree):
    if run_tree is None:
        return
    try:
        run_tree.patch()
    except Exception as exc:
        logger.debug("LangSmith patch error (%s)", type(exc).__name__)


def _apply_run_metadata(run_tree, attributes):
    if run_tree is None or not isinstance(attributes, dict):
        return
    safe_meta = _sanitize_value(attributes)
    usage = safe_meta.get("usage")
    usage_meta = _to_langsmith_usage(usage)
    if "model" in safe_meta and safe_meta.get("model"):
        safe_meta["ls_model_name"] = str(safe_meta["model"])
    if "protocol" in safe_meta and safe_meta.get("protocol"):
        safe_meta["ls_provider"] = str(safe_meta["protocol"])
    kwargs = {"metadata": safe_meta}
    if usage_meta is not None:
        kwargs["usage_metadata"] = usage_meta
    run_tree.set(**kwargs)


def _format_rag_documents(items):
    docs = []
    for i, item in enumerate(items):
        c_meta = _extract_chunk_trace_fields(item, index=i)
        docs.append({
            "page_content": c_meta["content"],
            "metadata": {
                "rank": i + 1,
                "chunk_id": c_meta["chunk_id"],
                "source": c_meta["source"],
                "breadcrumb": c_meta["breadcrumb"],
                "section_title": c_meta["section_title"],
            },
        })
    return docs


def _safe_snapshot_obj(value):
    """Create a thread-safe shallow/recursive copy of JSON-like structures without mutating live objects."""
    if isinstance(value, dict):
        copied = {}
        for key, item in list(value.items()):
            copied[key] = _safe_snapshot_obj(item)
        return copied
    if isinstance(value, list):
        return [_safe_snapshot_obj(item) for item in list(value)]
    if isinstance(value, tuple):
        return [_safe_snapshot_obj(item) for item in list(value)]
    return value


def _enrich_trace_for_debug(raw_record):
    """Build hierarchical waterfall spans, LLM/RAG/tool summaries, and KPI totals for UI debugging."""
    record = _safe_snapshot_obj(raw_record)
    raw_spans = record.get("spans")
    if not isinstance(raw_spans, list):
        raw_spans = []

    is_active = bool(record.get("is_active", False))
    now_ns = time_ns()
    root_started_ns = None
    for sp in raw_spans:
        if isinstance(sp, dict) and isinstance(sp.get("started_ns"), int):
            root_started_ns = sp["started_ns"]
            break

    known_ids = set()
    for idx, sp in enumerate(raw_spans):
        if not isinstance(sp, dict):
            continue
        sid = str(sp.get("span_id") or sp.get("run_id") or f"span_{idx + 1}")
        known_ids.add(sid)

    root_span_id = None
    normalized_spans = []
    span_by_id = {}
    standard_keys = {
        "span_id", "run_id", "parent_span_id", "parent_run_id",
        "run_type", "name", "started_ns", "duration_ms", "status", "attributes",
    }

    for idx, sp in enumerate(raw_spans):
        if not isinstance(sp, dict):
            continue
        sid = str(sp.get("span_id") or sp.get("run_id") or f"span_{idx + 1}")
        rid = str(sp.get("run_id") or sid)
        if root_span_id is None:
            root_span_id = sid

        raw_parent = sp.get("parent_span_id")
        if not raw_parent:
            raw_parent = sp.get("parent_run_id")
        if raw_parent:
            parent_id = str(raw_parent)
        else:
            parent_id = None

        if parent_id and parent_id in known_ids and parent_id != sid:
            effective_parent_id = parent_id
        elif sid != root_span_id:
            effective_parent_id = root_span_id
        else:
            effective_parent_id = None

        raw_attrs = sp.get("attributes")
        if isinstance(raw_attrs, dict):
            attrs = dict(raw_attrs)
        else:
            attrs = {}
        for extra_k, extra_v in sp.items():
            if extra_k not in standard_keys and extra_k not in attrs:
                attrs[extra_k] = extra_v
        attrs = _sanitize_value(attrs)

        started_ns = sp.get("started_ns")
        if isinstance(started_ns, int) and root_started_ns is not None:
            offset_ms = round(max(0.0, (started_ns - root_started_ns) / 1_000_000.0), _DURATION_ROUND_DIGITS)
        else:
            offset_ms = 0.0

        raw_duration = sp.get("duration_ms")
        status_val = str(sp.get("status") or _OK_STATUS)
        if isinstance(raw_duration, (int, float)):
            duration_ms = round(float(raw_duration), _DURATION_ROUND_DIGITS)
        elif is_active and isinstance(started_ns, int):
            duration_ms = round(max(0.0, (now_ns - started_ns) / 1_000_000.0), _DURATION_ROUND_DIGITS)
            if status_val == _OK_STATUS:
                status_val = "running"
        else:
            duration_ms = 0.0

        end_ms = round(offset_ms + duration_ms, _DURATION_ROUND_DIGITS)
        usage_obj = attrs.get("usage")
        if not isinstance(usage_obj, dict):
            usage_obj = attrs.get("usage_metadata")
        tokens_info = _to_langsmith_usage(usage_obj)

        span_name_val = str(sp.get("name") or f"span_{idx + 1}")
        run_type_val = str(sp.get("run_type") or "chain")
        role_attr = str(attrs.get("role") or "")
        is_embedding_span = False
        if run_type_val == "embedding":
            is_embedding_span = True
        elif role_attr == "embedding":
            is_embedding_span = True
        elif span_name_val == "query_embedding":
            is_embedding_span = True
        if is_embedding_span:
            run_type_val = "embedding"

        norm = {
            "span_id": sid,
            "run_id": rid,
            "parent_span_id": parent_id,
            "effective_parent_id": effective_parent_id,
            "depth": 0,
            "order": len(normalized_spans),
            "name": span_name_val,
            "run_type": run_type_val,
            "status": status_val,
            "started_ns": started_ns,
            "offset_ms": offset_ms,
            "duration_ms": duration_ms,
            "end_ms": end_ms,
            "self_duration_ms": duration_ms,
            "children_count": 0,
            "tokens": tokens_info,
            "attributes": attrs,
        }
        normalized_spans.append(norm)
        if sid not in span_by_id:
            span_by_id[sid] = norm

    children_map = {}
    for norm in normalized_spans:
        ep = norm["effective_parent_id"]
        if ep is not None:
            if ep not in children_map:
                children_map[ep] = []
            children_map[ep].append(norm)

    for norm in normalized_spans:
        depth = 0
        cursor = norm["effective_parent_id"]
        visited = {norm["span_id"]}
        while cursor is not None and cursor in span_by_id and cursor not in visited:
            visited.add(cursor)
            depth += 1
            cursor = span_by_id[cursor]["effective_parent_id"]
        norm["depth"] = depth

        direct_children = children_map.get(norm["span_id"], [])
        norm["children_count"] = len(direct_children)
        children_dur = 0.0
        for child in direct_children:
            children_dur += float(child["duration_ms"])
        norm["self_duration_ms"] = round(max(0.0, float(norm["duration_ms"]) - children_dur), _DURATION_ROUND_DIGITS)

    llm_calls = []
    rag_parsers = []
    rag_retrievals = []
    rag_rerankers = []
    rag_embeddings = []
    tool_calls = []
    graph_nodes = []
    error_spans = []

    total_in_tokens = 0
    total_out_tokens = 0
    total_all_tokens = 0
    models_used = []
    models_seen = set()

    has_query_embedding = any(item["name"] == "query_embedding" for item in normalized_spans)

    for norm in normalized_spans:
        s_name = norm["name"]
        s_type = norm["run_type"]
        attrs = norm["attributes"]
        role_attr = str(attrs.get("role") or "")
        is_embed = False
        if s_type == "embedding" or role_attr == "embedding" or s_name == "query_embedding":
            is_embed = True

        if norm["status"] == _ERROR_STATUS or attrs.get("result") == "failed" or bool(attrs.get("error_type")):
            error_spans.append(norm)

        if (s_name in ("model.attempt", "model.route") or s_type == "llm") and not is_embed:
            model_name = str(attrs.get("model") or attrs.get("ls_model_name") or "")
            if model_name and model_name not in models_seen:
                models_seen.add(model_name)
                models_used.append(model_name)

            alias_val = str(attrs.get("alias") or attrs.get("profile") or "")
            protocol_val = str(attrs.get("protocol") or attrs.get("ls_provider") or "")
            tok = norm["tokens"]
            if isinstance(tok, dict) and s_type == "llm":
                total_in_tokens += int(tok.get("input_tokens") or 0)
                total_out_tokens += int(tok.get("output_tokens") or 0)
                total_all_tokens += int(tok.get("total_tokens") or 0)

            llm_calls.append({
                "span_id": norm["span_id"],
                "name": s_name,
                "run_type": s_type,
                "status": norm["status"],
                "offset_ms": norm["offset_ms"],
                "duration_ms": norm["duration_ms"],
                "role": role_attr,
                "configured_role": str(attrs.get("configured_role") or ""),
                "model": model_name,
                "alias": alias_val,
                "protocol": protocol_val,
                "reason": str(attrs.get("reason") or ""),
                "attempt": attrs.get("attempt"),
                "result": str(attrs.get("result") or ""),
                "error_type": str(attrs.get("error_type") or ""),
                "http_status": attrs.get("http_status"),
                "input_bytes": attrs.get("input_bytes"),
                "input_budget_bytes": attrs.get("input_budget_bytes"),
                "context_window": attrs.get("context_window"),
                "temperature": attrs.get("temperature"),
                "reasoning_effort": attrs.get("reasoning_effort"),
                "max_tokens": attrs.get("max_tokens"),
                "max_completion_tokens": attrs.get("max_completion_tokens"),
                "output_tokens_limit": attrs.get("output_tokens_limit"),
                "repair": bool(attrs.get("repair", False)),
                "tokens": tok,
                "attributes": attrs,
            })

        if s_name == "query_parser" or s_type == "parser":
            rag_parsers.append(norm)
        elif s_name == "top_k_cross_encoder_reranker" or (s_type == "retriever" and "rerank" in s_name.lower()):
            rag_rerankers.append(norm)
        elif s_name == "top_k_hybrid_retrieval" or s_type == "retriever":
            rag_retrievals.append(norm)
        elif is_embed or "embed" in s_name.lower():
            if not (s_name == "model.attempt" and has_query_embedding):
                rag_embeddings.append(norm)

        if s_type == "tool":
            tool_calls.append(norm)

        if s_name.startswith("graph."):
            graph_nodes.append(norm)

    turn_snapshot = record.get("turn_snapshot")
    if not isinstance(turn_snapshot, dict):
        turn_snapshot = {}

    root_attrs = {}
    if normalized_spans:
        root_attrs = normalized_spans[0]["attributes"]

    intent_val = turn_snapshot.get("intent")
    if not intent_val:
        intent_val = root_attrs.get("intent")
    audit_val = turn_snapshot.get("audit_status")
    if not audit_val:
        audit_val = root_attrs.get("audit_status")
    if turn_snapshot.get("interaction"):
        if audit_val in (None, "", "PENDING", "UNKNOWN"):
            audit_val = "AWAITING_INPUT"
        if not intent_val and isinstance(turn_snapshot.get("interaction"), dict):
            kind_label = str(turn_snapshot["interaction"].get("kind") or "clarify").upper()
            intent_val = "INTERACTION_" + kind_label
    error_cat_val = turn_snapshot.get("error_category")
    if not error_cat_val:
        error_cat_val = root_attrs.get("error_category")

    citation_count_val = turn_snapshot.get("citations_count")
    if citation_count_val is None:
        citation_count_val = root_attrs.get("citation_count")
    if citation_count_val is None:
        citation_count_val = 0

    query_preview = str(turn_snapshot.get("query") or root_attrs.get("user_query") or "")
    if len(query_preview) > 200:
        query_preview = query_preview[:197] + "..."

    total_duration_ms = record.get("duration_ms")
    if not isinstance(total_duration_ms, (int, float)):
        if normalized_spans:
            total_duration_ms = normalized_spans[0]["duration_ms"]
        else:
            total_duration_ms = 0.0
    total_duration_ms = round(float(total_duration_ms), _DURATION_ROUND_DIGITS)

    return {
        "schema_version": record.get("schema_version", _SCHEMA_VERSION),
        "provider": record.get("provider", "langsmith"),
        "project_name": record.get("project_name", ""),
        "trace_id": str(record.get("trace_id") or ""),
        "run_id": str(record.get("run_id") or ""),
        "turn_id": str(record.get("turn_id") or ""),
        "timestamp": str(record.get("timestamp") or ""),
        "entrypoint": str(record.get("entrypoint") or ""),
        "session_ref": record.get("session_ref"),
        "outcome": str(record.get("outcome") or _DEFAULT_OUTCOME),
        "is_active": is_active,
        "duration_ms": total_duration_ms,
        "langsmith_run_url": record.get("langsmith_run_url"),
        "summary": {
            "span_count": len(normalized_spans),
            "llm_call_count": len(llm_calls),
            "tool_call_count": len(tool_calls),
            "retriever_call_count": len(rag_retrievals) + len(rag_rerankers),
            "parser_call_count": len(rag_parsers),
            "embedding_call_count": len(rag_embeddings),
            "graph_node_count": len(graph_nodes),
            "error_count": len(error_spans),
            "token_usage": {
                "input_tokens": total_in_tokens,
                "output_tokens": total_out_tokens,
                "total_tokens": total_all_tokens,
            },
            "models_used": models_used,
            "intent": intent_val,
            "audit_status": audit_val,
            "error_category": error_cat_val,
            "citation_count": int(citation_count_val),
            "query_preview": query_preview,
        },
        "spans": normalized_spans,
        "llm_calls": llm_calls,
        "rag_stages": {
            "query_parsers": rag_parsers,
            "retrievals": rag_retrievals,
            "rerankers": rag_rerankers,
            "embeddings": rag_embeddings,
        },
        "tool_calls": tool_calls,
        "graph_nodes": graph_nodes,
        "error_spans": error_spans,
        "turn_snapshot": turn_snapshot,
    }


def _build_trace_list_item(enriched):
    """Return compact summary entry for the debug trace sidebar list."""
    summary = enriched["summary"]
    return {
        "trace_id": enriched["trace_id"],
        "run_id": enriched["run_id"],
        "turn_id": enriched["turn_id"],
        "timestamp": enriched["timestamp"],
        "entrypoint": enriched["entrypoint"],
        "session_ref": enriched["session_ref"],
        "outcome": enriched["outcome"],
        "is_active": enriched["is_active"],
        "duration_ms": enriched["duration_ms"],
        "project_name": enriched["project_name"],
        "langsmith_run_url": enriched["langsmith_run_url"],
        "query_preview": summary["query_preview"],
        "intent": summary["intent"],
        "audit_status": summary["audit_status"],
        "error_category": summary["error_category"],
        "span_count": summary["span_count"],
        "llm_call_count": summary["llm_call_count"],
        "tool_call_count": summary["tool_call_count"],
        "error_count": summary["error_count"],
        "token_usage": summary["token_usage"],
        "models_used": summary["models_used"],
    }


class LangSmithTraceRun:
    """
    Unified LangSmith RunTree trace lifecycle for agent turns, RAG queries,
    and document ingestion.
    """

    def __init__(
        self,
        trace_dir,
        entrypoint,
        session_id="",
        *,
        tracer=None,
        query=None,
        run_type="chain",
        tags=None,
        metadata=None,
    ):
        self.trace_dir = trace_dir
        self.entrypoint = entrypoint
        self.session_ref = fingerprint(session_id)
        self.tracer = tracer
        self.query = query
        self.run_type = run_type
        self.tags = list(tags or [])
        if entrypoint and entrypoint not in self.tags:
            self.tags.append(entrypoint)
        self.extra_metadata = _sanitize_value(dict(metadata or {}))

        self.trace_id = None
        self.run_id = None
        self.turn_id = None
        self.spans = []
        self._spans_lock = Lock()
        self.outcome = _DEFAULT_OUTCOME
        self.details = {}
        self.start_time = time.time()
        self.finished = False
        self.root_run_tree = None
        self._active = True
        self.debug_snapshot = None
        self._capture_start_perf = None
        self._capture_timestamp = None

        if tracer is not None:
            self.enabled = bool(tracer.enabled)
            self.client = tracer.client
            self.project_name = tracer.project_name
            self.is_remote = bool(tracer.is_langsmith_active)
            if self.trace_dir is None and tracer.should_export_local():
                self.trace_dir = tracer.trace_dir
        else:
            self.enabled = self.trace_dir is not None
            self.client = OfflineLangSmithClient()
            self.project_name = TracingConfig.LANGSMITH_PROJECT or "ai-banking-assistant"
            self.is_remote = False

    def attach_debug_snapshot(self, snapshot):
        """Attach sanitized turn-level diagnostics in memory for the UI debug inspector."""
        if not isinstance(snapshot, dict):
            return
        safe_snap = _sanitize_value(_safe_snapshot_obj(snapshot))
        with self._spans_lock:
            self.debug_snapshot = safe_snap

    def get_langsmith_run_url(self):
        """Resolve LangSmith web URL for the active root run when remote tracing is active."""
        if not self.is_remote or self.root_run_tree is None or self.client is None:
            return None
        if not hasattr(self.client, "get_run_url"):
            return None
        try:
            return self.client.get_run_url(run=self.root_run_tree, project_name=self.project_name)
        except Exception:
            return None

    def build_debug_record(self, *, is_active=False, duration_ms=None):
        """Snapshot sanitized trace state in memory for live or completed debug inspection."""
        with self._spans_lock:
            spans_copy = _safe_snapshot_obj(self.spans)
            snapshot_copy = _safe_snapshot_obj(self.debug_snapshot)
        if not isinstance(snapshot_copy, dict):
            snapshot_copy = {}

        if isinstance(duration_ms, (int, float)):
            elapsed_ms = round(float(duration_ms), _DURATION_ROUND_DIGITS)
        elif self._capture_start_perf is not None:
            elapsed_ms = round((perf_counter() - self._capture_start_perf) * 1000, _DURATION_ROUND_DIGITS)
        else:
            elapsed_ms = round((time.time() - self.start_time) * 1000, _DURATION_ROUND_DIGITS)

        if is_active:
            outcome_val = "running"
        else:
            outcome_val = self.outcome

        timestamp_val = self._capture_timestamp
        if not timestamp_val:
            timestamp_val = datetime.now(timezone.utc).isoformat()

        return {
            "schema_version": _SCHEMA_VERSION,
            "provider": "langsmith",
            "project_name": self.project_name,
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "timestamp": timestamp_val,
            "entrypoint": self.entrypoint,
            "session_ref": self.session_ref,
            "outcome": outcome_val,
            "is_active": bool(is_active),
            "duration_ms": elapsed_ms,
            "langsmith_run_url": self.get_langsmith_run_url(),
            "spans": _sanitize_value(spans_copy),
            "turn_snapshot": _sanitize_value(snapshot_copy),
        }

    def link_turn(self, state):
        if self.trace_id and isinstance(state, dict) and state.get("agent_turn_id"):
            self.turn_id = state["agent_turn_id"]
            if self.root_run_tree is not None:
                self.root_run_tree.set(metadata={"turn_id": self.turn_id})

    def result(self, response):
        if self.trace_id and isinstance(response, dict):
            audit = response.get("audit_status")
            resp_status = response.get("status")
            if resp_status == "awaiting_input" and not audit:
                audit = "AWAITING_INPUT"
            if audit in _BLOCKED_AUDIT_STATUSES or audit == "AWAITING_INPUT":
                self.outcome = str(audit).lower()
            else:
                if resp_status:
                    self.outcome = resp_status
                else:
                    self.outcome = _DEFAULT_OUTCOME
            intent_val = response.get("intent")
            if not intent_val:
                intent_val = self.details.get("intent")
            err_cat = response.get("error_category")
            if not err_cat:
                err_cat = self.details.get("error_category")
            self.details.update(
                audit_status=audit,
                intent=intent_val,
                error_category=err_cat,
                citation_count=len(response.get("citations", [])),
            )
            response.update(trace_id=self.trace_id, turn_id=self.turn_id)
        return response

    def _build_root_inputs(self):
        inputs = {"entrypoint": self.entrypoint}
        if self.session_ref:
            inputs["session_ref"] = self.session_ref
        if self.query is not None:
            inputs["user_query"] = str(self.query)
        return inputs

    @contextmanager
    def capture(self):
        current = _current_run.get()
        if current is not None:
            parent_run = current[0]
            self.trace_id = parent_run.trace_id
            self.turn_id = parent_run.turn_id
            with parent_run.span(_ROOT_SPAN_PREFIX + self.entrypoint, run_type=self.run_type) as details:
                self.details = details
                self.run_id = details.get("_run_id")
                yield self
            return

        if not self.enabled:
            self._active = False
            yield self
            return

        span_name = _ROOT_SPAN_PREFIX + self.entrypoint
        if self.entrypoint == "query":
            span_name = _span_name("root_span_name")

        root_meta = {
            "schema_version": _SCHEMA_VERSION,
            "entrypoint": self.entrypoint,
            "session_ref": self.session_ref,
            **self.extra_metadata,
        }
        root_rt = RunTree(
            name=span_name,
            run_type=self.run_type,
            inputs=self._build_root_inputs(),
            project_name=self.project_name,
            tags=list(self.tags),
            extra={"metadata": root_meta},
            ls_client=self.client,
        )
        self.root_run_tree = root_rt
        self.trace_id = str(root_rt.trace_id)
        self.run_id = str(root_rt.id)
        self.turn_id = self.trace_id
        root_rt.set(metadata={"trace_id": self.trace_id, "turn_id": self.turn_id})
        _safe_post(root_rt)

        timestamp = datetime.now(timezone.utc).isoformat()
        start = perf_counter()
        self._capture_timestamp = timestamp
        self._capture_start_perf = start
        root_attrs = {}
        root_record = {
            "span_id": self.run_id,
            "run_id": self.run_id,
            "parent_span_id": None,
            "parent_run_id": None,
            "run_type": self.run_type,
            "name": span_name,
            "started_ns": time_ns(),
            "status": _OK_STATUS,
            "attributes": root_attrs,
        }
        self._append_span(root_record)
        self.details = root_attrs
        if self.tracer is not None and hasattr(self.tracer, "pop_pre_turn_spans"):
            for pre_span in self.tracer.pop_pre_turn_spans(self.session_ref):
                if not isinstance(pre_span, dict):
                    continue
                pre_sid = str(pre_span.get("span_id") or uuid4().hex)
                pre_span["span_id"] = pre_sid
                pre_span["run_id"] = str(pre_span.get("run_id") or pre_sid)
                pre_span["parent_span_id"] = self.run_id
                pre_span["parent_run_id"] = self.run_id
                self._append_span(pre_span)
        if self.tracer is not None and hasattr(self.tracer, "register_active_run"):
            self.tracer.register_active_run(self)

        token = _current_run.set((self, root_rt, root_attrs))
        error_msg = None
        try:
            with tracing_context(
                parent=root_rt,
                enabled=True,
                client=self.client,
                project_name=self.project_name,
            ):
                with tracing_v2_enabled(
                    project_name=self.project_name,
                    tags=list(self.tags),
                    client=self.client,
                ):
                    yield self
        except Exception as exc:
            self.outcome = _ERROR_STATUS
            root_record["status"] = _ERROR_STATUS
            error_msg = type(exc).__name__
            self.details.update(error_type=error_msg)
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                self.details["http_status"] = status
                if _REJECTED_HTTP_MIN <= status < _REJECTED_HTTP_MAX:
                    self.outcome = _REJECTED_OUTCOME
            raise
        finally:
            duration_ms = round((perf_counter() - start) * 1000, _DURATION_ROUND_DIGITS)
            root_record["duration_ms"] = duration_ms
            _current_run.reset(token)

            safe_details = _sanitize_value(root_attrs)
            root_outputs = {
                "outcome": self.outcome,
                "duration_ms": duration_ms,
                **safe_details,
            }
            if self.query is not None and "answer_preview" in safe_details:
                root_outputs["answer_preview"] = safe_details["answer_preview"]

            _apply_run_metadata(root_rt, {**safe_details, "outcome": self.outcome, "turn_id": self.turn_id})
            if error_msg:
                root_rt.end(outputs=root_outputs, error=error_msg)
            else:
                root_rt.end(outputs=root_outputs)
            _safe_patch(root_rt)

            if self.tracer is not None and hasattr(self.tracer, "record_completed_trace"):
                self.tracer.record_completed_trace(
                    self.build_debug_record(is_active=False, duration_ms=duration_ms)
                )
            if self.tracer is not None and hasattr(self.tracer, "unregister_active_run"):
                self.tracer.unregister_active_run(self.trace_id)

            if self.trace_dir is not None:
                self._write_local_record(timestamp, duration_ms)

    def _append_span(self, record):
        with self._spans_lock:
            self.spans.append(record)

    def _write_local_record(self, timestamp, duration_ms):
        with self._spans_lock:
            spans_copy = _safe_snapshot_obj(self.spans)
        record = {
            "schema_version": _SCHEMA_VERSION,
            "provider": "langsmith",
            "project_name": self.project_name,
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "timestamp": timestamp,
            "entrypoint": self.entrypoint,
            "session_ref": self.session_ref,
            "outcome": self.outcome,
            "duration_ms": duration_ms,
            "spans": _sanitize_value(spans_copy),
        }
        try:
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            with _write_lock:
                directory = Path(self.trace_dir)
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / _LANGSMITH_RUNS_FILENAME).open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(_SAVE_WARNING_MSG, self.trace_id, type(exc).__name__)

    @contextmanager
    def span(self, name, /, run_type="chain", **attributes):
        if not self.enabled or not self._active:
            yield attributes
            return

        current = _current_run.get()
        parent_ls = get_current_run_tree()
        parent_rt = None
        if parent_ls is not None and self.trace_id and str(parent_ls.trace_id) == self.trace_id:
            parent_rt = parent_ls
        elif current is not None and current[0] is self and current[1] is not None:
            parent_rt = current[1]
        elif self.root_run_tree is not None:
            parent_rt = self.root_run_tree

        parent_id = None
        if parent_rt is not None:
            parent_id = str(parent_rt.id)

        safe_attrs = _sanitize_value(attributes)
        child_rt = None
        if parent_rt is not None:
            child_rt = parent_rt.create_child(
                name=name,
                run_type=run_type,
                inputs=dict(safe_attrs),
                extra={"metadata": dict(safe_attrs)},
            )
            _safe_post(child_rt)
            span_id = str(child_rt.id)
        else:
            span_id = str(len(self.spans) + 1)

        start = perf_counter()
        record = {
            "span_id": span_id,
            "run_id": span_id,
            "parent_span_id": parent_id,
            "parent_run_id": parent_id,
            "run_type": run_type,
            "name": name,
            "started_ns": time_ns(),
            "status": _OK_STATUS,
            "attributes": safe_attrs,
        }
        self._append_span(record)
        token = _current_run.set((self, child_rt, record["attributes"]))
        err_str = None
        try:
            if child_rt is not None:
                with tracing_context(
                    parent=child_rt,
                    enabled=True,
                    client=self.client,
                    project_name=self.project_name,
                ):
                    yield record["attributes"]
            else:
                yield record["attributes"]
        except BaseException as exc:
            from langgraph.errors import GraphInterrupt

            if isinstance(exc, GraphInterrupt):
                record["status"] = _SUSPENDED_STATUS
                record["attributes"]["status"] = _SUSPENDED_STATUS
            else:
                record["status"] = _ERROR_STATUS
                err_str = type(exc).__name__
                record["attributes"]["error_type"] = err_str
            raise
        finally:
            duration_ms = round((perf_counter() - start) * 1000, _DURATION_ROUND_DIGITS)
            record["duration_ms"] = duration_ms
            record["attributes"] = _sanitize_value(record["attributes"])
            _current_run.reset(token)

            if child_rt is not None:
                _apply_run_metadata(child_rt, record["attributes"])
                safe_out = _sanitize_value(record["attributes"])
                if err_str:
                    child_rt.end(outputs=safe_out, error=err_str)
                else:
                    child_rt.end(outputs=safe_out)
                _safe_patch(child_rt)

    @contextmanager
    def node_span(self, name):
        """Enrich an active LangGraph node RunTree or open a child span when outside LangChainTracer."""
        if not self.enabled or not self._active:
            yield {}
            return
        active_rt = get_current_run_tree()
        if active_rt is None or active_rt.name != name or str(active_rt.trace_id) != str(self.trace_id):
            with self.span("graph." + name, run_type="chain") as details:
                yield details
            return

        parent_id = self.run_id
        if active_rt.parent_run_id is not None:
            candidate_parent = str(active_rt.parent_run_id)
            with self._spans_lock:
                known_ids = {str(sp.get("span_id") or "") for sp in self.spans}
            if candidate_parent == str(self.run_id) or candidate_parent in known_ids:
                parent_id = candidate_parent
        span_id = str(active_rt.id)
        node_attrs = {}
        start = perf_counter()
        record = {
            "span_id": span_id,
            "run_id": span_id,
            "parent_span_id": parent_id,
            "parent_run_id": parent_id,
            "run_type": "chain",
            "name": "graph." + name,
            "started_ns": time_ns(),
            "status": _OK_STATUS,
            "attributes": node_attrs,
        }
        self._append_span(record)
        token = _current_run.set((self, active_rt, node_attrs))
        try:
            yield node_attrs
        except BaseException as exc:
            from langgraph.errors import GraphInterrupt

            if isinstance(exc, GraphInterrupt):
                record["status"] = _SUSPENDED_STATUS
                node_attrs["status"] = _SUSPENDED_STATUS
            else:
                record["status"] = _ERROR_STATUS
                node_attrs["error_type"] = type(exc).__name__
            raise
        finally:
            record["duration_ms"] = round((perf_counter() - start) * 1000, _DURATION_ROUND_DIGITS)
            record["attributes"] = _sanitize_value(node_attrs)
            _current_run.reset(token)
            _apply_run_metadata(active_rt, {**record["attributes"], "status": record["status"]})

    # ------------------------------------------------------------------
    # Unified LangSmith RAG Stage Helpers (query, chunking, embedding, retrieval, rerank, guardrails)
    # ------------------------------------------------------------------
    def _emit_child_run(self, name, run_type, inputs, outputs, metadata, duration_ms=0.0):
        parent_rt = get_current_run_tree() or self.root_run_tree
        if parent_rt is None:
            return
        child = parent_rt.create_child(
            name=name,
            run_type=run_type,
            inputs=_sanitize_value(inputs),
            extra={"metadata": _sanitize_value({**metadata, "duration_ms": duration_ms})},
        )
        _safe_post(child)
        child.end(outputs=_sanitize_value(outputs))
        _safe_patch(child)

    def record_chunking_info(self, total_chunks, max_size=None, strategy=None, duration_ms=None):
        if max_size is None:
            max_size = int(_DEFAULTS_CFG.get("chunking_max_size") or 0)
        if strategy is None:
            strategy = _lbl("chunking_strategy")
        if duration_ms is None:
            duration_ms = float(_DEFAULTS_CFG.get("chunking_info_duration_ms") or 0.0)

        self._append_span({
            "name": _lbl("chunking_info_display"),
            "run_type": "chain",
            "duration_ms": duration_ms,
            "total_chunks_stored": total_chunks,
            "strategy": strategy,
            "max_size": max_size,
        })
        self._emit_child_run(
            name=_span_name("chunking_info_span_name"),
            run_type="chain",
            inputs={"chunking_algorithm": strategy, "chunk_size_limit": max_size},
            outputs={"status": _lbl("status_ready"), "total_indexed_chunks": total_chunks},
            metadata={"chunking.strategy": str(strategy), "chunking.total_indexed_chunks": int(total_chunks)},
            duration_ms=duration_ms,
        )

    def record_chunking(self, file_path, num_chunks, duration_ms, details=None, chunks=None, sample_chunks=None):
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

        extracted_list, total_chars, total_tokens, type_counts, avg_tok, avg_ch = _build_chunk_summary_list(raw_chunks)
        if num_chunks:
            final_count = num_chunks
        else:
            final_count = len(extracted_list)

        self._append_span({
            "name": _lbl("chunking_display"),
            "run_type": "chain",
            "duration_ms": duration_ms,
            "file_path": str(file_path),
            "num_chunks": final_count,
            "total_tokens": total_tokens,
            "chunk_type_counts": type_counts,
            "details": dt_dict,
        })
        self._emit_child_run(
            name=_span_name("chunking_span_name"),
            run_type="chain",
            inputs={"action": "document_layout_aware_chunking", "file_path": str(file_path), "options": dt_dict},
            outputs={
                "status": _lbl("status_success"),
                "total_chunks_extracted": final_count,
                "total_characters": total_chars,
                "total_tokens": total_tokens,
                "average_tokens_per_chunk": avg_tok,
                "average_characters_per_chunk": avg_ch,
                "extracted_chunks": extracted_list[:20],
            },
            metadata={"chunking.num_chunks": final_count, "chunking.total_tokens": total_tokens},
            duration_ms=duration_ms,
        )

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
        chunks=None,
    ):
        default_embed_model = str(_DEFAULTS_CFG.get("default_embedding_model") or "embedding-1024")
        if model_name:
            chosen_model = model_name
        elif self.tracer is not None:
            chosen_model = getattr(self.tracer.config, "EMBEDDING_MODEL_1024", default_embed_model)
        else:
            chosen_model = default_embed_model

        if is_query:
            disp_name = _lbl("query_embedding_display")
            if query_text:
                q_content = query_text
            else:
                q_content = str(self.query or "")
            inp_obj = {"action": "query_dense_embedding", "input_query": q_content, "model": chosen_model, "dimension": dimension}
            first_sample, norm_val, vector_preview_str, _ = _summarize_vector(query_vector, dimension)
            out_obj = {
                "status": _lbl("status_query_embedded"),
                "output_dimension": dimension,
                "vector_norm": norm_val,
                "cache_hits": cache_hits,
                "vector_preview": vector_preview_str,
                "first_10_floats": first_sample,
            }
        else:
            disp_name = _lbl("batch_embedding_display")
            inp_obj = {"action": "chunk_batch_embedding", "num_chunks": num_texts, "model": chosen_model, "dimension": dimension}
            out_obj = {
                "status": _lbl("status_batch_embedded"),
                "total_vectors_generated": num_texts,
                "dimension": dimension,
                "sqlite_cache_hits": cache_hits,
            }

        self._append_span({
            "name": disp_name,
            "run_type": "embedding",
            "duration_ms": duration_ms,
            "num_texts": num_texts,
            "dimension": dimension,
            "model": chosen_model,
            "cache_hits": cache_hits,
        })
        self._emit_child_run(
            name=_span_name("embedding_span_name"),
            run_type="embedding",
            inputs=inp_obj,
            outputs=out_obj,
            metadata={
                "ls_model_name": chosen_model,
                "embedding.num_texts": num_texts,
                "embedding.dimension": dimension,
                "embedding.cache_hits": cache_hits,
            },
            duration_ms=duration_ms,
        )

    def record_query_parser(self, input_query, parsed, duration_ms):
        norm_query = getattr(parsed, "normalized_query", getattr(parsed, "raw_query", str(parsed)))
        keywords = getattr(parsed, "keywords", [])
        acronyms = getattr(parsed, "detected_acronyms", getattr(parsed, "expanded_acronyms", []))
        legal_refs = getattr(parsed, "legal_references", [])

        self._append_span({
            "name": _lbl("query_parser_display"),
            "run_type": "parser",
            "duration_ms": duration_ms,
            "normalized_query": norm_query,
            "keywords": keywords,
            "acronyms": acronyms,
            "legal_references": legal_refs,
        })
        self._emit_child_run(
            name=_span_name("query_parser_span_name"),
            run_type="parser",
            inputs={"raw_user_query": input_query},
            outputs={
                "normalized_query": norm_query,
                "extracted_keywords": keywords,
                "detected_banking_acronyms": acronyms,
                "legal_references_found": legal_refs,
            },
            metadata={"query_parser.normalized_query": norm_query},
            duration_ms=duration_ms,
        )

    def record_retrieval(self, query, candidates, duration_ms, params=None):
        doc_count = len(candidates)
        formatted_docs = _format_rag_documents(candidates)
        self._append_span({
            "name": _lbl("retrieval_display"),
            "run_type": "retriever",
            "duration_ms": duration_ms,
            "retrieved_candidates": doc_count,
        })
        self._emit_child_run(
            name=_span_name("retrieval_span_name"),
            run_type="retriever",
            inputs={"query": query, "parameters": params or {}},
            outputs={"documents": formatted_docs},
            metadata={"retrieval.top_k": doc_count},
            duration_ms=duration_ms,
        )

    def record_rerank(self, query, candidates, reranked, duration_ms):
        formatted_docs = _format_rag_documents(reranked)
        self._append_span({
            "name": _lbl("rerank_display"),
            "run_type": "retriever",
            "duration_ms": duration_ms,
            "input_count": len(candidates),
            "top_reranked_count": len(reranked),
        })
        self._emit_child_run(
            name=_span_name("reranker_span_name"),
            run_type="retriever",
            inputs={"query": query, "input_count": len(candidates)},
            outputs={"documents": formatted_docs},
            metadata={"reranker.top_k": len(reranked)},
            duration_ms=duration_ms,
        )

    def record_guardrail_block(self, reason, details=None, duration_ms=0.0):
        if details:
            dt_str = str(details)
        else:
            dt_str = ""
        self._append_span({
            "name": _lbl("guardrail_block_display"),
            "run_type": "chain",
            "duration_ms": duration_ms,
            "blocked": True,
            "reason": reason,
            "details": dt_str,
        })
        self._emit_child_run(
            name=_span_name("guardrail_block_span_name"),
            run_type="chain",
            inputs={"details": dt_str},
            outputs={"blocked": True, "reason": reason},
            metadata={"guardrail.blocked": True, "guardrail.reason": str(reason)},
            duration_ms=duration_ms,
        )

    def record_guardrail_check(self, layer, is_safe, details=None, duration_ms=0.0):
        if details:
            dt_list = details
        else:
            dt_list = []
        self._append_span({
            "name": f"{_lbl('guardrail_check_prefix')}{layer}",
            "run_type": "chain",
            "duration_ms": duration_ms,
            "is_safe": is_safe,
            "details": dt_list,
        })
        span_prefix = _span_name("guardrail_check_span_prefix")
        self._emit_child_run(
            name=f"{span_prefix}{str(layer).lower().replace(' ', '_')}",
            run_type="chain",
            inputs={"layer": layer},
            outputs={"layer": layer, "is_safe": is_safe, "details": dt_list},
            metadata={"guardrail.is_safe": bool(is_safe), "guardrail.layer": str(layer)},
            duration_ms=duration_ms,
        )

    def finish(
        self,
        answer,
        citations=None,
        sources=None,
        is_blocked=False,
        guardrail_trigger=None,
    ):
        if self.finished:
            return
        self.finished = True
        latency_digits = int(_FMT_CFG.get("latency_round_digits") or 1)
        answer_preview_max = int(_FMT_CFG.get("answer_preview_max_chars") or 200)
        total_latency_ms = round((time.time() - self.start_time) * 1000, latency_digits)

        if answer:
            ans_rec = str(answer)[:answer_preview_max]
        else:
            ans_rec = ""

        if is_blocked:
            self.outcome = "blocked"
        else:
            self.outcome = _DEFAULT_OUTCOME

        self.details.update(
            answer_preview=ans_rec,
            is_blocked=is_blocked,
            guardrail_trigger=guardrail_trigger or "",
            legal_citations=citations or [],
            sources_used=sources or [],
        )
        if self.tracer is not None:
            self.tracer.print_trace_summary({
                "query": str(self.query or ""),
                "answer_preview": ans_rec,
                "is_blocked": is_blocked,
                "guardrail_trigger": guardrail_trigger or "",
                "legal_citations": citations or [],
                "sources_used": sources or [],
                "total_latency_ms": total_latency_ms,
                "spans": self.spans,
            })


@contextmanager
def trace_node_context(name):
    """Attach node-level diagnostics to LangGraph's LangSmith node RunTree or open a child span."""
    current = _current_run.get()
    if current is None:
        yield {}
        return
    with current[0].node_span(name) as details:
        yield details


@contextmanager
def trace_span(name, /, run_type="chain", **attributes):
    """Create a child LangSmith span under the active LangSmithTraceRun or RunTree."""
    current = _current_run.get()
    if current is None:
        yield attributes
        return
    with current[0].span(name, run_type=run_type, **attributes) as details:
        yield details


def annotate_span(**attributes):
    """Attach safe diagnostic metadata to the active LangSmith span and RunTree."""
    clean = _sanitize_value(attributes)
    current = _current_run.get()
    if current is not None:
        current[2].update(clean)
    active_rt = get_current_run_tree()
    if active_rt is not None:
        _apply_run_metadata(active_rt, clean)


def trace_stage(name, run_type="chain"):
    """Decorator wrapping a stage function in a child LangSmith span when inside a trace."""
    def decorate(operation):
        @wraps(operation)
        def traced(*args, **kwargs):
            if _current_run.get() is None:
                return operation(*args, **kwargs)
            with trace_span(name, run_type=run_type):
                return operation(*args, **kwargs)
        return traced
    return decorate


class BankingTracer:
    """
    Centralized LangSmith Tracing Manager for AI Banking Assistant.
    Coordinates LangSmith Client, LangChainTracer callbacks, and optional offline JSONL export.
    """

    def __init__(self, config=None):
        if config is None:
            from agent_banking.core.settings import Config
            self.config = Config()
        else:
            self.config = config

        self.enabled = bool(getattr(self.config, "ENABLE_TRACING", TracingConfig.ENABLE_TRACING))
        self.langsmith_tracing = self._resolve_langsmith_flag()
        self.project_name = str(
            getattr(self.config, "LANGSMITH_PROJECT", "")
            or os.getenv("LANGSMITH_PROJECT")
            or os.getenv("LANGCHAIN_PROJECT")
            or TracingConfig.LANGSMITH_PROJECT
            or "ai-banking-assistant"
        ).strip()
        self.endpoint = str(
            getattr(self.config, "LANGSMITH_ENDPOINT", "")
            or os.getenv("LANGSMITH_ENDPOINT")
            or os.getenv("LANGCHAIN_ENDPOINT")
            or TracingConfig.LANGSMITH_ENDPOINT
            or "https://api.smith.langchain.com"
        ).strip()
        self.web_url = str(
            getattr(self.config, "LANGSMITH_WEB_URL", "")
            or os.getenv("LANGSMITH_WEB_URL")
            or TracingConfig.LANGSMITH_WEB_URL
            or "https://smith.langchain.com"
        ).strip()
        self._api_key = str(
            getattr(self.config, "LANGSMITH_API_KEY", "")
            or os.getenv("LANGSMITH_API_KEY")
            or os.getenv("LANGCHAIN_API_KEY")
            or TracingConfig.LANGSMITH_API_KEY
            or ""
        ).strip()

        if hasattr(self.config, "TRACE_DIR"):
            self.trace_dir = Path(self.config.TRACE_DIR)
        else:
            self.trace_dir = Path(TracingConfig.TRACE_DIR)

        self.local_export_enabled = bool(
            getattr(self.config, "LOCAL_TRACE_EXPORT", TracingConfig.LOCAL_TRACE_EXPORT)
        )
        self.is_langsmith_active = False
        self.client = OfflineLangSmithClient(api_url=self.endpoint, web_url=self.web_url)
        self._store_lock = Lock()
        self._max_stored_traces = 200
        self._trace_store = {}
        self._active_runs = {}
        self._pre_turn_spans = {}

        if self.enabled and self.langsmith_tracing:
            self._init_langsmith()

    def current_trace_id(self):
        """Return the active ContextVar trace_id when inside a LangSmithTraceRun."""
        current = _current_run.get()
        if current is None:
            return None
        return getattr(current[0], "trace_id", None)

    def stash_pre_turn_span(self, session_id, span_payload):
        """Hold spans emitted right before trace_agent starts (e.g. Vision OCR pre-turn)."""
        if not session_id or not isinstance(span_payload, dict):
            return
        session_ref = fingerprint(session_id)
        if not session_ref:
            return
        safe_payload = _sanitize_value(_safe_snapshot_obj(span_payload))
        with self._store_lock:
            bucket = self._pre_turn_spans.get(session_ref)
            if bucket is None:
                bucket = []
                self._pre_turn_spans[session_ref] = bucket
            bucket.append(safe_payload)

    def pop_pre_turn_spans(self, session_ref_or_id):
        """Adopt and clear pre-turn spans matching a session fingerprint or raw id."""
        if not session_ref_or_id:
            return []
        fp_key = fingerprint(session_ref_or_id)
        with self._store_lock:
            spans = self._pre_turn_spans.pop(str(session_ref_or_id), None)
            if spans is None and fp_key:
                spans = self._pre_turn_spans.pop(fp_key, None)
        if not spans:
            return []
        return list(spans)

    def register_active_run(self, run):
        """Register an in-flight LangSmithTraceRun for live UI debugging."""
        trace_id = getattr(run, "trace_id", None)
        if not trace_id:
            return
        with self._store_lock:
            self._active_runs[str(trace_id)] = run

    def unregister_active_run(self, trace_id):
        """Remove a completed or failed LangSmithTraceRun from active tracking."""
        if not trace_id:
            return
        with self._store_lock:
            self._active_runs.pop(str(trace_id), None)

    def record_completed_trace(self, record):
        """Store a sanitized completed trace in the bounded FIFO in-memory buffer."""
        if not isinstance(record, dict):
            return
        trace_id = str(record.get("trace_id") or "").strip()
        if not trace_id:
            return
        safe_record = _sanitize_value(_safe_snapshot_obj(record))
        with self._store_lock:
            if trace_id in self._trace_store:
                self._trace_store.pop(trace_id, None)
            self._trace_store[trace_id] = safe_record
            while len(self._trace_store) > self._max_stored_traces:
                oldest_key = next(iter(self._trace_store))
                self._trace_store.pop(oldest_key, None)

    def _load_disk_traces_fallback(self, limit=50):
        """Read recent sanitized traces from local JSONL export when present on disk."""
        jsonl_path = Path(self.trace_dir) / _LANGSMITH_RUNS_FILENAME
        if not jsonl_path.is_file():
            return []
        try:
            raw_lines = jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

        non_empty = [line.strip() for line in raw_lines if line.strip()]
        if limit > 0 and len(non_empty) > limit * 2:
            non_empty = non_empty[-(limit * 2):]

        parsed = []
        for line in reversed(non_empty):
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and item.get("trace_id"):
                parsed.append(_sanitize_value(item))
        return parsed

    def list_debug_traces(self, *, session_id=None, scope="session", limit=50, include_active=True):
        """Return compact debug trace summaries ordered newest-first."""
        target_ref = None
        if scope == "session" and session_id:
            target_ref = fingerprint(session_id)

        with self._store_lock:
            active_runs = list(self._active_runs.values())
            completed_records = list(reversed(list(self._trace_store.values())))
            known_ids = set(self._trace_store.keys())
            mem_total = len(self._trace_store) + len(self._active_runs)

        active_items = []
        active_ids = set()
        if include_active:
            for run in active_runs:
                raw_active = run.build_debug_record(is_active=True)
                tid = str(raw_active.get("trace_id") or "")
                if target_ref is not None and raw_active.get("session_ref") != target_ref:
                    continue
                enriched_active = _enrich_trace_for_debug(raw_active)
                active_items.append(_build_trace_list_item(enriched_active))
                if tid:
                    active_ids.add(tid)
            active_items.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)

        completed_items = []
        for raw_rec in completed_records:
            tid = str(raw_rec.get("trace_id") or "")
            if tid in active_ids:
                continue
            if target_ref is not None and raw_rec.get("session_ref") != target_ref:
                continue
            enriched_rec = _enrich_trace_for_debug(raw_rec)
            completed_items.append(_build_trace_list_item(enriched_rec))

        if len(completed_items) < limit:
            disk_records = self._load_disk_traces_fallback(limit=max(limit, 50))
            for disk_rec in disk_records:
                tid = str(disk_rec.get("trace_id") or "")
                if not tid or tid in known_ids or tid in active_ids:
                    continue
                if target_ref is not None and disk_rec.get("session_ref") != target_ref:
                    continue
                known_ids.add(tid)
                enriched_disk = _enrich_trace_for_debug(disk_rec)
                completed_items.append(_build_trace_list_item(enriched_disk))
                if len(completed_items) >= limit:
                    break

        combined = active_items + completed_items
        return {
            "active_count": len(active_items),
            "total_stored": len(combined),
            "total_in_memory": mem_total,
            "session_ref": target_ref,
            "traces": combined[:limit],
        }

    def get_debug_trace(self, trace_id):
        """Return full enriched debug payload for a live or completed trace_id."""
        if not trace_id:
            return None
        key = str(trace_id).strip()
        if not key:
            return None

        active_run = None
        completed_rec = None
        with self._store_lock:
            active_run = self._active_runs.get(key)
            if active_run is None:
                completed_rec = _safe_snapshot_obj(self._trace_store.get(key))

        if active_run is not None:
            return _enrich_trace_for_debug(active_run.build_debug_record(is_active=True))
        if completed_rec is not None:
            return _enrich_trace_for_debug(completed_rec)

        for disk_rec in self._load_disk_traces_fallback(limit=200):
            if str(disk_rec.get("trace_id") or "") == key:
                return _enrich_trace_for_debug(disk_rec)
        return None

    def clear_debug_traces(self, *, session_id=None, scope="all"):
        """Clear completed in-memory traces matching scope (all or current session)."""
        target_ref = None
        if scope == "session" and session_id:
            target_ref = fingerprint(session_id)

        with self._store_lock:
            if target_ref is None:
                cleared = len(self._trace_store)
                self._trace_store.clear()
                return cleared
            to_delete = [
                tid for tid, rec in self._trace_store.items()
                if rec.get("session_ref") == target_ref
            ]
            for tid in to_delete:
                self._trace_store.pop(tid, None)
            return len(to_delete)

    def _resolve_langsmith_flag(self):
        env_flag = os.getenv("LANGSMITH_TRACING")
        if env_flag is None or not env_flag.strip():
            env_flag = os.getenv("LANGCHAIN_TRACING_V2")
        if env_flag is not None and env_flag.strip():
            return env_flag.strip().lower() == "true"
        return bool(getattr(self.config, "LANGSMITH_TRACING", TracingConfig.LANGSMITH_TRACING))

    def should_export_local(self):
        """Export local LangSmith JSONL only when opted in or when an isolated TRACE_DIR is set."""
        if not self.enabled:
            return False
        if self.local_export_enabled:
            return True
        default_dir = Path(TracingConfig.TRACE_DIR).resolve()
        return self.trace_dir.resolve() != default_dir

    def _init_langsmith(self):
        if not self._api_key:
            self.is_langsmith_active = False
            logger.debug(_ls_msg("missing_api_key"))
            return
        try:
            self.client = Client(
                api_url=self.endpoint,
                api_key=self._api_key,
                web_url=self.web_url,
                timeout_ms=int(_LS_CFG.get("timeout_ms") or 5000),
                auto_batch_tracing=bool(_LS_CFG.get("auto_batch_tracing", True)),
                anonymizer=_sanitize_value,
                hide_inputs=bool(_LS_CFG.get("hide_inputs", False)),
                hide_outputs=bool(_LS_CFG.get("hide_outputs", False)),
            )
            self.is_langsmith_active = True
            logger.info(_ls_msg("connected", project=self.project_name, endpoint=self.endpoint))
        except Exception as exc:
            self.is_langsmith_active = False
            self.client = OfflineLangSmithClient(api_url=self.endpoint, web_url=self.web_url)
            logger.warning(_ls_msg("connect_failed", error=exc))

    def create_langchain_tracer(self, tags=None, metadata=None):
        """Create a LangChainTracer configured for this BankingTracer's LangSmith client."""
        return LangChainTracer(
            project_name=self.project_name,
            client=self.client,
            tags=list(tags or []),
            metadata=_sanitize_value(dict(metadata or {})),
        )

    def trace_agent(self, entrypoint, session_id, *, trace_dir=None, tags=None, metadata=None):
        """Capture an agent turn, SSE stream, resume, or document run in LangSmith."""
        target_dir = trace_dir
        if target_dir is None and self.should_export_local():
            target_dir = self.trace_dir
        return LangSmithTraceRun(
            target_dir,
            entrypoint,
            session_id,
            tracer=self,
            tags=tags,
            metadata=metadata,
        ).capture()

    @contextmanager
    def trace_query(self, query, session_id=""):
        """Create and capture a unified LangSmith trace session for a RAG query."""
        target_dir = None
        if self.should_export_local():
            target_dir = self.trace_dir
        run = LangSmithTraceRun(
            target_dir,
            "query",
            session_id,
            tracer=self,
            query=query,
            run_type="chain",
        )
        with run.capture() as active_run:
            try:
                yield active_run
            finally:
                if not active_run.finished:
                    active_run.finish(answer="")

    def trace_chunking(self, file_path, num_chunks, duration_ms, details=None, chunks=None, sample_chunks=None):
        """Record DLA Layout-Aware chunking in LangSmith during ingestion."""
        if not self.enabled:
            return
        fname = Path(file_path).name
        with self.trace_query(query=f"{_lbl('ingestion_chunking_prefix')}{fname}") as run:
            run.record_chunking(
                file_path=file_path,
                num_chunks=num_chunks,
                duration_ms=duration_ms,
                details=details,
                chunks=chunks,
                sample_chunks=sample_chunks,
            )
            run.finished = True

    def trace_embedding(
        self,
        num_texts,
        dimension,
        duration_ms,
        model_name=None,
        cache_hits=0,
        chunks=None,
        vectors=None,
    ):
        """Record dense vector embedding in LangSmith during ingestion indexing."""
        if not self.enabled:
            return
        query_tpl = _lbl("ingestion_embedding_template")
        if query_tpl:
            query_label = query_tpl.format(num_texts=num_texts)
        else:
            query_label = str(num_texts)
        with self.trace_query(query=query_label) as run:
            run.record_embedding(
                num_texts=num_texts,
                dimension=dimension,
                duration_ms=duration_ms,
                model_name=model_name,
                cache_hits=cache_hits,
                is_query=False,
                chunks=chunks,
            )
            run.finished = True

    def flush(self):
        if self.client is not None and hasattr(self.client, "flush"):
            try:
                self.client.flush()
            except Exception as exc:
                logger.debug("LangSmith flush error (%s)", type(exc).__name__)

    def print_trace_summary(self, record):
        """Render terminal trace summary box."""
        box_width = int(_FMT_CFG.get("terminal_box_width") or 70)
        query_preview_max = int(_FMT_CFG.get("query_preview_max_chars") or 50)
        citations_limit = int(_FMT_CFG.get("citations_preview_limit") or 3)

        total_ms = record.get("total_latency_ms", 0.0)
        query = record.get("query", "")
        spans = record.get("spans", [])

        if self.is_langsmith_active:
            sync_tag = str(_SUMMARY_TPLS.get("langsmith_on_tag") or "LangSmith: ON")
        else:
            sync_tag = str(_SUMMARY_TPLS.get("langsmith_local_tag") or "LangSmith: LOCAL")
        sync_str = f" [{sync_tag}]"

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
            block_line = str(_SUMMARY_TPLS.get("block_line_template") or "").format(
                trigger=record.get("guardrail_trigger", "")
            )
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


tracer = BankingTracer()
