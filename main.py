#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Banking Assistant - Web Application Server
======================================================================
Launches web API service and static frontend connected to all RAG and Agentic nodes:

1. Deep agent chat: /api/chat, /api/agent/chat, /api/agent/stream
2. Config Endpoint: /api/config
3. Node Folders Management: /api/nodes, /api/nodes/{node_name}/{io_type}/{file_name}
4. Static Frontend: http://127.0.0.1:8000

MANDATORY RULES:
- 0% REGEX LAW: Strictly string methods only (in, split, startswith, endswith, strip, lower, replace, find).
- ZERO TERNARY: Explicit if/else syntax only (no x if c else y).
"""

import os
import sys
import time
import json
import base64
import asyncio
import threading
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

# Ensure UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from agent_banking.core.config.settings import CoreConfig, get_core_messages
from agent_banking.core.cloud_rate_limit import acquire_cloud_slot
from agent_banking.core.settings import Config
from agent_banking.document_viewer import (
    find_document_path,
    get_pdf_manifest,
    locate_pdf_passages,
    prewarm_pdf_citations,
    read_docx_media,
    render_docx_preview,
    render_pdf_page_png,
    render_pdf_region_png,
    render_xlsx_preview,
    search_pdf_document,
)
from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.node_pipeline import NodePipeline
from agent_banking.agent_pipeline import BankingAgentPipeline, InteractionError
from agent_banking.tracing.tracer import trace_span
from agent_banking.agent.config.response_profiles import (
    PERSONA_PROFILES,
    TONE_PROFILES,
    FORMAT_PROFILES
)

_API_CFG = CoreConfig.CORE_API
_AP_CFG = CoreConfig.CORE_AGENT_PIPELINE
_API_MESSAGES = get_core_messages(None, "api_messages.yaml", "API_MESSAGES")
_SSE_DESCRIPTIONS = dict(_API_MESSAGES.get("sse_node_descriptions", {}))
_FORMAT_PREFIXES = tuple(_API_MESSAGES.get("format_prefixes", []))
_STARTUP_BANNER = dict(_API_MESSAGES.get("startup_banner", {}))

_DEFAULT_PERSONA = str(_AP_CFG["default_persona"])
_DEFAULT_TONE = str(_AP_CFG["default_tone"])
_DEFAULT_FORMAT_TYPE = str(_AP_CFG["default_format_type"])
_DEFAULT_CUSTOM_PERSONA = str(_AP_CFG["default_custom_persona"])

_STATIC_DIR_NAME = str(_API_CFG["static_dir"])
_INDEX_FILE_NAME = str(_API_CFG["index_file"])
_STREAM_STEP_DELAY = float(_API_CFG["stream_step_delay_sec"])
_STREAM_TOKEN_DELAY = float(_API_CFG["stream_token_delay_sec"])
_FINAL_METADATA_KEYS = tuple(_API_CFG["final_metadata_keys"])
_TRACE_METADATA_KEYS = tuple(_API_CFG["trace_metadata_keys"])

# Initialize FastAPI application
app = FastAPI(
    title=str(_API_CFG["title"]),
    description=str(_API_CFG["description"]),
    version=str(_API_CFG["version"])
)

# Mount static directory for frontend assets
static_dir = ROOT_DIR / _STATIC_DIR_NAME
if static_dir.exists():
    app.mount("/" + _STATIC_DIR_NAME, StaticFiles(directory=str(static_dir)), name=_STATIC_DIR_NAME)

# Initialize Config & BankingRAGPipeline
config = Config()
use_memory_env = False
if os.environ.get(str(_API_CFG["use_memory_env"]), "0") == "1":
    use_memory_env = True

try:
    rag_pipeline = BankingRAGPipeline(config=config, use_memory=use_memory_env)
except RuntimeError as lock_err:
    if "already accessed by another instance" in str(lock_err):
        print(_API_MESSAGES["lock_warning"])
        rag_pipeline = BankingRAGPipeline(config=config, use_memory=True)
    else:
        raise

# Manage indexing and diagnostic folders.
node_pipe = NodePipeline()


_MAX_IMAGE_ATTACHMENTS = 3
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_VISION_SNIPPET_CHARS = 1200
_ALLOWED_IMAGE_MIMES = ("png", "jpeg", "jpg", "webp", "gif")
_DEFAULT_IMAGE_QUERY = "Hãy phân tích nội dung hình ảnh đính kèm và tư vấn hướng xử lý theo quy định của A-Bank."
_VISION_ERROR_MARKER = "[Không thể trích xuất chi tiết từ ảnh do lỗi kết nối Cloud Vision]"


# Pydantic Schemas
class ImageAttachment(BaseModel):
    name: str = Field(default="", max_length=120)
    data_url: str = Field(min_length=20, max_length=8_000_000)


class ChatRequest(BaseModel):
    message: str = ""
    persona: str | None = _DEFAULT_PERSONA
    tone: str | None = _DEFAULT_TONE
    format_style: str | None = _DEFAULT_FORMAT_TYPE
    custom_persona: str | None = _DEFAULT_CUSTOM_PERSONA
    session_id: str | None = None
    attachments: list[ImageAttachment] | None = None


class VisionAnalyzeRequest(BaseModel):
    message: str = ""
    attachments: list[ImageAttachment] = Field(min_length=1, max_length=3)


class SessionRequest(BaseModel):
    session_id: str = Field(
        min_length=int(_API_CFG["session_id_min_length"]),
        max_length=int(_API_CFG["session_id_max_length"])
    )


class ResumeRequest(SessionRequest):
    interaction_id: str = Field(
        min_length=int(_API_CFG["interaction_id_length"]),
        max_length=int(_API_CFG["interaction_id_length"])
    )
    choice_id: str | None = Field(
        default=None,
        min_length=int(_API_CFG["choice_id_min_length"]),
        max_length=int(_API_CFG["choice_id_max_length"])
    )
    choice_ids: list[str] | None = Field(default=None, min_length=1, max_length=10)
    text: str | None = Field(default=None, max_length=int(_API_CFG["resume_text_max_length"]))


class SessionCreateRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=128)
    title: str = Field(default="", max_length=120)


class SessionUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class PdfLocateRequest(BaseModel):
    source: str = Field(min_length=1, max_length=512)
    page_numbers: list[int] = Field(default_factory=list, max_length=12)
    chunk_id: str = Field(default="", max_length=160)
    excerpt: str = Field(default="", max_length=8000)
    section_title: str = Field(default="", max_length=400)
    images: list[dict] = Field(default_factory=list, max_length=24)


# -----------------------------------------------------------------------------
# 1. SERVE FRONTEND USER INTERFACE
# -----------------------------------------------------------------------------
@app.get("/")
def serve_index():
    index_file = ROOT_DIR / _STATIC_DIR_NAME / _INDEX_FILE_NAME
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": _API_MESSAGES["server_running"]}


# -----------------------------------------------------------------------------
# 2. SYSTEM CONFIGURATION & PERSONA PROFILES
# -----------------------------------------------------------------------------
@app.get("/api/persona/profiles")
def get_persona_profiles():
    """
    Return persona, tone, and format profiles for frontend configuration.
    """
    tone_list = []
    for k, v in TONE_PROFILES.items():
        if " (" in v:
            t_name = v.split(" (")[0]
        else:
            t_name = v
        tone_list.append({"id": k, "name": t_name, "description": v})

    format_list = []
    for k, v in FORMAT_PROFILES.items():
        if ":" in v:
            f_name = v.split(":")[0]
            for prefix in _FORMAT_PREFIXES:
                f_name = f_name.replace(prefix, "")
        else:
            f_name = v
        format_list.append({"id": k, "name": f_name, "description": v})

    return {
        "personas": [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"]
            }
            for k, v in PERSONA_PROFILES.items()
        ],
        "tones": tone_list,
        "formats": format_list
    }


@app.get("/api/config")
def get_config():
    total_points = rag_pipeline.vector_store.count()
    return {
        "default_model": config.AGENT_LLM_MODEL,
        "agent_mode": str(_API_CFG["agent_mode"]),
        "embedding_model": config.EMBEDDING_MODEL_1024,
        "reranker_model": config.RERANKER_MODEL,
        "qdrant_collection": config.QDRANT_COLLECTION,
        "points_count": total_points,
        "tracing_enabled": rag_pipeline.tracer.is_langsmith_active,
        "tracing_provider": "langsmith",
        "langsmith_project": rag_pipeline.tracer.project_name,
        "langsmith_url": str(_API_CFG["langsmith_url"]),
        "nodes": list(_API_CFG["nodes"])
    }


def _resolve_chat_style(req: ChatRequest) -> dict:
    """Normalize persona, tone, format_style, and custom_persona fields from ChatRequest."""
    persona_val = _DEFAULT_PERSONA
    if req.persona:
        persona_val = req.persona

    tone_val = _DEFAULT_TONE
    if req.tone:
        tone_val = req.tone

    format_style_val = _DEFAULT_FORMAT_TYPE
    if req.format_style:
        format_style_val = req.format_style

    custom_persona_val = _DEFAULT_CUSTOM_PERSONA
    if req.custom_persona:
        custom_persona_val = req.custom_persona

    return {
        "persona": persona_val,
        "tone": tone_val,
        "format_style": format_style_val,
        "custom_persona": custom_persona_val
    }


# -----------------------------------------------------------------------------
# 3. BANKING DEEP AGENT
# -----------------------------------------------------------------------------
_agent_pipeline_instance = BankingAgentPipeline(
    config=config, use_memory=rag_pipeline.use_memory, rag_pipeline=rag_pipeline
)


def get_agent_pipeline():
    return _agent_pipeline_instance


def _get_vision_client():
    from agent_banking.chunking.universal.models.vision_client import CloudVisionClient
    client = CloudVisionClient()
    client.disable_thinking = True
    return client


def _decode_image_data_url(data_url: str) -> bytes:
    """Validate and decode a base64 image data URL without regex."""
    if not isinstance(data_url, str):
        raise HTTPException(
            status_code=400,
            detail="Định dạng ảnh đính kèm không hợp lệ (chỉ hỗ trợ PNG, JPEG, WEBP, GIF)."
        )
    clean = data_url.strip()
    if not clean.startswith("data:image/") or ";base64," not in clean:
        raise HTTPException(
            status_code=400,
            detail="Định dạng ảnh đính kèm không hợp lệ (chỉ hỗ trợ PNG, JPEG, WEBP, GIF)."
        )
    header, encoded = clean.split(";base64,", 1)
    mime_sub = header[len("data:image/"):].strip().lower()
    if mime_sub not in _ALLOWED_IMAGE_MIMES:
        raise HTTPException(
            status_code=400,
            detail="Định dạng ảnh đính kèm không hợp lệ (chỉ hỗ trợ PNG, JPEG, WEBP, GIF)."
        )
    try:
        raw = base64.b64decode(encoded.strip(), validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Dữ liệu ảnh base64 không hợp lệ.") from None

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Tệp ảnh đính kèm rỗng.")
    if len(raw) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Mỗi ảnh đính kèm không được vượt quá 5 MB.")

    is_png = raw.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = raw.startswith(b"\xff\xd8")
    is_gif = raw.startswith((b"GIF87a", b"GIF89a"))
    is_webp = len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"
    if not (is_png or is_jpeg or is_gif or is_webp):
        raise HTTPException(
            status_code=400,
            detail="Định dạng ảnh đính kèm không hợp lệ (chỉ hỗ trợ PNG, JPEG, WEBP, GIF)."
        )
    return raw


def _validate_and_decode_attachments(attachments: list[ImageAttachment] | None) -> list[dict]:
    """Validate attachment count and decode all image payloads up front."""
    if not attachments:
        return []
    if len(attachments) > _MAX_IMAGE_ATTACHMENTS:
        raise HTTPException(status_code=400, detail="Tối đa 3 ảnh đính kèm cho mỗi lượt gửi.")
    decoded = []
    for idx, item in enumerate(attachments):
        raw_bytes = _decode_image_data_url(item.data_url)
        clean_name = str(item.name or "").strip()
        if not clean_name:
            clean_name = f"image_{idx + 1}"
        decoded.append({"name": clean_name, "bytes": raw_bytes})
    return decoded


def _analyze_single_image_bytes(
    image_bytes: bytes,
    file_name: str,
    user_hint: str,
    agent=None,
    session_id: str = "",
) -> str:
    """Run Cloud Vision OCR/analysis on one image attachment and sanitize output."""
    prompt = (
        "Hãy đọc và trích xuất chính xác toàn bộ nội dung hiển thị trong ảnh bằng tiếng Việt ngắn gọn, rõ ràng "
        "(tiêu đề, văn bản, số liệu, bảng biểu, mã lỗi nếu có như ABP-..., số tiền, tên sản phẩm/thẻ, "
        "biểu phí hoặc điều khoản). Chỉ nêu đúng những gì nhìn thấy trong ảnh, không bịa đặt và không cần "
        "liệt kê các mục không xuất hiện."
    )
    clean_hint = str(user_hint or "").strip()
    if clean_hint:
        prompt = prompt + "\nCâu hỏi kèm theo của khách hàng: " + clean_hint[:200]

    started_ns = time.time_ns()
    start_perf = time.perf_counter()
    stashed_attrs = {"role": "vision", "file_name": file_name}
    stashed_status = "ok"
    try:
        with trace_span("vision.attachment_ocr", run_type="llm", role="vision", file_name=file_name) as span_meta:
            try:
                client = _get_vision_client()
                model_name = getattr(client, "model", "")
                base_url = getattr(client, "base_url", "")
                if model_name:
                    span_meta["model"] = model_name
                    stashed_attrs["model"] = model_name
                if model_name and base_url:
                    acquire_cloud_slot(model_name, base_url)
                raw_text = client.ask(image_bytes, prompt=prompt, max_tokens=1500, temperature=0.0)
                cleaned = " ".join(str(raw_text or "").split())
                if cleaned:
                    runtime = getattr(agent, "runtime", None)
                    if runtime is not None and hasattr(runtime, "guard_artifact"):
                        guarded = runtime.guard_artifact(cleaned)
                        if not guarded:
                            cleaned = "[Nội dung văn bản trong ảnh đã bị chặn bởi bộ lọc an toàn]"
                        else:
                            cleaned = guarded
                if not cleaned:
                    cleaned = "[Không đọc được nội dung chữ rõ ràng từ ảnh]"
                if len(cleaned) > _MAX_VISION_SNIPPET_CHARS:
                    cleaned = cleaned[:_MAX_VISION_SNIPPET_CHARS - 3] + "..."
                span_meta["extracted_chars"] = len(cleaned)
                span_meta["status"] = "success"
                stashed_attrs.update(span_meta)
                return cleaned
            except Exception as exc:
                print(f"[Vision attachment OCR] {file_name}: {type(exc).__name__}: {exc}")
                stashed_status = "error"
                span_meta["status"] = "error"
                span_meta["error_type"] = type(exc).__name__
                stashed_attrs.update(span_meta)
                return _VISION_ERROR_MARKER
    finally:
        tracer_obj = _get_active_tracer()
        if tracer_obj is not None and hasattr(tracer_obj, "stash_pre_turn_span"):
            has_active = False
            if hasattr(tracer_obj, "current_trace_id") and tracer_obj.current_trace_id():
                has_active = True
            if not has_active and session_id:
                elapsed_ms = round((time.perf_counter() - start_perf) * 1000.0, 3)
                tracer_obj.stash_pre_turn_span(
                    session_id,
                    {
                        "name": "vision.attachment_ocr",
                        "run_type": "llm",
                        "started_ns": started_ns,
                        "duration_ms": elapsed_ms,
                        "status": stashed_status,
                        "attributes": stashed_attrs,
                    },
                )


def _extract_vision_text_from_attachments(
    decoded_items: list[dict],
    user_hint: str,
    agent=None,
    session_id: str = "",
) -> list[dict]:
    """Analyze each decoded image attachment and return structured OCR summaries."""
    analyses = []
    for idx, item in enumerate(decoded_items):
        summary = _analyze_single_image_bytes(
            item["bytes"],
            item["name"],
            user_hint,
            agent=agent,
            session_id=session_id,
        )
        analyses.append({
            "index": idx + 1,
            "name": item["name"],
            "summary": summary,
        })
    return analyses


def _prepare_query_with_attachments(
    query_text: str,
    decoded_items: list[dict],
    agent=None,
    session_id: str = "",
) -> tuple[str, list[dict]]:
    """Combine user query text with sanitized Cloud Vision extractions from attachments."""
    base_query = str(query_text or "").strip()
    if not decoded_items:
        return base_query, []
    if not base_query:
        base_query = _DEFAULT_IMAGE_QUERY
    analyses = _extract_vision_text_from_attachments(
        decoded_items,
        base_query,
        agent=agent,
        session_id=session_id,
    )
    parts = []
    for item in analyses:
        parts.append(f"Ảnh {item['index']} ({item['name']}): {item['summary']}")
    appendix = "[Thông tin trích xuất từ ảnh đính kèm của khách hàng: " + " | ".join(parts) + "]"
    max_total = 2600
    if len(base_query) + len(appendix) + 2 > max_total:
        avail = max(120, max_total - len(base_query) - 3)
        appendix = appendix[:avail - 3] + "...]"
    enriched_query = base_query + "\n\n" + appendix
    return enriched_query, analyses


@app.post("/api/vision/analyze")
def analyze_vision_attachments(req: VisionAnalyzeRequest):
    """Analyze up to 3 attached images via Cloud Vision and return sanitized summaries."""
    decoded_items = _validate_and_decode_attachments(req.attachments)
    agent = get_agent_pipeline()
    enriched_query, analyses = _prepare_query_with_attachments(req.message.strip(), decoded_items, agent=agent)
    return {
        "status": "completed",
        "analyses": analyses,
        "combined_query": enriched_query,
    }


@app.post("/api/chat")
@app.post("/api/agent/chat")
def handle_agent_chat(req: ChatRequest):
    """
    Execute a deep agent turn with persistent session memory.
    """
    query_text = req.message.strip()
    decoded_items = _validate_and_decode_attachments(req.attachments)
    if not query_text and not decoded_items:
        raise HTTPException(status_code=400, detail=_API_MESSAGES["empty_query_error"])

    session_id = req.session_id or uuid4().hex

    style = _resolve_chat_style(req)

    try:
        agent = get_agent_pipeline()
        final_query = query_text
        if decoded_items:
            final_query, _ = _prepare_query_with_attachments(
                query_text,
                decoded_items,
                agent=agent,
                session_id=session_id,
            )
        result = agent.chat(
            query=final_query,
            session_id=session_id,
            persona=style["persona"],
            tone=style["tone"],
            format_type=style["format_style"],
            custom_persona=style["custom_persona"]
        )
        if isinstance(result, dict):
            _schedule_pdf_prewarm(result.get("citations"))
        return result
    except Exception as error:
        print(f"[Deep agent chat] {type(error).__name__}")
        raise HTTPException(status_code=500, detail=_API_MESSAGES["chat_internal_error"]) from None


@app.post("/api/agent/status")
def agent_status(req: SessionRequest):
    return get_agent_pipeline().status(req.session_id)


@app.post("/api/agent/resume")
def agent_resume(req: ResumeRequest):
    try:
        return get_agent_pipeline().resume(
            req.session_id,
            req.interaction_id,
            req.choice_id,
            req.text,
            choice_ids=req.choice_ids,
        )
    except InteractionError as error:
        headers = None
        if getattr(error, "trace_id", None):
            headers = {"X-Trace-ID": error.trace_id}
        raise HTTPException(status_code=error.status_code, detail=str(error), headers=headers) from None
    except Exception as error:
        print(f"[Deep agent resume] {type(error).__name__}")
        headers = None
        if getattr(error, "trace_id", None):
            headers = {"X-Trace-ID": error.trace_id}
        raise HTTPException(status_code=500, detail=_API_MESSAGES["resume_internal_error"], headers=headers) from None


async def _emit_graph_sse_events(step_stream, session_id: str, stream_started: float, error_label: str, default_error_msg: str):
    final_response_text = ""
    audit_status_used = "PASS"
    citations = []
    sources = []
    images = []
    visualization = None
    follow_up_questions = []
    final_metadata = {}

    try:
        async for step_event in step_stream:
            for node_name, node_state in step_event.items():
                if node_name == "awaiting_input":
                    event = {"event_type": "awaiting_input", **node_state}
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    return
                if "audit_status" in node_state:
                    audit_status_used = str(node_state["audit_status"])

                node_description = _SSE_DESCRIPTIONS.get(node_name, node_name)
                if node_name == "answer_review" and node_state["audit_status"] == "RETRY":
                    node_description = _SSE_DESCRIPTIONS.get("answer_review_retry", node_description)
                action = node_state.get("last_action")
                if action:
                    node_description = f"{action['kind']} · {action['name']} · {action['status']}"
                if node_name == "output_guardrail":
                    final_response_text = node_state["final_answer"]
                    citations = node_state["citations"]
                    sources = node_state["sources"]
                    images = node_state.get("images", [])
                    visualization = node_state.get("visualization")
                    follow_up_questions = node_state.get("follow_up_questions", [])
                    _schedule_pdf_prewarm(citations)
                    final_metadata = {
                        key: node_state[key] for key in _FINAL_METADATA_KEYS
                    }
                    final_metadata.update({key: node_state[key] for key in _TRACE_METADATA_KEYS if key in node_state})
                    if "dag_telemetry" in node_state:
                        final_metadata["dag_telemetry"] = node_state["dag_telemetry"]
                    if "session_title" in node_state and node_state["session_title"]:
                        final_metadata["session_title"] = node_state["session_title"]
                    if "accepted_interaction_id" in node_state and node_state["accepted_interaction_id"]:
                        final_metadata["accepted_interaction_id"] = node_state["accepted_interaction_id"]

                step_payload = {
                    "event_type": "node_progress",
                    "node": node_name,
                    "description": node_description,
                    "action": action
                }
                if "trace_id" in node_state and node_state["trace_id"]:
                    step_payload["trace_id"] = node_state["trace_id"]
                if "plan" in node_state:
                    step_payload["plan"] = node_state["plan"]
                if node_name == "answer_review":
                    step_payload["audit_status"] = node_state["audit_status"]
                if "dag_node" in node_state:
                    step_payload["dag_node"] = node_state["dag_node"]
                if "dag_telemetry" in node_state:
                    step_payload["dag_telemetry"] = node_state["dag_telemetry"]
                data_str = json.dumps(step_payload, ensure_ascii=False)
                yield f"data: {data_str}\n\n"
                await asyncio.sleep(_STREAM_STEP_DELAY)

        words = []
        if final_response_text:
            words = final_response_text.split(" ")
        for word in words:
            token_payload = {
                "event_type": "token",
                "content": word + " "
            }
            token_str = json.dumps(token_payload, ensure_ascii=False)
            yield f"data: {token_str}\n\n"
            await asyncio.sleep(_STREAM_TOKEN_DELAY)

        stream_latency_ms = round((time.perf_counter() - stream_started) * 1000.0, 2)
        done_status = "completed"
        if audit_status_used == "CANCELED":
            done_status = "canceled"
        done_payload = {
            "event_type": "done",
            "status": done_status,
            "session_id": session_id,
            "audit_status": audit_status_used,
            "latency_ms": stream_latency_ms,
            "evaluation_metrics": None,
            "citations": citations,
            "sources": sources,
            "images": images,
            "visualization": visualization,
            "follow_up_questions": follow_up_questions,
            "full_answer": final_response_text,
            **final_metadata
        }
        done_str = json.dumps(done_payload, ensure_ascii=False)
        yield f"data: {done_str}\n\n"

    except InteractionError as err:
        error_payload = {
            "event_type": "error",
            "status_code": err.status_code,
            "message": str(err)
        }
        if getattr(err, "trace_id", None):
            error_payload["trace_id"] = err.trace_id
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
    except Exception as err:
        print(f"[{error_label}] {type(err).__name__}")
        error_payload = {
            "event_type": "error",
            "message": default_error_msg
        }
        if getattr(err, "trace_id", None):
            error_payload["trace_id"] = err.trace_id
        err_str = json.dumps(error_payload, ensure_ascii=False)
        yield f"data: {err_str}\n\n"


@app.post("/api/agent/resume/stream")
async def stream_agent_resume(req: ResumeRequest):
    """Server-Sent Events (SSE) streaming endpoint for resuming an interrupted interaction."""
    async def event_generator():
        stream_started = time.perf_counter()
        agent = get_agent_pipeline()
        step_stream = agent.astream_resume(
            req.session_id,
            req.interaction_id,
            req.choice_id,
            req.text,
            choice_ids=req.choice_ids,
        )
        async for chunk in _emit_graph_sse_events(
            step_stream,
            req.session_id,
            stream_started,
            "Deep agent resume stream",
            _API_MESSAGES["resume_internal_error"],
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.post("/api/agent/stream")
async def stream_agent_chat(req: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint for real-time node progress and token streaming.
    """
    query_text = req.message.strip()
    decoded_items = _validate_and_decode_attachments(req.attachments)
    if query_text == "" and not decoded_items:
        raise HTTPException(status_code=400, detail=_API_MESSAGES["empty_query_error"])

    session_id = req.session_id or uuid4().hex

    style = _resolve_chat_style(req)

    async def event_generator():
        stream_started = time.perf_counter()
        try:
            agent = get_agent_pipeline()
            final_query = query_text
            if decoded_items:
                vision_started = time.perf_counter()
                vision_start_payload = {
                    "event_type": "node_progress",
                    "node": "vision_ocr",
                    "description": "Đang phân tích hình ảnh đính kèm qua Cloud Vision...",
                    "action": {"kind": "vision", "name": "cloud_vision_ocr", "status": "RUNNING"},
                    "dag_node": {
                        "stage_id": "input_guardrail",
                        "stage_label": "Input Guardrail",
                        "node": "vision_ocr",
                        "status": "RUNNING",
                        "latency_ms": 0.0,
                        "model": "cloud-vision",
                        "profile": "STANDARD",
                        "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated": True},
                        "input_summary": {"attachments": len(decoded_items)},
                        "output_summary": {},
                    },
                }
                yield f"data: {json.dumps(vision_start_payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(_STREAM_STEP_DELAY)
                final_query, vision_analyses = await asyncio.to_thread(
                    _prepare_query_with_attachments,
                    query_text,
                    decoded_items,
                    agent,
                    session_id,
                )
                vision_ms = round((time.perf_counter() - vision_started) * 1000.0, 2)
                vision_all_failed = bool(vision_analyses) and all(
                    item.get("summary") == _VISION_ERROR_MARKER for item in vision_analyses
                )
                if vision_all_failed:
                    vision_desc = "Không thể đọc nội dung hình ảnh qua Cloud Vision"
                    vision_action_status = "ERROR"
                    vision_dag_status = "ERROR"
                else:
                    vision_desc = "Đã trích xuất thông tin từ hình ảnh đính kèm"
                    vision_action_status = "SUCCESS"
                    vision_dag_status = "PASS"
                vision_done_payload = {
                    "event_type": "node_progress",
                    "node": "vision_ocr",
                    "description": vision_desc,
                    "action": {"kind": "vision", "name": "cloud_vision_ocr", "status": vision_action_status},
                    "dag_node": {
                        "stage_id": "input_guardrail",
                        "stage_label": "Input Guardrail",
                        "node": "vision_ocr",
                        "status": vision_dag_status,
                        "latency_ms": vision_ms,
                        "model": "cloud-vision",
                        "profile": "STANDARD",
                        "tokens": {
                            "input_tokens": 128 * len(decoded_items),
                            "output_tokens": max(16, len(final_query) // 4),
                            "total_tokens": (128 * len(decoded_items)) + max(16, len(final_query) // 4),
                            "estimated": True,
                        },
                        "input_summary": {"attachments": len(decoded_items)},
                        "output_summary": {"extracted_images": len(vision_analyses)},
                    },
                }
                yield f"data: {json.dumps(vision_done_payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(_STREAM_STEP_DELAY)

            step_stream = agent.astream_chat(
                query=final_query,
                session_id=session_id,
                persona=style["persona"],
                tone=style["tone"],
                format_type=style["format_style"],
                custom_persona=style["custom_persona"]
            )
            async for chunk in _emit_graph_sse_events(
                step_stream,
                session_id,
                stream_started,
                "Deep agent stream",
                _API_MESSAGES["chat_internal_error"],
            ):
                yield chunk
        except Exception as err:
            print(f"[Deep agent stream] {type(err).__name__}")
            error_payload = {
                "event_type": "error",
                "message": _API_MESSAGES["chat_internal_error"]
            }
            if getattr(err, "trace_id", None):
                error_payload["trace_id"] = err.trace_id
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# -----------------------------------------------------------------------------
# 4. LIVE DEBUG & TRACING INSPECTOR API
# -----------------------------------------------------------------------------
def _get_active_tracer():
    agent = get_agent_pipeline()
    tracer_obj = getattr(agent, "tracer", None)
    if tracer_obj is None:
        tracer_obj = getattr(rag_pipeline, "tracer", None)
    return tracer_obj


def _build_debug_system_config(tracer_obj) -> dict:
    points_count = None
    vector_store = getattr(rag_pipeline, "vector_store", None)
    if vector_store is not None and hasattr(vector_store, "count"):
        try:
            points_count = vector_store.count()
        except Exception:
            points_count = None

    tracing_enabled = False
    langsmith_active = False
    langsmith_project = ""
    local_export_enabled = False
    if tracer_obj is not None:
        tracing_enabled = bool(getattr(tracer_obj, "enabled", False))
        langsmith_active = bool(getattr(tracer_obj, "is_langsmith_active", False))
        langsmith_project = str(getattr(tracer_obj, "project_name", "") or "")
        local_export_enabled = bool(getattr(tracer_obj, "local_export_enabled", False))

    return {
        "default_model": config.AGENT_LLM_MODEL,
        "agent_mode": str(_API_CFG["agent_mode"]),
        "embedding_model": config.EMBEDDING_MODEL_1024,
        "reranker_model": config.RERANKER_MODEL,
        "qdrant_collection": config.QDRANT_COLLECTION,
        "points_count": points_count,
        "tracing_enabled": tracing_enabled,
        "tracing_provider": "langsmith",
        "langsmith_active": langsmith_active,
        "langsmith_project": langsmith_project,
        "langsmith_url": str(_API_CFG["langsmith_url"]),
        "local_export_enabled": local_export_enabled,
    }


def _validate_trace_id(trace_id: str) -> str:
    if not isinstance(trace_id, str) or "\x00" in trace_id:
        raise HTTPException(status_code=400, detail="Invalid trace_id")
    clean = trace_id.strip()
    if not clean or len(clean) > 128:
        raise HTTPException(status_code=400, detail="Invalid trace_id")
    if "/" in clean or "\\" in clean or ".." in clean:
        raise HTTPException(status_code=400, detail="Invalid trace_id")
    return clean


@app.get("/api/debug/traces")
def list_debug_traces(
    session_id: str | None = None,
    scope: str = "session",
    limit: int = 50,
    include_active: bool = True,
):
    """Return trace summaries for the UI Debug & Tracing Inspector."""
    clean_scope = scope.strip().lower()
    if clean_scope not in ("session", "all"):
        raise HTTPException(status_code=400, detail="Invalid scope; expected 'session' or 'all'")
    bounded_limit = max(1, min(int(limit), 200))

    tracer_obj = _get_active_tracer()
    sys_cfg = _build_debug_system_config(tracer_obj)

    if tracer_obj is not None and hasattr(tracer_obj, "list_debug_traces"):
        payload = tracer_obj.list_debug_traces(
            session_id=session_id,
            scope=clean_scope,
            limit=bounded_limit,
            include_active=include_active,
        )
    else:
        payload = {
            "active_count": 0,
            "total_stored": 0,
            "total_in_memory": 0,
            "session_ref": None,
            "traces": [],
        }
    return {**payload, "system_config": sys_cfg}


@app.get("/api/debug/traces/{trace_id}")
def get_debug_trace_detail(trace_id: str):
    """Return full hierarchical spans, LLM/RAG/Tool/Agent state, and system config for a trace."""
    clean_id = _validate_trace_id(trace_id)
    tracer_obj = _get_active_tracer()
    if tracer_obj is None or not hasattr(tracer_obj, "get_debug_trace"):
        raise HTTPException(status_code=404, detail="Trace not found")
    detail = tracer_obj.get_debug_trace(clean_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {
        **detail,
        "system_config": _build_debug_system_config(tracer_obj),
    }


@app.delete("/api/debug/traces")
def clear_debug_traces(session_id: str | None = None, scope: str = "session"):
    """Clear stored in-memory debug traces for the current session or globally."""
    clean_scope = scope.strip().lower()
    if clean_scope not in ("session", "all"):
        raise HTTPException(status_code=400, detail="Invalid scope; expected 'session' or 'all'")
    tracer_obj = _get_active_tracer()
    cleared = 0
    if tracer_obj is not None and hasattr(tracer_obj, "clear_debug_traces"):
        cleared = int(tracer_obj.clear_debug_traces(session_id=session_id, scope=clean_scope))
    return {"cleared": cleared}


# -----------------------------------------------------------------------------
# 4B. MULTI-SESSION HISTORY MANAGEMENT (SYNCED WITH SQLITE CHECKPOINTER)
# -----------------------------------------------------------------------------
def _validate_session_id_param(session_id: str) -> str:
    if not isinstance(session_id, str) or "\x00" in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    clean = session_id.strip()
    if not clean or len(clean) > 128:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if "/" in clean or "\\" in clean or ".." in clean:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    return clean


@app.get("/api/sessions")
def list_chat_sessions(q: str = "", limit: int = 100):
    """List persisted chat sessions from SQLite checkpointer/session_meta."""
    bounded_limit = max(1, min(int(limit), 500))
    return get_agent_pipeline().list_sessions(
        search_query=q,
        limit=bounded_limit,
    )


@app.post("/api/sessions")
def create_chat_session(req: SessionCreateRequest):
    """Create or register a chat session in SQLite session_meta."""
    clean_id = None
    if req.session_id is not None and req.session_id.strip():
        clean_id = _validate_session_id_param(req.session_id)
    return get_agent_pipeline().create_session(session_id=clean_id, title=req.title)


@app.get("/api/sessions/{session_id}")
def get_chat_session_detail(session_id: str):
    """Restore full conversation history, citations, images, and DAG telemetry for a session."""
    clean_id = _validate_session_id_param(session_id)
    detail = get_agent_pipeline().get_session_detail(clean_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return detail


@app.patch("/api/sessions/{session_id}")
def update_chat_session(session_id: str, req: SessionUpdateRequest):
    """Rename a chat session."""
    clean_id = _validate_session_id_param(session_id)
    if req.title is None:
        raise HTTPException(status_code=400, detail="Nothing to update")
    updated = get_agent_pipeline().update_session_meta(
        clean_id,
        title=req.title,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return updated


@app.delete("/api/sessions/{session_id}")
def delete_chat_session(session_id: str):
    """Delete a chat session and its checkpoints/turns from SQLite."""
    clean_id = _validate_session_id_param(session_id)
    deleted = get_agent_pipeline().delete_session(clean_id)
    return {"deleted": bool(deleted), "session_id": clean_id}


# -----------------------------------------------------------------------------
# 5. NODE FOLDERS & FILES MANAGEMENT
# -----------------------------------------------------------------------------
@app.get("/api/nodes")
def get_nodes_info():
    """
    Return list of nodes and all files present in input/output directories.
    """
    folders = node_pipe.get_node_folders()
    result = {}
    for node_name, paths in folders.items():
        in_files = []
        if paths["input"].exists():
            for p in sorted(paths["input"].rglob("*")):
                if p.is_file():
                    rel_name = str(p.relative_to(paths["input"])).replace("\\", "/")
                    in_files.append({
                        "name": rel_name,
                        "size_bytes": p.stat().st_size,
                        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
                    })

        out_files = []
        if paths["output"].exists():
            for p in sorted(paths["output"].rglob("*")):
                if p.is_file():
                    rel_name = str(p.relative_to(paths["output"])).replace("\\", "/")
                    out_files.append({
                        "name": rel_name,
                        "size_bytes": p.stat().st_size,
                        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
                    })

        result[node_name] = {
            "node_name": node_name,
            "input_dir": str(paths["input"]),
            "output_dir": str(paths["output"]),
            "input_files": in_files,
            "output_files": out_files
        }
    return result


@app.get("/api/nodes/{node_name}/{io_type}/{file_name:path}")
def get_node_file_content(node_name: str, io_type: str, file_name: str):
    """
    Read detailed content of a file in the input or output folder of a node.
    """
    folders = node_pipe.get_node_folders()
    if node_name not in folders:
        raise HTTPException(status_code=404, detail=_API_MESSAGES["node_not_found"].format(node_name=node_name))
    if io_type not in ["input", "output"]:
        raise HTTPException(status_code=400, detail=_API_MESSAGES["invalid_io_type"])

    target_dir = folders[node_name][io_type]
    base_resolved = target_dir.resolve()
    target_file = (target_dir / file_name).resolve()

    if not target_file.is_relative_to(base_resolved):
        raise HTTPException(status_code=403, detail=_API_MESSAGES["invalid_file_path"])

    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail=_API_MESSAGES["file_not_found"].format(file_name=file_name))

    try:
        if target_file.suffix.lower() == ".json":
            with open(target_file, "r", encoding="utf-8", errors="replace") as fp:
                data = json.load(fp)
            return {"file_name": file_name, "is_json": True, "content": data}
        else:
            with open(target_file, "r", encoding="utf-8", errors="replace") as fp:
                raw_text = fp.read()
            return {"file_name": file_name, "is_json": False, "content": raw_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_API_MESSAGES["unable_to_read_file"].format(error=e))


_SHARED_ASSETS_DIR = (ROOT_DIR / "agent_banking" / "chunking" / "output" / "assets").resolve()
_PUBLICATION_KNOWLEDGE_DIR = (ROOT_DIR / "agent_banking" / "outputs" / "tuning_20261117" / "knowledge").resolve()
_ASSET_INDEX_CACHE = {}
_HEX_CHARS = "0123456789abcdef"


def _is_valid_asset_id(asset_id: str) -> bool:
    if not isinstance(asset_id, str) or len(asset_id) != 64:
        return False
    for ch in asset_id:
        if ch not in _HEX_CHARS:
            return False
    return True


def _find_asset_file(asset_id: str):
    if not _is_valid_asset_id(asset_id):
        return None
    filename = asset_id + ".png"

    if _SHARED_ASSETS_DIR.is_dir():
        candidate = (_SHARED_ASSETS_DIR / filename).resolve()
        if candidate.is_relative_to(_SHARED_ASSETS_DIR) and candidate.is_file():
            return candidate

    cached = _ASSET_INDEX_CACHE.get(asset_id)
    if cached is not None and cached.is_file():
        return cached

    if _PUBLICATION_KNOWLEDGE_DIR.is_dir():
        for pub_assets in _PUBLICATION_KNOWLEDGE_DIR.glob("*/assets"):
            if not pub_assets.is_dir():
                continue
            pub_base = pub_assets.resolve()
            candidate = (pub_base / filename).resolve()
            if candidate.is_relative_to(pub_base) and candidate.is_file():
                _ASSET_INDEX_CACHE[asset_id] = candidate
                return candidate
    return None


@app.get("/api/assets/{asset_id}")
def get_visual_asset(asset_id: str):
    """Serve a validated cropped or embedded visual asset PNG safely."""
    clean_id = asset_id.strip().lower()
    if not _is_valid_asset_id(clean_id):
        raise HTTPException(status_code=400, detail="Invalid visual asset identifier")
    target = _find_asset_file(clean_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Visual asset not found")
    return FileResponse(
        str(target),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"}
    )


_PDF_DOC_INDEX_CACHE = {}


def _allowed_pdf_dirs() -> list:
    """Return deduplicated resolved directories permitted for PDF preview."""
    raw_dirs = [
        Path(config.CHUNKING_INPUT_DIR),
        Path(config.CHUNKING_DATASET_DIR),
        Path(config.DATASET_DIR),
        ROOT_DIR / "dataset",
        ROOT_DIR / "dataset" / "A_Bank",
    ]
    resolved_dirs = []
    seen = set()
    for folder in raw_dirs:
        if not folder.is_dir():
            continue
        resolved = folder.resolve()
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            resolved_dirs.append(resolved)
    return resolved_dirs


def _normalize_pdf_source(source: str) -> str:
    """Validate and normalize a relative PDF source string without regex."""
    if not isinstance(source, str) or "\x00" in source:
        raise HTTPException(status_code=400, detail="Invalid PDF source parameter")
    clean = source.replace("\\", "/").strip()
    while clean.startswith("/"):
        clean = clean[1:]
    if not clean:
        raise HTTPException(status_code=400, detail="Empty PDF source parameter")
    if ":" in clean:
        raise HTTPException(status_code=403, detail="Invalid PDF document path")
    parts = clean.split("/")
    for seg in parts:
        if not seg:
            continue
        if seg == "." or seg == "..":
            raise HTTPException(status_code=403, detail="Invalid PDF document path")
    if not clean.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported")
    return clean


def _is_within_allowed_dirs(resolved: Path, allowed_dirs: list) -> bool:
    """Return True if resolved path resides inside any allowed base directory."""
    for allowed_dir in allowed_dirs:
        if resolved.is_relative_to(allowed_dir):
            return True
    return False


def _find_pdf_document(source: str):
    """Locate a PDF file inside allowed dataset/chunking directories."""
    clean = _normalize_pdf_source(source)
    allowed_dirs = _allowed_pdf_dirs()
    parts = [seg for seg in clean.split("/") if seg]

    for base_dir in allowed_dirs:
        candidate = base_dir
        for seg in parts:
            candidate = candidate / seg
        resolved = candidate.resolve()
        if not _is_within_allowed_dirs(resolved, allowed_dirs):
            raise HTTPException(status_code=403, detail="Invalid PDF document path")
        if resolved.is_file() and resolved.suffix.lower() == ".pdf":
            return resolved

    if "/" not in clean:
        cached = _PDF_DOC_INDEX_CACHE.get(clean)
        if cached is not None and cached.is_file():
            return cached
        target_lower = clean.lower()
        for base_dir in allowed_dirs:
            for item in base_dir.rglob("*.pdf"):
                if item.name == clean or item.name.lower() == target_lower:
                    resolved = item.resolve()
                    if _is_within_allowed_dirs(resolved, allowed_dirs) and resolved.is_file():
                        _PDF_DOC_INDEX_CACHE[clean] = resolved
                        return resolved
    return None


def _build_inline_content_disposition(filename: str) -> str:
    """Construct RFC 6266 inline Content-Disposition header for UTF-8 filenames."""
    ascii_chars = []
    for ch in filename:
        code = ord(ch)
        if 32 <= code < 127 and ch not in ('"', "\\"):
            ascii_chars.append(ch)
        else:
            ascii_chars.append("_")
    ascii_name = "".join(ascii_chars).strip()
    if not ascii_name:
        ascii_name = "document.pdf"
    encoded_name = quote(filename, safe="")
    return f'inline; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded_name}'


_PDF_OCR_CACHE_DB = (ROOT_DIR / "agent_banking" / "vector_db" / "pdf_ocr_cache.db").resolve()


def _schedule_pdf_prewarm(citations) -> None:
    """Pre-warm OCR bounding box cache in background for cited PDF pages."""
    if not isinstance(citations, list) or not citations:
        return
    has_pdf = False
    for cit in citations:
        if isinstance(cit, dict) and str(cit.get("source") or "").strip().lower().endswith(".pdf"):
            has_pdf = True
            break
    if not has_pdf:
        return
    snapshot = list(citations)
    allowed = _allowed_pdf_dirs()

    def _run():
        try:
            prewarm_pdf_citations(snapshot, allowed, _PDF_OCR_CACHE_DB)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def _resolve_safe_document(source: str, allowed_exts: tuple) -> Path:
    """Resolve a PDF, DOCX, or XLSX document safely or raise structured HTTP errors."""
    try:
        resolved = find_document_path(source, _allowed_pdf_dirs(), allowed_exts=allowed_exts)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Invalid document path") from None
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from None
    if resolved is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return resolved


@app.get("/api/documents/pdf")
def get_source_pdf(source: str):
    """Stream the original source PDF inline so browsers can jump directly to #page=N."""
    target = _find_pdf_document(source)
    if target is None:
        raise HTTPException(status_code=404, detail="PDF document not found")
    return FileResponse(
        str(target),
        media_type="application/pdf",
        headers={
            "Content-Disposition": _build_inline_content_disposition(target.name),
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/documents/file")
def get_source_document_file(source: str):
    """Stream or download an original PDF, DOCX, or XLSX source document safely."""
    target = _resolve_safe_document(source, (".pdf", ".docx", ".xlsx"))
    ext = target.suffix.lower()
    mime = "application/octet-stream"
    if ext == ".pdf":
        mime = "application/pdf"
    elif ext == ".docx":
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif ext == ".xlsx":
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return FileResponse(
        str(target),
        media_type=mime,
        headers={
            "Content-Disposition": _build_inline_content_disposition(target.name),
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/documents/docx/preview")
def preview_docx_document(source: str, chunk_id: str = ""):
    """Render a Word (.docx) document to structured HTML with 1:1 XPath anchors."""
    target = _resolve_safe_document(source, (".docx",))
    try:
        return render_docx_preview(
            target,
            source,
            chunk_id=chunk_id.strip(),
            knowledge_root=_PUBLICATION_KNOWLEDGE_DIR,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from None


@app.get("/api/documents/docx/media")
def get_docx_embedded_media(source: str, part: str):
    """Serve an embedded image from inside a Word (.docx) document safely."""
    target = _resolve_safe_document(source, (".docx",))
    try:
        raw_bytes, mime = read_docx_media(target, part)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Invalid DOCX media path") from None
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="DOCX media not found") from None
    return Response(
        content=raw_bytes,
        media_type=mime,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/documents/xlsx/preview")
def preview_xlsx_document(
    source: str,
    sheet: str = "",
    range: str = "",
    chunk_id: str = "",
):
    """Render an Excel (.xlsx) workbook with sheets, merged cells, and target range bounds."""
    target = _resolve_safe_document(source, (".xlsx",))
    try:
        return render_xlsx_preview(
            target,
            source,
            target_sheet=sheet,
            target_range=range,
            chunk_id=chunk_id.strip(),
            knowledge_root=_PUBLICATION_KNOWLEDGE_DIR,
        )
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Cannot parse XLSX: {type(err).__name__}") from None


@app.get("/api/documents/pdf/manifest")
def get_source_pdf_manifest(source: str):
    """Return total pages and dimensions for full in-UI PDF reading."""
    target = _resolve_safe_document(source, (".pdf",))
    return get_pdf_manifest(target, source)


@app.get("/api/documents/pdf/page")
def get_source_pdf_page_image(source: str, page: int = 1, dpi: int = 144):
    """Render a crisp PNG image of a single PDF page for the in-UI reader."""
    target = _resolve_safe_document(source, (".pdf",))
    try:
        png_bytes = render_pdf_page_png(target, page, dpi=dpi)
    except IndexError:
        raise HTTPException(status_code=404, detail="PDF page out of range") from None
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/documents/pdf/region")
def get_source_pdf_region_image(source: str, page: int, bbox: str):
    """Serve a high-resolution PNG of a cited PDF region."""
    try:
        if len(bbox) > 128:
            raise ValueError("Invalid PDF region")
        coordinates = tuple(float(value) for value in bbox.split(","))
        target = _resolve_safe_document(source, (".pdf",))
        png_bytes = render_pdf_region_png(target, page, coordinates)
    except IndexError:
        raise HTTPException(status_code=404, detail="PDF page out of range") from None
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PDF region") from None
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/documents/pdf/locate")
def locate_pdf_citation_passages(req: PdfLocateRequest):
    """Locate Level-2 exact bounding boxes for a citation inside its PDF pages."""
    target = _resolve_safe_document(req.source, (".pdf",))
    return locate_pdf_passages(
        target,
        _PDF_OCR_CACHE_DB,
        req.page_numbers,
        excerpt=req.excerpt,
        section_title=req.section_title,
        chunk_id=req.chunk_id,
        images=req.images,
        knowledge_root=_PUBLICATION_KNOWLEDGE_DIR,
    )


@app.get("/api/documents/pdf/search")
def search_source_pdf(source: str, q: str = ""):
    """Search inside a PDF across chunks, native text, and OCR cache."""
    target = _resolve_safe_document(source, (".pdf",))
    return search_pdf_document(
        target,
        source,
        q,
        _PDF_OCR_CACHE_DB,
        knowledge_root=_PUBLICATION_KNOWLEDGE_DIR,
    )


if __name__ == "__main__":
    host = str(_API_CFG["host"])
    port = int(_API_CFG["port"])
    print("=" * 70)
    print(_STARTUP_BANNER["title"])
    print(_STARTUP_BANNER["web_ui_prefix"].format(host=host, port=port))
    print(_STARTUP_BANNER["api_doc_prefix"].format(host=host, port=port))
    print("=" * 70)
    uvicorn.run(app, host=host, port=port)
