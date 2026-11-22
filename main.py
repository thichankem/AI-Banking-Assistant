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
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from agent_banking.core.config.settings import CoreConfig, get_core_messages
from agent_banking.core.cloud_rate_limit import acquire_cloud_slot
from agent_banking.core.settings import Config
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
_MAX_VISION_SNIPPET_CHARS = 400
_ALLOWED_IMAGE_MIMES = ("png", "jpeg", "jpg", "webp", "gif")
_DEFAULT_IMAGE_QUERY = "Hãy phân tích nội dung hình ảnh đính kèm và tư vấn hướng xử lý theo quy định của A-Bank."


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
    return CloudVisionClient()


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
        "Hãy đọc và trích xuất chính xác toàn bộ nội dung hiển thị trong hình ảnh này bằng tiếng Việt ngắn gọn: "
        "tiêu đề màn hình ứng dụng A-Bank Plus, mã lỗi (ví dụ ABP-...), số tiền, tên sản phẩm/thẻ, "
        "bảng biểu phí hoặc điều khoản quan trọng. Không bịa đặt thông tin không có trong ảnh."
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
                raw_text = client.ask(image_bytes, prompt=prompt, max_tokens=600, temperature=0.0)
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
                stashed_status = "error"
                span_meta["status"] = "error"
                span_meta["error_type"] = type(exc).__name__
                stashed_attrs.update(span_meta)
                return "[Không thể trích xuất chi tiết từ ảnh do lỗi kết nối Cloud Vision]"
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
    max_total = 1450
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
        final_response_text = ""
        audit_status_used = "PASS"
        citations = []
        sources = []
        images = []
        final_metadata = {}

        try:
            agent = get_agent_pipeline()
            final_query = query_text
            if decoded_items:
                vision_start_payload = {
                    "event_type": "node_progress",
                    "node": "vision_ocr",
                    "description": "Đang phân tích hình ảnh đính kèm qua Cloud Vision...",
                    "action": {"kind": "vision", "name": "cloud_vision_ocr", "status": "RUNNING"},
                }
                yield f"data: {json.dumps(vision_start_payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(_STREAM_STEP_DELAY)
                final_query, _ = await asyncio.to_thread(
                    _prepare_query_with_attachments,
                    query_text,
                    decoded_items,
                    agent,
                    session_id,
                )
                vision_done_payload = {
                    "event_type": "node_progress",
                    "node": "vision_ocr",
                    "description": "Đã trích xuất thông tin từ hình ảnh đính kèm",
                    "action": {"kind": "vision", "name": "cloud_vision_ocr", "status": "SUCCESS"},
                }
                yield f"data: {json.dumps(vision_done_payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(_STREAM_STEP_DELAY)

            async for step_event in agent.astream_chat(
                query=final_query,
                session_id=session_id,
                persona=style["persona"],
                tone=style["tone"],
                format_type=style["format_style"],
                custom_persona=style["custom_persona"]
            ):
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
                        final_metadata = {
                            key: node_state[key] for key in _FINAL_METADATA_KEYS
                        }
                        final_metadata.update({key: node_state[key] for key in _TRACE_METADATA_KEYS if key in node_state})

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
                    data_str = json.dumps(step_payload, ensure_ascii=False)
                    yield f"data: {data_str}\n\n"
                    await asyncio.sleep(_STREAM_STEP_DELAY)

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
            done_payload = {
                "event_type": "done",
                "status": "completed",
                "session_id": session_id,
                "audit_status": audit_status_used,
                "latency_ms": stream_latency_ms,
                "evaluation_metrics": None,
                "citations": citations,
                "sources": sources,
                "images": images,
                "full_answer": final_response_text,
                **final_metadata
            }
            done_str = json.dumps(done_payload, ensure_ascii=False)
            yield f"data: {done_str}\n\n"

        except Exception as err:
            print(f"[Deep agent stream] {type(err).__name__}")
            error_payload = {
                "event_type": "error",
                "message": _API_MESSAGES["chat_internal_error"]
            }
            if getattr(err, "trace_id", None):
                error_payload["trace_id"] = err.trace_id
            err_str = json.dumps(error_payload, ensure_ascii=False)
            yield f"data: {err_str}\n\n"

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


if __name__ == "__main__":
    host = str(_API_CFG["host"])
    port = int(_API_CFG["port"])
    print("=" * 70)
    print(_STARTUP_BANNER["title"])
    print(_STARTUP_BANNER["web_ui_prefix"].format(host=host, port=port))
    print(_STARTUP_BANNER["api_doc_prefix"].format(host=host, port=port))
    print("=" * 70)
    uvicorn.run(app, host=host, port=port)
