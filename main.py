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
- NO TYPING LIBRARY: Use Python native types only.
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path
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
from agent_banking.core.settings import Config
from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.node_pipeline import NodePipeline
from agent_banking.agent_pipeline import BankingAgentPipeline, InteractionError
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


# Pydantic Schemas
class ChatRequest(BaseModel):
    message: str
    persona: str | None = _DEFAULT_PERSONA
    tone: str | None = _DEFAULT_TONE
    format_style: str | None = _DEFAULT_FORMAT_TYPE
    custom_persona: str | None = _DEFAULT_CUSTOM_PERSONA
    session_id: str | None = None


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
    choice_id: str = Field(
        min_length=int(_API_CFG["choice_id_min_length"]),
        max_length=int(_API_CFG["choice_id_max_length"])
    )
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
        "tracing_enabled": rag_pipeline.tracer.is_phoenix_active,
        "phoenix_url": str(_API_CFG["phoenix_url"]),
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


@app.post("/api/chat")
@app.post("/api/agent/chat")
def handle_agent_chat(req: ChatRequest):
    """
    Execute a deep agent turn with persistent session memory.
    """
    query_text = req.message.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail=_API_MESSAGES["empty_query_error"])

    session_id = req.session_id or uuid4().hex

    style = _resolve_chat_style(req)

    try:
        agent = get_agent_pipeline()
        result = agent.chat(
            query=query_text,
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
        return get_agent_pipeline().resume(req.session_id, req.interaction_id, req.choice_id, req.text)
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
    if query_text == "":
        raise HTTPException(status_code=400, detail=_API_MESSAGES["empty_query_error"])

    session_id = req.session_id or uuid4().hex

    style = _resolve_chat_style(req)

    async def event_generator():
        final_response_text = ""
        audit_status_used = "PASS"
        citations = []
        sources = []
        final_metadata = {}

        try:
            agent = get_agent_pipeline()
            async for step_event in agent.astream_chat(
                query=query_text,
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

            done_payload = {
                "event_type": "done",
                "status": "completed",
                "session_id": session_id,
                "audit_status": audit_status_used,
                "evaluation_metrics": None,
                "citations": citations,
                "sources": sources,
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
# 4. NODE FOLDERS & FILES MANAGEMENT
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


if __name__ == "__main__":
    host = str(_API_CFG["host"])
    port = int(_API_CFG["port"])
    print("=" * 70)
    print(_STARTUP_BANNER["title"])
    print(_STARTUP_BANNER["web_ui_prefix"].format(host=host, port=port))
    print(_STARTUP_BANNER["api_doc_prefix"].format(host=host, port=port))
    print("=" * 70)
    uvicorn.run(app, host=host, port=port)
