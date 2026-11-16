#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Banking Assistant - Web Application Server
======================================================================
Launches web API service and static frontend connected to all RAG and Agentic nodes:

1. ReAct chat: /api/chat, /api/agent/chat, /api/agent/stream
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
from pydantic import BaseModel
import uvicorn

from agent_banking.core.settings import Config
from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.node_pipeline import NodePipeline
from agent_banking.agent_pipeline import BankingAgentPipeline
from agent_banking.langgraph_agent.config.response_profiles import (
    PERSONA_PROFILES,
    TONE_PROFILES,
    FORMAT_PROFILES
)

# Initialize FastAPI application
app = FastAPI(
    title="AI Banking Assistant",
    description="Enterprise Digital Banking Assistant & Pipeline Orchestration API",
    version="3.0.0"
)

# Mount static directory for frontend assets
static_dir = ROOT_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize Config & BankingRAGPipeline
config = Config()
use_memory_env = False
if os.environ.get("USE_MEMORY_STORE", "0") == "1":
    use_memory_env = True

try:
    rag_pipeline = BankingRAGPipeline(config=config, use_memory=use_memory_env)
except RuntimeError as lock_err:
    if "already accessed by another instance" in str(lock_err):
        print("[Lock Warning] Vector store locked by another process -> Falling back to In-Memory store.")
        rag_pipeline = BankingRAGPipeline(config=config, use_memory=True)
    else:
        raise

# Manage indexing and diagnostic folders.
node_pipe = NodePipeline()


# Pydantic Schemas
class ChatRequest(BaseModel):
    message: str
    persona: str | None = "advisor"
    tone: str | None = "formal"
    format_style: str | None = "standard"
    custom_persona: str | None = ""
    session_id: str | None = None


# -----------------------------------------------------------------------------
# 1. SERVE FRONTEND USER INTERFACE
# -----------------------------------------------------------------------------
@app.get("/")
def serve_index():
    index_file = ROOT_DIR / "static" / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "AI Banking Assistant Server is running."}


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
            f_name = v.split(":")[0].replace("Format ", "").replace("Định dạng ", "")
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
        "embedding_model": config.EMBEDDING_MODEL_1024,
        "reranker_model": config.RERANKER_MODEL,
        "qdrant_collection": config.QDRANT_COLLECTION,
        "points_count": total_points,
        "tracing_enabled": rag_pipeline.tracer.is_phoenix_active,
        "phoenix_url": "http://127.0.0.1:6006",
        "nodes": [
            {"id": "chunking", "name": "Node 1: Chunking"},
            {"id": "embedding", "name": "Node 2: Embedding"},
            {"id": "vector_db", "name": "Node 3: Vector DB & BM25"},
            {"id": "query_parser", "name": "Node 4: Query Parser"},
            {"id": "retrieval", "name": "Node 5: Hybrid Retrieval"},
            {"id": "reranking", "name": "Node 6: Cross-Encoder Reranker"}
        ]
    }


def _resolve_chat_style(req: ChatRequest) -> dict:
    """Normalize persona, tone, format_style, and custom_persona fields from ChatRequest."""
    persona_val = "advisor"
    if req.persona:
        persona_val = req.persona

    tone_val = "formal"
    if req.tone:
        tone_val = req.tone

    format_style_val = "standard"
    if req.format_style:
        format_style_val = req.format_style

    custom_persona_val = ""
    if req.custom_persona:
        custom_persona_val = req.custom_persona

    return {
        "persona": persona_val,
        "tone": tone_val,
        "format_style": format_style_val,
        "custom_persona": custom_persona_val
    }


# -----------------------------------------------------------------------------
# 3. BANKING REACT CHAT
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
    Execute a ReAct turn with session memory.
    """
    query_text = req.message.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangGraph Agent execution error: {e}")


@app.post("/api/agent/stream")
async def stream_agent_chat(req: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint for real-time node progress and token streaming.
    """
    query_text = req.message.strip()
    if query_text == "":
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    session_id = req.session_id or uuid4().hex

    style = _resolve_chat_style(req)

    async def event_generator():
        final_response_text = ""
        audit_status_used = "PASS"
        citations = []
        sources = []

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
                    if "audit_status" in node_state:
                        audit_status_used = str(node_state["audit_status"])

                    descriptions = {
                        "input_guardrail": "Đã kiểm tra yêu cầu",
                        "react_decide": "Đã chọn bước tiếp theo",
                        "audit_generation": "Đã đối soát trích dẫn",
                        "output_guardrail": "Đã kiểm tra phản hồi"
                    }
                    node_description = descriptions.get(node_name, node_name)
                    action = node_state.get("last_action")
                    if action:
                        node_description = f"{action['kind']} · {action['name']} · {action['status']}"
                    if node_name == "output_guardrail":
                        final_response_text = node_state["final_answer"]
                        citations = node_state["citations"]
                        sources = node_state["sources"]

                    step_payload = {
                        "event_type": "node_progress",
                        "node": node_name,
                        "description": node_description,
                        "action": action
                    }
                    data_str = json.dumps(step_payload, ensure_ascii=False)
                    yield f"data: {data_str}\n\n"
                    await asyncio.sleep(0.02)

            words = final_response_text.split(" ")
            for word in words:
                token_payload = {
                    "event_type": "token",
                    "content": word + " "
                }
                token_str = json.dumps(token_payload, ensure_ascii=False)
                yield f"data: {token_str}\n\n"
                await asyncio.sleep(0.015)

            done_payload = {
                "event_type": "done",
                "session_id": session_id,
                "audit_status": audit_status_used,
                "evaluation_metrics": None,
                "citations": citations,
                "sources": sources,
                "full_answer": final_response_text
            }
            done_str = json.dumps(done_payload, ensure_ascii=False)
            yield f"data: {done_str}\n\n"

        except Exception as err:
            error_payload = {
                "event_type": "error",
                "message": f"Service connection error: {err}"
            }
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
        raise HTTPException(status_code=404, detail=f"Node not found: {node_name}")
    if io_type not in ["input", "output"]:
        raise HTTPException(status_code=400, detail="Directory type must be 'input' or 'output'")

    target_dir = folders[node_name][io_type]
    base_resolved = target_dir.resolve()
    target_file = (target_dir / file_name).resolve()

    if not target_file.is_relative_to(base_resolved):
        raise HTTPException(status_code=403, detail="Access denied: invalid file path")

    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_name}")

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
        raise HTTPException(status_code=500, detail=f"Unable to read file: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("  AI BANKING ASSISTANT SERVER")
    print("  * Web UI : http://127.0.0.1:8000")
    print("  * API Doc: http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000)
