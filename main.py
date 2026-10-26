#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Banking Assistant - Web Application Server
======================================================================
Khởi chạy dịch vụ web API & kết nối trực tiếp giao diện tĩnh (Static Frontend)
với toàn bộ các Node (Chunking, Embedding, Vector DB, Query Parser, Retrieval, Reranking, Generator):

1. Endpoint Chat: /api/chat (Xử lý toàn trình qua 7 nodes, trả về câu trả lời và vết chi tiết của từng node)
2. Endpoint Cấu hình: /api/config
3. Endpoint Quản lý Node Folders: /api/nodes, /api/nodes/{node_name}/{io_type}/{file_name}
4. Endpoint Kích hoạt Pipeline từ Folder: /api/pipeline/run-folders
5. Phục vụ Static Frontend: http://127.0.0.1:8000

NGUYÊN TẮC:
- 0% REGEX LAW: Tuyệt đối không dùng regex.
- ZERO TERNARY: Ưu tiên if/else tường minh.
- MINH BẠCH INPUT/OUTPUT: Từng node đều lưu vết vào thư mục input/output tương ứng.
"""

from __future__ import annotations

import os
import sys
import time
import json
import asyncio
from pathlib import Path

# Đảm bảo UTF-8 an toàn trên Windows console
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

from agent_banking.config import Config
from agent_banking.node_pipeline import NodePipeline
from agent_banking.generator.rag_generator import (
    PERSONA_PROFILES,
    TONE_PROFILES,
    FORMAT_PROFILES
)

# Khởi tạo FastAPI app
app = FastAPI(
    title="AI Banking Assistant",
    description="Hệ thống Trợ lý Ngân hàng Số & Pipeline Quản lý Node Dữ liệu (Folder-to-Folder)",
    version="3.0.0"
)

# Gắn thư mục static để phục vụ index.html, style.css, app.js
static_dir = ROOT_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Khởi tạo NodePipeline Singleton với cơ chế Fallback In-Memory khi tệp bị khóa
config = Config()
use_memory_env = os.environ.get("USE_MEMORY_STORE", "0") == "1"
try:
    node_pipe = NodePipeline(config=config, use_memory=use_memory_env)
except RuntimeError as lock_err:
    if "already accessed by another instance" in str(lock_err):
        print("⚠️ [Qdrant Lock] Thư mục Qdrant đang được giữ bởi tiến trình khác -> Chuyển sang In-Memory mode an toàn!")
        node_pipe = NodePipeline(config=config, use_memory=True)
    else:
        raise


# Pydantic Schemas
class ChatRequest(BaseModel):
    message: str
    persona: str | None = "advisor"
    tone: str | None = "formal"
    format_style: str | None = "standard"
    custom_persona: str | None = ""
    session_id: str | None = "default_session"


# -----------------------------------------------------------------------------
# 1. SERVE FRONTEND GIAO DIỆN CHÍNH
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
    return {"message": "AI Banking Assistant Server đang chạy thành công!"}


# -----------------------------------------------------------------------------
# 2. CẤU HÌNH HỆ THỐNG & PERSONA PROFILES
# -----------------------------------------------------------------------------
@app.get("/api/persona/profiles")
def get_persona_profiles():
    """
    Trả về danh mục cấu hình Persona, Tông giọng và Định dạng phản hồi cho giao diện Frontend.
    """
    return {
        "personas": [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"]
            }
            for k, v in PERSONA_PROFILES.items()
        ],
        "tones": [
            {
                "id": k,
                "name": v.split(" (")[0] if " (" in v else v,
                "description": v
            }
            for k, v in TONE_PROFILES.items()
        ],
        "formats": [
            {
                "id": k,
                "name": v.split(":")[0].replace("Định dạng ", "") if ":" in v else v,
                "description": v
            }
            for k, v in FORMAT_PROFILES.items()
        ]
    }


@app.get("/api/config")
def get_config():
    total_points = node_pipe.pipeline.vector_store.count()
    return {
        "default_model": node_pipe.config.LLM_MODEL,
        "embedding_model": node_pipe.config.EMBEDDING_MODEL_1024,
        "reranker_model": node_pipe.config.RERANKER_MODEL,
        "qdrant_collection": node_pipe.config.QDRANT_COLLECTION,
        "points_count": total_points,
        "tracing_enabled": node_pipe.pipeline.tracer.is_phoenix_active,
        "phoenix_url": "http://127.0.0.1:6006",
        "nodes": [
            {"id": "chunking", "name": "Node 1: Chunking"},
            {"id": "embedding", "name": "Node 2: Embedding"},
            {"id": "vector_db", "name": "Node 3: Vector DB & BM25"},
            {"id": "query_parser", "name": "Node 4: Query Parser"},
            {"id": "retrieval", "name": "Node 5: Hybrid Retrieval"},
            {"id": "reranking", "name": "Node 6: Cross-Encoder Reranker"},
            {"id": "generator", "name": "Node 7: Generator & Guardrails"}
        ]
    }


# -----------------------------------------------------------------------------
# 3. TRUY VẤN HỎI ĐÁP & THEO DÕI LIÊN HOÀN 7 NODES (KÈM DYNAMIC PERSONA & TRACING)
# -----------------------------------------------------------------------------
@app.post("/api/chat")
def handle_chat(req: ChatRequest):
    query_text = req.message.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Nội dung câu hỏi không được để trống.")

    persona_config = {
        "persona": req.persona or "advisor",
        "tone": req.tone or "formal",
        "format_style": req.format_style or "standard",
        "custom_persona": req.custom_persona or ""
    }

    try:
        result = node_pipe.run_single_query_with_node_trace(query_text, persona_config=persona_config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý đường ống RAG: {e}")


# -----------------------------------------------------------------------------
# 3.1 LANGGRAPH AI AGENT (SELF-REFLECTIVE RAG & BANKING ACTION TOOLS)
# -----------------------------------------------------------------------------
_agent_pipeline_instance = None

def get_agent_pipeline():
    global _agent_pipeline_instance
    if _agent_pipeline_instance is None:
        from agent_banking.agent_pipeline import BankingAgentPipeline
        _agent_pipeline_instance = BankingAgentPipeline(config=config, use_memory=node_pipe.use_memory)
    return _agent_pipeline_instance


@app.post("/api/agent/chat")
def handle_agent_chat(req: ChatRequest):
    """
    Endpoint LangGraph Agentic RAG: Phân loại ý định, Self-Reflection, gọi Banking Tools và quản lý Memory.
    """
    query_text = req.message.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Nội dung câu hỏi không được để trống.")

    session_id = req.session_id or "default_session"
    try:
        agent = get_agent_pipeline()
        result = agent.chat(
            query=query_text,
            session_id=session_id,
            persona=req.persona or "advisor",
            tone=req.tone or "formal",
            format_type=req.format_style or "standard"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý LangGraph Agent: {e}")


@app.post("/api/agent/stream")
async def stream_agent_chat(req: ChatRequest):
    """
    Endpoint phát Server-Sent Events (SSE) thời gian thực cho từng Node trong LangGraph v4.0
    và stream nội dung trả lời cuối cùng ra giao diện Web.
    """
    query_text = req.message.strip()
    if query_text == "":
        raise HTTPException(status_code=400, detail="Nội dung câu hỏi không được để trống.")

    session_id = "default_session"
    if req.session_id is not None:
        if req.session_id != "":
            session_id = req.session_id

    async def event_generator():
        config_thread = {
            "configurable": {
                "thread_id": session_id
            }
        }
        initial_input = {
            "user_query": query_text,
            "query": query_text,
            "original_query": query_text,
            "session_id": session_id
        }

        final_response_text = ""
        sub_agent_used = "GENERAL_ADVISOR"
        audit_status_used = "PASS"
        last_retrieved_chunks = []

        try:
            agent = get_agent_pipeline()
            for step_event in agent.graph.stream(initial_input, config=config_thread):
                for node_name, node_state in step_event.items():
                    if "sub_agent_role" in node_state:
                        sub_agent_used = str(node_state["sub_agent_role"])

                    if "audit_status" in node_state:
                        audit_status_used = str(node_state["audit_status"])

                    if "retrieved_chunks" in node_state:
                        last_retrieved_chunks = node_state["retrieved_chunks"]

                    if "final_answer" in node_state:
                        if node_state["final_answer"]:
                            final_response_text = str(node_state["final_answer"])
                    elif "draft_answer" in node_state:
                        if node_state["draft_answer"]:
                            if final_response_text == "":
                                final_response_text = str(node_state["draft_answer"])

                    node_description = node_name
                    if node_name == "input_guardrail":
                        node_description = "Đang kiểm duyệt an toàn & làm sạch trạng thái..."
                    elif node_name == "router":
                        node_description = f"Multi-Agent Supervisor: Định tuyến tới {sub_agent_used}..."
                    elif node_name == "retrieve":
                        node_description = "Đang truy xuất tri thức quy chế từ Qdrant & BM25..."
                    elif node_name == "grade_documents":
                        node_description = "Đang thẩm định độ liên quan của tài liệu trích xuất..."
                    elif node_name == "rewrite_query":
                        node_description = "Tài liệu chưa tối ưu, đang tự động viết lại câu hỏi..."
                    elif node_name == "banking_tool":
                        node_description = "Đang thực thi công cụ nghiệp vụ ngân hàng..."
                    elif node_name == "generate":
                        node_description = "Đang sinh câu trả lời căn cứ điều khoản Vietcombank..."
                    elif node_name == "audit_generation":
                        node_description = "Đang kiểm toán trích dẫn [Điều X, Khoản Y] & chống ảo giác..."
                    elif node_name == "output_guardrail":
                        node_description = "Kiểm duyệt an toàn đầu ra & hoàn tất phản hồi..."

                    step_payload = {
                        "event_type": "node_progress",
                        "node": node_name,
                        "description": node_description,
                        "sub_agent": sub_agent_used
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

            eval_metrics = {
                "context_precision": 1.0,
                "faithfulness": 1.0,
                "answer_relevance": 1.0,
                "overall_score": 1.0
            }
            if hasattr(agent, "evaluator"):
                if agent.evaluator is not None and len(last_retrieved_chunks) > 0:
                    cleaned_q = query_text.lower()
                    punctuation_chars = [".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "\"", "'", "\n", "\t", "*", "-", "_", "/", "\\", "|"]
                    for ch in punctuation_chars:
                        cleaned_q = cleaned_q.replace(ch, " ")
                    raw_tokens = cleaned_q.split(" ")
                    stopwords = {"cho", "tôi", "hỏi", "được", "không", "như", "thế", "nào", "vietcombank", "với", "các", "những", "có", "là", "và", "của", "tại"}
                    kws = []
                    for t in raw_tokens:
                        stripped_t = t.strip()
                        if len(stripped_t) > 2 and stripped_t not in stopwords:
                            kws.append(stripped_t)
                    if len(kws) == 0:
                        kws.append(query_text.lower())
                    eval_metrics = agent.evaluator.evaluate_single_case(
                        question=query_text,
                        retrieved_chunks=last_retrieved_chunks,
                        generated_answer=final_response_text,
                        expected_keywords=kws
                    )

            done_payload = {
                "event_type": "done",
                "session_id": session_id,
                "sub_agent": sub_agent_used,
                "audit_status": audit_status_used,
                "evaluation_metrics": eval_metrics,
                "full_answer": final_response_text
            }
            done_str = json.dumps(done_payload, ensure_ascii=False)
            yield f"data: {done_str}\n\n"

        except Exception as err:
            error_payload = {
                "event_type": "error",
                "message": f"Hệ thống đang bảo trì kết nối: {err}"
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
# 4. QUẢN LÝ FOLDERS & TẬP TIN CỦA TỪNG NODE
# -----------------------------------------------------------------------------
@app.get("/api/nodes")
def get_nodes_info():
    """
    Trả về danh sách 7 node và toàn bộ các tệp tin hiện diện trong input/output folder của mỗi node.
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
    Đọc nội dung chi tiết của một tệp tin trong thư mục input hoặc output của node.
    Hỗ trợ cả đường dẫn tệp lồng nhau trong thư mục con.
    """
    folders = node_pipe.get_node_folders()
    if node_name not in folders:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy node: {node_name}")
    if io_type not in ["input", "output"]:
        raise HTTPException(status_code=400, detail="Loại thư mục chỉ chấp nhận 'input' hoặc 'output'")

    target_dir = folders[node_name][io_type]
    target_file = target_dir / file_name

    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy tệp: {file_name}")

    try:
        # Nếu là file json, parse và trả về định dạng json
        if target_file.suffix.lower() == ".json":
            with open(target_file, "r", encoding="utf-8", errors="replace") as fp:
                data = json.load(fp)
            return {"file_name": file_name, "is_json": True, "content": data}
        else:
            with open(target_file, "r", encoding="utf-8", errors="replace") as fp:
                raw_text = fp.read()
            return {"file_name": file_name, "is_json": False, "content": raw_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể đọc tệp: {e}")


# -----------------------------------------------------------------------------
# 5. KÍCH HOẠT CHẠY PIPELINE TỪ FOLDER INPUT CỦA CÁC NODE
# -----------------------------------------------------------------------------
@app.post("/api/pipeline/run-folders")
def run_folders_pipeline():
    """
    Kích hoạt chạy liên hoàn toàn bộ 7 nodes từ các thư mục input/output nội bộ.
    """
    try:
        summary = node_pipe.run_node_directories_pipeline()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi thực thi pipeline từ folder: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("  🚀 KHỞI ĐỘNG AI BANKING ASSISTANT SERVER")
    print("  * Web UI : http://127.0.0.1:8000")
    print("  * API Doc: http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000)
