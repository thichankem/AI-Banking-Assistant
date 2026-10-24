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
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from rag_banking.config import Config
from rag_banking.node_pipeline import NodePipeline

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


# -----------------------------------------------------------------------------
# 1. SERVE FRONTEND GIAO DIỆN CHÍNH
# -----------------------------------------------------------------------------
@app.get("/")
def serve_index():
    index_file = ROOT_DIR / "static" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "AI Banking Assistant Server đang chạy thành công!"}


# -----------------------------------------------------------------------------
# 2. CẤU HÌNH HỆ THỐNG
# -----------------------------------------------------------------------------
@app.get("/api/config")
def get_config():
    total_points = node_pipe.pipeline.vector_store.count()
    return {
        "default_model": node_pipe.config.LLM_MODEL,
        "embedding_model": node_pipe.config.EMBEDDING_MODEL_1024,
        "reranker_model": node_pipe.config.RERANKER_MODEL,
        "qdrant_collection": node_pipe.config.QDRANT_COLLECTION,
        "points_count": total_points,
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
# 3. TRUY VẤN HỎI ĐÁP & THEO DÕI LIÊN HOÀN 7 NODES
# -----------------------------------------------------------------------------
@app.post("/api/chat")
def handle_chat(req: ChatRequest):
    query_text = req.message.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Nội dung câu hỏi không được để trống.")

    try:
        result = node_pipe.run_single_query_with_node_trace(query_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý đường ống RAG: {e}")


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
