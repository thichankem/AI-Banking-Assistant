# -*- coding: utf-8 -*-
"""
AI Banking Agent Pipeline (LangGraph Production Entrypoint)
======================================================================
Điểm khởi chạy chính (Entrypoint) cho hệ thống Agentic RAG Ngân hàng Vietcombank:
- Tích hợp toàn diện LangGraph StateGraph (Self-Reflective RAG + Tool Calling).
- Duy trì bộ nhớ phiên hội thoại (Session Memory với Thread ID).
- Tương thích 100% với giao diện người dùng và API Web hiện tại.
- Bảo toàn toàn bộ cơ chế An toàn (Guardrails) và Giám sát AI Observability (Arize Phoenix).
"""

from __future__ import annotations
import sys
import time
from typing import Any
from pathlib import Path

# Đảm bảo UTF-8 an toàn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.config import Config
from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.agents.graph import build_banking_agent_graph


class BankingAgentPipeline:
    """
    Bộ điều phối toàn trình AI Banking Agent ứng dụng LangGraph.
    """

    def __init__(self, config: Config | None = None, use_memory: bool = False):
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.use_memory = use_memory

        print("=" * 70)
        print("  🚀 KHỞI TẠO LANGGRAPH AI BANKING AGENT (VIETCOMBANK PRODUCTION)")
        print("=" * 70)

        # 1. Khởi tạo RAG Pipeline nền tảng (Dual Index Qdrant + BM25)
        self.rag_pipeline = BankingRAGPipeline(config=self.config, use_memory=self.use_memory)

        # 2. Xây dựng và biên dịch Đồ thị LangGraph StateGraph
        print("[Agentic] Đang kết nối các Nodes và biên dịch LangGraph StateGraph...")
        self.graph = build_banking_agent_graph(
            pipeline_instance=self.rag_pipeline,
            config=self.config
        )
        print("✅ [LangGraph] Đồ thị AI Agent đã sẵn sàng phục vụ (Self-Reflection & Tools Activated)!\n")

    def chat(
        self,
        query: str,
        session_id: str = "default_session",
        persona: str = "standard",
        tone: str = "professional",
        format_type: str = "standard"
    ) -> dict[str, Any]:
        """
        Thực thi phiên hội thoại với LangGraph Agent kèm Session Memory (Thread ID).
        """
        start_time = time.perf_counter()
        clean_query = query.strip()
        
        if not clean_query:
            return {
                "answer": "Kính chào Quý khách! Vui lòng nhập nội dung câu hỏi hoặc yêu cầu cần hỗ trợ.",
                "citations": [],
                "sources": [],
                "intent": "CHIT_CHAT",
                "retry_count": 0,
                "session_id": session_id,
                "latency_ms": 0.0,
                "nodes": {}
            }

        # Thiết lập Thread ID cho Checkpointer (Quản lý bộ nhớ phiên)
        thread_config = {"configurable": {"thread_id": session_id}}

        # Trạng thái khởi tạo đưa vào StateGraph (Làm mới các trường của turn mới)
        initial_state = {
            "query": clean_query,
            "original_query": clean_query,
            "session_id": session_id,
            "intent": "ROUTING",
            "input_guard": {"is_safe": True, "sanitized_query": clean_query},
            "output_guard": {},
            "retry_count": 0,
            "max_retries": self.config.MAX_RETRIES,
            "retrieved_docs": [],
            "tool_calls": [],
            "tool_outputs": [],
            "citations": [],
            "sources": [],
            "draft_answer": "",
            "final_answer": ""
        }

        # Kích hoạt thực thi qua LangGraph
        try:
            final_state = self.graph.invoke(initial_state, config=thread_config)
        except Exception as e:
            print(f"❌ [LangGraph Execution Error]: {e}")
            # Dự phòng an toàn nếu gặp sự cố runtime
            return {
                "answer": (
                    "Hệ thống đang bảo trì một số tính năng nâng cao. "
                    "Quý khách vui lòng liên hệ hotline Vietcombank 1900 54 54 13 để được trợ giúp tức thì."
                ),
                "citations": [],
                "sources": [],
                "intent": "ERROR",
                "retry_count": 0,
                "session_id": session_id,
                "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
                "nodes": {}
            }

        total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Trích xuất dữ liệu trả về cho Frontend UI Inspector
        answer = final_state.get("final_answer", final_state.get("draft_answer", ""))
        citations = final_state.get("citations", [])
        sources = final_state.get("sources", [])
        intent = final_state.get("intent", "RAG_SEARCH")
        retries = final_state.get("retry_count", 0)
        tool_outputs = final_state.get("tool_outputs", [])
        docs = final_state.get("retrieved_docs", [])

        # Đóng gói Node Inspector Data chi tiết
        nodes_detail = {
            "00_guardrail": {
                "name": "Node 0: Banking Input Guardrail",
                "is_safe": final_state.get("input_guard", {}).get("is_safe", True),
                "status": "passed" if final_state.get("input_guard", {}).get("is_safe", True) else "blocked"
            },
            "01_router": {
                "name": "Node 1: Intent Router (LangGraph)",
                "detected_intent": intent,
                "target_tool": final_state.get("target_tool")
            },
            "02_retrieval": {
                "name": "Node 2: Knowledge Retrieval (RAG)",
                "total_documents_found": len(docs),
                "document_quality": final_state.get("document_quality", "N/A"),
                "retry_count": retries
            },
            "03_tools": {
                "name": "Node 3: Banking Action Tools",
                "executed_tools": [to.get("tool") for to in tool_outputs]
            },
            "04_output_guard": {
                "name": "Node 4: Output Guardrail & Verification",
                "is_safe": final_state.get("output_guard", {}).get("is_safe", True),
                "flags": final_state.get("output_guard", {}).get("flags", [])
            }
        }

        return {
            "answer": answer,
            "citations": citations,
            "sources": sources,
            "intent": intent,
            "retry_count": retries,
            "tool_calls": tool_outputs,
            "session_id": session_id,
            "latency_ms": total_latency_ms,
            "nodes": nodes_detail
        }

    def query(self, query: str) -> dict[str, Any]:
        """
        Hàm giao tiếp tương thích ngược 100% với pipeline.py cũ.
        """
        return self.chat(query=query)


if __name__ == "__main__":
    # Thử nghiệm nhanh LangGraph Agent Pipeline
    agent = BankingAgentPipeline(use_memory=True)
    
    test_queries = [
        "Xin chào, bạn có thể giúp gì cho tôi?",
        "Lãi suất tiết kiệm kỳ hạn 12 tháng hôm nay bao nhiêu?",
        "Tỷ giá USD tại Vietcombank hiện tại là bao nhiêu?",
        "Hạn mức giao dịch mở tài khoản eKYC tối đa một ngày là bao nhiêu?"
    ]

    for q in test_queries:
        print(f"\n👉 Câu hỏi: {q}")
        res = agent.chat(q)
        print(f"🎯 Ý định: {res['intent']} (Retry: {res['retry_count']})")
        print(f"💬 Trả lời: {res['answer'][:150]}...")
        if res["citations"]:
            print(f"📚 Trích dẫn: {[c['label'] for c in res['citations']]}")
