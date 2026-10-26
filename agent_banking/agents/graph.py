# -*- coding: utf-8 -*-
"""
LangGraph StateGraph Builder for Banking Agent v4.0
======================================================================
Xây dựng và biên dịch đồ thị luân chuyển trạng thái (StateGraph) với:
- 9 Nodes toàn trình (bao gồm Node 7: audit_generation).
- Multi-Agent Supervisor Router (Chỉ định Sub-Agent chuyên trách).
- Persistent SQLite Checkpointer (SqliteSaver) lưu trữ lịch sử phiên bền vững.
- Chống lặp vô tận (Infinite Loop Protection) tại Grader (max 2 retries) và Audit (max 1 retry).
- 0% REGEX & ZERO TERNARY tuân thủ 100%.
"""

from __future__ import annotations
import os
import sqlite3
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent_banking.config import Config
from agent_banking.agents.state import AgentState
from agent_banking.agents.nodes import BankingAgentNodes
from agent_banking.agents.edges import (
    route_after_input_guardrail,
    route_after_router,
    route_after_grader,
    route_after_audit
)


def create_persistent_checkpointer(db_path: str = "agent_banking/tracing/agent_memory.db"):
    """
    Khởi tạo SqliteSaver để lưu trữ bền vững lịch sử chat qua các lần khởi động lại Server.
    Nếu môi trường thiếu thư viện sqlite checkpointer, tự động chuyển về MemorySaver an toàn.
    """
    parent_dir = os.path.dirname(db_path)
    if parent_dir != "":
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        return checkpointer
    except Exception as e:
        print(f"[Checkpointer] Chuyển về MemorySaver dự phòng (Lý do: {e})")
        fallback_memory = MemorySaver()
        return fallback_memory


def build_banking_agent_graph(
    pipeline_instance=None,
    config: Optional[Config] = None,
    checkpointer: Optional[Any] = None,
    sqlite_db_path: str = "agent_banking/tracing/agent_memory.db"
):
    """
    Xây dựng và biên dịch đồ thị LangGraph Agentic RAG phiên bản 4.0.
    """
    if config is not None:
        cfg = config
    else:
        cfg = Config()

    nodes = BankingAgentNodes(pipeline_instance=pipeline_instance, config=cfg)

    # 1. Khởi tạo StateGraph với AgentState
    workflow = StateGraph(AgentState)

    # 2. Đăng ký toàn bộ 9 Nodes
    workflow.add_node("input_guardrail", nodes.input_guardrail_node)
    workflow.add_node("router", nodes.supervisor_router_node)
    workflow.add_node("retrieve", nodes.retrieve_node)
    workflow.add_node("grade_documents", nodes.grade_documents_node)
    workflow.add_node("rewrite_query", nodes.rewrite_query_node)
    workflow.add_node("banking_tool", nodes.banking_tool_node)
    workflow.add_node("generate", nodes.generate_node)
    workflow.add_node("audit_generation", nodes.audit_generation_node)
    workflow.add_node("output_guardrail", nodes.output_guardrail_node)

    # 3. Nối luồng đi (Edges & Conditional Edges)
    workflow.add_edge(START, "input_guardrail")

    # Node 0 -> Node 1 (Router) hoặc Node 8 (Output Guardrail nếu bị từ chối)
    workflow.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {
            "safe": "router",
            "unsafe": "output_guardrail"
        }
    )

    # Node 1 (Supervisor Router) -> 3 nhánh
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "chit_chat": "generate",
            "banking_action": "banking_tool",
            "rag_search": "retrieve"
        }
    )

    # Nhánh RAG: Retrieve -> Thẩm định tài liệu
    workflow.add_edge("retrieve", "grade_documents")

    # Node 3 (Grade Documents) -> Generate hoặc Rewrite
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grader,
        {
            "relevant": "generate",
            "rewrite": "rewrite_query",
            "max_retries_reached": "generate"
        }
    )

    # Node 4 (Rewrite) -> Tìm kiếm lại (Retrieve)
    workflow.add_edge("rewrite_query", "retrieve")

    # Node 5 (Banking Tool) -> Tổng hợp kết quả tại Generate
    workflow.add_edge("banking_tool", "generate")

    # Node 6 (Generate) -> Node 7 (Audit Generation)
    workflow.add_edge("generate", "audit_generation")

    # Node 7 (Audit Generation) -> Sinh lại (Generate) hoặc Hoàn tất (Output Guardrail)
    workflow.add_conditional_edges(
        "audit_generation",
        route_after_audit,
        {
            "regenerate": "generate",
            "output_guardrail": "output_guardrail"
        }
    )

    # Node 8 (Output Guardrail) -> END
    workflow.add_edge("output_guardrail", END)

    # 4. Xác định Checkpointer
    active_checkpointer = checkpointer
    if active_checkpointer is None:
        active_checkpointer = create_persistent_checkpointer(sqlite_db_path)

    compiled_graph = workflow.compile(checkpointer=active_checkpointer)
    return compiled_graph
