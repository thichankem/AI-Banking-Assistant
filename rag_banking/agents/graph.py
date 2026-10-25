# -*- coding: utf-8 -*-
"""
LangGraph StateGraph Builder for Banking Agent
======================================================================
Xây dựng và biên dịch đồ thị luân chuyển trạng thái (StateGraph):
- Khởi tạo đầy đủ 8 Nodes nghiệp vụ.
- Thiết lập Conditional Edges định tuyến thông minh (Self-Reflective RAG + Tool Calling).
- Gắn MemorySaver Checkpointer để duy trì bộ nhớ phiên hội thoại (Session Memory).
- Bảo vệ chống lặp vô tận (Infinite Loop Protection).
"""

from __future__ import annotations
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent_banking.config import Config
from agent_banking.agents.state import AgentState
from agent_banking.agents.nodes import BankingAgentNodes
from agent_banking.agents.edges import (
    route_after_input_guardrail,
    route_by_intent,
    route_after_grading
)


def build_banking_agent_graph(
    pipeline_instance=None,
    config: Config | None = None,
    checkpointer: Any | None = None
):
    """
    Khởi tạo và biên dịch đồ thị LangGraph Agentic RAG cho Vietcombank.
    """
    cfg = config if config is not None else Config()
    nodes = BankingAgentNodes(pipeline_instance=pipeline_instance, config=cfg)

    # 1. Khởi tạo StateGraph với AgentState
    workflow = StateGraph(AgentState)

    # 2. Đăng ký toàn bộ 8 Nodes
    workflow.add_node("input_guardrail", nodes.input_guardrail_node)
    workflow.add_node("router", nodes.router_node)
    workflow.add_node("retrieve", nodes.retrieve_node)
    workflow.add_node("grade_documents", nodes.grade_documents_node)
    workflow.add_node("rewrite_query", nodes.rewrite_query_node)
    workflow.add_node("banking_tool", nodes.banking_tool_node)
    workflow.add_node("generate", nodes.generate_node)
    workflow.add_node("output_guardrail", nodes.output_guardrail_node)

    # 3. Nối các Edges & Conditional Edges
    # Điểm bắt đầu -> Kiểm duyệt đầu vào
    workflow.add_edge(START, "input_guardrail")

    # Sau Input Guardrail: Nếu an toàn -> Router, nếu vi phạm -> Output Guardrail
    workflow.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {
            "router": "router",
            "output_guardrail": "output_guardrail"
        }
    )

    # Sau Router: Rẽ 3 nhánh (RAG_SEARCH -> retrieve, BANKING_ACTION -> banking_tool, CHIT_CHAT -> generate)
    workflow.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "retrieve": "retrieve",
            "banking_tool": "banking_tool",
            "generate": "generate"
        }
    )

    # Nhánh RAG: Retrieve -> Thẩm định tài liệu
    workflow.add_edge("retrieve", "grade_documents")

    # Sau Thẩm định tài liệu: Hợp lệ -> generate, Không hợp lệ -> rewrite_query (có đếm retry)
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query"
        }
    )

    # Sau Rewrite: Quay lại bước Retrieve để tìm kiếm lại (Corrective RAG)
    workflow.add_edge("rewrite_query", "retrieve")

    # Nhánh Banking Tool: Sau khi chạy tool -> Tổng hợp kết quả tại generate
    workflow.add_edge("banking_tool", "generate")

    # Sau khi Generate: Bắt buộc qua Output Guardrail
    workflow.add_edge("generate", "output_guardrail")

    # Kết thúc luồng
    workflow.add_edge("output_guardrail", END)

    # 4. Gắn Checkpointer (Bộ nhớ phiên hội thoại)
    memory = checkpointer if checkpointer is not None else MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app
