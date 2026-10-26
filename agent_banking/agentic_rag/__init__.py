# -*- coding: utf-8 -*-
"""
Agentic RAG Architecture for Banking Assistant
======================================================================
Module Agentic RAG thế hệ mới kết hợp Adaptive Router + Corrective RAG (CRAG) + Self-RAG
xây dựng trên nền tảng LangGraph StateGraph:
- GraphState: Trạng thái luân chuyển toàn trình
- Schemas: Pydantic BaseModels cho .with_structured_output()
- Chains: Các LangChain Runnables tối ưu hóa cho nghiệp vụ ngân hàng
- Nodes: 12 Nodes thực thi phân tách theo nhóm chức năng
"""

from .state import GraphState
from .schemas import (
    RouteQuery,
    DecomposeQuery,
    GradeDocument,
    GradeHallucination,
    GradeAnswer
)
from .chains import (
    get_llm,
    get_router_chain,
    get_decomposer_chain,
    get_document_grader_chain,
    get_hallucination_grader_chain,
    get_answer_grader_chain,
    get_contextualize_chain,
    get_rewriter_chain,
    get_generation_chain,
    get_direct_chat_chain
)

__all__ = [
    "GraphState",
    "RouteQuery",
    "DecomposeQuery",
    "GradeDocument",
    "GradeHallucination",
    "GradeAnswer",
    "get_llm",
    "get_router_chain",
    "get_decomposer_chain",
    "get_document_grader_chain",
    "get_hallucination_grader_chain",
    "get_answer_grader_chain",
    "get_contextualize_chain",
    "get_rewriter_chain",
    "get_generation_chain",
    "get_direct_chat_chain"
]
