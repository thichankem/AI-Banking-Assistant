# -*- coding: utf-8 -*-
"""
AgentState Definition for Banking LangGraph Architecture v4.0
======================================================================
Định nghĩa cấu trúc trạng thái luân chuyển (State) giữa các Node trong StateGraph.
Tuân thủ nghiêm ngặt:
- 0% REGEX LAW: Tuyệt đối không dùng regex.
- ZERO TERNARY LAW: Không dùng toán tử 3 ngôi (x if c else y).
- NO LANGCHAIN BLOAT: Pure TypedDict của chuẩn Python typing.
- INFINITE LOOP PROTECTION: retry_count và hallucination_retries.
- MULTI-AGENT SUPERVISOR & SELF-AUDITING RAG v4.0.
"""

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """
    Trạng thái toàn trình của Banking Agentic Workflow v4.0.
    """
    # 1. Thông tin truy vấn và phiên làm việc
    session_id: str
    user_query: str
    query: str
    original_query: str
    rewritten_query: str
    messages: List[Dict[str, str]]
    chat_history: List[Dict[str, Any]]

    # 2. Bảo mật đầu vào (Node 0: input_guardrail)
    is_safe: bool
    guardrail_reason: str
    input_guard: Dict[str, Any]

    # 3. Điều phối Multi-Agent Supervisor & Intent (Node 1: supervisor_router)
    intent: str
    sub_agent_role: str
    target_tool: Optional[str]

    # 4. Truy xuất & Thẩm định tài liệu (Node 2, 3, 4: retrieve, grade_documents, rewrite_query)
    parsed_query: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    retrieved_docs: List[Dict[str, Any]]
    doc_quality: str
    document_quality: str
    retry_count: int
    max_retries: int

    # 5. Công cụ nghiệp vụ ngân hàng (Node 5: banking_tool)
    tool_name: str
    tool_output: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]

    # 6. Sinh câu trả lời & Kiểm toán ảo giác (Node 6, 7: generate, audit_generation)
    draft_answer: str
    final_answer: str
    audit_status: str
    audit_feedback: str
    hallucination_retries: int
    hallucination_grade: str

    # 7. Kiểm duyệt đầu ra & Trích dẫn pháp lý (Node 8: output_guardrail)
    output_guard: Dict[str, Any]
    citations: List[Dict[str, Any]]
    sources: List[str]

    # 8. Observability & Real-time Streaming Traces
    execution_steps: List[str]
    trace_metadata: Dict[str, Any]
