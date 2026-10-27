# -*- coding: utf-8 -*-
"""
Conditional Edges for Banking LangGraph Workflow v4.0
======================================================================
Định nghĩa toàn bộ các hàm điều hướng có điều kiện (Conditional Routing):
1. route_after_input_guardrail: Chặn rủi ro an toàn ngay từ đầu vào.
2. route_after_router: Rẽ nhánh theo Intent & Sub-Agent Role (CHIT_CHAT, BANKING_ACTION, RAG_SEARCH).
3. route_after_grader: Thẩm định tài liệu (Corrective RAG) hoặc chuyển sang Generate (chống lặp max_retries).
4. route_after_audit: Kiểm toán câu trả lời sau sinh (Post-Generation Hallucination & Citation Check).

NGUYÊN TẮC:
- 0% REGEX LAW: Tuyệt đối không dùng regex.
- ZERO TERNARY: Ưu tiên if/else tường minh (không dùng toán tử 3 ngôi).
- INFINITE LOOP PROTECTION: Kiểm soát chặt chẽ số lần retry.
"""

from __future__ import annotations
from agent_banking.langgraph_agent.state import AgentState


def route_after_input_guardrail(state: AgentState) -> str:
    """
    Điều hướng sau bước kiểm duyệt đầu vào:
    - Nếu vi phạm (Prompt Injection, Jailbreak, vi phạm dữ liệu): Đi thẳng tới output_guardrail.
    - Nếu an toàn: Chuyển sang router để phân loại ý định.
    """
    is_safe = state.get("is_safe", True)
    if is_safe is False:
        return "unsafe"

    input_guard = state.get("input_guard", {})
    if input_guard.get("is_safe", True) is False:
        return "unsafe"

    return "safe"


def route_after_router(state: AgentState) -> str:
    """
    Điều hướng sau Intent Router:
    - RAG_SEARCH: Tìm kiếm tài liệu pháp lý & quy chế Vietcombank.
    - BANKING_ACTION: Gọi công cụ ngân hàng (lãi suất, tỷ giá, tính lãi, khấu hao, sinh trắc học, ticket).
    - CHIT_CHAT: Chào hỏi, giao tiếp thông thường.
    """
    intent = state.get("intent", "RAG_SEARCH")
    if intent == "BANKING_ACTION":
        return "banking_action"
    elif intent == "CHIT_CHAT":
        return "chit_chat"
    else:
        return "rag_search"


# Alias để tương thích ngược với code cũ
route_by_intent = route_after_router


def route_after_grader(state: AgentState) -> str:
    """
    Điều hướng sau Document Grader (Corrective RAG):
    - Nếu tài liệu RELEVANT: Chuyển tiếp tới generate.
    - Nếu tài liệu IRRELEVANT:
        * Nếu retry_count < max_retries: Chuyển tới rewrite_query để tìm lại.
        * Nếu đã chạm ngưỡng max_retries: Chuyển sang generate với fallback an toàn.
    """
    quality = state.get("doc_quality")
    if quality is None:
        quality = state.get("document_quality", "RELEVANT")

    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 2))

    if quality == "RELEVANT":
        return "relevant"

    if retry_count < max_retries:
        return "rewrite"
    else:
        return "max_retries_reached"


# Alias để tương thích ngược với code cũ
route_after_grading = route_after_grader


def route_after_audit(state: AgentState) -> str:
    """
    Điều hướng sau Node 7 (audit_generation).
    Tuyệt đối tuân thủ Infinite Loop Protection (chỉ retry tối đa 1 lần).
    """
    status = state.get("audit_status", "PASS")
    retries = int(state.get("hallucination_retries", 0))

    if status == "FAIL":
        if retries <= 1:
            return "regenerate"

    return "output_guardrail"
