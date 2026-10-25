# -*- coding: utf-8 -*-
"""
Conditional Edges for Banking LangGraph Workflow
======================================================================
Định nghĩa toàn bộ các hàm điều hướng có điều kiện (Conditional Routing):
1. route_after_input_guardrail: Chặn rủi ro an toàn ngay từ đầu vào.
2. route_by_intent: Rẽ nhánh theo ý định (RAG_SEARCH vs BANKING_ACTION vs CHIT_CHAT).
3. route_after_grading: Tự sửa lỗi (Corrective RAG) hoặc chuyển sang Generate.
   -> BẢO VỆ CHỐNG LẶP VÔ TẬN: So sánh retry_count với max_retries (mặc định = 2).
"""

from __future__ import annotations
from typing import Literal
from agent_banking.agents.state import AgentState


def route_after_input_guardrail(state: AgentState) -> Literal["router", "output_guardrail"]:
    """
    Điều hướng sau bước kiểm duyệt đầu vào:
    - Nếu vi phạm (Prompt Injection, Jailbreak, vi phạm dữ liệu): Đi thẳng tới output_guardrail để trả lời từ chối.
    - Nếu an toàn: Chuyển sang router để phân loại ý định.
    """
    input_guard = state.get("input_guard", {})
    if not input_guard.get("is_safe", True):
        return "output_guardrail"
    return "router"


def route_by_intent(state: AgentState) -> Literal["retrieve", "banking_tool", "generate"]:
    """
    Điều hướng sau Intent Router:
    - RAG_SEARCH: Tìm kiếm tài liệu pháp lý & quy chế.
    - BANKING_ACTION: Gọi công cụ ngân hàng (lãi suất, tỷ giá, tính lãi, tạo ticket).
    - CHIT_CHAT: Chào hỏi, giao tiếp thông thường.
    """
    intent = state.get("intent", "RAG_SEARCH")
    if intent == "BANKING_ACTION":
        return "banking_tool"
    elif intent == "CHIT_CHAT":
        return "generate"
    else:
        return "retrieve"


def route_after_grading(state: AgentState) -> Literal["generate", "rewrite_query"]:
    """
    Điều hướng sau Document Grader (Corrective RAG):
    - Nếu tài liệu RELEVANT: Chuyển tiếp tới generate.
    - Nếu tài liệu IRRELEVANT:
        * Nếu retry_count < max_retries: Chuyển tới rewrite_query để tìm lại.
        * Nếu đã chạm ngưỡng max_retries: Buộc chuyển sang generate (tránh lặp vô tận).
    """
    quality = state.get("document_quality", "RELEVANT")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if quality == "RELEVANT":
        return "generate"

    # Tài liệu không liên quan: Kiểm tra giới hạn retry
    if retry_count < max_retries:
        return "rewrite_query"
    else:
        print(f"⚠️ [LangGraph Loop Guard] Đã chạm giới hạn tối đa {max_retries} lần retry -> Chuyển sang sinh câu trả lời fallback an toàn.")
        return "generate"
