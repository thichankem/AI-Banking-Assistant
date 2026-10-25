# -*- coding: utf-8 -*-
"""
AgentState Definition for Banking LangGraph Architecture
======================================================================
Định nghĩa cấu trúc trạng thái luân chuyển (State) giữa các Node trong StateGraph.
Tuân thủ:
- Pure TypedDict (Không phụ thuộc class phức tạp của LangChain).
- Minh bạch dữ liệu theo từng bước thực thi (Input Guard -> Route -> Tool/RAG -> Generate -> Output Guard).
- Lưu giữ biến đếm retry_count để bảo vệ chống lặp vô tận (Infinite Loop Protection).
"""

from __future__ import annotations
from typing import TypedDict, Any


class AgentState(TypedDict, total=False):
    """
    Trạng thái toàn trình của Banking Agentic Workflow.
    """
    # 1. Thông tin truy vấn & phiên làm việc
    query: str                       # Câu hỏi hiện tại (có thể là câu đã được rewrite)
    original_query: str              # Câu hỏi gốc ban đầu từ người dùng
    session_id: str                  # Mã định danh phiên hội thoại (Thread ID)
    chat_history: list[dict[str, Any]] # Lịch sử chat: [{"role": "user"|"assistant", "content": "..."}]
    
    # 2. Kiểm duyệt an toàn đầu vào (Banking Input Guardrails)
    input_guard: dict[str, Any]      # Kết quả kiểm duyệt: is_safe, sanitized_query, refusal_message
    
    # 3. Phân loại ý định & Định tuyến (Routing)
    intent: str                      # "RAG_SEARCH" | "BANKING_ACTION" | "CHIT_CHAT" | "REFUSAL"
    target_tool: str | None          # Tên công cụ mục tiêu nếu intent là BANKING_ACTION
    
    # 4. Truy xuất tài liệu & Tự đánh giá (Corrective RAG)
    parsed_query: dict[str, Any]     # Kết quả chuẩn hóa từ viết tắt và trích xuất thực thể
    retrieved_docs: list[dict[str, Any]] # Danh sách chunks sau Hybrid Retrieval & Cross-Encoder
    document_quality: str            # "RELEVANT" | "IRRELEVANT" (Đánh giá của Document Grader)
    
    # 5. Gọi công cụ nghiệp vụ ngân hàng (Banking Action Tools)
    tool_calls: list[dict[str, Any]] # Danh sách tool calls cần thực thi
    tool_outputs: list[dict[str, Any]] # Kết quả thực thi công cụ
    
    # 6. Tổng hợp câu trả lời & Kiểm duyệt đầu ra
    draft_answer: str                # Câu trả lời dự thảo trước khi qua output guardrail
    hallucination_grade: str         # "GROUNDED" | "NOT_GROUNDED" (Đánh giá tính trung thực)
    output_guard: dict[str, Any]     # Kết quả kiểm duyệt đầu ra: is_safe, sanitized_answer, flags
    final_answer: str                # Câu trả lời hoàn chỉnh cuối cùng trả về cho người dùng
    
    # 7. Trích dẫn pháp lý & Nguồn dữ liệu
    citations: list[dict[str, Any]]  # Danh sách trích dẫn điều khoản [Điều X, Khoản Y]
    sources: list[str]               # Danh sách tệp văn bản quy chế nguồn
    
    # 8. Kiểm soát vòng lặp & AI Observability (Arize Phoenix)
    retry_count: int                 # Biến đếm số lần thử lại (query rewrite hoặc correction)
    max_retries: int                 # Giới hạn số lần thử lại tối đa (từ Config.MAX_RETRIES)
    trace_metadata: dict[str, Any]   # Lưu vết thời gian và node traces phục vụ giám sát Phoenix
