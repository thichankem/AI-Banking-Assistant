# -*- coding: utf-8 -*-
"""
GraphState Definition for Banking Agentic RAG Architecture
======================================================================
Định nghĩa cấu trúc trạng thái luân chuyển (GraphState) cho hệ thống
Agentic RAG tích hợp Adaptive Router + Corrective RAG (CRAG) + Self-RAG
kèm các tham số quản trị doanh nghiệp chuyên sâu.

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. TYPED STATE: Định kiểu chuẩn xác với TypedDict, BaseMessage và Document.
"""

from __future__ import annotations
from typing import TypedDict, List, Optional, Any, Dict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """
    Trạng thái toàn trình của luồng Agentic RAG Ngân hàng (LangGraph State).
    Mỗi Node nhận vào state này và trả về dict chứa các trường cần cập nhật.
    """
    # 1. Truy vấn và Lịch sử hội thoại
    original_question: str                  # Câu hỏi gốc ban đầu của người dùng
    question: str                           # Câu hỏi hiện tại (sau contextualize hoặc rewrite)
    sub_questions: List[str]                # Danh sách câu hỏi phụ nếu câu hỏi phức tạp cần chia nhỏ
    chat_history: List[BaseMessage]         # Lịch sử hội thoại dạng LangChain BaseMessage

    # 2. Truy xuất tài liệu (Retrieval & Reranking)
    documents: List[Document]               # Danh sách tài liệu truy xuất được (LangChain Document)

    # 3. Định tuyến & Trạng thái điều hướng (Adaptive Routing)
    datasource: str                         # Kết quả định tuyến: "vectorstore", "web_search", "direct_chat"
    web_search_needed: bool                 # Cờ báo hiệu cần fallback sang web search hoặc rewrite

    # 4. Kiểm soát vòng lặp (Self-Correction & Infinite Loop Protection)
    retry_count: int                        # Đếm số lần rewrite/thử lại tìm kiếm để chống lặp vô tận

    # 5. Sinh phản hồi & Chấm điểm kiểm soát chất lượng (Self-RAG Grading)
    generation: str                         # Câu trả lời do LLM sinh ra
    hallucination_grade: str                # Đánh giá ảo giác: "grounded" hoặc "hallucinated"
    answer_grade: str                       # Đánh giá độ hữu ích: "useful" hoặc "not_useful"

    # 6. Mở rộng Quản trị Doanh nghiệp (Enterprise Banking Capabilities)
    complain_count: int                     # Đếm số lần phàn nàn/thái độ tiêu cực trong phiên (ngưỡng >= 2 để handoff)
    is_handoff_requested: bool              # Cờ yêu cầu chuyển phiên sang Chuyên viên Tư vấn (Live Agent Handoff)
    handoff_payload: Dict[str, Any]         # Dữ liệu điều phối hàng đợi (queue, segment, hotline)
    suggested_followups: List[str]          # Danh sách câu hỏi gợi ý tiếp theo (Follow-up Questions)
    rich_card: Dict[str, Any]               # Cấu trúc giao diện tương tác phong phú (Buttons, Quick Replies)
