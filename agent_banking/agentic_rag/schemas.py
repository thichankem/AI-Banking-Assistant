# -*- coding: utf-8 -*-
"""
Pydantic Schemas for Structured Output in Agentic RAG
======================================================================
Định nghĩa toàn bộ các Pydantic BaseModel phục vụ việc ép kiểu dữ liệu
đầu ra từ LLM thông qua phương thức `llm.with_structured_output(Schema)`:
- Chấm điểm & Định tuyến: RouteQuery, DecomposeQuery, GradeDocument, GradeHallucination, GradeAnswer
- Mở rộng Tương tác Doanh nghiệp: CardButton, RichCardPayload, SuggestFollowups

NGUYÊN TẮC:
1. 0% REGEX LAW: Tuyệt đối không dùng regex.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. CLEAR DESCRIPTIONS: Mô tả rõ ràng từng trường để LLM sinh kết quả chính xác.
"""

from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """
    Schema định tuyến câu hỏi của người dùng vào nguồn dữ liệu phù hợp:
    - vectorstore: Tra cứu chính sách, biểu phí, quy định tài khoản, thẻ, eKYC của ngân hàng.
    - web_search: Thông tin thời sự bên ngoài, tin tức vĩ mô, tỷ giá liên ngân hàng quốc tế.
    - direct_chat: Chào hỏi, cảm ơn, hội thoại xã giao thông thường.
    """
    datasource: Literal["vectorstore", "web_search", "direct_chat"] = Field(
        description="Nguồn dữ liệu mục tiêu để trả lời câu hỏi: 'vectorstore' (tri thức ngân hàng nội bộ), 'web_search' (tin tức/thời sự bên ngoài), hoặc 'direct_chat' (chào hỏi xã giao)."
    )
    reasoning: str = Field(
        description="Giải thích ngắn gọn lý do tại sao câu hỏi được định tuyến vào nguồn dữ liệu này."
    )


class DecomposeQuery(BaseModel):
    """
    Schema phân rã câu hỏi phức hợp thành các câu hỏi phụ độc lập (tối đa 3 câu).
    Nếu câu hỏi đơn giản, giữ nguyên câu hỏi gốc trong danh sách 1 phần tử.
    """
    sub_questions: List[str] = Field(
        default_factory=list,
        description="Danh sách các câu hỏi phụ độc lập (tối đa 3 câu). Nếu câu hỏi đơn giản, trả về danh sách chứa duy nhất câu hỏi hiện tại."
    )


class GradeDocument(BaseModel):
    """
    Schema thẩm định độ liên quan của tài liệu trích xuất (Corrective RAG Grader).
    """
    binary_score: Literal["yes", "no"] = Field(
        description="Đánh giá tài liệu có chứa thông tin hữu ích và liên quan trực tiếp đến câu hỏi hay không: 'yes' nếu có, 'no' nếu không liên quan."
    )
    reason: str = Field(
        default="",
        description="Giải thích ngắn gọn lý do đánh giá tài liệu liên quan hoặc không liên quan."
    )


class GradeHallucination(BaseModel):
    """
    Schema thẩm định ảo giác (Self-RAG Groundedness Grader).
    Kiểm tra câu trả lời có được chứng minh hoàn toàn từ tài liệu căn cứ hay không.
    """
    binary_score: Literal["yes", "no"] = Field(
        description="Đánh giá câu trả lời có bám sát tài liệu hay không: 'yes' nếu hoàn toàn dựa trên dữ kiện tài liệu (grounded), 'no' nếu có thông tin bịa đặt hoặc suy diễn vô căn cứ (hallucinated)."
    )
    explanation: str = Field(
        default="",
        description="Giải thích ngắn gọn căn cứ đối soát giữa các luận điểm trong câu trả lời với nội dung tài liệu."
    )


class GradeAnswer(BaseModel):
    """
    Schema thẩm định độ hữu ích của câu trả lời (Self-RAG Answer Utility Grader).
    Kiểm tra câu trả lời có thực sự giải quyết triệt để câu hỏi của khách hàng hay không.
    """
    binary_score: Literal["yes", "no"] = Field(
        description="Đánh giá câu trả lời có giải quyết được câu hỏi hay không: 'yes' nếu trả lời trúng trọng tâm và đầy đủ, 'no' nếu trả lời lạc đề hoặc thiếu sót nghiêm trọng."
    )
    reason: Optional[str] = Field(
        default="",
        description="Lý do đánh giá câu trả lời hữu ích hoặc chưa hữu ích."
    )


# =====================================================================
# CÁC SCHEMA MỞ RỘNG TƯƠNG TÁC DOANH NGHIỆP (CARD BINDING & FOLLOW-UP)
# =====================================================================

class CardButton(BaseModel):
    """
    Nút tương tác đính kèm tin nhắn (Open URL, Send Text, Call Hotline).
    """
    title: str = Field(description="Nhãn hiển thị trên nút bấm (vd: 'Mở VCB Digibank', 'Gọi 1900 54 54 13').")
    action_type: Literal["open_url", "send_text", "call_phone"] = Field(
        description="Hành động khi nhấn nút: 'open_url', 'send_text' hoặc 'call_phone'."
    )
    payload: str = Field(description="URL liên kết, nội dung tin nhắn gửi đi, hoặc số điện thoại.")


class RichCardPayload(BaseModel):
    """
    Cấu trúc tin nhắn tương tác dạng Card (Card Data Binding) cho giao diện người dùng.
    """
    title: str = Field(description="Tiêu đề chính của Card.")
    subtitle: Optional[str] = Field(default="", description="Phụ đề hoặc tóm tắt ngắn gọn.")
    buttons: List[CardButton] = Field(default_factory=list, description="Danh sách tối đa 3 nút tương tác.")
    quick_replies: List[str] = Field(default_factory=list, description="Danh sách các câu gợi ý trả lời nhanh (Quick Replies).")


class SuggestFollowups(BaseModel):
    """
    Schema tự động gợi ý các câu hỏi tiếp theo (Follow-up Questions) liên quan đến nội dung vừa giải đáp.
    """
    questions: List[str] = Field(
        default_factory=list,
        description="Danh sách từ 2 đến 3 câu hỏi gợi ý tiếp theo tự nhiên, bám sát nghiệp vụ vừa trao đổi."
    )
