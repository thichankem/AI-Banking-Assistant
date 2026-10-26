# -*- coding: utf-8 -*-
"""
Generation Nodes for Banking Agentic RAG
======================================================================
Cung cấp các Node tạo sinh câu trả lời cho khách hàng:
10. generate_answer_node: Sinh câu trả lời căn cứ vào tài liệu đã lọc và kiểm định (kèm trích dẫn pháp lý và cấu trúc XML Chunk).
11. direct_chat_node: Trả lời trực tiếp các câu chào hỏi, cảm ơn, xã giao theo ngữ cảnh thời gian (Time-aware Greeting).
12. fallback_node: Phản hồi thông báo lịch sự khi vượt quá số lần thử lại tối đa (chống vòng lặp vô tận) và chuyển kênh hỗ trợ.

NGUYÊN TẮC:
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. BANKING SAFETY: Tuân thủ quy chuẩn bảo mật và phong cách tư vấn Vietcombank.
"""

from __future__ import annotations
import logging
import datetime
from typing import Dict, Any, List

from langchain_core.documents import Document

from agent_banking.config import Config
from agent_banking.agentic_rag.state import GraphState
from agent_banking.agentic_rag.chains import (
    get_generation_chain,
    get_direct_chat_chain
)
from agent_banking.agentic_rag.nodes.routing_nodes import _format_chat_history

logger = logging.getLogger(__name__)


def _clean_think_tags(text: str) -> str:
    """
    Làm sạch thẻ suy nghĩ nội tâm <think>...</think> của LLM (0% Regex Law).
    """
    cleaned = text
    while True:
        start_idx = cleaned.find("<think>")
        if start_idx == -1:
            break
        end_idx = cleaned.find("</think>")
        if end_idx == -1:
            cleaned = cleaned[:start_idx]
            break
        else:
            cleaned = cleaned[:start_idx] + cleaned[end_idx + len("</think>"):]
    return cleaned.strip()


def _get_time_of_day_greeting() -> str:
    """
    Tạo lời chào theo 4 khung giờ trong ngày (Time-aware Greeting Process).
    - 5h - 11h: Chào buổi sáng
    - 12h - 18h: Chào buổi chiều
    - 19h - 22h: Chào buổi tối
    - 22h - 5h: Kính chào Quý khách (đêm)
    """
    hour = datetime.datetime.now().hour
    if hour >= 5:
        if hour < 12:
            return "Chào buổi sáng Quý khách!"
    if hour >= 12:
        if hour < 18:
            return "Chào buổi chiều Quý khách!"
    if hour >= 18:
        if hour < 22:
            return "Chào buổi tối Quý khách!"
    return "Kính chào Quý khách!"


def _format_documents_context(documents: List[Document]) -> str:
    """
    Tổng hợp danh sách Document thành văn bản cấu trúc XML (Structured XML Context Engineering):
    Mỗi đoạn dữ liệu được bao bọc trong thẻ <chunk> kèm <chunk_metadata> và <chunk_content>,
    giúp LLM dễ dàng bám sát sự thật và trích dẫn chuẩn xác nguồn căn cứ [Điều X, Khoản Y].
    """
    if not documents:
        return "Không có tài liệu căn cứ nào được cung cấp."

    xml_chunks: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Tài liệu Ngân hàng Vietcombank")
        breadcrumb = doc.metadata.get("breadcrumb", "")
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{idx}")
        pages = doc.metadata.get("page", doc.metadata.get("pages", ""))

        meta_lines = [
            f"    <id>{idx}</id>",
            f"    <chunk_id>{chunk_id}</chunk_id>",
            f"    <source>{source}</source>"
        ]
        if breadcrumb:
            meta_lines.append(f"    <breadcrumb>{breadcrumb}</breadcrumb>")
        if pages:
            meta_lines.append(f"    <pages>{pages}</pages>")

        metadata_xml = "\n".join(meta_lines)
        content_body = doc.page_content.strip()

        chunk_block = (
            f"<chunk>\n"
            f"  <chunk_metadata>\n"
            f"{metadata_xml}\n"
            f"  </chunk_metadata>\n"
            f"  <chunk_content>\n"
            f"{content_body}\n"
            f"  </chunk_content>\n"
            f"</chunk>"
        )
        xml_chunks.append(chunk_block)

    return "\n\n".join(xml_chunks)


def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 10: Generate Answer Node.
    Sinh câu trả lời từ danh sách tài liệu đã thẩm định và câu hỏi hiện tại.
    Áp dụng Banking System Guardrails: nghiêm cấm bịa đặt số liệu, bắt buộc trích dẫn
    nguồn điều khoản pháp lý và bảo mật thông tin tài khoản.
    """
    logger.info("---NODE: GENERATE ANSWER---")

    question = state.get("question")
    if question is None:
        question = state.get("original_question", "")
    target_question = str(question).strip()

    docs = state.get("documents", [])
    context_text = _format_documents_context(docs)

    try:
        gen_chain = get_generation_chain()
        raw_answer = gen_chain.invoke({
            "context": context_text,
            "question": target_question
        })
        answer = _clean_think_tags(str(raw_answer))
        logger.info("Đã sinh thành công câu trả lời từ tài liệu căn cứ.")
    except Exception as e:
        logger.error(f"Lỗi khi sinh câu trả lời từ LLM: {e}")
        # Chế độ dự phòng khi lỗi kết nối LLM
        answer = (
            "Dạ, hệ thống đang gặp gián đoạn tạm thời khi kết nối máy chủ xử lý ngôn ngữ. "
            "Dưới đây là các nội dung căn cứ được trích xuất từ tài liệu chính thức của Vietcombank:\n\n"
            f"{context_text[:1000]}...\n\n"
            "Quý khách vui lòng liên hệ Tổng đài 1900 54 54 13 để được tư vấn viên hỗ trợ trực tiếp."
        )

    # Tự động gợi ý 2-3 câu hỏi tiếp theo bám sát chủ đề (Follow-up Questions)
    suggested_followups: List[str] = []
    lower_q = target_question.lower()
    if "ekyc" in lower_q or "mở tài khoản" in lower_q:
        suggested_followups = [
            "Hạn mức giao dịch chuyển tiền qua eKYC tối đa bao nhiêu?",
            "Làm thế nào để nâng hạn mức tài khoản eKYC lên mức thông thường?",
            "Điều kiện và giấy tờ cần thiết khi mở tài khoản eKYC?"
        ]
    elif "thẻ" in lower_q or "visa" in lower_q:
        suggested_followups = [
            "Biểu phí thường niên và phí rút tiền mặt tại ATM của thẻ?",
            "Hồ sơ và điều kiện chứng minh thu nhập để phát hành thẻ tín dụng?",
            "Làm thế nào để kích hoạt và khóa thẻ khẩn cấp trên VCB Digibank?"
        ]
    elif "tiết kiệm" in lower_q or "lãi suất" in lower_q:
        suggested_followups = [
            "Bảng lãi suất tiền gửi tiết kiệm các kỳ hạn mới nhất hôm nay?",
            "Có thể rút tiền gửi tiết kiệm trước hạn được không và nhận lãi thế nào?",
            "Cách gửi tiết kiệm online qua VCB Digibank để nhận lãi suất ưu đãi?"
        ]

    # Đóng gói Card Data Binding cho giao diện tương tác
    rich_card = {
        "title": "Hỗ trợ Dịch vụ Ngân hàng Số Vietcombank",
        "subtitle": "Kênh tra cứu thông tin chính thức",
        "buttons": [
            {
                "title": "Mở ứng dụng VCB Digibank",
                "action_type": "open_url",
                "payload": "https://vcbdigibank.vietcombank.com.vn"
            },
            {
                "title": "Gọi Tổng đài 1900 54 54 13",
                "action_type": "call_phone",
                "payload": "1900545413"
            }
        ],
        "quick_replies": suggested_followups
    }

    return {
        "generation": answer,
        "suggested_followups": suggested_followups,
        "rich_card": rich_card
    }


def direct_chat_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 11: Direct Chat Node.
    Xử lý các câu hỏi chào hỏi, cảm ơn, hỏi thăm xã giao thông thường
    tích hợp lời chào động theo thời gian thực (Time-aware Greeting).
    """
    logger.info("---NODE: DIRECT CHAT---")

    question = state.get("question")
    if question is None:
        question = state.get("original_question", "")
    target_question = str(question).strip()

    chat_history = state.get("chat_history", [])
    history_str = _format_chat_history(chat_history)
    time_greeting = _get_time_of_day_greeting()

    try:
        chat_chain = get_direct_chat_chain()
        raw_response = chat_chain.invoke({
            "chat_history": history_str,
            "question": target_question
        })
        chat_response = _clean_think_tags(str(raw_response))
        logger.info("Đã tạo câu trả lời xã giao trực tiếp.")
    except Exception as e:
        logger.error(f"Lỗi khi tạo phản hồi xã giao: {e}")
        chat_response = (
            f"{time_greeting} Em là Trợ lý AI của Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank). "
            f"Em có thể hỗ trợ Quý khách tra cứu thông tin về mở tài khoản eKYC, hạn mức chuyển tiền, "
            f"biểu phí dịch vụ, thẻ ngân hàng hoặc các quy định giao dịch. Quý khách cần em hỗ trợ nội dung gì ạ?"
        )

    return {
        "generation": chat_response,
        "suggested_followups": [
            "Tra cứu lãi suất tiết kiệm Vietcombank hôm nay",
            "Hạn mức chuyển tiền eKYC tối đa là bao nhiêu?",
            "Biểu phí phát hành và thường niên thẻ tín dụng"
        ]
    }


def fallback_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 12: Fallback Node.
    Trả về câu thông báo lịch sự, chuẩn mực ngân hàng khi retry_count vượt ngưỡng
    giới hạn (chống lặp vô tận) và đề xuất kết nối Live Agent hoặc Hotline 24/7.
    """
    logger.info("---NODE: FALLBACK---")

    question = state.get("original_question")
    if question is None:
        question = state.get("question", "")
    target_question = str(question).strip()

    retry_count = state.get("retry_count", 0)
    logger.warning(f"Kích hoạt Fallback Node sau {retry_count} lần thử lại cho câu hỏi: '{target_question}'")

    fallback_message = (
        f"Kính thưa Quý khách,\n\n"
        f"Hệ thống Trợ lý AI Vietcombank đã tiến hành rà soát chuyên sâu trong cơ sở dữ liệu quy chế "
        f"nhưng hiện tại chưa tìm thấy thông tin văn bản chính thức đủ độ xác thực để giải đáp chi tiết câu hỏi:\n"
        f"👉 *\"{target_question}\"*\n\n"
        f"Để đảm bảo an toàn và quyền lợi tối đa cho Quý khách trong các giao dịch tài chính, "
        f"kính mời Quý khách tham khảo qua các kênh hỗ trợ chính thức sau:\n"
        f"1. 📞 **Trung tâm Hỗ trợ Khách hàng 24/7**: **1900 54 54 13** (hoặc +84 24 3824 3524).\n"
        f"2. 🏢 **Điểm giao dịch**: Quý khách có thể đến bất kỳ Chi nhánh hoặc Phòng giao dịch Vietcombank gần nhất để được Giao dịch viên hướng dẫn.\n"
        f"3. 🌐 **Website chính thức**: [https://www.vietcombank.com.vn](https://www.vietcombank.com.vn)\n\n"
        f"Trợ lý AI Vietcombank xin chân thành cảm ơn sự thông cảm của Quý khách!"
    )

    return {
        "generation": fallback_message,
        "is_handoff_requested": True,
        "handoff_payload": {
            "reason": "FALLBACK_MAX_RETRIES_EXCEEDED",
            "hotline": "1900 54 54 13",
            "unresolved_question": target_question
        }
    }
