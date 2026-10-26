# -*- coding: utf-8 -*-
"""
Routing & Pre-processing Nodes for Banking Agentic RAG
======================================================================
Cung cấp các Node điều hướng và tiền xử lý câu hỏi:
1. contextualize_query_node: Tái cấu trúc câu hỏi theo ngữ cảnh lịch sử chat.
2. route_query_node: Định tuyến câu hỏi sang vectorstore, web_search hoặc direct_chat.
3. decompose_query_node: Phân rã câu hỏi phức hợp thành tối đa 3 câu hỏi phụ độc lập.

NGUYÊN TẮC:
1. 0% REGEX LAW: Tuyệt đối không dùng regex.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. LOGGING: Có logger.info("---NODE: ...---") ở đầu mỗi node.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from agent_banking.agentic_rag.state import GraphState
from agent_banking.agentic_rag.chains import (
    get_contextualize_chain,
    get_router_chain,
    get_decomposer_chain
)

logger = logging.getLogger(__name__)


def _format_chat_history(chat_history: List[BaseMessage]) -> str:
    """
    Chuyển đổi danh sách BaseMessage thành chuỗi hội thoại văn bản.
    """
    if not chat_history:
        return "Không có lịch sử hội thoại trước đó."

    lines: List[str] = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            role_prefix = "Khách hàng"
        elif isinstance(msg, AIMessage):
            role_prefix = "Trợ lý AI"
        else:
            role_prefix = "Hệ thống"

        content = str(msg.content).strip()
        lines.append(f"{role_prefix}: {content}")

    return "\n".join(lines)


def contextualize_query_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 1: Contextualize Query Node.
    Nếu có lịch sử hội thoại (chat_history), viết lại câu hỏi để làm rõ ngữ cảnh.
    Đồng thời khởi tạo retry_count = 0 và lưu vết original_question.
    """
    logger.info("---NODE: CONTEXTUALIZE QUERY---")

    raw_question = state.get("question")
    if raw_question is None:
        raw_question = state.get("original_question", "")
    current_question = str(raw_question).strip()

    # Xác định câu hỏi gốc ban đầu
    existing_original = state.get("original_question")
    if existing_original:
        original_question = existing_original
    else:
        original_question = current_question

    chat_history = state.get("chat_history", [])
    if chat_history and len(chat_history) > 0:
        history_str = _format_chat_history(chat_history)
        try:
            chain = get_contextualize_chain()
            rephrased = chain.invoke({
                "chat_history": history_str,
                "question": current_question
            })
            if rephrased:
                clean_rephrased = str(rephrased).strip()
                if len(clean_rephrased) > 3:
                    current_question = clean_rephrased
        except Exception as e:
            logger.warning(f"Lỗi khi contextualize query: {e}. Giữ nguyên câu hỏi hiện tại.")

    logger.info(f"Contextualized Query: '{current_question}' (Original: '{original_question}')")

    return {
        "original_question": original_question,
        "question": current_question,
        "retry_count": 0
    }


def route_query_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 2: Route Query Node (Adaptive Router).
    Sử dụng RouteQuery Pydantic schema với .with_structured_output() để phân loại
    câu hỏi vào 1 trong 3 nguồn: 'vectorstore', 'web_search', 'direct_chat'.
    """
    logger.info("---NODE: ROUTE QUERY---")

    question = state.get("question")
    if question is None:
        question = state.get("original_question", "")
    target_question = str(question).strip()

    datasource = "vectorstore"
    try:
        router_chain = get_router_chain()
        result = router_chain.invoke({"question": target_question})
        if hasattr(result, "datasource"):
            if result.datasource in ["vectorstore", "web_search", "direct_chat"]:
                datasource = result.datasource
        logger.info(f"Định tuyến câu hỏi '{target_question}' -> [{datasource}] | Lý do: {getattr(result, 'reasoning', '')}")
    except Exception as e:
        logger.error(f"Lỗi khi định tuyến câu hỏi: {e}. Mặc định chọn 'vectorstore'.")
        datasource = "vectorstore"

    return {
        "datasource": datasource
    }


def decompose_query_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 3: Decompose Query Node.
    Phân rã câu hỏi phức tạp thành tối đa 3 câu hỏi phụ (sub_questions)
    để hỗ trợ truy xuất đa góc nhìn. Nếu câu hỏi đơn giản, giữ nguyên câu hỏi hiện tại.
    """
    logger.info("---NODE: DECOMPOSE QUERY---")

    question = state.get("question")
    if question is None:
        question = state.get("original_question", "")
    target_question = str(question).strip()

    sub_questions: List[str] = [target_question]

    try:
        decomposer_chain = get_decomposer_chain()
        result = decomposer_chain.invoke({"question": target_question})
        if hasattr(result, "sub_questions"):
            raw_subs = result.sub_questions
            if isinstance(raw_subs, list) and len(raw_subs) > 0:
                cleaned_subs: List[str] = []
                for sq in raw_subs[:3]:
                    sq_str = str(sq).strip()
                    if len(sq_str) > 0:
                        cleaned_subs.append(sq_str)
                if len(cleaned_subs) > 0:
                    sub_questions = cleaned_subs

        logger.info(f"Phân rã truy vấn thành {len(sub_questions)} câu hỏi phụ: {sub_questions}")
    except Exception as e:
        logger.warning(f"Lỗi khi phân rã câu hỏi: {e}. Giữ nguyên câu hỏi đơn.")
        sub_questions = [target_question]

    return {
        "sub_questions": sub_questions
    }
