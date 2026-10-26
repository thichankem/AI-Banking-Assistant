# -*- coding: utf-8 -*-
"""
Grading & Self-Correction Nodes for Banking Agentic RAG
======================================================================
Cung cấp các Node đánh giá tài liệu và tự sửa sai (Self-Correction):
7. grade_documents_node: Thẩm định độ liên quan của từng tài liệu (Corrective RAG).
8. rewrite_query_node: Tối ưu hóa lại câu truy vấn tìm kiếm và tăng retry_count.
9. evaluate_generation_node: Đánh giá ảo giác (groundedness) và độ hữu ích (utility) của câu trả lời.

NGUYÊN TẮC:
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. STRUCTURED GRADING: 100% sử dụng chains với Structured Output.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List

from langchain_core.documents import Document

from agent_banking.agentic_rag.state import GraphState
from agent_banking.agentic_rag.chains import (
    get_document_grader_chain,
    get_rewriter_chain,
    get_hallucination_grader_chain,
    get_answer_grader_chain
)

logger = logging.getLogger(__name__)


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 7: Grade Documents Node (Corrective RAG Grader).
    Duyệt qua từng tài liệu trong documents và chấm điểm bằng GradeDocument schema.
    Chỉ giữ lại các tài liệu có binary_score == 'yes'.
    Nếu danh sách tài liệu sau lọc bị rỗng (hoặc không đạt ngưỡng), bật cờ web_search_needed = True.
    """
    logger.info("---NODE: GRADE DOCUMENTS---")

    question = state.get("question")
    if question is None:
        question = state.get("original_question", "")
    target_question = str(question).strip()

    docs = state.get("documents", [])
    if not docs or len(docs) == 0:
        logger.warning("Không có tài liệu nào để chấm điểm -> Yêu cầu tìm kiếm web.")
        return {
            "documents": [],
            "web_search_needed": True
        }

    grader_chain = get_document_grader_chain()
    filtered_docs: List[Document] = []

    for idx, doc in enumerate(docs, start=1):
        content = doc.page_content
        source = doc.metadata.get("source", "Tài liệu Vietcombank")
        breadcrumb = doc.metadata.get("breadcrumb", "")

        try:
            grade_result = grader_chain.invoke({
                "question": target_question,
                "document": content
            })

            score = getattr(grade_result, "binary_score", "no")
            reason = getattr(grade_result, "reason", "")

            if score == "yes":
                logger.info(f"  + [Doc {idx}] RELEVANT (Nguồn: {source} | {breadcrumb}) - Lý do: {reason}")
                filtered_docs.append(doc)
            else:
                logger.info(f"  - [Doc {idx}] IRRELEVANT (Nguồn: {source} | {breadcrumb}) - Loại bỏ. Lý do: {reason}")
        except Exception as e:
            logger.error(f"Lỗi khi thẩm định tài liệu {idx}: {e}. Mặc định giữ lại an toàn.")
            filtered_docs.append(doc)

    if len(filtered_docs) == 0:
        logger.warning("Không có tài liệu nào vượt qua thẩm định -> Bật cờ web_search_needed = True")
        web_search_needed = True
    else:
        logger.info(f"Số lượng tài liệu liên quan được giữ lại: {len(filtered_docs)}/{len(docs)}")
        web_search_needed = False

    return {
        "documents": filtered_docs,
        "web_search_needed": web_search_needed
    }


def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 8: Rewrite Query Node (Self-Correction Loop).
    Viết lại câu hỏi để tối ưu hóa từ khóa nghiệp vụ ngân hàng cho lượt tìm kiếm kế tiếp,
    đồng thời tăng retry_count thêm 1 để chống vòng lặp vô tận.
    """
    logger.info("---NODE: REWRITE QUERY---")

    current_retries = state.get("retry_count", 0)
    new_retries = current_retries + 1

    orig_q = state.get("original_question")
    if orig_q is None:
        orig_q = state.get("question", "")
    original_question = str(orig_q).strip()

    curr_q = state.get("question")
    if curr_q is None:
        curr_q = original_question
    current_question = str(curr_q).strip()

    rewritten_query = current_question
    try:
        rewriter_chain = get_rewriter_chain()
        result = rewriter_chain.invoke({
            "original_question": original_question,
            "question": current_question
        })
        if result:
            clean_res = str(result).strip().strip('"\'')
            if len(clean_res) > 3:
                rewritten_query = clean_res
    except Exception as e:
        logger.error(f"Lỗi khi viết lại câu hỏi: {e}. Sử dụng mẫu quy định Vietcombank chuẩn.")
        rewritten_query = f"Quy định Vietcombank về {original_question}"

    logger.info(f"Lượt thử lại #{new_retries}: Câu hỏi viết lại -> '{rewritten_query}'")

    return {
        "question": rewritten_query,
        "retry_count": new_retries
    }


def evaluate_generation_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 9: Evaluate Generation Node (Self-RAG Evaluator).
    Kết hợp GradeHallucination và GradeAnswer:
    1. Kiểm tra câu trả lời có bịa đặt so với tài liệu không -> hallucination_grade ('grounded' / 'hallucinated').
    2. Kiểm tra câu trả lời có trả lời đúng câu hỏi không -> answer_grade ('useful' / 'not_useful').
    """
    logger.info("---NODE: EVALUATE GENERATION---")

    generation = state.get("generation", "").strip()
    question = state.get("question")
    if question is None:
        question = state.get("original_question", "")
    target_question = str(question).strip()

    docs = state.get("documents", [])
    doc_snippets: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "Vietcombank")
        bc = d.metadata.get("breadcrumb", "")
        doc_snippets.append(f"[{src} - {bc}]\n{d.page_content}")
    docs_context = "\n\n---\n\n".join(doc_snippets)

    # 1. Thẩm định ảo giác (Hallucination / Groundedness)
    hallucination_grade = "grounded"
    if docs_context:
        try:
            hall_chain = get_hallucination_grader_chain()
            hall_res = hall_chain.invoke({
                "documents": docs_context,
                "generation": generation
            })
            b_score = getattr(hall_res, "binary_score", "yes")
            if b_score == "yes":
                hallucination_grade = "grounded"
            else:
                hallucination_grade = "hallucinated"
            logger.info(f"Đánh giá ảo giác: [{hallucination_grade}] | Giải thích: {getattr(hall_res, 'explanation', '')}")
        except Exception as e:
            logger.error(f"Lỗi kiểm định ảo giác: {e}. Mặc định 'grounded'.")
            hallucination_grade = "grounded"
    else:
        # Nếu không có tài liệu căn cứ (ví dụ direct chat), mặc định grounded
        hallucination_grade = "grounded"

    # 2. Thẩm định độ hữu ích câu trả lời (Answer Utility)
    answer_grade = "useful"
    try:
        ans_chain = get_answer_grader_chain()
        ans_res = ans_chain.invoke({
            "question": target_question,
            "generation": generation
        })
        b_score = getattr(ans_res, "binary_score", "yes")
        if b_score == "yes":
            answer_grade = "useful"
        else:
            answer_grade = "not_useful"
        logger.info(f"Đánh giá độ hữu ích câu trả lời: [{answer_grade}] | Lý do: {getattr(ans_res, 'reason', '')}")
    except Exception as e:
        logger.error(f"Lỗi kiểm định độ hữu ích câu trả lời: {e}. Mặc định 'useful'.")
        answer_grade = "useful"

    return {
        "hallucination_grade": hallucination_grade,
        "answer_grade": answer_grade
    }
