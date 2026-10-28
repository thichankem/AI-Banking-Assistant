# -*- coding: utf-8 -*-
"""
AgentState Definition for Banking LangGraph Architecture v4.0
======================================================================
Định nghĩa cấu trúc trạng thái luân chuyển (State) giữa các Node trong StateGraph.
Tuân thủ nghiêm ngặt:
- 0% REGEX LAW: Tuyệt đối không dùng regex.
- ZERO TERNARY LAW: Không dùng toán tử 3 ngôi (x if c else y).
- NO TYPING LIBRARY: Native Python dict cho LangGraph StateGraph channels.
- INFINITE LOOP PROTECTION: retry_count và hallucination_retries.
- MULTI-AGENT SUPERVISOR & SELF-AUDITING RAG v4.0.

CÁC TRƯỜNG DỮ LIỆU CHÍNH TRONG AGENTSTATE:
1. Thông tin truy vấn & phiên:
   - session_id (str): Mã định danh phiên hội thoại.
   - user_query / query / original_query / rewritten_query (str): Câu hỏi của người dùng.
   - messages (list[dict]): Lịch sử tin nhắn dạng OpenAI [{"role": ..., "content": ...}].
   - chat_history (list[dict]): Lịch sử trao đổi lưu trữ trong bộ nhớ phiên.

2. Bảo mật đầu vào (Node 0: input_guardrail):
   - is_safe (bool): Trạng thái an toàn sau khi quét Prompt Injection, Jailbreak, Domain.
   - guardrail_reason (str): Lý do vi phạm bảo mật nếu có.
   - input_guard (dict): Chi tiết kết quả kiểm duyệt đầu vào.

3. Điều phối Multi-Agent Supervisor & Intent (Node 1: supervisor_router):
   - intent (str): "CHIT_CHAT" | "BANKING_ACTION" | "RAG_SEARCH" | "REFUSAL".
   - sub_agent_role (str): "GENERAL_ADVISOR" | "CREDIT_CARD_AGENT" | "LOAN_MORTGAGE_AGENT" | "COMPLIANCE_AUDIT_AGENT".
   - target_tool (str | None): Tên công cụ nghiệp vụ cần gọi nếu có.

4. Truy xuất & Thẩm định tài liệu (Node 2, 3, 4: retrieve, grade_documents, rewrite_query):
   - retrieved_chunks / retrieved_docs (list[dict]): Danh sách văn bản quy chuẩn trích xuất.
   - doc_quality / document_quality (str): "RELEVANT" | "IRRELEVANT".
   - retry_count (int): Số lần đã thử viết lại câu hỏi tìm kiếm lại (tối đa 2).
   - max_retries (int): Giới hạn số lần thử lại tối đa.

5. Công cụ nghiệp vụ ngân hàng (Node 5: banking_tool):
   - tool_name (str): Tên công cụ đã thực thi.
   - tool_output (dict): Dữ liệu kết quả từ công cụ.
   - tool_outputs / tool_calls (list[dict]): Danh sách kết quả các công cụ.

6. Sinh câu trả lời & Kiểm toán ảo giác (Node 6, 7: generate, audit_generation):
   - draft_answer (str): Câu trả lời dự thảo trước khi kiểm toán.
   - final_answer (str): Câu trả lời hoàn chỉnh sau kiểm toán.
   - audit_status (str): "PASS" | "FAIL".
   - audit_feedback (str): Nhận xét nhắc nhở LLM bổ sung căn cứ điều khoản.
   - hallucination_retries (int): Số lần tự sửa ảo giác/thiếu trích dẫn (tối đa 1).

7. Kiểm duyệt đầu ra & Trích dẫn pháp lý (Node 8: output_guardrail):
   - output_guard (dict): Kết quả kiểm duyệt an toàn đầu ra.
   - citations (list[dict]): Danh sách trích dẫn điều khoản pháp lý [Điều X, Khoản Y].
   - sources (list[str]): Nguồn tài liệu tham chiếu.

8. Observability & Bước thực thi:
   - execution_steps (list[str]): Chuỗi các Node đã chạy trong quy trình.
"""

from __future__ import annotations

# Native Python dictionary state schema for LangGraph StateGraph
AgentState = dict
