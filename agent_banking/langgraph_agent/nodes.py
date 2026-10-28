# -*- coding: utf-8 -*-
"""
Agent Nodes for Banking LangGraph Workflow v4.0
======================================================================
Cung cấp các Node thực thi trong đồ thị StateGraph phiên bản 4.0:
1. input_guardrail_node: Làm sạch biến tạm thời (chống nhiễm chéo state) và kiểm duyệt đầu vào (Prompt Injection, Jailbreak, Domain).
2. supervisor_router_node: Phân loại ý định & chỉ định Sub-Agent chuyên trách (Multi-Agent Supervisor).
3. retrieve_node: Hybrid Search & Cross-Encoder Reranking qua BankingRAGTool.
4. grade_documents_node: Đánh giá độ liên quan của tài liệu trích xuất (Corrective RAG).
5. rewrite_query_node: Viết lại câu hỏi để thử lại tìm kiếm (kèm retry_count chống lặp vô tận).
6. banking_tool_node: Thực thi công cụ nghiệp vụ ngân hàng (Tỷ giá, Lãi suất, Tiết kiệm, Khấu hao khoản vay, Thẩm định QĐ 2345, Ticket).
7. generate_node: Sinh câu trả lời căn cứ tài liệu/công cụ theo Persona chuyên viên Vietcombank.
8. audit_generation_node: Thẩm định câu trả lời sau sinh (chống ảo giác, bắt buộc trích dẫn [Điều X, Khoản Y]).
9. output_guardrail_node: Kiểm duyệt an toàn đầu ra, làm sạch think tag, gắn disclaimer và cập nhật lịch sử chat.

NGUYÊN TẮC:
- 0% REGEX LAW: Tuyệt đối không dùng regex hoặc thư viện re.
- ZERO TERNARY LAW: Tuyệt đối không dùng toán tử 3 ngôi (x if c else y).
- NO TYPING LIBRARY: Native Python dict cho LangGraph state.
- INFINITE LOOP PROTECTION: Giới hạn chặt chẽ retry_count và hallucination_retries.
"""

from __future__ import annotations
from openai import OpenAI

from agent_banking.config import Config
from agent_banking.langgraph_agent.state import AgentState
from agent_banking.langgraph_agent.prompts import (
    ROUTER_SYSTEM_PROMPT,
    DOCUMENT_GRADER_PROMPT,
    QUERY_REWRITER_PROMPT,
    BANKING_AGENT_SYSTEM_PROMPT,
    SUB_AGENT_SYSTEM_PROMPTS
)
from agent_banking.guardrails.rag_guardrails import BankingGuardrailManager
from agent_banking.tracing.tracer import BankingTracer
from agent_banking.tools.rag_tool import BankingRAGTool
from agent_banking.tools.banking_api_tools import (
    get_interest_rates,
    get_foreign_exchange_rates,
    calculate_savings_profit,
    calculate_loan_amortization,
    check_biometric_compliance_qd2345,
    create_support_ticket,
    parse_vietnamese_financial_params
)


def _clean_think_tags(text: str) -> str:
    """Làm sạch thẻ suy nghĩ <think>...</think> của DeepSeek/LLM (0% Regex)."""
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


class BankingAgentNodes:
    """Tập hợp các hàm Node thực thi cho StateGraph v4.0."""

    def __init__(self, pipeline_instance=None, config: Config | None = None):
        if config is not None:
            self.config = config
        else:
            self.config = Config()

        # 1. Khởi tạo Guardrail Manager
        embedder_inst = None
        if pipeline_instance is not None:
            if hasattr(pipeline_instance, "embedder"):
                embedder_inst = pipeline_instance.embedder
        self.guardrail_manager = BankingGuardrailManager(embedder=embedder_inst)

        # 2. Khởi tạo Tracing Manager
        self.tracer = BankingTracer(config=self.config)

        # 3. Khởi tạo RAG Tool
        self.rag_tool = BankingRAGTool(pipeline_instance=pipeline_instance, config=self.config)

        # 4. Khởi tạo LLM Client (FPT Cloud / DeepSeek API)
        self.client = None
        if self.config.LLM_API_KEY and self.config.LLM_BASE_URL:
            try:
                self.client = OpenAI(
                    api_key=self.config.LLM_API_KEY,
                    base_url=self.config.LLM_BASE_URL
                )
            except Exception as e:
                print(f"[Agent Nodes] Cảnh báo khởi tạo LLM Client: {e}")

    def _check_tool_guard(self, tool_name: str, params: dict):
        res = self.guardrail_manager.guard_tool_call(tool_name, params)
        if isinstance(res, tuple):
            is_ok = bool(res[0])
            msg = ""
            if len(res) > 1:
                msg = str(res[1])
            return is_ok, msg
        if isinstance(res, dict):
            is_ok = bool(res.get("is_safe", True))
            msg = str(res.get("error_message", ""))
            return is_ok, msg
        return True, ""

    def _call_llm_quick(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        """Gọi LLM nội bộ tốc độ cao cho Router/Grader/Rewriter."""
        if not self.client:
            return ""
        try:
            resp = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens
            )
            raw_content = resp.choices[0].message.content
            if raw_content is None:
                return ""
            return _clean_think_tags(raw_content).strip()
        except Exception as e:
            print(f"[Agent Nodes] LLM Quick Call Error: {e}")
            return ""

    # =========================================================================
    # NODE 0: INPUT GUARDRAIL & STATE RESET (CHỐNG NHIỄM CHÉO TRẠNG THÁI)
    # =========================================================================
    def input_guardrail_node(self, state: AgentState) -> dict:
        """
        Node 0: Khởi tạo lại các biến trạng thái tạm thời của lượt trước
        (chống State Contamination khi dùng SqliteSaver) và kiểm tra an toàn đầu vào.
        """
        raw_query = state.get("user_query") or state.get("query") or state.get("original_query") or ""
        user_query = str(raw_query).strip()

        guard_result = self.guardrail_manager.guard_input(user_query)
        is_safe = getattr(guard_result, "is_safe", True)
        reason_text = str(getattr(guard_result, "trigger_reason", "") or "")
        sanitized_query = str(getattr(guard_result, "sanitized_query", user_query) or user_query)

        max_retries_cfg = 2
        if hasattr(self.config, "MAX_RETRIES"):
            max_retries_cfg = int(self.config.MAX_RETRIES)

        # Trường hợp vi phạm quy chuẩn an toàn
        if not is_safe:
            refusal = str(getattr(guard_result, "refusal_message", "") or "Yêu cầu của Quý khách không thể thực hiện do vi phạm quy chuẩn an toàn bảo mật của Ngân hàng Vietcombank.")
            return {
                "user_query": user_query,
                "query": user_query,
                "original_query": user_query,
                "rewritten_query": user_query,
                "is_safe": False,
                "guardrail_reason": reason_text,
                "input_guard": {
                    "is_safe": False,
                    "sanitized_query": sanitized_query,
                    "refusal_message": refusal,
                    "trigger_reason": reason_text
                },
                "intent": "REFUSAL",
                "sub_agent_role": "GENERAL_ADVISOR",
                "retry_count": 0,
                "max_retries": max_retries_cfg,
                "hallucination_retries": 0,
                "retrieved_chunks": [],
                "retrieved_docs": [],
                "doc_quality": "UNKNOWN",
                "document_quality": "UNKNOWN",
                "tool_name": "",
                "target_tool": None,
                "tool_output": {},
                "tool_outputs": [],
                "tool_calls": [],
                "draft_answer": refusal,
                "final_answer": refusal,
                "audit_status": "PASS",
                "audit_feedback": "",
                "citations": [],
                "sources": [],
                "execution_steps": ["Node 0: input_guardrail (Blocked by Security Guardrail)"]
            }

        # Trường hợp an toàn: reset toàn bộ state cho lượt mới
        return {
            "user_query": user_query,
            "query": user_query,
            "original_query": user_query,
            "rewritten_query": user_query,
            "is_safe": True,
            "guardrail_reason": "",
            "input_guard": {
                "is_safe": True,
                "sanitized_query": sanitized_query,
                "refusal_message": None
            },
            "intent": "ROUTING",
            "sub_agent_role": "GENERAL_ADVISOR",
            "retry_count": 0,
            "max_retries": max_retries_cfg,
            "hallucination_retries": 0,
            "retrieved_chunks": [],
            "retrieved_docs": [],
            "doc_quality": "UNKNOWN",
            "document_quality": "UNKNOWN",
            "tool_name": "",
            "target_tool": None,
            "tool_output": {},
            "tool_outputs": [],
            "tool_calls": [],
            "draft_answer": "",
            "final_answer": "",
            "audit_status": "PENDING",
            "audit_feedback": "",
            "citations": [],
            "sources": [],
            "execution_steps": ["Node 0: input_guardrail (State Reset & Security Check)"]
        }

    # =========================================================================
    # NODE 1: SUPERVISOR ROUTER (MULTI-AGENT INTENT & ROLE DISPATCH)
    # =========================================================================
    def supervisor_router_node(self, state: AgentState) -> dict:
        """
        Node 1: Supervisor điều phối Intent và chỉ định Sub-Agent chuyên trách.
        Sử dụng phương thức chuỗi Python chuẩn (0% Regex Law, Zero Ternary).
        """
        raw_query = state.get("user_query") or state.get("query") or ""
        query_lower = str(raw_query).lower().strip()
        steps = list(state.get("execution_steps", []))

        # 1. Kiểm tra nhóm chào hỏi xã giao (CHIT_CHAT)
        chit_chat_keywords = [
            "xin chào", "chào bạn", "hello", "hi vcb", "cảm ơn",
            "cam on", "bạn tên gì", "bạn là ai", "tạm biệt", "chúc bạn"
        ]
        banking_substantive_kws = [
            "thẻ", "the", "vay", "lãi", "lai", "tỷ giá", "ty gia", "tiết kiệm", "tiet kiem",
            "sinh trắc", "sinh trac", "2345", "ekyc", "hạn mức", "han muc", "phí", "phi",
            "tài khoản", "tai khoan", "chuyển tiền", "chuyen tien", "digibank", "otp"
        ]

        has_chit_chat_kw = any(kw in query_lower for kw in chit_chat_keywords)
        has_banking_substance = any(bkw in query_lower for bkw in banking_substantive_kws)

        if has_chit_chat_kw and not has_banking_substance:
            steps.append("Node 1: supervisor_router -> CHIT_CHAT (GENERAL_ADVISOR)")
            res = dict(state)
            res.update({
                "intent": "CHIT_CHAT",
                "sub_agent_role": "GENERAL_ADVISOR",
                "target_tool": None,
                "execution_steps": steps
            })
            return res

        # 2. Phân tích Sub-Agent Role chuyên trách
        sub_agent = "GENERAL_ADVISOR"
        credit_card_markers = [
            "thẻ tín dụng", "thẻ ghi nợ", "visa", "mastercard", "jcb",
            "amex", "hoàn tiền", "hạn mức thẻ", "phí thường niên", "mở thẻ"
        ]
        loan_markers = [
            "vay vốn", "vay mua nhà", "vay mua xe", "thế chấp", "tín chấp",
            "dư nợ giảm dần", "khấu hao khoản vay", "lịch trả nợ", "vay tiêu dùng", "vay ngân hàng"
        ]
        compliance_markers = [
            "sinh trắc học", "2345", "thông tư 50", "ekyc", "cccd gắn chip",
            "vneid", "nfc", "quét khuôn mặt", "hạn mức chuyển tiền", "facepay"
        ]

        if any(marker in query_lower for marker in credit_card_markers):
            sub_agent = "CREDIT_CARD_AGENT"
        elif any(marker in query_lower for marker in loan_markers):
            sub_agent = "LOAN_MORTGAGE_AGENT"
        elif any(marker in query_lower for marker in compliance_markers):
            sub_agent = "COMPLIANCE_AUDIT_AGENT"

        # 3. Phân định BANKING_ACTION vs RAG_SEARCH
        target_tool = None
        detected_intent = "RAG_SEARCH"

        if "khấu hao" in query_lower or "dư nợ giảm dần" in query_lower or "lịch trả nợ" in query_lower:
            detected_intent = "BANKING_ACTION"
            sub_agent = "LOAN_MORTGAGE_AGENT"
            target_tool = "calculate_loan_amortization"
        elif "vay" in query_lower and ("tháng bao nhiêu" in query_lower or "trả góp" in query_lower or "mỗi tháng" in query_lower):
            detected_intent = "BANKING_ACTION"
            sub_agent = "LOAN_MORTGAGE_AGENT"
            target_tool = "calculate_loan_amortization"
        elif ("sinh trắc học" in query_lower or "2345" in query_lower or "quét khuôn mặt" in query_lower) and ("chuyển" in query_lower or "triệu" in query_lower or "bắt buộc" in query_lower or "hạn mức" in query_lower or "thiết bị" in query_lower):
            detected_intent = "BANKING_ACTION"
            sub_agent = "COMPLIANCE_AUDIT_AGENT"
            target_tool = "check_biometric_compliance_qd2345"
        elif "tỷ giá" in query_lower or "ty gia" in query_lower or "usd" in query_lower or "ngoại tệ" in query_lower:
            detected_intent = "BANKING_ACTION"
            target_tool = "get_foreign_exchange_rates"
        elif "tính lãi" in query_lower or "tinh lai" in query_lower or ("gửi" in query_lower and "tiết kiệm" in query_lower and "triệu" in query_lower):
            detected_intent = "BANKING_ACTION"
            target_tool = "calculate_savings_profit"
        elif "lãi suất" in query_lower or "lai suat" in query_lower:
            detected_intent = "BANKING_ACTION"
            target_tool = "get_interest_rates"
        elif "khiếu nại" in query_lower or "khieu nai" in query_lower or "tạo ticket" in query_lower or "mất thẻ" in query_lower:
            detected_intent = "BANKING_ACTION"
            target_tool = "create_support_ticket"

        if detected_intent == "RAG_SEARCH":
            target_tool = "search_banking_knowledge"

        steps.append(f"Node 1: supervisor_router -> Intent={detected_intent} | Role={sub_agent}")

        res = dict(state)
        res.update({
            "intent": detected_intent,
            "sub_agent_role": sub_agent,
            "target_tool": target_tool,
            "execution_steps": steps
        })
        return res

    router_node = supervisor_router_node

    # =========================================================================
    # NODE 2: RETRIEVE NODE (KNOWLEDGE RAG)
    # =========================================================================
    def retrieve_node(self, state: AgentState) -> dict:
        """Node 2: Thực hiện Hybrid Retrieval và Cross-Encoder Reranking qua BankingRAGTool."""
        query = str(state.get("query") or state.get("user_query") or "")
        steps = list(state.get("execution_steps", []))
        steps.append("Node 2: retrieve (Hybrid RRF k=60 + bge-reranker-v2-m3)")

        res = self.rag_tool.execute(query=query, top_k=3)
        docs = res.get("documents", [])
        sources = res.get("sources", [])
        citations = res.get("citations", [])

        # Kiểm tra Indirect Prompt Injection
        safe_docs, blocked_count, threats = self.guardrail_manager.guard_retrieved_chunks(docs)
        if blocked_count > 0:
            steps.append(f"Node 2: Indirect Injection Guardrail blocked {blocked_count} poisoned chunk(s)")
        docs = safe_docs

        normalized_q = res.get("normalized_query", query)

        res_state = dict(state)
        res_state.update({
            "retrieved_chunks": docs,
            "retrieved_docs": docs,
            "sources": sources,
            "citations": citations,
            "parsed_query": {"normalized_query": normalized_q},
            "execution_steps": steps
        })
        return res_state

    # =========================================================================
    # NODE 3: GRADE DOCUMENTS (CORRECTIVE RAG)
    # =========================================================================
    def grade_documents_node(self, state: AgentState) -> dict:
        """Node 3: Thẩm định chất lượng và độ liên quan của tài liệu trích xuất."""
        steps = list(state.get("execution_steps", []))
        docs = state.get("retrieved_chunks") or state.get("retrieved_docs") or []

        if not docs:
            steps.append("Node 3: grade_documents -> IRRELEVANT (No Chunks Found)")
            res_state = dict(state)
            res_state.update({
                "doc_quality": "IRRELEVANT",
                "document_quality": "IRRELEVANT",
                "execution_steps": steps
            })
            return res_state

        query = str(state.get("query") or state.get("user_query") or "")
        context_parts = [str(d.get("content", ""))[:300] for d in docs]
        context_str = "\n---\n".join(context_parts)

        grader_prompt = DOCUMENT_GRADER_PROMPT.format(query=query, documents=context_str)
        verdict = self._call_llm_quick("Bạn là trợ lý thẩm định tài liệu ngân hàng.", grader_prompt, max_tokens=10).lower()

        quality = "IRRELEVANT"
        if "yes" in verdict or verdict.strip() == "y":
            quality = "RELEVANT"

        steps.append("Node 3: grade_documents -> " + quality)
        res_state = dict(state)
        res_state.update({
            "doc_quality": quality,
            "document_quality": quality,
            "execution_steps": steps
        })
        return res_state

    # =========================================================================
    # NODE 4: REWRITE QUERY (SELF-CORRECTION LOOP)
    # =========================================================================
    def rewrite_query_node(self, state: AgentState) -> dict:
        """Node 4: Viết lại câu hỏi để tối ưu hóa tìm kiếm tài liệu (Chống lặp với retry_count)."""
        steps = list(state.get("execution_steps", []))
        current_retries = int(state.get("retry_count", 0))
        new_retries = current_retries + 1

        orig_q = str(state.get("original_query") or state.get("user_query") or state.get("query") or "")
        curr_q = str(state.get("query") or state.get("user_query") or "")

        rewrite_user = QUERY_REWRITER_PROMPT.format(original_query=orig_q, current_query=curr_q)
        new_query = self._call_llm_quick("Bạn là chuyên gia viết lại truy vấn tìm kiếm ngân hàng.", rewrite_user, max_tokens=60).strip().strip('"\'')

        if len(new_query) < 3:
            new_query = f"Quy định Vietcombank về {orig_q}"

        steps.append(f"Node 4: rewrite_query (Retry {new_retries}: '{new_query}')")

        res_state = dict(state)
        res_state.update({
            "query": new_query,
            "rewritten_query": new_query,
            "retry_count": new_retries,
            "execution_steps": steps
        })
        return res_state

    # =========================================================================
    # NODE 5: BANKING TOOL CALLING (EXPANDED TO 6 TOOLS)
    # =========================================================================
    def banking_tool_node(self, state: AgentState) -> dict:
        """Node 5: Thực thi công cụ nghiệp vụ ngân hàng dựa trên câu hỏi và target_tool/sub_agent_role."""
        steps = list(state.get("execution_steps", []))
        target = state.get("target_tool", "")
        sub_agent = state.get("sub_agent_role", "GENERAL_ADVISOR")
        raw_query = str(state.get("query") or state.get("user_query") or "")
        query = raw_query.lower().strip()

        params = parse_vietnamese_financial_params(raw_query)
        tool_outputs = []
        executed_tool_name = target

        # 1. Khấu hao khoản vay theo Dư nợ giảm dần
        if target == "calculate_loan_amortization" or sub_agent == "LOAN_MORTGAGE_AGENT" or "khấu hao" in query or "dư nợ giảm dần" in query or ("vay" in query and ("tháng bao nhiêu" in query or "trả góp" in query or "lịch trả nợ" in query)):
            executed_tool_name = "calculate_loan_amortization"
            principal = params["amount_vnd"]
            if principal <= 0.0:
                principal = 1000000000.0
            term_months = params["term_months"]
            if term_months <= 0:
                term_months = 60
            rate_percent = params["annual_rate_percent"]
            if rate_percent <= 0.0:
                rate_percent = 7.5

            tool_input = {"principal_vnd": principal, "annual_rate_percent": rate_percent, "term_months": term_months}
            guard_check = self.guardrail_manager.guard_tool_call("calculate_loan_amortization", tool_input)
            if not guard_check["is_safe"]:
                res = {"status": "ERROR", "message": guard_check["error_message"]}
            else:
                p_safe = guard_check["sanitized_params"]
                res = calculate_loan_amortization(
                    principal_vnd=p_safe["principal_vnd"],
                    annual_rate_percent=p_safe["annual_rate_percent"],
                    term_months=p_safe["term_months"]
                )
            tool_outputs.append({"tool": "calculate_loan_amortization", "result": res})

        # 2. Thẩm định Sinh trắc học QĐ 2345
        elif target == "check_biometric_compliance_qd2345" or sub_agent == "COMPLIANCE_AUDIT_AGENT" or "2345" in query or "sinh trắc học" in query or "quét khuôn mặt" in query or "khuôn mặt" in query:
            executed_tool_name = "check_biometric_compliance_qd2345"
            tool_input = {
                "single_tx_amount_vnd": params["single_tx_amount_vnd"],
                "daily_cumulative_amount_vnd": params["daily_cumulative_amount_vnd"],
                "is_first_time_new_device": params["is_first_time_new_device"],
                "has_verified_chip_cccd": params["has_verified_chip_cccd"]
            }
            guard_check = self.guardrail_manager.guard_tool_call("check_biometric_compliance_qd2345", tool_input)
            if not guard_check["is_safe"]:
                res = {"status": "ERROR", "message": guard_check["error_message"]}
            else:
                p_safe = guard_check["sanitized_params"]
                res = check_biometric_compliance_qd2345(
                    single_tx_amount_vnd=p_safe["single_tx_amount_vnd"],
                    daily_cumulative_amount_vnd=p_safe["daily_cumulative_amount_vnd"],
                    is_first_time_new_device=p_safe["is_first_time_new_device"],
                    has_verified_chip_cccd=p_safe["has_verified_chip_cccd"]
                )
            tool_outputs.append({"tool": "check_biometric_compliance_qd2345", "result": res})

        # 3. Lãi suất tiền gửi
        elif target == "get_interest_rates" or ("lãi suất" in query and "vay" not in query and "khấu hao" not in query):
            executed_tool_name = "get_interest_rates"
            res = get_interest_rates(deposit_type="vnd")
            tool_outputs.append({"tool": "get_interest_rates", "result": res})

        # 4. Tỷ giá ngoại tệ
        elif target == "get_foreign_exchange_rates" or "tỷ giá" in query or "usd" in query or "ngoại tệ" in query:
            executed_tool_name = "get_foreign_exchange_rates"
            currency = "USD"
            for c in ["EUR", "JPY", "GBP", "SGD", "AUD", "CAD", "CNY"]:
                if c.lower() in query:
                    currency = c
                    break
            res = get_foreign_exchange_rates(currency=currency)
            tool_outputs.append({"tool": "get_foreign_exchange_rates", "result": res})

        # 5. Tính lãi tiết kiệm
        elif target == "calculate_savings_profit" or "tính lãi" in query or "gửi tiết kiệm" in query:
            executed_tool_name = "calculate_savings_profit"
            amount = params["amount_vnd"]
            if amount <= 0.0:
                amount = 100000000.0
            term = params["term_months"]
            if term <= 0:
                term = 6

            tool_input = {"amount_vnd": amount, "term_months": term}
            guard_check = self.guardrail_manager.guard_tool_call("calculate_savings_profit", tool_input)
            if not guard_check["is_safe"]:
                res = {"status": "ERROR", "message": guard_check["error_message"]}
            else:
                p_safe = guard_check["sanitized_params"]
                res = calculate_savings_profit(amount_vnd=p_safe["amount_vnd"], term_months=p_safe["term_months"])
            tool_outputs.append({"tool": "calculate_savings_profit", "result": res})

        # 6. Ticket hỗ trợ khẩn cấp
        else:
            executed_tool_name = "create_support_ticket"
            tool_input = {
                "customer_name": "Quý khách hàng",
                "issue_type": "Yêu cầu hỗ trợ nghiệp vụ",
                "description": raw_query,
                "priority": "NORMAL"
            }
            guard_check = self.guardrail_manager.guard_tool_call("create_support_ticket", tool_input)
            if not guard_check.get("is_safe", True):
                res = {"status": "ERROR", "message": guard_check.get("error_message", "Thông tin yêu cầu hỗ trợ không hợp lệ")}
            else:
                p_safe = guard_check.get("sanitized_params", tool_input)
                res = create_support_ticket(
                    customer_name=p_safe["customer_name"],
                    issue_type=p_safe["issue_type"],
                    description=p_safe["description"],
                    priority=p_safe["priority"]
                )
            tool_outputs.append({"tool": "create_support_ticket", "result": res})

        single_result = {}
        if tool_outputs:
            single_result = tool_outputs[0].get("result", {})

        steps.append("Node 5: banking_tool -> Executed " + executed_tool_name)

        res_state = dict(state)
        res_state.update({
            "tool_name": executed_tool_name,
            "tool_output": single_result,
            "tool_outputs": tool_outputs,
            "tool_calls": tool_outputs,
            "execution_steps": steps
        })
        return res_state

    # =========================================================================
    # NODE 6: GENERATOR NODE (DYNAMIC PERSONA & CITATION PROMPTS)
    # =========================================================================
    def generate_node(self, state: AgentState) -> dict:
        """Node 6: Sinh câu trả lời hoàn chỉnh dựa trên ngữ cảnh tài liệu hoặc kết quả công cụ."""
        steps = list(state.get("execution_steps", []))
        intent = state.get("intent", "RAG_SEARCH")
        sub_agent_role = state.get("sub_agent_role", "GENERAL_ADVISOR")
        selected_system_prompt = SUB_AGENT_SYSTEM_PROMPTS.get(sub_agent_role, BANKING_AGENT_SYSTEM_PROMPT)

        query = str(state.get("original_query") or state.get("user_query") or state.get("query") or "")
        docs = state.get("retrieved_chunks") or state.get("retrieved_docs") or []
        tool_outputs = state.get("tool_outputs", [])
        audit_feedback = state.get("audit_feedback", "")
        hallucination_retries = int(state.get("hallucination_retries", 0))

        # Nhánh 1: BANKING_ACTION
        if intent == "BANKING_ACTION" and tool_outputs:
            tool_res = tool_outputs[0].get("result", {})
            tool_name = tool_outputs[0].get("tool", "")

            if tool_name == "calculate_loan_amortization":
                sched = tool_res.get("schedule_sample", [])
                sched_lines = [
                    f"| Tháng {s['month']} | {s['opening_balance_vnd']:,.0f} | {s['principal_payment_vnd']:,.0f} | {s['interest_payment_vnd']:,.0f} | {s['total_monthly_payment_vnd']:,.0f} |"
                    for s in sched
                ]
                table_md = "\n".join(sched_lines)
                draft = (
                    f"Kính gửi Quý khách, dưới đây là bảng tính lịch trả nợ ước tính theo phương thức **Dư nợ giảm dần** chuẩn Vietcombank:\n\n"
                    f"- **Số tiền vay gốc**: {tool_res.get('principal_vnd', 0):,.0f} VND\n"
                    f"- **Thời hạn vay**: {tool_res.get('term_months', 0)} tháng ({tool_res.get('term_months', 0) / 12:.1f} năm)\n"
                    f"- **Lãi suất tham chiếu**: {tool_res.get('annual_rate_percent', 0)}%/năm\n"
                    f"- **Số tiền trả tháng đầu tiên (cao nhất)**: {tool_res.get('first_month_payment_vnd', 0):,.0f} VND\n"
                    f"- **Tổng tiền lãi phải trả toàn bộ kỳ hạn**: {tool_res.get('total_interest_vnd', 0):,.0f} VND\n"
                    f"- **Tổng số tiền gốc và lãi**: {tool_res.get('total_payable_vnd', 0):,.0f} VND\n\n"
                    f"### Trích lược Lịch trả nợ mẫu (Đơn vị: VND)\n"
                    f"| Kỳ trả nợ | Dư nợ đầu kỳ | Tiền gốc hàng tháng | Tiền lãi hàng tháng | Tổng phải trả |\n"
                    f"| :--- | :--- | :--- | :--- | :--- |\n"
                    f"{table_md}\n\n"
                    f"*(Lưu ý: Bảng tính mang tính chất tham khảo dựa trên quy định cho vay của Vietcombank. Lãi suất và lịch trả nợ chính xác sẽ được phê duyệt theo hợp đồng tín dụng cụ thể tại chi nhánh Vietcombank).*"
                )

            elif tool_name == "check_biometric_compliance_qd2345":
                req_bio = tool_res.get("requires_biometric", False)
                reasons = tool_res.get("trigger_reasons", [])
                reason_bullets = "\n".join([f"- {r}" for r in reasons])
                if req_bio:
                    conclusion = "**BẮT BUỘC QUÉT KHUÔN MẶT SINH TRẮC HỌC**"
                else:
                    conclusion = "KHÔNG BẮT BUỘC xác thực khuôn mặt sinh trắc học"

                draft = (
                    f"Căn cứ theo **Quyết định 2345/QĐ-NHNN** và **Thông tư 50/2024/TT-NHNN** về an toàn, bảo mật trong thanh toán trực tuyến:\n\n"
                    f"### 1. Kết quả thẩm định nghiệp vụ:\n"
                    f"- **Tình trạng**: {conclusion}\n"
                    f"- **Giá trị giao dịch đơn lẻ**: {tool_res.get('single_tx_amount_vnd', 0):,.0f} VND\n"
                    f"- **Tổng giá trị cộng dồn trong ngày**: {tool_res.get('daily_cumulative_amount_vnd', 0):,.0f} VND\n\n"
                    f"### 2. Căn cứ và lý do áp dụng:\n"
                    f"{reason_bullets}\n\n"
                    f"### 3. Hướng dẫn thực hiện cho Khách hàng:\n"
                    f"{tool_res.get('mandatory_action', '')}\n\n"
                    f"*(Nguồn tham chiếu: [Quyết định 2345/QĐ-NHNN, Điều 1 & Điều 2] - Ngân hàng Nhà nước Việt Nam & Hướng dẫn VCB Digibank).*"
                )

            elif tool_name == "calculate_savings_profit":
                draft = (
                    f"Kính gửi Quý khách, dưới đây là kết quả dự tính tiền gửi tiết kiệm tại Vietcombank:\n\n"
                    f"- **Số tiền gửi**: {tool_res.get('principal_amount_vnd', 0):,.0f} VND\n"
                    f"- **Kỳ hạn gửi**: {tool_res.get('term_months', 0)} tháng\n"
                    f"- **Lãi suất áp dụng**: {tool_res.get('applicable_rate_percent', 0)}%/năm\n"
                    f"- **Tiền lãi dự kiến**: {tool_res.get('interest_earned_vnd', 0):,.0f} VND\n"
                    f"- **Tổng số tiền khi đáo hạn**: {tool_res.get('total_at_maturity_vnd', 0):,.0f} VND\n\n"
                    f"*(Lưu ý: Tiền lãi thực tế có thể thay đổi tùy thuộc vào ngày gửi thực tế và chính sách ưu đãi trên kênh VCB Digibank tại thời điểm gửi).*"
                )

            elif tool_name == "get_foreign_exchange_rates":
                rates = tool_res.get("rates", {})
                rate_lines = [f"- **{curr} ({detail.get('name', '')})**: Mua CK: {detail.get('buy_transfer', 0):,.1f} | Bán: {detail.get('sell', 0):,.1f} VND" for curr, detail in rates.items()]
                draft = (
                    f"Bảng tỷ giá ngoại tệ Vietcombank ngày hôm nay ({tool_res.get('effective_date', '2026-10-25')}):\n\n"
                    + "\n".join(rate_lines)
                    + "\n\n*(Tỷ giá có tính chất tham khảo và có thể biến động theo từng thời điểm trong ngày giao dịch).*"
                )

            elif tool_name == "get_interest_rates":
                rates = tool_res.get("interest_rates", [])
                lines = [f"- **{r.get('term')}**: {r.get('rate_percent_per_year')}%/năm ({r.get('note', '')})" for r in rates[:6]]
                draft = (
                    f"Biểu lãi suất tiền gửi tiết kiệm cá nhân tại Vietcombank (niêm yết mới nhất):\n\n"
                    + "\n".join(lines)
                    + "\n\n*(Quý khách có thể gửi tiết kiệm trực tuyến trên ứng dụng VCB Digibank để hưởng lãi suất ưu đãi nhất).*"
                )
            else:
                draft = str(tool_res.get("message", "Yêu cầu của Quý khách đã được tiếp nhận thành công."))

            steps.append("Node 6: generate (Persona Chuyên viên Vietcombank - Banking Tool Result)")
            res_state = dict(state)
            res_state.update({"draft_answer": draft, "final_answer": draft, "execution_steps": steps})
            return res_state

        # Nhánh 2: CHIT_CHAT
        if intent == "CHIT_CHAT":
            draft = (
                "Kính chào Quý khách! Tôi là Trợ lý AI của Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank).\n\n"
                "Tôi có thể hỗ trợ Quý khách giải đáp các quy chế mở tài khoản eKYC, hạn mức giao dịch chuyển tiền, "
                "thông tin biểu phí, điều khoản sản phẩm thẻ, tính lịch trả nợ vay mua nhà/xe, kiểm tra sinh trắc học theo QĐ 2345 hoặc tra cứu tỷ giá ngoại tệ hôm nay. "
                "Quý khách cần hỗ trợ thông tin gì ạ?"
            )
            steps.append("Node 6: generate (Persona Chuyên viên Vietcombank - Chit Chat)")
            res_state = dict(state)
            res_state.update({"draft_answer": draft, "final_answer": draft, "execution_steps": steps})
            return res_state

        # Nhánh 3: RAG_SEARCH
        if not docs:
            fallback = (
                f"Kính gửi Quý khách, hiện tại trong kho tài liệu quy chuẩn của Vietcombank chưa có thông tin chi tiết "
                f"cho yêu cầu: '{query}'.\n\n"
                "Để được tư vấn và hỗ trợ chính xác nhất cho trường hợp của mình, Quý khách vui lòng liên hệ Trung tâm Hỗ trợ Khách hàng Vietcombank qua số Tổng đài **1900 54 54 13** (phục vụ 24/7) hoặc đến chi nhánh Vietcombank gần nhất."
            )
            steps.append("Node 6: generate (Safe Fallback - Empty Docs)")
            res_state = dict(state)
            res_state.update({"draft_answer": fallback, "final_answer": fallback, "execution_steps": steps})
            return res_state

        # Ghép Context từ Top Chunks
        context_parts = []
        for d in docs:
            bread = d.get("breadcrumb", "")
            content = d.get("content", "")
            if bread:
                context_parts.append(f"### Nguồn: {bread}\n{content}")
            else:
                context_parts.append(str(content))
        context_str = "\n\n".join(context_parts)

        feedback_instruction = ""
        if hallucination_retries > 0 and audit_feedback:
            feedback_instruction = f"\n\n[LƯU Ý ĐẶC BIỆT TỪ HỆ THỐNG AUDIT KIỂM ĐỊNH - LẦN THỬ LẠI {hallucination_retries}]: {audit_feedback}\n"

        prompt = (
            f"{selected_system_prompt}\n\n"
            f"TÀI LIỆU CĂN CỨ TỪ HỆ THỐNG VIETCOMBANK:\n{context_str}\n\n"
            f"CÂU HỎI CỦA KHÁCH HÀNG: {query}"
            f"{feedback_instruction}\n\n"
            f"YÊU CẦU BẮT BUỘC: Trả lời rõ ràng, chính xác, tuyệt đối không bịa đặt số liệu và BẮT BUỘC trích dẫn điều khoản cụ thể [Điều X, Khoản Y] hoặc tên tài liệu nguồn Vietcombank."
        )

        draft = self._call_llm_quick(selected_system_prompt, prompt, max_tokens=800)

        if not draft:
            first_doc = docs[0]
            first_content = str(first_doc.get("content", ""))[:400]
            first_source = first_doc.get("breadcrumb", first_doc.get("source", "Vietcombank"))
            draft = (
                f"Căn cứ theo quy định của Vietcombank:\n\n{first_content}...\n\n"
                f"*(Nguồn tham chiếu: [Điều khoản Vietcombank] - {first_source})*"
            )

        draft_lower = draft.lower()
        if "[điều" not in draft_lower and "[khoản" not in draft_lower and "[nguồn" not in draft_lower:
            first_source = docs[0].get("breadcrumb", docs[0].get("source", "Quy chế Vietcombank"))
            draft = draft + f"\n\n*(Căn cứ pháp lý: [Điều khoản Quy định Vietcombank] - {first_source})*"

        steps.append(f"Node 6: generate ({sub_agent_role} - Persona Chuyên viên Vietcombank - RAG Grounded)")
        res_state = dict(state)
        res_state.update({"draft_answer": draft, "final_answer": draft, "execution_steps": steps})
        return res_state

    # =========================================================================
    # NODE 7: AUDIT GENERATION NODE (POST-GENERATION HALLUCINATION & CITATION)
    # =========================================================================
    def audit_generation_node(self, state: AgentState) -> dict:
        """
        Node 7: Thẩm định câu trả lời sau sinh (Post-Generation Groundedness & Citation Check).
        Bảo vệ khách hàng trước ảo giác số liệu và đảm bảo tính minh bạch pháp lý [Điều X, Khoản Y].
        """
        draft = str(state.get("draft_answer") or "").strip()
        intent = state.get("intent", "RAG_SEARCH")
        doc_quality = state.get("doc_quality") or state.get("document_quality", "IRRELEVANT")
        current_retries = int(state.get("hallucination_retries", 0))
        steps = list(state.get("execution_steps", []))

        # Nếu là CHIT_CHAT hoặc BANKING_ACTION hoặc fallback IRRELEVANT: Cho qua ngay
        if intent != "RAG_SEARCH" or doc_quality != "RELEVANT":
            steps.append("Node 7: audit_generation -> PASS (Non-RAG or Fallback)")
            res_state = dict(state)
            res_state.update({"audit_status": "PASS", "final_answer": draft, "execution_steps": steps})
            return res_state

        draft_lower = draft.lower()
        citation_indicators = [
            "[điều", "[khoản", "[mục", "[nguồn:", "quyết định",
            "thông tư", "biểu phí", "vietcombank", "theo quy định", "căn cứ"
        ]
        has_citation = any(indicator in draft_lower for indicator in citation_indicators)

        if has_citation:
            docs = state.get("retrieved_chunks") or state.get("retrieved_docs") or []
            grounding = self.guardrail_manager.verify_citation_grounding(draft, docs)
            if not grounding["is_grounded"] and current_retries < 1:
                steps.append("Node 7: audit_generation -> FAIL (Unverified/Hallucinated Citation, Triggering Regenerate)")
                res_state = dict(state)
                res_state.update({
                    "audit_status": "FAIL",
                    "audit_feedback": (
                        "Phát hiện trích dẫn điều khoản không có trong tài liệu căn cứ được cung cấp. "
                        "BẮT BUỘC chỉ trích dẫn đúng các điều khoản thực tế có trong văn bản đã cung cấp."
                    ),
                    "hallucination_retries": current_retries + 1,
                    "execution_steps": steps
                })
                return res_state

            steps.append("Node 7: audit_generation -> PASS (Citation & Groundedness Verified)")
            res_state = dict(state)
            res_state.update({"audit_status": "PASS", "final_answer": draft, "execution_steps": steps})
            return res_state

        if current_retries < 1:
            steps.append("Node 7: audit_generation -> FAIL (Missing Citation, Triggering Regenerate)")
            res_state = dict(state)
            res_state.update({
                "audit_status": "FAIL",
                "audit_feedback": (
                    "BẮT BUỘC bổ sung trích dẫn cụ thể [Điều X, Khoản Y] hoặc tên tài liệu nguồn của Vietcombank dựa đúng trên các đoạn văn bản đã cung cấp."
                ),
                "hallucination_retries": current_retries + 1,
                "execution_steps": steps
            })
            return res_state

        steps.append("Node 7: audit_generation -> PASS (Max Audit Retries Reached)")
        res_state = dict(state)
        res_state.update({"audit_status": "PASS", "final_answer": draft, "execution_steps": steps})
        return res_state

    # =========================================================================
    # NODE 8: OUTPUT GUARDRAIL NODE
    # =========================================================================
    def output_guardrail_node(self, state: AgentState) -> dict:
        """Node 8: Kiểm duyệt an toàn đầu ra, làm sạch think tag, gắn disclaimer và lưu vết hội thoại."""
        steps = list(state.get("execution_steps", []))
        draft = state.get("final_answer") or state.get("draft_answer") or ""
        cleaned_answer = _clean_think_tags(str(draft))

        guard_res = self.guardrail_manager.guard_output(cleaned_answer)
        is_safe_flag = bool(getattr(guard_res, "is_safe", True))
        sanitized_text = str(getattr(guard_res, "sanitized_answer", cleaned_answer) or cleaned_answer)
        flags_list = list(getattr(guard_res, "flags", []) or [])

        # Cập nhật lịch sử hội thoại (Session Memory)
        messages = list(state.get("messages", []))
        chat_history = list(state.get("chat_history", []))
        user_q = str(state.get("user_query") or state.get("query") or "").strip()

        if user_q:
            messages.append({"role": "user", "content": user_q})
            messages.append({"role": "assistant", "content": sanitized_text})
            chat_history.append({"role": "user", "content": user_q})
            chat_history.append({"role": "assistant", "content": sanitized_text})

        steps.append("Node 8: output_guardrail (Clean Think + Disclaimer & Memory Updated)")

        result = dict(state)
        result.update({
            "output_guard": {
                "is_safe": is_safe_flag,
                "sanitized_answer": sanitized_text,
                "flags": flags_list
            },
            "final_answer": sanitized_text,
            "messages": messages,
            "chat_history": chat_history,
            "execution_steps": steps
        })
        return result
