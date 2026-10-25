# -*- coding: utf-8 -*-
"""
Agent Nodes for Banking LangGraph Workflow
======================================================================
Cung cấp các Node thực thi trong đồ thị StateGraph:
1. input_guardrail_node: Kiểm duyệt an toàn đầu vào (Prompt Injection, Jailbreak, PII).
2. router_node: Phân loại ý định (RAG_SEARCH, BANKING_ACTION, CHIT_CHAT).
3. retrieve_node: Bọc gọi RAGTool thực hiện Hybrid Search & Reranking.
4. grade_documents_node: Đánh giá độ liên quan của tài liệu (Corrective RAG).
5. rewrite_query_node: Viết lại câu hỏi để thử lại tìm kiếm (kèm retry_count chống lặp vô tận).
6. banking_tool_node: Thực thi công cụ ngân hàng (Lãi suất, Tỷ giá, Tính tiền lãi, Tạo Ticket).
7. generate_node: Sinh câu trả lời qua LLM kèm văn phong ngân hàng và trích dẫn [Điều X, Khoản Y].
8. output_guardrail_node: Kiểm duyệt an toàn đầu ra, kiểm soát Hallucination và gắn disclaimer.

NGUYÊN TẮC:
- 0% REGEX LAW: Tuyệt đối không dùng regex methods.
- Tái sử dụng trực tiếp các class trong guardrails, tracing, pipeline và tools.
"""

from __future__ import annotations
import time
from typing import Any
from openai import OpenAI

from agent_banking.config import Config
from agent_banking.agents.state import AgentState
from agent_banking.agents.prompts import (
    ROUTER_SYSTEM_PROMPT,
    DOCUMENT_GRADER_PROMPT,
    QUERY_REWRITER_PROMPT,
    ANSWER_GRADER_PROMPT,
    BANKING_AGENT_SYSTEM_PROMPT
)
from agent_banking.guardrails.rag_guardrails import BankingGuardrailManager
from agent_banking.tracing.tracer import BankingTracer
from agent_banking.tools.rag_tool import BankingRAGTool
from agent_banking.tools.banking_api_tools import (
    get_interest_rates,
    get_foreign_exchange_rates,
    calculate_savings_profit,
    create_support_ticket
)


class BankingAgentNodes:
    """
    Tập hợp các hàm Node thực thi cho StateGraph.
    """

    def __init__(self, pipeline_instance=None, config: Config | None = None):
        if config is not None:
            self.config = config
        else:
            self.config = Config()

        # 1. Khởi tạo Guardrail Manager
        self.guardrail_manager = BankingGuardrailManager()

        # 2. Khởi tạo Tracing Manager
        self.tracer = BankingTracer(config=self.config)

        # 3. Khởi tạo RAG Tool
        self.rag_tool = BankingRAGTool(pipeline_instance=pipeline_instance, config=self.config)

        # 4. Khởi tạo LLM Client (OpenAI-compatible FPT Cloud / DeepSeek API)
        self.client = None
        if self.config.LLM_API_KEY and self.config.LLM_BASE_URL:
            try:
                self.client = OpenAI(
                    api_key=self.config.LLM_API_KEY,
                    base_url=self.config.LLM_BASE_URL
                )
            except Exception as e:
                print(f"[Agent Nodes] Cảnh báo khởi tạo LLM Client: {e}")

    # -------------------------------------------------------------------------
    # Helper: Gọi LLM nội bộ tốc độ cao cho Router/Grader/Rewriter
    # -------------------------------------------------------------------------
    def _call_llm_quick(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
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
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            print(f"[Agent Nodes] LLM Quick Call Error: {e}")
            return ""

    # =========================================================================
    # NODE 1: INPUT GUARDRAIL
    # =========================================================================
    def input_guardrail_node(self, state: AgentState) -> dict[str, Any]:
        """
        Kiểm duyệt câu hỏi đầu vào chống Prompt Injection, Jailbreak, rò rỉ PII.
        """
        query = state.get("query", "")
        guard_res = self.guardrail_manager.guard_input(query)

        if not guard_res.is_safe:
            refusal = guard_res.refusal_message or (
                "Yêu cầu của Quý khách không thể thực hiện do vi phạm quy tắc an toàn bảo mật "
                "thông tin của Ngân hàng Vietcombank. Vui lòng đặt câu hỏi phù hợp."
            )
            return {
                "input_guard": {
                    "is_safe": False,
                    "sanitized_query": guard_res.sanitized_query,
                    "refusal_message": refusal,
                    "trigger_reason": guard_res.trigger_reason
                },
                "intent": "REFUSAL",
                "draft_answer": refusal,
                "final_answer": refusal
            }

        return {
            "input_guard": {
                "is_safe": True,
                "sanitized_query": guard_res.sanitized_query,
                "refusal_message": None
            },
            "query": guard_res.sanitized_query,
            "original_query": state.get("original_query", guard_res.sanitized_query)
        }

    # =========================================================================
    # NODE 2: INTENT ROUTER
    # =========================================================================
    def router_node(self, state: AgentState) -> dict[str, Any]:
        """
        Định tuyến ý định: RAG_SEARCH vs BANKING_ACTION vs CHIT_CHAT (0% Regex Law).
        """
        query_lower = state.get("query", "").lower()

        # Heuristic 1: Nhận diện Hành động Ngân hàng (Action Tools)
        action_keywords = [
            "lãi suất", "lai suat", "tỷ giá", "ty gia", "usd", "ngoại tệ", "ngoai te",
            "tính lãi", "tinh lai", "gửi tiết kiệm", "gui tiet kiem",
            "khiếu nại", "khieu nai", "tạo ticket", "tao ticket", "mất thẻ", "mat the"
        ]
        is_action = False
        for kw in action_keywords:
            if kw in query_lower:
                is_action = True
                break

        if is_action:
            target_tool = "general_banking"
            if "lãi suất" in query_lower or "lai suat" in query_lower:
                target_tool = "get_interest_rates"
            if "tỷ giá" in query_lower or "ty gia" in query_lower or "usd" in query_lower or "ngoại tệ" in query_lower:
                target_tool = "get_foreign_exchange_rates"
            if "tính lãi" in query_lower or "tinh lai" in query_lower or "gửi tiết kiệm" in query_lower:
                target_tool = "calculate_savings_profit"
            if "khiếu nại" in query_lower or "khieu nai" in query_lower or "mất thẻ" in query_lower:
                target_tool = "create_support_ticket"

            return {
                "intent": "BANKING_ACTION",
                "target_tool": target_tool
            }

        # Heuristic 2: Nhận diện Chào hỏi (Chit-chat)
        chitchat_phrases = ["xin chào", "chào bạn", "hello", "hi", "cảm ơn", "cam on", "tên bạn là gì", "bạn là ai"]
        is_chitchat = False
        for cc in chitchat_phrases:
            if query_lower == cc or query_lower.startswith(cc + " ") or query_lower.endswith(" " + cc):
                is_chitchat = True
                break

        if is_chitchat:
            return {
                "intent": "CHIT_CHAT",
                "target_tool": None
            }

        # Heuristic 3: Nhận diện RAG Tri thức (eKYC, thẻ, hạn mức, hợp đồng, biểu phí)
        rag_keywords = [
            "ekyc", "hạn mức", "han muc", "thẻ", "the", "phí", "phi",
            "điều khoản", "dieu khoan", "digibank", "mở tài khoản", "mo tai khoan",
            "otp", "smart otp", "quy trình", "quy trinh", "vcb"
        ]
        for rk in rag_keywords:
            if rk in query_lower:
                return {
                    "intent": "RAG_SEARCH",
                    "target_tool": "search_banking_knowledge"
                }

        # Phân loại bổ trợ bằng LLM nếu không khớp từ khóa
        llm_intent = self._call_llm_quick(
            ROUTER_SYSTEM_PROMPT,
            f"Phân loại câu hỏi sau: '{state.get('query', '')}'"
        ).strip().upper()

        if "BANKING_ACTION" in llm_intent:
            classified_intent = "BANKING_ACTION"
        elif "CHIT_CHAT" in llm_intent:
            classified_intent = "CHIT_CHAT"
        else:
            classified_intent = "RAG_SEARCH"

        return {
            "intent": classified_intent,
            "target_tool": "search_banking_knowledge" if classified_intent == "RAG_SEARCH" else None
        }

    # =========================================================================
    # NODE 3: RETRIEVE NODE (KNOWLEDGE RAG)
    # =========================================================================
    def retrieve_node(self, state: AgentState) -> dict[str, Any]:
        """
        Thực hiện Hybrid Retrieval và Cross-Encoder Reranking qua BankingRAGTool.
        """
        query = state.get("query", "")
        res = self.rag_tool.execute(query=query, top_k=3)

        return {
            "retrieved_docs": res.get("documents", []),
            "sources": res.get("sources", []),
            "citations": res.get("citations", []),
            "parsed_query": {
                "normalized_query": res.get("normalized_query", query)
            }
        }

    # =========================================================================
    # NODE 4: GRADE DOCUMENTS (CORRECTIVE RAG)
    # =========================================================================
    def grade_documents_node(self, state: AgentState) -> dict[str, Any]:
        """
        Thẩm định chất lượng và độ liên quan của tài liệu trích xuất.
        """
        docs = state.get("retrieved_docs", [])
        if not docs:
            return {"document_quality": "IRRELEVANT"}

        query = state.get("query", "")
        context_str = "\n---\n".join([d.get("content", "")[:300] for d in docs])

        # Đánh giá nhanh bằng LLM Grader
        grader_prompt = DOCUMENT_GRADER_PROMPT.format(query=query, documents=context_str)
        verdict = self._call_llm_quick(
            "Bạn là trợ lý thẩm định tài liệu ngân hàng.",
            grader_prompt,
            max_tokens=10
        ).lower()

        if "yes" in verdict or "y" == verdict.strip():
            quality = "RELEVANT"
        else:
            quality = "IRRELEVANT"

        return {"document_quality": quality}

    # =========================================================================
    # NODE 5: REWRITE QUERY (SELF-CORRECTION LOOP)
    # =========================================================================
    def rewrite_query_node(self, state: AgentState) -> dict[str, Any]:
        """
        Viết lại câu hỏi để tối ưu hóa tìm kiếm tài liệu (Chống lặp với retry_count).
        """
        current_retries = state.get("retry_count", 0)
        new_retries = current_retries + 1

        orig_q = state.get("original_query", state.get("query", ""))
        curr_q = state.get("query", "")

        rewrite_user = QUERY_REWRITER_PROMPT.format(
            original_query=orig_q,
            current_query=curr_q
        )
        new_query = self._call_llm_quick(
            "Bạn là chuyên gia viết lại truy vấn tìm kiếm ngân hàng.",
            rewrite_user,
            max_tokens=60
        ).strip().strip('"\'')

        if not new_query or len(new_query) < 3:
            new_query = f"Quy định Vietcombank về {orig_q}"

        print(f"🔄 [LangGraph Corrective RAG] Viết lại câu hỏi lần {new_retries}: '{new_query}'")

        return {
            "query": new_query,
            "retry_count": new_retries
        }

    # =========================================================================
    # NODE 6: BANKING TOOL CALLING
    # =========================================================================
    def banking_tool_node(self, state: AgentState) -> dict[str, Any]:
        """
        Thực thi công cụ nghiệp vụ ngân hàng dựa trên câu hỏi và target_tool.
        """
        target = state.get("target_tool", "")
        query = state.get("query", "").lower()
        tool_outputs = []

        if target == "get_interest_rates" or "lãi suất" in query or "lai suat" in query:
            res = get_interest_rates(deposit_type="vnd")
            tool_outputs.append({"tool": "get_interest_rates", "result": res})

        elif target == "get_foreign_exchange_rates" or "tỷ giá" in query or "usd" in query:
            currency = "USD"
            for c in ["EUR", "JPY", "GBP", "SGD", "AUD", "CAD", "CNY"]:
                if c.lower() in query:
                    currency = c
                    break
            res = get_foreign_exchange_rates(currency=currency)
            tool_outputs.append({"tool": "get_foreign_exchange_rates", "result": res})

        elif target == "calculate_savings_profit" or "tính lãi" in query or "gửi" in query:
            # Trích xuất số tiền và kỳ hạn cơ bản
            amount = 100000000.0  # Mặc định 100 triệu VND nếu không nhận diện được
            term = 6              # Mặc định 6 tháng
            
            # Phân tách số tiền đơn giản (0% Regex)
            tokens = query.split()
            for i, tok in enumerate(tokens):
                tok_clean = tok.replace(".", "").replace(",", "")
                if tok_clean.isdigit():
                    val = float(tok_clean)
                    if i + 1 < len(tokens):
                        next_tok = tokens[i + 1]
                        if "triệu" in next_tok or "trieu" in next_tok or "tr" in next_tok:
                            amount = val * 1000000.0
                        elif "tỷ" in next_tok or "ty" in next_tok:
                            amount = val * 1000000000.0
                        elif "tháng" in next_tok or "thang" in next_tok:
                            term = int(val)

            res = calculate_savings_profit(amount_vnd=amount, term_months=term)
            tool_outputs.append({"tool": "calculate_savings_profit", "result": res})

        else:
            # Mặc định: Tạo ticket hỗ trợ khách hàng
            res = create_support_ticket(
                customer_name="Quý khách hàng",
                issue_type="Yêu cầu hỗ trợ nghiệp vụ",
                description=state.get("query", ""),
                priority="NORMAL"
            )
            tool_outputs.append({"tool": "create_support_ticket", "result": res})

        return {"tool_outputs": tool_outputs}

    # =========================================================================
    # NODE 7: GENERATOR NODE
    # =========================================================================
    def generate_node(self, state: AgentState) -> dict[str, Any]:
        """
        Sinh câu trả lời hoàn chỉnh dựa trên ngữ cảnh tài liệu hoặc kết quả công cụ.
        """
        intent = state.get("intent", "RAG_SEARCH")
        query = state.get("original_query", state.get("query", ""))
        docs = state.get("retrieved_docs", [])
        tool_outputs = state.get("tool_outputs", [])

        # Trường hợp 1: Nhánh công cụ nghiệp vụ
        if intent == "BANKING_ACTION" and tool_outputs:
            tool_res = tool_outputs[0].get("result", {})
            tool_name = tool_outputs[0].get("tool", "")

            if tool_name == "calculate_savings_profit":
                draft = (
                    f"Kính gửi Quý khách, dưới đây là kết quả dự tính tiền gửi tiết kiệm tại Vietcombank:\n\n"
                    f"- **Số tiền gửi**: {tool_res.get('principal_amount_vnd', 0):,.0f} VND\n"
                    f"- **Kỳ hạn gửi**: {tool_res.get('term_months', 0)} tháng\n"
                    f"- **Lãi suất áp dụng**: {tool_res.get('applicable_rate_percent', 0)}%/năm\n"
                    f"- **Tiền lãi dự kiến**: {tool_res.get('interest_earned_vnd', 0):,.0f} VND\n"
                    f"- **Tổng số tiền khi đáo hạn**: {tool_res.get('total_at_maturity_vnd', 0):,.0f} VND\n\n"
                    f"*(Lưu ý: Tiền lãi thực tế có thể thay đổi tùy thuộc vào ngày gửi thực tế và chính sách ưu đãi trên kênh VCB Digibank tại thời điểm gửi).* "
                )
            elif tool_name == "get_foreign_exchange_rates":
                rates = tool_res.get("rates", {})
                rate_lines = []
                for curr, detail in rates.items():
                    rate_lines.append(f"- **{curr} ({detail.get('name', '')})**: Mua CK: {detail.get('buy_transfer', 0):,.1f} | Bán: {detail.get('sell', 0):,.1f} VND")
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
                draft = tool_res.get("message", "Yêu cầu của Quý khách đã được tiếp nhận thành công.")

            return {"draft_answer": draft}

        # Trường hợp 2: Chào hỏi xã giao (Chit-chat)
        if intent == "CHIT_CHAT":
            draft = (
                "Kính chào Quý khách! Tôi là Trợ lý AI của Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank).\n\n"
                "Tôi có thể hỗ trợ Quý khách giải đáp các quy chế mở tài khoản eKYC, hạn mức giao dịch chuyển tiền, "
                "thông tin biểu phí, điều khoản sản phẩm thẻ, tính lãi tiết kiệm hoặc tra cứu tỷ giá ngoại tệ hôm nay. "
                "Quý khách cần hỗ trợ thông tin gì ạ?"
            )
            return {"draft_answer": draft}

        # Trường hợp 3: RAG Search - Trả lời từ tài liệu
        if not docs:
            fallback = (
                f"Kính gửi Quý khách, hiện tại trong kho tài liệu quy chuẩn của Vietcombank chưa có thông tin chi tiết "
                f"cho yêu cầu: '{query}'.\n\n"
                "Để được tư vấn và hỗ trợ chính xác nhất cho trường hợp của mình, Quý khách vui lòng liên hệ Trung tâm Hỗ trợ Khách hàng Vietcombank qua số Tổng đài **1900 54 54 13** (phục vụ 24/7) hoặc đến chi nhánh Vietcombank gần nhất."
            )
            return {"draft_answer": fallback}

        # Chuẩn bị Context Markdown từ Top Chunks
        context_parts = []
        for d in docs:
            bread = d.get("breadcrumb", "")
            content = d.get("content", "")
            if bread:
                context_parts.append(f"### Nguồn: {bread}\n{content}")
            else:
                context_parts.append(content)
        context_str = "\n\n".join(context_parts)

        # Gọi LLM sinh câu trả lời căn cứ
        prompt = (
            f"{BANKING_AGENT_SYSTEM_PROMPT}\n\n"
            f"TÀI LIỆU CĂN CỨ TỪ HỆ THỐNG VIETCOMBANK:\n{context_str}\n\n"
            f"CÂU HỎI CỦA KHÁCH HÀNG: {query}\n\n"
            f"Hãy trả lời rõ ràng, chính xác, có căn cứ và trích dẫn điều khoản pháp lý cụ thể [Điều X, Khoản Y]."
        )

        draft = self._call_llm_quick(
            "Bạn là trợ lý AI chuyên nghiệp của Vietcombank.",
            prompt,
            max_tokens=800
        )

        if not draft:
            draft = (
                f"Căn cứ theo quy định của Vietcombank:\n\n{docs[0].get('content', '')[:400]}...\n\n"
                f"*(Nguồn tham chiếu: {docs[0].get('breadcrumb', docs[0].get('source', 'Vietcombank'))})*"
            )

        return {"draft_answer": draft}

    # =========================================================================
    # NODE 8: OUTPUT GUARDRAIL
    # =========================================================================
    def output_guardrail_node(self, state: AgentState) -> dict[str, Any]:
        """
        Kiểm duyệt an toàn đầu ra, loại bỏ PII và áp dụng Banking Disclaimers.
        """
        draft = state.get("draft_answer") or state.get("final_answer", "")
        context_docs = state.get("retrieved_docs", [])
        context_str = " ".join([d.get("content", "") for d in context_docs])

        guard_res = self.guardrail_manager.guard_output(draft)

        return {
            "output_guard": {
                "is_safe": guard_res.is_safe,
                "sanitized_answer": guard_res.sanitized_answer,
                "flags": guard_res.flags
            },
            "final_answer": guard_res.sanitized_answer
        }
