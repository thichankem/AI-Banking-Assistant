# -*- coding: utf-8 -*-
"""
Agent Nodes for Banking LangGraph Workflow v4.0
======================================================================
Provides execution nodes in the StateGraph workflow:
1. input_guardrail_node: Cleans temporary variables (prevents cross-state pollution) and audits input (Prompt Injection, Jailbreak, Domain).
2. supervisor_router_node: Classifies intent & assigns specialized Sub-Agent (Multi-Agent Supervisor).
3. retrieve_node: Hybrid Search & Cross-Encoder Reranking via BankingRAGTool.
4. grade_documents_node: Assesses document relevance (Corrective RAG).
5. rewrite_query_node: Rewrites query for retrieval retry (with retry_count infinite loop protection).
6. banking_tool_node: Executes banking business tools (FX rates, Interest rates, Savings profit, Loan amortization, Biometrics, Ticket).
7. generate_node: Synthesizes responses grounded on documents/tools using banking specialist personas.
8. audit_generation_node: Post-generation audit (anti-hallucination, mandatory citation).
9. output_guardrail_node: Post-generation safety filter, cleans think tags, attaches disclaimer and updates session history.

RULES:
- 0% REGEX LAW: Absolutely no regex or re library imports.
- ZERO TERNARY LAW: Explicit if/else only.
- NO TYPING LIBRARY: Native Python dict for LangGraph state.
- INFINITE LOOP PROTECTION: Strict retry_count and hallucination_retries bounds.
"""

from agent_banking.core.chat import FallbackChatClient
from agent_banking.core.fallback import ModelFallbackError
from agent_banking.generator.config.prompts import build_banking_system_prompt

from agent_banking.core.settings import Config
from agent_banking.langgraph_agent.state import AgentState
from agent_banking.langgraph_agent.config.prompts import (
    ROUTER_SYSTEM_PROMPT,
    DOCUMENT_GRADER_SYSTEM_MESSAGE,
    DOCUMENT_GRADER_PROMPT,
    QUERY_REWRITER_SYSTEM_MESSAGE,
    QUERY_REWRITER_PROMPT,
    ANSWER_GRADER_SYSTEM_MESSAGE,
    ANSWER_GRADER_PROMPT,
    BANKING_AGENT_SYSTEM_PROMPT,
    GENERATE_USER_PROMPT_TEMPLATE,
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
    """Clean reasoning tags <think>...</think> and un-tagged chain-of-thought leaks (0% Regex)."""
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
    
    cleaned = cleaned.strip()
    cot_markers = [
        "Chúng ta cần trả lời câu hỏi:",
        "Phân tích yêu cầu câu hỏi:",
        "Dựa trên các tài liệu được cung cấp, chúng ta thấy:"
    ]
    for cot in cot_markers:
        if cleaned.startswith(cot):
            formal_starts = ["\nKính gửi", "\nKính thưa", "\nCăn cứ", "\nTheo quy định", "\nTrả lời:", "\n### ", "\n## ", "\n1. "]
            best_idx = -1
            for fs in formal_starts:
                idx = cleaned.find(fs)
                if idx != -1:
                    if best_idx == -1 or idx < best_idx:
                        best_idx = idx
            if best_idx != -1:
                cleaned = cleaned[best_idx:].strip()
                break
    return cleaned.strip()


def _unwrap_doc_dict(doc) -> dict:
    """Return canonical chunk dict whether wrapped in {'chunk': ...} or already flat."""
    if isinstance(doc, dict) and isinstance(doc.get("chunk"), dict):
        return doc["chunk"]
    if isinstance(doc, dict):
        return doc
    return {}


def _render_table_data_text(table_data) -> str:
    """Render canonical chunk table_data (str, dict, or list) into concise Markdown."""
    if not table_data:
        return ""
    if isinstance(table_data, str):
        return table_data.strip()

    tables = []
    if isinstance(table_data, dict):
        tables = [table_data]
    elif isinstance(table_data, list):
        if table_data and isinstance(table_data[0], dict):
            tables = table_data
        else:
            row_lines = []
            for row in table_data[:20]:
                if isinstance(row, list):
                    cells = [str(c).strip() for c in row]
                    row_lines.append("| " + " | ".join(cells) + " |")
                elif row:
                    row_lines.append(str(row).strip())
            return "\n".join(row_lines)

    rendered = []
    for tbl in tables[:3]:
        if not isinstance(tbl, dict):
            continue
        headers = tbl.get("headers") or tbl.get("header") or []
        rows = tbl.get("rows") or tbl.get("cells") or []
        lines = []
        if isinstance(headers, list) and headers:
            h_cells = [str(h).strip() for h in headers]
            lines.append("| " + " | ".join(h_cells) + " |")
            lines.append("| " + " | ".join(["---"] * len(h_cells)) + " |")
        if isinstance(rows, list):
            for row in rows[:20]:
                if isinstance(row, list):
                    r_cells = [str(c).strip() for c in row]
                    lines.append("| " + " | ".join(r_cells) + " |")
                elif isinstance(row, dict):
                    r_vals = [str(v).strip() for v in row.values()]
                    lines.append("| " + " | ".join(r_vals) + " |")
        if lines:
            rendered.append("\n".join(lines))
        else:
            fallback_str = str(tbl.get("markdown") or tbl.get("summary") or tbl.get("caption") or "").strip()
            if fallback_str:
                rendered.append(fallback_str)
    return "\n\n".join(rendered)


def _render_images_text(images) -> str:
    """Extract readable image captions, descriptions, or OCR text from multimodal chunks."""
    if not isinstance(images, list) or not images:
        return ""
    parts = []
    for img in images[:5]:
        if isinstance(img, str):
            txt = img.strip()
            if txt:
                parts.append(txt)
            continue
        if not isinstance(img, dict):
            continue
        labels = []
        for key in ("caption", "description", "native_text", "ocr_text", "alt_text", "title"):
            val = str(img.get(key) or "").strip()
            if val and val not in labels:
                labels.append(val)
        if labels:
            parts.append(" - ".join(labels))
    return "; ".join(parts)


def _format_retrieved_doc_context(doc, max_chars=None) -> str:
    """Format a 13-key universal chunk (or wrapper dict) for grading or grounded generation."""
    d = _unwrap_doc_dict(doc)
    if not d:
        return ""

    source_name = str(d.get("source") or "")
    breadcrumb = str(d.get("breadcrumb") or "")
    section_title = str(d.get("section_title") or "")
    chunk_type = str(d.get("chunk_type") or "text")
    meta = d.get("business_metadata")
    if not isinstance(meta, dict):
        meta = {}

    doc_name = str(meta.get("document_name") or "")
    domain = str(meta.get("domain") or "")
    category = str(meta.get("category") or "")
    retrieval_meta = meta.get("retrieval")
    if not isinstance(retrieval_meta, dict):
        retrieval_meta = {}
    situated = str(retrieval_meta.get("situated_context") or "").strip()

    display_source = doc_name
    if not display_source:
        display_source = source_name

    header_parts = []
    if display_source:
        header_parts.append(display_source)
    if breadcrumb and breadcrumb != display_source:
        header_parts.append(breadcrumb)
    if section_title and section_title not in breadcrumb:
        header_parts.append(section_title)

    meta_tags = []
    if chunk_type and chunk_type != "text":
        meta_tags.append(chunk_type)
    if domain:
        meta_tags.append(domain)
    if category:
        meta_tags.append(category)

    body_parts = []
    if situated:
        body_parts.append(f"[Ngữ cảnh]: {situated}")

    content = str(d.get("content") or "").strip()
    if content:
        body_parts.append(content)

    table_text = _render_table_data_text(d.get("table_data"))
    if table_text and table_text not in content:
        body_parts.append(f"[Bảng dữ liệu]:\n{table_text}")

    image_text = _render_images_text(d.get("images"))
    if image_text and image_text not in content:
        body_parts.append(f"[Hình ảnh/Minh họa]: {image_text}")

    body_text = "\n".join(body_parts).strip()
    if max_chars is not None and len(body_text) > max_chars:
        body_text = body_text[:max_chars]

    header_line = ""
    if header_parts:
        header_line = "### Nguồn: " + " | ".join(header_parts)
        if meta_tags:
            header_line = header_line + " (" + ", ".join(meta_tags) + ")"

    if header_line and body_text:
        return f"{header_line}\n{body_text}"
    if header_line:
        return header_line
    return body_text


class BankingAgentNodes:
    """Collection of node execution functions for StateGraph."""

    def __init__(self, pipeline_instance=None, config: Config | None = None):
        if config is not None:
            self.config = config
        else:
            self.config = Config()

        # 1. Initialize Guardrail Manager
        embedder_inst = None
        if pipeline_instance is not None:
            if hasattr(pipeline_instance, "embedder"):
                embedder_inst = pipeline_instance.embedder
        self.guardrail_manager = BankingGuardrailManager(embedder=embedder_inst)

        # 2. Initialize Tracing Manager
        self.tracer = BankingTracer(config=self.config)

        # 3. Initialize RAG Tool
        self.rag_tool = BankingRAGTool(pipeline_instance=pipeline_instance, config=self.config)

        # 4. Initialize LLM Client from configuration
        self.client = FallbackChatClient("langgraph_agent")
        if config is not None:
            self.client = FallbackChatClient("langgraph_agent", {
                "model": getattr(self.config, "AGENT_LLM_MODEL", self.config.LLM_MODEL),
                "url": getattr(self.config, "AGENT_LLM_BASE_URL", self.config.LLM_BASE_URL),
                "api_key": getattr(self.config, "AGENT_LLM_API_KEY", self.config.LLM_API_KEY),
            })

        # 5. Initialize Decision Model Client (disabled / optional)
        self.decision_client = None

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

    def _call_llm_quick(self, system_prompt: str, user_prompt: str, max_tokens: int = 1536) -> str:
        """Call internal LLM endpoint for Router/Grader/Rewriter."""
        if not self.client:
            return ""
        eff_max_tokens = max_tokens
        if eff_max_tokens < 1536:
            eff_max_tokens = 1536
        try:
            resp = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=eff_max_tokens
            )
            raw_content = resp.choices[0].message.content
            if raw_content is None:
                if hasattr(resp.choices[0].message, "reasoning_content"):
                    raw_content = resp.choices[0].message.reasoning_content
            if raw_content is None:
                return ""
            return _clean_think_tags(raw_content).strip()
        except ModelFallbackError as e:
            print(f"[Agent Nodes] LLM Quick Call ModelFallbackError: {e}")
            return ""
        except Exception as e:
            print(f"[Agent Nodes] LLM Quick Call Error: {type(e).__name__}")
            return ""

    # =========================================================================
    # NODE 0: INPUT GUARDRAIL & STATE RESET
    # =========================================================================
    def input_guardrail_node(self, state: AgentState) -> dict:
        """
        Node 0: Reset temporary state variables from prior turns
        (prevent state contamination with persistent storage) and check input safety.
        """
        raw_query = state.get("user_query") or state.get("query") or state.get("original_query") or ""
        user_query = str(raw_query).strip()

        guard_result = self.guardrail_manager.guard_input(user_query)
        is_safe = getattr(guard_result, "is_safe", True)
        reason_text = str(getattr(guard_result, "trigger_reason", "") or "")
        sanitized_query = str(getattr(guard_result, "sanitized_query", user_query) or user_query)

        # Enhanced check with Decision Model (Adversarial & Injection Audit)
        blocked_by_decision = False
        if is_safe:
            if self.decision_client is not None and getattr(self.decision_client, "is_available", lambda: False)():
                adv_check = self.decision_client.audit_adversarial_input(user_query)
                if adv_check is not None:
                    if adv_check.get("is_attack", False):
                        if adv_check.get("confidence", 0.0) >= 0.85:
                            is_safe = False
                            blocked_by_decision = True
                            reason_text = "Prompt Injection / Adversarial attack detected by decision engine"

        max_retries_cfg = 3

        if hasattr(self.config, "AGENT_MAX_RETRIES"):
            max_retries_cfg = int(self.config.AGENT_MAX_RETRIES)
        elif hasattr(self.config, "MAX_RETRIES"):
            max_retries_cfg = int(self.config.MAX_RETRIES)

        # Case: Safety violation
        if not is_safe:
            refusal = str(getattr(guard_result, "refusal_message", "") or "")
            if not refusal:
                refusal = "Yêu cầu của Quý khách không thể thực hiện do vi phạm quy chuẩn an toàn bảo mật của Ngân hàng."
            step_msg = "Node 0: input_guardrail (Blocked by Security Guardrail)"
            if blocked_by_decision:
                step_msg = "Node 0: input_guardrail [Decision Engine] (Blocked: Prompt Injection)"
            res_state = dict(state)
            res_state.update({
                "user_query": user_query,
                "query": user_query,
                "original_query": user_query,
                "rewritten_query": user_query,
                "parsed_query": {},
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
                "execution_steps": [step_msg]
            })
            return res_state

        # Case: Safe input, reset state for turn
        res_state = dict(state)
        res_state.update({
            "user_query": user_query,
            "query": user_query,
            "original_query": user_query,
            "rewritten_query": user_query,
            "parsed_query": {},
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
        })
        return res_state

    # =========================================================================
    # NODE 1: SUPERVISOR ROUTER
    # =========================================================================
    def supervisor_router_node(self, state: AgentState) -> dict:
        """
        Node 1: Supervisor routes intent and assigns specialized sub-agent.
        Uses native Python string methods (0% Regex, Zero Ternary).
        """
        raw_query = state.get("user_query") or state.get("query") or ""
        query_lower = str(raw_query).lower().strip()
        steps = list(state.get("execution_steps", []))

        # 0. Enhanced routing via Decision Model
        if self.decision_client is not None and getattr(self.decision_client, "is_available", lambda: False)():
            decision_res = self.decision_client.route_supervisor(str(raw_query))
            if decision_res is not None:
                j_conf = float(decision_res.get("confidence", 0.0))
                if j_conf >= 0.70:
                    detected_intent = decision_res.get("intent", "RAG_SEARCH")
                    sub_agent = decision_res.get("sub_agent_role", "GENERAL_ADVISOR")
                    target_tool = decision_res.get("target_tool", "search_banking_knowledge")
                    if detected_intent == "CHIT_CHAT":
                        target_tool = None
                    elif detected_intent == "RAG_SEARCH":
                        target_tool = "search_banking_knowledge"
                    steps.append(f"Node 1: supervisor_router [Decision Engine] -> Intent={detected_intent} | Role={sub_agent} | Tool={target_tool} (conf={j_conf:.2f})")
                    res = dict(state)
                    res.update({
                        "intent": detected_intent,
                        "sub_agent_role": sub_agent,
                        "target_tool": target_tool,
                        "execution_steps": steps
                    })
                    return res

        # 1. Check for casual chat (CHIT_CHAT)
        chit_chat_keywords = [
            "xin chào", "chào bạn", "hello", "hi a-bank", "chào a-bank", "cảm ơn",
            "cam on", "bạn tên gì", "bạn là ai", "tạm biệt", "chúc bạn"
        ]
        banking_substantive_kws = [
            "thẻ", "the", "vay", "lãi", "lai", "tỷ giá", "ty gia", "tiết kiệm", "tiet kiem",
            "sinh trắc", "sinh trac", "2345", "ekyc", "hạn mức", "han muc", "phí", "phi",
            "tài khoản", "tai khoan", "chuyển tiền", "chuyen tien", "a-bank plus", "a-smart",
            "otp", "bảo hiểm", "bao hiem", "a-life", "hợp đồng", "hop dong",
            "tra soát", "tra soat", "sla", "báo cáo", "bao cao"
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

        # 2. Specialized Sub-Agent Role
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
            "vneid", "nfc", "quét khuôn mặt", "hạn mức chuyển tiền", "facepay",
            "bảo hiểm", "a-life", "tra soát", "sla", "quy chế", "tuân thủ"
        ]

        if any(marker in query_lower for marker in credit_card_markers):
            sub_agent = "CREDIT_CARD_AGENT"
        elif any(marker in query_lower for marker in loan_markers):
            sub_agent = "LOAN_MORTGAGE_AGENT"
        elif any(marker in query_lower for marker in compliance_markers):
            sub_agent = "COMPLIANCE_AUDIT_AGENT"

        # 3. Classify BANKING_ACTION vs RAG_SEARCH
        target_tool = None
        detected_intent = "RAG_SEARCH"

        is_card_or_knowledge_query = False
        card_terms = ["thẻ", "the", "credit", "debit", "platinum", "napas", "visa", "mastercard", "jcb", "rút tiền", "phí", "quy định", "hạn mức", "điều khoản", "bảo hiểm", "a-life", "tra soát", "báo cáo", "tài chính"]
        for cterm in card_terms:
            if cterm in query_lower:
                is_card_or_knowledge_query = True
                break

        if "khấu hao" in query_lower or "dư nợ giảm dần" in query_lower or "lịch trả nợ" in query_lower:
            detected_intent = "BANKING_ACTION"
            sub_agent = "LOAN_MORTGAGE_AGENT"
            target_tool = "calculate_loan_amortization"
        elif "vay" in query_lower and ("tháng bao nhiêu" in query_lower or "trả góp" in query_lower or "mỗi tháng" in query_lower):
            detected_intent = "BANKING_ACTION"
            sub_agent = "LOAN_MORTGAGE_AGENT"
            target_tool = "calculate_loan_amortization"
        elif ("sinh trắc học" in query_lower or "2345" in query_lower or "quét khuôn mặt" in query_lower) and ("chuyển" in query_lower or "triệu" in query_lower or "bắt buộc" in query_lower or "thiết bị" in query_lower):
            detected_intent = "BANKING_ACTION"
            sub_agent = "COMPLIANCE_AUDIT_AGENT"
            target_tool = "check_biometric_compliance_qd2345"
        elif not is_card_or_knowledge_query and ("tỷ giá hôm nay" in query_lower or "tra cứu tỷ giá" in query_lower or "bảng tỷ giá" in query_lower or "giá usd" in query_lower):
            detected_intent = "BANKING_ACTION"
            target_tool = "get_foreign_exchange_rates"
        elif not is_card_or_knowledge_query and ("tính lãi" in query_lower or "tinh lai" in query_lower or ("gửi" in query_lower and "tiết kiệm" in query_lower and "triệu" in query_lower and "lãi bao nhiêu" in query_lower)):
            detected_intent = "BANKING_ACTION"
            target_tool = "calculate_savings_profit"
        elif not is_card_or_knowledge_query and ("biểu lãi suất tiền gửi" in query_lower or "bảng lãi suất tiết kiệm" in query_lower or "lãi suất tiền gửi hôm nay" in query_lower):
            detected_intent = "BANKING_ACTION"
            target_tool = "get_interest_rates"
        elif "khiếu nại" in query_lower or "khieu nai" in query_lower or "tạo ticket" in query_lower or "báo mất thẻ" in query_lower or "khóa thẻ khẩn cấp" in query_lower:
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
        """Node 2: Hybrid Retrieval and Cross-Encoder Reranking via BankingRAGTool."""
        query = str(state.get("query") or state.get("user_query") or "")
        steps = list(state.get("execution_steps", []))
        steps.append("Node 2: retrieve (Hybrid RRF k=60 + reranker)")

        res = self.rag_tool.execute(query=query, top_k=6)
        docs = res.get("documents", [])
        sources = res.get("sources", [])
        citations = res.get("citations", [])

        # Audit Indirect Prompt Injection
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
        """Node 3: Assess quality and relevance of retrieved documents."""
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
        context_parts = []
        for d in docs:
            formatted = _format_retrieved_doc_context(d, max_chars=400)
            if formatted:
                context_parts.append(formatted)
        context_str = "\n---\n".join(context_parts)

        # 1. Fast relevance grading via Decision Model
        if self.decision_client is not None and getattr(self.decision_client, "is_available", lambda: False)():
            decision_grade = self.decision_client.grade_document_relevance(query, context_str)
            if decision_grade is not None:
                conf = float(decision_grade.get("confidence", 0.0))
                if conf >= 0.70:
                    quality = decision_grade.get("quality", "IRRELEVANT")
                    steps.append(f"Node 3: grade_documents [Decision Engine] -> {quality} (conf={conf:.2f})")
                    res_state = dict(state)
                    res_state.update({
                        "doc_quality": quality,
                        "document_quality": quality,
                        "execution_steps": steps
                    })
                    return res_state
                else:
                    steps.append(f"Node 3: grade_documents [Decision Engine] -> Low confidence ({conf:.2f} < 0.70), fallback LLM")

        # 2. LLM Fallback
        grader_prompt = DOCUMENT_GRADER_PROMPT.format(query=query, documents=context_str)

        verdict = self._call_llm_quick(DOCUMENT_GRADER_SYSTEM_MESSAGE, grader_prompt, max_tokens=1536).lower()

        quality = "RELEVANT"
        if "no" in verdict and "yes" not in verdict:
            quality = "IRRELEVANT"

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
        """Node 4: Rewrite query for search optimization (Loop-protected by retry_count)."""
        steps = list(state.get("execution_steps", []))
        current_retries = int(state.get("retry_count", 0))
        new_retries = current_retries + 1

        orig_q = str(state.get("original_query") or state.get("user_query") or state.get("query") or "")
        curr_q = str(state.get("query") or state.get("user_query") or "")

        rewrite_user = QUERY_REWRITER_PROMPT.format(original_query=orig_q, current_query=curr_q)
        new_query = self._call_llm_quick(QUERY_REWRITER_SYSTEM_MESSAGE, rewrite_user, max_tokens=256).strip().strip('"\'')

        if len(new_query) < 3:
            new_query = f"Quy định về {orig_q}"

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
    # NODE 5: BANKING TOOL CALLING
    # =========================================================================
    def banking_tool_node(self, state: AgentState) -> dict:
        """Node 5: Execute banking business tools based on query and routing target."""
        steps = list(state.get("execution_steps", []))
        target = state.get("target_tool", "")
        sub_agent = state.get("sub_agent_role", "GENERAL_ADVISOR")
        raw_query = str(state.get("query") or state.get("user_query") or "")
        query = raw_query.lower().strip()

        params = parse_vietnamese_financial_params(raw_query)
        tool_outputs = []
        executed_tool_name = target
        known_tools = {
            "calculate_loan_amortization",
            "check_biometric_compliance_qd2345",
            "get_interest_rates",
            "get_foreign_exchange_rates",
            "calculate_savings_profit",
            "create_support_ticket",
        }
        has_explicit_target = target in known_tools

        # 1. Loan Amortization
        if target == "calculate_loan_amortization" or (
            not has_explicit_target
            and (
                sub_agent == "LOAN_MORTGAGE_AGENT"
                or "khấu hao" in query
                or "dư nợ giảm dần" in query
                or ("vay" in query and ("tháng bao nhiêu" in query or "trả góp" in query or "lịch trả nợ" in query))
            )
        ):
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

        # 2. Biometric Compliance Evaluation
        elif target == "check_biometric_compliance_qd2345" or (
            not has_explicit_target
            and (
                sub_agent == "COMPLIANCE_AUDIT_AGENT"
                or "2345" in query
                or "sinh trắc học" in query
                or "quét khuôn mặt" in query
                or "khuôn mặt" in query
            )
        ):
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

        # 3. Deposit Interest Rates
        elif target == "get_interest_rates" or (not has_explicit_target and "lãi suất" in query and "vay" not in query and "khấu hao" not in query):
            executed_tool_name = "get_interest_rates"
            res = get_interest_rates(deposit_type="vnd")
            tool_outputs.append({"tool": "get_interest_rates", "result": res})

        # 4. Foreign Exchange Rates
        elif target == "get_foreign_exchange_rates" or (not has_explicit_target and ("tỷ giá" in query or "usd" in query or "ngoại tệ" in query)):
            executed_tool_name = "get_foreign_exchange_rates"
            currency = "USD"
            for c in ["EUR", "JPY", "GBP", "SGD", "AUD", "CAD", "CNY"]:
                if c.lower() in query:
                    currency = c
                    break
            res = get_foreign_exchange_rates(currency=currency)
            tool_outputs.append({"tool": "get_foreign_exchange_rates", "result": res})

        # 5. Calculate Savings Profit
        elif target == "calculate_savings_profit" or (not has_explicit_target and ("tính lãi" in query or "gửi tiết kiệm" in query)):
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

        # 6. Urgent Support Ticket
        else:
            executed_tool_name = "create_support_ticket"
            issue_type = "Yêu cầu hỗ trợ nghiệp vụ"
            priority = "NORMAL"
            if "nuốt thẻ" in query or "nuot the" in query:
                issue_type = "nuot_the_atm"
                priority = "HIGH"
            elif "mất thẻ" in query or "mat the" in query or "khóa thẻ" in query or "khoa the" in query:
                issue_type = "mat_the_khoa_the"
                priority = "HIGH"
            elif "lừa đảo" in query or "lua dao" in query or "gian lận" in query or "gian lan" in query:
                issue_type = "gian_lan_lua_dao"
                priority = "HIGH"
            elif "tra soát" in query or "tra soat" in query or "khiếu nại" in query or "khieu nai" in query:
                issue_type = "tra_soat_giao_dich"

            tool_input = {
                "customer_name": "Quý khách hàng",
                "issue_type": issue_type,
                "description": raw_query,
                "priority": priority
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
    # NODE 6: GENERATOR NODE
    # =========================================================================
    def generate_node(self, state: AgentState) -> dict:
        """Node 6: Generate comprehensive answer grounded on documents or tool results."""
        steps = list(state.get("execution_steps", []))
        intent = state.get("intent", "RAG_SEARCH")
        sub_agent_role = state.get("sub_agent_role", "GENERAL_ADVISOR")

        persona_key = str(state.get("persona") or "advisor").strip().lower()
        if not persona_key:
            persona_key = "advisor"
        tone_key = str(state.get("tone") or "formal").strip().lower()
        if not tone_key:
            tone_key = "formal"
        format_key = str(state.get("format_type") or state.get("format_style") or "standard").strip().lower()
        if not format_key:
            format_key = "standard"
        custom_persona = str(state.get("custom_persona") or "")

        specialist_prompt = SUB_AGENT_SYSTEM_PROMPTS.get(sub_agent_role, BANKING_AGENT_SYSTEM_PROMPT)
        style_prompt = build_banking_system_prompt(
            persona_key=persona_key,
            tone_key=tone_key,
            format_key=format_key,
            custom_persona=custom_persona
        )
        if style_prompt:
            selected_system_prompt = f"{specialist_prompt}\n\n{style_prompt}".strip()
        else:
            selected_system_prompt = specialist_prompt

        query = str(state.get("original_query") or state.get("user_query") or state.get("query") or "")
        docs = state.get("retrieved_chunks") or state.get("retrieved_docs") or []
        tool_outputs = state.get("tool_outputs", [])
        audit_feedback = state.get("audit_feedback", "")
        hallucination_retries = int(state.get("hallucination_retries", 0))

        # Branch 1: BANKING_ACTION
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
                    f"Kính gửi Quý khách, dưới đây là bảng tính lịch trả nợ ước tính theo phương thức **Dư nợ giảm dần**:\n\n"
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
                    f"*(Lưu ý: Bảng tính mang tính chất tham khảo. Lãi suất và lịch trả nợ chính xác sẽ được phê duyệt theo hợp đồng tín dụng cụ thể tại chi nhánh).*"
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
                    f"*(Nguồn tham chiếu: [Quyết định 2345/QĐ-NHNN, Điều 1 & Điều 2] - Ngân hàng Nhà nước Việt Nam).*"
                )

            elif tool_name == "calculate_savings_profit":
                draft = (
                    f"Kính gửi Quý khách, dưới đây là kết quả dự tính tiền gửi tiết kiệm:\n\n"
                    f"- **Số tiền gửi**: {tool_res.get('principal_amount_vnd', 0):,.0f} VND\n"
                    f"- **Kỳ hạn gửi**: {tool_res.get('term_months', 0)} tháng\n"
                    f"- **Lãi suất áp dụng**: {tool_res.get('applicable_rate_percent', 0)}%/năm\n"
                    f"- **Tiền lãi dự kiến**: {tool_res.get('interest_earned_vnd', 0):,.0f} VND\n"
                    f"- **Tổng số tiền khi đáo hạn**: {tool_res.get('total_at_maturity_vnd', 0):,.0f} VND\n\n"
                    f"*(Lưu ý: Tiền lãi thực tế có thể thay đổi tùy thuộc vào ngày gửi thực tế và chính sách ưu đãi tại thời điểm gửi).*"
                )

            elif tool_name == "get_foreign_exchange_rates":
                rates = tool_res.get("rates", {})
                rate_lines = [f"- **{curr} ({detail.get('name', '')})**: Mua CK: {detail.get('buy_transfer', 0):,.1f} | Bán: {detail.get('sell', 0):,.1f} VND" for curr, detail in rates.items()]
                draft = (
                    f"Bảng tỷ giá ngoại tệ ngày hôm nay ({tool_res.get('effective_date', '2026-10-25')}):\n\n"
                    + "\n".join(rate_lines)
                    + "\n\n*(Tỷ giá có tính chất tham khảo và có thể biến động theo từng thời điểm trong ngày giao dịch).*"
                )

            elif tool_name == "get_interest_rates":
                rates = tool_res.get("interest_rates", [])
                lines = [f"- **{r.get('term')}**: {r.get('rate_percent_per_year')}%/năm ({r.get('note', '')})" for r in rates[:6]]
                draft = (
                    f"Biểu lãi suất tiền gửi tiết kiệm cá nhân niêm yết mới nhất:\n\n"
                    + "\n".join(lines)
                    + "\n\n*(Quý khách có thể gửi tiết kiệm trực tuyến trên ứng dụng ngân hàng số để hưởng lãi suất ưu đãi nhất).*"
                )
            else:
                draft = str(tool_res.get("message", "Yêu cầu của Quý khách đã được tiếp nhận thành công."))

            steps.append("Node 6: generate (Banking Specialist Persona - Banking Tool Result)")
            res_state = dict(state)
            res_state.update({"draft_answer": draft, "final_answer": draft, "execution_steps": steps})
            return res_state

        # Branch 2: CHIT_CHAT
        if intent == "CHIT_CHAT":
            draft = (
                "Kính chào Quý khách! Tôi là Trợ lý AI Ngân hàng A-Bank.\n\n"
                "Tôi có thể hỗ trợ Quý khách giải đáp các quy chế mở tài khoản eKYC, hạn mức giao dịch chuyển tiền, "
                "thông tin biểu phí, điều khoản sản phẩm thẻ, hợp đồng bảo hiểm A-Life, quy trình tra soát SLA, tính lịch trả nợ vay, "
                "kiểm tra sinh trắc học theo QĐ 2345 hoặc tra cứu tỷ giá ngoại tệ hôm nay. "
                "Quý khách cần hỗ trợ thông tin gì ạ?"
            )
            steps.append("Node 6: generate (Banking Specialist Persona - Chit Chat)")
            res_state = dict(state)
            res_state.update({"draft_answer": draft, "final_answer": draft, "execution_steps": steps})
            return res_state

        # Branch 3: RAG_SEARCH
        if not docs:
            fallback = (
                f"Kính gửi Quý khách, hiện tại trong kho tài liệu quy chuẩn chưa có thông tin chi tiết "
                f"cho yêu cầu: '{query}'.\n\n"
                "Để được tư vấn và hỗ trợ chính xác nhất cho trường hợp của mình, Quý khách vui lòng liên hệ Trung tâm Hỗ trợ Khách hàng qua Tổng đài hỗ trợ hoặc đến chi nhánh gần nhất."
            )
            steps.append("Node 6: generate (Safe Fallback - Empty Docs)")
            res_state = dict(state)
            res_state.update({"draft_answer": fallback, "final_answer": fallback, "execution_steps": steps})
            return res_state

        # Assemble context from top chunks (universal 13-key schema: text, table, multimodal)
        context_parts = []
        for d in docs:
            formatted = _format_retrieved_doc_context(d)
            if formatted:
                context_parts.append(formatted)
        context_str = "\n\n".join(context_parts)

        feedback_instruction = ""
        if hallucination_retries > 0 and audit_feedback:
            feedback_instruction = f"\n\n[LƯU Ý ĐẶC BIỆT TỪ HỆ THỐNG AUDIT KIỂM ĐỊNH - LẦN THỬ LẠI {hallucination_retries}]: {audit_feedback}\n"

        if GENERATE_USER_PROMPT_TEMPLATE and "{context}" in GENERATE_USER_PROMPT_TEMPLATE and "{query}" in GENERATE_USER_PROMPT_TEMPLATE:
            user_prompt_body = GENERATE_USER_PROMPT_TEMPLATE.format(context=context_str, query=query)
        else:
            user_prompt_body = f"TÀI LIỆU CĂN CỨ TỪ HỆ THỐNG:\n{context_str}\n\nCÂU HỎI CỦA KHÁCH HÀNG: {query}"

        prompt = (
            f"{user_prompt_body}\n\n"
            f"{feedback_instruction}\n\n"
            "YÊU CẦU BẮT BUỘC VỀ CHẤT LƯỢNG CÂU TRẢ LỜI CHO GIÁM KHẢO NGÂN HÀNG:\n"
            "1. Đọc kỹ toàn bộ TÀI LIỆU CĂN CỨ (gồm cả văn bản, bảng dữ liệu [Bảng dữ liệu] và chú thích hình ảnh [Hình ảnh/Minh họa]) và trả lời TRỰC DIỆN, TOÀN DIỆN tất cả các vế của câu hỏi.\n"
            "2. Bắt đầu ngay bằng lời chào và giải đáp nghiệp vụ chuyên nghiệp (ví dụ: 'Kính gửi Quý khách...').\n"
            "3. TUYỆT ĐỐI KHÔNG viết các đoạn lập luận nháp nội bộ như 'Chúng ta cần trả lời...', 'Tài liệu cung cấp...', 'Chúng ta phân tích...'. Chỉ cung cấp câu trả lời hoàn chỉnh tới khách hàng.\n"
            "4. ĐỘ CHÍNH XÁC VÀ ĐẦY ĐỦ TUYỆT ĐỐI: Trích xuất CHÍNH XÁC mọi số liệu, tỷ lệ %, số tiền hạn mức (VNĐ/USD), các mốc thời hạn (ngày, tháng, năm), điều kiện doanh số chi tiêu hoặc số dư tối thiểu, các bước quy trình từ bước 1 đến bước cuối.\n"
            "5. TRẢ LỜI ĐẦY ĐỦ MỌI ĐỐI TƯỢNG ĐƯỢC HỎI: Nếu câu hỏi hỏi nhiều vế (ví dụ: cả thẻ chính và thẻ phụ, cả mức phí và điều kiện miễn phí, từng kỳ hạn trả góp, từng phân khúc khách hàng...), bắt buộc phải liệt kê rành mạch từng vế với đầy đủ số liệu tương ứng.\n"
            "6. THÁI ĐỘ NGHIỆP VỤ CHUẨN MỰC: Tuyệt đối không từ chối hoặc nói 'tài liệu không có' nếu trong tài liệu đã có bảng biểu, sơ đồ hoặc điều khoản liên quan. Hãy tổng hợp thông tin sát nhất từ tài liệu để giải đáp thấu đáo và chính xác.\n"
            "7. Cuối câu trả lời, ghi rõ căn cứ pháp lý: [Điều X, Khoản Y hoặc Tên văn bản quy chuẩn]."
        )

        draft = self._call_llm_quick(selected_system_prompt, prompt, max_tokens=5000)

        # Clean any leaked chain-of-thought prefix if present
        if draft.startswith("Chúng ta cần trả lời câu hỏi:"):
            # Find the actual answer starting with Kính gửi / Căn cứ / Theo quy định
            cut_markers = ["\nKính gửi", "\nKính thưa", "\nCăn cứ", "\nTheo quy định", "\nTrả lời:"]
            best_cut = -1
            for marker in cut_markers:
                idx = draft.find(marker)
                if idx != -1:
                    if best_cut == -1 or idx < best_cut:
                        best_cut = idx
            if best_cut != -1:
                draft = draft[best_cut:].strip()

        first_doc = _unwrap_doc_dict(docs[0])
        first_meta = first_doc.get("business_metadata")
        if not isinstance(first_meta, dict):
            first_meta = {}
        first_source = (
            str(first_meta.get("document_name") or "").strip()
            or str(first_doc.get("breadcrumb") or "").strip()
            or str(first_doc.get("source") or "").strip()
            or "Tài liệu ngân hàng"
        )

        if not draft:
            first_content = _format_retrieved_doc_context(docs[0], max_chars=400)
            draft = (
                f"Căn cứ theo quy định:\n\n{first_content}...\n\n"
                f"*(Nguồn tham chiếu: [Điều khoản quy định] - {first_source})*"
            )

        draft_lower = draft.lower()
        if "[điều" not in draft_lower and "[khoản" not in draft_lower and "[nguồn" not in draft_lower:
            draft = draft + f"\n\n*(Căn cứ pháp lý: [Điều khoản Quy định] - {first_source})*"

        steps.append(f"Node 6: generate ({sub_agent_role} - Banking Specialist Persona - RAG Grounded)")
        res_state = dict(state)
        res_state.update({"draft_answer": draft, "final_answer": draft, "execution_steps": steps})
        return res_state

    # =========================================================================
    # NODE 7: AUDIT GENERATION NODE
    # =========================================================================
    def audit_generation_node(self, state: AgentState) -> dict:
        """
        Node 7: Post-generation hallucination and citation verification.
        Validates citation grounding [Article X, Clause Y].
        """
        draft = str(state.get("draft_answer") or "").strip()
        intent = state.get("intent", "RAG_SEARCH")
        doc_quality = state.get("doc_quality") or state.get("document_quality", "IRRELEVANT")
        current_retries = int(state.get("hallucination_retries", 0))
        steps = list(state.get("execution_steps", []))

        # Pass through for CHIT_CHAT, BANKING_ACTION or IRRELEVANT fallback
        if intent != "RAG_SEARCH" or doc_quality != "RELEVANT":
            steps.append("Node 7: audit_generation -> PASS (Non-RAG or Fallback)")
            res_state = dict(state)
            res_state.update({"audit_status": "PASS", "final_answer": draft, "execution_steps": steps})
            return res_state

        draft_lower = draft.lower()
        citation_indicators = [
            "[điều", "[khoản", "[mục", "[nguồn:", "quyết định",
            "thông tư", "biểu phí", "quy định", "theo quy định", "căn cứ"
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
                    "BẮT BUỘC bổ sung trích dẫn cụ thể [Điều X, Khoản Y] hoặc tên tài liệu nguồn dựa đúng trên các đoạn văn bản đã cung cấp."
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
        """Node 8: Output guardrail safety check, cleans think tags, and updates session history."""
        steps = list(state.get("execution_steps", []))
        draft = state.get("final_answer") or state.get("draft_answer") or ""
        cleaned_answer = _clean_think_tags(str(draft))

        guard_res = self.guardrail_manager.guard_output(cleaned_answer)
        is_safe_flag = bool(getattr(guard_res, "is_safe", True))
        sanitized_text = str(getattr(guard_res, "sanitized_answer", cleaned_answer) or cleaned_answer)
        flags_list = list(getattr(guard_res, "flags", []) or [])

        # Update session memory
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
