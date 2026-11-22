"""
Execution adapters for Agent Skills in the A-Bank runtime.
Clean, intuitive, and modular:
- BaseSkill receives a validated binding from the skill registry.
- Measures execution latency and handles exceptions safely.

MANDATORY RULES:
1. NO __INIT__.PY: Direct module imports.
3. ZERO TERNARY: Explicit if/else only.
4. 0% REGEX LAW: Absolutely no regular expressions.
5. KEEP IT SIMPLE: Clear, direct, readable code.
"""

import time
from dataclasses import dataclass, field
from agent_banking.core.chat import FallbackChatClient
from agent_banking.core.fallback import is_temporary_error

from agent_banking.core.settings import Config
from agent_banking.agent.document_context import _format_retrieved_doc_context
from agent_banking.skills.config.prompts import (
    AUDIT_ANSWER_PROMPT_TEMPLATE,
    FACT_CHECKER_SYSTEM_PROMPT,
    GRADE_DOCUMENTS_PROMPT_TEMPLATE,
    SKILL_TEMPLATES,
)
from agent_banking.skills.config.settings import get_skill_runtime_config

_RUNTIME_CFG = get_skill_runtime_config()
_META_DEFAULTS = _RUNTIME_CFG["metadata_defaults"]
_PREVIEW_CFG = _RUNTIME_CFG["doc_preview"]
_FACT_CFG = _RUNTIME_CFG["fact_checker"]
_RAG_CFG = _RUNTIME_CFG["rag_pipeline"]
_QD2345_CFG = _RUNTIME_CFG["compliance_qd2345"]
_FIN_CFG = _RUNTIME_CFG["financial_dispatcher"]
_INC_CFG = _RUNTIME_CFG["incident_dispatcher"]


def _unwrap_chunk(doc) -> tuple:
    """Return (outer_dict, inner_chunk_dict) for wrapped or flat chunk payloads."""
    if isinstance(doc, dict):
        inner = doc.get("chunk")
        if isinstance(inner, dict):
            return doc, inner
        return doc, doc
    return {}, {"content": str(doc)}


def _format_doc_preview(doc) -> str:
    """Format canonical 13-key chunk (including table_data and images) for grading/audit."""
    _, chunk = _unwrap_chunk(doc)
    parts = []
    bread = str(chunk.get("breadcrumb") or "")
    sec = str(chunk.get("section_title") or "")
    if bread:
        parts.append(f"[{bread}]")
    elif sec:
        parts.append(f"[{sec}]")
    content = str(chunk.get("content", chunk.get("text", "")) or "")
    if content:
        parts.append(content)
    max_rows = int(_PREVIEW_CFG["max_table_rows"])
    max_images = int(_PREVIEW_CFG["max_images"])
    table_prefix = SKILL_TEMPLATES["doc_preview_table_prefix"]
    image_prefix = SKILL_TEMPLATES["doc_preview_image_prefix"]
    table_data = chunk.get("table_data")
    if isinstance(table_data, dict):
        t_title = str(table_data.get("title") or "")
        headers = table_data.get("headers") or []
        if t_title:
            parts.append(table_prefix.format(title=t_title))
        if headers:
            parts.append(" | ".join([str(h) for h in headers]))
        for row in (table_data.get("rows") or [])[:max_rows]:
            if isinstance(row, list):
                parts.append(" | ".join([str(cell) for cell in row]))
    images = chunk.get("images") or []
    if isinstance(images, list):
        for img in images[:max_images]:
            if isinstance(img, dict):
                cap = str(img.get("caption") or img.get("description") or img.get("native_text") or "")
                if cap:
                    parts.append(image_prefix.format(caption=cap))
    return "\n".join(parts)


def _clean_think_tags(text: str) -> str:
    """Clean reasoning tags <think>...</think> from LLM output (0% Regex)."""
    cleaned = text
    while True:
        s = cleaned.find("<think>")
        if s == -1:
            break
        e = cleaned.find("</think>")
        if e == -1:
            cleaned = cleaned[:s]
            break
        cleaned = cleaned[:s] + cleaned[e + 8:]
    return cleaned.strip()


@dataclass
class SkillMetadata:
    """Skill metadata loaded from configuration."""
    skill_id: str
    display_name: str
    description: str
    version: str = str(_META_DEFAULTS["version"])
    category: str = str(_META_DEFAULTS["category"])
    capabilities: list[str] = field(default_factory=list)
    intent_triggers: list[str] = field(default_factory=list)
    parameters_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    execution_config: dict = field(default_factory=dict)


class BaseSkill:
    """Execute a skill through an A-Bank adapter or registered tool."""

    def __init__(
        self,
        metadata: SkillMetadata,
        config: Config | None = None,
        pipeline_instance=None
    ):
        self.metadata = metadata
        if config is not None:
            self.config = config
        else:
            self.config = Config()

        self.pipeline = pipeline_instance
        self.tool_registry = None
        self.llm_client = FallbackChatClient("skills", {
            "model": self.config.SKILLS_LLM_MODEL,
            "url": self.config.SKILLS_LLM_BASE_URL,
            "api_key": self.config.SKILLS_LLM_API_KEY,
        })

    @classmethod
    def from_spec(cls, data: dict, config=None, pipeline_instance=None):
        default_ver = str(_META_DEFAULTS["version"])
        default_cat = str(_META_DEFAULTS["category"])
        meta = SkillMetadata(
            skill_id=data["skill_id"],
            display_name=data.get("display_name", data["skill_id"]),
            description=data.get("description", ""),
            version=data.get("version", default_ver),
            category=data.get("category", default_cat),
            capabilities=data.get("capabilities", []),
            intent_triggers=data.get("intent_triggers", []),
            parameters_schema=data.get("parameters_schema", {}),
            output_schema=data.get("output_schema", {}),
            execution_config=data.get("execution", {})
        )
        return cls(metadata=meta, config=config, pipeline_instance=pipeline_instance)

    def execute(self, params: dict) -> dict:
        """Execute skill and return normalized result with latency in ms."""
        start_time = time.perf_counter()
        try:
            data = self._run(params)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "skill_id": self.metadata.skill_id,
                "status": "SUCCESS",
                "latency_ms": latency_ms,
                "data": data,
                "error": None
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "skill_id": self.metadata.skill_id,
                "status": "FAILED",
                "latency_ms": latency_ms,
                "data": None,
                "error": str(exc),
                "temporary": is_temporary_error(exc)
            }

    def _run(self, params: dict):
        """Dispatch execution based on execution config."""
        exec_cfg = self.metadata.execution_config
        exec_type = exec_cfg.get("type", "generic")

        if exec_type == "registered_tool":
            from jsonschema import Draft202012Validator

            from agent_banking.guardrails.tool_guardrail import get_tool_guardrail
            from agent_banking.skills.skill_registry import mapped_arguments
            from agent_banking.tools.tool_registry import BankingToolRegistry, ONCE_PER_TURN, is_extension, tool_input_schema

            registry = self.tool_registry
            if registry is None:
                registry = BankingToolRegistry(pipeline_instance=self.pipeline)
            name = exec_cfg["tool"]
            definition = registry.get_tool_definition(name)
            if definition is None or name in ONCE_PER_TURN:
                raise ValueError("Skill target must be a registered read-only tool")
            schema = {**self.metadata.parameters_schema, "additionalProperties": False}
            if not Draft202012Validator(schema).is_valid(params):
                raise ValueError("Skill arguments do not match the schema")
            arguments = mapped_arguments(exec_cfg, params)
            guarded = get_tool_guardrail().guard_tool_call(name, arguments, registered_readonly=is_extension(definition))
            if not guarded["is_safe"]:
                raise ValueError("Tool guard rejected arguments")
            tool_schema = {**tool_input_schema(definition), "additionalProperties": False}
            if not Draft202012Validator(tool_schema).is_valid(guarded["sanitized_params"]):
                raise ValueError("Tool arguments do not match the schema")
            result = registry.execute_tool(name, guarded["sanitized_params"], definition=definition)
            if result["status"] != "SUCCESS":
                raise ValueError("Registered tool could not complete the skill")
            return result["data"]

        # 1. Biometric Compliance Execution (Decision 2345)
        if exec_type == "tool_call":
            from agent_banking.tools.banking_api_tools import check_biometric_compliance_qd2345
            amount = params.get("single_tx_amount_vnd", params.get("transaction_amount", _QD2345_CFG["default_single_tx_amount_vnd"]))
            daily_amount = params.get("daily_cumulative_amount_vnd", params.get("daily_accumulated_amount", _QD2345_CFG["default_daily_cumulative_amount_vnd"]))
            is_new_device = params.get("is_first_time_new_device", params.get("is_new_device", _QD2345_CFG["default_is_first_time_new_device"]))
            has_cccd = params.get("has_verified_chip_cccd", params.get("has_cccd_chip_nfc", _QD2345_CFG["default_has_verified_chip_cccd"]))

            audit_result = check_biometric_compliance_qd2345(
                single_tx_amount_vnd=float(amount),
                daily_cumulative_amount_vnd=float(daily_amount),
                is_first_time_new_device=bool(is_new_device),
                has_verified_chip_cccd=bool(has_cccd)
            )
            return {
                "compliance_status": "SUCCESS",
                "requires_biometric": audit_result.get("requires_biometric", False),
                "action_required": audit_result.get("mandatory_action", ""),
                "regulatory_rules_triggered": audit_result.get("trigger_reasons", []),
                "legal_basis": audit_result.get("regulation_reference", _QD2345_CFG["default_legal_basis"]),
                "full_audit_details": audit_result
            }

        # 2. Financial Calculator & Rates Execution
        if exec_type == "financial_dispatcher":
            from agent_banking.tools.banking_api_tools import (
                calculate_loan_amortization,
                calculate_savings_profit,
                get_interest_rates,
                get_foreign_exchange_rates,
                parse_vietnamese_financial_params
            )
            raw_text = params.get("raw_text", "")
            extracted = {}
            if raw_text:
                extracted = parse_vietnamese_financial_params(raw_text)

            default_op = _FIN_CFG["default_operation"]
            fallback_op = _FIN_CFG["fallback_operation"]
            op_keywords = _FIN_CFG["operation_keywords"]
            fin_defaults = _FIN_CFG["defaults"]

            operation = params.get("operation", default_op)
            if operation == default_op:
                text_lower = raw_text.lower()
                if any(kw in text_lower for kw in op_keywords["loan_amortization"]):
                    operation = "loan_amortization"
                elif any(kw in text_lower for kw in op_keywords["savings_profit"]):
                    operation = "savings_profit"
                elif any(kw in text_lower for kw in op_keywords["fx_rates"]):
                    operation = "fx_rates"
                else:
                    operation = fallback_op

            if operation == "loan_amortization":
                loan_def = fin_defaults["loan_amortization"]
                default_principal = float(loan_def["principal_vnd"])
                default_rate = float(loan_def["annual_rate_percent"])
                default_term = int(loan_def["term_months"])
                principal = params.get("principal_vnd", params.get("principal", params.get("amount_vnd", extracted.get("amount_vnd", default_principal))))
                if float(principal or 0.0) <= 0.0:
                    principal = default_principal
                annual_rate = params.get("annual_rate_percent", extracted.get("annual_rate_percent", default_rate))
                if float(annual_rate or 0.0) <= 0.0:
                    annual_rate = default_rate
                term_months = params.get("term_months", extracted.get("term_months", default_term))
                if int(term_months or 0) <= 0:
                    term_months = default_term
                result = calculate_loan_amortization(
                    principal_vnd=float(principal),
                    annual_rate_percent=float(annual_rate),
                    term_months=int(term_months)
                )
                return {"operation": "loan_amortization", "calculation_data": result}

            if operation == "savings_profit":
                sav_def = fin_defaults["savings_profit"]
                default_amount = float(sav_def["amount_vnd"])
                default_term = int(sav_def["term_months"])
                principal = params.get("amount_vnd", params.get("principal", params.get("principal_vnd", extracted.get("amount_vnd", default_amount))))
                if float(principal or 0.0) <= 0.0:
                    principal = default_amount
                term_months = params.get("term_months", extracted.get("term_months", default_term))
                if int(term_months or 0) <= 0:
                    term_months = default_term
                result = calculate_savings_profit(
                    amount_vnd=float(principal),
                    term_months=int(term_months)
                )
                return {"operation": "savings_profit", "calculation_data": result}

            if operation == "fx_rates":
                currency = params.get("currency", fin_defaults["fx_rates"]["currency"])
                result = get_foreign_exchange_rates(currency=currency)
                return {"operation": "fx_rates", "rates_data": result}

            dep_type = params.get("deposit_type", fin_defaults["interest_rates"]["deposit_type"])
            result = get_interest_rates(deposit_type=dep_type)
            return {"operation": "interest_rates", "rates_data": result}

        # 3. Incident Support Execution
        if exec_type == "incident_dispatcher":
            from agent_banking.tools.banking_api_tools import create_support_ticket
            issue_type = params.get("issue_type", params.get("incident_type", _INC_CFG["default_issue_type"]))
            contact = params.get("user_contact", params.get("phone_number", params.get("customer_phone", _INC_CFG["default_contact"])))
            card_num = params.get("card_number", "")
            default_details = SKILL_TEMPLATES["incident_default_details"]
            details = params.get("details", params.get("atm_location", default_details))
            urgency = params.get("urgency", _INC_CFG["default_urgency"])

            critical_urgency = _INC_CFG["critical_urgency"]
            issue_lower = str(issue_type).lower()
            for kw in _INC_CFG["critical_keywords"]:
                if kw in issue_lower:
                    urgency = critical_urgency
                    break

            desc = details
            if card_num:
                desc = SKILL_TEMPLATES["incident_card_detail_format"].format(details=details, card_number=card_num)

            priority_val = urgency
            if urgency == critical_urgency:
                priority_val = _INC_CFG["critical_priority"]

            ticket_data = create_support_ticket(
                customer_name=contact,
                issue_type=issue_type,
                description=desc,
                priority=priority_val
            )

            if urgency == critical_urgency:
                guide = SKILL_TEMPLATES["incident_critical_guide"]
            else:
                guide = SKILL_TEMPLATES["incident_standard_guide"]

            return {
                "ticket_info": ticket_data,
                "immediate_action_guide": guide,
                "hotline_support": exec_cfg.get("hotline", _INC_CFG["default_hotline"])
            }

        # 4. RAG Knowledge Retrieval Execution
        if exec_type == "rag_pipeline":
            from agent_banking.tools.rag_tool import BankingRAGTool

            default_top_k = int(_RAG_CFG["default_top_k"])
            context_max_chars = int(_RAG_CFG["context_max_chars"])
            result = BankingRAGTool(pipeline_instance=self.pipeline, config=self.config).execute(
                params.get("query", ""), top_k=params.get("top_k", default_top_k), candidate_k=params.get("candidate_k"),
                exact=params.get("exact", "")
            )
            documents = result["documents"]
            previews = [_format_retrieved_doc_context(chunk, max_chars=context_max_chars) for chunk in documents]
            return {**result, "chunks": documents, "query_used": result.get("query", ""),
                    "formatted_context": "\n\n---\n\n".join(preview for preview in previews if preview)}

        # 5. Document Grader & Hallucination Audit Execution
        if exec_type == "fact_checker_engine":
            default_check = _FACT_CFG["default_check_type"]
            max_docs = int(_FACT_CFG["max_docs"])
            check_type = params.get("check_type", default_check)
            raw_docs = params.get("documents", params.get("retrieved_context", ""))
            if isinstance(raw_docs, list):
                doc_texts = [_format_doc_preview(d) for d in raw_docs[:max_docs]]
                doc_str = "\n---\n".join(doc_texts)
            else:
                doc_str = str(raw_docs)

            if check_type == "grade_documents":
                query = params.get("query", "")
                prompt = GRADE_DOCUMENTS_PROMPT_TEMPLATE.format(
                    query=query,
                    documents=doc_str
                )
                verdict = self._call_llm_checker(prompt)
                is_relevant = verdict == "yes"
                return {
                    "check_type": "grade_documents",
                    "is_relevant": is_relevant,
                    "is_grounded": is_relevant
                }

            draft = params.get("draft_answer", params.get("answer", ""))
            has_citation = "[" in draft and "]" in draft
            prompt = AUDIT_ANSWER_PROMPT_TEMPLATE.format(
                documents=doc_str,
                draft=draft
            )
            verdict = self._call_llm_checker(prompt)
            is_faithful = verdict == "yes"

            compliance_score = 0.0
            if is_faithful:
                compliance_score = 1.0

            return {
                "check_type": "audit_answer",
                "is_faithful": is_faithful,
                "is_grounded": is_faithful,
                "has_citation": has_citation,
                "compliance_score": compliance_score
            }

        msg_tpl = SKILL_TEMPLATES["generic_success_message"]
        return {"status": "SUCCESS", "message": msg_tpl.format(skill_id=self.metadata.skill_id)}

    def _call_llm_checker(self, prompt: str) -> str:
        temperature = float(_FACT_CFG["temperature"])
        max_tokens = int(_FACT_CFG["max_tokens"])
        valid_verdicts = tuple(_FACT_CFG["valid_verdicts"])
        response = self.llm_client.chat.completions.create(
            model=self.config.SKILLS_LLM_MODEL,
            messages=[
                {"role": "system", "content": FACT_CHECKER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        verdict = _clean_think_tags(response.choices[0].message.content or "").lower().strip(". \"'")
        if verdict not in valid_verdicts:
            raise ValueError("Invalid fact-check response")
        return verdict
