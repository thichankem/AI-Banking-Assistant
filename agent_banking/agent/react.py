"""Guarded banking capability execution and deterministic citation checks."""

import json

from jsonschema import Draft202012Validator

from agent_banking.core.chat import FallbackChatClient
from agent_banking.guardrails.rag_guardrails import BankingGuardrailManager
from agent_banking.agent.document_context import (
    _format_retrieved_doc_context, _unwrap_doc_dict
)
from agent_banking.skills.skill_registry import SkillRegistry, mapped_arguments
from agent_banking.tools.tool_registry import BankingToolRegistry, ONCE_PER_TURN, is_extension


MAX_EVIDENCE = 36
FALLBACK_ANSWER = "Hiện tôi chưa đủ thông tin để trả lời chính xác yêu cầu này. Quý khách vui lòng cung cấp thêm chi tiết hoặc thử lại."
SIMULATED_TOOLS = {
    "create_support_ticket", "transfer_to_agent", "rate_agent_csat",
    "get_interest_rates", "get_foreign_exchange_rates", "calculate_savings_profit"
}
FINANCIAL_OPERATIONS = {
    "loan_amortization": ("calculate_loan_amortization", {"principal_vnd": "principal",
                           "annual_rate_percent": "annual_rate_percent", "term_months": "term_months"}),
    "savings_profit": ("calculate_savings_profit", {"amount_vnd": "principal", "term_months": "term_months"}),
    "interest_rates": ("get_interest_rates", {"deposit_type": "deposit_type"}),
    "fx_rates": ("get_foreign_exchange_rates", {"currency": "currency"})
}


def _reference_numbers(answer: str, evidence_count: int) -> set[int]:
    references = set()
    if answer.lower().count("[nguồn ") != answer.count("[Nguồn "):
        raise ValueError("Invalid citation marker")
    for tail in answer.split("[Nguồn ")[1:]:
        label, closing, _ = tail.partition("]")
        if not closing or not label.isascii() or not label.isdecimal() or len(label) > 3:
            raise ValueError("Invalid citation marker")
        number = int(label)
        if number < 1 or number > evidence_count or label != str(number):
            raise ValueError("Unknown citation marker")
        references.add(number)
    return references


class BankingReActNodes:
    def __init__(self, pipeline, config):
        self.config = config
        self.guardrails = BankingGuardrailManager(embedder=getattr(pipeline, "embedder", None))
        self.tools = BankingToolRegistry(pipeline_instance=pipeline)
        self.skills = SkillRegistry(pipeline_instance=pipeline, config=config, tool_registry=self.tools)
        self.client = FallbackChatClient("agent", {
            "model": config.AGENT_LLM_MODEL,
            "url": config.AGENT_LLM_BASE_URL,
            "api_key": config.AGENT_LLM_API_KEY,
        })

    def snapshot(self) -> dict:
        tools = self.tools.snapshot()
        return {"tool": tools, "skill": self.skills.snapshot(tools)}

    def catalog_view(self, snapshot: dict) -> dict:
        return {
            "tool": {
                name: {"description": spec.get("description", ""), "parameters": spec["parameters"]}
                for name, spec in snapshot["tool"].items()
            },
            "skill": {
                name: {"description": spec.get("description", ""), "parameters": spec["parameters_schema"]}
                for name, spec in snapshot["skill"].items()
            },
        }

    def screen_input(self, query: str) -> dict:
        guard = self.guardrails.guard_input(query)
        if guard.is_safe:
            return {"query": guard.sanitized_query, "is_safe": True}
        return {"query": "", "is_safe": False, "refusal": guard.refusal_message or FALLBACK_ANSWER}

    def guard_artifact(self, text: str) -> str:
        """Check model-generated or externally supplied text before it crosses a trust boundary."""
        text = text.strip()
        lowered = text.lower()
        internal_tags = ("<think", "<thought", "<reasoning", "<scratchpad", "<thinking")
        if any(marker in lowered for marker in internal_tags):
            return ""
        leak, _, _ = self.guardrails._check_auth_leak(lowered)
        if leak or not self.guardrails.indirect_guardrail.scan_text(text)["is_safe"]:
            return ""
        return self.guardrails._mask_pii(text)

    def guard_payload(self, value):
        if isinstance(value, str):
            guarded = self.guard_artifact(value)
            if value.strip() and not guarded:
                raise ValueError("Unsafe text")
            return guarded
        if isinstance(value, dict):
            return {key: self.guard_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.guard_payload(item) for item in value]
        return value

    def input_guardrail(self, state: dict) -> dict:
        query = str(state.get("user_query") or state.get("query") or "").strip()
        screened = state.get("screened_input")
        if screened is None or screened["is_safe"]:
            screened = self.screen_input(query)
        state = {
            **state,
            "input_guard": {"is_safe": screened["is_safe"]},
            "is_safe": screened["is_safe"],
            "user_query": screened["query"],
            "intent": "ROUTING",
            "retry_count": 0,
            "search_count": 0,
            "hallucination_retries": 0,
            "retrieved_chunks": [],
            "tool_outputs": [],
            "citations": [],
            "sources": [],
            "final_answer": "",
            "audit_status": "PENDING",
            "audit_feedback": "",
            "execution_steps": ["input_guardrail"],
            "react_turn": {"observations": [], "attempts": [], "evidence": [], "decision": {}}
        }
        state.pop("screened_input", None)
        state.pop("query", None)
        if not screened["is_safe"]:
            state.update(intent="REFUSAL", final_answer=screened["refusal"], audit_status="BLOCKED")
        elif "capabilities" not in state:
            state["capabilities"] = self.snapshot()
        return state

    def _schema_error(self, schema: dict, params: dict) -> str:
        schema = {**schema, "additionalProperties": False}
        error = next(Draft202012Validator(schema).iter_errors(params), None)
        if error is None:
            return ""
        if error.validator == "required":
            missing = [key for key in error.validator_value if key not in params]
            return "Thiếu tham số bắt buộc: " + ", ".join(missing)
        path = ".".join(str(part) for part in error.path)
        return "Tham số không hợp lệ: " + (path or "arguments")

    def _guard_tool(self, name: str, params: dict, tools: dict) -> tuple[dict, str]:
        spec = tools.get(name)
        if spec is None:
            return {}, "Tool chưa đăng ký."
        error = self._schema_error(spec["parameters"], params)
        if error:
            return {}, error
        if name == "search_banking_knowledge":
            if not params["query"].strip() or not 1 <= params.get("top_k", 3) <= 6:
                return {}, "Cần query có nội dung và top_k từ 1 đến 6."
        result = self.guardrails.guard_tool_call(name, params, registered_readonly=is_extension(spec))
        if not result["is_safe"]:
            return {}, result["error_message"]
        return result["sanitized_params"], ""

    def _prepare_skill(self, spec: dict, params: dict, evidence: list, tools: dict) -> tuple[dict, str, str]:
        error = self._schema_error(spec["parameters_schema"], params)
        if error:
            return {}, "", error
        execution = spec["execution"]
        mode = execution.get("type")
        if mode == "registered_tool":
            target = execution["tool"]
            if target in ONCE_PER_TURN:
                return {}, "", "Skill cần trỏ tới tool chỉ đọc."
            guarded, error = self._guard_tool(target, mapped_arguments(execution, params), tools)
            return guarded, target, error
        if mode == "rag_pipeline":
            query = params.get("query", "")
            top_k = params.get("top_k", 3)
            candidate_k = params.get("candidate_k", max(5, top_k))
            if not query.strip() or not 1 <= top_k <= 6 or not top_k <= candidate_k <= 20:
                return {}, "", "Cần query có nội dung, top_k từ 1 đến 6 và candidate_k từ top_k đến 20."
            guarded, error = self._guard_tool("search_banking_knowledge", {"query": query, "top_k": top_k}, tools)
            return {**guarded, "candidate_k": candidate_k}, "", error
        if mode == "fact_checker_engine":
            if not evidence:
                return {}, "", "Cần tra cứu tài liệu trước khi đối soát."
            if params.get("check_type") == "grade_documents":
                required = "query"
            else:
                required = "draft_answer"
            if not params.get(required, "").strip():
                return {}, "", "Thiếu tham số bắt buộc: " + required
            return {**params, "documents": [entry["chunk"] for entry in evidence]}, "", ""
        if mode == "incident_dispatcher":
            if params.get("card_number"):
                return {}, "", "Không truyền số thẻ hoặc tài khoản vào phiếu hỗ trợ."
            request = {
                "customer_name": params.get("user_contact", "KHACH_HANG_HOTLINE"),
                "issue_type": params.get("issue_type", "tra_soat_giao_dich"),
                "description": params.get("details", ""),
                "priority": params.get("urgency", "HIGH").replace("CRITICAL", "URGENT")
            }
            guarded, error = self._guard_tool("create_support_ticket", request, tools)
            if error:
                return {}, "", error
            return {
                "user_contact": guarded["customer_name"], "issue_type": guarded["issue_type"],
                "details": guarded["description"], "urgency": guarded["priority"].replace("URGENT", "CRITICAL")
            }, "create_support_ticket", ""
        if mode == "financial_dispatcher":
            operation = FINANCIAL_OPERATIONS.get(params.get("operation"))
            if operation is None:
                return {}, "", "Chọn operation cụ thể cho skill; có thể dùng tool trích xuất tham số trước."
            tool_name, mapping = operation
        elif mode == "tool_call":
            tool_name, mapping = execution["function"], execution.get("arg_mapping", {})
        else:
            return {}, "", "Skill chưa có phương thức thực thi được hỗ trợ."

        request = {key: params[source] for key, source in mapping.items() if source in params}
        guarded, error = self._guard_tool(tool_name, request, tools)
        skill_params = {mapping[key]: value for key, value in guarded.items() if key in mapping}
        if "operation" in params:
            skill_params["operation"] = params["operation"]
        return skill_params, tool_name, error

    def _observe_retrieval(self, data: dict, evidence: list) -> dict:
        chunks = data.get("documents") or data.get("chunks", [])
        safe_chunks, blocked_count, _ = self.guardrails.guard_retrieved_chunks(chunks)
        references = []
        for chunk in safe_chunks:
            record = _unwrap_doc_dict(chunk)
            context = _format_retrieved_doc_context(record, max_chars=1800)
            if not context:
                continue
            scan = getattr(self.guardrails, "indirect_guardrail", None)
            if scan is not None and not scan.scan_text(context)["is_safe"]:
                blocked_count += 1
                continue
            source = str(record.get("source") or "Tài liệu ngân hàng")
            identity = (source, record.get("chunk_id"), context)
            entry = next((item for item in evidence if tuple(item["identity"]) == identity), None)
            if entry is None:
                if len(evidence) >= MAX_EVIDENCE:
                    continue
                entry = {
                    "identity": identity, "chunk": record, "context": context,
                    "index": len(evidence) + 1, "source": source
                }
                evidence.append(entry)
            references.append({"label": f"[Nguồn {entry['index']}]", "text": entry["context"]})
        return {"documents": references, "blocked_count": blocked_count}

    def act(self, state: dict) -> dict:
        turn = state["react_turn"]
        decision = turn["decision"]
        kind, name, params = decision["kind"], decision["name"], decision["arguments"]
        evidence = list(turn["evidence"])
        attempts = list(turn["attempts"])
        snapshot = state["capabilities"]
        tools = snapshot["tool"]
        definition = snapshot[kind].get(name)
        tool_name = name
        error = ""
        try:
            params = self.guard_payload(params)
        except ValueError:
            error = "Tham số bị guardrail từ chối."
        if definition is None:
            error = "Hành động chưa đăng ký."
            name = "unknown"
        elif not error and kind == "skill":
            params, tool_name, error = self._prepare_skill(definition, params, evidence, tools)
        elif not error:
            params, error = self._guard_tool(name, params, tools)
        key = ""
        if not error and state.get("worker") and tool_name in ONCE_PER_TURN:
            error = "Subagent chỉ được thực hiện tác vụ đọc hoặc tính toán."
        if not error:
            if tool_name in ONCE_PER_TURN:
                key = tool_name
            else:
                # Fact-checking documents come from state and need not be stored twice.
                fingerprint_input = {k: v for k, v in params.items() if k != "documents"}
                if kind == "skill" and name == "fact_checking_audit":
                    fingerprint_input["evidence_count"] = len(evidence)
                key = kind + ":" + name + ":" + json.dumps(fingerprint_input, ensure_ascii=False, sort_keys=True)
            if key in attempts:
                error = "Hành động này đã được thực hiện trong lượt hiện tại. Hãy dùng kết quả đã có hoặc đổi tham số."

        calls = list(state.get("tool_outputs", []))
        observation = {"kind": kind, "name": name, "status": "ERROR", "error": error}
        intent = state["intent"]
        searches = state.get("search_count", 0)
        if not error:
            attempts.append(key)
            retrieval = tool_name == "search_banking_knowledge"
            if kind == "skill" and definition["execution"]["type"] == "rag_pipeline":
                retrieval = True
            if kind == "tool" or definition["execution"]["type"] == "registered_tool":
                result = self.tools.execute_tool(tool_name, params, definition=tools[tool_name])
            else:
                result = self.skills.get_skill(name, definition=definition).execute(params)
            data = result.get("data")
            if result.get("status") == "SUCCESS" and isinstance(data, dict) and str(data.get("status", "")).lower() not in ("failed", "error"):
                observation = {"kind": kind, "name": name, "status": "SUCCESS"}
                simulated = tool_name in SIMULATED_TOOLS or data.get("simulation") is True
                documents = data.get("documents") or data.get("chunks")
                if isinstance(documents, list) and documents and all(_unwrap_doc_dict(item).get("content") for item in documents):
                    retrieval = True
                if retrieval:
                    observation["retrieval"] = True
                    data = self._observe_retrieval(data, evidence)
                    searches += 1
                    intent = "RAG_SEARCH"
                elif intent != "RAG_SEARCH":
                    intent = "BANKING_ACTION"
                if simulated:
                    observation["simulation"] = True
                try:
                    if retrieval:
                        observation["data"] = data
                    else:
                        observation["data"] = self.guard_payload(data)
                except ValueError:
                    observation.update(status="ERROR", error="Kết quả bị guardrail từ chối.")
            else:
                observation["error"] = "Công cụ chưa trả được kết quả cho yêu cầu này."

            if retrieval and isinstance(params.get("query"), str):
                observation["query_used"] = params["query"]

            calls.append({
                "kind": kind, "tool": name, "status": observation["status"],
                "latency_ms": result.get("latency_ms", 0.0),
                "simulation": observation.get("simulation", False)
            })
        return {
            **state, "intent": intent, "tool_outputs": calls,
            "search_count": searches, "retry_count": max(0, searches - 1),
            "retrieved_chunks": [entry["chunk"] for entry in evidence],
            "react_turn": {**turn, "attempts": attempts, "evidence": evidence,
                           "observations": turn["observations"] + [observation]},
            "last_action": {"kind": kind, "name": name, "status": observation["status"]},
            "execution_steps": state["execution_steps"] + [f"react_action: {kind} {name} ({observation['status']})"]
        }

    def validate_citations(self, decision: dict, evidence: list) -> list:
        draft = decision["answer"]
        used = _reference_numbers(draft, len(evidence))
        if evidence and not used and not decision.get("insufficient_evidence"):
            raise ValueError("Missing citation")
        documents = [evidence[index - 1]["chunk"] for index in sorted(used)]
        if documents:
            grounding = self.guardrails.verify_citation_grounding(draft, documents)
            if grounding["unsupported_citations"]:
                raise ValueError("Unsupported legal citation")

        citations = []
        for index in sorted(used):
            entry = evidence[index - 1]
            chunk = entry["chunk"]
            citations.append({
                "index": index, "label": f"[Nguồn {index}]", "source": entry["source"],
                "chunk_id": chunk.get("chunk_id", ""), "breadcrumb": chunk.get("breadcrumb", ""),
                "section_title": chunk.get("section_title", "")
            })
        return citations
