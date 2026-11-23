"""Guarded banking capability execution and deterministic citation checks."""

import json

from jsonschema import Draft202012Validator

from agent_banking.agent.config.response_profiles import get_messages, get_prompt
from agent_banking.agent.config.settings import (
    get_financial_operations,
    get_int_setting,
    get_setting,
)
from agent_banking.agent.document_context import (
    _format_retrieved_doc_context, _unwrap_doc_dict
)
from agent_banking.core.chat import FallbackChatClient
from agent_banking.core.models import auto_routing_enabled
from agent_banking.core.search_text import search_key
from agent_banking.guardrails.rag_guardrails import BankingGuardrailManager
from agent_banking.skills.skill_registry import SkillRegistry, mapped_arguments, resolve_skill
from agent_banking.tools.extensions.legal_web_search import MAX_SEARCHES_PER_TURN, TOOL_NAME as LEGAL_WEB_SEARCH_TOOL
from agent_banking.tools.tool_registry import BankingToolRegistry, ONCE_PER_TURN, is_extension, tool_contract, tool_input_schema
from agent_banking.tracing.tracer import trace_span


MAX_EVIDENCE = get_int_setting("max_evidence")
FALLBACK_ANSWER = get_prompt("FALLBACK_ANSWER")
SIMULATED_TOOLS = set(get_setting("simulated_tools", ()))
FINANCIAL_OPERATIONS = get_financial_operations()

_EVIDENCE_CONTEXT_MAX_CHARS = get_int_setting("evidence_context_max_chars")
_EVIDENCE_MERGED_MAX_CHARS = get_int_setting("evidence_merged_max_chars")
_SEARCH_TOP_K_MIN = get_int_setting("search_top_k_min")
_SEARCH_TOP_K_MAX = get_int_setting("search_top_k_max")
_SEARCH_TOP_K_DEFAULT = get_int_setting("search_top_k_default")
_SEARCH_CANDIDATE_K_MAX = get_int_setting("search_candidate_k_max")
_SEARCH_CANDIDATE_K_DEFAULT = get_int_setting("search_candidate_k_default")
_CALCULATION_TOLERANCE_VND = get_int_setting("calculation_tolerance_vnd")
_CALCULATION_MAX_VALUE = float(get_setting("calculation_max_value", 0))
_MAX_CITATION_DIGITS = get_int_setting("max_citation_number_digits")
_INTERNAL_THINKING_TAGS = tuple(get_setting("internal_thinking_tags", ()))
_INCIDENT_DEFAULTS = dict(get_setting("incident_defaults") or {})
_REACT_MESSAGES = get_messages("react")


def legal_searches_remaining(calls):
    return max(0, MAX_SEARCHES_PER_TURN - sum(call.get("tool") == LEGAL_WEB_SEARCH_TOOL for call in calls))


def verify_calculation(tool_name: str, data: dict, params: dict) -> bool:
    if tool_name == "calculate_loan_amortization":
        keys = ("principal_vnd", "total_interest_vnd", "total_payable_vnd",
                "annual_rate_percent", "first_month_payment_vnd")
        principal = params.get("principal_vnd", params.get("principal"))
    elif tool_name == "calculate_savings_profit":
        keys = ("principal_amount_vnd", "interest_earned_vnd", "total_at_maturity_vnd", "applicable_rate_percent")
        principal = params.get("amount_vnd", params.get("principal"))
    else:
        return True
    result = data.get("calculation_data", data)
    if not isinstance(result, dict) or result.get("status") != "SUCCESS":
        return False
    values = [result.get(key) for key in keys]
    if any(type(value) not in (int, float) or not 0 <= value <= _CALCULATION_MAX_VALUE for value in values):
        return False
    months = result.get("term_months")
    if values[0] != principal or type(months) is not int or months <= 0 or months != params.get("term_months"):
        return False
    monthly_rate = values[3] / 1200
    if tool_name == "calculate_loan_amortization":
        if values[3] != params.get("annual_rate_percent"):
            return False
        expected_interest = principal * monthly_rate * (months + 1) / 2
        if abs(values[4] - principal * (1 / months + monthly_rate)) > _CALCULATION_TOLERANCE_VND:
            return False
    else:
        expected_interest = principal * monthly_rate * months
    # Monetary fields are independently rounded to whole VND.
    return (
        abs(values[1] - expected_interest) <= _CALCULATION_TOLERANCE_VND
        and abs(values[0] + values[1] - values[2]) <= _CALCULATION_TOLERANCE_VND
    )


def _preview_trace_dict(payload, max_chars=300):
    """Build a bounded preview dict of tool inputs or outputs for tracing."""
    if not isinstance(payload, dict):
        return {}
    out = {}
    for k, v in list(payload.items())[:25]:
        if isinstance(v, str):
            if len(v) > max_chars:
                out[k] = v[:max_chars - 3] + "..."
            else:
                out[k] = v
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list):
            out[k] = f"[{len(v)} items]"
        elif isinstance(v, dict):
            out[k] = _preview_trace_dict(v, max_chars=max_chars)
        else:
            text_val = str(v)
            if len(text_val) > max_chars:
                out[k] = text_val[:max_chars - 3] + "..."
            else:
                out[k] = text_val
    return out


def _normalize_citation_markers(answer: str) -> str:
    """Normalize accepted citation bracket prefixes to canonical '[Nguồn '."""
    norm_answer = answer
    for prefix in ("[nguồn ", "[Nguon ", "[nguon ", "[NGUỒN ", "[Source ", "[source ", "[SOURCE "):
        if prefix in norm_answer:
            norm_answer = norm_answer.replace(prefix, "[Nguồn ")
    for colon_prefix in ("[Nguồn: ", "[Nguon: ", "[nguồn: ", "[nguon: ", "[Source: ", "[source: "):
        if colon_prefix in norm_answer:
            norm_answer = norm_answer.replace(colon_prefix, "[Nguồn ")
    return norm_answer


def _reference_numbers(answer: str, evidence_count: int) -> set[int]:
    references = set()
    norm_answer = _normalize_citation_markers(answer)
    if norm_answer.lower().count("[nguồn ") != norm_answer.count("[Nguồn "):
        raise ValueError("Invalid citation marker")
    for tail in norm_answer.split("[Nguồn ")[1:]:
        label, closing, _ = tail.partition("]")
        if not closing:
            raise ValueError("Invalid citation marker")
        parts = [part.strip() for part in label.replace(";", ",").split(",") if part.strip()]
        if not parts:
            raise ValueError("Invalid citation marker")
        for idx, part in enumerate(parts):
            token = part
            if idx > 0 and part.startswith("Nguồn "):
                token = part[len("Nguồn "):].strip()
            if " " in token:
                token = token.split()[0].rstrip(":,-")
            if not token.isascii() or not token.isdecimal() or len(token) > _MAX_CITATION_DIGITS:
                raise ValueError("Invalid citation marker")
            number = int(token)
            if number < 1 or number > evidence_count or token != str(number):
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
        }, routed=auto_routing_enabled())

    def snapshot(self) -> dict:
        tools = self.tools.snapshot()
        return {"tool": tools, "skill": self.skills.snapshot(tools)}

    def catalog_view(self, snapshot: dict) -> dict:
        return {
            "tool": {name: tool_contract(spec) for name, spec in snapshot["tool"].items()},
            "skill": {
                name: {"description": spec.get("description", ""), "category": spec.get("category", "GENERAL")}
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
        if any(marker in lowered for marker in _INTERNAL_THINKING_TAGS):
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
            "last_action": None,
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
            return _REACT_MESSAGES.get("missing_required_params_prefix", "") + ", ".join(missing)
        path = ".".join(str(part) for part in error.path)
        return _REACT_MESSAGES.get("invalid_params_prefix", "") + (path or "arguments")

    def _guard_tool(self, name: str, params: dict, tools: dict) -> tuple[dict, str]:
        spec = tools.get(name)
        if spec is None:
            return {}, _REACT_MESSAGES.get("tool_not_registered", "")
        error = self._schema_error(tool_input_schema(spec), params)
        if error:
            return {}, error
        if name == "search_banking_knowledge":
            if not params["query"].strip() or not _SEARCH_TOP_K_MIN <= params.get("top_k", _SEARCH_TOP_K_DEFAULT) <= _SEARCH_TOP_K_MAX:
                return {}, _REACT_MESSAGES.get("invalid_search_params", "")
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
        if mode == "guidance":
            return {}, "", _REACT_MESSAGES.get("skill_guidance_only", "")
        if mode == "registered_tool":
            target = execution["tool"]
            if target in ONCE_PER_TURN:
                return {}, "", _REACT_MESSAGES.get("skill_must_be_readonly", "")
            guarded, error = self._guard_tool(target, mapped_arguments(execution, params), tools)
            return guarded, target, error
        if mode == "rag_pipeline":
            query = params.get("query", "")
            top_k = params.get("top_k", _SEARCH_TOP_K_DEFAULT)
            candidate_k = params.get(
                "candidate_k",
                max(top_k, getattr(self.config, "HYBRID_CANDIDATE_K", _SEARCH_CANDIDATE_K_DEFAULT)),
            )
            if (
                not query.strip()
                or not _SEARCH_TOP_K_MIN <= top_k <= _SEARCH_TOP_K_MAX
                or not top_k <= candidate_k <= _SEARCH_CANDIDATE_K_MAX
            ):
                return {}, "", _REACT_MESSAGES.get("invalid_rag_skill_params", "")
            search = {"query": query, "top_k": top_k}
            if "exact" in params:
                search["exact"] = params["exact"]
            guarded, error = self._guard_tool("search_banking_knowledge", search, tools)
            return {**guarded, "candidate_k": candidate_k}, "", error
        if mode == "fact_checker_engine":
            if not evidence:
                return {}, "", _REACT_MESSAGES.get("fact_check_needs_evidence", "")
            if params.get("check_type") == "grade_documents":
                required = "query"
            else:
                required = "draft_answer"
            if not params.get(required, "").strip():
                return {}, "", _REACT_MESSAGES.get("missing_required_params_prefix", "") + required
            return {**params, "documents": [entry["chunk"] for entry in evidence]}, "", ""
        if mode == "incident_dispatcher":
            if params.get("card_number"):
                return {}, "", _REACT_MESSAGES.get("incident_forbids_card", "")
            request = {
                "customer_name": params.get("user_contact", _INCIDENT_DEFAULTS.get("customer_name", "")),
                "issue_type": params.get("issue_type", _INCIDENT_DEFAULTS.get("issue_type", "")),
                "description": params.get("details", ""),
                "priority": params.get("urgency", _INCIDENT_DEFAULTS.get("urgency", "")).replace("CRITICAL", "URGENT")
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
                return {}, "", _REACT_MESSAGES.get("financial_needs_operation", "")
            tool_name, mapping = operation
        elif mode == "tool_call":
            tool_name, mapping = execution["function"], execution.get("arg_mapping", {})
        else:
            return {}, "", _REACT_MESSAGES.get("skill_unsupported_execution", "")

        request = {key: params[source] for key, source in mapping.items() if source in params}
        guarded, error = self._guard_tool(tool_name, request, tools)
        skill_params = {mapping[key]: value for key, value in guarded.items() if key in mapping}
        if "operation" in params:
            skill_params["operation"] = params["operation"]
        return skill_params, tool_name, error

    def _observe_retrieval(self, data: dict, evidence: list) -> dict:
        previous_count = len(evidence)
        chunks = data.get("documents") or data.get("chunks", [])
        related = data.get("related_documents") or []
        safe_chunks, blocked_count, _ = self.guardrails.guard_retrieved_chunks([*chunks, *related])
        references = []
        anchors = set()
        default_source = _REACT_MESSAGES.get("default_source_name", "")
        for chunk in safe_chunks:
            record = _unwrap_doc_dict(chunk)
            context_for = record.get("context_for")
            source = str(record.get("source") or default_source)
            if context_for and (source, context_for) not in anchors:
                continue
            context = _format_retrieved_doc_context(record, max_chars=_EVIDENCE_CONTEXT_MAX_CHARS, query=data.get("query", ""))
            if not context:
                continue
            scan = getattr(self.guardrails, "indirect_guardrail", None)
            if scan is not None and not scan.scan_text(context)["is_safe"]:
                blocked_count += 1
                continue
            meta = record.get("business_metadata") or {}
            identity = (source, record.get("chunk_id"), meta.get("document_version"),
                        meta.get("publication_revision") or record.get("content", ""))
            entry = next((item for item in evidence if tuple(item["identity"]) == identity), None)
            if entry is None:
                if len(evidence) >= MAX_EVIDENCE:
                    continue
                entry = {
                    "identity": identity, "chunk": record, "context": context,
                    "index": len(evidence) + 1, "source": source
                }
                evidence.append(entry)
            elif context not in entry["context"] and len(entry["context"]) + len(context) <= _EVIDENCE_MERGED_MAX_CHARS:
                entry = {**entry, "context": entry["context"] + "\n\n" + context}
                evidence[entry["index"] - 1] = entry
            if not context_for:
                anchors.add((source, record.get("chunk_id")))
            reference = {"label": f"[Nguồn {entry['index']}]", "text": entry["context"]}
            if meta.get("excel_table_id"):
                reference["excel_table_id"] = meta["excel_table_id"]
                table = (record.get("table_data") or {}).get("excel") if isinstance(record.get("table_data"), dict) else None
                if table:
                    reference["excel_columns"] = table["columns"]
            if type(record.get("source_offset")) is int:
                reference["source_offset"] = record["source_offset"]
            references.append(reference)
        result = {"documents": references, "blocked_count": blocked_count,
                  "new_evidence": len(evidence) - previous_count}
        if isinstance(data.get("retrieval_quality"), dict):
            result["retrieval_quality"] = data["retrieval_quality"]
        if data.get("excel_match_count") is not None or str(data.get("status", "")).startswith("excel_"):
            result.update({key: data[key] for key in ("status", "excel_columns", "excel_units", "excel_match_count", "next_offset")
                           if key in data})
        return result

    def act(self, state: dict) -> dict:
        turn = state["react_turn"]
        decision = turn["decision"]
        kind, name, params = decision["kind"], decision["name"], decision["arguments"]
        evidence = list(turn["evidence"])
        attempts = list(turn["attempts"])
        snapshot = state["capabilities"]
        tools = snapshot["tool"]
        if kind == "skill":
            definition = resolve_skill(snapshot["skill"], name)
        else:
            definition = tools.get(name)
        tool_name = name
        error = ""
        try:
            params = self.guard_payload(params)
        except ValueError:
            error = _REACT_MESSAGES.get("guardrail_rejected_params", "")
        if definition is None:
            error = _REACT_MESSAGES.get("action_not_registered", "")
            name = "unknown"
        elif not error and kind == "skill":
            params, tool_name, error = self._prepare_skill(definition, params, evidence, tools)
        elif not error:
            params, error = self._guard_tool(name, params, tools)
        key = ""
        calls = list(state.get("tool_outputs", []))
        if not error and tool_name == LEGAL_WEB_SEARCH_TOOL and not legal_searches_remaining(calls):
            error = _REACT_MESSAGES.get("legal_search_limit") or "Legal search limit reached for this turn."
        if not error and state.get("worker") and tool_name in ONCE_PER_TURN:
            error = _REACT_MESSAGES.get("worker_readonly_only", "")
        if not error:
            if tool_name in ONCE_PER_TURN:
                key = tool_name
            else:
                # Fact-checking documents come from state and need not be stored twice.
                fingerprint_input = {k: v for k, v in params.items() if k != "documents"}
                if tool_name == LEGAL_WEB_SEARCH_TOOL:
                    fingerprint_input["query"] = search_key(params["query"])
                if kind == "tool" and name == "search_banking_knowledge" and params.get("offset") is not None and params.get("source"):
                    fingerprint_input.pop("query", None)
                    raw_src = str(params.get("source") or "").replace("\\", "/").casefold().rsplit("/", 1)[-1]
                    if "." in raw_src:
                        raw_stem = raw_src.rsplit(".", 1)[0]
                    else:
                        raw_stem = raw_src
                    src_parts = raw_stem.split()
                    if len(src_parts) >= 2 and src_parts[0].isalpha() and src_parts[1].isdigit():
                        fingerprint_input["source"] = src_parts[0] + " " + src_parts[1]
                    else:
                        fingerprint_input["source"] = raw_stem
                if kind == "skill" and name.replace("_", "-") == "fact-checking-audit":
                    fingerprint_input["evidence_count"] = len(evidence)
                if kind == "skill":
                    identity = definition["skill_id"]
                else:
                    identity = name
                key = kind + ":" + identity + ":" + json.dumps(fingerprint_input, ensure_ascii=False, sort_keys=True)
            if key in attempts:
                if kind == "tool" and name == "search_banking_knowledge" and params.get("offset") is not None:
                    error = _REACT_MESSAGES.get("offset_already_read_template", "").replace(
                        "{offset}", str(params.get("offset"))
                    )
                else:
                    error = _REACT_MESSAGES.get("action_already_executed", "")

        observation = {"kind": kind, "name": name, "status": "ERROR", "error": error}
        if error:
            observation["error_code"] = "invalid_action"
        intent = state["intent"]
        searches = state.get("search_count", 0)
        if not error:
            attempts.append(key)
            web_search = tool_name == LEGAL_WEB_SEARCH_TOOL
            retrieval = web_search or tool_name == "search_banking_knowledge"
            if kind == "skill" and definition["execution"]["type"] == "rag_pipeline":
                retrieval = True
            execution = self.tools.execute_tool
            arguments = (tool_name, params)
            kwargs = {}
            if kind == "tool" or definition["execution"]["type"] == "registered_tool":
                kwargs["definition"] = tools[tool_name]
            else:
                execution = self.skills.get_skill(name, definition=definition).execute
                arguments = (params,)
            with trace_span(
                f"{kind}.{name}",
                run_type="tool",
                kind=kind,
                action_name=name,
                tool_name=tool_name,
                params=_preview_trace_dict(params, max_chars=300),
            ) as span_details:
                result = execution(*arguments, **kwargs)
                retried = False
                elapsed = result.get("latency_ms", 0.0)
                if result.get("temporary") is True and tool_name not in ONCE_PER_TURN:
                    result = execution(*arguments, **kwargs)
                    elapsed += result.get("latency_ms", 0.0)
                    retried = True
                data = result.get("data")
                web_details = {}
                if web_search and isinstance(data, dict):
                    web_details = {key: data[key] for key in ("provider", "attempts") if key in data}
                    web_details["web_evidence_status"] = data.get("status")
                if result.get("status") == "SUCCESS" and isinstance(data, dict) and str(data.get("status", "")).lower() not in ("failed", "error"):
                    observation = {"kind": kind, "name": name, "status": "SUCCESS"}
                    simulated = tool_name in SIMULATED_TOOLS or data.get("simulation") is True
                    documents = data.get("documents") or data.get("chunks")
                    if isinstance(documents, list) and documents and all(_unwrap_doc_dict(item).get("content") for item in documents):
                        retrieval = True
                    if retrieval:
                        observation["retrieval"] = True
                        data = self._observe_retrieval({**data, "query": params.get("query", "")}, evidence)
                        data.update(web_details)
                        searches += 1
                        intent = "RAG_SEARCH"
                    elif intent != "RAG_SEARCH":
                        intent = "BANKING_ACTION"
                    if simulated:
                        observation["simulation"] = True
                    if retrieval:
                        observation["data"] = data
                        if not data["documents"]:
                            code = data.get("retrieval_quality", {}).get("source_status") or "insufficient_evidence"
                            observation.update(status="ERROR", error_code=code,
                                               error=_REACT_MESSAGES.get(code, _REACT_MESSAGES.get("insufficient_evidence", "")))
                    elif not verify_calculation(tool_name, data, params):
                        observation.update(status="ERROR", error_code="verification_failed",
                                           error=_REACT_MESSAGES.get("calculation_verification_failed", ""))
                    else:
                        try:
                            observation["data"] = self.guard_payload(data)
                        except ValueError:
                            observation.update(status="ERROR", error_code="unsafe_result",
                                               error=_REACT_MESSAGES.get("unsafe_result", ""))
                else:
                    observation["error"] = _REACT_MESSAGES.get("tool_failed", "")
                    observation["error_code"] = "tool_failed"
                    if result.get("error") == "ToolInputError":
                        observation.update(error_code="invalid_action", error=_REACT_MESSAGES.get("tool_input_error", ""))
                    elif result.get("error") == "ToolOutputError":
                        observation.update(error_code="invalid_tool_output", error=_REACT_MESSAGES.get("tool_output_error", ""))
                    elif result.get("temporary") is True:
                        observation["error_code"] = "temporary_failure"

                docs_count = None
                out_keys = []
                out_preview = None
                obs_data = observation.get("data")
                if observation.get("retrieval") and isinstance(obs_data, dict):
                    docs_val = obs_data.get("documents")
                    if isinstance(docs_val, list):
                        docs_count = len(docs_val)
                elif isinstance(obs_data, dict):
                    out_keys = list(obs_data.keys())[:20]
                    out_preview = _preview_trace_dict(obs_data, max_chars=240)

                span_details.update(
                    status=observation["status"],
                    error_code=observation.get("error_code"),
                    retried=retried,
                    latency_ms=elapsed,
                    simulation=bool(observation.get("simulation", False)),
                    retrieval=bool(observation.get("retrieval", False)),
                    documents_count=docs_count,
                    output_keys=out_keys,
                    output_preview=out_preview,
                    **web_details,
                )

            if retried:
                observation["retried"] = True

            if retrieval and isinstance(params.get("query"), str):
                observation["query_used"] = params["query"]

            calls.append({
                "kind": kind, "tool": name, "status": observation["status"],
                "latency_ms": elapsed,
                "simulation": observation.get("simulation", False),
                **web_details,
            })
            if retried:
                calls[-1]["retried"] = True
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
                "section_title": chunk.get("section_title", ""),
                "document_version": (chunk.get("business_metadata") or {}).get("document_version", "")
            })
        return citations
