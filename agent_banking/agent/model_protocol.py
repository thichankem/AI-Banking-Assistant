"""Structured model calls used by planning, delegation, memory and review."""

import json
import math
import os
import time

from jsonschema import Draft202012Validator

from agent_banking.agent.config.response_profiles import get_prompt
from agent_banking.agent.config.settings import get_float_setting, get_int_setting, get_setting
from agent_banking.core.cloud_rate_limit import retry_delay
from agent_banking.core.fallback import ModelFallbackError, is_temporary_error
from agent_banking.core.model_routing import apply_chat_options, ordered_candidates, request_metadata, usage_metadata
from agent_banking.tracing.agent_trace import agent_span


_CONTEXT_BYTES_DEFAULT = get_int_setting("context_bytes_default")
_PROMPT_OVERHEAD_BYTES = get_int_setting("prompt_overhead_bytes")
_MAX_STRUCTURED_RESPONSE_CHARS = get_int_setting("max_structured_response_chars")
_MAX_REQUEST_ATTEMPTS = get_int_setting("max_request_attempts")
_DEFAULT_REQUEST_MAX_TOKENS = get_int_setting("default_request_max_tokens")
_DEFAULT_TEMPERATURE = get_float_setting("default_temperature")
_RETRY_BACKOFF_CAP_SECONDS = get_int_setting("retry_backoff_cap_seconds")
_PACK_WEIGHTS = dict(get_setting("context_pack_weights") or {})
_REVIEW_FEEDBACK_MAX_CHARS = get_int_setting("review_feedback_max_chars")
_REVIEW_ISSUE_ITEM_MAX_CHARS = get_int_setting("review_issue_item_max_chars")
_REVIEW_ISSUE_PROBLEM_MAX_CHARS = get_int_setting("review_issue_problem_max_chars")

_JSON_REPAIR_STRICT_PROMPT = get_prompt("JSON_REPAIR_STRICT_PROMPT")
_JSON_REPAIR_ROUTED_PROMPT = get_prompt("JSON_REPAIR_ROUTED_PROMPT")
_ACTION_FIELDS_PREFIX = get_prompt("ACTION_FIELDS_PREFIX")


def object_schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object", "properties": properties,
        "required": required, "additionalProperties": False
    }


def text_schema(max_length: int) -> dict:
    return {"type": "string", "minLength": 1, "maxLength": max_length}


def action_schema(kind: str, properties: dict, optional: tuple = ()) -> dict:
    return object_schema(
        {"kind": {"const": kind}, **properties},
        ["kind"] + [name for name in properties if name not in optional]
    )


PLAN_ITEM = object_schema({
    "id": {"type": "integer", "minimum": 1, "maximum": 99},
    "task": text_schema(180),
    "status": {"enum": ["pending", "in_progress", "done", "blocked"]},
}, ["id", "task", "status"])

CLAIM = object_schema({
    "claim": text_schema(600),
    "draft_line": {"type": "integer", "minimum": 1, "maximum": 500},
    "source": text_schema(32),
    "quote": text_schema(720),
    "line": {"type": "integer", "minimum": 1, "maximum": 500},
}, ["source"])
CLAIM["allOf"] = [
    {"anyOf": [{"required": ["quote"]}, {"required": ["line"]}]},
    {"anyOf": [{"required": ["claim"]}, {"required": ["draft_line"]}]},
]

COVERAGE = object_schema({
    "item": text_schema(240),
    "status": {"enum": ["answered", "limited", "missing"]},
}, ["item", "status"])

REVIEW_ISSUE = object_schema({
    "item": text_schema(_REVIEW_ISSUE_ITEM_MAX_CHARS),
    "problem": text_schema(_REVIEW_ISSUE_PROBLEM_MAX_CHARS),
    "draft_line": {"type": "integer", "minimum": 1, "maximum": 500},
    "source": text_schema(32),
}, ["item", "problem"])

SCHEMAS = {
    "tool": action_schema("tool", {"name": text_schema(100), "arguments": {"type": "object"}}),
    "skill": action_schema("skill", {"name": text_schema(100), "arguments": {"type": "object"}}),
    "inspect_skill": action_schema("inspect_skill", {"name": text_schema(100), "reference": text_schema(240)},
                                   optional=("reference",)),
    "plan": action_schema("plan", {"items": {"type": "array", "items": PLAN_ITEM, "minItems": 1, "maxItems": 8}}),
    "delegate": action_schema("delegate", {"role": text_schema(50), "task": text_schema(600)}),
    "clarify": action_schema("clarify", {
        "question": text_schema(300),
        "options": {"type": "array", "items": text_schema(120), "minItems": 2, "maxItems": 4},
    }),
    "confirm": action_schema("confirm", {"proposal": text_schema(480)}),
    "remember": action_schema("remember", {
        "key": text_schema(48), "value": text_schema(300), "category": {"enum": ["preference", "fact", "pending"]}
    }),
    "forget": action_schema("forget", {"key": text_schema(48)}),
    "compact": action_schema("compact", {}),
    "final": action_schema("final", {
        "answer": text_schema(8000), "insufficient_evidence": {"type": "boolean"}
    }, optional=("insufficient_evidence",)),
}
REVIEW_SCHEMA = object_schema(
    {"verdict": {"enum": ["approve", "revise"]}, "feedback": {"type": "string", "maxLength": _REVIEW_FEEDBACK_MAX_CHARS},
     "coverage": {"type": "array", "items": COVERAGE, "minItems": 1, "maxItems": 16},
     "claims": {"type": "array", "items": CLAIM, "maxItems": 20},
     "issues": {"type": "array", "items": REVIEW_ISSUE, "maxItems": 4}},
    ["verdict", "feedback", "coverage", "claims"]
)
SUMMARY_SCHEMA = object_schema({"summary": text_schema(1800)}, ["summary"])


def _bound_review_advice(result):
    # Free-text suggestions are advisory. Evidence pointers and quotes stay strict.
    if isinstance(result.get("feedback"), str):
        result["feedback"] = result["feedback"][:_REVIEW_FEEDBACK_MAX_CHARS]
    issues = result.get("issues")
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, dict):
                for key, limit in (("item", _REVIEW_ISSUE_ITEM_MAX_CHARS), ("problem", _REVIEW_ISSUE_PROBLEM_MAX_CHARS)):
                    if isinstance(issue.get(key), str):
                        issue[key] = issue[key][:limit]
    coverage = result.get("coverage")
    if isinstance(coverage, list):
        for item in coverage:
            if isinstance(item, dict):
                if isinstance(item.get("item"), str):
                    item["item"] = item["item"][:240]
                if item.get("status") not in ("answered", "limited", "missing"):
                    item["status"] = "answered"


def pack_context(data: dict, budget: int, *, keep_evidence=False) -> str:
    """Keep complete records and reserve the request, live replies and review draft."""
    def encode(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    sized = tuple(_PACK_WEIGHTS.keys())
    if keep_evidence:
        sized = tuple(key for key in sized if key != "evidence")
    packed = {key: value for key, value in data.items() if key not in sized}
    total = len(encode(packed).encode("utf-8"))
    if total > budget:
        raise ModelFallbackError("agent", [("context", "required input exceeds the context budget")])
    weights = _PACK_WEIGHTS
    available = sum(weights[key] for key in sized if data.get(key))
    for key in sized:
        if key not in data:
            continue
        value = data[key]
        remaining = budget - total - len(key) - 6
        if remaining <= 0:
            continue
        if available and value:
            remaining = remaining * weights[key] // available
            available -= weights[key]
        if len(encode(value).encode("utf-8")) > remaining:
            if isinstance(value, list):
                reverse = key in ("history", "older_messages", "observations")
                if reverse:
                    items = reversed(value)
                else:
                    items = iter(value)
                selected = []
                for item in items:
                    size = len(encode(item).encode("utf-8")) + 1
                    if size <= remaining:
                        selected.append(item)
                        remaining -= size
                if reverse:
                    value = list(reversed(selected))
                else:
                    value = selected
            else:
                continue
        packed[key] = value
        total += len(encode(value).encode("utf-8")) + len(key) + 4
    result = encode(packed)
    if len(result.encode("utf-8")) > budget:
        raise ModelFallbackError("agent", [("context", "input exceeds the context budget")])
    return result


def _parse(text: str) -> dict:
    text = text.strip()
    while text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    fence = chr(96) * 3
    if text.startswith(fence) and text.endswith(fence):
        text = text.partition("\n")[2][:-3].strip()

    def reject_constant(value):
        raise ValueError("Non-finite JSON number")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            reject_constant(value)
        return number

    def distinct_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Repeated JSON key")
            result[key] = value
        return result

    result = json.loads(text, parse_constant=reject_constant, parse_float=finite_float, object_pairs_hook=distinct_keys)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    if result.get("kind") == "final":
        ans_text = str(result.get("answer") or "").strip()
        if not ans_text or not any(ch.isalnum() for ch in ans_text):
            raise ValueError("Final answer must contain readable text")
    return result


class StructuredModel:
    def __init__(self, client, model: str, *, review_client=None):
        self.client = client
        self.review_client = review_client
        self.model = model
        self.context_bytes = int(os.getenv("AGENT_CONTEXT_BYTES", str(_CONTEXT_BYTES_DEFAULT)))

    def request(self, system: str, data: dict, schemas: list[dict], max_tokens=None, *, concise=False,
                role="agent", deep_reason=None) -> dict:
        if max_tokens is None:
            max_tokens = _DEFAULT_REQUEST_MAX_TOKENS
        validators = [Draft202012Validator(schema) for schema in schemas]
        parsed = {}

        def validate(response):
            choice = response.choices[0]
            try:
                result = _parse(choice.message.content)
            except ValueError:
                if getattr(choice, "finish_reason", None) == "length":
                    raise ValueError("Structured response reached the token budget") from None
                raise
            if isinstance(result.get("kind"), str):
                fields = SCHEMAS.get(result["kind"], {}).get("properties")
            else:
                fields = None
            if fields:
                # A provider may add prose fields to an otherwise valid decision.
                # They have no effect on execution and must not enter the checkpoint.
                result = {key: value for key, value in result.items() if key in fields}
            if result.get("kind") == "final" and isinstance(result.get("answer"), str):
                result["answer"] = result["answer"][:8000]
            if any(schema is REVIEW_SCHEMA for schema in schemas):
                _bound_review_advice(result)
            if not any(validator.is_valid(result) for validator in validators):
                issues = []
                for validator in validators:
                    expected = validator.schema.get("properties", {}).get("kind", {}).get("const")
                    if expected is not None and result.get("kind") != expected:
                        continue
                    error = next(validator.iter_errors(result), None)
                    if error is not None:
                        path = ".".join(str(part) for part in error.path)
                        issues.append(str(error.validator) + ":" + path)
                raise ValueError("Model output does not match the response schema (" + ",".join(issues[:2]) + ")")
            if len(json.dumps(result, ensure_ascii=False)) > _MAX_STRUCTURED_RESPONSE_CHARS:
                raise ValueError("Structured response is too large")
            parsed.clear()
            parsed.update(result)

        contract = json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
        if getattr(self.client, "routed", False) is True:
            self._routed_request(system, data, contract, max_tokens, validate, role, deep_reason)
            return parsed
        # Account for prompts and schema repair on every attempt; reserve output headroom.
        reserved = len(system.encode("utf-8")) + len(contract.encode("utf-8")) + max_tokens * 4 + _PROMPT_OVERHEAD_BYTES
        payload = pack_context(data, self.context_bytes - reserved, keep_evidence=role == "agent_review")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": payload}]
        review_thinking_rejected = False
        for attempt in range(_MAX_REQUEST_ATTEMPTS):
            options = {}
            keep_review_concise = role == "agent_review" and not review_thinking_rejected
            if concise and (keep_review_concise or role != "agent_review" and attempt < _MAX_REQUEST_ATTEMPTS - 1):
                options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            if attempt == 1:
                options["response_format"] = {"type": "json_object"}
            try:
                self.client.create(
                    model=self.model, messages=messages, validate=validate,
                    temperature=_DEFAULT_TEMPERATURE, max_tokens=max_tokens, **options
                )
                return parsed
            except ModelFallbackError as error:
                cause = getattr(error, "last_error", None)
                rejected_mode = bool(options) and getattr(cause, "status_code", None) in (400, 422)
                if keep_review_concise and rejected_mode and "response_format" not in options:
                    # This request used only the thinking toggle as an optional provider setting.
                    review_thinking_rejected = True
                temporary = is_temporary_error(error)
                if attempt == _MAX_REQUEST_ATTEMPTS - 1 or not (temporary or rejected_mode or isinstance(cause, (ValueError, TypeError))):
                    raise
                if temporary:
                    time.sleep(retry_delay(error, 2 ** (attempt + 1)))
                    continue
                repair_suffix = _JSON_REPAIR_STRICT_PROMPT.replace("{contract}", contract)
                messages = [
                    {"role": "system", "content": system + "\n" + repair_suffix},
                    {"role": "user", "content": payload}
                ]
        raise RuntimeError("Unreachable structured model state")

    def _routed_request(self, system, data, contract, max_tokens, validate, role, deep_reason):
        review = role == "agent_review"
        clients = [self.client]
        if review and self.review_client is not None:
            clients.insert(0, self.review_client)
        routes = [(client, candidate) for client in clients
                  for candidate in ordered_candidates(client.candidates, bool(deep_reason))
                  if candidate.protocol == "chat"]
        failures = []
        last_error = None
        calls = 0
        repair = False
        context_limited = False
        while routes and calls < _MAX_REQUEST_ATTEMPTS:
            client, candidate = routes.pop(0)
            output_tokens = min(max_tokens, candidate.max_output_tokens or max_tokens)
            prompt = system
            if repair:
                prompt += "\n" + _JSON_REPAIR_ROUTED_PROMPT.replace("{contract}", contract)
            overhead = max(
                len(prompt.encode("utf-8")),
                len(system.encode("utf-8")) + len(contract.encode("utf-8")),
            ) + _PROMPT_OVERHEAD_BYTES
            budget = self.context_bytes - overhead - output_tokens * 4
            if candidate.context_window:
                # Bound unknown tokenizers conservatively using UTF-8 input bytes.
                budget = candidate.context_window - overhead - output_tokens
                if "AGENT_CONTEXT_BYTES" in os.environ:
                    budget = min(budget, self.context_bytes - overhead - output_tokens * 4)
            try:
                packed = pack_context(data, budget, keep_evidence=review)
            except ModelFallbackError:
                context_limited = True
                failures.append((candidate.profile, "context_budget_exceeded"))
                with agent_span("model.route", role=role, alias=candidate.profile,
                                reason="context_budget_exceeded", input_budget_bytes=budget):
                    pass
                continue
            payload = {"messages": [{"role": "system", "content": prompt}, {"role": "user", "content": packed}],
                       "temperature": _DEFAULT_TEMPERATURE}
            apply_chat_options(payload, candidate, output_tokens, repair=repair, sdk=True)
            reason = deep_reason or "standard"
            if last_error is not None:
                if repair:
                    reason = "contract_repair"
                else:
                    reason = "provider_fallback"
            elif context_limited:
                reason = "larger_context"
            elif candidate.profile == "standard" and deep_reason:
                reason = "standard_fallback"
            calls += 1
            with agent_span("model.attempt", role=role, configured_role=client.module,
                            reason=reason, attempt=calls, input_budget_bytes=budget,
                            input_bytes=len(packed.encode("utf-8")), **request_metadata(payload, candidate)) as details:
                try:
                    client.create(_candidate=candidate, _usage=lambda usage: details.update(usage=usage_metadata(usage)),
                                  validate=validate, **payload)
                    details["result"] = "success"
                    return
                except Exception as error:
                    last_error = error
                    status = getattr(error, "status_code", None)
                    repair = isinstance(error, (ValueError, TypeError)) or status in (400, 422)
                    details.update(result="failed", error_type=type(error).__name__)
                    if type(status) is int:
                        details["http_status"] = status
                    failures.append((candidate.profile, type(error).__name__))
            if calls < _MAX_REQUEST_ATTEMPTS and is_temporary_error(last_error):
                time.sleep(retry_delay(last_error, min(_RETRY_BACKOFF_CAP_SECONDS, 2 ** calls)))
            if repair or is_temporary_error(last_error) or calls < candidate.attempts:
                routes.append((client, candidate))
        error = ModelFallbackError(role, failures or [("configuration", "no eligible model")])
        error.last_error = last_error
        if calls == 0 and context_limited:
            error.category = "context_budget_exceeded"
        raise error from None

    def decide(self, system: str, data: dict, allowed: list[str], max_tokens=None, *, deep_reason=None) -> dict:
        if max_tokens is None:
            max_tokens = _DEFAULT_REQUEST_MAX_TOKENS
        # Show the action shapes up front to avoid a repair call for an unknown action shape.
        action_fields = {kind: list(SCHEMAS[kind]["properties"])[1:] for kind in allowed}
        system += "\n" + _ACTION_FIELDS_PREFIX + " " + json.dumps(action_fields, ensure_ascii=False)
        return self.request(system, data, [SCHEMAS[kind] for kind in allowed], max_tokens=max_tokens,
                            concise=True, deep_reason=deep_reason)
