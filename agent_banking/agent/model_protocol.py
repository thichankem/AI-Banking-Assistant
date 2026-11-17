"""Structured model calls used by planning, delegation, memory and review."""

import json
import math
import os

from jsonschema import Draft202012Validator

from agent_banking.core.fallback import ModelFallbackError


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
    "claim": text_schema(420),
    "source": text_schema(32),
    "quote": text_schema(720),
    "line": {"type": "integer", "minimum": 1, "maximum": 500},
}, ["claim", "source"])
CLAIM["anyOf"] = [{"required": ["quote"]}, {"required": ["line"]}]

COVERAGE = object_schema({
    "item": text_schema(240),
    "status": {"enum": ["answered", "limited", "missing"]},
}, ["item", "status"])

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
    {"verdict": {"enum": ["approve", "revise"]}, "feedback": {"type": "string", "maxLength": 640},
     "coverage": {"type": "array", "items": COVERAGE, "minItems": 1, "maxItems": 16},
     "claims": {"type": "array", "items": CLAIM, "maxItems": 20}},
    ["verdict", "feedback", "coverage", "claims"]
)
SUMMARY_SCHEMA = object_schema({"summary": text_schema(1800)}, ["summary"])


def pack_context(data: dict, budget: int) -> str:
    """Keep complete records and reserve the request, live replies and review draft."""
    def encode(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    sized = ("available_evidence", "evidence", "observations", "history", "older_messages", "memory", "user_memory")
    packed = {key: value for key, value in data.items() if key not in sized}
    total = len(encode(packed).encode("utf-8"))
    if total > budget:
        raise ModelFallbackError("agent", [("context", "required input exceeds the context budget")])
    weights = {"available_evidence": 56, "evidence": 56, "observations": 25, "history": 13,
               "older_messages": 80, "memory": 7, "user_memory": 7}
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
                items = reversed(value) if reverse else iter(value)
                selected = []
                for item in items:
                    size = len(encode(item).encode("utf-8")) + 1
                    if size <= remaining:
                        selected.append(item)
                        remaining -= size
                value = list(reversed(selected)) if reverse else selected
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
    return result


class StructuredModel:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.context_bytes = int(os.getenv("AGENT_CONTEXT_BYTES", "65536"))

    def request(self, system: str, data: dict, schemas: list[dict], max_tokens=3000, *, concise=False) -> dict:
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
            fields = SCHEMAS.get(result["kind"], {}).get("properties") if isinstance(result.get("kind"), str) else None
            if fields:
                # A provider may add prose fields to an otherwise valid decision.
                # They have no effect on execution and must not enter the checkpoint.
                result = {key: value for key, value in result.items() if key in fields}
            if not any(validator.is_valid(result) for validator in validators):
                raise ValueError("Model output does not match the response schema")
            if len(json.dumps(result, ensure_ascii=False)) > 16000:
                raise ValueError("Structured response is too large")
            parsed.update(result)

        contract = json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
        # Account for prompts and schema repair on every attempt; reserve output headroom.
        reserved = len(system.encode("utf-8")) + len(contract.encode("utf-8")) + max_tokens * 4 + 512
        payload = pack_context(data, self.context_bytes - reserved)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": payload}]
        for attempt in range(3):
            options = {}
            if concise and attempt < 2:
                options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            if attempt == 1:
                options["response_format"] = {"type": "json_object"}
            try:
                self.client.create(
                    model=self.model, messages=messages, validate=validate,
                    temperature=0.1, max_tokens=max_tokens, **options
                )
                return parsed
            except ModelFallbackError as error:
                cause = getattr(error, "last_error", None)
                rejected_mode = bool(options) and getattr(cause, "status_code", None) in (400, 422)
                if attempt == 2 or not (rejected_mode or isinstance(cause, (ValueError, TypeError))):
                    raise
                messages = [
                    {"role": "system", "content": system + (
                        "\nChỉ trả một đối tượng JSON đầy đủ theo một schema sau. Kiểm tra mọi trường bắt buộc: "
                        + contract
                    )},
                    {"role": "user", "content": payload}
                ]
        raise RuntimeError("Unreachable structured model state")

    def decide(self, system: str, data: dict, allowed: list[str], max_tokens=3000) -> dict:
        # Show the action shapes up front to avoid a repair call for an unknown action shape.
        action_fields = {kind: list(SCHEMAS[kind]["properties"])[1:] for kind in allowed}
        system += "\nTrường JSON theo kind: " + json.dumps(action_fields, ensure_ascii=False)
        return self.request(system, data, [SCHEMAS[kind] for kind in allowed], max_tokens=max_tokens)

