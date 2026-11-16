"""Structured model calls used by planning, delegation, memory and review."""

import json
import math

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

SCHEMAS = {
    "tool": action_schema("tool", {"name": text_schema(100), "arguments": {"type": "object"}}),
    "skill": action_schema("skill", {"name": text_schema(100), "arguments": {"type": "object"}}),
    "plan": action_schema("plan", {"items": {"type": "array", "items": PLAN_ITEM, "minItems": 1, "maxItems": 8}}),
    "delegate": action_schema("delegate", {"role": text_schema(50), "task": text_schema(600)}),
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
    {"verdict": {"enum": ["approve", "revise"]}, "feedback": {"type": "string", "maxLength": 640}},
    ["verdict", "feedback"]
)
SUMMARY_SCHEMA = object_schema({"summary": text_schema(1800)}, ["summary"])


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

    def request(self, system: str, data: dict, schemas: list[dict], max_tokens=3000) -> dict:
        validators = [Draft202012Validator(schema) for schema in schemas]
        parsed = {}

        def validate(response):
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise ValueError("Structured response reached the token budget")
            result = _parse(choice.message.content)
            if not any(validator.is_valid(result) for validator in validators):
                raise ValueError("Model output does not match the response schema")
            if len(json.dumps(result, ensure_ascii=False)) > 16000:
                raise ValueError("Structured response is too large")
            parsed.update(result)

        payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": payload}]
        for attempt in range(3):
            options = {}
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
                rejected_mode = attempt == 1 and getattr(cause, "status_code", None) in (400, 422)
                if attempt == 2 or not (rejected_mode or isinstance(cause, (ValueError, TypeError))):
                    raise
                contract = json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
                messages = [
                    {"role": "system", "content": system + (
                        "\nChỉ trả một đối tượng JSON đầy đủ theo một schema sau. Kiểm tra mọi trường bắt buộc: "
                        + contract
                    )},
                    {"role": "user", "content": payload}
                ]
        raise RuntimeError("Unreachable structured model state")

    def decide(self, system: str, data: dict, allowed: list[str], max_tokens=3000) -> dict:
        return self.request(system, data, [SCHEMAS[kind] for kind in allowed], max_tokens=max_tokens)

