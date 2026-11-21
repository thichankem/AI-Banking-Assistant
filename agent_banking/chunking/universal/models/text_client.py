"""Request and validate JSON responses from configured cloud model chains."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
from dataclasses import replace
import json
import logging
import time
import urllib.error
import urllib.request

from agent_banking.chunking.config.models import TEXT_CLIENT_SETTINGS
from agent_banking.core.model_routing import apply_chat_options, ordered_candidates, request_metadata, usage_metadata
from agent_banking.tracing.tracer import annotate_span, trace_span

logger = logging.getLogger("chunking.models")
_TEXT_CFG = TEXT_CLIENT_SETTINGS
MAX_CLOUD_ATTEMPTS = _TEXT_CFG["max_cloud_attempts"]
_DEFAULT_MAX_OUTPUT_TOKENS = _TEXT_CFG["default_max_output_tokens"]
_CONTEXT_RESERVE_TOKENS = _TEXT_CFG["context_reserve_tokens"]
_DEFAULT_TIMEOUT = _TEXT_CFG["default_timeout"]
_DEFAULT_RETRIES = _TEXT_CFG["default_retries"]
_MAX_BACKOFF_SECONDS = _TEXT_CFG["max_backoff_seconds"]
_STREAM_LOG_INTERVAL = _TEXT_CFG["stream_log_interval"]
_ERROR_BODY_READ_LIMIT = _TEXT_CFG["error_body_read_limit"]
_ERROR_DETAIL_MAX_CHARS = _TEXT_CFG["error_detail_max_chars"]
_LENGTH_RETRY_FLOOR = _TEXT_CFG["length_retry_floor"]
_LENGTH_RETRY_MULTIPLIER = _TEXT_CFG["length_retry_multiplier"]
_DEFAULT_LENGTH_FALLBACK = _TEXT_CFG["default_length_fallback"]
_ROUTED_MULTI_CANDIDATE_ATTEMPTS = _TEXT_CFG["routed_multi_candidate_attempts"]
_ROUTED_ESCALATION_REASONS = tuple(_TEXT_CFG["routed_escalation_reasons"])


class CloudAttemptLimit(RuntimeError):
    retryable = False


class CloudAttempts:
    """Share an HTTP attempt budget across retries of one model operation."""

    def __init__(self, limit=MAX_CLOUD_ATTEMPTS):
        self.remaining = min(limit, MAX_CLOUD_ATTEMPTS)

    def consume(self):
        if self.remaining <= 0:
            raise CloudAttemptLimit("Cloud request attempt limit reached")
        self.remaining -= 1


class IncompleteModelResponse(ValueError):
    retryable = False

    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage or {}


def ordered_parallel(operation, items, workers):
    if workers == 1:
        return [operation(item) for item in items]
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(copy_context().run, operation, item): index for index, item in enumerate(items)}
        results = [None] * len(futures)
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return results


def read_response(response, streaming=False):
    if not streaming:
        return json.load(response)
    content = []
    model = None
    usage = {}
    finish = None
    fields = []
    completed = False
    event_count = 0

    def consume(event):
        nonlocal model, usage, finish, completed, event_count
        if event == "[DONE]":
            completed = True
            return
        data = json.loads(event)
        if not isinstance(data, dict) or "error" in data:
            raise ValueError("Cloud stream returned an invalid event")
        event_count += 1
        if event_count % _STREAM_LOG_INTERVAL == 0:
            logger.info("Model stream active: %s events, %s final-content characters", event_count, sum(len(part) for part in content))
        model = data.get("model", model)
        if data.get("usage"):
            usage = data["usage"]
        for choice in data.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            text = (choice.get("delta") or {}).get("content")
            if isinstance(text, str):
                content.append(text)
            # Reasoning deltas are intentionally neither stored nor exposed.
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            fields.append(line[5:].lstrip())
        elif not line and fields:
            consume("\n".join(fields))
            fields = []
            if completed:
                break
    if fields:
        consume("\n".join(fields))
    if not completed or finish is None:
        raise ValueError("Cloud stream ended before a complete response")
    logger.info("Cloud stream finished: reason=%s, events=%s, final_chars=%s, usage=%s",
                finish, event_count, sum(len(part) for part in content), usage)
    return {"model": model, "usage": usage,
            "choices": [{"finish_reason": finish, "message": {"content": "".join(content)}}]}


def strict_json(text, allow_layout_list=False):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Model did not return final JSON content")
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[0].strip().casefold() in ("```", "```json") and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1])
    value = json.loads(candidate, parse_constant=_reject_nonfinite, object_pairs_hook=_strict_object)
    if allow_layout_list and isinstance(value, list) and all(isinstance(item, dict) and "type" in item for item in value):
        value = {"headings": [], "elements": value}
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def _reject_nonfinite(_):
    raise ValueError("Model JSON contains a non-finite number")


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Model JSON contains a duplicate object key")
        value[key] = item
    return value


def validate_decisions(result, questions):
    if isinstance(result, dict):
        answers = result.get("answers")
    else:
        answers = None
    if not isinstance(answers, dict):
        raise ValueError("Decision API did not return an answers object")
    for name, question in questions.items():
        answer = answers.get(name)
        if (not isinstance(answer, dict) or not isinstance(answer.get("choice"), str)
                or answer["choice"] not in question["criteria"]):
            raise ValueError("Decision API returned no valid choice for " + name)


def cloud_json(endpoint, key, payload, timeout=None, retries=None, allow_layout_list=False,
               max_output_tokens=_DEFAULT_MAX_OUTPUT_TOKENS, model_module=None, validate=None, dimension=None,
               configure_payload=None, attempt_budget=None, route_reason=None):
    from agent_banking.core.fallback import run_fallback
    from agent_banking.core.models import auto_routing_enabled, get_model_chain

    endpoint = endpoint.rstrip("/")
    if model_module is None:
        model_module = "chunking.text"
        if endpoint.endswith("/systemone"):
            model_module = "chunking.decision"
        else:
            for message in payload.get("messages", []):
                content = message.get("content")
                if isinstance(content, list) and any(
                    isinstance(part, dict) and part.get("type") == "image_url" for part in content
                ):
                    model_module = "chunking.vision"
    path = "/chat/completions"
    base_url = endpoint
    if model_module.endswith("decision") or model_module == "decision":
        path = "/systemone"
    elif model_module == "embedding":
        path = ""
    if path and base_url.endswith(path):
        base_url = base_url[:-len(path)]
    primary = {"model": payload.get("model"), "url": base_url, "api_key": key, "timeout": timeout}
    if dimension is not None:
        primary["dimension"] = dimension
    if retries is not None:
        primary["attempts"] = retries + 1
    chat_profile = model_module in ("chunking.text", "chunking.vision", "chunking.vision_review", "vision", "vision_review")
    chunking_route = chat_profile or model_module == "chunking.decision"
    if chunking_route:
        primary["timeout"] = None
        if attempt_budget is None:
            attempt_budget = CloudAttempts()
    candidates = get_model_chain(model_module, primary)
    routed = chunking_route and auto_routing_enabled()
    if routed:
        reason = route_reason
        if reason is None and model_module in ("chunking.vision_review", "vision_review"):
            reason = "visual_review"
        candidates = ordered_candidates(candidates, reason in _ROUTED_ESCALATION_REASONS)
        route_reason = reason
        if len(candidates) > 1:
            candidates = [replace(candidate, attempts=_ROUTED_MULTI_CANDIDATE_ATTEMPTS) for candidate in candidates]
    request_payloads = {}
    attempts = 0

    def call(candidate):
        nonlocal attempts
        if candidate not in request_payloads:
            request_payload = deepcopy(payload)
            request_payload["model"] = candidate.model
            if configure_payload is not None:
                configure_payload(request_payload, candidate)
            ceiling = max_output_tokens
            if chat_profile:
                if routed:
                    ceiling = _routed_chat_profile(request_payload, candidate, ceiling)
                else:
                    ceiling = apply_chat_profile(request_payload, candidate, ceiling)
            request_payloads[candidate] = (request_payload, ceiling)
        request_payload, ceiling = request_payloads[candidate]
        request_url = candidate.url.rstrip("/") + path
        request_timeout = candidate.timeout
        if chunking_route and timeout is not None:
            request_timeout = min(request_timeout, timeout)
        if attempts:
            span_reason = "provider_fallback"
        elif route_reason:
            span_reason = route_reason
        else:
            span_reason = "standard"
        with trace_span("model.attempt", run_type="llm", role=model_module,
                        reason=span_reason,
                        **request_metadata(request_payload, candidate)) as details:
            if routed and candidate.context_window and _input_size(request_payload, candidate) + _CONTEXT_RESERVE_TOKENS > candidate.context_window:
                details.update(result="skipped", reason="context_budget_exceeded")
                raise ModelContextLimit("context_budget_exceeded")
            if attempt_budget is not None:
                if not attempt_budget.remaining:
                    details["result"] = "skipped"
                attempt_budget.consume()
            attempts += 1
            try:
                result = _cloud_json(request_url, candidate.api_key, request_payload, request_timeout,
                                     0, allow_layout_list, ceiling)
                usage = (result.get("_api") or {}).get("usage") or result.get("usage")
                details["usage"] = usage_metadata(usage)
                if path == "/systemone":
                    validate_decisions(result, payload.get("questions", {}))
                if validate is not None:
                    validate(result)
            except Exception as error:
                details["result"] = "failed"
                if isinstance(error, IncompleteModelResponse):
                    details["usage"] = usage_metadata(error.usage)
                status = getattr(error, "status_code", None)
                if type(status) is int:
                    details["http_status"] = status
                raise
            details["result"] = "success"
            if routed and path == "/systemone":
                result["_routing"] = {"alias": candidate.profile, "model": candidate.model}
            return result

    return run_fallback(model_module, candidates, call)


class ModelContextLimit(ValueError):
    retryable = False


def _input_size(payload, candidate):
    if candidate.protocol == "systemone":
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    size = payload[candidate.token_parameter]
    for message in payload["messages"]:
        parts = message["content"]
        if isinstance(parts, str):
            size += len(parts.encode("utf-8"))
        else:
            for part in parts:
                if part["type"] == "image_url":
                    size += candidate.image_token_reserve
                else:
                    size += len(part.get("text", "").encode("utf-8"))
    return size


def _routed_chat_profile(payload, candidate, ceiling):
    requested = payload.get("max_tokens") or payload.get("max_completion_tokens") or ceiling
    ceiling = min(ceiling, candidate.max_output_tokens or ceiling)
    payload.pop("chat_template_kwargs", None)
    apply_chat_options(payload, candidate, min(requested, ceiling))
    for message in payload.get("messages", []):
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url" and candidate.image_detail:
                    part["image_url"].setdefault("detail", candidate.image_detail)
    return ceiling


def apply_chat_profile(payload, candidate, ceiling, *, structured=True):
    requested = payload.pop("max_tokens", None)
    completion = payload.pop("max_completion_tokens", None)
    ceiling = min(ceiling, candidate.max_output_tokens or ceiling)
    payload[candidate.token_parameter] = min(requested or completion or ceiling, ceiling)
    if structured and candidate.json_mode == "on":
        payload["response_format"] = {"type": "json_object"}
    elif candidate.json_mode == "off":
        payload.pop("response_format", None)
    if not candidate.send_temperature:
        payload.pop("temperature", None)
    elif candidate.temperature is not None:
        payload["temperature"] = candidate.temperature
    payload.pop("chat_template_kwargs", None)
    if candidate.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    for message in payload.get("messages", []):
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url" and candidate.image_detail:
                    part["image_url"].setdefault("detail", candidate.image_detail)
    return ceiling


def _cloud_json(endpoint, key, payload, timeout=_DEFAULT_TIMEOUT, retries=_DEFAULT_RETRIES,
                allow_layout_list=False, max_output_tokens=_DEFAULT_MAX_OUTPUT_TOKENS):
    from agent_banking.core.cloud_rate_limit import acquire_cloud_slot
    if not key:
        raise RuntimeError("Required cloud API credential is not configured")
    for attempt in range(retries + 1):
        acquire_cloud_slot(str(payload.get("model", "")), endpoint)
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = read_response(response, payload.get("stream", False))
            if isinstance(body, dict):
                annotate_span(usage=usage_metadata(body.get("usage")))
            if not isinstance(body, dict) or "error" in body:
                raise ValueError("Cloud API did not return a successful response object")
            if "choices" not in body:
                return body
            if not isinstance(body["choices"], list) or not body["choices"] or not isinstance(body["choices"][0], dict):
                raise ValueError("Empty model response choices")
            choice = body["choices"][0]
            if choice.get("finish_reason") not in (None, "stop"):
                limit_key = "max_tokens"
                if "max_completion_tokens" in payload:
                    limit_key = "max_completion_tokens"
                if choice.get("finish_reason") == "length" and payload.get(limit_key, 0) >= max_output_tokens:
                    raise IncompleteModelResponse("Incomplete model response: length", body.get("usage", {}))
                if choice.get("finish_reason") == "length" and payload.get(limit_key, 0) < max_output_tokens:
                    payload[limit_key] = min(
                        max_output_tokens,
                        max(_LENGTH_RETRY_FLOOR, payload.get(limit_key, _DEFAULT_LENGTH_FALLBACK) * _LENGTH_RETRY_MULTIPLIER),
                    )
                raise ValueError("Incomplete model response: " + str(choice.get("finish_reason")))
            message = choice.get("message")
            if not isinstance(message, dict):
                raise ValueError("Cloud API did not return a message object")
            parsed = strict_json(message.get("content"), allow_layout_list)
            parsed["_api"] = {"model": body.get("model") or payload.get("model"), "usage": body.get("usage") or {}}
            return parsed
        except urllib.error.HTTPError as error:
            if error.code < 500 and error.code != 429:
                detail = ""
                try:
                    body = json.loads(error.read(_ERROR_BODY_READ_LIMIT))
                    if isinstance(body, dict):
                        entry = body.get("error", body)
                    else:
                        entry = body
                    if isinstance(entry, dict):
                        detail = str(entry.get("message", entry.get("detail", entry.get("code", ""))))
                    elif isinstance(entry, str):
                        detail = entry
                except (ValueError, OSError):
                    pass
                detail = detail.replace(key, "[redacted]")[:_ERROR_DETAIL_MAX_CHARS]
                failure = RuntimeError("Cloud API rejected request with HTTP " + str(error.code) + ": " + detail)
                failure.status_code = error.code
                failure.retryable = error.code == 408
                raise failure from None
            if attempt == retries:
                failure = RuntimeError("Cloud API unavailable, HTTP " + str(error.code))
                failure.status_code = error.code
                raise failure from None
        except (OSError, TimeoutError, ValueError) as error:
            if isinstance(error, IncompleteModelResponse):
                raise
            logger.warning("Cloud request attempt %s failed (%s)", attempt + 1, type(error).__name__)
            if attempt == retries:
                raise
        time.sleep(min(_MAX_BACKOFF_SECONDS, 2 ** attempt))
    raise RuntimeError("Cloud request failed")
