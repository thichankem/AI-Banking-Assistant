"""Request and validate JSON responses from configured cloud model chains."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import logging
import time
import urllib.error
import urllib.request


logger = logging.getLogger("chunking.models")
MAX_CLOUD_ATTEMPTS = 3


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
        futures = {executor.submit(operation, item): index for index, item in enumerate(items)}
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
        if event_count % 500 == 0:
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
    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, dict):
        raise ValueError("Decision API did not return an answers object")
    for name, question in questions.items():
        answer = answers.get(name)
        if (not isinstance(answer, dict) or not isinstance(answer.get("choice"), str)
                or answer["choice"] not in question["criteria"]):
            raise ValueError("Decision API returned no valid choice for " + name)


def cloud_json(endpoint, key, payload, timeout=None, retries=None, allow_layout_list=False,
               max_output_tokens=131072, model_module=None, validate=None, dimension=None,
               configure_payload=None, attempt_budget=None):
    from agent_banking.core.fallback import run_fallback
    from agent_banking.core.models import get_model_chain

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
    request_payloads = {}

    def call(candidate):
        if candidate not in request_payloads:
            request_payload = deepcopy(payload)
            request_payload["model"] = candidate.model
            if configure_payload is not None:
                configure_payload(request_payload, candidate)
            ceiling = max_output_tokens
            if chat_profile:
                ceiling = apply_chat_profile(request_payload, candidate, ceiling)
            request_payloads[candidate] = (request_payload, ceiling)
        request_payload, ceiling = request_payloads[candidate]
        request_url = candidate.url.rstrip("/") + path
        request_timeout = candidate.timeout
        if chunking_route and timeout is not None:
            request_timeout = min(request_timeout, timeout)
        if attempt_budget is not None:
            attempt_budget.consume()
        result = _cloud_json(request_url, candidate.api_key, request_payload, request_timeout,
                             0, allow_layout_list, ceiling)
        if path == "/systemone":
            validate_decisions(result, payload.get("questions", {}))
        if validate is not None:
            validate(result)
        return result

    return run_fallback(model_module, candidates, call)


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


def _cloud_json(endpoint, key, payload, timeout=240, retries=2, allow_layout_list=False, max_output_tokens=131072):
    if not key:
        raise RuntimeError("Required cloud API credential is not configured")
    for attempt in range(retries + 1):
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = read_response(response, payload.get("stream", False))
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
                    payload[limit_key] = min(max_output_tokens, max(16384, payload.get(limit_key, 8192) * 2))
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
                    body = json.loads(error.read(16384))
                    entry = body.get("error", body) if isinstance(body, dict) else body
                    if isinstance(entry, dict):
                        detail = str(entry.get("message", entry.get("detail", entry.get("code", ""))))
                    elif isinstance(entry, str):
                        detail = entry
                except (ValueError, OSError):
                    pass
                detail = detail.replace(key, "[redacted]")[:1000]
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
        time.sleep(min(8, 2 ** attempt))
    raise RuntimeError("Cloud request failed")
