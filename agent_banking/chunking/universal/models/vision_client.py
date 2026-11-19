"""Strict cloud layout client; model failures never become source text."""
import base64
import hashlib
import json
from agent_banking.chunking.config.models import VISION_CLIENT_SETTINGS
from agent_banking.chunking.config.prompt_config import (
    DOCUMENT_VISION_SCHEMA_REPAIR_PROMPT,
    DOCUMENT_VISION_SYSTEM_PROMPT,
    QWEN_VISION_SCHEMA_REPAIR_PROMPT,
    QWEN_VISION_SYSTEM_PROMPT,
    VISION_DEFAULT_ASK,
    VISION_DEFAULT_INSTRUCTIONS,
)
from agent_banking.core.models import model_primary
from agent_banking.core.fallback import ModelFallbackError

_VISION_CFG = VISION_CLIENT_SETTINGS
_MIN_HEADING_LEVEL = _VISION_CFG["min_heading_level"]
_MAX_HEADING_LEVEL = _VISION_CFG["max_heading_level"]
_ALLOWED_IMAGE_DETAILS = tuple(_VISION_CFG["allowed_image_details"])


def _ensure_env_loaded():
    from agent_banking.core.environment import load_environment
    load_environment()


_ensure_env_loaded()


class VisionPageOverflow(RuntimeError):
    """The page adapter must subdivide with explicit, recorded coordinates."""


class InvalidVisionLayout(ValueError):
    """A repeated request needs schema feedback."""
    retryable = False


def _heading_level(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Vision heading level must be an integer")
    return max(_MIN_HEADING_LEVEL, min(_MAX_HEADING_LEVEL, int(value)))


class QwenVisionClient:
    allow_partial_layout = False
    SCHEMA_REPAIR_PROMPT = "\n" + QWEN_VISION_SCHEMA_REPAIR_PROMPT
    SYSTEM_PROMPT = QWEN_VISION_SYSTEM_PROMPT

    def __init__(self, base_url=None, api_key=None, model=None, timeout=None,
                 max_retries=None, backoff_factor=_VISION_CFG["backoff_factor"],
                 max_tokens=_VISION_CFG["default_max_tokens"],
                 model_module=_VISION_CFG["default_model_module"], image_detail=None):
        if image_detail not in _ALLOWED_IMAGE_DETAILS:
            raise ValueError("Image detail must be auto, high or low")
        self._configured_image_detail = image_detail
        self.image_detail = image_detail
        self.model_module = model_module
        primary = model_primary(model_module)
        self.base_url = (base_url or primary.url).rstrip("/")
        self.api_key = primary.api_key
        if api_key is not None:
            self.api_key = api_key
        self.model = model or primary.model
        if image_detail is None:
            self.image_detail = primary.image_detail or None
        self.endpoint = primary
        self.timeout = primary.timeout
        if timeout is not None:
            self.timeout = timeout
        self.max_retries = primary.attempts - 1
        if max_retries is not None:
            self.max_retries = max_retries
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("Vision max_retries must be a nonnegative integer")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("Vision max_tokens must be a positive integer")
        self.backoff_factor = backoff_factor
        self.max_tokens = max_tokens

    def identity(self):
        return {"model": self.model, "module": self.model_module, "max_tokens": self.max_tokens,
                "image_detail": self.image_detail, "image_detail_override": self._configured_image_detail,
                "provider_sha256": hashlib.sha256(self.base_url.encode()).hexdigest()}

    def analyze_page_layout(self, image_bytes, instructions=None, *, extra_images=(), attempt_budget=None,
                            single_request=False, route_reason=None):
        from .text_client import CloudAttempts, MAX_CLOUD_ATTEMPTS, IncompleteModelResponse

        payload = self._build_payload(image_bytes, instructions)
        for label, region in extra_images:
            payload["messages"][1]["content"].append({"type": "text", "text": label})
            payload["messages"][1]["content"].append(self._image_part(region))
        last_error = None
        if attempt_budget is None:
            attempt_budget = CloudAttempts()
        if single_request:
            allowance = 1
        else:
            allowance = min(self.max_retries + 1, MAX_CLOUD_ATTEMPTS)
        for _ in range(allowance):
            if not attempt_budget.remaining:
                break
            try:
                return self._parse_and_validate_layout(self._execute_request_with_retry(
                    payload, attempt_budget=attempt_budget, single_request=single_request,
                    route_reason=route_reason,
                ))
            except (ValueError, ModelFallbackError) as error:
                if isinstance(error, ModelFallbackError):
                    if any(reason == "IncompleteModelResponse" or reason.endswith(" HTTP 413")
                           for _, reason in error.failures):
                        raise VisionPageOverflow("Vision request exceeded a provider size limit") from error
                    if isinstance(getattr(error, "last_error", None), ValueError):
                        error = error.last_error
                    else:
                        raise
                if isinstance(error, IncompleteModelResponse):
                    raise VisionPageOverflow("Vision output overflow requires explicit regional review") from error
                last_error = error
                if not single_request and attempt_budget.remaining:
                    payload["messages"][1]["content"][0]["text"] = (instructions or VISION_DEFAULT_INSTRUCTIONS) + self.SCHEMA_REPAIR_PROMPT
        raise InvalidVisionLayout("Vision schema repair failed: " + str(last_error))

    def _image_part(self, image_bytes):
        if isinstance(image_bytes, str):
            if image_bytes.startswith("data:image/") or image_bytes.startswith("http://") or image_bytes.startswith("https://"):
                image_url = {"url": image_bytes}
                if self.image_detail is not None:
                    image_url["detail"] = self.image_detail
                return {"type": "image_url", "image_url": image_url}
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise TypeError("image_bytes must be bytes or bytearray")
        if not image_bytes:
            raise ValueError("Image is empty")
        mime = "image/png"
        if image_bytes.startswith(b"\xff\xd8"):
            mime = "image/jpeg"
        elif image_bytes.startswith((b"GIF87a", b"GIF89a")):
            mime = "image/gif"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            mime = "image/webp"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_url = {"url": "data:" + mime + ";base64," + encoded}
        if self.image_detail is not None:
            image_url["detail"] = self.image_detail
        return {"type": "image_url", "image_url": image_url}

    def ask(self, image_data, prompt=VISION_DEFAULT_ASK, max_tokens=_VISION_CFG["default_ask_max_tokens"],
            temperature=_VISION_CFG["default_ask_temperature"], stream=False):
        import urllib.request
        from .text_client import apply_chat_profile, read_response

        image_part = self._image_part(image_data)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_part,
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        apply_chat_profile(payload, self.endpoint, max_tokens, structured=False)
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = read_response(resp, streaming=stream)
        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if not content:
                content = message.get("reasoning_content", "")
            return content
        return ""

    def _build_payload(self, image_bytes, instructions=None):
        prompt = instructions or VISION_DEFAULT_INSTRUCTIONS
        payload = {"model": self.model, "messages": [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prompt}, self._image_part(image_bytes)]}],
            "temperature": 0, "max_tokens": self.max_tokens, "response_format": {"type": "json_object"}}
        return payload

    def _configure_payload(self, payload, candidate):
        # Fallback may change provider, so use its request fields and copy
        # the image parts before changing provider-specific detail settings.
        detail = self._configured_image_detail
        if detail is None:
            detail = candidate.image_detail or None
        system, user = payload["messages"]
        content = []
        for part in user["content"]:
            if part["type"] == "image_url":
                image_url = dict(part["image_url"])
                if detail is None:
                    image_url.pop("detail", None)
                else:
                    image_url["detail"] = detail
                part = {"type": "image_url", "image_url": image_url}
            content.append(part)
        payload["messages"] = [system, {**user, "content": content}]

    def _execute_request_with_retry(self, payload, *, attempt_budget, single_request, route_reason=None):
        from .text_client import cloud_json

        def validate(data):
            try:
                self._parse_and_validate_layout(json.dumps(data, ensure_ascii=False))
            except ValueError as error:
                raise InvalidVisionLayout(str(error)) from error

        if single_request:
            request_retries = 0
        else:
            request_retries = self.max_retries
        result = cloud_json(self.base_url + "/chat/completions", self.api_key, payload,
                            timeout=self.timeout, retries=request_retries, allow_layout_list=True,
                            max_output_tokens=self.max_tokens, model_module=self.model_module, validate=validate,
                            configure_payload=self._configure_payload, attempt_budget=attempt_budget,
                            route_reason=route_reason)
        return json.dumps(result, ensure_ascii=False)

    def _clean_trailing_commas(self, text):
        output = []
        quoted = False
        escaped = False
        for index, character in enumerate(text):
            if quoted:
                output.append(character)
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
                continue
            if character == '"':
                quoted = True
            if character == ",":
                end = index + 1
                while end < len(text) and text[end].isspace():
                    end += 1
                if end < len(text) and text[end] in "]}":
                    continue
            output.append(character)
        return "".join(output)

    def _try_parse_json(self, text):
        if not isinstance(text, str):
            return None
        candidate = text.strip()
        if candidate.startswith("```"):
            start = candidate.find("\n")
            end = candidate.rfind("```")
            if start >= 0 and end > start:
                candidate = candidate[start + 1:end].strip()
        for value in (candidate, self._clean_trailing_commas(candidate)):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                pass
        return None

    def _parse_and_validate_layout(self, text):
        data = self._try_parse_json(text)
        if not isinstance(data, dict):
            raise ValueError("Vision response is not a layout JSON object")
        if "elements" not in data:
            for alias in ("layout_elements", "layout", "blocks"):
                if isinstance(data.get(alias), list):
                    data["elements"] = data[alias]
                    break
        if "elements" not in data or not isinstance(data["elements"], list):
            raise ValueError("Vision response lacks elements; keys=" + str(list(data)))
        invalid_records = 0
        for field, expected in (("headings", list), ("review", dict), ("uncertainties", list), ("_api", dict)):
            if field not in data or field == "uncertainties" and data[field] is None:
                continue
            if not isinstance(data[field], expected):
                if not self.allow_partial_layout:
                    raise ValueError("Invalid vision response field: " + field)
                data.pop(field)
                invalid_records += 1
        headings = []
        elements = []
        for heading in data.get("headings", []):
            if isinstance(heading, str):
                heading = {"text": heading, "level": 1}
            try:
                if not isinstance(heading, dict) or not isinstance(heading.get("text"), str):
                    raise ValueError("Malformed vision heading")
                level = _heading_level(heading.get("level", 1))
            except ValueError:
                if not self.allow_partial_layout:
                    raise
                invalid_records += 1
                continue
            if not heading["text"].strip():
                continue
            item = {"level": level, "text": heading["text"]}
            if "bbox" in heading:
                item["bbox"] = heading["bbox"]
            headings.append(item)
        heading_map = {heading["text"]: heading["level"] for heading in headings}
        for element in data["elements"]:
            try:
                item = self._layout_element(element, heading_map)
            except ValueError:
                if not self.allow_partial_layout:
                    raise
                invalid_records += 1
                continue
            if item["type"] == "heading" and item["text"].strip() and item["text"] not in heading_map:
                headings.append({"text": item["text"], "level": item["level"]})
                heading_map[item["text"]] = item["level"]
            elements.append(item)
        return {"headings": headings, "elements": elements, "api": data.get("_api", {}),
                "uncertainties": data.get("uncertainties"), "review": data.get("review", {}),
                "invalid_records": invalid_records}

    def _layout_element(self, element, heading_map):
        if not isinstance(element, dict) or not isinstance(element.get("text", ""), str):
            raise ValueError("Malformed vision element")
        item = dict(element)
        if item.get("role") is not None and not isinstance(item["role"], str):
            raise ValueError("Malformed vision element role")
        kind = str(item.get("type", "paragraph")).lower()
        if kind in ("header", "footer", "page_number", "footnote", "endnote", "annotation", "furniture"):
            item["role"] = kind
            kind = "paragraph"
        if kind.startswith("heading_"):
            suffix = kind.rsplit("_", 1)[-1]
            if suffix.isdecimal():
                item.setdefault("level", int(suffix))
            kind = "heading"
        if kind in ("figure", "chart", "diagram", "picture"):
            kind = "image"
        elif kind in ("bullet_points", "list", "list_item", "text", "callout"):
            kind = "paragraph"
        elif kind == "table_html" and isinstance(item.get("table_html", item.get("content")), str):
            item.setdefault("table_html", item.get("content"))
            kind = "table"
        if kind not in ("heading", "paragraph", "table", "image", "caption"):
            raise ValueError("Unknown layout element type: " + kind)
        item["type"] = kind
        if kind != "table" and not item.get("text") and isinstance(item.get("content"), str):
            item["text"] = item["content"]
        item.setdefault("text", "")
        if kind == "heading":
            item["level"] = _heading_level(item.get("level", heading_map.get(item["text"], 1)))
        if item.get("table_data") is not None and not isinstance(item["table_data"], (list, dict)):
            raise ValueError("Malformed vision table_data")
        if item.get("image_info") is not None and not isinstance(item["image_info"], dict):
            raise ValueError("Malformed vision image_info")
        return item


class DocumentVisionClient(QwenVisionClient):
    """Focused extraction schema; interpretation is delegated to the text LLM."""
    allow_partial_layout = True
    SCHEMA_REPAIR_PROMPT = "\n" + DOCUMENT_VISION_SCHEMA_REPAIR_PROMPT
    SYSTEM_PROMPT = DOCUMENT_VISION_SYSTEM_PROMPT

    def _parse_and_validate_layout(self, text):
        from ..extraction.pdf.layout import place_reported_headings

        layout = super()._parse_and_validate_layout(text)
        place_reported_headings(layout, allow_unlocated=True)
        return layout


# Vendor-neutral aliases
CloudVisionClient = QwenVisionClient
