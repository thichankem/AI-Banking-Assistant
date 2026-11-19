"""Shared provider options and diagnostic metadata for model routing."""

from agent_banking.core.config.settings import CoreConfig
from agent_banking.core.models import model_configured

_MODELS_CFG = CoreConfig.CORE_MODELS
_ROUTING_CFG = CoreConfig.CORE_MODEL_ROUTING
_DEFAULT_PROFILE = _MODELS_CFG["default_profile"]
_DEEP_PROFILE = _MODELS_CFG["deep_profile"]
_METADATA_KEYS = tuple(_ROUTING_CFG["metadata_payload_keys"])
_USAGE_KEYS = tuple(_ROUTING_CFG["usage_keys"])
_VISION_PROTOCOL = _ROUTING_CFG["vision_protocol"]
_THINKING_TEMPLATE_KEY = _ROUTING_CFG["thinking_template_key"]
_THINKING_ENABLE_KEY = _ROUTING_CFG["thinking_enable_key"]


def ordered_candidates(candidates, prefer_deep=False):
    if prefer_deep:
        priority = _DEEP_PROFILE
    else:
        priority = _DEFAULT_PROFILE
    return sorted((item for item in candidates if model_configured(item)), key=lambda item: item.profile != priority)


def apply_chat_options(payload, candidate, output_tokens, *, repair=False, sdk=False):
    payload.pop("max_tokens", None)
    payload.pop("max_completion_tokens", None)
    payload[candidate.token_parameter] = output_tokens
    if candidate.send_temperature:
        if candidate.temperature is not None:
            payload["temperature"] = candidate.temperature
    else:
        payload.pop("temperature", None)
    if candidate.json_mode == "on" or (candidate.json_mode == "auto" and repair):
        payload["response_format"] = {"type": "json_object"}
    elif candidate.json_mode == "off":
        payload.pop("response_format", None)
    if candidate.reasoning_effort:
        payload["reasoning_effort"] = candidate.reasoning_effort
    thinking = candidate.thinking
    if thinking is None and candidate.disable_thinking:
        thinking = False
    if thinking is not None:
        setting = {_THINKING_TEMPLATE_KEY: {_THINKING_ENABLE_KEY: thinking}}
        if sdk:
            payload["extra_body"] = setting
        else:
            payload.update(setting)


def request_metadata(payload, candidate):
    if candidate.context_window:
        ctx_win = candidate.context_window
    else:
        ctx_win = None
    metadata = {
        "alias": candidate.profile,
        "model": candidate.model,
        "protocol": candidate.protocol,
        "context_window": ctx_win,
    }
    for key in _METADATA_KEYS:
        if key in payload:
            metadata[key] = payload[key]
    if "response_format" in payload:
        metadata["json_mode"] = "object"
    template = payload.get(
        _THINKING_TEMPLATE_KEY,
        payload.get("extra_body", {}).get(_THINKING_TEMPLATE_KEY, {}),
    )
    if _THINKING_ENABLE_KEY in template:
        metadata["thinking"] = template[_THINKING_ENABLE_KEY]
    if candidate.protocol == _VISION_PROTOCOL:
        metadata["image_detail"] = next(
            (
                part["image_url"].get("detail")
                for message in payload.get("messages", [])
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if part.get("type") == "image_url"
            ),
            None,
        )
    return metadata


def usage_metadata(usage):
    result = {}
    if usage is not None:
        for key in _USAGE_KEYS:
            if isinstance(usage, dict):
                value = usage.get(key)
            else:
                value = getattr(usage, key, None)
            if type(value) is int and value >= 0:
                result[key] = value
    return result
