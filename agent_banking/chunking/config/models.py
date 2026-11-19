"""Load cloud model role specifications from config/settings/models.yaml."""

import math
import os

from agent_banking.chunking.config.settings import load_yaml_settings


_MODEL_CONFIG = load_yaml_settings("models.yaml")
_DEFAULTS = _MODEL_CONFIG["endpoint_defaults"]
_ENDPOINTS = _MODEL_CONFIG["endpoints"]
_VALIDATION = _MODEL_CONFIG["validation"]

TEXT_CLIENT_SETTINGS = _MODEL_CONFIG["text_client"]
VISION_CLIENT_SETTINGS = _MODEL_CONFIG["vision_client"]
PREFLIGHT_SETTINGS = _MODEL_CONFIG["preflight"]


def _endpoint(entry):
    prefix = entry["prefix"]
    name = entry.get("name", _DEFAULTS["name"])
    legacy_model = tuple(entry.get("legacy_model", _DEFAULTS["legacy_model"]))
    legacy_url = tuple(entry.get("legacy_url", _DEFAULTS["legacy_url"]))
    legacy_key = tuple(entry.get("legacy_key", _DEFAULTS["legacy_key"]))
    return {
        "name": name,
        "prefix": prefix,
        "model_env": (prefix + "_MODEL", *legacy_model),
        "url_env": (prefix + "_BASE_URL", *legacy_url, "BASE_URL"),
        "api_key_env": (prefix + "_API_KEY", *legacy_key, "API_KEY"),
        "attempts": int(entry.get("attempts", _DEFAULTS["attempts"])),
        "timeout": float(entry.get("timeout", _DEFAULTS["timeout"])),
        "output_tokens": int(entry.get("output_tokens", _DEFAULTS["output_tokens"])),
        "json_mode": entry.get("json_mode", _DEFAULTS["json_mode"]),
        "token_parameter": entry.get("token_parameter", _DEFAULTS["token_parameter"]),
        "temperature": bool(entry.get("temperature", _DEFAULTS["temperature"])),
        "image_detail": entry.get("image_detail", _DEFAULTS["image_detail"]),
    }


TEXT_MODELS = [_endpoint(entry) for entry in _ENDPOINTS["text_models"]]
VISION_MODELS = [_endpoint(entry) for entry in _ENDPOINTS["vision_models"]]
VISION_REVIEW_MODELS = [_endpoint(entry) for entry in _ENDPOINTS["vision_review_models"]]
DECISION_MODELS = [_endpoint(entry) for entry in _ENDPOINTS["decision_models"]]

_BOOL_TRUE = tuple(_VALIDATION["boolean_true"])
_BOOL_FALSE = tuple(_VALIDATION["boolean_false"])
_ALLOWED_BOOLS = _BOOL_TRUE + _BOOL_FALSE
_ALLOWED_TOKEN_PARAMS = tuple(_VALIDATION["token_parameters"])
_ALLOWED_JSON_MODES = tuple(_VALIDATION["json_modes"])
_ALLOWED_IMAGE_DETAILS = tuple(_VALIDATION["image_details"])
_MIN_TEMPERATURE = float(_VALIDATION["min_temperature"])
_MAX_TEMPERATURE = float(_VALIDATION["max_temperature"])
_ZERO_OUTPUT_ALLOWED = tuple(_VALIDATION["zero_output_allowed_prefixes"])


def endpoint_enabled(spec):
    variable = spec["prefix"] + "_ENABLED"
    value = os.getenv(variable, "true").strip().lower()
    if value not in _ALLOWED_BOOLS:
        raise ValueError(variable + " must be true or false")
    return value in _BOOL_TRUE


def endpoint_options(spec):
    prefix = spec["prefix"]

    def option(suffix, default):
        return os.getenv(prefix + "_" + suffix, str(default)).strip()

    def boolean(suffix, default):
        value = option(suffix, str(default).lower()).lower()
        if value not in _ALLOWED_BOOLS:
            raise ValueError(prefix + "_" + suffix + " must be true or false")
        return value in _BOOL_TRUE

    timeout = float(option("TIMEOUT", spec["timeout"]))
    attempts = int(option("ATTEMPTS", spec["attempts"]))
    if prefix == "CHUNKING_TEXT":
        legacy_cap = os.getenv("CHUNKING_LLM_MAX_TOKENS")
    else:
        legacy_cap = None
    cap = int(option("MAX_OUTPUT_TOKENS", legacy_cap or spec["output_tokens"]))
    if not math.isfinite(timeout) or timeout <= 0 or attempts < 1 or cap < 0:
        raise ValueError(prefix + " requires a positive timeout, positive attempts and nonnegative output limit")
    if prefix not in _ZERO_OUTPUT_ALLOWED and not cap:
        raise ValueError(prefix + " requires a positive output limit")
    token_parameter = option("TOKEN_PARAMETER", spec["token_parameter"])
    json_mode = option("JSON_MODE", spec["json_mode"])
    detail = option("IMAGE_DETAIL", spec["image_detail"])
    if token_parameter not in _ALLOWED_TOKEN_PARAMS:
        raise ValueError(prefix + "_TOKEN_PARAMETER must be max_tokens or max_completion_tokens")
    if json_mode not in _ALLOWED_JSON_MODES or detail not in _ALLOWED_IMAGE_DETAILS:
        raise ValueError(prefix + " has unsupported JSON_MODE or IMAGE_DETAIL")
    # The legacy text route already used the provider's opt-out on structured tasks.
    legacy_text = prefix == "CHUNKING_TEXT" and not os.getenv("CHUNKING_TEXT_MODEL") and bool(os.getenv("MODEL"))
    raw_temperature = option("TEMPERATURE", "")
    if raw_temperature:
        temperature = float(raw_temperature)
    else:
        temperature = None
    if temperature is not None and (not math.isfinite(temperature) or not _MIN_TEMPERATURE <= temperature <= _MAX_TEMPERATURE):
        raise ValueError(prefix + "_TEMPERATURE must be between 0 and 2")
    return {
        "temperature": temperature,
        "timeout": timeout,
        "attempts": attempts,
        "max_output_tokens": cap,
        "token_parameter": token_parameter,
        "json_mode": json_mode,
        "image_detail": detail,
        "send_temperature": boolean("SEND_TEMPERATURE", spec["temperature"]),
        "disable_thinking": boolean("DISABLE_THINKING", legacy_text),
    }
