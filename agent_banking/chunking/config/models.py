"""Configure cloud roles through environment variables and explicit request capabilities."""

import math
import os


def _endpoint(prefix, name="primary", *, legacy_model=(), legacy_url=(), legacy_key=(),
              attempts=3, timeout=240, output_tokens=6144, json_mode="auto", token_parameter="max_tokens",
              temperature=True, image_detail=""):
    return {
        "name": name,
        "prefix": prefix,
        "model_env": (prefix + "_MODEL", *legacy_model),
        "url_env": (prefix + "_BASE_URL", *legacy_url, "BASE_URL"),
        "api_key_env": (prefix + "_API_KEY", *legacy_key, "API_KEY"),
        "attempts": attempts,
        "timeout": timeout,
        "output_tokens": output_tokens,
        "json_mode": json_mode,
        "token_parameter": token_parameter,
        "temperature": temperature,
        "image_detail": image_detail,
    }


TEXT_MODELS = [_endpoint("CHUNKING_TEXT", legacy_model=("MODEL",), output_tokens=8192)]
VISION_MODELS = [_endpoint("CHUNKING_VISION", legacy_model=("VISION_MODEL",), json_mode="on")]
VISION_REVIEW_MODELS = [
    _endpoint("CHUNKING_VISION_REVIEW", legacy_model=("VISION_REVIEW_MODEL",),
              token_parameter="max_completion_tokens", json_mode="off", temperature=False, image_detail="high"),
]
DECISION_MODELS = [
    _endpoint("CHUNKING_DECISION", legacy_model=("DECISION_MODEL",), legacy_url=("DECISION_BASE_URL",),
              legacy_key=("DECISION_API_KEY",), timeout=60, output_tokens=0)
]


def endpoint_enabled(spec):
    variable = spec["prefix"] + "_ENABLED"
    value = os.getenv(variable, "true").strip().lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError(variable + " must be true or false")
    return value in ("true", "1")


def endpoint_options(spec):
    prefix = spec["prefix"]

    def option(suffix, default):
        return os.getenv(prefix + "_" + suffix, str(default)).strip()

    def boolean(suffix, default):
        value = option(suffix, str(default).lower()).lower()
        if value not in ("true", "false", "1", "0"):
            raise ValueError(prefix + "_" + suffix + " must be true or false")
        return value in ("true", "1")

    timeout = float(option("TIMEOUT", spec["timeout"]))
    attempts = int(option("ATTEMPTS", spec["attempts"]))
    legacy_cap = os.getenv("CHUNKING_LLM_MAX_TOKENS") if prefix == "CHUNKING_TEXT" else None
    cap = int(option("MAX_OUTPUT_TOKENS", legacy_cap or spec["output_tokens"]))
    if not math.isfinite(timeout) or timeout <= 0 or attempts < 1 or cap < 0:
        raise ValueError(prefix + " requires a positive timeout, positive attempts and nonnegative output limit")
    if prefix != "CHUNKING_DECISION" and not cap:
        raise ValueError(prefix + " requires a positive output limit")
    token_parameter = option("TOKEN_PARAMETER", spec["token_parameter"])
    json_mode = option("JSON_MODE", spec["json_mode"])
    detail = option("IMAGE_DETAIL", spec["image_detail"])
    if token_parameter not in ("max_tokens", "max_completion_tokens"):
        raise ValueError(prefix + "_TOKEN_PARAMETER must be max_tokens or max_completion_tokens")
    if json_mode not in ("auto", "on", "off") or detail not in ("", "auto", "low", "high"):
        raise ValueError(prefix + " has unsupported JSON_MODE or IMAGE_DETAIL")
    # The legacy text route already used the provider's opt-out on structured tasks.
    legacy_text = prefix == "CHUNKING_TEXT" and not os.getenv("CHUNKING_TEXT_MODEL") and bool(os.getenv("MODEL"))
    raw_temperature = option("TEMPERATURE", "")
    temperature = float(raw_temperature) if raw_temperature else None
    if temperature is not None and (not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError(prefix + "_TEMPERATURE must be between 0 and 2")
    return {"temperature": temperature, "timeout": timeout, "attempts": attempts, "max_output_tokens": cap,
            "token_parameter": token_parameter, "json_mode": json_mode, "image_detail": detail,
            "send_temperature": boolean("SEND_TEMPERATURE", spec["temperature"]),
            "disable_thinking": boolean("DISABLE_THINKING", legacy_text)}
