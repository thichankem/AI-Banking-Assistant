"""Ordered, bounded fallback shared by every model consumer."""
from dataclasses import dataclass, field
import logging
import math
import time

from httpx import NetworkError, TimeoutException
from openai import APIConnectionError

from agent_banking.core.config.settings import CoreConfig, get_core_messages

logger = logging.getLogger(__name__)

_EP_DEFAULTS = CoreConfig.CORE_MODELS["endpoint_defaults"]
_FALLBACK_CFG = CoreConfig.CORE_FALLBACK
_DEFAULT_RETRY_DELAYS = tuple(_FALLBACK_CFG["retry_delays"])
_TEMP_STATUS_CODES = tuple(_FALLBACK_CFG["temporary_status_codes"])
_MESSAGES = get_core_messages("fallback", "core_messages.yaml", "CORE_FALLBACK_MESSAGES")


@dataclass(frozen=True)
class ModelEndpoint:
    name: str
    model: str
    url: str = field(repr=False)
    api_key: str = field(repr=False)
    timeout: float = _EP_DEFAULTS["timeout"]
    attempts: int = _EP_DEFAULTS["attempts"]
    dimension: int = _EP_DEFAULTS["dimension"]
    token_parameter: str = _EP_DEFAULTS["token_parameter"]
    max_output_tokens: int = _EP_DEFAULTS["max_output_tokens"]
    json_mode: str = _EP_DEFAULTS["json_mode"]
    send_temperature: bool = _EP_DEFAULTS["send_temperature"]
    temperature: float | None = _EP_DEFAULTS["temperature"]
    disable_thinking: bool = _EP_DEFAULTS["disable_thinking"]
    image_detail: str = _EP_DEFAULTS["image_detail"]
    context_window: int = _EP_DEFAULTS["context_window"]
    reasoning_effort: str = _EP_DEFAULTS["reasoning_effort"]
    thinking: bool | None = _EP_DEFAULTS["thinking"]
    profile: str = _EP_DEFAULTS["profile"]
    protocol: str = _EP_DEFAULTS["protocol"]
    image_token_reserve: int = _EP_DEFAULTS["image_token_reserve"]


class ModelFallbackError(RuntimeError):
    """Contains safe diagnostics only, never provider bodies or credentials."""

    def __init__(self, module, failures):
        self.module = module
        self.failures = tuple(failures)
        details = "; ".join(name + ": " + reason for name, reason in failures)
        super().__init__(module + ": all model candidates failed (" + details + ")")


def is_temporary_error(error):
    if isinstance(error, ModelFallbackError):
        error = getattr(error, "last_error", None)
    if isinstance(error, (TimeoutError, ConnectionError, APIConnectionError, NetworkError, TimeoutException)):
        return True
    return getattr(error, "status_code", None) in _TEMP_STATUS_CODES


def run_fallback(module, candidates, operation, retry_delays=None):
    """Restart from highest priority for each request; validate inside operation."""
    if retry_delays is None:
        retry_delays = _DEFAULT_RETRY_DELAYS
    failures = []
    last_error = None
    for index, candidate in enumerate(candidates, start=1):
        # Use positional labels so a misconfigured name cannot leak a secret.
        label = "candidate " + str(index)
        if not candidate.model or not candidate.url or not candidate.api_key:
            failures.append((label, _MESSAGES["missing_credential"]))
            continue
        if candidate.attempts < 1 or not math.isfinite(candidate.timeout) or candidate.timeout <= 0:
            failures.append((label, _MESSAGES["invalid_limits"]))
            continue
        for attempt in range(candidate.attempts):
            try:
                return operation(candidate)
            except Exception as error:
                last_error = error
                reason = type(error).__name__
                status = getattr(error, "status_code", getattr(error, "code", None))
                if isinstance(status, int):
                    reason += " HTTP " + str(status)
                if attempt + 1 == candidate.attempts or getattr(error, "retryable", True) is False:
                    failures.append((label, reason))
                    logger.warning("%s %s failed (%s); trying next priority", module, label, reason)
                    break
                elif retry_delays:
                    time.sleep(retry_delays[min(attempt, len(retry_delays) - 1)])
    if not failures:
        failures.append((_MESSAGES["empty_chain_source"], _MESSAGES["empty_chain"]))
    error = ModelFallbackError(module, failures)
    error.last_error = last_error
    raise error from None
