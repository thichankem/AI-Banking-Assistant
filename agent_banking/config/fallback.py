"""Ordered, bounded fallback shared by every model consumer."""
from dataclasses import dataclass, field
import logging
import math
import time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEndpoint:
    name: str
    model: str
    url: str = field(repr=False)
    api_key: str = field(repr=False)
    timeout: float = 60.0
    attempts: int = 1
    dimension: int = 0


class ModelFallbackError(RuntimeError):
    """Contains safe diagnostics only, never provider bodies or credentials."""

    def __init__(self, module, failures):
        self.module = module
        self.failures = tuple(failures)
        details = "; ".join(name + ": " + reason for name, reason in failures)
        super().__init__(module + ": all model candidates failed (" + details + ")")


def run_fallback(module, candidates, operation, retry_delays=(1.0, 2.0, 4.0)):
    """Restart from highest priority for each request; validate inside operation."""
    failures = []
    last_error = None
    for index, candidate in enumerate(candidates, start=1):
        # Use positional labels so a misconfigured name cannot leak a secret.
        label = "candidate " + str(index)
        if not candidate.model or not candidate.url or not candidate.api_key:
            failures.append((label, "missing model, endpoint or credential"))
            continue
        if candidate.attempts < 1 or not math.isfinite(candidate.timeout) or candidate.timeout <= 0:
            failures.append((label, "invalid attempts or timeout"))
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
        failures.append(("configuration", "empty model chain"))
    error = ModelFallbackError(module, failures)
    error.last_error = last_error
    raise error from None
