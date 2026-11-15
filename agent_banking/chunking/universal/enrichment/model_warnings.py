"""Optional cloud failures are diagnostic events, never source edits."""
import logging

from agent_banking.core.fallback import ModelFallbackError


logger = logging.getLogger("chunking.models")


def record_model_warning(ir, stage, error_type):
    logger.warning("Optional chunking stage %s kept source defaults (%s)", stage, error_type)
    if ir is not None:
        ir.setdefault("model_warnings", []).append({"stage": stage, "error_type": error_type})


def optional_model_call(ir, stage, operation):
    try:
        return operation()
    except (ModelFallbackError, TimeoutError, ConnectionError, ValueError, TypeError, KeyError) as error:
        record_model_warning(ir, stage, type(error).__name__)
        return None
