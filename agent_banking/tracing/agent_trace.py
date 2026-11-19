"""Per-request execution traces with metadata selected at the call site."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
import json
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter, time_ns
from uuid import uuid4

from agent_banking.tracing.config.settings import TracingConfig, load_tracing_prompts


_agent_cfg = dict(TracingConfig.AGENT_TRACE_SETTINGS or {})
_prompts = load_tracing_prompts()
_phoenix_msgs = dict(getattr(_prompts, "PHOENIX_MESSAGES", {}) or {})

_CONTEXT_VAR_NAME = str(_agent_cfg.get("context_var_name") or "")
_SCHEMA_VERSION = int(_agent_cfg.get("schema_version") or 0)
_FINGERPRINT_LENGTH = int(_agent_cfg.get("fingerprint_length") or 0)
_SPAN_ID_LENGTH = int(_agent_cfg.get("span_id_length") or 0)
_DURATION_ROUND_DIGITS = int(_agent_cfg.get("duration_round_digits") or 0)
_ROOT_SPAN_PREFIX = str(_agent_cfg.get("root_span_prefix") or "")
_DEFAULT_OUTCOME = str(_agent_cfg.get("default_outcome") or "")
_SUSPENDED_STATUS = str(_agent_cfg.get("suspended_status") or "")
_OK_STATUS = str(_agent_cfg.get("ok_status") or "")
_ERROR_STATUS = str(_agent_cfg.get("error_status") or "")
_BLOCKED_AUDIT_STATUSES = tuple(_agent_cfg.get("blocked_audit_statuses") or [])
_REJECTED_HTTP_MIN = int(_agent_cfg.get("rejected_http_status_min") or 0)
_REJECTED_HTTP_MAX = int(_agent_cfg.get("rejected_http_status_max") or 0)
_REJECTED_OUTCOME = str(_agent_cfg.get("rejected_outcome") or "")
_AGENT_RUNS_FILENAME = str(TracingConfig.AGENT_RUNS_FILENAME or "")
_SAVE_WARNING_MSG = str(_phoenix_msgs.get("agent_trace_save_warning") or "")

_current_span = ContextVar(_CONTEXT_VAR_NAME, default=None)
_write_lock = Lock()
logger = logging.getLogger(__name__)


def fingerprint(value):
    if value:
        return sha256(str(value).encode("utf-8", errors="replace")).hexdigest()[:_FINGERPRINT_LENGTH]
    return None


class AgentRun:
    def __init__(self, trace_dir, entrypoint, session_id):
        self.trace_dir = trace_dir
        self.trace_id = None
        if trace_dir is not None:
            self.trace_id = uuid4().hex
        self.turn_id = self.trace_id
        self.entrypoint = entrypoint
        self.session_ref = fingerprint(session_id)
        self.spans = []
        self.outcome = _DEFAULT_OUTCOME
        self.details = {}

    def link_turn(self, state):
        if self.trace_id and state.get("agent_turn_id"):
            self.turn_id = state["agent_turn_id"]

    def result(self, response):
        if self.trace_id:
            audit = response.get("audit_status")
            if audit in _BLOCKED_AUDIT_STATUSES:
                self.outcome = audit.lower()
            else:
                self.outcome = response.get("status", _DEFAULT_OUTCOME)
            self.details.update(audit_status=audit, intent=response.get("intent"),
                                error_category=response.get("error_category"),
                                citation_count=len(response.get("citations", [])))
            response.update(trace_id=self.trace_id, turn_id=self.turn_id)
        return response

    @contextmanager
    def capture(self):
        current = _current_span.get()
        if current is not None:
            # Document ingestion inside an existing request belongs to that trace.
            self.trace_id = current[0].trace_id
            self.turn_id = current[0].turn_id
            with current[0].span(_ROOT_SPAN_PREFIX + self.entrypoint) as details:
                self.details = details
                yield self
            return
        if not self.trace_id:
            yield self
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            with self.span(_ROOT_SPAN_PREFIX + self.entrypoint) as details:
                self.details = details
                yield self
        except Exception as exc:
            self.outcome = _ERROR_STATUS
            self.details.update(error_type=type(exc).__name__)
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                self.details["http_status"] = status
                if _REJECTED_HTTP_MIN <= status < _REJECTED_HTTP_MAX:
                    self.outcome = _REJECTED_OUTCOME
            raise
        finally:
            record = {"schema_version": _SCHEMA_VERSION, "trace_id": self.trace_id, "turn_id": self.turn_id,
                      "timestamp": timestamp, "entrypoint": self.entrypoint, "session_ref": self.session_ref,
                      "outcome": self.outcome, "duration_ms": self.spans[0]["duration_ms"], "spans": self.spans}
            try:
                line = json.dumps(record, ensure_ascii=False, allow_nan=False)
                with _write_lock:
                    directory = Path(self.trace_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    with (directory / _AGENT_RUNS_FILENAME).open("a", encoding="utf-8") as stream:
                        stream.write(line + "\n")
            except (OSError, ValueError, TypeError) as exc:
                logger.warning(_SAVE_WARNING_MSG, self.trace_id, type(exc).__name__)

    @contextmanager
    def span(self, name, **attributes):
        current = _current_span.get()
        parent_id = None
        if current and current[0] is self:
            parent_id = current[1]
        start = perf_counter()
        span_id = uuid4().hex[:_SPAN_ID_LENGTH]
        record = {"span_id": span_id, "parent_span_id": parent_id, "name": name,
                  "started_ns": time_ns(), "status": _OK_STATUS, "attributes": attributes}
        self.spans.append(record)
        token = _current_span.set((self, span_id, attributes))
        try:
            yield attributes
        except BaseException as exc:
            from langgraph.errors import GraphInterrupt

            if isinstance(exc, GraphInterrupt):
                record["status"] = _SUSPENDED_STATUS
            else:
                record["status"] = _ERROR_STATUS
                attributes["error_type"] = type(exc).__name__
            raise
        finally:
            record["duration_ms"] = round((perf_counter() - start) * 1000, _DURATION_ROUND_DIGITS)
            _current_span.reset(token)


@contextmanager
def agent_span(name, **attributes):
    current = _current_span.get()
    if current is None:
        yield attributes
    else:
        with current[0].span(name, **attributes) as details:
            yield details


def annotate_span(**attributes):
    current = _current_span.get()
    if current is not None:
        current[2].update(attributes)


def trace_stage(name):
    def decorate(operation):
        @wraps(operation)
        def traced(*args, **kwargs):
            if _current_span.get() is None:
                return operation(*args, **kwargs)
            with agent_span(name):
                return operation(*args, **kwargs)
        return traced
    return decorate
