# -*- coding: utf-8 -*-
"""
Chat adapter with ordered fallback and bounded retries.
Complies with 0% Regex Law, Zero Ternary Law.
"""

from agent_banking.core.cloud_rate_limit import acquire_cloud_slot
from agent_banking.core.config.settings import CoreConfig, get_core_messages
from agent_banking.core.fallback import run_fallback
from agent_banking.core.model_routing import request_metadata, usage_metadata
from agent_banking.core.models import chain_fingerprint, get_model_chain
from agent_banking.tracing.tracer import trace_span

_CHAT_CFG = CoreConfig.CORE_CHAT
_DEFAULT_RETRY_DELAYS = tuple(_CHAT_CFG["retry_delays"])
_SDK_MAX_RETRIES = int(_CHAT_CFG["openai_sdk_max_retries"])
_VALID_FINISH_REASONS = tuple([None] + list(_CHAT_CFG["valid_finish_reasons"]))
_MESSAGES = get_core_messages("chat", "core_messages.yaml", "CORE_CHAT_MESSAGES")


class FallbackChatClient:
    def __init__(self, module, primary=None, retry_delays=None, *, routed=False):
        self.module = module
        self.routed = routed
        self.candidates = get_model_chain(module, primary, routed=routed)
        self.fingerprint = chain_fingerprint(module, primary, routed=routed)
        if retry_delays is None:
            self.retry_delays = _DEFAULT_RETRY_DELAYS
        else:
            self.retry_delays = retry_delays
        self.chat = self
        self.completions = self

    def create(self, validate=None, *, _candidate=None, _usage=None, **kwargs):
        from openai import OpenAI

        def call(candidate):
            payload = dict(kwargs)
            payload["model"] = candidate.model
            acquire_cloud_slot(candidate.model, candidate.url)
            # SDK retries are disabled: the chain owns the complete retry budget.
            with OpenAI(
                api_key=candidate.api_key,
                base_url=candidate.url,
                timeout=candidate.timeout,
                max_retries=_SDK_MAX_RETRIES,
            ) as client:
                response = client.chat.completions.create(**payload)
            if _usage is not None:
                _usage(getattr(response, "usage", None))
            if not response.choices:
                raise ValueError(_MESSAGES["empty_choices"])
            message = response.choices[0].message
            content_str = message.content
            if not isinstance(content_str, str) or not content_str.strip():
                has_reasoning = False
                if hasattr(message, "reasoning_content"):
                    if isinstance(message.reasoning_content, str):
                        if message.reasoning_content.strip():
                            has_reasoning = True
                if has_reasoning:
                    message.content = message.reasoning_content
                else:
                    raise ValueError(_MESSAGES["empty_answer"])
            if response.choices[0].finish_reason not in _VALID_FINISH_REASONS:
                raise ValueError(_MESSAGES["incomplete_answer"])
            if validate is not None:
                validate(response)
            return response

        if _candidate is not None:
            return call(_candidate)

        trace_meta = getattr(self, "_pending_trace_meta", None)
        if hasattr(self, "_pending_trace_meta"):
            self._pending_trace_meta = None
        if not isinstance(trace_meta, dict):
            trace_meta = {}

        span_role = str(trace_meta.get("role") or self.module)
        configured_role = str(trace_meta.get("configured_role") or self.module)
        base_reason = str(trace_meta.get("reason") or "legacy_order")
        base_attempt = int(trace_meta.get("attempt") or 1)
        input_bytes = trace_meta.get("input_bytes")
        if input_bytes is None and isinstance(kwargs.get("messages"), list):
            input_bytes = sum(len(str(msg.get("content") or "").encode("utf-8")) for msg in kwargs["messages"])
        input_budget_bytes = trace_meta.get("input_budget_bytes")
        output_tokens_limit = trace_meta.get("output_tokens_limit")
        if output_tokens_limit is None:
            output_tokens_limit = kwargs.get("max_tokens")
            if output_tokens_limit is None:
                output_tokens_limit = kwargs.get("max_completion_tokens")
        is_repair = bool(trace_meta.get("repair", False))
        fallback_attempt_idx = 0

        def legacy_call(candidate):
            nonlocal _usage, fallback_attempt_idx
            attempt_num = base_attempt + fallback_attempt_idx
            if fallback_attempt_idx > 0 and base_reason == "standard":
                attempt_reason = "provider_fallback"
            else:
                attempt_reason = base_reason
            fallback_attempt_idx += 1
            span_attrs = {
                "role": span_role,
                "configured_role": configured_role,
                "reason": attempt_reason,
                "attempt": attempt_num,
                "input_bytes": input_bytes,
                "input_budget_bytes": input_budget_bytes,
                "output_tokens_limit": output_tokens_limit,
                "repair": is_repair,
            }
            span_attrs.update(request_metadata(kwargs, candidate))
            with trace_span(
                "model.attempt",
                run_type="llm",
                **span_attrs,
            ) as details:
                _usage = lambda usage: details.update(usage=usage_metadata(usage))
                try:
                    response = call(candidate)
                    details["result"] = "success"
                    return response
                except Exception as error:
                    status = getattr(error, "status_code", None)
                    details.update(
                        result="failed",
                        error_type=type(error).__name__,
                        error_message=str(error)[:240],
                    )
                    if type(status) is int:
                        details["http_status"] = status
                    raise

        return run_fallback(self.module, self.candidates, legacy_call, self.retry_delays)
