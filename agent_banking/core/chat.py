# -*- coding: utf-8 -*-
"""
Chat adapter with ordered fallback and bounded retries.
Complies with 0% Regex Law, Zero Ternary Law, and No Typing Library.
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

        def legacy_call(candidate):
            nonlocal _usage
            with trace_span(
                "model.attempt",
                run_type="llm",
                role=self.module,
                reason="legacy_order",
                **request_metadata(kwargs, candidate),
            ) as details:
                _usage = lambda usage: details.update(usage=usage_metadata(usage))
                response = call(candidate)
                details["result"] = "success"
                return response

        return run_fallback(self.module, self.candidates, legacy_call, self.retry_delays)
