"""Chat adapter with ordered fallback and bounded retries."""
from agent_banking.config.fallback import run_fallback
from agent_banking.config.models import get_model_chain, chain_fingerprint


class FallbackChatClient:
    def __init__(self, module, primary=None, retry_delays=(1.0, 2.0, 4.0)):
        self.module = module
        self.candidates = get_model_chain(module, primary)
        self.fingerprint = chain_fingerprint(module, primary)
        self.retry_delays = retry_delays
        self.chat = self
        self.completions = self

    def create(self, validate=None, **kwargs):
        from openai import OpenAI

        def call(candidate):
            payload = dict(kwargs)
            payload["model"] = candidate.model
            # SDK retries are disabled: the chain owns the complete retry budget.
            with OpenAI(api_key=candidate.api_key, base_url=candidate.url,
                        timeout=candidate.timeout, max_retries=0) as client:
                response = client.chat.completions.create(**payload)
            if not response.choices:
                raise ValueError("Empty chat choices")
            message = response.choices[0].message
            if not isinstance(message.content, str) or not message.content.strip():
                raise ValueError("Empty chat answer")
            if response.choices[0].finish_reason not in (None, "stop"):
                raise ValueError("Incomplete chat answer")
            if validate is not None:
                validate(response)
            return response

        return run_fallback(self.module, self.candidates, call, self.retry_delays)
