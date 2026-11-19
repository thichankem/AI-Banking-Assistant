"""Check cloud capabilities using synthetic, non-document inputs."""

from io import BytesIO

from PIL import Image, ImageDraw

from agent_banking.chunking.config.models import PREFLIGHT_SETTINGS
from agent_banking.chunking.config.prompt_config import PREFLIGHT_CANARIES
from agent_banking.core.models import auto_routing_enabled, get_model_chain
from .text_client import _cloud_json, _routed_chat_profile, apply_chat_profile, validate_decisions
from .vision_client import DocumentVisionClient

_PREFLIGHT = PREFLIGHT_SETTINGS
_CANARIES = PREFLIGHT_CANARIES
_CANARY_VALUE = _PREFLIGHT["canary_value"]


def _vision_probe():
    stream = BytesIO()
    width = _PREFLIGHT["vision_probe_width"]
    height = _PREFLIGHT["vision_probe_height"]
    pos = tuple(_PREFLIGHT["vision_probe_pos"])
    with Image.new("RGB", (width, height), "white") as image:
        ImageDraw.Draw(image).text(pos, str(_CANARY_VALUE), fill="black", font_size=_PREFLIGHT["vision_probe_font_size"])
        image.save(stream, format="PNG")
    return stream.getvalue()


def _probe(role, endpoint):
    url = endpoint.url.rstrip("/")
    timeout = min(endpoint.timeout, _PREFLIGHT["probe_timeout_cap"])
    if role == "decision":
        questions = _CANARIES["decision_questions"]
        path = "/systemone"
        state = {"number": _CANARY_VALUE}
        answer = _cloud_json(url.removesuffix(path) + path, endpoint.api_key,
                             {"model": endpoint.model, "state": state, "questions": questions}, timeout, 0)
        validate_decisions(answer, questions)
        if answer["answers"]["number"]["choice"] != str(_CANARY_VALUE):
            raise ValueError("Decision canary mismatch")
        return

    path = "/chat/completions"
    cap = min(endpoint.max_output_tokens or _PREFLIGHT["probe_max_tokens"], _PREFLIGHT["probe_max_tokens"])
    payload = {"model": endpoint.model, "temperature": 0, "max_tokens": cap}
    if role == "text":
        payload["messages"] = [{"role": "user", "content": _CANARIES["text_user_prompt"]}]
    else:
        client = DocumentVisionClient(model_module="chunking." + role)
        payload["messages"] = [{"role": "user", "content": [
            {"type": "text", "text": _CANARIES["vision_user_prompt"]},
            client._image_part(_vision_probe()),
        ]}]
    payload["messages"].insert(0, {"role": "system", "content": _CANARIES["system_prompt"]})
    if auto_routing_enabled():
        _routed_chat_profile(payload, endpoint, cap)
    else:
        apply_chat_profile(payload, endpoint, cap)
    answer = _cloud_json(url.removesuffix(path) + path, endpoint.api_key, payload, timeout, 0, max_output_tokens=cap)
    if str(answer.get("number")) != str(_CANARY_VALUE):
        raise ValueError("Cloud canary mismatch")


def check_models():
    checks = []
    required_roles = set(_PREFLIGHT["required_roles"])
    for role in _PREFLIGHT["roles"]:
        try:
            configured = [endpoint for endpoint in get_model_chain("chunking." + role) if endpoint.model]
        except (TypeError, ValueError):
            checks.append({"role": role, "status": "failed", "reason": "invalid_configuration"})
            continue
        if not configured:
            if role in required_roles:
                status = "failed"
            else:
                status = "disabled"
            checks.append({"role": role, "status": status, "reason": "model_missing"})
            continue
        for index, endpoint in enumerate(configured, 1):
            result = {"role": role, "candidate": index, "model": endpoint.model, "status": "ok"}
            try:
                if not endpoint.url or not endpoint.api_key:
                    raise ValueError("Endpoint URL or credential missing")
                _probe(role, endpoint)
            except (RuntimeError, OSError, ValueError, TypeError, KeyError) as error:
                result.update(status="failed", reason=type(error).__name__)
                if hasattr(error, "status_code"):
                    result["http_status"] = error.status_code
            checks.append(result)
    return checks
