"""Check cloud capabilities using synthetic, non-document inputs."""

from io import BytesIO

from PIL import Image, ImageDraw

from agent_banking.core.models import get_model_chain
from .text_client import _cloud_json, apply_chat_profile, validate_decisions
from .vision_client import DocumentVisionClient


def _vision_probe():
    stream = BytesIO()
    with Image.new("RGB", (256, 96), "white") as image:
        ImageDraw.Draw(image).text((32, 12), "42", fill="black", font_size=42)
        image.save(stream, format="PNG")
    return stream.getvalue()


def _probe(role, endpoint):
    url = endpoint.url.rstrip("/")
    timeout = min(endpoint.timeout, 30)
    if role == "decision":
        questions = {"number": {"type": "choice", "instructions": "Select the integer supplied in state.",
                                "criteria": {"42": "Forty-two", "24": "Twenty-four"}}}
        path = "/systemone"
        state = {"number": 42}
        answer = _cloud_json(url.removesuffix(path) + path, endpoint.api_key,
                             {"model": endpoint.model, "state": state, "questions": questions}, timeout, 0)
        validate_decisions(answer, questions)
        if answer["answers"]["number"]["choice"] != "42":
            raise ValueError("Decision canary mismatch")
        return

    path = "/chat/completions"
    cap = min(endpoint.max_output_tokens, 4096)
    payload = {"model": endpoint.model, "temperature": 0, "max_tokens": cap}
    if role == "text":
        payload["messages"] = [{"role": "user", "content": 'Return this exact JSON object: {"number":42}'}]
    else:
        client = DocumentVisionClient(model_module="chunking." + role)
        payload["messages"] = [{"role": "user", "content": [
            {"type": "text", "text": 'Read the image. Return JSON like {"number":"visible digits"}.'},
            client._image_part(_vision_probe()),
        ]}]
    payload["messages"].insert(0, {"role": "system", "content": "Return a single JSON object. Read any supplied image exactly."})
    apply_chat_profile(payload, endpoint, cap)
    answer = _cloud_json(url.removesuffix(path) + path, endpoint.api_key, payload, timeout, 0, max_output_tokens=cap)
    if str(answer.get("number")) != "42":
        raise ValueError("Cloud canary mismatch")


def check_models():
    checks = []
    for role in ("text", "vision", "vision_review", "decision"):
        try:
            configured = [endpoint for endpoint in get_model_chain("chunking." + role) if endpoint.model]
        except (TypeError, ValueError):
            checks.append({"role": role, "status": "failed", "reason": "invalid_configuration"})
            continue
        if not configured:
            required = role in ("text", "vision")
            checks.append({"role": role, "status": "failed" if required else "disabled", "reason": "model_missing"})
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
