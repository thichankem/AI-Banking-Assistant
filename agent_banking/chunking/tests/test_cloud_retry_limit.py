"""Count real transport attempts across validation, repair and fallback."""

from io import BytesIO
import json

import pymupdf
import pytest

from agent_banking.chunking.run_chunking import run_auto_chunking
from agent_banking.chunking.universal.enrichment.annotations import ANNOTATION_ARRAYS
from agent_banking.chunking.universal.enrichment.structure import _review_window
from agent_banking.chunking.universal.extraction.pdf.page_review import PdfVisualReviewer
from agent_banking.chunking.universal.models.response_cache import PageVisionCache
from agent_banking.chunking.universal.models.text_client import cloud_json
from agent_banking.core.models import model_primary


def _response(value):
    body = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}
    return BytesIO(json.dumps(body).encode())


@pytest.fixture
def roles(monkeypatch):
    for role in ("VISION", "VISION_REVIEW", "TEXT"):
        prefix = "CHUNKING_" + role
        monkeypatch.setenv(prefix + "_MODEL", role.lower())
        monkeypatch.setenv(prefix + "_BASE_URL", "https://example.invalid/v1")
        monkeypatch.setenv(prefix + "_API_KEY", "fixture")
        monkeypatch.setenv(prefix + "_ENABLED", "true")
        monkeypatch.setenv(prefix + "_ATTEMPTS", "9")
    monkeypatch.setattr("agent_banking.core.fallback.time.sleep", lambda delay: None)


@pytest.fixture
def page_png():
    with pymupdf.open() as document:
        page = document.new_page(width=180, height=120)
        page.insert_text((12, 45), "Source witness")
        return page.get_pixmap().tobytes("png")


@pytest.mark.parametrize("repaired", [True, False])
def test_pdf_fills_required_fields_with_at_most_three_total_calls(roles, page_png, monkeypatch, repaired):
    calls = []
    def send(request, timeout):
        payload = json.loads(request.data)
        calls.append(payload)
        result = {"elements": [{"type": "paragraph", "text": "Source witness", "bbox": [0.1, 0.1, 0.9, 0.4]}]}
        if len(calls) == 3 and repaired:
            result.update(review={"verdict": "accepted"}, uncertainties=[])
        return _response(result)

    monkeypatch.setattr("urllib.request.urlopen", send)
    with PageVisionCache(":memory:") as cache:
        reader = PdfVisualReviewer(None, cache, None, None, True)
        layout = reader.read(page_png, [], 180, 120)

    assert [call["model"] for call in calls] == ["vision", "vision", "vision"]
    assert "previous response did not match" in calls[1]["messages"][1]["content"][0]["text"]
    assert layout["elements"][0]["text"] == "Source witness"
    assert layout["visual_review"]["cloud_attempts"] == 3
    assert layout["visual_review"]["incomplete"] is not repaired
    assert ("review_metadata_missing" in layout["visual_review"]["issues"]) is not repaired


def test_document_retry_cannot_reset_ocr_attempt_limit(roles, monkeypatch, tmp_path):
    incoming = tmp_path / "input"
    incoming.mkdir()
    with pymupdf.open() as document:
        document.new_page().insert_text((24, 80), "Source witness")
        document.save(incoming / "scan.pdf")

    calls = []
    def send(request, timeout):
        calls.append(json.loads(request.data))
        return _response({})

    monkeypatch.setattr("urllib.request.urlopen", send)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="failed for every document"):
        run_auto_chunking(incoming, output, retry_delays=(0, 0, 0), cache_db_path=":memory:",
                          use_decision=False)
    assert [call["model"] for call in calls] == ["vision", "vision", "vision_review"]
    result = json.loads((output / "chunking_last_run.json").read_text(encoding="utf-8"))
    assert result["records"][0]["attempts"] == 1


def test_transport_retry_and_structure_repair_spend_same_budget(roles, monkeypatch):
    calls = []
    def send(request, timeout):
        calls.append(json.loads(request.data))
        if len(calls) == 1:
            raise TimeoutError("Cloud timeout")
        if len(calls) == 2:
            return _response({})
        return _response(dict.fromkeys(ANNOTATION_ARRAYS, []))

    monkeypatch.setattr("urllib.request.urlopen", send)
    endpoint = model_primary("chunking.text")
    window = {"blocks": [], "editable_block_ids": [], "context_source_ids": []}
    result = _review_window(0, window, lambda _: "", endpoint, cloud_json)
    assert {key: result[key] for key in ANNOTATION_ARRAYS} == dict.fromkeys(ANNOTATION_ARRAYS, [])
    assert len(calls) == 3
    assert "return all arrays" in calls[2]["messages"][-1]["content"]
