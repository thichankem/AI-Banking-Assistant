"""Offline regressions for extraction, review and cache boundary cases."""
from copy import deepcopy
from io import BytesIO
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pymupdf
import pytest

from agent_banking.chunking.universal.extraction.pdf.extractor import PdfExtractor
from agent_banking.chunking.universal.extraction.pdf.page_review import PdfVisualReviewer
from agent_banking.chunking.universal.extraction.pdf.tables import table_from_html, validate_table
from agent_banking.chunking.universal.enrichment.image_context import surrounding_sources, validate_context_selection
from agent_banking.chunking.universal.models.response_cache import PageVisionCache
from agent_banking.chunking.universal.models.vision_client import QwenVisionClient, VisionPageOverflow


def test_malformed_model_table_cannot_allocate_unbounded_header_grid():
    table = {"headers": [""], "rows": [], "header_row_count": 2**63}
    with pytest.raises(ValueError, match="requires? header_grid"):
        validate_table(table)


@pytest.mark.parametrize("html", [
    "<table><tr><td>amount</th></tr></table>",
    "<table><thead><tr><th>amount</th></tr></table>",
    "<table><thead><tr><th>amount</th></tr></tbody></table>",
    "<table><tr><td rowspan='2'>amount</td></tr></table>",
    "<table><tr><td>amount</td><td>rate</td></tr><tr><td>10</td></tr></table>",
])
def test_invalid_serialized_table_is_rejected(html):
    with pytest.raises(ValueError):
        table_from_html(html, 0)


def test_valid_serialized_table_retains_formula_spans_and_empty_cells():
    table = table_from_html(
        "<table><thead><tr><th colspan='2'>Costs</th></tr></thead>"
        "<tbody><tr><td rowspan='2'>A<sup>2</sup></td><td></td></tr>"
        "<tr><td>x<sub>1</sub></td></tr></tbody></table>", 1,
    )
    validate_table(table)
    assert table["headers"] == ["Costs", "Costs"]
    assert table["header_grid"] == [["Costs", ""]]
    assert table["rows"] == [["A<sup>2</sup>", ""], ["A<sup>2</sup>", "x<sub>1</sub>"]]


def test_cloud_pages_preserve_rotation_without_local_text_extraction(tmp_path):
    cache = Mock()
    cache.get.return_value = None
    boxes = [{"type": "paragraph", "text": "Cloud text", "bbox": [0.1, 0.2, 0.3, 0.4]}]
    client = layout_client([reviewed(boxes), reviewed(boxes)])
    adapter = PdfExtractor(tmp_path, cache, vision_client=client, use_decision=False, dpi=72)
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text((40, 45), "Native page")
        rotated = document.new_page(width=600, height=800)
        rotated.insert_text((80, 150), "Rotated native page")
        rotated.set_rotation(90)
        result = adapter.extract(document.tobytes(), "native.pdf")
    assert [item["text"] for item in result["blocks"]] == ["Cloud text", "Cloud text"]
    assert result["blocks"][-1]["provenance"]["rotation"] == 90
    assert client.analyze_page_layout.call_count == 2


def page_png():
    with Image.new("RGB", (64, 64), "white") as image:
        image.putpixel((10, 10), (0, 0, 0))
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


def reviewed(elements=(), **review):
    return {"elements": list(elements), "uncertainties": [], "review": {"verdict": "accepted", **review}}


def layout_client(response):
    return SimpleNamespace(model="test", analyze_page_layout=Mock(side_effect=response))


def test_invalid_review_shape_is_retried_after_corrupt_cached_metadata(tmp_path):
    cache = Mock()
    cache.get.return_value = {**reviewed(), "visual_review": None}
    accepted = reviewed([{"type": "paragraph", "text": "Amount", "bbox": [0.1, 0.1, 0.3, 0.2]}])
    reader = PdfVisualReviewer(tmp_path, cache, layout_client([accepted]), layout_client([None, deepcopy(accepted)]), False)
    result = reader.read(page_png(), [], 64, 64)
    assert result["elements"] == accepted["elements"]
    cache.put.assert_called_once()


def test_unavailable_primary_can_use_backup_cloud(tmp_path):
    cache = Mock()
    cache.get.return_value = None
    backup = layout_client([reviewed([{"type": "paragraph", "text": "Backup text"}], verdict="unresolved")])
    reader = PdfVisualReviewer(tmp_path, cache, layout_client([RuntimeError("unavailable")]), backup, False)
    result = reader.read(page_png(), [], 64, 64)
    assert result["elements"][0]["text"] == "Backup text"
    assert result["elements"][0]["bbox_granularity"] == "page"
    assert result["visual_review"]["incomplete"] is True
    assert result["visual_review"]["failed_candidates"] == ["RuntimeError"]


def test_full_page_overflow_recovers_a_partial_cloud_crop(tmp_path):
    cache = Mock()
    cache.get.return_value = None
    first = layout_client([VisionPageOverflow("full page"), reviewed([{
        "type": "paragraph", "text": "Visible crop", "bbox": [0.1, 0.1, 0.5, 0.5]
    }]), RuntimeError("unavailable"), RuntimeError("unavailable"), RuntimeError("unavailable")])
    reader = PdfVisualReviewer(tmp_path, cache, first, None, False)
    result = reader.read(page_png(), [], 64, 64)
    assert result["elements"][0]["text"] == "Visible crop"
    assert result["visual_review"]["incomplete"] is True
    assert len(result["region_audit"]) == 4
    assert sum("error" in item for item in result["region_audit"]) == 3


def test_batch_cache_copy_and_cross_connection_coherence(tmp_path):
    path = tmp_path / "cache.db"
    first = PageVisionCache(path)
    second = PageVisionCache(path)
    try:
        record = {"result": [1]}
        assert first.put_batch([("key", record), ("invalid", {} , "oops"), ("bad",), ("nan", float("nan"))]) == 1
        record["result"].append(2)
        assert first.get("key") == second.get("key") == {"result": [1]}
        assert second.put("key", {"result": [3]})
        assert first.get("key") == {"result": [3]}
        second.delete("key")
        assert not first.contains("key")
        assert first.get("key") is None
    finally:
        first.close()
        second.close()


def test_cache_transaction_failure_is_not_visible_in_memory():
    with PageVisionCache(":memory:") as cache:
        assert cache.put("example", {"answer": 10})
        cache._conn.execute("PRAGMA query_only=1")
        with pytest.raises(sqlite3.OperationalError):
            cache.put("example", {"answer": 20})
        assert cache.get("example") == {"answer": 10}
        assert cache.get("\ud800") is None
    with cache:
        assert cache.get("example") is None
        assert cache.put("example", {"answer": 20})


def test_explicit_caption_outside_radius_and_derived_title():
    figure = {"type": "image", "id": "image", "provenance": {"region": "body"},
              "image": {"title": "Generated title", "title_source": "decision_model", "source_title": "Source title"}}
    caption = {"type": "caption", "id": "caption", "text": "Authorized limit", "provenance": {"region": "body"}}
    ir = {"blocks": [figure, caption], "relations": [{"type": "caption_of", "from": "caption", "to": "image"}]}
    candidates = surrounding_sources(ir, 0, radius=0)
    assert {item["text"] for item in candidates} == {"Authorized limit", "Source title"}
    assert candidates[0]["explicit_caption_for_image"]
    with pytest.raises(ValueError, match="supplied source"):
        validate_context_selection({"include": True, "reason": "caption", "evidence": [
            {"source_id": [], "quote": "Authorized limit"},
        ]}, candidates)


def test_vision_fallback_rebuilds_provider_fields_for_each_image(monkeypatch):
    from agent_banking.chunking.universal.models import vision_client

    endpoint = SimpleNamespace(url="https://invalid.example", api_key="fixture", model="Azure:primary", timeout=1, attempts=1)
    monkeypatch.setattr(vision_client, "model_primary", lambda module: endpoint)
    client = QwenVisionClient()
    payload = client._build_payload(page_png())
    payload["messages"][1]["content"].append(client._image_part(page_png()))
    fallback = dict(payload)
    client._configure_payload(fallback, SimpleNamespace(model="qwen-fallback"))
    assert "max_completion_tokens" not in fallback and fallback["max_tokens"] == client.max_tokens
    assert "response_format" in fallback and "response_format" not in payload
    assert all("detail" not in part["image_url"] for part in fallback["messages"][1]["content"][1:])
    assert all(part["image_url"]["detail"] == "high" for part in payload["messages"][1]["content"][1:])
    with pytest.raises(ValueError, match="Image is empty"):
        client.analyze_page_layout(page_png(), extra_images=[("empty", b"")])
    with pytest.raises(ValueError, match="integer"):
        client._parse_and_validate_layout('{"elements":[{"type":"heading","text":"Example","level":1.5}]}')


def test_vision_request_uses_fallback_candidate_payload(monkeypatch):
    from agent_banking.chunking.universal.models import text_client as model_client, vision_client
    from agent_banking.core import models
    from agent_banking.core.fallback import ModelEndpoint

    first = ModelEndpoint("first", "Azure:example", "https://first.invalid", "fixture")
    second = ModelEndpoint("second", "Qwen-example", "https://second.invalid", "fixture")
    monkeypatch.setattr(vision_client, "model_primary", lambda module: first)
    monkeypatch.setattr(models, "get_model_chain", lambda *args: [first, second])
    sent = []

    def fake_cloud(url, key, payload, *args):
        sent.append(deepcopy(payload))
        if payload["model"] == first.model:
            return {"elements": [{"type": "heading", "text": "Example", "level": None}]}
        return reviewed([{"type": "heading", "text": "Example", "level": 1}])

    monkeypatch.setattr(model_client, "_cloud_json", fake_cloud)
    parsed = QwenVisionClient().analyze_page_layout(page_png())
    assert parsed["headings"] == [{"level": 1, "text": "Example"}]
    assert [payload["model"] for payload in sent] == [first.model, second.model]
    assert "max_completion_tokens" in sent[0] and "max_tokens" in sent[1]
