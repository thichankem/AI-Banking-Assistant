"""Regression tests for provider response variants and scan failure locations."""

from io import BytesIO
import json
from unittest.mock import Mock

from PIL import Image, ImageDraw
import pymupdf
import pytest

from agent_banking.chunking.run_chunking import run_auto_chunking
from agent_banking.chunking.universal.extraction.pdf.page_review import CloudOCRUnavailable, PdfVisualReviewer
from agent_banking.chunking.universal.models.vision_client import DocumentVisionClient, VisionPageOverflow
from agent_banking.core.fallback import ModelFallbackError


@pytest.mark.parametrize("markup_field", ["table_html", "content"])
def test_empty_heading_summary_does_not_discard_serialized_table(markup_field):
    table_html = "<table><tr><td>500,000,000 VND</td></tr></table>"
    client = DocumentVisionClient()
    parsed = client._parse_and_validate_layout(json.dumps({
        "headings": [{"text": "", "level": 1}],
        "elements": [{"type": "heading", "content": "Daily limits", "level": 1,
                      "bbox": [0.1, 0.05, 0.9, 0.15]},
                     {"type": "table_html", markup_field: table_html,
                      "header_row_count": 0, "bbox": [0.1, 0.2, 0.9, 0.8]}],
        "uncertainties": [], "review": {"verdict": "accepted"},
    }))
    assert parsed["headings"] == [{"text": "Daily limits", "level": 1}]
    assert parsed["elements"][0]["text"] == "Daily limits"
    assert parsed["elements"][1]["type"] == "table"
    assert parsed["elements"][1]["table_html"] == table_html


@pytest.mark.parametrize("reason", ["IncompleteModelResponse", "RuntimeError HTTP 413"])
def test_fallback_quota_error_does_not_hide_earlier_page_size_limit(monkeypatch, reason):
    failure = ModelFallbackError("chunking.vision", [("candidate 1", reason), ("candidate 2", "RuntimeError HTTP 429")])
    failure.last_error = RuntimeError("quota")
    client = DocumentVisionClient(max_retries=0)
    monkeypatch.setattr(client, "_execute_request_with_retry", Mock(side_effect=failure))
    with pytest.raises(VisionPageOverflow):
        client.analyze_page_layout(b"image")


def test_failed_scan_manifest_links_to_original_page_and_render(tmp_path, monkeypatch):
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    output = tmp_path / "output"
    with Image.new("RGB", (160, 110), "white") as image:
        ImageDraw.Draw(image).text((12, 20), "Visible text", fill="black")
        stream = BytesIO()
        image.save(stream, format="PNG")
    with pymupdf.open() as pdf:
        pdf.new_page(width=160, height=110)
        second = pdf.new_page(width=160, height=110)
        second.insert_image(second.rect, stream=stream.getvalue())
        pdf.save(source_dir / "scan.pdf")
    blank = {"elements": [], "review": {"verdict": "accepted"}, "visual_review": {"passes": 2}}
    monkeypatch.setattr(PdfVisualReviewer, "read", Mock(side_effect=[blank, CloudOCRUnavailable("Unresolved raster text")]))
    with pytest.raises(RuntimeError, match="failed for every document"):
        run_auto_chunking(source_dir, output, retry_delays=(), cache_db_path=":memory:",
                          use_llm=False, use_decision=False)
    record = json.loads((output / "chunking_last_run.json").read_text(encoding="utf-8"))["records"][0]
    assert record["failed_pdf_page"] == 2
    assert (output / "assets" / (record["failed_pdf_render_sha256"] + ".png")).is_file()
