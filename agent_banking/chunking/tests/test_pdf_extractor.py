"""Regression cases generated locally through the PDF adapter."""

from io import BytesIO
from copy import deepcopy
import json
from unittest.mock import Mock

import pymupdf
import pytest
from PIL import Image, ImageDraw

from agent_banking.chunking.run_chunking import load_published_chunks, run_auto_chunking
from agent_banking.chunking.universal.extraction.pdf.extractor import PdfExtractor
from agent_banking.chunking.universal.enrichment.enricher import ModelEnricher
from agent_banking.chunking.universal.planning.semantic_boundaries import SemanticBoundaryScorer


def small_scan(document, page_number=False):
    page = document.new_page(width=600, height=800)
    with Image.new("RGB", (500, 300), "white") as image:
        ImageDraw.Draw(image).text((30, 30), "Daily transfer limit: 5000", fill="black")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    page.insert_image(pymupdf.Rect(60, 200, 540, 488), stream=buffer.getvalue())
    if page_number:
        page.insert_text((275, 778), "Page 1")


def visual_client(elements=None, *, non_text_only=False):
    review = {"verdict": "accepted"}
    if non_text_only:
        review["non_text_only"] = True
    response = {"elements": elements or [], "review": review, "uncertainties": []}
    return Mock(model="fixture-vision", identity=Mock(return_value="fixture-vision"),
                analyze_page_layout=Mock(side_effect=lambda *args: deepcopy(response)))


def extractor(path, vision=None):
    cache = Mock()
    cache.get.return_value = None
    return PdfExtractor(path, cache, vision_client=vision, use_decision=False, dpi=72)


def test_all_pdf_pages_use_cloud_even_with_embedded_text(tmp_path):
    expected = "Cloud transcription"
    client = visual_client([{"type": "paragraph", "text": expected, "bbox": [0.1, 0.2, 0.6, 0.4]}])
    with pymupdf.open() as document:
        document.new_page().insert_text((40, 70), "Embedded text must not be used")
        small_scan(document, page_number=True)
        ir = extractor(tmp_path, client).extract(document.tobytes(), "mixed.pdf")
    assert [block["text"] for block in ir["blocks"]] == [expected, expected]
    assert ir["statistics"]["ocr_pages_processed"] == 2
    assert client.analyze_page_layout.call_count == 2


def test_partial_cloud_failure_can_publish_remaining_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelEnricher, "enrich", lambda self, ir: ir)
    monkeypatch.setattr(SemanticBoundaryScorer, "annotate", lambda self, ir: ir)
    incoming = tmp_path / "input"
    output = tmp_path / "output"
    incoming.mkdir()
    with pymupdf.open() as document:
        small_scan(document, page_number=True)
        small_scan(document, page_number=False)
        document.save(incoming / "scan.pdf")
    available = {"elements": [{"type": "paragraph", "text": "Hạn mức mỗi ngày là 5.000 đồng.",
                               "bbox": [0.1, 0.2, 0.6, 0.4]}],
                 "review": {"verdict": "unresolved"}, "uncertainties": [{"reason": "One word is unclear"}]}
    client = visual_client()
    client.analyze_page_layout.side_effect = [RuntimeError("Cloud unavailable"), available]
    chunks = run_auto_chunking(incoming, output, vision_client=client, use_decision=False, retry_delays=(),
                               cache_db_path=tmp_path / "models.db", dpi=72)
    published, manifest = load_published_chunks(output, return_manifest=True)
    quality = manifest["records"][0]["quality"]
    assert chunks == published
    assert quality["ocr_failed_pages"] == 1 and quality["ocr_failed_page_numbers"] == [1]
    assert quality["ocr_partial_pages"] == 1 and quality["index_ready"] is True


def test_total_cloud_outage_cannot_publish_empty_document(tmp_path):
    incoming = tmp_path / "input"
    output = tmp_path / "output"
    incoming.mkdir()
    with pymupdf.open() as document:
        document.new_page().insert_text((40, 70), "Do not fall back to embedded text")
        document.save(incoming / "scan.pdf")
    client = visual_client()
    client.analyze_page_layout.side_effect = RuntimeError("Cloud unavailable")
    with pytest.raises(RuntimeError, match="failed for every document"):
        run_auto_chunking(incoming, output, vision_client=client, use_decision=False, retry_delays=(),
                          cache_db_path=tmp_path / "models.db", dpi=72)
    assert not (output / "chunking_manifest.json").exists()


def test_blank_and_non_text_pages_use_cloud(tmp_path):
    client = visual_client(non_text_only=True)
    adapter = extractor(tmp_path, client)
    with pymupdf.open() as document:
        document.new_page()
        result = adapter.extract(document.tobytes(), "blank.pdf")
        assert result["blocks"] == []
        assert result["statistics"]["ocr_pages_processed"] == 1
        page = document[0]
        page.draw_rect(pymupdf.Rect(100, 100, 120, 120), color=None, fill=(0.1, 0.2, 0.3))
        result = adapter.extract(document.tobytes(), "illustration.pdf")
    assert result["blocks"] == []
    assert result["page_reviews"][0]["content"] == "non_text_only"
    assert client.analyze_page_layout.call_count == 2


def test_pdf_without_visual_coordinates_publishes_file_and_all_source_pages(tmp_path, monkeypatch):
    incoming = tmp_path / "input"
    category = incoming / "manuals"
    category.mkdir(parents=True)
    with pymupdf.open() as document:
        document.new_page().insert_text((30, 100), "Keep the first page.")
        document.new_page().insert_text((30, 100), "Keep the second page.")
        document.save(category / "Manual.pdf")

    accepted = {"review": {"verdict": "accepted"}, "uncertainties": []}
    client = visual_client()
    client.analyze_page_layout.side_effect = [
        {**accepted, "elements": [{"type": "paragraph", "text": "Keep the first page."}]},
        {**accepted, "elements": [{"type": "paragraph", "text": "Keep the second page.", "bbox": [0.1, 0.1, 0.4, 0.3]}]},
    ]
    monkeypatch.setattr(ModelEnricher, "enrich", lambda self, ir: ir)
    monkeypatch.setattr(SemanticBoundaryScorer, "annotate", lambda self, ir: ir)
    output = tmp_path / "output"
    run_auto_chunking(incoming, output, vision_client=client, use_decision=False, retry_delays=(),
                      cache_db_path=":memory:", dpi=72)
    chunks, manifest = load_published_chunks(output, return_manifest=True)
    (chunk,) = chunks

    assert client.analyze_page_layout.call_count == 2
    assert manifest["ocr_partial_pages"] == 0
    assert chunk["source"] == "manuals/Manual.pdf"
    assert chunk["business_metadata"]["citation"] == {"file_name": "Manual.pdf", "page_numbers": [1, 2]}
    assert "Keep the first page." in chunk["content"] and "Keep the second page." in chunk["content"]
    assert '"bbox"' not in json.dumps(chunk)
