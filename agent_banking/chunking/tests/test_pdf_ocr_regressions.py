"""OCR agreement, numeric glyphs, and source table metadata regressions."""

from copy import deepcopy
from io import BytesIO
import json
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image, ImageDraw
import pymupdf
import pytest

from agent_banking.chunking.universal.extraction.pdf.layout import merge_region_observation, normalize_visual_layout, validate_page
from agent_banking.chunking.universal.extraction.pdf.native_text import (
    native_table_elements,
    native_text_elements,
    page_requires_scan,
    unreliable_native_text,
)
from agent_banking.chunking.universal.extraction.pdf.page_review import PdfVisualReviewer
from agent_banking.chunking.universal.extraction.pdf.tables import table_from_html
from agent_banking.chunking.universal.models.vision_client import DocumentVisionClient


def _png(text, size=(300, 100), font_size=38):
    output = BytesIO()
    with Image.new("RGB", size, "white") as image:
        ImageDraw.Draw(image).text((10, 10), text, fill="black", font_size=font_size)
        image.save(output, format="PNG")
    return output.getvalue()


def _reading(*parts):
    return {
        "elements": [{"type": "paragraph", "text": part, "bbox": [0.1, 0.1, 0.9, 0.9]} for part in parts],
        "uncertainties": [],
        "review": {"verdict": "accepted"},
    }


def _reader(tmp_path, draft, reviewed):
    cache = Mock()
    cache.get.return_value = None
    draft_client = SimpleNamespace(model="draft", analyze_page_layout=Mock(side_effect=lambda *args: deepcopy(draft)))
    review_client = SimpleNamespace(model="review", analyze_page_layout=Mock(side_effect=lambda *args: deepcopy(reviewed)))
    return PdfVisualReviewer(tmp_path, cache, draft_client, review_client, False)


def test_usable_cloud_reading_does_not_need_second_model_agreement(tmp_path):
    first = _reading("Standard deadline: 14 days.", "Exception: 21 days.")
    first["review"]["verdict"] = "unresolved"
    first["uncertainties"] = [{"text": "Exception"}]
    reader = _reader(tmp_path, first, _reading("Standard deadline: 140 days."))
    result = reader.read(_png("14 days"), [], 300, 100)
    assert result["elements"] == first["elements"]
    assert result["visual_review"]["incomplete"] is True
    assert result["visual_review"]["provider_verdict"] == "unresolved"
    reader.review_client.analyze_page_layout.assert_not_called()
    reader.cache.put.assert_not_called()


def test_document_ocr_keeps_valid_records_when_another_record_is_invalid(tmp_path):
    source = _reading("Standard transaction limit: 14.")
    source["elements"][0]["representation"] = "provider field unused by the chunker"
    source["elements"].append({"type": "unknown", "text": ""})
    parsed = DocumentVisionClient()._parse_and_validate_layout(json.dumps(source))
    reader = _reader(tmp_path, parsed, _reading())

    result = reader.read(_png("14"), [], 300, 100)

    assert result["elements"] == _reading("Standard transaction limit: 14.")["elements"]
    assert result["visual_review"]["incomplete"] is True
    assert result["visual_review"]["skipped_elements"] == 1
    reader.cache.put.assert_not_called()


def test_cloud_cache_checks_content_and_position(tmp_path):
    source = _reading("Daily transfer limit: 14")
    pixels = _png("14")
    reader = _reader(tmp_path, source, source)
    result = reader.read(pixels, [], 300, 100)
    reader.cache.get.return_value = result
    assert reader.read(pixels, [], 300, 100) == result
    assert reader.vision_client.analyze_page_layout.call_count == 1

    altered = deepcopy(result)
    altered["elements"][0].update(text="Daily transfer limit: 140", bbox=[0.4, 0.4, 0.9, 0.9])
    reader.cache.get.return_value = altered
    assert reader.read(pixels, [], 300, 100)["elements"] == source["elements"]
    assert reader.vision_client.analyze_page_layout.call_count == 2


def test_cloud_table_with_incomplete_cells_keeps_reported_data_as_text(tmp_path):
    source = _reading()
    source["elements"] = [{
        "type": "table", "bbox": [0.1, 0.1, 0.9, 0.9],
        "table_data": {"headers": ["Fee"], "header_grid": [["Fee"]], "header_row_count": 1,
                       "rows": [["7%"]], "cells": []},
    }]
    result = _reader(tmp_path, source, _reading()).read(_png("7%"), [], 300, 100)
    assert result["elements"][0]["text"] == "Fee\n7%"
    assert result["elements"][0]["type"] == "paragraph"
    assert result["visual_review"]["downgraded_tables"] == 1


def test_overlapping_tables_with_same_grid_and_different_spans_are_unresolved():
    merged = table_from_html("<table><tr><th colspan='2'>Fees</th></tr></table>", 1)
    separate = table_from_html("<table><tr><th>Fees</th><th></th></tr></table>", 1)
    assert merged["header_grid"] == separate["header_grid"]
    element = {"type": "table", "bbox": [0.1, 0.1, 0.9, 0.9], "table_data": merged}
    with pytest.raises(ValueError, match="regions disagree"):
        merge_region_observation([element], {**element, "table_data": separate})


@pytest.mark.parametrize("second", ["transaction limit is 50,000", "limit is 50,000", "is 50,000"])
def test_overlapping_cropped_clauses_are_not_published_as_duplicate_text(second):
    left = {"type": "paragraph", "text": "Standard daily transaction limit",
            "bbox": [0.1, 0.2, 0.56, 0.25], "region_pixel_path": [[0, 0, 625, 625]]}
    right = {"type": "paragraph", "text": second,
             "bbox": [0.35, 0.2, 0.8, 0.25], "region_pixel_path": [[375, 0, 1000, 625]]}
    with pytest.raises(ValueError, match="partial readings"):
        merge_region_observation([left], right)


def test_visual_table_requires_every_coordinate():
    record = _reading()
    record["elements"] = [{
        "type": "table", "bbox": [0.1, 0.1, 0.9, 0.9],
        "table_data": {"headers": ["Label", "Value"], "header_row_count": 1,
                       "header_grid": [["Label", "Value"]], "rows": [["Daily limit", "7%"]],
                       "cells": []},
    }]
    with pytest.raises(ValueError, match="complete cell coordinates"):
        validate_page(record, [])


def test_reported_heading_with_location_is_preserved_in_order():
    record = _reading("Terms remain in effect.")
    record["headings"] = [{"level": 1, "text": "Transfer terms", "bbox": [0.1, 0.02, 0.6, 0.08]}]
    normalize_visual_layout(record)
    assert [e["text"] for e in record["elements"]] == ["Transfer terms", "Terms remain in effect."]
    normalize_visual_layout(record)
    assert len(record["elements"]) == 2


def test_standalone_heading_without_position_is_not_silently_dropped():
    record = _reading("Terms remain in effect.")
    record["headings"] = [{"level": 1, "text": "Transfer terms"}]
    with pytest.raises(ValueError, match="heading inside elements"):
        normalize_visual_layout(record)


def test_located_heading_already_in_a_paragraph_is_not_added_twice():
    record = _reading("Transfer terms\nTerms remain in effect.")
    record["headings"] = [{"level": 1, "text": "Transfer terms", "bbox": [0.1, 0.1, 0.6, 0.2]}]
    normalize_visual_layout(record)
    assert [e["text"] for e in record["elements"]] == ["Transfer terms\nTerms remain in effect."]


def test_native_pdf_checks_short_raster_rate_and_numeric_soft_hyphen():
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text((60, 250), "Interest rate for preferential loans:")
        page.insert_image(pymupdf.Rect(300, 232, 400, 282), stream=_png("7%", (200, 100)))
        assert page_requires_scan(page, native_text_elements(page))

        signed = document.new_page(width=600, height=800)
        signed.insert_text((60, 250), "Lower tolerance: -0.05")
        observations = native_text_elements(signed)
        observations[0]["text"] = "Lower tolerance: \u00ad0.05"
        assert unreliable_native_text(observations[0]["text"])
        assert not unreliable_native_text("inter\u00adnational")
        assert page_requires_scan(signed, observations)


def test_native_grouped_table_header_uses_complete_merged_rows():
    with pymupdf.open() as document:
        page = document.new_page(width=650, height=400)
        for y, begin in ((80, 40), (120, 230), (160, 40), (200, 40), (240, 40)):
            page.draw_line((begin, y), (600, y))
        for x, top in ((40, 80), (230, 80), (415, 120), (600, 80)):
            page.draw_line((x, top), (x, 240))
        for x, y, value in ((100, 120, "Specimen"), (320, 105, "Acceptance limits"),
                            (280, 145, "Lower"), (470, 145, "Upper"),
                            (100, 185, "Steel"), (280, 185, "-0.05"), (470, 185, "+0.05"),
                            (100, 225, "Glass"), (280, 225, "-0.005"), (470, 225, "+0.005")):
            page.insert_text((x, y), value)
        table, = native_table_elements(page)
    assert table["table_data"]["header_row_count"] == 2
    assert table["table_data"]["headers"] == ["Specimen", "Acceptance limits / Lower", "Acceptance limits / Upper"]
    assert [row[0] for row in table["table_data"]["rows"]] == ["Steel", "Glass"]
