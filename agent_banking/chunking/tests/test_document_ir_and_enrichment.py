from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pytest

from agent_banking.chunking.universal.extraction.docx.extractor import DocxExtractor
from agent_banking.chunking.universal.enrichment import enricher as model_enrichment
from agent_banking.chunking.universal.enrichment import review as document_review
from agent_banking.chunking.universal.enrichment.annotations import filter_optional_annotations, validate_annotations
from agent_banking.chunking.universal.models import text_client as model_client
from agent_banking.core.fallback import ModelEndpoint, run_fallback


def _bytes(document):
    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()


def test_docx_textbox_outline_and_links_stay_with_their_own_blocks(tmp_path):
    document = Document()
    paragraph = document.add_paragraph()
    link = OxmlElement("w:hyperlink")
    link.set(qn("w:anchor"), "Beginning")
    link.append(paragraph.add_run("Before")._element)
    paragraph._p.append(link)
    pict = OxmlElement("w:pict")
    textbox = OxmlElement("w:txbxContent")
    text = document.add_heading("Textbox heading", level=2)
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), "1")
    text._p.get_or_add_pPr().append(outline)
    textbox.append(text._p)
    pict.append(textbox)
    paragraph.add_run()._element.append(pict)
    paragraph.add_run("After")

    ir = DocxExtractor(tmp_path, use_decision=False).extract(_bytes(document), "nested.docx")
    before, heading, after = ir["blocks"]
    assert (before["type"], heading["type"], after["type"]) == ("paragraph", "heading", "paragraph")
    assert (before["text"], heading["text"], after["text"]) == ("Before", "Textbox heading", "After")
    assert before["hyperlinks"] == [{"text": "Before", "target": "#Beginning"}]
    assert after["hyperlinks"] == []
    assert len({unit["id"] for unit in ir["units"]}) == 3


def test_shared_docx_extractor_keeps_concurrent_documents_isolated(tmp_path):
    reader = DocxExtractor(tmp_path, use_decision=False)
    inputs = []
    for position in range(6):
        document = Document()
        document.add_paragraph("Source " + str(position))
        inputs.append((_bytes(document), f"{position}.docx"))
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda item: reader.extract(*item), inputs))
    assert [result["blocks"][0]["text"] for result in results] == [f"Source {i}" for i in range(6)]
    assert len({result["units"][0]["id"] for result in results}) == len(inputs)


def test_tracked_deleted_rows_do_not_supply_text_headers_or_extraction_warnings(tmp_path):
    document = Document()
    table = document.add_table(rows=3, cols=1)
    deleted, header, body = table.rows
    deleted.cells[0].text = "Removed interest rate"
    deleted._tr.get_or_add_trPr().append(OxmlElement("w:del"))
    deleted.cells[0]._tc.append(OxmlElement("w:altChunk"))
    header.cells[0].text = "Interest rate"
    header._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    body.cells[0].text = "7%"
    entirely_removed = document.add_table(rows=1, cols=1)
    entirely_removed.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:del"))

    ir = DocxExtractor(tmp_path, use_decision=False).extract(_bytes(document), "tracked.docx")
    block, = ir["blocks"]
    assert block["table_data"]["grid"] == [["Interest rate"], ["7%"]]
    assert block["table_data"]["header_row_count"] == 1
    assert [unit["text"] for unit in ir["units"]] == ["Interest rate", "7%"]
    assert ir["warnings"] == []


@pytest.mark.parametrize("response", [
    '{"x":1,"x":2}',
    '{"x":NaN}',
    '```json\n{"x":1}\n``` trailing',
])
def test_invalid_cloud_json_is_rejected(response):
    with pytest.raises(ValueError):
        model_client.strict_json(response)


def test_candidate_payload_is_configured_once_and_isolated_across_retries(monkeypatch):
    from agent_banking.core import fallback, models

    candidates = [
        ModelEndpoint("first", "First", "https://first.invalid", "test-key", attempts=2),
        ModelEndpoint("second", "Second", "https://second.invalid", "test-key"),
    ]
    monkeypatch.setattr(models, "get_model_chain", lambda module, primary: candidates)
    monkeypatch.setattr(fallback, "run_fallback",
                        lambda module, chain, call: run_fallback(module, chain, call, retry_delays=()))
    calls = []

    def configure(payload, candidate):
        payload["messages"][0]["content"] = candidate.model

    def transport(url, key, payload, *args):
        calls.append((payload["model"], payload["messages"][0]["content"], payload["max_tokens"]))
        if payload["model"] == "First":
            payload["max_tokens"] *= 2
            raise ValueError("retry")
        return {"success": True}

    monkeypatch.setattr(model_client, "_cloud_json", transport)
    original = {"model": "First", "messages": [{"role": "user", "content": "original"}], "max_tokens": 1024}
    assert model_client.cloud_json("https://first.invalid/chat/completions", "test-key", original,
                                   configure_payload=configure) == {"success": True}
    assert calls == [("First", "First", 1024), ("First", "First", 2048), ("Second", "Second", 1024)]
    assert original["messages"][0]["content"] == "original"


def test_bad_optional_context_quarantines_only_the_bad_annotation():
    window = {"blocks": [{"id": "b0", "type": "paragraph"}, {"id": "b1", "type": "paragraph"}],
              "editable_block_ids": ["b1"], "context_source_ids": ["b0", "b1"]}
    valid = {"block_id": "b1", "source_block_ids": ["b0"]}
    rejected = {"block_id": "b1", "source_block_ids": ["missing"]}
    result = {"heading_levels": [], "contexts": [valid, rejected], "relations": [], "warnings": [],
              "reading_order": []}
    clean = validate_annotations(result, window, lambda alias: "native", quarantine_context=True)
    assert clean["contexts"] == [valid]
    assert clean["_rejected_optional_contexts"] == [rejected]
    rejected["context_quotes"] = [{"source_block_id": "b0", "text": "invented"}]
    with pytest.raises(ValueError, match="verbatim"):
        validate_annotations(result, window, lambda alias: "native", quarantine_context=True)


def test_invalid_structure_proposals_do_not_block_valid_native_annotations():
    window = {
        "blocks": [{"id": item, "type": "paragraph"} for item in ("b0", "b1", "b2")],
        "editable_block_ids": ["b1", "b2"],
        "context_source_ids": ["b0", "b1", "b2"],
    }
    good_context = {"block_id": "b1", "source_block_ids": ["b0"]}
    raw = {
        "heading_levels": [
            {"block_id": "b0", "level": 1}, {"block_id": "b1", "level": 1},
            {"block_id": "b1", "level": 2}, {"block_id": "b2", "level": 3},
        ],
        "relations": [{"from": "b2", "to": "made-up", "type": "note_of"}],
        "contexts": [good_context, {"block_id": "b0", "source_block_ids": ["b1"]}],
        "warnings": [],
        "reading_order": ["b1", "b2"],
    }
    clean = filter_optional_annotations(raw, window, lambda alias: "native")
    assert clean["heading_levels"] == [{"block_id": "b2", "level": 3}]
    assert clean["contexts"] == [good_context]
    assert clean["reading_order"] == clean["relations"] == []
    assert len(clean["_rejected_annotations"]) == 6


def test_observation_keeps_conflicting_passages_together():
    ir = {"blocks": [{"id": "A", "type": "paragraph", "text": "15 million"},
                     {"id": "B", "type": "paragraph", "text": "20 million"}],
          "source_review": [{"code": "conflict", "reason": "Amounts differ", "block_ids": ["A", "B"]}]}
    observations = list(document_review._source_observations(ir))
    assert len(observations) == 1
    assert [ref["id"] for ref in observations[0]["native_evidence"]] == ["A", "B"]


def test_document_route_keeps_source_review_after_another_window_is_downgraded(monkeypatch):
    endpoint = SimpleNamespace(model="fake", url="https://example.invalid", api_key="test-key")
    monkeypatch.setattr(document_review, "model_primary", lambda _: endpoint)
    monkeypatch.setattr(document_review, "_observation_windows", lambda *_: [[0], [1]])
    monkeypatch.setattr(model_enrichment, "chain_fingerprint", lambda _: "fake-chain")
    enricher = model_enrichment.ModelEnricher(SimpleNamespace(get=lambda _: None, put=lambda *_: None), workers=1)

    def route(choice):
        return {"answers": {"review_route": {"choice": choice}}}

    monkeypatch.setattr(enricher, "_adjudicate_repair", lambda *_: route("VALIDATE"))
    responses = iter([route("REVIEW_SOURCE"), route("REPAIR_EXTRACTION")])
    ir = {"blocks": [], "units": [], "warnings": [],
          "deterministic_coverage": {"status": "verified_source_reconstruction"}}
    document_review.review_extraction(enricher, ir, lambda *_, **__: next(responses))
    assert ir["decision_review"]["answers"]["review_route"]["choice"] == "REVIEW_SOURCE"
    assert len(ir["decision_review"]["window_reviews"]) == 2


def test_reviewed_image_evidence_survives_offline_enrichment():
    image = {"id": "image", "type": "image", "text": "Visual transcript",
             "image": {"status": "reviewed", "ocr_text": "Visual transcript"}}
    ir = {"blocks": [image], "page_reviews": [{"page": 0}]}
    result = model_enrichment.ModelEnricher(None, use_llm=False, use_decision=False).enrich(ir)
    assert result["image_policy"] == "reviewed_visual_source_v1"
    assert image["image"]["ocr_text"] == "Visual transcript"
