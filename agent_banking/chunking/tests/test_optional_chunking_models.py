import json
from types import SimpleNamespace

from agent_banking.chunking.tests.test_chunk_planning_edges import CharacterTokenizer, document_ir, source_block, table_block
from agent_banking.chunking.universal.enrichment import enricher, review, structure
from agent_banking.chunking.universal.enrichment.decisions import engine
from agent_banking.chunking.universal.enrichment.decisions.tables import resolve_table_structures_and_titles
from agent_banking.chunking.universal.enrichment.image_context import excluded_image
from agent_banking.chunking.universal.planning import semantic_boundaries
from agent_banking.chunking.universal.planning.planner import ChunkPlanner
from agent_banking.chunking.universal.text.token_budget import TokenBudget
from agent_banking.core.fallback import ModelFallbackError


ENDPOINT = SimpleNamespace(model="test", url="https://example.invalid", api_key="test")


def unavailable(*args, **kwargs):
    raise ModelFallbackError("chunking.test", [("candidate 1", "timeout")])


def cache():
    return SimpleNamespace(get=lambda _: None, put=lambda *_: None)


def test_failed_decision_batch_keeps_successful_answers(monkeypatch):
    monkeypatch.setattr(engine, "model_primary", lambda _: ENDPOINT)
    monkeypatch.setattr(engine, "chain_fingerprint", lambda _: "chain")
    question = {"type": "choice", "criteria": {"KEEP": "Keep existing source"}}

    def respond(url, key, payload, **kwargs):
        if "missing" in payload["questions"]:
            unavailable()
        return {"answers": {"available": {"choice": "KEEP"}, "missing": {"choice": "KEEP"}}}

    monkeypatch.setattr(engine, "cloud_json", respond)
    ir = {}
    answers = engine.ask_decisions("test", {}, {"missing": question, "available": question}, cache(), batch_size=1, ir=ir)

    assert answers == {"available": {"choice": "KEEP"}}
    assert ir["model_warnings"] == [{"stage": "test", "error_type": "ModelFallbackError"}]


def test_default_decision_batches_keep_questions_with_their_evidence(monkeypatch):
    monkeypatch.setattr(engine, "model_primary", lambda _: ENDPOINT)
    monkeypatch.setattr(engine, "chain_fingerprint", lambda _: "chain")
    question = {"type": "choice", "criteria": {"KEEP": "Keep existing source"}}
    ids = [f"q{i}" for i in range(13)]
    batches = []

    def respond(url, key, payload, **kwargs):
        selected = list(payload["questions"])
        batches.append(selected)
        assert [entry["id"] for entry in payload["state"]["candidates"]] == selected
        return {"answers": {name: {"choice": "KEEP"} for name in selected}}

    monkeypatch.setattr(engine, "cloud_json", respond)
    answers = engine.ask_decisions("test", {"candidates": [{"id": name} for name in ids]},
                                   dict.fromkeys(ids, question), cache(), workers=1)

    assert [len(batch) for batch in batches] == [6, 6, 1]
    assert list(answers) == ids


def test_missing_table_title_decision_preserves_native_table_metadata():
    heading = source_block("h", "heading", "Điều khoản", level=1)
    block = table_block("table", [["Mức phí"], ["1.000đ"]], header_count=1, title="Tiêu đề gốc")
    original = dict(block["table_data"])

    resolve_table_structures_and_titles(document_ir([heading, block]), lambda *_, **__: {})

    assert block["table_data"] == original
    assert "caption" not in block


def test_invalid_table_header_response_does_not_reclassify_source_rows(monkeypatch):
    monkeypatch.setattr(enricher, "model_primary", lambda _: ENDPOINT)
    monkeypatch.setattr(enricher, "chain_fingerprint", lambda _: "chain")
    monkeypatch.setattr(enricher, "cloud_json", lambda *_, **__: {"header_row_count": True})
    model = enricher.ModelEnricher(cache(), workers=1)
    block = table_block("t", [["Phí thường niên"], ["100.000đ"]])
    ir = document_ir([block])

    model._table_headers(ir)

    assert block["table_data"]["rows"] == [["Phí thường niên"], ["100.000đ"]]
    assert block["table_data"]["header_row_count"] == 0
    assert ir["model_warnings"] == [{"stage": "table_header_review", "error_type": "ValueError"}]


def test_failed_image_context_preserves_literal_surrounding_text(monkeypatch):
    monkeypatch.setattr(enricher, "model_primary", lambda _: ENDPOINT)
    monkeypatch.setattr(enricher, "chain_fingerprint", lambda _: "chain")
    monkeypatch.setattr(enricher, "cloud_json", unavailable)
    model = enricher.ModelEnricher(cache(), workers=1)
    paragraph = source_block("p", "paragraph", "Xem thao tác xác nhận hạn mức giao dịch ở hình dưới.")
    image = source_block("i", "image", image={"sha256": "asset", "title": "Tiêu đề gốc"})
    ir = document_ir([paragraph, image])

    model._image_context(ir, 1, image)

    assert excluded_image(image)
    assert image["image"]["status"] == "skipped_context_unavailable"
    assert image["image"]["text_context_assessed"] is False
    assert image["image"]["title"] == "Tiêu đề gốc"
    assert paragraph["text"] == "Xem thao tác xác nhận hạn mức giao dịch ở hình dưới."


def test_failed_structure_window_does_not_discard_other_grounded_annotations(monkeypatch):
    monkeypatch.setattr(structure, "model_primary", lambda _: ENDPOINT)
    monkeypatch.setattr(enricher, "chain_fingerprint", lambda _: "chain")
    model = enricher.ModelEnricher(cache(), workers=1)
    ir = document_ir([source_block(str(i), "paragraph", "Dữ liệu gốc " + str(i)) for i in range(15)])
    ir["warnings"] = []

    def respond(url, key, payload, **kwargs):
        window = json.loads(payload["messages"][1]["content"])
        if window["editable_block_ids"][0] == "b0":
            unavailable()
        return {"heading_levels": [{"block_id": "b12", "level": 2}], "reading_order": [],
                "relations": [], "contexts": [], "warnings": []}

    structure.enrich_structure(model, ir, respond)

    assert ir["blocks"][0]["type"] == "paragraph"
    assert ir["blocks"][12]["type"] == "heading"
    assert [block["text"] for block in ir["blocks"]] == ["Dữ liệu gốc " + str(i) for i in range(15)]
    assert ir["model_warnings"] == [{"stage": "grounded_structure_window", "error_type": "ModelFallbackError"}]


def test_failed_heading_confirmation_restores_native_structure(monkeypatch):
    monkeypatch.setattr(review, "model_primary", lambda _: ENDPOINT)
    monkeypatch.setattr(enricher, "chain_fingerprint", lambda _: "chain")
    model = enricher.ModelEnricher(cache(), workers=1)
    block = source_block("p", "heading", "Văn bản gốc", level=2)
    ir = document_ir([block])
    ir["structure_proposals"] = [{"block_id": "p", "native_level": 0, "proposed_level": 2}]

    review.review_headings(model, ir, unavailable)

    assert block["type"] == "paragraph"
    assert block["text"] == "Văn bản gốc"
    assert ir["heading_decisions"][0]["decision"] == "KEEP_NATIVE"


def test_model_repair_route_is_advisory_when_cloud_adjudication_fails(monkeypatch):
    monkeypatch.setattr(review, "model_primary", lambda _: ENDPOINT)
    monkeypatch.setattr(enricher, "chain_fingerprint", lambda _: "chain")
    model = enricher.ModelEnricher(cache(), workers=1)
    monkeypatch.setattr(model, "_adjudicate_repair", unavailable)
    ir = document_ir([])
    ir.update(warnings=[], deterministic_coverage={"status": "verified_source_reconstruction"})

    review.review_extraction(model, ir, lambda *_, **__: {"answers": {"review_route": {"choice": "REPAIR_EXTRACTION"}}})

    assert ir["decision_review"]["status"] == "advisory"
    assert ir["decision_review"]["answers"]["review_route"]["choice"] == "REPAIR_EXTRACTION"
    assert {item["stage"] for item in ir["model_warnings"]} == {"extraction_adjudication", "review_route"}


def test_failed_semantic_window_keeps_other_cloud_boundaries(monkeypatch):
    monkeypatch.setattr(semantic_boundaries, "model_primary", lambda _: ENDPOINT)
    monkeypatch.setattr(semantic_boundaries, "chain_fingerprint", lambda _: "chain")
    ir = document_ir([source_block(str(i), "paragraph", "Dữ liệu gốc " + str(i)) for i in range(25)])
    scorer = semantic_boundaries.SemanticBoundaryScorer(cache(), TokenBudget(128, 256, CharacterTokenizer()), workers=1)

    def respond(url, key, payload, **kwargs):
        state = json.loads(payload["messages"][1]["content"])
        identifiers = state["editable_ids"]
        assert len(identifiers) <= 12
        assert len(state["atoms"]) <= 18
        if "a0" in identifiers:
            unavailable()
        return {"boundaries": [{"id": identifier, "score": 0.8} for identifier in identifiers]}

    monkeypatch.setattr(semantic_boundaries, "cloud_json", respond)
    scorer.annotate(ir)

    assert ir["semantic_boundaries"] == [
        {"block_id": str(index), "offset": len(ir["blocks"][index]["text"]), "score": 0.8}
        for index in range(12, 25)
    ]
    assert ir["segmentation_method"].endswith("-partial")
    assert ir["model_warnings"] == [{"stage": "semantic_boundaries", "error_type": "ModelFallbackError"}]
    planner = ChunkPlanner(128, 256, tokenizer=CharacterTokenizer(), use_decision=False)
    _, artifact = planner.plan(ir)
    assert artifact["quality"]["status"] == "verified_source_reconstruction"
