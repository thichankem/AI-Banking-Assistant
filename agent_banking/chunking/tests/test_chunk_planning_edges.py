from types import SimpleNamespace

import pytest

from agent_banking.chunking.universal.enrichment.decisions import images as decision_media
from agent_banking.chunking.universal.enrichment.decisions.tables import (
    merge_continued_tables,
    resolve_table_structures_and_titles,
)
from agent_banking.chunking.universal.planning.planner import ChunkPlanner
from agent_banking.chunking.universal.planning.table_rendering import render_structured_html_table
from agent_banking.chunking.universal.text.boundaries import protected_markup_spans, safe_boundaries
from agent_banking.chunking.universal.text.language import project_vietnamese
from agent_banking.chunking.universal.text.token_budget import TokenBudget, format_passage_fields


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return SimpleNamespace(ids=list(range(len(text))), offsets=[(i, i + 1) for i in range(len(text))])

    def to_str(self):
        return "test character tokenizer"


def source_block(identifier, kind, text="", page=None, **extra):
    return {"id": identifier, "type": kind, "text": text,
            "provenance": {"region": "body", "page_index": page}, **extra}


def table_block(identifier, grid, header_count=0, page=None, **extra):
    width = len(grid[0])
    headers = grid[0] if header_count else ["#" + str(i + 1) for i in range(width)]
    cells = [{"row": row, "column": column, "row_span": 1, "col_span": 1, "text": value}
             for row, values in enumerate(grid) for column, value in enumerate(values)]
    return source_block(identifier, "table", page=page, table_data={
        "table_id": identifier, "grid": grid, "header_row_count": header_count, "headers": headers,
        "header_grid": grid[:header_count], "rows": grid[header_count:], "cells": cells, **extra,
    })


def document_ir(blocks, units=()):
    return {"source": "planning.docx", "document_id": "d", "document_version": "a" * 64,
            "blocks": blocks, "units": list(units), "relations": [], "assets": {}}


def test_context_excerpt_skips_an_oversized_atomic_formula():
    budget = TokenBudget(32, 44, CharacterTokenizer())
    formula = "$$" + "x+" * 100 + "y$$"
    suffix = "Hạn mức giao dịch"
    retrieval = {"context": "Nhãn " * 20}

    budget.embedding_excerpt(formula + "\n" + suffix, retrieval)

    assert retrieval["passage_content"] == suffix
    assert budget.count(format_passage_fields(formula, retrieval)) <= 44


def test_foreign_projection_preserves_formula_inside_foreign_sentence():
    assert project_vietnamese("Please follow this rule $x=3$ and update your bank.") == "$x=3$"


def test_mixed_language_cell_retains_authoritative_text_and_offsets():
    value = "Khách hàng xem hạn mức. Please read this with your card."
    table = table_block("dt", [[value]])
    planner = ChunkPlanner(128, 256, tokenizer=CharacterTokenizer(), use_decision=False)

    (chunk,), _ = planner.plan(document_ir([table]))

    retrieval = chunk["business_metadata"]["retrieval"]
    assert "Khách hàng xem hạn mức." in retrieval["embedding_content"]
    assert "Please" not in retrieval["embedding_content"]
    assert chunk["table_data"]["rows"] == [[value]]
    element, = chunk["business_metadata"]["elements"]
    first, last = element["source_span"]
    assert element["text"] == value[first:last] == " Please read this with your card."


def test_table_position_mismatch_fails_even_when_value_exists_in_another_cell():
    table = table_block("dt", [["A", "B"]])
    planner = ChunkPlanner(128, 256, tokenizer=CharacterTokenizer(), use_decision=False)
    bad_unit = {"id": "dt:u0", "block_id": "dt", "text": "A", "row": 0, "column": 1}

    with pytest.raises(RuntimeError, match="1 text units"):
        planner.plan(document_ir([table], [bad_unit]))


def test_blank_table_is_preserved_without_indexing_punctuation():
    table = table_block("dt", [["", ""]])
    planner = ChunkPlanner(128, 256, tokenizer=CharacterTokenizer(), use_decision=False)

    (chunk,), _ = planner.plan(document_ir([table]))

    assert chunk["table_data"]["rows"] == [["", ""]]
    assert chunk["business_metadata"]["retrieval"]["role"] == "attachment_only"


def test_foreign_heading_preserves_section_boundary():
    ir = document_ir([
        source_block("h1", "heading", "Mở thẻ", level=1),
        source_block("p1", "paragraph", "Đăng ký phát hành thẻ."),
        source_block("h2", "heading", "How can you check the card?", level=1),
        source_block("p2", "paragraph", "Kiểm tra trạng thái thẻ."),
    ])
    planner = ChunkPlanner(128, 256, tokenizer=CharacterTokenizer(), use_decision=False)

    (opening, status), _ = planner.plan(ir)

    assert opening["parent_id"] != status["parent_id"]
    assert "Mở thẻ" not in status["breadcrumb"]
    assert status["business_metadata"]["elements"][0]["block_id"] == "h2"


def test_table_title_keeps_exact_long_source_and_valid_header_grid():
    title = "Điều kiện áp dụng " * 22 + "năm 2026"
    previous = source_block("p", "paragraph", title)
    table = table_block("dt", [["Dịch vụ"], ["Mở thẻ"]], header_detection="unmarked")

    def decision(role, state, questions, **kwargs):
        assert len(state["tables"][0]["prev_text"]) == 200
        assert "HEADERS_2" not in questions["t0_hdr"]["criteria"]
        return {"t0_hdr": {"choice": "HEADERS_1"}, "t0_title": {"choice": "PREV_BLOCK"}}

    ir = document_ir([previous, table])
    resolve_table_structures_and_titles(ir, decision)
    planner = ChunkPlanner(128, 256, tokenizer=CharacterTokenizer(), use_decision=False)
    chunks, _ = planner.plan(ir)
    data, = [chunk["table_data"] for chunk in chunks if chunk["chunk_type"] == "table"]
    assert data["title"] == title
    assert data["title_source_block_id"] == "p"
    assert data["header_grid"] == [["Dịch vụ"]]
    assert data["rows"] == [["Mở thẻ"]]


def test_continued_table_keeps_repeated_headers_and_canonical_unit_rows():
    blocks = [table_block("table" + str(i), [["Mức phí"], [str(i)]], header_count=1, page=i)
              for i in range(3)]
    units = [{"id": block["id"] + ":" + str(row), "block_id": block["id"],
              "row": row, "column": 0, "text": text}
             for block in blocks for row, (text,) in enumerate(block["table_data"]["grid"])]
    ir = document_ir(blocks, units)

    def decision(role, state, questions, **kwargs):
        return {key: {"choice": "MERGE_CONTINUED"} for key in questions}

    merge_continued_tables(ir, decision)
    table, = ir["blocks"]
    data = table["table_data"]
    assert [row["page_index"] for row in data["row_provenance"]] == [0, 0, 1, 2]
    assert len(data["header_occurrences"]) == 3
    assert data["rows"] == [["0"], ["1"], ["2"]]
    planner = ChunkPlanner(128, 256, tokenizer=CharacterTokenizer(), use_decision=False)
    _, artifact = planner.plan(ir)
    assert artifact["quality"]["covered_units"] == len(units)


def test_image_evidence_uses_original_heading_coordinates(monkeypatch):
    heading = source_block("h", "heading", "  Quy trình mở thẻ  ")
    image = source_block("i", "image", image={"title": "Tiêu đề gốc"})
    monkeypatch.setattr(decision_media, "model_primary", lambda role: SimpleNamespace(model="test"))

    def decision(role, state, questions, **kwargs):
        return {"img_0_title": {"choice": "SECTION_HEADING"}}

    decision_media.resolve_image_titles_and_context(document_ir([heading, image]), decision)

    (evidence,) = image["image"]["context_evidence"]
    first, last = evidence["source_span"]
    assert image["image"]["source_title"] == "Tiêu đề gốc"
    assert evidence["source_block_id"] == "h"
    assert evidence["quote"] == heading["text"][first:last] == "Quy trình mở thẻ"


def test_unicode_and_markup_boundaries_preserve_graphemes():
    assert safe_boundaries("a\u0302\r\n🇻🇳b") == [0, 2, 4, 6, 7]
    assert protected_markup_spans(r"\\$x$") == [(2, 5)]
    assert protected_markup_spans(r"\$x$") == []


def test_structured_table_default_body_offset_respects_header_row():
    cells = [
        {"row": 0, "column": 0, "text": "Tiêu đề"},
        {"row": 1, "column": 0, "text": "Ô gộp", "col_span": 2},
    ]
    html = render_structured_html_table(["A", "B"], [["Ô gộp", "Ô gộp"]], cells=cells)

    assert "<th>Tiêu đề</th><th>B</th>" in html
    assert '<td colspan="2">Ô gộp</td>' in html
