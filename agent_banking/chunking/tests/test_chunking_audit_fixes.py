"""Targeted regression tests covering all audited chunking bug fixes and upgrades."""

from io import BytesIO
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from docx import Document
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from PIL import Image, ImageDraw
import pymupdf
import pytest

from agent_banking.chunking.golden_suite import score
from agent_banking.chunking.tests.test_chunk_planning_edges import (
    CharacterTokenizer,
    document_ir,
    source_block,
    table_block,
)
from agent_banking.chunking.universal.enrichment.annotations import (
    ANNOTATION_ARRAYS,
    apply_annotations,
    filter_optional_annotations,
)
from agent_banking.chunking.universal.enrichment.decisions.tables import (
    compatible_table_headers,
    merge_continued_tables,
    resolve_table_structures_and_titles,
)
from agent_banking.chunking.universal.extraction.docx.openxml import native_math
from agent_banking.chunking.universal.extraction.pdf.extractor import PdfExtractor, _mark_pdf_furniture
from agent_banking.chunking.universal.extraction.pdf.layout import reading_order
from agent_banking.chunking.universal.extraction.pdf.page_review import _prepare_layout
from agent_banking.chunking.universal.pipeline import LayoutAwareChunker
from agent_banking.chunking.universal.planning.planner import ChunkPlanner
from agent_banking.chunking.universal.planning.schema import infer_business_metadata
from agent_banking.chunking.universal.planning.source_spans import find_sentence_gap_ends
from agent_banking.embedding.embedder import BankingEmbedder


def test_annotation_normalization_repairs_commas_and_missing_optional_arrays():
    window = {
        "blocks": [{"id": "b0", "type": "paragraph"}, {"id": "b1", "type": "paragraph"}],
        "editable_block_ids": ["b0", "b1"],
        "context_source_ids": ["b0", "b1"],
    }
    raw = {"heading_levels": [{"block_id": "b0,", "level": "2"}]}
    cleaned = filter_optional_annotations(raw, window, lambda _: "Điều khoản")
    assert cleaned["heading_levels"] == [{"block_id": "b0", "level": 2}]
    for key in ANNOTATION_ARRAYS:
        assert isinstance(cleaned[key], list)

    with pytest.raises(ValueError):
        filter_optional_annotations({}, window, lambda _: "")


def test_apply_annotations_honors_single_page_ocr_reading_order():
    blocks = [
        {"id": "d:v1:b0", "type": "paragraph", "text": "Right column", "level": 0, "provenance": {"page_index": 0}},
        {"id": "d:v1:b1", "type": "paragraph", "text": "Left column", "level": 0, "provenance": {"page_index": 0}},
    ]
    ir = {"blocks": blocks, "relations": [], "page_reviews": [{"page_index": 0}]}
    native_ids = {"b0": "d:v1:b0", "b1": "d:v1:b1"}
    by_id = {b["id"]: b for b in blocks}
    result = {
        "heading_levels": [],
        "relations": [],
        "contexts": [],
        "warnings": [],
        "reading_order": ["b1", "b0"],
    }
    apply_annotations(ir, [(0, result)], native_ids, by_id)
    assert [b["text"] for b in ir["blocks"]] == ["Left column", "Right column"]
    assert [b["reading_order"] for b in ir["blocks"]] == [0, 1]


def test_pdf_furniture_and_caption_order_enable_multi_page_table_merge():
    t1 = table_block("t1", [["Loại phí", "Mức thu"], ["Thường", "10.000đ"]], header_count=0, page=0)
    t1["table_data"]["header_detection"] = "visual_model_review"
    t1["table_data"]["headers"] = ["", ""]
    footer = source_block("f1", "paragraph", "Trang 1/3", page=0)
    footer["provenance"].update(bbox=[0.4, 0.94, 0.6, 0.98], bbox_granularity="region")
    hdr_p2 = source_block("rh2", "heading", "BÁO CÁO TÀI CHÍNH 2025", page=1, level=1)
    hdr_p2["provenance"].update(bbox=[0.1, 0.02, 0.9, 0.06], bbox_granularity="region")
    t2 = table_block("t2", [["Loại phí", "Mức thu"], ["Nhanh", "20.000đ"]], header_count=0, page=1)
    t2["table_data"]["header_detection"] = "visual_model_review"
    t2["table_data"]["headers"] = ["", ""]

    hdr_p1 = source_block("rh1", "heading", "BÁO CÁO TÀI CHÍNH 2025", page=0, level=1)
    hdr_p1["provenance"].update(bbox=[0.1, 0.02, 0.9, 0.06], bbox_granularity="region")
    hdr_p3 = source_block("rh3", "heading", "BÁO CÁO TÀI CHÍNH 2025", page=2, level=1)
    hdr_p3["provenance"].update(bbox=[0.1, 0.02, 0.9, 0.06], bbox_granularity="region")
    p3_body = source_block("p3", "paragraph", "Kết thúc báo cáo.", page=2)

    ir = document_ir([hdr_p1, t1, footer, hdr_p2, t2, hdr_p3, p3_body])
    _mark_pdf_furniture(ir, 3)

    assert hdr_p1["provenance"]["region"].startswith("furniture:")
    assert hdr_p2["provenance"]["region"].startswith("furniture:")
    assert footer["provenance"]["region"] == "furniture:page_number"

    def hdr_decision(role, state, questions, **kwargs):
        return {k: {"choice": "HEADERS_1" if k.endswith("_hdr") else "SECTION_HEADING"} for k in questions}

    resolve_table_structures_and_titles(ir, hdr_decision)
    assert t1["table_data"]["headers"] == ["Loại phí", "Mức thu"]
    assert t2["table_data"]["headers"] == ["Loại phí", "Mức thu"]

    merge_continued_tables(ir, lambda role, state, q, **kw: {k: {"choice": "MERGE_CONTINUED"} for k in q})
    tables = [b for b in ir["blocks"] if b["type"] == "table"]
    assert len(tables) == 1
    assert tables[0]["table_data"]["rows"] == [["Thường", "10.000đ"], ["Nhanh", "20.000đ"]]


def test_compatible_headers_accept_col_span_and_column_span_equivalently():
    first = {
        "header_row_count": 1,
        "header_grid": [["Hạn mức", ""]],
        "headers": ["Hạn mức", "Hạn mức"],
        "cells": [{"row": 0, "column": 0, "row_span": 1, "col_span": 2, "text": "Hạn mức"}],
    }
    second = {
        "header_row_count": 1,
        "header_grid": [["Hạn mức", ""]],
        "headers": ["Hạn mức", "Hạn mức"],
        "cells": [{"row": 0, "column": 0, "row_span": 1, "column_span": 2, "text": "Hạn mức"}],
    }
    assert compatible_table_headers(first, second) is True


def test_page_granularity_bbox_preserves_order_and_html_table_fallback_avoids_skip():
    mixed = [
        {"type": "heading", "text": "Tiêu đề trên", "level": 1, "bbox": [0.1, 0.05, 0.8, 0.1], "bbox_granularity": "region"},
        {"type": "paragraph", "text": "Nội dung dưới", "bbox": [0, 0, 1, 1], "bbox_granularity": "page"},
    ]
    assert [e["text"] for e in reading_order(mixed)] == ["Tiêu đề trên", "Nội dung dưới"]

    broken_html = {
        "elements": [{
            "type": "table",
            "table_html": "<table><tr><th>Phí dịch vụ</th><td>50.000đ</tr></table>",
            "bbox": [0.1, 0.2, 0.9, 0.5],
        }],
        "uncertainties": [],
        "review": {"verdict": "accepted"},
    }
    prepared = _prepare_layout(broken_html)
    assert prepared["visual_review"]["skipped_elements"] == 0
    assert prepared["visual_review"]["downgraded_tables"] == 1
    assert "Phí dịch vụ" in prepared["elements"][0]["text"]
    assert "50.000đ" in prepared["elements"][0]["text"]


def test_ancestor_heading_carry_eliminates_heading_only_chunks_and_widens_parent():
    ir = document_ir([
        source_block("h1", "heading", "CHƯƠNG I: QUY ĐỊNH CHUNG", level=1),
        source_block("h2", "heading", "Điều 1. Phạm vi áp dụng", level=2),
        source_block("p1", "paragraph", "Áp dụng cho toàn bộ khách hàng cá nhân tại A-Bank."),
        source_block("h3", "heading", "Điều 2. Đối tượng tham gia", level=2),
        source_block("p2", "paragraph", "Khách hàng từ 18 tuổi trở lên có năng lực hành vi dân sự."),
    ])
    planner = ChunkPlanner(128, 256, tokenizer=CharacterTokenizer(), use_decision=False)
    chunks, _ = planner.plan(ir)

    for chunk in chunks:
        assert chunk["content"].strip() != chunk["section_title"].strip()
        assert chunk["parent_content"].strip() != chunk["content"].strip()
    assert "CHƯƠNG I: QUY ĐỊNH CHUNG" in chunks[0]["content"]
    assert chunks[0]["breadcrumb"].endswith("CHƯƠNG I: QUY ĐỊNH CHUNG > Điều 1. Phạm vi áp dụng")


def test_table_fragment_parent_content_retains_sibling_rows_and_prose():
    table = table_block("tbl", [["Kỳ hạn", "Lãi suất"], ["1 tháng", "4.0%"], ["6 tháng", "5.5%"]], header_count=1)
    ir = document_ir([
        source_block("h1", "heading", "Biểu lãi suất tiền gửi", level=1),
        source_block("intro", "paragraph", "Bảng dưới đây quy định lãi suất tiết kiệm có kỳ hạn áp dụng toàn hệ thống từ ngày 01/01/2026 cho khách hàng cá nhân tại quầy và trực tuyến."),
        table,
    ])
    planner = ChunkPlanner(128, 256, table_rows=1, tokenizer=CharacterTokenizer(), use_decision=False)
    chunks, _ = planner.plan(ir)
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) == 2
    assert "4.0%" in table_chunks[1]["parent_content"]
    assert "5.5%" in table_chunks[0]["parent_content"]


def test_short_word_vietnamese_sentence_breaks_and_extended_omml():
    text = "Phí giao dịch là 10.000 đồng. Khách hàng được miễn phí với TS. Nguyễn Văn A."
    gaps = find_sentence_gap_ends(text, use_decision_for_dots=False)
    sentence_break = text.index("Khách")
    abbrev_pos = text.index("Nguyễn")
    assert sentence_break in gaps
    assert abbrev_pos not in gaps

    xml = parse_xml(
        f'<m:oMath {nsdecls("m")}><m:d><m:e><m:r><m:t>a+b</m:t></m:r></m:e></m:d></m:oMath>'
    )
    rendered = native_math(xml)
    assert not rendered.startswith("<code")
    assert "(a+b)" in rendered


def test_docx_ocr_images_emit_multimodal_chunks_with_images():
    doc = Document()
    doc.add_paragraph("Hướng dẫn bảo mật.")
    fig = BytesIO()
    with Image.new("RGB", (200, 60), "white") as img:
        ImageDraw.Draw(img).text((10, 10), "OTP 6 số", fill="black")
        img.save(fig, format="PNG")
    fig.seek(0)
    doc.add_picture(fig)
    data = BytesIO()
    doc.save(data)

    stub = SimpleNamespace(
        model="fixture-vision",
        analyze_page_layout=lambda pixels, prompt: {
            "elements": [{"type": "caption", "text": "OTP 6 số", "bbox": [0.1, 0.1, 0.9, 0.8]}],
            "review": {"verdict": "accepted"},
            "uncertainties": [],
        },
    )
    with LayoutAwareChunker(use_llm=False, use_decision=False, vision_client=stub, ocr_docx_images=True,
                            cache_db_path=":memory:") as chunker:
        chunks = chunker.chunk_docx(docx_bytes=data.getvalue(), source_name="guide.docx")
    multimodal = [c for c in chunks if c["chunk_type"] == "multimodal"]
    assert len(multimodal) == 1
    assert multimodal[0]["images"]
    assert "OTP 6 số" in multimodal[0]["content"]


def test_domain_metadata_and_embedder_passage_content_preserved(tmp_path):
    meta = infer_business_metadata("A_Bank/3. Văn bản pháp lý tuân thủ/PL 06 Quy chế.docx")
    assert meta["domain"] == "phap_ly_tuan_thu"
    assert meta["doc_type"] == "compliance_policy"

    embedder = BankingEmbedder(model_name="stub", dimension=2, api_key="k", embedding_url="http://localhost",
                               cache_db_path=tmp_path / "embed.db")
    passage = embedder.format_virtual_passage({
        "content": "Full very long raw content",
        "business_metadata": {
            "schema_version": "2.0",
            "retrieval": {"context": "Doc", "passage_content": "Budgeted excerpt"},
        },
    })
    assert "Budgeted excerpt" in passage
    assert "Full very long raw content" not in passage


def test_fingerprint_ignores_tests_and_code_obeys_gemini_laws(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.parts:
            continue
        assert path.name != "__init__.py", f"Forbidden __init__.py: {path}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        ternaries = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.IfExp)]
        assert not ternaries, f"Ternary operators found in {path}: {ternaries}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name in ("re", "regex", "typing") for alias in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in ("re", "regex", "typing"), path

    with LayoutAwareChunker(cache_db_path=":memory:", asset_dir=tmp_path) as chunker:
        assert all("tests/" not in key and key != "golden_suite.py" for key in chunker._code_snapshot)


def test_golden_suite_strict_gate_flags_partial_ocr_and_empty_headers(tmp_path):
    import hashlib
    from agent_banking.chunking.publication import _publish_document_files, publish_snapshot
    from agent_banking.chunking.universal.planning.schema import build_universal_chunk

    pub = tmp_path / "pub"
    source = "a.pdf"
    aid = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
    digest = "d" * 64
    fp = "f" * 64
    bad_chunk = build_universal_chunk(
        chunk_id="c1",
        parent_id="p1",
        source=source,
        chunk_type="table",
        breadcrumb="a > Tiêu đề",
        section_title="Tiêu đề",
        content="Tiêu đề",
        parent_content="Tiêu đề",
        table_data={"headers": ["", ""]},
        business_metadata={
            "document_id": aid,
            "document_version": digest,
            "pipeline_fingerprint": fp,
            "retrieval": {"role": "child"},
            "provenance": {"locations": [{"page_index": 0}]},
        },
    )
    chunks = [bad_chunk]
    art = {
        "document": {"source": source, "document_id": aid, "document_version": digest, "assets": {},
                     "extraction_method": "pdf_cloud_ocr"},
        "parents": {"p1": {"parent_id": "", "section_title": "Tiêu đề"}},
        "quality": {"index_ready": True, "visual_policy": "pdf_cloud_ocr"},
        "pipeline_fingerprint": fp,
    }
    record = _publish_document_files(pub, chunks, art, digest, fp)
    manifest = {
        "module": "chunking",
        "schema_version": "2.0",
        "pipeline_fingerprint": fp,
        "index_ready": True,
        "total_files": 1,
        "failed_files": 0,
        "total_chunks": 1,
        "ocr_failed_pages": 0,
        "ocr_partial_pages": 1,
        "records": [{**record, "rel_path": source, "status": "SUCCESS"}],
    }
    publish_snapshot(pub, chunks, manifest, {aid: chunks})
    report = score(pub, {source: ["Tiêu đề"]}, strict=True)
    assert report["passed"] is False
    assert report["ocr_partial_pages"] == 1
    assert report["empty_table_headers"] == 1
    assert report["heading_only_chunks"] == 1
    assert report["parent_eq_child_chunks"] == 1
