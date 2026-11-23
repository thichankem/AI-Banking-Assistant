"""Exercise the native workbook, Cloud OCR and published Excel row lookup paths."""

from contextlib import closing
from copy import deepcopy
from io import BytesIO
import json
from types import SimpleNamespace
from unittest.mock import Mock
from xml.sax.saxutils import escape
from zipfile import ZipFile

from docx import Document
from PIL import Image
import pymupdf
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from agent_banking.chunking import run_chunking
from agent_banking.chunking.universal.enrichment.enricher import ModelEnricher
from agent_banking.chunking.universal.extraction.xlsx import extractor as excel_extractor
from agent_banking.chunking.universal.pipeline import UniversalDocumentChunker
from agent_banking.chunking.universal.planning.semantic_boundaries import SemanticBoundaryScorer
from agent_banking.embedding.run_embedding import normalize_searchable_chunks
from agent_banking.tools.rag_tool import BankingRAGTool
from agent_banking.vector_db.qdrant_store import QdrantBankingStore, mark_publication_chunks


NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
DRAW = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
MAIN = "http://schemas.openxmlformats.org/drawingml/2006/main"
CHART = "http://schemas.openxmlformats.org/drawingml/2006/chart"


def _rels(*entries):
    return f'<Relationships xmlns="{PKG}">' + "".join(
        f'<Relationship Id="{name}" Type="{REL}/{kind}" Target="{target}" {mode}/>'
        for name, kind, target, *extra in entries for mode in [extra[0] if extra else ""]) + "</Relationships>"


def _text(coordinate, value):
    return f'<c r="{coordinate}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def _number(coordinate, value, style=0):
    return f'<c r="{coordinate}" s="{style}"><v>{value}</v></c>'


def _sheet(rows, extra=""):
    data = "".join(f'<row r="{number}">{cells}</row>' for number, cells in rows)
    return f'<worksheet xmlns="{NS}" xmlns:r="{REL}"><sheetData>{data}</sheetData>{extra}</worksheet>'


def _anchor(row, col):
    return f"<xdr:from><xdr:col>{col}</xdr:col><xdr:row>{row}</xdr:row></xdr:from>"


def _xlsx(sheets, *, parts=None, defined_names="", workbook_rels=()):
    parts = dict(parts or {})
    sequence = []
    relations = [("style", "styles", "styles.xml"), *workbook_rels]
    for number, (name, data, state) in enumerate(sheets, 1):
        path = f"worksheets/sheet{number}.xml"
        identifier = f"sheet{number}"
        sequence.append(f'<sheet name="{escape(name)}" sheetId="{number}" state="{state}" r:id="{identifier}"/>')
        relations.append((identifier, "worksheet", path))
        parts["xl/" + path] = data
    parts.update({
        "[Content_Types].xml": '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        "</Types>",
        "_rels/.rels": _rels(("office", "officeDocument", "xl/workbook.xml")),
        "xl/_rels/workbook.xml.rels": _rels(*relations),
        "xl/workbook.xml": f'<workbook xmlns="{NS}" xmlns:r="{REL}"><sheets>{"".join(sequence)}</sheets>{defined_names}</workbook>',
        "xl/styles.xml": f'<styleSheet xmlns="{NS}"><numFmts>'
        '<numFmt numFmtId="164" formatCode="00000"/><numFmt numFmtId="165" formatCode="#,##0 &quot;VND&quot;"/>'
        '</numFmts><cellXfs><xf numFmtId="0"/><xf numFmtId="164"/><xf numFmtId="165"/><xf numFmtId="10"/>'
        '<xf numFmtId="14"/></cellXfs></styleSheet>',
    })
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, value in parts.items():
            archive.writestr(name, value)
    return stream.getvalue()


def _image_parts():
    buffer = BytesIO()
    with Image.new("RGB", (45, 30), "white") as picture:
        picture.save(buffer, "PNG")
    drawing = (
        f'<xdr:wsDr xmlns:xdr="{DRAW}" xmlns:a="{MAIN}" xmlns:r="{REL}" xmlns:c="{CHART}">'
        f'<xdr:oneCellAnchor>{_anchor(2, 5)}<xdr:pic><a:blip r:embed="image"/></xdr:pic></xdr:oneCellAnchor>'
        f'<xdr:oneCellAnchor>{_anchor(3, 5)}<xdr:graphicFrame><c:chart r:id="chart"/></xdr:graphicFrame></xdr:oneCellAnchor>'
        f'<xdr:oneCellAnchor>{_anchor(4, 5)}<xdr:sp><a:p><a:r><a:t>Văn bản trong hình vẽ</a:t>'
        "</a:r></a:p></xdr:sp></xdr:oneCellAnchor></xdr:wsDr>"
    )
    chart = (f'<c:chartSpace xmlns:c="{CHART}" xmlns:a="{MAIN}"><c:chart>'
             '<c:title><a:p><a:r><a:t>Biểu đồ phí test</a:t></a:r></a:p></c:title>'
             '<c:plotArea><c:barChart><c:ser><c:val><c:numRef><c:f>Biểu!C2:C3</c:f>'
             '<c:numCache><c:pt idx="0"><c:v>15</c:v></c:pt></c:numCache>'
             '</c:numRef></c:val></c:ser></c:barChart></c:plotArea></c:chart></c:chartSpace>')
    return {
        "xl/media/image.png": buffer.getvalue(),
        "xl/drawings/drawing.xml": drawing,
        "xl/drawings/_rels/drawing.xml.rels": _rels(("image", "image", "../media/image.png"),
                                                  ("chart", "chart", "../charts/chart.xml")),
        "xl/charts/chart.xml": chart,
        "xl/comments.xml": f'<comments xmlns="{NS}"><commentList><comment ref="A2">'
        "<text><r><t>Ghi chú của bảng</t></r></text></comment></commentList></comments>",
    }


def _vision(text="Nội dung OCR online"):
    result = {"elements": [{"type": "paragraph", "text": text, "bbox": [.1, .1, .9, .8]}],
              "review": {"verdict": "accepted"}, "uncertainties": []}
    return SimpleNamespace(model="vision-test", identity=lambda: "vision-test",
                           analyze_page_layout=Mock(side_effect=lambda *_: deepcopy(result)))


@pytest.fixture()
def options(tmp_path):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return {"tokenizer": tokenizer, "cache_db_path": tmp_path / "model-cache.sqlite",
            "use_decision": False, "use_llm": False, "target_chunk_tokens": 55, "max_chunk_tokens": 85,
            "min_chunk_tokens": 15, "table_rows_per_chunk": 2}


def _table_chunks(chunks, sheet=None):
    return [chunk for chunk in chunks if (chunk["table_data"] or {}).get("excel")
            and (sheet is None or chunk["table_data"]["excel"]["sheet"] == sheet)]


def test_layout_reconstructs_headers_sparse_regions_and_long_faq(tmp_path, options):
    question = "Làm sao để đóng thẻ?"
    answer = " ".join(f"phần-{idx}." for idx in range(90))
    fees = _sheet([
        (1, _text("A1", "Dịch vụ") + _text("B1", "Biểu phí")),
        (2, _text("B2", "Phí VND") + _text("C2", "Tỷ lệ")),
        (3, _number("A3", 7, 1) + _number("B3", 0, 2) + _number("C3", ".025", 3)
            + _text("G3", "Mã") + _text("H3", "Tên")),
        (4, '<c r="B4"><f>1+2</f><v>918273645</v></c>' + _number("C4", ".04", 3)
            + _number("G4", 12, 1) + _text("H4", "Bảng cạnh bên")),
        (1048576, _text("XFD1048576", "Cuối sheet")),
    ], '<mergeCells><mergeCell ref="A1:A2"/><mergeCell ref="B1:C1"/><mergeCell ref="A3:A4"/></mergeCells>')
    faq = _sheet([(1, _text("A1", "Câu hỏi") + _text("B1", "Câu trả lời")),
                  (2, _text("A2", question) + _text("B2", answer))])
    raw = _xlsx([("Biểu", fees, "visible"), ("FAQ", faq, "visible"),
                 ("Hidden", _sheet([(1, _text("A1", "Hidden row"))]), "hidden")])
    with UniversalDocumentChunker(asset_dir=tmp_path / "assets", **options) as chunker:
        chunks = chunker.chunk_document(raw)
        artifact = chunker.last_artifact
    tables = _table_chunks(chunks, "Biểu")
    fee = next(chunk["table_data"]["excel"] for chunk in tables if chunk["table_data"]["headers"][1:2] == ["Biểu phí / Phí VND"])
    assert fee["columns"] == {"A": "Dịch vụ", "B": "Biểu phí / Phí VND", "C": "Biểu phí / Tỷ lệ"}
    assert fee["rows"][0]["values"]["A"]["text"] == "00007"
    assert fee["rows"][0]["values"]["B"]["value"] == "0"
    assert fee["rows"][1]["values"]["A"]["origin"] == "A3"
    assert fee["rows"][1]["values"]["B"]["type"] == "skipped"
    assert artifact["quality"]["excel_formula_cells_skipped"] == 1
    assert not any("918273645" in json.dumps(chunk) for chunk in chunks)
    assert any("G3:H4" == chunk["table_data"]["excel"]["range"] for chunk in tables)
    assert any("XFD1048576" in chunk["business_metadata"]["citation"]["cell_ranges"] for chunk in chunks)
    parts = [chunk for chunk in chunks if chunk["business_metadata"].get("excel_row_part")]
    assert len(parts) > 2
    assert all(question in chunk["content"] for chunk in parts)
    assert all(chunk["business_metadata"]["citation"]["sheet_name"] == "FAQ" for chunk in parts)
    spans = [cell for chunk in parts for cell in chunk["business_metadata"]["excel_row_part"]["cells"]
             if cell["address"] == "B2"]
    assert "".join(answer[span["start"]:span["end"]] for span in spans) == answer
    assert not any("Hidden row" in chunk["content"] or "Hidden row" in chunk["parent_content"] for chunk in chunks)


def test_cloud_ocr_for_sheet_with_native_data_cache_and_object_evidence(tmp_path, options):
    parts = _image_parts()
    parts["xl/worksheets/_rels/sheet1.xml.rels"] = _rels(
        ("draw", "drawing", "../drawings/drawing.xml"), ("notes", "comments", "../comments.xml"),
        ("external", "hyperlink", "https://example.invalid/", 'TargetMode="External"'))
    raw = _xlsx([("Biểu", _sheet([(1, _text("A1", "Tên") + _text("B1", "Phí")),
                                 (2, _text("A2", "Bảo trì") + _number("B2", 20))],
                                '<drawing r:id="draw"/><hyperlinks><hyperlink ref="A2" r:id="external"/></hyperlinks>'),
                  "visible")], parts=parts)
    primary = _vision()
    review = _vision("OCR model kiểm tra")
    draft = {"elements": [{"type": "paragraph", "text": "Bản OCR chưa chắc chắn", "bbox": [.1, .1, .9, .8]}],
             "review": {"verdict": "unresolved"}, "uncertainties": [{"reason": "Unclear text"}]}
    primary.analyze_page_layout.side_effect = lambda *_: deepcopy(draft)
    with UniversalDocumentChunker(asset_dir=tmp_path / "assets", vision_client=primary,
                                  review_vision_client=review, **options) as chunker:
        first = chunker.chunk_xlsx(xlsx_bytes=raw)
        second = chunker.chunk_xlsx(xlsx_bytes=raw)
        assert first == second
        image = chunker.last_artifact["document"]["excel_ocr"][0]
    primary.analyze_page_layout.assert_called_once()
    review.analyze_page_layout.assert_called_once()
    assert image["cell_range"] == "F3" and image["review"]["model"] == "vision-test"
    text = "\n".join(chunk["content"] for chunk in first)
    assert all(value in text for value in ("Bảo trì", "Ghi chú", "Văn bản trong hình vẽ",
                                           "Biểu đồ phí test", "https://example.invalid/", "OCR model kiểm tra"))
    ocr_chunk = next(chunk for chunk in first if "OCR model kiểm tra" in chunk["content"])
    assert "F3" in ocr_chunk["business_metadata"]["citation"]["cell_ranges"]
    assert ocr_chunk["images"]


def test_vertical_faq_repeats_question_and_keeps_exact_answer_offsets(tmp_path, options):
    question = "Hỏi: Có thể đổi số điện thoại như thế nào?"
    answer = "Đáp: " + " ".join(f"Bước {step} xử lý yêu cầu." for step in range(35))
    raw = _xlsx([("FAQ", _sheet([(2, _text("C2", question)), (4, _text("C4", answer))]), "visible")])
    with UniversalDocumentChunker(asset_dir=tmp_path / "assets", **options) as chunker:
        chunks = chunker.chunk_xlsx(xlsx_bytes=raw)
    fragments = [chunk for chunk in chunks if chunk["business_metadata"].get("excel_faq_part")]
    assert len(fragments) > 2
    reconstructed = []
    for chunk in fragments:
        assert chunk["content"].startswith(question)
        meta = chunk["business_metadata"]
        assert meta["citation"]["cell_ranges"] == ["C2", "C4"]
        source, = meta["provenance"]["source_spans"]
        reconstructed.append(chunk["content"][source["chunk_start"]:source["chunk_end"]])
    assert "".join(reconstructed) == answer


def test_in_cell_image_reaches_cloud_ocr(tmp_path, options):
    parts = _image_parts()
    parts.update({
        "xl/metadata.xml": f'<metadata xmlns="{NS}"><metadataTypes><metadataType name="XLRICHVALUE"/>'
        '</metadataTypes><futureMetadata name="XLRICHVALUE"><bk><extLst><ext><rvb i="0"/></ext></extLst></bk>'
        '</futureMetadata><valueMetadata><bk><rc t="1" v="0"/></bk></valueMetadata></metadata>',
        "xl/richData/rdrichvalue.xml": '<rvData><rv s="0"><v>0</v></rv></rvData>',
        "xl/richData/richValueRel.xml": f'<richValueRels xmlns:r="{REL}"><rel r:id="pic"/></richValueRels>',
        "xl/richData/_rels/richValueRel.xml.rels": _rels(("pic", "image", "../media/image.png")),
    })
    raw = _xlsx([("Ảnh", _sheet([(4, '<c r="D4" t="e" vm="1"><v>#VALUE!</v></c>')]), "visible")],
                parts=parts, workbook_rels=(("meta", "sheetMetadata", "metadata.xml"),
                                            ("rich", "rdRichValue", "richData/rdrichvalue.xml"),
                                            ("binding", "richValueRel", "richData/richValueRel.xml")))
    client = _vision()
    with UniversalDocumentChunker(asset_dir=tmp_path / "assets", vision_client=client, **options) as chunker:
        chunks = chunker.chunk_document(raw)
    client.analyze_page_layout.assert_called_once()
    assert any(chunk["images"] and chunk["business_metadata"]["citation"]["cell_ranges"] == ["D4"] for chunk in chunks)


def test_excel_ocr_failure_is_reported_and_uncached_image_is_retried(tmp_path, options):
    incoming = tmp_path / "input"
    incoming.mkdir()
    parts = _image_parts()
    parts["xl/worksheets/_rels/sheet1.xml.rels"] = _rels(("draw", "drawing", "../drawings/drawing.xml"))
    (incoming / "sheet.xlsx").write_bytes(_xlsx([("Bảng", _sheet([
        (1, _text("A1", "Dữ liệu gốc"))], '<drawing r:id="draw"/>'), "visible")], parts=parts))
    client = _vision()
    accepted = client.analyze_page_layout.side_effect
    client.analyze_page_layout.side_effect = RuntimeError("Cloud unavailable")
    output = tmp_path / "publication"
    config = {**options, "vision_client": client, "retry_delays": ()}
    first = run_chunking.run_auto_chunking(incoming, output, **config)
    assert any("Dữ liệu gốc" in chunk["content"] for chunk in first)
    _, manifest = run_chunking.load_published_chunks(output, return_manifest=True)
    issue, = manifest["records"][0]["quality"]["excel_issues_detail"]
    assert (issue["sheet_name"], issue["cell_range"], issue["status"]) == ("Bảng", "F3", "ocr_unavailable")
    client.analyze_page_layout.side_effect = accepted
    second = run_chunking.run_auto_chunking(incoming, output, **config)
    assert any("Nội dung OCR online" in chunk["content"] for chunk in second)
    _, manifest = run_chunking.load_published_chunks(output, return_manifest=True)
    assert manifest["excel_ocr_unresolved"] == 0


def test_native_dates_named_ranges_and_pivot_regions(tmp_path, options):
    view = _sheet([
        (1, _text("A1", "Mã tài liệu: BP-77") + _text("D1", "Ngày hiệu lực") + _number("E1", 45658, 4)),
        (3, _text("A3", "Mã") + _text("B3", "Phí") + _text("E3", "Nhóm") + _text("F3", "Tổng")),
        (4, _text("A4", "01") + _number("B4", 30) + _text("E4", "Khách hàng") + _number("F4", 4)),
    ])
    parts = {"xl/worksheets/_rels/sheet1.xml.rels": _rels(("pivot", "pivotTable", "../pivotTables/pivot.xml")),
             "xl/pivotTables/pivot.xml": f'<pivotTableDefinition xmlns="{NS}" name="Khách">'
             '<location ref="E3:F4"/></pivotTableDefinition>'}
    names = "<definedNames><definedName name=\"Price\">'Biểu phí'!$A$3:$B$4</definedName></definedNames>"
    raw = _xlsx([("Biểu phí", view, "visible")], parts=parts, defined_names=names)
    with UniversalDocumentChunker(asset_dir=tmp_path / "assets", **options) as chunker:
        chunks = chunker.chunk_xlsx(xlsx_bytes=raw)
    tables = [chunk["table_data"]["excel"] for chunk in _table_chunks(chunks)]
    assert any(table["title"] == "Price" and table["range"] == "A3:B4" for table in tables)
    assert any(table["title"] == "Khách" and table["range"] == "E3:F4" for table in tables)
    revision = chunks[0]["business_metadata"]["excel_revision"]
    assert (revision["document_key"], revision["effective_from"]) == ("BP-77", "2025-01-01")


def test_excel_structure_model_is_grounded_and_cached(tmp_path, options, monkeypatch):
    raw = _xlsx([("Bảng", _sheet([(1, _text("A1", "Thuộc tính") + _text("B1", "Mô tả")),
                                 (2, _text("A2", "Tính năng") + _text("B2", "Chi tiết"))]), "visible")])
    endpoint = SimpleNamespace(model="text-fixture", url="https://example.invalid", api_key="mock")
    monkeypatch.setattr(excel_extractor, "model_primary", lambda _: endpoint)
    output = {"header_rows": 1, "question_column": None, "answer_columns": []}
    cloud = Mock(side_effect=lambda *args, validate, **kwargs: (validate(output), deepcopy(output))[1])
    monkeypatch.setattr(excel_extractor, "cloud_json", cloud)
    decision = Mock(return_value={"structure": {"choice": "ACCEPT"}})
    monkeypatch.setattr(excel_extractor, "ask_decisions", decision)
    config = {**options, "use_llm": True, "use_decision": True}
    with UniversalDocumentChunker(asset_dir=tmp_path / "assets", **config) as chunker:
        chunks = chunker.chunk_document(raw)
        chunker.chunk_document(raw)
    cloud.assert_called_once()
    assert decision.called
    assert _table_chunks(chunks)[0]["table_data"]["headers"] == ["Thuộc tính", "Mô tả"]


def _fee_workbook(amount):
    return _xlsx([("Biểu phí", _sheet([
        (1, _text("A1", "Mã") + _text("B1", "Phí VND") + _text("C1", "Tỷ lệ") + _text("D1", "Ngày")),
        (2, _text("A2", "A") + _number("B2", amount, 2) + _number("C2", ".025", 3) + _number("D2", 45658, 4)),
        (3, _text("A3", "B") + _number("B3", 80, 2) + _number("C3", ".04", 3)),
    ], '<tableParts><tablePart r:id="rate"/></tableParts>'), "visible")],
        parts={"xl/worksheets/_rels/sheet1.xml.rels": _rels(("rate", "table", "../tables/rate.xml")),
               "xl/tables/rate.xml": f'<table xmlns="{NS}" ref="A1:D3" name="Rates"/>'})


def test_mixed_publication_and_qdrant_preserve_versions_and_failed_source(tmp_path, options, monkeypatch):
    incoming = tmp_path / "input"
    incoming.mkdir()
    output = tmp_path / "publication"
    before = incoming / "fees-v1.xlsx"
    after = incoming / "fees-v2.xlsx"
    before.write_bytes(_fee_workbook(25))
    after.write_bytes(_fee_workbook(110))
    metadata = incoming / "revisions.json"
    metadata.write_text(json.dumps({before.name: {"document_key": "BU-001", "business_version": "01",
                                                "effective_from": "2025-01-01"},
                                    after.name: {"document_key": "BU-001", "business_version": "02",
                                                 "effective_from": "2025-07-01"}}), encoding="utf-8")
    word = Document()
    word.add_paragraph("Tài liệu Word")
    word.save(incoming / "guide.docx")
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((20, 50), "Fixture PDF")
        pdf.save(incoming / "guide.pdf")
    config = {**options, "use_llm": True, "vision_client": _vision(),
              "excel_metadata_path": metadata, "retry_delays": ()}
    monkeypatch.setattr(ModelEnricher, "enrich", lambda self, ir: ir)
    monkeypatch.setattr(SemanticBoundaryScorer, "annotate", lambda self, ir: ir)
    chunks = run_chunking.run_auto_chunking(incoming, output, **config)
    assert run_chunking.load_published_chunks(output) == chunks
    assert {chunk["source"] for chunk in chunks} == {before.name, after.name, "guide.docx", "guide.pdf"}
    scope = "excel-test"
    table = _table_chunks(chunks)[0]["table_data"]["excel"]
    with closing(QdrantBankingStore(dimension=2, storage_path=tmp_path / "qdrant", collection_name="excel_test")) as store:
        marked = mark_publication_chunks(normalize_searchable_chunks(chunks), scope)
        store.add_chunks([{**chunk, "vector": [.6, .8]} for chunk in marked])
        pipeline = SimpleNamespace(config=SimpleNamespace(KNOWLEDGE_PUBLICATION_SCOPE=scope), vector_store=store)
        tool = BankingRAGTool(pipeline_instance=pipeline)
        result = tool.execute("Biểu phí", excel={"table_id": table["id"], "as_of": "2025-06-30", "filters": [
            {"column": "Phí VND", "op": "lt", "value": 30, "unit": "VND"},
            {"column": "Tỷ lệ", "op": "eq", "value": "2.5%"},
            {"column": "D", "op": "eq", "value": "2025-01-01"}]})
        assert result["excel_match_count"] == 1
        assert result["citations"][0]["cell_ranges"] == ["A2:D2"]
        assert result["citations"][0]["sheet_name"] == "Biểu phí"
        crossed = tool.execute("Biểu phí", excel={"table_id": table["id"], "as_of": "2025-07-01", "filters": [
            {"column": "B", "op": "gt", "value": 100}, {"column": "C", "op": "gte", "value": 4}]})
        assert crossed["status"] == "no_results"
        before.write_bytes(b"corrupted archive")
        refreshed = run_chunking.run_auto_chunking(incoming, output, **config)
        _, manifest = run_chunking.load_published_chunks(output, return_manifest=True)
        kept = [item["rel_path"] for item in manifest["records"] if item["status"] == "FAILED"]
        assert kept == [before.name]
        refreshed = mark_publication_chunks(normalize_searchable_chunks(refreshed), scope)
        store.add_chunks([{**chunk, "vector": [.6, .8]} for chunk in refreshed])
        store.prune_chunks({item["chunk_id"] for item in refreshed}, scope=scope, preserve_sources=kept)
        result = tool.execute("Biểu phí", excel={"table_id": table["id"], "as_of": "2025-01-01"})
        assert result["sources"] == [before.name]
    revisions = json.loads(metadata.read_text(encoding="utf-8"))
    revisions[after.name]["effective_from"] = "2025-07-02"
    metadata.write_text(json.dumps(revisions), encoding="utf-8")
    changed = run_chunking.run_auto_chunking(incoming, output, **config)
    assert next(chunk["business_metadata"]["excel_revision"]["effective_from"]
                for chunk in changed if chunk["source"] == after.name) == "2025-07-02"

