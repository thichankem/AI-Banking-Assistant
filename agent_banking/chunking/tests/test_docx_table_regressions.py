from io import BytesIO
from types import SimpleNamespace

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from agent_banking.chunking.universal.extraction.docx.extractor import DocxExtractor
from agent_banking.chunking.universal.pipeline import UniversalDocumentChunker


def _bytes(document):
    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()


def _textbox(document, paragraph, text):
    drawing = OxmlElement("w:pict")
    textbox = OxmlElement("w:txbxContent")
    textbox.append(document.add_paragraph(text)._p)
    drawing.append(textbox)
    paragraph.add_run()._r.append(drawing)
    return textbox


def test_table_cells_share_visible_inline_extraction_with_body(tmp_path):
    document = Document()
    removed, heading, body = document.add_table(rows=3, cols=1).rows
    removed._tr.get_or_add_trPr().append(OxmlElement("w:del"))
    removed.cells[0].text = "Hàng đã xóa"
    removed.cells[0]._tc.append(OxmlElement("w:altChunk"))
    heading._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    heading.cells[0].text = "Điều kiện"

    paragraph = body.cells[0].paragraphs[0]
    paragraph.add_run("Xem ").bold = True
    hidden = paragraph.add_run("hạn mức không còn áp dụng ")
    hidden.font.hidden = True
    hidden.bold = False
    hidden._r.append(OxmlElement("w:object"))
    deleted = OxmlElement("w:del")
    deleted.append(paragraph.add_run("nội dung bị xóa ")._r)
    paragraph._p.append(deleted)
    inserted = OxmlElement("w:ins")
    link = OxmlElement("w:hyperlink")
    link.set(qn("w:anchor"), "Policy")
    link.append(paragraph.add_run("quy định")._r)
    link[0].get_or_add_rPr().append(OxmlElement("w:b"))
    inserted.append(link)
    paragraph._p.append(inserted)

    ir = DocxExtractor(tmp_path, use_decision=False).extract(_bytes(document), "visible.docx")

    block, = ir["blocks"]
    table = block["table_data"]
    assert table["grid"] == [["Điều kiện"], ["Xem quy định"]]
    assert table["header_grid"] == [["Điều kiện"]]
    assert table["rows"] == [["Xem quy định"]]
    assert table["cells"][1]["hyperlinks"] == [{"text": "quy định", "target": "#Policy"}]
    assert table["cells"][1]["all_text_bold"] is True
    assert [unit["text"] for unit in ir["units"]] == ["Điều kiện", "Xem quy định"]
    assert ir["warnings"] == []


def test_combined_horizontal_and_vertical_merges_preserve_grid_and_origin(tmp_path):
    document = Document()
    table = document.add_table(rows=3, cols=3)
    table.cell(0, 0).merge(table.cell(1, 1)).text = "Mức ưu đãi"
    table.cell(0, 2).text = "Theo kỳ"
    table.cell(1, 2).text = "7%"
    for index, value in enumerate(["Cá nhân", "Tổ chức", "10%"]):
        table.cell(2, index).text = value
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))

    ir = DocxExtractor(tmp_path, use_decision=False).extract(_bytes(document), "merged.docx")

    table = ir["blocks"][0]["table_data"]
    assert table["headers"] == ["Mức ưu đãi", "Mức ưu đãi", "Theo kỳ"]
    assert table["rows"] == [["Mức ưu đãi", "Mức ưu đãi", "7%"], ["Cá nhân", "Tổ chức", "10%"]]
    merged, _, continuation, *_ = table["cells"]
    assert (merged["row"], merged["column"], merged["row_span"], merged["col_span"]) == (0, 0, 2, 2)
    assert continuation["merged_into"] == merged["id"]
    assert sum(unit["text"] == "Mức ưu đãi" for unit in ir["units"]) == 1


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return SimpleNamespace(ids=list(range(len(text))), offsets=[(i, i + 1) for i in range(len(text))])

    def to_str(self):
        return "test character tokenizer"


def test_table_attachment_positions_and_provenance_reach_chunks(tmp_path):
    document = Document()
    cell = document.add_table(rows=1, cols=1).cell(0, 0)
    paragraph = cell.paragraphs[0]
    paragraph.add_run("  Trước ")
    _textbox(document, paragraph, "Giải thích thứ nhất")
    paragraph.add_run(" sau  ")
    cell.add_table(rows=1, cols=1).cell(0, 0).text = "Giá trị bảng con"
    tail = cell.paragraphs[-1]
    textbox = _textbox(document, tail, "Giải thích thứ hai")
    inside_textbox = document.add_table(rows=1, cols=1)
    inside_textbox.cell(0, 0).text = "Bảng trong hộp văn bản"
    textbox.append(inside_textbox._tbl)
    tail.add_run("Cuối")
    _textbox(document, tail, "")
    removed = cell.add_table(rows=1, cols=1).rows[0]
    removed._tr.get_or_add_trPr().append(OxmlElement("w:del"))
    removed.cells[0].text = "Bảng con đã xóa"

    with UniversalDocumentChunker(cache_db_path=tmp_path / "models.db", asset_dir=tmp_path,
                                  use_llm=False, use_decision=False, search_language="all",
                                  tokenizer=_CharacterTokenizer()) as chunker:
        chunks = chunker.chunk_docx(docx_bytes=_bytes(document), source_name="nested.docx")
        artifact = chunker.last_artifact

    outer, first, nested, second, boxed_table = artifact["document"]["blocks"]
    outer_cell, = outer["table_data"]["cells"]
    assert outer_cell["text"] == "Trước  sau  \nCuối"
    assert [first["text"], second["text"]] == ["Giải thích thứ nhất", "Giải thích thứ hai"]
    assert outer_cell["attachments"] == [
        {"type": "textbox", "text_offset": len("Trước "), "block_ids": [first["id"]]},
        {"type": "table", "text_offset": len("Trước  sau  "), "table_id": nested["id"]},
        {"type": "textbox", "text_offset": len("Trước  sau  \n"), "block_ids": [second["id"], boxed_table["id"]]},
    ]
    for block in (first, nested, second, boxed_table):
        assert block["provenance"]["parent_cell"] == outer_cell["id"]
    tables = {chunk["table_data"]["table_id"]: chunk["table_data"] for chunk in chunks if chunk["table_data"]}
    assert tables[outer["id"]]["cells"][0]["attachments"] == outer_cell["attachments"]
    assert tables[nested["id"]]["parent_cell"] == tables[boxed_table["id"]]["parent_cell"] == outer_cell["id"]
    assert artifact["quality"]["covered_units"] == artifact["quality"]["source_units"] == 6
