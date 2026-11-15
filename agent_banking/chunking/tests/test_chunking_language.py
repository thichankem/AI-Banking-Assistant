from io import BytesIO
import urllib.request

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from agent_banking.chunking.universal.pipeline import LayoutAwareChunker
from agent_banking.chunking.universal.text.language import classify_text_blocks, vietnamese_projection


VIETNAMESE = "Khách hàng được miễn phí chuyển tiền nội bộ."
ENGLISH = "The customer must pay a service fee before using the card."
ENGLISH_2 = "Reimbursement may take up to five business days before funds are received."


@pytest.fixture
def chunker(tmp_path, monkeypatch):
    def block_network(*args, **kwargs):
        raise AssertionError("Unexpected network call in language regression")

    monkeypatch.setattr(urllib.request, "urlopen", block_network)
    with LayoutAwareChunker(
        asset_dir=tmp_path / "assets",
        cache_db_path=tmp_path / "cache.db",
        use_llm=False,
        use_decision=False,
    ) as chunker:
        yield chunker


def document_bytes(document):
    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()


def test_stale_word_proofing_language_keeps_vietnamese(chunker):
    document = Document()
    run = document.add_paragraph(VIETNAMESE).runs[0]
    language = OxmlElement("w:lang")
    language.set(qn("w:val"), "en-US")
    run._element.get_or_add_rPr().append(language)

    chunk, = chunker.chunk_document(document_bytes(document), source_name="language.docx")

    assert chunk["content"] == VIETNAMESE
    assert chunk["business_metadata"]["retrieval"]["role"] == "child"


def test_inline_foreign_sentences_keep_exact_source_spans(chunker):
    document = Document()
    document.add_heading("Hạn mức giao dịch! " + ENGLISH, 1)
    document.add_paragraph(VIETNAMESE + " " + ENGLISH_2)
    document.add_paragraph("1. " + ENGLISH + " " + VIETNAMESE)

    chunk, = chunker.chunk_document(document_bytes(document), source_name="language.docx")

    assert chunk["section_title"] == "Hạn mức giao dịch!"
    for key in ("content", "parent_content", "breadcrumb"):
        assert ENGLISH not in chunk[key]
        assert ENGLISH_2 not in chunk[key]
    elements = chunk["business_metadata"]["elements"]
    blocks = {block["id"]: block["text"] for block in chunker.last_artifact["document"]["blocks"]}
    assert [element["text"].strip() for element in elements] == [ENGLISH, ENGLISH_2, "1. " + ENGLISH]
    for element in elements:
        start, end = element["source_span"]
        assert blocks[element["block_id"]][start:end] == element["text"]


def test_ambiguous_vietnamese_and_math_stay_searchable():
    block = {"type": "paragraph", "text": "Vui long xem them. " + ENGLISH + " $x=3$"}

    classify_text_blocks([block])

    assert vietnamese_projection(block).split() == ["Vui", "long", "xem", "them.", "$x=3$"]


@pytest.mark.parametrize("with_header", [False, True])
@pytest.mark.parametrize("with_tail", [False, True])
def test_table_fragment_role_depends_on_its_own_content(chunker, with_header, with_tail):
    document = Document()
    header_count = int(with_header)
    table = document.add_table(rows=5 + header_count, cols=1)
    if with_header:
        table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        table.rows[0].cells[0].text = "Nội dung thẻ"
    for index, row in enumerate(table.rows[header_count:]):
        row.cells[0].text = VIETNAMESE if index == 0 else ENGLISH
    if with_tail:
        document.add_paragraph("Kiểm tra lại giao dịch.")

    first, last = chunker.chunk_document(document_bytes(document), source_name="table.docx")

    assert first["business_metadata"]["retrieval"]["role"] == "child"
    assert last["business_metadata"]["retrieval"]["role"] == ("child" if with_tail else "attachment_only")
    assert last["table_data"]["rows"] == [[ENGLISH]]


def test_numeric_cell_keeps_its_fragment_searchable(chunker):
    document = Document()
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = ENGLISH
    table.rows[0].cells[1].text = "5.000.000"

    chunk, = chunker.chunk_document(document_bytes(document), source_name="table.docx")

    retrieval = chunk["business_metadata"]["retrieval"]
    assert retrieval["role"] == "child"
    assert "5.000.000" in retrieval["embedding_content"]
