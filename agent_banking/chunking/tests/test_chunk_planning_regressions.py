import pytest

from agent_banking.chunking.tests.test_chunk_planning_edges import (
    CharacterTokenizer,
    document_ir,
    source_block,
    table_block,
)
from agent_banking.chunking.universal.enrichment.decisions.tables import resolve_table_structures_and_titles
from agent_banking.chunking.universal.enrichment.image_context import IMAGE_POLICY
from agent_banking.chunking.universal.planning.coverage import validate_coverage
from agent_banking.chunking.universal.planning.planner import ChunkPlanner


def planner(target=128, maximum=256):
    return ChunkPlanner(target, maximum, tokenizer=CharacterTokenizer(), use_decision=False)


def test_explicit_caption_follows_its_own_table():
    caption = source_block("c1", "caption", "Bảng 1. Biểu phí thẻ tín dụng")
    ir = document_ir([table_block("t1", [["Phí"], ["100.000đ"]], header_count=1), caption,
                      table_block("t2", [["Phí"], ["20.000đ"]], header_count=1)])
    ir["relations"] = [{"from": "c1", "to": "t1", "type": "caption_of"}]

    (first, second), _ = planner().plan(ir)

    assert first["content"].endswith(caption["text"])
    assert second["business_metadata"]["provenance"]["block_ids"] == ["t2"]
    assert caption["text"] not in second["content"]


@pytest.mark.parametrize("choice", ["PREV_BLOCK", "NEXT_BLOCK"])
def test_table_title_uses_its_own_caption_when_adjacent_captions_belong_elsewhere(choice):
    previous = source_block("prev", "caption", "Hình 1. Quy trình gửi hồ sơ")
    following = source_block("next", "caption", "Hình 2. Quy trình đối soát")
    own_caption = source_block("own", "caption", "Bảng 1. Hạn mức giao dịch")
    table = table_block("t1", [["Hạn mức"], ["100.000đ"]], header_count=1)
    table["caption"] = previous["text"]
    table["table_data"].update(title=previous["text"], title_source_block_id="prev")
    ir = document_ir([source_block("i1", "image"), previous, table, following,
                      source_block("i2", "image"), own_caption])
    ir["relations"] = [
        {"from": "prev", "to": "i1", "type": "caption_of"},
        {"from": "next", "to": "i2", "type": "caption_of"},
        {"from": "own", "to": "t1", "type": "caption_of"},
    ]

    def decision(role, state, questions, **kwargs):
        entry, = state["tables"]
        assert entry["prev_text"] == entry["next_text"] == ""
        assert {"PREV_BLOCK", "NEXT_BLOCK"}.isdisjoint(questions["t0_title"]["criteria"])
        return {"t0_title": {"choice": choice}}

    resolve_table_structures_and_titles(ir, decision)

    assert table["table_data"]["title"] == table["caption"] == own_caption["text"]
    assert table["table_data"]["title_source_block_id"] == "own"


@pytest.mark.parametrize("table_after_caption", [False, True])
def test_caption_for_excluded_image_stays_out_of_table_chunk(table_after_caption):
    image = source_block("i1", "image", image={"sha256": "x", "image_policy": IMAGE_POLICY,
                         "status": "skipped_no_text_support", "text_context_assessed": True})
    caption = source_block("c1", "caption", "Hình 1. Quy trình đăng ký thẻ")
    table = table_block("t1", [["Hạn mức"], ["100.000đ"]], header_count=1)
    ir = document_ir([image, caption, table] if table_after_caption else [table, image, caption])
    ir["relations"] = [{"from": "c1", "to": "i1", "type": "caption_of"}]

    chunks, _ = planner().plan(ir)

    image_caption, = [item for item in chunks if item["chunk_type"] == "text"]
    table_chunk, = [item for item in chunks if item["chunk_type"] == "table"]
    assert image_caption["content"] == caption["text"]
    assert caption["text"] not in table_chunk["content"]


def test_table_budget_measures_the_html_used_for_search():
    table = table_block("t1", [["Tên", "Mức", "Ghi chú", "Công thức"],
                               ["Phí một", "100.000đ", "VNĐ", "\n$$x=1$$\n"],
                               ["Phí hai", "20.000đ", "VNĐ", "\n$$x=2$$\n"]], header_count=1)
    chunk_planner = planner(target=160, maximum=220)

    chunks, artifact = chunk_planner.plan(document_ir([table]))

    assert [chunk["table_data"]["rows"] for chunk in chunks] == [[row] for row in table["table_data"]["rows"]]
    assert all(chunk["content"].startswith("<table>") for chunk in chunks)
    assert artifact["quality"]["source_preserved_over_budget"] == 0


@pytest.mark.parametrize("field, replacement", [
    ("headers", ["Sai"]), ("html", "<table></table>"), ("cells", []), ("header_cells", []),
    ("row_provenance", [{"page_index": 9}]), ("row_grid_offsets", [{"before": 1, "after": 0}]),
])
def test_coverage_rejects_changed_table_fragment_metadata(field, replacement):
    table = table_block("t1", [["Hạn mức"], ["100.000đ"]], header_count=1)
    table["table_data"]["row_provenance"] = [{"page_index": 1}, {"page_index": 2}]
    table["table_data"]["row_grid_offsets"] = [{"before": 0, "after": 0}] * 2
    chunk_planner = planner()
    (table_chunk,), _ = chunk_planner.plan(document_ir([table]))

    table_chunk["table_data"][field] = replacement

    with pytest.raises(RuntimeError, match=rf"Table fragment metadata changed: .*{field}"):
        validate_coverage(chunk_planner)


def test_table_citation_uses_original_page_of_each_row():
    table = table_block("t1", [["Hạn mức"], ["100.000đ"]], header_count=1)
    table["provenance"]["page_index"] = 3
    table["table_data"]["row_provenance"] = [
        {"page_index": 3, "bbox": [0.1, 0.1, 0.9, 0.3]},
        {"page_index": 4, "bbox": [0.1, 0.3, 0.9, 0.6]},
    ]
    ir = document_ir([table])
    ir["source"] = "folder/limits.pdf"
    chunk_planner = planner()
    (chunk,), _ = chunk_planner.plan(ir)
    assert chunk["business_metadata"]["citation"] == {"file_name": "limits.pdf", "page_numbers": [4, 5]}
    assert chunk["table_data"]["row_provenance"] == [{"page_index": 4}]
    validate_coverage(chunk_planner)
