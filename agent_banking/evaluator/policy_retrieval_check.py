"""Offline integration checks for Temporal Retrieval and GraphRAG."""

from copy import deepcopy
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

from docx import Document
import pymupdf
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from agent_banking.agent.document_context import _format_retrieved_doc_context
from agent_banking.agent.react import BankingReActNodes
from agent_banking.chunking.run_chunking import load_published_chunks, run_auto_chunking
from agent_banking.chunking.universal.planning.schema import build_universal_chunk
from agent_banking.core.policy_dates import extract_policy_dates, policy_query_date
from agent_banking.retrieval.hybrid_retriever import HybridRetriever
from agent_banking.retrieval.policy_graph import replace_links
from agent_banking.tools.extensions.banking_query import parse_banking_query
from agent_banking.tools.tool_registry import BankingToolRegistry
from agent_banking.vector_db.qdrant_store import QdrantBankingStore


@pytest.fixture
def knowledge(tmp_path):
    store = QdrantBankingStore(collection_name="policy_checks", dimension=2, use_memory=True)
    config = SimpleNamespace(HYBRID_CANDIDATE_K=1, KNOWLEDGE_PUBLICATION_SCOPE="pilot",
                             KNOWLEDGE_GRAPH_PATH=tmp_path / "graph.sqlite3")
    retriever = HybridRetriever(vector_store=store, embedding_model=Mock(embed_query=Mock(return_value=[1.0, 0.01])))
    reranker = SimpleNamespace(rerank=lambda query, candidates, top_k: candidates[:top_k])
    pipeline = SimpleNamespace(config=config, vector_store=store, retriever=retriever,
                               reranker=reranker, parse_query=parse_banking_query)
    try:
        yield store, config, BankingToolRegistry(pipeline_instance=pipeline)
    finally:
        store.close()


def _chunk(source, content, policy, version, start=None, end=None, uploaded=None, scope="pilot", **extra):
    revision = {"policy_id": policy, "document_version": version, "effective_from": start,
                "effective_to": end, "uploaded_at": uploaded}
    chunk = build_universal_chunk(source=source, content=content, section_title="Điều 4",
                                  business_metadata={"policy_revision": revision, "publication_scopes": [scope],
                                                     "citation": {"page_numbers": [2]}})
    return {**chunk, "vector": [1.0, 0.01], **extra}


def _search(registry, **args):
    call = registry.execute_tool("search_banking_knowledge", {"top_k": 1, **args})
    assert call["status"] == "SUCCESS", call
    return call["data"]


def test_effective_dates_are_published_from_pdf_and_word(tmp_path):
    incoming = tmp_path / "input"
    incoming.mkdir()
    word = Document()
    word.add_heading("Điều khoản thi hành", 1)
    word.add_paragraph("Quy định này có hiệu lực từ ngày 01/07/2025 đến hết ngày 31/12/2025.")
    word.save(incoming / "word.docx")
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 60), "Policy")
        pdf.save(incoming / "policy.pdf")
    response = {"elements": [{"type": "paragraph", "text": "Ngày hiệu lực: 02/07/2025",
                              "bbox": [0.1, 0.2, 0.6, 0.4]}],
                "review": {"verdict": "accepted"}, "uncertainties": []}
    vision = Mock(model="fixture-vision", identity=Mock(return_value="fixture-vision"),
                  analyze_page_layout=Mock(side_effect=lambda *args: deepcopy(response)))
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    output = tmp_path / "output"
    options = {"use_llm": False, "use_decision": False, "require_ready": False, "retry_delays": (),
               "tokenizer": tokenizer, "cache_db_path": tmp_path / "cache.db", "vision_client": vision, "dpi": 72}
    run_auto_chunking(incoming, output, **options)
    _, manifest = load_published_chunks(output, require_ready=False, return_manifest=True)
    revisions = {record["rel_path"]: record["policy_revision"] for record in manifest["records"]}
    assert revisions["word.docx"]["effective_from"] == "2025-07-01"
    assert revisions["word.docx"]["effective_to"] == "2025-12-31"
    assert "Điều khoản thi hành" in revisions["word.docx"]["effective_date_reference"]
    assert revisions["policy.pdf"]["effective_from"] == "2025-07-02"
    assert "Trang 1" in revisions["policy.pdf"]["effective_date_reference"]
    run_auto_chunking(incoming, output, force=True, **options)
    _, repeated = load_published_chunks(output, require_ready=False, return_manifest=True)
    assert {record["rel_path"]: record["policy_revision"] for record in repeated["records"]} == revisions


def test_policy_selection_and_reference_only_through_registered_tool(knowledge):
    store, _, registry = knowledge
    store.add_chunks([
        _chunk("SLA v1.pdf", "Thời hạn tra soát là ba ngày.", "sla", "a", "2024-01-01", "2025-06-30"),
        _chunk("SLA v2.pdf", "Thời hạn tra soát là năm ngày.", "sla", "b", "2025-07-01"),
        _chunk("SLA imported.pdf", "Thời hạn tra soát là bảy ngày.", "sla", "c", uploaded="2025-09-05T01:00:00+00:00"),
    ])
    for day, expected in (("2025-06-30", "SLA v1.pdf"), ("2025-07-01", "SLA v2.pdf")):
        result = _search(registry, query="Thời hạn tra soát?", as_of=day)
        assert result["sources"] == [expected]
        assert result["citations"][0]["temporal"]["status"] == "effective"
    query = "Ngày 01/07/2023 thời hạn tra soát là bao lâu?"
    result = _search(registry, query=query)
    assert result["sources"] == ["SLA imported.pdf"]
    assert result["retrieval_quality"]["temporal"]["statuses"] == ["reference_only"]
    assert result["retrieval_quality"]["temporal"]["status"] == "unverified_as_of"
    assert "Chưa xác minh được bản áp dụng ngày 2023-07-01" in _format_retrieved_doc_context(result["documents"][0])
    node = object.__new__(BankingReActNodes)
    node.guardrails = SimpleNamespace(guard_retrieved_chunks=lambda docs: (docs, 0, []),
                                     verify_citation_grounding=lambda answer, docs: {"unsupported_citations": []})
    evidence = []
    observation = node._observe_retrieval(result, evidence)
    assert observation["retrieval_quality"]["temporal"]["status"] == "unverified_as_of"
    answer = {"answer": "Chưa xác minh được bản áp dụng. Bản mới nhất dùng để tham khảo [Nguồn 1]."}
    with pytest.raises(ValueError, match="reference-only answer"):
        node.validate_citations(answer, evidence)
    assert node.validate_citations({**answer, "insufficient_evidence": True}, evidence)[0]["temporal"]["status"] == "reference_only"
    assert _search(registry, query="Thời hạn tra soát?", as_of="2025-02-31")["status"] == "invalid_as_of"
    assert policy_query_date("hôm qua", today=date(2025, 7, 2))[0] == date(2025, 7, 1)


def test_existing_index_and_latest_upload_without_effective_date(knowledge):
    store, _, registry = knowledge
    legacy = _chunk("Legacy.docx", "Ngày hiệu lực: 06/07/2025. Quy định tra soát giao dịch.", "old", "legacy")
    legacy["business_metadata"].pop("policy_revision")
    store.add_chunks([
        legacy,
        _chunk("Undated v1.pdf", "Quy định về tra soát giao dịch bản đầu.", "undated", "d1", uploaded="2025-09-05T01:00:00+00:00"),
        _chunk("Undated v2.pdf", "Quy định về tra soát giao dịch bản cập nhật.", "undated", "d2", uploaded="2025-09-06T01:00:00+00:00"),
    ])
    legacy_reply = _search(registry, query="Quy trình tra soát?", source="Legacy.docx", as_of="2025-07-06")
    assert legacy_reply["citations"][0]["temporal"]["status"] == "effective"
    assert legacy_reply["documents"][0]["business_metadata"]["policy_revision"]["effective_date_origin"] == "document"
    latest = _search(registry, query="Quy định mới nhất về tra soát?", source="Undated")
    assert latest["sources"] == ["Undated v2.pdf"]
    assert latest["citations"][0]["temporal"]["status"] == "latest_available"
    older = _search(registry, query="Quy trình tra soát?", source="Legacy.docx", as_of="2025-07-05")
    assert "Chưa xác định được bản mới nhất" in _format_retrieved_doc_context(older["documents"][0])


def test_strict_retrieval_does_not_relax_edition_filters():
    store = Mock()
    store.search_vector.side_effect = [[], [(build_universal_chunk(content="wrong edition"), 1.0)]]
    store.search_bm25.return_value = []
    retriever = HybridRetriever(vector_store=store)
    assert retriever.retrieve(query="tra soát", filter_dict={"source": ["eligible.pdf"]}, strict_filters=True) == []
    assert store.search_vector.call_count == store.search_bm25.call_count == 1


def test_date_of_another_policy_and_negated_date_are_not_effective_dates():
    chunks = [{"source": "policy.docx", "content": "Thông tư số 12 có hiệu lực từ ngày 01/01/2025. "
               "Quy định này không có hiệu lực từ ngày 01/02/2025."}]
    assert extract_policy_dates(chunks) == {}


def test_graph_increases_required_source_coverage_and_stops_at_one_hop(knowledge):
    store, config, registry = knowledge
    quote = "Phụ lục sửa đổi thời hạn tra soát tại Điều 4 của SLA A."
    store.add_chunks([
        _chunk("SLA A.pdf", "SLA A quy định xử lý tra soát.", "base", "a", "2024-01-01"),
        _chunk("Phu luc B.pdf", quote, "amendment", "b", "2025-07-01"),
        _chunk("C.pdf", "Tài liệu C được phụ lục B dẫn chiếu.", "unrelated", "c", "2025-07-01"),
        _chunk("Restricted.pdf", "Tài liệu này thuộc phạm vi khác.", "restricted", "d", "2024-01-01", scope="other"),
    ])
    links = [
        {"from_source": "Phu luc B.pdf", "to_source": "SLA A.pdf", "relation": "amends",
         "evidence_source": "Phu luc B.pdf", "evidence_quote": quote, "evidence_page": 2},
        {"from_source": "Phu luc B.pdf", "to_source": "C.pdf", "relation": "references",
         "evidence_source": "C.pdf", "evidence_quote": "Tài liệu C được phụ lục B dẫn chiếu.", "evidence_page": 2},
        {"from_source": "SLA A.pdf", "to_source": "Restricted.pdf", "relation": "references",
         "evidence_source": "Restricted.pdf", "evidence_quote": "Tài liệu này thuộc phạm vi khác.", "evidence_page": 2},
    ]
    replace_links(config.KNOWLEDGE_GRAPH_PATH, links)
    args = {"query": "Thời hạn tra soát theo SLA A?", "source": "SLA A.pdf", "as_of": "2025-07-02"}
    assert _search(registry, expand_graph=False, **args)["sources"] == ["SLA A.pdf"]
    expanded = _search(registry, **args)
    assert expanded["sources"] == ["Phu luc B.pdf", "SLA A.pdf"]
    evidence = expanded["related_documents"][0]
    assert evidence["source"] == "Phu luc B.pdf"
    assert evidence["business_metadata"]["graph_link"]["evidence_page"] == 2
    assert evidence["business_metadata"]["temporal"]["status"] == "effective"
    assert len(expanded["citations"]) == 2
    assert _search(registry, **{**args, "as_of": "2025-06-30"})["sources"] == ["SLA A.pdf"]
    replace_links(config.KNOWLEDGE_GRAPH_PATH, [{**links[0], "evidence_page": 99}])
    assert _search(registry, **args)["sources"] == ["SLA A.pdf"]
