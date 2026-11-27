"""Offline checks for retrieval, edition history, and cited source integrity."""

from datetime import date
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
from fastapi.testclient import TestClient
import pytest
from qdrant_client import QdrantClient

from agent_banking.agent.deep import validate_review
from agent_banking.agent.react import BankingReActNodes
from agent_banking.chunking.publication_io import atomic_json, file_hash
from agent_banking.core.attachment_privacy import prepare_vision_image
from agent_banking.core.document_revisions import archive_document
from agent_banking.core.policy_dates import extract_policy_dates, policy_query_date
from agent_banking.core.search_text import search_key
from agent_banking.evaluator.policy_retrieval_check import _chunk, _search, knowledge
from agent_banking.evaluator.rag_quality_gate import _anchor_hits, _expects_absent_evidence
from agent_banking.retrieval.hybrid_retriever import HybridRetriever, RetrievalUnavailable
from agent_banking.retrieval.policy_graph import replace_links
from agent_banking.retrieval.policy_versions import PolicyVersions
from agent_banking.tools.excel_rows import lookup_excel_rows, select_excel_version
from agent_banking.tools.extensions import legal_web_search as legal
from agent_banking.tools.rag_tool import _exact_excel_fee_evidence
from agent_banking.vector_db.qdrant_store import QdrantBankingStore, compute_corpus_table_paths
from agent_banking.vector_db.backfill_document_metadata import backfill

_FIXTURE_TABLE = "a" * 24


def test_dense_failure_uses_bm25_and_preserves_scope():
    chunk = _chunk("Policy.pdf", "Thời hạn tra soát", "policy", "v1")
    store = SimpleNamespace(search_vector=Mock(), search_bm25=Mock(return_value=[(chunk, 3.0)]))
    embedder = SimpleNamespace(embed_query=Mock(side_effect=RuntimeError("provider unavailable")))
    retriever = HybridRetriever(vector_store=store, embedding_model=embedder)
    scope = {"business_metadata.publication_scopes": "pilot"}
    found = retriever.retrieve("tra soát", filter_dict=scope, strict_filters=True)
    assert found[0]["chunk"]["source"] == "Policy.pdf"
    assert found[0]["retrieval_meta"]["dense_degraded"]
    assert store.search_bm25.call_args.kwargs["filter_dict"] == scope
    store.search_vector.assert_not_called()
    store.search_bm25.return_value = []
    with pytest.raises(RetrievalUnavailable):
        retriever.retrieve("tra soát", filter_dict=scope, strict_filters=True)


def test_persistent_index_refuses_lock_conflict(tmp_path, monkeypatch):
    open_index = Mock(side_effect=RuntimeError("locked"))
    monkeypatch.setattr("agent_banking.vector_db.qdrant_store.QdrantClient", open_index)
    with pytest.raises(RuntimeError, match="persistent knowledge index"):
        QdrantBankingStore(collection_name="locked_fixture", dimension=2, storage_path=tmp_path)
    open_index.assert_called_once_with(path=str(tmp_path))


def test_server_mode_refreshes_sparse_documents_after_a_published_change(tmp_path, monkeypatch):
    client = QdrantClient(":memory:")
    factory = Mock(return_value=client)
    monkeypatch.setattr("agent_banking.vector_db.qdrant_store.QdrantClient", factory)
    args = {"collection_name": "shared_test", "dimension": 2, "storage_path": tmp_path,
            "server_url": "http://qdrant.fixture", "server_api_key": "test"}
    writer = QdrantBankingStore(**args)
    reader = QdrantBankingStore(**args)
    try:
        assert reader.source_groups() == {}
        writer.add_chunks([_chunk("Updated.pdf", "Văn bản được xuất bản.", "policy", "v1")])
        assert set(reader.source_groups("pilot")) == {"Updated.pdf"}
        writer.prune_chunks(set(), scope="pilot")
        assert reader.source_groups("pilot") == {}
        assert factory.call_args.kwargs == {"url": "http://qdrant.fixture", "api_key": "test"}
    finally:
        writer.close()


def test_snapshot_keeps_an_old_indexed_policy_edition(knowledge):
    store, _, registry = knowledge
    old = _chunk("Limits.pdf", "Hạn mức là 3 triệu đồng.", "limits", "old", "2024-01-01", "2024-12-31")
    new = _chunk("Limits.pdf", "Hạn mức là 7 triệu đồng.", "limits", "new", "2025-01-01")
    for chunk in (old, new):
        chunk["business_metadata"]["document_version"] = chunk["business_metadata"]["policy_revision"]["document_version"]
        store.add_chunks([chunk])
    assert store.prune_chunks({new["chunk_id"]}, {"Limits.pdf"}, scope="pilot") == 0
    historic = _search(registry, query="Hạn mức theo chính sách?", as_of="2024-06-01")
    assert "3 triệu" in historic["documents"][0]["content"]
    assert historic["citations"][0]["temporal"]["status"] == "effective"
    current = _search(registry, query="Hạn mức theo chính sách?", as_of="2025-06-01")
    assert "7 triệu" in current["documents"][0]["content"]
    assert store.prune_chunks(set(), set(), scope="pilot") == 2


def test_metadata_backfill_rejects_a_new_ocr_publication_until_its_passages_are_indexed(tmp_path):
    source_root = tmp_path / "input"
    publication = tmp_path / "publication"
    index = tmp_path / "index"
    for directory in (source_root, publication, index):
        directory.mkdir()
    source = source_root / "Terms.pdf"
    source.write_bytes(b"fixture source")
    version = file_hash(source)
    artifact_path = publication / "artifact.json"
    chunks_path = publication / "chunks.json"
    atomic_json(artifact_path, {"document": {"source": source.name, "document_version": version}, "quality": {}})
    atomic_json(chunks_path, [{"chunk_id": "shared-id", "content": "Verified OCR passage"}])
    atomic_json(publication / "chunking_manifest.json", {"records": [{
        "status": "SUCCESS", "rel_path": source.name, "input_sha256": version,
        "artifact_file": artifact_path.name, "artifact_sha256": file_hash(artifact_path),
        "output_file": chunks_path.name, "output_sha256": file_hash(chunks_path),
    }]})
    atomic_json(index / "knowledge_inventory.json", {"publications": [{
        "publication": str(publication), "sources": [source.name],
    }], "source_hashes": {source.name: version}})
    store = SimpleNamespace(storage_path=index, update_document_metadata=Mock(), source_groups=lambda: {
        source.name: [{"chunk_id": "shared-id", "content": "Older OCR passage",
                      "business_metadata": {"document_version": version}}],
    })
    with pytest.raises(ValueError, match="Reindex this publication"):
        backfill(SimpleNamespace(CHUNKING_INPUT_DIR=source_root), store, apply=True)
    store.update_document_metadata.assert_not_called()
    assert not (index / "revisions").exists()


def test_event_dates_and_partial_dates_do_not_select_an_arbitrary_policy_day():
    today = date(2026, 11, 26)
    assert policy_query_date("Giao dịch ngày 20/11/2026, hiện giờ tra soát thế nào?", today=today)[0] == today
    assert policy_query_date("Giao dịch ngày 20/11/2026 của tôi bị treo", today=today)[0] is None
    assert policy_query_date("Chuyển tiền thực hiện tại quầy tính phí thế nào?", today=today)[0] is None
    assert policy_query_date("Hiện tại phí chuyển tiền tại quầy là bao nhiêu?", today=today)[0] == today
    with pytest.raises(ValueError, match="imprecise_as_of"):
        policy_query_date("Biểu phí áp dụng vào tháng 10/2025 là gì?", today=today)
    with pytest.raises(ValueError, match="ambiguous_as_of"):
        policy_query_date("So sánh phí ngày 01/01/2025 và ngày 01/02/2025", today=today)
    result = extract_policy_dates([{"source": "Policy.pdf", "content": (
        "Quy định này có hiệu lực thi hành kể từ ngày 01/10/2025. Ưu đãi mở thẻ có hiệu lực từ ngày 01/12/2025."
    )}])
    assert result["effective_from"] == "2025-10-01"


def test_related_graph_evidence_cannot_bypass_exact_or_wrong_date(knowledge):
    store, config, registry = knowledge
    original = _chunk("SLA A.pdf", "Mã DVC901 quy định thời hạn tra soát.", "sla", "a", "2024-01-01")
    quote = "Quy định này thay thế tài liệu SLA A về thời hạn tra soát."
    replacement = _chunk("Replacement.pdf", quote, "replacement", "b", "2025-07-01")
    store.add_chunks([original, replacement])
    replace_links(config.KNOWLEDGE_GRAPH_PATH, [{
        "from_source": "Replacement.pdf", "to_source": "SLA A.pdf", "relation": "replaces",
        "evidence_source": "Replacement.pdf", "evidence_quote": quote, "evidence_page": 2,
    }])
    early = _search(registry, query="Mã DVC901", exact="DVC901", source="SLA A.pdf", as_of="2025-06-30")
    assert early["sources"] == ["SLA A.pdf"]
    assert early["citations"][0]["temporal"]["status"] == "effective"
    later = _search(registry, query="Mã DVC901", exact="DVC901", source="SLA A.pdf", as_of="2025-07-01")
    assert later["sources"] == ["SLA A.pdf"]
    assert later["citations"][0]["temporal"]["status"] == "reference_only"
    search = _search(registry, query="Thời hạn tra soát", as_of="2025-07-01")
    assert search["sources"] == ["Replacement.pdf"]


def _workbook_chunk(source="Fees.xlsx", date_start="2025-01-01"):
    def row(number, kind):
        return {"row": number, "range": f"A{number}:B{number}", "values": {
            "A": {"type": "text", "text": kind, "value": kind, "origin": f"A{number}"},
        }}
    chunk = _chunk(source, "Danh sách ATM, CDM và chi nhánh.", "workbook", "one")
    chunk["business_metadata"].pop("policy_revision")
    chunk["business_metadata"].update(document_version="one", excel_table_id=_FIXTURE_TABLE,
                                      excel_revision={"document_key": "fees", "effective_from": date_start,
                                                      "effective_date_status": "known"})
    chunk["table_data"] = {"excel": {"sheet": "Danh sách", "columns": {"A": "Loại"},
                                   "rows": [row(2, "ATM"), row(3, "CDM"), row(4, "Chi nhánh")]}}
    return chunk


def test_excel_discovery_rows_and_version_selection(knowledge):
    store, _, registry = knowledge
    chunk = _workbook_chunk()
    store.add_chunks([chunk])
    discovered = _search(registry, query="Danh sách ATM", source="Fees.xlsx", as_of="2025-08-01")
    assert discovered["documents"][0]["business_metadata"]["excel_table_id"] == _FIXTURE_TABLE
    result = _search(registry, query="Danh sách ATM", as_of="2025-08-01", excel={
        "table_id": _FIXTURE_TABLE, "group_by": ["Loại"],
        "filters": [{"column": "Loại", "op": "in", "value": ["ATM", "CDM"]}],
    })
    assert result["excel_match_count"] == 2
    assert result["citations"][-1]["cell_ranges"] == ["A2:B3"]
    assert result["citations"][-1]["temporal"]["status"] == "effective"
    assert lookup_excel_rows([chunk], {"table_id": _FIXTURE_TABLE, "group_by": ["missing"]})["status"] == "excel_column_not_found"
    versions = {("Fees.xlsx", "one", "2025-01-01", "1", "known"): [],
                ("Fees.xlsx", "two", "2025-02-01", "2", "conflicting"): []}
    assert select_excel_version(versions, "2025-03-01") == (None, "excel_version_conflict")


def test_fee_lookup_resolves_the_cited_row_from_an_already_retrieved_workbook():
    fee = _workbook_chunk()
    table = fee["table_data"]["excel"]
    label = "Đóng Tài khoản\n(Miễn phí đối với cán bộ nhân viên)"
    table["columns"] = {"A": "Khoản mục", "B": "Mức phí"}
    table["rows"][2]["values"] = {
        "A": {"type": "text", "text": label, "value": label, "origin": "A4"},
        "B": {"type": "currency", "text": "10.000đ", "value": "10000", "origin": "B4"},
    }
    query = f" {search_key('Đóng tài khoản thì phí là bao nhiêu?')} "
    day = date(2025, 8, 1)
    policy = PolicyVersions([fee], "pilot", day, day)
    [found] = _exact_excel_fee_evidence([fee], query, policy, policy.labels)
    assert found["chunk"]["business_metadata"]["citation"]["cell_ranges"] == ["A4:B4", "A4", "B4"]
    assert "10.000đ" in found["chunk"]["content"] and "cán bộ nhân viên" in found["chunk"]["content"]
    outside = PolicyVersions([fee], "restricted", day, day)
    assert _exact_excel_fee_evidence([fee], query, outside, outside.labels) == []
    transfer = f" {search_key('Phí chuyển tiền tài khoản cán bộ nhân viên là bao nhiêu?')} "
    assert _exact_excel_fee_evidence([fee], transfer, policy, policy.labels) == []
    earlier = PolicyVersions([fee], "pilot", date(2024, 12, 1), day)
    assert _exact_excel_fee_evidence([fee], query, earlier, earlier.labels) == []


def test_fee_comparison_preserves_two_distinct_priced_transfer_rows():
    fee = _workbook_chunk()
    table = fee["table_data"]["excel"]
    table["columns"] = {"A": "Khoản mục", "B": "Mức phí"}
    table["rows"] = [
        {"row": number, "range": f"A{number}:B{number}", "values": {
            "A": {"type": "text", "text": service, "value": service},
            "B": {"type": "percent", "text": rate, "value": rate},
        }}
        for number, service, rate in (
            (2, "Chuyển tiền nhanh liên ngân hàng", "0,055%"),
            (3, "Chuyển tiền qua kênh thanh toán điện tử liên ngân hàng", "0,05%"),
        )
    ]
    query = "So sánh mức phí chuyển khoản liên ngân hàng nhanh 24/7 và chuyển khoản CITAD thực hiện tại quầy"
    requested, today = policy_query_date(query, today=date(2025, 8, 1))
    policy = PolicyVersions([fee], "pilot", requested, today)
    matched = _exact_excel_fee_evidence([fee], f" {search_key(query)} ", policy, policy.labels)
    assert {item["chunk"]["business_metadata"]["citation"]["cell_ranges"][0] for item in matched} == {"A2:B2", "A3:B3"}


def test_merged_excel_headers_provide_context_only_within_the_same_edition():
    def fragment(chunk_id, edition, row_start, row):
        chunk = _workbook_chunk()
        chunk["chunk_id"] = chunk_id
        chunk["business_metadata"]["document_version"] = edition
        chunk["table_data"].update(row_start=row_start, rows=[row])
        return chunk

    name = "Chuyển khoản cho người hưởng cùng hệ thống"
    header = fragment("section", "one", 1, ["4.1", name, name, name])
    first = fragment("row-one", "one", 2, ["-", "Cùng chủ tài khoản", "CT01", "Miễn phí"])
    second = fragment("row-two", "two", 2, ["-", "Khác chủ tài khoản", "CT02", "10.000đ"])
    paths, _ = compute_corpus_table_paths([header, second, first])
    assert name in paths["row-one"]
    assert "row-two" not in paths


def test_explicit_absence_reference_is_reported_separately_from_quote_matches():
    anchor = {"source_path": "Terms.pdf", "quote": "Không có trong tài liệu hiện hành",
              "section": "Không có trong tài liệu hiện hành (Terms)"}
    assert _expects_absent_evidence(anchor)
    assert _anchor_hits({"evidence": [anchor]}, []) == []
    assert _anchor_hits({"evidence": [{**anchor, "quote": "Phí được công bố theo phụ lục"}]}, []) == [False]


def test_web_providers_contribute_complementary_articles(monkeypatch):
    monkeypatch.setenv("LEGAL_WEB_SEARCH_PRIMARY", "brave")

    def provider(name, query):
        article = {"brave": "dieu 1", "tavily": "dieu 2"}[name]
        return {"status": "ok", "provider": name, "documents": [
            {"content": "Nội dung của " + article, "business_metadata": {"legal_article": article}}]}

    monkeypatch.setattr(legal, "search_provider", provider)
    result = legal.search_legal_web("Điều 1 và Điều 2")
    assert result["status"] == "ok"
    assert {doc["business_metadata"]["legal_article"] for doc in result["documents"]} == {"dieu 1", "dieu 2"}


def test_dated_reference_and_wrong_financial_number_are_rejected():
    node = object.__new__(BankingReActNodes)
    node.guardrails = SimpleNamespace(verify_citation_grounding=Mock(return_value={"unsupported_citations": []}))
    evidence = [{"source": "Policy.pdf", "chunk": {"business_metadata": {"temporal": {
        "requested_date": True, "status": "reference_only", "as_of": "2025-01-01",
    }}}}, {"source": "web", "chunk": {}}]
    with pytest.raises(ValueError, match="reference-only"):
        node.validate_citations({"answer": "Hạn mức áp dụng [Nguồn 1] [Nguồn 2]."}, evidence)
    draft = "Phí là 250.000 VND [Nguồn 1]."
    verdict = {"verdict": "approve", "coverage": [{"status": "answered"}],
               "claims": [{"draft_line": 1, "source": "[Nguồn 1]", "quote": "100.000 VND"}]}
    with pytest.raises(ValueError, match="Unsupported financial figure"):
        validate_review(verdict, draft, [{"label": "[Nguồn 1]"}],
                        [{"label": "[Nguồn 1]", "text": "Phí là 100.000 VND."}], {})


def test_archive_rejects_modified_source_and_preflight_masks_detected_fields(tmp_path, monkeypatch):
    from hashlib import sha256

    source = tmp_path / "Terms.docx"
    source.write_bytes(b"old source")
    version = sha256(b"old source").hexdigest()
    archived = archive_document(tmp_path / "revisions", source, version)
    source.write_bytes(b"updated source")
    assert archived.read_bytes() == b"old source"
    with pytest.raises(ValueError, match="Source changed"):
        archive_document(tmp_path / "other", source, version)
    image = BytesIO()
    Image.new("RGB", (24, 18), "white").save(image, format="PNG")
    monkeypatch.setattr("pytesseract.image_to_data", lambda *args, **kwargs: {
        "text": ["Mã", "000000"], "block_num": [1, 1], "par_num": [1, 1], "line_num": [1, 1],
        "left": [1, 10], "top": [2, 2], "width": [6, 12], "height": [7, 7],
    })
    output = prepare_vision_image(image.getvalue(), lambda text: text.replace("000000", "[ẩn]"),
                                  ocr_configure=lambda: "fixture")
    with Image.open(BytesIO(output)) as screened:
        assert screened.getpixel((2, 4)) == (255, 255, 255)
        assert screened.getpixel((12, 4)) == (0, 0, 0)


def test_source_endpoint_checks_scope_identity_and_serves_the_cited_version(tmp_path, monkeypatch):
    from hashlib import sha256
    import importlib.util

    from agent_banking import agent_pipeline, node_pipeline, pipeline

    archived_bytes = b"version cited in the answer"
    digest = sha256(archived_bytes).hexdigest()
    original = tmp_path / "Limits.pdf"
    original.write_bytes(archived_bytes)
    store = QdrantBankingStore(collection_name="view_history", dimension=2, use_memory=True, storage_path=tmp_path)
    try:
        archive_document(tmp_path / "revisions", original, digest)
        original.write_bytes(b"later revision in the source directory")
        cited = _chunk("Limits.pdf", "Thời hạn tra soát", "policy", digest)
        cited["business_metadata"]["document_version"] = digest
        store.add_chunks([cited])
        monkeypatch.setattr(pipeline, "BankingRAGPipeline", lambda **kwargs: SimpleNamespace(use_memory=True, vector_store=store))
        monkeypatch.setattr(agent_pipeline, "BankingAgentPipeline", lambda **kwargs: None)
        monkeypatch.setattr(node_pipeline, "NodePipeline", lambda: None)
        server_path = Path(__file__).resolve().parents[2] / "main.py"
        spec = importlib.util.spec_from_file_location("banking_rag_history_fixture", server_path)
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        monkeypatch.setattr(server.config, "KNOWLEDGE_PUBLICATION_SCOPE", "pilot")
        monkeypatch.setattr(server, "_allowed_pdf_dirs", lambda: [tmp_path])
        monkeypatch.setenv("ABANK_KNOWLEDGE_ACCESS_TOKEN", "fixture-access")
        with TestClient(server.app) as client:
            params = {"source": "Limits.pdf", "version": digest}
            assert client.get("/api/documents/file", params=params).status_code == 401
            headers = {"Authorization": "Bearer fixture-access"}
            reply = client.get("/api/documents/file", params=params, headers=headers)
            assert reply.status_code == 200
            assert reply.content == archived_bytes
            assert client.get("/api/documents/file", params={"source": "Other.pdf", "version": digest},
                              headers=headers).status_code == 404
    finally:
        store.close()
