from types import SimpleNamespace

import pytest
from docx import Document

from agent_banking.chunking import run_chunking
from agent_banking.chunking.universal.enrichment.enricher import ModelEnricher
from agent_banking.chunking.universal.planning.semantic_boundaries import SemanticBoundaryScorer
from agent_banking.embedding import run_embedding
from agent_banking.vector_db import run_vector_db
from agent_banking.vector_db.qdrant_store import QdrantBankingStore, mark_publication_chunks


def _chunk(identifier, source, context=""):
    return {
        "chunk_id": identifier,
        "source": source,
        "chunk_type": "text",
        "content": "Điều kiện giao dịch.",
        "business_metadata": {"schema_version": "2.0", "retrieval": {"context": context}},
    }


def _vectorize(chunks):
    return [{**chunk, "vector": [0.6, 0.8]} for chunk in chunks]


@pytest.fixture()
def local_index(tmp_path, monkeypatch):
    config = SimpleNamespace(
        EMBEDDING_DIM=2, EMBEDDING_MODEL_1024="stub", EMBEDDING_URL="", STORAGE_DIR=tmp_path,
        QDRANT_STORAGE_PATH=tmp_path / "qdrant", QDRANT_COLLECTION="audit",
    )
    for module in (run_vector_db, run_embedding):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(module, "Config", lambda: config)
    return config


def test_standalone_snapshot_rejects_stale_metadata(tmp_path, local_index, monkeypatch):
    output = tmp_path / "publication"
    embedded_file = tmp_path / "embedded_chunks.json"
    original = _chunk("same-id", "limits.docx", "Hạn mức cũ")
    changed = _chunk("same-id", "limits.docx", "Hạn mức mới")
    manifest = {"records": [{"rel_path": "limits.docx", "status": "SUCCESS"}]}
    monkeypatch.setattr(run_chunking, "load_published_chunks", lambda *a, **k: ([changed], manifest))
    run_chunking.atomic_json(embedded_file, _vectorize(run_embedding.normalize_searchable_chunks([original])))
    with pytest.raises(ValueError, match="rerun embedding"):
        run_vector_db.run_auto_indexing(
            embedded_file, tmp_path / "index_metadata", snapshot=True, publication_dir=output,
        )
    assert not local_index.QDRANT_STORAGE_PATH.exists()


def test_scoped_index_preserves_failed_then_prunes_deleted_and_empty(tmp_path, local_index, monkeypatch):
    class StubEmbedder:
        dimension = 2

        def __init__(self, **options):
            pass

        def format_virtual_passage(self, chunk):
            return chunk["content"]

        def _get_hash(self, passage):
            return passage

        def _get_cached_vector(self, passage):
            return None

        def embed_chunks(self, chunks, batch_size):
            return _vectorize(chunks)

    monkeypatch.setattr(run_embedding, "BankingEmbedder", StubEmbedder)
    monkeypatch.setattr(ModelEnricher, "enrich", lambda self, ir: ir)
    monkeypatch.setattr(SemanticBoundaryScorer, "annotate", lambda self, ir: ir)
    incoming = tmp_path / "input"
    incoming.mkdir()
    output = tmp_path / "publication"
    embedding_output = tmp_path / "embedding_output"

    def write_document(path, text):
        document = Document()
        document.add_paragraph(text)
        document.save(path)

    def index_current():
        run_chunking.run_auto_chunking(
            incoming, output, use_decision=False, cache_db_path=tmp_path / "cache.db", retry_delays=(),
        )
        run_embedding.run_auto_embedding(output, embedding_output)
        return run_vector_db.run_auto_indexing(
            embedding_output / "embedded_chunks.json", tmp_path / "index_metadata",
            snapshot=True, publication_dir=output,
        )

    first, second = incoming / "first.docx", incoming / "second.docx"
    write_document(first, "Hạn mức chuyển khoản là 10 triệu đồng.")
    write_document(second, "Thời hạn trả nợ là ba tháng.")
    assert index_current()["total_points"] == 2
    write_document(first, "Hạn mức chuyển khoản là 15 triệu đồng.")
    second.write_bytes(b"invalid DOCX")
    stats = index_current()
    assert (stats["removed_count"], stats["total_points"]) == (1, 2)
    second.unlink()
    stats = index_current()
    assert (stats["removed_count"], stats["total_points"]) == (1, 1)
    first.unlink()
    stats = index_current()
    assert (stats["removed_count"], stats["total_points"], stats["bm25_size"]) == (1, 0, 0)


def test_shared_chunk_ownership_survives_removal_and_restart(tmp_path):
    def open_store():
        return QdrantBankingStore(collection_name="audit", dimension=2, storage_path=tmp_path)

    store = open_store()
    try:
        shared = _vectorize([_chunk("shared", "shared.docx")])
        store.add_chunks(mark_publication_chunks(shared, "first"))
        store.add_chunks(mark_publication_chunks(shared, "second"))
        assert store.chunk_ids(scope="first") == {"shared"}
        assert store.chunk_ids(scope="second") == {"shared"}
        assert store.prune_chunks(set(), scope="second") == 0
    finally:
        store.close()

    store = open_store()
    try:
        assert store.chunk_ids(scope="first") == {"shared"}
        assert store.chunk_ids(scope="second") == set()
        assert store.prune_chunks(set(), scope="first") == 1
        assert store.chunk_ids() == set()
    finally:
        store.close()


def test_direct_ingest_relinks_around_attachments_and_preserves_other_sources(tmp_path, monkeypatch):
    from agent_banking import pipeline

    first = _chunk("first", "source.docx")
    attachment = _chunk("attachment", "source.docx")
    attachment["business_metadata"]["retrieval"]["role"] = "attachment_only"
    last = _chunk("last", "source.docx")
    store = QdrantBankingStore(collection_name="audit", dimension=2, storage_path=tmp_path / "qdrant")
    try:
        store.add_chunks(_vectorize([_chunk("other", "other.docx")]))
        received = []

        def ingest(chunks, batch_size, *, return_index_keys=False):
            received.extend(chunks)
            indexed = _vectorize(chunks)
            store.add_chunks(indexed)
            if return_index_keys:
                return len(chunks), {store.index_key(chunk) for chunk in indexed}
            return len(chunks)

        instance = SimpleNamespace(vector_store=store, ingest_chunks=ingest)
        manifest = {"records": [{"rel_path": "source.docx", "status": "SUCCESS"}]}
        monkeypatch.setattr(pipeline, "load_published_chunks", lambda *a, **k: ([first, attachment, last], manifest))
        ingest_published = pipeline.BankingRAGPipeline.ingest_published_chunks
        assert ingest_published(instance, tmp_path / "publication") == 2
        assert received[0]["next_chunk_id"] == "last"
        assert received[1]["prev_chunk_id"] == "first"
        assert store.chunk_ids() == {"first", "last", "other"}
        monkeypatch.setattr(pipeline, "load_published_chunks", lambda *a, **k: ([], {"records": []}))
        assert ingest_published(instance, tmp_path / "publication") == 0
        assert store.chunk_ids() == {"other"}
    finally:
        store.close()
