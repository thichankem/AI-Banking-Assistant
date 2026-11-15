from types import SimpleNamespace

import pytest
from docx import Document

from agent_banking.chunking.run_chunking import atomic_json, run_auto_chunking
from agent_banking.chunking.universal.enrichment.enricher import ModelEnricher
from agent_banking.chunking.universal.pipeline import LayoutAwareChunker
from agent_banking.chunking.universal.planning.semantic_boundaries import SemanticBoundaryScorer
from agent_banking.embedding import run_embedding
from agent_banking.vector_db import run_vector_db
from agent_banking.vector_db.qdrant_store import QdrantBankingStore, mark_publication_chunks


def _embedded(identifier, source, content="Hạn mức chuyển khoản.", vector=None):
    return {
        "chunk_id": identifier, "source": source, "content": content,
        "business_metadata": {}, "vector": vector if vector is not None else [1.0, 0.0],
    }


def test_document_source_is_identical_across_ingestion_entry_points(tmp_path, monkeypatch):
    from agent_banking import pipeline

    root = tmp_path / "input"
    directory = root / "A_Bank" / "Limits"
    directory.mkdir(parents=True)
    source = directory / "Quy định.docx"
    with source.open("wb") as file:
        document = Document()
        document.add_paragraph("Hạn mức chuyển khoản phải được phê duyệt.")
        document.save(file)
    monkeypatch.setattr(ModelEnricher, "enrich", lambda self, ir: ir)
    monkeypatch.setattr(SemanticBoundaryScorer, "annotate", lambda self, ir: ir)
    config = SimpleNamespace(CHUNKING_INPUT_DIR=root, DATASET_DIR=root / "A_Bank")
    settings = {"use_decision": False, "cache_db_path": tmp_path / "cache.db", "config": config}
    published, = run_auto_chunking(root, tmp_path / "publication", retry_delays=(), **settings)
    with LayoutAwareChunker(asset_dir=tmp_path / "assets", **settings) as chunker:
        direct, = chunker.chunk_file(source)
        relative = source.relative_to(root).as_posix()
        byte_input, = chunker.chunk_docx(docx_bytes=source.read_bytes(), source_name=relative)
        absolute_byte_input, = chunker.chunk_docx(docx_bytes=source.read_bytes(), source_name=str(source))
        collected = []
        instance = SimpleNamespace(config=config, dla_chunker=chunker,
                                   tracer=SimpleNamespace(trace_chunking=lambda **kwargs: None),
                                   ingest_chunks=lambda chunks: collected.extend(chunks))
        pipeline.BankingRAGPipeline.ingest_documents(instance)
        pipeline.BankingRAGPipeline.ingest_documents(
            instance, filenames=[source.name, "Limits\\Quy định.docx", relative, str(source)],
        )
        with pytest.raises(FileNotFoundError):
            pipeline.BankingRAGPipeline.ingest_documents(instance, filenames=["wrong/Quy định.docx"])
    assert len(collected) == 2
    assert all(chunk["source"] == relative and chunk["chunk_id"] == published["chunk_id"]
               for chunk in [published, direct, byte_input, absolute_byte_input, *collected])
    nested, = run_auto_chunking(directory, tmp_path / "nested-publication", retry_delays=(), **settings)
    assert nested["source"] == relative
    assert nested["chunk_id"] == published["chunk_id"]


def test_scoped_prune_resolves_only_unique_legacy_aliases(tmp_path):
    store = QdrantBankingStore(collection_name="audit", dimension=2, storage_path=tmp_path)
    try:
        new_source = "A_Bank/Credit/Hạn mức.docx"
        failed_source = "A_Bank/Loans/Frozen.docx"
        active = mark_publication_chunks([_embedded("published", new_source)], "scope")
        store.add_chunks([
            _embedded("basename", "Hạn mức.docx"),
            _embedded("relative", "Credit\\Hạn mức.docx"),
            _embedded("failed", "Frozen.docx"),
            _embedded("ambiguous", "Same.docx"),
            _embedded("unrelated", "Other/Hạn mức.docx"),
            _embedded("removed", "A_Bank/Other/Frozen.docx"),
        ])
        store.add_chunks(mark_publication_chunks([_embedded("owned-removal", "A_Bank/Other/Frozen.docx")], "scope"))
        store.add_chunks(active)
        removed = store.prune_chunks(
            {"published"}, {new_source, "A_Bank/First/Same.docx"}, scope="scope",
            preserve_sources={failed_source, "A_Bank/Second/Same.docx"},
            active_index_keys={store.index_key(active[0])},
        )
        assert removed == 3
        assert store.chunk_ids() == {"published", "failed", "ambiguous", "unrelated", "removed"}
        assert len(store._bm25_corpus) == store.count() == 5
    finally:
        store.close()


def test_publication_variants_keep_independent_payloads_and_vectors(tmp_path):
    from agent_banking.generator.run_generator import _compute_chunk_signature
    from agent_banking.reranking.run_reranking import extract_candidate_ids
    from agent_banking.retrieval.hybrid_retriever import HybridRetriever

    def open_store():
        return QdrantBankingStore(collection_name="audit", dimension=2, storage_path=tmp_path)

    first = _embedded("shared", "shared.docx", "Hạn mức đầu tiên", [1.0, 0.0])
    second = _embedded("shared", "shared.docx", "Hạn mức kế tiếp", [0.0, 1.0])
    store = open_store()
    try:
        store.add_chunks(mark_publication_chunks([first], "first"))
        store.add_chunks(mark_publication_chunks([first], "third"))
        store.add_chunks(mark_publication_chunks([second], "second"))
        assert store.count() == len(store._bm25_corpus) == 2
        assert store.chunk_ids(scope="first", index_keys=True) == store.chunk_ids(scope="third", index_keys=True)
        assert store.chunk_ids(scope="first", index_keys=True).isdisjoint(store.chunk_ids(scope="second", index_keys=True))
        second_filter = {"business_metadata.publication_scopes": "second"}
        assert store.search_vector([0.0, 1.0], filter_dict=second_filter)[0][1] == pytest.approx(1.0)
        assert store.search_bm25("kế tiếp", filter_dict=second_filter)[0][0]["content"] == second["content"]
        retriever = HybridRetriever(vector_store=store, embedder=SimpleNamespace(embed_query=lambda text: [1.0, 0.0]))
        merged = retriever.retrieve("Hạn mức", top_k=2)
        assert len(merged) == 2
        assert {hit["chunk_id"] for hit in merged} == {"shared"}
        assert len(set(extract_candidate_ids(merged))) == 2
        assert _compute_chunk_signature(merged[0]) != _compute_chunk_signature(merged[1])
        assert retriever.retrieve("Hạn mức", filter_dict={"business_metadata.publication_scopes": "unpublished"}) == []
        assert store.prune_chunks(set(), scope="second") == 1
        assert store.prune_chunks(set(), scope="third") == 0
    finally:
        store.close()
    store = open_store()
    try:
        assert store.count() == len(store._bm25_corpus) == 1
        first_filter = {"business_metadata.publication_scopes": "first"}
        hit, score = store.search_vector([1.0, 0.0], filter_dict=first_filter)[0]
        assert hit["content"] == first["content"]
        assert score == pytest.approx(1.0)
    finally:
        store.close()


def test_snapshot_replaces_same_chunk_id_revision_within_its_scope(tmp_path, monkeypatch):
    config = SimpleNamespace(QDRANT_STORAGE_PATH=tmp_path / "qdrant", QDRANT_COLLECTION="audit", EMBEDDING_DIM=2)
    monkeypatch.setattr(run_vector_db, "Config", lambda: config)
    monkeypatch.setattr(run_vector_db, "PROJECT_ROOT", tmp_path)
    source = "shared.docx"
    manifest = {"records": [{"rel_path": source, "status": "SUCCESS"}]}
    root = tmp_path / "index"
    first = _embedded("same-id", source, vector=[1.0, 0.0])
    second = _embedded("same-id", source, vector=[0.0, 1.0])

    def index(chunk, scope):
        return run_vector_db.run_auto_indexing(
            embedded_chunks=[chunk], output_dir=root, snapshot=True,
            publication_dir=tmp_path / scope, publication_manifest=manifest,
        )

    assert index(first, "first")["new_count"] == 1
    assert index(first, "second")["total_points"] == 1
    changed = index(second, "first")
    assert (changed["removed_count"], changed["total_points"]) == (0, 2)
    skipped = index(second, "first")
    assert (skipped["new_count"], skipped["total_points"]) == (0, 2)
    restored = index(first, "first")
    assert (restored["removed_count"], restored["total_points"]) == (1, 1)


def test_unready_chunks_fail_before_embedding_cache_or_qdrant(tmp_path, monkeypatch):
    from agent_banking import pipeline

    raw = _embedded("unready", "draft.docx")
    raw["business_metadata"]["quality"] = {"index_ready": False}
    config = SimpleNamespace(QDRANT_STORAGE_PATH=tmp_path / "index", EMBEDDING_DIM=2)
    for module in (run_embedding, run_vector_db):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(module, "Config", lambda: config)
    raw_file = tmp_path / "chunks.json"
    atomic_json(raw_file, [raw])
    with pytest.raises(ValueError, match="not ready for indexing"):
        run_embedding.run_auto_embedding(raw_file, tmp_path / "embedding")
    with pytest.raises(ValueError, match="not ready for indexing"):
        run_vector_db.run_auto_indexing(embedded_chunks=[raw], output_dir=tmp_path / "index-output")
    with pytest.raises(ValueError, match="not ready for indexing"):
        pipeline.BankingRAGPipeline.ingest_chunks(SimpleNamespace(), [raw])
    assert not config.QDRANT_STORAGE_PATH.exists()


def test_standalone_index_node_uses_current_publication(tmp_path, monkeypatch):
    from agent_banking import node_pipeline

    pipeline = node_pipeline.NodePipeline(base_dir=tmp_path)
    folders = pipeline.get_node_folders()
    publication = folders["chunking"]["output"]
    atomic_json(publication / "chunking_manifest.json", {"records": []})
    embed_out = folders["embedding"]["output"] / "embedded_chunks.json"
    with pytest.raises(FileNotFoundError, match="Embed the current chunking publication"):
        pipeline.run_node("indexing")
    atomic_json(embed_out, [])
    observed = {}

    def index(**options):
        observed.update(options)
        return {"status": "success"}

    monkeypatch.setattr(node_pipeline, "run_auto_indexing", index)
    assert pipeline.run_node("indexing")["manifest"]["status"] == "success"
    assert observed["input_file"] == embed_out
    assert observed["publication_dir"] == publication
    assert observed["snapshot"] is True
