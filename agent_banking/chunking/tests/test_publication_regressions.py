import hashlib
import json
from io import BytesIO
import unicodedata

from docx import Document
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from agent_banking.chunking import run_chunking
from agent_banking.chunking.universal.extraction.sources import discover_documents, resolve_document_source
from agent_banking.chunking.universal.pipeline import LayoutAwareChunker, _parse_markdown_table_to_2d_list
from agent_banking.chunking.universal.planning.schema import build_universal_chunk, validate_universal_chunk
from agent_banking.embedding.run_embedding import normalize_universal_chunk


def _write_word(target, text="Hạn mức chuyển khoản là 10 triệu đồng."):
    document = Document()
    document.add_paragraph(text)
    document.save(target)


@pytest.fixture
def publication(tmp_path):
    incoming = tmp_path / "input"
    incoming.mkdir()
    output = tmp_path / "output"
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    options = dict(cache_db_path=tmp_path / "cache.db", require_ready=False, use_llm=False,
                   use_decision=False, retry_delays=(), tokenizer=tokenizer)
    return incoming, output, options


def test_exact_filename_through_publication_and_embedding(publication):
    incoming, output, options = publication
    source = incoming / " Hạn mức.docx"
    _write_word(source)
    chunks = run_chunking.run_auto_chunking(incoming, output, **options)
    verified, manifest = run_chunking.load_published_chunks(output, require_ready=False, return_manifest=True)
    assert chunks == verified
    assert normalize_universal_chunk(verified[0])["source"] == manifest["records"][0]["rel_path"] == source.name


def test_clean_collects_alias_for_previously_removed_source(publication):
    incoming, output, options = publication
    source = incoming / "A.docx"
    _write_word(source)
    chunks = run_chunking.run_auto_chunking(incoming, output, **options)
    alias = output / (chunks[0]["business_metadata"]["document_id"] + ".json")
    source.unlink()
    run_chunking.run_auto_chunking(incoming, output, **options)
    assert alias.exists()
    run_chunking.run_auto_chunking(incoming, output, clean_output=True, **options)
    assert not alias.exists()


def test_manifest_with_missing_records_cannot_be_used_as_empty_snapshot(publication):
    incoming, output, options = publication
    run_chunking.run_auto_chunking(incoming, output, **options)
    manifest_path = output / "chunking_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    del manifest["records"]
    run_chunking.atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="document count mismatch"):
        run_chunking.load_published_chunks(output, require_ready=False)


def test_retry_only_loads_failed_documents(publication, monkeypatch):
    incoming, output, options = publication
    for name in ("A.docx", "B.docx"):
        _write_word(incoming / name)
    calls = []
    original = LayoutAwareChunker.chunk_file

    def extract(self, path, **options):
        calls.append(path.name)
        if calls == ["A.docx"]:
            raise RuntimeError("Transient extraction error")
        return original(self, path, **options)

    monkeypatch.setattr(LayoutAwareChunker, "chunk_file", extract)
    options["retry_delays"] = (0,)
    run_chunking.run_auto_chunking(incoming, output, **options)
    _, manifest = run_chunking.load_published_chunks(output, require_ready=False, return_manifest=True)
    assert calls == ["A.docx", "B.docx", "A.docx"]
    assert [record["attempts"] for record in manifest["records"]] == [2, 1]


def test_duplicate_document_names_include_unicode_and_copy_suffixes(tmp_path):
    first = tmp_path / "original"
    second = tmp_path / "copy"
    first.mkdir()
    second.mkdir()
    source = first / "Hạn mức.docx"
    duplicate = second / (unicodedata.normalize("NFD", source.stem.upper()) + " (1).DOCX")
    source.touch()
    duplicate.touch()
    with pytest.raises(ValueError, match="Duplicate document names") as error:
        discover_documents(tmp_path)
    assert source.relative_to(tmp_path).as_posix() in str(error.value)
    assert duplicate.relative_to(tmp_path).as_posix() in str(error.value)


def test_writer_lock_prevents_overlapping_publications(publication):
    incoming, output, options = publication
    with run_chunking.publication_lock(output):
        with pytest.raises(RuntimeError, match="Another chunking writer"):
            run_chunking.run_auto_chunking(incoming, output, **options)
        assert not (output / "chunking_manifest.json").exists()
    assert run_chunking.run_auto_chunking(incoming, output, **options) == []


def test_publication_reads_shared_config_and_honors_explicit_budget(publication):
    incoming, output, options = publication
    _write_word(incoming / "A.docx")
    config = type("Config", (), {"CHUNK_SIZE_TOKENS": 64, "CHUNK_MAX_TOKENS": 96, "CHUNK_MIN_TOKENS": 32})
    (chunk,) = run_chunking.run_auto_chunking(incoming, output, config=config, max_chunk_tokens=112, **options)
    assert chunk["business_metadata"]["retrieval"]["max_tokens"] == 112


def test_legacy_manual_file_cannot_override_source_content(publication):
    incoming, output, options = publication
    source = incoming / "A.docx"
    _write_word(source)
    (base,) = run_chunking.run_auto_chunking(incoming, output, **options)
    foreign = "これは日本語の文章です。"
    edit_file = output / "manual_edits" / (source.name + ".json")
    run_chunking.atomic_json(edit_file, {
        "document_version": base["business_metadata"]["document_version"],
        "replacements": {base["chunk_id"]: foreign},
    })
    saved_edit = edit_file.read_bytes()
    (chunk,) = run_chunking.run_auto_chunking(incoming, output, clean_output=True, **options)
    assert chunk == base
    assert edit_file.read_bytes() == saved_edit
    _, manifest = run_chunking.load_published_chunks(output, require_ready=False, return_manifest=True)
    assert manifest["records"][0]["status"] == "SUCCESS"
    assert "base_output_file" not in manifest["records"][0]


@pytest.mark.parametrize("flag", ["--manual-edits", "--reviewed-pdf-transcripts"])
def test_removed_content_override_flags_are_rejected(publication, flag, monkeypatch):
    incoming, output, _ = publication
    monkeypatch.setattr("sys.argv", ["run_chunking", "--input", str(incoming), "--output", str(output), flag, str(incoming)])
    with pytest.raises(SystemExit) as error:
        run_chunking.main()
    assert error.value.code == 2
    assert not (output / "chunking_manifest.json").exists()


@pytest.mark.parametrize("override", ["manual", "transcript"])
def test_old_override_snapshot_is_readable_for_audit_but_requires_rebuild_to_index(publication, override):
    incoming, output, options = publication
    source = incoming / "A.docx"
    _write_word(source)
    chunks = run_chunking.run_auto_chunking(incoming, output, **options)
    _, manifest = run_chunking.load_published_chunks(output, require_ready=False, return_manifest=True)
    record, = manifest["records"]
    artifact = run_chunking.read_verified_json(output / record["artifact_file"], record["artifact_sha256"])
    artifact["quality"]["index_ready"] = True
    manifest["index_ready"] = True
    if override == "manual":
        record["status"] = "MANUALLY_EDITED"
    else:
        artifact["document"]["extraction_method"] = "pdf_reviewed_transcript_and_cloud_vision"
    record.update(run_chunking._publish_document_files(
        output, chunks, artifact, record["input_sha256"], record["pipeline_fingerprint"],
    ))
    run_chunking.atomic_json(output / "chunking_manifest.json", manifest)
    assert run_chunking.load_published_chunks(output, require_ready=False) == chunks
    with pytest.raises(ValueError, match="retired content override; rerun chunking"):
        run_chunking.load_published_chunks(output)


def test_legacy_native_pdf_snapshot_requires_cloud_rebuild(publication):
    _, output, _ = publication
    source = "old.PDF"
    digest = "1" * 64
    fingerprint = "legacy-pdf"
    artifact = {
        "document": {"source": source, "document_version": digest, "assets": {},
                     "document_id": hashlib.sha256(source.encode()).hexdigest()[:20]},
        "pipeline_fingerprint": fingerprint, "parents": {},
        "quality": {"index_ready": True, "visual_policy": "native_pdf_no_image_ocr"},
    }
    record = run_chunking._publish_document_files(output, [], artifact, digest, fingerprint)
    snapshot = {"schema_version": "2.0", "module": "chunking", "total_files": 1,
                "failed_files": 0, "total_chunks": 0, "index_ready": True, "pipeline_fingerprint": fingerprint,
                "records": [{**record, "rel_path": source, "status": "SUCCESS"}]}
    run_chunking.publish_snapshot(output, [], snapshot, {})
    assert run_chunking.load_published_chunks(output, require_ready=False) == []
    with pytest.raises(ValueError, match="retired extraction policy; rerun chunking"):
        run_chunking.load_published_chunks(output)


def test_failed_pdf_page_counts_survive_total_publication_failure(publication, monkeypatch):
    incoming, output, options = publication
    (incoming / "failure.pdf").write_bytes(b"PDF fixture")

    def fail(self, path, **kwargs):
        error = RuntimeError("None of the three pages returned usable cloud OCR")
        error.pdf_page_number = 1
        error.pdf_ocr_statistics = {"source_pages": 3, "pages_requiring_ocr": 3, "ocr_pages_processed": 0,
                                    "ocr_partial_pages": 0, "ocr_failed_pages": 3,
                                    "ocr_failed_page_numbers": [1, 2, 3]}
        raise error

    monkeypatch.setattr(LayoutAwareChunker, "chunk_file", fail)
    with pytest.raises(RuntimeError, match="failed for every document"):
        run_chunking.run_auto_chunking(incoming, output, **options)
    report = json.loads((output / "chunking_last_run.json").read_text(encoding="utf-8"))
    assert report["failed_files"] == 1 and report["source_pages"] == report["ocr_failed_pages"] == 3
    assert report["records"][0]["quality"]["ocr_failed_page_numbers"] == [1, 2, 3]


def test_schema_rejects_wrong_field_types_and_mixed_keys():
    valid, issues = validate_universal_chunk({**build_universal_chunk(), "content": 5, "images": {},
                                              "next_chunk_id": None, "extra": {}, 4: {}})
    assert not valid
    for name in ("content", "images", "next_chunk_id", "'extra'", "'4'"):
        assert any(name in issue for issue in issues)


def test_legacy_markdown_table_parser_preserves_tex_and_literal_pipe():
    markdown = "| Giá trị | Ký hiệu |\n| --- | --- |\n| - | - |\n" + r"| A\|B | $\alpha$ |"
    assert _parse_markdown_table_to_2d_list(markdown) == [["Giá trị", "Ký hiệu"], ["-", "-"], ["A|B", r"$\alpha$"]]


def test_docx_byte_entrypoint_and_file_type_validation(publication):
    _, _, options = publication
    data = BytesIO()
    _write_word(data)
    with LayoutAwareChunker(use_llm=False, use_decision=False, cache_db_path=options["cache_db_path"],
                            tokenizer=options["tokenizer"]) as chunker:
        chunks = chunker.chunk_docx(docx_bytes=data.getvalue(), source_name=r"D:\files\A.docx")
        assert chunks[0]["source"] == "A.docx"
        with pytest.raises(ValueError, match="PDF adapter requires"):
            chunker.chunk_pdf(pdf_bytes=data.getvalue(), source_name="A.docx")
        assert chunker.last_artifact is None
    with pytest.raises(ValueError, match="filename"):
        resolve_document_source(source_bytes=data.getvalue(), source_name="invalid/")
