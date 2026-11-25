"""Index verified tuning publications into an explicitly isolated Qdrant store."""

import argparse
import json
from pathlib import Path

from agent_banking.evaluator.config.settings import EvaluatorConfig

_KB_CFG = EvaluatorConfig.KNOWLEDGE_BUILD
_UNAVAILABLE_ISSUES = set(_KB_CFG["unavailable_image_issues"])
_EXCLUDED_ROLE = str(_KB_CFG["excluded_retrieval_role"])


def refresh_legacy_domain_metadata(chunks):
    from agent_banking.chunking.universal.planning.schema import infer_business_metadata

    refreshed = []
    for chunk in chunks:
        metadata = chunk.get("business_metadata")
        if isinstance(metadata, dict) and metadata.get("domain") == "A_Bank":
            inferred = infer_business_metadata(chunk.get("source"))
            updated_meta = {
                **metadata,
                "domain": inferred["domain"],
                "doc_type": inferred["doc_type"],
            }
            chunk = {**chunk, "business_metadata": updated_meta}
        refreshed.append(chunk)
    return refreshed


def annotate_page_quality(chunks, document):
    uncertain = {review["page_index"] for review in document.get("page_reviews", [])
                 if review.get("incomplete") or review.get("content") == "unavailable"}
    uncertain_images = {review.get("asset_sha256") for review in document.get("docx_image_reviews", []) if review.get("incomplete")}
    result = []
    for chunk in chunks:
        metadata = chunk.get("business_metadata", {})
        locations = metadata.get("provenance", {}).get("locations", [])
        pages = {location.get("page_index") for location in locations}
        affected = sorted(page + 1 for page in uncertain.intersection(pages))
        image_affected = any(location.get("image_asset_sha256") in uncertain_images
                             for location in locations if location.get("image_asset_sha256"))
        if affected or image_affected:
            quality = {**metadata.get("quality", {})}
            if affected:
                quality["ocr_uncertain_pages"] = affected
            if image_affected:
                quality["ocr_uncertain_images"] = True
            chunk = {**chunk, "business_metadata": {**metadata, "quality": quality}}
        result.append(chunk)
    return result


def unavailable_images(document):
    return [review for review in document.get("docx_image_reviews", []) if _UNAVAILABLE_ISSUES.intersection(review.get("issues", []))]


def publication_matches(current, published):
    from agent_banking.vector_db.qdrant_store import _index_payload

    if len(current) != len(published):
        return False
    return {chunk["chunk_id"]: _index_payload(chunk, chunk["chunk_id"]) for chunk in current} == {
        chunk["chunk_id"]: _index_payload(chunk, chunk["chunk_id"]) for chunk in published}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publications", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-docx-images", action="store_true")
    args = parser.parse_args()
    from agent_banking.core.settings import Config
    from agent_banking.pipeline import BankingRAGPipeline
    from agent_banking.chunking.run_chunking import load_published_chunks
    from agent_banking.vector_db.qdrant_store import mark_publication_chunks, publication_scope
    from agent_banking.chunking.universal.extraction.sources import discover_documents
    from agent_banking.chunking.publication_io import file_hash
    from agent_banking.chunking.universal.planning.schema import payload_index_key

    config = Config()
    config.ABANK_QDRANT_STORAGE_PATH = str(args.index.resolve())
    pipeline = BankingRAGPipeline(config=config)
    indexed = []
    pending = []
    prune = []
    verified = {}
    image_failures = []
    try:
        verified_scopes = set()
        for manifest in sorted(args.publications.glob("*/chunking_manifest.json")):
            folder = manifest.parent
            chunks, metadata = load_published_chunks(folder, return_manifest=True)
            if len(metadata["records"]) != 1:
                raise ValueError("Expected one document per tuning publication")
            sources = {record["rel_path"] for record in metadata["records"] if record["status"] != "FAILED"}
            for record in metadata["records"]:
                path = config.CHUNKING_INPUT_DIR / record["rel_path"]
                if not path.is_file() or file_hash(path) != record["input_sha256"]:
                    raise ValueError("Source changed after publication: " + record["rel_path"])
                verified[record["rel_path"]] = record
            artifact = json.loads((folder / metadata["records"][0]["artifact_file"]).read_text(encoding="utf-8"))
            if unavailable_images(artifact["document"]):
                image_failures.append(metadata["records"][0]["rel_path"])
            chunks = annotate_page_quality(chunks, artifact["document"])
            chunks = refresh_legacy_domain_metadata(chunks)
            chunks = [chunk for chunk in chunks
                      if chunk.get("business_metadata", {}).get("retrieval", {}).get("role") != _EXCLUDED_ROLE]
            scope = publication_scope(folder)
            verified_scopes.add(scope)
            chunks = mark_publication_chunks(chunks, scope)
            current = [chunk for chunk in pipeline.vector_store._bm25_corpus
                       if scope in chunk.get("business_metadata", {}).get("publication_scopes", [])]
            expected = {chunk["chunk_id"] for chunk in chunks}
            if not publication_matches(current, chunks):
                pending.extend(chunks)
                prune.append((expected, sources, scope))
            indexed.append({"sources": sorted(sources), "chunks": len(chunks), "publication": str(folder.resolve())})
        expected_sources = {path.relative_to(config.CHUNKING_INPUT_DIR).as_posix()
                    for path in discover_documents(config.CHUNKING_INPUT_DIR)}
        complete = bool(expected_sources) and set(verified) == expected_sources
        if args.require_complete and not complete:
            raise ValueError("Verified publications do not cover the complete current corpus")
        if args.require_complete and any(record.get("quality", {}).get("ocr_failed_pages", 0) for record in verified.values()):
            raise ValueError("Some PDF pages still have failed OCR; rebuild those publications")
        if args.require_docx_images and any(source.endswith(".docx") and not record.get("ocr_docx_images")
                                            for source, record in verified.items()):
            raise ValueError("Some DOCX publications have not processed their embedded images")
        if args.require_docx_images and image_failures:
            raise ValueError("Some embedded source images still lack usable OCR: " + ", ".join(image_failures))
        if pending:
            _, keys = pipeline.ingest_chunks(pending, return_index_keys=True)
            active_by_scope = {}
            for expected, sources, scope in prune:
                active_by_scope[scope] = {payload_index_key(chunk) for chunk in pipeline.vector_store._bm25_corpus
                                         if scope in chunk.get("business_metadata", {}).get("publication_scopes", [])
                                         and chunk["chunk_id"] in expected and payload_index_key(chunk) in keys}
            pipeline.vector_store.prune_publication_scopes(active_by_scope)
        if args.require_complete:
            stale_scopes = {}
            for chunk in pipeline.vector_store._bm25_corpus:
                for owner in chunk.get("business_metadata", {}).get("publication_scopes", []):
                    if owner not in verified_scopes:
                        stale_scopes[owner] = set()
            if stale_scopes:
                pipeline.vector_store.prune_publication_scopes(stale_scopes)
        indexed_sources = {chunk["source"] for chunk in pipeline.vector_store._bm25_corpus}
        if args.require_complete and indexed_sources != expected_sources:
            raise ValueError("The resulting index does not contain exactly the current source corpus")
        report = {"documents": sum(len(item["sources"]) for item in indexed),
                  "chunks": pipeline.vector_store.count(), "publications": indexed,
                  "complete_source_coverage": complete, "expected_documents": len(expected_sources),
                  "docx_image_ocr_documents": sum(bool(record.get("ocr_docx_images")) for record in verified.values()),
                  "ocr_partial_pages": sum(record.get("quality", {}).get("ocr_partial_pages", 0) for record in verified.values()),
                  "source_hashes": {source: record["input_sha256"] for source, record in sorted(verified.items())}}
        report["ocr_uncertain_passages"] = sum(bool(chunk.get("business_metadata", {}).get("quality", {}).get("ocr_uncertain_pages")
                                                   or chunk.get("business_metadata", {}).get("quality", {}).get("ocr_uncertain_images"))
                                                for chunk in pipeline.vector_store._bm25_corpus)
        report["unavailable_image_documents"] = image_failures
        (args.index / "knowledge_inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Verified index:", report["documents"], "documents,", report["chunks"], "chunks", flush=True)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
