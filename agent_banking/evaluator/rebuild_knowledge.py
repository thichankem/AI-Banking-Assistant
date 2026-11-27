"""Build isolated, source-verified publications using configured cloud models.

Resumable per document. Does not use golden questions or replace the live index.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

from agent_banking.evaluator.config.settings import EvaluatorConfig

_KB_CFG = EvaluatorConfig.KNOWLEDGE_BUILD


def build_document(path, output, ocr_docx_images=False, recheck_partial=False, include_hidden_sheets=False):
    from agent_banking.core.settings import Config
    from agent_banking.chunking.run_chunking import load_published_chunks
    from agent_banking.chunking.policy_revision import attach_policy_revision, read_policy_history
    from agent_banking.chunking.publication import (
        _publish_document_files, _validate_document, publish_snapshot, use_relative_asset_paths,
    )
    from agent_banking.chunking.publication_io import file_hash, read_verified_json
    from agent_banking.chunking.universal.extraction.pdf.extractor import PartialOCRNotImproved
    from agent_banking.chunking.universal.pipeline import LayoutAwareChunker
    from agent_banking.core.document_revisions import archive_document
    from agent_banking.evaluator.index_knowledge import unavailable_images

    started = time.perf_counter()
    config = Config()
    source = path.resolve().relative_to(config.CHUNKING_INPUT_DIR.resolve()).as_posix()
    prefix_len = int(_KB_CFG["folder_hash_prefix_len"])
    folder = output / sha256(source.encode("utf-8")).hexdigest()[:prefix_len]
    digest = file_hash(path)
    result = {"source": source, "publication": str(folder), "status": "ERROR"}
    try:
        previous_partial = 0
        review_page_indexes = None
        cached_chunks = 0
        if (folder / "chunking_manifest.json").exists():
            chunks, manifest = load_published_chunks(folder, return_manifest=True)
            record = manifest["records"][0]
            quality = record.get("quality", {})
            usable = not (quality.get("ocr_failed_pages", 0) or quality.get("excel_ocr_unresolved", 0))
            if ocr_docx_images:
                previous = read_verified_json(folder / record["artifact_file"], record["artifact_sha256"])
                usable = usable and not unavailable_images(previous["document"])
            if usable and record["input_sha256"] == digest and bool(record.get("ocr_docx_images")) == ocr_docx_images and bool(record.get("include_hidden_sheets")) == include_hidden_sheets:
                previous_partial = quality.get("ocr_partial_pages", 0)
                if recheck_partial and previous_partial:
                    previous = read_verified_json(folder / record["artifact_file"], record["artifact_sha256"])
                    review_page_indexes = {review["page_index"] for review in previous["document"]["page_reviews"]
                                           if review.get("incomplete")}
                if (not recheck_partial or not previous_partial) and (
                    path.suffix.lower() == ".xlsx" or record.get("policy_revision")
                ):
                    archive_document(folder / "revisions", path, digest)
                    return {**result, "status": "CACHED", "chunks": len(chunks), "seconds": 0}
                cached_chunks = len(chunks)
        chunker_options = {}
        if recheck_partial:
            chunker_options = {"workers": 1}
        with LayoutAwareChunker(config=config, asset_dir=folder / "assets", ocr_docx_images=ocr_docx_images,
                                review_incomplete_pdf=recheck_partial, previous_partial_pdf_pages=previous_partial,
                                review_pdf_page_indexes=review_page_indexes,
                                include_hidden_sheets=include_hidden_sheets,
                                **chunker_options) as chunker:
            chunks = chunker.chunk_file(path, source_name=source)
            artifact = chunker.last_artifact
            if previous_partial and (not artifact["quality"].get("index_ready") or artifact["quality"].get("ocr_failed_pages", 0)
                                     or artifact["quality"].get("ocr_partial_pages", 0) >= previous_partial):
                return {**result, "status": "RETAINED", "chunks": cached_chunks, "ocr_partial_pages": previous_partial,
                        "seconds": round(time.perf_counter() - started, 2)}
            use_relative_asset_paths(folder, chunks, artifact)
            history = read_policy_history(folder / "chunking_manifest.json", "2.0")
            revision = attach_policy_revision(artifact["document"], chunks, history.get(source, {}).get(digest))
            if revision is not None:
                history.setdefault(source, {})[digest] = revision
            fingerprint = chunker.fingerprint()
            _validate_document(folder, chunks, artifact, source, digest, fingerprint)
            if not artifact["quality"].get("index_ready"):
                raise ValueError("Document interpretation is unresolved")
            if artifact["quality"].get("ocr_failed_pages", 0) or unavailable_images(artifact["document"]):
                raise ValueError("Some source pixels still have no usable cloud transcription")
            archive_document(folder / "revisions", path, digest)
            record = _publish_document_files(folder, chunks, artifact, digest, fingerprint)
            record.update(file_name=path.name, rel_path=source, status="SUCCESS", quality=artifact["quality"],
                          ocr_docx_images=ocr_docx_images, include_hidden_sheets=include_hidden_sheets)
            manifest = {"module": "chunking", "schema_version": "2.0", "pipeline_fingerprint": fingerprint,
                        "profile": "accuracy", "total_files": 1, "failed_files": 0, "skipped_files": 0,
                        "new_files": 1, "total_chunks": len(chunks), "index_ready": True, "records": [record]}
            if history:
                manifest["policy_revision_history"] = history
            publish_snapshot(folder, chunks, manifest, {artifact["document"]["document_id"]: chunks})
            result.update(status="SUCCESS", chunks=len(chunks), quality=artifact["quality"])
    except PartialOCRNotImproved:
        result.update(status="RETAINED", chunks=cached_chunks, ocr_partial_pages=previous_partial)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=int(_KB_CFG["default_workers"]))
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--extensions", nargs="*", default=[*_KB_CFG["default_extensions"], ".xlsx"])
    parser.add_argument("--ocr-docx-images", action="store_true")
    parser.add_argument("--include-hidden-sheets", action="store_true",
                        help="Include standard hidden Excel sheets when rebuilding XLSX publications.")
    parser.add_argument("--recheck-partial", action="store_true", help="Retry partially read PDF pages and keep the better publication.")
    parser.add_argument("--only-partial", action="store_true", help="Process existing publications with partially read PDF pages")
    args = parser.parse_args()
    from agent_banking.core.settings import Config
    from agent_banking.chunking.universal.extraction.sources import discover_documents

    files = discover_documents(Config.CHUNKING_INPUT_DIR, extensions=args.extensions)
    if args.names:
        files = [path for path in files if any(path.name.startswith(name) for name in args.names)]
    if args.only_partial:
        args.recheck_partial = True
        config = Config()
        partial_sources = set()
        for manifest in args.output.glob("*/chunking_manifest.json"):
            for record in json.loads(manifest.read_bytes())["records"]:
                if record.get("quality", {}).get("ocr_partial_pages", 0):
                    partial_sources.add(record["rel_path"])
        files = [path for path in files if path.resolve().relative_to(config.CHUNKING_INPUT_DIR.resolve()).as_posix() in partial_sources]
    # Small documents establish usable coverage before long annual reports.
    files.sort(key=lambda path: path.stat().st_size)
    args.output.mkdir(parents=True, exist_ok=True)
    print("Building", len(files), "source publications", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                build_document, path, args.output.resolve(), args.ocr_docx_images,
                args.recheck_partial, args.include_hidden_sheets,
            )
            for path in files
        ]
        for future in as_completed(futures):
            record = future.result()
            if record["status"] == "ERROR":
                failures += 1
            with (args.output / "build_results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            print(record["source"], record["status"], record["seconds"], record.get("error_type", ""), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
