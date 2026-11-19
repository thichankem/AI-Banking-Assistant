"""Run versioned PDF/DOCX ingestion and publish verified snapshots."""

import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.chunking.config.settings import ChunkingConfig, get_config_section
from agent_banking.chunking.publication import (
    _load_snapshot,
    _publish_document_files,
    _read_cached,
    _validate_document,
    publication_asset_path,
    publish_snapshot,
    use_relative_asset_paths,
)
from agent_banking.chunking.publication_io import (
    _is_indirection,
    _publication_cleanup_candidates,
    _remove_obsolete_publication_files,
    atomic_bytes as atomic_bytes,
    atomic_json as atomic_json,
    content_hash as content_hash,
    file_hash,
    publication_lock,
    read_verified_json as read_verified_json,
    report,
)
from agent_banking.chunking.universal.extraction.pdf.page_review import CloudOCRUnavailable
from agent_banking.chunking.universal.extraction.sources import discover_documents, document_source_root
from agent_banking.chunking.universal.pipeline import LayoutAwareChunker

_PIPELINE = get_config_section("pipeline")
_DEFAULT_RETRY_DELAYS = tuple(_PIPELINE["retry_delays"])
_LOAD_ATTEMPTS = _PIPELINE["load_attempts"]
_SCHEMA_VERSION = _PIPELINE["schema_version"]
_PROFILE = _PIPELINE["profile"]
_DEFAULT_IMAGE_MODE = _PIPELINE["default_image_mode"]
_DEFAULT_INPUT_SUBDIR = _PIPELINE["input_subdir"]
_DEFAULT_OUTPUT_SUBDIR = _PIPELINE["output_subdir"]
_DIAGNOSTICS_PREFIX = _PIPELINE["diagnostics_prefix"]
_MANIFEST_FILENAME = _PIPELINE["manifest_filename"]
_LAST_RUN_FILENAME = _PIPELINE["last_run_filename"]
_STATE_TRACKER_FILENAME = _PIPELINE["state_tracker_filename"]
_QUALITY_METRICS = tuple(_PIPELINE["quality_metrics"])


def load_published_chunks(output_dir, require_ready=True, *, return_manifest=False):
    output = Path(output_dir)
    manifest_path = output / _MANIFEST_FILENAME
    manifest_bytes = manifest_path.read_bytes()
    for attempt in range(_LOAD_ATTEMPTS):
        manifest = json.loads(manifest_bytes)
        try:
            chunks = _load_snapshot(output, manifest, require_ready)
        except (OSError, ValueError):
            latest = manifest_path.read_bytes()
            if attempt == _LOAD_ATTEMPTS - 1 or manifest_bytes == latest:
                raise
            # A successful --clean may have collected files from this earlier manifest.
            manifest_bytes = latest
        else:
            if return_manifest:
                return chunks, manifest
            return chunks


def run_auto_chunking(input_dir=None, output_dir=None, force=False, max_files=None, *,
                      require_ready=None, clean_output=False,
                      retry_delays=None, **chunker_options):
    if retry_delays is None:
        retry_delays = _DEFAULT_RETRY_DELAYS
    config = chunker_options.get("config")
    for option, setting in (("target_chunk_tokens", "CHUNK_SIZE_TOKENS"),
                            ("max_chunk_tokens", "CHUNK_MAX_TOKENS"),
                            ("min_chunk_tokens", "CHUNK_MIN_TOKENS"),
                            ("table_rows_per_chunk", "TABLE_ROWS_PER_CHUNK"),
                            ("search_language", "CHUNK_SEARCH_LANGUAGE")):
        if option not in chunker_options:
            chunker_options[option] = getattr(config, setting, getattr(ChunkingConfig, setting))
    output = Path(output_dir or Path(__file__).resolve().parent / _DEFAULT_OUTPUT_SUBDIR)
    if max_files is not None:
        if type(max_files) is not int or max_files < 0:
            raise ValueError("max_files must be a nonnegative integer")
        output = output / "diagnostics" / (_DIAGNOSTICS_PREFIX + str(max_files))
    if clean_output and (output.exists() or output.is_symlink()) and _is_indirection(output.lstat()):
        raise ValueError("Refusing to clean an output directory link or junction")
    with publication_lock(output):
        return _run_auto_chunking(input_dir, output, force, max_files,
                                  require_ready=require_ready, clean_output=clean_output,
                                  retry_delays=retry_delays, **chunker_options)


def _run_auto_chunking(input_dir=None, output_dir=None, force=False, max_files=None, *,
                       require_ready=None, clean_output=False,
                       retry_delays=None, **chunker_options):
    if retry_delays is None:
        retry_delays = _DEFAULT_RETRY_DELAYS
    module = Path(__file__).resolve().parent
    incoming = Path(input_dir or module / _DEFAULT_INPUT_SUBDIR).resolve()
    output = Path(output_dir or module / _DEFAULT_OUTPUT_SUBDIR)
    if not incoming.is_dir():
        raise FileNotFoundError("Input directory not found: " + str(incoming))
    source_root = document_source_root(incoming, getattr(chunker_options.get("config"), "CHUNKING_INPUT_DIR", module / _DEFAULT_INPUT_SUBDIR))
    discovered = discover_documents(incoming, reject_unsupported=True)
    files = {path.relative_to(source_root).as_posix(): path for path in discovered}
    active_sources = set(files)
    output.mkdir(parents=True, exist_ok=True)
    previous_files = set()
    if clean_output:
        if incoming.resolve().is_relative_to(output.resolve()):
            raise ValueError("Input must be outside the output directory when using --clean")
        cleanup_root, previous_files = _publication_cleanup_candidates(output)
        force = True
    if max_files is not None:
        files = {path.relative_to(source_root).as_posix(): path for path in discovered[:max_files]}
    state_path = output / _STATE_TRACKER_FILENAME
    records = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") == _SCHEMA_VERSION:
                records = {name: value for name, value in state.get("records", {}).items() if name in active_sources}
        except (OSError, ValueError, TypeError, AttributeError):
            records = {}
    started = time.perf_counter()
    all_chunks = []
    completed = {}
    errors = {}
    cloud_failures = set()
    aliases = {}
    refreshed = set()
    active_asset_paths = set()
    with LayoutAwareChunker(asset_dir=output / "assets", **chunker_options) as chunker:
        fingerprint = chunker.fingerprint()
        if require_ready is None:
            require_ready = True
        def process(source, path, attempt):
            beginning = time.perf_counter()
            try:
                digest = file_hash(path)
                record = records.get(source, {})
                if force and source not in refreshed:
                    cached = None
                else:
                    cached = _read_cached(output, record, source, digest, fingerprint)
                if cached is not None:
                    chunks, artifact = cached
                    status = "SKIPPED_ALREADY_PROCESSED"
                else:
                    chunks = chunker.chunk_file(path, source_name=source)
                    artifact = chunker.last_artifact
                    use_relative_asset_paths(output, chunks, artifact)
                    _validate_document(output, chunks, artifact, source, digest, fingerprint)
                    status = "SUCCESS"
                if require_ready and not artifact["quality"].get("index_ready"):
                    raise ValueError("Document has unresolved interpretation; index publication refused")
                if cached is None:
                    records[source] = _publish_document_files(output, chunks, artifact, digest, fingerprint)
                    atomic_json(state_path, {"schema_version": _SCHEMA_VERSION, "records": records})
                    refreshed.add(source)
                for asset in artifact["document"]["assets"].values():
                    active_asset_paths.add(publication_asset_path(output, asset["path"]))
                    if asset.get("visual_path"):
                        active_asset_paths.add(publication_asset_path(output, asset["visual_path"]))
                aliases[artifact["document"]["document_id"]] = chunks
                completed[source] = (chunks, {
                    "file_name": path.name, "rel_path": source, "input_rel_path": path.relative_to(incoming).as_posix(),
                    "status": status, "attempts": attempt, "quality": artifact["quality"],
                    "duration_sec": round(time.perf_counter() - beginning, 3), **records[source],
                })
                errors.pop(source, None)
                detail = ""
                quality = artifact["quality"]
                if quality.get("source_pages"):
                    processed = quality.get("ocr_pages_processed", 0)
                    partial = quality.get("ocr_partial_pages", 0)
                    failed = quality.get("ocr_failed_pages", 0)
                    detail = f" (cloud OCR: {processed}/{quality['source_pages']} pages, {partial} partial, {failed} failed)"
                report(status + ": " + source + " -> " + str(len(chunks)) + " chunks" + detail)
            except Exception as error:
                if isinstance(error, CloudOCRUnavailable):
                    cloud_failures.add(source)
                errors[source] = {"file_name": path.name, "rel_path": source, "input_rel_path": path.relative_to(incoming).as_posix(),
                                  "status": "FAILED", "attempts": attempt,
                                  "error_type": type(error).__name__, "error": str(error), "chunk_count": 0}
                ocr_statistics = getattr(error, "pdf_ocr_statistics", None)
                if ocr_statistics is not None:
                    errors[source]["quality"] = ocr_statistics
                page = getattr(error, "pdf_page_number", None)
                location = ""
                if page is not None:
                    errors[source]["failed_pdf_page"] = page
                    location = f", page {page}"
                render_sha = getattr(error, "pdf_render_sha256", None)
                if render_sha is not None:
                    errors[source]["failed_pdf_render_sha256"] = render_sha
                report(f"ATTEMPT {attempt} FAILED: {source} ({type(error).__name__}{location})")

        for source, path in files.items():
            process(source, path, 1)
        for attempt, delay in enumerate(retry_delays, 2):
            retry_sources = errors.keys() - cloud_failures
            if not retry_sources:
                break
            report(f"Retrying {len(retry_sources)} failed document(s) in {delay}s (attempt {attempt})")
            time.sleep(delay)
            for source in files:
                if source in retry_sources:
                    process(source, files[source], attempt)
        manifests = []
        for source in files:
            if source in completed:
                chunks, record = completed[source]
                all_chunks.extend(chunks)
                manifests.append(record)
            else:
                manifests.append(errors[source])
        ids = [chunk["chunk_id"] for chunk in all_chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Cross-document chunk ID collision; index publication refused")
        manifest = {"module": "chunking", "schema_version": _SCHEMA_VERSION, "pipeline_fingerprint": fingerprint,
                    "profile": _PROFILE, "total_files": len(files), "failed_files": len(errors),
                    "skipped_files": sum(item["status"] == "SKIPPED_ALREADY_PROCESSED" for item in manifests),
                    "new_files": sum(item["status"] == "SUCCESS" for item in manifests),
                    "total_chunks": len(all_chunks), "total_elapsed_sec": round(time.perf_counter() - started, 3),
                    "index_ready": (not files or bool(completed))
                    and all(record["quality"]["index_ready"] for _, record in completed.values()),
                    "records": manifests}
        for metric in _QUALITY_METRICS:
            manifest[metric] = sum(item.get("quality", {}).get(metric, 0) for item in manifests)
        atomic_json(state_path, {"schema_version": _SCHEMA_VERSION, "records": records})
        if errors and not completed:
            atomic_json(output / _LAST_RUN_FILENAME, manifest)
            raise RuntimeError("Chunking failed for every document; previous publication retained")
        snapshot = publish_snapshot(output, all_chunks, manifest, aliases)
        if clean_output and not errors:
            active_files = {cleanup_root / snapshot, *active_asset_paths}
            active_files.update(cleanup_root / (identifier + ".json") for identifier in aliases)
            for record in manifests:
                for key in ("output_file", "artifact_file"):
                    active_files.add(cleanup_root / record[key])
            _remove_obsolete_publication_files(cleanup_root, previous_files, active_files)
        if errors:
            report(f"Published {len(completed)} document(s). {len(errors)} failed after all attempts.")
    return all_chunks


def main():
    import argparse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run the cloud chunking pipeline")
    cleaning = parser.add_mutually_exclusive_group()
    cleaning.add_argument("--clean", action="store_true", help="Rebuild and remove obsolete publication files after a successful run")
    cleaning.add_argument("--no-clean", action="store_true", help="Preserve existing publications (default)")
    rerun = parser.add_mutually_exclusive_group()
    rerun.add_argument("--force", action="store_true", help="Rebuild per-document artifacts using the model cache")
    rerun.add_argument("--keep-cache", action="store_true", help="Reuse validated document artifacts (default)")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most this many files and publish under output/diagnostics")
    parser.add_argument("--image-mode", choices=["text_context", "vision"], default=_DEFAULT_IMAGE_MODE,
                        help="Deprecated compatibility option. Standalone images are ignored in both modes.")
    parser.add_argument("--check-models", action="store_true", help="Probe configured cloud endpoints without processing documents")
    parser.add_argument("--ocr-docx-images", action="store_true", help="Transcribe embedded DOCX raster images via cloud vision")
    parser.add_argument("--search-language", choices=["vi", "all"], default=None,
                        help="Retain all source languages for search (default) or opt into Vietnamese-only search")
    parser.add_argument("--input", type=Path, default=None, help="Input directory")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()
    if args.check_models:
        from agent_banking.chunking.universal.models.preflight import check_models
        checks = check_models()
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        if any(check["status"] == "failed" for check in checks):
            return 1
        return 0

    module_dir = Path(__file__).resolve().parent
    in_dir = args.input or module_dir / _DEFAULT_INPUT_SUBDIR
    options = {"ocr_docx_images": args.ocr_docx_images}
    if args.search_language is not None:
        options["search_language"] = args.search_language
    run_auto_chunking(
        input_dir=in_dir,
        output_dir=args.output,
        force=args.force,
        clean_output=args.clean,
        max_files=args.max_files,
        image_mode=args.image_mode,
        **options,
    )
    output = args.output or module_dir / _DEFAULT_OUTPUT_SUBDIR
    if args.max_files is not None:
        output = output / "diagnostics" / (_DIAGNOSTICS_PREFIX + str(args.max_files))
    last_run = json.loads((output / _LAST_RUN_FILENAME).read_text(encoding="utf-8"))
    if last_run["failed_files"]:
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
