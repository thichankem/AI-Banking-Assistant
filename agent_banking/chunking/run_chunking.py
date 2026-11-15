"""Run versioned PDF/DOCX ingestion and publish verified snapshots."""

import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.chunking.config.settings import ChunkingConfig
from agent_banking.chunking.manual_edits import _apply_manual_edits
from agent_banking.chunking.publication import (
    _load_snapshot,
    _publish_document_files,
    _read_cached,
    _validate_document,
    publish_snapshot,
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
from agent_banking.chunking.universal.universal_chunker import LayoutAwareChunker, discover_documents


def load_published_chunks(output_dir, require_ready=True, *, return_manifest=False):
    output = Path(output_dir)
    manifest_path = output / "chunking_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    for attempt in range(3):
        manifest = json.loads(manifest_bytes)
        try:
            chunks = _load_snapshot(output, manifest, require_ready)
        except (OSError, ValueError):
            latest = manifest_path.read_bytes()
            if attempt == 2 or manifest_bytes == latest:
                raise
            # A successful --clean may have collected files from this earlier manifest.
            manifest_bytes = latest
        else:
            if return_manifest:
                return chunks, manifest
            return chunks


def run_auto_chunking(input_dir=None, output_dir=None, force=False, max_files=None, *,
                      require_ready=None, clean_output=False, manual_edits_dir=None,
                      retry_delays=(30, 60, 120), **chunker_options):
    config = chunker_options.get("config")
    for option, setting in (("target_chunk_tokens", "CHUNK_SIZE_TOKENS"),
                            ("max_chunk_tokens", "CHUNK_MAX_TOKENS"),
                            ("min_chunk_tokens", "CHUNK_MIN_TOKENS")):
        if option not in chunker_options:
            chunker_options[option] = getattr(config, setting, getattr(ChunkingConfig, setting))
    output = Path(output_dir or Path(__file__).resolve().parent / "output")
    if max_files is not None:
        if type(max_files) is not int or max_files < 0:
            raise ValueError("max_files must be a nonnegative integer")
        output = output / "diagnostics" / f"accuracy-first-{max_files}"
    if clean_output and (output.exists() or output.is_symlink()) and _is_indirection(output.lstat()):
        raise ValueError("Refusing to clean an output directory link or junction")
    with publication_lock(output):
        return _run_auto_chunking(input_dir, output, force, max_files,
                                  require_ready=require_ready, clean_output=clean_output,
                                  manual_edits_dir=manual_edits_dir, retry_delays=retry_delays, **chunker_options)


def _run_auto_chunking(input_dir=None, output_dir=None, force=False, max_files=None, *,
                       require_ready=None, clean_output=False, manual_edits_dir=None,
                       retry_delays=(30, 60, 120), **chunker_options):
    module = Path(__file__).resolve().parent
    incoming = Path(input_dir or module / "input")
    output = Path(output_dir or module / "output")
    if not incoming.is_dir():
        raise FileNotFoundError("Input directory not found: " + str(incoming))
    files = discover_documents(incoming)
    active_sources = {path.name for path in files}
    edits_dir = Path(manual_edits_dir) if manual_edits_dir is not None else output / "manual_edits"
    output.mkdir(parents=True, exist_ok=True)
    previous_files = set()
    if clean_output:
        if incoming.resolve().is_relative_to(output.resolve()):
            raise ValueError("Input must be outside the output directory when using --clean")
        cleanup_root, previous_files = _publication_cleanup_candidates(output)
        force = True
    if max_files is not None:
        files = files[:max_files]
    state_path = output / ".state_tracker.json"
    records = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") == "2.0":
                records = {name: value for name, value in state.get("records", {}).items() if name in active_sources}
        except (OSError, ValueError, TypeError, AttributeError):
            records = {}
    started = time.perf_counter()
    all_chunks = []
    completed = {}
    errors = {}
    aliases = {}
    refreshed = set()
    active_asset_paths = set()
    with LayoutAwareChunker(asset_dir=output / "assets", **chunker_options) as chunker:
        fingerprint = chunker.fingerprint()
        if require_ready is None:
            require_ready = True
        def process(path, attempt):
            source = path.name
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
                    chunks = chunker.chunk_file(path)
                    artifact = chunker.last_artifact
                    _validate_document(output, chunks, artifact, source, digest, fingerprint)
                    status = "SUCCESS"
                if require_ready and not artifact["quality"].get("index_ready"):
                    raise ValueError("Document has unresolved interpretation; index publication refused")
                if cached is None:
                    records[source] = _publish_document_files(output, chunks, artifact, digest, fingerprint)
                    atomic_json(state_path, {"schema_version": "2.0", "records": records})
                    refreshed.add(source)
                publication = records[source]
                base_output_file = publication["output_file"]
                edited = _apply_manual_edits(edits_dir / (source + ".json"), chunks, artifact, chunker.planner.budget)
                if edited is not None:
                    chunks, artifact = edited
                    _validate_document(output, chunks, artifact, source, digest, fingerprint)
                    publication = _publish_document_files(output, chunks, artifact, digest, fingerprint)
                    status = "MANUALLY_EDITED"
                for asset in artifact["document"]["assets"].values():
                    active_asset_paths.add(Path(asset["path"]).resolve())
                    if asset.get("visual_path"):
                        active_asset_paths.add(Path(asset["visual_path"]).resolve())
                aliases[artifact["document"]["document_id"]] = chunks
                completed[source] = (chunks, {
                    "file_name": source, "rel_path": source, "input_rel_path": path.relative_to(incoming).as_posix(),
                    "status": status, "attempts": attempt, "quality": artifact["quality"],
                    "base_output_file": base_output_file,
                    "duration_sec": round(time.perf_counter() - beginning, 3), **publication,
                })
                errors.pop(source, None)
                detail = ""
                if artifact["quality"].get("pages_requiring_ocr"):
                    detail = f" ({artifact['quality']['pages_requiring_ocr']} page(s) require OCR)"
                report(status + ": " + source + " -> " + str(len(chunks)) + " chunks" + detail)
            except Exception as error:
                errors[source] = {"file_name": source, "rel_path": source,
                                  "input_rel_path": path.relative_to(incoming).as_posix(),
                                  "status": "FAILED", "attempts": attempt,
                                  "error_type": type(error).__name__, "error": str(error), "chunk_count": 0}
                report(f"ATTEMPT {attempt} FAILED: {source} ({type(error).__name__})")

        for path in files:
            process(path, 1)
        for attempt, delay in enumerate(retry_delays, 2):
            if not errors:
                break
            report(f"Retrying {len(errors)} failed document(s) in {delay}s (attempt {attempt})")
            time.sleep(delay)
            for path in files:
                if path.name in errors:
                    process(path, attempt)
        manifests = []
        for path in files:
            if path.name in completed:
                chunks, record = completed[path.name]
                all_chunks.extend(chunks)
                manifests.append(record)
            else:
                manifests.append(errors[path.name])
        ids = [chunk["chunk_id"] for chunk in all_chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Cross-document chunk ID collision; index publication refused")
        manifest = {"module": "chunking", "schema_version": "2.0", "pipeline_fingerprint": fingerprint,
                    "profile": "accuracy", "total_files": len(files), "failed_files": len(errors),
                    "skipped_files": sum(item["status"] == "SKIPPED_ALREADY_PROCESSED" for item in manifests),
                    "new_files": sum(item["status"] in {"SUCCESS", "MANUALLY_EDITED"} for item in manifests),
                    "total_chunks": len(all_chunks), "total_elapsed_sec": round(time.perf_counter() - started, 3),
                    "index_ready": (not files or bool(completed))
                    and all(record["quality"]["index_ready"] for _, record in completed.values()),
                    "records": manifests}
        for metric in ("source_pages", "pages_without_native_text", "pages_requiring_ocr", "ocr_pages_processed", "reviewed_pages"):
            manifest[metric] = sum(item.get("quality", {}).get(metric, 0) for item in manifests)
        atomic_json(state_path, {"schema_version": "2.0", "records": records})
        if errors and not completed:
            atomic_json(output / "chunking_last_run.json", manifest)
            raise RuntimeError("Chunking failed for every document; previous publication retained")
        snapshot = publish_snapshot(output, all_chunks, manifest, aliases)
        if clean_output and not errors:
            active_files = {cleanup_root / snapshot, *active_asset_paths}
            active_files.update(cleanup_root / (identifier + ".json") for identifier in aliases)
            for record in manifests:
                for key in ("output_file", "artifact_file", "base_output_file"):
                    active_files.add(cleanup_root / record[key])
                active_files.add(cleanup_root / records[record["rel_path"]]["artifact_file"])
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
    parser.add_argument("--image-mode", choices=["text_context", "vision"], default="text_context",
                        help="Deprecated compatibility option. Standalone images are ignored in both modes.")
    parser.add_argument("--manual-edits", type=Path, default=None, help="Directory containing manual per-document corrections")
    parser.add_argument("--input", type=Path, default=None, help="Input directory")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    in_dir = args.input or module_dir / "input"
    run_auto_chunking(
        input_dir=in_dir,
        output_dir=args.output,
        force=args.force,
        clean_output=args.clean,
        max_files=args.max_files,
        image_mode=args.image_mode,
        manual_edits_dir=args.manual_edits,
    )
    output = args.output or module_dir / "output"
    if args.max_files is not None:
        output = output / "diagnostics" / f"accuracy-first-{args.max_files}"
    last_run = json.loads((output / "chunking_last_run.json").read_text(encoding="utf-8"))
    if last_run["failed_files"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
