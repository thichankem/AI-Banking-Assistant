"""Versioned PDF/DOCX ingestion with verified caches and atomic publication."""
import hashlib
import json
import os
import stat
from pathlib import Path
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.chunking.unified_chunk.layout_aware_chunker import LayoutAwareChunker
from agent_banking.chunking.universal.schema_assembler import validate_chunk_sequence



def clean_output_directory(output_dir):
    """Remove pipeline-owned publication artifacts, preserving unrelated data.

    ``output`` can also contain benchmarks, downloaded models or user files.
    Their presence is never permission to erase them. Manifest paths are not
    deletion targets: legacy aliases are derived from recorded source names.
    Refuse directory links/junctions and validate the entire owned tree before
    deleting any item, then recheck each path immediately before removing it.
    """
    output = Path(output_dir)
    if not output.exists():
        return

    def is_indirection(info):
        return stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )

    if is_indirection(output.lstat()):
        raise ValueError("Refusing to clean an output directory link or junction")
    root = output.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Output cleanup requires a directory")

    def checked(path):
        info = path.lstat()
        if is_indirection(info):
            raise ValueError("Refusing to clean a publication link or junction: " + str(path))
        try:
            relative = path.resolve(strict=True).relative_to(root)
        except ValueError:
            raise ValueError("Publication cleanup target escapes output: " + str(path)) from None
        if not relative.parts:
            raise ValueError("Publication cleanup cannot delete its output root")
        if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
            raise ValueError("Unsupported publication cleanup target: " + str(path))
        return info

    names = {"chunking_manifest.json", "chunking_last_run.json", ".state_tracker.json", "all_chunks.json",
             "snapshots", "artifacts", "assets", "chunks_per_file", "runtime"}
    manifest_path = root / "chunking_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        info = checked(manifest_path)
        if stat.S_ISREG(info.st_mode):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = None
            if isinstance(manifest, dict) and manifest.get("module") == "chunking" and manifest.get("schema_version") == "2.0":
                records = manifest.get("records", [])
                if isinstance(records, list):
                    for record in records:
                        if isinstance(record, dict):
                            source = record.get("rel_path")
                        else:
                            source = None
                        if isinstance(source, str) and source:
                            identifier = hashlib.sha256(source.replace("\\", "/").encode()).hexdigest()[:20]
                            names.add(identifier + ".json")

    targets = []
    for name in sorted(names):
        path = root / name
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        targets.append(path)

    def validate_tree(path):
        if stat.S_ISDIR(checked(path).st_mode):
            for child in path.iterdir():
                validate_tree(child)

    # Detect unsafe trees before removing even the first manifest or snapshot.
    for path in targets:
        validate_tree(path)

    def remove_owned(path):
        if stat.S_ISDIR(checked(path).st_mode):
            for child in path.iterdir():
                remove_owned(child)
            checked(path)
            path.rmdir()
        else:
            checked(path)
            path.unlink()

    for path in targets:
        remove_owned(path)

def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def content_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def report(message):
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(message.encode(encoding, errors="backslashreplace").decode(encoding), flush=True)


def load_published_chunks(output_dir, require_ready=True):
    output = Path(output_dir)
    manifest = json.loads((output / "chunking_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "2.0" or "active_aggregate_file" not in manifest:
        raise ValueError("Chunking output uses an obsolete publication contract; rerun chunking")
    if require_ready and not manifest.get("index_ready"):
        raise ValueError("Published chunks are not ready for indexing")
    path = output / manifest["active_aggregate_file"]
    path.resolve().relative_to(output.resolve())
    if file_hash(path) != manifest["aggregate_sha256"]:
        raise ValueError("Published aggregate checksum mismatch")
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if len(chunks) != manifest["total_chunks"]:
        raise ValueError("Published aggregate count mismatch")
    offset = 0
    for record in manifest.get("records", []):
        cached = _read_cached(output, record, record["rel_path"], record["input_sha256"], manifest["pipeline_fingerprint"])
        if cached is None:
            raise ValueError("Published document artifact failed integrity checks: " + record["rel_path"])
        document_chunks, artifact = cached
        if require_ready and not artifact["quality"].get("index_ready"):
            raise ValueError("Published document is not ready for indexing: " + record["rel_path"])
        if chunks[offset:offset + len(document_chunks)] != document_chunks:
            raise ValueError("Published aggregate disagrees with its document artifacts")
        offset += len(document_chunks)
    if offset != len(chunks):
        raise ValueError("Published aggregate contains unaccounted document chunks")
    identifiers = [chunk["chunk_id"] for chunk in chunks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Published aggregate contains duplicate chunk IDs")
    return chunks


def _read_cached(output, record, source, digest, fingerprint):
    if record.get("input_sha256") != digest or record.get("pipeline_fingerprint") != fingerprint:
        return None
    try:
        chunk_path = output / record["output_file"]
        artifact_path = output / record["artifact_file"]
        # Do not follow arbitrary paths from a corrupt state file.
        chunk_path.resolve().relative_to(output.resolve())
        artifact_path.resolve().relative_to(output.resolve())
        if file_hash(chunk_path) != record["output_sha256"] or file_hash(artifact_path) != record["artifact_sha256"]:
            return None
        chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        document = artifact["document"]
        if document.get("source") != source or document.get("document_version") != digest or artifact.get("pipeline_fingerprint") != fingerprint:
            return None
        if not isinstance(artifact.get("quality"), dict):
            return None
        valid, issues = validate_chunk_sequence(chunks)
        if not valid or len(chunks) != record["chunk_count"]:
            return None
        if any(chunk["source"] != source or chunk["business_metadata"].get("document_version") != digest
               or chunk["business_metadata"].get("pipeline_fingerprint") != fingerprint
               or chunk["parent_id"] not in artifact["parents"] for chunk in chunks):
            return None
        for asset in artifact["document"]["assets"].values():
            if not Path(asset["path"]).is_file() or file_hash(asset["path"]) != asset["sha256"]:
                return None
            if asset.get("visual_path") and (not Path(asset["visual_path"]).is_file() or file_hash(asset["visual_path"]) != asset["visual_sha256"]):
                return None
        return chunks, artifact
    except (OSError, ValueError, KeyError, TypeError):
        return None


def run_auto_chunking(input_dir=None, output_dir=None, force=False, max_files=None,
                      profile=None, require_ready=None, clean_output=False, **chunker_options):
    from agent_banking.chunking.universal.publication_lock import publication_lock
    output = Path(output_dir or Path(__file__).resolve().parent / "output")
    with publication_lock(output):
        return _run_auto_chunking(input_dir, output, force, max_files, profile, require_ready, clean_output, **chunker_options)


def _run_auto_chunking(input_dir=None, output_dir=None, force=False, max_files=None,
                       profile=None, require_ready=None, clean_output=False, **chunker_options):
    module = Path(__file__).resolve().parent
    incoming = Path(input_dir or module / "input")
    output = Path(output_dir or module / "output")
    if not incoming.is_dir():
        raise FileNotFoundError("Input directory not found: " + str(incoming))
    if max_files is not None and max_files < 0:
        raise ValueError("max_files cannot be negative")
    output.mkdir(parents=True, exist_ok=True)
    if clean_output:
        if incoming.resolve().is_relative_to(output.resolve()):
            raise ValueError("Input must be outside the output directory when using --clean")
        clean_output_directory(output)
        force = True
        report("[Pipeline] Cleaned publisher-owned publication data; retained benchmark, models, and other files.")
    files = sorted(path for path in incoming.rglob("*") if path.is_file() and path.suffix.lower() in {".docx", ".pdf"} and not path.name.startswith("~$"))
    active_sources = {path.relative_to(incoming).as_posix() for path in files}
    if max_files is not None:
        files = files[:max_files]
    state_path = output / ".state_tracker.json"
    records = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") == "2.0":
                records = {name: value for name, value in state.get("records", {}).items() if name in active_sources}
        except (ValueError, TypeError, AttributeError):
            records = {}
    started = time.perf_counter()
    all_chunks = []
    manifests = []
    failures = []
    aliases = {}
    with LayoutAwareChunker(profile=profile, asset_dir=output / "assets", **chunker_options) as chunker:
        fingerprint = chunker.fingerprint()
        if require_ready is None:
            require_ready = chunker.docx_chunker.profile == "accuracy"
        for path in files:
            source = path.relative_to(incoming).as_posix()
            beginning = time.perf_counter()
            try:
                digest = file_hash(path)
                record = records.get(source, {})
                if force:
                    cached = None
                else:
                    cached = _read_cached(output, record, source, digest, fingerprint)
                if cached is not None:
                    chunks, artifact = cached
                    status = "SKIPPED_ALREADY_PROCESSED"
                else:
                    chunks = chunker.chunk_file(path, source_name=source)
                    artifact = chunker.last_artifact
                    if artifact is None:
                        raise ValueError("Missing document validation artifact")
                    if artifact["document"]["document_version"] != digest:
                        raise ValueError("Input changed while being read; retry the document")
                    status = "SUCCESS"
                if require_ready and not artifact["quality"].get("index_ready"):
                    raise ValueError("Document has unresolved interpretation; index publication refused")
                document_id = artifact["document"]["document_id"]
                prefix = document_id + "_" + digest[:16] + "_" + fingerprint[:16]
                chunk_file = Path("chunks_per_file") / (prefix + "_" + content_hash(chunks)[:16] + ".json")
                artifact_file = Path("artifacts") / (prefix + "_" + content_hash(artifact)[:16] + ".json")
                if cached is None:
                    atomic_json(output / chunk_file, chunks)
                    atomic_json(output / artifact_file, artifact)
                else:
                    chunk_file = Path(record["output_file"])
                    artifact_file = Path(record["artifact_file"])
                records[source] = {"input_sha256": digest, "pipeline_fingerprint": fingerprint,
                                   "output_file": chunk_file.as_posix(), "artifact_file": artifact_file.as_posix(),
                                   "output_sha256": file_hash(output / chunk_file),
                                   "artifact_sha256": file_hash(output / artifact_file), "chunk_count": len(chunks)}
                if cached is None:
                    # An interrupted scan batch can resume from each completed document.
                    atomic_json(state_path, {"schema_version": "2.0", "records": records})
                aliases[document_id] = chunks
                all_chunks.extend(chunks)
                manifests.append({"file_name": path.name, "rel_path": source, "status": status,
                                  "chunk_count": len(chunks), "quality": artifact["quality"],
                                  "duration_sec": round(time.perf_counter() - beginning, 3), **records[source]})
                detail = ""
                if artifact["quality"].get("pages_requiring_ocr"):
                    detail = f" ({artifact['quality']['pages_requiring_ocr']} page(s) require OCR)"
                report(status + ": " + source + " -> " + str(len(chunks)) + " chunks" + detail)
            except Exception as error:
                failures.append(source)
                manifests.append({"file_name": path.name, "rel_path": source, "status": "FAILED",
                                  "error_type": type(error).__name__, "error": str(error), "chunk_count": 0})
                report("FAILED: " + source + " (" + type(error).__name__ + ")")
        ids = [chunk["chunk_id"] for chunk in all_chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Cross-document chunk ID collision; index publication refused")
        manifest = {"module": "chunking", "schema_version": "2.0", "pipeline_fingerprint": fingerprint,
                    "profile": chunker.docx_chunker.profile, "total_files": len(files), "failed_files": len(failures),
                    "skipped_files": sum(item["status"] == "SKIPPED_ALREADY_PROCESSED" for item in manifests),
                    "new_files": sum(item["status"] == "SUCCESS" for item in manifests),
                    "total_chunks": len(all_chunks), "total_elapsed_sec": round(time.perf_counter() - started, 3),
                    "index_ready": bool(manifests) and not failures and all(item.get("quality", {}).get("index_ready", False) for item in manifests),
                    "records": manifests}
        for metric in ("source_pages", "pages_without_native_text", "pages_requiring_ocr", "reviewed_pages"):
            manifest[metric] = sum(item.get("quality", {}).get(metric, 0) for item in manifests)
        atomic_json(state_path, {"schema_version": "2.0", "records": records})
        if failures:
            atomic_json(output / "chunking_last_run.json", manifest)
            raise RuntimeError("Chunking failed for " + str(len(failures)) + " document(s); aggregate was not replaced")
        # Rebuild from precisely the current selected input set, including empty
        # sets and removals. Never return an unscoped old aggregate.
        aggregate = json.dumps(all_chunks, ensure_ascii=False, indent=2).encode("utf-8")
        aggregate_hash = hashlib.sha256(aggregate).hexdigest()
        snapshot = Path("snapshots") / (aggregate_hash + ".json")
        if not (output / snapshot).is_file() or file_hash(output / snapshot) != aggregate_hash:
            atomic_bytes(output / snapshot, aggregate)
        manifest["active_aggregate_file"] = snapshot.as_posix()
        manifest["aggregate_sha256"] = aggregate_hash
        atomic_bytes(output / "all_chunks.json", aggregate)
        for document_id, chunks in aliases.items():
            atomic_json(output / (document_id + ".json"), chunks)
        # The manifest is the commit pointer. Failed attempts retain the previous
        # manifest, snapshot and immutable document artifacts intact.
        atomic_json(output / "chunking_manifest.json", manifest)
        atomic_json(output / "chunking_last_run.json", manifest)
    return all_chunks


def main(default_profile="native"):
    import argparse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run Chunking Pipeline")
    cleaning = parser.add_mutually_exclusive_group()
    cleaning.add_argument("--clean", action="store_true", help="Delete existing pipeline-owned publications before running")
    cleaning.add_argument("--no-clean", action="store_true", help="Preserve existing publications (default)")
    rerun = parser.add_mutually_exclusive_group()
    rerun.add_argument("--force", action="store_true", help="Rebuild per-document artifacts using the model cache")
    rerun.add_argument("--keep-cache", action="store_true", help="Reuse validated document artifacts (default)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--profile", choices=["accuracy", "native"], default=default_profile, help="Execution profile")
    parser.add_argument("--image-mode", choices=["text_context", "vision"], default="vision", help="Image processing mode")
    parser.add_argument("--input", type=Path, default=None, help="Input directory")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    in_dir = args.input or module_dir / "input"
    out_dir = args.output
    if out_dir is None:
        out_dir = module_dir / "output"
        if args.profile == "native" or args.max_files is not None:
            if args.max_files is None:
                scope = "all"
            else:
                scope = f"first-{args.max_files}"
            out_dir = out_dir / "diagnostics" / f"{args.profile}-{scope}"

    run_auto_chunking(
        input_dir=in_dir,
        output_dir=out_dir,
        force=args.force,
        clean_output=args.clean,
        max_files=args.max_files,
        profile=args.profile,
        image_mode=args.image_mode,
    )


if __name__ == "__main__":
    main()
