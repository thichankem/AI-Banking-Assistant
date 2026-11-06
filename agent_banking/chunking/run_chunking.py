"""Versioned DOCX ingestion with verified caches and atomic index publication."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.chunking.unified_chunk.layout_aware_chunker import LayoutAwareChunker
from agent_banking.chunking.universal.schema_assembler import validate_chunk_sequence


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
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
        document_chunks = cached[0]
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
                      profile=None, require_ready=None, **chunker_options):
    module = Path(__file__).resolve().parent
    incoming = Path(input_dir or module / "input")
    output = Path(output_dir or module / "output")
    if not incoming.is_dir():
        raise FileNotFoundError("Input directory not found: " + str(incoming))
    if max_files is not None and max_files < 0:
        raise ValueError("max_files cannot be negative")
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in incoming.rglob("*") if path.is_file() and path.suffix.lower() == ".docx" and not path.name.startswith("~$"))
    active_sources = {path.relative_to(incoming).as_posix() for path in files}
    if max_files is not None:
        files = files[:max_files]
    state_path = output / ".state_tracker.json"
    records = {}
    if state_path.exists() and not force:
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
            digest = file_hash(path)
            record = records.get(source, {})
            cached = None
            if not force:
                cached = _read_cached(output, record, source, digest, fingerprint)
            beginning = time.perf_counter()
            try:
                if cached is not None:
                    chunks, artifact = cached
                    status = "SKIPPED_ALREADY_PROCESSED"
                else:
                    chunks = chunker.chunk_docx(path, source_name=source)
                    artifact = chunker.last_artifact
                    if artifact is None:
                        raise ValueError("Missing document validation artifact")
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
                aliases[document_id] = chunks
                records[source] = {"input_sha256": digest, "pipeline_fingerprint": fingerprint,
                                   "output_file": chunk_file.as_posix(), "artifact_file": artifact_file.as_posix(),
                                   "output_sha256": file_hash(output / chunk_file),
                                   "artifact_sha256": file_hash(output / artifact_file), "chunk_count": len(chunks)}
                all_chunks.extend(chunks)
                manifests.append({"file_name": path.name, "rel_path": source, "status": status,
                                  "chunk_count": len(chunks), "quality": artifact["quality"],
                                  "duration_sec": round(time.perf_counter() - beginning, 3), **records[source]})
                report(status + ": " + source + " -> " + str(len(chunks)) + " chunks")
            except Exception as error:
                failures.append(source)
                records.pop(source, None)
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
                    "index_ready": not failures and all(item.get("quality", {}).get("index_ready", False) for item in manifests),
                    "records": manifests}
        atomic_json(state_path, {"schema_version": "2.0", "records": records})
        if failures:
            atomic_json(output / "chunking_last_run.json", manifest)
            raise RuntimeError("Chunking failed for " + str(len(failures)) + " document(s); aggregate was not replaced")
        # Rebuild from precisely the current selected input set, including empty
        # sets and removals. Never return an unscoped old aggregate.
        snapshot = Path("snapshots") / (content_hash(all_chunks) + ".json")
        if not (output / snapshot).exists():
            atomic_json(output / snapshot, all_chunks)
        manifest["active_aggregate_file"] = snapshot.as_posix()
        manifest["aggregate_sha256"] = file_hash(output / snapshot)
        atomic_json(output / "all_chunks.json", all_chunks)
        for document_id, chunks in aliases.items():
            atomic_json(output / (document_id + ".json"), chunks)
        # The manifest is the commit pointer. Failed attempts retain the previous
        # manifest, snapshot and immutable document artifacts intact.
        atomic_json(output / "chunking_manifest.json", manifest)
        atomic_json(output / "chunking_last_run.json", manifest)
    return all_chunks


if __name__ == "__main__":
    run_auto_chunking()
