"""Write and verify document artifacts and published chunk snapshots."""

import hashlib
import json
from pathlib import Path, PureWindowsPath

from .config.settings import get_config_section, get_rules
from .publication_io import atomic_bytes, atomic_json, content_hash, file_hash, read_verified_json
from .universal.planning.schema import validate_chunk_sequence

_PUB_RULES = get_rules("publication")
_PIPELINE_CFG = get_config_section("pipeline")

SUCCESS_STATUSES = set(_PUB_RULES["success_statuses"])
ASSET_REFERENCES = tuple(tuple(pair) for pair in _PUB_RULES["asset_references"])
_SCHEMA_VERSION = _PUB_RULES["schema_version"]
_DOCUMENT_ID_HEX_LENGTH = _PUB_RULES["document_id_hex_length"]
_PREFIX_HASH_LENGTH = _PUB_RULES["prefix_hash_length"]
_RETIRED_EXTRACTION_METHODS = tuple(_PUB_RULES["retired_extraction_methods"])
_REQUIRED_PDF_VISUAL_POLICY = _PUB_RULES["required_pdf_visual_policy"]
_MAX_VALIDATION_ISSUES_SHOWN = _PUB_RULES["max_validation_issues_shown"]
_MANIFEST_FILENAME = _PIPELINE_CFG["manifest_filename"]
_LAST_RUN_FILENAME = _PIPELINE_CFG["last_run_filename"]
_AGGREGATE_FILENAME = _PIPELINE_CFG["aggregate_filename"]


def publication_asset_path(output, filename):
    root = output.resolve()
    reference = Path(filename)
    windows_reference = PureWindowsPath(filename)
    if reference.is_absolute() or windows_reference.is_absolute():
        # Legacy publications stored absolute paths to their flat asset directory.
        reference = Path("assets") / windows_reference.name
    target = (root / reference).resolve()
    if not target.is_relative_to(root / "assets"):
        raise ValueError("Document asset must be inside the publication assets directory")
    return target


def use_relative_asset_paths(output, chunks, artifact):
    root = output.resolve()
    items = list(artifact["document"]["assets"].values())
    items.extend(image for chunk in chunks for image in chunk["images"] if isinstance(image, dict))
    for asset in items:
        for path_key, _ in ASSET_REFERENCES:
            if asset.get(path_key):
                path = Path(asset[path_key])
                if path.is_absolute():
                    path = path.resolve()
                    if not path.is_relative_to(root / "assets"):
                        raise ValueError("Document asset must be inside the publication assets directory")
                    asset[path_key] = path.relative_to(root).as_posix()


def _publication_path(output, filename, directory):
    relative = Path(filename)
    if relative.parent != Path(directory) or relative.suffix != ".json":
        raise ValueError("Unexpected chunking publication path: " + str(filename))
    path = output / relative
    if path.resolve().parent != output.resolve() / directory:
        raise ValueError("Chunking publication path leaves its expected directory")
    return path


def _snapshot_records(manifest):
    records = manifest.get("records")
    if not isinstance(records, list) or type(manifest.get("total_files")) is not int or len(records) != manifest["total_files"]:
        raise ValueError("Chunking publication document count mismatch")
    sources = set()
    failed = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid document record in chunking publication")
        source = record.get("rel_path")
        if not isinstance(source, str) or not source.strip() or source in sources:
            raise ValueError("Missing or duplicate source in chunking publication")
        sources.add(source)
        status = record.get("status")
        if status == "FAILED":
            failed += 1
        elif not isinstance(status, str) or status not in SUCCESS_STATUSES:
            raise ValueError("Invalid document status in chunking publication")
    if type(manifest.get("failed_files")) is not int or failed != manifest["failed_files"]:
        raise ValueError("Chunking publication failure count mismatch")
    return records


def _load_snapshot(output, manifest, require_ready):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _SCHEMA_VERSION or "active_aggregate_file" not in manifest:
        raise ValueError("Chunking output uses an obsolete publication contract; rerun chunking")
    if type(manifest.get("index_ready")) is not bool or (require_ready and not manifest["index_ready"]):
        raise ValueError("Published chunks are not ready for indexing")
    records = _snapshot_records(manifest)
    try:
        path = _publication_path(output, manifest["active_aggregate_file"], "snapshots")
        digest = manifest["aggregate_sha256"]
        fingerprint = manifest["pipeline_fingerprint"]
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid chunking publication metadata") from error
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Missing chunking publication fingerprint")
    chunks = read_verified_json(path, digest)
    if not isinstance(chunks, list) or type(manifest.get("total_chunks")) is not int or len(chunks) != manifest["total_chunks"]:
        raise ValueError("Published aggregate count mismatch")
    offset = 0
    for record in records:
        if record["status"] == "FAILED":
            continue
        cached = _read_cached(output, record, record["rel_path"], record.get("input_sha256"), fingerprint)
        if cached is None:
            raise ValueError("Published document artifact failed integrity checks: " + record["rel_path"])
        document_chunks, artifact = cached
        extraction = artifact["document"].get("extraction_method")
        if require_ready and (record["status"] == "MANUALLY_EDITED" or artifact["quality"].get("manually_edited_chunks")
                              or extraction in _RETIRED_EXTRACTION_METHODS):
            raise ValueError("Published document uses a retired content override; rerun chunking: " + record["rel_path"])
        if (require_ready and Path(record["rel_path"]).suffix.lower() == ".pdf"
                and artifact["quality"].get("visual_policy") != _REQUIRED_PDF_VISUAL_POLICY):
            raise ValueError("Published PDF uses a retired extraction policy; rerun chunking: " + record["rel_path"])
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


def _validate_document(output, chunks, artifact, source, digest, fingerprint):
    if not isinstance(artifact, dict):
        raise ValueError("Missing document validation artifact")
    document = artifact["document"]
    if document["source"] != source or document["document_version"] != digest:
        raise ValueError("Document identity or version does not match the input")
    document_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:_DOCUMENT_ID_HEX_LENGTH]
    if document["document_id"] != document_id or artifact["pipeline_fingerprint"] != fingerprint:
        raise ValueError("Document artifact does not match this pipeline run")
    if not isinstance(artifact.get("quality"), dict) or type(artifact["quality"].get("index_ready")) is not bool:
        raise ValueError("Document artifact is missing its index readiness result")
    valid, issues = validate_chunk_sequence(chunks)
    if not valid:
        raise ValueError("Invalid chunk sequence: " + "; ".join(issues[:_MAX_VALIDATION_ISSUES_SHOWN]))
    for chunk in chunks:
        metadata = chunk["business_metadata"]
        if (chunk["source"] != source or metadata.get("document_id") != document_id
                or metadata.get("document_version") != digest or metadata.get("pipeline_fingerprint") != fingerprint
                or chunk["parent_id"] not in artifact["parents"]):
            raise ValueError("Chunk identity, parent or pipeline fingerprint does not match the document")
        if metadata.get("policy_revision") != document.get("policy_revision"):
            raise ValueError("Chunk policy revision does not match the document")
    for asset in document["assets"].values():
        for path_key, hash_key in ASSET_REFERENCES:
            if path_key == "visual_path" and not asset.get(path_key):
                continue
            asset_path = publication_asset_path(output, asset[path_key])
            if file_hash(asset_path) != asset[hash_key]:
                raise ValueError("Document asset failed integrity checks")


def _read_cached(output, record, source, digest, fingerprint):
    if (not isinstance(record, dict) or not isinstance(digest, str) or not digest
            or record.get("input_sha256") != digest or record.get("pipeline_fingerprint") != fingerprint):
        return None
    try:
        chunk_path = _publication_path(output, record["output_file"], "chunks_per_file")
        artifact_path = _publication_path(output, record["artifact_file"], "artifacts")
        chunks = read_verified_json(chunk_path, record["output_sha256"])
        artifact = read_verified_json(artifact_path, record["artifact_sha256"])
        _validate_document(output, chunks, artifact, source, digest, fingerprint)
        if record.get("policy_revision") != artifact["document"].get("policy_revision"):
            return None
        if type(record.get("chunk_count")) is not int or len(chunks) != record["chunk_count"]:
            return None
        return chunks, artifact
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _publish_document_files(output, chunks, artifact, digest, fingerprint):
    n = _PREFIX_HASH_LENGTH
    prefix = artifact["document"]["document_id"] + "_" + digest[:n] + "_" + fingerprint[:n]
    chunk_file = Path("chunks_per_file") / (prefix + "_" + content_hash(chunks)[:n] + ".json")
    artifact_file = Path("artifacts") / (prefix + "_" + content_hash(artifact)[:n] + ".json")
    for path, value in ((chunk_file, chunks), (artifact_file, artifact)):
        atomic_json(_publication_path(output, path, path.parent.name), value)
    record = {"input_sha256": digest, "pipeline_fingerprint": fingerprint,
              "output_file": chunk_file.as_posix(), "artifact_file": artifact_file.as_posix(),
              "output_sha256": file_hash(output / chunk_file),
              "artifact_sha256": file_hash(output / artifact_file), "chunk_count": len(chunks)}
    if "policy_revision" in artifact["document"]:
        record["policy_revision"] = artifact["document"]["policy_revision"]
    return record


def publish_snapshot(output, chunks, manifest, aliases):
    aggregate = json.dumps(chunks, ensure_ascii=False, indent=2).encode("utf-8")
    digest = hashlib.sha256(aggregate).hexdigest()
    snapshot = Path("snapshots") / (digest + ".json")
    snapshot_file = _publication_path(output, snapshot, "snapshots")
    if not snapshot_file.is_file() or file_hash(snapshot_file) != digest:
        atomic_bytes(snapshot_file, aggregate)
    manifest["active_aggregate_file"] = snapshot.as_posix()
    manifest["aggregate_sha256"] = digest
    atomic_bytes(output / _AGGREGATE_FILENAME, aggregate)
    for document_id, document_chunks in aliases.items():
        atomic_json(output / (document_id + ".json"), document_chunks)
    # The manifest commits a snapshot only after its artifacts exist.
    atomic_json(output / _MANIFEST_FILENAME, manifest)
    atomic_json(output / _LAST_RUN_FILENAME, manifest)
    return snapshot
