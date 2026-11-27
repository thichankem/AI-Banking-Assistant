"""Backfill source revisions and extraction quality in an existing local index."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from agent_banking.chunking.policy_revision import attach_policy_revision
from agent_banking.chunking.publication_io import atomic_json, file_hash, read_verified_json
from agent_banking.chunking.universal.enrichment.source_quality import annotate_page_quality
from agent_banking.chunking.universal.planning.schema import payload_index_key
from agent_banking.core.document_revisions import archive_document, is_document_version
from agent_banking.core.config.settings import CoreConfig
from agent_banking.core.settings import Config
from agent_banking.vector_db.qdrant_store import QdrantBankingStore, configured_qdrant_connection

_EXCLUDED_ROLE = CoreConfig.CORE_RAG_PIPELINE["excluded_retrieval_role"]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_publication_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_dir():
        return path.resolve(strict=True)
    if not path.is_absolute():
        rel_cand = (_PROJECT_ROOT / path).resolve()
        if rel_cand.is_dir():
            return rel_cand
    norm = str(raw_path).replace("\\", "/")
    marker = "/agent_banking/"
    pos = norm.find(marker)
    if pos >= 0:
        mapped = (_PROJECT_ROOT / norm[pos + 1:]).resolve()
        if mapped.is_dir():
            return mapped
    return path.resolve(strict=True)


def _publication_artifacts(inventory):
    for publication in inventory["publications"]:
        root = _resolve_publication_dir(publication["publication"])
        manifest = json.loads((root / "chunking_manifest.json").read_bytes())
        for record in manifest["records"]:
            if record["status"] == "FAILED" or record["rel_path"] not in publication["sources"]:
                continue
            artifact_path = (root / record["artifact_file"]).resolve(strict=True)
            chunk_path = (root / record["output_file"]).resolve(strict=True)
            if not artifact_path.is_relative_to(root) or not chunk_path.is_relative_to(root):
                raise ValueError("Invalid source artifact location")
            artifact = read_verified_json(artifact_path, record["artifact_sha256"])
            chunks = read_verified_json(chunk_path, record["output_sha256"])
            document = artifact["document"]
            if (document["source"] != record["rel_path"] or document["document_version"] != record["input_sha256"]):
                raise ValueError("Source artifact does not match the publication")
            searchable = {chunk["chunk_id"]: chunk["content"] for chunk in chunks
                          if chunk.get("business_metadata", {}).get("retrieval", {}).get("role") != _EXCLUDED_ROLE}
            yield (document["source"], document["document_version"]), artifact, searchable


def backfill(config, store, *, apply=False):
    index = Path(store.storage_path)
    inventory_path = index / "knowledge_inventory.json"
    inventory = json.loads(inventory_path.read_bytes())
    source_root = Path(config.CHUNKING_INPUT_DIR).resolve()
    group_map = store.source_groups()
    updates = {}
    sources = []
    policy_count = 0
    statuses = defaultdict(int)
    uncertain_count = 0
    partial_pages = 0
    searchable_counts = {}
    for (source, version), artifact, searchable in _publication_artifacts(inventory):
        if not is_document_version(version) or inventory["source_hashes"].get(source) != version:
            raise ValueError("Publication digest does not match the index inventory")
        input_path = (source_root / source).resolve(strict=True)
        if not input_path.is_relative_to(source_root) or file_hash(input_path) != version:
            raise ValueError("The original source file has changed since publication")
        chunks = [chunk for chunk in group_map.get(source, ()) if chunk["business_metadata"].get("document_version") == version]
        if not chunks:
            raise ValueError("Publication document is absent from the configured index")
        if {chunk["chunk_id"]: chunk["content"] for chunk in chunks} != searchable:
            raise ValueError(f"Published passages differ from the active index for {source}. "
                             "Reindex this publication before backfilling metadata.")
        searchable_counts[source] = len(searchable)
        sources.append((input_path, version))
        document = dict(artifact["document"])
        partial_pages += artifact["quality"].get("ocr_partial_pages", 0)
        annotated = annotate_page_quality([
            {**chunk, "business_metadata": dict(chunk["business_metadata"])} for chunk in chunks
        ], document)
        existing = chunks[0]["business_metadata"].get("policy_revision") or document.get("policy_revision") or {}
        revision = attach_policy_revision(document, annotated, existing)
        if revision:
            policy_count += 1
            statuses[revision["effective_date_status"]] += 1
        for old, new in zip(chunks, annotated, strict=True):
            old_metadata, new_metadata = old["business_metadata"], new["business_metadata"]
            quality = dict(new_metadata.get("quality", {}))
            for field in ("excel_formula_cells_skipped", "ocr_partial_pages", "ocr_failed_page_numbers"):
                if artifact["quality"].get(field):
                    quality[field] = artifact["quality"][field]
            if quality.get("ocr_uncertain_pages") or quality.get("ocr_uncertain_images"):
                uncertain_count += 1
            fields = {}
            if quality and quality != old_metadata.get("quality"):
                fields["quality"] = quality
            if revision and revision != old_metadata.get("policy_revision"):
                fields["policy_revision"] = revision
            if fields:
                updates[payload_index_key(old)] = fields
    if len({path for path, _ in sources}) != len(inventory["source_hashes"]):
        raise ValueError("Publications do not cover the complete source inventory")
    report = {"sources_checked": len(sources), "policy_documents": policy_count,
              "policy_date_status": dict(statuses), "ocr_uncertain_passages": uncertain_count,
              "ocr_partial_pages": partial_pages,
              "points_to_update": len(updates), "applied": apply}
    if apply:
        for path, version in sources:
            archive_document(index / "revisions", path, version)
        store.update_document_metadata(updates)
        publications = [{**publication, "chunks": sum(searchable_counts[source] for source in publication["sources"])}
                        for publication in inventory["publications"]]
        atomic_json(inventory_path, {**inventory, "chunks": store.count(), "publications": publications,
                                     "ocr_uncertain_passages": uncertain_count,
                                     "ocr_partial_pages": partial_pages,
                                     "policy_documents": policy_count, "policy_date_status": dict(statuses)})
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Persist the verified metadata and original document revisions")
    args = parser.parse_args()
    config = Config()
    store = QdrantBankingStore(collection_name=config.QDRANT_COLLECTION, dimension=config.EMBEDDING_DIM,
                              **configured_qdrant_connection(config))
    try:
        print(json.dumps(backfill(config, store, apply=args.apply)))
    finally:
        store.close()


if __name__ == "__main__":
    main()
