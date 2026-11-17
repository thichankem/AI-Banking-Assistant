"""Build an isolated index only from a complete, verified A-Bank publication."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

from agent_banking.chunking.publication_io import file_hash
from agent_banking.chunking.run_chunking import load_published_chunks
from agent_banking.chunking.universal.extraction.sources import discover_documents, document_source_root
from agent_banking.core.settings import Config
from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.vector_db.qdrant_store import publication_scope


def build_index(publication: Path, index: Path, corpus: Path) -> dict:
    config = Config()
    publication, index, corpus = (path.resolve() for path in (publication, index, corpus))
    current = (config.ROOT_DIR / config.QDRANT_STORAGE_PATH).resolve()
    if index.is_relative_to(current) or current.is_relative_to(index):
        raise ValueError("Supply a separate index directory")
    if index.exists() and any(index.iterdir()):
        raise ValueError("Index destination must be empty")
    _, manifest = load_published_chunks(publication, require_ready=True, return_manifest=True)
    corpus_root = document_source_root(corpus, config.CHUNKING_INPUT_DIR)
    expected = {path.relative_to(corpus_root).as_posix(): path for path in discover_documents(corpus)}
    records = {item["rel_path"]: item for item in manifest["records"]}
    if not expected or set(expected) != set(records) or manifest["failed_files"]:
        raise ValueError("A-Bank publication does not cover all source documents")
    if any(records[name]["input_sha256"] != file_hash(path) for name, path in expected.items()):
        raise ValueError("A source document changed after publication")
    config.ABANK_QDRANT_STORAGE_PATH = str(index)
    config.TRACE_DIR = index.parent / "index-traces"
    pipeline = BankingRAGPipeline(config=config)
    try:
        count = pipeline.ingest_published_chunks(publication, require_ready=True)
        points = pipeline.vector_store.count()
        if not count or points != count:
            raise ValueError("Unexpected point count in isolated collection")
    finally:
        pipeline.close()
    sqlite_path = index / "collection" / config.QDRANT_COLLECTION / "storage.sqlite"
    with sqlite_path.open("rb") as contents:
        digest = file_digest(contents, "sha256").hexdigest()
    report = {"collection": config.QDRANT_COLLECTION, "points": points, "documents": len(records),
              "scope": publication_scope(publication), "index_digest": digest,
              "publication_digest": manifest["aggregate_sha256"]}
    (index / "harness_snapshot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Config.CHUNKING_DATASET_DIR)
    args = parser.parse_args()
    print(json.dumps(build_index(args.publication, args.index, args.corpus)))
