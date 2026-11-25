"""Chunk and index DS-01 multi-version Excel fixtures into qdrant_excel_versions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_banking.chunking.universal.pipeline import LayoutAwareChunker
from agent_banking.core.settings import Config
from agent_banking.pipeline import BankingRAGPipeline

FIXTURES_DIR = Path("dataset/A_Bank_eval_excel_ops/fixtures")
DEFAULT_PUB_DIR = Path("agent_banking/outputs/ds01_excel_versions/knowledge")
DEFAULT_INDEX_DIR = Path("agent_banking/vector_db/qdrant_excel_versions")


def index_excel_versions(
    fixtures_dir: Path = FIXTURES_DIR,
    pub_dir: Path = DEFAULT_PUB_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> dict[str, Any]:
    """Chunk versioned fixtures, stamp revisions metadata, and ingest into isolated Qdrant."""
    revs_data = json.loads((fixtures_dir / "revisions.json").read_text(encoding="utf-8"))
    rev_by_file: dict[str, dict[str, Any]] = {}
    for entry in revs_data.get("revisions", []):
        rev_by_file[str(entry["file"])] = entry

    pub_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    all_chunks: list[dict[str, Any]] = []

    for xlsx_path in sorted(fixtures_dir.glob("*.xlsx")):
        fname = xlsx_path.name
        rev_meta = rev_by_file.get(fname, {})
        source_rel = f"A_Bank/8. Biểu phí dịch vụ/{fname}"
        asset_dir = pub_dir / xlsx_path.stem / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        with LayoutAwareChunker(config=cfg, asset_dir=asset_dir) as chunker:
            chunks = chunker.chunk_file(xlsx_path, source_name=source_rel)
        for chunk in chunks:
            bm = chunk.get("business_metadata")
            if not isinstance(bm, dict):
                bm = {}
                chunk["business_metadata"] = bm
            ex_rev = bm.get("excel_revision")
            if not isinstance(ex_rev, dict):
                ex_rev = {}
                bm["excel_revision"] = ex_rev
            ex_rev["business_version"] = rev_meta.get("business_version")
            ex_rev["effective_from"] = rev_meta.get("effective_from")
            ex_rev["effective_to"] = rev_meta.get("effective_to")
            ex_rev["effective_date_status"] = rev_meta.get("effective_date_status", "unknown")
        out_file = pub_dir / xlsx_path.stem / "all_chunks.json"
        out_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        all_chunks.extend(chunks)
        print(f"Chunked {len(chunks)} chunks for {fname}")

    index_dir.mkdir(parents=True, exist_ok=True)
    cfg.QDRANT_STORAGE_PATH = str(index_dir)
    cfg.ABANK_QDRANT_STORAGE_PATH = str(index_dir)
    rag = BankingRAGPipeline(config=cfg)
    try:
        rag.vector_store.recreate_collection()
        embedded_chunks = rag.embedder.embed_chunks(all_chunks)
        rag.vector_store.upsert_chunks(embedded_chunks)
        summary = {
            "index_dir": str(index_dir).replace("\\", "/"),
            "documents": len(rev_by_file),
            "chunks_indexed": len(all_chunks),
            "vector_store_count": rag.vector_store.count(),
        }
    finally:
        rag.close()
    (index_dir / "versions_index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build isolated Qdrant index for DS-01 version fixtures")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--pub", type=Path, default=DEFAULT_PUB_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args()
    out = index_excel_versions(args.fixtures, args.pub, args.index)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
