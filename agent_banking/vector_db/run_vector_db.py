"""
Automated Vector DB & BM25 Indexing Runner (Auto Vector DB Node with Deduplication).
===================================================================================
Features:
1. Ingests from agent_banking/embedding/output/embedded_chunks.json.
2. Fallback input from agent_banking/vector_db/input/embedded_chunks.json.
3. Computes SHA-256 hash across chunk content and vector properties.
4. Checks StateTracker: Skips previously indexed chunks.
5. Supports clean rebuild with --force flag.
6. Emits indexing_manifest.json report with canonical fields.

MANDATORY INVARIANTS:
1. 0% REGEX LAW: No regex library or import re.
2. ZERO TERNARY LAW: Explicit if/else statements only.
4. NO __INIT__.PY: Direct module imports.
5. 100% MERMAID: No draw.io.
6. ZERO NOTEBOOKLM: No NotebookLM / NLM.
"""

import sys
import json
import time
import shutil
import argparse
from math import isfinite
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.core.settings import Config
from agent_banking.chunking.universal.extraction.sources import legacy_source_resolver, source_key
from agent_banking.chunking.universal.planning.schema import require_index_ready
from agent_banking.vector_db.config.settings import VectorDBConfig
from agent_banking.vector_db.qdrant_store import QdrantBankingStore, mark_publication_chunks, publication_scope
from agent_banking.state_tracker import StateTracker


def compute_chunk_hash(tracker, chunk, default_idx=0):
    return tracker.compute_dict_hash(chunk)


def run_auto_indexing(
    input_file=None,
    output_dir=None,
    force=False,
    use_memory=False,
    *,
    snapshot=False,
    preserve_sources=(),
    publication_dir=None,
    publication_manifest=None,
    embedded_chunks=None,
):
    """
    Tu dong lap chi muc kep Qdrant & BM25 kem co che chong trung lap StateTracker.
    Ho tro Clean Rebuild khi force=True.
    """
    if isinstance(input_file, bool):
        force = input_file
        input_file = None

    cfg = Config()
    in_dir = PROJECT_ROOT / VectorDBConfig.VECTOR_DB_INPUT_DIR
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / VectorDBConfig.VECTOR_DB_OUTPUT_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / VectorDBConfig.VECTOR_DB_STATE_TRACKER_FILENAME
    tracker = StateTracker(state_file)
    previous_records = tracker._active_records().copy()
    if force and not snapshot:
        tracker.clear()

    # 1. Tim nguon va dong bo embedded chunks
    target_in_file = None
    in_chunks_name = VectorDBConfig.VECTOR_DB_INPUT_CHUNKS_FILENAME
    in_copy = in_dir / in_chunks_name

    if embedded_chunks is None:
        if input_file is not None:
            target_in_file = Path(input_file)
        else:
            embed_out = PROJECT_ROOT / VectorDBConfig.VECTOR_DB_EMBEDDING_OUTPUT_CHUNKS_PATH
            if embed_out.exists():
                need_copy = not in_copy.exists() or embed_out.read_bytes() != in_copy.read_bytes()
                if need_copy:
                    try:
                        if embed_out.resolve() != in_copy.resolve():
                            shutil.copyfile(str(embed_out), str(in_copy))
                    except Exception as e_sync:
                        print(f"⚠️ [Sync Warning] Khong the dong bo {in_chunks_name}: {e_sync}")
                target_in_file = in_copy
            elif in_copy.exists():
                target_in_file = in_copy
        if target_in_file is None:
            print(f"⚠️ Khong tim thay tap tin embedded chunks trong: {in_dir}")
            print(f"Vui long bo file '{in_chunks_name}' vao {VectorDBConfig.VECTOR_DB_INPUT_DIR}/")
            return {}
        if not target_in_file.exists():
            print(f"⚠️ Tap tin embedded chunks khong ton tai: {target_in_file}")
            return {}
        with open(target_in_file, "r", encoding="utf-8") as fp:
            raw_chunks = json.load(fp)
        input_label = target_in_file.name
    else:
        raw_chunks = embedded_chunks
        input_label = "Embedding result"

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 3: VECTOR DB & BM25] Input: {input_label} -> Output: {out_path}")
    print(f"{'='*75}")

    if not isinstance(raw_chunks, list):
        raise ValueError("Indexing input must be a list of embedded chunks")
    require_index_ready(raw_chunks)
    scope = None
    published_sources = set()
    if publication_manifest is not None or publication_dir is not None:
        if not snapshot or publication_dir is None:
            raise ValueError("Scoped publication indexing requires snapshot mode and a publication directory")
        if publication_manifest is None:
            from agent_banking.chunking.run_chunking import load_published_chunks
            from agent_banking.embedding.run_embedding import normalize_searchable_chunks

            published, publication_manifest = load_published_chunks(publication_dir, return_manifest=True)
            expected = normalize_searchable_chunks(published)
            if len(expected) != len(raw_chunks) or any(
                source != {key: value for key, value in embedded.items() if key != "vector"}
                for source, embedded in zip(expected, raw_chunks)
            ):
                raise ValueError("Embedded chunks do not match the current publication; rerun embedding")
        failed_sources = set()
        for record in publication_manifest["records"]:
            if record["status"] == "FAILED":
                failed_sources.add(record["rel_path"])
            else:
                published_sources.add(record["rel_path"])
        scope = publication_scope(publication_dir)
        if any(chunk.get("source") not in published_sources for chunk in raw_chunks):
            raise ValueError("Embedded chunks do not match the supplied publication sources")
        preserve_sources = (*preserve_sources, *failed_sources)
        raw_chunks = mark_publication_chunks(raw_chunks, scope)
    if not raw_chunks and not snapshot:
        print("⚠️ Danh sach embedded chunks rong.")
        return {}

    if embedded_chunks is None and input_file is None and output_dir is None:
        if target_in_file.resolve() != in_copy.resolve():
            try:
                shutil.copyfile(str(target_in_file), str(in_copy))
            except Exception:
                pass

    canonical_payload_indexes = list(VectorDBConfig.VECTOR_DB_PAYLOAD_INDEX_FIELDS)

    manifest_file = out_path / VectorDBConfig.VECTOR_DB_MANIFEST_FILENAME
    collection_name = getattr(cfg, "QDRANT_COLLECTION", VectorDBConfig.QDRANT_COLLECTION)
    embedding_dim = getattr(cfg, "EMBEDDING_DIM", VectorDBConfig.QDRANT_VECTOR_SIZE)

    # Kiem tra nhanh: Neu khong force va khong use_memory, kiem tra xem tat ca chunks da xu ly chua
    if not force and not snapshot:
        if not use_memory:
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as fp:
                        cached_manifest = json.load(fp)
                    raw_hash_map = {}
                    for idx, c in enumerate(raw_chunks):
                        cid = c.get("chunk_id")
                        if not cid:
                            cid = f"chunk_{idx}"
                        raw_hash_map[cid] = compute_chunk_hash(tracker, c, idx)

                    if tracker.are_all_items_processed(raw_hash_map):
                        print(f"  ⏭️ [SKIP ALL] Tat ca {len(raw_chunks)} chunks da duoc lap chi muc trong Qdrant & BM25 tu truoc. Bo qua toan bo trong 0.0s!")

                        cached_manifest["status"] = "success"
                        cached_manifest["received_count"] = len(raw_chunks)
                        cached_manifest["new_count"] = 0
                        cached_manifest["skipped_count"] = len(raw_chunks)
                        if "total_points" not in cached_manifest:
                            cached_manifest["total_points"] = cached_manifest.get("total_points_in_collection", len(raw_chunks))
                        if "bm25_size" not in cached_manifest:
                            cached_manifest["bm25_size"] = cached_manifest.get("bm25_corpus_size", len(raw_chunks))
                        if "payload_indexes" not in cached_manifest:
                            cached_manifest["payload_indexes"] = canonical_payload_indexes
                        cached_manifest["execution_time"] = 0.0
                        cached_manifest["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        cached_manifest["total_chunks"] = len(raw_chunks)
                        cached_manifest["indexed_count"] = 0
                        cached_manifest["created_payload_indexes"] = canonical_payload_indexes

                        with open(manifest_file, "w", encoding="utf-8") as fp:
                            json.dump(cached_manifest, fp, ensure_ascii=False, indent=2)

                        total_pts = cached_manifest["total_points"]
                        print(f"✅ [Vector DB Hoan Tat] Collection '{collection_name}' hien co {total_pts} points trong 0.0s")
                        return cached_manifest
                except Exception:
                    pass

    raw_storage = getattr(cfg, "QDRANT_STORAGE_PATH", VectorDBConfig.QDRANT_STORAGE_PATH)
    storage_candidate = Path(str(raw_storage))
    if storage_candidate.is_absolute():
        resolved_storage = str(storage_candidate)
    else:
        resolved_storage = str((PROJECT_ROOT / storage_candidate).resolve())

    store = QdrantBankingStore(
        collection_name=collection_name,
        dimension=embedding_dim,
        storage_path=resolved_storage,
        use_memory=use_memory
    )
    if snapshot and not use_memory and store.use_memory:
        store.close()
        raise RuntimeError("The persistent index is unavailable")

    # Snapshot mode upserts verified replacements before removing obsolete chunks.
    if force and not snapshot:
        print(f"[*] Co --force duoc kich hoat -> Thuc hien Clean Rebuild collection '{collection_name}'...")
        store.recreate_collection()
        tracker.clear()

    # 3. Loc chunks moi chua duoc lap chi muc
    start_t = time.time()
    chunks_to_index = []
    skipped_count = 0
    if snapshot and not force:
        existing_ids = store.chunk_ids(scope=scope, index_keys=True)
    else:
        existing_ids = set()
    active_index_keys = set()

    for idx, c in enumerate(raw_chunks):
        cid = c.get("chunk_id")
        if not cid:
            cid = f"chunk_{idx}"
        c_hash = compute_chunk_hash(tracker, c, idx)

        if snapshot:
            vector = c.get("vector")
            if (not isinstance(vector, list) or len(vector) != embedding_dim
                    or not all(type(value) in (int, float) and isfinite(value) for value in vector)
                    or not any(vector)):
                store.close()
                raise ValueError("Invalid vector for published chunk: " + cid)
            index_key = store.index_key(c, cid)
            active_index_keys.add(index_key)
        is_done = tracker.is_processed(cid, c_hash)
        if snapshot and is_done:
            is_done = index_key in existing_ids
        if is_done:
            if not force:
                if not use_memory:
                    skipped_count += 1
                else:
                    chunks_to_index.append((c, cid, c_hash))
            else:
                chunks_to_index.append((c, cid, c_hash))
        else:
            chunks_to_index.append((c, cid, c_hash))

    print(f"[*] Tong so chunks nhan vao: {len(raw_chunks)} | Da lap chi muc truoc do: {skipped_count}")
    print(f"[*] So chunks can nap moi: {len(chunks_to_index)}")

    # 4. Nap chunks moi vao Qdrant va BM25
    if len(chunks_to_index) > 0:
        t0 = time.time()
        raw_to_index = []
        for item in chunks_to_index:
            raw_to_index.append(item[0])
        store.add_chunks(raw_to_index)
        dt_idx = round(time.time() - t0, 2)
        print(f"[*] Da nap thanh cong {len(chunks_to_index)} chunks moi vao Qdrant & BM25 trong {dt_idx}s.")

        for c, cid, c_hash in chunks_to_index:
            src_val = c.get("source")
            if src_val is None:
                src_val = ""
            metadata = {"source": src_val}
            if scope is not None:
                metadata["publication_scope"] = scope
            tracker.record_processed(cid, c_hash, metadata)
    else:
        print("[*] Toan bo chunks da duoc lap chi muc tu truoc. Khong can nap lai!")

    active_ids = {chunk.get("chunk_id") or f"chunk_{idx}" for idx, chunk in enumerate(raw_chunks)}
    removed_count = 0
    if snapshot:
        if scope is None:
            managed_sources = {item.get("source") for item in raw_chunks}
            managed_sources.update(value.get("source") for value in previous_records.values() if isinstance(value, dict))
        else:
            managed_sources = published_sources
        preserve = {source_key(source) for source in preserve_sources}
        active_sources = {source_key(source) for source in managed_sources if isinstance(source, str)} - preserve
        resolve_legacy = legacy_source_resolver(active_sources | preserve)
        known_sources = managed_sources | {value.get("source") for value in tracker._active_records().values()
                                           if isinstance(value, dict)}
        retained_sources = {source for source in known_sources if isinstance(source, str)
                            and (source_key(source) in preserve or resolve_legacy(source) in preserve)}
        removed_count = store.prune_chunks(active_ids, managed_sources - retained_sources,
                                          scope=scope, preserve_sources=preserve_sources,
                                          active_index_keys=active_index_keys)
        kept_records = {}
        for key, value in tracker._active_records().items():
            if key in active_ids:
                kept_records[key] = value
                continue
            if not isinstance(value, dict):
                continue
            source = value.get("source")
            owner = value.get("publication_scope")
            outside_scope = False
            if scope is not None:
                if owner is None:
                    outside_scope = resolve_legacy(source) not in active_sources
                else:
                    outside_scope = owner != scope
            if source in retained_sources or outside_scope:
                kept_records[key] = value
        tracker.records = kept_records
    tracker.save()
    total_elapsed = round(time.time() - start_t, 4)

    total_count = store.count()
    bm25_count = len(store._bm25_corpus)

    manifest_data = {
        "status": "success",
        "received_count": len(raw_chunks),
        "new_count": len(chunks_to_index),
        "skipped_count": skipped_count,
        "removed_count": removed_count,
        "total_points": total_count,
        "bm25_size": bm25_count,
        "payload_indexes": canonical_payload_indexes,
        "execution_time": total_elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Cac truong tuong thich nguoc (compatibility aliases)
        "total_chunks": len(raw_chunks),
        "indexed_count": len(chunks_to_index),
        "created_payload_indexes": canonical_payload_indexes,
        "total_chunks_received": len(raw_chunks),
        "skipped_chunks": skipped_count,
        "new_indexed_chunks": len(chunks_to_index),
        "total_points_in_collection": total_count,
        "bm25_corpus_size": bm25_count,
        "total_elapsed_sec": total_elapsed,
        "module": "vector_db",
        "collection_name": collection_name,
        "dimension": embedding_dim
    }

    with open(manifest_file, "w", encoding="utf-8") as fp:
        json.dump(manifest_data, fp, ensure_ascii=False, indent=2)

    store.close()

    print(f"✅ [Vector DB Hoan Tat] Collection '{collection_name}' hien co {total_count} points (Nap moi: {len(chunks_to_index)}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Vector DB & BM25 Indexing Pipeline")
    parser.add_argument("--force", action="store_true", help="Clean rebuild flag to recreate collection and clear state tracker")
    parser.add_argument("--input", type=str, default=None, help="Optional custom input json file path")
    parser.add_argument("--output", type=str, default=None, help="Optional custom output manifest directory path")
    parser.add_argument("--use-memory", action="store_true", help="Optional in-memory Qdrant execution")
    parser.add_argument("--snapshot", action="store_true", help="Remove outdated chunk IDs for tracked source files")
    parser.add_argument("--publication", type=Path, help="Verified chunking output directory for scoped snapshot indexing")
    args = parser.parse_args()

    run_auto_indexing(
        input_file=args.input,
        output_dir=args.output,
        force=args.force,
        use_memory=args.use_memory,
        snapshot=args.snapshot,
        publication_dir=args.publication
    )
