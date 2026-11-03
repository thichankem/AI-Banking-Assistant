"""
Module Tu Dong Lap Chi Muc Vector DB & BM25 (Auto Vector DB Node with Deduplication).
=====================================================================================
Dac diem:
1. Dong bo va nap truc tiep tu agent_banking/embedding/output/embedded_chunks.json (2.627 chunks).
2. Du phong nap tu agent_banking/vector_db/input/embedded_chunks.json neu khong tim thay nguon embedding.
3. Tinh toan SHA-256 hash toan dien: chunk_id, chunk_type, content, vector head/tail.
4. Kiem tra StateTracker: Chunk nao da lap chi muc thi bo qua (SKIP).
5. Ho tro Clean Rebuild voi co --force: xoa sach collection cu va nap lai tu dau.
6. Xuat bao cao indexing_manifest.json voi day du cac truong canonical va aliases tuong thich.

NGUYEN TAC BAT BUOC (GEMINI.md & AGENTS.md):
1. 0% REGEX LAW: Tuyet doi khong dung thu vien re hoac bieu thuc chinh quy.
2. ZERO TERNARY LAW: Bat buoc dung if/else tuong minh tung dong, khong dung cu phap ternary.
3. NO TYPING LIBRARY: Tuyet doi khong su dung kieu tu typing module, chi dung kieu co san.
4. KHONG TAO HOAC GIU FILE __INIT__.PY.
5. 100% MERMAID: Khong dung draw.io.
6. ZERO NOTEBOOKLM: Khong su dung NotebookLM / NLM.
"""

import sys
import json
import time
import shutil
import argparse
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

from agent_banking.config import Config
from agent_banking.vector_db.qdrant_store import QdrantBankingStore
from agent_banking.state_tracker import StateTracker


def compute_chunk_hash(tracker, chunk, default_idx=0):
    """
    Tinh ma hash toan dien ket hop chunk_id, chunk_type, content, vector head va tail.
    Tuan thu 0% Regex va Zero Ternary.
    """
    cid = chunk.get("chunk_id")
    if not cid:
        cid = f"chunk_{default_idx}"

    ctype = chunk.get("chunk_type")
    if not ctype:
        ctype = "text"

    content = chunk.get("content")
    if content is None:
        content = ""
    else:
        content = str(content)

    vec = chunk.get("vector")
    if vec is None:
        vec = chunk.get("embedding")
    if vec is None:
        vec = []

    vec_head = ""
    vec_tail = ""
    if len(vec) > 0:
        vec_head = str(vec[:3])
        vec_tail = str(vec[-3:])

    hash_key = f"{cid}|{ctype}|{content}|{vec_head}|{vec_tail}"
    return tracker.compute_text_hash(hash_key)


def run_auto_indexing(
    input_file=None,
    output_dir=None,
    force=False,
    use_memory=False
):
    """
    Tu dong lap chi muc kep Qdrant & BM25 kem co che chong trung lap StateTracker.
    Ho tro Clean Rebuild khi force=True.
    """
    if isinstance(input_file, bool):
        force = input_file
        input_file = None

    cfg = Config()
    in_dir = PROJECT_ROOT / "agent_banking" / "vector_db" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "vector_db" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / ".state_tracker.json"
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 1. Tim nguon va dong bo embedded chunks
    target_in_file = None
    in_copy = in_dir / "embedded_chunks.json"

    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        embed_out = PROJECT_ROOT / "agent_banking" / "embedding" / "output" / "embedded_chunks.json"
        if embed_out.exists():
            need_copy = True
            if in_copy.exists():
                if embed_out.stat().st_size == in_copy.stat().st_size:
                    need_copy = False
            if need_copy:
                try:
                    if embed_out.resolve() != in_copy.resolve():
                        shutil.copyfile(str(embed_out), str(in_copy))
                except Exception as e_sync:
                    print(f"⚠️ [Sync Warning] Khong the dong bo embedded_chunks.json: {e_sync}")
            target_in_file = in_copy
        else:
            if in_copy.exists():
                target_in_file = in_copy
            else:
                rag_embed_out = PROJECT_ROOT / "rag_banking" / "embedding" / "output" / "embedded_chunks.json"
                if rag_embed_out.exists():
                    try:
                        if rag_embed_out.resolve() != in_copy.resolve():
                            shutil.copyfile(str(rag_embed_out), str(in_copy))
                    except Exception:
                        pass
                    target_in_file = in_copy

    if target_in_file is None:
        print(f"⚠️ Khong tim thay tap tin embedded chunks trong: {in_dir}")
        print("Vui long bo file 'embedded_chunks.json' vao agent_banking/vector_db/input/")
        return {}

    if not target_in_file.exists():
        print(f"⚠️ Tap tin embedded chunks khong ton tai: {target_in_file}")
        return {}

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 3: VECTOR DB & BM25] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_chunks = json.load(fp)

    if not raw_chunks:
        print("⚠️ Danh sach embedded chunks rong.")
        return {}

    if target_in_file.resolve() != in_copy.resolve():
        try:
            shutil.copyfile(str(target_in_file), str(in_copy))
        except Exception:
            pass

    canonical_payload_indexes = [
        "source",
        "chunk_type",
        "parent_id",
        "business_metadata.domain",
        "business_metadata.category"
    ]

    manifest_file = out_path / "indexing_manifest.json"

    # Kiem tra nhanh: Neu khong force va khong use_memory, kiem tra xem tat ca chunks da xu ly chua
    if not force:
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
                        print(f"✅ [Vector DB Hoan Tat] Collection '{cfg.QDRANT_COLLECTION}' hien co {total_pts} points trong 0.0s")
                        return cached_manifest
                except Exception:
                    pass

    store = QdrantBankingStore(
        collection_name=cfg.QDRANT_COLLECTION,
        dimension=cfg.EMBEDDING_DIM,
        storage_path=str(cfg.STORAGE_DIR / "qdrant_db"),
        use_memory=use_memory
    )

    # 2. Xu ly Clean Rebuild neu force=True
    if force:
        print(f"[*] Co --force duoc kich hoat -> Thuc hien Clean Rebuild collection '{cfg.QDRANT_COLLECTION}'...")
        store.recreate_collection()
        tracker.clear()

    # 3. Loc chunks moi chua duoc lap chi muc
    start_t = time.time()
    chunks_to_index = []
    skipped_count = 0

    for idx, c in enumerate(raw_chunks):
        cid = c.get("chunk_id")
        if not cid:
            cid = f"chunk_{idx}"
        c_hash = compute_chunk_hash(tracker, c, idx)

        is_done = tracker.is_processed(cid, c_hash)
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
            tracker.record_processed(cid, c_hash, {"source": src_val})
    else:
        print("[*] Toan bo chunks da duoc lap chi muc tu truoc. Khong can nap lai!")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 4)

    total_count = store.count()
    bm25_count = len(store._bm25_corpus)

    manifest_data = {
        "status": "success",
        "received_count": len(raw_chunks),
        "new_count": len(chunks_to_index),
        "skipped_count": skipped_count,
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
        "collection_name": cfg.QDRANT_COLLECTION,
        "dimension": cfg.EMBEDDING_DIM
    }

    with open(manifest_file, "w", encoding="utf-8") as fp:
        json.dump(manifest_data, fp, ensure_ascii=False, indent=2)

    store.close()

    print(f"✅ [Vector DB Hoan Tat] Collection '{cfg.QDRANT_COLLECTION}' hien co {total_count} points (Nap moi: {len(chunks_to_index)}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Vector DB & BM25 Indexing Pipeline")
    parser.add_argument("--force", action="store_true", help="Clean rebuild flag to recreate collection and clear state tracker")
    parser.add_argument("--input", type=str, default=None, help="Optional custom input json file path")
    parser.add_argument("--output", type=str, default=None, help="Optional custom output manifest directory path")
    parser.add_argument("--use-memory", action="store_true", help="Optional in-memory Qdrant execution")
    args = parser.parse_args()

    run_auto_indexing(
        input_file=args.input,
        output_dir=args.output,
        force=args.force,
        use_memory=args.use_memory
    )
