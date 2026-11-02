"""
Module Tu Dong Lap Chi Muc Vector DB & BM25 (Auto Vector DB Node with Deduplication).
=====================================================================================
Dac diem:
1. Doc tu vector_db/input/embedded_chunks.json (hoac tu dong lay tu embedding/output).
2. Tinh toan SHA-256 hash cua chunk_id va noi dung vector.
3. Kiem tra StateTracker: Chunk nao da lap chi muc roi thi bo qua (SKIP).
4. Chi upsert vao Qdrant va BM25 nhung chunks moi hoac noi dung bi sua doi.
5. Xuat ket qua: indexing_manifest.json.

NGUYEN TAC BAT BUOC:
1. KHONG DUNG __INIT__.PY.
2. KHONG DUNG THU VIEN TYPING.
3. ZERO TERNARY: if/else tuong minh.
4. 0% REGEX LAW.
"""

from __future__ import annotations
import sys
import json
import time
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


def run_auto_indexing(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
    use_memory: bool = False
) -> dict:
    """
    Tu dong lap chi muc kép Qdrant & BM25 kem co che chong trung lap.
    """
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

    # 1. Tim nguon embedded chunks
    target_in_file = None
    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        default_in = in_dir / "embedded_chunks.json"
        if default_in.exists():
            target_in_file = default_in
        else:
            embed_out = PROJECT_ROOT / "agent_banking" / "embedding" / "output" / "embedded_chunks.json"
            if embed_out.exists():
                target_in_file = embed_out
            else:
                rag_embed_out = PROJECT_ROOT / "rag_banking" / "embedding" / "output" / "embedded_chunks.json"
                if rag_embed_out.exists():
                    target_in_file = rag_embed_out

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Khong tim thay tap tin embedded chunks trong: {in_dir}")
        print("Vui long bo file 'embedded_chunks.json' vao agent_banking/vector_db/input/")
        return {}

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 3: VECTOR DB & BM25] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_chunks = json.load(fp)

    if not raw_chunks:
        print("⚠️ Danh sach embedded chunks rong.")
        return {}

    # Dong bo file vao in_dir
    in_copy = in_dir / "embedded_chunks.json"
    if target_in_file != in_copy:
        try:
            with open(in_copy, "w", encoding="utf-8") as fp:
                json.dump(raw_chunks, fp, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Kiem tra nhanh neu tat ca chunks da duoc lap chi muc
    manifest_file = out_path / "indexing_manifest.json"
    if not force and manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as fp:
                cached_manifest = json.load(fp)
            raw_hash_map = {}
            for idx, c in enumerate(raw_chunks):
                cid = c.get("chunk_id")
                if not cid:
                    cid = f"chunk_{idx}"
                content = c.get("content", "")
                vec = c.get("vector", c.get("embedding", []))
                vec_sample = ""
                if vec:
                    vec_sample = str(vec[:3])
                raw_hash_map[cid] = tracker.compute_text_hash(content + ":" + vec_sample)
            if tracker.are_all_items_processed(raw_hash_map):
                print(f"  ⏭️ [SKIP ALL] Tat ca {len(raw_chunks)} chunks da duoc lap chi muc trong Qdrant & BM25 tu truoc. Bo qua toan bo trong 0.0s!")
                print(f"✅ [Vector DB Hoan Tat] Collection '{cfg.QDRANT_COLLECTION}' hien co {cached_manifest.get('total_points_in_collection', len(raw_chunks))} points trong 0.0s")
                return cached_manifest
        except Exception:
            pass

    store = QdrantBankingStore(
        collection_name=cfg.QDRANT_COLLECTION,
        dimension=cfg.EMBEDDING_DIM,
        storage_path=str(cfg.STORAGE_DIR / "qdrant_db"),
        use_memory=use_memory
    )

    # 2. Loc chunks moi chua duoc lap chi muc
    start_t = time.time()
    chunks_to_index = []
    skipped_count = 0

    for idx, c in enumerate(raw_chunks):
        cid = c.get("chunk_id")
        if not cid:
            cid = f"chunk_{idx}"
        content = c.get("content", "")
        # Tinh hash ket hop content va vector head
        vec = c.get("vector", c.get("embedding", []))
        vec_sample = ""
        if vec:
            vec_sample = str(vec[:3])
        c_hash = tracker.compute_text_hash(content + ":" + vec_sample)

        is_done = tracker.is_processed(cid, c_hash)
        if is_done and not force:
            skipped_count += 1
        else:
            chunks_to_index.append(c)

    print(f"[*] Tong so chunks: {len(raw_chunks)} | Da lap chi muc truoc do: {skipped_count}")
    print(f"[*] So chunks can lap chi muc moi: {len(chunks_to_index)}")

    # 3. Lap chi muc neu co chunks moi
    if chunks_to_index:
        t0 = time.time()
        store.add_chunks(chunks_to_index)
        dt_idx = round(time.time() - t0, 2)
        print(f"[*] Da nap thanh cong {len(chunks_to_index)} chunks moi vao Qdrant & BM25 trong {dt_idx}s.")

        for c in chunks_to_index:
            cid = c.get("chunk_id")
            content = c.get("content", "")
            vec = c.get("vector", c.get("embedding", []))
            vec_sample = ""
            if vec:
                vec_sample = str(vec[:3])
            c_hash = tracker.compute_text_hash(content + ":" + vec_sample)
            tracker.record_processed(cid, c_hash, {"source": c.get("source", "")})
    else:
        print("[*] Toan bo chunks da duoc lap chi muc tu truoc. Khong can nap lai!")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    total_count = store.count()
    manifest_file = out_path / "indexing_manifest.json"
    manifest_data = {
        "module": "vector_db",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "collection_name": cfg.QDRANT_COLLECTION,
        "dimension": cfg.EMBEDDING_DIM,
        "total_chunks_received": len(raw_chunks),
        "skipped_chunks": skipped_count,
        "new_indexed_chunks": len(chunks_to_index),
        "total_points_in_collection": total_count,
        "bm25_corpus_size": len(store._bm25_corpus),
        "total_elapsed_sec": total_elapsed,
        "status": "READY_FOR_RETRIEVAL"
    }
    with open(manifest_file, "w", encoding="utf-8") as fp:
        json.dump(manifest_data, fp, ensure_ascii=False, indent=2)

    print(f"✅ [Vector DB Hoan Tat] Collection '{cfg.QDRANT_COLLECTION}' hien co {total_count} points (Nap moi: {len(chunks_to_index)}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return manifest_data


if __name__ == "__main__":
    run_auto_indexing()
