"""
Module Tu Dong Tao Vector Nhuong (Auto Embedding Node with Deduplication).
==========================================================================
Dac diem:
1. Doc chunks tu embedding/input/chunks.json (hoac tu dong lay tu chunking/output/all_chunks.json).
2. Tinh toan SHA-256 hash cua content tung chunk.
3. Kiem tra StateTracker & SQLite Cache: Chunk nao da co vector thi bo qua (SKIP).
4. Chi gui API batch nhung chunks moi hoac noi dung bi sua doi.
5. Xuat ket qua: embedded_chunks.json, sample_embeddings.json, embeddings_manifest.json.

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
from agent_banking.embedding.embedder import BankingEmbedder
from agent_banking.state_tracker import StateTracker


def run_auto_embedding(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
    batch_size: int = 32
) -> list[dict]:
    """
    Tu dong tao vector nhung tu input sang output kem co che chong trung lap.
    """
    cfg = Config()
    in_dir = PROJECT_ROOT / "agent_banking" / "embedding" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "embedding" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / ".state_tracker.json"
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 1. Tim nguon chunks dau vao
    target_in_file = None
    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        default_in = in_dir / "chunks.json"
        if default_in.exists():
            target_in_file = default_in
        else:
            # Tu dong lien ket tu chunking/output/all_chunks.json
            chunking_out = PROJECT_ROOT / "agent_banking" / "chunking" / "output" / "all_chunks.json"
            if chunking_out.exists():
                target_in_file = chunking_out
            else:
                rag_chunking_out = PROJECT_ROOT / "rag_banking" / "chunking" / "output" / "all_chunks.json"
                if rag_chunking_out.exists():
                    target_in_file = rag_chunking_out

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Khong tim thay tap tin chunks dau vao trong: {in_dir}")
        print("Vui long bo file 'chunks.json' vao agent_banking/embedding/input/")
        return []

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 2: EMBEDDING] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_chunks = json.load(fp)

    if not raw_chunks:
        print("⚠️ Danh sach chunks rong.")
        return []

    # Dong bo file vao in_dir neu chua co
    in_copy = in_dir / "chunks.json"
    if target_in_file != in_copy:
        try:
            with open(in_copy, "w", encoding="utf-8") as fp:
                json.dump(raw_chunks, fp, ensure_ascii=False, indent=2)
        except Exception:
            pass

    embedder = BankingEmbedder(
        dimension=cfg.EMBEDDING_DIM,
        model_name=cfg.EMBEDDING_MODEL_1024,
        embedding_url=cfg.EMBEDDING_URL,
        api_key=cfg.LLM_API_KEY,
        cache_db_path=str(cfg.STORAGE_DIR / "embedding_cache.db")
    )

    # 2. Doc file output cu neu co de tai su dung vector
    existing_vectors = {}
    prev_out_file = out_path / "embedded_chunks.json"
    if prev_out_file.exists() and not force:
        try:
            with open(prev_out_file, "r", encoding="utf-8") as fp:
                prev_list = json.load(fp)
                for item in prev_list:
                    cid = item.get("chunk_id")
                    v = item.get("vector", item.get("embedding"))
                    if cid and v:
                        existing_vectors[cid] = v
            # Kiem tra nhanh neu tat ca chunks da co vector trong output va tracker
            if len(existing_vectors) >= len(raw_chunks) and len(raw_chunks) > 0:
                chunk_hash_map = {}
                for idx, c in enumerate(raw_chunks):
                    cid = c.get("chunk_id")
                    if not cid:
                        cid = f"chunk_{idx}"
                    content = c.get("content", "")
                    chunk_hash_map[cid] = tracker.compute_text_hash(content)
                if tracker.are_all_items_processed(chunk_hash_map):
                    print(f"  ⏭️ [SKIP ALL] Tat ca {len(raw_chunks)} chunks da duoc tao vector truoc do. Bo qua toan bo trong 0.0s!")
                    print(f"✅ [Embedding Hoan Tat] {len(prev_list)} vector da san sang trong: {prev_out_file} (Bo qua: {len(raw_chunks)}, Moi: 0) trong 0.0s")
                    return prev_list
        except Exception:
            existing_vectors = {}

    start_t = time.time()
    chunks_to_embed = []
    chunks_result_map = {}
    skipped_count = 0

    for idx, c in enumerate(raw_chunks):
        cid = c.get("chunk_id")
        if not cid:
            cid = f"chunk_{idx}"
        content = c.get("content", "")
        c_hash = tracker.compute_text_hash(content)

        is_done = tracker.is_processed(cid, c_hash)
        vec = None
        if is_done and cid in existing_vectors:
            vec = existing_vectors[cid]
        else:
            cached_v = embedder._get_cached_vector(content)
            if cached_v is not None:
                vec = cached_v

        if vec is not None:
            c_copy = dict(c)
            c_copy["vector"] = vec
            chunks_result_map[cid] = c_copy
            tracker.record_processed(cid, c_hash, {"dimension": len(vec), "source": "cache"})
            skipped_count += 1
        else:
            chunks_to_embed.append(c)

    print(f"[*] Tong so chunks: {len(raw_chunks)} | Da co san trong cache/output: {skipped_count} chunks")
    print(f"[*] So chunks can goi nhung moi: {len(chunks_to_embed)} chunks")

    # 3. Nhung vector cho cac chunks moi
    new_embedded_count = 0
    if chunks_to_embed:
        t0 = time.time()
        newly_done = embedder.embed_chunks(chunks_to_embed, batch_size=batch_size)
        dt_embed = round(time.time() - t0, 2)
        new_embedded_count = len(newly_done)
        print(f"[*] Da nhung thanh cong {new_embedded_count} chunks moi trong {dt_embed}s.")

        for c in newly_done:
            cid = c.get("chunk_id")
            content = c.get("content", "")
            c_hash = tracker.compute_text_hash(content)
            vec = c.get("vector", c.get("embedding", []))
            tracker.record_processed(cid, c_hash, {"dimension": len(vec), "source": "fpt_api"})
            chunks_result_map[cid] = c

    # 4. Sap xep lai dung thu tu ban dau
    final_embedded_chunks = []
    for idx, c in enumerate(raw_chunks):
        cid = c.get("chunk_id")
        if not cid:
            cid = f"chunk_{idx}"
        if cid in chunks_result_map:
            final_embedded_chunks.append(chunks_result_map[cid])
        else:
            final_embedded_chunks.append(c)

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 5. Xuat ket qua
    out_file = out_path / "embedded_chunks.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(final_embedded_chunks, fp, ensure_ascii=False, indent=2)

    # Mau 10 vector de kiem tra
    sample_vectors = []
    for c in final_embedded_chunks[:10]:
        vec = c.get("vector", c.get("embedding", []))
        v_head = []
        if vec:
            v_head = vec[:5]
        v_tail = []
        if vec:
            v_tail = vec[-5:]
        sample_vectors.append({
            "chunk_id": c.get("chunk_id"),
            "source": c.get("source"),
            "section_title": c.get("section_title"),
            "dimension": len(vec),
            "vector_head_5": v_head,
            "vector_tail_5": v_tail
        })

    sample_file = out_path / "sample_embeddings.json"
    with open(sample_file, "w", encoding="utf-8") as fp:
        json.dump(sample_vectors, fp, ensure_ascii=False, indent=2)

    manifest_file = out_path / "embeddings_manifest.json"
    manifest_data = {
        "module": "embedding",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": cfg.EMBEDDING_MODEL_1024,
        "dimension": cfg.EMBEDDING_DIM,
        "total_chunks": len(final_embedded_chunks),
        "skipped_chunks": skipped_count,
        "new_embedded_chunks": new_embedded_count,
        "total_elapsed_sec": total_elapsed
    }
    with open(manifest_file, "w", encoding="utf-8") as fp:
        json.dump(manifest_data, fp, ensure_ascii=False, indent=2)

    print(f"✅ [Embedding Hoan Tat] {len(final_embedded_chunks)} vector da luu vao: {out_file} (Moi: {new_embedded_count}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return final_embedded_chunks


if __name__ == "__main__":
    run_auto_embedding()
