"""
Module Tự Động Tạo Vector Nhúng (Auto Embedding Node with Deduplication).
==========================================================================
Đặc điểm:
1. Đọc chunks ưu tiên từ agent_banking/chunking/output/all_chunks.json.
2. Chuẩn hóa 100% chunks về Target Universal Chunking Schema (13 trường chuẩn).
3. Loại bỏ toàn bộ các trường legacy (stats, location, element_ids, doc_name...).
4. Định dạng Type-Aware Virtual Passage cho multilingual-e5-large.
5. Khử trùng lặp qua StateTracker và SQLite WAL Cache trên chính Virtual Passage.
6. Xuất kết quả: embedded_chunks.json (13 trường + 'vector' 1024-d).

NGUYÊN TẮC BẮT BUỘC (GEMINI.md & AGENTS.md):
1. KHÔNG DÙNG __INIT__.PY.
2. KHÔNG DÙNG THƯ VIỆN TYPING.
3. ZERO TERNARY: if/else tường minh từng dòng, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex hay import re.
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
from agent_banking.chunking.universal.schema_assembler import (
    REQUIRED_UNIVERSAL_SCHEMA_KEYS,
    normalize_chunk_type,
    link_chunk_sequence,
    infer_business_metadata,
)


def normalize_universal_chunk(chunk, vector=None):
    """
    Chuẩn hóa chunk về đúng 13 trường của Target Universal Schema (R3) + trường 'vector' (nếu có).
    Loại bỏ hoàn toàn các trường legacy (stats, location, element_ids, word_count, token_count, doc_name...).
    Bảo toàn 100% nội dung gốc của chunk['content'].
    """
    c_id = chunk.get("chunk_id")
    if not c_id:
        c_id = ""
    else:
        c_id = str(c_id).strip()

    p_id = chunk.get("parent_id")
    if not p_id:
        p_id = ""
    else:
        p_id = str(p_id).strip()

    source = chunk.get("source")
    if not source:
        source = chunk.get("doc_name")
    if not source:
        source = "unknown.docx"
    else:
        source = str(source).strip()

    raw_c_type = chunk.get("chunk_type")
    c_type = normalize_chunk_type(raw_c_type)
    images_list = chunk.get("images")
    if images_list is None:
        images_list = []
    elif not isinstance(images_list, list):
        images_list = []

    if c_type == "text":
        if len(images_list) > 0:
            c_type = "multimodal"

    breadcrumb = chunk.get("breadcrumb")
    if breadcrumb is None:
        breadcrumb = ""
    else:
        breadcrumb = str(breadcrumb).strip()

    sec_title = chunk.get("section_title")
    if sec_title is None:
        sec_title = ""
    else:
        sec_title = str(sec_title).strip()

    content = chunk.get("content")
    if content is None:
        content = chunk.get("original_text")
    if content is None:
        content = ""
    else:
        content = str(content)

    parent_content = chunk.get("parent_content")
    if parent_content is None:
        parent_content = ""
    else:
        parent_content = str(parent_content)

    b_meta = chunk.get("business_metadata")
    if isinstance(b_meta, dict) and "domain" in b_meta:
        has_prohibited = False
        for k, v in b_meta.items():
            k_lower = str(k).lower()
            v_lower = str(v).lower()
            for term in ("the_tin_dung", "tiet_kiem", "vay_von", "nghiep_vu_bank", "chuong_trinh_the", "van_ban_phap_ly", "huong_dan_app", "bao_mat_ekyc", "the_ghi_no", "phap_ly_tuan_thu"):
                if term in k_lower or term in v_lower:
                    has_prohibited = True
                    break
            if has_prohibited:
                break
        if has_prohibited:
            business_metadata = infer_business_metadata(source)
        else:
            business_metadata = dict(b_meta)
    else:
        business_metadata = infer_business_metadata(source)

    prev_id = chunk.get("prev_chunk_id")
    if prev_id is None:
        prev_id = ""
    else:
        prev_id = str(prev_id).strip()

    next_id = chunk.get("next_chunk_id")
    if next_id is None:
        next_id = ""
    else:
        next_id = str(next_id).strip()

    table_data = chunk.get("table_data")
    if not isinstance(table_data, (dict, list)):
        table_data = None

    normalized = {
        "chunk_id": c_id,
        "parent_id": p_id,
        "source": source,
        "chunk_type": c_type,
        "breadcrumb": breadcrumb,
        "section_title": sec_title,
        "content": content,
        "parent_content": parent_content,
        "business_metadata": business_metadata,
        "prev_chunk_id": prev_id,
        "next_chunk_id": next_id,
        "table_data": table_data,
        "images": images_list,
    }

    if vector is not None:
        normalized["vector"] = vector

    return normalized


def run_auto_embedding(
    input_file=None,
    output_dir=None,
    force=False,
    batch_size=32
):
    """
    Tự động tạo vector nhúng từ input sang output kèm cơ chế chống trùng lặp.
    Bảo toàn 100% 13 trường của Target Universal Schema kèm trường 'vector' 1024-d.
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

    # 1. Tìm nguồn chunks đầu vào (Ưu tiên all_chunks.json chuẩn Universal Schema)
    target_in_file = None
    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        chunking_out = PROJECT_ROOT / "agent_banking" / "chunking" / "output" / "all_chunks.json"
        if chunking_out.exists():
            target_in_file = chunking_out
        else:
            default_in = in_dir / "chunks.json"
            if default_in.exists():
                target_in_file = default_in
            else:
                rag_chunking_out = PROJECT_ROOT / "rag_banking" / "chunking" / "output" / "all_chunks.json"
                if rag_chunking_out.exists():
                    target_in_file = rag_chunking_out

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Không tìm thấy tập tin chunks đầu vào: {target_in_file}")
        print("Vui lòng bỏ file 'all_chunks.json' hoặc 'chunks.json' vào hệ thống.")
        return []

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 2: EMBEDDING] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_chunks = json.load(fp)

    if not raw_chunks:
        print("⚠️ Danh sách chunks rỗng.")
        return []

    # Đồng bộ file vào in_dir nếu chưa có
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

    # Chuẩn hóa toàn bộ raw_chunks về chuẩn 13 trường Universal Schema và liên kết chuỗi
    normalized_chunks = []
    for idx, raw_c in enumerate(raw_chunks):
        norm_c = normalize_universal_chunk(raw_c)
        if not norm_c.get("chunk_id"):
            norm_c["chunk_id"] = f"chunk_{idx}"
        normalized_chunks.append(norm_c)

    link_chunk_sequence(normalized_chunks)

    # 2. Đọc file output cũ nếu có để tái sử dụng vector
    existing_vectors = {}
    prev_out_file = out_path / "embedded_chunks.json"
    if prev_out_file.exists():
        if not force:
            try:
                with open(prev_out_file, "r", encoding="utf-8") as fp:
                    prev_list = json.load(fp)
                    for item in prev_list:
                        cid = item.get("chunk_id")
                        v = item.get("vector")
                        if not v:
                            v = item.get("embedding")
                        if cid:
                            if v:
                                if len(v) == embedder.dimension:
                                    existing_vectors[cid] = v

                # Kiểm tra nhanh nếu tất cả chunks đã có vector trong output và StateTracker
                if len(existing_vectors) >= len(normalized_chunks):
                    if len(normalized_chunks) > 0:
                        chunk_hash_map = {}
                        for c in normalized_chunks:
                            cid = c.get("chunk_id")
                            vp = embedder.format_virtual_passage(c)
                            chunk_hash_map[cid] = embedder._get_hash(vp)
                        if tracker.are_all_items_processed(chunk_hash_map):
                            print(f"  ⏭️ [SKIP ALL] Tất cả {len(normalized_chunks)} chunks đã được tạo vector trước đó. Bỏ qua toàn bộ trong 0.0s!")
                            # Đảm bảo 100% chunks trả về đúng chuẩn 13 trường + vector
                            clean_prev_list = []
                            for p_chunk in prev_list:
                                p_vec = p_chunk.get("vector")
                                if not p_vec:
                                    p_vec = p_chunk.get("embedding")
                                clean_prev_list.append(normalize_universal_chunk(p_chunk, p_vec))
                            link_chunk_sequence(clean_prev_list)
                            with open(prev_out_file, "w", encoding="utf-8") as fp:
                                json.dump(clean_prev_list, fp, ensure_ascii=False, indent=2)
                            print(f"✅ [Embedding Hoàn Tất] {len(clean_prev_list)} vector đã sẵn sàng trong: {prev_out_file} (Bỏ qua: {len(normalized_chunks)}, Mới: 0) trong 0.0s")
                            return clean_prev_list
            except Exception:
                existing_vectors = {}

    start_t = time.time()
    chunks_to_embed = []
    chunks_result_map = {}
    skipped_count = 0

    for idx, c in enumerate(normalized_chunks):
        cid = c.get("chunk_id")
        vp = embedder.format_virtual_passage(c)
        vp_hash = embedder._get_hash(vp)

        is_done = tracker.is_processed(cid, vp_hash)
        vec = None
        if is_done:
            if cid in existing_vectors:
                vec = existing_vectors[cid]

        if vec is None:
            cached_v = embedder._get_cached_vector(vp)
            if cached_v is not None:
                vec = cached_v

        if vec is not None:
            normalized_with_vec = normalize_universal_chunk(c, vec)
            chunks_result_map[cid] = normalized_with_vec
            tracker.record_processed(cid, vp_hash, {"dimension": len(vec), "source": "cache"})
            skipped_count += 1
        else:
            chunks_to_embed.append(c)

    print(f"[*] Tổng số chunks: {len(normalized_chunks)} | Đã có sẵn trong cache/output: {skipped_count} chunks")
    print(f"[*] Số chunks cần gọi nhúng mới: {len(chunks_to_embed)} chunks")

    # 3. Nhúng vector cho các chunks mới
    new_embedded_count = 0
    if chunks_to_embed:
        t0 = time.time()
        newly_done = embedder.embed_chunks(chunks_to_embed, batch_size=batch_size)
        dt_embed = round(time.time() - t0, 2)
        new_embedded_count = len(newly_done)
        print(f"[*] Đã nhúng thành công {new_embedded_count} chunks mới trong {dt_embed}s.")

        for c in newly_done:
            cid = c.get("chunk_id")
            vp = embedder.format_virtual_passage(c)
            vp_hash = embedder._get_hash(vp)
            vec = c.get("vector")
            if not vec:
                vec = c.get("embedding", [])
            tracker.record_processed(cid, vp_hash, {"dimension": len(vec), "source": "fpt_api"})
            chunks_result_map[cid] = normalize_universal_chunk(c, vec)

    # 4. Sắp xếp lại đúng thứ tự ban đầu
    final_embedded_chunks = []
    for c in normalized_chunks:
        cid = c.get("chunk_id")
        if cid in chunks_result_map:
            final_embedded_chunks.append(chunks_result_map[cid])
        else:
            final_embedded_chunks.append(normalize_universal_chunk(c, [0.0] * embedder.dimension))

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 5. Xuất kết quả
    out_file = out_path / "embedded_chunks.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(final_embedded_chunks, fp, ensure_ascii=False, indent=2)

    # Mẫu 10 vector để kiểm tra trực quan
    sample_vectors = []
    for c in final_embedded_chunks[:10]:
        vec = c.get("vector", [])
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
            "chunk_type": c.get("chunk_type"),
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

    print(f"✅ [Embedding Hoàn Tất] {len(final_embedded_chunks)} vector đã lưu vào: {out_file} (Mới: {new_embedded_count}, Bỏ qua: {skipped_count}) trong {total_elapsed}s")
    return final_embedded_chunks


if __name__ == "__main__":
    run_auto_embedding()
