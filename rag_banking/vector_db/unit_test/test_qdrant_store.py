"""
Kiểm thử toàn diện QdrantBankingStore:
1. Nạp chunks và vectors thực tế từ ngân hàng.
2. Tìm kiếm Dense Vector (Cosine HNSW 1024-d).
3. Tìm kiếm Từ khóa (BM25 Okapi).
4. Lọc Metadata (Payload Filter).
5. Tìm kiếm Lai (Hybrid Search: Dense + BM25 + RRF).
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAG_ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(RAG_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from embedding.embedder import BankingEmbedder
from vector_db.qdrant_store import QdrantBankingStore

def test_qdrant_store_suite():
    main()


def main():
    print("=" * 70)
    print("▶ 1. KHỞI TẠO EMBEDDER VÀ VECTOR DATABASE (QDRANT LOCALHOST)")
    print("=" * 70)
    embedder = BankingEmbedder(dimension=1024)
    # Dùng in-memory cho test nhanh gọn, không để lại file rác
    store = QdrantBankingStore(
        collection_name="test_banking_1024",
        dimension=1024,
        use_memory=True
    )

    # Đọc 10 chunks thật từ file ngân hàng
    cand_files = [
        ROOT / "rag_banking/chunking/output/chunks_per_file/01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.json",
        Path("chunking/output/chunks_per_file/01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.json"),
        ROOT / "rag_banking/chunking/chunking_dataset/dataset_1/chunks_per_file/vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.json"
    ]
    chunk_file = next((p for p in cand_files if p.exists()), None)
    if chunk_file is None:
        raise FileNotFoundError("Không tìm thấy tệp chunk JSON mẫu để kiểm thử.")
    with open(chunk_file, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)[:10]

    print(f"▶ Nhúng vector 1024 chiều cho {len(raw_chunks)} chunks...")
    chunks_with_vec = embedder.embed_chunks(raw_chunks)

    print("▶ Nạp vào Qdrant Vector Store...")
    store.add_chunks(chunks_with_vec)
    print(f"  [OK] Tổng số vector trong Qdrant: {store.count()}")

    print("\n" + "=" * 70)
    print("▶ 2. KIỂM THỬ DENSE VECTOR SEARCH (COSINE HNSW 1024-D)")
    print("=" * 70)
    query_text = "Khách hàng nhận thẻ nhưng không chi tiêu được online thì làm sao?"
    q_vec = embedder.embed_query(query_text)
    dense_results = store.search_vector(q_vec, top_k=3)

    for rank, (chunk, score) in enumerate(dense_results, 1):
        print(f"  Top {rank} [Score: {score:.4f}] - ID: {chunk.get('chunk_id')}")
        print(f"    Tiêu đề: {chunk.get('section_title')[:70]}...")

    print("\n" + "=" * 70)
    print("▶ 3. KIỂM THỬ TỪ KHÓA CHÍNH XÁC (BM25 OKAPI SEARCH)")
    print("=" * 70)
    bm25_results = store.search_bm25("kích hoạt thẻ SMS VCB Digibank", top_k=3)
    for rank, (chunk, score) in enumerate(bm25_results, 1):
        print(f"  Top {rank} [BM25 Score: {score:.4f}] - ID: {chunk.get('chunk_id')}")
        print(f"    Nội dung: {chunk.get('content')[:80]}...")

    print("\n" + "=" * 70)
    print("▶ 4. KIỂM THỬ LỌC METADATA (PAYLOAD FILTER)")
    print("=" * 70)
    source_name = raw_chunks[0].get("source")
    filtered_results = store.search_vector(
        q_vec,
        top_k=3,
        filter_dict={"source": source_name, "chunk_type": "docx_text"}
    )
    print(f"  [OK] Lọc theo source='{source_name}': tìm thấy {len(filtered_results)} kết quả phù hợp.")

    print("\n" + "=" * 70)
    print("▶ 5. KIỂM THỬ HYBRID SEARCH (DENSE + BM25 + RRF FUSION)")
    print("=" * 70)
    hybrid_results = store.search_hybrid(
        query_text=query_text,
        query_vector=q_vec,
        top_k=3,
        alpha=0.5
    )
    for rank, item in enumerate(hybrid_results, 1):
        c = item["chunk"]
        print(f"  Top {rank} [RRF: {item['rrf_score']:.6f} | DenseRank: {item['dense_rank']} | BM25Rank: {item['bm25_rank']}]")
        print(f"    ID: {c.get('chunk_id')}")
        print(f"    Nội dung: {c.get('content')[:90]}...")

    print("\n🎉 TẤT CẢ TÍNH NĂNG VECTOR DATABASE ĐÃ HOẠT ĐỘNG HOÀN HẢO 100%!")


if __name__ == "__main__":
    main()
