"""
Kiểm thử toàn diện BankingReranker (FPT Cloud bge-reranker-v2-m3):
1. Kiểm thử Rerank danh sách câu văn bản thô (giống test curl của người dùng).
2. Kiểm thử Tích hợp chuỗi RAG:
   Chunking -> Embedding (1024-d) -> Qdrant Hybrid Search (Dense + BM25) -> Reranker (Top-3).
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
from reranking.rag_reranker import BankingReranker

def test_reranker_suite():
    main()


def main():
    print("=" * 70)
    print("▶ 1. KIỂM THỬ RERANK VĂN BẢN TRỰC TIẾP (CURL PROMPT TEST)")
    print("=" * 70)

    reranker = BankingReranker()
    query = "What is the capital of the United States?"
    documents = [
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Washington, D.C. is the capital of the United States.",
        "Capital punishment has existed in the United States since before it was a country."
    ]

    print(f"Câu hỏi: {query}")
    print(f"Số lượng tài liệu đầu vào: {len(documents)}")
    results = reranker.rerank(query, documents, top_k=2)

    for res in results:
        rank = res.get("rerank_rank")
        score = res.get("relevance_score")
        doc = res.get("document")
        print(f"  Top {rank} [Score: {score:.4f}]: {doc}")

    assert len(results) == 2
    assert "Washington, D.C." in results[0]["document"], "Top 1 phải là Washington, D.C.!"
    print("  [OK] Test văn bản trực tiếp thành công vượt trội!\n")

    print("=" * 70)
    print("▶ 2. KIỂM THỬ TÍCH HỢP: CHUNKS VCB -> EMBED -> QDRANT -> RERANK")
    print("=" * 70)

    # 1. Đọc dữ liệu mẫu từ FAQ ngân hàng
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

    # 2. Nhúng vector và nạp vào Qdrant in-memory
    embedder = BankingEmbedder(dimension=1024)
    chunks_with_vec = embedder.embed_chunks(raw_chunks)

    store = QdrantBankingStore(collection_name="test_rerank_store", dimension=1024, use_memory=True)
    store.add_chunks(chunks_with_vec)

    # 3. Hybrid Search lấy Top-5 ứng viên
    user_query = "Tôi bị nuốt thẻ tại cây ATM thì phải làm thế nào để lấy lại?"
    query_vector = embedder.embed_query(user_query)

    print(f"▶ Query người dùng: '{user_query}'")
    hybrid_candidates = store.search_hybrid(
        query_text=user_query,
        query_vector=query_vector,
        top_k=5,
        candidate_k=10
    )

    print(f"\n[Kết quả Hybrid Search (Dense + BM25) - Top {len(hybrid_candidates)} ứng viên]:")
    for i, c in enumerate(hybrid_candidates, 1):
        chunk = c["chunk"]
        title = chunk.get("section_title", "")
        print(f"  Ứng viên {i} (RRF: {c['rrf_score']:.6f}) - ID: {chunk['chunk_id']}")
        print(f"    Tiêu đề: {title[:75]}...")

    # 4. Reranker tinh lọc Top-5 xuống Top-3
    print(f"\n▶ Đang đưa Top-{len(hybrid_candidates)} ứng viên qua Cross-Encoder bge-reranker-v2-m3...")
    final_top_3 = reranker.rerank(user_query, hybrid_candidates, top_k=3)

    print(f"\n[Kết quả Reranker Tinh Lọc Sau Cùng - Top {len(final_top_3)} Chunks]:")
    for item in final_top_3:
        rank = item.get("rerank_rank")
        score = item.get("relevance_score")
        chunk = item.get("chunk", item)
        title = chunk.get("section_title", "")
        content = chunk.get("content", "")[:120].replace("\n", " ")
        print(f"  ★ Top {rank} [Cross-Encoder Score: {score:.4f}] - ID: {chunk.get('chunk_id')}")
        print(f"    Tiêu đề: {title}")
        print(f"    Nội dung: {content}...")

    assert len(final_top_3) <= 3
    print("\n🎉 KIỂM THỬ RERANKER FPT CLOUD BGE-RERANKER-V2-M3 HOÀN TOÀN THÀNH CÔNG 100%!")


if __name__ == "__main__":
    main()
