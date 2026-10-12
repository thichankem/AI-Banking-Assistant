"""
Script kiểm thử toàn trình Hệ thống AI Banking Dual-Indexing RAG Pipeline (5 Bước)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from rag_banking.pipeline import BankingRAGPipeline
from rag_banking.config import Config

def main():
    pipe = BankingRAGPipeline()

    # Nạp các tệp tài liệu nghiệp vụ tiêu biểu (Pháp lý eKYC, Quy trình, FAQ, Ngân hàng số)
    print("\n--- BẮT ĐẦU PHA INGESTION & DUAL-INDEXING ---", flush=True)
    dataset_dir = Config.RETRIEVAL_DATASET_DIR
    target_files = [
        "phap_ly__dieu_kien_dieu_khoan_mo_tai_khoan_truc_tuyen_ekyc.md",
        "phap_ly__dieu_kien_dieu_khoan_mo_va_su_dung_tai_khoan_khcn.md",
        "quy_trinh__mo_tai_khoan_thanh_toan_ekyc_va_mo_so_dep.md",
        "faq__ngan_hang_so_vcb_digibank_va_smart_otp.md",
        "ngan_hang_so__quy_chuan_dich_vu_digibank_va_han_muc.md",
        "cho_vay__vay_tieu_dung_co_tai_san_bao_dam.md"
    ]

    all_chunks = []
    for fname in target_files:
        fpath = dataset_dir / fname
        if fpath.exists():
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            chunks = pipe.chunker.chunk_text(text, source_name=fname)
            all_chunks.extend(chunks)

    print(f" -> Đã trích xuất tổng cộng {len(all_chunks)} chunks có bảo toàn Breadcrumbs.", flush=True)

    # Nạp song song vào Dual-Index:
    # 1. Cosine HNSW 1024-d
    print(" -> Lập chỉ mục Dense Vector Store (Cosine HNSW 1024 chiều)...", flush=True)
    texts_to_embed = [c["content"] for c in all_chunks]
    vectors = pipe.embedding_model.embed_documents(texts_to_embed, batch_size=32)
    pipe.vector_store.add_chunks(all_chunks, vectors)

    # 2. Sparse Okapi BM25
    print(" -> Lập chỉ mục Sparse Okapi BM25...", flush=True)
    pipe.bm25_search.index(all_chunks)
    pipe.indexed_chunks = all_chunks
    print(f"[Thành công] Hoàn tất nạp {len(all_chunks)} chunks vào Dual-Index!", flush=True)

    # Thử nghiệm câu hỏi điển hình chứa các từ viết tắt chuyên ngành: eKYC, TKTT, Smart OTP, VCB Digibank
    query = "Hạn mức giao dịch và điều kiện mở TKTT trực tuyến qua eKYC trên app VCB Digibank xác thực Smart OTP là gì?"
    result = pipe.run_pipeline(query)

    print("\n" + "=" * 65, flush=True)
    print("1. KẾT QUẢ BƯỚC 1: QUERY PARSER", flush=True)
    print("=" * 65, flush=True)
    pq = result["step1_parsed_query"]
    print(f"Câu hỏi gốc: {pq.raw_query}", flush=True)
    print(f"Câu chuẩn hóa: {pq.normalized_query}", flush=True)
    print(f"Từ viết tắt phát hiện: {pq.detected_acronyms}", flush=True)
    print(f"Từ khóa BM25: {pq.keywords[:10]}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("2. KẾT QUẢ BƯỚC 2: HYBRID RETRIEVAL (TOP-5 ỨNG VIÊN - RRF k=60)", flush=True)
    print("=" * 65, flush=True)
    for c in result["step2_candidates_top5"]:
        chunk = c["chunk"]
        print(f"Top {c['candidate_rank']}: RRF={c['rrf_score']} | DenseRank={c['dense_rank']} | BM25Rank={c['bm25_rank']}", flush=True)
        print(f"  + Nguồn: {chunk.get('source')} | Phân mục: {chunk.get('breadcrumb', '')[:60]}", flush=True)
        print(f"  + Pháp lý: {chunk.get('legal_ref', 'N/A')}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("3. KẾT QUẢ BƯỚC 3: RERANKING (TINH LỌC TOP-3 CHUNKS)", flush=True)
    print("=" * 65, flush=True)
    for c in result["step3_top3_chunks"]:
        chunk = c["chunk"]
        print(f"Top {c['rerank_rank']} (Gốc: Top {c['original_hybrid_rank']}): FinalScore={c['final_rerank_score']}", flush=True)
        print(f"  + Canonical Matching: {c['canonical_matching_score']} | Lexical Overlap: {c['lexical_overlap_score']}", flush=True)
        print(f"  + Nguồn: {chunk.get('source')} | Pháp lý: {chunk.get('legal_ref', 'N/A')}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("4. KẾT QUẢ BƯỚC 4: CONTEXT SYNTHESIZER (MARKDOWN & TOKEN BUDGET)", flush=True)
    print("=" * 65, flush=True)
    ctx = result["step4_synthesized_context"]
    print(f"Tổng Tokens: {ctx.total_tokens} (Ngân sách tối đa: 2000)", flush=True)
    print(f"Thẻ căn cứ pháp lý: {ctx.legal_citations}", flush=True)
    print(f"Nguồn tài liệu: {ctx.sources_used}", flush=True)
    print(f"Số chunks đã gói: {ctx.chunks_included}", flush=True)
    print("Preview Markdown:\n" + ctx.markdown_context[:400] + "...\n", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("5. KẾT QUẢ BƯỚC 5: LLM SERVING (DEEPSEEK-V4-FLASH + GUARDRAILS)", flush=True)
    print("=" * 65, flush=True)
    print(result["answer"], flush=True)

    print("\n" + "=" * 65, flush=True)
    print("THỐNG KÊ ĐỘ TRỄ TOÀN HỆ THỐNG (LATENCY METRICS)", flush=True)
    print("=" * 65, flush=True)
    for step, ms in result["latency"].items():
        print(f"  - {step}: {ms} ms", flush=True)

if __name__ == "__main__":
    main()
