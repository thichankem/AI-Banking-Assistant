"""
Kiểm thử End-to-End: Nghiệp vụ Hạn mức mở tài khoản eKYC Vietcombank (Mục 1.4)
Câu hỏi: "Hạn mức mở tài khoản eKYC tối đa bao nhiêu 1 ngày?"
Code siêu clean, siêu đơn giản, minh bạch từng bước và đo lường độ trễ thực tế.
"""

import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

from rag_banking.pipeline import BankingRAGPipeline
from rag_banking.config import Config

def main():
    print("=" * 70)
    print("1.4. KẾT QUẢ KIỂM THỬ END-TO-END VÀ ĐỘ TRỄ THỰC TẾ (VIETCOMBANK EKYC)")
    print("=" * 70)

    # 1. Khởi tạo Pipeline
    pipeline = BankingRAGPipeline()

    # 2. Pha Ingestion: Nạp tài liệu quy chế hạn mức eKYC
    dataset_dir = Config.RETRIEVAL_DATASET_DIR
    target_file = dataset_dir / "phap_ly__quy_che_ekyc_vietcombank_han_muc.md"
    
    text = target_file.read_text(encoding="utf-8")
    chunks = pipeline.chunker.chunk_text(text, source_name="phap_ly__quy_che_ekyc_vietcombank_han_muc.md")
    
    # Dual-Indexing: Vector Store 1024-d + BM25
    vectors = pipeline.embedding_model.embed_documents([c["content"] for c in chunks])
    pipeline.vector_store.add_chunks(chunks, vectors)
    pipeline.bm25_search.index(chunks)
    pipeline.indexed_chunks = chunks
    print(f"[*] Đã lập chỉ mục kép (Dual-Index) thành công {len(chunks)} chunks quy chế eKYC.\n")

    # 3. Thực thi truy vấn End-to-End
    query = "Hạn mức mở tài khoản eKYC tối đa bao nhiêu 1 ngày?"
    print(f"[CÂU HỎI NGHIỆP VỤ]: \"{query}\"\n")

    t_start = time.time()
    result = pipeline.run_pipeline(query)
    total_latency = round((time.time() - t_start) * 1000, 1)

    # 4. Hiển thị kết quả chuẩn theo yêu cầu mục 1.4
    print("=" * 70)
    print("KẾT QUẢ TRUY VẤN & SUY LUẬN TỪ HỆ THỐNG:")
    print("=" * 70)
    print(f"Câu trả lời của bot:\n{result['answer']}\n")
    print(f"Dẫn chứng nguồn (Citations): {result['legal_citations']}\n")

    print("-" * 70)
    print("BẢNG THỐNG KÊ ĐỘ TRỄ CHI TIẾT (LATENCY BENCHMARK):")
    print("-" * 70)
    lat = result["latency"]
    print(f"  1. Query Parser (Chuẩn hóa từ khóa):            {lat['step1_query_parser_ms']:>6.1f} ms")
    print(f"  2. Hybrid Retrieval (Cosine HNSW + BM25 + RRF): {lat['step2_hybrid_retrieval_ms']:>6.1f} ms")
    print(f"  3. Reranking (So khớp câu hỏi chuẩn & từ vựng): {lat['step3_reranking_ms']:>6.1f} ms")
    print(f"  4. Context Synthesizer (Đóng gói Markdown):     {lat['step4_context_synthesizer_ms']:>6.1f} ms")
    print(f"  5. LLM Serving (DeepSeek-V4-Flash Guardrails):   {lat['step5_llm_serving_ms']:>6.1f} ms")
    print(f"  -> TỔNG THỜI GIAN PHẢN HỒI (END-TO-END):        {total_latency:>6.1f} ms ({total_latency/1000:.2f}s)")
    print("=" * 70)

if __name__ == "__main__":
    main()
