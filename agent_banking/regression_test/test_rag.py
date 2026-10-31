"""
Script Kiểm thử Toàn trình Hệ thống RAG Banking
Sử dụng dữ liệu thực tế DOCX từ: dataset/chunking_dataset (Chuẩn hóa Canonical Ingestion)
Kiểm tra:
1. Document Layout-Aware Chunking (Table / Header Extraction)
2. Weaviate Vector Database
3. BM25 Keyword Search
4. Hybrid Retrieval (RRF)
5. RAG Generation (LLM Answer with Citations)
"""

import sys
from pathlib import Path

# Đảm bảo in tiếng Việt chuẩn trên Windows terminal
sys.stdout.reconfigure(encoding="utf-8")

# Thêm đường dẫn gốc dự án vào sys.path để import
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.pipeline import BankingRAGPipeline

def run_test():
    print("\n" + "#" * 70)
    print("      CHƯƠNG TRÌNH KIỂM THỬ RAG BANKING (AI ASSISTANT)")
    print("#" * 70 + "\n")

    # 1. Khởi tạo Pipeline RAG
    pipeline = BankingRAGPipeline()

    from agent_banking.config import Config
    dataset_dir = Config.CHUNKING_DATASET_DIR
    print(f"[*] Đang nạp dữ liệu kiểm thử DOCX chuẩn hóa từ: {dataset_dir}")

    # Kiểm thử định dạng DOCX chuẩn hóa của hệ thống:
    target_docs = [
        "vcb_md_15__tien_ich__bieu_lai_suat_tien_gui_va_ty_gia_20_ngoai_te.docx",
        "vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.docx"
    ]
    pipeline.ingest_documents(dataset_dir=dataset_dir, filenames=target_docs)

    test_queries = [
        # Test Case 1: Tra cứu Biểu lãi suất và tỷ giá (DOCX Layout-Aware)
        "Lãi suất tiền gửi tiết kiệm VND tại quầy kỳ hạn 12 tháng là bao nhiêu %/năm và tỷ giá USD mua chuyển khoản là bao nhiêu?",

        # Test Case 2: Tra cứu DOCX FAQ & Điều khoản thẻ (DOCX Layout-Aware)
        "Khi bị mất thẻ tín dụng hoặc phát hiện giao dịch thẻ lạ, khách hàng cần thực hiện những bước khẩn cấp nào?"
    ]

    print("\n" + "=" * 70)
    print("      BẮT ĐẦU CHẠY CÁC TEST CASES TRUY VẤN VÀ SINH CÂU TRẢ LỜI")
    print("=" * 70 + "\n")

    for idx, query in enumerate(test_queries, start=1):
        print(f"\n{'='*25} TEST CASE {idx} {'='*25}")
        print(f"CÂU HỎI: {query}\n")

        # Thực thi RAG: Hybrid Retrieve + Sinh câu trả lời với top_k=5
        response = pipeline.ask(query, top_k=5)

        # Hiển thị các chunks được Hybrid Retrieval lựa chọn
        print("\n--- [TOP CHUNKS TRÍCH XUẤT BỞI HYBRID RETRIEVAL (RRF)] ---")
        for c_idx, item in enumerate(response["retrieved_chunks"], start=1):
            chunk = item["chunk"]
            print(f"\n[Chunk {c_idx}]")
            print(f" - Loại chunk : {chunk.get('chunk_type')} (Layout-Aware)")
            print(f" - Nguồn file : {chunk.get('source')}")
            print(f" - Phân cấp   : {chunk.get('breadcrumb')}")
            print(f" - Điểm số    : RRF = {item['rrf_score']} | Dense Score = {item['dense_score']} (Rank {item['dense_rank']}) | BM25 Score = {item['bm25_score']} (Rank {item['bm25_rank']})")
            print(f" - Trích đoạn : {chunk.get('content')[:180].strip()}...")

        # Hiển thị câu trả lời hoàn chỉnh từ AI
        print("\n--- [CÂU TRẢ LỜI CỦA AI BANKING ASSISTANT] ---")
        print(response["answer"])

        print("\n--- [NGUỒN THAM CHIẾU ĐÃ DÙNG] ---")
        for src in response["sources_used"]:
            print(f" * {src}")

        print("\n" + "-" * 70)

    # Đóng tài nguyên
    pipeline.close()
    print("\n[Hoàn thành] Đã kết thúc toàn bộ bài kiểm thử RAG Banking thành công rực rỡ!\n")

if __name__ == "__main__":
    run_test()
