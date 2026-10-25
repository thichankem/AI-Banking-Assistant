"""
Script Kiểm thử Toàn trình Hệ thống RAG Banking
Sử dụng dữ liệu thực tế từ: dataset/retrieval/retrieval_dataset hoặc dataset/chunking_dataset
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
    chunking_dir = Config.CHUNKING_DATASET_DIR
    retrieval_dir = Config.RETRIEVAL_DATASET_DIR
    if chunking_dir.exists():
        dataset_dir = chunking_dir
    else:
        dataset_dir = retrieval_dir
    print(f"[*] Đang nạp dữ liệu kiểm thử từ: {dataset_dir}")
    
    if dataset_dir == chunking_dir:
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
    else:
        target_docs = [
            "tien_ich__bieu_lai_suat_tien_gui_va_ty_gia_20_ngoai_te.md",
            "the_tin_dung__vcb_vibe_platinum_dac_quyen_va_bieu_phi.md",
            "quy_trinh__mo_tai_khoan_thanh_toan_ekyc_va_xac_thuc_nfc.md",
            "bao_mat__nguyen_tac_an_toan_va_canh_bao_5_hinh_thuc_lua_dao.md"
        ]
        pipeline.ingest_documents(dataset_dir=dataset_dir, filenames=target_docs)

        test_queries = [
            "Lãi suất tiền gửi tiết kiệm tại quầy cho kỳ hạn 12 tháng là bao nhiêu %/năm?",
            "Tỷ giá mua chuyển khoản và giá bán ra của đồng USD là bao nhiêu VNĐ?",
            "Mức trách nhiệm bồi thường bảo hiểm du lịch toàn cầu tối đa của thẻ Vibe Platinum là bao nhiêu?",
            "Quy trình chụp ảnh CCCD gắn chip và quét chip NFC khi mở tài khoản eKYC trên VCB Digibank được thực hiện như thế nào?"
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
