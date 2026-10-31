"""
Kiểm thử End-to-End: Nghiệp vụ Hạn mức mở tài khoản eKYC (Mục 1.4)
Câu hỏi: "Hạn mức mở tài khoản eKYC tối đa bao nhiêu 1 ngày?"
Sử dụng tài liệu chuẩn hóa DOCX, không dùng text/md flow.
"""

import sys
import time
from pathlib import Path

# Đảm bảo UTF-8 trên Windows terminal
sys.stdout.reconfigure(encoding="utf-8")

# Thêm đường dẫn gốc dự án vào sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.config import Config

def main():
    print("=" * 70)
    print("1.4. KẾT QUẢ KIỂM THỬ END-TO-END VÀ ĐỘ TRỄ THỰC TẾ (EKYC)")
    print("=" * 70)

    # 1. Khởi tạo Pipeline
    pipeline = BankingRAGPipeline()

    # 2. Pha Ingestion: Nạp tài liệu quy chế hạn mức eKYC (.docx chuẩn hóa)
    candidates = [
        Config.CHUNKING_DATASET_DIR / "dataset_1" / "docx",
        Config.ROOT_DIR / "dataset" / "dataset_1" / "docx",
        Config.ROOT_DIR / "agent_banking" / "chunking" / "chunking_dataset" / "dataset_1" / "docx"
    ]
    dataset_dir = candidates[0]
    for c in candidates:
        if c.exists():
            dataset_dir = c
            break

    target_file = dataset_dir / "okbank_014__nhom1_tai_khoan__quy_trinh_mo_tai_khoan_ekyc_nfc.docx"
    if not target_file.exists():
        docx_files = list(dataset_dir.glob("*ekyc*.docx"))
        if docx_files:
            target_file = docx_files[0]
        else:
            docx_files = list(dataset_dir.glob("*.docx"))
            if docx_files:
                target_file = docx_files[0]

    chunks = pipeline.chunker.chunk_docx(target_file)
    
    # Dual-Indexing: Vector Store 1024-d + BM25
    pipeline.ingest_chunks(chunks)
    print(f"[*] Đã lập chỉ mục kép (Dual-Index) thành công {len(chunks)} chunks quy chế eKYC từ {target_file.name}.\n")

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
    print(f" - B1: Query Parser        : {lat['step1_query_parser_ms']} ms")
    print(f" - B2: Hybrid Retrieval    : {lat['step2_hybrid_retrieval_ms']} ms (Dense: {lat['dense_retrieval_ms']}ms, BM25: {lat['bm25_retrieval_ms']}ms, RRF: {lat['rrf_fusion_ms']}ms)")
    print(f" - B3: Cross-Encoder Rerank: {lat['step3_rerank_ms']} ms")
    print(f" - B4: Context Synthesis   : {lat['step4_context_ms']} ms")
    print(f" - B5: LLM Generation      : {lat['step5_generation_ms']} ms (Model: {result['model']})")
    if total_latency < 3000:
        sla_status = "ĐẠT"
    else:
        sla_status = "CHƯA ĐẠT"
    print(f" >> TỔNG ĐỘ TRỄ TOÀN TRÌNH : {total_latency} ms (SLA < 3000ms: {sla_status})")
    print("=" * 70)

if __name__ == "__main__":
    main()
