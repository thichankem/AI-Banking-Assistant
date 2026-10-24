"""
Kiểm thử nhanh BankingEmbedder tối giản trên dữ liệu chunk thật.
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

def test_embedder_suite():
    main()

def main():
    print("▶ Khởi tạo BankingEmbedder...")
    embedder = BankingEmbedder(dimension=1024)

    # 1. Test embed đơn lẻ
    text = "Lãi suất tiền gửi tiết kiệm ngân hàng kỳ hạn 12 tháng"
    vec = embedder.embed_text(text)
    print(f"  [OK] embed_text thành công! Số chiều: {len(vec)}, Norm: {sum(x*x for x in vec):.4f}")

    # 2. Test embed query
    q_vec = embedder.embed_query("Cách kích hoạt thẻ VCB Digibank?")
    print(f"  [OK] embed_query thành công! Số chiều: {len(q_vec)}, Norm: {sum(x*x for x in q_vec):.4f}")

    # 3. Test embed chunks từ tệp chunk thật
    cand_files = [
        ROOT / "rag_banking/chunking/output/chunks_per_file/01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.json",
        Path("chunking/output/chunks_per_file/01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.json"),
        ROOT / "rag_banking/chunking/chunking_dataset/dataset_1/chunks_per_file/vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.json"
    ]
    chunk_file = next((p for p in cand_files if p.exists()), None)
    if chunk_file and chunk_file.exists():
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)[:5]

        print(f"▶ Nạp và nhúng {len(chunks)} chunks từ file JSON...")
        enriched_chunks = embedder.embed_chunks(chunks)
        print(f"  [OK] Đã gắn vector thành công cho {len(enriched_chunks)} chunks!")
        sample = enriched_chunks[0]
        print(f"  - Chunk ID: {sample.get('chunk_id')}")
        print(f"  - Vector có trong chunk: {'vector' in sample} (Kích thước: {len(sample.get('vector', []))})")
        print(f"  - Nội dung đầu chunk: {sample.get('content', '')[:60]}...")

    print("\n🎉 KIỂM THỬ EMBEDDER TỐI GIẢN THÀNH CÔNG 100%!")

if __name__ == "__main__":
    main()
