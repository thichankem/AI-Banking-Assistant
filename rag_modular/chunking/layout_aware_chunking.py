"""
Thuật toán Document Layout-Aware Chunking (Table / Header Extraction)
Được triển khai cho thư mục rag_modular_study theo chuẩn mô-đun hóa độc lập.
"""

import sys
from pathlib import Path

# Tái sử dụng LayoutAwareChunker từ rag_banking
from rag_banking.chunking.layout_aware_chunker import LayoutAwareChunker

def run_sample():
    sys.stdout.reconfigure(encoding="utf-8")
    chunker = LayoutAwareChunker()
    dataset_dir = Path("dataset/retrieval/retrieval_dataset")
    sample_file = dataset_dir / "tien_ich__bieu_lai_suat_tien_gui_va_ty_gia_20_ngoai_te.md"
    
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunker.chunk_document(content, source_name=sample_file.name)
        print(f"Trích xuất thành công {len(chunks)} chunks với Document Layout-Aware Chunking!")
        types = {}
        for c in chunks:
            t = c['chunk_type']
            types[t] = types.get(t, 0) + 1
        print("Phân bố loại chunk:", types)

if __name__ == "__main__":
    run_sample()
