# AI Banking Assistant: Chunking Evaluation & Benchmark Suite

Bộ khung kiểm thử, đo lường tự động (Evaluation Suite) và kịch bản Ablation Study chuyên sâu phục vụ hệ thống phân đoạn nhận thức bố cục `LayoutAwareChunker / ChunkingPipeline` (Vietcombank RAG).

---

## 1. Cấu trúc Phân hệ Cô lập (`benchmarks/`)

Toàn bộ mã nguồn, dữ liệu kiểm thử và báo cáo được cô lập 100% trong thư mục này, tuyệt đối không can thiệp vào codebase gốc hay `experiments/core_upgrades/`:

```
benchmarks/
├── datasets/
│   ├── __init__.py
│   ├── synthetic_qa_generator.py   # Trình sinh câu hỏi kiểm thử bám sát 3 ca khó
│   └── golden_qa.json              # Tập dữ liệu kiểm thử chuẩn (30 mẫu đa dạng)
├── evaluators/
│   ├── __init__.py
│   ├── ocr_evaluator.py            # Đo lường CER, WER và đối sánh lọc mộc đỏ HSV
│   ├── structure_evaluator.py      # Đo lường Heading F1, Breadcrumb & Table Integrity
│   ├── retrieval_evaluator.py      # Đo lường Recall@5, Recall@10, MRR@10, NDCG@10 (BM25)
│   └── system_profiler.py          # Đo lường Latency/page, Throughput, Peak RAM & VRAM (RTX 4060)
├── ablation_runners/
│   ├── __init__.py
│   └── run_ablation.py             # Script đối sánh Baseline vs Pipeline v1 vs Pipeline v2
├── reports/
│   ├── __init__.py
│   ├── report_generator.py         # Trình xuất bản báo cáo đa định dạng (MD, CSV, JSON)
│   ├── benchmark_summary.md        # Bảng tổng hợp đối sánh Markdown trực quan
│   ├── benchmark_summary.csv       # Dữ liệu bảng tính chi tiết
│   ├── benchmark_summary.json      # Kết quả phân tích có cấu trúc
│   └── ablation_raw_results.json   # Dữ liệu đo lường thô
└── README.md                       # Tài liệu hướng dẫn CLI
```

---

## 2. Hướng dẫn Chạy Kiểm thử (CLI Commands)

> **Lưu ý:** Chạy các lệnh dưới đây với môi trường virtual environment của dự án (`.venv`).

### 2.1. Sinh bộ câu hỏi kiểm thử Golden QA
Chạy `synthetic_qa_generator.py` trên một tập chunks bất kỳ (hỗ trợ các backend: `heuristic` offline, `ollama` hoặc `openrouter`):
```bash
python -m benchmarks.datasets.synthetic_qa_generator \
  --input_chunks "dataset/chunking_dataset/chunks.json" \
  --output_file "benchmarks/datasets/golden_qa.json" \
  --backend heuristic \
  --samples_per_cat 10
```

### 2.2. Chạy kịch bản Đối sánh Ablation Study toàn diện
Chạy đồng thời **Config 0 (Baseline)**, **Config 1 (Pipeline v1)** và **Config 2 (Pipeline v2)** trên cùng tập tệp đầu vào (`.docx`, `.pdf` số, `.pdf` scan):
```bash
python -m benchmarks.ablation_runners.run_ablation \
  --files "dataset/chunking_dataset/pdf/vcb_pdf_22_bieu_phi_the_tdqt_2026.pdf" "dataset/chunking_dataset/pdf/vcb_pdf_27_dieu_kien_dieu_khoan_tktt_truc_tuyen.pdf" \
  --golden_qa "benchmarks/datasets/golden_qa.json" \
  --output_dir "benchmarks/reports"
```

### 2.3. Tái xuất bản Báo cáo Tổng hợp (Markdown & CSV)
Tổng hợp lại kết quả đo lường thành bảng biểu trực quan:
```bash
python -m benchmarks.reports.report_generator "benchmarks/reports/ablation_raw_results.json"
```

---

## 3. Cơ chế Mở rộng (Upgrade Hook Interface)

Khi phân hệ `experiments/core_upgrades/` (ví dụ: Config 3: Unified Hybrid PDF hoặc Config 4: Chunk Budget Profiles) hoàn thành ở phiên làm việc song song, có thể dễ dàng cắm vào chạy benchmark chung bằng hook `register_upgrade_pipeline`:

```python
from benchmarks.ablation_runners.run_ablation import register_upgrade_pipeline, run_ablation_benchmark

# 1. Định nghĩa wrapper function cho cấu hình mới
def run_hybrid_pdf_config(file_paths):
    # Khởi tạo mô-đun từ core_upgrades
    from experiments.core_upgrades.hybrid_pdf import HybridPDFChunker
    chunker = HybridPDFChunker()
    all_chunks = []
    for fp in file_paths:
        all_chunks.extend(chunker.chunk_file(fp))
    return all_chunks

# 2. Đăng ký cấu hình mới vào registry
register_upgrade_pipeline("Config 3 - Unified Hybrid PDF", run_hybrid_pdf_config)

# 3. Kích hoạt benchmark đối sánh cùng Baseline, v1, v2
run_ablation_benchmark(
    sample_files=["path/to/test.pdf"],
    golden_qa=loaded_qa_data,
    output_dir="benchmarks/reports"
)
```

---

## 4. Định nghĩa 4 Nhóm Metrics

| Nhóm Metric | Chỉ số chính | Mục đích đo lường |
|:---|:---|:---|
| **1. OCR Quality** | `CER` (Character Error Rate), `WER` (Word Error Rate) | Đánh giá sai số ký tự & từ, chứng minh hiệu quả giảm nhiễu của bộ lọc mộc đỏ HSV. |
| **2. Structure & Breadcrumb** | `Heading F1-Score`, `Breadcrumb Coverage`, `Table Header Retain` | Đánh giá khả năng bảo toàn cấu trúc văn bản, phả hệ đa tầng và tính nguyên tử của bảng biểu. |
| **3. Retrieval Effectiveness** | `Recall@5`, `Recall@10`, `MRR@10`, `NDCG@10` | Đánh giá xác suất tìm đúng chunk mục tiêu trên 3 nhóm câu hỏi khó (Pháp lý đa tầng, Bảng biểu, Nối đoạn xuyên trang). |
| **4. System Profiler** | `Latency (ms/trang)`, `Throughput (trang/s)`, `Peak RAM (MB)`, `Peak VRAM (MB)` | Giám sát tài nguyên hệ thống và đảm bảo an toàn bộ nhớ VRAM trên GPU NVIDIA RTX 4060. |
