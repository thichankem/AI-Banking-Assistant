# HỆ THỐNG ĐO ĐẠC VÀ ĐÁNH GIÁ MA TRẬN ĐỘ ĐO OCR LOCAL CHO NGÂN HÀNG
## Thư mục: `benchmark/ocr/eval_script/`

> [!IMPORTANT]
> - **Nguyên tắc cốt lõi**: Toàn bộ hệ thống đo đạc và đánh giá chạy **100% Offline Local** trên máy tính nội bộ, không gọi bất kỳ API bên ngoài nào, bảo đảm tuyệt đối tính an toàn và bảo mật dữ liệu ngân hàng.
> - **Hỗ trợ chuyên biệt**: Đo đạc toàn diện tiếng Việt (chuẩn hóa Unicode NFC, Diacritic-CER, GCER) và các thực thể tài chính (số tiền, tỷ lệ %, ngày tháng, số tài khoản, KIE eKYC).

---

## 📁 1. Cấu Trúc Thư Mục `eval_script`

```
benchmark/ocr/eval_script/
├── ocr_metrics.py          # Module lõi tính toán các chỉ số độ đo (CER, WER, NED, GCER, Diacritic, Numeric, KIE)
├── eval_benchmark.py       # Runner CLI quét toàn bộ thư mục mẫu, đối soát đối đầu các engine OCR và xuất báo cáo
├── eval_results.json       # Kết quả định lượng chi tiết dạng JSON
├── EVAL_SUMMARY_REPORT.md  # Báo cáo tổng hợp đối đầu dạng Markdown
└── README.md               # Tài liệu hướng dẫn sử dụng
```

---

## 📐 2. Các Độ Đo Được Cài Đặt Trong `ocr_metrics.py`

### 1. Ký tự & Từ (Character & Word Level)
* **`calculate_cer(ref, hyp)`**: Character Error Rate dựa trên khoảng cách Levenshtein:
  $$CER = \frac{S + D + I}{N_{\text{ref}}} \times 100\%$$
* **`calculate_wer(ref, hyp)`**: Word Error Rate dựa trên khoảng cách Levenshtein ở cấp độ từ (whitespace tokenization).
* **`calculate_ned(ref, hyp)`**: Normalized Edit Distance ($S_{ED}$):
  $$S_{ED} = \max\left(0, 1 - \frac{\text{EditDist}(ref, hyp)}{\max(|ref|, |hyp|)}\right)$$

### 2. Dấu Tiếng Việt & Ngữ Âm (Vietnamese Diacritics & Grapheme)
* **`calculate_diacritic_metrics(ref, hyp)`**:
  * `diacritic_accuracy`: Tỷ lệ bảo toàn các nguyên âm có dấu tiếng Việt (`á, à, ả, ã, ạ, ă, ắ...`).
  * `diacritic_cer`: Tỷ lệ lỗi tính riêng trên tập ký tự có dấu thanh/mũ/móc.
* **`calculate_gcer(ref, hyp)`**: Grapheme Cluster Error Rate — đo lỗi theo cụm ký tự ngữ âm thay vì tách mã code point Unicode NFD.

### 3. Số Liệu Tài Chính & Ngân Hàng (Financial Fact-Level)
* **`calculate_numeric_metrics(ref, hyp)`**:
  * Tự động trích xuất số tiền (`15.000.000 VND`, `15,000,000.50`), tỷ lệ % (`8.5%`), ngày tháng (`16/12/2025`), số tài khoản/CCCD (`0011001234567`).
  * Tính toán `numeric_precision`, `numeric_recall`, `numeric_f1`, và `numeric_accuracy`.

### 4. Bóc Tách Biểu Mẫu & eKYC (Key Information Extraction - KIE)
* **`calculate_kie_field_metrics(ref_fields, hyp_fields, anls_threshold=0.5)`**:
  * Đánh giá theo từng trường dữ liệu: `Họ tên`, `Số CCCD`, `Ngày sinh`, `Địa chỉ`.
  * Tính Exact Match (EM) và Average Normalized Levenshtein Similarity (ANLS).

---

## 🚀 3. Hướng Dẫn Sử Dụng

### Cách 1: Sử dụng trong code Python
```python
from benchmark.ocr.eval_script.ocr_metrics import evaluate_ocr_pair

gt_text = "NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM THÔNG BÁO BIỂU PHÍ NGÀY 16/12/2025: 50.000 VND"
ocr_text = "NGAN HANG TMCP NGOAI THUONG VIET NAM THONG BAO BIEU PHI NGAY 16/12/2025: 50.000 VND"

results = evaluate_ocr_pair(gt_text, ocr_text)

print(f"CER: {results['cer']}%")
print(f"WER: {results['wer']}%")
print(f"Bảo toàn dấu: {results['diacritic_accuracy']}%")
print(f"Lỗi dấu Diacritic-CER: {results['diacritic_cer']}%")
print(f"Độ chính xác số tiền: {results['numeric_accuracy']}%")
```

### Cách 2: Chạy Runner Benchmark CLI trên toàn bộ tập mẫu
```powershell
# Chạy đánh giá mặc định (RapidOCR vs Tesseract v5 vie trên 20 tài liệu ngân hàng)
python benchmark/ocr/eval_script/eval_benchmark.py

# Chỉ định thư mục tùy chỉnh
python benchmark/ocr/eval_script/eval_benchmark.py `
    --gt_dir "benchmark/ocr/vision_groundtruth" `
    --tess_dir "benchmark/ocr/tesseract_output" `
    --rapid_dir "benchmark/ocr/rapidocr_output" `
    --out_json "benchmark/ocr/eval_script/eval_results.json" `
    --out_md "benchmark/ocr/eval_script/EVAL_SUMMARY_REPORT.md"
```

---

## 📊 4. Kết Quả Benchmark Thực Nghiệm Trên 20 Tài Liệu Vietcombank

| Chỉ số Đánh Giá | RapidOCR (ONNX) | Tesseract v5 (`vie`) | Mức Chênh Lệch (Δ) | Đánh Giá Tác Động |
|---|---|---|---|---|
| **Bảo Toàn Dấu Tiếng Việt** | **12.93%** | **94.83%** | **+81.90%** | Tesseract bảo toàn dấu xuất sắc |
| **Tỷ lệ Lỗi Dấu (Diacritic-CER)** | **96.69%** | **27.22%** | **-69.47%** | Giảm gần 70 điểm % lỗi dấu |
| **Tỷ lệ Lỗi Ký tự (Macro-CER)** | **54.81%** | **26.33%** | **-28.48%** | Giảm mạnh tỷ lệ sai lệch ký tự |
| **Tỷ lệ Lỗi Cấp Từ (Macro-WER)** | **100.28%** | **30.35%** | **-69.93%** | Triệt tiêu lỗi dính chữ / ngắt từ |
| **Độ Tương Đồng Chuẩn Hóa (NED)** | **0.5325** | **0.8002** | **+0.2677** | Độ tương đồng tăng đáng kể |
| **Độ Chính Xác Số Liệu Tài Chính** | **72.66%** | **85.74%** | **+13.08%** | Bảo toàn chuẩn xác số tiền & tỷ giá |
| **Thời Gian Xử Lý Bình Quân** | < 0.1s | **1.47s** | +1.47s | Tốc độ đáp ứng tốt trên CPU |
