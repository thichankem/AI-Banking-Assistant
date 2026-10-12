# BÁO CÁO ĐỐI SOÁT ĐỊNH LƯỢNG OCR LOCAL TOÀN DIỆN (OCR LEADERBOARD)

- **Ngày thực hiện**: 2026-10-12 17:13:26
- **Số lượng tài liệu khảo sát**: 20 mẫu đại diện chính thức của Vietcombank
- **Môi trường**: 100% Offline Local (CPU / ONNX / Tesseract / VietOCR)

---

## 1. Bảng Xếp Hạng Hiệu Năng Tổng Thể (Overall Leaderboard)

| Chỉ số Đánh Giá (Metric) | **RapidOCR (ONNX)** | **Tesseract v5 (vie)** | **VietOCR (Transformer)** | Ý Nghĩa Thực Tế |
|---|---|---|---|---|
| **Bảo Toàn Dấu Tiếng Việt** | 12.93% | 94.83% | 90.13% | Càng cao càng bảo toàn đúng ngữ nghĩa |
| **Tỷ lệ Lỗi Dấu (Diacritic-CER)** | 96.69% | 27.22% | 29.86% | Đo lỗi riêng trên thanh điệu tiếng Việt |
| **Tỷ lệ Lỗi Ký Tự (Macro-CER)** | 54.81% | 26.33% | 30.06% | Chỉ số vàng đo độ chính xác toàn trang |
| **Tỷ lệ Lỗi Cấp Từ (Macro-WER)** | 100.28% | 30.35% | 34.14% | Đo lỗi ranh giới từ và dính từ |
| **Độ Tương Đồng Chuẩn Hóa (NED)** | 0.5325 | 0.8002 | 0.7358 | Chuẩn hóa về [0, 1], càng gần 1 càng tốt |
| **Lỗi Cụm Ngữ Âm (GCER)** | 54.81% | 26.33% | 30.06% | Đo lỗi trên cụm ký tự gốc + dấu ghép |
| **Độ Chính Xác Số Liệu Tài Chính** | 72.66% | 85.74% | 83.73% | Bảo toàn số tiền, ngày tháng, tỷ lệ % |
| **Thời Gian Xử Lý / Trang** | 0.00s | 1.47s | 18.41s | Tốc độ xử lý thực tế trên CPU |

---

## 2. Bảng Chi Tiết CER Từng Tài Liệu Khảo Sát

| STT | Mã Tài Liệu | CER RapidOCR (ONNX) | CER Tesseract v5 (vie) | CER VietOCR (Transformer) |
|---|---|---|---|---|
| 01 | `docx_01_faq_the_ngan_hang` | 44.8% | 31.0% | 22.0% |
| 02 | `docx_02_le_loyalty` | 57.0% | 33.1% | 44.5% |
| 03 | `docx_03_dkdk_ekyc` | 44.1% | 15.2% | 21.7% |
| 04 | `docx_04_faq_digibank_otp` | 110.6% | 91.7% | 84.3% |
| 05 | `docx_05_dkdk_tai_khoan_khcn` | 60.4% | 24.4% | 37.1% |
| 06 | `docx_06_faq_kieu_hoi` | 73.3% | 53.4% | 48.3% |
| 07 | `docx_07_faq_loyalty_doi_qua` | 55.7% | 24.6% | 43.6% |
| 08 | `docx_08_faq_cho_vay_khcn` | 34.9% | 36.9% | 7.3% |
| 09 | `docx_09_urbox_evergreen` | 64.8% | 30.9% | 53.2% |
| 10 | `docx_10_vibe_platinum` | 96.0% | 83.4% | 73.3% |
| 11 | `pdf_01_bctc_2024_hop_nhat` | 67.7% | 0.0% | 0.7% |
| 12 | `pdf_02_bctc_2024_rieng_le` | 68.4% | 1.9% | 1.9% |
| 13 | `pdf_03_trai_phieu_cbtt` | 39.0% | 8.1% | 16.1% |
| 14 | `pdf_04_nq_hdqt_vcbs` | 40.8% | 8.1% | 18.8% |
| 15 | `pdf_05_bieu_phi_the_tdqt` | 32.0% | 15.4% | 19.9% |
| 16 | `pdf_06_bieu_phi_vay_khcn` | 49.1% | 7.8% | 24.8% |
| 17 | `pdf_07_bieu_phi_forex_digi` | 45.0% | 23.6% | 23.3% |
| 18 | `pdf_08_bieu_phi_so_dep` | 18.0% | 15.4% | 3.9% |
| 19 | `pdf_09_bieu_phi_debit_haha` | 36.4% | 17.7% | 13.9% |
| 20 | `pdf_10_dkdk_tktt_truc_tuyen` | 58.1% | 4.1% | 42.7% |
