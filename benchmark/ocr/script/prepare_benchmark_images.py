"""
Script kết xuất hình ảnh chuẩn từ 10 file PDF và 10 file DOCX phục vụ OCR Benchmark.
- PDF: Kết xuất trang đại diện có độ phức tạp cao (bảng biểu, dấu mộc, văn bản) qua pypdfium2.
- DOCX: Chuyển đổi DOCX sang PDF qua Word API (win32com) và kết xuất trang đầu sang PNG.
"""

import sys
from pathlib import Path
import pypdfium2 as pdfium
import win32com.client

# Đảm bảo UTF-8 cho console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BENCHMARK_DIR.parent.parent
PDF_DIR = PROJECT_ROOT / "dataset" / "chunking_dataset" / "pdf"
DOCX_DIR = PROJECT_ROOT / "dataset" / "chunking_dataset" / "docx"

OUTPUT_IMG_DIR = BENCHMARK_DIR / "rendered_images"
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
TEMP_PDF_DIR = OUTPUT_IMG_DIR / "_temp_pdf"
TEMP_PDF_DIR.mkdir(parents=True, exist_ok=True)

PDF_FILES = [
    ("pdf_01_bctc_2024_hop_nhat", "vcb_pdf_01_bctc_2024_hop_nhat_kiem_toan.pdf", 0), # Trang bìa có mộc & EY
    ("pdf_02_bctc_2024_rieng_le", "vcb_pdf_02_bctc_2024_rieng_le_kiem_toan.pdf", 0),   # BCTC riêng lẻ
    ("pdf_03_trai_phieu_cbtt", "vcb_pdf_20_cbtt_trai_phieu_rieng_le_2025.pdf", 0),      # CBTT phát hành TP
    ("pdf_04_nq_hdqt_vcbs", "vcb_pdf_21_nq_hdqt_cctg_vcbs.pdf", 0),                    # Nghị quyết mộc đỏ
    ("pdf_05_bieu_phi_the_tdqt", "vcb_pdf_22_bieu_phi_the_tdqt_2026.pdf", 0),           # Bảng biểu phí thẻ
    ("pdf_06_bieu_phi_vay_khcn", "vcb_pdf_23_bieu_phi_cho_vay_khcn_2025.pdf", 0),       # Bảng biểu phí vay
    ("pdf_07_bieu_phi_forex_digi", "vcb_pdf_24_bieu_phi_chuyen_tien_ngoai_te_digibank.pdf", 0), # Phí ngoại tệ
    ("pdf_08_bieu_phi_so_dep", "vcb_pdf_25_bieu_phi_tai_khoan_so_dep.pdf", 0),          # Phí TK số đẹp
    ("pdf_09_bieu_phi_debit_haha", "vcb_pdf_26_bieu_phi_the_thanh_toan_debit_haha.pdf", 0), # Phí debit
    ("pdf_10_dkdk_tktt_truc_tuyen", "vcb_pdf_27_dieu_kien_dieu_khoan_tktt_truc_tuyen.pdf", 0), # ĐKĐK TKTT
]

DOCX_FILES = [
    ("docx_01_faq_the_ngan_hang", "vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.docx"),
    ("docx_02_le_loyalty", "vcb_md_02__phap_ly__the_le_chi_tiet_chuong_trinh_vcb_loyalty.docx"),
    ("docx_03_dkdk_ekyc", "vcb_md_03__phap_ly__dieu_kien_dieu_khoan_mo_tai_khoan_truc_tuyen_ekyc.docx"),
    ("docx_04_faq_digibank_otp", "vcb_md_04__faq__ngan_hang_so_vcb_digibank_va_smart_otp.docx"),
    ("docx_05_dkdk_tai_khoan_khcn", "vcb_md_05__phap_ly__dieu_kien_dieu_khoan_mo_va_su_dung_tai_khoan_khcn.docx"),
    ("docx_06_faq_kieu_hoi", "vcb_md_06__faq__chuyen_tien_va_nhan_tien_kieu_hoi.docx"),
    ("docx_07_faq_loyalty_doi_qua", "vcb_md_07__faq__vcb_loyalty_tich_diem_va_doi_qua.docx"),
    ("docx_08_faq_cho_vay_khcn", "vcb_md_08__faq__cho_vay_khach_hang_ca_nhan.docx"),
    ("docx_09_urbox_evergreen", "vcb_md_09__phap_ly__the_le_chuong_trinh_uu_dai_urbox_evergreen_visa.docx"),
    ("docx_10_vibe_platinum", "vcb_md_10__the_tin_dung__vcb_vibe_platinum_dac_quyen_va_bieu_phi.docx"),
]


def render_all_samples():
    print("=" * 70)
    print("  KẾT XUẤT 10 ẢNH PDF VÀ 10 ẢNH DOCX PHỤC VỤ OCR BENCHMARK")
    print("=" * 70)

    # 1. Kết xuất 10 tệp PDF
    print("\n[1/2] Đang kết xuất 10 tệp PDF sang ảnh PNG độ phân giải cao...")
    for key, filename, page_idx in PDF_FILES:
        pdf_path = PDF_DIR / filename
        out_png = OUTPUT_IMG_DIR / f"{key}.png"
        try:
            doc = pdfium.PdfDocument(str(pdf_path))
            page = doc[page_idx]
            img = page.render(scale=2).to_pil()
            img.save(str(out_png))
            print(f"  + [PDF] {key}.png (kích thước: {img.size})")
        except Exception as e:
            print(f"  - [Lỗi PDF] {filename}: {e}")

    # 2. Kết xuất 10 tệp DOCX qua Word -> PDF -> PNG
    print("\n[2/2] Đang chuyển đổi và kết xuất 10 tệp DOCX sang ảnh PNG...")
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        for key, filename in DOCX_FILES:
            docx_path = DOCX_DIR / filename
            temp_pdf = TEMP_PDF_DIR / f"{key}.pdf"
            out_png = OUTPUT_IMG_DIR / f"{key}.png"
            try:
                # Convert docx to pdf
                doc_word = word.Documents.Open(str(docx_path.resolve()))
                doc_word.SaveAs(str(temp_pdf.resolve()), FileFormat=17) # wdFormatPDF = 17
                doc_word.Close()

                # Render page 0 to PNG
                doc_pdf = pdfium.PdfDocument(str(temp_pdf))
                img = doc_pdf[0].render(scale=2).to_pil()
                img.save(str(out_png))
                print(f"  + [DOCX] {key}.png (kích thước: {img.size})")
            except Exception as e:
                print(f"  - [Lỗi DOCX] {filename}: {e}")
    finally:
        word.Quit()

    # Dọn dẹp thư mục tạm
    import shutil
    try:
        shutil.rmtree(TEMP_PDF_DIR, ignore_errors=True)
    except Exception:
        pass

    print(f"\n[Hoàn tất] Toàn bộ 20 ảnh mẫu đã lưu tại: {OUTPUT_IMG_DIR}\n")


if __name__ == "__main__":
    render_all_samples()
