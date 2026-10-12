"""
Script trích xuất và lưu trữ Ground Truth chuẩn xác (Vision Reference) cho 10 file PDF và 10 file Word (DOCX).
Tạo 20 tệp Markdown tại: benchmark/ocr/vision_groundtruth/
"""

import sys
import json
from pathlib import Path
import pypdf
import docx

# Đảm bảo UTF-8 cho console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PDF_DIR = PROJECT_ROOT / "dataset" / "chunking_dataset" / "pdf"
DOCX_DIR = PROJECT_ROOT / "dataset" / "chunking_dataset" / "docx"

OUT_VISION_DIR = PROJECT_ROOT / "benchmark" / "ocr" / "vision_groundtruth"
OUT_VISION_DIR.mkdir(parents=True, exist_ok=True)

# 10 PDF Định nghĩa Ground Truth thị giác (Vision Ground Truth)
PDF_VISION_DATA = {
    "pdf_01_bctc_2024_hop_nhat": """# BÁO CÁO TÀI CHÍNH HỢP NHẤT 2024 (VIETCOMBANK)
**Nguồn**: `vcb_pdf_01_bctc_2024_hop_nhat_kiem_toan.pdf` | **Trang**: 1/90

---

Ngân hàng Thương mại Cổ phần
Ngoại thương Việt Nam

Báo cáo tài chính hợp nhất cho năm kết thúc ngày 31 tháng 12 năm 2024

EY
Building a better
working world
""",

    "pdf_02_bctc_2024_rieng_le": """# BÁO CÁO TÀI CHÍNH RIÊNG LẺ 2024 (VIETCOMBANK)
**Nguồn**: `vcb_pdf_02_bctc_2024_rieng_le_kiem_toan.pdf` | **Trang**: 1/86

---

Ngân hàng Thương mại Cổ phần
Ngoại thương Việt Nam

Báo cáo tài chính riêng lẻ cho năm kết thúc ngày 31 tháng 12 năm 2024

EY
Building a better
working world
""",

    "pdf_03_trai_phieu_cbtt": """# CÔNG BỐ THÔNG TIN PHÁT HÀNH TRÁI PHIẾU RIÊNG LẺ
**Nguồn**: `vcb_pdf_20_cbtt_trai_phieu_rieng_le_2025.pdf` | **Trang**: 1/2

---

NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM
Số: 2025/CBTT-VCB
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
Hà Nội, ngày 15 tháng 01 năm 2025

CÔNG BỐ THÔNG TIN KẾT QUẢ PHÁT HÀNH TRÁI PHIẾU RIÊNG LẺ ĐỢT 1 NĂM 2025
Kính gửi:
- Sở Giao dịch Chứng khoán Hà Nội (HNX);
- Ủy ban Chứng khoán Nhà nước.

1. Thông tin về tổ chức phát hành:
- Tên tổ chức phát hành: NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM (VIETCOMBANK)
- Địa chỉ trụ sở chính: 198 Trần Quang Khải, phường Lý Thái Tổ, quận Hoàn Kiếm, thành phố Hà Nội
- Mã chứng khoán: VCB
- Giấy chứng nhận đăng ký doanh nghiệp số: 0100112437 do Sở Kế hoạch và Đầu tư TP Hà Nội cấp.

2. Kết quả phát hành trái phiếu riêng lẻ:
- Tên trái phiếu: Trái phiếu Vietcombank phát hành riêng lẻ đợt 1 năm 2025
- Mã trái phiếu: VCBH2528001
- Loại hình trái phiếu: Trái phiếu không chuyển đổi, không kèm chứng quyền, không có bảo đảm
- Kỳ hạn trái phiếu: 03 năm
- Ngày phát hành: 15/01/2025
- Ngày đáo hạn: 15/01/2028
- Tổng khối lượng phát hành dự kiến: 2.000.000.000.000 VNĐ (Hai nghìn tỷ đồng)
- Tổng khối lượng phát hành thực tế: 2.000.000.000.000 VNĐ (Tỷ lệ thành công 100%).
""",

    "pdf_04_nq_hdqt_vcbs": """# NGHỊ QUYẾT HĐQT PHÊ DUYỆT THỎA THUẬN HỢP TÁC VCBS
**Nguồn**: `vcb_pdf_21_nq_hdqt_cctg_vcbs.pdf` | **Trang**: 1/1

---

NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM
Số: 28154 /VCB-CLTKHĐQT
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
Hà Nội, ngày 31 tháng 12 năm 2025

V/v HĐQT phê duyệt nội dung thỏa thuận hợp tác v/v thực hiện giao dịch CCTG do VCB phát hành bằng phương tiện điện tử giữa KHCN của VCB và VCBS trên hệ thống của VCB

CÔNG BỐ THÔNG TIN BẤT THƯỜNG
Kính gửi:
- Ủy ban Chứng khoán Nhà nước;
- Sở Giao dịch Chứng khoán Việt Nam;
- Sở Giao dịch Chứng khoán Hà Nội;
- Sở Giao dịch Chứng khoán TP HCM.

1. Tên tổ chức: Ngân hàng TMCP Ngoại thương Việt Nam
Mã chứng khoán: VCB
Địa chỉ trụ sở chính: 198 Trần Quang Khải, Hoàn Kiếm, Hà Nội
Điện thoại: 0243 - 9343137
Email: cltkhdqt.ho@vietcombank.com.vn.

2. Nội dung thông tin công bố:
Ngày 30/12/2025, Hội đồng Quản trị Ngân hàng TMCP Ngoại thương Việt Nam (VCB) đã ban hành Nghị quyết số 728/NQ-VCB-HĐQT phê duyệt nội dung Thỏa thuận hợp tác về việc thực hiện giao dịch chứng chỉ tiền gửi do VCB phát hành bằng phương tiện điện tử giữa khách hàng cá nhân của VCB và Công ty TNHH Chứng khoán Ngân hàng Ngoại thương Việt Nam (VCBS) trên hệ thống của VCB (VCBS là công ty con của VCB).

3. Thông tin này được đăng tải lên website của Ngân hàng kể từ ngày 31/12/2025 tại đường dẫn: https://www.vietcombank.com.vn/vi-VN/Nha-dau-tu
Chúng tôi xin cam kết các thông tin công bố trên đây là đúng sự thật và hoàn toàn chịu trách nhiệm trước pháp luật về nội dung các thông tin đã công bố./.

Nơi nhận:
- Như kính gửi;
- CT HĐQT, TGĐ (để b/c);
- TV HĐQT, TV BĐH (để biết);
- Lưu: VT, CL&TKHĐQT.

NGƯỜI ĐƯỢC ỦY QUYỀN CBTT
PHÓ TỔNG GIÁM ĐỐC
(Dấu mộc tròn đỏ: M.S.T: 0100112437 - C.T.C.P NGÂN HÀNG THƯƠNG MẠI CỔ PHẦN NGOẠI THƯƠNG VIỆT NAM - THÀNH PHỐ HÀ NỘI)
Lê Hoàng Tùng
""",

    "pdf_05_bieu_phi_the_tdqt": """# BIỂU PHÍ DỊCH VỤ VÀ LÃI SUẤT THẺ TÍN DỤNG QUỐC TẾ
**Nguồn**: `vcb_pdf_22_bieu_phi_the_tdqt_2026.pdf` | **Trang**: 1/7

---

Vietcombank
Chung niềm tin vững tương lai

BIỂU PHÍ DỊCH VỤ VÀ LÃI SUẤT THẺ TÍN DỤNG QUỐC TẾ
ÁP DỤNG CHO KHÁCH HÀNG CÁ NHÂN TẠI VIETCOMBANK
(Có hiệu lực từ ngày 22/01/2026)

| STT | Dịch vụ | Mức phí |
|---|---|---|
| 1 | Phí phát hành (*) | Miễn phí |
| 2 | Phí thường niên (*) | |
| 2.1 | Thẻ Vietcombank Visa Infinite | 30.000.000 VNĐ/thẻ/năm (Miễn phí cho khách hàng Priority phân hạng Kim cương Elite và Kim cương của VCB) |
| 2.2 | Thẻ Vietcombank Visa Signature | 3.000.000 VNĐ/thẻ/năm (Miễn phí cho khách hàng Priority của VCB)¹ |
| 2.3 | Thẻ Vietcombank Visa Platinum (Visa Platinum)/ Vietcombank Vibe Platinum (Vibe Platinum)/ Vietcombank Cashplus Platinum American Express (Amex Cashplus)/ Vietcombank Mastercard World (Mastercard World)/ Vietcombank JCB Platinum (JCB Platinum)/ Vietcombank Vietnam Airlines Platinum American Express (Amex Platinum) | |
| 2.3.1 | Thẻ chính | 800.000 VNĐ/thẻ/năm |
| 2.3.2 | Thẻ phụ | 500.000 VNĐ/thẻ/năm |
| 2.4 | Phí thường niên thẻ Vietcombank Vibe | |
| 2.4.1 | Thẻ chính | 400.000 VNĐ/thẻ/năm |
| 2.4.2 | Thẻ phụ | 200.000 VNĐ/thẻ/năm |
| 2.5 | Thẻ Vietcombank Visa, JCB, Mastercard | |
| 2.5.1 | Thẻ chính | 200.000 VNĐ/thẻ/năm |
| 2.5.2 | Thẻ phụ | 100.000 VNĐ/thẻ/năm |
| 2.6 | Thẻ Vietcombank American Express | |

¹ VCB sẽ áp dụng thu phí thường niên thẻ Vietcombank Visa Signature theo quy định trong trường hợp KH xuống hạng Priority

NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM
Trụ sở chính: 198 Trần Quang Khải, Quận Hoàn Kiếm, Thành phố Hà Nội, Việt Nam | Tel: (+84)4 39343137 | Fax: (+84)4 38269067
www.vietcombank.com.vn | Trung tâm dịch vụ khách hàng 24/7, Tel: (+84)4 38243524/1900 545413
""",

    "pdf_06_bieu_phi_vay_khcn": """# BIỂU PHÍ CHO VAY KHÁCH HÀNG CÁ NHÂN VIETCOMBANK
**Nguồn**: `vcb_pdf_23_bieu_phi_cho_vay_khcn_2025.pdf` | **Trang**: 1/4

---

Vietcombank
BIỂU PHÍ DỊCH VỤ CHO VAY KHÁCH HÀNG CÁ NHÂN TẠI VIETCOMBANK
(Áp dụng từ năm 2025)

| STT | Loại phí | Mức phí quy định |
|---|---|---|
| 1 | Phí thẩm định hồ sơ vay vốn | Miễn phí |
| 2 | Phí cam kết rút vốn | 0,1% - 0,5% trên số tiền cam kết chưa rút |
| 3 | Phí trả nợ trước hạn gói vay lãi suất ưu đãi | Theo quy định cụ thể từng chương trình (từ 1% - 2% số tiền trả trước hạn) |
| 4 | Phí trả nợ trước hạn thông thường | 0,5% - 1,5% số tiền trả nợ trước hạn trong 3 năm đầu, miễn phí từ năm thứ 5 |
| 5 | Phí quản lý tài sản bảo đảm | Miễn phí quản lý định kỳ |
| 6 | Phí chuyển nhượng hợp đồng vay | 0,2% giá trị dư nợ chuyển nhượng (tối thiểu 1.000.000 VNĐ) |
""",

    "pdf_07_bieu_phi_forex_digi": """# BIỂU PHÍ CHUYỂN TIỀN NGOẠI TỆ QUA VCB DIGIBANK
**Nguồn**: `vcb_pdf_24_bieu_phi_chuyen_tien_ngoai_te_digibank.pdf` | **Trang**: 1/3

---

Vietcombank
BIỂU PHÍ CHUYỂN TIỀN NGOẠI TỆ ĐI NƯỚC NGOÀI QUA KÊNH VCB DIGIBANK
(Áp dụng cho Khách hàng cá nhân)

1. Phạm vi áp dụng: Chuyển tiền quốc tế trực tuyến phục vụ mục đích du học, chữa bệnh, trợ cấp thân nhân.
2. Biểu phí chi tiết:
- Phí chuyển tiền của Vietcombank: 0,15% số tiền chuyển (Tối thiểu 5 USD, Tối đa 150 USD hoặc ngoại tệ tương đương).
- Điện phí SWIFT: 5 USD / điện chuyển tiền.
- Phí ngân hàng trung gian / ngân hàng nhận (OUR fee): Thu theo thực tế biểu phí đại lý quốc tế (bình quân 20 - 30 USD/giao dịch).
- Tỷ giá quy đổi: Áp dụng tỷ giá bán chuyển khoản hiện hành của Vietcombank tại thời điểm thực hiện giao dịch.
""",

    "pdf_08_bieu_phi_so_dep": """# BIỂU PHÍ MỞ TÀI KHOẢN SỐ ĐẸP VIETCOMBANK
**Nguồn**: `vcb_pdf_25_bieu_phi_tai_khoan_so_dep.pdf` | **Trang**: 1/2

---

NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM
BIỂU PHÍ DỊCH VỤ CHỌN TÀI KHOẢN THANH TOÁN SỐ ĐẸP
(Áp dụng trên toàn hệ thống Vietcombank và ứng dụng VCB Digibank)

| Loại tài khoản số đẹp | Mô tả cấu trúc số | Mức phí niêm yết (chưa VAT) |
|---|---|---|
| Lộc phát / Thần tài | Đuôi số 68, 86, 79, 39 | 1.000.000 - 5.000.000 VNĐ |
| Tam hoa | 3 chữ số cuối giống nhau (ví dụ: ...666, ...888, ...999) | 2.000.000 - 10.000.000 VNĐ |
| Tứ quý | 4 chữ số cuối giống nhau (ví dụ: ...8888, ...9999) | 10.000.000 - 50.000.000 VNĐ |
| Ngũ quý / Lục quý | 5 hoặc 6 chữ số cuối giống nhau | 50.000.000 VNĐ trở lên (theo phê duyệt Hội sở) |
| Tiến đơn / Tiến đôi | Dãy số tiến đều (ví dụ: ...123456, ...6789) | 3.000.000 - 20.000.000 VNĐ |
| Số ngày sinh / Kỷ niệm | Trùng ngày tháng năm sinh khách hàng | Miễn phí hoặc 500.000 VNĐ |
""",

    "pdf_09_bieu_phi_debit_haha": """# BIỂU PHÍ THẺ GHI NỢ NỘI ĐỊA VÀ QUỐC TẾ (DEBIT)
**Nguồn**: `vcb_pdf_26_bieu_phi_the_thanh_toan_debit_haha.pdf` | **Trang**: 1/4

---

Vietcombank
BIỂU PHÍ DỊCH VỤ THẺ GHI NỢ (DEBIT CARD) VIETCOMBANK
(Bao gồm thẻ Connect24, Visa Debit, Mastercard Debit, JCB Debit)

1. Phí phát hành thẻ vật lý:
- Thẻ phát hành lần đầu: 50.000 VNĐ/thẻ (Miễn phí phát hành thẻ phi vật lý VCB DigiCard).
- Thẻ phát hành lại / thay thế: 50.000 VNĐ/thẻ.
- Phát hành nhanh (trong 24h-48h): 50.000 VNĐ/lần phụ thu.

2. Phí duy trì và quản lý tài khoản thẻ:
- Phí duy trì thẻ ghi nợ nội địa Connect24: Miễn phí nếu duy trì số dư bình quân tháng theo quy định.
- Phí thường niên thẻ ghi nợ quốc tế Visa/Mastercard Debit chuẩn: 60.000 VNĐ/năm.
- Thẻ ghi nợ quốc tế Platinum: Miễn phí thường niên năm đầu.
""",

    "pdf_10_dkdk_tktt_truc_tuyen": """# ĐIỀU KIỆN ĐIỀU KHOẢN MỞ TÀI KHOẢN THANH TOÁN TRỰC TUYẾN
**Nguồn**: `vcb_pdf_27_dieu_kien_dieu_khoan_tktt_truc_tuyen.pdf` | **Trang**: 1/5

---

CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

ĐIỀU KIỆN VÀ ĐIỀU KHOẢN MỞ VÀ SỬ DỤNG TÀI KHOẢN THANH TOÁN BẰNG PHƯƠNG THỨC ĐIỆN TỬ (eKYC) TẠI VIETCOMBANK

Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng
1. Văn bản này quy định về điều kiện, điều khoản mở và sử dụng tài khoản thanh toán cá nhân bằng phương thức điện tử (eKYC) tại Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank).
2. Đối tượng áp dụng là khách hàng cá nhân mang quốc tịch Việt Nam, đủ 18 tuổi trở lên, có đầy đủ năng lực hành vi dân sự, có Căn cước công dân gắn chip hợp lệ và thiết bị di động có tính năng đọc NFC.

Điều 2. Quy trình xác thực danh tính điện tử
1. Khách hàng thực hiện chụp ảnh mặt trước, mặt sau thẻ Căn cước công dân gắn chip.
2. Khách hàng thực hiện đọc thông tin chip sinh trắc học thông qua kết nối NFC trên thiết bị di động.
3. Khách hàng thực hiện xác thực khuôn mặt động (Liveness Detection) theo chuyển động ngẫu nhiên yêu cầu bởi hệ thống VCB Digibank.
"""
}


def generate_vision_and_word_groundtruth():
    print("=" * 70)
    print("  TẠO 20 FILE GROUND TRUTH VISION (10 PDF + 10 DOCX)")
    print("=" * 70)

    # 1. Ghi 10 PDF Vision Ground Truth
    for key, content in PDF_VISION_DATA.items():
        out_p = OUT_VISION_DIR / f"{key}_vision.md"
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  + [PDF Vision] Đã tạo {out_p.name} ({len(content)} ký tự)")

    # 2. Tạo 10 DOCX Vision Ground Truth từ chính file Word gốc kết hợp Vision layout
    docx_file_map = [
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

    for key, filename in docx_file_map:
        docx_path = DOCX_DIR / filename
        doc = docx.Document(str(docx_path))
        lines = []
        lines.append(f"# VISION GROUND TRUTH: `{filename}`")
        lines.append(f"**Nguồn**: `{filename}` | **Trang đầu kết xuất**\n")
        lines.append("---\n")

        # Thu thập các đoạn văn và danh sách của trang đầu (khoảng 25-30 paragraphs đầu)
        for p in doc.paragraphs[:30]:
            t = p.text.strip()
            if not t:
                continue
            style = p.style.name if p.style else ""
            if "Heading 1" in style or style == "Title":
                lines.append(f"# {t}\n")
            elif "Heading 2" in style:
                lines.append(f"## {t}\n")
            elif "Heading 3" in style:
                lines.append(f"### {t}\n")
            elif "List Bullet" in style:
                lines.append(f"- {t}")
            elif "List Number" in style:
                lines.append(f"1. {t}")
            else:
                lines.append(f"{t}\n")

        content_str = "\n".join(lines)
        out_p = OUT_VISION_DIR / f"{key}_vision.md"
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(content_str)
        print(f"  + [DOCX Vision] Đã tạo {out_p.name} ({len(content_str)} ký tự)")

    print(f"\n[Hoàn tất] 20 file Vision Ground Truth đã lưu tại: {OUT_VISION_DIR}\n")


if __name__ == "__main__":
    generate_vision_and_word_groundtruth()
