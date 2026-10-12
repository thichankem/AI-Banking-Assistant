"""
Bộ Dữ liệu Đánh giá và Tinh chỉnh Chuẩn (Ground-Truth Evaluation Dataset)
Bao gồm 20 câu hỏi nghiệp vụ ngân hàng thực tế đại diện cho tất cả các phân hệ:
1. Bảng lãi suất & Tỷ giá ngoại tệ (Excel .xlsx đa sheet & Markdown table)
2. Biểu phí & Lãi suất thẻ tín dụng quốc tế (Digital PDF & Markdown)
3. Văn bản pháp lý & Nghị quyết HĐQT (Scanned OCR PDF)
4. Ngân hàng số, eKYC, xác thực NFC CCCD gắn chip (Markdown & PDF)
5. Cho vay khách hàng cá nhân & Biểu phí phạt trả trước hạn (Excel & PDF)
6. Quy trình bảo mật, chống lừa đảo, khóa thẻ khẩn cấp (Markdown FAQ)
7. Chương trình khách hàng thân thiết VCB Loyalty & Bảo hiểm FWD
"""

BANKING_EVALUATION_DATASET = [
    {
        "id": "EVAL_01",
        "query": "Lãi suất tiền gửi tiết kiệm VND tại quầy của Vietcombank kỳ hạn 12 tháng là bao nhiêu %/năm?",
        "target_source_contains": "vcb_bieu_lai_suat_va_ty_gia_ngoai_te",
        "target_keywords": ["12 tháng", "5,9", "Quầy"],
        "domain": "Lãi suất & Tiền gửi (Excel)"
    },
    {
        "id": "EVAL_02",
        "query": "Lãi suất tiền gửi trực tuyến trên VCB Digibank kỳ hạn 24 tháng là bao nhiêu %/năm?",
        "target_source_contains": "vcb_bieu_lai_suat_va_ty_gia_ngoai_te",
        "target_keywords": ["24 tháng", "6,1", "Digibank"],
        "domain": "Lãi suất & Tiền gửi (Excel)"
    },
    {
        "id": "EVAL_03",
        "query": "Phí thường niên thẻ chính Vietcombank Vibe theo biểu phí dịch vụ thẻ tín dụng quốc tế 2026 là bao nhiêu?",
        "target_source_contains": "vcb_pdf_22_bieu_phi_the_tdqt_2026",
        "target_keywords": ["Vietcombank Vibe", "400.000", "Thẻ chính"],
        "domain": "Biểu phí thẻ (Digital PDF)"
    },
    {
        "id": "EVAL_04",
        "query": "Phí phát hành thẻ tín dụng quốc tế tại Vietcombank là bao nhiêu tiền?",
        "target_source_contains": "vcb_pdf_22_bieu_phi_the_tdqt_2026",
        "target_keywords": ["Phí phát hành", "Miễn phí"],
        "domain": "Biểu phí thẻ (Digital PDF)"
    },
    {
        "id": "EVAL_05",
        "query": "Nghị quyết HĐQT số 28154/VCB-CLTKHDQT ngày 31/12/2025 phê duyệt nội dung gì về chứng chỉ tiền gửi CCTG?",
        "target_source_contains": "vcb_pdf_21_nq_hdqt_cctg_vcbs",
        "target_keywords": ["28154", "CCTG", "VCBS"],
        "domain": "Nghị quyết HĐQT (Scanned OCR PDF)"
    },
    {
        "id": "EVAL_06",
        "query": "Khi thẻ ngân hàng bị thất lạc hoặc mất cắp, khách hàng cần thực hiện thao tác gì đầu tiên?",
        "target_source_contains": "vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang",
        "target_keywords": ["Khóa thẻ", "Digibank", "6167"],
        "domain": "Bảo mật & Thẻ (Markdown)"
    },
    {
        "id": "EVAL_07",
        "query": "Khách hàng không sử dụng thẻ nhưng nhận được tin nhắn báo phát sinh giao dịch thì phải xử lý ra sao?",
        "target_source_contains": "vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang",
        "target_keywords": ["khóa thẻ", "khiếu nại", "tra soát"],
        "domain": "Bảo mật & Thẻ (Markdown)"
    },
    {
        "id": "EVAL_08",
        "query": "Đặc quyền bảo hiểm du lịch toàn cầu của thẻ tín dụng Vietcombank Vibe Platinum có mức bồi thường tối đa bao nhiêu?",
        "target_source_contains": "vcb_md_10__the_tin_dung__vcb_vibe_platinum_dac_quyen_va_bieu_phi",
        "target_keywords": ["bảo hiểm", "du lịch", "tối đa", "Platinum"],
        "domain": "Sản phẩm Thẻ cao cấp (Markdown)"
    },
    {
        "id": "EVAL_09",
        "query": "Quy trình chụp ảnh CCCD gắn chip và quét chip NFC khi mở tài khoản eKYC trên VCB Digibank gồm những bước nào?",
        "target_source_contains": "vcb_md_12__quy_trinh__mo_tai_khoan_thanh_toan_ekyc_va_xac_thuc_nfc",
        "target_keywords": ["CCCD", "NFC", "khuôn mặt"],
        "domain": "eKYC & Ngân hàng số (Markdown)"
    },
    {
        "id": "EVAL_10",
        "query": "Hạn mức chuyển tiền tối đa qua VCB Digibank khi sử dụng phương thức xác thực Smart OTP là bao nhiêu một ngày?",
        "target_source_contains": "vcb_md_04__faq__ngan_hang_so_vcb_digibank_va_smart_otp",
        "target_keywords": ["Smart OTP", "hạn mức", "ngày"],
        "domain": "Ngân hàng số (Markdown)"
    },
    {
        "id": "EVAL_11",
        "query": "Quy định điều kiện miễn phí duy trì dịch vụ VCB Digibank hiện nay như thế nào?",
        "target_source_contains": "vcb_md_04__faq__ngan_hang_so_vcb_digibank_va_smart_otp",
        "target_keywords": ["duy trì", "miễn phí", "Digibank"],
        "domain": "Ngân hàng số (Markdown)"
    },
    {
        "id": "EVAL_12",
        "query": "Biểu phí trả nợ trước hạn đối với khoản vay cá nhân tại Vietcombank trong năm đầu tiên là bao nhiêu %?",
        "target_source_contains": "vcb_pdf_23_bieu_phi_cho_vay_khcn_2025",
        "target_keywords": ["trả nợ trước hạn", "năm đầu", "phí"],
        "domain": "Tín dụng cá nhân (Digital PDF)"
    },
    {
        "id": "EVAL_13",
        "query": "Biểu phí dịch vụ tài khoản thanh toán số đẹp chọn theo đuôi tứ quý hoặc thần tài tại Vietcombank?",
        "target_source_contains": "vcb_pdf_25_bieu_phi_tai_khoan_so_dep",
        "target_keywords": ["số đẹp", "tứ quý", "thần tài"],
        "domain": "Tài khoản số đẹp (Digital PDF)"
    },
    {
        "id": "EVAL_14",
        "query": "Hạn mức giao dịch và thời gian miễn lãi tối đa của thẻ tín dụng Vietcombank là bao nhiêu ngày?",
        "target_source_contains": "vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang",
        "target_keywords": ["miễn lãi", "45 ngày", "thẻ tín dụng"],
        "domain": "Thẻ tín dụng (Markdown)"
    },
    {
        "id": "EVAL_15",
        "query": "Thời gian giải quyết tra soát khiếu nại đối với giao dịch thẻ quốc tế Vietcombank theo quy định là bao nhiêu ngày?",
        "target_source_contains": "vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang",
        "target_keywords": ["tra soát", "khiếu nại", "ngày"],
        "domain": "Quy trình Thẻ (Markdown)"
    },
    {
        "id": "EVAL_16",
        "query": "Điều kiện điều khoản mở và sử dụng tài khoản thanh toán trực tuyến qua eKYC yêu cầu độ tuổi tối thiểu là bao nhiêu?",
        "target_source_contains": "vcb_md_03__phap_ly__dieu_kien_dieu_khoan_mo_tai_khoan_truc_tuyen_ekyc",
        "target_keywords": ["tuổi", "eKYC", "tài khoản"],
        "domain": "Pháp lý mở tài khoản (Markdown)"
    },
    {
        "id": "EVAL_17",
        "query": "Cách thức tích lũy điểm thưởng VCB Loyalty và giá trị quy đổi điểm thưởng ra quà tặng hoặc tiền như thế nào?",
        "target_source_contains": "vcb_md_07__faq__vcb_loyalty_tich_diem_va_doi_qua",
        "target_keywords": ["Loyalty", "điểm", "đổi quà"],
        "domain": "Chăm sóc khách hàng (Markdown)"
    },
    {
        "id": "EVAL_18",
        "query": "Thủ tục và giấy tờ cần thiết khi chuyển tiền ra nước ngoài qua kênh SWIFT hoặc MoneyGram tại Vietcombank?",
        "target_source_contains": "vcb_md_06__faq__chuyen_tien_va_nhan_tien_kieu_hoi",
        "target_keywords": ["chuyển tiền", "nước ngoài", "giấy tờ"],
        "domain": "Chuyển tiền kiều hối (Markdown)"
    },
    {
        "id": "EVAL_19",
        "query": "Nguyên tắc an toàn khi sử dụng ngân hàng số và các hình thức lừa đảo giả mạo tin nhắn Brandname ngân hàng?",
        "target_source_contains": "bao_mat__nguyen_tac_an_toan_va_canh_bao_5_hinh_thuc_lua_dao",
        "target_keywords": ["lừa đảo", "Brandname", "mật khẩu"],
        "domain": "An toàn bảo mật (Markdown)"
    },
    {
        "id": "EVAL_20",
        "query": "Quy định về thời hạn tái tục gốc và lãi tiền gửi tiết kiệm tự động trên VCB Digibank khi đến ngày đáo hạn?",
        "target_source_contains": "vcb_pdf_39_thoa_thuan_tien_gui_online_digibank",
        "target_keywords": ["đáo hạn", "tự động", "tái tục"],
        "domain": "Tiền gửi tiết kiệm (Digital PDF)"
    }
]
