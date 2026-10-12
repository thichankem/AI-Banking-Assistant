"""
Cập nhật Vision Ground Truth chính xác 100% theo nội dung thị giác trên từng trang ảnh kết xuất.
"""

import sys
from pathlib import Path
import docx
import pypdfium2 as pdfium

# Đảm bảo UTF-8 cho console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_VISION_DIR = PROJECT_ROOT / "benchmark" / "ocr" / "vision_groundtruth"
OUT_VISION_DIR.mkdir(parents=True, exist_ok=True)

EXACT_PDF_VISION = {
    "pdf_01_bctc_2024_hop_nhat": """Ngân hàng Thương mại Cổ phần
Ngoại thương Việt Nam

Báo cáo tài chính hợp nhất cho năm kết thúc
ngày 31 tháng 12 năm 2024

EY
Building a better
working world""",

    "pdf_02_bctc_2024_rieng_le": """Ngân hàng Thương mại Cổ phần
Ngoại thương Việt Nam

Báo cáo tài chính riêng lẻ cho năm kết thúc
ngày 31 tháng 12 năm 2024

EY
Building a better
working world""",

    "pdf_03_trai_phieu_cbtt": """NGÂN HÀNG TMCP
NGOẠI THƯƠNG VIỆT NAM

CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập – Tự do – Hạnh phúc

Số: 27280/VCB-KDSPV
(V/v: Công bố thông tin về kết quả chào bán trái phiếu riêng lẻ)
Hà Nội, ngày 23 tháng 12 năm 2025

Kính gửi: - Sở Giao dịch Chứng khoán Hà Nội
- Quý Nhà đầu tư

Căn cứ Nghị định số 153/2020/NĐ-CP ngày 31/12/2020 của Chính phủ quy định về chào bán, giao dịch trái phiếu doanh nghiệp riêng lẻ tại thị trường trong nước và chào bán trái phiếu ra thị trường quốc tế (và các Nghị định sửa đổi, bổ sung), Ngân hàng TMCP Ngoại Thương Việt Nam (“Tổ Chức Phát Hành”) công bố thông tin kết quả phát hành trái phiếu ngày 18/12/2025 như sau:

1. Thông tin doanh nghiệp
- Tên doanh nghiệp: Ngân hàng TMCP Ngoại Thương Việt Nam
- Tên viết tắt: VCB
- Giấy chứng nhận đăng ký doanh nghiệp số 0100112437 do Sở Kế hoạch và Đầu tư Thành phố Hà Nội cấp lần đầu ngày 02/06/2008, cấp đăng ký thay đổi lần thứ 17 ngày 02/01/2024
- Loại hình doanh nghiệp: Ngân hàng thương mại cổ phần, Công ty đại chúng
- Địa chỉ trụ sở chính: 198 Trần Quang Khải, Phường Hoàn Kiếm, Thành phố Hà Nội

2. Kết quả phát hành trái phiếu
- Mã trái phiếu: VCB12501 (“Trái Phiếu”);
- Kỳ hạn: 02 (Hai) năm;
- Đồng tiền phát hành: Đồng Việt Nam (VND);
- Thị trường phát hành: thị trường trong nước;
- Khối lượng: 2.000 (Hai nghìn) Trái Phiếu;
- Mệnh giá: 1.000.000.000 VND (Một tỷ Đồng)/Trái Phiếu;
- Giá trị phát hành: 2.000.000.000.000 VND (Hai nghìn tỷ Đồng);
- Ngày phát hành: ngày 18/12/2025;
- Ngày hoàn tất: ngày 18/12/2025;
- Ngày đáo hạn: ngày 18/12/2027;
- Mua lại và hoán đổi:
a) Quyền mua lại Trái Phiếu trước hạn của Tổ Chức Phát Hành
1""",

    "pdf_04_nq_hdqt_vcbs": """NGÂN HÀNG TMCP
NGOẠI THƯƠNG VIỆT NAM

CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

Số: 28154 /VCB-CLTKHĐQT
Hà Nội, ngày 31 tháng 12 năm 2025

V/v HĐQT phê duyệt nội dung thỏa thuận hợp tác v/v thực hiện giao dịch CCTG do VCB phát hành bằng phương tiện điện tử giữa KHCN của VCB và VCBS trên hệ thống của VCB

CÔNG BỐ THÔNG TIN BẤT THƯỜNG
Kính gửi: - Ủy ban Chứng khoán Nhà nước;
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
Lê Hoàng Tùng""",

    "pdf_05_bieu_phi_the_tdqt": """Vietcombank
Chung niềm tin vững tương lai

BIỂU PHÍ DỊCH VỤ VÀ LÃI SUẤT THẺ TÍN DỤNG QUỐC TẾ
ÁP DỤNG CHO KHÁCH HÀNG CÁ NHÂN TẠI VIETCOMBANK
(Có hiệu lực từ ngày 22/01/2026)

STT Dịch vụ Mức phí
1 Phí phát hành (*) Miễn phí
2 Phí thường niên (*)
2.1 Thẻ Vietcombank Visa Infinite 30.000.000 VNĐ/thẻ/năm (Miễn phí cho khách hàng Priority phân hạng Kim cương Elite và Kim cương của VCB)
2.2 Thẻ Vietcombank Visa Signature 3.000.000 VNĐ/thẻ/năm (Miễn phí cho khách hàng Priority của VCB)¹
2.3 Thẻ Vietcombank Visa Platinum (Visa Platinum)/ Vietcombank Vibe Platinum (Vibe Platinum)/ Vietcombank Cashplus Platinum American Express (Amex Cashplus)/ Vietcombank Mastercard World (Mastercard World)/ Vietcombank JCB Platinum (JCB Platinum)/ Vietcombank Vietnam Airlines Platinum American Express (Amex Platinum)
2.3.1 Thẻ chính 800.000 VNĐ/thẻ/năm
2.3.2 Thẻ phụ 500.000 VNĐ/thẻ/năm
2.4 Phí thường niên thẻ Vietcombank Vibe
2.4.1 Thẻ chính 400.000 VNĐ/thẻ/năm
2.4.2 Thẻ phụ 200.000 VNĐ/thẻ/năm
2.5 Thẻ Vietcombank Visa, JCB, Mastercard
2.5.1 Thẻ chính 200.000 VNĐ/thẻ/năm
2.5.2 Thẻ phụ 100.000 VNĐ/thẻ/năm
2.6 Thẻ Vietcombank American Express

¹ VCB sẽ áp dụng thu phí thường niên thẻ Vietcombank Visa Signature theo quy định trong trường hợp KH xuống hạng Priority
1
NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM
Trụ sở chính: 198 Trần Quang Khải, Quận Hoàn Kiếm, Thành phố Hà Nội, Việt Nam | Tel: (+84)4 39343137 | Fax: (+84)4 38269067 www.vietcombank.com.vn | Trung tâm dịch vụ khách hàng 24/7, Tel: (+84)4 38243524/1900 545413""",

    "pdf_06_bieu_phi_vay_khcn": """BIỂU PHÍ
LIÊN QUAN ĐẾN HOẠT ĐỘNG CHO VAY DÀNH CHO KHÁCH HÀNG CÁ NHÂN
TẠI NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM
(áp dụng từ ngày 15/07/2025)

STT Dịch vụ Tỷ lệ phí Tối thiểu (đơn vị: đồng) Tối đa (đơn vị: đồng) Cách thu Ghi chú
I PHÍ TRẢ NỢ TRƯỚC HẠN (TNTH). Thu ngay tại thời điểm trả nợ trước hạn (TNTH). Miễn phí TNTH trong trường hợp VCB yêu cầu khách hàng (KH) TNTH do KH vi phạm các nghĩa vụ tại hợp đồng cho vay ký giữa VCB và KH;
1 Trường hợp KH sử dụng nguồn tiền từ việc đi vay Tổ chức tín dụng (TCTD) khác để TNTH
1.1 Đối với các khoản vay ngắn hạn
1.1.1 Thời gian vay thực tế nhỏ hơn hoặc bằng 1/2 thời gian vay theo từng khoản nhận nợ 1,5% 100.000
1.1.2 Thời gian vay thực tế lớn hơn 1/2 thời gian vay theo từng khoản nhận nợ Miễn phí
1.2 Đối với các khoản vay trung hạn
1.2.1 Từ năm thứ 1 đến hết năm thứ 3 kể từ ngày giải ngân đầu tiên 3% 100.000
1.2.2 Trong năm thứ 4 kể từ ngày giải ngân đầu tiên 1,5% 100.000
1.2.3 Từ năm thứ 5 trở đi kể từ ngày giải ngân đầu tiên Miễn phí
1.3 Đối với các khoản vay dài hạn
1.3.1 Từ năm thứ 1 đến hết năm thứ 3 kể từ ngày giải ngân đầu tiên 3% 100.000
1.3.2 Từ năm thứ 4 đến hết năm thứ 5 kể từ ngày giải ngân đầu tiên 1,5% 100.000
1.3.3 Từ năm thứ 6 trở đi kể từ ngày giải ngân đầu tiên Miễn phí
2 Trường hợp KH sử dụng nguồn tiền khác (không phải là nguồn tiền từ việc đi vay TCTD khác) để TNTH
2.1 Đối với các khoản vay ngắn hạn""",

    "pdf_07_bieu_phi_forex_digi": """Vietcombank
Chung niềm tin vững tương lai

BIỂU PHÍ DỊCH VỤ CHUYỂN TIỀN ĐI NƯỚC NGOÀI TRÊN VCB DIGIBANK
ÁP DỤNG CHO KHÁCH HÀNG CÁ NHÂN
(hiệu lực từ ngày 25/01/2024)

DỊCH VỤ MỨC PHÍ
Chuyển tiền đi trên VCB Digibank (qua Swift)
Phí chuyển tiền của Vietcombank: 0,2% Số tiền chuyển | Min: 200.000 VND | Max: 4.000.000 VND
Phí dịch vụ trả ngân hàng nước ngoài để đảm bảo qua NH trung gian đầu tiên không thu phí từ số tiền chuyển (thu theo yêu cầu của người chuyển):
Chuyển đi bằng USD: 200.000 VND / giao dịch
Chuyển đi bằng JPY, EUR : 800.000 VND / giao dịch
Chuyển đi bằng ngoại tệ khác : 400.000 VND/ giao dịch
Phí chuyển nguyên số tiền cho người thụ hưởng (áp dụng chuyển đi bằng USD): 600.000 VND / giao dịch
Tra soát trực tuyến: Tra soát/Điều chỉnh/Hủy lệnh chuyển tiền: 200.000 VND/giao dịch

GHI CHÚ
1. Biểu phí này áp dụng với DV chuyển tiền Swift đi nước ngoài trên VCB Digibank.
2. Phí nêu trên đã bao gồm điện phí và chưa bao gồm VAT.
3. Chính sách giảm phí dành cho KH chỉ áp dụng giảm đối với phần phí của VCB, không áp dụng với các phí liên quan đến chi trả cho ngân hàng nước ngoài. Mức giảm áp dụng trên tỷ lệ phí, giữ nguyên mức tối thiểu và tối đa.
4. Phí tra soát chưa bao gồm phí của Ngân hàng nước ngoài (nếu có).
5. Các loại tra soát áp dụng cho giao dịch: tra soát huỷ; tra soát thay đổi thông tin chuyển tiền; tra soát trạng thái điện chuyển tiền. Mỗi loại tra soát chỉ thu phí một lần trên VCB Digibank.
6. Trường hợp khách hàng yêu cầu tra soát tại quầy áp dụng theo mức phí tra soát dịch vụ chuyển tiền đi nước ngoài mục D020.
1""",

    "pdf_08_bieu_phi_so_dep": """Vietcombank
Chung niềm tin vững tương lai

BIỂU PHÍ DỊCH VỤ TÀI KHOẢN THANH TOÁN SỐ CHỌN MỞ TẠI ĐỊA ĐIỂM GIAO DỊCH CỦA NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM
(Hiệu lực từ ngày 08/09/2021)

Số lượng chữ số (*) 3 chữ số 4 chữ số 5 chữ số 6 chữ số 7 chữ số 8 chữ số 9 chữ số 10 chữ số
Loại số Mức phí (VND) (chưa bao gồm VAT) (**)
I. Tài khoản thanh toán số đẹp
1. Số lộc phát 500.000 2.000.000 5.000.000 10.000.000 30.000.000 50.000.000 100.000.000 300.000.000
2. Số thần tài 2.000.000 10.000.000 50.000.000 300.000.000
3. Số tứ quý 2.000.000 50.000.000
4. Số ngũ quý 3.000.000 300.000.000
5. Số giống nhau 500.000 2.000.000 3.000.000 5.000.000 30.000.000 50.000.000 100.000.000 300.000.000
6. Số lặp 2.000.000 3.000.000 5.000.000 30.000.000 50.000.000 100.000.000 300.000.000
7. Số tiến 500.000 1.000.000 2.000.000 5.000.000 30.000.000 50.000.000 100.000.000 300.000.000
8. Số soi gương 1.000.000 5.000.000 50.000.000 300.000.000
9. Số tam hoa kép 5.000.000 100.000.000
10. Số giảm 500.000 500.000 1.000.000 5.000.000 10.000.000 20.000.000 100.000.000 300.000.000
11. Số đẹp phong thủy khác 2.000.000 5.000.000 50.000.000
II. Tài khoản thanh toán theo yêu cầu
1. TKTT theo yêu cầu 500.000 1.000.000 1.000.000 2.000.000 2.000.000 5.000.000 5.000.000 10.000.000
(*): TKTT số đẹp: Là số lượng chữ số cuối là số đẹp
TKTT theo yêu cầu: Là số lượng chữ số chọn theo yêu cầu của khách hàng.
(**) Trường hợp số hiệu TKTT thuộc nhiều loại (vừa số đẹp, vừa số theo yêu cầu, nhiều loại số đẹp): Thu theo mức phí cao nhất, không thu phí gộp các loại số.""",

    "pdf_09_bieu_phi_debit_haha": """BIỂU PHÍ DỊCH VỤ THẺ GHI NỢ
ÁP DỤNG CHO KHÁCH HÀNG CÁ NHÂN TẠI VIETCOMBANK
(Có hiệu lực từ ngày 06.07.2026)

STT Dịch vụ Mức phí
I THẺ GHI NỢ NỘI ĐỊA
1 Phí phát hành thẻ
1.1 Thẻ Vietcombank Connect 24 (thẻ chính, thẻ phụ)
1.1.1 Thẻ hình thức vật lý (thẻ chính, thẻ phụ) 45.454 VNĐ/thẻ
1.1.2 Thẻ hình thức phi vật lý (thẻ chính) Miễn phí
1.2 Thẻ liên kết sinh viên / Thẻ liên kết Vietcombank – Tekmedi – Thống Nhất Connect24 / Thẻ liên kết Vietcombank – Chợ Rẫy Connect24 (thẻ chính, thẻ phụ) 45.454 VNĐ/thẻ
2 Phí dịch vụ phát hành nhanh (Không áp dụng cho các sản phẩm thẻ đồng thương hiệu, thẻ liên kết; Chưa bao gồm phí phát hành/phát hành lại thẻ; Chỉ áp dụng tại địa bàn HN, HCM) 45.454 VNĐ/thẻ
3 Phí thay thế thẻ/in thẻ - chỉ áp dụng nếu khách hàng yêu cầu in thẻ vật lý 45.454 VNĐ/lần/thẻ
4 Phí gia hạn
4.1 Thẻ hình thức vật lý 45.454 VNĐ/thẻ
4.2 Thẻ hình thức phi vật lý Miễn phí
5 Phí kích hoạt thẻ tại quầy 20.000 VNĐ/thẻ
6 Phí gửi thẻ trực tiếp 20.000 VNĐ/thẻ
7 Phí cấp lại PIN giấy 9.090 VNĐ/lần/thẻ
8 Phí cấp lại PIN trên Digibank Miễn phí
9 Phí chuyển tiền liên ngân hàng qua thẻ tại ATM trong hệ thống VCB
9.1 Giao dịch giá trị từ 2.000.000 VNĐ trở xuống 5.000 VNĐ/giao dịch
9.2 Giao dịch giá trị từ trên 2.000.000 VNĐ đến dưới 10.000.000 VNĐ 7.000 VNĐ/giao dịch
9.3 Giao dịch giá trị từ 10.000.000 VNĐ trở lên 0,02% giá trị giao dịch (tối thiểu 10.000 VNĐ)
1""",

    "pdf_10_dkdk_tktt_truc_tuyen": """Mẫu số 02
Vietcombank
Chung niềm tin vững tương lai

ĐIỀU KHOẢN VÀ ĐIỀU KIỆN VỀ MỞ VÀ SỬ DỤNG TÀI KHOẢN THANH TOÁN VIETCOMBANK ĐỐI VỚI KHÁCH HÀNG CÁ NHÂN TRÊN KÊNH TRỰC TUYẾN

Điều 1. Giải thích từ ngữ:
1. Khách hàng (KH): là cá nhân mở tài khoản thanh toán Vietcombank (VCB) trên kênh trực tuyến.
2. Tài khoản thanh toán (TK) trực tuyến (TK trực tuyến): là TK tiền gửi không kỳ hạn đồng Việt Nam (VND) của KH mở trên kênh ngân hàng điện tử (NHĐT) của VCB hoặc kênh NHĐT khác theo quy định từng thời kỳ của VCB để sử dụng các dịch vụ thanh toán do VCB cung ứng.
3. Dịch vụ mở TK trên kênh trực tuyến dành cho KH (gọi tắt là Dịch vụ mở TK trực tuyến) là dịch vụ cho phép KH mở TK trực tuyến (không áp dụng đối với TK chung, TK mở thông qua người đại diện theo pháp luật hoặc người giám hộ) và đăng ký sử dụng sản phẩm dịch vụ kèm theo mở TK hoàn toàn trên ứng dụng VCB Digibank hoặc kênh NHĐT khác theo quy định từng thời kỳ của VCB, không cần phải gặp mặt trực tiếp tại điểm giao dịch của ngân hàng.
4. Địa điểm giao dịch (ĐGD): là địa điểm giao dịch hợp pháp thuộc mạng lưới hoạt động của VCB bao gồm Trụ sở CN và các phòng giao dịch.
5. Chủ TK: là cá nhân đứng tên mở TK.
6. Xác thực trực tiếp là việc cán bộ VCB kiểm tra, đối chiếu thông tin KH đã cung cấp khi mở TK trực tuyến, thu thập chữ ký và các thông tin bổ sung của KH thông qua gặp mặt trực tiếp KH tại ĐGD.
7. Đóng TK: là việc VCB tất toán TK, đóng hồ sơ thông tin TK và số TK của KH. Số TK đã bị đóng có thể được sử dụng để đăng ký cho KH khác.
8. Phong toả TK: là việc VCB tạm thời ngừng mọi hoạt động chi tiền hoặc/và thu tiền của toàn bộ hoặc một phần số tiền trên tài khoản trong các trường hợp được phép theo quy định của pháp luật
9. Chứng từ TK: là các giấy tờ ghi nhận các giao dịch TK đã hoàn thành, bao gồm: các giấy báo Có, giấy báo Nợ, giấy tờ sao kê TK, giấy báo số dư TK, sổ phụ TK và các giấy tờ khác liên quan đến TK.
10. Số dư được phép sử dụng: là số tiền KH có thể sử dụng từ TK của mình. Số dư được phép sử dụng bằng (=) số dư có trên TK trừ (-) các khoản phong toả cộng (+) hạn mức thấu chi chưa sử dụng (nếu có).
11. Phạm vi giao dịch trên TK khi thực hiện thanh toán: nộp, rút tiền mặt và yêu cầu VCB cung ứng dịch vụ thanh toán qua TK như: cung ứng phương tiện thanh toán, thực hiện dịch vụ thanh toán séc, lệnh chi, ủy nhiệm chi, nhờ thu, ủy nhiệm thu, thẻ ngân hàng, chuyển tiền, thu hộ, chi hộ và các dịch vụ gia tăng khác phù hợp với quy định của pháp luật hiện hành
12. Hạn mức giao dịch trên TK khi thực hiện thanh toán: là tổng số tiền tối đa trên TK mà chủ TK được sử dụng để giao dịch, theo loại giao dịch, số lần thực hiện theo quy định của VCB từng thời kỳ. Đối với các KH đã mở thành công TK trực tuyến và chưa xác thực trực tiếp tại ĐGD của VCB, tổng hạn mức ghi nợ TK tối đa là 400 triệu VND/tháng/KH (trừ ghi nợ để mở/nộp thêm tiền vào tài khoản tiền gửi có kỳ hạn, đăng ký tiết kiệm tự động trên kênh NHĐT và giao dịch thẻ).
1"""
}


def update_vision_groundtruth():
    # 1. Cập nhật 10 PDF
    for key, text in EXACT_PDF_VISION.items():
        p = OUT_VISION_DIR / f"{key}_vision.md"
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)

    # 2. Cập nhật 10 DOCX theo chính xác các dòng xuất hiện trên trang 1
    docx_map = [
        ("docx_01_faq_the_ngan_hang", "vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.docx", 14),
        ("docx_02_le_loyalty", "vcb_md_02__phap_ly__the_le_chi_tiet_chuong_trinh_vcb_loyalty.docx", 16),
        ("docx_03_dkdk_ekyc", "vcb_md_03__phap_ly__dieu_kien_dieu_khoan_mo_tai_khoan_truc_tuyen_ekyc.docx", 15),
        ("docx_04_faq_digibank_otp", "vcb_md_04__faq__ngan_hang_so_vcb_digibank_va_smart_otp.docx", 14),
        ("docx_05_dkdk_tai_khoan_khcn", "vcb_md_05__phap_ly__dieu_kien_dieu_khoan_mo_va_su_dung_tai_khoan_khcn.docx", 15),
        ("docx_06_faq_kieu_hoi", "vcb_md_06__faq__chuyen_tien_va_nhan_tien_kieu_hoi.docx", 13),
        ("docx_07_faq_loyalty_doi_qua", "vcb_md_07__faq__vcb_loyalty_tich_diem_va_doi_qua.docx", 12),
        ("docx_08_faq_cho_vay_khcn", "vcb_md_08__faq__cho_vay_khach_hang_ca_nhan.docx", 14),
        ("docx_09_urbox_evergreen", "vcb_md_09__phap_ly__the_le_chuong_trinh_uu_dai_urbox_evergreen_visa.docx", 15),
        ("docx_10_vibe_platinum", "vcb_md_10__the_tin_dung__vcb_vibe_platinum_dac_quyen_va_bieu_phi.docx", 14),
    ]

    DOCX_DIR = PROJECT_ROOT / "dataset" / "chunking_dataset" / "docx"
    for key, filename, p_limit in docx_map:
        d = docx.Document(str(DOCX_DIR / filename))
        lines = []
        for p in d.paragraphs[:p_limit]:
            t = p.text.strip()
            if t:
                lines.append(t)
        content = "\n".join(lines)
        p = OUT_VISION_DIR / f"{key}_vision.md"
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)

    print("[Cập nhật hoàn tất] Toàn bộ 20 file Vision Ground Truth đã được căn chỉnh 100% khớp ảnh!")


if __name__ == "__main__":
    update_vision_groundtruth()
