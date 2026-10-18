"""
Unit Tests cho Bộ Phục hồi Cấu trúc Tài liệu Scan (ScanStructureReconstructor)
Kiểm thử toàn diện 7 tiêu chí:
1. Regex chịu lỗi OCR (Đicu, Dieu, ĐIÈU, O/o/l/I -> 0/1) kết hợp tín hiệu bố cục.
2. Kiểm tra tính liên tục của số Điều (loại trừ tham chiếu chéo như 'theo quy định tại Điều 12').
3. Ghép dòng tên Chương khi mã chương nằm riêng 1 dòng ('CHƯƠNG I' + 'QUY ĐỊNH CHUNG').
4. Lọc Header/Footer (vùng 8% trên/dưới và lặp lại >= 50% số trang).
5. State Machine BreadcrumbTracker (duy trì stack, dừng ở cấp Điều, Khoản/Điểm vào metadata).
6. Kiểm chứng Mục lục TOC (đối chiếu tên điều, cộng điểm tin cậy).
7. Reset stack khi phát hiện văn bản mới ghép trong PDF (Quyết định số, Thông tư số...).
8. Thang dự phòng 3 mức (regex+layout -> layout_visual -> page_fallback).
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import unittest
from rag_banking.chunking.pdf_chunker.scan_structure_reconstructor import (
    BreadcrumbTracker,
    TOCVerifier,
    ScanStructureReconstructor,
    fix_num
)


class TestScanStructureReconstructor(unittest.TestCase):

    def test_01_ocr_error_correction_and_fix_num(self):
        """Kiểm tra sửa lỗi OCR trong số điều (O, o, l, I)."""
        self.assertEqual(fix_num("O1"), 1)
        self.assertEqual(fix_num("l2"), 12)
        self.assertEqual(fix_num("I5"), 15)
        self.assertEqual(fix_num("2O"), 20)
        self.assertEqual(fix_num("Điều l0."), 10)

    def test_02_breadcrumb_tracker_state_machine(self):
        """Kiểm tra state machine duy trì stack, cấp N xóa từ N trở xuống, dừng ở cấp Điều."""
        tracker = BreadcrumbTracker(doc_title="Quy chế eKYC VCB")

        # 1. Ban đầu fallback Trang 1
        path, src, conf = tracker.path(page_num=1)
        self.assertEqual(path, "Quy chế eKYC VCB > Trang 1")
        self.assertEqual(src, "page_fallback")

        # 2. Gặp Chương I
        tracker.on_heading("chuong", "Chương I. Quy định chung", number=1, score=0.95, source="regex+layout")
        path, src, conf = tracker.path(page_num=1)
        self.assertEqual(path, "Quy chế eKYC VCB > Chương I. Quy định chung")
        self.assertEqual(src, "regex+layout")

        # 3. Gặp Điều 1
        tracker.on_heading("dieu", "Điều 1. Phạm vi điều chỉnh", number=1, score=0.92)
        path, src, conf = tracker.path(page_num=1)
        self.assertEqual(path, "Quy chế eKYC VCB > Chương I. Quy định chung > Điều 1. Phạm vi điều chỉnh")

        # 4. Gặp Điều 2 (cùng cấp -> thay thế Điều 1, giữ nguyên Chương I)
        tracker.on_heading("dieu", "Điều 2. Đối tượng áp dụng", number=2, score=0.94)
        path, src, conf = tracker.path(page_num=2)
        self.assertEqual(path, "Quy chế eKYC VCB > Chương I. Quy định chung > Điều 2. Đối tượng áp dụng")

        # 5. Gặp Chương II (cấp cha -> xóa Điều 2)
        tracker.on_heading("chuong", "Chương II. Quy trình mở tài khoản", number=2, score=0.96)
        path, src, conf = tracker.path(page_num=3)
        self.assertEqual(path, "Quy chế eKYC VCB > Chương II. Quy trình mở tài khoản")

        # 6. Gặp Điều 3 ở Chương II
        tracker.on_heading("dieu", "Điều 3. Xác minh eKYC", number=3, score=0.90)
        path, src, conf = tracker.path(page_num=3)
        self.assertEqual(path, "Quy chế eKYC VCB > Chương II. Quy trình mở tài khoản > Điều 3. Xác minh eKYC")

    def test_03_toc_verifier(self):
        """Kiểm tra đối chiếu và kiểm chứng Mục lục TOC."""
        toc = TOCVerifier()
        mock_toc_lines = [
            {"text": "MỤC LỤC QUY CHẾ"},
            {"text": "Điều 1. Phạm vi điều chỉnh .................... 3"},
            {"text": "Điều 2. Đối tượng áp dụng ..................... 5"},
            {"text": "Điều 3. Giải thích từ ngữ ..................... 7"},
        ]
        toc.parse_toc_page(mock_toc_lines)
        self.assertTrue(toc.toc_found)
        self.assertEqual(len(toc.expected_dieus), 3)
        self.assertEqual(toc.expected_dieus[1], "Phạm vi điều chỉnh")

        # Kiểm tra đối chiếu Điều 1 khi OCR bị lỗi nhẹ
        matched, corrected, bonus = toc.verify_dieu(1, "Pham vi dieu chinh")
        self.assertTrue(matched)
        self.assertEqual(corrected, "Phạm vi điều chỉnh")
        self.assertGreater(bonus, 0.0)

    def test_04_heading_detection_with_layout_signals(self):
        """Kiểm tra nhận diện heading 100% bằng đặc trưng hình học OCR (0% Regex) & loại trừ tham chiếu chéo."""
        reconstructor = ScanStructureReconstructor(doc_title="Quy chế eKYC VCB")
        median_h = 16.0
        page_w = 800.0

        # Case 1: Heading hợp lệ (chữ to, căn giữa, cách lề trên) -> OCR dự đoán là Heading
        valid_heading_line = {
            "text": "ĐIỀU 1. PHẠM VI ÁP DỤNG",
            "height": 22.0,  # > 1.35 * median_h
            "center_x": 400.0,
            "y_min": 150.0,
            "width": 300.0,
            "score": 0.95
        }
        res1 = reconstructor.detect_heading(valid_heading_line, median_h, page_w)
        self.assertIsNotNone(res1)
        self.assertEqual(res1["source"], "ocr_geometry_prediction")

        # Cập nhật tracker đã qua Điều 1
        reconstructor.tracker.on_heading("h2", "Điều 1. Phạm vi áp dụng", number=1)

        # Case 2: Tham chiếu chéo trong câu ("theo quy định tại Điều 12...") có cỡ chữ thân bài bình thường
        # -> OCR dự đoán là BODY TEXT, tuyệt đối không bị nhận nhầm là Heading!
        cross_ref_line = {
            "text": "theo quy định tại Điều 12 của Quy chế này thì khách hàng phải...",
            "height": 16.0,  # Không to hơn median
            "center_x": 350.0,
            "width": 600.0,
            "score": 0.90
        }
        res2 = reconstructor.detect_heading(cross_ref_line, median_h, page_w)
        self.assertIsNone(res2, "Tham chiếu chéo trong câu phải được OCR phân loại là Body Text, không phải Heading!")

        # Case 3: Câu bình thường trong thân bài có font size thân bài -> BỊ LOẠI BỎ
        normal_body_line = {
            "text": "Khách hàng có trách nhiệm bảo mật thông tin đăng nhập theo quy định.",
            "height": 16.0,
            "center_x": 300.0,
            "width": 550.0,
            "score": 0.88
        }
        res3 = reconstructor.detect_heading(normal_body_line, median_h, page_w)
        self.assertIsNone(res3)

        # Case 4: Ghép 2 dòng cho Tiêu đề lớn (Dòng 1: CHƯƠNG I, Dòng 2: QUY ĐỊNH CHUNG)
        ch_line = {
            "text": "CHƯƠNG I",
            "height": 22.0,
            "center_x": 400.0,
            "width": 150.0,
            "score": 0.96
        }
        ch_title_line = {
            "text": "QUY ĐỊNH CHUNG",
            "height": 20.0,
            "center_x": 400.0,
            "width": 250.0,
            "score": 0.95
        }
        res4 = reconstructor.detect_heading(ch_line, median_h, page_w, next_line=ch_title_line)
        self.assertIsNotNone(res4)
        self.assertIn("QUY ĐỊNH CHUNG", res4["label"])

    def test_05_reset_on_new_document(self):
        """Kiểm tra reset stack khi phát hiện văn bản mới ghép trong PDF."""
        tracker = BreadcrumbTracker(doc_title="Quy chế cũ")
        tracker.on_heading("chuong", "Chương I", number=1)
        tracker.on_heading("dieu", "Điều 1", number=1)

        # Reset khi gặp Quyết định số 123
        tracker.reset_for_new_doc("QUYẾT ĐỊNH SỐ 123/QĐ-VCB")
        self.assertEqual(tracker.doc_title, "QUYẾT ĐỊNH SỐ 123/QĐ-VCB")
        self.assertEqual(tracker.last_dieu, 0)
        self.assertEqual(len(tracker.stack), 0)

    def test_06_filter_header_footer(self):
        """Kiểm tra loại bỏ header/footer 8% trên/dưới và lặp lại >= 50% trang."""
        reconstructor = ScanStructureReconstructor(doc_title="Test Doc")
        page_h = 1000.0
        pages = [
            [
                {"text": "NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM", "y_min": 30.0, "y_max": 50.0, "page_height": page_h}, # 4% -> header
                {"text": "Nội dung trang 1", "y_min": 200.0, "y_max": 220.0, "page_height": page_h},
                {"text": "Trang 1/2", "y_min": 950.0, "y_max": 970.0, "page_height": page_h}, # 96% -> footer
            ],
            [
                {"text": "NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM", "y_min": 30.0, "y_max": 50.0, "page_height": page_h}, # header
                {"text": "Nội dung trang 2", "y_min": 250.0, "y_max": 270.0, "page_height": page_h},
                {"text": "Trang 2/2", "y_min": 950.0, "y_max": 970.0, "page_height": page_h}, # footer
            ]
        ]
        cleaned = reconstructor.filter_header_footer_lines(pages)
        self.assertEqual(len(cleaned[0]), 1)
        self.assertEqual(cleaned[0][0]["text"], "Nội dung trang 1")
        self.assertEqual(len(cleaned[1]), 1)
        self.assertEqual(cleaned[1][0]["text"], "Nội dung trang 2")

    def test_07_ocr_layout_hierarchy_predictor(self):
        """Kiểm tra OCRLayoutHierarchyPredictor dự đoán chuẩn 100% bằng hình học (0% Regex)."""
        from rag_banking.chunking.pdf_chunker.ocr_layout_predictor import OCRLayoutHierarchyPredictor, LayoutHierarchyLevel

        predictor = OCRLayoutHierarchyPredictor()
        page_w = 800.0
        page_h = 1100.0
        stats = {"median_h": 16.0, "base_left": 50.0, "median_gap": 8.0}

        # 1. Dự đoán Title (chữ to 24px, căn giữa)
        title_line = {"text": "QUY TRÌNH QUẢN LÝ TÀI CHÍNH", "height": 24.0, "width": 300.0, "x_min": 250.0, "y_min": 150.0, "center_x": 400.0}
        pred_title = predictor.predict_line_hierarchy(title_line, stats, page_w, page_h)
        self.assertEqual(pred_title["predicted_level"], LayoutHierarchyLevel.TITLE)
        self.assertTrue(pred_title["is_heading"])

        # 2. Dự đoán H1 (chữ 20px > 1.18 * 16)
        h1_line = {"text": "PHẦN I. TỔNG QUAN HỆ THỐNG", "height": 20.0, "width": 250.0, "x_min": 50.0, "y_min": 220.0, "center_x": 175.0}
        pred_h1 = predictor.predict_line_hierarchy(h1_line, stats, page_w, page_h)
        self.assertEqual(pred_h1["predicted_level"], LayoutHierarchyLevel.H1)
        self.assertTrue(pred_h1["is_heading"])

        # 3. Dự đoán Body Text (chữ 16px, bám lề 50px)
        body_line = {"text": "Đây là nội dung thân bài thông thường chảy liên tục.", "height": 16.0, "width": 650.0, "x_min": 50.0, "y_min": 260.0, "center_x": 375.0}
        pred_body = predictor.predict_line_hierarchy(body_line, stats, page_w, page_h)
        self.assertEqual(pred_body["predicted_level"], LayoutHierarchyLevel.BODY)
        self.assertFalse(pred_body["is_heading"])
        self.assertEqual(pred_body["indent_level"], 0)

        # 4. Dự đoán Khoản (Thụt lề x_min=75 so với base_left=50 -> indent=25px)
        sub_item_line = {"text": "1. Khách hàng thực hiện định danh trực tuyến.", "height": 16.0, "width": 550.0, "x_min": 75.0, "y_min": 290.0, "center_x": 350.0}
        pred_sub = predictor.predict_line_hierarchy(sub_item_line, stats, page_w, page_h)
        self.assertEqual(pred_sub["predicted_level"], LayoutHierarchyLevel.SUB_ITEM)
        self.assertEqual(pred_sub["indent_level"], 1)

        # 5. Dự đoán Điểm (Thụt lề sâu x_min=105 so với base_left=50 -> indent=55px)
        deep_item_line = {"text": "a) Cung cấp giấy tờ tùy thân hợp lệ.", "height": 16.0, "width": 500.0, "x_min": 105.0, "y_min": 320.0, "center_x": 355.0}
        pred_deep = predictor.predict_line_hierarchy(deep_item_line, stats, page_w, page_h)
        self.assertEqual(pred_deep["predicted_level"], LayoutHierarchyLevel.DEEP_ITEM)
        self.assertEqual(pred_deep["indent_level"], 2)

        # 6. Dự đoán Header mép trên (y_min < 0.08 * 1100 = 88)
        header_line = {"text": "NGÂN HÀNG NGOẠI THƯƠNG", "height": 14.0, "width": 200.0, "x_min": 50.0, "y_min": 30.0, "center_x": 150.0}
        pred_header = predictor.predict_line_hierarchy(header_line, stats, page_w, page_h)
        self.assertEqual(pred_header["predicted_level"], LayoutHierarchyLevel.HEADER_FOOTER)


if __name__ == "__main__":
    unittest.main()
