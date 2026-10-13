"""
Unit Test & Công Cụ Kiểm Thử: Nhận Diện Trang PDF Scan / Digital (Scan Detection)
================================================================================
Tệp kiểm thử chuẩn hóa cho khối:
"Trang Scan? (img_area > 40% hoặc lỗi font)" trong Flowchart Pipeline.

Mục đích:
1. Kiểm chuẩn tự động (Automated Benchmark): Chạy qua 100 tệp PDF mẫu (50 scan, 50 digital)
   để xác thực độ chính xác (Accuracy, Precision, Recall, F1 = 100%).
2. Kiểm tra đơn lập trực quan (Interactive Inspector): Kiểm tra chi tiết 1 tệp PDF bất kỳ
   hoặc mẫu đại diện để xem các chỉ số hình học và lý do quyết định.

Cách sử dụng:
-------------
1. Chạy toàn bộ Unit Test (chuẩn unittest / CI):
   python -m unittest rag_banking/chunking/unit_test/test_scan_detection.py

2. Chạy kiểm thử & in bảng tổng kết trực quan:
   python rag_banking/chunking/unit_test/test_scan_detection.py

3. Kiểm tra chi tiết một tệp PDF tùy ý:
   python rag_banking/chunking/unit_test/test_scan_detection.py "duong_dan_den_file.pdf"
"""

import sys
import json
import argparse
import unittest
from datetime import datetime
from pathlib import Path

# Đảm bảo console Windows hỗ trợ in Unicode tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR
for p in [CURRENT_DIR] + list(CURRENT_DIR.parents):
    if (p / "rag_banking").is_dir():
        REPO_ROOT = p
        break
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from rag_banking.chunking.pdf_chunker.scan_detector import (
    detect_is_scanned_page,
    is_scanned_page,
    detect_pdf_file_scan_status,
    ScanDetector
)

LABELS_PATH = CURRENT_DIR / "ground_truth_labels.json" if (CURRENT_DIR / "ground_truth_labels.json").exists() else CURRENT_DIR / "labels.json"
DATASET_DIR = CURRENT_DIR / "dataset"
SCANNED_DIR = DATASET_DIR / "scanned"
DIGITAL_DIR = DATASET_DIR / "digital"


def inspect_single_pdf(file_path: Path, max_pages: int = 5):
    """Kiểm tra và in chi tiết chỉ số nhận diện scan cho từng trang của 1 tệp PDF."""
    if not file_path.exists():
        print(f"[Lỗi] Không tìm thấy tệp: {file_path}")
        return

    doc = fitz.open(str(file_path))
    total_pages = len(doc)
    pages_to_check = min(total_pages, max_pages)

    print("\n" + "=" * 76)
    print(f"KẾT QUẢ KIỂM TRA NHẬN DIỆN SCAN: {file_path.name}")
    print("=" * 76)
    print(f"• Đường dẫn tệp : {file_path}")
    print(f"• Tổng số trang : {total_pages} (Kiểm tra {pages_to_check} trang đầu)")
    print("-" * 76)

    for idx in range(pages_to_check):
        page = doc[idx]
        is_scanned, metrics = detect_is_scanned_page(page, image_area_ratio_threshold=0.40)
        tag = "🔴 [SCAN (Cần OCR)]" if is_scanned else "🟢 [DIGITAL (Trích xuất Text)]"

        print(f"Trang {idx + 1:02d}/{total_pages:02d} -> {tag}")
        print(f"  - Quyết định      : {'Bản Scan Vật Lý' if is_scanned else 'Văn Bản Số Hóa'}")
        print(f"  - Lý do           : {metrics['reason']}")
        print(f"  - Tỷ lệ ảnh       : {metrics['img_area_ratio']:.2%} (Ngưỡng kích hoạt: >= 40.00%)")
        print(f"  - Số ảnh XObject  : {metrics['img_count']}")
        print(f"  - Số lượng ký tự  : {metrics['char_count']}")
        print(f"  - Số lượng từ     : {metrics['word_count']}")
        print(f"  - Tiếng Việt chuẩn: {'Hợp lệ' if metrics['is_valid_vietnamese'] else 'Lỗi font / rác ký tự'}")
        print("-" * 76)

    doc.close()


def print_visual_benchmark_summary():
    """Chạy và in bảng tổng kết trực quan nhanh cho tập dữ liệu."""
    if not LABELS_PATH.exists():
        print(f"[Cảnh báo] Không tìm thấy {LABELS_PATH}, bỏ qua in bảng tổng kết.")
        return

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    tp, tn, fp, fn = 0, 0, 0, 0

    for sample in samples:
        pdf_path = CURRENT_DIR / sample["relative_path"]
        if not pdf_path.exists():
            continue
        doc = fitz.open(str(pdf_path))
        is_scanned, _ = detect_is_scanned_page(doc[0])
        doc.close()

        actual = sample["is_scanned"]
        if actual and is_scanned:
            tp += 1
        elif not actual and not is_scanned:
            tn += 1
        elif not actual and is_scanned:
            fp += 1
        elif actual and not is_scanned:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 76)
    print("BẢNG TỔNG KẾT ĐÁNH GIÁ ĐỘ CHÍNH XÁC NHẬN DIỆN SCAN (BENCHMARK 100 TRANG)")
    print("=" * 76)
    print(f"• Tổng số mẫu kiểm thử      : {total} trang (50 Scan, 50 Digital)")
    print(f"• True Positives (Scan đúng): {tp}/50 trang")
    print(f"• True Negatives (Số đúng)  : {tn}/50 trang")
    print(f"• False Positives (Nhầm scan): {fp} trang")
    print(f"• False Negatives (Bỏ sót)  : {fn} trang")
    print("-" * 76)
    print(f"• Accuracy  (Độ chính xác)  : {accuracy * 100:.2f}%")
    print(f"• Precision (Độ chuẩn xác)  : {precision * 100:.2f}%")
    print(f"• Recall    (Độ thu hồi)    : {recall * 100:.2f}%")
    print(f"• F1-Score  (Điểm điều hòa) : {f1 * 100:.2f}%")
    print("=" * 76)


class TestScanDetection(unittest.TestCase):
    """Suite kiểm thử tự động toàn diện cho bộ nhận diện trang Scan / Digital."""

    @classmethod
    def setUpClass(cls):
        cls.labels_file = LABELS_PATH
        cls.assertTrue(cls.labels_file.exists(), f"Không tìm thấy nhãn kiểm chuẩn: {cls.labels_file}")
        with open(cls.labels_file, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)
        cls.samples = cls.manifest["samples"]
        cls.scanned_samples = [s for s in cls.samples if s["is_scanned"]]
        cls.digital_samples = [s for s in cls.samples if not s["is_scanned"]]

    def test_01_dataset_integrity(self):
        """1. Kiểm tra tính toàn vẹn của bộ dữ liệu: đúng 50 scan, 50 digital, tổng cộng 100 mẫu."""
        self.assertEqual(len(self.scanned_samples), 50, "Số lượng trang scan phải là 50!")
        self.assertEqual(len(self.digital_samples), 50, "Số lượng trang digital phải là 50!")
        self.assertEqual(len(self.samples), 100, "Tổng số mẫu kiểm thử phải là 100!")

    def test_02_detect_scanned_pages(self):
        """2. Kiểm tra bộ nhận diện trên 50/50 trang PDF scan vật lý (yêu cầu 100% nhận diện là Scan)."""
        failed_cases = []
        for sample in self.scanned_samples:
            pdf_path = CURRENT_DIR / sample["relative_path"]
            self.assertTrue(pdf_path.exists(), f"Không tìm thấy file: {pdf_path}")

            doc = fitz.open(str(pdf_path))
            self.assertEqual(len(doc), 1, f"Tệp {sample['file_name']} phải có đúng 1 trang!")
            is_scanned, metrics = detect_is_scanned_page(doc[0])
            doc.close()

            if not is_scanned:
                failed_cases.append((sample["file_name"], metrics))

        self.assertEqual(
            len(failed_cases), 0,
            f"Có {len(failed_cases)} trang scan bị nhận nhầm thành digital: {failed_cases}"
        )

    def test_03_detect_digital_pages(self):
        """3. Kiểm tra bộ nhận diện trên 50/50 trang PDF văn bản số (yêu cầu 100% nhận diện là Digital)."""
        failed_cases = []
        for sample in self.digital_samples:
            pdf_path = CURRENT_DIR / sample["relative_path"]
            self.assertTrue(pdf_path.exists(), f"Không tìm thấy file: {pdf_path}")

            doc = fitz.open(str(pdf_path))
            self.assertEqual(len(doc), 1, f"Tệp {sample['file_name']} phải có đúng 1 trang!")
            is_scanned, metrics = detect_is_scanned_page(doc[0])
            doc.close()

            if is_scanned:
                failed_cases.append((sample["file_name"], metrics))

        self.assertEqual(
            len(failed_cases), 0,
            f"Có {len(failed_cases)} trang digital bị nhận nhầm thành scan: {failed_cases}"
        )

    def test_04_benchmark_accuracy_and_f1(self):
        """4. Đánh giá số liệu toàn diện: Accuracy >= 98% và đạt 100% trên dataset chuẩn."""
        tp, tn, fp, fn = 0, 0, 0, 0
        results_list = []
        for sample in self.samples:
            pdf_path = CURRENT_DIR / sample["relative_path"]
            doc = fitz.open(str(pdf_path))
            is_scanned, metrics = detect_is_scanned_page(doc[0])
            doc.close()

            actual = sample["is_scanned"]
            is_pass = (is_scanned == actual)
            if actual and is_scanned:
                tp += 1
            elif not actual and not is_scanned:
                tn += 1
            elif not actual and is_scanned:
                fp += 1
            elif actual and not is_scanned:
                fn += 1

            results_list.append({
                "file_name": sample["file_name"],
                "relative_path": sample["relative_path"],
                "is_scanned_expected": actual,
                "is_scanned_predicted": is_scanned,
                "status": "PASS" if is_pass else "FAIL",
                "metrics": {
                    "text_chars": metrics.get("text_chars", 0),
                    "images_count": metrics.get("images_count", 0),
                    "image_area_ratio": round(metrics.get("image_area_ratio", 0.0), 4),
                    "reason": metrics.get("reason", "")
                }
            })

        accuracy = (tp + tn) / len(self.samples)
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        out_json_path = CURRENT_DIR / "test_results.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "node_id": "Node 01",
                "node_name": "Scan Detector",
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(self.samples),
                "passed": tp + tn,
                "failed": fp + fn,
                "accuracy_percent": round(accuracy * 100, 2),
                "f1_score_percent": round(f1 * 100, 2),
                "status": "PASS" if (fp + fn) == 0 else "FAIL",
                "summary": {
                    "true_positives": tp,
                    "true_negatives": tn,
                    "false_positives": fp,
                    "false_negatives": fn
                },
                "results": results_list
            }, f, ensure_ascii=False, indent=2)

        self.assertGreaterEqual(accuracy, 0.98, "Accuracy phải đạt tối thiểu 98%!")
        self.assertEqual(accuracy, 1.0, "Accuracy trên tập dữ liệu chuẩn đạt 100%!")
        self.assertEqual(f1, 1.0, "F1-Score trên tập dữ liệu chuẩn đạt 1.0 (100%)!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Chế độ kiểm tra 1 tệp tùy chọn từ dòng lệnh
        target_file = Path(sys.argv[1])
        inspect_single_pdf(target_file)
    else:
        # In tóm tắt trước
        print_visual_benchmark_summary()
        # Chạy unittest
        unittest.main()
