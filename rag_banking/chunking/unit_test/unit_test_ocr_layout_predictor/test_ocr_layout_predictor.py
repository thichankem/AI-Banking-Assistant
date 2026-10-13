"""
Unit Test: OCR Layout Hierarchy Predictor (Node 02)
Kiểm thử bộ dự đoán cấp bậc bố cục thị giác OCR hoàn toàn bằng hình học (0% Regex).
Kiểm tra độ chính xác phân loại Title, H1, H2, H3, Sub-item (Khoản), Deep-item (Điểm), Body, Header/Footer.
"""

import sys
import json
import unittest
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Tìm root repository động để chạy được từ bất kỳ đâu
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR
for p in [CURRENT_DIR] + list(CURRENT_DIR.parents):
    if (p / "rag_banking").is_dir():
        REPO_ROOT = p
        break
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_banking.chunking.pdf_chunker.ocr_layout_predictor import (
    OCRLayoutHierarchyPredictor,
    LayoutHierarchyLevel
)


class TestOCRLayoutHierarchyPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = OCRLayoutHierarchyPredictor()
        cls.dataset_dir = CURRENT_DIR / "dataset"
        cls.labels_file = CURRENT_DIR / "ground_truth_labels.json" if (CURRENT_DIR / "ground_truth_labels.json").exists() else CURRENT_DIR / "labels.json"

        with open(cls.labels_file, "r", encoding="utf-8") as f:
            cls.labels = json.load(f)

    def test_all_60_layout_samples(self):
        """Chạy kiểm thử toàn bộ 60 mẫu ảnh PNG bố cục hình học OCR."""
        total = len(self.labels)
        self.assertGreaterEqual(total, 50, "Dataset phải có ít nhất 50 mẫu theo tiêu chuẩn!")

        passed = 0
        failed_cases = []
        results_list = []

        for png_name, label_info in self.labels.items():
            png_path = self.dataset_dir / png_name
            # 1. Thẩm định file ảnh PNG tồn tại thực tế và có kích thước hợp lệ
            self.assertTrue(png_path.exists(), f"Không tìm thấy file ảnh PNG: {png_name}")
            self.assertGreater(png_path.stat().st_size, 1000, f"File ảnh PNG rỗng hoặc hỏng: {png_name}")

            # 2. Đọc metadata cấu trúc hình học tương ứng của trang ảnh
            json_ref = label_info.get("json_ref", png_name.replace(".png", ".json"))
            json_path = self.dataset_dir / json_ref
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            page_w = data["page_width"]
            page_h = data["page_height"]
            lines = data["lines"]
            target_idx = data["target_idx"]

            # 3. Phân loại bố cục bằng mô hình OCRLayoutHierarchyPredictor (0% Regex)
            annotated_lines = self.predictor.classify_page_lines(lines, page_w, page_h)
            pred = annotated_lines[target_idx]

            expected_level = label_info["expected_level"]
            expected_heading = label_info["expected_heading"]
            expected_indent = label_info["expected_indent"]

            pred_level = pred["predicted_level"]
            pred_heading = pred["is_heading"]
            pred_indent = pred["indent_level"]

            is_correct = (
                pred_level == expected_level and
                pred_heading == expected_heading and
                pred_indent == expected_indent
            )

            results_list.append({
                "sample": png_name,
                "target_text": label_info.get("target_text", ""),
                "expected": {
                    "level": expected_level,
                    "is_heading": expected_heading,
                    "indent_level": expected_indent
                },
                "predicted": {
                    "level": pred_level,
                    "is_heading": pred_heading,
                    "indent_level": pred_indent
                },
                "status": "PASS" if is_correct else "FAIL",
                "layout_features": pred.get("layout_features", {})
            })

            if is_correct:
                passed += 1
            else:
                failed_cases.append({
                    "image": png_name,
                    "target_text": label_info.get("target_text", ""),
                    "expected": (expected_level, expected_heading, expected_indent),
                    "got": (pred_level, pred_heading, pred_indent),
                    "features": pred.get("layout_features", {})
                })

        accuracy = (passed / total) * 100.0

        # Xuất kết quả chi tiết ra test_results.json
        out_json_path = CURRENT_DIR / "test_results.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "node_id": "Node 02",
                "node_name": "OCR Layout Hierarchy Predictor (0% Regex)",
                "timestamp": datetime.now().isoformat(),
                "total_samples": total,
                "passed": passed,
                "failed": len(failed_cases),
                "accuracy_percent": round(accuracy, 2),
                "status": "PASS" if len(failed_cases) == 0 else "FAIL",
                "results": results_list
            }, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print(f"[*] KẾT QUẢ TEST NODE 02: OCR Layout Hierarchy Predictor")
        print(f"    - Tổng số mẫu kiểm thử: {total}")
        print(f"    - Đạt chuẩn (PASS)   : {passed}/{total}")
        print(f"    - Độ chính xác       : {accuracy:.2f}%")
        if failed_cases:
            print("    - Các ca lỗi:")
            for fc in failed_cases[:10]:
                print(f"      + {fc['image']} ({fc['target_text']}): Exp {fc['expected']} != Got {fc['got']} (Features: {fc['features']})")
        print("=" * 70)

        self.assertEqual(passed, total, f"Có {len(failed_cases)} mẫu không đạt kết quả kỳ vọng!")


if __name__ == "__main__":
    unittest.main()
