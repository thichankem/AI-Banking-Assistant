"""
Unit Test: Scan Structure Reconstructor (Node 03)
Kiểm thử bộ tái tạo cấu trúc văn bản scan và State Machine BreadcrumbTracker (0% Regex DLA).
Xác minh tính đúng đắn của việc lọc Header/Footer, phân cấp Chương -> Điều -> Khoản -> Điểm,
và tuân thủ schema (có source, không có ref, legal_ref, raw_content, char_length, format).
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

from rag_banking.chunking.pdf_chunker.scan_structure_reconstructor import ScanStructureReconstructor


class TestScanStructureReconstructor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = CURRENT_DIR / "dataset"
        cls.labels_file = CURRENT_DIR / "ground_truth_labels.json" if (CURRENT_DIR / "ground_truth_labels.json").exists() else CURRENT_DIR / "labels.json"

        with open(cls.labels_file, "r", encoding="utf-8") as f:
            cls.labels = json.load(f)

    def test_all_scan_reconstructor_samples(self):
        """Chạy kiểm thử toàn bộ các tài liệu scan qua ScanStructureReconstructor."""
        total = len(self.labels)
        self.assertGreaterEqual(total, 50, "Dataset phải có ít nhất 50 mẫu theo tiêu chuẩn!")

        passed = 0
        failed_cases = []
        results_list = []

        for fname, label_info in self.labels.items():
            fpath = self.dataset_dir / fname
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            doc_title = data["doc_title"]
            pages_lines = data["pages_lines"]

            reconstructor = ScanStructureReconstructor(doc_title=doc_title)
            chunks = reconstructor.reconstruct_document_chunks(
                pages_lines=pages_lines,
                source_name=f"{fname}.pdf",
                max_chunk_size=800,
                min_section_size=50
            )

            # 1. Kiểm tra số lượng chunk tối thiểu
            min_chunks = label_info["expected_min_chunks"]
            has_enough_chunks = len(chunks) >= min_chunks

            # 2. Kiểm tra schema: Có source, KHÔNG có ref, legal_ref, raw_content, char_length, format
            valid_schema = True
            for c in chunks:
                if "source" not in c or any(k in c for k in ["ref", "legal_ref", "raw_content", "char_length", "format"]):
                    valid_schema = False
                    break

            # 3. Kiểm tra breadcrumb có chứa các từ khóa mong đợi
            all_breadcrumbs = " || ".join(c.get("breadcrumb", "") for c in chunks)
            breadcrumbs_ok = True
            for kw in label_info["expected_breadcrumbs_contain"]:
                if kw not in all_breadcrumbs:
                    breadcrumbs_ok = False
                    break

            # 4. Kiểm tra Header/Footer không bị lọt vào nội dung
            all_content = " ".join(c.get("content", "") for c in chunks)
            no_garbage = True
            for bad_kw in label_info["unexpected_in_content"]:
                if bad_kw in all_content:
                    no_garbage = False
                    break

            is_success = has_enough_chunks and valid_schema and breadcrumbs_ok and no_garbage

            results_list.append({
                "sample": fname,
                "doc_title": doc_title,
                "chunks_count": len(chunks),
                "expected_min_chunks": min_chunks,
                "valid_schema": valid_schema,
                "breadcrumbs_ok": breadcrumbs_ok,
                "no_garbage": no_garbage,
                "sample_breadcrumbs": [c.get("breadcrumb", "") for c in chunks[:3]],
                "status": "PASS" if is_success else "FAIL"
            })

            if is_success:
                passed += 1
            else:
                failed_cases.append({
                    "file": fname,
                    "chunks": len(chunks),
                    "min_expected": min_chunks,
                    "valid_schema": valid_schema,
                    "breadcrumbs_ok": breadcrumbs_ok,
                    "no_garbage": no_garbage,
                    "breadcrumbs": all_breadcrumbs
                })

        accuracy = (passed / total) * 100.0

        # Xuất kết quả chi tiết ra test_results.json
        out_json_path = CURRENT_DIR / "test_results.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "node_id": "Node 03",
                "node_name": "Scan Structure Reconstructor",
                "timestamp": datetime.now().isoformat(),
                "total_samples": total,
                "passed": passed,
                "failed": len(failed_cases),
                "accuracy_percent": round(accuracy, 2),
                "status": "PASS" if len(failed_cases) == 0 else "FAIL",
                "results": results_list
            }, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print(f"[*] KẾT QUẢ TEST NODE 03: Scan Structure Reconstructor")
        print(f"    - Tổng số tài liệu kiểm thử: {total}")
        print(f"    - Đạt chuẩn (PASS)         : {passed}/{total}")
        print(f"    - Độ chính xác             : {accuracy:.2f}%")
        if failed_cases:
            print("    - Các ca lỗi:")
            for fc in failed_cases[:5]:
                print(f"      + {fc['file']}: chunks={fc['chunks']} (exp>={fc['min_expected']}), schema={fc['valid_schema']}, breadcrumb={fc['breadcrumbs_ok']}, clean={fc['no_garbage']}")
        print("=" * 70)

        self.assertEqual(passed, total, f"Có {len(failed_cases)} tài liệu scan không đạt kết quả kỳ vọng!")


if __name__ == "__main__":
    unittest.main()
