"""
Unit Test: Excel Table Layout-Aware Chunker (Node 05)
Kiểm thử bộ trích xuất và phân đoạn ma trận bảng biểu Excel (.xlsx) ngân hàng:
1. Multi-Sheet Extraction (trích xuất toàn bộ các sheet dữ liệu).
2. Dynamic Header Row Detection (bỏ qua hàng trống và tiêu đề trang đầu).
3. Linearized Table Rows (ghép ô với tên cột tương ứng 'Cột: Giá trị').
4. Schema tuân thủ chuẩn: có 'source', KHÔNG có 'ref', 'legal_ref', 'raw_content', 'char_length', 'format'.
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

from rag_banking.chunking.unified_chunk.excel_chunker import ExcelChunker


class TestExcelChunker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunker = ExcelChunker(rows_per_chunk=3)
        cls.dataset_dir = CURRENT_DIR / "dataset"
        cls.labels_file = CURRENT_DIR / "ground_truth_labels.json" if (CURRENT_DIR / "ground_truth_labels.json").exists() else CURRENT_DIR / "labels.json"

        with open(cls.labels_file, "r", encoding="utf-8") as f:
            cls.labels = json.load(f)

    def test_all_excel_samples(self):
        """Chạy kiểm thử toàn bộ các tệp Excel thực tế qua ExcelChunker."""
        total = len(self.labels)
        self.assertGreaterEqual(total, 50, "Dataset phải có ít nhất 50 mẫu theo tiêu chuẩn!")

        passed = 0
        failed_cases = []
        results_list = []

        for fname, label_info in self.labels.items():
            fpath = self.dataset_dir / fname
            chunks = self.chunker.chunk_excel(fpath)

            # 1. Kiểm tra số lượng chunk tối thiểu
            min_chunks = label_info["expected_min_chunks"]
            has_enough_chunks = len(chunks) >= min_chunks

            # 2. Kiểm tra Schema: Có 'source', KHÔNG có 'ref', 'legal_ref', 'raw_content', 'char_length', 'format'
            valid_schema = True
            for c in chunks:
                if "source" not in c or any(k in c for k in ["ref", "legal_ref", "raw_content", "char_length", "format"]):
                    valid_schema = False
                    break

            # 3. Kiểm tra chunk_type
            valid_types = all(c.get("chunk_type") == "excel_table" for c in chunks)

            # 4. Kiểm tra toàn bộ expected_sheets đã được bóc tách
            all_breadcrumbs = " || ".join(c.get("breadcrumb", "") for c in chunks)
            sheets_ok = True
            for sheet in label_info["expected_sheets"]:
                if sheet not in all_breadcrumbs:
                    sheets_ok = False
                    break

            # 5. Kiểm tra Linearized Header (Cột: Giá trị)
            all_content = " ".join(c.get("content", "") for c in chunks)
            headers_ok = True
            for hdr in label_info["expected_headers"]:
                if hdr not in all_content:
                    headers_ok = False
                    break

            is_success = has_enough_chunks and valid_schema and valid_types and sheets_ok and headers_ok

            results_list.append({
                "sample": fname,
                "chunks_count": len(chunks),
                "expected_min_chunks": min_chunks,
                "expected_sheets": label_info.get("expected_sheets", []),
                "valid_schema": valid_schema,
                "valid_types": valid_types,
                "sheets_ok": sheets_ok,
                "headers_ok": headers_ok,
                "sample_preview": [c.get("content", "")[:120] + "..." for c in chunks[:2]],
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
                    "valid_types": valid_types,
                    "sheets_ok": sheets_ok,
                    "headers_ok": headers_ok
                })

        accuracy = (passed / total) * 100.0

        # Xuất kết quả chi tiết ra test_results.json
        out_json_path = CURRENT_DIR / "test_results.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "node_id": "Node 05",
                "node_name": "Excel Table Layout Chunker",
                "timestamp": datetime.now().isoformat(),
                "total_samples": total,
                "passed": passed,
                "failed": len(failed_cases),
                "accuracy_percent": round(accuracy, 2),
                "status": "PASS" if len(failed_cases) == 0 else "FAIL",
                "results": results_list
            }, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print(f"[*] KẾT QUẢ TEST NODE 05: Excel Table Layout Chunker")
        print(f"    - Tổng số tệp Excel kiểm thử: {total}")
        print(f"    - Đạt chuẩn (PASS)          : {passed}/{total}")
        print(f"    - Độ chính xác              : {accuracy:.2f}%")
        if failed_cases:
            print("    - Các ca lỗi:")
            for fc in failed_cases[:5]:
                print(f"      + {fc['file']}: chunks={fc['chunks']} (exp>={fc['min_expected']}), schema={fc['valid_schema']}, sheets={fc['sheets_ok']}, headers={fc['headers_ok']}")
        print("=" * 70)

        self.assertEqual(passed, total, f"Có {len(failed_cases)} tệp Excel không đạt kết quả kỳ vọng!")


if __name__ == "__main__":
    unittest.main()
