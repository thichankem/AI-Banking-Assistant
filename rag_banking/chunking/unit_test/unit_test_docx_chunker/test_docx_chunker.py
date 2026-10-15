"""
Unit Test: DOCX Layout-Aware & Advanced Table Chunker (Node 04)
Kiểm thử bộ phân đoạn tài liệu Word (.docx) ngân hàng nhận thức bố cục DLA:
1. Heuristic Heading Detection (nhận diện in đậm không phụ thuộc Word style).
2. Table Atomicity & Multi-level Table Header (lặp lại header 2 tầng ở mọi chunk bảng).
3. Section-Length-Aware Anti-Fragmentation (tự gộp các mục < 120 từ).
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

from rag_banking.chunking.docx_chunker.docx_chunker import DOCXChunker


class TestDOCXChunker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunker = DOCXChunker(
            max_chunk_size=800,
            min_section_size=120,
            table_rows_per_chunk=4
        )
        cls.dataset_dir = CURRENT_DIR / "dataset"
        cls.labels_file = CURRENT_DIR / "ground_truth_labels.json" if (CURRENT_DIR / "ground_truth_labels.json").exists() else CURRENT_DIR / "labels.json"

        with open(cls.labels_file, "r", encoding="utf-8") as f:
            cls.labels = json.load(f)

    def test_all_docx_samples(self):
        """Chạy kiểm thử toàn bộ các tệp DOCX thực tế qua DOCXChunker."""
        total = len(self.labels)
        self.assertGreaterEqual(total, 50, "Dataset phải có ít nhất 50 mẫu theo tiêu chuẩn!")

        passed = 0
        failed_cases = []
        results_list = []

        for fname, label_info in self.labels.items():
            fpath = self.dataset_dir / fname
            chunks = self.chunker.chunk_docx(fpath)

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
            valid_types = True
            for c in chunks:
                if c.get("chunk_type") not in ["docx_text", "docx_table"]:
                    valid_types = False
                    break

            # 4. Kiểm tra breadcrumbs
            all_breadcrumbs = " || ".join(c.get("breadcrumb", "") for c in chunks)
            breadcrumbs_ok = True
            for kw in label_info["expected_breadcrumb_keywords"]:
                if kw not in all_breadcrumbs:
                    breadcrumbs_ok = False
                    break

            # 5. Kiểm tra Table Atomicity (header lặp lại trên mọi table chunk)
            table_chunks = [c for c in chunks if c.get("chunk_type") == "docx_table"]
            table_headers_ok = True
            if label_info.get("has_table") and len(table_chunks) > 1:
                for tc in table_chunks:
                    content = tc.get("content", "")
                    if "Hàng tiêu đề:" not in content and "Cột dữ liệu:" not in content and "Cột:" not in content:
                        table_headers_ok = False
                        break

            is_success = has_enough_chunks and valid_schema and valid_types and breadcrumbs_ok and table_headers_ok

            results_list.append({
                "sample": fname,
                "chunks_count": len(chunks),
                "expected_min_chunks": min_chunks,
                "has_table": label_info.get("has_table", False),
                "table_chunks_count": len(table_chunks),
                "valid_schema": valid_schema,
                "valid_types": valid_types,
                "breadcrumbs_ok": breadcrumbs_ok,
                "table_headers_ok": table_headers_ok,
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
                    "valid_types": valid_types,
                    "breadcrumbs_ok": breadcrumbs_ok,
                    "table_headers_ok": table_headers_ok
                })

        accuracy = (passed / total) * 100.0

        # Xuất kết quả chi tiết ra test_results.json
        out_json_path = CURRENT_DIR / "test_results.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "node_id": "Node 04",
                "node_name": "DOCX Layout-Aware Chunker",
                "timestamp": datetime.now().isoformat(),
                "total_samples": total,
                "passed": passed,
                "failed": len(failed_cases),
                "accuracy_percent": round(accuracy, 2),
                "status": "PASS" if len(failed_cases) == 0 else "FAIL",
                "results": results_list
            }, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print(f"[*] KẾT QUẢ TEST NODE 04: DOCX Layout-Aware Chunker")
        print(f"    - Tổng số tệp DOCX kiểm thử: {total}")
        print(f"    - Đạt chuẩn (PASS)         : {passed}/{total}")
        print(f"    - Độ chính xác             : {accuracy:.2f}%")
        if failed_cases:
            print("    - Các ca lỗi:")
            for fc in failed_cases[:5]:
                print(f"      + {fc['file']}: chunks={fc['chunks']} (exp>={fc['min_expected']}), schema={fc['valid_schema']}, breadcrumb={fc['breadcrumbs_ok']}, tbl_hdr={fc['table_headers_ok']}")
        print("=" * 70)

        self.assertEqual(passed, total, f"Có {len(failed_cases)} tệp DOCX không đạt kết quả kỳ vọng!")


if __name__ == "__main__":
    unittest.main()
