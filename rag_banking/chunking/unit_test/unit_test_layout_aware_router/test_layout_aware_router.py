"""
Unit Test: Multi-Format Layout-Aware Router & Pipeline (Node 07)
Kiểm thử bộ định tuyến tài liệu ngân hàng đa định dạng:
1. Định tuyến chuẩn xác: .docx -> DOCXChunker, .pdf -> PDFChunker, .xlsx -> ExcelChunker, .md -> Markdown Chunker.
2. Unified Schema Enforcement: Mọi chunk đều BẮT BUỘC có trường 'source' và TUYỆT ĐỐI KHÔNG CÓ trường 'ref', 'legal_ref', 'raw_content', 'char_length', 'format'.
3. Khả năng tương thích trọn vẹn xuyên suốt toàn bộ Phase 1 Chunking Pipeline.
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

from rag_banking.chunking.pipeline import ChunkingPipeline


class TestLayoutAwareRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = ChunkingPipeline()
        cls.dataset_dir = CURRENT_DIR / "dataset"
        cls.labels_file = CURRENT_DIR / "ground_truth_labels.json" if (CURRENT_DIR / "ground_truth_labels.json").exists() else CURRENT_DIR / "labels.json"

        with open(cls.labels_file, "r", encoding="utf-8") as f:
            cls.labels = json.load(f)

    def test_all_router_samples(self):
        """Chạy kiểm thử toàn bộ các tệp đa định dạng qua ChunkingPipeline Router."""
        total = len(self.labels)
        self.assertGreaterEqual(total, 30, "Dataset phải có ít nhất 30 mẫu DOCX và PDF theo tiêu chuẩn!")

        passed = 0
        failed_cases = []
        results_list = []

        for fname, label_info in self.labels.items():
            fpath = self.dataset_dir / fname
            chunks = self.pipeline.process_file(fpath, max_pdf_pages=3)

            # 1. Kiểm tra số lượng chunk tối thiểu
            min_chunks = label_info["expected_min_chunks"]
            has_enough_chunks = len(chunks) >= min_chunks

            # 2. Kiểm tra Schema: BẮT BUỘC có 'source', KHÔNG có 'ref', 'legal_ref', 'raw_content', 'char_length', 'format'
            valid_schema = True
            for c in chunks:
                if "source" not in c or any(k in c for k in ["ref", "legal_ref", "raw_content", "char_length", "format"]):
                    valid_schema = False
                    break

            # 3. Kiểm tra chunk_type được ánh xạ đúng
            valid_types = True
            exp_types = label_info.get("expected_chunk_types", [])
            for c in chunks:
                ctype = c.get("chunk_type")
                if exp_types and ctype not in exp_types and ctype != "table":
                    valid_types = False
                    break

            # 4. Kiểm tra content và breadcrumb không rỗng
            valid_content = all(bool(c.get("content")) and bool(c.get("breadcrumb")) for c in chunks)

            is_success = has_enough_chunks and valid_schema and valid_types and valid_content

            results_list.append({
                "sample": fname,
                "suffix": label_info.get("suffix"),
                "chunks_count": len(chunks),
                "expected_min_chunks": min_chunks,
                "valid_schema": valid_schema,
                "valid_types": valid_types,
                "valid_content": valid_content,
                "detected_chunk_types": list(set([c.get("chunk_type") for c in chunks])),
                "sample_preview": [c.get("content", "")[:120] + "..." for c in chunks[:2]],
                "status": "PASS" if is_success else "FAIL"
            })

            if is_success:
                passed += 1
            else:
                failed_cases.append({
                    "file": fname,
                    "suffix": label_info.get("suffix"),
                    "chunks": len(chunks),
                    "min_expected": min_chunks,
                    "valid_schema": valid_schema,
                    "valid_types": valid_types,
                    "valid_content": valid_content,
                    "got_types": [c.get("chunk_type") for c in chunks]
                })

        accuracy = (passed / total) * 100.0

        # Xuất kết quả chi tiết ra test_results.json
        out_json_path = CURRENT_DIR / "test_results.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "node_id": "Node 07",
                "node_name": "Multi-Format Layout-Aware Router",
                "timestamp": datetime.now().isoformat(),
                "total_samples": total,
                "passed": passed,
                "failed": len(failed_cases),
                "accuracy_percent": round(accuracy, 2),
                "status": "PASS" if len(failed_cases) == 0 else "FAIL",
                "results": results_list
            }, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print(f"[*] KẾT QUẢ TEST NODE 07: Multi-Format Layout-Aware Router")
        print(f"    - Tổng số tệp kiểm thử: {total} (.docx, .pdf, .xlsx, .md)")
        print(f"    - Đạt chuẩn (PASS)     : {passed}/{total}")
        print(f"    - Độ chính xác         : {accuracy:.2f}%")
        if failed_cases:
            print("    - Các ca lỗi:")
            for fc in failed_cases[:5]:
                print(f"      + {fc['file']} ({fc['suffix']}): chunks={fc['chunks']} (exp>={fc['min_expected']}), schema={fc['valid_schema']}, types={fc['got_types']}")
        print("=" * 70)

        self.assertEqual(passed, total, f"Có {len(failed_cases)} tệp không đạt kết quả kỳ vọng!")


if __name__ == "__main__":
    unittest.main()
