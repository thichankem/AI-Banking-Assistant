"""
Unit Test: Canonical Entity & Legal Structure Resolver (Node 06)
Kiểm thử module NLP chuyên biệt cho văn bản ngân hàng và thể thức quy phạm pháp luật:
1. Advanced Sentence Splitting (bảo vệ số tiền, lãi suất, Điều 12., TP.HCM, v.v.).
2. Dehyphenation & Hard-line breaking (nối từ bị ngắt dòng gạch nối).
3. Document Identifier & Cross-citations (chuẩn thể thức Nghị định 30/2020/NĐ-CP).
4. Hierarchical Units Resolution (Chương -> Mục -> Điều -> Khoản -> Điểm -> FAQ).
5. Vietnamese Text Quality Checker (lọc bỏ văn bản scan rác).
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

from rag_banking.chunking.unified_chunk.canonical_resolver import (
    split_sentences_banking,
    clean_and_dehyphenate_text,
    extract_document_identifier,
    extract_hierarchical_units,
    is_valid_vietnamese_text
)


class TestCanonicalResolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = CURRENT_DIR / "dataset"
        cls.labels_file = CURRENT_DIR / "ground_truth_labels.json" if (CURRENT_DIR / "ground_truth_labels.json").exists() else CURRENT_DIR / "labels.json"

        with open(cls.labels_file, "r", encoding="utf-8") as f:
            cls.labels = json.load(f)

    def test_all_52_canonical_samples(self):
        """Chạy kiểm thử toàn bộ 52 mẫu NLP ngân hàng qua Canonical Resolver."""
        total = len(self.labels)
        self.assertGreaterEqual(total, 50, "Dataset phải có ít nhất 50 mẫu theo tiêu chuẩn!")

        passed = 0
        failed_cases = []
        results_list = []

        for fname, label_info in self.labels.items():
            fpath = self.dataset_dir / fname
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            test_type = label_info["test_type"]
            text = data["text"]
            is_success = False
            detail = ""

            if test_type == "sentence_split":
                parts = split_sentences_banking(text)
                exp_count = label_info["expected_count"]
                protected = label_info.get("protected_tokens", [])
                
                count_ok = (len(parts) == exp_count)
                tokens_ok = all(any(tok in p for p in parts) for tok in protected)
                is_success = count_ok and tokens_ok
                detail = f"count={len(parts)}/{exp_count}, parts={parts}"

            elif test_type == "dehyphenate":
                cleaned = clean_and_dehyphenate_text(text)
                exp_text = label_info["expected_text"]
                is_success = (cleaned == exp_text)
                detail = f"got='{cleaned}', exp='{exp_text}'"

            elif test_type in ["doc_identifier", "document_identifier", "legal_ref"]:
                import re
                doc_id = extract_document_identifier("", text)
                exp_id = label_info["expected_identifier"]
                clean_doc = re.sub(r"\bsố\s+", "", doc_id, flags=re.IGNORECASE).strip().lower()
                clean_exp = re.sub(r"\bsố\s+", "", exp_id, flags=re.IGNORECASE).strip().lower()
                is_success = (clean_exp in clean_doc) or (clean_doc in clean_exp)
                detail = f"got='{doc_id}', exp='{exp_id}'"

            elif test_type == "hierarchical_unit":
                unit = extract_hierarchical_units(text)
                exp_unit = label_info["expected_unit"]
                is_success = (unit == exp_unit)
                detail = f"got='{unit}', exp='{exp_unit}'"

            elif test_type == "vietnamese_quality":
                valid = is_valid_vietnamese_text(text)
                exp_valid = label_info["expected_valid"]
                is_success = (valid == exp_valid)
                detail = f"got={valid}, exp={exp_valid}"

            results_list.append({
                "sample": fname,
                "test_type": test_type,
                "detail": detail,
                "input_preview": text[:100] + "..." if len(text) > 100 else text,
                "status": "PASS" if is_success else "FAIL"
            })

            if is_success:
                passed += 1
            else:
                failed_cases.append({
                    "file": fname,
                    "type": test_type,
                    "detail": detail
                })

        accuracy = (passed / total) * 100.0

        # Xuất kết quả chi tiết ra test_results.json
        out_json_path = CURRENT_DIR / "test_results.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "node_id": "Node 06",
                "node_name": "Canonical Entity & Legal Resolver",
                "timestamp": datetime.now().isoformat(),
                "total_samples": total,
                "passed": passed,
                "failed": len(failed_cases),
                "accuracy_percent": round(accuracy, 2),
                "status": "PASS" if len(failed_cases) == 0 else "FAIL",
                "results": results_list
            }, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print(f"[*] KẾT QUẢ TEST NODE 06: Canonical Entity & Legal Resolver")
        print(f"    - Tổng số mẫu NLP kiểm thử: {total}")
        print(f"    - Đạt chuẩn (PASS)         : {passed}/{total}")
        print(f"    - Độ chính xác             : {accuracy:.2f}%")
        if failed_cases:
            print("    - Các ca lỗi:")
            for fc in failed_cases[:5]:
                print(f"      + {fc['file']} ({fc['type']}): {fc['detail']}")
        print("=" * 70)

        self.assertEqual(passed, total, f"Có {len(failed_cases)} mẫu không đạt kết quả kỳ vọng!")


if __name__ == "__main__":
    unittest.main()
