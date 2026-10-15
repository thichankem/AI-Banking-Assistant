"""
Helper script to assemble exactly 50 balanced Golden QA samples across 4 categories.
"""
import json
import sys
from collections import Counter
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_banking.chunking import LayoutAwareChunker

chunker = LayoutAwareChunker()
qa_p = Path(__file__).parent / "golden_qa.json"
with open(qa_p, "r", encoding="utf-8") as f:
    existing_qa = json.load(f)

doc_scan1 = chunker.chunk_file(PROJECT_ROOT / "dataset/chunking_dataset/pdf/vcb_pdf_21_nq_hdqt_cctg_vcbs.pdf")
doc_scan2 = chunker.chunk_file(PROJECT_ROOT / "dataset/chunking_dataset/pdf/vcb_pdf_20_cbtt_trai_phieu_rieng_le_2025.pdf", max_pdf_pages=4)
doc_scan = doc_scan1 + doc_scan2
doc_faq = chunker.chunk_file(PROJECT_ROOT / "dataset/chunking_dataset/docx/vcb_md_01__faq__the_tin_dung_va_the_ghi_no_ngan_hang.docx")
doc_terms = chunker.chunk_file(PROJECT_ROOT / "dataset/chunking_dataset/pdf/vcb_pdf_27_dieu_kien_dieu_khoan_tktt_truc_tuyen.pdf", max_pdf_pages=8)

by_cat = {
    "multi_level_legal": [],
    "complex_table": [],
    "noisy_scan_ocr": [],
    "cross_page_defragmented": []
}

from rag_banking.chunking.pdf_chunker import PDFChunker
pdf_legacy_chunker = PDFChunker(use_unified_stream=False)
doc_tables = pdf_legacy_chunker.chunk_pdf(PROJECT_ROOT / "dataset/chunking_dataset/pdf/vcb_pdf_22_bieu_phi_the_tdqt_2026.pdf", max_pages=6)

for chk in doc_tables:
    if len(by_cat["complex_table"]) >= 13:
        break
    if chk.get("chunk_type") == "pdf_table":
        lines = [l.strip() for l in chk["content"].split("\n") if "* Hàng" in l or ("|" in l and len(l) > 20)]
        for l in lines:
            if len(by_cat["complex_table"]) >= 13:
                break
            q_text = f"Mức biểu phí thẻ tín dụng quốc tế quy định cho hạng mục: '{l[:50]}...' là bao nhiêu?"
            by_cat["complex_table"].append({
                "question": q_text,
                "ground_truth_answer": l,
                "ground_truth_context_span": l,
                "expected_chunk_ids": [chk["chunk_id"]],
                "expected_breadcrumb_prefix": chk["breadcrumb"].split(" > ")[0],
                "category": "complex_table"
            })

# Bổ sung Category 3: Red-stamp / Noisy Scan OCR
for chk in doc_scan:
    if len(by_cat["noisy_scan_ocr"]) >= 12:
        break
    lines = [l.strip() for l in chk["content"].split("\n") if len(l.strip()) > 30 and not l.strip().startswith("[")]
    for l in lines:
        if len(by_cat["noisy_scan_ocr"]) >= 12:
            break
        q_text = f"Nghị quyết HĐQT phê duyệt nội dung chi tiết: '{l[:50]}...' như thế nào?"
        by_cat["noisy_scan_ocr"].append({
            "question": q_text,
            "ground_truth_answer": l,
            "ground_truth_context_span": l,
            "expected_chunk_ids": [chk["chunk_id"]],
            "expected_breadcrumb_prefix": chk["breadcrumb"].split(" > ")[0],
            "category": "noisy_scan_ocr"
        })

# Bổ sung Category 4: Cross-page & Short Defrag
for chk in doc_faq + doc_terms:
    if len(by_cat["cross_page_defragmented"]) >= 12:
        break
    lines = [l.strip() for l in chk["content"].split("\n") if len(l.strip()) > 30 and not l.strip().startswith("[")]
    for l in lines:
        if len(by_cat["cross_page_defragmented"]) >= 12:
            break
        if any(kw in l.lower() for kw in ["sau khi", "trường hợp", "thời gian", "khiếu nại", "kích hoạt", "phí"]):
            q_text = f"Theo hướng dẫn quy chuẩn dịch vụ ngân hàng, nội dung: '{l[:50]}...' được quy định ra sao?"
            by_cat["cross_page_defragmented"].append({
                "question": q_text,
                "ground_truth_answer": l,
                "ground_truth_context_span": l,
                "expected_chunk_ids": [chk["chunk_id"]],
                "expected_breadcrumb_prefix": chk["breadcrumb"].split(" > ")[0],
                "category": "cross_page_defragmented"
            })

# Bổ sung Category 1: Multi-level Legal
for chk in doc_terms:
    if len(by_cat["multi_level_legal"]) >= 13:
        break
    lines = [l.strip() for l in chk["content"].split("\n") if len(l.strip()) > 30 and not l.strip().startswith("[")]
    for l in lines:
        if len(by_cat["multi_level_legal"]) >= 13:
            break
        q_text = f"Quy định chi tiết về quyền và nghĩa vụ liên quan đến: '{l[:50]}...' được áp dụng thế nào?"
        by_cat["multi_level_legal"].append({
            "question": q_text,
            "ground_truth_answer": l,
            "ground_truth_context_span": l,
            "expected_chunk_ids": [chk["chunk_id"]],
            "expected_breadcrumb_prefix": chk["breadcrumb"].split(" > ")[0],
            "category": "multi_level_legal"
        })

final_50 = []
for c_name in ["multi_level_legal", "complex_table", "noisy_scan_ocr", "cross_page_defragmented"]:
    limit = 13 if c_name in ("multi_level_legal", "complex_table") else 12
    final_50.extend(by_cat[c_name][:limit])

with open(qa_p, "w", encoding="utf-8") as f:
    json.dump(final_50, f, ensure_ascii=False, indent=2)

print("SUCCESS: Exactly 50 Golden QA samples generated!")
print("Distribution:", Counter(item["category"] for item in final_50))
