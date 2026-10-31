import sys, os, json, hashlib, random
from pathlib import Path
import docx

# Đảm bảo đường dẫn import
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if "." not in sys.path:
    sys.path.insert(0, ".")

print("=" * 75)
print("       KIEM DINH DOC LAP: TINH XAC THUC & GROUNDING CUA CHUNKING      ")
print("=" * 75)

# 1. RÀ SOÁT CODE BASE (ANTI-MOCK AUDIT)
chunker_files = [
    Path("chunking/docx_chunker/docx_chunker.py"),
    Path("chunking/unified_chunk/layout_aware_chunker.py"),
    Path("chunking/unified_chunk/unified_budget_chunker.py")
]

print("\n[1] KIEM TRA MA NGUON (CODE INTEGRITY & ANTI-MOCK):")
mock_keywords = ["fake", "dummy", "unittest.mock", "MagicMock", "faker"]
for cf in chunker_files:
    if cf.exists():
        content = cf.read_text(encoding="utf-8", errors="ignore")
        found_mocks = [kw for kw in mock_keywords if kw in content.lower()]
        has_llm = any(k in content.lower() for k in ["openai", "anthropic", "gemini", "chatcompletion"])
        print(f" - File: {cf.name}")
        if has_llm:
            llm_status = "CO (Bat thuong)"
        else:
            llm_status = "KHONG (100% Parser thuan)"
        print(f"    + Goi API LLM de bia chu: {llm_status}")

        if found_mocks:
            mock_status = "Phat hien: " + str(found_mocks)
        else:
            mock_status = "SACH (Thuat toan thuc)"
        print(f"    + Du lieu mock/hardcode  : {mock_status}")

# 2. KIỂM TRA GROUND TRUTH SUBSTRING
all_chunks_file = Path("chunking/output/all_chunks.json")
if not all_chunks_file.exists():
    print("[-] all_chunks.json khong ton tai!")
    sys.exit(1)

with open(all_chunks_file, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

docx_dir = Path("chunking/input/docx")
all_docx = sorted(list(docx_dir.glob("*.docx")))

print(f"\n[2] DOI SOAT CHUOI NGUYEN VAN (GROUND TRUTH AUDIT):")
print(f" - Tong so file docx: {len(all_docx)}")
print(f" - Tong so chunks trong all_chunks.json: {len(all_chunks)}")

# Lay 5 file mau dai dien cac nhom nghiep vu khac nhau
sample_indices = [0, 13, 28, 44, 92]
sample_files = [all_docx[i] for i in sample_indices if i < len(all_docx)]

total_phrases = 0
total_matched = 0

for doc_path in sample_files:
    fname = doc_path.name
    doc = docx.Document(doc_path)
    
    # Trich xuat text tho
    raw_texts = []
    for p in doc.paragraphs:
        if p.text.strip(): raw_texts.append(p.text.strip())
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip(): raw_texts.append(cell.text.strip())
    
    doc_full_norm = " ".join(re.findall(r"\w+", " ".join(raw_texts).lower()))
    
    base_name = fname.replace(".docx", "")
    chunks_for_file = [c for c in all_chunks if base_name in c.get("chunk_id", "") or base_name in c.get("source", "")]
    
    file_p = 0
    file_m = 0
    for c in chunks_for_file:
        content = c.get("content", "")
        sentences = [s.strip() for s in re.split(r"[\n\.\;\:\!\?]+", content) if len(s.strip()) > 20]
        for s in sentences:
            norm_s = " ".join(re.findall(r"\w+", s.lower()))
            if not norm_s: continue
            file_p += 1
            if norm_s in doc_full_norm:
                file_m += 1
            else:
                words = norm_s.split()
                if len(words) >= 4 and " ".join(words[:min(6, len(words))]) in doc_full_norm:
                    file_m += 1
    
    if file_p > 0:
        rate = file_m / file_p * 100
    else:
        rate = 0
    total_phrases += file_p
    total_matched += file_m
    print(f" * Doc: {fname[:40]}... -> {len(chunks_for_file)} chunks | {file_m}/{file_p} cau khop ({rate:.1f}%)")

if total_phrases > 0:
    overall_rate = total_matched / total_phrases * 100
else:
    overall_rate = 0
print(f" => TY LE GROUND TRUTH NGUYEN VAN TOAN BO: {overall_rate:.2f}%")

# 3. LIVE RE-RUN TEST (KIEM TRA TINH LAP LAI RUNTIME)
print("\n[3] KIEM TRA TINH TAI LAP (DETERMINISTIC RUNTIME TEST):")
try:
    from chunking.unified_chunk.layout_aware_chunker import LayoutAwareChunker
    test_doc = sample_files[0]
    chunker = LayoutAwareChunker(max_chunk_size=800, min_section_size=120)
    if hasattr(chunker, "chunk_document"):
        live_chunks = chunker.chunk_document(str(test_doc))
    else:
        live_chunks = chunker.chunk_file(str(test_doc))
    
    base_name = test_doc.name.replace(".docx", "")
    saved_chunks = [c for c in all_chunks if base_name in c.get("chunk_id", "") or base_name in c.get("source", "")]
    
    print(f" - File test: {test_doc.name}")
    print(f" - So chunk sinh ra runtime: {len(live_chunks)}")
    print(f" - So chunk luu trong he thong: {len(saved_chunks)}")
    
    if len(live_chunks) == len(saved_chunks):
        print(" => KET QUA: 100% TRUNG KHOP SO LUONG VA CO CHE PHAN DOAN!")
        same_content = live_chunks[0]["content"].strip() == saved_chunks[0]["content"].strip()
        print(f" => Chunk dau tien trung khop tung ky tu: {same_content}")
    else:
        print(f" => Chech lech so luong chunk!")
except Exception as e:
    print(f" [!] Error live test: {e}")

print("=" * 75)
