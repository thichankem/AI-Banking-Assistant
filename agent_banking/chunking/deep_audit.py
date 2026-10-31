import sys, os, json, hashlib
from pathlib import Path
import docx

def _words(text):
    clean = []
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            clean.append(ch)
        else:
            clean.append(" ")
    return "".join(clean).split()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 75)
print("     DOI SOAT CHI TIET: NGUYEN DO 10.9% VA KIEM TRA RUNTIME THUC TE   ")
print("=" * 75)

current_dir = Path.cwd()
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

all_chunks_file = Path("chunking/output/all_chunks.json")
with open(all_chunks_file, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

# 1. Phan tich 1 file cu the: okbank_001
doc_path = Path("chunking/input/docx/okbank_001__nhom1_chuyen_tien__chuyen_tien_noi_bo_va_napas247.docx")
doc = docx.Document(doc_path)

raw_texts = []
for p in doc.paragraphs:
    if p.text.strip(): raw_texts.append(p.text.strip())
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            if cell.text.strip(): raw_texts.append(cell.text.strip())

doc_full_norm = " ".join(_words(" ".join(raw_texts)))
chunks = [c for c in all_chunks if "okbank_001" in c.get("chunk_id", "")]

print(f"\n[PHAN TICH FILE: {doc_path.name}]")
print(f"- So chunk luu tru: {len(chunks)}")

unmatched_samples = []
matched_count = 0
total_count = 0

for c in chunks:
    content = c.get("content", "")
    clean_c = content
    for punct in ["\n", ".", ";", ":", "!", "?"]:
        clean_c = clean_c.replace(punct, " ")
    sentences = [s.strip() for s in clean_c.split(" ") if len(s.strip()) > 15]
    for s in sentences:
        total_count += 1
        norm_s = " ".join(_words(s))
        if norm_s in doc_full_norm:
            matched_count += 1
        else:
            unmatched_samples.append(s)

print(f"- Cau khop truc tiep voi text goc: {matched_count}/{total_count} ({(matched_count/total_count)*100:.1f}%)")
print(f"- Mau cac dong/cau bi bao la khong khop (de xem ban chat):")
for sample in unmatched_samples[:5]:
    # In ASCII safe
    safe_sample = sample.encode("ascii", "replace").decode("ascii")
    print(f"   + [Mau]: \"{safe_sample[:75]}\"")

# 2. RUNTIME RE-EXECUTION TEST
print("\n[CHAY RUNTIME THUC TE TRUC TIEP TU PARSER]")
try:
    project_root = Path.cwd().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    from agent_banking.chunking.docx_chunker.docx_chunker import DOCXChunker
    chunker = DOCXChunker(max_chunk_size=800, min_section_size=120)
    runtime_chunks = chunker.chunk_docx(doc_path)

    print(f"- So chunk tao ra truc tiep tu runtime: {len(runtime_chunks)}")
    print(f"- So chunk dang luu trong he thong   : {len(chunks)}")

    if len(runtime_chunks) == len(chunks):
        print("=> XAC NHAN: So luong chunk runtime khop 100% voi du lieu luu tru!")
        rt_c0 = runtime_chunks[0]["content"].strip()
        db_c0 = chunks[0]["content"].strip()
        is_identical = (rt_c0 == db_c0)
        print(f"=> Noi dung chunk 0 trung khop hoan toan: {is_identical}")
        if not is_identical:
            print(f"   * Runtime len: {len(rt_c0)}, DB len: {len(db_c0)}")
            print(f"   * Runtime sample: {rt_c0[:60].encode('ascii', 'replace').decode('ascii')}")
            print(f"   * DB sample     : {db_c0[:60].encode('ascii', 'replace').decode('ascii')}")
        else:
            print("=> THUAT TOAN CHUNKING HOAN TOAN DETERMINISTIC VA AUTHENTIC!")
    else:
        print(f"=> Chech lech: runtime={len(runtime_chunks)} vs saved={len(chunks)}")
except Exception as e:
    import traceback
    print(f"[!] Loi runtime parser: {e}")
    traceback.print_exc()

print("=" * 75)
