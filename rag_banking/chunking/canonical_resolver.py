"""
Canonical Entity & Legal Structure Resolver
Module phân giải định danh văn bản và cấu trúc pháp lý ngân hàng tổng quát hóa (Non-Overfitting):
1. Document Identifier: Bóc tách tên/số hiệu văn bản động từ Breadcrumbs Root hoặc regex thể thức quốc gia.
2. Hierarchical Units: Bóc tách thứ bậc: Chương -> Mục -> Điều -> Khoản -> Điểm -> Câu hỏi FAQ.
3. Cross Statutory Citations: Quét các văn bản viện dẫn chéo (Quyết định NHNN, Thông tư, Nghị định...).
4. Advanced Sentence Splitting: Tách câu nhận thức ngữ cảnh ngân hàng, bảo vệ Điều 12., TP.HCM, 1.000.000 VND, 5,5%/năm.
5. Dehyphenation & Line-merge: Nối từ bị ngắt dòng và nối câu bị gãy dòng cứng.
6. Vietnamese Text Quality Checker: Kiểm tra tỷ lệ ký tự tiếng Việt hợp lệ để phát hiện lỗi font VNI/TCVN3 hoặc text scan rác.
7. Header/Footer Remover: Tự động phát hiện và loại bỏ header/footer lặp lại qua nhiều trang.
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple


# =============================================================================
# CÁC BIỂU THỨC CHÍNH QUY (PRE-COMPILED REGEX CONSTANTS)
# =============================================================================

# Bắt số hiệu văn bản quy phạm pháp luật (theo thể thức Nghị định 30/2020/NĐ-CP)
DOC_NUMBER_PATTERN = re.compile(
    r"\b((?:Luật|Nghị quyết|Quyết định|Thông tư|Nghị định|Quy chế|Quy định)\s+(?:số\s+)?\d+(?:/[A-ZĐ0-9\-_]+)+)\b",
    re.IGNORECASE
)

# Bắt các viện dẫn chéo văn bản quy phạm pháp luật
CROSS_CITATION_PATTERN = re.compile(
    r"\b((?:Quyết định|Thông tư|Nghị định|Nghị quyết|Luật)\s+(?:số\s+)?\d+(?:/[A-ZĐ0-9\-_]+)+)\b",
    re.IGNORECASE
)

# Bắt các cấp bậc phân cấp trong văn bản ngân hàng
DIEU_PATTERN = re.compile(r"\bĐiều\s+(\d+[a-zA-Z]?)\b", re.IGNORECASE)
KHOAN_WORD_PATTERN = re.compile(r"\bKhoản\s+(\d+)\b", re.IGNORECASE)
KHOAN_DIGIT_PATTERN = re.compile(r"(?:^|\n)\s*(\d{1,2})\.\s+[A-ZÀ-Ỹ]")
DIEM_WORD_PATTERN = re.compile(r"\bĐiểm\s+([a-zđ])\b", re.IGNORECASE)
DIEM_CHAR_PATTERN = re.compile(r"(?:^|\n)\s*([a-zđ])\)\s+[A-ZÀ-Ỹa-zà-ỹ]")
CHUONG_PATTERN = re.compile(r"\b(?:Chương|Phần)\s+([IVXLCDM\d]+)\b", re.IGNORECASE)
MUC_PATTERN = re.compile(r"\b(?:Tiểu\s+mục|Mục)\s+([IVXLCDM\d]+|\d+)\b", re.IGNORECASE)
FAQ_PATTERN = re.compile(r"\b(?:Câu|Câu\s+hỏi)\s+(\d+)\b", re.IGNORECASE)

# Bắt số trang dạng 'Trang 1/10' hoặc '1' đứng riêng một mình
PAGE_NUMBER_PATTERN = re.compile(r"^\s*(?:Trang\s+)?\d+(?:/\d+)?\s*$", re.IGNORECASE)

# Tập hợp nguyên âm tiếng Việt chuẩn Unicode
VIETNAMESE_VOWELS = set("aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ")


# =============================================================================
# HÀM TIỆN ÍCH CƠ BẢN (STRING UTILITIES)
# =============================================================================

def strip_vietnamese_accents(text: str) -> str:
    """
    Chuyển đổi văn bản tiếng Việt có dấu thành không dấu (chuẩn Unicode NFKD).
    Hữu ích cho fuzzy matching và đối chiếu Mục lục chống lỗi OCR.
    """
    if not text:
        return ""
    text_clean = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text_clean)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def count_syllables_vietnamese(text: str) -> int:
    """Đếm số âm tiết tiếng Việt (tương đương số từ có ký tự chữ)."""
    if not text:
        return 0
    words = re.findall(r"\b\w+\b", text)
    return len(words)


def count_words(text: str) -> int:
    """Đếm số từ phân tách bằng khoảng trắng."""
    if not text:
        return 0
    return len(text.split())


# =============================================================================
# ĐỊNH DANH VĂN BẢN & PHÂN CẤP PHÁP LÝ (CANONICAL RESOLUTION)
# =============================================================================

def extract_document_identifier(
    breadcrumb_text: str,
    full_context: str = "",
    source_name: str = ""
) -> str:
    """
    Trích xuất tên hoặc số hiệu văn bản định danh một cách tổng quát:
    Tự động nhận diện từ Số hiệu thể thức, Breadcrumb Root, hoặc Tên tệp nguồn.
    Tuyệt đối không hardcode tên văn bản cụ thể.
    """
    # 1. Bắt số hiệu văn bản chính thức xuất hiện ở đầu văn bản (Nghị định 30/2020)
    match_code = DOC_NUMBER_PATTERN.search(breadcrumb_text or full_context[:200])
    if match_code:
        return match_code.group(1).strip()

    # 2. Lấy phần tử gốc đầu tiên của Breadcrumb (Root Node)
    if breadcrumb_text:
        root_part = breadcrumb_text.split(" > ")[0].strip()
        clean_root = re.sub(r"^\[.*?\]\s*", "", root_part)
        clean_root = re.sub(r"^(?:vcb_)?(?:md|pdf|docx|excel)_\d+__?", "", clean_root, flags=re.IGNORECASE)
        clean_root = clean_root.replace("_", " ").strip()
        if len(clean_root) > 45:
            split_title = re.split(r"[-–—:]", clean_root)[0].strip()
            if len(split_title) >= 8:
                clean_root = split_title
        if clean_root:
            return clean_root

    # 3. Fallback: Chuẩn hóa từ tên tệp tin nguồn
    if source_name:
        stem = Path(source_name).stem
        stem = re.sub(r"^(?:vcb_)?(?:md|pdf|docx|excel)_\d+__?", "", stem, flags=re.IGNORECASE)
        stem = stem.replace("_", " ").strip().title()
        if stem:
            return stem

    return ""


def extract_hierarchical_units(full_context: str, text_chunk: str = "") -> str:
    """
    Bóc tách các đơn vị phân cấp chuẩn: Chương > Mục > Điều > Khoản > Điểm > FAQ.
    """
    context = full_context or text_chunk
    dieu_match = DIEU_PATTERN.search(context)

    # Nhận diện Khoản
    khoan_match = KHOAN_WORD_PATTERN.search(context)
    if not khoan_match and dieu_match:
        khoan_match = KHOAN_DIGIT_PATTERN.search(text_chunk or full_context)

    # Nhận diện Điểm
    diem_match = DIEM_WORD_PATTERN.search(context)
    if not diem_match and khoan_match:
        diem_match = DIEM_CHAR_PATTERN.search(text_chunk or full_context)

    # Nhận diện Chương / Mục / FAQ
    chuong_match = CHUONG_PATTERN.search(context)
    muc_match = MUC_PATTERN.search(context)
    faq_match = FAQ_PATTERN.search(context)

    # Ghép chuỗi phân cấp theo thứ tự ưu tiên
    if dieu_match:
        res = f"Điều {dieu_match.group(1)}"
        if khoan_match:
            res += f", Khoản {khoan_match.group(1)}"
            if diem_match:
                res += f", Điểm {diem_match.group(1)}"
        return res
    if faq_match:
        return f"Câu {faq_match.group(1)}"
    if muc_match:
        res = f"Mục {muc_match.group(1)}"
        if khoan_match:
            res += f", Khoản {khoan_match.group(1)}"
        return res
    if chuong_match:
        return f"Chương {chuong_match.group(1)}"
    if khoan_match:
        return f"Khoản {khoan_match.group(1)}"

    return ""


def extract_cross_statutory_citations(full_context: str, current_doc: str = "") -> List[str]:
    """Quét và trích xuất mọi văn bản pháp quy được viện dẫn chéo trong nội dung."""
    matches = CROSS_CITATION_PATTERN.findall(full_context)
    citations: List[str] = []
    current_lower = current_doc.lower()

    for m in matches:
        clean_m = m.strip()
        lower_m = clean_m.lower()
        if lower_m not in [c.lower() for c in citations] and lower_m not in current_lower:
            citations.append(clean_m)

    return citations


def extract_canonical_legal_ref(
    breadcrumb_text: str,
    text_chunk: str,
    source_name: str = ""
) -> str:
    """
    Trích xuất tham chiếu pháp lý chuẩn tắc: [Tên/Số hiệu văn bản - Điều X, Khoản Y; Viện dẫn chéo].
    """
    full_context = f"{breadcrumb_text} {text_chunk}"
    doc_identifier = extract_document_identifier(breadcrumb_text, full_context, source_name)
    hierarchical_unit = extract_hierarchical_units(full_context, text_chunk)
    cross_citations = extract_cross_statutory_citations(full_context, doc_identifier)

    parts: List[str] = []
    if doc_identifier and hierarchical_unit:
        parts.append(f"{doc_identifier} - {hierarchical_unit}")
    elif hierarchical_unit:
        parts.append(hierarchical_unit)
    elif doc_identifier:
        parts.append(doc_identifier)

    for c in cross_citations:
        if c not in parts:
            parts.append(c)

    return f"[{'; '.join(parts)}]" if parts else ""


# =============================================================================
# HẬU XỬ LÝ VĂN BẢN VÀ TÁCH CÂU (TEXT PROCESSING & SPLITTING)
# =============================================================================

def split_sentences_banking(text: str) -> List[str]:
    """
    Tách câu nhận thức ngữ cảnh ngân hàng và pháp lý:
    Bảo vệ an toàn: Điều 12., TP.HCM, 1.000.000 VND, 5,5%/năm, v.v.
    """
    if not text:
        return []

    text_norm = unicodedata.normalize("NFC", text)
    protected_tokens: Dict[str, str] = {}
    counter = 0

    def _replace_with_token(match: re.Match) -> str:
        nonlocal counter
        token = f"__TOKEN_GUARD_{counter}__"
        protected_tokens[token] = match.group(0)
        counter += 1
        return token

    # 1. Bảo vệ số tiền, tỷ lệ phần trăm (1.000.000 VND, 5,5%/năm)
    temp = re.sub(r"\b\d+(?:[\.,]\d+)+(?:%|\s*(?:VND|VNĐ|USD|EUR))?", _replace_with_token, text_norm)

    # 2. Bảo vệ tiêu đề pháp lý (Điều 12., Khoản 1., Mục 2.)
    temp = re.sub(
        r"\b(?:Điều|Khoản|Mục|Tiểu\s+mục|Chương|Phần|Câu)\s+\d+[a-zA-Z]?\.",
        _replace_with_token,
        temp,
        flags=re.IGNORECASE
    )

    # 3. Bảo vệ từ viết tắt có dấu chấm (TP.HCM, v.v., TNHH., ThS.)
    temp = re.sub(
        r"\b(?:TP\.HCM|T\.P|v\.v\.|ThS\.|TS\.|PGS\.|GS\.|TNHH\.|CP\.)",
        _replace_with_token,
        temp,
        flags=re.IGNORECASE
    )

    # 4. Tách câu sau dấu kết thúc [.?!] kèm khoảng trắng và chữ hoa [A-ZÀ-Ỹ]
    raw_parts = re.split(r"(?<=[.?!])\s+(?=[A-ZÀ-Ỹ])|(?:\n{2,})", temp)

    # 5. Phục hồi các token đã được bảo vệ
    results: List[str] = []
    for part in raw_parts:
        clean_part = part.strip()
        if not clean_part:
            continue
        for token, original in protected_tokens.items():
            clean_part = clean_part.replace(token, original)
        results.append(clean_part)

    return results if results else [text]


def group_sentences_banking(sentences: List[str], max_size: int, overlap: int = 100) -> List[str]:
    """
    Gom nhóm các câu an toàn vào chunk có độ dài phù hợp (không cắt giữa chừng câu).
    Dùng chung cho cả PDF Chunker và DOCX Chunker.
    """
    if not sentences:
        return [""]

    parts: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue

        if len(current_chunk) + len(s) + 1 <= max_size:
            current_chunk = f"{current_chunk} {s}".strip() if current_chunk else s
        else:
            if current_chunk:
                parts.append(current_chunk)
            current_chunk = s

    if current_chunk:
        parts.append(current_chunk)

    return parts or (sentences if sentences else [""])


def clean_and_dehyphenate_text(text: str) -> str:
    """
    Khử ngắt dòng gạch nối (ngân-\nhàng -> ngân hàng), nối dòng gãy cứng giữa câu,
    và làm sạch khoảng trắng dư thừa.
    """
    if not text:
        return ""

    text_norm = unicodedata.normalize("NFC", text)

    # 1. Khử gạch nối ở cuối dòng
    text_norm = re.sub(r"(\b\w+)[-–—]\n\s*(\w+\b)", r"\1\2", text_norm)

    # 2. Nối dòng gãy cứng giữa câu
    lines = text_norm.split("\n")
    merged_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged_lines.append("")
            continue

        if merged_lines and merged_lines[-1] != "":
            prev = merged_lines[-1]
            if not prev.endswith((".", ":", ";", "!", "?", "---", "***")) and not prev.startswith(("#", "-", "*", "|", "•")):
                if stripped[0].islower() or (len(stripped) > 1 and stripped[0].isalnum() and not stripped.startswith(("Điều", "Khoản", "Chương"))):
                    merged_lines[-1] = f"{prev} {stripped}"
                    continue

        merged_lines.append(stripped)

    # 3. Loại bỏ khoảng trắng thừa
    res = "\n".join(merged_lines)
    res = re.sub(r"[ \t]+", " ", res)
    res = re.sub(r"\n{3,}", "\n\n", res)
    return res.strip()


def is_valid_vietnamese_text(text: str, min_valid_ratio: float = 0.25) -> bool:
    """
    Kiểm tra chất lượng văn bản:
    Phát hiện văn bản scan rác hoặc lỗi font VNI/TCVN3 dựa trên tỷ lệ từ có nguyên âm tiếng Việt.
    """
    if not text or len(text.strip()) < 10:
        return False

    text_lower = text.lower()
    # Kiểm tra ký tự lỗi unicode không in được
    weird_chars = len(re.findall(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", text))
    if weird_chars / max(len(text), 1) > 0.05:
        return False

    words = re.findall(r"\b\w+\b", text_lower)
    if not words:
        return False

    # Đếm số từ có chứa ít nhất 1 nguyên âm
    words_with_vowels = sum(1 for w in words if any(c in VIETNAMESE_VOWELS for c in w))
    ratio = words_with_vowels / len(words)

    return ratio >= min_valid_ratio


def detect_and_remove_headers_footers(pages_text: List[str]) -> List[str]:
    """
    Tự động phát hiện và loại bỏ Header/Footer lặp lại và số trang qua nhiều trang.
    """
    if len(pages_text) < 3:
        # Nếu ít hơn 3 trang, chỉ loại bỏ số trang đơn thuần
        cleaned: List[str] = []
        for p in pages_text:
            lines = [l for l in p.split("\n") if not PAGE_NUMBER_PATTERN.match(l.strip())]
            cleaned.append("\n".join(lines))
        return cleaned

    top_freq: Dict[str, int] = {}
    bottom_freq: Dict[str, int] = {}

    for p in pages_text:
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        if not lines:
            continue
        for l in lines[:2]:
            if len(l) < 100 and not l.startswith(("#", "Điều", "CHƯƠNG")):
                top_freq[l] = top_freq.get(l, 0) + 1
        for l in lines[-2:]:
            if len(l) < 100:
                bottom_freq[l] = bottom_freq.get(l, 0) + 1

    threshold = max(3, int(len(pages_text) * 0.4))
    repeated_headers = {l for l, c in top_freq.items() if c >= threshold}
    repeated_footers = {l for l, c in bottom_freq.items() if c >= threshold}

    cleaned_pages: List[str] = []
    for p in pages_text:
        lines = p.split("\n")
        filtered: List[str] = []
        for l in lines:
            s = l.strip()
            if PAGE_NUMBER_PATTERN.match(s):
                continue
            if s in repeated_headers or s in repeated_footers:
                continue
            filtered.append(l)
        cleaned_pages.append("\n".join(filtered))

    return cleaned_pages
