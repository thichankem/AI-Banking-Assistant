"""
Module Canonical Entity & Legal Structure Resolver
===================================================
Module xử lý ngôn ngữ tự nhiên (NLP) và bóc tách cấu trúc pháp lý chuyên biệt
cho văn bản ngân hàng và tài chính Việt Nam (Vietcombank RAG Assistant).

Các nhóm chức năng chính:
1. Chuẩn hóa chuỗi ký tự & Âm tiết:
   - `strip_vietnamese_accents()`: Khử dấu tiếng Việt (NFKD) chống lỗi OCR.
   - `count_words()`, `count_syllables_vietnamese()`: Đếm từ và số âm tiết tiếng Việt.
   - `clean_and_dehyphenate_text()`: Khử gạch nối ngắt dòng, nối câu gãy dòng cứng.

2. Định danh văn bản & Cấu trúc pháp lý (Nghị định 30/2020/NĐ-CP):
   - `extract_document_identifier()`: Bóc tách số hiệu văn bản (Thông tư, Nghị định, Quyết định).
   - `extract_hierarchical_units()`: Bóc tách đơn vị thứ bậc (Chương, Mục, Điều, Khoản, Điểm, FAQ).

3. Tách câu an toàn nhận thức ngữ cảnh ngân hàng:
   - `split_sentences_banking()`: Bảo vệ số tiền, lãi suất, ký hiệu Điều/Khoản, từ viết tắt TP.HCM, v.v.
   - `group_sentences_banking()`: Gom cụm câu theo kích thước trần max_chunk_size có overlap.

4. Kiểm định chất lượng văn bản & Khử Header/Footer:
   - `is_valid_vietnamese_text()`: Lọc bỏ văn bản scan rác hoặc lỗi font VNI/TCVN3 bằng âm vị học.
   - `detect_and_remove_headers_footers()`: Tự động phát hiện và loại bỏ header/footer lặp qua trang.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple


# =============================================================================
# 1. BIỂU THỨC CHÍNH QUY CHUẨN HÓA (PRE-COMPILED REGEX CONSTANTS)
# =============================================================================

# Thể thức số hiệu văn bản quy phạm pháp luật theo Nghị định 30/2020/NĐ-CP
# Ví dụ: "Thông tư số 17/2024/TT-NHNN", "Nghị định 52/2024/NĐ-CP", "Quyết định 2345/QĐ-NHNN"
DOC_NUMBER_PATTERN: re.Pattern = re.compile(
    r"\b((?:Luật|Nghị quyết|Quyết định|Thông tư|Nghị định|Quy chế|Quy định)\s+(?:số\s+)?\d+(?:/[A-ZĐ0-9\-_]+)+)\b",
    re.IGNORECASE
)

# Nhận diện thứ bậc pháp lý trong văn bản
DIEU_PATTERN: re.Pattern = re.compile(r"\bĐiều\s+(\d+[a-zA-Z]?)\b", re.IGNORECASE)
KHOAN_WORD_PATTERN: re.Pattern = re.compile(r"\bKhoản\s+(\d+)\b", re.IGNORECASE)
KHOAN_DIGIT_PATTERN: re.Pattern = re.compile(r"(?:^|\n)\s*(\d{1,2})\.\s+[A-ZÀ-Ỹ]")
DIEM_WORD_PATTERN: re.Pattern = re.compile(r"\bĐiểm\s+([a-zđ])\b", re.IGNORECASE)
DIEM_CHAR_PATTERN: re.Pattern = re.compile(r"(?:^|\n)\s*([a-zđ])\)\s+[A-ZÀ-Ỹa-zà-ỹ]")
CHUONG_PATTERN: re.Pattern = re.compile(r"\b(?:Chương|Phần)\s+([IVXLCDM\d]+)\b", re.IGNORECASE)
MUC_PATTERN: re.Pattern = re.compile(r"\b(?:Tiểu\s+mục|Mục)\s+([IVXLCDM\d]+|\d+)\b", re.IGNORECASE)
FAQ_PATTERN: re.Pattern = re.compile(r"\b(?:Câu|Câu\s+hỏi)\s+(\d+)\b", re.IGNORECASE)

# Bắt số trang dạng 'Trang 1/10' hoặc '1' đứng riêng lẻ
PAGE_NUMBER_PATTERN: re.Pattern = re.compile(r"^\s*(?:Trang\s+)?\d+(?:/\d+)?\s*$", re.IGNORECASE)

# Bảng nguyên âm tiếng Việt chuẩn Unicode phục vụ kiểm tra ngôn ngữ
VIETNAMESE_VOWELS: Set[str] = set("aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ")


# =============================================================================
# 2. CHUẨN HÓA KÝ TỰ & ĐẾM TỪ (STRING UTILITIES)
# =============================================================================

def strip_vietnamese_accents(text: str) -> str:
    """
    Chuyển đổi văn bản tiếng Việt có dấu thành không dấu (chuẩn Unicode NFKD).
    Hỗ trợ fuzzy matching tiêu đề mục lục và đối chiếu chống sai lệch OCR.
    """
    if not text:
        return ""
    text_clean = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text_clean)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def count_words(text: str) -> int:
    """Đếm số từ phân tách bởi khoảng trắng (Whitespace Token Count)."""
    if not text:
        return 0
    return len(text.split())


def count_syllables_vietnamese(text: str) -> int:
    """Đếm số âm tiết tiếng Việt (tương đương số từ có ký tự chữ)."""
    if not text:
        return 0
    words = re.findall(r"\b\w+\b", text)
    return len(words)


def clean_and_dehyphenate_text(text: str) -> str:
    """
    Hàn gắn văn bản:
    1. Khử gạch nối ngắt dòng tiếng Việt (ngân-\nhàng -> ngân hàng, thanh-\ntoán -> thanh toán).
    2. Nối dòng gãy cứng giữa câu (hard line breaks trong cùng câu văn).
    3. Chuẩn hóa khoảng trắng dư thừa và ngắt dòng tối đa 2 lần.
    """
    if not text:
        return ""

    text_norm = unicodedata.normalize("NFC", text)

    # 1. Khử gạch nối ngắt dòng ở cuối dòng (các âm tiết ghép tiếng Việt cách nhau bằng dấu cách)
    text_norm = re.sub(r"(\b\w+)[-–—]\n\s*(\w+\b)", r"\1 \2", text_norm)

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
            # Nếu dòng trước không kết thúc bằng dấu ngắt câu hoặc ký tự tiêu đề/danh sách
            if not prev.endswith((".", ":", ";", "!", "?", "---", "***")) and not prev.startswith(("#", "-", "*", "|", "•")):
                # Nếu dòng sau bắt đầu bằng chữ thường hoặc không phải tiêu đề điều khoản
                if stripped[0].islower() or (len(stripped) > 1 and stripped[0].isalnum() and not stripped.startswith(("Điều", "Khoản", "Chương"))):
                    merged_lines[-1] = f"{prev} {stripped}"
                    continue

        merged_lines.append(stripped)

    # 3. Làm sạch khoảng trắng thừa
    result = "\n".join(merged_lines)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# =============================================================================
# 3. ĐỊNH DANH PHÁP LÝ & CẤU TRÚC THỨ BẬC (CANONICAL RESOLUTION)
# =============================================================================

def extract_document_identifier(
    breadcrumb_text: str = "",
    full_context: str = "",
    source_name: str = ""
) -> str:
    """
    Trích xuất tên hoặc số hiệu văn bản định danh một cách tổng quát:
    Ưu tiên:
    1. Số hiệu thể thức quy phạm pháp luật (Nghị định 30/2020) ở đầu context.
    2. Phần tử gốc đầu tiên của Breadcrumb (Root Node).
    3. Tên tệp nguồn được làm sạch.
    """
    # 1. Bắt số hiệu văn bản chính thức xuất hiện ở đầu văn bản
    target_str = breadcrumb_text or full_context[:200]
    match_code = DOC_NUMBER_PATTERN.search(target_str)
    if match_code:
        return match_code.group(1).strip()

    # 2. Lấy phần tử gốc đầu tiên của Breadcrumb
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

    # 3. Fallback từ tên tệp nguồn
    if source_name:
        stem = Path(source_name).stem
        stem = re.sub(r"^(?:vcb_)?(?:md|pdf|docx|excel)_\d+__?", "", stem, flags=re.IGNORECASE)
        stem = stem.replace("_", " ").strip().title()
        if stem:
            return stem

    return ""


def extract_hierarchical_units(full_context: str, text_chunk: str = "") -> str:
    """
    Bóc tách các đơn vị phân cấp chuẩn:
    Chương -> Mục -> Điều -> Khoản -> Điểm -> FAQ.
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


# =============================================================================
# 4. TÁCH CÂU AN TOÀN NGÂN HÀNG (SENTENCE SPLITTING & GROUPING)
# =============================================================================

def split_sentences_banking(text: str) -> List[str]:
    """
    Tách câu nhận thức ngữ cảnh ngân hàng và pháp lý:
    Bảo vệ an toàn không bị cắt gãy các thực thể:
    - Số tiền & Tỷ lệ: 1.000.000 VND, 5,5%/năm, 100 USD.
    - Điều khoản pháp lý: Điều 12., Khoản 1., Mục 2.
    - Viết tắt có dấu chấm: TP.HCM, v.v., TNHH., ThS.
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

    # 4. Tách câu sau dấu kết thúc [.?!] kèm khoảng trắng và chữ hoa [A-ZÀ-Ỹ] hoặc ngắt dòng kép
    raw_parts = re.split(r"(?<=[.?!])\s+(?=[A-ZÀ-Ỹ])|(?:\n{2,})", temp)

    # 5. Khôi phục các token đã bảo vệ
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
    Gom nhóm các câu thành các chunk vừa vặn với kích thước trần max_size (ký tự).
    Không bao giờ cắt giữa chừng một câu văn.
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


# =============================================================================
# 5. ĐÁNH GIÁ CHẤT LƯỢNG TIẾNG VIỆT & LỌC HEADER/FOOTER
# =============================================================================

def is_valid_vietnamese_text(text: str, min_valid_ratio: float = 0.25) -> bool:
    """
    Kiểm tra chất lượng văn bản:
    Phát hiện văn bản scan rác hoặc lỗi font VNI/TCVN3 dựa trên 2 quy tắc:
    1. Âm vị học tiếng Việt (Phonotactics): Tiếng Việt không bao giờ có >= 4 phụ âm liên tiếp.
    2. Tỷ lệ từ có nguyên âm tiếng Việt hợp lệ >= min_valid_ratio (mặc định 25%).
    """
    if not text or len(text.strip()) < 10:
        return False

    text_lower = text.lower()

    # Kiểm tra ký tự lỗi unicode lạ (\ufffd hoặc byte điều khiển)
    weird_chars = len(re.findall(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", text))
    if weird_chars / max(len(text), 1) > 0.05:
        return False

    words = re.findall(r"\b\w+\b", text_lower)
    if not words:
        return False

    # 1. Phát hiện cụm phụ âm vô nghĩa (>= 4 phụ âm liên tiếp, ví dụ: asdkjh, zxclkj)
    invalid_consonant_clusters = sum(
        1 for w in words if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", w)
    )
    if invalid_consonant_clusters / max(len(words), 1) > 0.20:
        return False

    # 2. Đếm số từ chứa ít nhất 1 nguyên âm
    words_with_vowels = sum(1 for w in words if any(c in VIETNAMESE_VOWELS for c in w))
    ratio = words_with_vowels / len(words)

    return ratio >= min_valid_ratio


def detect_and_remove_headers_footers(pages_text: List[str]) -> List[str]:
    """
    Tự động phát hiện và loại bỏ Header/Footer và số trang lặp lại qua nhiều trang.
    """
    if len(pages_text) < 3:
        # Với tài liệu ngắn (< 3 trang), chỉ lọc số trang đơn thuần
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
