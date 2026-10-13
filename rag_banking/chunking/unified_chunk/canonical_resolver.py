"""
Module Canonical Entity & Legal Structure Resolver
===================================================
Module xử lý ngôn ngữ tự nhiên (NLP) và bóc tách cấu trúc tài liệu đa lĩnh vực
hoàn toàn KHÔNG sử dụng biểu thức chính quy (0% Regex Engine - Pure Python String Traversal).

Các nhóm chức năng chính:
1. Chuẩn hóa chuỗi ký tự & Âm tiết:
   - `strip_vietnamese_accents()`: Khử dấu tiếng Việt (NFKD) chống lỗi OCR.
   - `count_words()`, `count_syllables_vietnamese()`: Đếm từ và số âm tiết tiếng Việt.
   - `clean_and_dehyphenate_text()`: Khử gạch nối ngắt dòng, nối câu gãy dòng cứng.

2. Định danh văn bản & Cấu trúc thứ bậc (Pure String Heuristics):
   - `extract_document_identifier()`: Bóc tách số hiệu thể thức văn bản và root title.
   - `extract_hierarchical_units()`: Bóc tách đơn vị thứ bậc (Chương, Mục, Điều, Khoản, Điểm, FAQ).

3. Tách câu an toàn nhận thức cấu trúc:
   - `split_sentences_banking()`: Tách câu ký tự thuần túy, bảo vệ số thực, tiền tệ, từ viết tắt.
   - `group_sentences_banking()`: Gom cụm câu theo kích thước trần max_chunk_size có overlap.

4. Kiểm định chất lượng văn bản & Khử Header/Footer:
   - `is_valid_vietnamese_text()`: Lọc văn bản rác bằng mật độ âm vị học và nguyên âm thuần túy.
   - `detect_and_remove_headers_footers()`: Tự động phát hiện và loại bỏ header/footer lặp qua trang.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path


# =============================================================================
# 1. HẰNG SỐ BẢNG NGUYÊN ÂM (UNICODE CHARACTER SETS)
# =============================================================================

# Bảng nguyên âm tiếng Việt và tiếng Latin chuẩn Unicode phục vụ kiểm tra ngôn ngữ tự nhiên
VIETNAMESE_VOWELS: set[str] = set("aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ")
LATIN_VOWELS: set[str] = set("aeiouy")


# =============================================================================
# 2. CHUẨN HÓA KÝ TỰ & ĐẾM TỪ (PURE STRING UTILITIES)
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
    """Đếm số âm tiết tiếng Việt (tương đương số từ có ký tự chữ/số)."""
    if not text:
        return 0
    return len([w for w in text.split() if any(c.isalnum() for c in w)])


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
    lines = text_norm.split("\n")
    merged_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged_lines.append("")
            continue

        if merged_lines and merged_lines[-1] != "":
            prev = merged_lines[-1]
            # Xử lý gạch nối cuối dòng
            if prev and prev[-1] in ("-", "–", "—"):
                prev_base = prev[:-1].rstrip()
                merged_lines[-1] = f"{prev_base} {stripped}"
                continue

            # Nối dòng gãy cứng giữa câu
            if not prev.endswith((".", ":", ";", "!", "?", "---", "***")) and not prev.startswith(("#", "-", "*", "|", "•")):
                first_toks = stripped.split()
                is_outline_heading = (
                    stripped.isupper()
                    or (len(first_toks) >= 2 and any(c.isdigit() for c in first_toks[1]) and first_toks[0][0].isupper())
                    or (first_toks and first_toks[0].rstrip(".").isdigit())
                )
                if stripped[0].islower() or (len(stripped) > 1 and stripped[0].isalnum() and not is_outline_heading):
                    merged_lines[-1] = f"{prev} {stripped}"
                    continue

        merged_lines.append(stripped)

    # Làm sạch khoảng trắng thừa
    clean_blocks: list[str] = []
    for line in merged_lines:
        clean_line = " ".join(line.split())
        clean_blocks.append(clean_line)

    result = "\n".join(clean_blocks)
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result.strip()


# =============================================================================
# 3. ĐỊNH DANH PHÁP LÝ & CẤU TRÚC THỨ BẬC (ZERO REGEX STRING PARSER)
# =============================================================================

def extract_document_identifier(
    breadcrumb_text: str = "",
    full_context: str = "",
    source_name: str = ""
) -> str:
    """
    Trích xuất tên hoặc số hiệu văn bản định danh một cách tổng quát đa lĩnh vực (0% Regex):
    Ưu tiên:
    1. Số hiệu thể thức văn bản chứa dấu '/' kèm chữ số (vd: Thông tư 17/2024/TT-NHNN, Nghị định 52/2024/NĐ-CP).
    2. Phần tử gốc đầu tiên của Breadcrumb (Root Node) sau khi làm sạch slug/prefix.
    3. Tên tệp nguồn được làm sạch tổng quát.
    """
    # 1. Bắt số hiệu văn bản chính thức xuất hiện ở đầu văn bản
    target_str = breadcrumb_text or full_context[:300]
    tokens = target_str.replace(":", " ").replace(",", " ").replace(";", " ").split()
    for i, tok in enumerate(tokens):
        if "/" in tok and any(c.isdigit() for c in tok) and len(tok) >= 5:
            prefix = ""
            if i > 0:
                prev = tokens[i - 1]
                if prev.lower() in ("số", "no.", "no"):
                    if i > 1:
                        prefix = f"{tokens[i - 2]} "
                else:
                    prefix = f"{prev} "
            return f"{prefix}{tok}".strip()

    # 2. Lấy phần tử gốc đầu tiên của Breadcrumb
    if breadcrumb_text:
        root_part = breadcrumb_text.split(" > ")[0].strip()
        if root_part.startswith("[") and "]" in root_part:
            root_part = root_part.split("]", 1)[1].strip()
        if "__" in root_part:
            root_part = root_part.split("__", 1)[1].strip()
        clean_root = root_part.replace("_", " ").strip()
        if clean_root:
            return clean_root

    # 3. Fallback từ tên tệp nguồn
    if source_name:
        stem = Path(source_name).stem
        if "__" in stem:
            stem = stem.split("__", 1)[1].strip()
        clean_stem = stem.replace("_", " ").strip().title()
        if clean_stem:
            return clean_stem

    return ""


def extract_hierarchical_units(full_context: str, text_chunk: str = "") -> str:
    """
    Bóc tách các đơn vị phân cấp chuẩn đa lĩnh vực (0% Regex):
    Chương / Chapter -> Mục / Section -> Điều / Article -> Khoản / Clause -> Điểm / Sub-clause -> FAQ.
    """
    context = full_context or text_chunk
    tokens = context.replace(":", " : ").replace(".", " . ").replace(",", " , ").replace(")", " ) ").split()

    dieu = ""
    khoan = ""
    diem = ""
    chuong = ""
    muc = ""
    faq = ""

    i = 0
    while i < len(tokens):
        t_lower = tokens[i].lower()
        if t_lower in ("điều", "article", "section") and i + 1 < len(tokens):
            nxt = tokens[i + 1].rstrip(".,;:")
            if any(c.isdigit() for c in nxt):
                dieu = f"Điều {nxt}"
                i += 2
                continue
        elif t_lower in ("khoản", "clause", "paragraph") and i + 1 < len(tokens):
            nxt = tokens[i + 1].rstrip(".,;:")
            if nxt.isdigit():
                khoan = f"Khoản {nxt}"
                i += 2
                continue
        elif t_lower in ("điểm", "point", "sub-clause") and i + 1 < len(tokens):
            nxt = tokens[i + 1].rstrip(".,;:)").lower()
            if len(nxt) == 1 and nxt.isalpha():
                diem = f"Điểm {nxt}"
                i += 2
                continue
        elif t_lower in ("chương", "phần", "chapter", "part") and i + 1 < len(tokens):
            nxt = tokens[i + 1].rstrip(".,;:")
            chuong = f"Chương {nxt}"
            i += 2
            continue
        elif t_lower in ("mục", "tiểu mục") and i + 1 < len(tokens):
            nxt = tokens[i + 1].rstrip(".,;:")
            muc = f"Mục {nxt}"
            i += 2
            continue
        elif t_lower in ("câu", "question", "q") and i + 1 < len(tokens):
            if tokens[i + 1].lower() == "hỏi" and i + 2 < len(tokens):
                nxt = tokens[i + 2].rstrip(".,;:")
                if any(c.isdigit() for c in nxt):
                    faq = f"Câu {nxt}"
                    i += 3
                    continue
            else:
                nxt = tokens[i + 1].rstrip(".,;:")
                if any(c.isdigit() for c in nxt):
                    faq = f"Câu {nxt}"
                    i += 2
                    continue
        i += 1

    # Ghép chuỗi phân cấp theo thứ tự ưu tiên
    if dieu:
        res = dieu
        if khoan:
            res += f", {khoan}"
            if diem:
                res += f", {diem}"
        return res
    if faq:
        return faq
    if muc:
        res = muc
        if khoan:
            res += f", {khoan}"
        return res
    if chuong:
        return chuong
    if khoan:
        return khoan

    return ""


# =============================================================================
# 4. TÁCH CÂU AN TOÀN ĐA LĨNH VỰC (ZERO REGEX CHARACTER TRAVERSAL)
# =============================================================================

def split_sentences_banking(text: str) -> list[str]:
    """
    Tách câu tổng quát hóa hoàn toàn KHÔNG sử dụng biểu thức chính quy (0% Regex):
    - Bảo vệ số thực, số tiền, tỷ lệ: 500.000.000 VND, 6,5%/năm, 50 USD, 3.14
    - Bảo vệ ký hiệu phân cấp, đề mục: 1., 2., 12., Điều 12., Khoản 2., Mục 1.
    - Bảo vệ từ viết tắt: TP.HCM, v.v., e.g., i.e., TNHH., INC.
    - Tách câu an toàn tại [.?!] theo sau bởi khoảng trắng và ký tự viết hoa.
    """
    if not text:
        return []

    text_norm = unicodedata.normalize("NFC", text).strip()
    if not text_norm:
        return []

    sentences: list[str] = []
    current_chars: list[str] = []
    n = len(text_norm)
    i = 0

    while i < n:
        ch = text_norm[i]
        current_chars.append(ch)

        # Kiểm tra dấu kết thúc câu tiềm năng
        if ch in (".", "?", "!"):
            if i + 1 < n and text_norm[i + 1] in (" ", "\t", "\n"):
                is_boundary = True

                pre_words = "".join(current_chars).strip().split()
                if pre_words:
                    last_tok = pre_words[-1]

                    # A. Kiểm tra số thập phân / số tiền có dấu chấm (vd: 500.000, 1.000.000.000)
                    if ch == "." and i > 0 and text_norm[i - 1].isdigit():
                        if last_tok.count(".") >= 1 and not last_tok.rstrip(".").isdigit():
                            is_boundary = False
                        elif len(pre_words) <= 2 and last_tok.rstrip(".").isdigit():
                            is_boundary = False

                    # B. Kiểm tra đề mục phân cấp đi kèm chữ số (Từ viết hoa kèm số: CapitalizedWord Number.)
                    if (
                        last_tok.rstrip(".").isdigit()
                        or (len(last_tok) > 1 and last_tok[-1] == "." and any(c.isdigit() for c in last_tok[:-1]))
                    ):
                        if len(pre_words) >= 2 and pre_words[-2] and pre_words[-2][0].isupper():
                            is_boundary = False
                        elif len(pre_words) <= 3:
                            is_boundary = False

                    # C. Kiểm tra từ viết tắt (TP.HCM, v.v., e.g., TNHH., LLC.)
                    if last_tok.count(".") >= 2:
                        is_boundary = False
                    elif last_tok.rstrip(".").isupper() and len(last_tok.rstrip(".")) <= 5:
                        is_boundary = False

                # D. Ký tự tiếp theo sau khoảng trắng phải là ký tự in hoa hoặc xuống dòng
                j = i + 1
                while j < n and text_norm[j] in (" ", "\t"):
                    j += 1
                if j < n:
                    next_non_space = text_norm[j]
                    if not (next_non_space.isupper() or next_non_space == "\n" or text_norm[i + 1] == "\n"):
                        is_boundary = False
                else:
                    is_boundary = True

                if is_boundary:
                    sent = "".join(current_chars).strip()
                    if sent:
                        sentences.append(sent)
                    current_chars.clear()
                    while i + 1 < n and text_norm[i + 1] in (" ", "\t"):
                        i += 1
        i += 1

    rem = "".join(current_chars).strip()
    if rem:
        sentences.append(rem)

    return sentences if sentences else [text]


# Alias tổng quát hóa không gắn mác riêng ngành ngân hàng
split_sentences_generalized = split_sentences_banking


def group_sentences_banking(sentences: list[str], max_size: int, overlap: int = 100) -> list[str]:
    """
    Gom nhóm các câu thành các chunk vừa vặn với kích thước trần max_size (ký tự).
    Không bao giờ cắt giữa chừng một câu văn.
    """
    if not sentences:
        return [""]

    parts: list[str] = []
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
# 5. ĐÁNH GIÁ CHẤT LƯỢNG VĂN BẢN TỰ NHIÊN & LỌC HEADER/FOOTER (0% REGEX)
# =============================================================================

def is_valid_vietnamese_text(text: str, min_valid_ratio: float = 0.25) -> bool:
    """
    Kiểm tra chất lượng văn bản tự nhiên đa ngôn ngữ (0% Regex):
    1. Lọc ký tự lỗi byte điều khiển lạ (> 5% ký tự không hợp lệ).
    2. Phân bố thống kê âm vị học (Phonotactic Consonant Entropy): phát hiện cụm >= 5 phụ âm liên tiếp.
    3. Tỷ lệ từ chứa nguyên âm hợp lệ >= min_valid_ratio.
    """
    if not text or len(text.strip()) < 10:
        return False

    text_lower = text.lower()

    # 1. Ký tự byte điều khiển lạ
    weird_count = sum(1 for c in text if (ord(c) < 32 and c not in "\t\n\r") or c == "\ufffd")
    if weird_count / max(len(text), 1) > 0.05:
        return False

    # 2. Tách từ thuần túy không dùng regex
    words = [w.strip(".,;:?!\"'()[]{}<>-–—/") for w in text_lower.split()]
    words = [w for w in words if w]
    if not words:
        return False

    # 3. Phân bố phụ âm: cụm >= 5 phụ âm liên tiếp
    def has_extreme_consonants(word: str) -> bool:
        consecutive = 0
        for ch in word:
            if ch in "bcdfghjklmnpqrstvwxyz":
                consecutive += 1
                if consecutive >= 5:
                    return True
            else:
                consecutive = 0
        return False

    extreme_clusters = sum(1 for w in words if has_extreme_consonants(w))
    if extreme_clusters / len(words) > 0.20:
        return False

    # 4. Tỷ lệ từ chứa nguyên âm
    all_valid_vowels = VIETNAMESE_VOWELS | LATIN_VOWELS
    words_with_vowels = sum(1 for w in words if any(c in all_valid_vowels for c in w))
    ratio = words_with_vowels / len(words)

    return ratio >= min_valid_ratio


# Alias tổng quát hóa đánh giá chất lượng văn bản
is_valid_natural_text = is_valid_vietnamese_text


def detect_and_remove_headers_footers(pages_text: list[str]) -> list[str]:
    """
    Tự động phát hiện và loại bỏ Header/Footer và số trang lặp lại qua nhiều trang (0% Regex).
    """
    def is_page_number(line_str: str) -> bool:
        s = line_str.strip().lower()
        if not s:
            return False
        clean = s.replace("trang", "").replace("page", "").replace("of", "/").strip()
        parts = clean.split("/")
        return all(p.strip().isdigit() for p in parts if p.strip()) and any(p.strip().isdigit() for p in parts)

    if len(pages_text) < 3:
        cleaned: list[str] = []
        for p in pages_text:
            lines = [l for l in p.split("\n") if not is_page_number(l)]
            cleaned.append("\n".join(lines))
        return cleaned

    top_freq: dict[str, int] = {}
    bottom_freq: dict[str, int] = {}

    for p in pages_text:
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        if not lines:
            continue
        for l in lines[:2]:
            if len(l) < 100 and not l.startswith("#"):
                top_freq[l] = top_freq.get(l, 0) + 1
        for l in lines[-2:]:
            if len(l) < 100:
                bottom_freq[l] = bottom_freq.get(l, 0) + 1

    threshold = max(3, int(len(pages_text) * 0.4))
    repeated_headers = {l for l, c in top_freq.items() if c >= threshold}
    repeated_footers = {l for l, c in bottom_freq.items() if c >= threshold}

    cleaned_pages: list[str] = []
    for p in pages_text:
        lines = p.split("\n")
        filtered: list[str] = []
        for l in lines:
            if is_page_number(l):
                continue
            if l.strip() in repeated_headers or l.strip() in repeated_footers:
                continue
            filtered.append(l)
        cleaned_pages.append("\n".join(filtered))

    return cleaned_pages
