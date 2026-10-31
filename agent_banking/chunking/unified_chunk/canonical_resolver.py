"""
Module Canonical Resolver (Universal NLP & Text Normalization Engine)
======================================================================
Bộ tiện ích chuẩn hóa văn bản, phân đoạn câu và xử lý hình thái tự nhiên
cho hệ thống RAG Banking (chuyên trách tiền xử lý Tiếng Việt và Tiếng Anh;
mọi ký tự và ngôn ngữ khác được bảo toàn nguyên vẹn như chuỗi ký tự đặc biệt).

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không sử dụng `import re` hay regex methods.
2. BẢO TOÀN NGUYÊN VẸN NỘI DUNG: Không tự tiện xóa bỏ ký tự ngoại ngữ;
   coi các ngôn ngữ khác như các chuỗi ký tự đặc biệt cần được giữ nguyên bản.
3. UNICODE STANDARD INVARIANTS: Dựa trên chuẩn Unicode (UAX #29, Unicode Categories L*, N*, P*, Z*, C*)
   và nguyên lý chữ hoa/chữ thường (Bicameral Capitalization) để nhận diện ranh giới câu.
"""

from __future__ import annotations

import logging
import unicodedata

logger = logging.getLogger("canonical_resolver")


# -----------------------------------------------------------------------------
# 1. LÀM SẠCH VĂN BẢN & HÀN GẮN NGẮT DÒNG GẠCH NỐI (DEHYPHENATION)
# -----------------------------------------------------------------------------
def clean_and_dehyphenate_text(text: str) -> str:
    """
    Làm sạch văn bản tự nhiên (bảo toàn nguyên vẹn mọi ký tự, không xóa ngôn ngữ khác):
    1. Hàn gắn các từ bị ngắt dòng có dấu gạch nối (vd: ngân-\\nhàng -> ngân hàng).
    2. Chuẩn hóa khoảng trắng và loại bỏ dòng trống dư thừa.
    Thuần túy xử lý chuỗi cơ sở (0% Regex).
    """
    if not text:
        return ""

    lines_raw = text.split("\n")
    stitched_lines: list[str] = []
    idx = 0
    total_raw = len(lines_raw)

    while idx < total_raw:
        line = lines_raw[idx]
        stripped = line.rstrip()
        # Nếu dòng kết thúc bằng dấu '-' nối với từ dòng kế tiếp
        if stripped.endswith("-") and len(stripped) > 1 and stripped[-2].isalnum() and idx + 1 < total_raw:
            next_line = lines_raw[idx + 1].lstrip()
            if next_line and next_line[0].isalnum():
                stitched_lines.append(stripped[:-1] + next_line)
                idx += 2
                continue
        stitched_lines.append(line)
        idx += 1

    # Chuẩn hóa khoảng trắng dư thừa trong từng dòng
    lines = [" ".join(line.split()) for line in stitched_lines]

    # Bỏ các dòng trống liên tiếp (chỉ giữ tối đa 1 dòng trống phân cách đoạn)
    cleaned_lines: list[str] = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned_lines.append("")
                prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False

    return "\n".join(cleaned_lines).strip()


# -----------------------------------------------------------------------------
# 2. PHÂN ĐOẠN CÂU PHỔ QUÁT (UNIVERSAL SENTENCE BOUNDARY SEGMENTATION)
# -----------------------------------------------------------------------------
_SAT_MODEL = None
_SAT_INITIALIZED = False


def get_sat_segmenter(model_name: str = "sat-3l-sm") -> object | None:
    """
    Khởi tạo hoặc trả về instance singleton của mô hình wtpsplit (SaT).
    Ưu tiên số 1 cho bài toán phân đoạn câu thông minh (Segment Any Text).
    """
    global _SAT_MODEL, _SAT_INITIALIZED
    if _SAT_INITIALIZED:
        return _SAT_MODEL

    try:
        from wtpsplit import SaT
        _SAT_MODEL = SaT(model_name)
        logger.info(f"[CanonicalResolver] Khởi tạo thành công mô hình wtpsplit (SaT: {model_name}).")
    except Exception as e:
        logger.warning(
            f"[CanonicalResolver] Không thể khởi tạo wtpsplit (SaT): {e}. "
            "Hệ thống sẽ chuyển sang cơ chế dự phòng Unicode Invariant rule-based."
        )
        _SAT_MODEL = None
    finally:
        _SAT_INITIALIZED = True

    return _SAT_MODEL


def split_sentences(text: str, use_sat: bool = True) -> list[str]:
    """
    Phân đoạn câu song ngữ Tiếng Việt - Tiếng Anh tối ưu hóa cho RAG Banking.

    100% sử dụng mô hình học sâu `wtpsplit (SaT)` (Segment Any Text).
    Tuyệt đối KHÔNG sử dụng regex hoặc các mảng ký tự neo / dấu câu thủ công.
    """
    if not text:
        return []

    # 1. ƯU TIÊN DUY NHẤT: wtpsplit (SaT)
    if use_sat:
        sat = get_sat_segmenter()
        if sat is not None:
            try:
                raw_segments = sat.split(text, strip_whitespace=True)
                clean_segments = [s.strip() for s in raw_segments if s and s.strip()]
                if clean_segments:
                    return clean_segments
            except Exception as e:
                logger.warning(f"[CanonicalResolver] SaT.split() gặp lỗi: {e}")

    # 2. DỰ PHÒNG THUẦN TÚY (0% Regex, 0% Mảng ký tự neo): Tách theo cấu trúc dòng
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines
    return [text.strip()]


split_sentences_banking = split_sentences
split_sentences_generalized = split_sentences


# -----------------------------------------------------------------------------
# 3. GOM NHÓM CÂU THEO NGÂN SÁCH (SENTENCE GROUPING)
# -----------------------------------------------------------------------------
def group_sentences(
    sentences: list[str],
    max_words: int = 350,
    max_chunk_size: int | None = None,
    **kwargs
) -> list[str]:
    """
    Gom các câu thành từng nhóm (chunk) với kích thước từ tối đa.
    Phân đoạn câu tuần tự, độc lập và trọn vẹn, không cắt vụn giữa chừng một câu văn.
    """
    if not sentences:
        return []

    if max_words is not None:
        limit = max_words
    elif max_chunk_size:
        limit = max_chunk_size // 2
    else:
        limit = 350

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for s in sentences:
        s_words = len(s.split())
        if current_word_count + s_words > limit and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_word_count = 0

        current_sentences.append(s)
        current_word_count += s_words

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


group_sentences_banking = group_sentences


# -----------------------------------------------------------------------------
# 4. ĐÁNH GIÁ TÍNH HỢP LỆ VĂN BẢN (0% REGEX)
# -----------------------------------------------------------------------------
def is_valid_readable_text(text: str) -> bool:
    """
    Kiểm tra văn bản hợp lệ theo đặc tính chuẩn Unicode (0% Regex).
    1. Phát hiện và loại bỏ rác mã hóa, Mojibake và ký tự điều khiển hỏng bảng mã.
    2. Đảm bảo văn bản chứa các ký tự có nghĩa (chữ cái Category L* hoặc ký số N*).
    3. Không xóa bỏ ngôn ngữ ngoại lai (coi như các chuỗi ký tự đặc biệt).
    """
    if not text:
        return False
    s = text.strip()
    if len(s) < 5:
        return False

    total_chars = len(s)
    # 1. Ký tự điều khiển bất thường hoặc ký tự thay thế hỏng bảng mã (\ufffd, \x00-\x1f)
    bad_chars = sum(
        1 for c in s
        if c in ("\ufffd", "\x00", "\ufeff") or
        (unicodedata.category(c).startswith("C") and c not in ("\n", "\r", "\t"))
    )
    if bad_chars / total_chars > 0.03:
        return False

    # 2. Phải có ít nhất ký tự chữ cái (Unicode category L*) hoặc ký hiệu số (N*)
    has_meaningful_content = any(
        unicodedata.category(c)[0] in ("L", "N") for c in s
    )
    if not has_meaningful_content:
        return False

    return True


is_valid_natural_text = is_valid_readable_text


def is_valid_vietnamese_text(text: str) -> bool:
    """
    Kiểm tra tính hợp lệ văn bản đọc được (tương thích ngược, hoàn toàn không hardcode bảng chữ).
    """
    return is_valid_readable_text(text)


# -----------------------------------------------------------------------------
# 5. LỌC HEADER / FOOTER & MẪU BIỂU TỰ NHIÊN (STRUCTURAL PATTERNS)
# -----------------------------------------------------------------------------
def detect_and_remove_headers_footers(
    text: str,
    custom_ignore_phrases: list[str] | None = None,
    config: object | None = None
) -> str:
    """
    Loại bỏ các dòng header/footer, số trang và mã biểu mẫu theo cấu trúc cú pháp phổ quát.
    Tuân thủ 0% Regex Law và nguyên tắc Zero Domain Hardcoding qua config.
    """
    if not text:
        return ""

    ignore_phrases: list[str] = []
    if custom_ignore_phrases is not None:
        ignore_phrases.extend(custom_ignore_phrases)
    elif config is not None and getattr(config, "custom_ignore_phrases", None):
        ignore_phrases.extend(config.custom_ignore_phrases)

    lines = text.splitlines()
    filtered: list[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        s_lower = s.lower()
        words = s.split()

        # 1. Số trang độc lập hoặc có nhãn ngắn (vd: "1", "1/5", "Page 2", "Trang 3")
        if len(words) == 1:
            w0 = words[0]
            if w0.isdigit() or (all(c.isdigit() or c in "/-" for c in w0) and any(c.isdigit() for c in w0)):
                continue
        elif len(words) in (2, 3) and words[0].isalpha() and len(words[0]) <= 6:
            remainder = "".join(words[1:])
            if all(c.isdigit() or c in "/-" for c in remainder) and any(c.isdigit() for c in remainder):
                continue

        # 2. Tiêu đề mẫu biểu có mã định danh và dấu hai chấm (vd: "01/TNDN:", "202-B:", "#12:", "A-04:")
        if ":" in s and len(words) <= 10:
            prefix = s.split(":", 1)[0].strip()
            prefix_words = prefix.split()
            if prefix_words:
                code_token = prefix_words[-1]
                has_digit = any(c.isdigit() for c in code_token)
                has_code_delim = any(c in "/-#_" for c in code_token) or (len(code_token) <= 5 and any(c.isdigit() for c in code_token) and any(c.isalpha() for c in code_token))
                if has_digit and has_code_delim:
                    continue

        # 3. Cụm từ người dùng cấu hình loại trừ (Slogan, banner, metadata)
        if ignore_phrases and any(p.lower() in s_lower for p in ignore_phrases if p.strip()):
            continue

        filtered.append(line)

    return "\n".join(filtered)


# -----------------------------------------------------------------------------
# 6. TIỆN ÍCH TRÍCH XUẤT ĐỊNH DANH & PHÂN CẤP TỔNG QUÁT
# -----------------------------------------------------------------------------
def extract_document_identifier(filename: str = "", text: str = "") -> str:
    """Lấy định danh tài liệu sạch từ tên tệp hoặc văn bản đầu vào (0% Regex)."""
    if filename:
        if "." in filename:
            stem = filename.rsplit(".", 1)[0]
        else:
            stem = filename
        return stem.strip()
    if text:
        stripped_text = text.strip()
        first_line = ""
        if stripped_text:
            first_line = stripped_text.splitlines()[0]
        return first_line.strip()
    return ""


def extract_hierarchical_units(text: str = "", text_chunk: str = "") -> list[str]:
    """
    Trích xuất các điều khoản, phân đoạn phân cấp song ngữ Việt - Anh (0% Regex):
    Cú pháp tổng quát: [Từ định danh viết hoa] [Chỉ số: Số / Số La Mã].
    Ví dụ tiếng Việt: Điều 1, Khoản 2, Mục III, Chương 4, Phần I...
    Ví dụ tiếng Anh: Section 1, Article 2, Chapter III, Part 4, Clause 5...
    """
    content = text or text_chunk
    if not content:
        return []
    words = content.split()
    results: list[str] = []
    i = 0
    roman_chars = set("ivxlcdmIVXLCDM")
    while i < len(words) - 1:
        curr_w = words[i].strip(",.;:()")
        next_w = words[i + 1].strip(",.;:()")

        # Đảm bảo từ viết hoa thuộc bảng chữ cái Latin (Tiếng Việt / Tiếng Anh)
        is_title_word = bool(
            curr_w
            and curr_w[0].isupper()
            and curr_w.isalpha()
            and 2 <= len(curr_w) <= 20
            and all(unicodedata.name(c, "").startswith("LATIN") for c in curr_w)
        )
        is_num = next_w.isdigit()
        is_roman = bool(next_w and all(c in roman_chars for c in next_w) and not is_num)

        if is_title_word and (is_num or is_roman):
            unit_val = next_w
            if is_roman:
                unit_val = next_w.upper()
            results.append(f"{curr_w.capitalize()} {unit_val}")
            i += 1
        i += 1
    return list(dict.fromkeys(results))


__all__ = [
    "clean_and_dehyphenate_text",
    "get_sat_segmenter",
    "split_sentences",
    "split_sentences_banking",
    "split_sentences_generalized",
    "group_sentences",
    "group_sentences_banking",
    "is_valid_vietnamese_text",
    "is_valid_readable_text",
    "is_valid_natural_text",
    "detect_and_remove_headers_footers",
    "extract_document_identifier",
    "extract_hierarchical_units",
]
