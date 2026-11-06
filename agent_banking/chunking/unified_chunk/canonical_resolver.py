# unified_chunk/canonical_resolver.py
# Universal NLP & Text Normalization Engine (100% Domain-Agnostic)
# Complies with GEMINI.md: 0% regex, no typing library, zero ternary, zero __init__.py

import logging
import unicodedata

logger = logging.getLogger("canonical_resolver")


# -----------------------------------------------------------------------------
# 1. TEXT CLEANING & DEHYPHENATION (UNIVERSAL)
# -----------------------------------------------------------------------------
def clean_and_dehyphenate_text(text):
    """
    Cleans natural text without discarding characters or favoring specific domains:
    1. Removes UTF-8 BOM (\ufeff) and zero-width spaces (\u200b).
    2. Stitches line breaks split by hyphens (- or \xad):
       - Vietnamese syllables: joined with a space.
       - Foreign words: stitched without space.
    3. Normalizes whitespace and removes redundant empty lines.
    Strictly complies with 0% Regex Law and Zero Ternary Law.
    """
    if not text:
        return ""

    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines_raw = text.split("\n")
    stitched_lines = []
    idx = 0
    total_raw = len(lines_raw)

    while idx < total_raw:
        line = lines_raw[idx]
        stripped = line.rstrip()

        is_hyphenated = False
        if stripped.endswith("-") or stripped.endswith("\xad"):
            if len(stripped) > 1 and stripped[-2].isalnum() and idx + 1 < total_raw:
                is_hyphenated = True

        if is_hyphenated:
            next_line = lines_raw[idx + 1].lstrip()
            if next_line and next_line[0].isalnum():
                prefix_part = stripped[:-1]
                w1_tokens = prefix_part.split()
                w1 = ""
                if w1_tokens:
                    w1 = w1_tokens[-1]
                w2_tokens = next_line.split()
                w2 = ""
                if w2_tokens:
                    w2 = w2_tokens[0]

                is_vn = False
                if w1 and strip_vietnamese_accents(w1) != w1:
                    is_vn = True
                else:
                    if w2 and strip_vietnamese_accents(w2) != w2:
                        is_vn = True

                if is_vn:
                    stitched_lines.append(prefix_part + " " + next_line)
                else:
                    stitched_lines.append(prefix_part + next_line)
                idx += 2
                continue

        stitched_lines.append(line)
        idx += 1

    lines = [" ".join(line.replace("\xad", "").split()) for line in stitched_lines]

    cleaned_lines = []
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
# 2. UNIVERSAL SENTENCE BOUNDARY SEGMENTATION (0% REGEX, DOMAIN-AGNOSTIC)
# -----------------------------------------------------------------------------
_SAT_MODEL = None
_SAT_INITIALIZED = False


def get_sat_segmenter(model_name="sat-3l-sm"):
    """
    Initializes and caches the wtpsplit (SaT) model instance.
    Falls back to Unicode rule-based splitting if model is unavailable.
    """
    global _SAT_MODEL, _SAT_INITIALIZED
    if _SAT_INITIALIZED:
        return _SAT_MODEL

    try:
        from wtpsplit import SaT
        import torch
        _SAT_MODEL = SaT(model_name)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            _SAT_MODEL.half().to(device)
        else:
            _SAT_MODEL.to(device)
        logger.info("[CanonicalResolver] Loaded wtpsplit (SaT: " + model_name + ") on " + device)
    except Exception as e:
        logger.warning("[CanonicalResolver] wtpsplit unavailable: " + str(e) + ". Using rule-based fallback.")
        _SAT_MODEL = None
    finally:
        _SAT_INITIALIZED = True

    return _SAT_MODEL


# Universal abbreviations across academic, technical, business, and general text
# Zero hardcoded banking/currency units (no đ/tháng, đ/năm, tr., tỷ., đ.)
UNIVERSAL_KNOWN_ABBREVIATIONS = {
    "tp.", "tp.hcm", "hcm.", "hn.", "v.v.", "ths.", "ts.", "bs.", "mr.", "mrs.", "ms.",
    "dr.", "prof.", "assoc.", "stt.", "no.", "inc.", "corp.", "ltd.", "co.", "approx.",
    "dept.", "est.", "fig.", "vs.", "etc.", "e.g.", "i.e.", "vol.", "pp.", "ed."
}


def split_universal_sentences(text):
    """
    Universal domain-agnostic sentence boundary splitter.
    Preserves decimals (10.5), thousand separators (10,000 / 10.000), percentages (12.5%),
    universal abbreviations, and structural markers.
    Strictly 0% Regex Law, Zero Ternary Law, No Typing Library.
    """
    if not text:
        return []

    raw_lines = [l.strip() for l in text.splitlines() if l.strip()]
    all_sentences = []
    structural_delims = ("-", "*", "•", "\"", "'", "“", "”", "(", "[")

    for line in raw_lines:
        n = len(line)
        start_idx = 0
        i = 0
        while i < n:
            ch = line[i]
            is_boundary = False

            if ch in (".", "!", "?"):
                # 1. Ellipsis check: "..."
                if i + 1 < n and line[i + 1] == ".":
                    i += 1
                    continue
                if i > 0 and line[i - 1] == ".":
                    i += 1
                    continue

                # 2. Decimal / number check: "10.5" or "10.000" or "1.5%"
                if i > 0 and line[i - 1].isdigit() and i + 1 < n and line[i + 1].isdigit():
                    i += 1
                    continue

                # 3. Universal abbreviation check
                word_start = i
                while word_start > 0 and not line[word_start - 1].isspace():
                    word_start -= 1
                curr_word = line[word_start : i + 1].lower()
                if curr_word in UNIVERSAL_KNOWN_ABBREVIATIONS:
                    i += 1
                    continue

                # 4. End of line or boundary confirmation
                if i == n - 1:
                    is_boundary = True
                else:
                    if line[i + 1].isspace():
                        next_char_idx = i + 1
                        while next_char_idx < n and line[next_char_idx].isspace():
                            next_char_idx += 1
                        if next_char_idx < n:
                            next_ch = line[next_char_idx]
                            if next_ch.isupper() or next_ch.isdigit() or next_ch in structural_delims:
                                is_boundary = True
                        else:
                            is_boundary = True

            if is_boundary:
                sent = line[start_idx : i + 1].strip()
                if sent:
                    all_sentences.append(sent)
                start_idx = i + 1
                while start_idx < n and line[start_idx].isspace():
                    start_idx += 1
                i = start_idx
            else:
                i += 1

        if start_idx < n:
            remaining = line[start_idx:].strip()
            if remaining:
                all_sentences.append(remaining)

    return all_sentences


def split_sentences(text, use_sat=True):
    """
    Universal sentence segmentation facade.
    Uses wtpsplit (SaT) if available, falling back to split_universal_sentences.
    """
    if not text:
        return []

    if use_sat:
        sat = get_sat_segmenter()
        if sat is not None:
            try:
                raw_segments = sat.split(text, strip_whitespace=True)
                clean_segments = [s.strip() for s in raw_segments if s and s.strip()]
                if clean_segments:
                    return clean_segments
            except Exception as e:
                logger.warning("[CanonicalResolver] SaT.split error: " + str(e))

    return split_universal_sentences(text)


# Backwards compatibility aliases (0% breaking change for existing callers)
split_banking_sentences = split_universal_sentences
split_sentences_banking = split_universal_sentences
split_sentences_generalized = split_sentences


# -----------------------------------------------------------------------------
# 3. SENTENCE GROUPING (BUDGET BASED)
# -----------------------------------------------------------------------------
def group_sentences(sentences, max_words=350, max_chunk_size=None, **kwargs):
    """
    Groups sentences into paragraph chunks without breaking sentence boundaries.
    """
    if not sentences:
        return []

    if max_words is not None:
        limit = max_words
    else:
        if max_chunk_size:
            limit = max_chunk_size // 2
        else:
            limit = 350

    chunks = []
    current_sentences = []
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
# 4. TEXT VALIDITY CHECKS (UNICODE INVARIANTS, 0% REGEX)
# -----------------------------------------------------------------------------
def is_valid_readable_text(text):
    """
    Validates text readability via Unicode categories (0% Regex).
    Detects mojibake, control character corruption, and verifies meaningful content.
    """
    if not text:
        return False
    s = text.replace("\ufeff", "").strip()
    if len(s) < 5:
        return False

    total_chars = len(s)
    bad_chars = 0
    for c in s:
        if c in ("\ufffd", "\x00", "\ufeff"):
            bad_chars += 1
        else:
            cat = unicodedata.category(c)
            if cat == "Cc" or (cat.startswith("C") and c not in ("\n", "\r", "\t", "\u200c", "\u200d", "\u200e", "\u200f")):
                bad_chars += 1

    if (bad_chars / total_chars) > 0.03:
        return False

    has_meaningful_content = False
    for c in s:
        cat = unicodedata.category(c)
        if cat[0] in ("L", "N"):
            has_meaningful_content = True
            break

    return has_meaningful_content


is_valid_natural_text = is_valid_readable_text
is_valid_vietnamese_text = is_valid_readable_text


# -----------------------------------------------------------------------------
# 5. REMOVE HEADER / FOOTER STRUCTURAL PATTERNS (DOMAIN-AGNOSTIC)
# -----------------------------------------------------------------------------
def detect_and_remove_headers_footers(text, custom_ignore_phrases=None, config=None):
    """
    Removes header/footer noise, page numbers, and form codes in a domain-agnostic manner.
    """
    if not text:
        return ""

    ignore_phrases = []
    if custom_ignore_phrases is not None:
        ignore_phrases.extend(custom_ignore_phrases)
    else:
        if config is not None and getattr(config, "custom_ignore_phrases", None):
            ignore_phrases.extend(config.custom_ignore_phrases)

    lines = text.splitlines()
    filtered = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        s_lower = s.lower()
        words = s.split()

        # Page numbers: "1", "1/5", "Page 2", "Trang 3"
        if len(words) == 1:
            w0 = words[0]
            if w0.isdigit() or (all(c.isdigit() or c in "/-" for c in w0) and any(c.isdigit() for c in w0)):
                continue
        else:
            if len(words) in (2, 3) and words[0].isalpha() and len(words[0]) <= 6:
                remainder = "".join(words[1:])
                if all(c.isdigit() or c in "/-" for c in remainder) and any(c.isdigit() for c in remainder):
                    continue

        # Form codes with colons: "01/TNDN:", "FORM-A1:"
        if ":" in s and len(words) <= 10:
            prefix = s.split(":", 1)[0].strip()
            prefix_words = prefix.split()
            if prefix_words:
                code_token = prefix_words[-1]
                has_digit = any(c.isdigit() for c in code_token)
                has_code_delim = any(c in "/-#_" for c in code_token) or (
                    len(code_token) <= 5 and any(c.isdigit() for c in code_token) and any(c.isalpha() for c in code_token)
                )
                if has_digit and has_code_delim:
                    continue

        if ignore_phrases:
            should_skip = False
            for p in ignore_phrases:
                if p.strip() and p.lower() in s_lower:
                    should_skip = True
                    break
            if should_skip:
                continue

        filtered.append(line)

    return "\n".join(filtered)


# -----------------------------------------------------------------------------
# 6. UNIVERSAL DOCUMENT IDENTIFIER & STRUCTURAL UNITS
# -----------------------------------------------------------------------------
def extract_document_identifier(filename="", text=""):
    """Extracts clean document stem from filename or input text."""
    if filename:
        if "." in filename:
            stem = filename.rsplit(".", 1)[0]
        else:
            stem = filename
        return stem.strip()
    if text:
        stripped_text = text.strip()
        if stripped_text:
            return stripped_text.splitlines()[0].strip()
    return ""


def extract_hierarchical_units(text="", text_chunk=""):
    """
    Extracts structural section units (e.g. Chapter 1, Section 2, Article III, Điều 1, Khoản 2).
    Purely pattern-driven: [Latin Title Word] [Number / Roman Numeral].
    """
    content = text or text_chunk
    if not content:
        return []
    words = content.split()
    results = []
    i = 0
    roman_chars = set("ivxlcdmIVXLCDM")
    while i < len(words) - 1:
        curr_w = words[i].strip(",.;:()")
        next_w = words[i + 1].strip(",.;:()")

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
            results.append(curr_w.capitalize() + " " + unit_val)
            i += 1
        i += 1

    unique_results = []
    seen = set()
    for item in results:
        if item not in seen:
            seen.add(item)
            unique_results.append(item)
    return unique_results


def strip_vietnamese_accents(text):
    """
    Normalizes and strips Vietnamese accents without regex.
    Replaces 'đ'/'Đ' with 'd'/'D' and decomposes via Unicode NFD.
    """
    if not text:
        return ""

    clean_text = text.replace("đ", "d").replace("Đ", "D")
    nfd_chars = unicodedata.normalize("NFD", clean_text)
    filtered_chars = []
    for c in nfd_chars:
        if unicodedata.category(c) != "Mn":
            filtered_chars.append(c)
    return unicodedata.normalize("NFC", "".join(filtered_chars))


__all__ = [
    "clean_and_dehyphenate_text",
    "get_sat_segmenter",
    "split_sentences",
    "split_universal_sentences",
    "split_banking_sentences",
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
    "strip_vietnamese_accents",
]
