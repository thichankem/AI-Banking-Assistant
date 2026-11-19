"""Conservative block language detection for Vietnamese chunk boundaries."""
import unicodedata

from agent_banking.chunking.config.settings import get_rules
from .boundaries import protected_markup_spans

_LANG_RULES = get_rules("language")

VIETNAMESE_MARKS = frozenset(_LANG_RULES["vietnamese_marks"])
VIETNAMESE_TONES = frozenset(_LANG_RULES["vietnamese_tones"])
VIETNAMESE_WORDS = set(_LANG_RULES["vietnamese_words"])
OTHER_LATIN_WORDS = set(_LANG_RULES["other_latin_words"])
TEXT_TYPES = set(_LANG_RULES["text_types"])
SENTENCE_ENDS = _LANG_RULES["sentence_ends"]
SENTENCE_CLOSERS = _LANG_RULES["sentence_closers"]
INLINE_MARKUP = _LANG_RULES["inline_markup"]
_IDENTIFIER_PUNCT = frozenset(_LANG_RULES["identifier_punct"])
_VIETNAMESE_STRUCTURAL_CHARS = _LANG_RULES["vietnamese_structural_chars"]
_VIETNAMESE_DIACRITICS = _LANG_RULES["vietnamese_diacritics"]
_VIETNAMESE_HOOK_DOT = tuple(_LANG_RULES["vietnamese_hook_dot"])
_VIETNAMESE_EXPLICIT_TAGS = set(_LANG_RULES["vietnamese_explicit_tags"])
_FOREIGN_MIN_LATIN_WORDS = _LANG_RULES["foreign_min_latin_words"]
_NON_LATIN_MIN_RUN = _LANG_RULES["non_latin_min_run"]
_DEFAULT_FOREIGN_LABEL = _LANG_RULES["default_foreign_label"]


def _is_identifier_char(ch):
    if ch in _IDENTIFIER_PUNCT or ch.isalpha() or ch.isdecimal():
        return True
    return unicodedata.category(ch).startswith("M")


def _iter_identifier_tokens(text):
    start = None
    for idx, ch in enumerate(text):
        if _is_identifier_char(ch):
            if start is None:
                start = idx
        elif start is not None:
            yield start, idx, text[start:idx]
            start = None
    if start is not None:
        yield start, len(text), text[start:]


def _words(text):
    chars = []
    for char in text.lower():
        if char.isalpha():
            chars.append(char)
        else:
            chars.append(" ")
    return "".join(chars).split()


def _profile(text):
    text = unicodedata.normalize("NFC", text)
    latin = 0
    other = 0
    strong_vi = False
    typical_tones = 0
    words = _words(text)
    for char in text:
        if not char.isalpha():
            continue
        if unicodedata.name(char, "").startswith("LATIN"):
            latin += 1
            decomposed = unicodedata.normalize("NFD", char)
            tones = any(mark in VIETNAMESE_TONES for mark in decomposed[1:])
            structural = ord(char) > 127 and any((
                char in _VIETNAMESE_STRUCTURAL_CHARS, char in VIETNAMESE_MARKS and tones,
                any(mark in _VIETNAMESE_DIACRITICS for mark in decomposed[1:]) and tones,
                _VIETNAMESE_HOOK_DOT[0] in decomposed or _VIETNAMESE_HOOK_DOT[1] in decomposed,
            ))
            if structural:
                strong_vi = True
            if tones:
                typical_tones += 1
        else:
            other += 1
    accentless = ["".join(ch for ch in unicodedata.normalize("NFD", word) if ch.isalpha())
                  .replace("đ", "d") for word in words]
    vi = sum(word in VIETNAMESE_WORDS for word in accentless)
    foreign = sum(word in OTHER_LATIN_WORDS for word in words)
    return latin, other, strong_vi or (typical_tones and vi), vi, foreign


def _foreign_from_profile(profile):
    latin, other, marked, vi, foreign = profile
    if other and (not latin or other >= latin and not marked and not vi):
        return "non_latin"
    if not other and not marked and foreign >= _FOREIGN_MIN_LATIN_WORDS and foreign > vi + 1:
        return "non_vietnamese_latin"
    return None


def foreign_language(text):
    """Keep ambiguous Latin text in context, including Vietnamese without dấu."""
    return _foreign_from_profile(_profile(text))


def _sentence_spans(text):
    """Yield spans at sentence punctuation followed by whitespace."""
    start = 0
    body_end = len(text.rstrip())
    for position, char in enumerate(text):
        if char not in SENTENCE_ENDS:
            continue
        end = position + 1
        while end < len(text) and text[end] in SENTENCE_CLOSERS:
            end += 1
        if end < body_end and text[end].isspace():
            # A numbered item keeps its rendered label with its own sentence.
            if char == "." and text[start:position].strip().isdecimal():
                continue
            yield start, end
            start = end
    if start < len(text):
        yield start, len(text)


def _append_span(spans, start, end, language):
    if spans and spans[-1][1] == start and spans[-1][2] == language:
        spans[-1] = (spans[-1][0], end, language)
    else:
        spans.append((start, end, language))


def foreign_spans(text):
    """Find foreign lines, sentences and script runs in source coordinates."""
    spans = []
    offset = 0
    for line in text.splitlines(keepends=True):
        end = offset + len(line)
        if not line.strip():
            offset = end
            continue
        # A clear sentence can be attached even on a line with separate
        # Vietnamese text. Keep ambiguous neighbouring sentences in context.
        for left, right in _sentence_spans(line):
            value = line[left:right]
            profile = _profile(value)
            language = _foreign_from_profile(profile)
            if language and (language != "non_latin" or not profile[0]):
                _append_span(spans, offset + left, offset + right, language)
                continue
            for first, last in _non_latin_runs(value):
                _append_span(spans, offset + left + first, offset + left + last, "non_latin")
        offset = end
    return spans


def _mask_protected_text(text, spans):
    """Mask formulas and mixed-script identifiers without changing offsets."""
    parts = []
    start = 0
    for left, right in spans:
        parts.extend((text[start:left], INLINE_MARKUP * (right - left)))
        start = right
    parts.append(text[start:])
    return "".join(parts)


def _non_latin_runs(text):
    left = None
    last = None
    count = 0
    for index, char in enumerate(text):
        if char.isalpha():
            if not unicodedata.name(char, "").startswith("LATIN"):
                if left is None:
                    left = index
                last = index + 1
                count += 1
            else:
                if count >= _NON_LATIN_MIN_RUN:
                    yield left, last
                left, count = None, 0
        elif char == INLINE_MARKUP:
            if count >= _NON_LATIN_MIN_RUN:
                yield left, last
            left, count = None, 0
        elif left is not None and unicodedata.category(char).startswith("M"):
            last = index + 1
        elif char.isdecimal() or not char.isspace():
            if count >= _NON_LATIN_MIN_RUN:
                yield left, last
            left, count = None, 0
    if count >= _NON_LATIN_MIN_RUN:
        yield left, last


def _projection_gap(text, left, right):
    return 0 < left < right < len(text) and not (text[left - 1].isspace() or text[right].isspace())


def without_foreign_spans(text, spans):
    if not spans:
        return text
    parts = []
    start = 0
    for left, right, _ in spans:
        parts.append(text[start:left])
        if _projection_gap(text, left, right):
            parts.append(" ")
        start = right
    parts.append(text[start:])
    return "".join(parts)


def projected_offset(text, spans, offset):
    """Translate an offset in retained source text into its search projection."""
    removed = sum(right - left - _projection_gap(text, left, right)
                  for left, right, _ in spans if right <= offset)
    return offset - removed


def vietnamese_projection(block):
    if block.get("foreign_language"):
        return ""
    return without_foreign_spans(block.get("text", ""), block.get("foreign_spans", []))


def project_vietnamese(text):
    """Exclude detected foreign text while preserving formulas and mixed-script identifiers."""
    return without_foreign_spans(text, source_foreign_spans(text))


def project_search_text(text, *, search_language="vi"):
    if search_language == "all":
        return text
    return project_vietnamese(text)


def _protected_spans(text):
    spans = protected_markup_spans(text)
    # A script transition within an identifier (such as 12中文-34 or AB中文CD)
    # is insufficient evidence to remove anything from the token.
    for left, right, token in _iter_identifier_tokens(text):
        if token.isascii():
            continue
        non_latin = False
        identifier = False
        for char in token:
            if char.isalpha():
                if unicodedata.name(char, "").startswith("LATIN"):
                    identifier = True
                else:
                    non_latin = True
            elif char.isdecimal():
                identifier = True
            if non_latin and identifier:
                spans.append((left, right))
                break
    merged = []
    for left, right in sorted(spans):
        if merged and left <= merged[-1][1]:
            start, end = merged[-1]
            merged[-1] = (start, max(right, end))
        else:
            merged.append((left, right))
    return merged


def source_foreign_spans(text):
    protected_spans = _protected_spans(text)
    spans = foreign_spans(_mask_protected_text(text, protected_spans))
    if not spans or not protected_spans:
        return spans
    # A foreign sentence can surround a protected token. Keep that token
    # searchable even when the surrounding language is filtered.
    result = []
    protected = iter(protected_spans)
    first, last = next(protected, (len(text), len(text)))
    for left, right, language in spans:
        while last <= left and first < len(text):
            first, last = next(protected, (len(text), len(text)))
        while first < right:
            if left < first:
                _append_span(result, left, first, language)
            left = max(left, last)
            if last >= right:
                break
            first, last = next(protected, (len(text), len(text)))
        if left < right:
            _append_span(result, left, right, language)
    return result


def classify_text_blocks(blocks, *, search_language="vi"):
    for block in blocks:
        if block["type"] not in TEXT_TYPES:
            continue
        block.pop("foreign_language", None)
        block.pop("foreign_spans", None)
        if search_language == "all":
            continue
        text = block.get("text", "")
        if not text.strip():
            continue
        explicit = block.get("source_language")
        language = None
        if isinstance(explicit, str) and explicit.strip():
            language = explicit.strip().casefold()
            if language.split("-", 1)[0] in _VIETNAMESE_EXPLICIT_TAGS:
                language = None
        # Word's proofing language can reflect the active keyboard or a
        # template. Use it only to label text independently classified as foreign.
        spans = source_foreign_spans(text)
        if not spans:
            continue
        if without_foreign_spans(text, spans).strip():
            block["foreign_spans"] = spans
        else:
            kinds = {kind for _, _, kind in spans}
            if language:
                resolved = language
            elif len(kinds) == 1:
                resolved = kinds.pop()
            else:
                resolved = _DEFAULT_FOREIGN_LABEL
            block["foreign_language"] = resolved
