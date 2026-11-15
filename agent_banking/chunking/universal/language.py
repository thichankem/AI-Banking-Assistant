"""Conservative block language detection for Vietnamese chunk boundaries."""
import unicodedata

from .source_boundaries import protected_markup_spans


VIETNAMESE_MARKS = frozenset("ăâđêôơưĂÂĐÊÔƠƯ")
VIETNAMESE_TONES = frozenset("\u0300\u0301\u0303\u0309\u0323")
VIETNAMESE_WORDS = set("""
    va cua cho voi trong khong duoc nhung nguoi mot cac nay khi theo tiep tuc hien tai lieu
    tieng viet thuc phai nhu the nao nam ve co la da de neu thi hay xem them doi sau truoc
    tren duoi huong dan yeu cau noi dung
""".split())
OTHER_LATIN_WORDS = set("""
    the and with for this that which from is are was were when where have has please only should
    would must shall before after may not will can your you our these those step instructions requirements
    les des une est dans avec pour vous aux pas sur eine einer auch nicht sind und der die das
    dem den ein einen für mit ist zur sie bei auf para las los una del con por que esta este esto
    ellos como está donde antes después durante può della degli delle uno sono nei nella ogni com
    uma dos pelo pela isso você também não
""".split())
TEXT_TYPES = {"heading", "paragraph", "caption"}
SENTENCE_ENDS = ".!?…"
SENTENCE_CLOSERS = "\"'”’»)]}"
INLINE_MARKUP = "\ufffc"


def _words(text):
    return "".join(char if char.isalpha() else " " for char in text.lower()).split()


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
                char in "ăđơưĂĐƠƯ", char in VIETNAMESE_MARKS and tones,
                any(mark in "\u0302\u0306\u031b" for mark in decomposed[1:]) and tones,
                "\u0309" in decomposed or "\u0323" in decomposed,
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
    if not other and not marked and foreign >= 2 and foreign > vi + 1:
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


def _without_technical_markup(text, spans):
    """Mask formulas while keeping all language span offsets in source coordinates."""
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
                if count >= 2:
                    yield left, last
                left, count = None, 0
        elif char == INLINE_MARKUP:
            if count >= 2:
                yield left, last
            left, count = None, 0
        elif left is not None and not char.isdecimal() and not char.isspace():
            last = index + 1
    if count >= 2:
        yield left, last


def without_foreign_spans(text, spans):
    if not spans:
        return text
    parts = []
    start = 0
    for left, right, _ in spans:
        parts.append(text[start:left])
        start = right
    parts.append(text[start:])
    return "".join(parts)


def vietnamese_projection(block):
    if block.get("foreign_language"):
        return ""
    return without_foreign_spans(block.get("text", ""), block.get("foreign_spans", []))


def project_vietnamese(text):
    """Exclude detected foreign text while keeping technical markup."""
    return without_foreign_spans(text, source_foreign_spans(text))


def source_foreign_spans(text):
    markup = protected_markup_spans(text)
    spans = foreign_spans(_without_technical_markup(text, markup))
    if not spans or not markup:
        return spans
    # A foreign sentence can surround a formula even when classification
    # ignored its symbols. Keep the formula searchable and its text intact.
    result = []
    protected = iter(markup)
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


def classify_text_blocks(blocks):
    for block in blocks:
        if block["type"] not in TEXT_TYPES:
            continue
        block.pop("foreign_language", None)
        block.pop("foreign_spans", None)
        text = block.get("text", "")
        if not text.strip():
            continue
        explicit = block.get("source_language")
        language = None
        if isinstance(explicit, str) and explicit.strip():
            language = explicit.strip().casefold()
            if language.split("-", 1)[0] in {"vi", "vie", "und", "zxx", "x"}:
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
            block["foreign_language"] = language or (kinds.pop() if len(kinds) == 1 else "non_vietnamese")
