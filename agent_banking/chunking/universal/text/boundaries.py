"""Source offsets at safe Unicode and authored-markup boundaries."""
import unicodedata

from agent_banking.chunking.config.settings import get_rules

_BOUNDARY_RULES = get_rules("boundaries")
MARKUP_DELIMITERS = tuple(tuple(pair) for pair in _BOUNDARY_RULES["markup_delimiters"])


def _is_regional_indicator(code):
    return 0x1F1E6 <= code <= 0x1F1FF


def _is_extend(char, code):
    if code in (0x200C, 0x200D) or 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF:
        return True
    if unicodedata.category(char) in ("Mn", "Me", "Mc"):
        return True
    return False


def _is_control(char, code):
    if code in (0x200C, 0x200D):
        return False
    return char in ("\r", "\n") or unicodedata.category(char) in ("Cc", "Zl", "Zp")


def _is_pictographic(code):
    return (
        0x1F000 <= code <= 0x1FAFF
        or 0x2600 <= code <= 0x27BF
        or 0x2300 <= code <= 0x23FF
        or 0x2B00 <= code <= 0x2BFF
        or code in (0x203C, 0x2049, 0x2122, 0x2139, 0x3030, 0x303D, 0x3297, 0x3299)
    )


def _hangul_kind(code):
    if 0x1100 <= code <= 0x115F or 0xA960 <= code <= 0xA97C:
        return "L"
    if 0x1160 <= code <= 0x11A7 or 0xD7B0 <= code <= 0xD7C6:
        return "V"
    if 0x11A8 <= code <= 0x11FF or 0xD7CB <= code <= 0xD7FB:
        return "T"
    if 0xAC00 <= code <= 0xD7A3:
        if (code - 0xAC00) % 28 == 0:
            return "LV"
        return "LVT"
    return None


def _grapheme_end_positions(text):
    """Yield end indices of Unicode Extended Grapheme Clusters without regex."""
    n = len(text)
    if not n:
        return
    ri_odd = False
    pic_extend = False
    seen_zwj = False
    for idx in range(n):
        ch = text[idx]
        code = ord(ch)
        if idx == 0:
            ri_odd = _is_regional_indicator(code)
            pic_extend = _is_pictographic(code)
            seen_zwj = False
            continue
        prev = text[idx - 1]
        prev_code = ord(prev)
        can_join = False
        if prev == "\r" and ch == "\n":
            can_join = True
        elif _is_control(prev, prev_code) or _is_control(ch, code):
            can_join = False
        elif _is_extend(ch, code):
            can_join = True
        elif seen_zwj and _is_pictographic(code):
            can_join = True
        elif ri_odd and _is_regional_indicator(code):
            can_join = True
        else:
            ph = _hangul_kind(prev_code)
            ch_h = _hangul_kind(code)
            if ph == "L" and ch_h in ("L", "V", "LV", "LVT"):
                can_join = True
            elif ph in ("LV", "V") and ch_h in ("V", "T"):
                can_join = True
            elif ph in ("LVT", "T") and ch_h == "T":
                can_join = True

        if not can_join:
            yield idx
            ri_odd = _is_regional_indicator(code)
            pic_extend = _is_pictographic(code)
            seen_zwj = False
        else:
            if _is_regional_indicator(code):
                ri_odd = not ri_odd
            if code == 0x200D and pic_extend:
                seen_zwj = True
            else:
                seen_zwj = False
            if _is_pictographic(code):
                pic_extend = True
            elif not _is_extend(ch, code):
                pic_extend = False
    yield n


def _escaped_at(text, position):
    start = position
    while start > 0 and text[start - 1] == "\\":
        start -= 1
    return (position - start) % 2 == 1


def protected_markup_spans(text):
    """Identify authored Markdown math and the base words of inline scripts."""
    if "<" not in text and "$" not in text and "\\" not in text:
        return []
    last_closer = {closer: text.rfind(closer) for _, closer in MARKUP_DELIMITERS}
    spans = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] not in "<$\\":
            cursor += 1
            continue
        for opener, closer in MARKUP_DELIMITERS:
            if last_closer[closer] <= cursor + len(opener) or not text.startswith(opener, cursor) or _escaped_at(text, cursor):
                continue
            if opener == "$" and ((cursor and text[cursor - 1] == "$") or text.startswith("$$", cursor)):
                continue
            end = text.find(closer, cursor + len(opener))
            while closer in ("$", "$$", "\\)", "\\]") and end > cursor and _escaped_at(text, end):
                end = text.find(closer, end + len(closer))
            if end <= cursor + len(opener):
                continue
            if opener == "$" and (text[cursor + 1].isspace() or text[end - 1].isspace()
                                  or end + 1 < len(text) and text[end + 1].isdigit()):
                continue
            left, right = cursor, end + len(closer)
            if opener in ("<sup>", "<sub>"):
                while left > 0 and not text[left - 1].isspace():
                    left -= 1
                while right < len(text) and (text[right].isalnum() or unicodedata.category(text[right]).startswith("M")):
                    right += 1
            while spans and left < spans[-1][1]:
                previous, _ = spans.pop()
                left = min(left, previous)
            spans.append((left, right))
            cursor = right
            break
        else:
            cursor += 1
    return spans


def safe_boundaries(text, *, protect_markup=True):
    """Use Unicode grapheme boundaries outside protected authored markup."""
    boundaries = [0]
    if protect_markup:
        blocked = iter(protected_markup_spans(text))
    else:
        blocked = iter(())
    left, right = next(blocked, (len(text), len(text)))
    for position in _grapheme_end_positions(text):
        while position >= right:
            following = next(blocked, None)
            if following is None:
                left = right = len(text)
                break
            left, right = following
        if not left < position < right:
            boundaries.append(position)
    return boundaries
