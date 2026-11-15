"""Source offsets at safe Unicode and authored-markup boundaries."""
import unicodedata

import regex


MARKUP_DELIMITERS = (("<sup>", "</sup>"), ("<sub>", "</sub>"), ("\\(", "\\)"), ("\\[", "\\]"),
                     ('<code data-format="omml">', "</code>"), ("$$", "$$"), ("$", "$"))
GRAPHEME = regex.compile(r"\X")


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
    blocked = iter(protected_markup_spans(text) if protect_markup else ())
    left, right = next(blocked, (len(text), len(text)))
    for grapheme in GRAPHEME.finditer(text):
        position = grapheme.end()
        while position >= right:
            following = next(blocked, None)
            if following is None:
                left = right = len(text)
                break
            left, right = following
        if not left < position < right:
            boundaries.append(position)
    return boundaries
