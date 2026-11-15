"""Source offsets at safe Unicode and authored-markup boundaries."""
import unicodedata


MARKUP_DELIMITERS = (("<sup>", "</sup>"), ("<sub>", "</sub>"), ("\\(", "\\)"), ("\\[", "\\]"),
                     ('<code data-format="omml">', "</code>"), ("$$", "$$"), ("$", "$"))


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
    """Preserve Vietnamese combining marks and authored inline math delimiters."""
    boundaries = [0]
    blocked = iter(protected_markup_spans(text) if protect_markup else ())
    left, right = next(blocked, (len(text), len(text)))
    regional_count = 0
    for position in range(1, len(text)):
        while position >= right:
            left, right = next(blocked, (len(text), len(text)))
        char, previous = text[position], text[position - 1]
        if "\U0001f1e6" <= previous <= "\U0001f1ff":
            regional_count += 1
        else:
            regional_count = 0
        if (left < position < right or previous == "\u200d" or char == "\u200d"
                or unicodedata.category(char).startswith("M")
                or (regional_count % 2 and "\U0001f1e6" <= char <= "\U0001f1ff")
                or "\U000e0020" <= char <= "\U000e007f"
                or "\U0001f3fb" <= char <= "\U0001f3ff" or (previous == "\r" and char == "\n")):
            continue
        boundaries.append(position)
    if text:
        boundaries.append(len(text))
    return boundaries
