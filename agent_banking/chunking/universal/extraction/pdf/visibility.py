"""Identify native glyphs that the page does not visibly paint."""


def _glyph_key(text, origin):
    return text, tuple(round(value, 3) for value in origin)


def hidden_glyphs(page):
    """Find nonpainting glyphs and glyphs covered by later opaque rectangles."""
    from pymupdf import Rect

    log = page.get_bboxlog()
    painted_text = [index for index, (kind, _) in enumerate(log)
                    if kind in ("fill-text", "stroke-text")]
    has_hidden = any(kind == "ignore-text" for kind, _ in log)
    covers = []
    if painted_text and any(kind == "fill-path" for kind, _ in log[painted_text[0] + 1:]):
        for drawing in page.get_drawings(extended=True):
            if (drawing["type"] in ("f", "fs") and drawing.get("fill_opacity") == 1
                    and not drawing.get("level") and len(drawing["items"]) == 1
                    and drawing["items"][0][0] == "re"):
                covers.append((drawing["seqno"], drawing["items"][0][1]))
    if not covers and not has_hidden:
        return set()

    hidden = set()
    visible = set()
    for span in page.get_texttrace():
        sequence = span["seqno"]
        invisible = span["type"] == 3 or span.get("opacity") == 0
        relevant = [cover for position, cover in covers
                    if position > sequence and cover.intersects(Rect(span["bbox"]))]
        for codepoint, _, origin, bounds in span["chars"]:
            if not 0 <= codepoint <= 0x10FFFF:
                continue
            key = _glyph_key(chr(codepoint), origin)
            if invisible or any(cover.contains(Rect(bounds)) for cover in relevant):
                hidden.add(key)
            else:
                visible.add(key)
    # A matching visible glyph takes precedence over an identical overlay or
    # OCR glyph at the same position. Span alpha still excludes nonpainted runs.
    return hidden - visible


def visible_span(span, hidden):
    """Return text and its visible character bounds for native extraction."""
    if span.get("alpha") == 0:
        return "", []
    if "chars" not in span:
        return span["text"], [span["bbox"]]

    characters = []
    bounds = []
    skipped = False
    for character in span["chars"]:
        text = character["c"]
        if _glyph_key(text, character["origin"]) in hidden:
            skipped = True
            continue
        if skipped and characters and not characters[-1].isspace() and not text.isspace():
            characters.append(" ")
        skipped = False
        characters.append(text)
        bounds.append(character["bbox"])
    return "".join(characters), bounds
