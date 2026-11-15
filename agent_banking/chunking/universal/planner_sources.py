"""Helpers for source regions, exact offsets, and visible text."""

def is_furniture_block(block):
    region = block.get("provenance", {}).get("region", "body")
    return str(region).startswith("furniture:")


def section_region(block):
    return block.get("provenance", {}).get("region", "body").split(":", 1)[0]


def build_lead_in(blocks):
    non_empty = [b for b in (blocks or []) if has_visible_text(b.get("text", ""))]
    if not non_empty:
        return "", [], []
    parts = []
    spans = []
    ids = []
    offset = 0
    for b in non_empty:
        txt = b["text"]
        b_len = len(txt)
        source_start = b.get("source_start", 0)
        parts.append(txt)
        ids.append(b["id"])
        spans.append({
            "block_id": b["id"],
            "start": source_start,
            "end": source_start + b_len,
            "chunk_start": offset,
            "chunk_end": offset + b_len,
        })
        offset += b_len + 2
    return "\n\n".join(parts), spans, ids


def find_word_starts(text):
    starts = []
    in_word = False
    for i, ch in enumerate(text):
        if not ch.isspace():
            if not in_word:
                starts.append(i)
                in_word = True
        else:
            in_word = False
    return starts


def find_sentence_gap_ends(text, use_decision_for_dots=True, cache=None):
    ends = []
    ambiguous_dots = []
    i = 0
    word_start = 0
    punctuation = ('.', '!', '?', '…', ';')
    closing = ('"', "'", '”', '’', '»', ')', ']', '}')
    n = len(text)
    while i < n:
        if text[i] in ('\r', '\n'):
            while i < n and text[i] in ('\r', '\n'):
                i += 1
            ends.append(i)
            word_start = i
        elif text[i] in punctuation:
            dot_index = i
            punct_char = text[i]
            token_length = i + 1 - word_start
            raw_token = text[word_start:i + 1] if token_length <= 6 else ""
            i += 1
            while i < n and text[i] in closing:
                i += 1
            if i < n and text[i].isspace():
                line_gap = False
                while i < n and text[i].isspace():
                    line_gap = line_gap or text[i] in ('\r', '\n')
                    i += 1
                word_start = i
                if line_gap:
                    ends.append(i)
                elif i < n and (text[i].isalpha() or text[i].isdigit()):
                    if punct_char == '.' and token_length <= 6:
                        if use_decision_for_dots:
                            ambiguous_dots.append((dot_index, i, raw_token))
                    else:
                        ends.append(i)
        else:
            if text[i].isspace():
                word_start = i + 1
            i += 1
    if ambiguous_dots:
        from .decision_engine import decide_period_boundaries

        valid_dot_breaks = decide_period_boundaries(text, ambiguous_dots, cache=cache)
        for gap_end in valid_dot_breaks:
            ends.append(gap_end)
        ends.sort()
    return ends


def normalize(text):
    return " ".join(str(text).split())


def has_visible_text(text):
    return any(not char.isspace() and char not in "\u200b\u200c\u200d\ufeff" for char in text)
