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


from agent_banking.chunking.config.settings import get_rules

_BOUND_RULES = get_rules("boundaries")
_SHORT_ABBREVIATIONS = frozenset(_BOUND_RULES["short_abbreviations"])
_MAX_ABBREV_TOKEN_LEN = _BOUND_RULES["max_abbrev_token_len"]
_SENTENCE_PUNCTUATION = tuple(_BOUND_RULES["sentence_punctuation"])
_SENTENCE_CLOSING = tuple(_BOUND_RULES["sentence_closing"])
_INVISIBLE_TEXT_CHARS = _BOUND_RULES["invisible_text_chars"]


def _is_probable_abbreviation(raw_token):
    lowered = raw_token.strip().casefold()
    if not lowered or lowered in _SHORT_ABBREVIATIONS:
        return True
    stem = lowered[:-1]
    if not stem or stem.isdecimal():
        return True
    # Dots inside a short token (e.g. "v.v.", "A.B.") or single-character
    # initials (e.g. "A.", "I.") are typically prefixes or abbreviations.
    if "." in stem or len(stem) == 1:
        return True
    return False


def find_sentence_gap_ends(text, use_decision_for_dots=True, cache=None, ir=None):
    ends = []
    ambiguous_dots = []
    i = 0
    word_start = 0
    punctuation = _SENTENCE_PUNCTUATION
    closing = _SENTENCE_CLOSING
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
            if token_length <= _MAX_ABBREV_TOKEN_LEN:
                raw_token = text[word_start:i + 1]
            else:
                raw_token = ""
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
                    if punct_char == '.' and token_length <= _MAX_ABBREV_TOKEN_LEN:
                        if _is_probable_abbreviation(raw_token):
                            if use_decision_for_dots and not raw_token[:-1].strip().isdecimal():
                                ambiguous_dots.append((dot_index, i, raw_token))
                        elif text[i].isupper() or text[i].isdigit():
                            ends.append(i)
                        elif use_decision_for_dots:
                            ambiguous_dots.append((dot_index, i, raw_token))
                    else:
                        ends.append(i)
        else:
            if text[i].isspace():
                word_start = i + 1
            i += 1
    if ambiguous_dots:
        from ..enrichment.decisions.engine import decide_period_boundaries

        valid_dot_breaks = decide_period_boundaries(text, ambiguous_dots, cache=cache, ir=ir)
        for gap_end in valid_dot_breaks:
            ends.append(gap_end)
        ends.sort()
    return ends


def normalize(text):
    return " ".join(str(text).split())


def has_visible_text(text):
    return any(not char.isspace() and char not in _INVISIBLE_TEXT_CHARS for char in text)
