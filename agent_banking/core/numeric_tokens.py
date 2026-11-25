"""Deterministic Vietnamese currency & percentage canonical token extractor.

DESIGN INVARIANTS (GEMINI.md):
- 0% REGEX LAW: Pure string and character parsing only.
- ZERO TERNARY: Explicit if/else blocks only.
- NO __INIT__.PY: Direct module imports.
"""

import unicodedata

_SPLIT_DELIMS = (
    "/", "|", ";", "(", ")", "[", "]", "{", "}",
    ":", "!", "?", "\"", "'", "\u201c", "\u201d",
    "\u2018", "\u2019", "\n", "\r", "\t",
)
_TRIM_PUNCT = ",.;:()[]{}\"'\u201c\u201d\u2018\u2019!?*+<>="

_MULT_UNITS = (
    (frozenset(("t\u1ef7", "ty", "t\u1ec9", "b")), 1_000_000_000),
    (frozenset(("tri\u1ec7u", "trieu", "tr", "m")), 1_000_000),
    (frozenset(("ngh\u00ecn", "ng\u00e0n", "nghin", "ngan", "k")), 1_000),
    (frozenset(("\u0111", "\u0111\u1ed3ng", "dong", "vnd", "vn\u0111")), 1),
)
_PERCENT_SPACED_SECOND = frozenset(("tram", "tr\u0103m"))
_PERCENT_SPACED_FIRST = frozenset(("phan", "ph\u1ea7n"))


def _clean_token(tok: str) -> str:
    return tok.strip(_TRIM_PUNCT)


def _is_digits_only(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        if not ("0" <= ch <= "9"):
            return False
    return True


def _parse_number(raw_num: str, allow_decimal: bool = True):
    """Parse Vietnamese formatted number (thousands '.' or ',' and decimals) into float/int."""
    if not raw_num:
        return None
    for ch in raw_num:
        if not (("0" <= ch <= "9") or ch in (".", ",")):
            return None
    if raw_num[0] in (".", ",") or raw_num[-1] in (".", ","):
        return None

    dot_cnt = raw_num.count(".")
    comma_cnt = raw_num.count(",")
    if dot_cnt > 0 and comma_cnt > 0:
        last_dot = raw_num.rfind(".")
        last_comma = raw_num.rfind(",")
        if last_comma > last_dot:
            dec_part = raw_num[last_comma + 1:]
            int_part = raw_num[:last_comma].replace(".", "").replace(",", "")
            if allow_decimal and _is_digits_only(int_part) and _is_digits_only(dec_part):
                return float(f"{int_part}.{dec_part}")
            return None
        dec_part = raw_num[last_dot + 1:]
        int_part = raw_num[:last_dot].replace(",", "").replace(".", "")
        if allow_decimal and _is_digits_only(int_part) and _is_digits_only(dec_part):
            return float(f"{int_part}.{dec_part}")
        return None

    if dot_cnt > 1 or comma_cnt > 1:
        sep = "."
        if comma_cnt > 1:
            sep = ","
        segs = raw_num.split(sep)
        if not _is_digits_only(segs[0]) or not (1 <= len(segs[0]) <= 3) or segs[0][0] == "0":
            return None
        for seg in segs[1:]:
            if not _is_digits_only(seg) or len(seg) != 3:
                return None
        return int("".join(segs))

    if dot_cnt == 1 or comma_cnt == 1:
        sep = "."
        if comma_cnt == 1:
            sep = ","
        left, right = raw_num.split(sep)
        if not _is_digits_only(left) or not _is_digits_only(right):
            return None
        if len(right) == 3 and left != "0" and 1 <= len(left) <= 3:
            return int(left + right)
        if allow_decimal:
            return float(f"{left}.{right}")
        if len(right) == 3 and len(left) <= 3:
            return int(left + right)
        return None

    if _is_digits_only(raw_num):
        return int(raw_num)
    return None


def _is_standalone_grouped_int(raw_num: str) -> bool:
    """True strictly for thousand-grouped currency numbers like 50.000.000 or 500.000."""
    dot_cnt = raw_num.count(".")
    comma_cnt = raw_num.count(",")
    if dot_cnt > 0 and comma_cnt > 0:
        return False
    if dot_cnt == 0 and comma_cnt == 0:
        return False
    sep = "."
    if comma_cnt > 0:
        sep = ","
    segs = raw_num.split(sep)
    if len(segs) < 2:
        return False
    if not _is_digits_only(segs[0]) or not (1 <= len(segs[0]) <= 3) or segs[0][0] == "0":
        return False
    for seg in segs[1:]:
        if not _is_digits_only(seg) or len(seg) != 3:
            return False
    return True


def _unit_multiplier(unit_str: str) -> int:
    for tokens, mult in _MULT_UNITS:
        if unit_str in tokens:
            return mult
    return 0


def _format_pct_token(raw_num: str):
    if not raw_num:
        return None
    norm = raw_num.replace(",", ".")
    if "." in norm:
        parts = norm.split(".")
        if len(parts) != 2 or not _is_digits_only(parts[0]) or not _is_digits_only(parts[1]):
            return None
        dec = parts[1].rstrip("0")
        int_part = str(int(parts[0]))
        if dec:
            return f"~pct_{int_part}_{dec}"
        return f"~pct_{int_part}"
    if _is_digits_only(norm):
        return f"~pct_{int(norm)}"
    return None


def extract_canonical_numeric_tokens(text: str, max_tokens: int = 32) -> list:
    """Extract canonical numeric tokens (~vnd_50000000, ~pct_0_03) without regex or ternaries."""
    if not text:
        return []
    work = unicodedata.normalize("NFC", str(text)).lower()
    for delim in _SPLIT_DELIMS:
        if delim in work:
            work = work.replace(delim, " ")
    raw_words = work.split()
    if not raw_words:
        return []

    out = []
    seen = set()

    def _add(tok: str):
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)

    n_words = len(raw_words)
    for idx in range(n_words):
        if len(out) >= max_tokens:
            break
        tok = _clean_token(raw_words[idx])
        if not tok:
            continue

        # 1. Attached percentage (e.g., 0,03% / 0.03%)
        if tok.endswith("%") and len(tok) > 1:
            pct_tok = _format_pct_token(tok[:-1].strip())
            if pct_tok:
                _add(pct_tok)
                continue

        # 2. Spaced percentage (e.g., 0,03 % or 0,03 phần trăm)
        if idx + 1 < n_words:
            nxt = _clean_token(raw_words[idx + 1])
            if nxt == "%":
                pct_tok = _format_pct_token(tok)
                if pct_tok:
                    _add(pct_tok)
                    continue
            if idx + 2 < n_words and nxt in _PERCENT_SPACED_FIRST:
                nxt2 = _clean_token(raw_words[idx + 2])
                if nxt2 in _PERCENT_SPACED_SECOND:
                    pct_tok = _format_pct_token(tok)
                    if pct_tok:
                        _add(pct_tok)
                        continue

        # 3. Attached VND amount unit (e.g., 50tr, 500.000đ, 50.000.000vnđ)
        cut = -1
        for pos, ch in enumerate(tok):
            if not (("0" <= ch <= "9") or ch in (".", ",")):
                cut = pos
                break
        if cut > 0:
            num_part = tok[:cut]
            unit_part = tok[cut:].strip()
            mult = _unit_multiplier(unit_part)
            if mult > 0:
                val = _parse_number(num_part, allow_decimal=True)
                if val is not None:
                    vnd_val = int(round(float(val) * mult))
                    if vnd_val >= 1_000 and vnd_val <= 10_000_000_000_000:
                        _add(f"~vnd_{vnd_val}")
                        continue

        # 4. Spaced VND amount unit (e.g., 50 triệu, 5 tỷ, 50.000.000 vnđ)
        if idx + 1 < n_words:
            nxt = _clean_token(raw_words[idx + 1])
            mult = _unit_multiplier(nxt)
            if mult > 0:
                allow_dec = mult > 1
                val = _parse_number(tok, allow_decimal=allow_dec)
                if val is not None:
                    vnd_val = int(round(float(val) * mult))
                    if vnd_val >= 1_000 and vnd_val <= 10_000_000_000_000:
                        _add(f"~vnd_{vnd_val}")
                        continue

        # 5. Standalone thousand-grouped VND figure (e.g., 50.000.000 or 500.000)
        if _is_standalone_grouped_int(tok):
            val = _parse_number(tok, allow_decimal=False)
            if isinstance(val, int) and val >= 10_000 and val <= 10_000_000_000_000:
                _add(f"~vnd_{val}")

    return out[:max_tokens]
