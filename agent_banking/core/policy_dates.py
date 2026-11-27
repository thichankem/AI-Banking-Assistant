"""Effective-date parsing and query as-of extraction (0% Regex, Zero Ternary)."""

from datetime import date, datetime, timedelta, timezone

from agent_banking.core.search_text import search_key

_VIETNAM = timezone(timedelta(hours=7))

_SELF_DOCUMENT_CUES = (
    "quy dinh nay",
    "quy che nay",
    "quy trinh nay",
    "van ban nay",
    "tai lieu nay",
    "bieu phi nay",
    "chinh sach nay",
    "huong dan nay",
    "dieu khoan nay",
    "quy tac nay",
    "quyet dinh nay",
)

_PROMO_CUES = (
    "uu dai",
    "khuyen mai",
    "khuyen nghi",
    "chuong trinh",
    "mien phi nam dau",
)

_EXTERNAL_SUBJECT_CUES = (
    "thong tu",
    "nghi dinh",
    "luat ",
    "bo luat",
    "phap lenh",
    "chi thi",
    "cong van",
    "nghi quyet",
)

_NEGATION_CUES = (
    "khong co hieu luc",
    "chua co hieu luc",
    "khong ap dung",
)

_START_TRIGGERS = (
    "co hieu luc thi hanh ke tu ngay",
    "co hieu luc thi hanh tu ngay",
    "co hieu luc ap dung thong nhat ke tu ngay",
    "co hieu luc ap dung ke tu ngay",
    "co hieu luc ap dung tu ngay",
    "co hieu luc ap dung ke tu",
    "co hieu luc ap dung tu",
    "co hieu luc ke tu ngay",
    "co hieu luc tu ngay",
    "hieu luc ap dung thong nhat ke tu ngay",
    "hieu luc ap dung ke tu ngay",
    "hieu luc ap dung tu ngay",
    "hieu luc thi hanh ke tu ngay",
    "hieu luc thi hanh tu ngay",
    "hieu luc ke tu ngay",
    "hieu luc tu ngay",
    "ngay co hieu luc la ngay",
    "ngay co hieu luc:",
    "ngay co hieu luc",
    "ngay hieu luc la ngay",
    "ngay hieu luc:",
    "ngay hieu luc",
    "ap dung ke tu ngay",
    "ap dung tu ngay",
)

_END_TRIGGERS = (
    "co hieu luc den het ngay",
    "co hieu luc den ngay",
    "hieu luc den het ngay",
    "hieu luc den ngay",
    "ap dung den het ngay",
    "ap dung den ngay",
    "den het ngay",
    "den ngay",
    "het hieu luc ke tu ngay",
    "het hieu luc tu ngay",
    "het hieu luc ngay",
)

_EVENT_BEFORE_CUES = (
    "giao dich",
    "chuyen tien",
    "chuyen khoan",
    "thanh toan",
    "khoan vay",
    "mo the",
    "phat hanh",
    "ky hop dong",
    "hop dong",
    "tra soat",
    "phat sinh",
    "bi treo",
    "su co",
    "rut tien",
    "gui tien",
    "tat toan",
)

_POLICY_AFTER_OR_BEFORE_CUES = (
    "hieu luc",
    "ap dung",
    "quy dinh",
    "chinh sach",
    "bieu phi",
    "quy che",
    "quy trinh",
    "tai thoi diem",
    "tinh den",
    "tinh tai",
    "ban hanh",
)

_CURRENT_TIME_PHRASES = (
    "hom nay",
    "hien nay",
    "hien gio",
    "bay gio",
    "hien tai",
)


def _split_digits_token(text, start_idx):
    """Consume contiguous ASCII digits starting at start_idx."""
    pos = start_idx
    n = len(text)
    while pos < n and text[pos].isdigit():
        pos += 1
    return text[start_idx:pos], pos


def _build_iso_date(year, month, day):
    """Validate and format YYYY-MM-DD, returning None if invalid."""
    if not (1990 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31):
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _try_parse_date_at(text, start_idx):
    """
    Attempt to parse a numeric or Vietnamese spelled-out calendar date at start_idx.
    Returns (iso_date, end_idx) or (None, start_idx).
    """
    n = len(text)
    pos = start_idx
    while pos < n and text[pos] in " \t,:;-":
        pos += 1
    if pos >= n:
        return None, start_idx

    low = text[pos:pos + 16].lower()
    if low.startswith("ngay "):
        pos += 5
        while pos < n and text[pos] == " ":
            pos += 1
        if pos >= n or not text[pos].isdigit():
            return None, start_idx

    if not text[pos].isdigit():
        return None, start_idx

    first_str, next_pos = _split_digits_token(text, pos)

    # Case 1: YYYY-MM-DD or YYYY/MM/DD
    if len(first_str) == 4 and next_pos < n and text[next_pos] in "-/":
        sep = text[next_pos]
        if next_pos + 1 < n and text[next_pos + 1].isdigit():
            mon_str, after_mon = _split_digits_token(text, next_pos + 1)
            if after_mon < n and text[after_mon] == sep and after_mon + 1 < n and text[after_mon + 1].isdigit():
                day_str, end_pos = _split_digits_token(text, after_mon + 1)
                if len(mon_str) in (1, 2) and len(day_str) in (1, 2):
                    if end_pos >= n or not text[end_pos].isdigit():
                        iso = _build_iso_date(int(first_str), int(mon_str), int(day_str))
                        if iso is not None:
                            return iso, end_pos

    # Case 2: DD/MM/YYYY, DD-MM-YYYY, or DD.MM.YYYY
    if len(first_str) in (1, 2) and next_pos < n and text[next_pos] in "/-.":
        sep = text[next_pos]
        if next_pos + 1 < n and text[next_pos + 1].isdigit():
            mon_str, after_mon = _split_digits_token(text, next_pos + 1)
            if after_mon < n and text[after_mon] == sep and after_mon + 1 < n and text[after_mon + 1].isdigit():
                year_str, end_pos = _split_digits_token(text, after_mon + 1)
                if len(mon_str) in (1, 2) and len(year_str) == 4:
                    if end_pos >= n or not text[end_pos].isdigit():
                        iso = _build_iso_date(int(year_str), int(mon_str), int(first_str))
                        if iso is not None:
                            return iso, end_pos

    # Case 3: DD thang MM nam YYYY (also covers unaccented or accented "tháng"/"năm")
    if len(first_str) in (1, 2):
        rest = search_key(text[next_pos:next_pos + 40])
        if rest.startswith("thang "):
            raw_after_day = text[next_pos:].lstrip()
            lower_after_day = search_key(raw_after_day)
            if lower_after_day.startswith("thang "):
                cur = next_pos
                while cur < n and text[cur] == " ":
                    cur += 1
                # skip word "thang" / "tháng"
                while cur < n and not text[cur].isspace():
                    cur += 1
                while cur < n and text[cur] == " ":
                    cur += 1
                if cur < n and text[cur].isdigit():
                    mon_str, after_mon = _split_digits_token(text, cur)
                    if len(mon_str) in (1, 2):
                        cur2 = after_mon
                        while cur2 < n and text[cur2] == " ":
                            cur2 += 1
                        if search_key(text[cur2:cur2 + 8]).startswith("nam "):
                            while cur2 < n and not text[cur2].isspace():
                                cur2 += 1
                            while cur2 < n and text[cur2] == " ":
                                cur2 += 1
                            if cur2 < n and text[cur2].isdigit():
                                year_str, end_pos = _split_digits_token(text, cur2)
                                if len(year_str) == 4 and (end_pos >= n or not text[end_pos].isdigit()):
                                    iso = _build_iso_date(int(year_str), int(mon_str), int(first_str))
                                    if iso is not None:
                                        return iso, end_pos

    return None, start_idx


def parse_calendar_date(raw_date):
    """Parse a calendar date string (YYYY-MM-DD, DD/MM/YYYY, DD tháng MM năm YYYY) -> ISO YYYY-MM-DD or None."""
    if not raw_date:
        return None
    text = str(raw_date).strip()
    iso, _ = _try_parse_date_at(text, 0)
    return iso


def _find_all_dates(text):
    """Return list of (start_idx, end_idx, iso_date) occurrences across text."""
    results = []
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos].isdigit():
            if pos == 0 or not text[pos - 1].isdigit():
                iso, end_pos = _try_parse_date_at(text, pos)
                if iso is not None:
                    results.append((pos, end_pos, iso))
                    pos = end_pos
                    continue
        pos += 1
    return results


def _has_month_year_without_full_date(norm_query, full_dates):
    """Detect imprecise month/year query expressions like 'thang 10/2025' or 'thang 10 nam 2025'."""
    if full_dates:
        return False
    tokens = norm_query.split()
    for idx, tok in enumerate(tokens):
        if tok == "thang" and idx + 1 < len(tokens):
            nxt = tokens[idx + 1]
            parts = nxt.replace("-", "/").split("/")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and len(parts[1]) == 4:
                mon = int(parts[0])
                yr = int(parts[1])
                if 1 <= mon <= 12 and 1990 <= yr <= 2099:
                    return True
            if nxt.isdigit() and 1 <= int(nxt) <= 12:
                if idx + 2 < len(tokens) and tokens[idx + 2].isdigit() and len(tokens[idx + 2]) == 4:
                    yr = int(tokens[idx + 2])
                    if 1990 <= yr <= 2099:
                        return True
                if idx + 3 < len(tokens) and tokens[idx + 2] == "nam" and tokens[idx + 3].isdigit() and len(tokens[idx + 3]) == 4:
                    yr = int(tokens[idx + 3])
                    if 1990 <= yr <= 2099:
                        return True
    return False


def _contains_token_phrase(padded_norm, phrase):
    """Check whether phrase appears as whole space-delimited tokens inside padded search_key text."""
    target = " " + phrase + " "
    return target in padded_norm


def _is_event_date(raw_text, start_idx, end_idx):
    """Distinguish a customer transaction date from a policy effective-date filter."""
    before_raw = raw_text[max(0, start_idx - 64):start_idx]
    before = " " + search_key(before_raw) + " "
    after_raw = raw_text[end_idx:min(len(raw_text), end_idx + 48)]
    after = " " + search_key(after_raw) + " "

    has_policy_cue = False
    for cue in _POLICY_AFTER_OR_BEFORE_CUES:
        if _contains_token_phrase(before[-48:], cue) or _contains_token_phrase(after[:40], cue):
            has_policy_cue = True
            break
    if has_policy_cue:
        return False

    tail_before = before[-40:]
    for ev in _EVENT_BEFORE_CUES:
        if _contains_token_phrase(tail_before, ev):
            return True
    return False


def policy_query_date(query, as_of=None, *, today=None):
    """
    Resolve (requested_policy_date, current_day) from explicit as_of or natural-language query.
    Raises ValueError on invalid_as_of, imprecise_as_of, or ambiguous_as_of.
    """
    if today is not None:
        if isinstance(today, datetime):
            current_day = today.date()
        else:
            current_day = today
    else:
        current_day = datetime.now(_VIETNAM).date()

    if as_of is not None and str(as_of).strip():
        raw_as_of = str(as_of).strip()
        parsed_iso = parse_calendar_date(raw_as_of)
        if parsed_iso is None:
            raise ValueError("invalid_as_of")
        return date.fromisoformat(parsed_iso), current_day

    raw_query = str(query or "").strip()
    if not raw_query:
        return None, current_day

    norm_query = search_key(raw_query)
    padded_norm = " " + norm_query + " "

    all_dates = _find_all_dates(raw_query)
    policy_isos = []
    seen_iso = set()
    for start_idx, end_idx, iso in all_dates:
        if not _is_event_date(raw_query, start_idx, end_idx):
            if iso not in seen_iso:
                seen_iso.add(iso)
                policy_isos.append(iso)

    if len(policy_isos) > 1:
        raise ValueError("ambiguous_as_of")

    if len(policy_isos) == 1:
        return date.fromisoformat(policy_isos[0]), current_day

    if _contains_token_phrase(padded_norm, "hom qua"):
        return current_day - timedelta(days=1), current_day

    # Prevent 'thực hiện tại (quầy)' -> 'thuc hien tai' from false-matching 'hien tai'.
    time_padded = padded_norm.replace(" thuc hien tai ", " ")
    for cur_phrase in _CURRENT_TIME_PHRASES:
        if _contains_token_phrase(time_padded, cur_phrase):
            return current_day, current_day

    if _has_month_year_without_full_date(norm_query, all_dates):
        raise ValueError("imprecise_as_of")

    return None, current_day


def _split_clauses(text):
    """Split text into sentence-like clauses without splitting numeric dates like 01.01.2026."""
    clauses = []
    current = []
    n = len(text)
    for idx, ch in enumerate(text):
        if ch in "\n\r;|•":
            piece = "".join(current).strip()
            if piece:
                clauses.append(piece)
            current = []
            continue
        if ch == ".":
            prev_is_dig = idx > 0 and text[idx - 1].isdigit()
            next_is_dig = idx + 1 < n and text[idx + 1].isdigit()
            if prev_is_dig and next_is_dig:
                current.append(ch)
            else:
                piece = "".join(current).strip()
                if piece:
                    clauses.append(piece)
                current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        clauses.append(tail)
    return clauses


def _strip_vietnamese_for_match(clause):
    """Map raw clause to lowercase ASCII-folded text and index tracking helper via search_key."""
    return search_key(clause)


def _extract_clause_dates(clause, source_is_circular=False):
    """
    Extract (is_self_doc, is_promo, start_iso, end_iso, snippet) from a single clause.
    Rejects negated clauses and references to external Circulars/Decrees/Laws.
    """
    norm = _strip_vietnamese_for_match(clause)
    if not norm:
        return None

    has_eff_kw = (
        "hieu luc" in norm
        or "ap dung ke tu ngay" in norm
        or "ap dung tu ngay" in norm
        or "co hieu luc den" in norm
        or "ap dung den" in norm
    )
    if not has_eff_kw:
        return None

    padded = " " + norm + " "
    for neg in _NEGATION_CUES:
        if neg in norm:
            return None

    is_self = False
    for cue in _SELF_DOCUMENT_CUES:
        if cue in norm:
            is_self = True
            break

    is_promo = False
    for promo in _PROMO_CUES:
        if promo in norm:
            is_promo = True
            break

    # Skip dates explicitly attached to another statutory/external norm ("Thông tư...", "Luật...", "Nghị định...")
    if not is_self and not source_is_circular:
        check_prefix = padded
        for trig in _START_TRIGGERS + _END_TRIGGERS:
            pos_t = padded.find(" " + trig + " ")
            if pos_t != -1:
                check_prefix = padded[:pos_t + 1]
                break
        for ext in _EXTERNAL_SUBJECT_CUES:
            if ext in check_prefix:
                return None

    # Find dates directly in original clause while inspecting the text window right before each date
    all_dates = _find_all_dates(clause)
    if not all_dates:
        return None

    start_iso = None
    end_iso = None
    for d_start, _d_end, iso in all_dates:
        prefix_raw = clause[max(0, d_start - 72):d_start]
        prefix_norm = search_key(prefix_raw)
        if not prefix_norm:
            continue

        # Ensure external statutory phrase wasn't right before THIS specific date
        if not is_self and not source_is_circular:
            ext_before = False
            for ext in _EXTERNAL_SUBJECT_CUES:
                if ext.strip() in prefix_norm:
                    ext_before = True
                    break
            if ext_before:
                continue

        matched_end = False
        for etrig in _END_TRIGGERS:
            if etrig in ("den het ngay", "den ngay") and start_iso is None and "hieu luc" not in prefix_norm and "ap dung" not in prefix_norm:
                continue
            if prefix_norm.endswith(etrig) or (etrig + " ngay") in prefix_norm[-45:]:
                if end_iso is None:
                    end_iso = iso
                matched_end = True
                break
        if matched_end:
            continue

        for strig in _START_TRIGGERS:
            clean_trig = strig.rstrip(":")
            if prefix_norm.endswith(clean_trig) or (clean_trig + " ngay") in prefix_norm[-45:] or clean_trig in prefix_norm[-48:]:
                if start_iso is None:
                    start_iso = iso
                break

    if start_iso is None and end_iso is None:
        return None
    if not is_self and is_promo:
        return None
    snippet = " ".join(clause.split())
    if len(snippet) > 160:
        snippet = snippet[:160].strip()
    return {
        "is_self": is_self,
        "is_promo": is_promo,
        "effective_from": start_iso,
        "effective_to": end_iso,
        "snippet": snippet,
    }


def _chunk_location_label(chunk):
    """Build location prefix matching chunk provenance ('Trang X' or section title)."""
    bmeta = chunk.get("business_metadata")
    if isinstance(bmeta, dict):
        citation = bmeta.get("citation")
        if isinstance(citation, dict):
            pages = citation.get("page_numbers")
            if isinstance(pages, list) and pages:
                return f"Trang {pages[0]}"
    sec = str(chunk.get("section_title") or "").strip()
    if sec:
        return sec
    bc = str(chunk.get("breadcrumb") or "").strip()
    if bc:
        return bc
    return "Tài liệu"


def extract_policy_dates(chunks):
    """
    Scan ordered chunks of a single document to extract its policy effective dates.
    Returns empty dict if unknown, {'effective_date_status': 'conflict'} on conflicting non-promo starts,
    or populated policy_revision date metadata.
    """
    if not chunks:
        return {}

    self_candidates = []
    other_candidates = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        source = str(chunk.get("source") or "")
        source_norm = search_key(source)
        source_is_circular = source_norm.startswith("thong tu ") or source_norm.startswith("nghi dinh ")
        loc = _chunk_location_label(chunk)

        texts = []
        content = str(chunk.get("content") or "").strip()
        if content:
            texts.append(content)
        parent = str(chunk.get("parent_content") or "").strip()
        if parent and parent not in content:
            texts.append(parent)

        for block in texts:
            for clause in _split_clauses(block):
                hit = _extract_clause_dates(clause, source_is_circular=source_is_circular)
                if hit is None:
                    continue
                hit["loc"] = loc
                if hit["is_self"]:
                    self_candidates.append(hit)
                else:
                    other_candidates.append(hit)

    chosen_pool = self_candidates
    if not chosen_pool:
        chosen_pool = other_candidates
    if not chosen_pool:
        return {}

    starts = []
    ends = []
    refs = []
    for item in chosen_pool:
        if item["is_promo"] and self_candidates:
            non_promo_exists = any(not c["is_promo"] for c in self_candidates)
            if non_promo_exists:
                continue
        if item["effective_from"] is not None:
            if item["effective_from"] not in starts:
                starts.append(item["effective_from"])
            ref = item["loc"] + " — " + item["snippet"]
            if ref not in refs:
                refs.append(ref)
        if item["effective_to"] is not None:
            if item["effective_to"] not in ends:
                ends.append(item["effective_to"])
            ref = item["loc"] + " — " + item["snippet"]
            if ref not in refs:
                refs.append(ref)

    if len(starts) > 1 or len(ends) > 1:
        return {"effective_date_status": "conflict"}

    effective_from = None
    if starts:
        effective_from = starts[0]
    effective_to = None
    if ends:
        effective_to = ends[0]

    if effective_from is None and effective_to is None:
        return {}

    status = "unknown"
    if effective_from is not None:
        status = "known"

    reference = ""
    if refs:
        reference = refs[0]

    return {
        "effective_from": effective_from,
        "effective_to": effective_to,
        "effective_date_status": status,
        "effective_date_reference": reference,
        "effective_date_origin": "document",
    }
