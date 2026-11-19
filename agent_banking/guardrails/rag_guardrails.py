# -*- coding: utf-8 -*-
"""
Banking Guardrails System (Streamlined & Robust Protection)
===========================================================
Checks ReAct input, tool parameters, retrieved documents, and final output.

CORE DESIGN INVARIANTS:
- 0% Regex Law: No regex methods or import re.
- Zero Ternary Law: Explicit if/else statements only.
- No Typing Library: No typing module imports.
- No __init__.py: Direct module imports.
"""

import unicodedata
from threading import Lock

from agent_banking.core.settings import Config
from agent_banking.guardrails.config.settings import (
    GuardrailsConfig,
    load_guardrail_patterns,
    load_guardrail_prompts,
    load_guardrail_rules,
)
from agent_banking.guardrails.indirect_injection_guardrail import (
    condense_spaced_text,
    strip_vietnamese_accents,
)


_patterns = load_guardrail_patterns()
_rules = load_guardrail_rules()
_prompts = load_guardrail_prompts()

_inj_heuristics = dict(_patterns.get("prompt_injection_heuristics") or {})
_fraud_policy = dict(_rules.get("fraud_policy_rules") or {})
_domain_rules = dict(_rules.get("domain_boundary_rules") or {})
_auth_rules = dict(_rules.get("auth_leak_rules") or {})
_pii_rules = dict(_rules.get("pii_masking_rules") or {})
_output_rules = dict(_rules.get("output_guard_rules") or {})
_citation_rules = dict(_rules.get("citation_grounding_rules") or {})

_redaction_tokens = dict(getattr(_prompts, "REDACTION_TOKENS", {}) or {})
_guardrail_logs = dict(getattr(_prompts, "GUARDRAIL_LOG_MESSAGES", {}) or {})


def _unwrap_guardrail_chunk(raw_item):
    """Unwrap {'chunk': <13-key dict>, ...} or flat chunk dict cleanly."""
    if isinstance(raw_item, dict) and "chunk" in raw_item:
        nested = raw_item.get("chunk")
        if isinstance(nested, dict):
            return nested, raw_item
    return raw_item, raw_item


def _extract_table_text_parts(table_data):
    """Collect searchable text from v2.0 table_data dicts."""
    if not isinstance(table_data, dict) or not table_data:
        return []
    parts = []
    for key in ["title", "column", "row_key", "table_id"]:
        val = str(table_data.get(key) or "").strip()
        if val and key != "table_id":
            parts.append(val)
    headers = table_data.get("headers")
    if isinstance(headers, list):
        for h in headers:
            hs = str(h).strip()
            if hs:
                parts.append(hs)
    rows = table_data.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, list):
                for cell in row:
                    cs = str(cell).strip()
                    if cs:
                        parts.append(cs)
            elif isinstance(row, dict):
                for rk, rv in row.items():
                    parts.append(f"{rk} {rv}")
    for cell_list_key in ["header_cells", "cells"]:
        cl = table_data.get(cell_list_key)
        if isinstance(cl, list):
            for c_item in cl:
                if isinstance(c_item, dict):
                    txt = str(c_item.get("text") or c_item.get("value") or "").strip()
                    if txt:
                        parts.append(txt)
                elif c_item is not None:
                    cs = str(c_item).strip()
                    if cs:
                        parts.append(cs)
    return parts


def _extract_images_text_parts(images):
    """Collect searchable text from v2.0 multimodal images list."""
    if not isinstance(images, list) or not images:
        return []
    parts = []
    for img in images:
        if not isinstance(img, dict):
            continue
        for key in ["title", "caption", "description", "alt_text", "native_text", "section_title"]:
            val = str(img.get(key) or "").strip()
            if val:
                parts.append(val)
        nvals = img.get("native_values")
        if isinstance(nvals, list):
            for nv in nvals:
                nvs = str(nv).strip()
                if nvs:
                    parts.append(nvs)
        elif isinstance(nvals, dict):
            for nk, nv in nvals.items():
                parts.append(f"{nk} {nv}")
        elif nvals is not None:
            nvs = str(nvals).strip()
            if nvs:
                parts.append(nvs)
    return parts


def _collect_chunk_text(raw_item, include_parent=True, include_source=True):
    """Gather all relevant text across the 13-field universal chunk schema for guardrails."""
    if isinstance(raw_item, str):
        return raw_item
    if not isinstance(raw_item, dict):
        return ""
    if "chunk" in raw_item and isinstance(raw_item.get("chunk"), str):
        return raw_item.get("chunk", "")

    chunk_data, _ = _unwrap_guardrail_chunk(raw_item)
    if not isinstance(chunk_data, dict):
        return ""

    parts = []
    if include_source:
        src = str(chunk_data.get("source") or "").strip()
        if src:
            parts.append(src)
    for key in ["section_title", "breadcrumb", "content"]:
        val = str(chunk_data.get(key) or "").strip()
        if val:
            parts.append(val)
    if not chunk_data.get("content"):
        for fallback_key in ["document", "text"]:
            val = str(chunk_data.get(fallback_key) or "").strip()
            if val:
                parts.append(val)
                break
    if include_parent:
        parent_val = str(chunk_data.get("parent_content") or "").strip()
        if parent_val:
            parts.append(parent_val)

    parts.extend(_extract_table_text_parts(chunk_data.get("table_data")))
    parts.extend(_extract_images_text_parts(chunk_data.get("images")))

    bmeta = chunk_data.get("business_metadata")
    if isinstance(bmeta, dict):
        for mk in ["document_name", "domain", "category"]:
            mv = str(bmeta.get(mk) or "").strip()
            if mv:
                parts.append(mv)
        ret = bmeta.get("retrieval")
        if isinstance(ret, dict):
            for rk in ["context", "situated_context"]:
                rv = str(ret.get(rk) or "").strip()
                if rv:
                    parts.append(rv)

    return " ".join(parts)


def _collect_chunks_grounding_text(retrieved_chunks):
    """Combine searchable text across all retrieved chunks cleanly."""
    all_parts = []
    for c in retrieved_chunks:
        txt = _collect_chunk_text(c, include_parent=True, include_source=True)
        if txt:
            all_parts.append(txt)
    return " ".join(all_parts)


# =====================================================================
# 1. GUARDRAIL RESULT DATA STRUCTURES
# =====================================================================

class InputGuardResult:
    """Result of input security evaluation."""
    def __init__(self, is_safe, sanitized_query, trigger_reason=None, refusal_message=None, warning_message=None, details=None):
        self.is_safe = is_safe
        self.sanitized_query = sanitized_query
        self.trigger_reason = trigger_reason
        self.refusal_message = refusal_message
        self.warning_message = warning_message
        if details is None:
            self.details = {}
        else:
            self.details = details


class OutputGuardResult:
    """Result of output sanitization and grounding."""
    def __init__(self, is_safe, sanitized_answer, flags=None):
        self.is_safe = is_safe
        self.sanitized_answer = sanitized_answer
        if flags is None:
            self.flags = []
        else:
            self.flags = flags


# =====================================================================
# 2. ATTACK AND OUT-OF-DOMAIN DETECTION CATALOG (CONFIG-DRIVEN, 0% REGEX)
# =====================================================================

INJECTION_PHRASES = list(_patterns.get("injection_phrases") or [])

UNACCENTED_INJECTION_PHRASES = []
for _p in INJECTION_PHRASES:
    _u = strip_vietnamese_accents(_p)
    if _u not in UNACCENTED_INJECTION_PHRASES:
        UNACCENTED_INJECTION_PHRASES.append(_u)

OFF_TOPIC_DOMAINS = list(_rules.get("off_topic_domains") or [])
FRAUD_CATEGORIES = dict(_rules.get("fraud_categories") or {})
ENTERPRISE_GUARDRAIL_TOPICS = FRAUD_CATEGORIES
REFUSAL_MESSAGES = dict(getattr(_prompts, "REFUSAL_MESSAGES", {}) or {})
PROMPT_LEAK_SIGNATURES = list(_patterns.get("prompt_leak_signatures") or [])

_DEFAULT_REFUSAL_MESSAGE = str(getattr(_prompts, "DEFAULT_REFUSAL_MESSAGE", "") or "")
_EMPTY_QUERY_REFUSAL_MESSAGE = str(getattr(_prompts, "EMPTY_QUERY_REFUSAL_MESSAGE", "") or "")

_ACTION_KEYWORDS = list(_inj_heuristics.get("action_keywords") or [])
_TARGET_KEYWORDS = list(_inj_heuristics.get("target_keywords") or [])
_COOCCURRENCE_WINDOW_CHARS = int(_inj_heuristics.get("cooccurrence_window_chars", 0))
_LEETSPEAK_CHAR_MAP = dict(_inj_heuristics.get("leetspeak_char_map") or {})
_ZERO_WIDTH_CHARS = list(_inj_heuristics.get("zero_width_chars") or [])


# =====================================================================
# 3. Banking Guardrail Manager
# =====================================================================

class BankingGuardrailManager:
    """
    Streamlined Banking Guardrails Manager (3-stage RAG defense pipeline).
    """

    def __init__(self, embedder=None, decision_client=None, use_decision_model=True):
        self.embedder = embedder
        self.use_decision_model = use_decision_model
        if decision_client is not None:
            self.decision_client = decision_client
        else:
            self.decision_client = None

        from agent_banking.guardrails.semantic_guardrail import SemanticBankingGuardrail
        from agent_banking.guardrails.indirect_injection_guardrail import IndirectInjectionGuardrail
        from agent_banking.guardrails.tool_guardrail import BankingToolGuardrail

        self.semantic_guardrail = SemanticBankingGuardrail(
            embedder=self.embedder,
            threshold=Config.SEMANTIC_SIMILARITY_THRESHOLD,
            decision_client=self.decision_client,
            use_decision_model=self.use_decision_model
        )
        self.indirect_guardrail = IndirectInjectionGuardrail()
        self.tool_guardrail = BankingToolGuardrail()

        # L1 In-Memory LRU Decision Cache
        self._decision_cache = {}
        self._decision_cache_lock = Lock()
        self._cache_max_size = GuardrailsConfig.DECISION_CACHE_MAX_SIZE

    def _get_refusal_for_category(self, category):
        """Retrieve category-specific refusal message for guardrail violation."""
        if category in REFUSAL_MESSAGES:
            return REFUSAL_MESSAGES[category]
        return _DEFAULT_REFUSAL_MESSAGE

    def _normalize_leetspeak(self, text):
        """Normalize Leetspeak, Cyrillic homoglyphs, and whitespace (0% Regex)."""
        chars = []
        for ch in text.lower():
            if ch in _LEETSPEAK_CHAR_MAP:
                chars.append(_LEETSPEAK_CHAR_MAP[ch])
            else:
                chars.append(ch)
        normalized = " ".join("".join(chars).split())
        return normalized

    def _scan_cooccurrence(self, text_val):
        for act in _ACTION_KEYWORDS:
            if act in text_val:
                act_idx = text_val.find(act)
                window = text_val[act_idx:act_idx + len(act) + _COOCCURRENCE_WINDOW_CHARS]
                for tgt in _TARGET_KEYWORDS:
                    if tgt in window:
                        return f"{act} ... {tgt}"
        return None

    def _check_prompt_injection(self, text_lower):
        """Scan Prompt Injection / Jailbreak, Spaced Text, Unaccented Vietnamese, and Leetspeak (0% Regex)."""
        # 1. Scan direct keyword taxonomy on raw text
        for phrase in INJECTION_PHRASES:
            if phrase in text_lower:
                return phrase

        # 2. Dual-Keyword Co-occurrence Heuristic
        cooccur = self._scan_cooccurrence(text_lower)
        if cooccur is not None:
            return cooccur

        # 3. Compress fragmented characters with spaces or dots/underscores
        text_clean = text_lower.replace(".", " ").replace("_", " ")
        condensed = condense_spaced_text(text_clean)
        if condensed != text_lower:
            for phrase in INJECTION_PHRASES:
                if phrase in condensed:
                    return phrase
            cooccur = self._scan_cooccurrence(condensed)
            if cooccur is not None:
                return cooccur

        # 4. Unaccented Vietnamese Normalization
        unacc = strip_vietnamese_accents(condensed)
        for phrase in UNACCENTED_INJECTION_PHRASES:
            if phrase in unacc:
                return phrase
        cooccur = self._scan_cooccurrence(unacc)
        if cooccur is not None:
            return cooccur

        # 5. Leetspeak Normalization and re-scan
        leet_norm = self._normalize_leetspeak(unacc)
        for phrase in UNACCENTED_INJECTION_PHRASES:
            if phrase in leet_norm:
                return phrase
        cooccur = self._scan_cooccurrence(leet_norm)
        if cooccur is not None:
            return cooccur

        return None

    def _check_auth_leak(self, text_lower):
        """Check credential leak risks (Password, OTP, PIN, CVV)."""
        leak_markers = list(_auth_rules.get("leak_markers") or [])
        explicit_markers = list(_auth_rules.get("explicit_assignment_markers") or [])
        cvv_checkers = list(_auth_rules.get("cvv_checkers") or [])
        auth_code_markers = tuple(_auth_rules.get("auth_code_markers") or [])
        auth_code_rules = list(_auth_rules.get("auth_code_rules") or [])
        password_keywords = list(_auth_rules.get("password_keywords") or [])
        non_secret_words = list(_auth_rules.get("non_secret_words") or [])
        pw_length_unit_words = list(_auth_rules.get("password_length_unit_words") or [])
        otp_keywords = list(_auth_rules.get("otp_keywords") or [])
        pin_keywords = list(_auth_rules.get("pin_keywords") or [])
        connector_words = tuple(_auth_rules.get("connector_words") or [])

        pw_min_len = int(_auth_rules.get("password_min_length", 0))
        explicit_min_len = int(_auth_rules.get("explicit_assignment_min_length", 0))
        pw_hint_max_digits = int(_auth_rules.get("password_length_hint_max_digits", 0))
        lookahead_chars = int(_auth_rules.get("lookahead_chars", 0))
        lookahead_words = int(_auth_rules.get("lookahead_words", 0))
        otp_digit_len = int(_auth_rules.get("otp_digit_length", 0))
        pin_digit_lengths = tuple(_auth_rules.get("pin_digit_lengths") or [])
        cvv_digit_lengths = tuple(_auth_rules.get("cvv_digit_lengths") or [])

        strip_chars = str(_pii_rules.get("punctuation_strip_chars") or "")
        strip_with_space = strip_chars + " "

        has_leak_marker = False
        matched_marker = None
        for m in leak_markers:
            if m in text_lower:
                has_leak_marker = True
                matched_marker = m
                break

        has_direct_cvv = False
        for cvk in cvv_checkers:
            if cvk in text_lower:
                idx = text_lower.find(cvk)
                after_part = text_lower[idx + len(cvk):].strip(" :,.")
                parts = after_part.split()
                if len(parts) > 0:
                    cand = parts[0].strip(strip_chars)
                    if len(cand) in cvv_digit_lengths and cand.isdigit():
                        has_direct_cvv = True
                        break

        has_auth_code = any(marker in text_lower for marker in auth_code_markers)
        for word in text_lower.split():
            token = word.strip(strip_chars)
            for rule in auth_code_rules:
                marker = str(rule.get("marker") or "")
                lengths = tuple(rule.get("lengths") or [])
                category = str(rule.get("category") or "")
                suffix = token.removeprefix(marker).lstrip(":=")
                if token.startswith(marker) and suffix.isdigit() and len(suffix) in lengths:
                    return True, category, self._get_refusal_for_category(category)

        if not has_leak_marker and not has_direct_cvv and not has_auth_code:
            return False, None, None

        # 1. Password leak (distinguished from password policy inquiries)
        if has_leak_marker:
            for pw in password_keywords:
                if pw in text_lower:
                    after_text = text_lower.split(pw, 1)[1].lstrip()
                    assignment = ""
                    for marker in leak_markers:
                        if after_text.startswith(marker):
                            assignment = marker
                            break
                    if not assignment:
                        continue
                    after_text = after_text[len(assignment):].strip()
                    tokens = after_text.split()

                    if len(tokens) > 0:
                        cand = tokens[0].strip(strip_chars)
                        is_non_secret = False
                        if cand.lower() in non_secret_words:
                            is_non_secret = True

                        if cand.isdigit() and len(cand) <= pw_hint_max_digits:
                            if len(tokens) > 1:
                                next_tok = tokens[1].strip(strip_chars).lower()
                                if next_tok in pw_length_unit_words:
                                    is_non_secret = True

                        if not is_non_secret:
                            is_pw_leak = False
                            if len(cand) >= pw_min_len:
                                is_pw_leak = True
                            elif any(c.isdigit() for c in cand) and any(c.isalpha() for c in cand):
                                is_pw_leak = True
                            elif matched_marker in explicit_markers and len(cand) >= explicit_min_len:
                                is_pw_leak = True

                            if is_pw_leak:
                                msg = self._get_refusal_for_category("AUTH_LEAK_PASSWORD")
                                return True, "AUTH_LEAK_PASSWORD", msg

        # 2. Card security code CVV/CVC leak
        for cvk in cvv_checkers:
            if cvk in text_lower:
                idx = text_lower.find(cvk)
                after_part = text_lower[idx + len(cvk):].strip(" :,.")
                parts = after_part.split()
                if len(parts) > 0:
                    cand = parts[0].strip(strip_chars)
                    if len(cand) in cvv_digit_lengths and cand.isdigit():
                        msg = self._get_refusal_for_category("AUTH_LEAK_CVV")
                        return True, "AUTH_LEAK_CVV", msg

                if has_leak_marker:
                    after_marker = text_lower[idx + len(cvk):].strip(" :,.")
                    for marker in leak_markers:
                        if after_marker.startswith(marker):
                            after_marker = after_marker[len(marker):].strip(" :,.")
                            break
                    m_parts = after_marker.split()
                    if len(m_parts) > 0:
                        cand = m_parts[0].strip(strip_chars)
                        if len(cand) in cvv_digit_lengths and cand.isdigit():
                            msg = self._get_refusal_for_category("AUTH_LEAK_CVV")
                            return True, "AUTH_LEAK_CVV", msg

        # 3. OTP code leak
        for ok in otp_keywords:
            if ok in text_lower:
                start = text_lower.find(ok) + len(ok)
                clean_t = text_lower[start:start + lookahead_chars].replace(":", " ").replace(",", " ")
                words = clean_t.split()
                for w in words[:lookahead_words]:
                    cw = w.strip(".,;:?!")
                    if len(cw) == otp_digit_len and cw.isdigit():
                        msg = self._get_refusal_for_category("AUTH_LEAK_OTP")
                        return True, "AUTH_LEAK_OTP", msg
                    if cw not in connector_words:
                        break

        # 4. PIN code leak
        for pk in pin_keywords:
            if pk in text_lower:
                start = text_lower.find(pk) + len(pk)
                clean_t = text_lower[start:start + lookahead_chars].replace(":", " ").replace(",", " ")
                words = clean_t.split()
                for w in words[:lookahead_words]:
                    cw = w.strip(".,;:?!")
                    if len(cw) in pin_digit_lengths and cw.isdigit():
                        msg = self._get_refusal_for_category("AUTH_LEAK_PIN")
                        return True, "AUTH_LEAK_PIN", msg
                    if cw not in connector_words:
                        break

        return False, None, None

    def _mask_pii(self, text):
        """Mask PII values (card, email, phone, citizen ID, account, CVV) using 0% Regex."""
        sanitized = text

        card_group_digits = int(_pii_rules.get("card_group_digits", 0))
        hyphen_card_len = int(_pii_rules.get("hyphen_card_len", 0))
        card_digits_len = int(_pii_rules.get("card_digits_len", 0))
        phone_digits_len = int(_pii_rules.get("phone_digits_len", 0))
        cid_12_len = int(_pii_rules.get("citizen_id_12_len", 0))
        cid_9_len = int(_pii_rules.get("citizen_id_9_len", 0))
        acc_min_len = int(_pii_rules.get("account_min_len", 0))
        acc_max_len = int(_pii_rules.get("account_max_len", 0))
        cur_min_len = int(_pii_rules.get("currency_min_len", 0))
        cur_max_len = int(_pii_rules.get("currency_max_len", 0))
        year_min = int(_pii_rules.get("year_seq_min", 0))
        year_max = int(_pii_rules.get("year_seq_max", 0))
        cvv_window = int(_pii_rules.get("cvv_context_window_tokens", 0))

        official_domains = tuple(_pii_rules.get("official_email_domains") or [])
        official_suffixes = tuple(_pii_rules.get("official_email_suffixes") or [])
        vn_mobile_prefixes = tuple(_pii_rules.get("vn_mobile_prefixes") or [])
        vn_country_prefix = str(_pii_rules.get("vn_country_code_prefix") or "")
        currency_units = tuple(_pii_rules.get("currency_unit_words") or [])
        strip_chars = str(_pii_rules.get("punctuation_strip_chars") or "")
        currency_strip_chars = str(_pii_rules.get("currency_strip_chars") or "")

        cvv_markers = list(_auth_rules.get("cvv_checkers") or [])
        connector_words = tuple(_auth_rules.get("connector_words") or [])
        cvv_digit_lengths = tuple(_auth_rules.get("cvv_digit_lengths") or [])

        tok_card = str(_redaction_tokens.get("card_number") or "")
        tok_email = str(_redaction_tokens.get("email") or "")
        tok_phone = str(_redaction_tokens.get("phone_number") or "")
        tok_cvv = str(_redaction_tokens.get("cvv") or "")
        tok_cid = str(_redaction_tokens.get("citizen_id") or "")
        tok_account = str(_redaction_tokens.get("account_number") or "")

        # 1. Mask space-separated credit card numbers (4 groups of 4 digits)
        words = sanitized.split(" ")
        idx = 0
        while idx <= len(words) - 4:
            raw0 = words[idx]
            raw1 = words[idx + 1]
            raw2 = words[idx + 2]
            raw3 = words[idx + 3]
            if (raw0.endswith(",") or raw0.endswith(";") or
                    raw1.endswith(",") or raw1.endswith(";") or
                    raw2.endswith(",") or raw2.endswith(";") or
                    raw1.startswith(",") or raw1.startswith(";") or
                    raw2.startswith(",") or raw2.startswith(";") or
                    raw3.startswith(",") or raw3.startswith(";")):
                idx += 1
                continue
            w0 = raw0.strip(strip_chars)
            w1 = raw1.strip(strip_chars)
            w2 = raw2.strip(strip_chars)
            w3 = raw3.strip(strip_chars)
            if (len(w0) == card_group_digits and w0.isdigit() and
                    len(w1) == card_group_digits and w1.isdigit() and
                    len(w2) == card_group_digits and w2.isdigit() and
                    len(w3) == card_group_digits and w3.isdigit()):
                n0 = int(w0)
                n1 = int(w1)
                n2 = int(w2)
                n3 = int(w3)
                is_year_seq = False
                if year_min <= n0 <= year_max and year_min <= n1 <= year_max and year_min <= n2 <= year_max and year_min <= n3 <= year_max:
                    if (n1 - n0 == 1 and n2 - n1 == 1 and n3 - n2 == 1) or (n0 - n1 == 1 and n1 - n2 == 1 and n2 - n3 == 1):
                        is_year_seq = True
                if is_year_seq:
                    idx += 1
                    continue
                target_str = raw0 + " " + raw1 + " " + raw2 + " " + raw3
                sanitized = sanitized.replace(target_str, tok_card)
                words = sanitized.split(" ")
                idx = 0
            else:
                idx += 1

        # 2. Process each token
        tokens = sanitized.split(" ")
        out_tokens = []
        for index, token in enumerate(tokens):
            clean_tok = token.strip(strip_chars)

            # 2.1. Mask Email (preserve official domains)
            if "@" in clean_tok:
                email_parts = clean_tok.split("@")
                if len(email_parts) == 2:
                    loc_p, dom_p = email_parts[0], email_parts[1]
                    if len(loc_p) > 0 and len(dom_p) > 0 and "." in dom_p:
                        d_parts = dom_p.split(".")
                        if len(d_parts[-1]) >= 2:
                            dom_lower = dom_p.lower()
                            is_official = dom_lower in official_domains
                            if not is_official:
                                for suffix in official_suffixes:
                                    if dom_lower.endswith(suffix):
                                        is_official = True
                                        break
                            if is_official:
                                out_tokens.append(token)
                            else:
                                out_tokens.append(token.replace(clean_tok, tok_email))
                            continue

            # 2.2. Mask credit card with hyphens or dots
            if len(clean_tok) == hyphen_card_len and ("-" in clean_tok or "." in clean_tok):
                card_digits = clean_tok.replace("-", "").replace(".", "")
                if len(card_digits) == card_digits_len and card_digits.isdigit():
                    out_tokens.append(token.replace(clean_tok, tok_card))
                    continue

            # 2.3. Mask phone numbers with separators
            if ("." in clean_tok or "-" in clean_tok) and not clean_tok.startswith("http"):
                phone_digits = clean_tok.replace(".", "").replace("-", "")
                if vn_country_prefix and phone_digits.startswith(vn_country_prefix):
                    phone_digits = "0" + phone_digits[len(vn_country_prefix):]
                if len(phone_digits) == phone_digits_len and phone_digits.isdigit():
                    if phone_digits.startswith(vn_mobile_prefixes):
                        out_tokens.append(token.replace(clean_tok, tok_phone))
                        continue

            # 2.4. Mask international phone number
            if vn_country_prefix and clean_tok.startswith(vn_country_prefix):
                norm_int_phone = "0" + clean_tok[len(vn_country_prefix):]
                if len(norm_int_phone) == phone_digits_len and norm_int_phone.isdigit():
                    out_tokens.append(token.replace(clean_tok, tok_phone))
                    continue

            # 2.5. Mask CVV/CVC when context and number follow recognition keyword
            is_cvv = False
            if clean_tok.isdigit() and len(clean_tok) in cvv_digit_lengths:
                for cvk in cvv_markers:
                    context = " ".join(tokens[max(0, index - cvv_window):index + 1]).lower()
                    if cvk in context:
                        tail = context.rsplit(cvk, 1)[1].replace(":", " ").split()
                        if tail and all(part in connector_words for part in tail[:-1]):
                            is_cvv = True
                            break

            if is_cvv:
                out_tokens.append(token.replace(clean_tok, tok_cvv))
                continue

            # 2.6. Mask contiguous numbers
            if clean_tok.isdigit():
                following = tokens[index + 1:index + 2]
                currency_amount = False
                if following and cur_min_len <= len(clean_tok) <= cur_max_len and not clean_tok.startswith("0"):
                    currency_amount = following[0].strip(currency_strip_chars).lower() in currency_units
                if currency_amount:
                    out_tokens.append(token)
                elif len(clean_tok) == card_digits_len:
                    out_tokens.append(token.replace(clean_tok, tok_card))
                elif len(clean_tok) == cid_12_len and clean_tok.startswith("0"):
                    out_tokens.append(token.replace(clean_tok, tok_cid))
                elif len(clean_tok) == cid_9_len:
                    out_tokens.append(token.replace(clean_tok, tok_cid))
                elif len(clean_tok) == phone_digits_len and clean_tok.startswith(vn_mobile_prefixes):
                    out_tokens.append(token.replace(clean_tok, tok_phone))
                elif acc_min_len <= len(clean_tok) <= acc_max_len:
                    out_tokens.append(token.replace(clean_tok, tok_account))
                else:
                    out_tokens.append(token)
            else:
                out_tokens.append(token)

        return " ".join(out_tokens)

    def _check_fraud_and_crime(self, text_lower):
        """Scan Financial Fraud & Cyber Crime patterns (0% Regex)."""
        payroll_subject = str(_fraud_policy.get("payroll_subject") or "")
        payroll_markers = list(_fraud_policy.get("payroll_markers") or [])
        if payroll_subject and payroll_subject in text_lower:
            for pm in payroll_markers:
                if pm in text_lower:
                    return None, None

        benign_markers = list(_fraud_policy.get("benign_markers") or [])
        has_benign = False
        for bm in benign_markers:
            if bm in text_lower:
                has_benign = True
                break

        asks_for_attack_markers = tuple(_fraud_policy.get("asks_for_attack_markers") or [])
        asks_for_attack = any(marker in text_lower for marker in asks_for_attack_markers)

        perm_markers = list(_fraud_policy.get("permission_question_markers") or [])
        permission_question = True
        if not perm_markers:
            permission_question = False
        else:
            for pm in perm_markers:
                if pm not in text_lower:
                    permission_question = False
                    break

        perm_exempt_categories = tuple(_fraud_policy.get("permission_exempt_categories") or [])
        defensive_category = str(_fraud_policy.get("defensive_context_category") or "")
        defensive_markers = tuple(_fraud_policy.get("defensive_context_markers") or [])
        defensive_context = any(marker in text_lower for marker in defensive_markers)

        crisis_category = str(_fraud_policy.get("crisis_response_category") or "")
        crisis_markers = tuple(_fraud_policy.get("crisis_response_markers") or [])
        crisis_response = any(marker in text_lower for marker in crisis_markers)

        violence_category = str(_fraud_policy.get("violence_category") or "")
        insurance_subject = str(_fraud_policy.get("insurance_context_subject") or "")
        insurance_markers = tuple(_fraud_policy.get("insurance_context_markers") or [])
        harmful_violence_kw = tuple(_fraud_policy.get("harmful_violence_keywords") or [])
        harmful_req_markers = tuple(_fraud_policy.get("harmful_request_markers") or [])
        direct_malicious = list(_fraud_policy.get("direct_malicious_markers") or [])

        for category, phrases in FRAUD_CATEGORIES.items():
            for p in phrases:
                if p in text_lower:
                    if not asks_for_attack:
                        if category in perm_exempt_categories and permission_question:
                            continue
                        if category == defensive_category and defensive_context:
                            continue
                        if category == crisis_category and crisis_response:
                            continue
                    if category == violence_category and p in harmful_violence_kw:
                        insurance_context = (insurance_subject in text_lower) and any(
                            marker in text_lower for marker in insurance_markers
                        )
                        harmful_request = any(marker in text_lower for marker in harmful_req_markers)
                        if insurance_context and not harmful_request:
                            continue
                    if has_benign:
                        is_direct = False
                        for dm in direct_malicious:
                            if dm in text_lower:
                                is_direct = True
                                break
                        if not is_direct:
                            continue
                    return category, p
        return None, None

    def _check_domain_boundary(self, text_lower):
        """Scan Banking Domain Boundaries to reject off-topic requests (0% Regex)."""
        lucky_subject = str(_domain_rules.get("lucky_account_subject") or "")
        lucky_words = list(_domain_rules.get("lucky_account_context_words") or [])
        if lucky_subject and lucky_subject in text_lower:
            for lw in lucky_words:
                if lw in text_lower:
                    return False, None

        banking_context_words = list(_domain_rules.get("banking_context_words") or [])
        has_banking_context = False
        for bw in banking_context_words:
            if bw in text_lower:
                has_banking_context = True
                break

        if has_banking_context:
            exempt_locations = list(_domain_rules.get("exempt_locations") or [])
            for el in exempt_locations:
                if el in text_lower:
                    return False, None

        medical_phrases = tuple(_domain_rules.get("medical_policy_phrases") or [])
        insurance_subjects = tuple(_domain_rules.get("medical_insurance_subjects") or [])
        coverage_markers = tuple(_domain_rules.get("medical_coverage_markers") or [])
        clinical_markers = tuple(_domain_rules.get("clinical_request_markers") or [])

        for phrase in OFF_TOPIC_DOMAINS:
            if phrase in text_lower:
                if phrase in medical_phrases:
                    coverage_question = any(marker in text_lower for marker in insurance_subjects) and any(
                        marker in text_lower for marker in coverage_markers
                    )
                    clinical_request = any(marker in text_lower for marker in clinical_markers)
                    if coverage_question and not clinical_request:
                        continue
                return True, phrase
        return False, None

    # -----------------------------------------------------------------
    # STAGE 1: INPUT GUARDRAIL (INJECTION & OUT-OF-DOMAIN PROTECTION)
    # -----------------------------------------------------------------
    def guard_input(self, query, use_decision_model=None):
        """
        Validate input query (optimized ordering & caching):
        1. L1 Decision Cache Hit
        2. Empty check
        3. Auth Leak check (Password, OTP, PIN, CVV)
        4. Fast-path Prompt Injection (Heuristics & Co-occurrence)
        5. Financial Fraud & Cyber Crime check
        6. Domain Boundary check
        7. Deferred PII Masking
        8. Semantic Vector Guardrail
        9. Store result in L1 Decision Cache
        """
        if not query:
            return InputGuardResult(
                is_safe=False,
                sanitized_query="",
                trigger_reason="EMPTY_QUERY",
                refusal_message=_EMPTY_QUERY_REFUSAL_MESSAGE
            )
        if not str(query).strip():
            return InputGuardResult(
                is_safe=False,
                sanitized_query="",
                trigger_reason="EMPTY_QUERY",
                refusal_message=_EMPTY_QUERY_REFUSAL_MESSAGE
            )

        raw_query = unicodedata.normalize("NFC", str(query)).strip()
        for zwc in _ZERO_WIDTH_CHARS:
            if zwc in raw_query:
                raw_query = raw_query.replace(zwc, "")

        raw_query = raw_query.strip()
        if not raw_query:
            return InputGuardResult(
                is_safe=False,
                sanitized_query="",
                trigger_reason="EMPTY_QUERY",
                refusal_message=_EMPTY_QUERY_REFUSAL_MESSAGE
            )

        norm_key = raw_query.lower()

        # 1. Check L1 Decision Cache
        with self._decision_cache_lock:
            cached = self._decision_cache.pop(norm_key, None)
            if cached is not None:
                self._decision_cache[norm_key] = cached
        if cached is not None:
            return InputGuardResult(
                is_safe=cached.is_safe,
                sanitized_query=cached.sanitized_query,
                trigger_reason=cached.trigger_reason,
                refusal_message=cached.refusal_message,
                warning_message=cached.warning_message,
                details=dict(cached.details)
            )

        # 2. Scan authentication leak risks
        is_leak, leak_reason, leak_msg = self._check_auth_leak(norm_key)
        if is_leak:
            res = InputGuardResult(
                is_safe=False,
                sanitized_query="",
                trigger_reason=leak_reason,
                refusal_message=leak_msg,
                details={"leak_type": leak_reason}
            )
            return res

        # 3. Fast-Path Prompt Injection / Jailbreak
        matched_inject = self._check_prompt_injection(norm_key)
        if matched_inject:
            refusal = self._get_refusal_for_category("PROMPT_INJECTION_DETECTED")
            res = InputGuardResult(
                is_safe=False,
                sanitized_query=raw_query,
                trigger_reason="PROMPT_INJECTION_DETECTED",
                refusal_message=refusal,
                details={"matched_phrase": matched_inject}
            )
            self._save_to_decision_cache(norm_key, res)
            return res

        # 4. Scan Financial Fraud & Cyber Security
        fraud_category, matched_phrase = self._check_fraud_and_crime(norm_key)
        if fraud_category:
            refusal = self._get_refusal_for_category(fraud_category)
            res = InputGuardResult(
                is_safe=False,
                sanitized_query=raw_query,
                trigger_reason=fraud_category,
                refusal_message=refusal,
                details={"matched_phrase": matched_phrase}
            )
            self._save_to_decision_cache(norm_key, res)
            return res

        # 5. Scan Domain Boundary
        is_off_topic, off_topic_phrase = self._check_domain_boundary(norm_key)
        if is_off_topic:
            refusal = self._get_refusal_for_category("OUT_OF_DOMAIN_OFF_TOPIC")
            res = InputGuardResult(
                is_safe=False,
                sanitized_query=raw_query,
                trigger_reason="OUT_OF_DOMAIN_OFF_TOPIC",
                refusal_message=refusal,
                details={"matched_phrase": off_topic_phrase}
            )
            self._save_to_decision_cache(norm_key, res)
            return res

        # 6. Deferred PII masking (runs only on legitimate queries)
        sanitized_query = self._mask_pii(raw_query)

        should_use_decision = self.use_decision_model
        if use_decision_model is not None:
            should_use_decision = use_decision_model

        # 7. Decision Verification Tier
        if should_use_decision is True:
            if self.decision_client is not None:
                try:
                    if self.decision_client.is_available():
                        audit = self.decision_client.audit_adversarial_input(raw_query)
                        if isinstance(audit, dict):
                            is_attack = False
                            if audit.get("is_adversarial") is True:
                                is_attack = True
                            elif audit.get("is_attack") is True:
                                is_attack = True
                            elif audit.get("choice") == "ATTACK":
                                is_attack = True

                            if is_attack is True:
                                refusal = self._get_refusal_for_category("PROMPT_INJECTION_DETECTED")
                                res = InputGuardResult(
                                    is_safe=False,
                                    sanitized_query=raw_query,
                                    trigger_reason="PROMPT_INJECTION_DETECTED",
                                    refusal_message=refusal,
                                    details={
                                        "decision_model": audit.get("model", ""),
                                        "confidence": audit.get("confidence", 1.0),
                                        "audit": audit
                                    }
                                )
                                self._save_to_decision_cache(norm_key, res)
                                return res
                except Exception as e:
                    tpl = str(_guardrail_logs.get("decision_client_fallback") or "")
                    if tpl:
                        print(tpl.format(error=e))

        # 8. Semantic Vector Guardrail Scan
        if Config.ENABLE_SEMANTIC_GUARDRAIL is True:
            sem_res = self.semantic_guardrail.check_query(raw_query, use_decision_model=False)
            if sem_res["is_safe"] is False:
                cat = sem_res["category"]
                matched_tpl = sem_res["matched_template"]
                sim_score = sem_res["max_similarity"]
                refusal = self._get_refusal_for_category(cat)
                res = InputGuardResult(
                    is_safe=False,
                    sanitized_query=sanitized_query,
                    trigger_reason=f"SEMANTIC_{cat}",
                    refusal_message=refusal,
                    details={
                        "category": cat,
                        "similarity": sim_score,
                        "matched_template": matched_tpl
                    }
                )
                self._save_to_decision_cache(norm_key, res)
                return res

        res = InputGuardResult(is_safe=True, sanitized_query=sanitized_query)
        self._save_to_decision_cache(norm_key, res)
        return res

    def _save_to_decision_cache(self, key, result):
        """Save decision to L1 cache with bounded FIFO/LRU eviction."""
        with self._decision_cache_lock:
            if self._cache_max_size > 0 and len(self._decision_cache) >= self._cache_max_size:
                oldest_key = next(iter(self._decision_cache))
                del self._decision_cache[oldest_key]
            self._decision_cache[key] = result

    def guard_retrieved_chunks(self, chunks):
        """Scan retrieved RAG chunks against indirect prompt injection."""
        if Config.ENABLE_INDIRECT_INJECTION_SCAN is True:
            return self.indirect_guardrail.scan_chunks(chunks)
        return chunks, 0, []

    # -----------------------------------------------------------------
    # STAGE 3: POST-GENERATION GROUNDING & FAITHFULNESS
    # -----------------------------------------------------------------
    def guard_output(self, answer):
        """
        Sanitize generated output without regex:
        - Remove <think>...</think> blocks
        - Filter system prompt leak markers
        - Trim analytical/self-censoring monologue
        """
        if not answer:
            return OutputGuardResult(is_safe=True, sanitized_answer="")

        sanitized = unicodedata.normalize("NFC", answer).strip()
        flags = []

        thinking_tags = list(_output_rules.get("thinking_tags") or [])
        markdown_thinking = list(_output_rules.get("markdown_thinking_blocks") or [])
        greeting_prefixes = list(_output_rules.get("greeting_prefixes") or [])
        pre_greeting_markers = list(_output_rules.get("pre_greeting_monologue_markers") or [])
        conclusive_markers = list(_output_rules.get("conclusive_answer_markers") or [])
        conclusive_min_len = int(_output_rules.get("conclusive_min_length", 0))
        tail_markers = list(_output_rules.get("tail_self_censor_markers") or [])
        tail_min_prefix_len = int(_output_rules.get("tail_min_prefix_length", 0))

        # 1. Remove XML/HTML thinking blocks and markdown thought blocks
        for tag in thinking_tags:
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"
            while open_tag in sanitized:
                start_idx = sanitized.find(open_tag)
                curr_pos = start_idx + len(open_tag)
                depth = 1
                while depth > 0 and curr_pos < len(sanitized):
                    next_open = sanitized.find(open_tag, curr_pos)
                    next_close = sanitized.find(close_tag, curr_pos)

                    if next_close == -1:
                        break

                    if next_open != -1 and next_open < next_close:
                        depth += 1
                        curr_pos = next_open + len(open_tag)
                    else:
                        depth -= 1
                        curr_pos = next_close + len(close_tag)

                if depth > 0:
                    if start_idx > 0:
                        sanitized = sanitized[:start_idx].strip()
                    else:
                        remainder = sanitized[len(open_tag):].strip()
                        if "\n\n" in remainder:
                            sanitized = remainder.split("\n\n")[-1].strip()
                        elif "..." in remainder:
                            parts = remainder.split("...")
                            sanitized = parts[-1].strip()
                        else:
                            sanitized = remainder
                    break
                else:
                    sanitized = sanitized[:start_idx] + sanitized[curr_pos:]

            if close_tag in sanitized:
                sanitized = sanitized.replace(close_tag, "")

        for mb in markdown_thinking:
            while mb in sanitized:
                s_idx = sanitized.find(mb)
                e_idx = sanitized.find("```", s_idx + len(mb))
                if e_idx != -1:
                    sanitized = sanitized[:s_idx] + sanitized[e_idx + 3:]
                else:
                    sanitized = sanitized[:s_idx]
                    break

        # 2. Filter out System Prompt leak markers
        for sig in PROMPT_LEAK_SIGNATURES:
            if sig.lower() in sanitized.lower():
                flags.append(f"SYSTEM_PROMPT_LEAK_{sig}")
                sanitized = sanitized.replace(sig, "")

        # 3. Remove preliminary analytical monologue before greeting
        for greeting in greeting_prefixes:
            if greeting in sanitized:
                g_idx = sanitized.find(greeting)
                if g_idx > 0:
                    prefix_low = sanitized[:g_idx].lower()
                    if any(k in prefix_low for k in pre_greeting_markers):
                        sanitized = sanitized[g_idx:]
                break

        # 4. Capture conclusive answer phrases after draft
        for fa in conclusive_markers:
            if fa in sanitized:
                candidate = sanitized.split(fa)[-1].strip()
                if len(candidate) > conclusive_min_len:
                    sanitized = candidate
                    break

        if sanitized.startswith('"') or sanitized.startswith("'"):
            sanitized = sanitized[1:].strip()

        # 5. Trim guardrail self-censoring monologue if present at tail
        for ca in tail_markers:
            if ca in sanitized:
                c_idx = sanitized.find(ca)
                if c_idx > tail_min_prefix_len:
                    sanitized = sanitized[:c_idx].strip()
                    break

        if sanitized.endswith('"') or sanitized.endswith("'"):
            sanitized = sanitized[:-1].strip()

        return OutputGuardResult(is_safe=True, sanitized_answer=sanitized.strip(), flags=flags)

    def verify_citation_grounding(self, answer, retrieved_chunks):
        """Verify legal citations in answer against retrieved chunks (0% Regex, Zero Ternary)."""
        if not answer:
            return {"is_grounded": True, "verified_citations": [], "unverified_citations": [], "unsupported_citations": []}
        if not retrieved_chunks:
            return {"is_grounded": True, "verified_citations": [], "unverified_citations": [], "unsupported_citations": []}

        citation_keywords = list(_citation_rules.get("citation_keywords") or [])
        max_numeric_len = int(_citation_rules.get("max_numeric_citation_len", 0))
        separators = list(_citation_rules.get("separators") or [])
        anchor_words = list(_citation_rules.get("anchor_words") or [])
        code_prefixes = list(_citation_rules.get("code_prefixes") or [])
        core_kw = list(_citation_rules.get("core_keywords") or [])
        min_phrase_word_len = int(_citation_rules.get("min_phrase_word_len", 0))

        answer_lower = answer.lower()
        found_citations = []

        start_idx = 0
        while True:
            lb = answer_lower.find("[", start_idx)
            if lb == -1:
                break
            rb = answer_lower.find("]", lb)
            if rb == -1:
                break
            tag = answer_lower[lb:rb + 1]
            inner = tag[1:-1].strip()
            is_citation = False
            for ck in citation_keywords:
                if ck in tag:
                    is_citation = True
                    break
            if not is_citation:
                if inner.isdigit():
                    if len(inner) <= max_numeric_len:
                        is_citation = True

            if is_citation:
                found_citations.append(tag)
            start_idx = rb + 1

        all_chunk_text = _collect_chunks_grounding_text(retrieved_chunks).lower()

        verified = []
        unverified = []

        for cit in found_citations:
            clean_cit = cit.replace("[", "").replace("]", "").strip()
            if not clean_cit:
                continue
            if clean_cit.isdigit():
                idx_num = int(clean_cit)
                if idx_num <= len(retrieved_chunks) and idx_num > 0:
                    verified.append(cit)
                else:
                    unverified.append(cit)
                continue

            tokenized = clean_cit
            for sep in separators:
                tokenized = tokenized.replace(sep, " ")
            tokens = tokenized.split()

            is_cit_verified = False

            for i in range(len(tokens) - 1):
                if tokens[i] in anchor_words:
                    term = tokens[i] + " " + tokens[i + 1]
                    if term in all_chunk_text:
                        is_cit_verified = True
                        break

            if not is_cit_verified:
                for i in range(len(tokens) - 1):
                    if tokens[i] in code_prefixes:
                        code = tokens[i] + " " + tokens[i + 1]
                        if code in all_chunk_text:
                            is_cit_verified = True
                            break

            if not is_cit_verified:
                words = [w for w in tokens if len(w) > min_phrase_word_len]
                for i in range(len(words) - 2):
                    phrase = " ".join(words[i:i + 3])
                    if phrase in all_chunk_text:
                        is_cit_verified = True
                        break

            if not is_cit_verified:
                for w in tokens:
                    if w in core_kw:
                        if w in all_chunk_text:
                            is_cit_verified = True
                            break

            if is_cit_verified:
                verified.append(cit)
            else:
                unverified.append(cit)

        is_grounded = True
        if len(found_citations) > 0:
            if len(verified) == 0:
                is_grounded = False
            else:
                is_grounded = True
        else:
            is_grounded = True

        return {
            "is_grounded": is_grounded,
            "verified_citations": verified,
            "unverified_citations": unverified,
            "unsupported_citations": unverified
        }

    def guard_tool_call(self, tool_name, params, *, registered_readonly=False):
        """Validate tool parameters via BankingToolGuardrail."""
        if hasattr(self, "tool_guardrail") and self.tool_guardrail is not None:
            return self.tool_guardrail.guard_tool_call(tool_name, params, registered_readonly=registered_readonly)
        return {"is_safe": True, "sanitized_params": params, "error_message": None}


# Initialize singleton & backward-compatible aliases
guardrail_manager = BankingGuardrailManager()
RAGGuardrailManager = BankingGuardrailManager
GuardrailManager = BankingGuardrailManager
