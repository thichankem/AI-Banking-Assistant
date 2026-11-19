# -*- coding: utf-8 -*-
"""
Indirect Prompt Injection Guardrail Module (Layer 3 - RAG Security)
===================================================================
Protects the system against indirect injection attacks embedded within
retrieved RAG knowledge documents (Indirect Prompt Injection - OWASP LLM01).

Inspects and neutralizes covert instructions, malicious payloads, or hijack
commands embedded in retrieved text chunks before passing them to the generator.

DESIGN INVARIANTS:
1. 0% REGEX LAW: No regex methods or import re.
2. ZERO TERNARY LAW: Explicit if/else statements only.
3. NO TYPING LIBRARY: No typing library imports.
4. NO __INIT__.PY: Direct module imports.
"""

import unicodedata

from agent_banking.guardrails.config.settings import (
    GuardrailsConfig,
    load_guardrail_patterns,
    load_guardrail_prompts,
)


_patterns = load_guardrail_patterns()
_indirect_cfg = dict(_patterns.get("indirect_injection") or {})
_indirect_limits = dict(GuardrailsConfig.INDIRECT_INJECTION_SETTINGS or {})
_prompts = load_guardrail_prompts()
_redaction_tokens = dict(getattr(_prompts, "REDACTION_TOKENS", {}) or {})

INDIRECT_INJECTION_SIGNATURES = list(_indirect_cfg.get("signatures") or [])
DANGEROUS_HTML_ELEMENTS = list(_indirect_cfg.get("dangerous_html_elements") or [])
DANGEROUS_DATA_URIS = list(_indirect_cfg.get("dangerous_data_uris") or [])
OVERRIDE_VERBS = list(_indirect_cfg.get("override_verbs") or [])
TARGET_WORDS = list(_indirect_cfg.get("target_words") or [])

_SUSPICIOUS_IMAGE_URL_KEYWORDS = list(_indirect_cfg.get("suspicious_image_url_keywords") or [])
_DANGEROUS_IMAGE_URL_PREFIXES = tuple(_indirect_cfg.get("dangerous_image_url_prefixes") or [])
_EXTERNAL_HTTP_SCHEMES = list(_indirect_cfg.get("external_http_schemes") or [])
_DANGEROUS_IMAGE_URL_SUBSTRINGS = list(_indirect_cfg.get("dangerous_image_url_substrings") or [])
_CRITICAL_THREAT_MARKERS = list(_indirect_cfg.get("critical_threat_markers") or [])
_TABLE_TEXT_FIELDS = list(_indirect_cfg.get("table_text_fields") or [])
_IMAGE_TEXT_FIELDS = list(_indirect_cfg.get("image_text_fields") or [])
_IMAGE_URL_FIELDS = list(_indirect_cfg.get("image_url_fields") or [])
_CHUNK_TEXT_FIELDS = list(_indirect_cfg.get("chunk_text_fields") or [])
_RETRIEVAL_CONTEXT_FIELDS = list(_indirect_cfg.get("retrieval_context_fields") or [])

_NOSPACE_MIN_LENGTH = int(_indirect_limits.get("nospace_min_length", 0))
_OVERRIDE_WINDOW_WORDS = int(_indirect_limits.get("override_window_words", 2))
_CRITICAL_THREAT_MIN_COUNT = int(_indirect_limits.get("critical_threat_min_count", 0))
_URL_PREVIEW_CHARS = int(_indirect_limits.get("url_preview_chars", 0))

_REDACTION_INJECTION = str(_redaction_tokens.get("injection_command") or "")
_REDACTION_SCRIPT = str(_redaction_tokens.get("script") or "")
_REDACTION_IMAGE = str(_redaction_tokens.get("image_tag") or "")


def strip_vietnamese_accents(text):
    """Normalize Vietnamese text to unaccented form (0% Regex, Unicode NFD)."""
    if not text:
        return ""
    nfd = unicodedata.normalize("NFD", str(text))
    clean_chars = []
    for char in nfd:
        if unicodedata.category(char) != "Mn":
            clean_chars.append(char)
    stripped = "".join(clean_chars)
    stripped = stripped.replace("đ", "d")
    stripped = stripped.replace("Đ", "D")
    return stripped


def condense_spaced_text(text):
    """
    Detect and condense fragmented spaced characters ('i g n o r e' -> 'ignore').
    0% regex: Pure string splitting and manipulation.
    """
    if not text:
        return ""

    if "  " in text:
        chunks = text.split("  ")
        condensed_chunks = []
        for ch in chunks:
            ch_str = ch.strip()
            if ch_str:
                condensed_chunks.append(condense_spaced_text(ch_str))
        return " ".join(condensed_chunks)

    words = text.split(" ")
    if len(words) < 2:
        return text

    result_words = []
    single_chars = []
    for w in words:
        if len(w) == 1:
            single_chars.append(w)
        else:
            if len(single_chars) >= 2:
                result_words.append("".join(single_chars))
                single_chars = []
            else:
                if len(single_chars) > 0:
                    result_words.extend(single_chars)
                    single_chars = []
            if w:
                result_words.append(w)

    if len(single_chars) >= 2:
        result_words.append("".join(single_chars))
    else:
        if len(single_chars) > 0:
            result_words.extend(single_chars)

    return " ".join(result_words)


def extract_and_validate_markdown_images(text):
    """
    Parse and inspect Markdown image tags '![alt](url)' without regex.
    Returns: (malicious_images, all_image_tags)
    """
    if not text:
        return [], []

    malicious_images = []
    all_image_tags = []
    pos = 0

    while True:
        img_start = text.find("![", pos)
        if img_start == -1:
            break

        close_bracket = text.find("]", img_start + 2)
        if close_bracket == -1:
            break

        paren_start = text.find("(", close_bracket)
        if paren_start == -1:
            break

        between_brackets = text[close_bracket + 1 : paren_start].strip()
        if between_brackets != "":
            pos = close_bracket + 1
            continue

        paren_end = text.find(")", paren_start + 1)
        if paren_end == -1:
            break

        full_tag = text[img_start : paren_end + 1]
        url = text[paren_start + 1 : paren_end].strip()
        url_lower = url.lower()
        all_image_tags.append((full_tag, url))

        is_malicious = False
        has_external_http = False
        for scheme in _EXTERNAL_HTTP_SCHEMES:
            if scheme in url_lower:
                has_external_http = True
                break

        if has_external_http:
            if "?" in url_lower:
                is_malicious = True
            for kw in _SUSPICIOUS_IMAGE_URL_KEYWORDS:
                if kw in url_lower:
                    is_malicious = True
                    break

        if _DANGEROUS_IMAGE_URL_PREFIXES and url_lower.startswith(_DANGEROUS_IMAGE_URL_PREFIXES):
            is_malicious = True

        if is_malicious is True:
            malicious_images.append(full_tag)

        pos = paren_end + 1

    return malicious_images, all_image_tags


class IndirectInjectionGuardrail:
    """
    Scanner and neutralizer for indirect prompt injection in retrieved RAG contexts.
    """

    def __init__(self):
        self.unaccented_signatures = []
        for sig in INDIRECT_INJECTION_SIGNATURES:
            self.unaccented_signatures.append(strip_vietnamese_accents(sig.lower()))

        self.unaccented_override_verbs = []
        for v in OVERRIDE_VERBS:
            self.unaccented_override_verbs.append(strip_vietnamese_accents(v.lower()))

        self.unaccented_targets = []
        for t in TARGET_WORDS:
            self.unaccented_targets.append(strip_vietnamese_accents(t.lower()))

    def scan_text(self, text):
        """
        Scan a text fragment for indirect prompt injection indicators.
        Returns dict:
            is_safe (bool)
            threats (list)
            sanitized_text (str)
        """
        if not text:
            return {
                "is_safe": True,
                "threats": [],
                "sanitized_text": ""
            }

        text_lower = text.lower()
        text_unaccented = strip_vietnamese_accents(text_lower)
        text_condensed = condense_spaced_text(text_lower)
        text_condensed_unaccented = strip_vietnamese_accents(text_condensed)

        threats_found = []

        # 1. Direct accented signatures
        for sig in INDIRECT_INJECTION_SIGNATURES:
            if sig in text_lower:
                threats_found.append(f"INJECTION_SIGNATURE:{sig}")

        # 2. Unaccented signatures
        for u_sig in self.unaccented_signatures:
            if u_sig in text_unaccented:
                threat_tag = f"UNACCENTED_INJECTION_SIGNATURE:{u_sig}"
                if threat_tag not in threats_found:
                    threats_found.append(threat_tag)

        # 3. Condensed spaced signatures
        for sig in INDIRECT_INJECTION_SIGNATURES:
            if sig in text_condensed:
                threat_tag = f"SPACED_INJECTION_SIGNATURE:{sig}"
                if threat_tag not in threats_found:
                    threats_found.append(threat_tag)

        for u_sig in self.unaccented_signatures:
            if u_sig in text_condensed_unaccented:
                threat_tag = f"UNACCENTED_SPACED_INJECTION_SIGNATURE:{u_sig}"
                if threat_tag not in threats_found:
                    threats_found.append(threat_tag)

        # No-space signatures
        text_nospace = text_lower.replace(" ", "")
        text_unaccented_nospace = text_unaccented.replace(" ", "")

        for sig in INDIRECT_INJECTION_SIGNATURES:
            sig_nospace = sig.replace(" ", "")
            if len(sig_nospace) >= _NOSPACE_MIN_LENGTH:
                if sig_nospace in text_nospace:
                    threat_tag = f"SPACED_INJECTION_SIGNATURE:{sig}"
                    if threat_tag not in threats_found:
                        threats_found.append(threat_tag)

        for u_sig in self.unaccented_signatures:
            u_sig_nospace = u_sig.replace(" ", "")
            if len(u_sig_nospace) >= _NOSPACE_MIN_LENGTH:
                if u_sig_nospace in text_unaccented_nospace:
                    threat_tag = f"UNACCENTED_SPACED_INJECTION_SIGNATURE:{u_sig}"
                    if threat_tag not in threats_found:
                        threats_found.append(threat_tag)

        # 4. Fuzzy verb + target co-occurrence
        # A verb only counts as an override when its target follows within a few words
        # ("bo qua huong dan", "ignore previous instructions"). Distant co-occurrence
        # such as "quen mat khau ... bao mat" is normal banking text.
        words = text_unaccented.split()
        found_pair = ""
        for v in self.unaccented_override_verbs:
            verb_words = v.split()
            for index in range(len(words) - len(verb_words) + 1):
                if words[index:index + len(verb_words)] != verb_words:
                    continue
                window = " ".join(words[index + len(verb_words):index + len(verb_words) + _OVERRIDE_WINDOW_WORDS + 3])
                for t in self.unaccented_targets:
                    if window.startswith(t) or (" " + t) in (" " + window):
                        near = window.split(t)[0].split()
                        if len(near) <= _OVERRIDE_WINDOW_WORDS:
                            found_pair = v + "+" + t
                            break
                if found_pair:
                    break
            if found_pair:
                break

        if found_pair:
            cooccur_tag = f"COOCCURRENCE_INJECTION:{found_pair}"
            if cooccur_tag not in threats_found:
                threats_found.append(cooccur_tag)

        # 5. Dangerous HTML elements
        for elem in DANGEROUS_HTML_ELEMENTS:
            if elem in text_lower:
                threats_found.append(f"DANGEROUS_HTML:{elem}")

        # 6. Dangerous data URIs
        for uri in DANGEROUS_DATA_URIS:
            if uri in text_lower:
                threats_found.append(f"DANGEROUS_DATA_URI:{uri}")

        # 7. Exfiltration image tags
        malicious_images, _ = extract_and_validate_markdown_images(text)
        if len(malicious_images) > 0:
            threats_found.append("DATA_EXFILTRATION_IMAGE_TAG")

        is_safe = True
        sanitized = text

        if len(threats_found) > 0:
            is_safe = False

            for sig in INDIRECT_INJECTION_SIGNATURES:
                if sig in sanitized.lower():
                    idx = sanitized.lower().find(sig)
                    while idx != -1:
                        end_pos = idx + len(sig)
                        sanitized = sanitized[:idx] + _REDACTION_INJECTION + sanitized[end_pos:]
                        idx = sanitized.lower().find(sig)

            for elem in DANGEROUS_HTML_ELEMENTS:
                if elem in sanitized.lower():
                    idx = sanitized.lower().find(elem)
                    while idx != -1:
                        end_pos = idx + len(elem)
                        sanitized = sanitized[:idx] + _REDACTION_SCRIPT + sanitized[end_pos:]
                        idx = sanitized.lower().find(elem)

            for mal_img in malicious_images:
                if mal_img in sanitized:
                    sanitized = sanitized.replace(mal_img, _REDACTION_IMAGE)

        return {
            "is_safe": is_safe,
            "threats": threats_found,
            "sanitized_text": sanitized
        }

    def _scan_and_sanitize_table_data(self, table_data):
        """Scan table_data text fields (title, headers, rows, split keys) and return (threats, cleaned_table, infected)."""
        if not isinstance(table_data, dict) or not table_data:
            return [], table_data, False

        threats = []
        infected = False
        cleaned = dict(table_data)

        for str_key in _TABLE_TEXT_FIELDS:
            if str_key in cleaned and isinstance(cleaned[str_key], str) and cleaned[str_key]:
                res = self.scan_text(cleaned[str_key])
                if res["is_safe"] is False:
                    infected = True
                    threats.extend(res["threats"])
                    cleaned[str_key] = res["sanitized_text"]

        if isinstance(cleaned.get("headers"), list):
            new_headers = []
            for h in cleaned["headers"]:
                if isinstance(h, str) and h:
                    res = self.scan_text(h)
                    if res["is_safe"] is False:
                        infected = True
                        threats.extend(res["threats"])
                        new_headers.append(res["sanitized_text"])
                    else:
                        new_headers.append(h)
                else:
                    new_headers.append(h)
            cleaned["headers"] = new_headers

        if isinstance(cleaned.get("rows"), list):
            new_rows = []
            for row in cleaned["rows"]:
                if isinstance(row, list):
                    new_cells = []
                    for cell in row:
                        if isinstance(cell, str) and cell:
                            res = self.scan_text(cell)
                            if res["is_safe"] is False:
                                infected = True
                                threats.extend(res["threats"])
                                new_cells.append(res["sanitized_text"])
                            else:
                                new_cells.append(cell)
                        else:
                            new_cells.append(cell)
                    new_rows.append(new_cells)
                elif isinstance(row, dict):
                    new_row_dict = dict(row)
                    for rk, rv in new_row_dict.items():
                        if isinstance(rv, str) and rv:
                            res = self.scan_text(rv)
                            if res["is_safe"] is False:
                                infected = True
                                threats.extend(res["threats"])
                                new_row_dict[rk] = res["sanitized_text"]
                    new_rows.append(new_row_dict)
                else:
                    new_rows.append(row)
            cleaned["rows"] = new_rows

        return threats, cleaned, infected

    def _scan_and_sanitize_images(self, images):
        """Scan multimodal image context dicts without exposing external/injection payloads."""
        if not isinstance(images, list) or not images:
            return [], images, False

        threats = []
        infected = False
        cleaned_list = []

        for img in images:
            if not isinstance(img, dict):
                cleaned_list.append(img)
                continue
            cleaned_img = dict(img)
            for str_key in _IMAGE_TEXT_FIELDS:
                if str_key in cleaned_img and isinstance(cleaned_img[str_key], str) and cleaned_img[str_key]:
                    res = self.scan_text(cleaned_img[str_key])
                    if res["is_safe"] is False:
                        infected = True
                        threats.extend(res["threats"])
                        cleaned_img[str_key] = res["sanitized_text"]
            for url_key in _IMAGE_URL_FIELDS:
                if url_key in cleaned_img and isinstance(cleaned_img[url_key], str):
                    u_low = cleaned_img[url_key].strip().lower()
                    is_dangerous_url = False
                    for scheme in _EXTERNAL_HTTP_SCHEMES:
                        if u_low.startswith(scheme):
                            is_dangerous_url = True
                            break
                    if not is_dangerous_url:
                        for sub in _DANGEROUS_IMAGE_URL_SUBSTRINGS:
                            if sub in u_low:
                                is_dangerous_url = True
                                break
                    if is_dangerous_url:
                        infected = True
                        threats.append(f"EXTERNAL_OR_DANGEROUS_IMAGE_URL:{u_low[:_URL_PREVIEW_CHARS]}")
                        cleaned_img[url_key] = ""
            cleaned_list.append(cleaned_img)

        return threats, cleaned_list, infected

    def _scan_single_chunk_dict(self, target_dict):
        """Scan all text-bearing fields of a chunk dict without injecting extra top-level schema keys."""
        if not isinstance(target_dict, dict):
            return "safe", target_dict, []

        all_threats = []
        cleaned = dict(target_dict)

        for field_key in _CHUNK_TEXT_FIELDS:
            val = cleaned.get(field_key)
            if isinstance(val, str) and val:
                res = self.scan_text(val)
                if res["is_safe"] is False:
                    all_threats.extend(res["threats"])
                    cleaned[field_key] = res["sanitized_text"]

        if "table_data" in cleaned and isinstance(cleaned["table_data"], dict):
            t_threats, t_clean, t_inf = self._scan_and_sanitize_table_data(cleaned["table_data"])
            if t_inf:
                all_threats.extend(t_threats)
                cleaned["table_data"] = t_clean

        if "images" in cleaned and isinstance(cleaned["images"], list):
            i_threats, i_clean, i_inf = self._scan_and_sanitize_images(cleaned["images"])
            if i_inf:
                all_threats.extend(i_threats)
                cleaned["images"] = i_clean

        bmeta = cleaned.get("business_metadata")
        if isinstance(bmeta, dict) and isinstance(bmeta.get("retrieval"), dict):
            ret_copy = dict(bmeta["retrieval"])
            ret_changed = False
            for rk in _RETRIEVAL_CONTEXT_FIELDS:
                rv = ret_copy.get(rk)
                if isinstance(rv, str) and rv:
                    res = self.scan_text(rv)
                    if res["is_safe"] is False:
                        all_threats.extend(res["threats"])
                        ret_copy[rk] = res["sanitized_text"]
                        ret_changed = True
            if ret_changed:
                bmeta_copy = dict(bmeta)
                bmeta_copy["retrieval"] = ret_copy
                cleaned["business_metadata"] = bmeta_copy

        if not all_threats:
            return "safe", target_dict, []

        is_critical = False
        if len(all_threats) >= _CRITICAL_THREAT_MIN_COUNT:
            is_critical = True
        for t in all_threats:
            for marker in _CRITICAL_THREAT_MARKERS:
                if marker in t:
                    is_critical = True
                    break
            if is_critical is True:
                break

        if is_critical is True:
            return "blocked", None, all_threats
        return "sanitized", cleaned, all_threats

    def scan_chunks(self, chunks):
        """
        Scan all retrieved RAG chunks across 13-field universal schema fields
        (content, parent_content, section_title, breadcrumb, table_data, images, business_metadata)
        plus wrapped {"chunk": ...} and legacy dicts, preserving strict 13-key schema keys.
        """
        if not chunks:
            return [], 0, []

        safe_chunks = []
        blocked_count = 0
        threat_details = []

        for chunk in chunks:
            if isinstance(chunk, str):
                res = self.scan_text(chunk)
                if res["is_safe"] is True:
                    safe_chunks.append(chunk)
                else:
                    threat_details.extend(res["threats"])
                    if len(res["threats"]) >= _CRITICAL_THREAT_MIN_COUNT:
                        blocked_count += 1
                    else:
                        safe_chunks.append(res["sanitized_text"])
                continue

            if not isinstance(chunk, dict):
                safe_chunks.append(chunk)
                continue

            if "chunk" in chunk and isinstance(chunk["chunk"], dict):
                status, cleaned_inner, c_threats = self._scan_single_chunk_dict(chunk["chunk"])
                threat_details.extend(c_threats)
                if status == "blocked":
                    blocked_count += 1
                elif status == "sanitized":
                    cleaned_outer = dict(chunk)
                    cleaned_outer["chunk"] = cleaned_inner
                    safe_chunks.append(cleaned_outer)
                else:
                    safe_chunks.append(chunk)
            else:
                status, cleaned_chunk, c_threats = self._scan_single_chunk_dict(chunk)
                threat_details.extend(c_threats)
                if status == "blocked":
                    blocked_count += 1
                elif status == "sanitized":
                    safe_chunks.append(cleaned_chunk)
                else:
                    safe_chunks.append(chunk)

        return safe_chunks, blocked_count, threat_details


_DEFAULT_INDIRECT_GUARDRAIL = None


def get_indirect_injection_guardrail():
    """Get or initialize singleton instance of IndirectInjectionGuardrail."""
    global _DEFAULT_INDIRECT_GUARDRAIL
    if _DEFAULT_INDIRECT_GUARDRAIL is None:
        _DEFAULT_INDIRECT_GUARDRAIL = IndirectInjectionGuardrail()
    return _DEFAULT_INDIRECT_GUARDRAIL


def scan_text(text):
    """Module-level wrapper for scan_text."""
    guardrail = get_indirect_injection_guardrail()
    return guardrail.scan_text(text)


def scan_chunks(chunks):
    """Module-level wrapper for scan_chunks."""
    guardrail = get_indirect_injection_guardrail()
    return guardrail.scan_chunks(chunks)
