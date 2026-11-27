"""Normalize banking queries for the analysis tool and hybrid retrieval."""

from dataclasses import dataclass, field
import unicodedata

from agent_banking.core.numeric_tokens import extract_canonical_numeric_tokens
from agent_banking.tools.config.settings import get_query_rules

_RULES = get_query_rules()
DEFAULT_BANKING_ACRONYMS = dict(_RULES["banking_acronyms"])
VIETNAMESE_STOPWORDS = set(_RULES["stopwords"])
BANK_RECOGNITION_MAP = dict(_RULES["bank_recognition_map"])
_AMBIGUOUS_TERMS = {k: tuple(v) for k, v in _RULES["ambiguous_terms"].items()}
_STRIP_PUNCT = str(_RULES["strip_punctuation"])
_DENSE_PREFIX = str(_RULES["dense_query_prefix"])
_LEGAL_RULES = _RULES["legal_extraction"]
_AMOUNT_RULES = _RULES["financial_extraction"]
_INTENT_RULES = _RULES["intent_classification"]
_DEFAULT_INTENT = _INTENT_RULES["defaults"]
_TERM_BRIDGES = list(_RULES.get("terminology_bridges", ()))


@dataclass
class ParsedQuery:
    """
    Banking query parse result contract.
    Maintains 100% backward compatibility with Retrieval, Tracer, and Pipeline modules.
    """
    raw_query: str
    normalized_query: str
    keywords: list
    keywords_str: str
    detected_acronyms: list = field(default_factory=list)
    legal_references: list = field(default_factory=list)
    financial_amounts: list = field(default_factory=list)
    banks_mentioned: list = field(default_factory=list)
    intent: str = str(_DEFAULT_INTENT["intent"])
    sub_intent: str = str(_DEFAULT_INTENT["sub_intent"])
    domain: str = str(_DEFAULT_INTENT["domain"])
    suggested_filters: dict = field(default_factory=dict)
    expanded_query: str = ""

    @property
    def entities(self):
        """Backward-compatible alias for detected_acronyms."""
        return self.detected_acronyms

    def to_dict(self):
        """Export dictionary structure for JSON serialization."""
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "keywords": self.keywords,
            "keywords_str": self.keywords_str,
            "detected_acronyms": self.detected_acronyms,
            "legal_references": self.legal_references,
            "financial_amounts": self.financial_amounts,
            "banks_mentioned": self.banks_mentioned,
            "intent": self.intent,
            "sub_intent": self.sub_intent,
            "domain": self.domain,
            "suggested_filters": self.suggested_filters,
            "expanded_query": self.expanded_query,
            "entities": self.detected_acronyms
        }


def _has_digit(text: str) -> bool:
    for ch in text:
        if ch.isdigit():
            return True
    return False


def _match_sub_rules(lower_q: str, words_set: set, branch: dict) -> tuple:
    sub_intent = branch["default_sub_intent"]
    filters = dict(branch.get("suggested_filters", {}))
    for sub in branch.get("sub_rules", ()):
        matched_word = any(w in words_set for w in sub.get("any_words", ()))
        matched_phrase = any(p in lower_q for p in sub.get("any_phrases", ()))
        if matched_word or matched_phrase:
            sub_intent = sub["sub_intent"]
            if "suggested_filters" in sub:
                filters.update(sub["suggested_filters"])
            break
    return sub_intent, filters


class BankingQueryParser:
    """
    Normalize Unicode, expand known terms and extract search metadata.
    """

    def __init__(self, acronym_dict=None):
        if acronym_dict is not None:
            raw_dict = acronym_dict
        else:
            raw_dict = DEFAULT_BANKING_ACRONYMS

        # Normalize all dictionary keys to lowercase Unicode NFC
        self.acronym_dict = {}
        for k, v in raw_dict.items():
            norm_k = unicodedata.normalize("NFC", k.strip().lower())
            self.acronym_dict[norm_k] = v

        # Sort longer phrases first to prioritize multi-word matches over single words
        self.sorted_acronyms = sorted(
            self.acronym_dict.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

    def normalize_acronyms(self, query):
        """
        Normalize and expand banking acronyms (0% Regex).
        Explicit word-boundary validation prevents substring collisions.
        """
        if not query:
            return "", []

        text = unicodedata.normalize("NFC", query)
        lower_text = text.lower()
        t_len = len(text)
        matches = []

        occupied = bytearray(t_len)

        for phrase, replacement in self.sorted_acronyms:
            p_len = len(phrase)
            if phrase not in lower_text:
                continue

            pos = 0
            while pos < t_len:
                idx = lower_text.find(phrase, pos)
                if idx == -1:
                    break

                end_idx = idx + p_len

                # 1. Check leading word boundary
                before_ok = False
                if idx == 0:
                    before_ok = True
                else:
                    prev_char = text[idx - 1]
                    if not prev_char.isalnum():
                        if prev_char in ("-", "/") and idx >= 2 and text[idx - 2].isalnum():
                            before_ok = False
                        else:
                            before_ok = True

                # 2. Check trailing word boundary
                after_ok = False
                if end_idx == t_len:
                    after_ok = True
                else:
                    next_char = text[end_idx]
                    if not next_char.isalnum():
                        if next_char in ("-", "/") and end_idx + 1 < t_len and text[end_idx + 1].isalnum():
                            after_ok = False
                        else:
                            after_ok = True

                if before_ok and after_ok:
                    if b"\x01" not in occupied[idx:end_idx]:
                        occupied[idx:end_idx] = b"\x01" * p_len
                        matches.append((idx, end_idx, phrase, replacement))
                        pos = end_idx
                    else:
                        pos = idx + 1
                else:
                    pos = idx + 1

        if not matches:
            clean_words = text.split()
            normalized = " ".join(clean_words)
            return normalized, []

        matches.sort(key=lambda x: x[0])

        parts = []
        detected = []
        last_pos = 0

        for start_idx, end_idx, phrase, replacement in matches:
            parts.append(text[last_pos:start_idx])
            parts.append(replacement)
            original_token = text[start_idx:end_idx]
            detected.append((original_token, replacement))
            last_pos = end_idx

        parts.append(text[last_pos:])
        joined_text = "".join(parts)
        clean_words = joined_text.split()
        normalized = " ".join(clean_words)
        return normalized, detected

    @staticmethod
    def extract_legal_entities(query):
        """
        Extract banking legal entities and state regulatory citations (0% Regex),
        driven by tools/config/settings/query_rules.yaml.
        """
        if not query:
            return []

        text = unicodedata.normalize("NFC", query)
        words = text.split()
        refs = []
        num_words = len(words)
        conjunctions = tuple(_LEGAL_RULES["conjunctions"])
        numbered_units = _LEGAL_RULES["numbered_units"]
        doc_rules = _LEGAL_RULES["document_rules"]
        num_markers = tuple(_LEGAL_RULES["number_marker_tokens"])

        for i in range(num_words):
            raw_w = words[i]
            clean_token = raw_w.strip(_STRIP_PUNCT)
            clean_lower = clean_token.lower()

            matched_numbered = False
            for unit in numbered_units:
                if clean_lower in unit["tokens"]:
                    matched_numbered = True
                    label = unit["label"]
                    if i + 1 < num_words:
                        next_raw = words[i + 1].strip(_STRIP_PUNCT)
                        if _has_digit(next_raw):
                            refs.append(f"{label} {next_raw}")
                        if i + 3 < num_words:
                            sep_word = words[i + 2].strip(_STRIP_PUNCT).lower()
                            if sep_word in conjunctions:
                                comp_raw = words[i + 3].strip(_STRIP_PUNCT)
                                if _has_digit(comp_raw):
                                    refs.append(f"{label} {comp_raw}")
                    break
            if matched_numbered:
                continue

            if clean_lower in _LEGAL_RULES["point_tokens"]:
                if i + 1 < num_words:
                    next_raw = words[i + 1].strip(_STRIP_PUNCT)
                    if len(next_raw) <= int(_LEGAL_RULES["point_max_length"]) and next_raw.isalnum():
                        refs.append(f"{_LEGAL_RULES['point_label']} {next_raw}")
                continue

            if clean_lower in _LEGAL_RULES["chapter_tokens"]:
                if i + 1 < num_words:
                    next_raw = words[i + 1].strip(_STRIP_PUNCT)
                    if len(next_raw) <= int(_LEGAL_RULES["chapter_max_length"]):
                        refs.append(f"{_LEGAL_RULES['chapter_label']} {next_raw}")
                continue

            if clean_lower in _LEGAL_RULES["section_tokens"]:
                if i + 1 < num_words:
                    next_raw = words[i + 1].strip(_STRIP_PUNCT)
                    if len(next_raw) <= int(_LEGAL_RULES["section_max_length"]) and next_raw.isalnum():
                        refs.append(f"{_LEGAL_RULES['section_label']} {next_raw}")
                continue

            if clean_lower in _LEGAL_RULES["appendix_first_tokens"]:
                if i + 1 < num_words:
                    w2 = words[i + 1].strip(_STRIP_PUNCT).lower()
                    if w2 in _LEGAL_RULES["appendix_second_tokens"]:
                        if i + 2 < num_words:
                            next_raw = words[i + 2].strip(_STRIP_PUNCT)
                            if next_raw and len(next_raw) <= int(_LEGAL_RULES["appendix_max_length"]):
                                refs.append(f"{_LEGAL_RULES['appendix_label']} {next_raw}")
                continue

            for rule in doc_rules:
                label = rule["label"]
                allow_num = bool(rule["allow_number_marker"])
                if clean_lower in rule["first_tokens"]:
                    if i + 1 < num_words:
                        w2 = words[i + 1].strip(_STRIP_PUNCT).lower()
                        if w2 in rule["second_tokens"]:
                            target_idx = i + 2
                            if allow_num and target_idx < num_words:
                                w3 = words[target_idx].strip(_STRIP_PUNCT).lower()
                                if w3 in num_markers:
                                    target_idx = target_idx + 1
                            if target_idx < num_words:
                                ref_code = words[target_idx].strip(_STRIP_PUNCT)
                                if _has_digit(ref_code):
                                    refs.append(f"{label} {ref_code}")
                    break
                if clean_lower in rule["short_tokens"]:
                    target_idx = i + 1
                    if allow_num and target_idx < num_words:
                        w2 = words[target_idx].strip(_STRIP_PUNCT).lower()
                        if w2 in num_markers:
                            target_idx = target_idx + 1
                    if target_idx < num_words:
                        ref_code = words[target_idx].strip(_STRIP_PUNCT)
                        if _has_digit(ref_code):
                            refs.append(f"{label} {ref_code}")
                    break
                matched_prefix = False
                for pfx in rule["prefixes"]:
                    if clean_lower.startswith(pfx) and len(clean_token) > len(pfx):
                        code_part = ""
                        p_len = len(pfx)
                        if clean_token[p_len].isdigit():
                            code_part = clean_token[p_len:]
                        elif clean_token[p_len] in rule["separators"] and len(clean_token) > p_len + 1:
                            if clean_token[p_len + 1].isdigit():
                                code_part = clean_token[p_len + 1:]
                        if code_part:
                            refs.append(f"{label} {code_part}")
                        matched_prefix = True
                        break
                if matched_prefix:
                    break

        lower_q = text.lower()
        for law in _LEGAL_RULES["statutory_laws"]:
            if any(trigger in lower_q for trigger in law["triggers"]):
                refs.append(law["label"])

        unique_refs = []
        for r in refs:
            if r not in unique_refs:
                unique_refs.append(r)
        return unique_refs

    @staticmethod
    def _parse_amount_value(num_str, multiplier):
        """
        Convert numeric string combined with multiplier to float/int.
        """
        clean = num_str.strip()
        if not clean:
            return None

        dot_count = clean.count(".")
        comma_count = clean.count(",")

        if dot_count > 1 or comma_count > 1:
            raw_digits = clean.replace(".", "").replace(",", "")
            if raw_digits.isdigit():
                return int(raw_digits) * multiplier
            return None

        if dot_count == 1 or comma_count == 1:
            sep = "."
            if comma_count == 1:
                sep = ","
            parts = clean.split(sep)
            if len(parts) == 2:
                left = parts[0]
                right = parts[1]
                if left.isdigit() and right.isdigit():
                    if multiplier > 1 and len(right) <= 2:
                        val_float = float(f"{left}.{right}")
                        return int(round(val_float * multiplier))
                    if len(right) == 3 and multiplier == 1:
                        return int(left + right)
                    if multiplier > 1:
                        val_float = float(f"{left}.{right}")
                        return int(round(val_float * multiplier))

        if clean.isdigit():
            return int(clean) * multiplier

        return None

    @staticmethod
    def extract_financial_amounts(query):
        """
        Extract financial amounts & currency formats in banking queries (0% Regex),
        driven by tools/config/settings/query_rules.yaml.
        """
        if not query:
            return []

        words = query.lower().split()
        amounts = []
        num_words = len(words)
        attached_units = _AMOUNT_RULES["attached_units"]
        spaced_units = _AMOUNT_RULES["spaced_units"]
        min_vnd = int(_AMOUNT_RULES["min_vnd_amount"])
        currency_suffix = str(_AMOUNT_RULES["currency_suffix"])

        for i in range(num_words):
            w = words[i].strip(_STRIP_PUNCT)
            if not w:
                continue

            cut_idx = -1
            for ci, char in enumerate(w):
                if not char.isdigit() and char not in (".", ","):
                    cut_idx = ci
                    break
            if cut_idx > 0:
                raw_num_part = w[:cut_idx]
                unit_part = w[cut_idx:].strip()
                multiplier = 0
                for unit in attached_units:
                    if unit_part in unit["tokens"]:
                        multiplier = int(unit["multiplier"])
                        break

                if multiplier > 0:
                    num_val = BankingQueryParser._parse_amount_value(raw_num_part, multiplier)
                    if num_val is not None and num_val > 0:
                        if multiplier != 1 or num_val >= min_vnd:
                            formatted_str = f"{num_val:,}".replace(",", ".") + currency_suffix
                            amounts.append({
                                "raw": w,
                                "value": num_val,
                                "formatted": formatted_str
                            })

            if i + 1 < num_words:
                next_w = words[i + 1].strip(_STRIP_PUNCT)
                multiplier = 0
                for unit in spaced_units:
                    if next_w in unit["tokens"]:
                        multiplier = int(unit["multiplier"])
                        break

                if multiplier > 0:
                    num_val = BankingQueryParser._parse_amount_value(w, multiplier)
                    if num_val is not None and num_val > 0:
                        if multiplier != 1 or num_val >= min_vnd:
                            raw_str = f"{w} {next_w}"
                            formatted_str = f"{num_val:,}".replace(",", ".") + currency_suffix
                            amounts.append({
                                "raw": raw_str,
                                "value": num_val,
                                "formatted": formatted_str
                            })

        unique_amounts = []
        seen_vals = set()
        for a in amounts:
            val = a["value"]
            if val not in seen_vals:
                seen_vals.add(val)
                unique_amounts.append(a)
        return unique_amounts

    @staticmethod
    def extract_bank_mentions(query):
        """
        Identify commercial bank and payment intermediary names mentioned in the query.
        """
        if not query:
            return []

        lower_q = query.lower()
        banks = []
        q_len = len(lower_q)

        for code, bname in BANK_RECOGNITION_MAP.items():
            pos = 0
            p_len = len(code)
            while pos < q_len:
                idx = lower_q.find(code, pos)
                if idx == -1:
                    break

                before_ok = False
                if idx == 0:
                    before_ok = True
                else:
                    prev_c = lower_q[idx - 1]
                    if not prev_c.isalnum():
                        before_ok = True

                after_ok = False
                if idx + p_len == q_len:
                    after_ok = True
                else:
                    next_c = lower_q[idx + p_len]
                    if not next_c.isalnum():
                        after_ok = True

                if before_ok and after_ok:
                    if bname not in banks:
                        banks.append(bname)
                    break
                else:
                    pos = idx + 1

        return banks

    @staticmethod
    def classify_intent_and_domain(normalized_query, detected_acronyms=None, legal_refs=None):
        """
        Classify intent (12 Intents Taxonomy) and banking domain (8 Domains) (0% Regex),
        driven by tools/config/settings/query_rules.yaml.
        """
        lower_q = normalized_query.lower()

        words_list = lower_q.split()
        words_set = set()
        for w in words_list:
            clean_w = w.strip(_STRIP_PUNCT)
            if clean_w:
                words_set.add(clean_w)

        intent = str(_DEFAULT_INTENT["intent"])
        sub_intent = str(_DEFAULT_INTENT["sub_intent"])
        domain = str(_DEFAULT_INTENT["domain"])
        suggested_filters = {}

        for rule in _INTENT_RULES["rules"]:
            excludes = rule.get("exclude_phrases", ())
            if any(ep in lower_q for ep in excludes):
                continue

            has_legal = bool(rule.get("match_legal_refs")) and legal_refs is not None and len(legal_refs) > 0
            has_phrase = any(p in lower_q for p in rule.get("any_phrases", ()))
            if not (has_legal or has_phrase):
                continue

            if "disambiguate" in rule:
                disamb = rule["disambiguate"]
                has_explicit_transfer = any(k in lower_q for k in disamb["transfer_phrases"])
                is_opening_or_management = False
                if any(w in words_set for w in disamb["management_words"]):
                    is_opening_or_management = True
                elif any(k in lower_q for k in disamb["management_phrases"]):
                    is_opening_or_management = True

                if has_explicit_transfer and not is_opening_or_management:
                    branch = disamb["transfer_branch"]
                else:
                    branch = disamb["account_branch"]
                intent = branch["intent"]
                domain = branch["domain"]
                sub_intent, suggested_filters = _match_sub_rules(lower_q, words_set, branch)
                break

            intent = rule["intent"]
            domain = rule["domain"]
            sub_intent, suggested_filters = _match_sub_rules(lower_q, words_set, rule)
            break

        return intent, sub_intent, domain, suggested_filters

    @staticmethod
    def extract_keywords(normalized_query):
        """
        Extract salient keywords for BM25 retrieval (0% Regex).
        """
        if not normalized_query:
            return []

        text_nfc = unicodedata.normalize("NFC", normalized_query)

        masked_chars = list(text_nfc)
        t_len = len(text_nfc)
        for i, ch in enumerate(text_nfc):
            if ch in (".", ",") and i > 0 and i + 1 < t_len:
                if text_nfc[i - 1].isdigit() and text_nfc[i + 1].isdigit():
                    left_start = i - 1
                    while left_start > 0 and text_nfc[left_start - 1].isdigit():
                        left_start = left_start - 1

                    right_end = i + 1
                    while right_end + 1 < t_len and text_nfc[right_end + 1].isdigit():
                        right_end = right_end + 1
                    right_len = right_end - i

                    has_percent = False
                    after_idx = right_end + 1
                    while after_idx < t_len and text_nfc[after_idx] == " ":
                        after_idx = after_idx + 1
                    if after_idx < t_len and text_nfc[after_idx] == "%":
                        has_percent = True

                    if right_len <= 2 and (ch == "," or has_percent):
                        # Preserve decimal notation inside tokens ('1,5' -> '1.5' and '1,5') instead of blanking it.
                        if ch == ",":
                            masked_chars[i] = "."
                            continue

        cleaned_chars = []
        for idx_c, c in enumerate(masked_chars):
            if c.isalnum() or c.isspace():
                cleaned_chars.append(c)
            elif c == "." and 0 < idx_c < t_len - 1 and masked_chars[idx_c - 1].isdigit() and masked_chars[idx_c + 1].isdigit():
                cleaned_chars.append(c)
            else:
                cleaned_chars.append(" ")
        cleaned_text = "".join(cleaned_chars)

        words = cleaned_text.lower().split()
        keywords = []

        for w in words:
            if len(w) > 1 or w.isdigit():
                if w not in VIETNAMESE_STOPWORDS:
                    if w not in keywords:
                        keywords.append(w)

        return keywords

    def parse(self, query):
        """
        Prepare the raw query for hybrid search.
        Takes raw user query string -> Returns fully standardized ParsedQuery object.
        """
        if not query:
            clean_query = ""
        else:
            clean_query = " ".join(unicodedata.normalize("NFC", str(query).strip()).split())

        if not clean_query:
            return ParsedQuery(
                raw_query="",
                normalized_query="",
                keywords=[],
                keywords_str="",
                detected_acronyms=[],
                legal_references=[],
                financial_amounts=[],
                banks_mentioned=[],
                intent=str(_DEFAULT_INTENT["intent"]),
                sub_intent=str(_DEFAULT_INTENT["sub_intent"]),
                domain=str(_DEFAULT_INTENT["domain"]),
                suggested_filters={},
                expanded_query=""
            )

        # 1. Normalize acronyms
        normalized_query, detected_acronyms = self.normalize_acronyms(clean_query)

        # 2. Extract legal entities (merged from original and expanded text)
        refs_raw = self.extract_legal_entities(clean_query)
        refs_norm = self.extract_legal_entities(normalized_query)
        legal_refs = []
        for r in refs_raw + refs_norm:
            if r not in legal_refs:
                legal_refs.append(r)

        # 3. Extract BM25 keywords
        keywords = self.extract_keywords(normalized_query)
        compound_codes = []
        for raw_tok in clean_query.split():
            trimmed = raw_tok.strip(_STRIP_PUNCT)
            if ("-" in trimmed or "/" in trimmed) and _has_digit(trimmed):
                if len(trimmed) <= 28 and trimmed not in compound_codes:
                    compound_codes.append(trimmed)
        keywords_parts = list(keywords)
        for code_tok in compound_codes:
            low_code = code_tok.lower()
            if low_code not in keywords_parts:
                keywords_parts.append(low_code)

        lower_norm = normalized_query.lower()
        bridge_expansions = []
        for bridge in _TERM_BRIDGES:
            triggers = bridge.get("triggers", ())
            if any(trig in lower_norm or trig in clean_query.lower() for trig in triggers):
                exp_text = str(bridge.get("expansion") or "").strip()
                if exp_text and exp_text not in bridge_expansions:
                    bridge_expansions.append(exp_text)
                    for b_tok in exp_text.lower().split():
                        b_clean = b_tok.strip(_STRIP_PUNCT)
                        if len(b_clean) >= 2 and b_clean not in keywords_parts:
                            keywords_parts.append(b_clean)

        for num_tok in extract_canonical_numeric_tokens(clean_query, max_tokens=8):
            if num_tok not in keywords_parts:
                keywords_parts.append(num_tok)
        keywords_str = " ".join(keywords_parts)

        # 4. Extract monetary amounts and mentioned banks
        financial_amounts = self.extract_financial_amounts(clean_query)
        banks_mentioned = self.extract_bank_mentions(clean_query)

        # 5. Classify intent and domain
        intent, sub_intent, domain, suggested_filters = self.classify_intent_and_domain(
            normalized_query=normalized_query,
            detected_acronyms=detected_acronyms,
            legal_refs=legal_refs
        )

        # 6. Construct expanded query for dense search
        expanded_parts = [normalized_query]
        if legal_refs:
            for ref in legal_refs:
                if ref.lower() not in normalized_query.lower():
                    expanded_parts.append(ref)
        for exp_text in bridge_expansions:
            if exp_text.lower() not in normalized_query.lower():
                expanded_parts.append(exp_text)
        combined_text = " ".join(expanded_parts).strip()
        if not combined_text.startswith(_DENSE_PREFIX):
            expanded_query = f"{_DENSE_PREFIX}{combined_text}"
        else:
            expanded_query = combined_text

        return ParsedQuery(
            raw_query=clean_query,
            normalized_query=normalized_query,
            keywords=keywords,
            keywords_str=keywords_str,
            detected_acronyms=detected_acronyms,
            legal_references=legal_refs,
            financial_amounts=financial_amounts,
            banks_mentioned=banks_mentioned,
            intent=intent,
            sub_intent=sub_intent,
            domain=domain,
            suggested_filters=suggested_filters,
            expanded_query=expanded_query
        )


def decompose_subqueries(clean_query: str) -> list:
    """Deterministically decompose comparison or multi-clause banking queries into <= 2 focused sub-queries."""
    if not clean_query:
        return []
    words = clean_query.split()
    if len(words) < 10:
        return []
    work = " ".join(unicodedata.normalize("NFC", clean_query).split())
    low = work.lower()

    # 1. Comparison query: "So sánh <chủ đề> giữa <A> và <B>"
    comp_markers = ("so sánh ", "khác nhau ")
    if any(m in low for m in comp_markers) and " giữa " in low:
        giua_idx = low.find(" giữa ")
        topic = work[:giua_idx].strip()
        for prefix in ("so sánh ", "sự khác nhau ", "điểm khác nhau "):
            if topic.lower().startswith(prefix):
                topic = topic[len(prefix):].strip()
                break
        rest = work[giua_idx + len(" giữa "):].strip(" ?.!:,")
        for tail_split in (" tại ", " ở ", " trên ", " của "):
            t_idx = rest.lower().find(tail_split)
            if t_idx > 0 and len(rest[:t_idx].split()) >= 5:
                rest = rest[:t_idx].strip()
                break
        va_idx = rest.lower().rfind(" và ")
        if va_idx > 0:
            side_a = rest[:va_idx].strip(" ,;:")
            side_b = rest[va_idx + len(" và "):].strip(" ,;:")
            if side_a.lower().startswith("là "):
                side_a = side_a[3:].strip()
            if side_b.lower().startswith("là "):
                side_b = side_b[3:].strip()
            if len(side_a.split()) >= 2 and len(side_b.split()) >= 2:
                sub_1 = f"{topic} {side_a}".strip()
                sub_2 = f"{topic} {side_b}".strip()
                if sub_1.lower() != sub_2.lower():
                    return [sub_1, sub_2]

    # 2. Compound multi-clause banking queries
    if len(words) >= 14:
        splitters = (
            " đồng thời ",
            " bên cạnh đó ",
            " ngoài ra ",
            " và chịu chi phí ",
            " và chịu khoản phí ",
            " và chịu phí ",
            " và mức phí ",
            " và thời gian xử lý ",
        )
        for sp in splitters:
            sp_idx = low.find(sp)
            if sp_idx > 0:
                left = work[:sp_idx].strip(" ,;.")
                if sp.startswith(" và "):
                    right = work[sp_idx + len(" và "):].strip(" ,;.?")
                else:
                    right = work[sp_idx + len(sp):].strip(" ,;.?")
                if len(left.split()) >= 3 and len(right.split()) >= 3:
                    subj_prefix = " ".join(left.split()[:6])
                    if len(right.split()) < 7 and subj_prefix.lower() not in right.lower():
                        right = f"{subj_prefix} {right}".strip()
                    return [left, right]
    return []


_PARSER = BankingQueryParser()


def parse_banking_query(query: str) -> ParsedQuery:
    """Prepare the search query without a model call."""
    return _PARSER.parse(query)


def analyze_banking_query(query: str) -> dict:
    """Expose query analysis to the agent as structured data."""
    parsed = parse_banking_query(query)
    ambiguities = {}
    for term, _ in parsed.detected_acronyms:
        candidates = _AMBIGUOUS_TERMS.get(term.lower())
        if candidates:
            ambiguities[term] = list(candidates)
    acronyms = [list(item) for item in parsed.detected_acronyms]
    return {
        **parsed.to_dict(), "detected_acronyms": acronyms, "entities": acronyms,
        "ambiguous_acronyms": ambiguities
    }
