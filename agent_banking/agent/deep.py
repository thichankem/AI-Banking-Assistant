"""Banking agent harness with planning, isolated workers and independent review."""

import json
from unicodedata import normalize
from urllib.parse import quote, urlencode
from uuid import uuid4

from langgraph.types import interrupt

from agent_banking.agent.config.conversation_policy import (
    ANSWER_SCOPE,
    COORDINATION_POLICY,
    REVIEW_SCOPE,
    SUMMARY_INSTRUCTION,
    SUMMARY_MAX_POINTS,
    SUMMARY_MAX_WORDS,
)
from agent_banking.agent.config.response_profiles import (
    FORMAT_PROFILES,
    PERSONA_PROFILES,
    TONE_PROFILES,
    get_messages,
    get_prompt,
)
from agent_banking.agent.config.settings import (
    get_int_setting,
    get_setting,
    get_workers,
)
from agent_banking.agent.memory import (
    MAX_COMPACT_PER_TURN,
    RECENT_MESSAGES,
    change_note,
    compact_memory,
    memory_view,
    public_memory,
    should_compact,
)
from agent_banking.agent.model_protocol import ARTIFACT_REVIEW_SCHEMA, REVIEW_SCHEMA, StructuredModel
from agent_banking.core.document_revisions import is_document_version
from agent_banking.core.numeric_tokens import extract_canonical_numeric_tokens
from agent_banking.core.policy_dates import policy_query_date
from agent_banking.agent.simulator import SIMULATOR_SCHEMA, SKILL_ID as SIMULATOR_SKILL
from agent_banking.agent.visualization import (
    MAX_RECENT as MAX_RECENT_VISUALIZATIONS,
    SKILL_ID as VISUALIZATION_SKILL,
    VISUALIZATION_SCHEMA,
    recent_visualizations,
)
from agent_banking.agent.react import (
    BankingReActNodes,
    FALLBACK_ANSWER,
    MAX_EVIDENCE,
    _normalize_citation_markers,
    _reference_numbers,
    _strip_invalid_citation_markers,
    legal_searches_remaining,
)
from agent_banking.core.fallback import ModelFallbackError, is_temporary_error
from agent_banking.core.search_text import leading_step_number, search_key
from agent_banking.skills.skill_registry import resolve_skill
from agent_banking.tools.extensions.banking_query import parse_banking_query
from agent_banking.tools.extensions.legal_web_search import TOOL_NAME as LEGAL_WEB_SEARCH_TOOL
from agent_banking.tools.extensions.present_visualization import TOOL_NAME as PRESENT_VISUALIZATION_TOOL, prepare_presentation, present_visualization
from agent_banking.tracing.tracer import annotate_span


MAX_ACTIONS = get_int_setting("max_actions")
MAX_DELEGATIONS = get_int_setting("max_delegations")
MAX_WORKER_ACTIONS = get_int_setting("max_worker_actions")
MAX_REVISIONS = get_int_setting("max_revisions")
MAX_INTERACTIONS = get_int_setting("max_interactions")
MAX_ACTIVE_SKILL_BYTES = get_int_setting("max_active_skill_bytes")
RESTRICTED_CARD_FIELDS = tuple(get_setting("restricted_card_fields", ()))
WORKERS = get_workers()

BASE_PROMPT = get_prompt("BASE_PROMPT")
COORDINATOR_PROMPT = get_prompt("COORDINATOR_PROMPT")
WORKER_PROMPT = get_prompt("WORKER_PROMPT")
REVIEW_PROMPT = get_prompt("REVIEW_PROMPT")
PARTIAL_REVIEW_PROMPT = get_prompt("PARTIAL_REVIEW_PROMPT")

_RECOVER_PROMPT = get_prompt("RECOVER_PROMPT")
_PARTIAL_RECOVERY_PROMPT = get_prompt("PARTIAL_RECOVERY_PROMPT")
_COORDINATOR_SKILL_HINT = get_prompt("COORDINATOR_SKILL_HINT")
_REVISION_INSTRUCTION = get_prompt("REVISION_INSTRUCTION")
_REVIEW_REPAIR_INSTRUCTION = get_prompt("REVIEW_REPAIR_INSTRUCTION")
_DEEP_MESSAGES = get_messages("deep")

_REVIEW_NUMERIC_MARKERS = tuple(get_setting("review_numeric_markers", ()))
_COMPACT_OBSERVATION_MAX_BYTES = get_int_setting("compact_observation_max_bytes")
_MAX_SOURCE_INVENTORY_ITEMS = get_int_setting("max_source_inventory_items")
_PRESERVED_RETRIEVAL_KEYS = tuple(get_setting("preserved_retrieval_keys", ()))
_SUPPORT_HISTORY_LIMIT = get_int_setting("support_history_messages_limit")
_SUPPORT_NOTES_LIMIT = get_int_setting("support_notes_limit")
_SUPPORT_MESSAGE_MAX_CHARS = get_int_setting("support_message_max_chars")
_SUPPORT_USER_TEXT_MAX_CHARS = get_int_setting("support_user_text_max_chars")
_SUPPORT_TOOL_JSON_MAX_CHARS = get_int_setting("support_tool_json_max_chars")
_QUERY_HINTS_MAX_QUERY_CHARS = get_int_setting("query_hints_max_query_chars")
_QUERY_HINTS_MAX_TEXT_CHARS = get_int_setting("query_hints_max_text_chars")
_QUERY_HINTS_MAX_KEYWORDS = get_int_setting("query_hints_max_keywords")
_QUERY_HINTS_MAX_ACRONYMS = get_int_setting("query_hints_max_acronyms")
_HISTORY_MESSAGE_MAX_CHARS = get_int_setting("history_message_max_chars")
_REVIEW_UNLINKED_EVIDENCE_LIMIT = get_int_setting("review_unlinked_evidence_limit")
_DECIDE_MAX_TOKENS = get_int_setting("decide_max_tokens")
_RECOVER_MAX_TOKENS = get_int_setting("recover_max_tokens")
_WORKER_MAX_TOKENS = get_int_setting("worker_max_tokens")
_PARTIAL_RECOVERY_MAX_TOKENS = get_int_setting("partial_recovery_max_tokens")
_PARTIAL_RECOVERY_MAX_SOURCES = get_int_setting("partial_recovery_max_sources")
_REVIEW_MAX_TOKENS = get_int_setting("review_max_tokens")
_PARTIAL_REVIEW_MAX_TOKENS = get_int_setting("partial_review_max_tokens")
_DEFAULT_PERSONA = str(get_setting("default_persona", ""))
_DEFAULT_TONE = str(get_setting("default_tone", ""))
_DEFAULT_FORMAT_TYPE = str(get_setting("default_format_type", ""))


def _label(index: int) -> str:
    prefix = _DEEP_MESSAGES.get("source_label_prefix", "[Nguồn ")
    suffix = _DEEP_MESSAGES.get("source_label_suffix", "]")
    return f"{prefix}{index}{suffix}"


def _alnum_or_space(text: str) -> str:
    chars = []
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append(" ")
    return " " + " ".join("".join(chars).split()) + " "


_SIMULATOR_INPUT_FIELDS = ("id", "label", "unit", "value", "min", "max", "step", "whole", "control")
_SIMULATOR_CALC_FIELDS = ("id", "label", "unit", "op", "args", "show", "decimals")
_LAYOUT_BLOCK_FIELDS = ("type", "children", "columns", "text", "ref", "tone")
_CONTAINER_BLOCK_TYPES = ("column", "row", "grid")
_LAYOUT_TEXT_MAX = 180
_LAYOUT_REF_MAX = 24


def _fold_layout_children(children: list) -> list:
    """Group overflow layout children (>12) into column sub-containers without dropping controls."""
    if len(children) <= 12:
        return children
    folded = list(children[:10])
    idx = 10
    while idx < len(children) and len(folded) < 12:
        folded.append({"type": "column", "children": list(children[idx:idx + 10])})
        idx += 10
    return folded


def _sanitize_layout_block(block, result_tones=None):
    """Strip disallowed non-grid 'columns', out-of-range text, and null optional keys from layout blocks."""
    if not isinstance(block, dict):
        return block
    btype = block.get("type")
    cleaned = {}
    for key, val in block.items():
        if val is None or key not in _LAYOUT_BLOCK_FIELDS:
            continue
        if key == "columns":
            if btype != "grid" or isinstance(val, bool) or not isinstance(val, int):
                continue
            if val < 1:
                val = 1
            elif val > 4:
                val = 4
        elif key == "text" and isinstance(val, str):
            val = " ".join(val.split())
            if not val:
                continue
            if len(val) > _LAYOUT_TEXT_MAX:
                val = val[:_LAYOUT_TEXT_MAX - 3].rstrip() + "..."
        elif key == "ref" and isinstance(val, str):
            val = val.strip()
            if not val:
                continue
            if len(val) > _LAYOUT_REF_MAX:
                val = val[:_LAYOUT_REF_MAX]
        elif key == "tone" and val not in ("plain", "card", "accent"):
            continue
        elif key == "children" and isinstance(val, list):
            sanitized_children = [_sanitize_layout_block(child, result_tones) for child in val]
            if btype not in _CONTAINER_BLOCK_TYPES and not sanitized_children:
                continue
            val = _fold_layout_children(sanitized_children)
        cleaned[key] = val
    if btype == "result" and result_tones:
        ref_id = cleaned.get("ref")
        if isinstance(ref_id, str) and ref_id in result_tones:
            if cleaned.get("tone", "plain") == "plain":
                cleaned["tone"] = result_tones[ref_id]
    return cleaned


def _ensure_layout_completeness(artifact: dict, result_tones: dict) -> None:
    """Append any missing declared controls/renderers into a container layout without dropping existing blocks."""
    layout = artifact.get("layout")
    if not isinstance(layout, dict) or layout.get("type") not in _CONTAINER_BLOCK_TYPES:
        return
    used_refs = set()
    stack = [layout]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        ntype = node.get("type")
        if ntype in _CONTAINER_BLOCK_TYPES:
            children = node.get("children")
            if isinstance(children, list):
                stack.extend(children)
            continue
        ref = node.get("ref")
        if ntype not in ("input", "result") and ref == ntype:
            ref = None
        if ntype:
            used_refs.add((ntype, ref))

    missing_blocks = []
    engine = artifact.get("engine")
    if engine == "simulator" and isinstance(artifact.get("inputs"), list) and isinstance(artifact.get("calculations"), list):
        for inp in artifact["inputs"]:
            if isinstance(inp, dict) and isinstance(inp.get("id"), str) and ("input", inp["id"]) not in used_refs:
                missing_blocks.append({"type": "input", "ref": inp["id"]})
                used_refs.add(("input", inp["id"]))
        for calc in artifact["calculations"]:
            if (isinstance(calc, dict) and isinstance(calc.get("id"), str)
                    and calc.get("show", True) is not False
                    and ("result", calc["id"]) not in used_refs):
                blk = {"type": "result", "ref": calc["id"]}
                if calc["id"] in result_tones:
                    blk["tone"] = result_tones[calc["id"]]
                missing_blocks.append(blk)
                used_refs.add(("result", calc["id"]))
    elif engine == "echarts" and ("chart", None) not in used_refs:
        missing_blocks.append({"type": "chart"})
    elif engine == "mermaid" and ("diagram", None) not in used_refs:
        missing_blocks.append({"type": "diagram"})

    if not missing_blocks:
        return
    existing = layout.get("children")
    if isinstance(existing, list):
        combined = list(existing) + missing_blocks
    else:
        combined = missing_blocks
    layout["children"] = _fold_layout_children(combined)


def _sanitize_calculator_item(item: dict, allowed_fields: tuple[str, ...]) -> dict:
    """Drop unsupported keys and null optionals from simulator input or calculation dicts."""
    cleaned = {}
    for key, val in item.items():
        if key not in allowed_fields or val is None:
            continue
        if key == "decimals":
            if isinstance(val, bool) or not isinstance(val, int):
                continue
            if val < 0:
                val = 0
            elif val > 4:
                val = 4
        cleaned[key] = val
    return cleaned


def _sanitize_presentation_arguments(arguments):
    """Normalize minor LLM layout/calculation quirks before schema and Pydantic validation."""
    if not isinstance(arguments, dict):
        return arguments
    artifact = arguments.get("artifact")
    if not isinstance(artifact, dict):
        return arguments
    result_tones = {}
    raw_calcs = artifact.get("calculations")
    if isinstance(raw_calcs, list):
        for calc in raw_calcs:
            if isinstance(calc, dict):
                calc_id = calc.get("id")
                tone = calc.get("tone")
                if isinstance(calc_id, str) and tone in ("card", "accent"):
                    result_tones[calc_id] = tone
    clean_art = {}
    has_full_body = "inputs" in artifact or "presentation" in artifact or "data" in artifact or "source" in artifact
    for key, val in artifact.items():
        if val is None and key in ("update_of", "layout"):
            continue
        if has_full_body and key in ("description", "operation", "defaults"):
            continue
        if key == "inputs" and isinstance(val, list):
            sanitized_inputs = []
            for item in val:
                if isinstance(item, dict):
                    sanitized_inputs.append(_sanitize_calculator_item(item, _SIMULATOR_INPUT_FIELDS))
                else:
                    sanitized_inputs.append(item)
            clean_art[key] = sanitized_inputs
        elif key == "calculations" and isinstance(val, list):
            sanitized_calcs = []
            for item in val:
                if isinstance(item, dict):
                    sanitized_calcs.append(_sanitize_calculator_item(item, _SIMULATOR_CALC_FIELDS))
                else:
                    sanitized_calcs.append(item)
            clean_art[key] = sanitized_calcs
        elif key == "layout" and isinstance(val, dict):
            clean_art[key] = _sanitize_layout_block(val, result_tones=result_tones)
        else:
            clean_art[key] = val
    _ensure_layout_completeness(clean_art, result_tones)
    return {**arguments, "artifact": clean_art}


def _evidence_view(evidence: list[dict]) -> list[dict]:
    result = []
    for item in evidence:
        chunk = item["chunk"]
        meta = chunk.get("business_metadata") or {}
        record = {"label": _label(item["index"]), "text": item["context"], "source": item["source"],
                  "version": meta.get("document_version", "")}
        temporal = meta.get("temporal")
        if isinstance(temporal, dict) and (
            temporal.get("requested_date") or temporal.get("status") in ("conflict", "effective")
        ):
            record["temporal"] = temporal
        if type(chunk.get("source_offset")) is int:
            record["source_offset"] = chunk["source_offset"]
        if chunk.get("context_for"):
            record["context_neighbor"] = True
        pages = _extract_page_numbers(meta)
        if pages:
            record["pages"] = pages
        section_title = str(chunk.get("section_title") or "").strip()
        if section_title:
            record["section"] = section_title
        breadcrumb = str(chunk.get("breadcrumb") or "").strip()
        if breadcrumb and breadcrumb != section_title:
            record["breadcrumb"] = breadcrumb
        prov_summary = _extract_citation_provenance(meta)
        locs = prov_summary.get("locations")
        if isinstance(locs, list) and locs:
            compact_locs = []
            for loc in locs[:2]:
                entry = {}
                for k in ("part", "xpath", "region", "page_number", "page_index", "sheet_name", "cell_range", "source_row"):
                    if k in loc:
                        entry[k] = loc[k]
                if entry:
                    compact_locs.append(entry)
            if compact_locs:
                record["locations"] = compact_locs
        cit_meta = meta.get("citation")
        if isinstance(cit_meta, dict) and cit_meta.get("sheet_name"):
            record["sheet_name"] = str(cit_meta["sheet_name"])
            if cit_meta.get("cell_ranges"):
                record["cell_ranges"] = list(cit_meta["cell_ranges"])
        result.append(record)
    return result


def _split_user_and_attachment(text: str) -> tuple[str, str]:
    marker = "[Thông tin trích xuất từ ảnh đính kèm của khách hàng:"
    pos = text.find(marker)
    if pos < 0:
        return text, ""
    user_part = text[:pos].strip()
    attach_part = text[pos:].strip()
    return user_part, attach_part


def _support_view(state: dict) -> dict:
    turn = state["react_turn"]
    user_messages = []
    assistant_messages = []
    attachment_segments = []
    for item in state.get("chat_history", [])[-_SUPPORT_HISTORY_LIMIT:]:
        if item["role"] == "user":
            u_part, a_part = _split_user_and_attachment(str(item["content"]))
            if u_part:
                user_messages.append(u_part)
            if a_part:
                attachment_segments.append(a_part)
        elif item["role"] == "assistant":
            a_text = str(item["content"]).strip()
            if a_text and a_text != FALLBACK_ANSWER:
                assistant_messages.append(a_text)
    cur_u, cur_a = _split_user_and_attachment(str(state["user_query"]))
    if cur_u:
        user_messages.append(cur_u)
    elif not cur_a:
        user_messages.append(str(state["user_query"]))
    if cur_a:
        attachment_segments.append(cur_a)
    user_messages += [reply["response"] for reply in state["human_replies"]]
    notes = [item["value"] for item in memory_view(state)["notes"].values()]
    user_messages = notes[:_SUPPORT_NOTES_LIMIT] + user_messages
    user_label = _DEEP_MESSAGES.get("user_support_label", "[Khách]")
    tool_prefix = _DEEP_MESSAGES.get("tool_support_prefix", "[Tool ")
    result = {user_label: "\n".join(text[:_SUPPORT_MESSAGE_MAX_CHARS] for text in user_messages)[-_SUPPORT_USER_TEXT_MAX_CHARS:]}
    if attachment_segments:
        attach_budget = max(_SUPPORT_MESSAGE_MAX_CHARS, 3600)
        result["[Ảnh đính kèm]"] = "\n".join(seg[:attach_budget] for seg in attachment_segments)[-attach_budget:]
    if assistant_messages:
        result["[Lịch sử trả lời trước]"] = "\n---\n".join(
            text[:_SUPPORT_MESSAGE_MAX_CHARS] for text in assistant_messages
        )[-_SUPPORT_USER_TEXT_MAX_CHARS:]

    def add(observation):
        if observation["status"] == "SUCCESS" and not observation.get("retrieval") and observation["kind"] in ("tool", "skill"):
            value = {"result": observation.get("data", {}), "simulation": observation.get("simulation", False)}
            result[f"{tool_prefix}{len(result)}]"] = json.dumps(value, ensure_ascii=False)[:_SUPPORT_TOOL_JSON_MAX_CHARS]

    for observation in turn["observations"]:
        add(observation)
        for item in observation.get("data", {}).get("worker_observations", []):
            add(item)
    return result


def _strip_thousands_separators(text: str) -> str:
    """Remove '.' and ',' thousands separators between digit groups without altering decimals."""
    out = []
    length = len(text)
    for i, ch in enumerate(text):
        if ch in (".", ",") and i > 0 and text[i - 1].isdigit() and i + 3 < length:
            if text[i + 1:i + 4].isdigit() and (i + 4 == length or not text[i + 4].isdigit()):
                continue
        out.append(ch)
    return "".join(out)


def _fold_quote(value: str) -> str:
    canonical = normalize("NFC", value).casefold()
    # Formatting may differ between Markdown, plain text, and a quoted table cell.
    return _alnum_or_space(canonical)


def _matches_table_cells(quote, source_text):
    # Allow a quote made from complete cells in order on one row.
    for line in source_text.splitlines():
        if line.count("|") < 2:
            continue
        remaining = quote
        for cell in line.split("|"):
            text = _fold_quote(_strip_thousands_separators(cell)).strip()
            if not text:
                continue
            if remaining == text:
                return True
            if remaining.startswith(text + " "):
                remaining = remaining[len(text) + 1:]
    return False


_FX_CURRENCY_UNITS = frozenset(("usd", "eur", "gbp", "jpy", "aud", "sgd", "chf", "cad", "cny", "krw", "thb"))
_FX_SPLIT_CHARS = "/|;()[]{}:\"'“”‘’\n\r\t"


def _extract_fx_tokens(text: str) -> set[str]:
    """Extract normalized foreign-currency numeric tokens (~fx_usd_100) without regex."""
    if not text:
        return set()
    work = text.casefold()
    for ch in _FX_SPLIT_CHARS:
        if ch in work:
            work = work.replace(ch, " ")
    words = [w.strip(",.;!?*+<>=") for w in work.split()]
    tokens = set()
    for idx in range(len(words) - 1):
        amt = words[idx]
        unit = words[idx + 1]
        if unit not in _FX_CURRENCY_UNITS or not amt or not amt[0].isdigit():
            continue
        clean_num = amt.replace(",", ".")
        if clean_num.count(".") == 1 and len(clean_num.split(".")[1]) == 3 and clean_num.split(".")[0] != "0":
            clean_num = clean_num.replace(".", "")
        elif clean_num.count(".") > 1:
            clean_num = clean_num.replace(".", "")
        valid = True
        dot_seen = 0
        for ch in clean_num:
            if ch == ".":
                dot_seen += 1
            elif not ch.isdigit():
                valid = False
                break
        if valid and dot_seen <= 1:
            if "." in clean_num:
                norm = clean_num.rstrip("0").rstrip(".")
            else:
                norm = clean_num
            if not norm:
                norm = "0"
            tokens.add(f"~fx_{unit}_{norm}")
    return tokens


_NON_PCT_PREV = frozenset((
    "ngày", "ngay", "tháng", "thang", "năm", "nam",
    "điều", "dieu", "khoản", "khoan", "mục", "muc",
    "trang", "chương", "chuong", "hình", "hinh", "bảng", "bang", "bước", "buoc", "lần", "lan",
))


def _extract_bare_pct_tokens(text: str) -> set[str]:
    """Extract decimal/bounded percentage values from table/percentage-labeled sources lacking '%' suffix."""
    if not text:
        return set()
    low = normalize("NFC", text).casefold()
    if "%" not in low and "phần trăm" not in low and "phan tram" not in low and "tỷ lệ" not in low and "ty le" not in low:
        return set()
    work = low
    for ch in _FX_SPLIT_CHARS:
        if ch in work:
            work = work.replace(ch, " ")
    words = work.split()
    tokens = set()
    for idx, raw_tok in enumerate(words):
        if idx > 0:
            prev_word = words[idx - 1].strip(",.;:!?\"'()[]*+-")
            if prev_word in _NON_PCT_PREV:
                continue
        tok = raw_tok.strip(",.;:!?\"'()[]*+-")
        if not tok or not tok[0].isdigit():
            continue
        if tok.count(",") + tok.count(".") > 1:
            continue
        norm = tok.replace(",", ".")
        if "." in norm:
            left, right = norm.split(".", 1)
            if not left.isdigit() or not right.isdigit() or len(right) > 2:
                continue
            val = float(norm)
            if val > 100.0:
                continue
            dec = right.rstrip("0")
            int_part = str(int(left))
            if dec:
                tokens.add(f"~pct_{int_part}_{dec}")
            else:
                tokens.add(f"~pct_{int_part}")
        elif tok.isdigit():
            val = int(tok)
            if 0 <= val <= 100:
                tokens.add(f"~pct_{val}")
    return tokens


def _derived_numeric_tokens(source_numbers: set[str]) -> set[str]:
    """Derive arithmetic and percentage deltas supported by pairs of grounded source figures."""
    derived = set()
    vnd_vals = set()
    pct_vals = set()
    for tok in source_numbers:
        if tok.startswith("~vnd_"):
            raw = tok[5:]
            if raw.isdigit():
                vnd_vals.add(int(raw))
        elif tok.startswith("~pct_"):
            raw = tok[5:].replace("_", ".")
            try:
                pct_vals.add(float(raw))
            except ValueError:
                pass
    if 2 <= len(vnd_vals) <= 150:
        ordered_vnd = sorted(vnd_vals)
        for i, val_a in enumerate(ordered_vnd):
            for val_b in ordered_vnd[i + 1:]:
                diff_v = abs(val_b - val_a)
                if diff_v >= 1_000:
                    derived.add(f"~vnd_{diff_v}")
                if val_a > 0:
                    rel_pct = diff_v * 100.0 / val_a
                    for digits in (0, 1, 2):
                        formatted = f"{rel_pct:.{digits}f}"
                        if "." in formatted:
                            formatted = formatted.rstrip("0").rstrip(".")
                        if not formatted:
                            formatted = "0"
                        derived.add("~pct_" + formatted.replace(".", "_"))
    if 2 <= len(pct_vals) <= 150:
        ordered_pct = sorted(pct_vals)
        for i, pct_a in enumerate(ordered_pct):
            for pct_b in ordered_pct[i + 1:]:
                diff_p = abs(pct_b - pct_a)
                for digits in (1, 2):
                    formatted = f"{diff_p:.{digits}f}".rstrip("0").rstrip(".")
                    if not formatted:
                        formatted = "0"
                    derived.add("~pct_" + formatted.replace(".", "_"))
    if vnd_vals and pct_vals and len(vnd_vals) * len(pct_vals) <= 400:
        for val_v in vnd_vals:
            for val_p in pct_vals:
                if val_v > 0 and val_p > 0:
                    prod = int(round(val_v * val_p / 100.0))
                    if prod >= 1_000:
                        derived.add(f"~vnd_{prod}")
                        derived.add(f"~vnd_{val_v + prod}")
                        if val_v > prod:
                            derived.add(f"~vnd_{val_v - prod}")
    return derived


def validate_review(verdict: dict, draft: str, citations: list, evidence: list, support: dict) -> None:
    sources = {item["label"]: item["text"] for item in evidence}
    sources.update(support)
    if any(item["status"] == "missing" for item in verdict["coverage"]):
        raise ValueError("Missing requested item")
    claimed = set()
    draft_text = _fold_quote(draft)
    plain = draft.casefold()
    for citation in citations:
        plain = plain.replace(citation["label"].casefold(), "")
    if not verdict["claims"] and any(char.isdigit() for char in plain) and any(
        marker in plain for marker in _REVIEW_NUMERIC_MARKERS
    ):
        raise ValueError("Unsupported financial figure or contact")
    source_labels = {_fold_quote(label): label for label in sources}
    for item in verdict["claims"]:
        source = source_labels.get(_fold_quote(item["source"]))
        excerpt = item.get("claim", "")
        if "draft_line" in item:
            draft_lines = draft.splitlines()
            pointer = item["draft_line"]
            if type(pointer) is not int or not 1 <= pointer <= len(draft_lines):
                raise ValueError("Invalid draft evidence line")
            candidate_line = draft_lines[pointer - 1]
            if _fold_quote(candidate_line).strip() and _fold_quote(candidate_line) in draft_text:
                excerpt = candidate_line
            elif item.get("claim") and _fold_quote(item["claim"]).strip() in draft_text:
                excerpt = item["claim"]
            else:
                excerpt = candidate_line
        claim = _fold_quote(excerpt)
        if source is None:
            raise ValueError("Invalid claim source label")
        if not claim.strip() or claim not in draft_text:
            raise ValueError("Invalid claim excerpt from draft")
        full_source_text = sources[source]
        source_text = full_source_text
        if "line" in item:
            lines = source_text.splitlines()
            line = item["line"]
            if type(line) is not int or not 1 <= line <= len(lines) or not _fold_quote(lines[line - 1]).strip():
                raise ValueError("Invalid claim evidence line")
            source_text = lines[line - 1]
        quote = item.get("quote")
        if quote is not None:
            clean_quote = str(quote).strip()
            q_parts = clean_quote.split(" ", 1)
            if len(q_parts) == 2 and len(q_parts[0]) >= 2 and q_parts[0][0] in ("L", "l", "D", "d") and q_parts[0][1:].isdigit():
                clean_quote = q_parts[1].strip()
            folded_q = _fold_quote(clean_quote).strip()
            if not folded_q:
                raise ValueError("Invalid claim evidence excerpt")
            full_folded = _fold_quote(full_source_text)
            src_folded = _fold_quote(source_text)
            compact_q = _fold_quote(_strip_thousands_separators(clean_quote)).strip()
            compact_full = _fold_quote(_strip_thousands_separators(full_source_text))
            if (not any(f" {folded_q} " in value for value in (src_folded, full_folded))
                    and f" {compact_q} " not in compact_full and not _matches_table_cells(compact_q, source_text)):
                raise ValueError("Invalid claim evidence excerpt")
        claimed.add(source)
    if verdict.get("verdict", "approve") == "approve":
        missing = [item["label"] for item in citations if item["label"] not in claimed]
        if citations and not any(item["label"] in claimed for item in citations):
            raise ValueError("Citation missing claim grounding: " + ", ".join(missing))
        labels = {item["label"] for item in citations}
        ungrounded_line_cites = []
        for line in _normalize_citation_markers(draft).replace(". ", ".\n").splitlines():
            line_labels = [label for label in labels if label.casefold() in line.casefold()]
            if not line_labels:
                continue
            if not any(lbl in claimed for lbl in line_labels):
                for lbl in line_labels:
                    if lbl not in claimed and lbl not in ungrounded_line_cites:
                        ungrounded_line_cites.append(lbl)
            numbers = {token for token in extract_canonical_numeric_tokens(line)
                       if token.startswith(("~pct_", "~vnd_"))}
            fx_numbers = _extract_fx_tokens(line)
            source_numbers = set()
            source_fx = set()
            user_label = _DEEP_MESSAGES.get("user_support_label", "[Khách]")
            tool_prefix = _DEEP_MESSAGES.get("tool_support_prefix", "[Tool ")
            grounded_support = {
                key for key in support
                if key == user_label or key.startswith(tool_prefix)
            }
            for label in set(line_labels) | (claimed & support.keys()) | grounded_support:
                src_val = sources.get(label, "")
                source_numbers.update(extract_canonical_numeric_tokens(src_val, max_tokens=512))
                source_numbers.update(_extract_bare_pct_tokens(src_val))
                source_fx.update(_extract_fx_tokens(src_val))
            derived_numbers = _derived_numeric_tokens(source_numbers)
            bad_tokens = (numbers - (source_numbers | derived_numbers)) | (fx_numbers - source_fx)
            if bad_tokens:
                detail = ", ".join(sorted(bad_tokens)[:4])
                raise ValueError(f"Unsupported financial figure in cited answer line ({detail})")
        if ungrounded_line_cites:
            raise ValueError("Citation missing claim grounding: " + ", ".join(ungrounded_line_cites))


def _compact_observations(observations: list[dict]) -> list[dict]:
    bounded = []
    for observation in observations:
        if observation["kind"] == "inspect_skill":
            bounded.append({key: value for key, value in observation.items() if key != "data"})
            continue
        data = observation.get("data")
        if not isinstance(data, dict):
            bounded.append(observation)
            continue
        if observation.get("retrieval"):
            compact_docs = []
            for item in data.get("documents", []):
                if isinstance(item, dict) and "label" in item:
                    compact_docs.append(item["label"])
                else:
                    compact_docs.append(item)
            compact_related = []
            for item in data.get("related_documents", []):
                if isinstance(item, dict) and "label" in item:
                    compact_related.append(item["label"])
                else:
                    compact_related.append(item)
            data = {**data, "documents": compact_docs, "related_documents": compact_related}
        encoded = json.dumps(data, ensure_ascii=False)
        if len(encoded) > _COMPACT_OBSERVATION_MAX_BYTES:
            if observation.get("retrieval"):
                preserved = {}
                for keep_key in _PRESERVED_RETRIEVAL_KEYS:
                    if keep_key in data:
                        preserved[keep_key] = data[keep_key]
                preserved["truncated"] = True
                if len(json.dumps(preserved, ensure_ascii=False)) <= _COMPACT_OBSERVATION_MAX_BYTES:
                    data = preserved
                else:
                    if isinstance(preserved.get("retrieval_quality"), dict):
                        rq = dict(preserved["retrieval_quality"])
                        if isinstance(rq.get("source_inventory"), list) and len(rq["source_inventory"]) > _MAX_SOURCE_INVENTORY_ITEMS:
                            rq["source_inventory"] = rq["source_inventory"][:_MAX_SOURCE_INVENTORY_ITEMS]
                        preserved["retrieval_quality"] = rq
                    if len(json.dumps(preserved, ensure_ascii=False)) <= _COMPACT_OBSERVATION_MAX_BYTES:
                        data = preserved
                    else:
                        data = {"preview": encoded[:_COMPACT_OBSERVATION_MAX_BYTES], "truncated": True}
            else:
                data = {"preview": encoded[:_COMPACT_OBSERVATION_MAX_BYTES], "truncated": True}
        bounded.append({**observation, "data": data})
    return bounded


_DIRECT_SOLICITATION_CUES = (
    " nhap ", " cung cap ", " cho biet ", " chia se ",
    " dien ma ", " dien so ", " dien thong tin ", " vui long dien ",
    " doc ma ", " doc so ", " doc thong tin ",
    " xac minh so ", " doi chieu so ", " xac nhan ma ",
    " gui ma otp ", " gui ma pin ", " gui otp ", " gui pin ",
    " gui cvv ", " gui cvc ", " gui mat khau ", " gui so the ",
    " gui so tai khoan ", " gui so cccd ", " gui cccd ",
)
_POSSESSIVE_SOLICITATION_CUES = (
    " bao nhieu ", " la gi ", " cua ban ", " cua quy khach ", " de tiep tuc ",
)
_GUI_SAFE_RECIPIENT_CUES = (
    " bo phan ", " phong ", " yeu cau ", " phieu ", " khieu nai ", " ho tro ",
)
_STRICT_SECRET_FIELDS = {"otp", "pin", "cvv", "cvc", "mat khau", "mật khẩu", "password"}
_TOPIC_CONTEXT_CUES = (
    " tinh huong ", " van de ", " dich vu ", " loi ", " chu de ", " nhom ",
    " quy trinh ", " chinh sach ", " huong dan ", " mo khoa ", " quen ",
    " khong nhan duoc ", " tao phieu ", " lap phieu ", " phieu ho tro ",
    " bo phan ", " chuyen vien ", " khieu nai ", " tra soat ", " bi khoa ",
    " su co ", " ho tro ",
)


def _requests_restricted_field(text: str) -> bool:
    normalized = _alnum_or_space(text.casefold().replace(".", ""))
    folded = " " + search_key(text) + " "
    has_field = any(f" {field} " in normalized or f" {search_key(field)} " in folded for field in RESTRICTED_CARD_FIELDS)
    if not has_field:
        return False
    if any(cue in folded for cue in _DIRECT_SOLICITATION_CUES):
        return True
    words = folded.split()
    for idx, tok in enumerate(words):
        if tok == "gui":
            win_text = " " + " ".join(words[idx + 1:idx + 6]) + " "
            if any(neg in win_text for neg in _GUI_SAFE_RECIPIENT_CUES):
                continue
            if any(f" {search_key(field)} " in win_text for field in RESTRICTED_CARD_FIELDS):
                return True
    if any(cue in folded for cue in (" bao nhieu ", " la gi ")):
        return True
    has_topic_context = any(cue in folded for cue in _TOPIC_CONTEXT_CUES)
    if any(cue in folded for cue in _POSSESSIVE_SOLICITATION_CUES) and not has_topic_context:
        return True
    has_secret = any(f" {field} " in normalized or f" {search_key(field)} " in folded for field in _STRICT_SECRET_FIELDS)
    if has_secret and not has_topic_context:
        return True
    return False


def _requests_brief_summary(state: dict) -> bool:
    replies = (reply["response"] for reply in state["human_replies"])
    request = " " + search_key(" ".join((state["user_query"], *replies))) + " "
    full_detail = (" khong tom tat ", " khong can tom tat ", " dung tom tat ",
                   " day du ", " toan bo ", " tat ca ", " full ", " do not summarize ")
    if any(phrase in request for phrase in full_detail):
        return False
    if not any(phrase in request for phrase in (" tom tat ", " tom gon ", " summarize ", " summarise ")):
        return False
    size_units = {"tu", "chu", "dong", "y", "buoc", "words", "lines", "points", "steps"}
    words = request.split()
    return not any(count.isdecimal() and unit in size_units for count, unit in zip(words, words[1:]))


def _brief_summary_counts(draft: str) -> tuple[int, int]:
    cleaned = _normalize_citation_markers(draft).replace("][", "] [")
    # Count prose words without penalizing [Nguồn N] tags or standalone list symbols.
    words = 0
    for tok in cleaned.split():
        bare = tok.strip(".,;:!?\"'()").lower()
        if not bare or bare in ("-", "*", "+"):
            continue
        if bare == "[nguồn" or (bare.endswith("]") and bare[:-1].isdigit()):
            continue
        words += 1
    line_starts = (line.lstrip().split(" ", 1)[0] for line in draft.splitlines())
    list_items = sum(
        start in ("-", "*", "+") or (start[-1:] in (".", ")") and start[:-1].isdigit())
        for start in line_starts
    )
    return words, list_items


def _fits_brief_summary(draft: str) -> bool:
    words, list_items = _brief_summary_counts(draft)
    return words <= SUMMARY_MAX_WORDS and list_items <= SUMMARY_MAX_POINTS


_SUPPORT_LINE_MARKERS = (
    "hotline",
    "tong dai",
    "1900",
    "cskh",
    "cham soc khach hang",
    "kenh ho tro",
    "lien he ho tro",
    "ho tro 24 7",
    "ho tro khach hang",
)
_ACTION_STEP_MARKERS = (
    "buoc ",
    "chon ",
    "nhan ",
    "nhap ",
    "bam ",
    "quet ",
    "chup ",
    "xac thuc",
    "dang nhap",
    "mo ung dung",
    "tai ung dung",
    "man hinh",
    "thao tac",
    "truy cap",
    "gat nut",
)
_APP_VISUAL_REGION_TYPES = ("app_step", "slide_visual")


def _is_hex64(value: str) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    for ch in value:
        if ch not in "0123456789abcdef":
            return False
    return True


def _parse_frame_index_from_caption(text: str) -> int:
    if not text:
        return 999999
    lower_text = str(text).lower()
    pos = lower_text.find("khung ")
    if pos < 0:
        return 999999
    tail = lower_text[pos + 6:].strip()
    slash_idx = tail.find("/")
    if slash_idx <= 0:
        return 999999
    num_str = tail[:slash_idx].strip()
    if num_str.isdigit():
        val = int(num_str)
        if val > 0:
            return val - 1
    return 999999


def _image_order_key(img: dict) -> tuple:
    page_num = 999999
    raw_page = img.get("page_number")
    raw_page_idx = img.get("page_index")
    if type(raw_page) is int and raw_page > 0:
        page_num = raw_page
    elif type(raw_page_idx) is int and raw_page_idx >= 0:
        page_num = raw_page_idx + 1

    region_idx = 999999
    raw_region_idx = img.get("region_index")
    if type(raw_region_idx) is int and raw_region_idx >= 0:
        region_idx = raw_region_idx
    else:
        cap_text = str(img.get("display_caption") or img.get("caption") or "")
        region_idx = _parse_frame_index_from_caption(cap_text)

    bbox_y = 999999.0
    bbox_x = 999999.0
    bbox = img.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
        try:
            bbox_x = float(bbox[0])
            bbox_y = float(bbox[1])
        except (TypeError, ValueError):
            bbox_x = 999999.0
            bbox_y = 999999.0

    return (page_num, region_idx, bbox_y, bbox_x)


def _normalize_image_entry(img: dict, citation: dict, doc_name: str):
    if not isinstance(img, dict):
        return None
    asset_id = str(img.get("asset_id") or img.get("sha256") or "").strip().lower()
    if not _is_hex64(asset_id):
        return None

    page_number = None
    raw_page = img.get("page_number")
    raw_page_idx = img.get("page_index")
    if type(raw_page) is int and raw_page > 0:
        page_number = raw_page
    elif type(raw_page_idx) is int and raw_page_idx >= 0:
        page_number = raw_page_idx + 1

    region_index = None
    raw_region_idx = img.get("region_index")
    if type(raw_region_idx) is int and raw_region_idx >= 0:
        region_index = raw_region_idx

    region_count = None
    raw_region_cnt = img.get("region_count")
    if type(raw_region_cnt) is int and raw_region_cnt > 0:
        region_count = raw_region_cnt

    caption = str(
        img.get("display_caption")
        or img.get("caption")
        or img.get("title")
        or citation.get("section_title")
        or doc_name
    ).strip()

    entry = {
        "asset_id": asset_id,
        "url": "/api/assets/" + asset_id,
        "citation_label": str(citation.get("label") or ""),
        "source": str(citation.get("source") or ""),
        "document_name": doc_name,
        "section_title": str(citation.get("section_title") or ""),
        "page_number": page_number,
        "region_index": region_index,
        "region_count": region_count,
        "region_type": str(img.get("region_type") or "figure"),
        "caption": caption,
        "width": int(img.get("width") or 0),
        "height": int(img.get("height") or 0),
    }
    step = img.get("step_number")
    if type(step) is int and step > 0:
        entry["step_number"] = step
    source = entry["source"]
    bbox = img.get("bbox")
    if (page_number and source.lower().endswith(".pdf")
            and isinstance(bbox, (list, tuple)) and len(bbox) == 4
            and all(type(coordinate) in (int, float) for coordinate in bbox)
            and 0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
        params = {
            "source": source,
            "page": page_number,
            "bbox": ",".join(map(str, bbox)),
        }
        if is_document_version(citation.get("document_version")):
            params["version"] = citation["document_version"]
        entry["detail_url"] = "/api/documents/pdf/region?" + urlencode(params)
    return entry


def _extract_chunk_images(citation: dict, chunk: dict) -> list:
    if not isinstance(chunk, dict):
        return []
    raw_images = chunk.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        return []
    meta = chunk.get("business_metadata")
    if not isinstance(meta, dict):
        meta = {}
    src = str(citation.get("source") or chunk.get("source") or "")
    doc_name = _extract_document_name(src, meta)

    by_asset = {}
    for raw_img in raw_images:
        entry = _normalize_image_entry(raw_img, citation, doc_name)
        if entry is None:
            continue
        asset_id = entry["asset_id"]
        direct_rank = 1
        if raw_img.get("match_strength") == "direct":
            direct_rank = 0
        order_key = _image_order_key(raw_img)
        existing = by_asset.get(asset_id)
        if existing is None or direct_rank < existing[0]:
            by_asset[asset_id] = (direct_rank, order_key, entry)

    items = list(by_asset.values())
    items.sort(key=lambda pair: pair[1])
    return [pair[2] for pair in items]


def _line_citation_position(line: str, label: str, cit_index) -> int | None:
    if type(cit_index) is not int or cit_index <= 0:
        if label and (offset := line.find(label)) >= 0:
            return offset
        return None
    target = str(cit_index)
    cursor = 0
    while (open_idx := line.find("[", cursor)) >= 0:
        close_idx = line.find("]", open_idx + 1)
        if close_idx < 0:
            break
        inner = line[open_idx + 1:close_idx]
        if label and line[open_idx:close_idx + 1] == label:
            return open_idx
        folded = inner.casefold()
        if folded.startswith(("nguồn", "nguon", "source")):
            part_offset = open_idx + 1
            for part in inner.replace(";", ",").split(","):
                token = part.strip().casefold()
                for prefix in ("nguồn", "nguon", "source"):
                    token = token.removeprefix(prefix)
                token = token.lstrip(" :")
                if token and token.split()[0] == target:
                    return part_offset
                part_offset += len(part) + 1
        cursor = close_idx + 1
    return None


def _citation_answer_context(citation: dict, answer_text: str) -> tuple:
    if not answer_text:
        return ((0, 0), False, False)
    label = str(citation.get("label") or "")
    cit_index = citation.get("index")
    first_pos = None
    support_only = True
    shows_action = False
    for pos, line in enumerate(answer_text.splitlines()):
        offset = _line_citation_position(line, label, cit_index)
        if offset is None:
            continue
        if first_pos is None:
            first_pos = (pos, offset)
        key = " " + search_key(line) + " "
        action = any(marker in key for marker in _ACTION_STEP_MARKERS)
        shows_action = shows_action or action
        support_only = support_only and not action and any(marker in key for marker in _SUPPORT_LINE_MARKERS)
    return (first_pos, support_only, shows_action)


def _image_source_key(source) -> str:
    return str(source or "").replace("\\", "/").strip().casefold()


def _image_workflow(chunk: dict) -> tuple:
    parts = str(chunk.get("breadcrumb") or "").split(" > ")
    for index, part in enumerate(parts):
        if leading_step_number(part) is not None:
            return tuple(parts[:index])
    return ()


def _resolve_cited_images(
    citations: list,
    retrieved_chunks: list,
    status: str,
    max_images: int | None = None,
    answer_text: str = "",
) -> list:
    """Show answer-linked illustrations, with each workflow in document reading order."""
    if status == "BLOCKED" or not citations or not retrieved_chunks:
        return []
    by_chunk_id = {}
    by_source_section = {}
    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id:
            by_chunk_id.setdefault(chunk_id, chunk)
        source_key = _image_source_key(chunk.get("source"))
        section = str(chunk.get("section_title") or "")
        if source_key:
            by_source_section.setdefault((source_key, section), []).append(chunk)

    best_by_asset = {}
    for cit_pos, citation in enumerate(citations):
        if not isinstance(citation, dict):
            continue
        first_pos, support_only, shows_action = _citation_answer_context(citation, answer_text)
        if first_pos is None:
            continue
        source_key = _image_source_key(citation.get("source"))
        section = str(citation.get("section_title") or "")
        chunk = by_chunk_id.get(str(citation.get("chunk_id") or ""))
        if chunk is None and source_key:
            matches = by_source_section.get((source_key, section), [])
            if len(matches) == 1:
                chunk = matches[0]
        if chunk is None:
            continue
        chunk_source = _image_source_key(chunk.get("source"))
        if source_key and chunk_source and source_key != chunk_source:
            continue
        raw_images = chunk.get("images")
        if not isinstance(raw_images, list) or not raw_images:
            continue

        meta = chunk.get("business_metadata")
        if not isinstance(meta, dict):
            meta = {}
        source = str(citation.get("source") or chunk.get("source") or "")
        doc_name = _extract_document_name(source, meta)
        source_key = _image_source_key(source or doc_name)
        section = str(chunk.get("section_title") or section)
        step = leading_step_number(section)
        workflow = _image_workflow(chunk)
        source_pages = set(_extract_page_numbers(meta))
        items = []
        app_frames = {}
        for raw_img in raw_images:
            entry = _normalize_image_entry(raw_img, {**citation, "source": source}, doc_name)
            if entry is None:
                continue
            is_app = entry["region_type"] in _APP_VISUAL_REGION_TYPES
            page = entry["page_number"]
            if is_app and page:
                app_frames.setdefault(page, set()).add(entry["asset_id"])
            items.append((raw_img, entry, is_app))

        for raw_img, entry, is_app in items:
            if raw_img.get("match_strength") == "page_context":
                continue
            page = entry["page_number"]
            if page and source_pages and page not in source_pages:
                continue
            if is_app:
                if support_only or (answer_text and not shows_action and step is None):
                    continue
                frame_step = entry.get("step_number")
                if frame_step is not None and step is not None and frame_step != step:
                    continue
                count = len(app_frames.get(page, ()))
                region_count = entry["region_count"]
                # Legacy whole-page and cross-page attachments were also marked direct.
                if frame_step is None:
                    if len(app_frames) > 1 or (count > 1 and (region_count is None or count >= region_count)):
                        continue
                if step is not None and frame_step is None and count <= 1:
                    entry["step_number"] = step
            direct = raw_img.get("match_strength") == "direct"
            if is_app:
                app_workflow = workflow
            else:
                app_workflow = ()
            group = (source_key, app_workflow)
            preference = (not direct, -bool(entry.get("step_number")), len(items), first_pos, cit_pos)
            asset_id = entry["asset_id"]
            previous = best_by_asset.get(asset_id)
            if previous is None or preference < previous[0]:
                best_by_asset[asset_id] = (preference, group, _image_order_key(raw_img), entry, is_app)

    surviving = list(best_by_asset.values())
    group_positions = {}
    for preference, group, _, _, _ in surviving:
        position = preference[-2:]
        group_positions[group] = min(position, group_positions.get(group, position))

    surviving.sort(key=lambda item: (group_positions[item[1]], item[1], item[2], item[0][-2:], item[3]["asset_id"]))
    if max_images is None:
        max_images = 8
        if any(item[4] for item in surviving):
            max_images = 16
    return [item[3] for item in surviving[:max(0, max_images)]]


def _extract_page_numbers(business_metadata: dict) -> list:
    """Collect sorted unique 1-based page numbers from citation or provenance metadata."""
    if not isinstance(business_metadata, dict):
        return []
    pages = []
    seen = set()
    citation_meta = business_metadata.get("citation")
    if isinstance(citation_meta, dict):
        raw_pages = citation_meta.get("page_numbers")
        if isinstance(raw_pages, list):
            for p_num in raw_pages:
                if type(p_num) is int and p_num > 0 and p_num not in seen:
                    seen.add(p_num)
                    pages.append(p_num)
    if not pages:
        prov = business_metadata.get("provenance")
        if isinstance(prov, dict):
            locations = prov.get("locations")
            if isinstance(locations, list):
                for loc in locations:
                    if not isinstance(loc, dict):
                        continue
                    page_num = loc.get("page_number")
                    page_idx = loc.get("page_index")
                    resolved_num = None
                    if type(page_num) is int and page_num > 0:
                        resolved_num = page_num
                    elif type(page_idx) is int and page_idx >= 0:
                        resolved_num = page_idx + 1
                    if resolved_num is not None and resolved_num not in seen:
                        seen.add(resolved_num)
                        pages.append(resolved_num)
    pages.sort()
    return pages


def _extract_document_name(source: str, business_metadata: dict) -> str:
    """Resolve human-readable document name without extension when metadata omits it."""
    if isinstance(business_metadata, dict):
        doc_name = str(business_metadata.get("document_name") or "").strip()
        if doc_name:
            return doc_name
    clean = str(source or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if "." in clean:
        return clean.rsplit(".", 1)[0]
    return clean


def _build_pdf_url(source: str, page_numbers: list, version="") -> str:
    """Build browser-ready inline PDF URL with #page=N anchor for PDF sources."""
    clean = str(source or "").replace("\\", "/").strip()
    while clean.startswith("/"):
        clean = clean[1:]
    if not clean or not clean.lower().endswith(".pdf"):
        return ""
    url = "/api/documents/pdf?source=" + quote(clean, safe="")
    if is_document_version(version):
        url += "&version=" + version
    if isinstance(page_numbers, list) and page_numbers:
        first_page = page_numbers[0]
        if type(first_page) is int and first_page > 0:
            url = url + "#page=" + str(first_page)
    return url


def _format_citation_excerpt(chunk: dict, max_chars: int = 4000) -> str:
    """Return literal chunk text or markdown table fallback, bounded to max_chars."""
    if not isinstance(chunk, dict):
        return ""
    text = str(chunk.get("content") or "").strip()
    if not text:
        table_data = chunk.get("table_data")
        rows = []
        if isinstance(table_data, dict):
            grid = table_data.get("grid")
            if isinstance(grid, list):
                rows = grid
            else:
                headers = table_data.get("header_grid")
                body = table_data.get("rows")
                if isinstance(headers, list):
                    rows.extend(headers)
                if isinstance(body, list):
                    rows.extend(body)
        elif isinstance(table_data, list):
            rows = table_data
        md_lines = []
        for idx, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or not row:
                continue
            cells = []
            for cell in row:
                cells.append(str(cell).replace("|", "\\|").replace("\n", " ").strip())
            md_lines.append("| " + " | ".join(cells) + " |")
            if idx == 0:
                sep = ["---" for _ in cells]
                md_lines.append("| " + " | ".join(sep) + " |")
        text = "\n".join(md_lines).strip()

    if len(text) > max_chars:
        cutoff = text.rfind(" ", max(0, max_chars - 120), max_chars)
        if cutoff > max_chars // 2:
            text = text[:cutoff].rstrip() + "..."
        else:
            text = text[:max_chars].rstrip() + "..."
    return text


def _extract_file_type(source: str, business_metadata: dict) -> str:
    """Resolve normalized file extension type (pdf, docx, xlsx) without leading dot."""
    clean_src = str(source or "").strip().lower()
    for ext in ("pdf", "docx", "xlsx"):
        if clean_src.endswith("." + ext):
            return ext
    if isinstance(business_metadata, dict):
        meta_ft = str(business_metadata.get("file_type") or "").strip().lower()
        if meta_ft.startswith("."):
            meta_ft = meta_ft[1:]
        if meta_ft in ("pdf", "docx", "xlsx"):
            return meta_ft
    return ""


def _extract_citation_provenance(business_metadata: dict) -> dict:
    """Extract compact, frontend-safe location anchors from chunk business_metadata."""
    if not isinstance(business_metadata, dict):
        return {}
    prov = business_metadata.get("provenance")
    if not isinstance(prov, dict):
        prov = {}

    allowed_loc_keys = (
        "part",
        "xpath",
        "region",
        "parent_cell",
        "sheet_name",
        "cell_range",
        "source_row",
        "page_number",
        "page_index",
        "page_asset_sha256",
    )
    compact_locs = []
    raw_locs = prov.get("locations")
    if isinstance(raw_locs, list):
        for loc in raw_locs[:24]:
            if not isinstance(loc, dict):
                continue
            cleaned = {}
            for k in allowed_loc_keys:
                if k in loc and loc[k] is not None and loc[k] != "":
                    cleaned[k] = loc[k]
            if cleaned:
                compact_locs.append(cleaned)

    compact_spans = []
    raw_spans = prov.get("source_spans")
    if isinstance(raw_spans, list):
        for sp in raw_spans[:12]:
            if not isinstance(sp, dict):
                continue
            sp_entry = {}
            for k in ("start", "end", "chunk_start", "chunk_end"):
                if type(sp.get(k)) is int:
                    sp_entry[k] = sp[k]
            if sp_entry:
                compact_spans.append(sp_entry)

    out = {}
    if prov.get("method"):
        out["method"] = str(prov["method"])
    if compact_locs:
        out["locations"] = compact_locs
    if compact_spans:
        out["source_spans"] = compact_spans
    if business_metadata.get("excel_table_id"):
        out["excel_table_id"] = str(business_metadata["excel_table_id"])
    return out


def _enrich_citations(citations: list, retrieved_chunks: list, cited_images: list, status: str) -> list:
    """Attach source document metadata, page numbers, original excerpt, and images to citations."""
    if status == "BLOCKED" or not citations:
        return []
    by_chunk_id = {}
    by_source_section = {}
    if isinstance(retrieved_chunks, list):
        for chunk in retrieved_chunks:
            if not isinstance(chunk, dict):
                continue
            cid = str(chunk.get("chunk_id") or "")
            if cid and cid not in by_chunk_id:
                by_chunk_id[cid] = chunk
            src = str(chunk.get("source") or "")
            sec = str(chunk.get("section_title") or "")
            if src and (src, sec) not in by_source_section:
                by_source_section[(src, sec)] = chunk

    images_by_label = {}
    if isinstance(cited_images, list):
        for img in cited_images:
            if not isinstance(img, dict):
                continue
            lbl = str(img.get("citation_label") or "")
            if lbl:
                images_by_label.setdefault(lbl, []).append(img)

    enriched = []
    for item in citations:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("chunk_id") or "")
        src = str(item.get("source") or "")
        sec = str(item.get("section_title") or "")
        chunk = by_chunk_id.get(cid)
        if chunk is None and src:
            chunk = by_source_section.get((src, sec))
        if chunk is None:
            chunk = {}

        meta = chunk.get("business_metadata")
        if not isinstance(meta, dict):
            meta = {}
        chunk_type = str(chunk.get("chunk_type") or "text")
        label = str(item.get("label") or "")
        enriched_item = dict(item)
        pages = _extract_page_numbers(meta)
        enriched_item["document_name"] = _extract_document_name(src, meta)
        enriched_item["file_type"] = _extract_file_type(src, meta)
        if meta.get("source_url"):
            enriched_item["source_url"] = meta["source_url"]
        quality = meta.get("quality")
        if not isinstance(quality, dict):
            quality = {}
        if quality.get("ocr_uncertain_pages"):
            enriched_item["ocr_uncertain_pages"] = list(quality["ocr_uncertain_pages"])
        if quality.get("ocr_uncertain_images"):
            enriched_item["ocr_uncertain_images"] = True
        enriched_item["page_numbers"] = pages
        excel_citation = meta.get("citation")
        if isinstance(excel_citation, dict) and excel_citation.get("sheet_name"):
            enriched_item["sheet_name"] = excel_citation["sheet_name"]
            enriched_item["cell_ranges"] = excel_citation.get("cell_ranges", [])
        enriched_item["pdf_url"] = _build_pdf_url(src, pages, enriched_item.get("document_version", ""))
        enriched_item["chunk_type"] = chunk_type
        enriched_item["excerpt"] = _format_citation_excerpt(chunk)
        prov_summary = _extract_citation_provenance(meta)
        if prov_summary:
            enriched_item["provenance"] = prov_summary
        chunk_imgs = _extract_chunk_images(item, chunk)
        if chunk_imgs:
            enriched_item["images"] = chunk_imgs
        else:
            enriched_item["images"] = list(images_by_label.get(label, []))
        enriched.append(enriched_item)
    return enriched


class BankingDeepAgent(BankingReActNodes):
    def __init__(self, pipeline, config):
        super().__init__(pipeline, config)
        review_client = None
        if self.client.routed:
            from agent_banking.core.chat import FallbackChatClient
            review_client = FallbackChatClient("agent_review", routed=True)
        self.model = StructuredModel(self.client, config.AGENT_LLM_MODEL, review_client=review_client)

    def input_guardrail(self, state: dict) -> dict:
        guarded = super().input_guardrail(state)
        guarded.update(plan=[], plan_revision=0, delegations=[], reviews=[], compactions_this_turn=0,
                       active_skills={})
        guarded.update(interaction=None, human_replies=[], accepted_interaction_id=None,
                       approved_scope=None, confirmation_required=False, follow_up_questions=[])
        guarded["session_memory"] = memory_view(guarded)
        guarded["memory_at_start"] = guarded["session_memory"]
        guarded["history_at_start"] = guarded.get("chat_history", [])
        guarded["prior_unresolved_interactions"] = int(guarded.get("prior_unresolved_interactions") or 0)
        return guarded

    def prepare_context(self, state: dict) -> dict:
        parsed = parse_banking_query(state["user_query"][:_QUERY_HINTS_MAX_QUERY_CHARS])
        hints = {
            "normalized": parsed.normalized_query[:_QUERY_HINTS_MAX_TEXT_CHARS],
            "accent_folded": search_key(parsed.raw_query)[:_QUERY_HINTS_MAX_TEXT_CHARS],
            "keywords": parsed.keywords[:_QUERY_HINTS_MAX_KEYWORDS],
            "abbreviations": parsed.detected_acronyms[:_QUERY_HINTS_MAX_ACRONYMS],
            "intent": parsed.intent,
            "sub_intent": parsed.sub_intent,
            "domain": parsed.domain,
        }
        if parsed.legal_references:
            hints["legal_references"] = parsed.legal_references[:_QUERY_HINTS_MAX_ACRONYMS]
        if parsed.financial_amounts:
            amounts = [
                item["formatted"] for item in parsed.financial_amounts[:_QUERY_HINTS_MAX_ACRONYMS]
                if isinstance(item, dict) and item.get("formatted")
            ]
            if amounts:
                hints["financial_amounts"] = amounts
        try:
            resolved_date, _ = policy_query_date(state["user_query"])
        except ValueError:
            resolved_date = None
        if resolved_date is None and state.get("human_replies"):
            reply_text = " ".join(
                str(item.get("response") or "") for item in state["human_replies"]
                if isinstance(item, dict)
            ).strip()
            if reply_text:
                try:
                    resolved_date, _ = policy_query_date(reply_text)
                except ValueError:
                    resolved_date = None
        if resolved_date is not None:
            hints["as_of"] = resolved_date.isoformat()
        state = {**state, "query_hints": hints, "last_action": None}
        if not should_compact(state.get("chat_history", [])):
            return state
        result, compacted = compact_memory(state, self.model, self.guard_artifact)
        if compacted:
            result["compactions_this_turn"] += 1
            result["execution_steps"] = state["execution_steps"] + ["memory_compact: automatic"]
            result["last_action"] = {"kind": "memory", "name": "compact", "status": "SUCCESS"}
        return result

    def _decide_input(self, state: dict) -> dict:
        turn = state["react_turn"]
        evidence = _evidence_view(turn["evidence"])
        # A revised search must expose its new results even when older sources
        # have filled the context budget. Labels themselves stay stable.
        recent = []
        for observation in reversed(turn["observations"]):
            if observation.get("retrieval"):
                for document in observation.get("data", {}).get("documents", []):
                    label = document.get("label")
                    if label and label not in recent:
                        recent.append(label)
        positions = {label: index for index, label in enumerate(recent)}
        evidence.sort(key=lambda item: positions.get(item["label"], len(positions)))
        return {
            "request": state["user_query"],
            "query_hints": state.get("query_hints", {}),
            "active_skills": state.get("active_skills", {}),
            "history": state.get("chat_history", []),
            "memory": memory_view(state),
            "plan": state["plan"],
            "observations": _compact_observations(turn["observations"]),
            "available_evidence": evidence,
            "remaining_actions": max(0, MAX_ACTIONS - len(turn["observations"])),
            "remaining_delegations": max(0, MAX_DELEGATIONS - len(state["delegations"])),
            "remaining_interactions": max(
                0,
                MAX_INTERACTIONS - len(state["human_replies"]) - int(state.get("prior_unresolved_interactions") or 0),
            ),
            "human_replies": state["human_replies"],
            "approved_scope": state["approved_scope"],
            "confirmation_required": state["confirmation_required"],
            "brief_summary_requested": _requests_brief_summary(state),
        }

    def decide(self, state: dict) -> dict:
        data = self._decide_input(state)
        data["remaining_legal_searches"] = legal_searches_remaining(state.get("tool_outputs", []))
        allowed = ["final"]
        system = BASE_PROMPT
        prepared = state.get("prepared_visualization")
        if data["remaining_interactions"] and not prepared:
            allowed += ["clarify", "confirm"]
        if data["remaining_actions"] and not prepared:
            allowed += ["plan", "remember", "forget"]
            if not state["confirmation_required"]:
                allowed.append("tool")
                if state["capabilities"]["skill"]:
                    allowed += ["skill", "inspect_skill"]
                if data["remaining_delegations"]:
                    allowed.append("delegate")
            if state["compactions_this_turn"] < MAX_COMPACT_PER_TURN and len(data["history"]) > RECENT_MESSAGES:
                allowed.append("compact")
            catalog = self.catalog_view(state["capabilities"])
            if not data["remaining_legal_searches"]:
                catalog["tool"].pop(LEGAL_WEB_SEARCH_TOOL, None)
            system += "\n" + COORDINATOR_PROMPT + _DEEP_MESSAGES.get("catalog_header", "\n") + json.dumps(catalog, ensure_ascii=False)
            system += _DEEP_MESSAGES.get("subagents_header", "\n") + json.dumps(
                {name: role["description"] for name, role in WORKERS.items()}, ensure_ascii=False
            )
            if "search_banking_knowledge" in catalog["tool"]:
                system += (
                    "\nFor search_banking_knowledge, omit 'source' on broad fee, policy, or account lookups unless the customer "
                    "explicitly names a document code/title (e.g., 'NV 01', 'BP 01', 'HDSD 05') or asks about mobile app navigation steps. "
                    "For comparison across specific dates, search each date separately with its explicit as_of. "
                    "When an applicability date is only a month or is ambiguous, clarify the date or state that the period needs verification. "
                    "A transaction date changes the policy lookup date only if the customer asks for the policy in force on that date."
                )
            if _COORDINATOR_SKILL_HINT:
                system += "\n" + _COORDINATOR_SKILL_HINT
            if PRESENT_VISUALIZATION_TOOL in catalog["tool"] and "tool" in allowed:
                system += (
                    "\nWhen the customer requests a diagram, flowchart, chart, or visual comparison (e.g. 'sơ đồ', 'biểu đồ', 'chart', 'diagram'), "
                    "or when a visual materially helps explain the requested information: "
                    "do NOT respond with final or plain text tables alone. "
                    "First activate the appropriate skill from the catalog using action 'skill' (e.g. kind 'skill' with name 'visualization' for charts and diagrams, "
                    "or 'interactive-simulator' for adjustable calculators) to load its visual schema, then present the visual in the next step. "
                    "For current rates or trends use verified evidence. Label illustrations as hypothetical."
                )
        persona = PERSONA_PROFILES.get(state.get("persona"), PERSONA_PROFILES.get(_DEFAULT_PERSONA, {}))
        tone = TONE_PROFILES.get(state.get("tone"), TONE_PROFILES.get(_DEFAULT_TONE, ""))
        format_style = FORMAT_PROFILES.get(state.get("format_type"), FORMAT_PROFILES.get(_DEFAULT_FORMAT_TYPE, ""))
        persona_role_text = persona.get("role_definition")
        if not persona_role_text:
            persona_role_text = persona.get("description", "")
        custom_persona_text = str(state.get("custom_persona") or "").strip()
        if state.get("persona") == "custom" and custom_persona_text:
            persona_role_text = persona_role_text + "\nCustom role instruction from customer: " + custom_persona_text
        system += _DEEP_MESSAGES.get("style_header", "\n") + "\n".join(
            [persona_role_text, tone, format_style]
        )
        system += _DEEP_MESSAGES.get("allowed_kinds_header", "\n") + ", ".join(allowed)
        if state.get("persona") == "custom":
            data["style_preference"] = custom_persona_text
        if state.get("audit_feedback"):
            data["review_feedback"] = state["audit_feedback"]
            data["previous_draft"] = state.get("draft_answer", "")
            data["revision_attempt"] = int(state.get("hallucination_retries") or 0)
            if _REVISION_INSTRUCTION:
                system += "\n" + _REVISION_INSTRUCTION
            system += (
                "\nREVISION DISCIPLINE: Never output previous_draft unchanged. If review failed on missing citations or missing facts "
                "and a previous search used a restrictive 'source' parameter (such as a folder prefix), call search_banking_knowledge "
                "WITHOUT 'source' if tools remain available; otherwise embed valid [Nguồn N] tags matching current available_evidence "
                "and drop/disclose unverified figures with insufficient_evidence=true."
            )

        recent = state.get("recent_visualizations", [])
        active_skills = state.get("active_skills", {})
        if VISUALIZATION_SKILL in active_skills and "tool" in allowed:
            system += (
                "\nVisualization skill is ALREADY active — do NOT call inspect_skill again. "
                "Gather necessary evidence first, then either call present_visualization with arguments={} "
                "(so the dedicated designer builds and validates the full schema automatically), OR return kind='final' "
                "and the designer will generate and validate the visual."
            )
        if SIMULATOR_SKILL in active_skills and "tool" in allowed:
            system += (
                "\nInteractive-simulator skill is ALREADY active — do NOT call inspect_skill again. "
                "Gather rates/rules if needed, then either call present_visualization with arguments={} "
                "(or {'artifact': {'engine': 'simulator', 'update_of': '<id>', 'defaults': {...}}} when updating defaults "
                "on a recent simulator) — the dedicated calculator designer will build and validate the full schema automatically — "
                "OR return kind='final' (including with insufficient_evidence=true for hypothetical scenarios)."
            )
        if prepared:
            data["active_skills"] = {key: value for key, value in active_skills.items()
                                     if key not in (SIMULATOR_SKILL, VISUALIZATION_SKILL)}
            prepared_vis = prepared.get("visualization") or {}
            extra_vis_hint = ""
            if prepared_vis.get("engine") == "mermaid":
                extra_vis_hint = (" Because engine='mermaid', also include the fenced ```mermaid source block in final.answer "
                                  "alongside the explanation and verified citations.")
            elif prepared_vis.get("engine") == "simulator":
                extra_vis_hint = (" Name the Interactive Simulator calculation operation(s) (e.g. operation='pmt'/'subtract'/'multiply') "
                                  "and adjustable slider defaults in final.answer without inventing ungrounded fee codes or rates.")
            if type(prepared_vis.get("revision")) is int and prepared_vis["revision"] > 1:
                extra_vis_hint += " Explicitly describe how this visual was updated from the previous version."
            if state["react_turn"].get("evidence"):
                extra_vis_hint += (
                    " Because available_evidence is non-empty, you MUST include valid [Nguồn N] citations "
                    "(e.g. [Nguồn 1]) next to any fee/policy facts taken from the retrieved documents so review does not fail on missing citations."
                )
            system += ("\nThe visualization is ready and will appear in the conversation. Return final with a helpful "
                       "customer-facing explanation and any necessary citations. Explain assumptions or trade-offs."
                       + extra_vis_hint)
        if recent:
            available_skills = state["capabilities"]["skill"]
            artifacts = []
            for item in recent_visualizations(recent):
                if item["engine"] == "simulator":
                    skill = SIMULATOR_SKILL
                else:
                    skill = VISUALIZATION_SKILL
                if skill in active_skills:
                    artifacts.append(item)
                elif skill in available_skills:
                    artifacts.append({**{key: item[key] for key in ("id", "engine", "kind", "title")}, "skill": skill})
            if artifacts:
                data["recent_visualizations"] = artifacts
                if not prepared:
                    system += "\nTo change a recent visualization or calculator, inspect its skill if it has not been activated."

        deep_reason = None
        if state.get("audit_feedback"):
            deep_reason = "review_revision"
        elif any(item["status"] == "SUCCESS" for item in state["delegations"]):
            deep_reason = "worker_evidence"
        elif len({item["chunk"].get("source") for item in state["react_turn"]["evidence"]
                  if item["chunk"].get("source")}) >= 2:
            deep_reason = "multiple_sources"
        turn = state["react_turn"]
        search_obs = []
        for item in turn.get("observations", []):
            if item.get("knowledge_search") or (item.get("kind"), item.get("name")) == ("tool", "search_banking_knowledge"):
                search_obs.append(item)
        num_searches = len(search_obs)
        zero_evidence_streak = 0
        for obs in reversed(search_obs):
            obs_data = obs.get("data")
            if isinstance(obs_data, dict):
                if obs_data.get("new_evidence", 0) == 0:
                    zero_evidence_streak += 1
                else:
                    break
            else:
                break

        search_limit_reached = num_searches >= 4 or zero_evidence_streak >= 2
        if search_limit_reached:
            system += (
                "\nThe internal document search budget is exhausted. Use clarify if available and a detail "
                "from the customer is still required. Otherwise answer the verified portion in scope and note "
                "any gaps. Do not call search_banking_knowledge again."
            )
        system += "\n\n" + ANSWER_SCOPE + "\n\n" + COORDINATION_POLICY
        mem_forgotten = [
            str(obs.get("key") or "").strip()
            for obs in turn.get("observations", [])
            if obs.get("kind") == "memory" and obs.get("name") == "forget" and obs.get("status") == "SUCCESS"
        ]
        if mem_forgotten:
            deleted_label = ", ".join(k for k in mem_forgotten if k)
            system += (
                f"\nMemory note(s) ({deleted_label}) already deleted in this turn (memory:forget SUCCESS). "
                "Do not call forget again for already-deleted keys; confirm deletion to the customer."
            )
        if data["brief_summary_requested"]:
            system += "\n" + SUMMARY_INSTRUCTION
            if state.get("audit_feedback"):
                system += (
                    f"\nSTRICT WORD LIMIT RETRY: Count every space-separated word carefully and keep the "
                    f"ENTIRE response under {max(80, SUMMARY_MAX_WORDS - 30)} words and at most {SUMMARY_MAX_POINTS} bullets. "
                    "Omit greetings, introductions, repetition, and closing filler."
                )

        try:
            decision = self.model.decide(system, data, allowed, max_tokens=_DECIDE_MAX_TOKENS, deep_reason=deep_reason)
        except ModelFallbackError as error:
            category = getattr(error, "category", "model_unavailable")
            if isinstance(getattr(error, "last_error", None), (ValueError, TypeError)):
                category = "model_contract"
            elif is_temporary_error(error):
                category = "provider_temporary"
            decision = self.recover_decision(data, state["confirmation_required"])
            if decision is None:
                return {
                    **state, "audit_status": "ERROR", "final_answer": FALLBACK_ANSWER,
                    "error_category": category,
                    "execution_steps": state["execution_steps"] + ["deep_decide: model_error"]
                }
            state = {**state, "execution_steps": state["execution_steps"] + ["deep_decide: recovered_contract"]}
        target_name = decision.get("name", "").strip()
        skill_target = None
        if decision["kind"] == "skill":
            skill_target = resolve_skill(catalog["skill"], target_name)
        if search_limit_reached and (
            (decision["kind"] == "tool" and target_name == "search_banking_knowledge")
            or (skill_target and skill_target["execution"]["type"] == "rag_pipeline")
        ):
            decision = self.recover_decision(data, state["confirmation_required"], can_clarify="clarify" in allowed)
            if decision is None:
                return {**state, "audit_status": "ERROR", "final_answer": FALLBACK_ANSWER}
        prefers_sim = self._prefers_simulator(state["user_query"], recent)
        if (decision["kind"] == "final" and (not decision.get("insufficient_evidence") or prefers_sim)
                and "tool" in allowed
                and not state.get("presentation_arguments_requested")
                and int(state.get("visual_revisions") or 0) < 2
                and self._explicit_chart_request(state["user_query"], recent)):
            definition = state["capabilities"]["tool"].get(PRESENT_VISUALIZATION_TOOL)
            if definition:
                state = {**state, "presentation_arguments_requested": True}
                if prefers_sim:
                    selected_skill_id = SIMULATOR_SKILL
                else:
                    selected_skill_id = VISUALIZATION_SKILL
                skill_entry = active_skills.get(selected_skill_id)
                if not isinstance(skill_entry, dict) or not skill_entry.get("instructions"):
                    skill_spec = state["capabilities"]["skill"].get(selected_skill_id)
                    if skill_spec and hasattr(self.skills, "instructions"):
                        try:
                            skill_entry = self.skills.instructions(skill_spec)
                        except (OSError, ValueError):
                            skill_entry = {}
                    else:
                        skill_entry = {}
                target_skills = {selected_skill_id: skill_entry}
                arguments = _sanitize_presentation_arguments(
                    self._design_presentation_arguments(definition, target_skills, data, recent)
                )
                if self._valid_presentation_arguments(arguments, recent):
                    decision = {"kind": "tool", "name": PRESENT_VISUALIZATION_TOOL, "arguments": arguments}
        if decision["kind"] == "tool" and "tool" in allowed:
            registered = state["capabilities"]["tool"]
            arguments = decision["arguments"]
            artifact = arguments.get("artifact")
            name = decision["name"].strip().lstrip(":").strip()
            if not state.get("worker") and name in ("create_support_ticket", "transfer_to_agent") and "confirm" in allowed:
                approved_confirm = any(
                    item.get("kind") == "confirm" and "approve" in item.get("selected_ids", [])
                    for item in state.get("human_replies", [])
                )
                if not approved_confirm:
                    if name == "transfer_to_agent":
                        proposal = (
                            "Xác nhận kết nối yêu cầu của Quý khách tới hàng đợi Chuyên viên Tư vấn "
                            "(trường hợp khẩn cấp vui lòng gọi ngay Hotline 1900 6688). Quý khách muốn tiếp tục không?"
                        )
                    else:
                        issue_desc = str(arguments.get("description") or arguments.get("issue_type") or "hỗ trợ khẩn cấp").strip()
                        proposal = (
                            f"Xác nhận tạo phiếu yêu cầu hỗ trợ ({issue_desc[:140]}). "
                            "Trường hợp khẩn cấp vui lòng liên hệ Hotline 1900 6688. Quý khách có đồng ý tiếp tục không?"
                        )
                    decision = {"kind": "confirm", "proposal": proposal}
            if decision["kind"] == "tool":
                if name == "export_visualization_json" and set(arguments) <= {"filename", "artifact"}:
                    name = PRESENT_VISUALIZATION_TOOL
                    if artifact is not None:
                        arguments = {"artifact": artifact}
                    else:
                        arguments = {}
                if (name not in registered and set(arguments) == {"artifact"}
                        and isinstance(artifact, dict) and PRESENT_VISUALIZATION_TOOL in registered):
                    engine = artifact.get("engine")
                    if ((engine == "simulator" and SIMULATOR_SKILL in active_skills)
                            or (engine in ("echarts", "mermaid") and VISUALIZATION_SKILL in active_skills)):
                        name = PRESENT_VISUALIZATION_TOOL
                if name == PRESENT_VISUALIZATION_TOOL and name in registered:
                    if int(state.get("visual_revisions") or 0) >= 2:
                        valid = False
                        state = {**state, "presentation_arguments_requested": True}
                    else:
                        if not arguments and not state.get("presentation_arguments_requested"):
                            valid = False
                        else:
                            arguments = _sanitize_presentation_arguments(arguments)
                            valid = self._valid_presentation_arguments(arguments, recent)
                        if not valid and not state.get("presentation_arguments_requested"):
                            state = {**state, "presentation_arguments_requested": True}
                            arguments = _sanitize_presentation_arguments(
                                self._design_presentation_arguments(registered[name], active_skills, data, recent)
                            )
                            valid = self._valid_presentation_arguments(arguments, recent)
                    if valid:
                        decision = {**decision, "name": name, "arguments": arguments}
                    else:
                        try:
                            decision = self.model.decide(
                                system
                                + "\nThis turn's visualization is unavailable after validation. "
                                + "Do NOT emit kind='tool' or claim any simulator or chart was created; "
                                + "output ONLY kind='final' with verified formulas and figures in text.",
                                data, ["final"], max_tokens=_RECOVER_MAX_TOKENS,
                            )
                        except ModelFallbackError:
                            decision = {"kind": "final", "answer": FALLBACK_ANSWER, "insufficient_evidence": True}
        if state["confirmation_required"] and decision["kind"] == "final":
            return {
                **state,
                "audit_status": "NEEDS_INPUT",
                "final_answer": _DEEP_MESSAGES.get("confirmation_required_answer", ""),
            }
        if decision["kind"] in ("clarify", "confirm"):
            try:
                decision = self.guard_payload(decision)
            except ValueError:
                return self._unsafe_interaction(state)
            texts = [decision.get("question", decision.get("proposal", "")), *decision.get("options", [])]
            if any(not text or "REDACTED]" in text.upper() or _requests_restricted_field(text) for text in texts):
                return self._unsafe_interaction(state)
        return {
            **state, "react_turn": {**state["react_turn"], "decision": decision},
            "execution_steps": state["execution_steps"] + ["deep_decide: " + decision["kind"]]
        }

    @staticmethod
    def _prefers_simulator(query, recent):
        text = " " + search_key(query) + " "
        sim_tokens = (
            " mo phong ", " bo mo phong ", " trinh mo phong ", " interactive simulator ",
            " simulator ", " thanh truot ", " slider ", " cong cu mo phong ",
        )
        if any(token in text for token in sim_tokens):
            return True
        if recent and (recent[0].get("visualization") or {}).get("engine") == "simulator":
            return True
        return False

    @staticmethod
    def _explicit_chart_request(query, recent):
        text = " " + search_key(query) + " "
        visual_tokens = (
            " so do ", " bieu do ", " chart ", " diagram ", " flowchart ", " mindmap ",
            " ve bieu do ", " tao bieu do ", " ve so do ", " tao so do ", " show so do ",
            " cho xem so do ", " xem so do ", " hien thi so do ", " do thi ",
            " mo phong ", " bo mo phong ", " trinh mo phong ", " interactive simulator ",
            " simulator ", " thanh truot ", " slider ", " cong cu mo phong ",
        )
        for token in visual_tokens:
            if token in text:
                return True
        if recent:
            update_tokens = (
                " doi sang ", " chuyen sang ", " change it to ", " switch to ",
                " cap nhat ", " update ", " chinh sua ", " sua lai ",
                " thay doi ", " dieu chinh ", " bo sung ", " doi mau ", " them nhom ",
            )
            for switch_token in update_tokens:
                if switch_token in text:
                    return True
        return False

    @staticmethod
    def _valid_presentation_arguments(arguments, recent):
        artifact = arguments.get("artifact")
        if set(arguments) != {"artifact"} or not isinstance(artifact, dict):
            return False
        update_of = artifact.get("update_of")
        if update_of and not any(item["visualization"]["id"] == update_of for item in recent):
            return False
        try:
            present_visualization(artifact)
        except (ValueError, TypeError, RecursionError):
            return False
        return True

    def _design_presentation_arguments(self, definition, active_skills, data, recent):
        available = {SIMULATOR_SKILL: SIMULATOR_SCHEMA, VISUALIZATION_SKILL: VISUALIZATION_SCHEMA}
        selected = [skill for skill in available if skill in active_skills] or list(available)
        schemas = [available[skill] for skill in selected]
        base = definition["inputSchema"]
        definitions = {key: value for artifact in schemas for key, value in artifact["$defs"].items()}
        for name in ("ChartSelection", "DiagramSelection", "SimulatorSelection"):
            if name in definitions:
                full = definitions[name]
                definitions[name] = {
                    **full, "required": [*full["required"], "layout"],
                    "properties": {**full["properties"], "layout": {"$ref": "#/$defs/LayoutBlock"}},
                }
        artifact_branches = []
        for artifact in schemas:
            branch = {key: value for key, value in artifact.items() if key != "$defs"}
            if not recent and "anyOf" in branch:
                filtered = [item for item in branch["anyOf"] if "Update" not in str(item.get("$ref", ""))]
                if len(filtered) == 1:
                    branch = filtered[0]
            artifact_branches.append(branch)
        if len(artifact_branches) == 1:
            artifact_schema = artifact_branches[0]
        else:
            artifact_schema = {"anyOf": artifact_branches}
        schema = {
            **base,
            "$defs": definitions,
            "properties": {
                **base["properties"],
                "artifact": artifact_schema,
            },
        }
        system = (
            "Design a useful customer-facing visualization. Return only the JSON object with artifact. "
            "Every complete artifact needs a layout. For charts the layout includes a chart block, "
            "for diagrams a diagram block, and for calculators add input and result blocks by ID. "
            "ONLY 'grid' layout blocks accept 'columns' (never put 'columns' on 'column' or 'row'). "
            "Never add extra keys such as 'description' or top-level 'operation' on artifact. "
            "Use verified or supplied data and clearly label hypothetical scenarios. "
            "For a chart presentation edit use update_of and presentation with the changed fields, "
            "preserving the original data. For an interactive simulator default edit (SimulatorDefaultUpdate) "
            "use engine='simulator', update_of (exact id from recent_visualizations), and defaults with the "
            "updated input id->value pairs (do NOT resend inputs/formulas unless changing schema). "
            "For calculator formulas use one operation per row; put 'tone' ('plain'|'card'|'accent') ONLY on layout blocks, "
            "NEVER inside calculations items; keep layout 'text' non-empty and at most 180 characters. "
            "When available_evidence contains exact fee rates, floors, caps, or interest rates matching the customer's scenario, "
            "set simulator input defaults to those verified figures (and cite the fee code in assumptions); "
            "only use clearly labeled hypothetical defaults when no matching figure exists in available_evidence. "
            "If review_feedback is provided, fix every visualization issue it names.\n"
            + "\n".join(active_skills.get(skill, {}).get("instructions", "") for skill in selected)
            + "\nJSON Schema: " + json.dumps(schema, ensure_ascii=False)
            + "\nReturn the arguments themselves, without a tool name, action wrapper or commentary."
        )
        context = {key: data[key] for key in ("request", "observations", "available_evidence", "human_replies", "review_feedback") if key in data}
        if recent:
            context["recent_visualizations"] = recent_visualizations(recent)
        try:
            return self.model.request(
                system, context, [schema], max_tokens=_REVIEW_MAX_TOKENS,
                concise=True, normalize=_sanitize_presentation_arguments,
            )
        except ModelFallbackError:
            return {}

    def recover_decision(self, data: dict, confirmation_required: bool, *, can_clarify=False) -> dict | None:
        if confirmation_required:
            return None
        has_success = False
        for item in data.get("observations", []):
            if item.get("kind") in ("tool", "skill", "delegate"):
                if item.get("status") == "SUCCESS":
                    if "data" in item:
                        has_success = True
                        break
        if not has_success and not can_clarify:
            return None
        instruction = "\n\n".join((BASE_PROMPT, _RECOVER_PROMPT, ANSWER_SCOPE))
        if data["brief_summary_requested"]:
            instruction += "\n" + SUMMARY_INSTRUCTION
        allowed = ["final"]
        if can_clarify:
            allowed.append("clarify")
            instruction += "\nA clarify action is also allowed if a missing detail changes the answer.\n" + COORDINATION_POLICY
        try:
            return self.model.decide(instruction, data, allowed, max_tokens=_RECOVER_MAX_TOKENS)
        except ModelFallbackError:
            return None

    def _record(self, state: dict, observation: dict) -> dict:
        turn = state["react_turn"]
        note = {key: observation[key] for key in ("kind", "name", "status")}
        return {
            **state,
            "react_turn": {**turn, "observations": turn["observations"] + [observation]},
            "last_action": note,
            "execution_steps": state["execution_steps"] + [
                f"{note['kind']}: {note['name']} ({note['status']})"
            ],
        }

    def _available_skills(self, state: dict) -> dict:
        definitions = state["capabilities"]["skill"]
        if state.get("worker_role"):
            definitions = self._worker_snapshot(state["worker_role"], state["capabilities"])["skill"]
        return definitions

    def act(self, state: dict) -> dict:
        decision = state["react_turn"]["decision"]
        if decision["kind"] == "skill":
            spec = resolve_skill(self._available_skills(state), decision["name"])
            if spec is None and state.get("worker_role"):
                return self._record(state, {"kind": "skill", "name": "restricted", "status": "ERROR",
                                            "error": _DEEP_MESSAGES.get("skill_restricted_worker", "")})
            if spec and "revision" in spec and (
                spec["skill_id"] not in state.get("active_skills", {})
                or spec["skill_id"] in (SIMULATOR_SKILL, VISUALIZATION_SKILL)
            ):
                return self.inspect_skill(state)
        if decision.get("kind") == "tool" and decision.get("name") == PRESENT_VISUALIZATION_TOOL and state.get("prepared_visualization"):
            return {**state, "last_action": None}
        result = super().act(state)
        if decision.get("kind") != "tool" or decision.get("name") != PRESENT_VISUALIZATION_TOOL:
            return result
        observation = result["react_turn"]["observations"][-1]
        if observation["status"] != "SUCCESS":
            observation["error"] = "Provide a valid artifact with an authored layout or a supported update for an exact session ID."
            return result
        prepared = prepare_presentation(observation["data"]["artifact"], state.get("recent_visualizations", []), self.guard_artifact)
        if prepared:
            return {**result, "prepared_visualization": prepared}
        observation.pop("data", None)
        observation.update(status="ERROR", error_code="invalid_visualization", error="Check the artifact and its session revision ID.")
        result["last_action"]["status"] = "ERROR"
        result["tool_outputs"][-1]["status"] = "ERROR"
        result["execution_steps"][-1] = f"react_action: tool {PRESENT_VISUALIZATION_TOOL} (ERROR)"
        return result

    def inspect_skill(self, state: dict) -> dict:
        decision = state["react_turn"]["decision"]
        spec = resolve_skill(self._available_skills(state), decision["name"])
        if spec is None:
            return self._record(state, {"kind": "inspect_skill", "name": "unknown", "status": "ERROR",
                                        "error": _DEEP_MESSAGES.get("skill_not_found", "")})
        name = spec["skill_id"]
        active = state.get("active_skills", {})
        reference = decision.get("reference")
        event = {"kind": "inspect_skill", "name": name, "status": "SUCCESS"}
        try:
            if reference is None:
                if name in active:
                    return self._record(state, {**event, "status": "ALREADY_ACTIVE"})
                loaded = self.skills.instructions(spec)
            else:
                event["reference"] = reference
                if name not in active:
                    raise ValueError("Activate the skill before reading its references")
                current = active[name]
                resources = current.get("loaded_references", {})
                if reference in resources:
                    return self._record(state, {**event, "status": "ALREADY_ACTIVE"})
                text = self.skills.reference(spec, reference, current["references"])
                loaded = {**current, "loaded_references": {**resources, reference: text}}
            updated = {**active, name: loaded}
            if len(json.dumps(updated, ensure_ascii=False).encode("utf-8")) > MAX_ACTIVE_SKILL_BYTES:
                raise ValueError("Active skill context budget exceeded")
        except (OSError, ValueError, RecursionError):
            return self._record(state, {**event, "status": "ERROR",
                                        "error": _DEEP_MESSAGES.get("skill_load_error", "")})
        return self._record({**state, "active_skills": updated}, event)

    def update_plan(self, state: dict) -> dict:
        decision = state["react_turn"]["decision"]
        items = decision["items"]
        try:
            guarded = self.guard_payload(items)
            if len({item["id"] for item in guarded}) != len(guarded):
                raise ValueError("Repeated plan ids")
            if any(not item["task"] for item in guarded):
                raise ValueError("Empty plan task")
            if guarded == state["plan"]:
                return self._record(state, {
                    "kind": "plan", "name": "unchanged", "status": "ERROR",
                    "error": _DEEP_MESSAGES.get("plan_unchanged", "")
                })
        except ValueError:
            return self._record(state, {
                "kind": "plan", "name": "update", "status": "ERROR",
                "error": _DEEP_MESSAGES.get("plan_invalid", "")
            })
        requires_confirmation = state["confirmation_required"]
        if state["approved_scope"] is not None:
            requires_confirmation = self.plan_scope(guarded) != state["approved_scope"]["plan"]
        return self._record({**state, "plan": guarded, "plan_revision": state["plan_revision"] + 1,
                             "confirmation_required": requires_confirmation},
                            {"kind": "plan", "name": "update", "status": "SUCCESS"})

    @staticmethod
    def plan_scope(plan: list) -> list:
        return [{"id": item["id"], "task": item["task"]} for item in plan]

    def prepare_interaction(self, state: dict, *, durable: bool) -> dict:
        decision = state["react_turn"]["decision"]
        mode = decision["kind"]
        field = "question"
        if mode == "confirm":
            field = "proposal"
        message = decision[field]
        labels = decision.get("options", [])
        normalized = [" ".join(label.casefold().split()) for label in labels]
        reserved_labels = set(_DEEP_MESSAGES.get("clarify_reserved_labels", ()))
        if len(set(normalized)) != len(labels) or reserved_labels.intersection(normalized):
            return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_invalid_options", ""))
        scope = []
        if mode == "confirm":
            scope = self.plan_scope(state["plan"])
        multi_select = False
        if mode == "clarify" and decision.get("multi_select") is True:
            multi_select = True
        signature = [mode, " ".join(message.casefold().split()), scope]
        used_interactions = len(state["human_replies"]) + int(state.get("prior_unresolved_interactions") or 0)
        if used_interactions >= MAX_INTERACTIONS or any(
            reply["signature"] == signature for reply in state["human_replies"]
        ):
            if mode == "clarify":
                message = (
                    f"Đã qua tối đa {MAX_INTERACTIONS} lượt làm rõ nên hệ thống xin dừng hỏi thêm. "
                    "Quý khách vui lòng bổ sung trực tiếp các tham số cụ thể (như số tiền, kỳ hạn, sản phẩm hoặc nhu cầu) để tôi tư vấn tiếp."
                )
            return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_exhausted_prefix", "") + message)
        if mode == "confirm":
            options = [
                {"id": "approve", "label": _DEEP_MESSAGES.get("confirm_approve_label", "")},
                {"id": "reject", "label": _DEEP_MESSAGES.get("confirm_reject_label", "")},
            ]
        else:
            options = [{"id": f"option_{i}", "label": label} for i, label in enumerate(labels, 1)]
            options += [
                {"id": "other", "label": _DEEP_MESSAGES.get("clarify_other_label", "")},
                {"id": "cancel", "label": _DEEP_MESSAGES.get("clarify_cancel_label", "")},
            ]
        if not durable:
            answer = " ".join([message, " / ".join(labels),
                               _DEEP_MESSAGES.get("non_durable_interaction_suffix", "")])
            return self._interaction_stop(state, answer)
        return {**state, "interaction": {"id": uuid4().hex, "kind": mode, "message": message,
                                         "options": options, "scope": scope, "multi_select": multi_select},
                "interaction_signature": signature,
                "execution_steps": state["execution_steps"] + ["interaction: awaiting_input"]}

    def await_interaction(self, state: dict) -> dict:
        card = state["interaction"]
        reply = interrupt(card)
        if not isinstance(reply, dict) or reply.get("interaction_id") != card["id"]:
            return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_mismatch_id", ""))
        raw_ids = []
        choice_ids = reply.get("choice_ids")
        if isinstance(choice_ids, list) and choice_ids:
            for item in choice_ids:
                if not isinstance(item, str) or not item:
                    return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_mismatch_choice", ""))
                if item not in raw_ids:
                    raw_ids.append(item)
        else:
            single_id = reply.get("choice_id")
            if isinstance(single_id, str) and single_id:
                raw_ids = [single_id]
        if not raw_ids:
            return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_mismatch_choice", ""))

        option_map = {option["id"]: option for option in card["options"]}
        selected = []
        for cid in raw_ids:
            option = option_map.get(cid)
            if option is None:
                return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_mismatch_choice", ""))
            selected.append(option)

        allow_multi = card.get("multi_select") is True
        if len(selected) > 1:
            if not allow_multi:
                return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_mismatch_choice", ""))
            for opt in selected:
                if opt["id"] in ("cancel", "reject"):
                    return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_mismatch_choice", ""))

        state = {**state, "interaction": None, "accepted_interaction_id": card["id"]}
        if len(selected) == 1 and selected[0]["id"] in ("cancel", "reject"):
            return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_canceled", ""), status="CANCELED")

        parts = []
        for opt in selected:
            if opt["id"] == "other":
                custom = self.guard_artifact(reply.get("text") or "")
                if not custom:
                    return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_unusable_reply", ""))
                parts.append(custom)
            else:
                parts.append(opt["label"])
        response = "; ".join(parts)

        if card["kind"] == "confirm":
            state["approved_scope"] = {"proposal": card["message"], "plan": card["scope"]}
            state["confirmation_required"] = False
        entry = {"kind": card["kind"], "question": card["message"], "response": response,
                 "selected_ids": [opt["id"] for opt in selected],
                 "signature": state["interaction_signature"]}
        state["human_replies"] = state["human_replies"] + [entry]
        return self._record(state, {"kind": card["kind"], "name": "human_input", "status": "SUCCESS"})

    @staticmethod
    def _interaction_stop(state: dict, answer: str, status: str = "NEEDS_INPUT") -> dict:
        return {**state, "interaction": None, "final_answer": answer, "audit_status": status,
                "execution_steps": state["execution_steps"] + ["interaction: " + status.lower()]}

    def _unsafe_interaction(self, state: dict) -> dict:
        return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_unsafe", ""))

    def manage_memory(self, state: dict) -> dict:
        decision = state["react_turn"]["decision"]
        kind = decision["kind"]
        error = ""
        if kind == "compact":
            state, compacted = compact_memory(state, self.model, self.guard_artifact)
            if compacted:
                state["compactions_this_turn"] += 1
            else:
                error = _DEEP_MESSAGES.get("memory_already_compact", "")
        else:
            user_messages = [state["user_query"]] + [
                item["content"] for item in state.get("chat_history", []) if item["role"] == "user"
            ]
            memory, error = change_note(memory_view(state), decision, user_messages, self.guard_artifact)
            state = {**state, "session_memory": memory}
        status = "SUCCESS"
        if error:
            status = "ERROR"
        return self._record(state, {
            "kind": "memory", "name": kind, "key": str(decision.get("key") or "").strip(),
            "status": status, "error": error
        })

    def _worker_snapshot(self, role: str, snapshot: dict) -> dict:
        tools = {}
        for name, spec in snapshot["tool"].items():
            if "policy" in spec:
                allowed_tool = role in spec["policy"].get("workers", [])
            else:
                allowed_tool = name in WORKERS[role]["tool"]
            if allowed_tool:
                tools[name] = spec
        skills = {}
        for name, spec in snapshot["skill"].items():
            execution = spec["execution"]
            if execution["type"] == "registered_tool":
                if execution["tool"] in tools:
                    skills[name] = spec
            elif name.replace("-", "_") in WORKERS[role]["skill"]:
                skills[name] = spec

        # A skill may guard against a registered tool without exposing that tool to the worker.
        return {"tool": tools, "skill": skills}

    def _run_worker(self, role: str, task: str, snapshot: dict) -> dict:
        catalog = self.catalog_view(self._worker_snapshot(role, snapshot))
        system = (
            BASE_PROMPT + "\n" + WORKER_PROMPT
            + _DEEP_MESSAGES.get("worker_role_header", "\n") + WORKERS[role]["description"]
            + _DEEP_MESSAGES.get("worker_catalog_header", "\n") + json.dumps(catalog, ensure_ascii=False)
        )
        worker = self.input_guardrail({"user_query": task, "capabilities": snapshot})
        if not worker["is_safe"]:
            return {"error": _DEEP_MESSAGES.get("worker_guardrail_rejected", "")}
        worker["worker"] = True
        worker["worker_role"] = role
        for step in range(MAX_WORKER_ACTIONS + 1):
            turn = worker["react_turn"]
            allowed = ["final"]
            if step < MAX_WORKER_ACTIONS:
                allowed.append("tool")
                if catalog["skill"]:
                    allowed += ["skill", "inspect_skill"]
            request = {
                "task": worker["user_query"], "allowed_kinds": allowed,
                "remaining_actions": MAX_WORKER_ACTIONS - step,
                "observations": _compact_observations(turn["observations"]),
                "active_skills": worker.get("active_skills", {}),
                "available_evidence": _evidence_view(turn["evidence"])
            }
            deep_reason = None
            if len({item["chunk"].get("source") for item in turn["evidence"]
                    if item["chunk"].get("source")}) >= 2:
                deep_reason = "multiple_sources"
            decision = self.model.decide(system, request, allowed, max_tokens=_WORKER_MAX_TOKENS, deep_reason=deep_reason)
            kind = decision["kind"]
            if kind == "final":
                raw = self.guard_artifact(decision["answer"])
                screened = self.guardrails.guard_output(raw)
                if not screened.is_safe or not screened.sanitized_answer or screened.flags:
                    return {"error": _DEEP_MESSAGES.get("worker_report_guardrail_rejected", "")}
                report = screened.sanitized_answer
                self.validate_citations({**decision, "answer": report}, turn["evidence"])
                return {"report": report, "evidence": turn["evidence"], "calls": worker["tool_outputs"],
                        "observations": turn["observations"], "searches": worker["search_count"]}
            name = decision["name"]
            if kind == "inspect_skill":
                group = "skill"
            else:
                group = kind
            allowed_member = name in catalog[group]
            if group == "skill":
                allowed_member = resolve_skill(catalog[group], name) is not None
            if not allowed_member:
                worker = self._record(worker, {"kind": kind, "name": "restricted", "status": "ERROR",
                                               "error": _DEEP_MESSAGES.get("worker_restricted_tool", "")})
                continue
            worker["react_turn"] = {**turn, "decision": decision}
            if kind == "inspect_skill":
                worker = self.inspect_skill(worker)
            else:
                worker = self.act(worker)
        return {"error": _DEEP_MESSAGES.get("worker_exhausted", "")}

    def _merge_worker_evidence(self, parent: list, child: list, preferred_indices: set | None = None) -> tuple[list, dict]:
        merged = list(parent)
        known = {tuple(item["identity"]): item["index"] for item in merged}
        mapping = {}
        ordered_child = list(child)
        if preferred_indices:
            ordered_child.sort(key=lambda item: item["index"] not in preferred_indices)
        for item in ordered_child:
            key = tuple(item["identity"])
            if key not in known:
                if len(merged) >= MAX_EVIDENCE:
                    if preferred_indices and item["index"] in preferred_indices:
                        raise ValueError("Turn evidence limit reached")
                    continue
                known[key] = len(merged) + 1
                merged.append({**item, "index": known[key]})
            mapping[item["index"]] = known[key]
        return merged, mapping

    def delegate(self, state: dict) -> dict:
        turn = state["react_turn"]
        decision = turn["decision"]
        role = decision["role"]
        task = self.guard_artifact(decision["task"])
        status = "ERROR"
        data = {}
        error = _DEEP_MESSAGES.get("delegate_unregistered_or_unsafe", "")
        result = {}
        if role not in WORKERS:
            role = "unknown"
        elif task:
            repeated = any(item["role"] == role and item["task"] == task for item in state["delegations"])
            if repeated:
                error = _DEEP_MESSAGES.get("delegate_repeated", "")
            else:
                try:
                    result = self._run_worker(role, task, state["capabilities"])
                    error = result.get("error", "")
                    if not error:
                        norm_report = _normalize_citation_markers(result["report"])
                        parts = norm_report.split("[Nguồn ")
                        cited_worker_indices = set()
                        for part in parts[1:]:
                            label_body, closing, _ = part.partition("]")
                            if not closing:
                                raise ValueError("Invalid worker citation")
                            for tok in label_body.replace(";", ",").split(","):
                                clean_tok = tok.strip()
                                if clean_tok.startswith("Nguồn "):
                                    clean_tok = clean_tok[len("Nguồn "):].strip()
                                if " " in clean_tok:
                                    clean_tok = clean_tok.split()[0].rstrip(":,-")
                                if clean_tok.isdigit():
                                    cited_worker_indices.add(int(clean_tok))
                        merged, mapping = self._merge_worker_evidence(
                            turn["evidence"], result["evidence"], preferred_indices=cited_worker_indices
                        )
                        report = parts[0]
                        for part in parts[1:]:
                            label_body, closing, tail = part.partition("]")
                            if not closing:
                                raise ValueError("Invalid worker citation")
                            raw_tokens = [tok.strip() for tok in label_body.replace(";", ",").split(",") if tok.strip()]
                            if not raw_tokens:
                                raise ValueError("Invalid worker citation")
                            remapped = []
                            for idx_pos, tok in enumerate(raw_tokens):
                                clean_tok = tok
                                if idx_pos > 0 and clean_tok.startswith("Nguồn "):
                                    clean_tok = clean_tok[len("Nguồn "):].strip()
                                if " " in clean_tok:
                                    clean_tok = clean_tok.split()[0].rstrip(":,-")
                                remapped.append(_label(mapping[int(clean_tok)]))
                            report += "".join(remapped) + tail
                        used_indices = []
                        for item in result["evidence"]:
                            if item["index"] in mapping:
                                mapped_idx = mapping[item["index"]]
                                if mapped_idx not in used_indices:
                                    used_indices.append(mapped_idx)
                        docs = [merged[index - 1] for index in used_indices]
                        work = [
                            item for item in result["observations"] if not item.get("retrieval")
                        ]
                        searches_used = [
                            {
                                "query_used": item["query_used"],
                                "status": item["status"],
                                "documents_found": len(item.get("data", {}).get("documents", []))
                            }
                            for item in result["observations"] if "query_used" in item
                        ]
                        data = {
                            "report": report, "documents": _evidence_view(docs),
                            "worker_observations": work, "search_attempts": searches_used
                        }
                        turn = {**turn, "evidence": merged}
                        status = "SUCCESS"
                except (ModelFallbackError, ValueError):
                    error = _DEEP_MESSAGES.get("delegate_unverified", "")
        calls = state["tool_outputs"]
        searches = state["search_count"]
        intent = state["intent"]
        if status == "SUCCESS":
            calls = calls + [{**item, "worker": role} for item in result["calls"]]
            searches += result["searches"]
            if searches:
                intent = "RAG_SEARCH"
            elif calls:
                intent = "BANKING_ACTION"
        state = {
            **state, "intent": intent, "react_turn": turn, "tool_outputs": calls,
            "retrieved_chunks": [item["chunk"] for item in turn["evidence"]],
            "search_count": searches, "retry_count": max(0, searches - 1),
            "delegations": state["delegations"] + [{"role": role, "task": task, "status": status}]
        }
        return self._record(state, {
            "kind": "delegate", "name": role, "status": status, "data": data, "error": error,
            "retrieval": "documents" in data
        })

    def _retry_review(self, state: dict, draft: str, feedback: str, outcome: str) -> dict:
        retries = state["hallucination_retries"]
        revised = {
            **state, "audit_feedback": feedback, "draft_answer": draft,
            "reviews": state["reviews"] + [outcome],
            "execution_steps": state["execution_steps"] + ["review: " + outcome]
        }
        if retries < MAX_REVISIONS:
            return {**revised, "audit_status": "RETRY", "hallucination_retries": retries + 1}
        return {**revised, "audit_status": "FAIL", "final_answer": FALLBACK_ANSWER}

    def _review_feedback(self, verdict, draft, evidence):
        feedback = self.guard_artifact(verdict["feedback"])
        scope_feedback = self.guard_artifact(verdict.get("scope_issue", ""))
        labels = {item["label"] for item in evidence}
        line_count = len(draft.splitlines())
        items = []
        if scope_feedback:
            items.append(scope_feedback)
        if feedback:
            items.append(feedback)
        for issue in verdict.get("issues", []):
            description = self.guard_artifact(issue["item"] + ": " + issue["problem"])
            if not description:
                continue
            pointer = issue.get("draft_line")
            if type(pointer) is int and 1 <= pointer <= line_count:
                description = f"D{pointer} — " + description
            if issue.get("source") in labels:
                description += " " + issue["source"]
            items.append(description)
        return "\n".join(items) or _DEEP_MESSAGES.get("review_default_feedback", "")

    def _has_non_rag_grounding(self, state: dict) -> bool:
        support = _support_view(state)
        tool_prefix = _DEEP_MESSAGES.get("tool_support_prefix", "[Tool ")
        if "[Ảnh đính kèm]" in support or "[Lịch sử trả lời trước]" in support:
            return True
        if any(key.startswith(tool_prefix) for key in support):
            return True
        return bool(state.get("search_count"))

    def review(self, state: dict) -> dict:
        result = self._review_draft(state)
        can_recover = bool(state["react_turn"]["evidence"] or self._has_non_rag_grounding(state))
        if (result["audit_status"] != "FAIL" or state.get("confirmation_required")
                or result["reviews"][-1] not in ("revise", "claim_failed", "citation_failed", "unavailable")
                or not can_recover or not result.get("draft_answer")):
            return result

        # A single final-only attempt after ordinary revisions are used up.
        # It cannot call tools, and its new answer must pass the normal reviewer.
        data = self._decide_input(result)
        ev_items = data.get("available_evidence") or []
        if ev_items and _PARTIAL_RECOVERY_MAX_SOURCES > 0:
            kept_sources = set()
            bounded_ev = []
            for ev_item in ev_items:
                src_name = str(ev_item.get("source") or "")
                if src_name not in kept_sources:
                    if len(kept_sources) >= _PARTIAL_RECOVERY_MAX_SOURCES:
                        continue
                    kept_sources.add(src_name)
                bounded_ev.append(ev_item)
            data["available_evidence"] = bounded_ev
        data["review_feedback"] = result["audit_feedback"]
        data["previous_draft"] = result["draft_answer"]
        instruction = "\n\n".join((BASE_PROMPT, _PARTIAL_RECOVERY_PROMPT, ANSWER_SCOPE))
        if data.get("available_evidence"):
            instruction += (
                "\nMANDATORY CITATION: available_evidence is non-empty; embed valid [Nguồn N] tags "
                "matching available_evidence in your final answer."
            )
        if data["brief_summary_requested"]:
            prev_words = len(str(result.get("draft_answer") or "").split())
            target_words = max(80, SUMMARY_MAX_WORDS - 25)
            instruction += (
                f"\n{SUMMARY_INSTRUCTION} "
                f"(Bản nháp trước dài {prev_words} từ; hãy viết bản tóm tắt mới dưới {target_words} từ và tối đa {SUMMARY_MAX_POINTS} ý.)"
            )
        try:
            decision = self.model.decide(instruction, data, ["final"], max_tokens=_PARTIAL_RECOVERY_MAX_TOKENS,
                                         deep_reason="review_revision")
        except ModelFallbackError:
            return result
        decision = {**decision, "insufficient_evidence": True}
        turn = {**result["react_turn"], "decision": decision}
        limited = self._review_draft({**result, "react_turn": turn}, partial=True)
        if limited["audit_status"] == "PASS":
            return limited
        return {**result, "reviews": limited["reviews"], "execution_steps": limited["execution_steps"]}

    def _screen_follow_up_questions(self, questions: list[str], query: str) -> list[str]:
        seen = {search_key(query)}
        screened = []
        for question in questions:
            question = " ".join(question.split())
            key = search_key(question)
            if not key or key in seen or not question.endswith("?") or _requests_restricted_field(question):
                continue
            if self.guard_artifact(question) != question:
                continue
            output = self.guardrails.guard_output(question)
            if not output.is_safe or output.flags or output.sanitized_answer != question:
                continue
            seen.add(key)
            screened.append(question)
        return screened

    def _review_draft(self, state: dict, *, partial=False) -> dict:
        turn = state["react_turn"]
        decision = turn["decision"]
        raw = self.guard_artifact(decision["answer"])
        output = self.guardrails.guard_output(raw)
        draft = output.sanitized_answer
        if not output.is_safe or not draft or output.flags:
            return self._retry_review(state, "", _DEEP_MESSAGES.get("review_draft_blocked", ""), "blocked")
        if _requests_brief_summary(state) and not partial:
            word_count, point_count = _brief_summary_counts(draft)
            if word_count > SUMMARY_MAX_WORDS or point_count > SUMMARY_MAX_POINTS:
                target_words = max(80, SUMMARY_MAX_WORDS - 30)
                over_words = max(0, word_count - SUMMARY_MAX_WORDS)
                cut_need = max(1, word_count - target_words)
                feedback = (
                    f"{SUMMARY_INSTRUCTION} "
                    f"(Bản nháp hiện tại có {word_count} từ — vượt {over_words} từ — và {point_count}/{SUMMARY_MAX_POINTS} ý. "
                    f"Hãy cắt bỏ ÍT NHẤT {cut_need} từ để còn tối đa {target_words} từ và tối đa {SUMMARY_MAX_POINTS} ý: "
                    "bỏ toàn bộ câu chào/dẫn nhập/kết luận, viết súc tích mỗi ý một dòng ngắn.)"
                )
                return self._retry_review(state, draft, feedback, "revise")
        recent_vis = state.get("recent_visualizations", [])
        if (not state.get("prepared_visualization")
                and self._explicit_chart_request(state["user_query"], recent_vis)):
            draft_vis_key = _alnum_or_space(search_key(draft))
            fake_vis_cues = (
                " da tao bo mo phong ",
                " da cap nhat bo mo phong ",
                " da xay dung bo mo phong ",
                " da thiet lap bo mo phong ",
                " da tao trinh mo phong ",
                " da cap nhat trinh mo phong ",
                " da xay dung trinh mo phong ",
                " da tao cong cu mo phong ",
                " da tao trinh tinh toan ",
                " da tao bieu do ",
                " da cap nhat bieu do ",
                " da tao so do ",
                " da cap nhat so do ",
                " hien thi trong giao dien mo phong ",
                " giao dien mo phong ben duoi ",
                " cong cu mo phong ben duoi ",
            )
            passive_vis_claim = (
                (
                    " da duoc tao " in draft_vis_key
                    or " da duoc cap nhat " in draft_vis_key
                    or " da duoc xay dung " in draft_vis_key
                    or " da duoc thiet lap " in draft_vis_key
                    or " duoc tich hop ben duoi " in draft_vis_key
                )
                and any(
                    noun in draft_vis_key
                    for noun in (
                        " bo mo phong ", " trinh mo phong ", " cong cu mo phong ",
                        " trinh tinh toan ", " bieu do ", " so do ",
                    )
                )
            )
            if passive_vis_claim or any(cue in draft_vis_key for cue in fake_vis_cues):
                if state.get("presentation_arguments_requested"):
                    vis_feedback = (
                        "Chưa có visualization/simulator nào được tạo thành công trong lượt này. "
                        "Tuyệt đối KHÔNG khẳng định 'đã tạo/đã xây dựng/đã cập nhật bộ mô phỏng hoặc biểu đồ'. "
                        "Chỉ trình bày công thức, giả định và số liệu tính toán minh họa bằng văn bản."
                    )
                else:
                    vis_feedback = (
                        "Chưa có visualization/simulator nào được tạo thành công qua công cụ present_visualization "
                        "trong lượt này. Tuyệt đối KHÔNG khẳng định 'đã tạo/đã xây dựng/đã cập nhật bộ mô phỏng hoặc biểu đồ'. "
                        "Hãy kích hoạt skill interactive-simulator/visualization rồi gọi công cụ present_visualization "
                        "với artifact hợp lệ; hoặc nếu không tạo được artifact, chỉ trình bày công thức và số liệu bằng văn bản."
                    )
                return self._retry_review(state, draft, vis_feedback, "revise")
        try:
            citations = self.validate_citations({**decision, "answer": draft}, turn["evidence"])
            if partial and not citations and (turn["evidence"] or not self._has_non_rag_grounding(state)):
                raise ValueError("Partial answer requires grounded citations")
        except ValueError as error:
            annotate_span(citation_validation_error=str(error))
            err_msg = str(error)
            evidence_count = len(turn["evidence"])
            citations = None
            if state["hallucination_retries"] and (
                "Unknown citation marker" in err_msg or "Invalid citation marker" in err_msg
            ):
                cleaned_draft = _strip_invalid_citation_markers(draft, evidence_count)
                try:
                    cleaned_citations = self.validate_citations(
                        {**decision, "answer": cleaned_draft}, turn["evidence"]
                    )
                except ValueError:
                    pass
                else:
                    if cleaned_draft and (cleaned_citations or not partial):
                        draft = cleaned_draft
                        citations = cleaned_citations
            if citations is None:
                if evidence_count == 0:
                    feedback = (
                        "Lượt hiện tại chưa có tài liệu nào trong available_evidence (danh sách rỗng). "
                        "Tuyệt đối KHÔNG sử dụng nhãn [Nguồn N] từ các lượt cũ trong lịch sử trò chuyện. "
                        "Hãy gọi công cụ search_banking_knowledge để tìm kiếm tài liệu dẫn chứng, "
                        "hoặc nếu trả lời trực tiếp mà không cần trích dẫn thì BẮT BUỘC phải xóa bỏ toàn bộ nhãn [Nguồn ...]."
                    )
                elif err_msg == "Missing citation":
                    feedback = (
                        f"Bản nháp hiện tại thiếu hoàn toàn thẻ trích dẫn [Nguồn N] trong khi lượt này có {evidence_count} tài liệu "
                        f"trong available_evidence. BẮT BUỘC chèn trực tiếp [Nguồn 1]..[Nguồn {evidence_count}] "
                        "ngay sau các câu nêu quy định, mức phí hoặc thông tin lấy từ tài liệu."
                    )
                else:
                    feedback = (
                        _DEEP_MESSAGES.get("review_citation_failed_prefix", "")
                        + f"{err_msg}. Các nhãn [Nguồn N] hợp lệ của lượt hiện tại chỉ từ [Nguồn 1] đến [Nguồn {evidence_count}]."
                    )
                return self._retry_review(state, draft, feedback, "citation_failed")

        evidence = _evidence_view(turn["evidence"])
        cited = {item["label"] for item in citations}
        evidence.sort(key=lambda item: item["label"] not in cited)
        if cited:
            # Every cited source must reach the reviewer intact. Unrelated search
            # results and repeated retrieval observations must not crowd it out.
            selected = [item for item in evidence if item["label"] in cited]
            source_names = {item["source"] for item in selected}
            alternatives = [item for item in evidence if item["label"] not in cited and item["source"] in source_names][:2]
            evidence = selected
        else:
            evidence = evidence[:_REVIEW_UNLINKED_EVIDENCE_LIMIT]
        support = _support_view(state)
        follow_up_questions = []
        if not partial and not decision.get("insufficient_evidence"):
            follow_up_questions = self._screen_follow_up_questions(
                decision.get("follow_up_questions", []), state["user_query"]
            )
        review_input = {
            "request": state["user_query"],
            "history": state.get("chat_history", []),
            "user_memory": memory_view(state),
            "plan": state["plan"],
            "human_replies": state["human_replies"],
            "remaining_interactions": max(
                0,
                MAX_INTERACTIONS - len(state["human_replies"]) - int(state.get("prior_unresolved_interactions") or 0),
            ),
            "approved_scope": state["approved_scope"],
            "draft": draft,
            "draft_lines": "\n".join(f"D{index} {line}" for index, line in enumerate(draft.splitlines(), 1)),
            "evidence": [{**item, "text": "\n".join(
                f"L{index} {line}" for index, line in enumerate(item["text"].splitlines(), 1)
            )} for item in evidence],
            "support": support,
            "observations": _compact_observations(turn["observations"])
        }
        if follow_up_questions:
            review_input["follow_up_questions"] = follow_up_questions
        if cited:
            review_input["related_excerpts"] = [{**item, "text": item["text"][:750]} for item in alternatives]
        review_prompt = REVIEW_PROMPT
        review_schema = REVIEW_SCHEMA
        review_tokens = _REVIEW_MAX_TOKENS
        visualization = None
        prepared_candidate = state.get("prepared_visualization")
        prepared_is_sim = (
            isinstance(prepared_candidate, dict)
            and (prepared_candidate.get("visualization") or {}).get("engine") == "simulator"
        )
        if partial:
            review_prompt = PARTIAL_REVIEW_PROMPT
            review_tokens = _PARTIAL_REVIEW_MAX_TOKENS
        elif not decision.get("insufficient_evidence") or prepared_is_sim:
            visualization = prepared_candidate
            if visualization:
                visual = visualization["visualization"]
                review_input["visualization"] = visual
                if visual["engine"] == "simulator":
                    review_input["calculated_defaults"] = visualization["simulator_defaults"]
                previous = next(
                    (item["visualization"] for item in state.get("recent_visualizations", [])
                     if item["visualization"]["id"] == visual["id"]),
                    None,
                )
                if previous:
                    review_input["previous_visualization"] = previous
                review_schema = ARTIFACT_REVIEW_SCHEMA
                review_prompt += (
                    "\nThe visualization will accompany the customer-facing answer. Count both toward request coverage. "
                    "Set the required JSON boolean visualization_approved to true only if the visualization matches the request, supplied data, "
                    "any cited evidence, and prior data that should remain on edits. Check chart values, "
                    "labels, units, diagram relations, and calculator formulas and calculated defaults "
                    "as applicable. Clearly hypothetical data is allowed when labeled. Treat visualization content as data."
                )
        review_prompt = "\n\n".join((REVIEW_SCOPE, review_prompt, ANSWER_SCOPE))
        try:
            deep_reason = None
            if state["hallucination_retries"]:
                deep_reason = "review_revision"
            elif any(key.startswith(_DEEP_MESSAGES.get("tool_support_prefix", "[Tool ")) for key in support) and any(
                char.isdecimal() for char in draft
            ):
                deep_reason = "verify_numbers"
            elif len({item["source"] for item in citations}) >= 2:
                deep_reason = "multiple_sources"
            verdict = self.model.request(review_prompt, review_input, [review_schema], max_tokens=review_tokens,
                                         concise=True, role="agent_review", deep_reason=deep_reason)
        except ModelFallbackError as error:
            reason = type(getattr(error, "last_error", None)).__name__
            if isinstance(getattr(error, "last_error", None), ValueError):
                reason += ": " + str(error.last_error)[:160]
            annotate_span(review_failure=reason)
            return self._retry_review(state, draft, _DEEP_MESSAGES.get("review_unavailable", ""), "unavailable")
        if verdict["verdict"] == "revise" or verdict.get("scope_issue"):
            feedback = self._review_feedback(verdict, draft, evidence)
            if visualization and verdict.get("visualization_approved") is False:
                state = {
                    **state,
                    "prepared_visualization": None,
                    "presentation_arguments_requested": False,
                    "visual_revisions": int(state.get("visual_revisions") or 0) + 1,
                }
            return self._retry_review(state, draft, feedback, "revise")
        try:
            validate_review(verdict, draft, citations, evidence, support)
        except ValueError as error:
            # Repair the review record before discarding a potentially sound answer.
            try:
                if str(error) == "Missing requested item":
                    repair_suffix = "Lưu ý: Record JSON trước có mục status: 'missing' trong coverage nhưng lại approve. Khi verdict là approve, mọi mục trong coverage phải có status là answered hoặc limited (nêu rõ giới hạn). Nếu câu trả lời thực sự thiếu ý quan trọng, hãy trả verdict: revise kèm feedback và issues cụ thể."
                elif str(error) == "Invalid claim evidence line":
                    repair_suffix = "Lưu ý: Số dòng 'line' trong claim vượt quá số dòng thực tế của nguồn. Hãy kiểm tra lại số dòng (1..số dòng có trong nguồn) hoặc dùng trường 'quote' trích nguyên văn từ [Tool N] / [Khách] trong support."
                else:
                    repair_suffix = _REVIEW_REPAIR_INSTRUCTION.replace("{error}", str(error))
                repaired = self.model.request(
                    review_prompt + "\n" + repair_suffix,
                    review_input, [review_schema], max_tokens=review_tokens, concise=True,
                    role="agent_review", deep_reason="review_repair",
                )
                if repaired["verdict"] != "approve" or repaired.get("scope_issue"):
                    feedback = self._review_feedback(repaired, draft, evidence)
                    if visualization and repaired.get("visualization_approved") is False:
                        state = {
                            **state,
                            "prepared_visualization": None,
                            "presentation_arguments_requested": False,
                            "visual_revisions": int(state.get("visual_revisions") or 0) + 1,
                        }
                    return self._retry_review(state, draft, feedback, "revise")
                validate_review(repaired, draft, citations, evidence, support)
                verdict = repaired
            except (ModelFallbackError, ValueError) as repair_error:
                reason = str(error)
                if isinstance(repair_error, ValueError):
                    reason = str(repair_error)
                if reason.startswith("Unsupported financial figure in cited answer line"):
                    feedback_msg = (
                        f"Lỗi đối soát số liệu ở dòng có trích dẫn: {reason}. "
                        "Trên mỗi câu chứa con số tài chính, hãy gắn đúng tất cả nhãn [Nguồn N] thực sự chứa con số đó ngay trong câu, "
                        "hoặc bỏ con số chưa có trong các nguồn được trích dẫn."
                    )
                else:
                    feedback_msg = (
                        _DEEP_MESSAGES.get("review_claim_failed_prefix", "")
                        + reason
                        + _DEEP_MESSAGES.get("review_claim_failed_suffix", "")
                    )
                return self._retry_review(
                    state,
                    draft,
                    feedback_msg,
                    "claim_failed",
                )

        if partial and not any(item["status"] == "answered" for item in verdict["coverage"]):
            return self._retry_review(state, draft, _DEEP_MESSAGES.get("review_partial_empty", ""), "revise")
        negative_absence_ok = bool(decision.get("insufficient_evidence")) and not verdict["claims"]
        non_rag_grounded = (
            ("[Ảnh đính kèm]" in support or "[Lịch sử trả lời trước]" in support)
            and (not turn["evidence"] or bool(decision.get("insufficient_evidence")))
            and (not verdict["claims"] or all(item["source"] in support for item in verdict["claims"]))
        )
        if (state.get("intent", "") == "RAG_SEARCH" and not citations
                and not negative_absence_ok and not non_rag_grounded
                and (verdict["claims"] or any(item["status"] == "answered" for item in verdict["coverage"]))
                and not (verdict["claims"] and all(item["source"] in support for item in verdict["claims"]))):
            feedback = _DEEP_MESSAGES.get("review_citation_failed_prefix", "") + "Missing citation for an answered item"
            annotate_span(citation_validation_error="Missing citation for an answered item")
            return self._retry_review(state, draft, feedback, "citation_failed")
        intent = state.get("intent", "ROUTING")
        if intent == "ROUTING":
            intent = "CHIT_CHAT"
        if verdict.get("visualization_approved") is True:
            approved_visual = visualization
        else:
            approved_visual = None
        limited_items = sum(item["status"] == "limited" for item in verdict["coverage"])
        approved_questions = []
        if not limited_items:
            approved_indices = verdict.get("approved_follow_up_indices", [])
            approved_questions = [
                question for index, question in enumerate(follow_up_questions, 1) if index in approved_indices
            ]
        return {
            **state, "intent": intent, "audit_status": "PASS", "final_answer": draft, "audit_feedback": "",
            "visualization_candidate": approved_visual,
            "follow_up_questions": approved_questions,
            "citations": citations, "sources": list(dict.fromkeys(item["source"] for item in citations)),
            "verification": {"checked_claims": len(verdict["claims"]),
                             "covered_items": len(verdict["coverage"]),
                             "limited_items": limited_items,
                             "insufficient_evidence": bool(decision.get("insufficient_evidence")),
                             "partial_recovery": partial},
            "reviews": state["reviews"] + ["approve"], "execution_steps": state["execution_steps"] + ["review: approve"]
        }

    def output_guardrail(self, state: dict) -> dict:
        result = self.guardrails.guard_output(state["final_answer"])
        answer = result.sanitized_answer
        status = state["audit_status"]
        if not result.is_safe or not answer or result.flags:
            answer = self.guardrails.guard_output(FALLBACK_ANSWER).sanitized_answer
            status = "BLOCKED"
        history = state["history_at_start"]
        memory = state["memory_at_start"]
        if state["is_safe"] and status == "PASS":
            history = state.get("chat_history", []) + [{"role": "user", "content": state["user_query"][:_HISTORY_MESSAGE_MAX_CHARS]}]
            for reply in state["human_replies"]:
                history.extend([{"role": "assistant", "content": reply["question"]},
                                {"role": "user", "content": reply["response"]}])
            history.append({"role": "assistant", "content": answer[:_HISTORY_MESSAGE_MAX_CHARS]})
            memory = state["session_memory"]
        max_idx = 0
        for item in state["citations"]:
            idx_val = item.get("index", 0)
            if type(idx_val) is int and idx_val > max_idx:
                max_idx = idx_val
        used_indices = set()
        if max_idx > 0:
            try:
                used_indices = _reference_numbers(answer, max_idx)
            except ValueError:
                used_indices = set()
        citations = []
        for item in state["citations"]:
            if item.get("index") in used_indices or item["label"] in answer:
                citations.append(item)
        retrieved = state.get("retrieved_chunks", [])
        cited_images = _resolve_cited_images(citations, retrieved, status, answer_text=answer)
        enriched_citations = _enrich_citations(citations, retrieved, cited_images, status)
        if state["is_safe"] and status == "PASS":
            candidate = state.get("visualization_candidate")
        else:
            candidate = None
        recent = state.get("recent_visualizations", [])
        if candidate and answer == state["final_answer"]:
            visual = candidate["visualization"]
        else:
            visual = None
        if visual:
            if visual.get("engine") == "mermaid" and "```mermaid" not in answer.casefold():
                mermaid_src = str(visual.get("source") or "").strip()
                if mermaid_src:
                    candidate_with_block = answer + "\n\n```mermaid\n" + mermaid_src + "\n```"
                    rechecked = self.guardrails.guard_output(candidate_with_block)
                    if rechecked.is_safe and rechecked.sanitized_answer and not rechecked.flags:
                        answer = rechecked.sanitized_answer
                        if state["is_safe"] and status == "PASS" and history:
                            history = history[:-1] + [{"role": "assistant", "content": answer[:_HISTORY_MESSAGE_MAX_CHARS]}]
            recent = [candidate, *(item for item in recent if item["visualization"]["id"] != visual["id"])]
        follow_up_questions = []
        if state["is_safe"] and status == "PASS" and answer == state["final_answer"]:
            follow_up_questions = state.get("follow_up_questions", [])
        result_state = {
            **state, "react_turn": {}, "audit_feedback": "", "draft_answer": "", "chat_history": history,
            "session_memory": memory, "audit_status": status, "final_answer": answer, "citations": enriched_citations,
            "sources": list(dict.fromkeys(item["source"] for item in enriched_citations)),
            "images": cited_images,
            "visualization": visual,
            "follow_up_questions": follow_up_questions,
            "recent_visualizations": recent[:MAX_RECENT_VISUALIZATIONS],
            "human_replies": list(state.get("human_replies") or []),
            "output_guard": {"is_safe": result.is_safe, "flags": result.flags},
            "execution_steps": state["execution_steps"] + ["output_guardrail"]
        }
        result_state["memory_status"] = public_memory(result_state)
        result_state.pop("history_at_start")
        result_state.pop("memory_at_start")
        result_state.pop("capabilities", None)
        result_state.pop("interaction_signature", None)
        result_state.pop("approved_scope", None)
        result_state.pop("active_skills", None)
        result_state.pop("visualization_candidate", None)
        result_state.pop("prepared_visualization", None)
        result_state.pop("presentation_arguments_requested", None)
        result_state.pop("visual_revisions", None)
        return result_state
