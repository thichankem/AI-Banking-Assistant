"""Banking agent harness with planning, isolated workers and independent review."""

import json
from unicodedata import normalize
from urllib.parse import quote
from uuid import uuid4

from langgraph.types import interrupt

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
from agent_banking.agent.model_protocol import REVIEW_SCHEMA, StructuredModel
from agent_banking.agent.react import (
    BankingReActNodes,
    FALLBACK_ANSWER,
    MAX_EVIDENCE,
    _normalize_citation_markers,
    _reference_numbers,
)
from agent_banking.core.fallback import ModelFallbackError, is_temporary_error
from agent_banking.core.search_text import search_key
from agent_banking.skills.skill_registry import resolve_skill
from agent_banking.tools.extensions.banking_query import parse_banking_query
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
_PARTIAL_RECOVERY_MAX_SOURCES = get_int_setting("partial_recovery_max_sources")
_DECIDE_MAX_TOKENS = get_int_setting("decide_max_tokens")
_RECOVER_MAX_TOKENS = get_int_setting("recover_max_tokens")
_WORKER_MAX_TOKENS = get_int_setting("worker_max_tokens")
_PARTIAL_RECOVERY_MAX_TOKENS = get_int_setting("partial_recovery_max_tokens")
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


def _evidence_view(evidence: list[dict]) -> list[dict]:
    result = []
    for item in evidence:
        chunk = item["chunk"]
        record = {"label": _label(item["index"]), "text": item["context"], "source": item["source"],
                  "version": (chunk.get("business_metadata") or {}).get("document_version", "")}
        if type(chunk.get("source_offset")) is int:
            record["source_offset"] = chunk["source_offset"]
        if chunk.get("context_for"):
            record["context_neighbor"] = True
        result.append(record)
    return result


def _support_view(state: dict) -> dict:
    turn = state["react_turn"]
    user_text = [
        item["content"] for item in state.get("chat_history", [])[-_SUPPORT_HISTORY_LIMIT:]
        if item["role"] == "user"
    ]
    user_text += [state["user_query"]] + [reply["response"] for reply in state["human_replies"]]
    notes = [item["value"] for item in memory_view(state)["notes"].values()]
    user_text = notes[:_SUPPORT_NOTES_LIMIT] + user_text
    user_label = _DEEP_MESSAGES.get("user_support_label", "[Khách]")
    tool_prefix = _DEEP_MESSAGES.get("tool_support_prefix", "[Tool ")
    result = {user_label: "\n".join(text[:_SUPPORT_MESSAGE_MAX_CHARS] for text in user_text)[-_SUPPORT_USER_TEXT_MAX_CHARS:]}

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
            if folded_q not in src_folded and folded_q not in full_folded and compact_q not in compact_full:
                words_match = False
                q_words = compact_q.split()
                if len(q_words) >= 4:
                    half_len = max(2, len(q_words) // 2)
                    first_half = " ".join(q_words[:half_len])
                    second_half = " ".join(q_words[half_len:])
                    if first_half in compact_full or second_half in compact_full:
                        words_match = True
                if not words_match and len(q_words) >= 3:
                    full_tokens = set(full_folded.split()) | set(compact_full.split())
                    numeric_q = [tok for tok in q_words if tok.isdigit()]
                    if all(tok in full_tokens for tok in numeric_q):
                        matched_count = 0
                        for tok in q_words:
                            if tok in full_tokens:
                                matched_count += 1
                        if matched_count * 5 >= len(q_words) * 4:
                            words_match = True
                if not words_match:
                    raise ValueError("Invalid claim evidence excerpt")
        claimed.add(source)
    if verdict.get("verdict", "approve") == "approve":
        missing = [item["label"] for item in citations if item["label"] not in claimed]
        if missing and not claimed:
            raise ValueError("Citation missing claim grounding: " + ", ".join(missing))


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


def _requests_restricted_field(text: str) -> bool:
    normalized = _alnum_or_space(text.casefold().replace(".", ""))
    return any(f" {field} " in normalized for field in RESTRICTED_CARD_FIELDS)


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

    return {
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


def _line_contains_citation(line: str, label: str, cit_index) -> bool:
    if not line:
        return False
    if label and label in line:
        return True
    if type(cit_index) is not int or cit_index <= 0:
        return False
    target = str(cit_index)
    cursor = 0
    while cursor < len(line):
        open_idx = line.find("[", cursor)
        if open_idx < 0:
            break
        close_idx = line.find("]", open_idx + 1)
        if close_idx < 0:
            break
        inner = line[open_idx + 1:close_idx]
        lower_inner = inner.lower()
        if "nguồn" in lower_inner or "nguon" in lower_inner:
            normalized = inner.replace(";", ",")
            for part in normalized.split(","):
                token = part.strip()
                lower_token = token.lower()
                if lower_token.startswith("nguồn"):
                    token = token[5:].strip()
                elif lower_token.startswith("nguon"):
                    token = token[5:].strip()
                if " " in token:
                    token = token.split(" ")[0].strip()
                if token == target:
                    return True
        cursor = close_idx + 1
    return False


def _citation_answer_stats(citation: dict, answer_text: str) -> tuple:
    if not answer_text:
        return (999999999, False)
    label = str(citation.get("label") or "")
    cit_index = citation.get("index")
    first_pos = 999999999
    matched_lines = []
    offset = 0
    for line in answer_text.splitlines():
        if _line_contains_citation(line, label, cit_index):
            if first_pos == 999999999:
                first_pos = offset
            matched_lines.append(line)
        offset += len(line) + 1

    if not matched_lines:
        return (first_pos, False)

    all_support_only = True
    for line in matched_lines:
        line_key = " " + search_key(line) + " "
        has_support = False
        for marker in _SUPPORT_LINE_MARKERS:
            if marker in line_key:
                has_support = True
                break
        has_action = False
        for marker in _ACTION_STEP_MARKERS:
            if marker in line_key:
                has_action = True
                break
        if not has_support or has_action:
            all_support_only = False
            break

    return (first_pos, all_support_only)


def _resolve_cited_images(
    citations: list,
    retrieved_chunks: list,
    status: str,
    max_images: int | None = None,
    answer_text: str = "",
) -> list:
    """Collect deduplicated visual assets from cited chunks in canonical step/page order."""
    if status == "BLOCKED" or not citations or not retrieved_chunks:
        return []
    by_chunk_id = {}
    by_source_section = {}
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

    cit_records = []
    for cit_pos, citation in enumerate(citations):
        if not isinstance(citation, dict):
            continue
        cid = str(citation.get("chunk_id") or "")
        src = str(citation.get("source") or "")
        sec = str(citation.get("section_title") or "")
        chunk = by_chunk_id.get(cid)
        if chunk is None and src:
            chunk = by_source_section.get((src, sec))
        if chunk is None:
            continue
        raw_images = chunk.get("images")
        if not isinstance(raw_images, list) or not raw_images:
            continue

        meta = chunk.get("business_metadata")
        if not isinstance(meta, dict):
            meta = {}
        effective_src = str(src or chunk.get("source") or "")
        doc_name = _extract_document_name(effective_src, meta)
        doc_key = str(effective_src or doc_name).replace("\\", "/").strip().lower()
        first_pos, is_support_only = _citation_answer_stats(citation, answer_text)

        cit_idx_raw = citation.get("index")
        if type(cit_idx_raw) is int:
            cit_idx = cit_idx_raw
        else:
            cit_idx = cit_pos + 1

        valid_items = []
        has_app_visual = False
        app_pages = []
        seen_app_pages = set()
        for raw_img in raw_images:
            entry = _normalize_image_entry(raw_img, citation, doc_name)
            if entry is None:
                continue
            order_key = _image_order_key(raw_img)
            is_direct = raw_img.get("match_strength") == "direct"
            valid_items.append((entry, order_key, is_direct))
            if entry["region_type"] in _APP_VISUAL_REGION_TYPES:
                has_app_visual = True
                p_num = entry.get("page_number")
                if type(p_num) is int and p_num > 0 and p_num not in seen_app_pages:
                    seen_app_pages.add(p_num)
                    app_pages.append(p_num)

        if not valid_items:
            continue

        cit_records.append({
            "cit_idx": cit_idx,
            "doc_key": doc_key,
            "first_pos": first_pos,
            "is_support_only": is_support_only,
            "has_app_visual": has_app_visual,
            "app_pages": app_pages,
            "chunk_img_count": len(valid_items),
            "items": valid_items,
            "drop_app": False,
        })

    if not cit_records:
        return []

    # Filter A: ignore app/slide visuals attached only to Hotline/CSKH support citations
    has_non_support_app = False
    for rec in cit_records:
        if rec["has_app_visual"] and not rec["is_support_only"]:
            has_non_support_app = True
            break
    if has_non_support_app:
        for rec in cit_records:
            if rec["has_app_visual"] and rec["is_support_only"]:
                rec["drop_app"] = True

    # Filter B: drop isolated cross-document HDSD stray when a dominant HDSD doc has >=60% & >=2 citations
    app_doc_counts = {}
    total_app_cits = 0
    for rec in cit_records:
        if rec["has_app_visual"] and not rec["drop_app"]:
            dk = rec["doc_key"]
            app_doc_counts[dk] = app_doc_counts.get(dk, 0) + 1
            total_app_cits += 1

    has_dominant_app_doc = False
    for dk, cnt in app_doc_counts.items():
        if cnt >= 2 and cnt * 5 >= total_app_cits * 3:
            has_dominant_app_doc = True
            break
    if has_dominant_app_doc:
        for rec in cit_records:
            if rec["has_app_visual"] and not rec["drop_app"]:
                if app_doc_counts.get(rec["doc_key"], 0) == 1:
                    rec["drop_app"] = True

    # Filter C: within an HDSD doc (>=4 app citations), drop isolated distant whole-page dumps
    by_app_doc = {}
    for rec in cit_records:
        if rec["has_app_visual"] and not rec["drop_app"]:
            by_app_doc.setdefault(rec["doc_key"], []).append(rec)

    for dk, doc_recs in by_app_doc.items():
        if len(doc_recs) < 4:
            continue
        page_cit_counts = {}
        for rec in doc_recs:
            for p_num in rec["app_pages"]:
                page_cit_counts[p_num] = page_cit_counts.get(p_num, 0) + 1

        to_drop = []
        for rec in doc_recs:
            if rec["chunk_img_count"] < 3 or len(rec["app_pages"]) != 1:
                continue
            p_num = rec["app_pages"][0]
            if page_cit_counts.get(p_num, 0) != 1:
                continue
            is_distant = True
            has_other_page = False
            for other_p in page_cit_counts:
                if other_p == p_num:
                    continue
                has_other_page = True
                if abs(p_num - other_p) < 2:
                    is_distant = False
                    break
            if has_other_page and is_distant:
                to_drop.append(rec)

        if to_drop and len(to_drop) < len(doc_recs):
            for rec in to_drop:
                rec["drop_app"] = True

    # Deduplicate by asset_id preferring direct match and most specific (fewest images) chunk
    best_by_asset = {}
    for rec in cit_records:
        drop_app = rec["drop_app"]
        for entry, order_key, is_direct in rec["items"]:
            if drop_app and entry["region_type"] in _APP_VISUAL_REGION_TYPES:
                continue
            asset_id = entry["asset_id"]
            direct_rank = 1
            if is_direct:
                direct_rank = 0
            spec_key = (direct_rank, rec["chunk_img_count"], rec["first_pos"], rec["cit_idx"])
            candidate = (spec_key, rec["doc_key"], order_key, rec["first_pos"], rec["cit_idx"], entry)
            existing = best_by_asset.get(asset_id)
            if existing is None or spec_key < existing[0]:
                best_by_asset[asset_id] = candidate

    surviving = list(best_by_asset.values())
    if not surviving:
        return []

    has_any_app_visual = False
    doc_stats = {}
    for item in surviving:
        doc_key = item[1]
        first_pos = item[3]
        cit_idx = item[4]
        entry = item[5]
        is_app = entry["region_type"] in _APP_VISUAL_REGION_TYPES
        if is_app:
            has_any_app_visual = True
        stat = doc_stats.get(doc_key)
        if stat is None:
            doc_stats[doc_key] = {
                "has_app": is_app,
                "cit_indices": {cit_idx},
                "min_first_pos": first_pos,
                "min_cit_idx": cit_idx,
            }
        else:
            if is_app:
                stat["has_app"] = True
            stat["cit_indices"].add(cit_idx)
            if first_pos < stat["min_first_pos"]:
                stat["min_first_pos"] = first_pos
            if cit_idx < stat["min_cit_idx"]:
                stat["min_cit_idx"] = cit_idx

    doc_keys = list(doc_stats.keys())
    if has_any_app_visual:
        def _app_doc_sort_key(dk):
            st = doc_stats[dk]
            app_pri = 1
            if st["has_app"]:
                app_pri = 0
            return (app_pri, -len(st["cit_indices"]), st["min_first_pos"], st["min_cit_idx"])
        doc_keys.sort(key=_app_doc_sort_key)
    else:
        def _non_app_doc_sort_key(dk):
            st = doc_stats[dk]
            return (st["min_first_pos"], st["min_cit_idx"])
        doc_keys.sort(key=_non_app_doc_sort_key)

    doc_rank = {}
    for rank_idx, dk in enumerate(doc_keys):
        doc_rank[dk] = rank_idx

    surviving.sort(
        key=lambda item: (
            doc_rank[item[1]],
            item[2][0],
            item[2][1],
            item[2][2],
            item[2][3],
            item[3],
            item[4],
        )
    )

    if max_images is not None:
        limit = max_images
    elif has_any_app_visual:
        limit = 16
    else:
        limit = 8

    if limit <= 0:
        return []
    return [item[5] for item in surviving[:limit]]


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


def _build_pdf_url(source: str, page_numbers: list) -> str:
    """Build browser-ready inline PDF URL with #page=N anchor for PDF sources."""
    clean = str(source or "").replace("\\", "/").strip()
    while clean.startswith("/"):
        clean = clean[1:]
    if not clean or not clean.lower().endswith(".pdf"):
        return ""
    url = "/api/documents/pdf?source=" + quote(clean, safe="")
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
        enriched_item["page_numbers"] = pages
        excel_citation = meta.get("citation")
        if isinstance(excel_citation, dict) and excel_citation.get("sheet_name"):
            enriched_item["sheet_name"] = excel_citation["sheet_name"]
            enriched_item["cell_ranges"] = excel_citation.get("cell_ranges", [])
        enriched_item["pdf_url"] = _build_pdf_url(src, pages)
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
                       approved_scope=None, confirmation_required=False)
        guarded["session_memory"] = memory_view(guarded)
        guarded["memory_at_start"] = guarded["session_memory"]
        guarded["history_at_start"] = guarded.get("chat_history", [])
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
            "remaining_interactions": max(0, MAX_INTERACTIONS - len(state["human_replies"])),
            "human_replies": state["human_replies"],
            "approved_scope": state["approved_scope"],
            "confirmation_required": state["confirmation_required"],
        }

    def decide(self, state: dict) -> dict:
        data = self._decide_input(state)
        allowed = ["final"]
        system = BASE_PROMPT
        if data["remaining_actions"]:
            allowed += ["plan", "remember", "forget"]
            if not state["confirmation_required"]:
                allowed.append("tool")
                if state["capabilities"]["skill"]:
                    allowed += ["skill", "inspect_skill"]
                if data["remaining_delegations"]:
                    allowed.append("delegate")
            if data["remaining_interactions"]:
                allowed += ["clarify", "confirm"]
            if state["compactions_this_turn"] < MAX_COMPACT_PER_TURN and len(data["history"]) > RECENT_MESSAGES:
                allowed.append("compact")
            system += "\n" + COORDINATOR_PROMPT + _DEEP_MESSAGES.get("catalog_header", "\n") + json.dumps(
                self.catalog_view(state["capabilities"]), ensure_ascii=False
            )
            system += _DEEP_MESSAGES.get("subagents_header", "\n") + json.dumps(
                {name: role["description"] for name, role in WORKERS.items()}, ensure_ascii=False
            )
            if _COORDINATOR_SKILL_HINT:
                system += "\n" + _COORDINATOR_SKILL_HINT
        persona = PERSONA_PROFILES.get(state.get("persona"), PERSONA_PROFILES.get(_DEFAULT_PERSONA, {}))
        tone = TONE_PROFILES.get(state.get("tone"), TONE_PROFILES.get(_DEFAULT_TONE, ""))
        format_style = FORMAT_PROFILES.get(state.get("format_type"), FORMAT_PROFILES.get(_DEFAULT_FORMAT_TYPE, ""))
        persona_role_text = persona.get("role_definition")
        if not persona_role_text:
            persona_role_text = persona.get("description", "")
        custom_persona_text = str(state.get("custom_persona") or "").strip()
        if state.get("persona") == "custom" and custom_persona_text:
            persona_role_text = persona_role_text + "\nChỉ thị vai trò tùy chỉnh từ khách hàng: " + custom_persona_text
        system += _DEEP_MESSAGES.get("style_header", "\n") + "\n".join(
            [persona_role_text, tone, format_style]
        )
        system += _DEEP_MESSAGES.get("allowed_kinds_header", "\n") + ", ".join(allowed)
        if state.get("persona") == "custom":
            data["style_preference"] = custom_persona_text
        if state.get("audit_feedback"):
            data["review_feedback"] = state["audit_feedback"]
            data["previous_draft"] = state.get("draft_answer", "")
            if _REVISION_INSTRUCTION:
                system += "\n" + _REVISION_INSTRUCTION

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
            if item.get("kind") == "tool":
                if item.get("name") == "search_banking_knowledge":
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

        if num_searches >= 3 or (num_searches >= 2 and zero_evidence_streak >= 2):
            system += "\nLƯU Ý: Đã thực hiện tra cứu tài liệu đủ số lượt. Hãy tổng hợp toàn bộ căn cứ và trích dẫn hiện có để đưa ra câu trả lời cuối cùng bằng action final ngay bây giờ. Không tiếp tục tra cứu thêm."

        try:
            decision = self.model.decide(system, data, allowed, max_tokens=_DECIDE_MAX_TOKENS, deep_reason=deep_reason)
        except ModelFallbackError as error:
            category = getattr(error, "category", "model_unavailable")
            if isinstance(getattr(error, "last_error", None), (ValueError, TypeError)):
                category = "model_contract"
            elif is_temporary_error(error):
                category = "provider_temporary"
            decision = self.recover_decision(data, category, state["confirmation_required"])
            if decision is None:
                return {
                    **state, "audit_status": "ERROR", "final_answer": FALLBACK_ANSWER,
                    "error_category": category,
                    "execution_steps": state["execution_steps"] + ["deep_decide: model_error"]
                }
            state = {**state, "execution_steps": state["execution_steps"] + ["deep_decide: recovered_contract"]}
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
        if decision.get("kind") == "tool":
            if decision.get("name") == "search_banking_knowledge":
                if num_searches >= 4 or (num_searches >= 2 and zero_evidence_streak >= 2):
                    recovered = self.recover_decision(data, "search_cap", state["confirmation_required"])
                    if recovered is not None:
                        decision = recovered
        return {
            **state, "react_turn": {**state["react_turn"], "decision": decision},
            "execution_steps": state["execution_steps"] + ["deep_decide: " + decision["kind"]]
        }

    def recover_decision(self, data: dict, category: str, confirmation_required: bool) -> dict | None:
        if confirmation_required:
            return None
        has_success = False
        for item in data.get("observations", []):
            if item.get("kind") in ("tool", "skill", "delegate"):
                if item.get("status") == "SUCCESS":
                    if "data" in item:
                        has_success = True
                        break
        if not has_success:
            return None
        instruction = BASE_PROMPT + "\n" + _RECOVER_PROMPT
        try:
            return self.model.decide(instruction, data, ["final"], max_tokens=_RECOVER_MAX_TOKENS)
        except ModelFallbackError:
            pass
        evidence = data.get("available_evidence", [])
        if evidence:
            primary_text = str(evidence[0].get("text") or "").strip()
            if primary_text:
                label = evidence[0].get("label", "[Nguồn 1]")
                clean_lines = []
                for line in primary_text.splitlines():
                    s_line = line.strip()
                    if s_line:
                        if not s_line.startswith("["):
                            clean_lines.append(s_line)
                snippet = " ".join(clean_lines[:4])
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                ans = "Dựa trên tài liệu tra cứu (" + label + "): " + snippet
                return {"kind": "final", "answer": ans}
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
            if spec and "revision" in spec and spec["skill_id"] not in state.get("active_skills", {}):
                return self.inspect_skill(state)
        return super().act(state)

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
        if len(state["human_replies"]) >= MAX_INTERACTIONS or any(
            reply["signature"] == signature for reply in state["human_replies"]
        ):
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
            "kind": "memory", "name": kind, "status": status, "error": error
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

    def _merge_worker_evidence(self, parent: list, child: list) -> tuple[list, dict]:
        merged = list(parent)
        known = {tuple(item["identity"]): item["index"] for item in merged}
        mapping = {}
        for item in child:
            key = tuple(item["identity"])
            if key not in known:
                if len(merged) >= MAX_EVIDENCE:
                    raise ValueError("Turn evidence limit reached")
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
                        merged, mapping = self._merge_worker_evidence(turn["evidence"], result["evidence"])
                        norm_report = _normalize_citation_markers(result["report"])
                        parts = norm_report.split("[Nguồn ")
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
                        docs = [merged[index - 1] for index in dict.fromkeys(mapping.values())]
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
        labels = {item["label"] for item in evidence}
        line_count = len(draft.splitlines())
        items = []
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

    def review(self, state: dict) -> dict:
        result = self._review_draft(state)
        if (result["audit_status"] != "FAIL" or state.get("confirmation_required")
                or result["reviews"][-1] not in ("revise", "claim_failed", "citation_failed")
                or not state["react_turn"]["evidence"] or not result.get("draft_answer")):
            return result

        # A single final-only attempt after ordinary revisions are used up.
        # It cannot call tools, and its new answer must pass the normal reviewer.
        data = self._decide_input(result)
        data["review_feedback"] = result["audit_feedback"]
        data["previous_draft"] = result["draft_answer"]
        instruction = BASE_PROMPT + "\n" + _PARTIAL_RECOVERY_PROMPT
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

    def _review_draft(self, state: dict, *, partial=False) -> dict:
        turn = state["react_turn"]
        decision = turn["decision"]
        raw = self.guard_artifact(decision["answer"])
        output = self.guardrails.guard_output(raw)
        draft = output.sanitized_answer
        if not output.is_safe or not draft or output.flags:
            return self._retry_review(state, "", _DEEP_MESSAGES.get("review_draft_blocked", ""), "blocked")
        try:
            citations = self.validate_citations({**decision, "answer": draft}, turn["evidence"])
            if partial and not citations:
                raise ValueError("Partial answer requires grounded citations")
            if partial and len({item["source"] for item in citations}) > _PARTIAL_RECOVERY_MAX_SOURCES:
                raise ValueError("Partial answer must focus on at most three documents")
        except ValueError as error:
            annotate_span(citation_validation_error=str(error))
            return self._retry_review(
                state, draft, _DEEP_MESSAGES.get("review_citation_failed_prefix", "") + str(error), "citation_failed"
            )

        evidence = _evidence_view(turn["evidence"])
        cited = {item["label"] for item in citations}
        evidence.sort(key=lambda item: item["label"] not in cited)
        if cited:
            # Every cited source must reach the reviewer intact. Unrelated search
            # results and repeated retrieval observations must not crowd it out.
            evidence = [item for item in evidence if item["label"] in cited]
        else:
            evidence = evidence[:_REVIEW_UNLINKED_EVIDENCE_LIMIT]
        support = _support_view(state)
        review_input = {
            "request": state["user_query"],
            "history": state.get("chat_history", []),
            "user_memory": memory_view(state),
            "plan": state["plan"],
            "human_replies": state["human_replies"],
            "approved_scope": state["approved_scope"],
            "draft": draft,
            "draft_lines": "\n".join(f"D{index} {line}" for index, line in enumerate(draft.splitlines(), 1)),
            "evidence": [{**item, "text": "\n".join(
                f"L{index} {line}" for index, line in enumerate(item["text"].splitlines(), 1)
            )} for item in evidence],
            "support": support,
            "observations": _compact_observations(turn["observations"])
        }
        review_prompt = REVIEW_PROMPT
        review_tokens = _REVIEW_MAX_TOKENS
        if partial:
            review_prompt = PARTIAL_REVIEW_PROMPT
            review_tokens = _PARTIAL_REVIEW_MAX_TOKENS
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
            verdict = self.model.request(review_prompt, review_input, [REVIEW_SCHEMA], max_tokens=review_tokens,
                                         concise=True, role="agent_review", deep_reason=deep_reason)
        except ModelFallbackError as error:
            reason = type(getattr(error, "last_error", None)).__name__
            if isinstance(getattr(error, "last_error", None), ValueError):
                reason += ": " + str(error.last_error)[:160]
            annotate_span(review_failure=reason)
            return self._retry_review(state, draft, _DEEP_MESSAGES.get("review_unavailable", ""), "unavailable")
        if verdict["verdict"] == "revise":
            feedback = self._review_feedback(verdict, draft, evidence)
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
                    review_input, [REVIEW_SCHEMA], max_tokens=review_tokens, concise=True,
                    role="agent_review", deep_reason="review_repair",
                )
                if repaired["verdict"] != "approve":
                    feedback = self._review_feedback(repaired, draft, evidence)
                    return self._retry_review(state, draft, feedback, "revise")
                validate_review(repaired, draft, citations, evidence, support)
                verdict = repaired
            except (ModelFallbackError, ValueError) as repair_error:
                reason = str(error)
                if isinstance(repair_error, ValueError):
                    reason = str(repair_error)
                return self._retry_review(
                    state,
                    draft,
                    _DEEP_MESSAGES.get("review_claim_failed_prefix", "")
                    + reason
                    + _DEEP_MESSAGES.get("review_claim_failed_suffix", ""),
                    "claim_failed",
                )

        if partial and not any(item["status"] == "answered" for item in verdict["coverage"]):
            return self._retry_review(state, draft, _DEEP_MESSAGES.get("review_partial_empty", ""), "revise")
        intent = state["intent"]
        if intent == "ROUTING":
            intent = "CHIT_CHAT"
        return {
            **state, "intent": intent, "audit_status": "PASS", "final_answer": draft, "audit_feedback": "",
            "citations": citations, "sources": list(dict.fromkeys(item["source"] for item in citations)),
            "verification": {"checked_claims": len(verdict["claims"]),
                             "covered_items": len(verdict["coverage"]),
                             "limited_items": sum(item["status"] == "limited" for item in verdict["coverage"]),
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
        result_state = {
            **state, "react_turn": {}, "audit_feedback": "", "draft_answer": "", "chat_history": history,
            "session_memory": memory, "audit_status": status, "final_answer": answer, "citations": enriched_citations,
            "sources": list(dict.fromkeys(item["source"] for item in enriched_citations)),
            "images": cited_images,
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
        return result_state
