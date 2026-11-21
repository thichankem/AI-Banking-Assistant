"""Cached structural decisions for document chunking."""
from copy import deepcopy
from functools import partial
import hashlib
import json
import math
from pathlib import Path

from agent_banking.chunking.config.prompt_config import (
    BREADCRUMB_HIERARCHY_PROMPT,
    DECISION_ENGINE_VERSION,
    PERIOD_BOUNDARIES_PROMPT,
    PROMPT_PAIRS_PROMPT,
    render_prompt,
)
from agent_banking.chunking.config.settings import get_config_section
from agent_banking.core.models import auto_routing_enabled, chain_fingerprint, model_primary
from agent_banking.tracing.tracer import annotate_span, trace_stage
from ...extraction.docx.extractor import MAX_HEADING_LEVEL
from .images import attach_visual_captions, resolve_image_titles_and_context
from .tables import (
    compatible_table_headers as compatible_table_headers,
    merge_continued_tables,
    record_header_occurrence as record_header_occurrence,
    resolve_table_structures_and_titles,
    table_header_cells as table_header_cells,
    table_row_provenance as table_row_provenance,
)
from ..model_warnings import optional_model_call, record_model_warning
from ...text.language import classify_text_blocks
from ...models.text_client import cloud_json, ordered_parallel, validate_decisions
from ...models.response_cache import PageVisionCache

_DECISIONS_CFG = get_config_section("enrichment")["decisions"]
_CACHE_CFG = get_config_section("cache")
_BREADCRUMB_CFG = _DECISIONS_CFG["breadcrumb"]
_PAIR_CFG = _DECISIONS_CFG["prompt_pairs"]
_PERIOD_CFG = _DECISIONS_CFG["period_boundaries"]
_DEFAULT_CACHE = None


def get_decision_cache(cache=None):
    global _DEFAULT_CACHE
    if cache is not None:
        return cache
    if _DEFAULT_CACHE is None:
        db_path = Path(__file__).resolve().parents[3] / get_config_section("pipeline")["cache_db_relpath"]
        _DEFAULT_CACHE = PageVisionCache(str(db_path))
    return _DEFAULT_CACHE


def ask_decisions(role, state, questions, cache=None, batch_size=_DECISIONS_CFG["batch_size"],
                  workers=_DECISIONS_CFG["workers"], *, ir=None):
    """Execute and cache batched /systemone decision queries."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Decision batch size must be a positive integer")
    if type(workers) is not int or workers < 1:
        raise ValueError("Decision worker count must be a positive integer")
    if not questions:
        return {}
    active_cache = get_decision_cache(cache)
    endpoint = optional_model_call(ir, role, lambda: model_primary("chunking.decision"))
    if endpoint is None:
        return {}
    model = endpoint.model
    routing = {}
    if auto_routing_enabled():
        reason = _DECISIONS_CFG["route_reasons"]
        routing["route_reason"] = reason.get(role, _DECISIONS_CFG["default_route_reason"])
    namespace = {
        "engine": DECISION_ENGINE_VERSION,
        "decisions": chain_fingerprint("chunking.decision"),
        "provider": endpoint.url,
        **routing,
    }
    q_items = list(questions.items())
    batches = []
    for start in range(0, len(q_items), batch_size):
        batches.append(dict(q_items[start:start + batch_size]))

    @trace_stage("document.decision")
    def run_batch(batch_questions):
        annotate_span(role=role, questions=len(batch_questions))
        batch_state = {}
        def relevant(identifier, selected):
            return isinstance(identifier, str) and any(
                key == identifier or key.startswith(identifier + "_") for key in selected
            )
        for name, value in state.items():
            if (isinstance(value, list) and value and all(isinstance(item, dict) and isinstance(item.get("id"), str)
                                                          for item in value)):
                value = [item for item in value if relevant(item["id"], batch_questions)]
            batch_state[name] = value
        payload_seed = {
            "version": DECISION_ENGINE_VERSION,
            "namespace": namespace,
            "role": role,
            "model": model,
            "state": batch_state,
            "questions": batch_questions,
        }
        key = hashlib.sha256(
            json.dumps(payload_seed, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cached = active_cache.get(key)
        if cached is not None:
            try:
                validate_decisions(cached, batch_questions)
                annotate_span(cache_hit=True)
                return answers_with_provenance(cached, batch_questions)
            except (ValueError, TypeError, KeyError):
                active_cache.delete(key)
        def request():
            result = cloud_json(
                endpoint.url.rstrip("/") + "/systemone",
                endpoint.api_key,
                {"model": model, "state": batch_state, "questions": batch_questions},
                timeout=_DECISIONS_CFG["timeout"],
                model_module="chunking.decision",
                **routing,
            )
            validate_decisions(result, batch_questions)
            return result

        result = optional_model_call(ir, role, request)
        if result is None:
            return {}
        active_cache.put(key, result)
        return answers_with_provenance(result, batch_questions)

    def answers_with_provenance(result, batch_questions):
        answers = {name: deepcopy(result["answers"][name]) for name in batch_questions}
        provenance = result.get("_routing")
        if provenance:
            annotate_span(**provenance)
            for answer in answers.values():
                answer["_routing"] = deepcopy(provenance)
            if ir is not None:
                ir.setdefault("decision_model_routes", []).append({"role": role, **provenance})
        return answers

    if len(batches) == 1:
        return run_batch(batches[0])
    batch_results = ordered_parallel(run_batch, batches, min(_DECISIONS_CFG["max_parallel_workers"], workers, len(batches)))
    merged = {}
    for ans_dict in batch_results:
        merged.update(ans_dict)
    return merged


def decide_breadcrumb_hierarchy(ir, cache=None):
    """Infer heading levels, captions and body text from document evidence."""
    classify_text_blocks(ir["blocks"], search_language=ir.get("search_language", "vi"))
    body = [
        block for block in ir["blocks"]
        if block.get("provenance", {}).get("region", "body") == "body"
        and not block.get("foreign_language")
    ]
    if not body:
        return
    char_by_size = {}
    for block in body:
        typo = block.get("typography")
        if typo is not None:
            for sz, count in typo.get("size_chars", {}).items():
                char_by_size[sz] = char_by_size.get(sz, 0) + count
    has_typography = bool(char_by_size)
    body_font_pt = _BREADCRUMB_CFG["default_body_font_pt"]
    max_chars = -1
    for sz, count in char_by_size.items():
        if count > max_chars:
            max_chars = count
            body_font_pt = sz
    if has_typography:
        ir["typography_profile"] = {
            "body_font_pt": body_font_pt,
            "char_distribution": {str(k): v for k, v in sorted(char_by_size.items())},
        }
    else:
        ir["typography_profile"] = {"body_font_pt": None, "char_distribution": {}}
    candidates = []
    questions = {}
    candidate_blocks = {}
    level_map = {"LEVEL_" + str(level): level for level in range(1, MAX_HEADING_LEVEL + 1)}
    criteria = {
        "LEVEL_1": BREADCRUMB_HIERARCHY_PROMPT["level_1_description"],
        **{key: render_prompt(BREADCRUMB_HIERARCHY_PROMPT["level_n_description"], level=level)
           for key, level in level_map.items() if level > 1},
        "CAPTION": BREADCRUMB_HIERARCHY_PROMPT["caption_description"],
        "BODY": BREADCRUMB_HIERARCHY_PROMPT["body_description"],
    }
    max_heading_words = _BREADCRUMB_CFG["max_heading_words"]
    max_heading_lines = _BREADCRUMB_CFG["max_heading_lines"]
    bold_label_words = _BREADCRUMB_CFG["starts_bold_label_max_words"]
    text_preview_chars = _BREADCRUMB_CFG["text_preview_chars"]
    upper_num = _BREADCRUMB_CFG["upper_ratio_num"]
    upper_den = _BREADCRUMB_CFG["upper_ratio_den"]
    for idx, block in enumerate(body):
        if block["type"] not in ("paragraph", "heading", "caption"):
            continue
        if block.get("authored_level") or block.get("heading_format") == "openxml_outline":
            continue
        text = block.get("text", "").strip()
        if not text or len(text.split()) > max_heading_words:
            continue
        typo = block.get("typography")
        if typo is not None:
            if len(text.splitlines()) > max_heading_lines:
                continue
            is_distinct = (
                typo.get("all_bold")
                or typo.get("centered")
                or typo.get("all_italic")
                or (typo.get("starts_bold_label") and len(text.split()) <= bold_label_words and block["type"] == "heading")
                or typo.get("font_pt") != body_font_pt
                or block["type"] == "heading"
            )
            if not is_distinct:
                continue
        elif block["type"] not in ("heading", "caption"):
            continue
        cid = "c" + str(len(candidates))
        if idx > 0:
            prev_type = body[idx - 1]["type"]
        else:
            prev_type = "none"
        if idx + 1 < len(body):
            next_type = body[idx + 1]["type"]
        else:
            next_type = "none"
        page_index = block.get("provenance", {}).get("page_index")
        if typo is not None:
            font_pt = typo["font_pt"]
            font_delta = round(typo["font_pt"] - body_font_pt, 1)
            all_bold = typo["all_bold"]
            all_italic = typo["all_italic"]
            centered = typo["centered"]
            is_upper = typo["is_upper"]
        else:
            letters = [ch for ch in text if ch.isalpha()]
            is_upper = bool(letters) and sum(1 for ch in letters if ch.isupper()) * upper_num >= len(letters) * upper_den
            font_pt = None
            font_delta = 0.0
            all_bold = False
            all_italic = False
            centered = False
        candidates.append({
            "id": cid,
            "block_index": idx,
            "page_index": page_index,
            "text": text[:text_preview_chars],
            "font_pt": font_pt,
            "font_delta": font_delta,
            "all_bold": all_bold,
            "all_italic": all_italic,
            "centered": centered,
            "is_upper": is_upper,
            "prev_type": prev_type,
            "next_type": next_type,
            "source_role": block["type"],
            "source_level": block.get("level"),
        })
        candidate_blocks[cid] = (block, typo)
        questions[cid] = {
            "type": "choice",
            "instructions": render_prompt(BREADCRUMB_HIERARCHY_PROMPT["instructions"], cid=cid),
            "criteria": criteria,
        }
    if not candidates:
        return
    # Every batch sees the same compact outline so peer headings in different
    # batches receive consistent levels.
    outline_limit = _BREADCRUMB_CFG["outline_max_candidates"]
    outline_chars = _BREADCRUMB_CFG["outline_text_chars"]
    outline = [{"text": item["text"][:outline_chars], "font_pt": item["font_pt"], "all_bold": item["all_bold"],
                "centered": item["centered"], "is_upper": item["is_upper"]} for item in candidates[:outline_limit]]
    state = {
        "typography_profile": ir["typography_profile"],
        "document_outline": outline,
        "candidates": candidates,
    }
    answers = ask_decisions("breadcrumb_hierarchy", state, questions, cache=cache, ir=ir)
    min_conf = _BREADCRUMB_CFG["min_confidence_same_size"]
    for cid, (block, typo) in candidate_blocks.items():
        ans = answers.get(cid)
        if ans is None:
            continue
        choice = ans["choice"]
        try:
            raw_confidence = ans.get("confidence", 1.0)
            if isinstance(raw_confidence, bool):
                confidence = math.nan
            else:
                confidence = float(raw_confidence)
        except (ValueError, TypeError, OverflowError):
            confidence = math.nan
        if not 0 <= confidence <= 1:
            record_model_warning(ir, "breadcrumb_hierarchy", "invalid_confidence")
            continue
        if choice == "CAPTION":
            block["type"] = "caption"
            block.pop("level", None)
            evidence_meta = {"source": "decision_model_hierarchy", "confidence": confidence}
            if typo is not None:
                evidence_meta["font_pt"] = typo["font_pt"]
                evidence_meta["body_font_pt"] = body_font_pt
            block["caption_evidence"] = evidence_meta
        elif choice in level_map:
            if typo is not None and typo["font_pt"] <= body_font_pt and confidence < min_conf and block["type"] != "heading":
                block["type"] = "paragraph"
                block.pop("level", None)
                continue
            lvl = level_map[choice]
            block["type"] = "heading"
            block["level"] = lvl
            evidence_meta = {"source": "decision_model_hierarchy", "level": lvl, "confidence": confidence}
            if typo is not None:
                evidence_meta["font_pt"] = typo["font_pt"]
                evidence_meta["body_font_pt"] = body_font_pt
            block["heading_evidence"] = evidence_meta
        else:
            if block["type"] == "heading":
                block["type"] = "paragraph"
                block.pop("level", None)



def decide_caption_attachments(ir, cache=None):
    """Resolve captions, table structures, and image context."""
    attach_visual_captions(ir, partial(ask_decisions, ir=ir), cache=cache)
    decide_table_structures_and_titles(ir, cache=cache)
    decide_image_titles_and_context(ir, cache=cache)


def decide_table_structures_and_titles(ir, cache=None):
    resolve_table_structures_and_titles(ir, partial(ask_decisions, ir=ir), cache=cache)


def decide_image_titles_and_context(ir, cache=None):
    resolve_image_titles_and_context(ir, partial(ask_decisions, ir=ir), cache=cache)


def decide_continued_tables(ir, cache=None):
    merge_continued_tables(ir, partial(ask_decisions, ir=ir), cache=cache)


def decide_prompt_pairs(blocks, cache=None, ir=None):
    """Use decision model to identify prompt/question blocks that must stay paired with the next block."""
    candidates = []
    questions = {}
    index_map = {}
    max_first = _PAIR_CFG["max_first_words"]
    max_second = _PAIR_CFG["max_second_words"]
    first_chars = _PAIR_CFG["first_text_chars"]
    second_chars = _PAIR_CFG["second_text_chars"]
    for idx in range(len(blocks) - 1):
        block = blocks[idx]
        next_block = blocks[idx + 1]
        if block.get("type") != "paragraph" or next_block.get("type") != "paragraph":
            continue
        text = block.get("text", "").strip()
        next_text = next_block.get("text", "").strip()
        if not text or not next_text:
            continue
        w_count = len(text.split())
        next_words = len(next_text.split())
        if w_count > max_first or next_words > max_second:
            continue
        typo = block.get("typography") or {}
        has_signal = (
            text.endswith("?")
            or text.endswith(":")
            or typo.get("all_bold", False)
            or typo.get("starts_bold_label", False)
        )
        if not has_signal:
            continue
        pid = "pair_" + str(len(candidates))
        candidates.append({
            "id": pid,
            "first_text": text[:first_chars],
            "second_text": next_text[:second_chars],
            "first_bold": typo.get("all_bold", False),
            "starts_bold_label": typo.get("starts_bold_label", False),
        })
        index_map[pid] = (block.get("id", str(idx)), next_block.get("id", str(idx + 1)))
        pair_spec = deepcopy(PROMPT_PAIRS_PROMPT)
        pair_spec["instructions"] = render_prompt(pair_spec["instructions"], pid=pid)
        questions[pid] = pair_spec
    if not questions:
        return set()
    answers = ask_decisions("prompt_response_pairing", {"pairs": candidates}, questions, cache=cache, ir=ir)
    paired_ids = set()
    for pid, (id1, id2) in index_map.items():
        if answers.get(pid, {}).get("choice") == "PAIR_TOGETHER":
            paired_ids.add(frozenset((id1, id2)))
    return paired_ids


def decide_period_boundaries(text, candidate_positions, cache=None, ir=None):
    """Use decision model to disambiguate ambiguous period '.' positions (sentence break vs abbreviation)."""
    if not candidate_positions:
        return set()
    items = []
    questions = {}
    pos_map = {}
    ctx_chars = _PERIOD_CFG["context_window_chars"]
    for idx, (dot_index, gap_end, token) in enumerate(candidate_positions):
        qid = "dot_" + str(idx)
        left_start = max(0, dot_index - ctx_chars)
        right_end = min(len(text), gap_end + ctx_chars)
        items.append({
            "id": qid,
            "token": token,
            "left_context": text[left_start:dot_index + 1],
            "right_context": text[gap_end:right_end],
        })
        pos_map[qid] = gap_end
        period_spec = deepcopy(PERIOD_BOUNDARIES_PROMPT)
        period_spec["instructions"] = render_prompt(period_spec["instructions"], token=token, qid=qid)
        questions[qid] = period_spec
    answers = ask_decisions("period_boundary", {"candidates": items}, questions, cache=cache, ir=ir)
    valid_breaks = set()
    for qid, gap_end in pos_map.items():
        if answers.get(qid, {}).get("choice") == "SENTENCE_BREAK":
            valid_breaks.add(gap_end)
    return valid_breaks
