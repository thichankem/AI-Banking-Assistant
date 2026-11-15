"""Cached structural decisions for document chunking."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from agent_banking.core.models import chain_fingerprint, model_primary
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
from ...text.language import classify_text_blocks
from ...models.text_client import cloud_json, ordered_parallel, validate_decisions
from ...models.response_cache import PageVisionCache

DECISION_ENGINE_VERSION = "decision-engine-v2.0"
_DEFAULT_CACHE = None


def get_decision_cache(cache=None):
    global _DEFAULT_CACHE
    if cache is not None:
        return cache
    if _DEFAULT_CACHE is None:
        db_path = Path(__file__).resolve().parents[3] / "cache" / "grounded_models.db"
        _DEFAULT_CACHE = PageVisionCache(str(db_path))
    return _DEFAULT_CACHE


def ask_decisions(role, state, questions, cache=None, batch_size=12, workers=2):
    """Execute and cache batched /systemone decision queries."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Decision batch size must be a positive integer")
    if type(workers) is not int or workers < 1:
        raise ValueError("Decision worker count must be a positive integer")
    if not questions:
        return {}
    active_cache = get_decision_cache(cache)
    endpoint = model_primary("chunking.decision")
    model = endpoint.model
    namespace = {
        "engine": DECISION_ENGINE_VERSION,
        "decisions": chain_fingerprint("chunking.decision"),
        "provider": endpoint.url,
    }
    q_items = list(questions.items())
    batches = []
    for start in range(0, len(q_items), batch_size):
        batches.append(dict(q_items[start:start + batch_size]))

    def run_batch(batch_questions):
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
                return deepcopy(cached.get("answers", {}))
            except (ValueError, TypeError, KeyError):
                active_cache.delete(key)
        result = cloud_json(
            endpoint.url.rstrip("/") + "/systemone",
            endpoint.api_key,
            {"model": model, "state": batch_state, "questions": batch_questions},
            timeout=60,
            model_module="chunking.decision",
        )
        validate_decisions(result, batch_questions)
        active_cache.put(key, result)
        return deepcopy(result.get("answers", {}))

    if len(batches) == 1:
        return run_batch(batches[0])
    batch_results = ordered_parallel(run_batch, batches, min(2, workers, len(batches)))
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
    if not char_by_size:
        return
    body_font_pt = 10.5
    max_chars = -1
    for sz, count in char_by_size.items():
        if count > max_chars:
            max_chars = count
            body_font_pt = sz
    ir["typography_profile"] = {
        "body_font_pt": body_font_pt,
        "char_distribution": {str(k): v for k, v in sorted(char_by_size.items())},
    }
    candidates = []
    questions = {}
    candidate_blocks = {}
    level_map = {"LEVEL_" + str(level): level for level in range(1, MAX_HEADING_LEVEL + 1)}
    criteria = {
        "LEVEL_1": "Document title or primary top-level heading, based on peer heading patterns",
        **{key: "Heading at document outline level " + str(level)
           + "; determine relative depth from the surrounding hierarchy and recurring typography"
           for key, level in level_map.items() if level > 1},
        "CAPTION": "Text directly labeling a nearby figure or table",
        "BODY": "Paragraph, list item, or label with no evidence that it begins a section",
    }
    for idx, block in enumerate(body):
        if block["type"] not in ("paragraph", "heading", "caption"):
            continue
        if block.get("heading_format") == "openxml_outline":
            continue
        typo = block.get("typography")
        if typo is None or not typo.get("single_line"):
            continue
        text = block.get("text", "").strip()
        if not text or len(text.split()) > 45:
            continue
        is_distinct = (
            typo.get("all_bold")
            or typo.get("centered")
            or typo.get("all_italic")
            or typo.get("font_pt") != body_font_pt
            or block["type"] == "heading"
        )
        if not is_distinct:
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
        candidates.append({
            "id": cid,
            "block_index": idx,
            "text": text[:220],
            "font_pt": typo["font_pt"],
            "font_delta": round(typo["font_pt"] - body_font_pt, 1),
            "all_bold": typo["all_bold"],
            "all_italic": typo["all_italic"],
            "centered": typo["centered"],
            "is_upper": typo["is_upper"],
            "prev_type": prev_type,
            "next_type": next_type,
            "source_role": block["type"],
            "source_level": block.get("level"),
        })
        candidate_blocks[cid] = (block, typo)
        questions[cid] = {
            "type": "choice",
            "instructions": (
                "Classify the structural breadcrumb hierarchy level or block role for candidate "
                + cid
                + " using its typography (font_pt vs body_font_pt, bold, italic, centered, uppercase), "
                + "neighboring block roles and text. Keep peer headings at consistent levels. "
                + "Proximity to an image does not determine the role or level."
            ),
            "criteria": criteria,
        }
    if not candidates:
        return
    state = {
        "typography_profile": ir["typography_profile"],
        "candidates": candidates,
    }
    answers = ask_decisions("breadcrumb_hierarchy", state, questions, cache=cache)
    for cid, (block, typo) in candidate_blocks.items():
        ans = answers.get(cid, {})
        choice = ans.get("choice", "BODY")
        confidence = float(ans.get("confidence", 1.0))
        if choice == "CAPTION":
            block["type"] = "caption"
            block.pop("level", None)
            block["caption_evidence"] = {
                "source": "decision_model_hierarchy",
                "font_pt": typo["font_pt"],
                "body_font_pt": body_font_pt,
                "confidence": confidence,
            }
        elif choice in level_map:
            if typo["font_pt"] <= body_font_pt and confidence < 0.72 and block["type"] != "heading":
                block["type"] = "paragraph"
                block.pop("level", None)
                continue
            lvl = level_map[choice]
            block["type"] = "heading"
            block["level"] = lvl
            block["heading_evidence"] = {
                "source": "decision_model_hierarchy",
                "font_pt": typo["font_pt"],
                "body_font_pt": body_font_pt,
                "level": lvl,
                "confidence": confidence,
            }
        else:
            if block["type"] == "heading":
                block["type"] = "paragraph"
                block.pop("level", None)



def decide_caption_attachments(ir, cache=None):
    """Resolve captions, table structures, and image context."""
    attach_visual_captions(ir, ask_decisions, cache=cache)
    decide_table_structures_and_titles(ir, cache=cache)
    decide_image_titles_and_context(ir, cache=cache)


def decide_table_structures_and_titles(ir, cache=None):
    resolve_table_structures_and_titles(ir, ask_decisions, cache=cache)


def decide_image_titles_and_context(ir, cache=None):
    resolve_image_titles_and_context(ir, ask_decisions, cache=cache)


def decide_continued_tables(ir, cache=None):
    merge_continued_tables(ir, ask_decisions, cache=cache)


def decide_prompt_pairs(blocks, cache=None):
    """Use decision model to identify prompt/question blocks that must stay paired with the next block."""
    candidates = []
    questions = {}
    index_map = {}
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
        if w_count > 45 or next_words > 220:
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
            "first_text": text[:200],
            "second_text": next_text[:240],
            "first_bold": typo.get("all_bold", False),
            "starts_bold_label": typo.get("starts_bold_label", False),
        })
        index_map[pid] = (block.get("id", str(idx)), next_block.get("id", str(idx + 1)))
        questions[pid] = {
            "type": "choice",
            "instructions": "Decide whether " + pid + " is a tightly coupled prompt-response (Q&A) or lead-in pair that must stay in the same retrieval chunk.",
            "criteria": {
                "PAIR_TOGETHER": "First block is a question, prompt, or lead-in directly answered or completed by the second block",
                "SEPARATE": "The two blocks are ordinary sequential paragraphs that do not form a prompt-response pair",
            },
        }
    if not questions:
        return set()
    answers = ask_decisions("prompt_response_pairing", {"pairs": candidates}, questions, cache=cache)
    paired_ids = set()
    for pid, (id1, id2) in index_map.items():
        if answers.get(pid, {}).get("choice") == "PAIR_TOGETHER":
            paired_ids.add(frozenset((id1, id2)))
    return paired_ids


def decide_period_boundaries(text, candidate_positions, cache=None):
    """Use decision model to disambiguate ambiguous period '.' positions (sentence break vs abbreviation)."""
    if not candidate_positions:
        return set()
    items = []
    questions = {}
    pos_map = {}
    for idx, (dot_index, gap_end, token) in enumerate(candidate_positions):
        qid = "dot_" + str(idx)
        left_start = max(0, dot_index - 45)
        right_end = min(len(text), gap_end + 45)
        items.append({
            "id": qid,
            "token": token,
            "left_context": text[left_start:dot_index + 1],
            "right_context": text[gap_end:right_end],
        })
        pos_map[qid] = gap_end
        questions[qid] = {
            "type": "choice",
            "instructions": "Decide whether the period in token '" + token + "' at " + qid + " marks the end of a sentence or an inline abbreviation/title/number prefix.",
            "criteria": {
                "SENTENCE_BREAK": "The period ends a complete sentence and is followed by a new sentence",
                "INLINE_ABBREVIATION": "The period is part of an abbreviation, title, honorific, code, or numbering prefix inside the same sentence",
            },
        }
    answers = ask_decisions("period_boundary", {"candidates": items}, questions, cache=cache)
    valid_breaks = set()
    for qid, gap_end in pos_map.items():
        if answers.get(qid, {}).get("choice") == "SENTENCE_BREAK":
            valid_breaks.add(gap_end)
    return valid_breaks
