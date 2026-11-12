"""Centralized Decision Model (/systemone) engine for structural chunking decisions.

Replaces heuristic rules across breadcrumb hierarchy inference, caption attachment,
multi-page table continuation, prompt-response pairing, section cohesion merging,
sentence-break disambiguation, and document taxonomy classification.
Complies with 0% Regex Law, Zero Ternary Law, and No Typing Library.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from agent_banking.core.models import chain_fingerprint, model_primary
from .model_enrichment import cloud_json, ordered_parallel, validate_decisions
from .vision_cache import PageVisionCache

DECISION_ENGINE_VERSION = "decision-engine-v1.0"
_DEFAULT_CACHE = None


def get_decision_cache(cache=None):
    global _DEFAULT_CACHE
    if cache is not None:
        return cache
    if _DEFAULT_CACHE is None:
        db_path = Path(__file__).resolve().parent.parent / "cache" / "grounded_models.db"
        _DEFAULT_CACHE = PageVisionCache(str(db_path))
    return _DEFAULT_CACHE


def ask_decisions(role, state, questions, cache=None, batch_size=25, workers=4):
    """Execute and cache batched /systemone decision queries."""
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
        payload_seed = {
            "version": DECISION_ENGINE_VERSION,
            "namespace": namespace,
            "role": role,
            "model": model,
            "state": state,
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
            {"model": model, "state": state, "questions": batch_questions},
            timeout=60,
            model_module="chunking.decision",
        )
        validate_decisions(result, batch_questions)
        active_cache.put(key, result)
        return deepcopy(result.get("answers", {}))

    if len(batches) == 1:
        return run_batch(batches[0])
    batch_results = ordered_parallel(run_batch, batches, min(workers, len(batches)))
    merged = {}
    for ans_dict in batch_results:
        merged.update(ans_dict)
    return merged


def decide_breadcrumb_hierarchy(ir, cache=None):
    """Use the decision model to classify heading levels (1..4), captions, and body text."""
    body = [
        block for block in ir["blocks"]
        if block.get("provenance", {}).get("region", "body") == "body"
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
    for idx, block in enumerate(body):
        if block["type"] not in ("paragraph", "heading", "caption"):
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
        })
        candidate_blocks[cid] = (block, typo)
        questions[cid] = {
            "type": "choice",
            "instructions": (
                "Classify the structural breadcrumb hierarchy level or block role for candidate "
                + cid
                + " using its typography (font_pt vs body_font_pt, bold, italic, centered, uppercase), "
                + "adjacent block types, and text."
            ),
            "criteria": {
                "LEVEL_1": "Main document title banner at the top of the document",
                "LEVEL_2": "Top-level Part or Chapter heading",
                "LEVEL_3": "Major Section, Article heading, or standalone FAQ question heading",
                "LEVEL_4": "Numbered sub-clause or deep sub-item heading",
                "CAPTION": "Explicit caption, note, or title directly labeling an adjacent table or figure",
                "BODY": "Regular body paragraph, diagram internal box/step label, footer/boilerplate, or inline list item",
            },
        }
    if not candidates:
        return
    state = {
        "typography_profile": ir["typography_profile"],
        "candidates": candidates,
    }
    answers = ask_decisions("breadcrumb_hierarchy", state, questions, cache=cache)
    level_map = {
        "LEVEL_1": 1,
        "LEVEL_2": 2,
        "LEVEL_3": 3,
        "LEVEL_4": 4,
    }
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
            if block["type"] == "heading" and block.get("heading_format") != "openxml_outline":
                block["type"] = "paragraph"
                block.pop("level", None)


def decide_caption_attachments(ir, cache=None):
    """Link caption blocks to adjacent image/table blocks, using decision model when ambiguous."""
    blocks = ir["blocks"]
    existing = {(rel["from"], rel["to"], rel["type"]) for rel in ir.get("relations", [])}
    ambiguous = []
    questions = {}
    ambiguous_map = {}
    for index, block in enumerate(blocks):
        if block["type"] != "caption":
            continue
        prev_visual = None
        next_visual = None
        if index > 0 and blocks[index - 1]["type"] in ("image", "table"):
            prev_visual = blocks[index - 1]
        if index + 1 < len(blocks) and blocks[index + 1]["type"] in ("image", "table"):
            next_visual = blocks[index + 1]
        if prev_visual is not None and next_visual is None:
            _attach_caption(ir, block, prev_visual, existing)
        elif next_visual is not None and prev_visual is None:
            _attach_caption(ir, block, next_visual, existing)
        elif prev_visual is not None and next_visual is not None:
            qid = "cap_" + str(len(ambiguous))
            ambiguous.append({
                "id": qid,
                "caption_text": block["text"][:200],
                "prev_visual_type": prev_visual["type"],
                "prev_visual_preview": prev_visual.get("text", "")[:120],
                "next_visual_type": next_visual["type"],
                "next_visual_preview": next_visual.get("text", "")[:120],
            })
            ambiguous_map[qid] = (block, prev_visual, next_visual)
            questions[qid] = {
                "type": "choice",
                "instructions": "Determine whether caption " + qid + " labels the preceding visual or introduces the following visual.",
                "criteria": {
                    "ATTACH_PREV": "Caption describes or summarizes the immediately preceding visual/table",
                    "ATTACH_NEXT": "Caption/title introduces the immediately following visual/table",
                    "STANDALONE": "Independent text not labeling either visual",
                },
            }
    if questions:
        answers = ask_decisions("caption_attachment", {"candidates": ambiguous}, questions, cache=cache)
        for qid, (block, prev_visual, next_visual) in ambiguous_map.items():
            choice = answers.get(qid, {}).get("choice", "ATTACH_NEXT")
            if choice == "ATTACH_PREV":
                _attach_caption(ir, block, prev_visual, existing)
            elif choice == "ATTACH_NEXT":
                _attach_caption(ir, block, next_visual, existing)


def _attach_caption(ir, caption_block, target_block, existing):
    target_block["caption"] = caption_block["text"]
    if target_block["type"] == "image" and "image" in target_block:
        target_block["image"]["caption"] = caption_block["text"]
    rel_key = (caption_block["id"], target_block["id"], "caption_of")
    if rel_key not in existing:
        ir["relations"].append({"from": caption_block["id"], "to": target_block["id"], "type": "caption_of"})
        existing.add(rel_key)


def decide_continued_tables(ir, cache=None):
    """Use decision model to determine whether consecutive multi-page PDF tables should merge."""
    blocks = ir["blocks"]
    if len(blocks) < 2:
        return
    pairs = []
    questions = {}
    pair_indices = {}
    for idx in range(1, len(blocks)):
        prev = blocks[idx - 1]
        curr = blocks[idx]
        if prev["type"] != "table" or curr["type"] != "table":
            continue
        prev_page = prev.get("provenance", {}).get("page_index", -2)
        curr_page = curr.get("provenance", {}).get("page_index", -1)
        t1 = prev["table_data"]
        t2 = curr["table_data"]
        if curr_page == prev_page + 1 and len(t1["headers"]) == len(t2["headers"]) and len(t1["headers"]) > 0:
            pid = "tbl_" + str(len(pairs))
            prev_tail = []
            if t1["rows"]:
                prev_tail = t1["rows"][-1]
            curr_head = []
            if t2["rows"]:
                curr_head = t2["rows"][0]
            pairs.append({
                "id": pid,
                "prev_page": prev_page,
                "curr_page": curr_page,
                "prev_bbox": prev.get("provenance", {}).get("bbox", []),
                "curr_bbox": curr.get("provenance", {}).get("bbox", []),
                "prev_headers": t1["headers"],
                "curr_headers": t2["headers"],
                "prev_last_row": prev_tail,
                "curr_first_row": curr_head,
            })
            pair_indices[pid] = idx
            questions[pid] = {
                "type": "choice",
                "instructions": "Decide if table pair " + pid + " across consecutive pages is a single continued table split by a page break.",
                "criteria": {
                    "MERGE_CONTINUED": "Second table is a direct continuation of the first table across the page boundary",
                    "KEEP_SEPARATE": "Second table is a distinct, independent table",
                },
            }
    if not questions:
        return
    answers = ask_decisions("pdf_table_continuation", {"table_pairs": pairs}, questions, cache=cache)
    merge_indices = {
        idx for pid, idx in pair_indices.items()
        if answers.get(pid, {}).get("choice") == "MERGE_CONTINUED"
    }
    if not merge_indices:
        return
    merged_blocks = []
    redirect_block_id = {}
    for idx, block in enumerate(blocks):
        if idx in merge_indices and merged_blocks and merged_blocks[-1]["type"] == "table":
            prev = merged_blocks[-1]
            t1 = prev["table_data"]
            t2 = block["table_data"]
            row_offset = len(t1["header_grid"]) + len(t1["rows"])
            t2_header_count = t2["header_row_count"]
            for cell in t2.get("cells", []):
                if cell["row"] < t2_header_count:
                    continue
                shifted = deepcopy(cell)
                shifted["row"] = row_offset + (cell["row"] - t2_header_count)
                t1["cells"].append(shifted)
            t1["rows"].extend(t2["rows"])
            t1["grid"] = t1["header_grid"] + t1["rows"]
            redirect_block_id[block["id"]] = prev["id"]
            continue
        merged_blocks.append(block)
    if redirect_block_id:
        for index, block in enumerate(merged_blocks):
            block["order"] = index
        for unit in ir["units"]:
            target = redirect_block_id.get(unit["block_id"])
            if target is not None:
                unit["block_id"] = target
        ir["blocks"] = merged_blocks


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


def decide_furniture_lines(lines, cache=None):
    """Use decision model to filter header/footer/page-number/form-code noise from short lines."""
    candidates = []
    questions = {}
    idx_map = {}
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        words = s.split()
        if len(words) <= 6 and (any(ch.isdigit() for ch in s) or ":" in s or "|" in s):
            lid = "ln_" + str(len(candidates))
            candidates.append({"id": lid, "line_index": idx, "text": s[:160]})
            idx_map[lid] = idx
            questions[lid] = {
                "type": "choice",
                "instructions": "Decide whether short line " + lid + " is page furniture (page number, header/footer boilerplate, bare form code) or substantive document content.",
                "criteria": {
                    "FURNITURE_NOISE": "Bare page number, running header/footer boilerplate, or standalone form code metadata",
                    "KEEP_CONTENT": "Substantive heading, list item, clause, or body content",
                },
            }
    if not questions:
        return set()
    answers = ask_decisions("furniture_line_filter", {"lines": candidates}, questions, cache=cache)
    noise_indices = set()
    for lid, idx in idx_map.items():
        if answers.get(lid, {}).get("choice") == "FURNITURE_NOISE":
            noise_indices.add(idx)
    return noise_indices


def decide_tiny_section_merges(sections, min_words, min_chars, section_size_fn, cache=None):
    """Use decision model to choose merge direction (MERGE_NEXT vs MERGE_PREV) for tiny sections."""
    if len(sections) <= 1:
        return {}
    candidates = []
    questions = {}
    sec_map = {}
    for idx, section in enumerate(sections):
        words, chars = section_size_fn(section)
        if words >= min_words and chars >= min_chars:
            continue
        has_prev = idx > 0
        has_next = idx + 1 < len(sections)
        if not has_prev and not has_next:
            continue
        if has_prev and not has_next:
            sec_map["s_" + str(idx)] = (idx, "MERGE_PREV")
            continue
        if has_next and not has_prev:
            sec_map["s_" + str(idx)] = (idx, "MERGE_NEXT")
            continue
        sid = "sec_" + str(len(candidates))
        prev_sec = sections[idx - 1]
        next_sec = sections[idx + 1]
        cur_preview = " ".join(b.get("text", "").strip() for b in section["blocks"] if b.get("text", "").strip())[:180]
        prev_preview = " ".join(b.get("text", "").strip() for b in prev_sec["blocks"] if b.get("text", "").strip())[:180]
        next_preview = " ".join(b.get("text", "").strip() for b in next_sec["blocks"] if b.get("text", "").strip())[:180]
        candidates.append({
            "id": sid,
            "current_headings": section["headings"],
            "current_text": cur_preview,
            "prev_headings": prev_sec["headings"],
            "prev_text": prev_preview,
            "next_headings": next_sec["headings"],
            "next_text": next_preview,
        })
        sec_map[sid] = (idx, None)
        questions[sid] = {
            "type": "choice",
            "instructions": "Tiny section " + sid + " is below the minimum chunk size. Decide whether it belongs with the preceding section or introduces the following section.",
            "criteria": {
                "MERGE_NEXT": "Current section is a parent heading or introductory lead-in that belongs with the following section",
                "MERGE_PREV": "Current section is a trailing note, sub-item, or continuation that belongs with the preceding section",
            },
        }
    decisions = {}
    for key, (idx, fixed) in sec_map.items():
        if fixed is not None:
            decisions[idx] = fixed
    if questions:
        answers = ask_decisions("tiny_section_merge", {"sections": candidates}, questions, cache=cache)
        for sid, (idx, _) in sec_map.items():
            if sid in answers:
                decisions[idx] = answers[sid].get("choice", "MERGE_NEXT")
    return decisions


def decide_document_taxonomy(raw_path, cache=None):
    """Use decision model to classify document category and doc_type without hardcoded keyword maps."""
    clean_path = str(raw_path or "").replace("\\", "/").strip()
    if not clean_path:
        return "UNKNOWN", "UNKNOWN"
    state = {"document_path": clean_path}
    questions = {
        "category": {
            "type": "choice",
            "instructions": "Classify the high-level banking domain category for this document path.",
            "criteria": {
                "THE": "Card products, credit/debit/prepaid cards, cardholder policies",
                "BAO_HIEM": "Bancassurance, life/health insurance policies and riders",
                "TIN_DUNG": "Credit, retail/corporate loans, overdrafts, collateral, mortgages",
                "NGHIEP_VU": "Core banking operations, accounts, deposits, payments, trade finance, treasury",
                "PHAP_LY": "Legal compliance, AML/CDD, risk management, governance, cybersecurity, ESG",
                "HDSD_APP": "Digital banking mobile app user guides and step-by-step tutorials",
                "BAO_CAO": "Annual reports, audited financial statements, corporate governance reports",
                "VAN_HANH": "Operational procedures, SLA commitments, customer support FAQs, dispute handling",
                "KHAC": "Other general banking documents",
            },
        },
        "doc_type": {
            "type": "choice",
            "instructions": "Classify the functional document type for this banking document.",
            "criteria": {
                "QUY_DINH_SAN_PHAM": "Product regulations, terms and conditions, policy rules, legal compliance rules",
                "QUY_TRINH_NGHIEP_VU": "Operational workflows, appraisal/disbursement/handling processes, SLA procedures",
                "HUONG_DAN_SU_DUNG": "Step-by-step user guides and digital app feature instructions",
                "BAO_CAO_TAI_CHINH": "Annual reports and financial statements",
                "HOI_DAP_NGHIEP_VU": "Customer support Q&A and operational FAQs",
                "BIEU_PHI_LAI_SUAT": "Fee schedules, interest rate tables, and limit matrices",
            },
        },
    }
    answers = ask_decisions("document_taxonomy", state, questions, cache=cache)
    cat = answers.get("category", {}).get("choice", "KHAC")
    dtype = answers.get("doc_type", {}).get("choice", "QUY_DINH_SAN_PHAM")
    return cat, dtype
