"""Centralized Decision Model (/systemone) engine for structural chunking decisions.

Handles breadcrumb hierarchy inference, table header & title assignment,
image title & context bundling, multi-page table continuation,
prompt-response pairing, and sentence-break period disambiguation.
Complies with 0% Regex Law, Zero Ternary Law, and No Typing Library.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from agent_banking.core.models import chain_fingerprint, model_primary
from .image_context import IMAGE_POLICY
from .model_enrichment import cloud_json, ordered_parallel, validate_decisions
from .vision_cache import PageVisionCache

DECISION_ENGINE_VERSION = "decision-engine-v2.0"
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
        steps_to_image = None
        for look_ahead in range(1, 6):
            if idx + look_ahead < len(body) and body[idx + look_ahead]["type"] == "image":
                steps_to_image = look_ahead
                break
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
            "steps_to_next_image": steps_to_image,
        })
        candidate_blocks[cid] = (block, typo)
        questions[cid] = {
            "type": "choice",
            "instructions": (
                "Classify the structural breadcrumb hierarchy level or block role for candidate "
                + cid
                + " using its typography (font_pt vs body_font_pt, bold, italic, centered, uppercase), "
                + "adjacent block types, steps_to_next_image, and text."
            ),
            "criteria": {
                "LEVEL_1": "Main document title banner at the top of the document",
                "LEVEL_2": "Top-level Part (Phần), Chapter (Chương), or Appendix (Phụ lục) section heading",
                "LEVEL_3": "Major Section (Mục), Article (Điều) heading, or standalone FAQ question heading (never a diagram box label when steps_to_next_image is 1..5)",
                "LEVEL_4": "Numbered sub-clause (1.1, 1.2) or deep sub-item heading (never a diagram box label when steps_to_next_image is 1..5)",
                "CAPTION": "Explicit figure or table caption directly labeling an adjacent visual (e.g. Hình:, Sơ đồ:, Bảng:), never a Part/Chapter/Appendix heading",
                "BODY": "Regular body paragraph, diagram box/step label preceding a figure (when steps_to_next_image is 1..5 and not a Phần/Chương/Mục/Điều/Phụ lục heading), or inline list item",
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


def _attach_caption(ir, caption_block, target_block, existing):
    target_block["caption"] = caption_block["text"]
    if target_block["type"] == "image" and "image" in target_block:
        target_block["image"]["caption"] = caption_block["text"]
    if target_block["type"] == "table" and "table_data" in target_block:
        target_block["table_data"]["caption"] = caption_block["text"]
    rel_key = (caption_block["id"], target_block["id"], "caption_of")
    if rel_key not in existing:
        ir["relations"].append({"from": caption_block["id"], "to": target_block["id"], "type": "caption_of"})
        existing.add(rel_key)


def decide_caption_attachments(ir, cache=None):
    """Link captions and resolve optimal titles & structures for all tables and images via decision model."""
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
                "next_visual_type": next_visual["type"],
            })
            ambiguous_map[qid] = (block, prev_visual, next_visual)
            questions[qid] = {
                "type": "choice",
                "instructions": "Determine whether caption " + qid + " labels the preceding visual or introduces the following visual.",
                "criteria": {
                    "ATTACH_PREV": "Caption describes or labels the immediately preceding visual/table",
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
    decide_table_structures_and_titles(ir, cache=cache)
    decide_image_titles_and_context(ir, cache=cache)


def decide_table_structures_and_titles(ir, cache=None):
    """Use decision model to resolve unmarked table header counts and assign descriptive table titles."""
    blocks = ir["blocks"]
    table_entries = []
    questions = {}
    table_map = {}
    current_heading = ""
    for idx, block in enumerate(blocks):
        if block["type"] == "heading":
            current_heading = block.get("text", "").strip()
            continue
        if block["type"] != "table":
            continue
        table = block["table_data"]
        grid = table.get("grid", [])
        if not grid:
            continue
        tid = "t" + str(len(table_entries))
        width = len(grid[0])
        top_rows = grid[:min(4, len(grid))]
        row_bold = []
        for r_idx in range(min(3, len(grid))):
            r_cells = [c for c in table.get("cells", []) if c["row"] == r_idx and c.get("text", "").strip()]
            is_b = bool(r_cells) and all(c.get("all_text_bold", False) for c in r_cells)
            row_bold.append(is_b)
        r0_unique = []
        for val in grid[0]:
            if val.strip() and val.strip() not in r0_unique:
                r0_unique.append(val.strip())
        row_0_banner = len(r0_unique) == 1 and width > 1
        prev_text = ""
        prev_type = "none"
        if idx > 0 and blocks[idx - 1]["type"] in ("paragraph", "caption", "heading"):
            prev_text = blocks[idx - 1].get("text", "").strip()[:200]
            prev_type = blocks[idx - 1]["type"]
        next_text = ""
        next_type = "none"
        if idx + 1 < len(blocks) and blocks[idx + 1]["type"] in ("caption", "paragraph"):
            next_text = blocks[idx + 1].get("text", "").strip()[:200]
            next_type = blocks[idx + 1]["type"]
        entry = {
            "id": tid,
            "width": width,
            "total_rows": len(grid),
            "top_rows": top_rows,
            "row_bold": row_bold,
            "row_0_banner": row_0_banner,
            "existing_header_count": table.get("header_row_count", 0),
            "prev_type": prev_type,
            "prev_text": prev_text,
            "next_type": next_type,
            "next_text": next_text,
            "section_heading": current_heading[:200],
        }
        table_entries.append(entry)
        table_map[tid] = (idx, block, entry)
        if table.get("header_detection") == "unmarked" and len(grid) >= 2:
            questions[tid + "_hdr"] = {
                "type": "choice",
                "instructions": "Determine how many leading rows in table " + tid + " are column header rows (not data records).",
                "criteria": {
                    "HEADERS_1": "First row is the column header row",
                    "HEADERS_2": "First two rows form grouped/merged column headers (or banner + column headers)",
                    "HEADERS_0": "No header row; all rows are data records",
                },
            }
        questions[tid + "_title"] = {
            "type": "choice",
            "instructions": "Select the best descriptive title source for table " + tid + ".",
            "criteria": {
                "PREV_BLOCK": "The immediately preceding block introduces or titles this table",
                "NEXT_BLOCK": "The immediately following block is the caption/title of this table",
                "ROW_0_BANNER": "Row 0 of the table itself is a full-width banner title",
                "SECTION_HEADING": "The enclosing section heading is the best title for this table",
            },
        }
    if not table_entries:
        return
    answers = ask_decisions("table_structure_and_title", {"tables": table_entries}, questions, cache=cache)
    for tid, (idx, block, entry) in table_map.items():
        table = block["table_data"]
        grid = table["grid"]
        width = entry["width"]
        hdr_key = tid + "_hdr"
        if hdr_key in answers:
            hdr_choice = answers[hdr_key].get("choice", "HEADERS_1")
            if hdr_choice == "HEADERS_2" and len(grid) > 2:
                table["header_row_count"] = 2
                table["header_detection"] = "decision_model_headers"
            elif hdr_choice == "HEADERS_1" and len(grid) > 1:
                table["header_row_count"] = 1
                table["header_detection"] = "decision_model_headers"
            else:
                table["header_row_count"] = 0
        title_choice = answers.get(tid + "_title", {}).get("choice", "SECTION_HEADING")
        resolved_title = ""
        if title_choice == "ROW_0_BANNER" and entry["row_0_banner"]:
            resolved_title = grid[0][0].strip()
        elif title_choice == "NEXT_BLOCK" and entry["next_text"]:
            resolved_title = entry["next_text"]
        elif title_choice == "PREV_BLOCK" and entry["prev_text"]:
            resolved_title = entry["prev_text"]
        elif entry["section_heading"]:
            resolved_title = entry["section_heading"]
        if not resolved_title:
            resolved_title = block.get("caption", "") or entry["section_heading"] or entry["prev_text"]
        header_count = table.get("header_row_count", 0)
        if header_count:
            start_hdr_row = 0
            if entry["row_0_banner"] and header_count >= 2:
                start_hdr_row = 1
                if not resolved_title:
                    resolved_title = grid[0][0].strip()
            headers = []
            for column in range(width):
                labels = []
                for row in grid[start_hdr_row:header_count]:
                    if row[column] and row[column] not in labels:
                        labels.append(row[column])
                headers.append(" / ".join(labels))
            table["headers"] = headers
            table["rows"] = grid[header_count:]
        table["title"] = resolved_title
        if resolved_title and not block.get("caption"):
            block["caption"] = resolved_title


def decide_image_titles_and_context(ir, cache=None):
    """Use decision model to assign titles, captions, and surrounding context to all images."""
    blocks = ir["blocks"]
    img_entries = []
    questions = {}
    img_map = {}
    current_heading = ""
    current_heading_id = None
    section_paragraphs = []
    for idx, block in enumerate(blocks):
        if block.get("provenance", {}).get("region", "body") != "body":
            continue
        if block["type"] == "heading":
            current_heading = block.get("text", "").strip()
            current_heading_id = block["id"]
            section_paragraphs = []
            continue
        if block["type"] == "paragraph":
            section_paragraphs.append((idx, block))
            continue
        if block["type"] != "image":
            continue
        iid = "img_" + str(len(img_entries))
        info = block["image"]
        prev_blocks_info = [
            {"block_id": b["id"], "text": b.get("text", "").strip()[:180]}
            for _, b in section_paragraphs[-4:]
            if b.get("text", "").strip()
        ]
        prev_1_text = ""
        prev_1_id = None
        if idx > 0 and blocks[idx - 1]["type"] in ("paragraph", "caption", "heading"):
            prev_1_text = blocks[idx - 1].get("text", "").strip()
            prev_1_id = blocks[idx - 1]["id"]
        next_1_text = ""
        next_1_id = None
        next_1_type = "none"
        if idx + 1 < len(blocks) and blocks[idx + 1]["type"] in ("caption", "paragraph"):
            next_1_text = blocks[idx + 1].get("text", "").strip()
            next_1_id = blocks[idx + 1]["id"]
            next_1_type = blocks[idx + 1]["type"]
        embedded = info.get("title", "") or info.get("alt_text", "") or info.get("native_text", "")
        entry = {
            "id": iid,
            "section_heading": current_heading[:200],
            "prev_1_text": prev_1_text[:200],
            "prev_paragraphs_count": len(prev_blocks_info),
            "prev_paragraphs_preview": [p["text"] for p in prev_blocks_info],
            "next_1_type": next_1_type,
            "next_1_text": next_1_text[:200],
            "embedded_text": embedded[:200],
        }
        img_entries.append(entry)
        img_map[iid] = {
            "idx": idx,
            "block": block,
            "heading_text": current_heading,
            "heading_id": current_heading_id,
            "prev_1_text": prev_1_text,
            "prev_1_id": prev_1_id,
            "prev_blocks": [b for _, b in section_paragraphs[-4:] if b.get("text", "").strip()],
            "next_1_text": next_1_text,
            "next_1_id": next_1_id,
            "next_1_type": next_1_type,
            "embedded": embedded,
        }
        questions[iid + "_title"] = {
            "type": "choice",
            "instructions": "Select the best source for the descriptive title of image/figure " + iid + ".",
            "criteria": {
                "NEXT_CAPTION": "The immediately following block (e.g. Hình:, Sơ đồ:) is the explicit title/caption of this figure",
                "PREV_BLOCK": "The immediately preceding block is the banner title or direct lead-in title of this figure",
                "SECTION_HEADING": "The enclosing section heading best titles this figure",
                "EMBEDDED_META": "The embedded alt_text/title/native_text best titles this figure",
                "DECORATIVE_SKIP": "Decorative image with no meaningful surrounding title, caption, or explanation",
            },
        }
        if prev_blocks_info:
            questions[iid + "_bundle"] = {
                "type": "choice",
                "instructions": "Decide whether the preceding short blocks in this section (" + str(len(prev_blocks_info)) + " blocks) are diagram box labels / direct lead-in steps that belong inside the same multimodal chunk as image " + iid + ".",
                "criteria": {
                    "BUNDLE_PRECEDING": "Preceding blocks in this section are diagram labels, step boxes, or direct lead-in text for this figure and should be bundled with it",
                    "CAPTION_ONLY": "Preceding blocks are independent body prose; bundle only the figure's caption/heading",
                },
            }
        section_paragraphs = []
    if not img_entries:
        return
    answers = ask_decisions("image_title_and_context", {"images": img_entries}, questions, cache=cache)
    existing_rels = {(rel["from"], rel["to"], rel["type"]) for rel in ir.get("relations", [])}
    for iid, ctx in img_map.items():
        block = ctx["block"]
        info = block["image"]
        title_choice = answers.get(iid + "_title", {}).get("choice", "NEXT_CAPTION")
        bundle_choice = answers.get(iid + "_bundle", {}).get("choice", "CAPTION_ONLY")
        if title_choice == "DECORATIVE_SKIP" and not ctx["next_1_text"] and not ctx["prev_1_text"] and not ctx["heading_text"]:
            info.update({
                "image_policy": IMAGE_POLICY,
                "visual_content_read": False,
                "text_context_assessed": True,
                "status": "skipped_no_text_support",
                "context_evidence": [],
                "description": "",
                "title": "",
            })
            block["text"] = ""
            continue
        title_text = ""
        caption_text = info.get("caption", "")
        evidence_blocks = []
        if ctx["next_1_text"] and (title_choice == "NEXT_CAPTION" or ctx["next_1_type"] == "caption"):
            title_text = ctx["next_1_text"]
            caption_text = ctx["next_1_text"]
            next_b = blocks[ctx["idx"] + 1]
            next_b["type"] = "caption"
            _attach_caption(ir, next_b, block, existing_rels)
            evidence_blocks.append(next_b)
        if title_choice == "PREV_BLOCK" and ctx["prev_1_text"]:
            if not title_text:
                title_text = ctx["prev_1_text"]
            prev_b = blocks[ctx["idx"] - 1]
            if prev_b not in evidence_blocks:
                evidence_blocks.append(prev_b)
        if title_choice == "SECTION_HEADING" and ctx["heading_text"]:
            if not title_text:
                title_text = ctx["heading_text"]
        if not title_text:
            title_text = caption_text or ctx["next_1_text"] or ctx["prev_1_text"] or ctx["heading_text"] or ctx["embedded"]
        if not caption_text:
            caption_text = title_text
        if bundle_choice == "BUNDLE_PRECEDING":
            for pb in ctx["prev_blocks"]:
                if pb not in evidence_blocks:
                    evidence_blocks.append(pb)
        if not evidence_blocks:
            if ctx["next_1_id"] is not None:
                evidence_blocks.append(blocks[ctx["idx"] + 1])
            elif ctx["prev_1_id"] is not None:
                evidence_blocks.append(blocks[ctx["idx"] - 1])
        evidence = []
        quotes = []
        for eb in evidence_blocks:
            eb_text = eb.get("text", "").strip()
            if not eb_text:
                continue
            start_pos = eb.get("text", "").find(eb_text)
            if start_pos < 0:
                start_pos = 0
            evidence.append({
                "source_block_id": eb["id"],
                "quote": eb_text,
                "source_span": [start_pos, start_pos + len(eb_text)],
                "location": eb["provenance"],
                "source_text_sha256": hashlib.sha256(eb.get("text", "").encode("utf-8")).hexdigest(),
            })
            if eb_text not in quotes:
                quotes.append(eb_text)
        if not evidence and title_text:
            src_id = ctx["heading_id"] or block["id"]
            evidence.append({
                "source_block_id": src_id,
                "quote": title_text,
                "source_span": [0, len(title_text)],
                "location": block["provenance"],
                "source_text_sha256": hashlib.sha256(title_text.encode("utf-8")).hexdigest(),
            })
            quotes.append(title_text)
        info.update({
            "title": title_text,
            "caption": caption_text,
            "section_title": ctx["heading_text"],
            "bundle_preceding": bundle_choice == "BUNDLE_PRECEDING",
            "image_policy": IMAGE_POLICY,
            "visual_content_read": False,
            "text_context_assessed": True,
            "status": "text_context_supported",
            "context_evidence": evidence,
            "context_reason": "decision_model_image_title_and_context",
            "context_model": model_primary("chunking.decision").model,
            "description": "\n\n".join(quotes),
        })
        block["caption"] = caption_text
        text_parts = []
        if info.get("native_text", "").strip():
            text_parts.append(info["native_text"].strip())
        if info["description"].strip():
            text_parts.append(info["description"].strip())
        block["text"] = "\n\n".join(text_parts)


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
