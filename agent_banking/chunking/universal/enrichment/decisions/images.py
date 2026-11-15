"""Caption links and literal source evidence for figure chunks."""
from agent_banking.core.models import model_primary

from ..image_context import apply_context_selection, region_key


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


def attach_visual_captions(ir, ask_decisions, cache=None):
    """Link caption blocks to nearby visual blocks within their source region."""
    blocks = ir["blocks"]
    existing = {(rel["from"], rel["to"], rel["type"]) for rel in ir.get("relations", [])}
    attached = {start for start, _, kind in existing if kind == "caption_of"}
    ambiguous = []
    questions = {}
    ambiguous_map = {}
    for index, block in enumerate(blocks):
        if block["type"] != "caption" or block["id"] in attached or not block["text"].strip():
            continue
        prev_visual = adjacent_source(blocks, index, -1, {"image", "table"})
        next_visual = adjacent_source(blocks, index, 1, {"image", "table"})
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


def adjacent_source(blocks, index, distance, kinds):
    neighbor = index + distance
    if not 0 <= neighbor < len(blocks):
        return None
    block = blocks[index]
    candidate = blocks[neighbor]
    if candidate["type"] not in kinds or region_key(candidate) != region_key(block):
        return None
    if candidate["provenance"].get("page_index") != block["provenance"].get("page_index"):
        return None
    return candidate


def _text(block):
    if block is None:
        return ""
    return block.get("text", "").strip()


def _native_title(info):
    if info.get("title_source") == "decision_model":
        return info.get("source_title", "")
    return info.get("title", "")


def _image_inputs(ir):
    blocks = ir["blocks"]
    by_id = {block["id"]: block for block in blocks}
    captions = {}
    for relation in ir.get("relations", []):
        if relation["type"] == "caption_of" and relation["from"] in by_id:
            captions.setdefault(relation["to"], []).append(by_id[relation["from"]])
    claimed_captions = {block["id"] for entries in captions.values() for block in entries}
    heading = None
    paragraphs = []
    for index, block in enumerate(blocks):
        if region_key(block) != "body":
            continue
        if block["type"] == "heading":
            heading = block
            paragraphs = []
        elif block["type"] == "paragraph":
            if _text(block):
                paragraphs.append(block)
                paragraphs = paragraphs[-4:]
        elif block["type"] == "image":
            preceding = adjacent_source(blocks, index, -1, {"paragraph", "caption", "heading"})
            following = adjacent_source(blocks, index, 1, {"paragraph", "caption"})
            explicit = captions.get(block["id"], [])
            if following and following["id"] in claimed_captions and following not in explicit:
                following = None
            if preceding and preceding["id"] in claimed_captions and preceding not in explicit:
                preceding = None
            yield block, heading, preceding, following, paragraphs, explicit
            paragraphs = []


def _record_evidence(context, choice, bundle, model, ir, existing):
    block, heading, preceding, following, paragraphs, explicit = context
    info = block["image"]
    original_title = _native_title(info)
    if original_title:
        info["source_title"] = original_title
    embedded = next((field for field in ("source_title", "alt_text", "native_text")
                     if isinstance(info.get(field), str) and info[field].strip()), None)
    title = ""
    sources = []
    if choice != "DECORATIVE_SKIP":
        if choice == "NEXT_CAPTION" and _text(following):
            following["type"] = "caption"
            _attach_caption(ir, following, block, existing)
            sources.append(following)
        elif choice == "PREV_BLOCK" and _text(preceding):
            sources.append(preceding)
        elif choice == "SECTION_HEADING" and _text(heading):
            sources.append(heading)
        if explicit and not sources:
            sources.extend(explicit)
        if choice == "EMBEDDED_META" and embedded:
            title = info[embedded].strip()
        elif sources:
            title = _text(sources[0])
        if bundle:
            sources.extend(paragraphs)
        if not sources and not title:
            fallback = next((candidate for candidate in (following, preceding, heading) if _text(candidate)), None)
            if fallback:
                sources.append(fallback)
                title = _text(fallback)
            elif embedded:
                title = info[embedded].strip()

    candidates = []
    for source in sources:
        if not _text(source) or any(item["block_id"] == source["id"] for item in candidates):
            continue
        candidates.append({"source_id": str(len(candidates)), "block_id": source["id"],
                           "text": source["text"], "location": source["provenance"]})
    if choice != "DECORATIVE_SKIP" and embedded and (choice == "EMBEDDED_META" or not candidates):
        candidates.append({"source_id": str(len(candidates)), "block_id": block["id"],
                           "source_attribute": embedded, "text": info[embedded], "location": block["provenance"]})
    if choice != "DECORATIVE_SKIP" and not candidates and info.get("caption", "").strip():
        candidates.append({"source_id": "0", "block_id": block["id"], "source_attribute": "caption",
                           "text": info["caption"], "location": block["provenance"]})
        title = info["caption"].strip()
    if not title and candidates:
        title = candidates[0]["text"].strip()
    evidence = [{"source_id": candidate["source_id"], "quote": candidate["text"].strip()} for candidate in candidates]
    apply_context_selection(block, candidates, {
        "include": bool(evidence),
        "evidence": evidence,
        "reason": "decision_model_image_title_and_context",
    }, model)
    info.update({"title": title, "title_source": "decision_model",
                 "section_title": _text(heading), "bundle_preceding": bundle and bool(evidence)})


def resolve_image_titles_and_context(ir, ask_decisions, cache=None):
    """Assign image titles from selected literal source blocks or native metadata."""
    entries = []
    questions = {}
    contexts = {}
    for context in _image_inputs(ir):
        block, heading, preceding, following, paragraphs, explicit = context
        iid = "img_" + str(len(entries))
        info = block["image"]
        embedded = _native_title(info) or info.get("alt_text", "") or info.get("native_text", "")
        next_type = "none" if following is None else following["type"]
        entries.append({
            "id": iid,
            "section_heading": _text(heading)[:200],
            "prev_1_text": _text(preceding)[:200],
            "prev_paragraphs_count": len(paragraphs),
            "prev_paragraphs_preview": [_text(item)[:180] for item in paragraphs],
            "next_1_type": next_type,
            "next_1_text": _text(following)[:200],
            "embedded_text": embedded[:200],
            "attached_captions": [_text(item)[:200] for item in explicit],
        })
        contexts[iid] = context
        questions[iid + "_title"] = {
            "type": "choice",
            "instructions": "Select the best source for the descriptive title of image/figure " + iid + ".",
            "criteria": {
                "NEXT_CAPTION": "The immediately following block is an explicit title or caption",
                "PREV_BLOCK": "The immediately preceding block is a direct title or lead-in",
                "SECTION_HEADING": "The enclosing section heading best titles the figure",
                "EMBEDDED_META": "The embedded alt text, title or native text best titles the figure",
                "DECORATIVE_SKIP": "Decorative image with no meaningful surrounding title, caption or explanation",
            },
        }
        if paragraphs:
            questions[iid + "_bundle"] = {
                "type": "choice",
                "instructions": "Decide whether the preceding blocks should be bundled with image " + iid + ".",
                "criteria": {
                    "BUNDLE_PRECEDING": "Preceding blocks are diagram labels, step boxes or direct lead-in",
                    "CAPTION_ONLY": "Preceding blocks are independent body prose",
                },
            }
    if not entries:
        return
    answers = ask_decisions("image_title_and_context", {"images": entries}, questions, cache=cache)
    model = model_primary("chunking.decision").model
    existing = {(rel["from"], rel["to"], rel["type"]) for rel in ir.get("relations", [])}
    for iid, context in contexts.items():
        title_choice = answers.get(iid + "_title", {}).get("choice", "NEXT_CAPTION")
        bundle = answers.get(iid + "_bundle", {}).get("choice") == "BUNDLE_PRECEDING"
        _record_evidence(context, title_choice, bundle, model, ir, existing)
