"""Associate images with literal surrounding text; never inspect image pixels."""
import hashlib

IMAGE_POLICY = "surrounding_text_only_v1"
RESOLVED_IMAGE_STATUSES = ("text_context_supported", "skipped_no_text_support")


def source_text(block):
    if block.get("type") == "table":
        grid = block.get("grid")
        if grid is None:
            grid = block.get("table_data", {}).get("grid", [])
        return "\n".join(" | ".join(str(cell) for cell in row) for row in grid)
    if block.get("type") in ("paragraph", "heading", "caption"):
        return block.get("text", "")
    return ""


def region_key(block):
    region = block.get("provenance", {}).get("region", "body")
    if region.startswith("body"):
        return "body"
    return region


def surrounding_sources(ir, image_index, radius=8):
    """Native neighbors and source-authored annotations, per image occurrence.

    Identical image bytes in different sections do not share an assessment.
    Header/footer and note regions never borrow unrelated body paragraphs.
    """
    blocks = ir["blocks"]
    image = blocks[image_index]
    region = region_key(image)
    candidates = []
    positions = list(range(max(0, image_index - radius), min(len(blocks), image_index + radius + 1)))
    for index in range(image_index - 1, -1, -1):
        if region_key(blocks[index]) == region and blocks[index]["type"] == "heading":
            if index not in positions:
                positions.insert(0, index)
            break
    for index in positions:
        block = blocks[index]
        text = source_text(block)
        if not text.strip() or region_key(block) != region:
            continue
        candidates.append({"source_id": "s" + str(len(candidates)), "block_id": block["id"],
                           "type": block["type"], "distance": index - image_index,
                           "text": text, "location": block["provenance"],
                           "explicit_caption_for_image": any(relation["from"] == block["id"] and relation["to"] == image["id"]
                                                             and relation["type"] == "caption_of" for relation in ir["relations"])})
    for field in ("caption", "alt_text", "title"):
        text = image["image"].get(field, "")
        if isinstance(text, str) and text.strip() and not any(item["text"] == text for item in candidates):
            candidates.append({"source_id": "s" + str(len(candidates)), "block_id": image["id"],
                               "type": "source_" + field, "source_attribute": field, "distance": 0,
                               "text": text, "location": image["provenance"]})
    return candidates


def validate_context_selection(result, candidates):
    if not isinstance(result, dict) or not isinstance(result.get("include"), bool):
        raise ValueError("Image context decision requires boolean include")
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        raise ValueError("Image context decision requires a reason")
    evidence = result.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("Image context evidence must be an array")
    if result["include"] != bool(evidence):
        raise ValueError("Include requires evidence; skip requires an empty evidence array")
    sources = {candidate["source_id"]: candidate for candidate in candidates}
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("Image evidence must be an object")
        source = sources.get(item.get("source_id"))
        quote = item.get("quote")
        if source is None or not isinstance(quote, str) or not quote.strip() or quote not in source["text"]:
            raise ValueError("Image evidence must quote a supplied source verbatim")


def apply_context_selection(block, candidates, result, model):
    validate_context_selection(result, candidates)
    sources = {candidate["source_id"]: candidate for candidate in candidates}
    info = block["image"]
    # An old visual interpretation must never survive a policy migration.
    for field in ("ocr_text", "nodes", "edges", "uncertainties", "visual_review_passes", "draft_sha256", "draft_model"):
        info.pop(field, None)
    evidence = []
    quotes = []
    for item in result["evidence"]:
        source = sources[item["source_id"]]
        quote = item["quote"]
        start = source["text"].find(quote)
        record = {"source_block_id": source["block_id"], "quote": quote,
                  "source_span": [start, start + len(quote)], "location": source["location"],
                  "source_text_sha256": hashlib.sha256(source["text"].encode("utf-8")).hexdigest()}
        if source.get("source_attribute"):
            record["source_attribute"] = source["source_attribute"]
        if record not in evidence:
            evidence.append(record)
        if quote not in quotes:
            quotes.append(quote)
    info.update({"image_policy": IMAGE_POLICY, "visual_content_read": False,
                 "text_context_assessed": True, "context_evidence": evidence,
                 "context_reason": result["reason"], "context_model": model,
                 "description": "\n\n".join(quotes), "status": "skipped_no_text_support"})
    block["text"] = ""
    if result["include"]:
        info["status"] = "text_context_supported"
        block["text"] = "\n\n".join(value for value in (info.get("native_text", ""), info["description"]) if value)


def mark_unassessed_images(ir):
    ir["image_policy"] = IMAGE_POLICY
    for block in ir["blocks"]:
        if block["type"] == "image":
            if block["image"].get("text_context_assessed") is True:
                continue
            block["image"].update({"image_policy": IMAGE_POLICY, "visual_content_read": False,
                                   "text_context_assessed": False, "status": "skipped_context_not_assessed",
                                   "context_evidence": [], "description": ""})
            block["text"] = ""


def excluded_image(block, profile):
    info = block["image"]
    if info.get("image_policy") != IMAGE_POLICY:
        return False
    if info.get("status") == "skipped_no_text_support" and info.get("text_context_assessed") is True and not info.get("context_evidence"):
        return True
    return profile == "native" and info.get("status") == "skipped_context_not_assessed"
