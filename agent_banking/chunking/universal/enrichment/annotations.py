"""Validate model annotations against their window and apply grounded metadata."""
from agent_banking.chunking.config.settings import get_config_section, get_rules
from .image_context import source_text

_ENRICH_RULES = get_rules("enrichment_rules")
_DOCX_CFG = get_config_section("extraction")["docx"]
ANNOTATION_ARRAYS = tuple(_ENRICH_RULES["annotation_arrays"])
RELATION_TYPES = tuple(_ENRICH_RULES["relation_types"])
_MAX_HEADING_LEVEL = _DOCX_CFG["max_heading_level"]


class OptionalContextReferenceError(ValueError):
    """An optional ID-only context annotation references unavailable text."""


def _known(identifier, choices):
    return isinstance(identifier, str) and identifier in choices


def _validate_heading_levels(items, supplied, editable):
    levels = {}
    for item in items:
        identifier = item.get("block_id")
        level = item.get("level")
        if not _known(identifier, editable):
            raise ValueError("Heading target must be an editable block ID")
        if supplied[identifier]["type"] not in ("paragraph", "heading"):
            raise ValueError("Heading targets must be paragraphs or headings")
        if type(level) is not int or not 0 <= level <= _MAX_HEADING_LEVEL:
            raise ValueError("Heading level must be 0.." + str(_MAX_HEADING_LEVEL))
        if identifier in levels and levels[identifier] != level:
            raise ValueError("Conflicting heading levels for the same source ID")
        levels[identifier] = level


def _validate_context(item, editable, sources, native_text):
    target = item.get("block_id")
    if not _known(target, editable):
        raise ValueError("Context target must be an editable block ID")
    quotes = item.get("context_quotes", [])
    if not isinstance(quotes, list):
        raise ValueError("Context quotes must be an array")
    for quote in quotes:
        if not isinstance(quote, dict):
            raise ValueError("Context quote entries must be objects")
        identifier = quote.get("source_block_id")
        text = quote.get("text")
        if (identifier == target or not _known(identifier, sources) or not isinstance(text, str)
                or not text.strip() or text not in native_text(identifier)):
            raise ValueError("Context must quote a distinct verbatim source from the supplied window")
    identifiers = item.get("source_block_ids", [])
    if not isinstance(identifiers, list) or any(
        identifier == target or not _known(identifier, sources) for identifier in identifiers
    ):
        raise OptionalContextReferenceError("Context source must be a distinct ID from context_source_ids")


def validate_annotations(result, window, native_text, *, quarantine_context=False):
    if not isinstance(result, dict):
        raise ValueError("Structure response must be an object")
    for key in ANNOTATION_ARRAYS:
        if not isinstance(result.get(key), list):
            raise ValueError("Expected annotation array: " + key)
        if key != "reading_order" and any(not isinstance(item, dict) for item in result[key]):
            raise ValueError("Annotation entries must be objects: " + key)
    supplied = {block["id"]: block for block in window["blocks"]}
    editable = set(window["editable_block_ids"])
    _validate_heading_levels(result["heading_levels"], supplied, editable)
    reading_order = result["reading_order"]
    if reading_order and (any(not _known(identifier, supplied) for identifier in reading_order)
                          or len(reading_order) != len(supplied) or len(set(reading_order)) != len(supplied)):
        raise ValueError("Reading order must be empty or a permutation of supplied block IDs")
    for relation in result["relations"]:
        begin, end = relation.get("from"), relation.get("to")
        if begin == end or not _known(begin, supplied) or not _known(end, supplied):
            raise ValueError("Relation endpoints must be distinct supplied block IDs")
        if relation.get("type") not in RELATION_TYPES:
            raise ValueError("Unsupported relation type")
    for warning in result["warnings"]:
        identifiers = warning.get("block_ids", [])
        if not isinstance(identifiers, list) or any(not _known(identifier, supplied) for identifier in identifiers):
            raise ValueError("Warning block_ids must reference supplied source blocks")
        if any(not isinstance(warning.get(key), str) or not warning[key].strip() for key in ("code", "reason")):
            raise ValueError("Warnings require a code and a reason")
    sources = set(window["context_source_ids"])
    valid, rejected = [], []
    for item in result["contexts"]:
        try:
            _validate_context(item, editable, sources, native_text)
            valid.append(item)
        except OptionalContextReferenceError:
            if not quarantine_context or item.get("context_quotes"):
                raise
            rejected.append(item)
    if rejected:
        return {**result, "contexts": valid, "_rejected_optional_contexts": rejected}
    return result


def _clean_identifier(value):
    """Remove separator noise that models append to IDs, such as "b0," or " b3 "."""
    if not isinstance(value, str):
        return value
    return value.strip().strip(",;:.").strip().strip("\"'").strip()


def _clean_identifier_list(values):
    if not isinstance(values, list):
        return values
    return [_clean_identifier(value) for value in values]


def normalize_annotation_response(result):
    """Fill omitted empty arrays and clean ID punctuation before validation.

    Models frequently omit empty optional arrays or emit IDs with trailing commas.
    When at least one annotation array is present, missing optional arrays are
    defaulted to [] instead of costing another cloud round trip; completely empty
    or unrelated objects (such as {}) still trigger schema repair.
    """
    if not isinstance(result, dict):
        return result
    if not any(isinstance(result.get(key), list) for key in ANNOTATION_ARRAYS):
        return result
    clean = dict(result)
    for key in ANNOTATION_ARRAYS:
        if key not in clean:
            clean[key] = []
    clean["reading_order"] = _clean_identifier_list(clean["reading_order"])
    for key, fields, list_fields in (("heading_levels", ("block_id",), ()),
                                     ("relations", ("from", "to"), ()),
                                     ("contexts", ("block_id",), ("source_block_ids",)),
                                     ("warnings", (), ("block_ids",))):
        items = clean[key]
        if not isinstance(items, list):
            continue
        repaired = []
        for item in items:
            if not isinstance(item, dict):
                repaired.append(item)
                continue
            item = dict(item)
            for field in fields:
                item[field] = _clean_identifier(item.get(field))
            for field in list_fields:
                if field in item:
                    item[field] = _clean_identifier_list(item[field])
            if key == "contexts" and isinstance(item.get("context_quotes"), list):
                quotes = []
                for quote in item["context_quotes"]:
                    if isinstance(quote, dict):
                        quote = {**quote, "source_block_id": _clean_identifier(quote.get("source_block_id"))}
                    quotes.append(quote)
                item["context_quotes"] = quotes
            if key == "heading_levels" and isinstance(item.get("level"), str) and item["level"].strip().isdigit():
                item["level"] = int(item["level"].strip())
            repaired.append(item)
        clean[key] = repaired
    return clean


def filter_optional_annotations(result, window, native_text):
    """Keep grounded annotations even if the model also returned bad references."""
    result = normalize_annotation_response(result)
    if not isinstance(result, dict) or any(not isinstance(result.get(key), list) for key in ANNOTATION_ARRAYS):
        raise ValueError("Structure response requires all annotation arrays")
    empty = dict.fromkeys(ANNOTATION_ARRAYS, [])
    clean = {**result, **empty}
    rejected = []
    levels = {}
    for key in ANNOTATION_ARRAYS:
        items = result[key]
        if key == "reading_order" and items:
            candidates = [items]
        else:
            candidates = items
        accepted = []
        for item in candidates:
            if key == "reading_order":
                entry = item
            else:
                entry = [item]
            try:
                validate_annotations({**empty, key: entry}, window, native_text)
            except ValueError as error:
                rejected.append({"kind": key, "proposal": item, "reason": str(error)})
                continue
            if key == "heading_levels":
                levels.setdefault(item["block_id"], set()).add(item["level"])
            accepted.append(item)
        if key == "reading_order" and accepted:
            clean[key] = accepted[0]
        else:
            clean[key] = accepted
    headings = []
    for item in clean["heading_levels"]:
        if len(levels[item["block_id"]]) == 1:
            headings.append(item)
        else:
            rejected.append({"kind": "heading_levels", "proposal": item, "reason": "conflicting levels"})
    clean["heading_levels"] = headings
    if rejected:
        clean["_rejected_annotations"] = rejected
    return validate_annotations(clean, window, native_text)


def apply_annotations(ir, results, native_ids, by_id):
    proposed = {}
    warnings = {}
    related = {(item["from"], item["to"], item["type"]) for item in ir["relations"]}
    for start, result in results:
        if result.get("_rejected_annotations"):
            ir.setdefault("annotation_rejections", []).append({
                "window_start": start, "role": "structure", "proposals": result["_rejected_annotations"],
            })
        if result.get("_rejected_optional_contexts"):
            ir.setdefault("annotation_rejections", []).append({
                "window_start": start,
                "role": "optional_context",
                "reason": "invalid_source_reference_after_schema_repair",
                "proposals": result["_rejected_optional_contexts"],
            })
        for item in result["heading_levels"]:
            identifier = native_ids[item["block_id"]]
            block = by_id[identifier]
            if identifier in proposed or block.get("authored_level") or block.get("heading_format") == "openxml_outline":
                continue
            level = item["level"]
            proposed[identifier] = {"block_id": identifier, "native_level": block.get("level", 0),
                                    "proposed_level": level}
            if level:
                block["type"] = "heading"
            else:
                block["type"] = "paragraph"
            block["level"] = level
        for relation in result["relations"]:
            key = (native_ids[relation["from"]], native_ids[relation["to"]], relation["type"])
            if key not in related:
                ir["relations"].append({"from": key[0], "to": key[1], "type": key[2]})
                related.add(key)
        for warning in result["warnings"]:
            ids = list(dict.fromkeys(native_ids[alias] for alias in warning.get("block_ids", [])))
            key = (warning["code"], warning["reason"], frozenset(ids))
            warnings.setdefault(key, {"code": warning["code"], "reason": warning["reason"], "block_ids": ids})
        for item in result["contexts"]:
            target = by_id[native_ids[item["block_id"]]]
            quotes = target.setdefault("context_quotes", [])
            candidates = []
            for alias in dict.fromkeys(item.get("source_block_ids", [])):
                identifier = native_ids[alias]
                candidates.append({"source_block_id": identifier, "text": source_text(by_id[identifier])})
            for quote in item.get("context_quotes", []):
                candidates.append({"source_block_id": native_ids[quote["source_block_id"]], "text": quote["text"]})
            for quote in candidates:
                if quote not in quotes:
                    quotes.append(quote)
    ir["structure_proposals"] = list(proposed.values())
    ir["source_review"] = list(warnings.values())
    if ir.get("page_reviews"):
        _apply_reading_orders(ir, results, native_ids)
    for index, block in enumerate(ir["blocks"]):
        block["reading_order"] = index


def _apply_reading_orders(ir, results, native_ids):
    """Apply model reading orders on OCR pages, only inside one contiguous page run.

    Authored DOCX order is never changed. For OCR pages the model may fix column
    order, but only for blocks on the same page that are adjacent in the IR, so a
    window can never move content across pages or across another window.
    """
    applied = []
    for start, result in results:
        order = result.get("reading_order") or []
        if not order:
            continue
        identifiers = [native_ids[alias] for alias in order if alias in native_ids]
        if len(identifiers) != len(order):
            continue
        positions = {block["id"]: index for index, block in enumerate(ir["blocks"])}
        if any(identifier not in positions for identifier in identifiers):
            continue
        indexes = sorted(positions[identifier] for identifier in identifiers)
        if indexes[-1] - indexes[0] + 1 != len(indexes):
            continue
        pages = {ir["blocks"][index]["provenance"].get("page_index") for index in indexes}
        if len(pages) != 1:
            continue
        reordered = [ir["blocks"][positions[identifier]] for identifier in identifiers]
        ir["blocks"][indexes[0]:indexes[-1] + 1] = reordered
        applied.append(start)
    if applied:
        ir["reading_order_windows"] = applied
