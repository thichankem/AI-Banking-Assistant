"""Validate model annotations against their window and apply grounded metadata."""
from .image_context import source_text


ANNOTATION_ARRAYS = ("heading_levels", "relations", "contexts", "warnings", "reading_order")
RELATION_TYPES = ("continues", "answer_to", "note_of", "exception_to", "refers_to")


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
        if type(level) is not int or not 0 <= level <= 7:
            raise ValueError("Heading level must be 0..7")
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


def apply_annotations(ir, results, native_ids, by_id):
    proposed = {}
    warnings = {}
    related = {(item["from"], item["to"], item["type"]) for item in ir["relations"]}
    for start, result in results:
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
            block["type"] = "heading" if level else "paragraph"
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
    for index, block in enumerate(ir["blocks"]):
        block["reading_order"] = index
