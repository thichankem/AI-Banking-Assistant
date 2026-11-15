"""Ask the configured text model for windowed structural annotations."""
import json
import logging
import os

from agent_banking.core.fallback import ModelFallbackError
from agent_banking.core.models import model_primary
from ..config.prompt_config import STRUCTURE_SYSTEM_PROMPT
from .image_context import IMAGE_POLICY, source_text
from .language import vietnamese_projection
from .model_client import IncompleteModelResponse, ordered_parallel
from .model_evidence import evidence_preview, table_preview
from .structure_annotations import (
    ANNOTATION_ARRAYS,
    OptionalContextReferenceError,
    apply_annotations,
    validate_annotations,
)


logger = logging.getLogger("chunking.models")


def structure_request_options(model, json_mode=None):
    maximum = int(os.getenv("CHUNKING_LLM_MAX_TOKENS", "65536"))
    if maximum <= 0:
        raise ValueError("CHUNKING_LLM_MAX_TOKENS must be positive")
    if json_mode is None:
        names = os.getenv("MODELS_DISABLE_JSON_MODE", os.getenv("MODEL", ""))
        disabled = [name.strip().casefold() for name in names.split(",") if name.strip()]
        normalized = model.casefold()
        json_mode = "reasoner" not in normalized and not any(name in normalized for name in disabled)
    options = {"temperature": 0.1, "max_tokens": maximum}
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    else:
        options["chat_template_kwargs"] = {"enable_thinking": False}
    return options


def _source_preview(block, alias):
    location = block["provenance"]
    preview = {
        "id": alias, "type": block["type"], "text": evidence_preview(vietnamese_projection(block), 1200),
        "level": block.get("level", 0),
        "location": {key: location[key] for key in ("region", "page_index", "bbox", "xpath") if key in location},
    }
    if "xpath" in preview["location"]:
        preview["location"]["xpath"] = evidence_preview(preview["location"]["xpath"], 360)
    for key in ("heading_format", "heading_evidence", "authored_level"):
        if key in block:
            preview[key] = block[key]
    if block["type"] == "table":
        preview["grid"] = table_preview(block["table_data"]["grid"], rows=6)
    if block["type"] == "image":
        preview["image_context_status"] = block["image"].get("status", "unresolved")
    return preview


def _structure_state(ir):
    state = {"image_policy": ir.get("image_policy", IMAGE_POLICY), "task": (
        "Interpret document structure only. Native text and numbers are immutable. Return raw JSON: "
        "{heading_levels:[{block_id,level}], relations:[{from,to,type}], reading_order:[], "
        "contexts:[{block_id,source_block_ids:[]}], warnings:[{code,block_ids,reason}]}. "
        "Use existing block IDs only; heading level 1..7, or level 0 to correct an inferred heading back to body text. "
        "Preserve authored_level headings, and recognize headings across all document genres. "
        "Use heading_format and heading_evidence together with section wording and neighboring blocks. "
        "A uniform bold font alone does not establish a heading. "
        "Only propose changes when needed. Never infer missing articles, facts or dates. "
        "Image text comes only from supplied native surrounding text; skipped image blocks are intentionally empty. "
        "No pixel reading/OCR is expected. Do not flag skipped_no_text_support as missing extraction, invent visual facts or omit empty image IDs from reading_order. "
        "Link lead-ins/lists, question/answer, table/note/exception, continued content and cross-references. "
        "Relation type must be one of continues, answer_to, note_of, exception_to, refers_to. "
        "Return reading_order:[] to keep source order. Text and table excerpts in this request can be incomplete. "
        "For ambiguous short passages choose context SOURCE BLOCK IDs only; the backend will extract exact source text. "
        "Do not emit context text or quotes. Context sources must be text, heading, caption or table blocks, not image descriptions. "
        "Omit contexts when section titles already suffice. "
        "Report conflicting claims as source warnings; do not repair their text.")}
    if ir.get("page_reviews"):
        state["task"] += (
            " These blocks also contain completed, reviewed visual transcriptions. "
            "Their page/figure coordinates are available as source evidence. "
            "Retain visual labels and table values exactly; do not treat scan text as absent. "
            "Connect continued tables/paragraphs across pages with source-ID relations.")

    state["task"] += (
        " Annotate headings and context targets only for editable_block_ids. Surrounding blocks are context. "
        "Choose optional context sources from context_source_ids. Omit unsupported optional annotations."
    )
    return state


def _window_ranges(blocks):
    serialized_sizes = [len(json.dumps(block, ensure_ascii=False)) + 2 for block in blocks]
    start = 0
    preceding = None
    while start < len(blocks):
        end, size = start, 0
        while end < min(start + 12, len(blocks)):
            size += serialized_sizes[end]
            if end > start and size > 9000:
                break
            end += 1
        yield start, end, preceding
        for block in blocks[start:end]:
            if block["type"] == "heading":
                preceding = {"text": evidence_preview(block["text"], 300), "level": block["level"]}
        start = end


def _review_window(start, window, native_text, endpoint, cloud_json):
    messages = [{"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(window, ensure_ascii=False)}]
    json_mode = None
    decode_mode_changed = False
    for repair in range(5):
        options = structure_request_options(endpoint.model, json_mode)

        def configure_payload(payload, candidate):
            payload.pop("response_format", None)
            payload.pop("chat_template_kwargs", None)
            payload.update(structure_request_options(candidate.model, json_mode))

        try:
            result = cloud_json(endpoint.url.rstrip("/") + "/chat/completions", endpoint.api_key,
                                {"model": endpoint.model, "messages": messages, **options},
                                timeout=240, configure_payload=configure_payload)
        except ModelFallbackError as error:
            cause = error.last_error
            if (decode_mode_changed or repair == 4 or not isinstance(cause, ValueError)
                    or isinstance(cause, IncompleteModelResponse)):
                raise
            json_mode = "response_format" not in options
            decode_mode_changed = True
            logger.warning("Switching JSON mode for structure window %s after response decoding failed", start)
            continue
        try:
            return validate_annotations(result, window, native_text)
        except ValueError as error:
            if repair == 4:
                if isinstance(error, OptionalContextReferenceError):
                    return validate_annotations(result, window, native_text, quarantine_context=True)
                raise
            if any(key not in result for key in ANNOTATION_ARRAYS):
                json_mode = False
            logger.warning("Repairing structure window %s: %s", start, error)
            messages[2:] = [
                {"role": "assistant", "content": evidence_preview(json.dumps(result, ensure_ascii=False), 2400)},
                {"role": "user", "content": (
                    "Repair the JSON annotations. Validation error: " + str(error)
                    + ". Use only supplied IDs and return all arrays. "
                    'Schema: {"heading_levels":[],"relations":[],"reading_order":[],"contexts":[],"warnings":[]}.'
                )},
            ]
    raise RuntimeError("Structure repair exhausted")


def enrich_structure(enricher, ir, cloud_json):
    by_id = {block["id"]: block for block in ir["blocks"]}
    native_ids = {}
    blocks = []
    for block in ir["blocks"]:
        if block.get("foreign_language"):
            continue
        alias = "b" + str(len(blocks))
        native_ids[alias] = block["id"]
        blocks.append(_source_preview(block, alias))
    if not blocks:
        return

    def native_text(alias):
        return source_text(by_id[native_ids[alias]])

    endpoint = model_primary("chunking.text")
    state = _structure_state(ir)

    def review_window(bounds):
        start, end, preceding = bounds
        # Neighbors only supply context. Keep expensive grid previews in the editable core.
        before = [{key: block[key] for key in block if key != "grid"} for block in blocks[max(0, start - 1):start]]
        after = [{key: block[key] for key in block if key != "grid"} for block in blocks[end:end + 1]]
        core = blocks[start:end]
        selected = before + core + after
        window = {**state, "blocks": selected, "previous_heading": preceding,
                  "editable_block_ids": [block["id"] for block in core],
                  "context_source_ids": [block["id"] for block in selected
                                         if block.get("grid") or (block["type"] != "image" and block["text"].strip())]}
        result = enricher._cached(
            "grounded_structure_window", endpoint.model, window,
            lambda: _review_window(start, window, native_text, endpoint, cloud_json),
            lambda value: validate_annotations(value, window, native_text),
        )
        return start, result

    results = ordered_parallel(review_window, _window_ranges(blocks), enricher.workers)
    apply_annotations(ir, results, native_ids, by_id)
