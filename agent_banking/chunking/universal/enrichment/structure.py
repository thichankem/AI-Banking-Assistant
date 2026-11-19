"""Ask the configured text model for windowed structural annotations."""
import json
import logging
import os

from agent_banking.core.fallback import ModelFallbackError
from agent_banking.core.models import model_primary
from agent_banking.chunking.config.prompt_config import (
    STRUCTURE_REPAIR_PROMPT,
    STRUCTURE_SYSTEM_PROMPT,
    STRUCTURE_TASK_BASE,
    STRUCTURE_TASK_EDITABLE_SUFFIX,
    STRUCTURE_TASK_OCR_ADDITION,
    render_prompt,
)
from agent_banking.chunking.config.settings import get_config_section
from .image_context import IMAGE_POLICY, source_text
from ..text.language import vietnamese_projection
from ..models.text_client import CloudAttempts, MAX_CLOUD_ATTEMPTS, IncompleteModelResponse, ordered_parallel
from .evidence import evidence_preview, table_preview
from .model_warnings import optional_model_call
from .annotations import (
    ANNOTATION_ARRAYS,
    apply_annotations,
    filter_optional_annotations,
    validate_annotations,
)


logger = logging.getLogger("chunking.models")
_STRUCT_CFG = get_config_section("enrichment")["structure"]
_STRUCTURE_WINDOW_BLOCKS = _STRUCT_CFG["window_blocks"]
_STRUCTURE_WINDOW_BYTES = _STRUCT_CFG["window_bytes"]
_STRUCTURE_COMPLETION_TOKENS = _STRUCT_CFG["completion_tokens"]
_DEFAULT_LLM_MAX_TOKENS = _STRUCT_CFG["default_llm_max_tokens"]
_TEMPERATURE = _STRUCT_CFG["temperature"]
_REQUEST_TIMEOUT = _STRUCT_CFG["request_timeout"]


def structure_request_options(model, json_mode=None):
    maximum = int(os.getenv("CHUNKING_LLM_MAX_TOKENS", str(_DEFAULT_LLM_MAX_TOKENS)))
    if maximum <= 0:
        raise ValueError("CHUNKING_LLM_MAX_TOKENS must be positive")
    if json_mode is None:
        names = os.getenv("MODELS_DISABLE_JSON_MODE", "")
        disabled = [name.strip().casefold() for name in names.split(",") if name.strip()]
        normalized = model.casefold()
        json_mode = not any(name in normalized for name in disabled)
    options = {"temperature": _TEMPERATURE, "max_tokens": min(maximum, _STRUCTURE_COMPLETION_TOKENS)}
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    else:
        options["chat_template_kwargs"] = {"enable_thinking": False}
    return options


def _source_preview(block, alias):
    location = block["provenance"]
    preview = {
        "id": alias, "type": block["type"], "text": evidence_preview(vietnamese_projection(block), _STRUCT_CFG["text_preview_chars"]),
        "level": block.get("level", 0),
        "location": {key: location[key] for key in ("region", "page_index", "bbox", "xpath") if key in location},
    }
    if "xpath" in preview["location"]:
        preview["location"]["xpath"] = evidence_preview(preview["location"]["xpath"], _STRUCT_CFG["xpath_preview_chars"])
    for key in ("heading_format", "heading_evidence", "authored_level"):
        if key in block:
            preview[key] = block[key]
    if block["type"] == "table":
        cell_limit = _STRUCT_CFG["table_cell_preview_chars"]
        preview["grid"] = [[evidence_preview(cell, cell_limit) for cell in row]
                           for row in table_preview(block["table_data"]["grid"], rows=_STRUCT_CFG["table_preview_rows"])]
    if block["type"] == "image":
        preview["image_context_status"] = block["image"].get("status", "unresolved")
    return preview


def _structure_state(ir):
    task = STRUCTURE_TASK_BASE
    if ir.get("page_reviews"):
        task += " " + STRUCTURE_TASK_OCR_ADDITION
    task += " " + STRUCTURE_TASK_EDITABLE_SUFFIX
    return {"image_policy": ir.get("image_policy", IMAGE_POLICY), "task": task}


def _context_preview(block, text_limit):
    preview = {"id": block["id"], "type": block["type"], "level": block["level"],
               "text": evidence_preview(block["text"], text_limit)}
    if block.get("authored_level"):
        preview["authored_level"] = block["authored_level"]
    return preview


def _window_evidence(core, neighbors, preceding):
    before, after = neighbors
    selected = before + core + after
    return {"blocks": selected, "previous_heading": preceding,
            "editable_block_ids": [block["id"] for block in core],
            "context_source_ids": [block["id"] for block in selected
                                   if block.get("grid") or (block["type"] != "image" and block["text"].strip())]}


def _window_size(evidence):
    return len(json.dumps(evidence, ensure_ascii=False).encode("utf-8"))


def _structure_windows(blocks):
    start = 0
    preceding = None
    neighbor_chars = _STRUCT_CFG["neighbor_context_chars"]
    compact_chars = _STRUCT_CFG["compact_context_chars"]
    compact_rows = _STRUCT_CFG["compact_table_rows"]
    compact_cols = _STRUCT_CFG["compact_table_cols"]
    compact_cell = _STRUCT_CFG["compact_cell_chars"]
    heading_chars = _STRUCT_CFG["preceding_heading_chars"]
    while start < len(blocks):
        before = [_context_preview(block, neighbor_chars) for block in blocks[max(0, start - 1):start]]
        end = start
        window = None
        while end < min(start + _STRUCTURE_WINDOW_BLOCKS, len(blocks)):
            next_end = end + 1
            after = [_context_preview(block, neighbor_chars) for block in blocks[next_end:next_end + 1]]
            candidate = _window_evidence(blocks[start:next_end], (before, after), preceding)
            if _window_size(candidate) > _STRUCTURE_WINDOW_BYTES:
                break
            end, window = next_end, candidate
        if window is None:
            # Keep a large block editable in its own request without crowding its neighbors.
            block = blocks[start]
            compact = _context_preview(block, compact_chars)
            if block.get("grid"):
                compact["grid"] = [[evidence_preview(cell, compact_cell) for cell in row[:compact_cols]]
                                   for row in block["grid"][:compact_rows]]
            end = start + 1
            window = _window_evidence([compact], ([], []), preceding)
        yield start, window
        for block in blocks[start:end]:
            if block["type"] == "heading":
                preceding = {"text": evidence_preview(block["text"], heading_chars), "level": block["level"]}
        start = end


def _review_window(start, window, native_text, endpoint, cloud_json):
    messages = [{"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(window, ensure_ascii=False)}]
    json_mode = None
    decode_mode_changed = False
    attempt_budget = CloudAttempts()
    for repair in range(MAX_CLOUD_ATTEMPTS):
        options = structure_request_options(endpoint.model, json_mode)

        def configure_payload(payload, candidate):
            payload.pop("response_format", None)
            payload.pop("chat_template_kwargs", None)
            payload.update(structure_request_options(candidate.model, json_mode))

        try:
            result = cloud_json(endpoint.url.rstrip("/") + "/chat/completions", endpoint.api_key,
                                {"model": endpoint.model, "messages": messages, **options},
                                timeout=_REQUEST_TIMEOUT, max_output_tokens=options["max_tokens"],
                                configure_payload=configure_payload, attempt_budget=attempt_budget)
        except ModelFallbackError as error:
            cause = getattr(error, "last_error", None)
            if (decode_mode_changed or repair == MAX_CLOUD_ATTEMPTS - 1 or not attempt_budget.remaining or not isinstance(cause, ValueError)
                    or isinstance(cause, IncompleteModelResponse)):
                raise
            json_mode = "response_format" not in options
            decode_mode_changed = True
            logger.warning("Switching JSON mode for structure window %s after response decoding failed", start)
            continue
        try:
            return filter_optional_annotations(result, window, native_text)
        except ValueError as error:
            if repair == MAX_CLOUD_ATTEMPTS - 1 or not attempt_budget.remaining:
                raise
            if any(key not in result for key in ANNOTATION_ARRAYS):
                json_mode = False
            logger.warning("Repairing structure window %s: %s", start, error)
            messages[2:] = [
                {"role": "assistant", "content": evidence_preview(json.dumps(result, ensure_ascii=False), _STRUCT_CFG["repair_preview_chars"])},
                {"role": "user", "content": render_prompt(STRUCTURE_REPAIR_PROMPT, error=error)},
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

    endpoint = optional_model_call(ir, "grounded_structure_window", lambda: model_primary("chunking.text"))
    if endpoint is None:
        return
    state = _structure_state(ir)

    def review_window(item):
        start, evidence = item
        window = {**state, **evidence}
        result = enricher._optional_cached(
            ir, "grounded_structure_window", endpoint.model, window,
            lambda: _review_window(start, window, native_text, endpoint, cloud_json),
            lambda value: validate_annotations(value, window, native_text),
        )
        return start, result

    results = ordered_parallel(review_window, _structure_windows(blocks), enricher.workers)
    apply_annotations(ir, [(start, result) for start, result in results if result is not None], native_ids, by_id)
