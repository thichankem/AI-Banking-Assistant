"""Enrich native document blocks with cached cloud model annotations."""
from copy import deepcopy
import hashlib
import json
import logging

from agent_banking.core.models import chain_fingerprint, model_primary
from agent_banking.chunking.config.prompt_config import (
    IMAGE_CONTEXT_REPAIR_PROMPT,
    IMAGE_CONTEXT_TASK_PROMPT,
    PROMPT_VERSION,
    STRUCTURE_SYSTEM_PROMPT,
    TABLE_HEADER_CONFIRM_QUESTION,
    TABLE_HEADER_REPAIR_PROMPT,
    TABLE_HEADER_TASK_PROMPT,
    render_prompt,
)
from agent_banking.chunking.config.settings import get_config_section
from .review import adjudicate_repair, review_extraction, review_headings
from .model_warnings import optional_model_call
from .image_context import (
    CONTEXT_UNAVAILABLE,
    IMAGE_POLICY,
    RESOLVED_IMAGE_STATUSES,
    apply_context_selection,
    surrounding_sources,
    validate_context_selection,
)
from ..models.text_client import (
    CloudAttempts,
    MAX_CLOUD_ATTEMPTS,
    IncompleteModelResponse,
    _cloud_json,
    cloud_json,
    ordered_parallel,
    read_response,
    strict_json,
    validate_decisions,
)
from .evidence import evidence_preview, table_preview
from .annotations import OptionalContextReferenceError as OptionalContextReferenceError
from .structure import enrich_structure, structure_request_options


logger = logging.getLogger("chunking.models")
_ENRICH_CFG = get_config_section("enrichment")
_IMAGE_CFG = _ENRICH_CFG["image_context"]
_HEADER_CFG = _ENRICH_CFG["table_header_review"]
_DEFAULT_WORKERS = _ENRICH_CFG["default_workers"]
_DEFAULT_TEMP = _ENRICH_CFG["temperature"]
_CACHE_MAX_TOKENS = _ENRICH_CFG["cache_max_tokens"]
_DECISION_ROLES = tuple(_ENRICH_CFG["decision_roles"])


class ModelEnricher:
    def __init__(self, cache, vision_client=None, workers=_DEFAULT_WORKERS, use_llm=True, use_decision=True, coverage_validator=None, review_vision_client=None):
        self.cache = cache
        self.workers = max(1, int(workers))
        self.use_llm = use_llm
        self.use_decision = use_decision
        self.coverage_validator = coverage_validator
        self.events = []

    def _cached(self, role, model, input_data, operation, validator=None):
        use_decision = role in _DECISION_ROLES
        if use_decision:
            chain = "chunking.decision"
        else:
            chain = "chunking.text"
        namespace = {"chain": chain_fingerprint(chain)}
        if not use_decision:
            namespace["system_prompt"] = STRUCTURE_SYSTEM_PROMPT
            namespace["options"] = {"temperature": _DEFAULT_TEMP, "chat_template_kwargs": {"enable_thinking": False},
                                    "max_tokens": _CACHE_MAX_TOKENS}
        if role == "grounded_structure_window":
            namespace["options"] = structure_request_options(model)
        seed = json.dumps({"version": PROMPT_VERSION, "namespace": namespace, "role": role, "model": model, "input": input_data},
                          ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        result = self.cache.get(key)
        if result is not None:
            try:
                if not isinstance(result, dict):
                    raise ValueError("Invalid cached model result")
                if validator is not None:
                    validator(result)
            except (ValueError, TypeError, KeyError):
                self.cache.delete(key)
                result = None
        hit = result is not None
        if result is None:
            logger.info("Calling %s (%s)", role, model)
            result = operation()
            if not isinstance(result, dict):
                raise ValueError("Invalid structured model result")
            if validator is not None:
                validator(result)
            self.cache.put(key, result)
            logger.info("Completed %s (%s)", role, model)
        api = result.get("_api") or result.get("api") or result
        if not isinstance(api, dict):
            api = result
        event = {"role": role, "model": model, "resolved_model": api.get("model", model),
                 "cache_hit": hit, "cache_key": key}
        if hit:
            event["cached_result_usage"] = api.get("usage", {})
            event["request_usage"] = {}
        else:
            event["request_usage"] = api.get("usage", {})
        self.events.append(event)
        return deepcopy(result)

    def _optional_cached(self, ir, role, model, input_data, operation, validator=None):
        return optional_model_call(ir, role, lambda: self._cached(role, model, input_data, operation, validator))

    def enrich(self, ir, pages=None):
        self.events = []
        ir["image_policy"] = "ignore_standalone_images"
        reviewed_visual = bool(ir.get("page_reviews"))
        if reviewed_visual:
            ir["image_policy"] = _IMAGE_CFG["reviewed_visual_policy"]
        ir["layout_method"] = ir.get("extraction_method", "native_openxml_text_structure")
        images = [(index, block) for index, block in enumerate(ir["blocks"])
                  if (block["type"] == "image" and not reviewed_visual
                      and (block["image"].get("status") not in RESOLVED_IMAGE_STATUSES
                           or block["image"].get("text_context_assessed") is not True))]
        if images and not self.use_llm:
            raise ValueError("Image context assessment requires the text LLM")
        if self.use_llm:
            ordered_parallel(lambda pair: self._image_context(ir, pair[0], pair[1]), images, self.workers)
            self._table_headers(ir)
            self._structure(ir)
        if self.use_decision:
            self._review_headings(ir)
            if self.coverage_validator is not None:
                ir["deterministic_coverage"] = self.coverage_validator(ir)
                self._decision(ir)
        ir["model_events"] = self.events
        return ir

    def review(self, ir):
        self._decision(ir)

    def _image_context(self, ir, index, block):
        candidates = surrounding_sources(ir, index)
        if not candidates:
            apply_context_selection(block, [], {"include": False, "evidence": [],
                                                "reason": _IMAGE_CFG["no_text_reason"]}, "")
            return
        endpoint = optional_model_call(ir, "image_text_context", lambda: model_primary("chunking.text"))
        if endpoint is None:
            self._skip_unassessed_image(block)
            return
        model = endpoint.model
        source_limit = _IMAGE_CFG["source_preview_chars"]
        previews = [{**source, "text": source["text"][:source_limit]} for source in candidates]
        state = {"image_occurrence_id": block["id"], "policy": IMAGE_POLICY,
                 "sources": previews,
                 "task": IMAGE_CONTEXT_TASK_PROMPT}
        def validate(value):
            validate_context_selection(value, previews)
        def operation():
            messages = [{"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(state, ensure_ascii=False)}]
            attempt_budget = CloudAttempts()
            for attempt in range(MAX_CLOUD_ATTEMPTS):
                result = cloud_json(endpoint.url.rstrip("/") + "/chat/completions", endpoint.api_key,
                                    {"model": model, "messages": messages, "temperature": _DEFAULT_TEMP,
                                     "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": _CACHE_MAX_TOKENS},
                                    attempt_budget=attempt_budget)
                try:
                    validate(result)
                    return result
                except (ValueError, TypeError, KeyError) as error:
                    if attempt == MAX_CLOUD_ATTEMPTS - 1 or not attempt_budget.remaining:
                        raise
                    messages.extend([{"role": "assistant", "content": evidence_preview(json.dumps(result, ensure_ascii=False), _IMAGE_CFG["repair_preview_chars"])},
                                     {"role": "user", "content": render_prompt(IMAGE_CONTEXT_REPAIR_PROMPT, error=error)}])
        result = self._optional_cached(ir, "image_text_context", model, state, operation, validate)
        if result is None:
            self._skip_unassessed_image(block)
        else:
            apply_context_selection(block, candidates, result, model)

    @staticmethod
    def _skip_unassessed_image(block):
        apply_context_selection(block, [], {"include": False, "evidence": [],
                                "reason": _IMAGE_CFG["unavailable_reason"]}, "")
        block["image"].update(status=CONTEXT_UNAVAILABLE, text_context_assessed=False)

    def _table_headers(self, ir):
        """Resolve unmarked/multirow headers without generating any cell text."""
        for block in ir["blocks"]:
            if block["type"] != "table":
                continue
            table = block["table_data"]
            if (table["header_row_count"] or len(table["grid"]) < 2 or "row_provenance" in table
                    or table.get("header_occurrences")
                    or table.get("header_detection") in ("decision_model_headers", "model_selected_native_rows")
                    or "header_review" in table):
                continue
            maximum = min(_HEADER_CFG["max_header_rows"], len(table["grid"]) - 1)
            allowed = [count for count in range(maximum + 1) if not any(
                cell["row"] < count < cell["row"] + cell["row_span"] for cell in table["cells"]
            )]
            if allowed == [0]:
                continue
            endpoint = optional_model_call(ir, "table_header_review", lambda: model_primary("chunking.text"))
            if endpoint is None:
                continue
            model = endpoint.model
            max_cols = _HEADER_CFG["max_cell_columns"]
            cell_preview = _HEADER_CFG["cell_text_preview_chars"]
            state = {"grid_preview": table_preview(table["grid"], maximum + 1, native=True),
                     "cells": [{"row": cell["row"], "column": cell["column"],
                                "row_span": cell["row_span"], "col_span": cell.get("col_span", cell.get("column_span", 1)),
                                "text": evidence_preview(cell["text"], cell_preview)}
                               for cell in table["cells"] if cell["row"] <= maximum and cell["column"] < max_cols],
                     "allowed_header_row_counts": allowed,
                     "task": TABLE_HEADER_TASK_PROMPT}
            def validate(result):
                count = result.get("header_row_count")
                if type(count) is not int or count not in allowed:
                    raise ValueError("Table header count must be a supplied row count")
            def operation():
                messages = [{"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(state, ensure_ascii=False)}]
                attempt_budget = CloudAttempts()
                for attempt in range(MAX_CLOUD_ATTEMPTS):
                    result = cloud_json(endpoint.url.rstrip("/") + "/chat/completions", endpoint.api_key,
                                        {"model": model, "messages": messages, "temperature": _DEFAULT_TEMP,
                                         "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": _CACHE_MAX_TOKENS},
                                    attempt_budget=attempt_budget)
                    try:
                        validate(result)
                        return result
                    except ValueError:
                        if attempt == MAX_CLOUD_ATTEMPTS - 1 or not attempt_budget.remaining:
                            raise
                        messages.append({"role": "user", "content": TABLE_HEADER_REPAIR_PROMPT})
            result = self._optional_cached(ir, "table_header_review", model, state, operation, validate)
            if result is None:
                continue
            count = result["header_row_count"]
            review = {"proposed_count": count, "reason": result.get("reason", ""), "model": model}
            if self.use_decision and count:
                questions = deepcopy(TABLE_HEADER_CONFIRM_QUESTION)
                evidence = {"grid": state["grid_preview"], "cells": state["cells"], "proposed_count": count}
                decision_endpoint = optional_model_call(ir, "table_header_confirmation", lambda: model_primary("chunking.decision"))
                if decision_endpoint is None:
                    continue
                decision_model = decision_endpoint.model
                def decision():
                    return cloud_json(decision_endpoint.url.rstrip("/") + "/systemone", decision_endpoint.api_key,
                                      {"model": decision_model, "state": evidence, "questions": questions},
                                      timeout=_HEADER_CFG["decision_timeout"])
                checked = self._optional_cached(ir, "table_header_confirmation", decision_model,
                                                {"state": evidence, "questions": questions}, decision,
                                                lambda value: validate_decisions(value, questions))
                if checked is None:
                    continue
                review["decision"] = checked["answers"]["header"]["choice"]
                if review["decision"] == "KEEP_DATA":
                    count = 0
            table["header_review"] = review
            if count:
                table["headers"] = [" / ".join(dict.fromkeys(row[column] for row in table["grid"][:count] if row[column]))
                                    for column in range(len(table["headers"]))]
                table["rows"] = table["grid"][count:]
                if "header_grid" in table:
                    table["header_grid"] = table["grid"][:count]
                table["header_row_count"] = count
                table["header_detection"] = "model_selected_native_rows"

    def _structure(self, ir):
        enrich_structure(self, ir, cloud_json)

    def _review_headings(self, ir):
        review_headings(self, ir, cloud_json)

    def _decision(self, ir):
        review_extraction(self, ir, cloud_json)

    def _adjudicate_repair(self, ir, initial, state, observations):
        return adjudicate_repair(self, ir, initial, state, observations, cloud_json)
