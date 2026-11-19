"""Repair invalid visual table structure and verify it against page pixels."""
from copy import deepcopy
import hashlib
import json

from agent_banking.chunking.config.prompt_config import (
    TABLE_ADJUDICATION_QUESTION,
    TABLE_PIXEL_AUDIT_PROMPT,
    TABLE_REPAIR_RETRY_PROMPT,
    TABLE_REPAIR_SYSTEM_PROMPT,
    render_prompt,
)
from agent_banking.chunking.config.settings import get_config_section
from .layout import compact
from .tables import table_from_html, validate_table

_PDF_CFG = get_config_section("extraction")["pdf"]


def numeric_signature(text):
    """Preserve ordered numeric lexemes during font reconciliation, never cuts."""
    numbers = []
    current = ""
    for index, character in enumerate(text):
        signed_start = character in "+-−" and index + 1 < len(text) and text[index + 1].isdecimal()
        if character.isdecimal() or signed_start or current and character in ".,%+-−":
            current += character
        else:
            if current:
                numbers.append(current.rstrip(".,"))
                current = ""
    if current:
        numbers.append(current.rstrip(".,"))
    return numbers


def repair_tables(layout, pixels, reviewer, asset_dir, use_decision):
    """Text-model schema repair grounded in raw VLM cells, then pixel audit.

    This is model interpretation of table structure, never padding ragged
    rows, dropping cells or manufacturing values with formatting heuristics.
    """
    from ...models.text_client import cloud_json
    from agent_banking.core.models import model_primary
    for element in layout.get("elements", []):
        if element.get("type") != "table":
            continue
        if element.get("table_html") is not None:
            element["table_data"] = table_from_html(element["table_html"], element.get("header_row_count"))
        original = element.get("table_data")
        try:
            validate_table(original, require_complete=True)
            continue
        except (ValueError, TypeError, KeyError):
            pass
        endpoint = model_primary("chunking.text")
        labels = []
        def collect(value):
            if isinstance(value, str):
                labels.append(value)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
        collect(original)
        system = TABLE_REPAIR_SYSTEM_PROMPT
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps({"raw_table": original}, ensure_ascii=False)}]
        repair_attempts = _PDF_CFG["table_repair_attempts"]
        repair_temp = _PDF_CFG["table_repair_temperature"]
        repair_max_tokens = _PDF_CFG["table_repair_max_tokens"]
        enable_thinking = _PDF_CFG["table_repair_enable_thinking"]
        for repair_attempt in range(repair_attempts):
            value = cloud_json(endpoint.url.rstrip("/") + "/chat/completions", endpoint.api_key,
                {"model": endpoint.model, "messages": messages,
                 "temperature": repair_temp, "chat_template_kwargs": {"enable_thinking": enable_thinking},
                 "max_tokens": repair_max_tokens})
            repaired = value.get("table_data")
            try:
                validate_table(repaired, require_complete=True)
                break
            except (ValueError, TypeError, KeyError) as error:
                if repair_attempt == repair_attempts - 1:
                    from agent_banking.chunking.publication_io import atomic_json
                    digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
                    atomic_json(asset_dir / "review_failures" / (digest + "_table.json"), {"raw_table": original, "proposed": repaired, "error": str(error)})
                    raise
                messages.extend([{"role": "assistant", "content": json.dumps(value, ensure_ascii=False)},
                    {"role": "user", "content": render_prompt(TABLE_REPAIR_RETRY_PROMPT, error=error)}])
        grid = repaired["header_grid"] + repaired["rows"]
        strings = [cell for row in grid for cell in row if cell]
        if any(not any(cell in source for source in labels) for cell in strings):
            raise ValueError("Table repair invented source text")
        before = {number for source in labels for number in numeric_signature(source)}
        after = {number for cell_text in strings for number in numeric_signature(cell_text)}
        if before != after:
            raise ValueError("Table repair changed or dropped numeric values")
        combined = compact("\n".join(strings))
        if any(compact(source) and compact(source) not in combined for source in labels):
            raise ValueError("Table repair dropped a source label or value")
        audit = reviewer.analyze_page_layout(
            pixels,
            render_prompt(TABLE_PIXEL_AUDIT_PROMPT, proposed_table=json.dumps(repaired, ensure_ascii=False)),
        )
        adjudication = None
        if isinstance(audit, dict):
            review = audit.get("review")
        else:
            review = None
        if not isinstance(review, dict):
            raise ValueError("Table pixel review must return a review object")
        if audit.get("uncertainties") == [] and review.get("verdict") == "unresolved":
            adjudication = _adjudicate_table_cells(repaired, audit, use_decision)
        accepted = audit.get("uncertainties") == [] and (
            review.get("verdict") == "accepted" or adjudication is not None)
        if not accepted:
            from agent_banking.chunking.publication_io import atomic_json
            digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
            atomic_json(asset_dir / "review_failures" / (digest + "_table_audit.json"),
                        {"raw_table": original, "proposed": repaired, "visual_review": audit})
            raise ValueError("Pixel review rejected structured table repair")
        element["table_data"] = repaired
        element["table_schema_repair"] = {"model": endpoint.model, "grounded_literal_values": True,
            "visual_audit": audit, "adjudication": adjudication,
            "raw_table_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()}


def _adjudicate_table_cells(proposed, audit, use_decision):
    """Resolve an inconclusive verdict only with complete matching cell OCR.

    A classifier never waives missing, contradictory or uncertain visual
    evidence. Every origin cell and every covered grid coordinate is checked
    before decision model can consider the disagreement.
    """
    if not use_decision:
        return None
    expected = {(cell["row"], cell["column"], cell["text"]) for cell in proposed["cells"]}
    observations = audit.get("elements")
    if not expected or not isinstance(observations, list) or any(
                           not isinstance(item, dict)
                           or not isinstance(item.get("row"), int) or isinstance(item.get("row"), bool)
                          or not isinstance(item.get("column"), int) or isinstance(item.get("column"), bool)
                          or not isinstance(item.get("text"), str) for item in observations):
        return None
    witnessed = {(item["row"], item["column"], item["text"]) for item in observations}
    if witnessed != expected:
        return None
    grid = proposed["header_grid"] + proposed["rows"]
    occupied = {(row, column) for cell in proposed["cells"]
                for row in range(cell["row"], cell["row"] + cell["row_span"])
                for column in range(cell["column"], cell["column"] + cell["column_span"])}
    if occupied != {(row, column) for row in range(len(grid)) for column in range(len(proposed["headers"]))}:
        return None
    from ...models.text_client import cloud_json, validate_decisions
    from agent_banking.core.models import model_primary
    endpoint = model_primary("chunking.decision")
    questions = deepcopy(TABLE_ADJUDICATION_QUESTION)
    result = cloud_json(endpoint.url.rstrip("/") + "/systemone", endpoint.api_key,
        {"model": endpoint.model, "state": {"proposed": proposed, "independent_visual_cells": observations,
         "initial_verdict": audit["review"], "all_cells_exact": True}, "questions": questions}, model_module="chunking.decision")
    validate_decisions(result, questions)
    if result["answers"]["table"]["choice"] == "ACCEPT":
        return {"model": endpoint.model, "decision": result, "basis": "complete_exact_independent_cell_witnesses"}
    return None
