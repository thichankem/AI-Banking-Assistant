"""Review heading proposals and arbitrate document-level extraction decisions."""
from copy import deepcopy
import json

from agent_banking.core.models import model_primary
from agent_banking.chunking.config.prompt_config import (
    ADJUDICATE_REPAIR_RETRY_PROMPT,
    ADJUDICATE_REPAIR_TASK,
    ADJUDICATE_REPAIR_TASK_OCR,
    EXTRACTION_ROUTE_QUESTION,
    HEADINGS_REVIEW_QUESTION,
    REPAIR_CONFIRM_QUESTION,
    STRUCTURE_SYSTEM_PROMPT,
    render_prompt,
)
from agent_banking.chunking.config.settings import get_config_section
from .image_context import IMAGE_POLICY, RESOLVED_IMAGE_STATUSES, source_text as context_source_text
from ..models.text_client import CloudAttempts, MAX_CLOUD_ATTEMPTS, ordered_parallel, validate_decisions
from .evidence import evidence_preview
from .model_warnings import optional_model_call, record_model_warning


_REVIEW_CFG = get_config_section("enrichment")["review"]
_ADJ_CFG = _REVIEW_CFG["adjudication"]
ROUTES = tuple(_REVIEW_CFG["routes"])
_HEADING_REVIEW_BATCH = _REVIEW_CFG["heading_review_batch"]
_REVIEW_WINDOW_BYTES = _REVIEW_CFG["window_bytes"]


def review_headings(enricher, ir, cloud_json):
    by_id = {block["id"]: block for block in ir["blocks"]}
    changed = [proposal for proposal in ir.get("structure_proposals", [])
               if proposal["native_level"] != proposal["proposed_level"]]
    reviews = []
    if changed:
        endpoint = optional_model_call(ir, "heading_review", lambda: model_primary("chunking.decision"))
    else:
        endpoint = None
    heading_preview = _REVIEW_CFG["heading_text_preview_chars"]
    for start in range(0, len(changed), _HEADING_REVIEW_BATCH):
        batch = changed[start:start + _HEADING_REVIEW_BATCH]
        evidence = []
        questions = {}
        for index, proposal in enumerate(batch):
            block = by_id[proposal["block_id"]]
            evidence.append({**proposal, "text": evidence_preview(block["text"], heading_preview),
                             "native_region": block["provenance"].get("region", "body")})
            for key in ("heading_format", "heading_evidence"):
                if key in block:
                    evidence[-1][key] = block[key]
            q_spec = deepcopy(HEADINGS_REVIEW_QUESTION)
            q_spec["instructions"] = render_prompt(q_spec["instructions"], index=index)
            questions["heading_" + str(index)] = q_spec
        def operation():
            return cloud_json(endpoint.url.rstrip("/") + "/systemone", endpoint.api_key,
                              {"model": endpoint.model, "state": {"candidates": evidence}, "questions": questions},
                              timeout=_REVIEW_CFG["decision_timeout"])
        result = None
        if endpoint is not None:
            result = enricher._optional_cached(ir, "heading_review", endpoint.model, {"state": evidence, "questions": questions},
                                               operation, lambda result: validate_decisions(result, questions))
        for index, proposal in enumerate(batch):
            answer = (result or {}).get("answers", {}).get("heading_" + str(index), {}).get("choice", "KEEP_NATIVE")
            if answer == "KEEP_NATIVE":
                block = by_id[proposal["block_id"]]
                block["level"] = proposal["native_level"]
                block["type"] = "paragraph"
                if block["level"]:
                    block["type"] = "heading"
            reviews.append({**proposal, "decision": answer})
    ir["heading_decisions"] = reviews


def _source_observations(ir):
    by_id = {block["id"]: block for block in ir["blocks"]}
    max_evidence = _REVIEW_CFG["observation_max_evidence"]
    chunk_step = _REVIEW_CFG["observation_chunk_step"]
    for warning in ir.get("source_review", []):
        summary = {"code": evidence_preview(str(warning.get("code", "")), _REVIEW_CFG["observation_code_chars"]),
                   "reason": evidence_preview(str(warning.get("reason", "")), _REVIEW_CFG["observation_reason_chars"])}
        evidence = []
        for identifier in dict.fromkeys(warning.get("block_ids", [])):
            block = by_id.get(identifier)
            if block is not None:
                text = context_source_text(block)
                evidence.append({"id": identifier, "type": block["type"],
                                 "native_text": evidence_preview(text, _REVIEW_CFG["observation_native_text_chars"]),
                                 "total_source_characters": len(text)})
        if len(evidence) <= max_evidence:
            yield {"unverified_model_observation": summary, "native_evidence": evidence}
        else:
            for start in range(1, len(evidence), chunk_step):
                yield {"unverified_model_observation": summary,
                       "native_evidence": evidence[:1] + evidence[start:start + chunk_step]}


def _review_state(ir):
    coverage = ir.get("deterministic_coverage", {})
    err_chars = _REVIEW_CFG["structural_error_code_chars"]
    max_errs = _REVIEW_CFG["max_structural_errors"]
    state = {"source_units": len(ir["units"]), "image_statuses": sorted({
                block["image"].get("status", "unresolved") for block in ir["blocks"] if block["type"] == "image"}),
             "image_policy": ir.get("image_policy", IMAGE_POLICY),
             "deterministic_coverage": {key: coverage[key] for key in (
                 "status", "source_units", "covered_units", "accounted_units", "oversized_passages"
             ) if key in coverage},
             "structural_errors": [evidence_preview(warning["code"], err_chars) for warning in ir["warnings"][:max_errs]],
             "heading_changes_reviewed": len(ir.get("heading_decisions", []))}
    return state


def _observation_windows(ir, state):
    def size(observations):
        return len(json.dumps({**state, "source_observations": observations}, ensure_ascii=False).encode("utf-8"))

    compact_chars = _REVIEW_CFG["observation_compact_text_chars"]
    windows = []
    pending = []
    for observation in _source_observations(ir):
        if size([observation]) > _REVIEW_WINDOW_BYTES:
            observation = {**observation, "native_evidence": [
                {**item, "native_text": evidence_preview(item["native_text"], compact_chars)}
                for item in observation["native_evidence"]
            ]}
        if pending and size(pending + [observation]) > _REVIEW_WINDOW_BYTES:
            windows.append(pending)
            pending = []
        pending.append(observation)
    if pending or not windows:
        windows.append(pending)
    return windows


def _route(result):
    return result["answers"]["review_route"]["choice"]


def review_extraction(enricher, ir, cloud_json):
    endpoint = optional_model_call(ir, "review_route", lambda: model_primary("chunking.decision"))
    if endpoint is None:
        ir["decision_review"] = {"status": "unavailable"}
        return
    model = endpoint.model
    state = _review_state(ir)
    route_spec = deepcopy(EXTRACTION_ROUTE_QUESTION["review_route"])
    ocr_instructions = route_spec.pop("ocr_instructions", "")
    questions = {"review_route": route_spec}
    if ir.get("page_reviews"):
        state["completed_visual_review_count"] = sum(
            page.get("content") != "unavailable" for page in ir["page_reviews"]
        )
        failed_pages = ir.get("statistics", {}).get("ocr_failed_page_numbers", [])
        state["ocr_failed_page_count"] = len(failed_pages)
        state["ocr_failed_page_numbers"] = failed_pages[:_REVIEW_CFG["max_ocr_failed_pages_preview"]]
        questions["review_route"]["instructions"] = ocr_instructions
    windows = _observation_windows(ir, state)

    def route(window):
        evidence = {**state, "source_observations": window}

        def operation():
            return cloud_json(endpoint.url.rstrip("/") + "/systemone", endpoint.api_key,
                              {"model": model, "state": evidence, "questions": questions},
                              timeout=_REVIEW_CFG["decision_timeout"])
        return enricher._optional_cached(ir, "review_route", model, {"state": evidence, "questions": questions}, operation,
                                        lambda result: validate_decisions(result, questions))
    results = ordered_parallel(route, windows, enricher.workers)
    final = []
    for value, window in zip(results, windows):
        if value is None:
            continue
        if _route(value) == "REPAIR_EXTRACTION" and enricher.use_llm and ir.get("deterministic_coverage"):
            adjudicated = optional_model_call(ir, "extraction_adjudication", lambda: enricher._adjudicate_repair(ir, value, state, window))
            if adjudicated is not None:
                value = adjudicated
        final.append(value)
    if not final:
        ir["decision_review"] = {"status": "unavailable"}
        return
    choice = max((_route(value) for value in final), key=ROUTES.index)
    result = final[0]
    if len(final) > 1:
        result = {"model": model, "aggregation": "strictest_route_across_all_evidence_windows",
                  "answers": {"review_route": {"choice": choice}}, "window_reviews": final}
    ir["decision_review"] = result
    if choice == "REPAIR_EXTRACTION":
        cov = ir.get("deterministic_coverage") or {}
        has_missing = any(cov.get(k) for k in ("missing_units", "missing_assets", "missing_image_occurrences", "oversized_passages"))
        has_warnings = any(w.get("requires_review") for w in ir.get("warnings", []))
        unresolved_imgs = any(b.get("image", {}).get("status") not in RESOLVED_IMAGE_STATUSES
                              for b in ir.get("blocks", []) if b.get("type") == "image")
        if cov.get("status") == "verified_source_reconstruction" and not has_missing and not has_warnings and not unresolved_imgs:
            result["status"] = "advisory"
        else:
            result["status"] = "repair_required"
        record_model_warning(ir, "review_route", "cloud_requested_repair")


def adjudicate_repair(enricher, ir, initial, state, observations, cloud_json):
    """Resolve model/deterministic disagreement; never bypass a failed gate."""
    coverage = ir["deterministic_coverage"]
    if (coverage.get("status") != "verified_source_reconstruction"
            or any(coverage.get(key) for key in (
                "missing_units", "missing_assets", "missing_image_occurrences", "oversized_passages"
            ))
            or any(warning.get("requires_review") for warning in ir["warnings"])):
        return initial
    if any(block["image"].get("status") not in RESOLVED_IMAGE_STATUSES for block in ir["blocks"] if block["type"] == "image"):
        return initial
    images_by_status = {}
    max_examples = _ADJ_CFG["max_examples_per_status"]
    example_chars = _ADJ_CFG["example_reason_chars"]
    for block in ir["blocks"]:
        if block["type"] == "image":
            info = block["image"]
            status = info["status"]
            group = images_by_status.setdefault(status, {"total": 0, "examples": []})
            group["total"] += 1
            if len(group["examples"]) < max_examples:
                group["examples"].append({"block_id": block["id"],
                                          "reason": evidence_preview(info.get("context_reason", ""), example_chars)})
    task = ADJUDICATE_REPAIR_TASK
    if ir.get("page_reviews"):
        task += " " + ADJUDICATE_REPAIR_TASK_OCR
    evidence = {**state, "source_observations": observations, "image_context_decisions": images_by_status,
                "initial_decision": {"choice": _route(initial), "reason": evidence_preview(
                    str(initial["answers"]["review_route"].get("reason", "")), _ADJ_CFG["initial_reason_chars"]
                )},
                "task": task}
    identifiers = {block["id"] for block in ir["blocks"]}
    def validate(value):
        if (not isinstance(value, dict) or value.get("route") not in ROUTES
                or not isinstance(value.get("reason"), str) or not value["reason"].strip()):
            raise ValueError("Invalid extraction adjudication")
        entries = value.get("extraction_evidence")
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict) or not isinstance(entry.get("block_id"), str)
            or entry["block_id"] not in identifiers or not isinstance(entry.get("reason"), str)
            or not entry["reason"].strip() for entry in entries
        ):
            raise ValueError("Adjudication evidence must reference actual source blocks")
        if value["route"] == "REPAIR_EXTRACTION" and not entries:
            raise ValueError("Extraction repair requires concrete grounded evidence")
    text_endpoint = model_primary("chunking.text")
    adj_max_tokens = _ADJ_CFG["max_tokens"]
    adj_temp = _ADJ_CFG["temperature"]
    def adjudicate():
        messages = [{"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}]
        attempt_budget = CloudAttempts()
        for attempt in range(MAX_CLOUD_ATTEMPTS):
            value = cloud_json(text_endpoint.url.rstrip("/") + "/chat/completions", text_endpoint.api_key,
                               {"model": text_endpoint.model, "messages": messages, "temperature": adj_temp,
                                "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": adj_max_tokens},
                               max_output_tokens=adj_max_tokens, attempt_budget=attempt_budget)
            try:
                validate(value)
                return value
            except ValueError as error:
                if attempt == MAX_CLOUD_ATTEMPTS - 1 or not attempt_budget.remaining:
                    raise
                messages.append({"role": "user", "content": render_prompt(ADJUDICATE_REPAIR_RETRY_PROMPT, error=error)})
    audit = enricher._cached("extraction_adjudication", text_endpoint.model, evidence, adjudicate, validate)
    confirm_spec = deepcopy(REPAIR_CONFIRM_QUESTION["confirm"])
    ocr_confirm = confirm_spec.pop("ocr_instructions", "")
    questions = {"confirm": confirm_spec}
    if ir.get("page_reviews"):
        questions["confirm"]["instructions"] = ocr_confirm
    confirmation_state = {"coverage": state["deterministic_coverage"],
                          "adjudication": {"route": audit["route"], "reason": evidence_preview(audit["reason"], _ADJ_CFG["audit_reason_chars"]),
                                           "extraction_evidence": [
                                               {"block_id": entry["block_id"], "reason": evidence_preview(entry["reason"], _ADJ_CFG["entry_reason_chars"])}
                                               for entry in audit["extraction_evidence"][:_ADJ_CFG["audit_evidence_preview"]]
                                           ]},
                          "image_statuses": state["image_statuses"]}
    decision_endpoint = model_primary("chunking.decision")
    decision_model = decision_endpoint.model
    def confirm():
        return cloud_json(decision_endpoint.url.rstrip("/") + "/systemone", decision_endpoint.api_key,
                          {"model": decision_model, "state": confirmation_state, "questions": questions},
                          timeout=_REVIEW_CFG["decision_timeout"])
    confirmed = enricher._cached("repair_confirmation", decision_model, {"state": confirmation_state, "questions": questions}, confirm,
                             lambda value: validate_decisions(value, questions))
    route = "REPAIR_EXTRACTION"
    if confirmed["answers"]["confirm"]["choice"] == "ACCEPT_ADJUDICATION":
        route = audit["route"]
    return {"model": decision_model, "answers": {"review_route": {"choice": route}},
            "initial_review": initial, "adjudication": audit, "confirmation": confirmed,
            "aggregation": "grounded_adjudication_confirmed_by_decision_model"}
