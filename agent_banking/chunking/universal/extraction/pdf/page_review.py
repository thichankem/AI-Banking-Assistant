"""Read PDF pixels through the configured cloud vision candidates."""
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import io
import json

from agent_banking.chunking.config.prompt_config import (
    CLOUD_OCR_PAGE_PROMPT as SCAN_PAGE_PROMPT,
    PDF_PAGE_REPAIR_PROMPT,
    PDF_PROMPT_VERSION,
    PDF_REGION_CROP_INSTRUCTION,
    PDF_SCHEMA_VERSION,
)
from agent_banking.chunking.config.settings import get_config_section, get_rules
from agent_banking.core.numeric_tokens import extract_canonical_numeric_tokens
from agent_banking.core.search_text import search_key
from agent_banking.tracing.tracer import annotate_span, trace_stage
from .layout import (
    has_meaningful_layout, layout_text, merge_region_observation, normalize_visual_layout, reading_order,
    validate_page, validate_scan_pixels,
)
from .tables import table_from_html, validate_table

_PDF_CFG = get_config_section("extraction")["pdf"]
_PDF_RULES = get_rules("pdf")

OCR_MAX_OUTPUT_TOKENS = _PDF_CFG["ocr_max_output_tokens"]
_SCHEMA_ISSUES = set(_PDF_RULES["schema_issues"])
_ACCEPTED_VERDICTS = tuple(_PDF_RULES["accepted_verdicts"])
_REPAIR_INSTRUCTION = "\n" + PDF_PAGE_REPAIR_PROMPT
_MAX_CROP_DEPTH = _PDF_CFG["max_crop_depth"]
_CROP_PAD_RATIO = _PDF_CFG["crop_pad_ratio"]



class CloudOCRUnavailable(ValueError):
    """No usable cloud transcription was returned for this page."""


def _retains_partial_evidence(previous, candidate):
    if candidate["visual_review"]["incomplete"]:
        return False
    old_text, new_text = layout_text(previous), layout_text(candidate)
    old_terms = set(search_key(old_text).split())
    new_terms = set(search_key(new_text).split())
    old_tables = sum(element["type"] == "table" for element in previous["elements"])
    new_tables = sum(element["type"] == "table" for element in candidate["elements"])
    financial = set(extract_canonical_numeric_tokens(old_text, max_tokens=128))
    return (new_tables >= old_tables and old_terms <= new_terms
            and financial <= set(extract_canonical_numeric_tokens(new_text, max_tokens=128)))


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("tr", "p", "div", "li", "br"):
            self.parts.append("\n")
        elif tag in ("td", "th") and self.parts and not self.parts[-1].endswith(("\n", " | ")):
            self.parts.append(" | ")

    def handle_data(self, data):
        text = " ".join(data.split())
        if text:
            if self.parts and not self.parts[-1].endswith(("\n", " | ")):
                self.parts.append(" ")
            self.parts.append(text)


def _strip_html_text(markup):
    if not isinstance(markup, str) or not markup.strip():
        return ""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:
        return ""
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _needs_table_repair(raw):
    if not isinstance(raw, dict):
        return False
    for element in raw.get("elements") or ():
        if not isinstance(element, dict) or element.get("type") != "table":
            continue
        try:
            if element.get("table_html") is not None:
                parsed = table_from_html(element["table_html"], element.get("header_row_count"))
            else:
                parsed = element.get("table_data")
            validate_table(parsed, require_complete=True)
        except (ValueError, TypeError, KeyError):
            return True
    return False


class PdfVisualReviewer:
    def __init__(self, asset_dir, cache, client, reviewer, use_decision):
        self.asset_dir = asset_dir
        self.cache = cache
        self.vision_client = client
        self.review_client = reviewer
        self.use_decision = bool(use_decision)

    @trace_stage("document.vision")
    def read(self, pixels, native, width, height, region_depth=0, *, allow_ocr_additions=False, attempt_budget=None,
             review_incomplete=False, prefer_halves=False):
        from ...models.text_client import CloudAttempts
        from ...models.vision_client import DocumentVisionClient, QwenVisionClient, VisionPageOverflow
        from agent_banking.core.fallback import ModelFallbackError
        from agent_banking.core.models import auto_routing_enabled, chain_fingerprint, get_model_chain

        primary = self.vision_client or DocumentVisionClient(max_tokens=OCR_MAX_OUTPUT_TOKENS)
        backup = self.review_client
        if (backup is None and self.vision_client is None
                and any(endpoint.model for endpoint in get_model_chain("chunking.vision_review"))):
            backup = DocumentVisionClient(model_module="chunking.vision_review", max_tokens=OCR_MAX_OUTPUT_TOKENS)
        if backup is None or backup is primary:
            clients = (primary,)
        else:
            clients = (primary, backup)
        routed = auto_routing_enabled()
        if routed and region_depth and len(clients) == 2:
            clients = (backup, primary)
        identity = {
            "schema": PDF_SCHEMA_VERSION, "version": PDF_PROMPT_VERSION, "prompt": SCAN_PAGE_PROMPT,
            "pixels": hashlib.sha256(pixels).hexdigest(), "depth": region_depth,
            "clients": [_client_identity(client) for client in clients],
            "models": [chain_fingerprint(role) for role in ("chunking.vision", "chunking.vision_review")],
        }
        base_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        key = base_key
        if review_incomplete:
            identity["review_incomplete"] = True
            if prefer_halves:
                identity["prefer_halves"] = True
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        for cache_key in dict.fromkeys((key, base_key)):
            cached = self.cache.get(cache_key)
            if isinstance(cached, dict):
                try:
                    evidence = cached["visual_review"]
                    if not evidence.get("incomplete") and evidence.get("layout_sha256") == _layout_hash(cached):
                        _validate_layout(cached, pixels)
                        annotate_span(cache_hit=True)
                        return deepcopy(cached)
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass

        instructions = SCAN_PAGE_PROMPT
        if region_depth:
            instructions += "\n" + PDF_REGION_CROP_INSTRUCTION
        if attempt_budget is None:
            attempt_budget = CloudAttempts()
        initial_budget = attempt_budget.remaining
        failures = []
        can_crop = False
        partial = None
        repair = False
        reserved = 0
        if prefer_halves and review_incomplete and region_depth == 0 and initial_budget >= 3 and region_depth < _MAX_CROP_DEPTH:
            reserved = 2
        for index, client in enumerate(clients):
            allowance = max(0, attempt_budget.remaining - reserved)
            if isinstance(client, QwenVisionClient):
                allowance = min(allowance, client.max_retries + 1)
            for _ in range(allowance):
                if not attempt_budget.remaining:
                    break
                if index == 0 and len(clients) > 1 and attempt_budget.remaining == 1 and partial is None:
                    break
                if repair:
                    prompt = instructions + _REPAIR_INSTRUCTION
                else:
                    prompt = instructions
                try:
                    if isinstance(client, QwenVisionClient):
                        routing = {}
                        if routed and (repair or region_depth or index):
                            routing["route_reason"] = "visual_review"
                        raw = client.analyze_page_layout(pixels, prompt, attempt_budget=attempt_budget,
                                                         single_request=True, **routing)
                    else:
                        attempt_budget.consume()
                        raw = client.analyze_page_layout(pixels, prompt)
                    if self.use_decision and self.asset_dir is not None and _needs_table_repair(raw):
                        reviewer = backup or client
                        if reviewer is not None:
                            try:
                                from pathlib import Path
                                from .table_review import repair_tables
                                candidate_raw = deepcopy(raw)
                                repair_tables(candidate_raw, pixels, reviewer, Path(self.asset_dir), self.use_decision)
                                raw = candidate_raw
                            except Exception:
                                pass
                    layout = _prepare_layout(raw)
                    _validate_layout(layout, pixels)
                except VisionPageOverflow as error:
                    failures.append(type(error).__name__)
                    can_crop = True
                    break
                except (ValueError, TypeError, KeyError) as error:
                    failures.append(type(error).__name__)
                    repair = True
                    can_crop = True
                    if (routed or review_incomplete) and index == 0 and len(clients) > 1:
                        break
                    continue
                except (RuntimeError, OSError) as error:
                    failures.append(type(error).__name__)
                    if isinstance(error, ModelFallbackError):
                        cause = getattr(error, "last_error", None)
                    else:
                        cause = None
                    if cause is not None and getattr(cause, "retryable", True) and index == len(clients) - 1:
                        continue
                    break
                evidence = layout["visual_review"]
                evidence["model"] = (layout.get("api") or {}).get("model") or getattr(client, "model", "unknown")
                evidence["scope"] = "cloud_single_reading"
                if (_SCHEMA_ISSUES.intersection(evidence["issues"])
                        or (routed or review_incomplete) and (evidence["incomplete"] or layout["uncertainties"])):
                    if partial is None or len(evidence["issues"]) < len(partial["visual_review"]["issues"]):
                        partial = layout
                    repair = True
                    if (routed or review_incomplete) and index == 0 and len(clients) > 1:
                        break
                    continue
                return self._finish(key, layout, width, height, failures, initial_budget - attempt_budget.remaining)
        if reserved and attempt_budget.remaining >= reserved and (partial is not None or can_crop):
            try:
                cropped = self._overflow_regions(pixels, width, height, region_depth, failures, attempt_budget,
                                                 review_incomplete=True, halves=True)
            except CloudOCRUnavailable:
                cropped = None
            if cropped is not None and (partial is None or _retains_partial_evidence(partial, cropped)):
                return self._finish(key, cropped, width, height, failures, initial_budget - attempt_budget.remaining)
        if partial is not None:
            return self._finish(key, partial, width, height, failures, initial_budget - attempt_budget.remaining)
        if can_crop and attempt_budget.remaining and region_depth < _MAX_CROP_DEPTH:
            layout = self._overflow_regions(pixels, width, height, region_depth, failures, attempt_budget,
                                            review_incomplete=review_incomplete)
            return self._finish(key, layout, width, height, failures, initial_budget - attempt_budget.remaining)
        raise CloudOCRUnavailable("Cloud OCR returned no usable page content (" + ", ".join(failures) + ")")

    def _finish(self, key, layout, width, height, failures, attempts):
        evidence = layout["visual_review"]
        evidence.update(failed_candidates=failures, cloud_attempts=attempts, layout_sha256=_layout_hash(layout))
        if not evidence["incomplete"]:
            self.cache.put(key, layout, width, height)
        return deepcopy(layout)

    def _overflow_regions(self, pixels, width, height, depth, failures, attempt_budget, *, review_incomplete=False,
                          halves=False):
        from PIL import Image
        from ...models.text_client import CloudAttempts

        observations = []
        audit = []
        non_text = False
        incomplete = False
        pad_x, pad_y = max(1, width // _CROP_PAD_RATIO), max(1, height // _CROP_PAD_RATIO)
        x_bands = ((0, min(width, width // 2 + pad_x)), (max(0, width // 2 - pad_x), width))
        if halves:
            x_bands = ((0, width),)
        y_bands = ((0, min(height, height // 2 + pad_y)), (max(0, height // 2 - pad_y), height))
        with Image.open(io.BytesIO(pixels)) as image:
            for top, bottom in y_bands:
                for left, right in x_bands:
                    box = [left, top, right, bottom]
                    entry = {"pixel_box": box}
                    audit.append(entry)
                    if not attempt_budget.remaining:
                        entry["error"] = "Cloud request attempt limit reached"
                        incomplete = True
                        continue
                    stream = io.BytesIO()
                    image.crop(tuple(box)).save(stream, "PNG")
                    region_budget = attempt_budget
                    if halves:
                        region_budget = CloudAttempts(1)
                    try:
                        crop = self.read(stream.getvalue(), [], right - left, bottom - top, depth + 1,
                                         attempt_budget=region_budget, review_incomplete=review_incomplete)
                    except CloudOCRUnavailable as error:
                        entry["error"] = str(error)
                        incomplete = True
                        continue
                    finally:
                        if halves and not region_budget.remaining:
                            attempt_budget.consume()
                    entry["visual_review"] = crop["visual_review"]
                    entry["children"] = crop.get("region_audit", [])
                    incomplete |= crop["visual_review"]["incomplete"]
                    non_text |= crop["review"].get("non_text_only") is True
                    for element in crop["elements"]:
                        item = deepcopy(element)
                        x0, y0, x1, y1 = item["bbox"]
                        item["bbox"] = [(left + x0 * (right - left)) / width,
                                        (top + y0 * (bottom - top)) / height,
                                        (left + x1 * (right - left)) / width,
                                        (top + y1 * (bottom - top)) / height]
                        item["representation"] = "supplemental_region_observation"
                        item["region_pixel_path"] = [box] + item.get("region_pixel_path", [])
                        item["region_pixel_box"] = box
                        try:
                            merge_region_observation(observations, item)
                        except ValueError:
                            entry["overlap_conflict"] = True
                            incomplete = True
        if not observations and incomplete:
            raise CloudOCRUnavailable("Cloud OCR returned no usable content from the page regions")
        region_issues = []
        if incomplete:
            region_issues = ["region_incomplete"]
        layout = {"elements": reading_order(observations), "uncertainties": [],
                  "review": {"non_text_only": non_text and not observations},
                  "region_audit": audit,
                  "visual_review": {"scope": "cloud_regions", "failed_candidates": failures,
                                    "incomplete": incomplete, "issues": region_issues}}
        try:
            _validate_layout(layout, pixels)
        except ValueError as error:
            raise CloudOCRUnavailable("Cloud OCR regions yielded no usable page content") from error
        return layout


def _client_identity(client):
    identity = getattr(client, "identity", None)
    if callable(identity):
        return identity()
    return getattr(client, "model", type(client).__name__)


def _prepare_layout(result):
    layout = deepcopy(result)
    normalize_visual_layout(layout)
    elements = []
    skipped = layout.get("invalid_records", 0)
    downgraded = 0
    for element in layout["elements"]:
        # Only local crop assembly may create provenance about nested regions.
        for field in ("representation", "region_pixel_box", "region_pixel_path", "typography"):
            element.pop(field, None)
        if element["type"] == "table":
            try:
                if element.get("table_html") is not None:
                    element["table_data"] = table_from_html(element["table_html"], element.get("header_row_count"))
                validate_table(element.get("table_data"), require_complete=True)
            except (ValueError, TypeError, KeyError):
                table = element.get("table_data")
                if isinstance(table, dict) and all(isinstance(table.get(key, []), list) for key in ("header_grid", "rows")):
                    rows = table.get("header_grid", []) + table.get("rows", [])
                else:
                    rows = []
                text = element.get("text", "").strip()
                if not text and rows:
                    text = "\n".join(
                        " | ".join(cell for cell in row if isinstance(cell, str))
                        for row in rows if isinstance(row, list)
                    ).strip()
                if not text and element.get("table_html"):
                    text = _strip_html_text(element.get("table_html"))
                if not text:
                    skipped += 1
                    continue
                element = {"type": "paragraph", "text": text, "bbox": element["bbox"],
                           "bbox_granularity": element.get("bbox_granularity", "region"),
                           "order_evidence": "cloud_table_text_only"}
                downgraded += 1
        elif not element.get("text", "").strip():
            skipped += 1
            continue
        else:
            for field in ("table_data", "table_html"):
                element.pop(field, None)
        elements.append(element)
    layout["elements"] = elements
    review = layout.get("review")
    if not isinstance(review, dict):
        review = {}
    if elements or skipped:
        review.pop("non_text_only", None)
    layout["review"] = review
    uncertainties = layout.get("uncertainties")
    issues = []
    if not isinstance(uncertainties, list) or not isinstance(review.get("verdict"), str):
        issues.append("review_metadata_missing")
    if uncertainties or str(review.get("verdict", "")).strip().casefold() == "unresolved":
        issues.append("model_uncertainty")
    if skipped:
        issues.append("elements_skipped")
    if downgraded:
        issues.append("table_text_only")
    if isinstance(review.get("verdict"), str) and review["verdict"].strip().casefold() not in _ACCEPTED_VERDICTS:
        issues.append("unrecognized_review_verdict")
    if isinstance(uncertainties, list):
        uncertainty_count = len(uncertainties)
    else:
        uncertainty_count = None
    layout["visual_review"] = {
        "issues": issues,
        "provider_verdict": review.get("verdict"),
        "uncertainty_count": uncertainty_count,
        "skipped_elements": skipped,
        "downgraded_tables": downgraded,
        "incomplete": bool(issues),
    }
    return layout


def _validate_layout(layout, pixels):
    # Validate the shape used by the chunker; the provider's actual verdict is
    # retained in the output and does not act as a second OCR gate.
    validated = {**layout, "uncertainties": [], "review": {**layout["review"], "verdict": "accepted"}}
    validate_page(validated, [])
    validate_scan_pixels(validated, pixels)


def _layout_hash(layout):
    data = [layout["elements"], layout["review"], layout.get("uncertainties"), layout.get("region_audit", [])]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
