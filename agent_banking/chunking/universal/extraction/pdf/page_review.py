"""Read PDF pixels through the configured cloud vision candidates."""
from copy import deepcopy
import hashlib
import io
import json

from agent_banking.chunking.config.prompt_config import CLOUD_OCR_PAGE_PROMPT as SCAN_PAGE_PROMPT, PDF_PROMPT_VERSION
from .layout import has_meaningful_layout, merge_region_observation, normalize_visual_layout, reading_order, validate_page, validate_scan_pixels
from .tables import table_from_html, validate_table


PDF_SCHEMA_VERSION = "pdf-ir-v7-page-citations"
OCR_MAX_OUTPUT_TOKENS = 6_144
_SCHEMA_ISSUES = {"review_metadata_missing", "unrecognized_review_verdict", "elements_skipped",
                  "table_text_only"}
_REPAIR_INSTRUCTION = (
    "\nThe previous response did not match the requested schema. Read the same image again and return a complete JSON object. "
    'Include elements, uncertainties, and review with verdict "accepted" or "unresolved". '
    "Each element must include its type. Include table cell structure for every table. "
    "Use uncertainties for unreadable source details."
)



class CloudOCRUnavailable(ValueError):
    """No usable cloud transcription was returned for this page."""


class PdfVisualReviewer:
    def __init__(self, asset_dir, cache, client, reviewer, use_decision):
        self.cache = cache
        self.vision_client = client
        self.review_client = reviewer

    def read(self, pixels, native, width, height, region_depth=0, *, allow_ocr_additions=False, attempt_budget=None):
        from ...models.text_client import CloudAttempts
        from ...models.vision_client import DocumentVisionClient, QwenVisionClient, VisionPageOverflow
        from agent_banking.core.fallback import ModelFallbackError
        from agent_banking.core.models import chain_fingerprint, get_model_chain

        primary = self.vision_client or DocumentVisionClient(max_tokens=OCR_MAX_OUTPUT_TOKENS)
        backup = self.review_client
        if (backup is None and self.vision_client is None
                and any(endpoint.model for endpoint in get_model_chain("chunking.vision_review"))):
            backup = DocumentVisionClient(model_module="chunking.vision_review", max_tokens=OCR_MAX_OUTPUT_TOKENS)
        clients = (primary,) if backup is None or backup is primary else (primary, backup)
        identity = {
            "schema": PDF_SCHEMA_VERSION, "version": PDF_PROMPT_VERSION, "prompt": SCAN_PAGE_PROMPT,
            "pixels": hashlib.sha256(pixels).hexdigest(), "depth": region_depth,
            "clients": [_client_identity(client) for client in clients],
            "models": [chain_fingerprint(role) for role in ("chunking.vision", "chunking.vision_review")],
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cached = self.cache.get(key)
        if isinstance(cached, dict):
            try:
                evidence = cached["visual_review"]
                if evidence.get("layout_sha256") == _layout_hash(cached):
                    _validate_layout(cached, pixels)
                    return deepcopy(cached)
            except (AttributeError, KeyError, TypeError, ValueError):
                pass

        instructions = SCAN_PAGE_PROMPT
        if region_depth:
            instructions += ("\nRead only the supplied crop. Keep visible fragments and do not complete clipped text. "
                             "If available, include normalized bbox coordinates to help match overlapping crops.")
        if attempt_budget is None:
            attempt_budget = CloudAttempts()
        initial_budget = attempt_budget.remaining
        failures = []
        can_crop = False
        partial = None
        repair = False
        for index, client in enumerate(clients):
            allowance = attempt_budget.remaining
            if isinstance(client, QwenVisionClient):
                allowance = min(allowance, client.max_retries + 1)
            for _ in range(allowance):
                if not attempt_budget.remaining:
                    break
                if index == 0 and len(clients) > 1 and attempt_budget.remaining == 1 and partial is None:
                    break
                prompt = instructions + _REPAIR_INSTRUCTION if repair else instructions
                try:
                    if isinstance(client, QwenVisionClient):
                        raw = client.analyze_page_layout(pixels, prompt, attempt_budget=attempt_budget,
                                                         single_request=True)
                    else:
                        attempt_budget.consume()
                        raw = client.analyze_page_layout(pixels, prompt)
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
                    continue
                except (RuntimeError, OSError) as error:
                    failures.append(type(error).__name__)
                    cause = getattr(error, "last_error", None) if isinstance(error, ModelFallbackError) else None
                    if cause is not None and getattr(cause, "retryable", True) and index == len(clients) - 1:
                        continue
                    break
                evidence = layout["visual_review"]
                evidence["model"] = (layout.get("api") or {}).get("model") or getattr(client, "model", "unknown")
                evidence["scope"] = "cloud_single_reading"
                if _SCHEMA_ISSUES.intersection(evidence["issues"]):
                    if partial is None:
                        partial = layout
                    repair = True
                    continue
                return self._finish(key, layout, width, height, failures, initial_budget - attempt_budget.remaining)
        if partial is not None:
            return self._finish(key, partial, width, height, failures, initial_budget - attempt_budget.remaining)
        if can_crop and attempt_budget.remaining and region_depth < 2:
            layout = self._overflow_regions(pixels, width, height, region_depth, failures, attempt_budget)
            return self._finish(key, layout, width, height, failures, initial_budget - attempt_budget.remaining)
        raise CloudOCRUnavailable("Cloud OCR returned no usable page content (" + ", ".join(failures) + ")")

    def _finish(self, key, layout, width, height, failures, attempts):
        evidence = layout["visual_review"]
        evidence.update(failed_candidates=failures, cloud_attempts=attempts, layout_sha256=_layout_hash(layout))
        if not evidence["incomplete"]:
            self.cache.put(key, layout, width, height)
        return deepcopy(layout)

    def _overflow_regions(self, pixels, width, height, depth, failures, attempt_budget):
        from PIL import Image

        observations = []
        audit = []
        non_text = False
        incomplete = False
        pad_x, pad_y = max(1, width // 8), max(1, height // 8)
        x_bands = ((0, min(width, width // 2 + pad_x)), (max(0, width // 2 - pad_x), width))
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
                    try:
                        crop = self.read(stream.getvalue(), [], right - left, bottom - top, depth + 1,
                                         attempt_budget=attempt_budget)
                    except CloudOCRUnavailable as error:
                        entry["error"] = str(error)
                        incomplete = True
                        continue
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
        layout = {"elements": reading_order(observations), "uncertainties": [],
                  "review": {"non_text_only": non_text and not observations},
                  "region_audit": audit,
                  "visual_review": {"scope": "cloud_regions", "failed_candidates": failures,
                                    "incomplete": incomplete, "issues": ["region_incomplete"] if incomplete else []}}
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
                rows = table.get("header_grid", []) + table.get("rows", []) if isinstance(table, dict) and all(
                    isinstance(table.get(key, []), list) for key in ("header_grid", "rows")
                ) else []
                text = element.get("text", "").strip() or "\n".join(
                    " | ".join(cell for cell in row if isinstance(cell, str))
                    for row in rows if isinstance(row, list)
                ).strip()
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
    if isinstance(review.get("verdict"), str) and review["verdict"].strip().casefold() not in ("accepted", "ok", "unresolved"):
        issues.append("unrecognized_review_verdict")
    layout["visual_review"] = {
        "issues": issues,
        "provider_verdict": review.get("verdict"),
        "uncertainty_count": len(uncertainties) if isinstance(uncertainties, list) else None,
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
