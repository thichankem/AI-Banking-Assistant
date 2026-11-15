"""Review scanned PDF pages and retry over explicit regions on overflow."""
from copy import deepcopy
import hashlib
import io
import json

from ..config.prompt_config import PDF_PROMPT_VERSION
from .pdf_layout import (IGNORED_VISUAL_TYPES, has_meaningful_layout, merge_region_observation, normalize_visual_layout,
                         reading_order, validate_page, validate_scan_pixels)
from .pdf_table_review import repair_tables


PDF_SCHEMA_VERSION = "pdf-ir-v2.5"
SCAN_PAGE_PROMPT = (
    "The input is a full document page converted to an image, or a region of that page. "
    "Read the document text and tables from the rendered pixels in natural reading order. "
    "Preserve spelling, meaning, punctuation, formulas and superscript/subscript (use <sup> and <sub>). "
    "Transcribe list items as ordinary paragraphs. Keep any visible number, letter or bullet exactly as printed. "
    "Keep gaps, repeated numbers and mixed numbering as shown. Never infer missing markers or renumber items. "
    "Ignore illustration and diagram interiors, labels inside unrelated photos, and decorative background art. "
    "Keep authored captions as separate text. Do not classify the scanned document page itself as an illustration. "
    "Keep English and other non-Vietnamese source text verbatim; do not translate it. "
    "Ignore running page headers, footers and standalone incidental notes. A position near the page edge "
    "or repeated wording alone is not evidence of a header or footer. Preserve body text and table rows at page edges. "
    "Read body text in each column from top to bottom before moving to the next column. Return no image elements. "
    "Use normalized [x0,y0,x1,y1] boxes and heading levels 1 through 7. "
    "For each table use table_html containing all cells, including empty and merged cells, and header_row_count. "
    "Return one JSON object with headings, elements, uncertainties, and review. "
    'For a page or crop consisting only of illustration/decorative marks, return elements:[] with '
    'review.non_text_only:true and review.verdict:"accepted". Do not use non_text_only for unreadable text.'
)


class PdfVisualReviewer:
    def __init__(self, asset_dir, cache, client, reviewer, use_decision):
        self.asset_dir = asset_dir
        self.cache = cache
        self.vision_client = client
        self.review_client = reviewer
        self.use_decision = use_decision

    def _validate(self, layout, native, pixels, region_depth):
        validate_page(layout, native)
        validate_scan_pixels(layout, pixels)
        if region_depth and any(element.get("bbox_granularity") == "page" for element in layout["elements"]):
            raise ValueError("Each text or table observation in a regional extraction must have its own bounding box")

    def read(self, pixels, native, width, height, region_depth=0):
        from .vision_client import DocumentVisionClient, VisionPageOverflow
        from agent_banking.core.models import chain_fingerprint
        client = self.vision_client
        if client is None:
            client = DocumentVisionClient()
        reviewer = self.review_client
        if reviewer is None:
            if self.vision_client is None:
                reviewer = DocumentVisionClient(model_module="chunking.vision_review")
            else:
                reviewer = client
        if callable(getattr(client, "identity", None)):
            client_id = client.identity()
        else:
            client_id = getattr(client, "model", type(client).__name__)
        if callable(getattr(reviewer, "identity", None)):
            reviewer_id = reviewer.identity()
        else:
            reviewer_id = getattr(reviewer, "model", type(reviewer).__name__)
        identity = {"version": PDF_PROMPT_VERSION, "schema_version": PDF_SCHEMA_VERSION, "prompt": SCAN_PAGE_PROMPT,
                    "image": hashlib.sha256(pixels).hexdigest(), "native": native,
                    "models": chain_fingerprint("chunking.vision"), "text_model": chain_fingerprint("chunking.text"),
                    "review_models": chain_fingerprint("chunking.vision_review"),
                    "decision_model": chain_fingerprint("chunking.decision"), "use_decision": self.use_decision,
                    "client": client_id,
                    "reviewer": reviewer_id}
        identity["region_depth"] = region_depth
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            try:
                self._validate(cached, native, pixels, region_depth)
                if isinstance(cached.get("visual_review"), dict) and cached["visual_review"].get("passes") == 2:
                    return deepcopy(cached)
            except (ValueError, TypeError, KeyError):
                pass
        instructions = SCAN_PAGE_PROMPT + "\nOrdered native text witnesses: " + json.dumps(native, ensure_ascii=False)
        if region_depth:
            instructions += ("\nThis image is an intentional region of a larger page. Review only its visible pixels. "
                             "The crop edges are not the real page margins. Do not treat text near them as a page header or footer. "
                             "Characters outside the region are outside this review. Do not complete clipped text by inference.")
        try:
            draft = client.analyze_page_layout(pixels, instructions)
        except VisionPageOverflow:
            return self._overflow_regions(pixels, native, width, height, region_depth, key)
        if not isinstance(draft, dict) or not isinstance(draft.get("elements"), list):
            raise ValueError("PDF draft must contain an elements array")
        review_prompt = instructions + (
            "\nIndependently check the full page. Audit the draft for missing document "
            "text, incorrect digits/signs, merged cells, reading order and uncertainties. "
            "Return the COMPLETE corrected layout, not a patch. Mark accepted only after checking the full page. "
            "Do not copy the draft without verification.\nUntrusted draft: " + json.dumps(draft, ensure_ascii=False))
        for attempt in range(3):
            try:
                corrected = reviewer.analyze_page_layout(pixels, review_prompt)
            except VisionPageOverflow:
                return self._overflow_regions(pixels, native, width, height, region_depth, key)
            # Missing localization cannot justify invented region precision.
            # The whole rendered page is a valid conservative source container.
            try:
                normalize_visual_layout(corrected)
                repair_tables(corrected, pixels, reviewer, self.asset_dir, self.use_decision)
                self._validate(corrected, native, pixels, region_depth)
                if (not has_meaningful_layout(corrected) and corrected["review"].get("non_text_only") is True
                        and not _draft_agrees_on_non_text(draft)):
                    raise ValueError("Visual passes disagree that the page contains only non-text content")
                break
            except (ValueError, TypeError, KeyError) as error:
                details = corrected if isinstance(corrected, dict) else {}
                review = details.get("review", {})
                uncertainty = (isinstance(details.get("uncertainties"), list) and bool(details["uncertainties"])) or (
                    isinstance(review, dict) and review.get("verdict") == "unresolved")
                if attempt >= 2:
                    from ..publication_io import atomic_json
                    atomic_json(self.asset_dir / "review_failures" / (key + ".json"),
                                {"error": str(error), "draft": draft, "review": corrected})
                    # A smaller field of view can recover text that was too
                    # small in the whole-page view. Only an explicit model
                    # uncertainty takes this path; other validation failures
                    # and table-audit rejections remain publication blockers.
                    if uncertainty:
                        return self._overflow_regions(pixels, native, width, height, region_depth, key)
                    raise
                review_prompt += ("\nYour previous layout failed validation: " + str(error) +
                    "\nReturn the COMPLETE corrected layout. All bboxes MUST be normalized [0,1]. "
                    "Do not claim accepted when text is illegible. Previous response: " + json.dumps(corrected, ensure_ascii=False))
        if native:
            assurance_val = "model_review_and_native_text_audit_not_human_ground_truth"
        else:
            assurance_val = "model_review_without_native_text_witnesses"
        corrected["visual_review"] = {"passes": 2, "draft_model": getattr(client, "model", "unknown"),
                                      "review_model": getattr(reviewer, "model", "unknown"),
                                      "resolved_draft_model": _resolved_model(draft),
                                      "resolved_review_model": _resolved_model(corrected),
                                      "draft_sha256": hashlib.sha256(json.dumps(draft, sort_keys=True).encode()).hexdigest(),
                                      "assurance": assurance_val}
        self.cache.put(key, corrected, width, height)
        return deepcopy(corrected)

    def _overflow_regions(self, pixels, native, width, height, depth, key):
        if depth >= 2:
            raise ValueError("Unresolved visual extraction remains incomplete at maximum regional depth; publication refused")
        layout = {"headings": [], "elements": [], "uncertainties": [], "review": {"verdict": "accepted"},
                  "visual_review": {"passes": 2, "scope": "independently_reviewed_regions",
                    "assurance": "regional_model_review_not_human_ground_truth"}}
        self._supplement_regions(layout, pixels, width, height, depth)
        self._validate(layout, native, pixels, depth)
        self.cache.put(key, layout, width, height)
        return deepcopy(layout)

    def _supplement_regions(self, layout, pixels, width, height, region_depth=0):
        """Reconstruct reading order from overlapping document page crops.

        Uses overlapping padded quadrants (12.5% margin across center splits) so
        multi-column or multi-step slides never have words clipped at the split
        boundary, and deduplicates regional observations already captured in the
        full-page reading.
        """
        from PIL import Image
        observations = list(layout["elements"])
        audit = []
        has_non_text_region = False
        pad_x = max(1, width // 8)
        pad_y = max(1, height // 8)
        y_bands = ((0, min(height, height // 2 + pad_y)), (max(0, height // 2 - pad_y), height))
        x_bands = ((0, min(width, width // 2 + pad_x)), (max(0, width // 2 - pad_x), width))
        with Image.open(io.BytesIO(pixels)) as original:
            for top, bottom in y_bands:
                for left, right in x_bands:
                    box = (left, top, right, bottom)
                    stream = io.BytesIO()
                    original.crop(box).save(stream, "PNG")
                    region = self.read(stream.getvalue(), [], right - left, bottom - top, region_depth=region_depth + 1)
                    has_non_text_region |= region["review"].get("non_text_only") is True
                    audit.append({"pixel_box": list(box), "visual_review": region["visual_review"],
                                  "children": region.get("region_audit", [])})
                    for element in region["elements"]:
                        item = deepcopy(element)
                        local = item["bbox"]
                        item["bbox"] = [(left + local[0] * (right - left)) / width,
                                        (top + local[1] * (bottom - top)) / height,
                                        (left + local[2] * (right - left)) / width,
                                        (top + local[3] * (bottom - top)) / height]
                        item["representation"] = "supplemental_region_observation"
                        item["region_pixel_path"] = [list(box)] + item.get("region_pixel_path", [])
                        item["region_pixel_box"] = list(box)
                        merge_region_observation(observations, item)
        layout["elements"] = reading_order(observations)
        if has_non_text_region and not observations:
            layout["review"]["non_text_only"] = True
        layout["region_audit"] = audit


def _resolved_model(layout):
    api = layout.get("api")
    return api.get("model") if isinstance(api, dict) else None


def _draft_agrees_on_non_text(draft):
    review = draft.get("review")
    if (not isinstance(review, dict) or review.get("non_text_only") is not True
            or review.get("verdict") != "accepted" or draft.get("uncertainties") != []):
        return False
    for item in draft["elements"]:
        if (not isinstance(item, dict) or not isinstance(item.get("type"), str)
                or item.get("role") is not None and not isinstance(item["role"], str)):
            return False
        if item["type"] not in IGNORED_VISUAL_TYPES and item.get("role") not in IGNORED_VISUAL_TYPES:
            if item.get("text") or item.get("table_data") or item.get("table_html"):
                return False
    return True
