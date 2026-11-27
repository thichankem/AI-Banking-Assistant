"""Extract PDF documents into the chunking intermediate representation."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from agent_banking.chunking.config.settings import get_config_section, get_rules
from .layout import has_meaningful_layout
from .page_review import CloudOCRUnavailable, PdfVisualReviewer

_PIPELINE = get_config_section("pipeline")
_PDF_CFG = get_config_section("extraction")["pdf"]
_FURN_CFG = _PDF_CFG["furniture"]
_PDF_RULES = get_rules("pdf")
_PUB_RULES = get_rules("publication")
_FURNITURE_PREFIXES = tuple(_PDF_RULES["furniture_prefixes"])
_NON_FURNITURE_PREFIXES = tuple(_PDF_RULES["non_furniture_prefixes"])
_MIN_DPI = _PIPELINE["min_dpi"]
_MAX_DPI = _PIPELINE["max_dpi"]
_DOC_ID_HEX_LEN = _PUB_RULES["document_id_hex_length"]
_PREFIX_HEX_LEN = _PUB_RULES["prefix_hash_length"]


class PartialOCRNotImproved(Exception):
    """A maintenance read did not improve on the published page coverage."""


class PdfExtractor:
    def __init__(self, asset_dir, cache, vision_client=None, review_client=None,
                 dpi=_PIPELINE["dpi"], max_pages=_PIPELINE["max_pdf_pages"],
                 max_page_pixels=_PIPELINE["max_page_pixels"], use_decision=True, *, search_language="vi", workers=1,
                 review_incomplete=False, previous_partial_pages=0, review_page_indexes=None):
        if isinstance(dpi, bool) or not isinstance(dpi, int) or not _MIN_DPI <= dpi <= _MAX_DPI or isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1:
            raise ValueError("PDF limits require " + str(_MIN_DPI) + ".." + str(_MAX_DPI) + " DPI and positive max_pages")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.asset_dir = Path(asset_dir)
        self.cache = cache
        self.vision_client = vision_client
        self.review_client = review_client
        self.dpi = dpi
        self.max_pages = max_pages
        if isinstance(max_page_pixels, bool) or not isinstance(max_page_pixels, int) or max_page_pixels < 1:
            raise ValueError("PDF pixel limit must be positive")
        self.max_page_pixels = max_page_pixels
        self.use_decision = use_decision
        self.search_language = search_language
        self.workers = workers
        self.review_incomplete = review_incomplete
        self.previous_partial_pages = previous_partial_pages
        self.review_page_indexes = review_page_indexes

    def extract(self, source, source_name):
        import pymupdf
        from ...models.text_client import ordered_parallel

        data = source
        if not isinstance(data, (bytes, bytearray)):
            data = Path(source).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        identity = str(source_name).replace("\\", "/")
        document_id = hashlib.sha256(identity.encode()).hexdigest()[:_DOC_ID_HEX_LEN]
        ir = {"version": _PIPELINE["schema_version"], "document_id": document_id, "document_version": digest,
              "source": identity, "blocks": [], "units": [], "assets": {}, "warnings": [], "relations": [],
              "profile": _PIPELINE["profile"], "page_reviews": [], "search_language": self.search_language}
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("Encrypted PDF requires decryption before ingestion")
            if len(document) > self.max_pages:
                raise ValueError("PDF exceeds configured page limit; no partial ingestion")
            ir["extraction_method"] = "pdf_cloud_ocr"
            ir["visual_policy"] = "pdf_cloud_ocr"
            page_count = len(document)
            rendered = []
            for page_index, page in enumerate(document):
                rendered.append(self._render_page(ir, page, page_index))

            def process_rendered(item):
                page_index, pixels, width, height, sha, rotation = item
                return self._read_rendered_page(pixels, width, height, sha, page_index, rotation)

            if self.workers > 1 and len(rendered) > 1:
                outcomes = ordered_parallel(process_rendered, rendered, min(self.workers, len(rendered)))
            else:
                outcomes = [process_rendered(item) for item in rendered]

            failed_pages = []
            for outcome in outcomes:
                page_index, sha, rotation, layout, error = outcome
                if error is not None:
                    error.pdf_page_number = page_index + 1
                    failed_pages.append(error)
                    ir["page_reviews"].append({
                        "page_index": page_index, "method": "pdf_cloud_ocr",
                        "content": "unavailable", "issues": ["ocr_unavailable"], "error": str(error),
                        "render_sha256": sha,
                    })
                    continue
                if has_meaningful_layout(layout):
                    content = "text"
                elif layout["review"].get("non_text_only") is True:
                    content = "non_text_only"
                else:
                    content = "blank"
                ir["page_reviews"].append({"page_index": page_index, **layout["visual_review"], "method": "pdf_cloud_ocr",
                                           "content": content, "render_sha256": sha,
                                           "region_audit": layout.get("region_audit", [])})
                for element in layout["elements"]:
                    self._append(ir, element, page_index, sha, rotation)
            ocr_statistics = {
                "source_pages": page_count,
                "pages_requiring_ocr": page_count,
                "ocr_pages_processed": page_count - len(failed_pages),
                "ocr_partial_pages": sum(review.get("incomplete") is True for review in ir["page_reviews"]),
                "ocr_failed_pages": len(failed_pages),
                "ocr_failed_page_numbers": [error.pdf_page_number for error in failed_pages],
            }
            if self.previous_partial_pages and (failed_pages or ocr_statistics["ocr_partial_pages"] >= self.previous_partial_pages):
                raise PartialOCRNotImproved
            if failed_pages and not ir["blocks"]:
                failed_pages[0].pdf_ocr_statistics = ocr_statistics
                raise failed_pages[0]
            _mark_pdf_furniture(ir, page_count)
            if self.use_decision:
                from ...enrichment.decisions.engine import decide_continued_tables
                from ..docx.extractor import apply_typography_hierarchy, link_visual_captions
                apply_typography_hierarchy(ir, cache=self.cache)
                link_visual_captions(ir, cache=self.cache)
                decide_continued_tables(ir, cache=self.cache)
            ir["statistics"] = {
                "pages": page_count, "blocks": len(ir["blocks"]), "units": len(ir["units"]),
                **ocr_statistics,
                "pdf_processing_mode": ir["visual_policy"],
                "tables": sum(block["type"] == "table" for block in ir["blocks"]),
            }
        return ir

    def _render_page(self, ir, page, page_index):
        import pymupdf

        expected_pixels = math.ceil(page.rect.width * self.dpi / 72) * math.ceil(page.rect.height * self.dpi / 72)
        if expected_pixels > self.max_page_pixels:
            raise ValueError("PDF page exceeds configured render pixel limit; no partial ingestion")
        pixmap = page.get_pixmap(dpi=self.dpi, colorspace=pymupdf.csRGB, alpha=False, annots=False)
        pixels = pixmap.tobytes("png")
        sha = hashlib.sha256(pixels).hexdigest()
        path = self.asset_dir / (sha + ".png")
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            from agent_banking.chunking.publication_io import atomic_bytes
            atomic_bytes(path, pixels)
        asset = ir["assets"].setdefault(sha, {
            "path": str(path.resolve()), "sha256": sha, "kind": "page", "page_index": page_index,
            "page_indices": [], "width": pixmap.width, "height": pixmap.height,
        })
        asset["page_indices"].append(page_index)
        from .visual_cropper import extract_page_visual_crops

        for crop in extract_page_visual_crops(
            pixels, page_index, sha, self.asset_dir, source_name=ir.get("source", "")
        ):
            ir["assets"].setdefault(crop["sha256"], crop)
        return page_index, pixels, pixmap.width, pixmap.height, sha, page.rotation

    def _read_rendered_page(self, pixels, width, height, sha, page_index, rotation):
        try:
            reader = PdfVisualReviewer(self.asset_dir, self.cache, self.vision_client, self.review_client, self.use_decision)
            review_page = self.review_incomplete and (self.review_page_indexes is None or page_index in self.review_page_indexes)
            layout = reader.read(pixels, [], width, height, review_incomplete=review_page, prefer_halves=review_page)
            return page_index, sha, rotation, layout, None
        except CloudOCRUnavailable as error:
            error.pdf_render_sha256 = sha
            return page_index, sha, rotation, None, error
        except (OSError, RuntimeError, ValueError) as error:
            error.pdf_render_sha256 = sha
            error.pdf_page_number = page_index + 1
            raise

    def _append(self, ir, element, page, sha, rotation):
        identifier = ir["document_id"] + ":v" + ir["document_version"][:_PREFIX_HEX_LEN] + ":b" + str(len(ir["blocks"]))
        kind = element["type"]
        block = {"id": identifier, "type": kind, "text": element.get("text", ""), "order": len(ir["blocks"]),
                 "provenance": {"page_index": page, "page_number": page + 1, "region": "body",
                                "bbox": element["bbox"], "coordinate_system": "normalized_rendered_page",
                                "bbox_granularity": element.get("bbox_granularity", "region"),
                                "order_evidence": element.get("order_evidence", "visual_reading_order"),
                                "rotation": rotation,
                                "method": "pdf_cloud_ocr", "page_asset_sha256": sha}}
        if element.get("typography"):
            block["typography"] = deepcopy(element["typography"])
        if element.get("representation"):
            block["provenance"]["representation"] = element["representation"]
            block["provenance"]["region_pixel_box"] = element["region_pixel_box"]
            block["provenance"]["region_pixel_path"] = element.get("region_pixel_path", [element["region_pixel_box"]])
        if kind == "heading":
            block["level"] = element["level"]
        if kind == "table":
            table = deepcopy(element["table_data"])
            table["table_id"] = identifier
            table["grid"] = table["header_grid"] + table["rows"]
            table["header_detection"] = "visual_model_review"
            block["table_data"] = table
            if element.get("table_schema_repair"):
                block["table_schema_repair"] = deepcopy(element["table_schema_repair"])
            for row_index, row in enumerate(table["grid"]):
                for column, value in enumerate(row):
                    self._unit(ir, block, value, {"row": row_index, "column": column})
        else:
            self._unit(ir, block, block["text"], {})
        ir["blocks"].append(block)

    def _unit(self, ir, block, text, location):
        if text.strip():
            ir["units"].append({"id": block["id"] + ":u" + str(len(ir["units"])),
                                "block_id": block["id"], "text": text, **location})


def _strip_page_digits(text):
    tokens = []
    for raw in " ".join(str(text).split()).casefold().split():
        cleaned = raw.strip(".,:;()[]|/-–—")
        if not cleaned:
            continue
        if cleaned in ("trang", "page"):
            tokens.append("#page")
        elif all(ch.isdecimal() or ch in "/-" for ch in cleaned):
            tokens.append("#num")
        else:
            tokens.append(cleaned)
    return " ".join(tokens)


def _is_explicit_page_label(text, page_index, page_count):
    clean = " ".join(str(text).split()).casefold().strip("-–—|()[] ")
    if not clean or len(clean) > _FURN_CFG["max_page_label_len"]:
        return False
    for prefix in _FURNITURE_PREFIXES:
        if clean.startswith(prefix):
            rest = clean[len(prefix):].strip()
            parts = [part.strip() for part in rest.split("/") if part.strip()]
            if parts and len(parts) <= 2 and all(part.isdecimal() for part in parts):
                if int(parts[0]) in (page_index, page_index + 1) or (len(parts) == 2 and int(parts[1]) == page_count):
                    return True
    if "/" in clean:
        parts = [part.strip() for part in clean.split("/")]
        if len(parts) == 2 and all(part.isdecimal() for part in parts):
            if int(parts[1]) == page_count and 1 <= int(parts[0]) <= page_count:
                return True
    return False


def _mark_pdf_furniture(ir, page_count):
    """Mark unambiguous cross-page running headers/footers/page_numbers as furniture."""
    if page_count < 2:
        return
    by_page = {}
    for block in ir["blocks"]:
        page_index = block.get("provenance", {}).get("page_index")
        if page_index is not None:
            by_page.setdefault(page_index, []).append(block)
    candidates = []
    signature_pages = {}
    max_words = _FURN_CFG["max_words"]
    top_bbox = _FURN_CFG["top_bbox_threshold"]
    bottom_bbox = _FURN_CFG["bottom_bbox_threshold"]
    top_pos = _FURN_CFG["top_pos_limit"]
    bottom_offset = _FURN_CFG["bottom_pos_offset"]
    min_repeat_pages = _FURN_CFG["min_repeat_pages"]
    for page_index, page_blocks in by_page.items():
        total = len(page_blocks)
        for pos, block in enumerate(page_blocks):
            if block["type"] == "table":
                continue
            text = block.get("text", "").strip()
            if not text or len(text.split()) > max_words:
                continue
            prov = block["provenance"]
            bbox = prov.get("bbox", [0, 0, 1, 1])
            has_box = prov.get("bbox_granularity") != "page"
            if has_box:
                at_top = bbox[3] <= top_bbox
                at_bottom = bbox[1] >= bottom_bbox
            else:
                at_top = pos < top_pos
                at_bottom = pos >= max(0, total - bottom_offset)
            if not (at_top or at_bottom):
                continue
            if _is_explicit_page_label(text, page_index, page_count):
                prov["region"] = "furniture:page_number"
                block.pop("level", None)
                block["type"] = "paragraph"
                continue
            if page_count >= min_repeat_pages:
                band = "top"
                if at_bottom and not at_top:
                    band = "bottom"
                sig = (band, _strip_page_digits(text))
                if sig[1] and not sig[1].startswith(_NON_FURNITURE_PREFIXES):
                    candidates.append((block, page_index, sig))
                    signature_pages.setdefault(sig, set()).add(page_index)
    min_repeat = max(min_repeat_pages, int(page_count * _FURN_CFG["min_repeat_ratio"] + _FURN_CFG["repeat_round_up"]))
    for block, _page_index, sig in candidates:
        if len(signature_pages.get(sig, ())) >= min_repeat:
            if sig[0] == "top":
                block["provenance"]["region"] = "furniture:running_header"
            else:
                block["provenance"]["region"] = "furniture:running_footer"
            block.pop("level", None)
            block["type"] = "paragraph"
