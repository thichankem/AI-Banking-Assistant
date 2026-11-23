"""Worksheet notes, DrawingML, saved charts and existing Cloud OCR."""

from hashlib import sha256

from PIL import Image
from lxml import etree
import pymupdf

from agent_banking.chunking.config.settings import get_config_section
from agent_banking.chunking.publication_io import atomic_bytes, file_hash

from ..docx.image_ocr import _raster_image
from ..pdf.page_review import CloudOCRUnavailable, PdfVisualReviewer
from .package import CellRange, a1, attribute, child, children, local_name, text_runs


_MAX_IMAGE_BYTES = get_config_section("extraction")["docx"]["max_image_bytes"]


def _render_picture(blob, part, max_pixels):
    if not part.casefold().endswith(".svg"):
        return _raster_image(blob, max_pixels)
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    root = etree.fromstring(blob, parser)
    if root.getroottree().docinfo.doctype:
        raise ValueError("Unsupported SVG document type")
    for node in root.iter():
        target = attribute(node, "href")
        if target and not target.startswith(("#", "data:")):
            raise ValueError("SVG uses an external resource")
    with pymupdf.open("svg", blob) as document:
        page = document[0]
        width, height = page.rect.width, page.rect.height
        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise ValueError("Embedded SVG exceeds the pixel limit")
        pixels = page.get_pixmap()
        return pixels.tobytes("png"), pixels.width, pixels.height


def _point(marker):
    if marker is None:
        return None
    try:
        return int(child(marker, "row").text) + 1, int(child(marker, "col").text) + 1
    except (AttributeError, TypeError, ValueError):
        return None


def _anchor(node):
    first = _point(child(node, "from"))
    last = _point(child(node, "to"))
    if first and last and last[0] >= first[0] and last[1] >= first[1]:
        return CellRange(*first, *last).label()
    if first:
        return a1(*first)
    client_data = next((part for part in node.iter() if local_name(part) == "ClientData"), None)
    if client_data is not None:
        value = child(client_data, "Anchor")
        if value is not None and value.text:
            try:
                left, _, top, _, right, _, bottom, _ = (int(part) for part in value.text.split(","))
                return CellRange(top + 1, left + 1, bottom + 1, right + 1).label()
            except ValueError:
                pass
    return None


def chart_lines(root):
    title = next((node for node in root.iter() if local_name(node) == "title"), None)
    title_text = text_runs(title)
    if title is not None and not title_text:
        title_text = " ".join(node.text for node in title.iter() if local_name(node) == "v" and node.text)
    if title_text.strip():
        yield title_text.strip()
    for series in (node for node in root.iter() if local_name(node) == "ser"):
        fields = {}
        for field in series:
            role = local_name(field)
            if role not in {"tx", "cat", "val", "xVal", "yVal", "bubbleSize"}:
                continue
            labels = {}
            for point in (node for node in field.iter() if local_name(node) == "pt"):
                key = point.get("idx", "")
                value = child(point, "v")
                if key.isascii() and key.isdecimal() and value is not None and value.text is not None:
                    labels.setdefault(int(key), []).append(value.text)
            reference = next((node.text for node in field.iter() if local_name(node) == "f" and node.text), None)
            literal = child(field, "v")
            if literal is not None and literal.text:
                labels[0] = [literal.text]
            if labels or reference:
                fields[role] = labels
                if reference:
                    yield f"{role}: {reference}"
        names = fields.pop("tx", {})
        name = " / ".join(names.get(0, []))
        if name:
            yield name
        for idx in sorted({idx for values in fields.values() for idx in values}):
            values = [f"{role}: {' / '.join(points[idx])}" for role, points in fields.items() if idx in points]
            yield " | ".join(filter(None, [name, *values]))


def cell_image_relationships(package, workbook_rels):
    parts = {rel.kind.casefold(): rel.target for rel in workbook_rels.values() if not rel.external}
    metadata_part = parts.get("sheetmetadata")
    binding_part = parts.get("richvaluerel")
    values_part = parts.get("rdrichvalue") or parts.get("richvalue")
    if not all((metadata_part, binding_part, values_part)):
        return {}
    metadata = package.xml(metadata_part)
    value_metadata = child(metadata, "valueMetadata")
    if value_metadata is None:
        return {}
    future = next((node for node in children(metadata, "futureMetadata") if node.get("name") == "XLRICHVALUE"), None)
    names = child(metadata, "metadataTypes")
    if future is None or names is None:
        return {}
    types = [node.get("name") for node in children(names, "metadataType")]
    rich_bindings = []
    for selection in children(future, "bk"):
        binder = next((node for node in selection.iter() if local_name(node) == "rvb"), None)
        rich_bindings.append(attribute(binder, "i"))
    values = [next((item.text for item in children(node, "v")), "") for node in children(package.xml(values_part), "rv")]
    binding_rels = package.relationships(binding_part)
    bindings = [binding_rels.get(attribute(node, "id")) for node in children(package.xml(binding_part), "rel")]
    result = {}
    for idx, record in enumerate(children(value_metadata, "bk"), 1):
        for item in children(record, "rc"):
            try:
                typ = int(item.get("t", "0"))
                link = int(item.get("v", "-1"))
                if typ <= 0 or link < 0 or types[typ - 1] != "XLRICHVALUE":
                    continue
                rich_value = int(rich_bindings[link])
                slot = int(values[rich_value])
                if rich_value < 0 or slot < 0:
                    continue
                rel = bindings[slot]
            except (ValueError, TypeError, IndexError):
                continue
            if rel is not None and rel.kind == "image":
                result[str(idx)] = rel
    return result


class WorkbookObjects:
    def __init__(self, package, ir, asset_dir, cache, vision, review, max_pixels):
        self.package = package
        self.ir = ir
        self.asset_dir = asset_dir
        self.max_pixels = max_pixels
        self.ocr = PdfVisualReviewer(asset_dir, cache, vision, review, False)

    def issue(self, sheet, anchor, kind, status, *, part=None):
        issue = {"sheet_name": sheet, "cell_range": anchor, "kind": kind, "status": status}
        if part:
            issue["part"] = part
        self.ir["excel_issues"].append(issue)

    def notes(self, sheet, relationships):
        for rel in relationships.values():
            if rel.external or rel.kind not in {"comments", "threadedComment", "threadedComments"}:
                continue
            try:
                root = self.package.xml(rel.target)
            except (KeyError, ValueError, etree.XMLSyntaxError) as error:
                self.issue(sheet, None, "note", type(error).__name__, part=rel.target)
                continue
            for note in (node for node in root.iter() if local_name(node) in {"comment", "ThreadedComment", "threadedComment"}):
                text = text_runs(note)
                if not text:
                    value = child(note, "text")
                    if value is not None:
                        text = value.text or ""
                if text.strip():
                    yield note.get("ref"), text, "note", {}

    def drawings(self, sheet, rel):
        if rel is None or rel.external:
            self.issue(sheet, None, "drawing", "unavailable_relationship")
            return
        root = self.package.xml(rel.target)
        links = self.package.relationships(rel.target)
        for shape in root:
            loc = _anchor(shape)
            paragraphs = [text_runs(node) for node in shape.iter() if local_name(node) == "p"]
            if any(text.strip() for text in paragraphs):
                yield loc, "\n".join(paragraphs), "drawing_text", {}
            for item in shape.iter():
                kind = local_name(item)
                if kind in {"blip", "imagedata", "svgBlip"}:
                    identifier = attribute(item, "embed") or attribute(item, "id") or attribute(item, "link")
                    yield from self.image(sheet, loc, links.get(identifier))
                elif kind == "chart":
                    target = links.get(attribute(item, "id"))
                    if target is None or target.external:
                        self.issue(sheet, loc, "chart", "unavailable_relationship")
                        continue
                    try:
                        lines = list(chart_lines(self.package.xml(target.target)))
                    except (KeyError, ValueError, etree.XMLSyntaxError) as error:
                        self.issue(sheet, loc, "chart", type(error).__name__, part=target.target)
                        continue
                    if lines:
                        yield loc, "\n".join(lines), "chart", {"part": target.target}
                    else:
                        self.issue(sheet, loc, "chart", "no_saved_chart_text", part=target.target)

    def image(self, sheet, loc, rel):
        if rel is None or rel.external or rel.kind != "image":
            self.issue(sheet, loc, "image", "unavailable_relationship")
            return
        review = {"sheet_name": sheet, "cell_range": loc, "part": rel.target}
        self.ir["excel_ocr"].append(review)
        try:
            blob = self.package.read(rel.target, limit=_MAX_IMAGE_BYTES)
            pixels, width, height = _render_picture(blob, rel.target, self.max_pixels)
        except (OSError, KeyError, ValueError, RuntimeError, etree.XMLSyntaxError,
                Image.DecompressionBombError, Image.DecompressionBombWarning):
            review["status"] = "unsupported_image"
            self.issue(sheet, loc, "image", review["status"], part=rel.target)
            return
        digest = sha256(pixels).hexdigest()
        target = self.asset_dir / (digest + ".png")
        if not target.is_file() or file_hash(target) != digest:
            atomic_bytes(target, pixels)
        self.ir["assets"][digest] = {"path": str(target.resolve()), "sha256": digest,
                                    "kind": "embedded_image", "width": width, "height": height}
        review["asset_sha256"] = digest
        try:
            result = self.ocr.read(pixels, [], width, height, review_incomplete=True)
        except CloudOCRUnavailable:
            review["status"] = "ocr_unavailable"
            self.issue(sheet, loc, "image", review["status"], part=rel.target)
            return
        review["review"] = result["visual_review"]
        review["status"] = "complete"
        if result["visual_review"].get("issues") or result["visual_review"].get("incomplete"):
            review["status"] = "partial"
            self.issue(sheet, loc, "image", "partial_ocr", part=rel.target)
        for item in result["elements"]:
            text = item.get("text", "")
            if item["type"] == "table":
                data = item["table_data"]
                text = "\n".join(" | ".join(row) for row in data["header_grid"] + data["rows"])
            if text.strip():
                yield loc, text, "image_ocr", {"asset_sha256": digest, "part": rel.target}

