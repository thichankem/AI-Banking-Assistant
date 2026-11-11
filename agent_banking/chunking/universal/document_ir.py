"""Lossless DOCX structural extraction before model-assisted interpretation.

Native text is authoritative. Model annotations never overwrite it. The same
block/cell/asset representation can be supplied by a future PDF adapter.
"""
import hashlib
import io
from pathlib import Path
import zipfile

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import parse_xml
from lxml import etree


def match_numbered_heading(source):
    s = source.lower()
    sep_chars = (':', '.', '-', '–', '—')
    if s.startswith('chương '):
        rest = s[7:].lstrip()
        idx = 0
        while idx < len(rest) and (rest[idx].isdigit() or rest[idx] in 'ivxlcdm'):
            idx += 1
        if idx > 0:
            after = rest[idx:].lstrip()
            if after and after[0] in sep_chars:
                tail = after[1:].lstrip()
                if tail:
                    return 'chapter'
    if s.startswith('điều '):
        rest = s[5:].lstrip()
        idx = 0
        while idx < len(rest) and rest[idx].isdigit():
            idx += 1
        if idx > 0:
            if idx < len(rest) and 'a' <= rest[idx] <= 'z':
                idx += 1
            after = rest[idx:].lstrip()
            if after and after[0] in sep_chars:
                tail = after[1:].lstrip()
                if tail:
                    return 'article'
    if s.startswith('phụ lục'):
        rest = s[7:].lstrip()
        if not rest:
            return None
        if rest[0] in sep_chars:
            tail = rest[1:].lstrip()
            if tail:
                return 'appendix'
        idx = 0
        while idx < len(rest) and (rest[idx].isdigit() or rest[idx] in 'ivxlcdm'):
            idx += 1
        if idx > 0:
            after = rest[idx:].lstrip()
            if after and after[0] in sep_chars:
                tail = after[1:].lstrip()
                if tail:
                    return 'appendix'
    return None


def local_name(node):
    if not isinstance(node.tag, str):
        return ""
    return node.tag.rsplit("}", 1)[-1]


def native_math(node):
    """Retain mathematical operators and argument boundaries, not just glyphs."""
    name = local_name(node)
    if name == "t":
        return node.text or ""
    if name in ("rPr", "ctrlPr", "argPr", "oMathParaPr"):
        return ""
    if name in ("oMath", "oMathPara", "r", "e", "num", "den", "sup", "sub", "deg"):
        return "".join(native_math(child) for child in node)
    operations = {"f": ("frac", ("num", "den")), "sSup": ("superscript", ("e", "sup")),
                  "sSub": ("subscript", ("e", "sub")), "sSubSup": ("subsup", ("e", "sub", "sup")),
                  "rad": ("root", ("deg", "e"))}
    if name in operations:
        label, arguments = operations[name]
        values = []
        for argument in arguments:
            child = next((value for value in node if local_name(value) == argument), None)
            text = ""
            if child is not None:
                text = native_math(child)
            values.append(argument + "=" + text)
        return label + "(" + "; ".join(values) + ")"
    # Less common OMML constructs keep their complete source representation.
    # Flattening a matrix, limit or integral into adjacent text changes meaning.
    return "[OMML " + etree.tostring(node, encoding="unicode", with_tail=False) + "]"


def native_text(node):
    name = local_name(node)
    if name in ("del", "moveFrom", "drawing", "pict", "txbxContent"):
        return ""
    if name in ("t", "instrText"):
        if name == "instrText":
            return ""
        return node.text or ""
    if name in ("br", "cr"):
        return "\n"
    if name == "tab":
        return "\t"
    if name == "noBreakHyphen":
        return "‑"
    if name == "softHyphen":
        return "\u00ad"
    if name == "sym":
        return chr(int(node.get(qn("w:char"), "0"), 16))
    if name in ("footnoteReference", "endnoteReference"):
        return " [" + name.replace("Reference", "") + ":" + str(node.get(qn("w:id"), "")) + "] "
    if name in ("oMath", "oMathPara"):
        return native_math(node)
    return "".join(native_text(child) for child in node)


def heading_level(paragraph, ordinal):
    text = paragraph.text.strip()
    if not text:
        return 0
    outline = paragraph._p.find(".//" + qn("w:outlineLvl"))
    if outline is None and paragraph.style is not None:
        outline = paragraph.style.element.find(".//" + qn("w:outlineLvl"))
    if outline is not None:
        value = int(outline.get(qn("w:val"), "9"))
        if value < 9:
            return value + 1
    # Only authored OpenXML outline metadata is structural evidence. Content,
    # capitalisation, font size and language-specific words are not rules.
    return 0


class DocxExtractor:
    def __init__(self, asset_dir, render_figures=False):
        self.asset_dir = Path(asset_dir)
        self.render_figures = render_figures

    def extract(self, source, source_name):
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            data = Path(source).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        self._source_bytes = data
        identity = str(source_name).replace("\\", "/")
        document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        self.document = docx.Document(io.BytesIO(data))
        self.ir = {"version": "2.0", "document_id": document_id,
                   "document_version": digest, "source": identity, "blocks": [],
                   "units": [], "assets": {}, "warnings": [], "relations": []}
        self._unit_ids = set()
        self._seen_drawings = set()
        self._list_counters = {}
        self._paragraph_values = {}
        self._walk(self.document.element.body, self.document.part, "body")
        self._infer_numbered_outline()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in ("word/footnotes.xml", "word/endnotes.xml"):
                if name in archive.namelist():
                    root = parse_xml(archive.read(name))
                    owner = next((part for part in self.document.part.package.parts if str(part.partname) == "/" + name), self.document.part)
                    for note in root:
                        note_type = note.get(qn("w:type"), "normal")
                        if note_type not in ("separator", "continuationSeparator"):
                            self._walk(note, owner, name + ":" + str(note.get(qn("w:id"))))
        # Header/footer content is preserved once per part, outside the body.
        seen_parts = set()
        for relationship in self.document.part.rels.values():
            if relationship.is_external:
                continue
            part = relationship.target_part
            name = str(part.partname)
            if ("/header" in name or "/footer" in name) and name not in seen_parts:
                seen_parts.add(name)
                self._walk(part.element, part, "furniture:" + name)
        self._link_captions()
        self._link_notes()
        self.ir["statistics"] = {"blocks": len(self.ir["blocks"]), "units": len(self.ir["units"]),
                                  "image_occurrences": sum(block["type"] == "image" for block in self.ir["blocks"]),
                                  "tables": sum(block["type"] == "table" for block in self.ir["blocks"])}
        return self.ir

    def _append(self, kind, text, node, region, **attributes):
        index = len(self.ir["blocks"])
        block_id = self.ir["document_id"] + ":v" + self.ir["document_version"][:16] + ":b" + str(index)
        part_name = "word/document.xml"
        if region.startswith("furniture:"):
            part_name = region.split(":", 1)[1]
        elif not region.startswith("body"):
            part_name = region.split(":", 1)[0]
        block = {"id": block_id, "type": kind, "text": text, "order": index,
                 "provenance": {"part": part_name,
                                "xpath": node.getroottree().getpath(node), "region": region}}
        block.update(attributes)
        self.ir["blocks"].append(block)
        return block

    def _unit(self, text, node, region, block_id):
        if not text.strip():
            return None
        key = self.ir["document_id"] + self.ir["document_version"] + region + ":" + node.getroottree().getpath(node)
        unit_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        if unit_id in self._unit_ids:
            return unit_id
        self._unit_ids.add(unit_id)
        self.ir["units"].append({"id": unit_id, "block_id": block_id, "text": text,
                                  "region": region, "xpath": node.getroottree().getpath(node)})
        return unit_id

    def _walk(self, container, part, region):
        for node in container:
            name = local_name(node)
            if name in ("del", "moveFrom", "sectPr", "sdtPr", "sdtEndPr", "pPr", "tblPr", "tblGrid"):
                continue
            if name == "p":
                self._paragraph(node, part, region)
            elif name == "tbl":
                self._table(node, part, region)
            elif name in ("altChunk", "object"):
                self.ir["warnings"].append({"code": "UNSUPPORTED_EMBEDDED_CONTENT", "region": region,
                                               "xpath": node.getroottree().getpath(node), "requires_review": True})
            else:
                self._walk(node, part, region)

    def _paragraph(self, node, part, region):
        paragraph = docx.text.paragraph.Paragraph(node, self.document)
        text = self._paragraph_text(node)
        level = heading_level(paragraph, len(self.ir["blocks"]))
        events = list(self._inline_events(node))
        prefix = ""
        plain = native_text(node)
        if text.endswith(plain) and len(text) > len(plain):
            prefix = text[:len(text) - len(plain)]
        buffer = prefix
        segment = 0
        def emit(value):
            nonlocal segment
            if not value.strip():
                return
            kind = "paragraph"
            if level and segment == 0:
                kind = "heading"
            elif paragraph.style is not None and paragraph.style.style_id == "Caption":
                kind = "caption"
            block = self._append(kind, value, node, region, level=level, inline_segment=segment)
            if segment == 0:
                runs = [run for run in paragraph.runs if run.text.strip()]
                sizes = {run.font.size for run in runs}
                if runs and all(run.bold is True for run in runs) and len(sizes) == 1 and None not in sizes:
                    block["heading_format"] = {"bold": True, "font_pt": sizes.pop().pt,
                                               "centered": paragraph.alignment == WD_ALIGN_PARAGRAPH.CENTER}
            unit_id = self._unit(value, node, region + ":inline" + str(segment), block["id"])
            block["unit_ids"] = [unit_id]
            segment += 1
            num = node.find(".//" + qn("w:numPr"))
            if num is not None:
                block["numbering"] = {local_name(item): item.get(qn("w:val")) for item in num}
            block["note_references"] = [{"type": local_name(item), "id": item.get(qn("w:id"))}
                for item in node.iter() if local_name(item) in ("footnoteReference", "endnoteReference")]
            hyperlinks = []
            for link in node.iter(qn("w:hyperlink")):
                rel_id = link.get(qn("r:id"))
                if rel_id and rel_id in part.rels:
                    hyperlinks.append({"text": native_text(link), "target": part.rels[rel_id].target_ref})
            block["hyperlinks"] = hyperlinks
        for kind, value in events:
            if kind == "text":
                buffer += value
            else:
                emit(buffer)
                buffer = ""
                self._pictures(value, part, region)
                for textbox in value.iter(qn("w:txbxContent")):
                    self._walk(textbox, part, region + ":textbox")
        emit(buffer)

    def _infer_numbered_outline(self):
        """Recover manually formatted Vietnamese regulatory outlines."""
        body = [block for block in self.ir["blocks"] if block["provenance"]["region"] == "body"]
        if any(block["type"] == "heading" for block in body):
            return
        paragraphs = [block for block in body if block["type"] == "paragraph"]
        candidates = []
        for block in paragraphs:
            source = block["text"].strip()
            if not block.get("heading_format") or "\n" in source or len(source.split()) > 45:
                continue
            heading_role = match_numbered_heading(source)
            if heading_role is not None:
                candidates.append((block, heading_role))
        if sum(role in ("chapter", "article") for _, role in candidates) < 2:
            return

        def mark(block, level, role):
            block["type"] = "heading"
            block["level"] = level
            block["heading_evidence"] = {"source": "numbered_outline_formatting", "role": role}

        title = paragraphs[0]
        title_style = title.get("heading_format", {})
        has_title = title_style.get("centered") and title_style["font_pt"] > max(
            block["heading_format"]["font_pt"] for block, _ in candidates
        )
        if has_title:
            section_level = 2
        else:
            section_level = 1
        if has_title:
            mark(title, 1, "title")
        has_parent = False
        for block, role in candidates:
            level = section_level + (role == "article" and has_parent)
            mark(block, level, role)
            has_parent = has_parent or role in ("chapter", "appendix")

    def _inline_events(self, node):
        for child in node:
            name = local_name(child)
            if name in ("pPr", "rPr", "del", "moveFrom"):
                continue
            if name in ("drawing", "pict", "txbxContent"):
                yield "visual", child
            elif name in ("t", "tab", "br", "cr", "noBreakHyphen", "softHyphen", "sym", "footnoteReference", "endnoteReference", "oMath", "oMathPara"):
                yield "text", native_text(child)
            else:
                yield from self._inline_events(child)

    def _pictures(self, node, part, region):
        for image_node in node.iter():
            name = local_name(image_node)
            ancestor = image_node.getparent()
            nested_textbox = False
            while ancestor is not None and ancestor is not node:
                if local_name(ancestor) == "txbxContent":
                    nested_textbox = True
                    break
                ancestor = ancestor.getparent()
            if nested_textbox:
                continue
            rel_id = None
            asset_kind = "image"
            if name == "blip":
                rel_id = image_node.get(qn("r:embed")) or image_node.get(qn("r:link"))
            elif name == "imagedata":
                rel_id = image_node.get(qn("r:id"))
            elif name == "chart":
                rel_id = image_node.get(qn("r:id"))
                asset_kind = "ooxml_chart"
            elif name == "relIds":
                rel_id = image_node.get(qn("r:dm"))
                asset_kind = "ooxml_diagram"
            if not rel_id:
                continue
            key = str(part.partname) + ":" + image_node.getroottree().getpath(image_node)
            if key in self._seen_drawings:
                continue
            self._seen_drawings.add(key)
            relationship = part.rels.get(rel_id)
            if relationship is None or relationship.is_external:
                self.ir["warnings"].append({"code": "UNRESOLVED_IMAGE", "relationship_id": rel_id, "requires_review": True})
                continue
            asset = relationship.target_part
            blob = asset.blob
            digest = hashlib.sha256(blob).hexdigest()
            extension = Path(str(asset.partname)).suffix or ".bin"
            destination = self.asset_dir / (digest + extension)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                from ..run_chunking import atomic_bytes
                atomic_bytes(destination, blob)
            info = {"image_id": digest[:24], "sha256": digest, "path": str(destination.resolve()),
                    "part": str(asset.partname), "content_type": asset.content_type,
                    "relationship_id": rel_id, "status": "native_asset", "caption": "", "description": ""}
            info["asset_kind"] = asset_kind
            source_text = ""
            if asset_kind != "image":
                xml = parse_xml(blob)
                values = [item.text for item in xml.iter() if local_name(item) in ("t", "v") and item.text]
                info["native_values"] = values
                source_text = "Dữ liệu hình gốc:\n" + "\n".join(values)
                info["native_text"] = source_text
                if self.render_figures:
                    from .office_figures import drawing_preview
                    preview = drawing_preview(self._source_bytes, self.document.element, image_node)
                    visual_hash = hashlib.sha256(preview).hexdigest()
                    visual_path = self.asset_dir / (visual_hash + ".png")
                    from ..run_chunking import atomic_bytes
                    atomic_bytes(visual_path, preview)
                    info["visual_path"] = str(visual_path.resolve())
                    info["visual_sha256"] = visual_hash
            anchor = image_node
            while anchor.getparent() is not None and local_name(anchor) not in ("drawing", "pict"):
                anchor = anchor.getparent()
            properties = [item for item in anchor.iter() if local_name(item) == "docPr"]
            if properties:
                info["alt_text"] = properties[0].get("descr", "")
                info["title"] = properties[0].get("title", "")
            info["drawing_position"] = [{"type": local_name(item), "attributes": dict(item.attrib), "text": item.text}
                for item in anchor.iter() if local_name(item) in ("extent", "positionH", "positionV", "align", "posOffset")]
            crops = [item for item in anchor.iter() if local_name(item) == "srcRect"]
            transforms = [item for item in anchor.iter() if local_name(item) == "xfrm"]
            if crops:
                info["crop"] = dict(crops[0].attrib)
            styled = any(any(int(item.get(name, "0")) != 0 for name in ("l", "t", "r", "b")) for item in crops)
            styled = styled or any(item.get("rot", "0") != "0" or item.get("flipH", "0") not in ("0", "false")
                                   or item.get("flipV", "0") not in ("0", "false") for item in transforms)
            if asset_kind == "image" and styled and self.render_figures:
                from .office_figures import drawing_preview
                preview = drawing_preview(self._source_bytes, self.document.element, image_node, allow_blank=True)
                visual_hash = hashlib.sha256(preview).hexdigest()
                visual_path = self.asset_dir / (visual_hash + ".png")
                from ..run_chunking import atomic_bytes
                atomic_bytes(visual_path, preview)
                info["visual_path"] = str(visual_path.resolve())
                info["visual_sha256"] = visual_hash
                info["visual_method"] = "office_rendered_crop_or_transform"
            self.ir["assets"][digest] = info
            block = self._append("image", source_text, image_node, region, image=info.copy(), unit_ids=[])
            if source_text:
                unit_id = self._unit(source_text, image_node, str(asset.partname), block["id"])
                block["unit_ids"].append(unit_id)

    def _table(self, node, part, region, parent_cell=None):
        rows = node.findall(qn("w:tr"))
        grid = []
        cells = []
        previous = {}
        nested = []
        units = []
        header_count = 0
        for row_index, row in enumerate(rows):
            props = row.find(qn("w:trPr"))
            before = 0
            is_header = False
            if props is not None:
                entry = props.find(qn("w:gridBefore"))
                if entry is not None:
                    before = int(entry.get(qn("w:val"), "0"))
                header = props.find(qn("w:tblHeader"))
                is_header = header is not None and header.get(qn("w:val"), "true").lower() not in ("0", "false", "off")
            if is_header and header_count == row_index:
                header_count += 1
            values = [""] * before
            column = before
            for cell in row.findall(qn("w:tc")):
                cell_props = cell.find(qn("w:tcPr"))
                colspan = 1
                merge = None
                if cell_props is not None:
                    span = cell_props.find(qn("w:gridSpan"))
                    if span is not None:
                        colspan = int(span.get(qn("w:val"), "1"))
                        if colspan < 1:
                            raise ValueError("Invalid nonpositive table gridSpan")
                    merge = cell_props.find(qn("w:vMerge"))
                paragraphs = []
                for child in cell:
                    self._cell_content(child, paragraphs, nested, part, region)
                text = "\n".join(self._paragraph_text(paragraph) for paragraph in paragraphs).strip()
                cell_id = self.ir["document_id"] + ":v" + self.ir["document_version"][:16] + ":cell:" + str(len(self.ir["blocks"])) + ":" + str(row_index) + ":" + str(column)
                record = {"id": cell_id, "row": row_index, "column": column,
                          "row_span": 1, "col_span": colspan, "text": text,
                          "xpath": cell.getroottree().getpath(cell)}
                runs = [run for paragraph in paragraphs for run in docx.text.paragraph.Paragraph(paragraph, self.document).runs if run.text.strip()]
                record["all_text_bold"] = bool(runs) and all(run.bold for run in runs)
                if merge is not None and merge.get(qn("w:val")) != "restart" and column in previous:
                    origin = previous[column]
                    origin["row_span"] += 1
                    record["merged_into"] = origin["id"]
                    if not text:
                        text = origin["text"]
                else:
                    for offset in range(colspan):
                        previous[column + offset] = record
                if merge is None:
                    for offset in range(colspan):
                        previous.pop(column + offset, None)
                cells.append(record)
                values.extend([text] * colspan)
                column += colspan
                for paragraph in paragraphs:
                    units.append((self._paragraph_text(paragraph), paragraph))
            grid.append(values)
        width = max([len(row) for row in grid] + [0])
        for row in grid:
            row.extend([""] * (width - len(row)))
        header_detection = "unmarked"
        if header_count:
            header_detection = "explicit_repeat_header"
        elif self._has_grouped_header(cells, width):
            header_count = 2
            header_detection = "merged_bold_header_rows"
        # Ambiguous unmarked tables retain every row as data.
        if header_count:
            headers = []
            for column in range(width):
                labels = []
                for row in grid[:header_count]:
                    if row[column] and row[column] not in labels:
                        labels.append(row[column])
                headers.append(" / ".join(labels))
        else:
            headers = ["Cột " + str(index + 1) for index in range(width)]
        table = {"headers": headers, "rows": grid[header_count:], "grid": grid,
                 "header_row_count": header_count, "header_detection": header_detection, "cells": cells, "parent_cell": parent_cell}
        block = self._append("table", "", node, region, table_data=table, unit_ids=[])
        table["table_id"] = block["id"]
        for text, paragraph in units:
            unit_id = self._unit(text, paragraph, region, block["id"])
            if unit_id:
                block["unit_ids"].append(unit_id)
        for cell in cells:
            self.ir["relations"].append({"from": block["id"], "to": cell["id"], "type": "contains_cell"})
        for child in nested:
            self._table(child, part, region + ":nested", block["id"])

    @staticmethod
    def _has_grouped_header(cells, width):
        """Require visual and grid evidence for two unmarked header rows."""
        groups, labels, first_record = (
            [cell for cell in cells if cell["row"] == row and cell["text"]]
            for row in range(3)
        )
        if not all(cell["all_text_bold"] and cell["row"] + cell["row_span"] <= 2
                   for cell in groups + labels):
            return False
        if not any(not cell["all_text_bold"] for cell in first_record):
            return False
        # A whole-width merged title above a table provides no column mapping.
        return any(
            1 < group["col_span"] < width
            and sum(group["column"] <= label["column"] < group["column"] + group["col_span"]
                    for label in labels) >= 2
            for group in groups
        )

    def _cell_content(self, node, paragraphs, nested, part, region):
        name = local_name(node)
        if name in ("del", "moveFrom", "tcPr"):
            return
        if name == "tbl":
            nested.append(node)
            return
        if name == "p":
            paragraphs.append(node)
            self._pictures(node, part, region + ":cell")
            for textbox in node.iter(qn("w:txbxContent")):
                self._walk(textbox, part, region + ":cell:textbox")
            return
        for child in node:
            self._cell_content(child, paragraphs, nested, part, region)

    def _link_captions(self):
        blocks = self.ir["blocks"]
        for index, block in enumerate(blocks):
            if block["type"] != "caption":
                continue
            candidates = []
            if index:
                candidates.append(blocks[index - 1])
            if index + 1 < len(blocks):
                candidates.append(blocks[index + 1])
            for candidate in candidates:
                if candidate["type"] in ("image", "table"):
                    candidate["caption"] = block["text"]
                    if candidate["type"] == "image":
                        candidate["image"]["caption"] = block["text"]
                    self.ir["relations"].append({"from": block["id"], "to": candidate["id"], "type": "caption_of"})
                    break

    def _link_notes(self):
        for block in self.ir["blocks"]:
            for reference in block.get("note_references", []):
                part = "word/footnotes.xml:"
                if reference["type"] == "endnoteReference":
                    part = "word/endnotes.xml:"
                target = part + str(reference["id"])
                matches = [candidate for candidate in self.ir["blocks"] if candidate["provenance"]["region"] == target]
                if not matches:
                    self.ir["warnings"].append({"code": "UNRESOLVED_NOTE", "block_id": block["id"], "target": target, "requires_review": True})
                for candidate in matches:
                    self.ir["relations"].append({"from": candidate["id"], "to": block["id"], "type": "note_of"})

    def _paragraph_text(self, node):
        if node in self._paragraph_values:
            return self._paragraph_values[node]
        text = native_text(node)
        paragraph = docx.text.paragraph.Paragraph(node, self.document)
        num = node.find("./" + qn("w:pPr") + "/" + qn("w:numPr"))
        if num is None and paragraph.style is not None:
            num = paragraph.style.element.find("./" + qn("w:pPr") + "/" + qn("w:numPr"))
        if num is not None and text.strip():
            num_id = num.find(qn("w:numId"))
            level = num.find(qn("w:ilvl"))
            if num_id is not None and num_id.get(qn("w:val")) != "0":
                identifier = num_id.get(qn("w:val"))
                depth = 0
                if level is not None:
                    depth = int(level.get(qn("w:val"), "0"))
                marker = self._list_marker(identifier, depth)
                if marker:
                    text = marker + " " + text
        self._paragraph_values[node] = text
        return text

    def _list_marker(self, identifier, depth):
        numbering = self.document.part.numbering_part.element
        definitions = numbering.findall(qn("w:num"))
        actual = next((item for item in definitions if item.get(qn("w:numId")) == identifier), None)
        if actual is None:
            self.ir["warnings"].append({"code": "UNRESOLVED_NUMBERING", "num_id": identifier, "requires_review": True})
            return ""
        abstract_id = actual.find(qn("w:abstractNumId")).get(qn("w:val"))
        abstract = next(item for item in numbering.findall(qn("w:abstractNum")) if item.get(qn("w:abstractNumId")) == abstract_id)
        levels = {int(item.get(qn("w:ilvl"), "0")): item for item in abstract.findall(qn("w:lvl"))}
        level = levels.get(depth)
        if level is None:
            self.ir["warnings"].append({"code": "UNRESOLVED_LIST_LEVEL", "num_id": identifier, "requires_review": True})
            return ""
        def value(parent, tag, default):
            item = parent.find(qn("w:" + tag))
            if item is not None:
                return item.get(qn("w:val"), default)
            return default
        start = int(value(level, "start", "1"))
        for override in actual.findall(qn("w:lvlOverride")):
            if int(override.get(qn("w:ilvl"), "0")) == depth:
                start = int(value(override, "startOverride", str(start)))
                replacement = override.find(qn("w:lvl"))
                if replacement is not None:
                    level = replacement
        key = (identifier, depth)
        self._list_counters[key] = self._list_counters.get(key, start - 1) + 1
        for old_key in list(self._list_counters):
            if old_key[0] == identifier and old_key[1] > depth:
                del self._list_counters[old_key]
        marker = value(level, "lvlText", "%" + str(depth + 1) + ".")
        for index in range(depth + 1):
            placeholder = "%" + str(index + 1)
            if placeholder not in marker:
                continue
            counter = self._list_counters.get((identifier, index), 1)
            format_level = levels.get(index, level)
            if index == depth:
                format_level = level
            fmt = value(format_level, "numFmt", "decimal")
            label = str(counter)
            if fmt == "decimalZero":
                label = str(counter).zfill(2)
            elif fmt == "none":
                label = ""
            elif fmt in ("lowerLetter", "upperLetter"):
                label = ""
                remaining = counter
                while remaining:
                    remaining, digit = divmod(remaining - 1, 26)
                    label = chr(97 + digit) + label
                if fmt == "upperLetter":
                    label = label.upper()
            elif fmt in ("lowerRoman", "upperRoman"):
                label = ""
                remaining = counter
                for amount, numeral in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
                    while remaining >= amount:
                        label += numeral
                        remaining -= amount
                if fmt == "lowerRoman":
                    label = label.lower()
            elif fmt != "decimal":
                self.ir["warnings"].append({"code": "UNSUPPORTED_NUMBERING_FORMAT", "num_id": identifier,
                                             "format": fmt, "level": index, "requires_review": True})
                return ""
            marker = marker.replace(placeholder, label)
        return marker
