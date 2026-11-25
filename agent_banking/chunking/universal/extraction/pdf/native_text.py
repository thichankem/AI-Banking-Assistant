"""Extract native PDF text and reconstruct tables and page furniture."""
from collections import Counter
from threading import Lock

from agent_banking.chunking.config.settings import get_config_section, get_rules
from .layout import compact, inside_box, is_inline_list_marker, layout_text, reading_order
from .raster_text import region_may_contain_text
from .tables import validate_table
from .visibility import hidden_glyphs, visible_span

_NATIVE_CFG = get_config_section("extraction")["pdf"]["native_text"]
_FURNITURE_PREFIXES = tuple(get_rules("pdf")["furniture_prefixes"])
NATIVE_TABLE_LOCK = Lock()


def _has_soft_hyphen_digit(text):
    cursor = text.find("\u00ad")
    while cursor != -1:
        pos = cursor + 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos < len(text) and text[pos].isdecimal():
            return True
        cursor = text.find("\u00ad", cursor + 1)
    return False


def unreliable_native_text(text):
    return "\ufffd" in text or _has_soft_hyphen_digit(text)


def normalized_page_box(page, bounds):
    from pymupdf import Rect
    rect = (Rect(bounds) * page.rotation_matrix) & page.rect
    if rect.is_empty:
        return None
    width, height = page.rect.width, page.rect.height
    return [(rect.x0 - page.rect.x0) / width, (rect.y0 - page.rect.y0) / height,
            (rect.x1 - page.rect.x0) / width, (rect.y1 - page.rect.y0) / height]


def native_text_elements(page, clip=None, *, hidden=None):
    import pymupdf
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES & ~pymupdf.TEXT_USE_CID_FOR_UNKNOWN_UNICODE
    if hidden is None:
        hidden = hidden_glyphs(page)
    if hidden:
        form = "rawdict"
    else:
        form = "dict"
    blocks = page.get_text(form, clip=clip, flags=flags)["blocks"]
    elements = []
    for block in blocks:
        lines = []
        for line in block["lines"]:
            fragments = []
            visible = [(span, *visible_span(span, hidden)) for span in line["spans"]]
            spans = [span for span, value, _ in visible if value.strip()]
            largest = max(spans, key=lambda span: span["size"], default=None)
            bounds = [box for _, value, boxes in visible if value.strip() for box in boxes]
            if largest is None or not bounds:
                continue
            for span, value, _ in visible:
                if value.strip():
                    if span["flags"] & pymupdf.TEXT_FONT_SUPERSCRIPT:
                        value = "<sup>" + value + "</sup>"
                    elif (largest is not None and line["dir"][0] > _NATIVE_CFG["line_dir_threshold"]
                          and span["size"] < largest["size"] * _NATIVE_CFG["subscript_size_ratio"]
                          and span["origin"][1] - largest["origin"][1] > largest["size"] * _NATIVE_CFG["subscript_origin_ratio"]):
                        value = "<sub>" + value + "</sub>"
                fragments.append(value)
            text = "".join(fragments)
            bbox = normalized_page_box(page, [min(box[0] for box in bounds), min(box[1] for box in bounds),
                                              max(box[2] for box in bounds), max(box[3] for box in bounds)])
            if text.strip() and bbox is not None:
                lines.append((text, bbox, largest["size"]))
        # MuPDF groups distant text on the same row into one block. Recover
        # the individual line boxes where such a block crosses a column gap.
        col_gap = _NATIVE_CFG["column_gap_threshold"]
        separated = any(
            min(right[3], left[3]) - max(right[1], left[1]) >= min(right[3] - right[1], left[3] - left[1]) / 2
            and max(right[0] - left[2], left[0] - right[2]) >= col_gap
            for index, (_, left, _) in enumerate(lines) for _, right, _ in lines[index + 1:]
        )
        if separated:
            groups = [[item] for item in lines]
        else:
            groups = [lines]
        for group in groups:
            if not group:
                continue
            boxes = [bbox for _, bbox, _ in group]
            bbox = [min(box[0] for box in boxes), min(box[1] for box in boxes),
                    max(box[2] for box in boxes), max(box[3] for box in boxes)]
            elements.append({"type": "paragraph", "text": "\n".join(text for text, _, _ in group),
                             "bbox": bbox, "native_font_size": max(size for _, _, size in group),
                             "order_evidence": "native_reading_order"})
    return reading_order(elements)


def page_requires_scan(page, elements):
    """Recognize missing text and pages dominated by a raster document image."""
    from pymupdf import Rect
    if any(unreliable_native_text(item["text"]) for item in elements):
        return True
    page_area = page.rect.get_area()
    if page_area <= 0:
        return False
    body_elements = [element for element in elements if furniture_key(element) is None]
    if not body_elements:
        # A page with visible painting and no native text needs visual review.
        # This includes cropped raster scans and outlined text on a page
        # that also has a native page number.
        return any(
            kind != "ignore-text" and (not elements or not kind.endswith("-text"))
            and ((Rect(box) * page.rotation_matrix) & page.rect).get_area() > 0
            for kind, box in page.get_bboxlog()
        )
    # Check large scan layers by area. A smaller raster can still contain a
    # pasted paragraph or table among visible native text.
    boxes = [(Rect(image["bbox"]) * page.rotation_matrix) & page.rect for image in page.get_image_info()]
    boxes = {tuple(box) for box in boxes if not box.is_empty}
    if _image_area_reaches(boxes, page_area * _NATIVE_CFG["scan_image_area_ratio"]):
        return True
    return any(region_may_contain_text(page, box, elements) for box in sorted(boxes))


def _image_area_reaches(boxes, threshold):
    areas = [(right - left) * (bottom - top) for left, top, right, bottom in boxes]
    if sum(areas) < threshold:
        return False
    if max(areas) >= threshold:
        return True
    # Count overlapping tiles only once, even when distant artwork makes
    # the bounding box much larger than the area actually painted.
    horizontal_edges = sorted({edge for left, _, right, _ in boxes for edge in (left, right)})
    area = 0.0
    for start, end in zip(horizontal_edges, horizontal_edges[1:]):
        intervals = sorted((top, bottom) for left, top, right, bottom in boxes if left <= start < right)
        if not intervals:
            continue
        covered = 0.0
        low, high = intervals[0]
        for top, bottom in intervals[1:]:
            if top > high:
                covered += high - low
                low, high = top, bottom
            else:
                high = max(high, bottom)
        area += (end - start) * (covered + high - low)
        if area >= threshold:
            return True
    return False


def furniture_key(element):
    top, bottom = element["bbox"][1], element["bbox"][3]
    text = compact(element["text"])
    if bottom <= _NATIVE_CFG["furniture_top_max"]:
        return "top", text
    if top >= _NATIVE_CFG["furniture_bottom_min"]:
        return "bottom", text
    return None


def furniture_signature(element, key):
    return key, tuple(round(value, 2) for value in element["bbox"]), round(element.get("native_font_size", 0), 1)


def native_page_number(text):
    text = text.strip("-–— ")
    prefixed = False
    for prefix in _FURNITURE_PREFIXES:
        if text.casefold().startswith(prefix):
            text = text[len(prefix):]
            prefixed = True
            break
    fragments = text.split("/")
    if len(fragments) > 2 or not all(part.strip().isdecimal() for part in fragments):
        return None
    try:
        return prefixed, tuple(int(part.strip()) for part in fragments)
    except ValueError:
        return None


def is_furniture(element, repeated, page_elements=(), page_number=None, page_count=None):
    key = furniture_key(element)
    if key is None:
        return False
    # A list marker can occupy its own text box near the edge of a page.
    if is_inline_list_marker(element, page_elements):
        return False
    left, top, right, bottom = element["bbox"]
    side = key[0]
    extreme_margin = (side == "top" and bottom <= _NATIVE_CFG["extreme_top_max"]) or (side == "bottom" and top >= _NATIVE_CFG["extreme_bottom_min"])
    number = native_page_number(key[1])
    if number:
        prefixed, values = number
        # Bare numbers at the edge may be amounts or section markers.
        center = (left + right) / 2
        matching_label = values[0] == page_number and (len(values) == 1 or values[1] == page_count)
        return (prefixed and matching_label
                or extreme_margin and center >= _NATIVE_CFG["center_min"] and (prefixed or matching_label))
    if furniture_signature(element, key) not in repeated:
        return False
    if not extreme_margin:
        return False
    body = [item for item in page_elements if furniture_key(item) is None]
    if not body:
        return False
    gaps = []
    for item in body:
        if side == "top":
            gaps.append(item["bbox"][1] - bottom)
        else:
            gaps.append(top - item["bbox"][3])
    nearest_gap = min(gaps)
    # Repetition alone is ordinary in procedures and multi-page tables. Only
    # exclude isolated marginal text with supporting typographic evidence.
    size = element.get("native_font_size", 0)
    body_size = sorted(item.get("native_font_size", 0) for item in body)[len(body) // 2]
    return nearest_gap >= _NATIVE_CFG["furniture_nearest_gap_min"] and 0 < size <= _NATIVE_CFG["furniture_font_ratio_max"] * body_size


def native_table_elements(page, *, hidden=None):
    """Copy native table cells and their spans before PyMuPDF reuses page state."""
    from pymupdf import Rect
    elements = []
    if hidden is None:
        hidden = hidden_glyphs(page)
    with NATIVE_TABLE_LOCK:
        # Use the PDF's vector grid to preserve merged cells and multirow headers.
        for table in page.find_tables(use_layout=False).tables:
            bbox = normalized_page_box(page, table.bbox)
            source_rows = [row.cells for row in table.rows]
            if bbox is None or len(source_rows) < 2 or len(source_rows[0]) < 2:
                continue
            columns = len(source_rows[0])
            if any(len(row) != columns or not any(row) for row in source_rows):
                continue
            grid = [[""] * columns for _ in source_rows]
            x_positions = sorted({cell[0] for row in source_rows for cell in row if cell})
            y_positions = [min(cell[1] for cell in row if cell) for row in source_rows]
            cells = []
            for row_index, row in enumerate(source_rows):
                for column_index, cell in enumerate(row):
                    if cell is None:
                        continue
                    span_tol = _NATIVE_CFG["table_span_tolerance"]
                    text = "\n".join(item["text"] for item in native_text_elements(page, clip=Rect(cell), hidden=hidden))
                    row_span = 1 + sum(y_positions[row_index] < y < cell[3] - span_tol for y in y_positions)
                    column_span = 1 + sum(x_positions[column_index] < x < cell[2] - span_tol for x in x_positions)
                    cells.append({"row": row_index, "column": column_index, "row_span": row_span,
                                  "column_span": column_span, "text": text})
                    grid[row_index][column_index] = text
                    for r in range(row_index, row_index + row_span):
                        for c in range(column_index, column_index + column_span):
                            if r < len(grid) and c < len(grid[r]):
                                grid[r][c] = text
            if not any(any(row) for row in grid):
                continue
            header_count = getattr(table, "header_rows", 0)
            if not isinstance(header_count, int) or not 0 <= header_count < len(grid):
                header_count = 0
            if not header_count and not table.header.external:
                # The detected header box can extend over several physical rows
                # when its first cell spans the group. Require merged-cell
                # evidence and align the boundary to a real row start.
                stop = table.header.bbox[3]
                span_tol = _NATIVE_CFG["table_span_tolerance"]
                count = sum(y < stop - span_tol for y in y_positions)
                if (1 < count < len(grid) and abs(y_positions[count] - stop) <= span_tol
                        and any(cell["row"] < count and cell["column_span"] > 1 for cell in cells)
                        and not any(cell["row"] < count < cell["row"] + cell["row_span"] for cell in cells)):
                    header_count = count
            headers = [""] * len(grid[0])
            if header_count:
                for index in range(len(headers)):
                    labels = [cell["text"] for cell in cells if cell["row"] < header_count and
                              cell["column"] <= index < cell["column"] + cell["column_span"] and cell["text"]]
                    headers[index] = " / ".join(labels)
            data = {"headers": headers, "header_grid": grid[:header_count], "header_row_count": header_count,
                    "rows": grid[header_count:], "cells": cells}
            try:
                validate_table(data)
            except ValueError:
                # Keep the source text in native blocks if the detected table
                # geometry is inconsistent (e.g. skewed or intersecting rules).
                continue
            elements.append({"type": "table", "text": "", "bbox": bbox, "table_data": data,
                             "order_evidence": "native_reading_order"})
    return elements


def native_page(page, repeated_margins):
    hidden = hidden_glyphs(page)
    elements = native_text_elements(page, hidden=hidden)
    native = elements
    for table in native_table_elements(page, hidden=hidden):
        # Replace the corresponding native blocks only when the matrix
        # accounts for all their words, including repeated occurrences.
        encompassed = [item for item in native if inside_box(item["bbox"], table["bbox"])]
        written = "\n".join(item["text"] for item in encompassed)
        extracted = layout_text({"elements": [table]})
        if not encompassed or Counter(compact(written).split()) != Counter(compact(extracted).split()):
            continue
        native = [item for item in native if item not in encompassed]
        native.append(table)
    native = [element for element in native if element["type"] == "table"
              or not is_furniture(element, repeated_margins, elements, page.number + 1, len(page.parent))]
    return reading_order(native)
