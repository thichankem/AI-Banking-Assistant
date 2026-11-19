"""Coordinate, ordering and validation rules for PDF page layouts."""
import io
import unicodedata

from agent_banking.chunking.config.settings import get_config_section, get_rules
from .tables import validate_table

_PDF_RULES = get_rules("pdf")
_LAYOUT_CFG = get_config_section("extraction")["pdf"]["layout"]

IGNORED_VISUAL_TYPES = set(_PDF_RULES["ignored_visual_types"])
_ALLOWED_ELEMENT_TYPES = set(_PDF_RULES["allowed_element_types"])
_INLINE_MARKER_SYMBOLS = set(_PDF_RULES["inline_marker_symbols"])


def compact(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def normalized_box(box):
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1 for value in box)
        and box[0] < box[2]
        and box[1] < box[3]
    )


def inside_box(inner, outer):
    tolerance = _LAYOUT_CFG["inside_tolerance"]
    return (outer[0] - tolerance <= inner[0] and outer[1] - tolerance <= inner[1]
            and inner[2] <= outer[2] + tolerance and inner[3] <= outer[3] + tolerance)


def region_signature(element):
    if element["type"] == "table":
        table = element["table_data"]
        cells = sorted(
            (cell["row"], cell["column"], cell["row_span"],
             cell.get("column_span", cell.get("col_span", 1)), compact(cell["text"]))
            for cell in table["cells"]
        )
        return ("table", len(table["header_grid"]) + len(table["rows"]), len(table["headers"]),
                table["header_row_count"], cells)
    value = compact(element.get("text", ""))
    if value:
        return ("text", value)
    return None


def region_overlap(first, second):
    if first.get("bbox_granularity") == "page" or second.get("bbox_granularity") == "page":
        return 0.0, 0.0
    left, top, right, bottom = first["bbox"]
    other_left, other_top, other_right, other_bottom = second["bbox"]
    overlap = (max(0, min(right, other_right) - max(left, other_left))
               * max(0, min(bottom, other_bottom) - max(top, other_top)))
    areas = ((right - left) * (bottom - top), (other_right - other_left) * (other_bottom - other_top))
    return overlap / min(areas), overlap / max(areas)


def merge_region_observation(elements, incoming):
    value = region_signature(incoming)
    if value is None:
        return
    containment_min = _LAYOUT_CFG["containment_threshold"]
    similarity_conflict = _LAYOUT_CFG["similarity_conflict_threshold"]
    for index, previous in enumerate(elements):
        containment, similarity = region_overlap(incoming, previous)
        if not containment:
            continue
        other = region_signature(previous)
        if (other is not None and other[0] == value[0] == "text"
                and (other != value or containment < containment_min)
                and _crossing_text_boxes(previous, incoming)):
            raise ValueError("Overlapping PDF regions contain partial readings of the same text")
        if containment < containment_min:
            continue
        if other == value:
            return
        if other is not None and other[0] == value[0] == "text":
            if value[1] in other[1] and inside_box(incoming["bbox"], previous["bbox"]):
                return
            if other[1] in value[1] and inside_box(previous["bbox"], incoming["bbox"]):
                elements[index] = incoming
                return
        if similarity >= similarity_conflict:
            raise ValueError("Overlapping PDF regions disagree on the same text")
    elements.append(incoming)


def _crossing_text_boxes(first, second):
    if (not first.get("region_pixel_path") or not second.get("region_pixel_path")
            or first["region_pixel_path"] == second["region_pixel_path"]):
        return False
    first_box, second_box = first["bbox"], second["bbox"]
    overlap_ratio = _LAYOUT_CFG["crossing_overlap_ratio"]
    cross_ratio = _LAYOUT_CFG["crossing_cross_ratio"]
    for axis, cross_axis in ((0, 1), (1, 0)):
        low, high = sorted((first_box, second_box), key=lambda box: box[axis])
        if not low[axis] < high[axis] < low[axis + 2] < high[axis + 2]:
            continue
        overlap = low[axis + 2] - high[axis]
        size = min(low[axis + 2] - low[axis], high[axis + 2] - high[axis])
        cross_overlap = min(low[cross_axis + 2], high[cross_axis + 2]) - max(low[cross_axis], high[cross_axis])
        cross_size = min(low[cross_axis + 2] - low[cross_axis], high[cross_axis + 2] - high[cross_axis])
        # Ignore small bbox rounding drift. Substantial crossing from two
        # different crops needs a reading that spans the disputed area.
        if overlap >= overlap_ratio * size and cross_overlap >= cross_ratio * cross_size:
            return True
    return False


def reading_order(elements):
    """Use a two-dimensional whitespace cut before falling back to row order."""
    if any(item.get("bbox_granularity") == "page" for item in elements):
        return list(elements)

    col_min_gap = _LAYOUT_CFG["column_min_gap"]
    col_min_overlap = _LAYOUT_CFG["column_min_vertical_overlap"]
    col_min_width = _LAYOUT_CFG["column_min_width"]
    col_min_lines = _LAYOUT_CFG["column_min_lines"]
    band_min_gap = _LAYOUT_CFG["band_min_gap"]
    band_broad_gap = _LAYOUT_CFG["band_broad_gap"]

    def whitespace_cuts(items, lower, upper, minimum):
        ordered = sorted(items, key=lambda item: item["bbox"][lower])
        edge = ordered[0]["bbox"][upper]
        cuts = []
        for index, item in enumerate(ordered[1:], start=1):
            gap = item["bbox"][lower] - edge
            if gap > minimum:
                cuts.append((gap, index))
            edge = max(edge, item["bbox"][upper])
        return ordered, cuts

    def widest_cut(ordered, cuts):
        return max(cuts, key=lambda cut: (cut[0], -abs(cut[1] * 2 - len(ordered))))[1]

    def column_split(items):
        if len(items) < 2:
            return None
        ordered, cuts = whitespace_cuts(items, 0, 2, col_min_gap)
        if not cuts:
            return None
        cut = widest_cut(ordered, cuts)
        split = ordered[:cut], ordered[cut:]
        top = max(min(item["bbox"][1] for item in group) for group in split)
        bottom = min(max(item["bbox"][3] for item in group) for group in split)
        if bottom - top <= col_min_overlap:
            return None
        # Both sides need vertically distributed text. A label or a list
        # marker next to a paragraph belongs to the same reading band.
        for group in split:
            if sum(len(item["text"].splitlines()) for item in group
                   if item["type"] != "table" and item["bbox"][2] - item["bbox"][0] >= col_min_width) < col_min_lines:
                return None
        return split

    def band_split(items):
        ordered, cuts = whitespace_cuts(items, 1, 3, band_min_gap)
        if cuts:
            # A title or footer that overlaps the column gutter can sit
            # closer to the columns than the whitespace between body rows.
            for (_, cut), at_top in ((cuts[0], True), (cuts[-1], False)):
                first, second = ordered[:cut], ordered[cut:]
                if at_top:
                    column_items, outer_band = second, first
                else:
                    column_items, outer_band = first, second
                columns = column_split(column_items)
                if columns:
                    left, right = columns
                    center = (max(item["bbox"][2] for item in left) + min(item["bbox"][0] for item in right)) / 2
                    if any(item["bbox"][0] < center < item["bbox"][2] for item in outer_band):
                        return first, second
        broad_cuts = [cut for cut in cuts if cut[0] > band_broad_gap]
        if broad_cuts:
            cut = widest_cut(ordered, broad_cuts)
            return ordered[:cut], ordered[cut:]
        return None

    result = []
    pending = [elements]
    while pending:
        items = pending.pop()
        if len(items) < 2:
            result.extend(items)
            continue
        # An established gutter takes priority over whitespace between rows
        # in the same columns. Spanning headings still separate page bands.
        split = column_split(items)
        if split is None:
            split = band_split(items)
        if split is not None:
            pending.extend(reversed(split))
        else:
            result.extend(sorted(items, key=lambda item: (round(item["bbox"][1], 3), item["bbox"][0])))
    return result


def layout_text(layout):
    values = []
    for element in layout["elements"]:
        if element["type"] == "table":
            table = element["table_data"]
            grid = table.get("header_grid", []) + table["rows"]
            span_fill = {
                (row, column)
                for cell in table.get("cells", [])
                for row in range(cell["row"], cell["row"] + cell["row_span"])
                for column in range(cell["column"], cell["column"] + cell.get("column_span", cell.get("col_span", 1)))
                if (row, column) != (cell["row"], cell["column"])
            }
            values.extend(value for row, cells in enumerate(grid) for column, value in enumerate(cells)
                          if (row, column) not in span_fill)
        else:
            values.append(element.get("text", ""))
    return "\n".join(values)


def has_meaningful_layout(layout):
    return bool(compact(layout_text(layout)))


def validate_scan_pixels(layout, pixels):
    if has_meaningful_layout(layout) or layout["review"].get("non_text_only") is True:
        return
    from PIL import Image, ImageChops
    with Image.open(io.BytesIO(pixels)) as original:
        rgb = original.convert("RGB")
        if ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox():
            raise ValueError("Nonblank page cannot have an empty visual extraction")


def ordered_witnesses_present(text, witnesses):
    """Check that each complete witness occurs in order, with repetition."""
    combined = " " + compact(text) + " "
    cursor = 0
    for value in witnesses:
        part = compact(value)
        if not part:
            continue
        position = combined.find(" " + part + " ", cursor)
        if position < 0:
            return False
        cursor = position + len(part) + 1
    return True


def validate_page(layout, native_lines, *, allow_ocr_additions=False):
    if not isinstance(layout, dict) or not isinstance(layout.get("elements"), list):
        raise ValueError("PDF layout requires an elements array")
    if layout.get("uncertainties") != [] or not isinstance(layout.get("review"), dict) or layout["review"].get("verdict") != "accepted":
        raise ValueError("Unresolved visual extraction; publication refused")
    max_level = _LAYOUT_CFG["max_heading_level"]
    bbox_preview = _LAYOUT_CFG["bbox_error_preview_chars"]
    for element in layout["elements"]:
        if (not isinstance(element, dict) or not isinstance(element.get("type"), str)
                or element["type"] not in _ALLOWED_ELEMENT_TYPES or not isinstance(element.get("text", ""), str)):
            raise ValueError("Invalid PDF element")
        if element["type"] != "table" and (element.get("table_data") is not None or element.get("table_html") is not None):
            raise ValueError("table_data requires type table; preserve the complete table as its own element")
        box = element.get("bbox")
        if not normalized_box(box):
            raise ValueError("PDF element bbox must be a nonempty [x0,y0,x1,y1] in [0,1]; received " + repr(box)[:bbox_preview])
        if element["type"] == "table":
            validate_table(element.get("table_data"), require_complete=True)
        if element["type"] == "heading":
            level = element.get("level")
            if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= max_level:
                raise ValueError("PDF heading must have a level 1.." + str(max_level))
        if element["type"] != "table" and not element.get("text", "").strip():
            raise ValueError("PDF text elements require visible text")
    if layout["review"].get("non_text_only") is True and has_meaningful_layout(layout):
        raise ValueError("PDF visual review marks visible text as non-text")
    if native_lines:
        text = layout_text(layout)
        if allow_ocr_additions:
            matches = ordered_witnesses_present(text, native_lines)
        else:
            matches = compact(text) == compact("\n".join(native_lines))
        if not matches:
            raise ValueError("Visual extraction disagrees with the ordered native text")


def is_inline_list_marker(element, page_elements):
    text = element.get("text", "")
    if not isinstance(text, str):
        return False
    text = compact(text)
    marker = text.rstrip(".)、）:：")
    max_len = _LAYOUT_CFG["inline_marker_max_len"]
    looks_like_marker = (text in _INLINE_MARKER_SYMBOLS
                         or 0 < len(marker) <= max_len and marker.isalnum()
                         or all(part.strip().isdecimal() for part in text.split("/")))
    if not looks_like_marker:
        return False
    left, top, right, bottom = element["bbox"]
    max_gap = _LAYOUT_CFG["inline_marker_max_gap"]
    wrap_top_max = _LAYOUT_CFG["inline_marker_wrap_top_max"]
    wrap_gap_max = _LAYOUT_CFG["inline_marker_wrap_gap_max"]
    for peer in page_elements:
        if peer is element:
            continue
        peer_text = peer.get("text", "")
        if not isinstance(peer_text, str) or not any(char.isalnum() for char in peer_text):
            continue
        peer_left, peer_top, peer_right, peer_bottom = peer["bbox"]
        first_direction = next((unicodedata.bidirectional(char) for char in peer_text if char.isalpha()), None)
        if first_direction in {"R", "AL"}:
            gap = left - peer_right
        else:
            gap = peer_left - right
        overlap = min(bottom, peer_bottom) - max(top, peer_top)
        wraps_at_top = first_direction is not None and top < wrap_top_max and 0 <= peer_top - bottom <= wrap_gap_max
        if 0 <= gap <= max_gap and (overlap >= (bottom - top) / 2 or wraps_at_top):
            return True
    return False


def normalize_visual_layout(layout):
    if not isinstance(layout, dict) or not isinstance(layout.get("elements"), list):
        raise ValueError("PDF layout requires an elements array")
    for item in layout["elements"]:
        if (not isinstance(item, dict) or not isinstance(item.get("type"), str)
                or item.get("role") is not None and not isinstance(item["role"], str)):
            raise ValueError("Invalid PDF element")
    place_reported_headings(layout)
    body = [item for item in layout["elements"]
            if item["type"] not in IGNORED_VISUAL_TYPES and item.get("role") not in IGNORED_VISUAL_TYPES
            and normalized_box(item.get("bbox"))]
    max_level = _LAYOUT_CFG["max_heading_level"]
    kept = []
    for element in layout["elements"]:
        if element.get("type") in IGNORED_VISUAL_TYPES or element.get("role") in IGNORED_VISUAL_TYPES:
            is_page_number = element.get("type") == "page_number" or element.get("role") == "page_number"
            if (not is_page_number or not normalized_box(element.get("bbox"))
                    or not is_inline_list_marker(element, body)):
                continue
            element["type"] = "paragraph"
            element.pop("role", None)
        if element.get("type") == "heading":
            level = element.get("level", 1)
            if isinstance(level, int) and not isinstance(level, bool):
                element["level"] = min(level, max_level)
        box = element.get("bbox")
        if not normalized_box(box):
            element["bbox"] = [0, 0, 1, 1]
            element["bbox_granularity"] = "page"
        kept.append(element)
    layout["elements"] = kept


def place_reported_headings(layout, *, allow_unlocated=False):
    """Reconcile a legacy heading array without inventing source positions."""
    headings = layout.get("headings", [])
    if not isinstance(headings, list):
        raise ValueError("PDF headings must be an array")
    visible = []
    for item in layout["elements"]:
        role = item.get("role")
        if role is not None and not isinstance(role, str):
            raise ValueError("Invalid PDF element role")
        if item["type"] not in IGNORED_VISUAL_TYPES and role not in IGNORED_VISUAL_TYPES:
            visible.append(item)
    represented = []
    for item in visible:
        source = item.get("text", "")
        represented.append((item, {compact(source), *(compact(line) for line in source.splitlines())}))
    additions = []
    unlocated = []
    for heading in headings:
        if not isinstance(heading, dict) or not isinstance(heading.get("text"), str) or not heading["text"].strip():
            raise ValueError("Invalid reported PDF heading")
        text = compact(heading["text"])
        if any(text in readings and _same_heading_location(heading, item) for item, readings in represented):
            continue
        if (not normalized_box(heading.get("bbox"))
                or any(not normalized_box(item.get("bbox")) for item in layout["elements"])):
            if not allow_unlocated:
                raise ValueError("Put every reported heading inside elements at its actual reading position")
            item = {**heading, "type": "heading", "order_evidence": "cloud_heading_order_unknown"}
            if not normalized_box(item.get("bbox")):
                item.update(bbox=[0, 0, 1, 1], bbox_granularity="page")
            unlocated.append(item)
            represented.append((item, {text}))
            continue
        additions.append({**heading, "type": "heading"})
        represented.append((additions[-1], {text}))
    if additions:
        layout["elements"] = reading_order(layout["elements"] + additions)
    layout["elements"].extend(unlocated)


def _same_heading_location(heading, element):
    if (not normalized_box(heading.get("bbox")) or not normalized_box(element.get("bbox"))
            or element.get("bbox_granularity") == "page"):
        return True
    return region_overlap(heading, element)[0] >= _LAYOUT_CFG["containment_threshold"]
