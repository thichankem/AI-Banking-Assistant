"""Read the physical DOCX grid, authored headers and cell spans."""
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from .inline_content import paragraph_parts
from .typography import visible_text_runs
from .openxml import (
    EMBEDDED_CONTENT,
    MC,
    content_children,
    is_noncontent,
    local_name,
    native_math,
    table_items,
)


def read_table(node, document, part, script_styles, hidden_run, cell_prefix, parent_cell):
    rows = table_items(node, qn("w:tr"))
    grid = []
    cells = []
    row_grid_offsets = []
    previous = {}
    attachments = []
    units = []
    header_count = 0
    declared_width = len(node.findall("./" + qn("w:tblGrid") + "/" + qn("w:gridCol")))
    for row_index, row in enumerate(rows):
        props = row.find(qn("w:trPr"))
        before = after = 0
        is_header = False
        if props is not None:
            entry_before = props.find(qn("w:gridBefore"))
            entry_after = props.find(qn("w:gridAfter"))
            if entry_before is not None:
                before = int(entry_before.get(qn("w:val"), "0"))
            if entry_after is not None:
                after = int(entry_after.get(qn("w:val"), "0"))
            if before < 0 or after < 0:
                raise ValueError("Table grid offsets must be nonnegative")
            header = props.find(qn("w:tblHeader"))
            is_header = header is not None and header.get(qn("w:val"), "true").lower() not in ("0", "false", "off")
        if is_header and header_count == row_index:
            header_count += 1
        row_grid_offsets.append({"before": before, "after": after})
        values = [""] * before
        column = before
        current = {}
        for cell in table_items(row, qn("w:tc")):
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
            cell_id = cell_prefix + str(row_index) + ":" + str(column)
            text, cell_units, hyperlinks, children = _read_cell(cell, part, script_styles, hidden_run)
            record = {"id": cell_id, "row": row_index, "column": column,
                      "row_span": 1, "col_span": colspan, "text": text,
                      "xpath": cell.getroottree().getpath(cell)}
            if hyperlinks:
                record["hyperlinks"] = hyperlinks
            attachments.extend((child, record, position) for child, position in children)
            runs = [run for _, paragraph in cell_units if paragraph.tag == qn("w:p")
                    for run in visible_text_runs(Paragraph(paragraph, document), hidden_run)]
            record["all_text_bold"] = bool(runs) and all(run.bold for run in runs)
            if merge is not None:
                merge_kind = merge.get(qn("w:val"), "continue")
                origin = previous.get(column)
                if (merge_kind == "continue" and origin is not None
                        and origin["column"] == column and origin["col_span"] == colspan
                        and all(previous.get(col) is origin for col in range(column, column + colspan))
                        and (not text or text == origin["text"])):
                    origin["row_span"] += 1
                    record["merged_into"] = origin["id"]
                    text = origin["text"]
                else:
                    origin = record
                    if merge_kind == "continue":
                        record["merge_status"] = "inconsistent_continuation"
                for col in range(column, column + colspan):
                    current[col] = origin
            cells.append(record)
            values.extend([text] * colspan)
            column += colspan
            units.extend(cell_units)
        grid.append(values + [""] * after)
        previous = current
    width = max([len(row) for row in grid] + [declared_width])
    for row in grid:
        row.extend([""] * (width - len(row)))
    header_detection = "unmarked"
    if header_count:
        header_detection = "explicit_repeat_header"
        headers = []
        for column in range(width):
            labels = []
            for row in grid[:header_count]:
                if row[column] and row[column] not in labels:
                    labels.append(row[column])
            headers.append(" / ".join(labels))
    else:
        headers = ["#" + str(index + 1) for index in range(width)]
    table = {"headers": headers, "rows": grid[header_count:], "grid": grid, "header_grid": grid[:header_count],
             "header_row_count": header_count, "header_detection": header_detection,
             "cells": cells, "parent_cell": parent_cell, "row_grid_offsets": row_grid_offsets}
    return table, units, attachments


def _cell_content(node):
    for child in content_children(node):
        name = local_name(child)
        if is_noncontent(child) or name in EMBEDDED_CONTENT or child.tag == MC + "AlternateContent":
            continue
        if name in ("tbl", "p", "oMath", "oMathPara"):
            yield name, child
        else:
            yield from _cell_content(child)


def _read_cell(cell, part, script_styles, hidden_run):
    units = []
    hyperlinks = []
    attachments = []
    offset = 0
    for kind, node in _cell_content(cell):
        if kind == "tbl":
            attachments.append((node, {"type": "table", "text_offset": offset}))
            continue
        if units:
            offset += 1  # The newline between authored paragraphs or equations.
        fragments = []
        if kind == "p":
            parts = paragraph_parts(node, part, script_styles, hidden_run)
        else:
            parts = (("text", native_math(node), []),)
        for part_kind, value, links in parts:
            if part_kind == "textbox":
                attachments.append((value, {"type": "textbox", "text_offset": offset}))
            else:
                fragments.append(value)
                offset += len(value)
                hyperlinks.extend(links)
        units.append(("".join(fragments), node))
    raw = "\n".join(value for value, _ in units)
    text = raw.strip()
    leading = len(raw) - len(raw.lstrip())
    for _, position in attachments:
        position["text_offset"] = max(0, min(len(text), position["text_offset"] - leading))
    return text, units, hyperlinks, attachments
