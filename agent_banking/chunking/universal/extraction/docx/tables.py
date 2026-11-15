"""Read the physical DOCX grid, authored headers and cell spans."""
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from .typography import visible_text_runs
from .openxml import (
    EMBEDDED_CONTENT,
    MC,
    NONCONTENT,
    content_children,
    local_name,
    native_text,
    table_items,
)


def read_table(node, document, script_styles, cell_prefix, parent_cell):
    rows = table_items(node, qn("w:tr"))
    grid = []
    cells = []
    row_grid_offsets = []
    previous = {}
    nested = []
    textbox_paragraphs = []
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
            paragraphs = []
            child_tables = []
            for child in content_children(cell):
                _cell_content(child, paragraphs, child_tables, textbox_paragraphs)
            nested.extend((child, cell_id) for child in child_tables)
            cell_units = [(native_text(paragraph, script_styles), paragraph) for paragraph in paragraphs]
            text = "\n".join(value for value, _ in cell_units).strip()
            record = {"id": cell_id, "row": row_index, "column": column,
                      "row_span": 1, "col_span": colspan, "text": text,
                      "xpath": cell.getroottree().getpath(cell)}
            runs = [run for paragraph in paragraphs if paragraph.tag == qn("w:p")
                    for run in visible_text_runs(Paragraph(paragraph, document))]
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
    table = {"headers": headers, "rows": grid[header_count:], "grid": grid,
             "header_row_count": header_count, "header_detection": header_detection,
             "cells": cells, "parent_cell": parent_cell, "row_grid_offsets": row_grid_offsets}
    return table, units, nested, textbox_paragraphs


def _cell_content(node, paragraphs, nested, textbox_paragraphs):
    name = local_name(node)
    if name in NONCONTENT or name in EMBEDDED_CONTENT or node.tag == MC + "AlternateContent":
        return
    if name == "tbl":
        nested.append(node)
        return
    if name == "p":
        paragraphs.append(node)
        textbox_paragraphs.append(node)
        return
    if name in ("oMath", "oMathPara"):
        paragraphs.append(node)
        return
    for child in content_children(node):
        _cell_content(child, paragraphs, nested, textbox_paragraphs)
