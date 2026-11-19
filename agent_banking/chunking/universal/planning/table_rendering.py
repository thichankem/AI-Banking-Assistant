"""Markdown and geometry-preserving HTML for source tables."""
from bisect import bisect_left
from html import unescape

from ..text.boundaries import protected_markup_spans


def markdown_row(row):
    rendered = []
    for value in row:
        value = str(value)
        if "\\" not in value:
            rendered.append(value.replace("|", "\\|").replace("\n", "<br>"))
            continue
        start = 0
        pieces = []
        for left, right in protected_markup_spans(value):
            part = value[left:right]
            if part.startswith(("$", "\\(", "\\[")):
                pieces.append(value[start:left].replace("\\", "\\\\"))
                pieces.append(part)
                start = right
        pieces.append(value[start:].replace("\\", "\\\\"))
        rendered.append("".join(pieces).replace("|", "\\|").replace("\n", "<br>"))
    return "| " + " | ".join(rendered) + " |"


def table_markdown(headers, rows):
    return "\n".join([markdown_row(headers), markdown_row(["---"] * len(headers))]
                     + [markdown_row(row) for row in rows])


def table_fragment_data(block, rows, row_start):
    """Select row geometry and source metadata for one table fragment."""
    table = block["table_data"]
    headers = table["headers"]
    title = table.get("title", "") or block.get("caption", "")
    header_count = table["header_row_count"]
    header_grid = table["grid"][:header_count]
    row_end = row_start + len(rows)
    header_cells = [cell for cell in table["cells"] if cell["row"] < header_count]
    cells = [cell for cell in table["cells"] if header_count <= cell["row"] < row_end
             and cell["row"] + cell.get("row_span", 1) > row_start]
    data = {"table_id": table["table_id"], "title": title, "headers": headers, "rows": rows,
            "row_start": row_start, "row_end": row_end, "header_row_count": header_count,
            "header_grid": header_grid, "header_cells": header_cells, "cells": cells,
            "html": render_structured_html_table(headers, rows, title, cells=[*header_cells, *cells],
                                                 header_grid=header_grid, row_start=row_start)}
    row_provenance = table.get("row_provenance", [])
    if table["rows"]:
        source_rows = row_provenance[row_start:row_end]
    else:
        source_rows = row_provenance[:header_count]
    if source_rows:
        data["row_provenance"] = source_rows
    for key in ("parent_cell", "title_source_block_id", "header_occurrences"):
        if table.get(key):
            data[key] = table[key]
    if table.get("row_grid_offsets"):
        data["row_grid_offsets"] = table["row_grid_offsets"][row_start:row_end]
        data["header_grid_offsets"] = table["row_grid_offsets"][:header_count]
    return data


def escape_html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_cell_text(value):
    escaped = []
    cursor = 0
    while cursor < len(value):
        index = value.find("<", cursor)
        if index == -1:
            escaped.append(escape_html(value[cursor:]))
            break
        escaped.append(escape_html(value[cursor:index]))
        recognized = False
        for opener, closer in (("<sup>", "</sup>"), ("<sub>", "</sub>"), ('<code data-format="omml">', "</code>")):
            if not value.startswith(opener, index):
                continue
            end = value.find(closer, index + len(opener))
            if end != -1:
                escaped.append(opener + escape_html(unescape(value[index + len(opener):end])) + closer)
                cursor = end + len(closer)
                recognized = True
                break
        if not recognized:
            escaped.append("&lt;")
            cursor = index + 1
    return "".join(escaped).replace("\n", "<br>")


def render_structured_html_table(headers, rows, title="", *, cells=(), header_grid=None, row_start=None):
    """Render source geometry for headers and the selected consecutive rows."""
    parts = ["<table>"]
    if title:
        parts.append("<caption>" + escape_html(title) + "</caption>")
    if header_grid is None:
        if headers:
            header_grid = [headers]
        else:
            header_grid = []
    if row_start is None:
        row_start = len(header_grid)
    selected = list(enumerate(header_grid)) + list(enumerate(rows, row_start))
    selected_indices = [index for index, _ in selected]
    anchors = {}
    hidden = set()
    for cell in cells:
        if cell.get("merged_into"):
            continue
        row, column = cell["row"], cell["column"]
        first = bisect_left(selected_indices, row)
        end = bisect_left(selected_indices, row + cell.get("row_span", 1))
        if first == end:
            continue
        width = cell.get("col_span", cell.get("column_span", 1))
        anchors[first, column] = (end - first, width, cell["text"])
        hidden.update((index, col) for index in range(first, end) for col in range(column, column + width)
                      if (index, col) != (first, column))
    # A single HTML row group also supports a native span crossing the
    # heading/body boundary, which separate thead/tbody groups cannot render.
    parts.append("<tbody>")
    for index, (_, row) in enumerate(selected):
        if index < len(header_grid):
            tag = "th"
        else:
            tag = "td"
        parts.append("<tr>")
        for column, value in enumerate(row):
            if (index, column) in hidden:
                continue
            height, width, value = anchors.get((index, column), (1, 1, value))
            attributes = ""
            if height > 1:
                attributes += f' rowspan="{height}"'
            if width > 1:
                attributes += f' colspan="{width}"'
            parts.append(f"<{tag}{attributes}>" + _render_cell_text(value) + f"</{tag}>")
        parts.append("</tr>")
    parts.append("</tbody>")
    parts.append("</table>")
    return "".join(parts)
