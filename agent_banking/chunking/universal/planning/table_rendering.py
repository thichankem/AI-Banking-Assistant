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
        header_grid = [headers] if headers else []
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
        tag = "th" if index < len(header_grid) else "td"
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
