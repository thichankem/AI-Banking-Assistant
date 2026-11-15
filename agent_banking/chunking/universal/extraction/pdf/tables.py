"""Parse and validate lossless cell grids for PDF tables."""
from html.parser import HTMLParser


class _Cells(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cells = []
        self.occupied = set()
        self.row = -1
        self.column = 0
        self.in_table = False
        self.in_row = False
        self.finished = False
        self.cell = None
        self.cell_tag = None
        self.section = None
        self.formats = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.in_table or self.finished:
                raise ValueError("Expected exactly one nonnested serialized table")
            self.in_table = True
        elif tag in ("thead", "tbody", "tfoot"):
            if not self.in_table or self.in_row or self.section is not None:
                raise ValueError("Invalid table section")
            self.section = tag
        elif tag == "tr":
            if not self.in_table or self.in_row:
                raise ValueError("Invalid table row")
            self.row += 1
            if self.row >= 4096:
                raise ValueError("Serialized table exceeds row limit")
            self.column = 0
            self.in_row = True
        elif tag in ("td", "th"):
            if not self.in_row or self.cell is not None:
                raise ValueError("Invalid serialized cell")
            while (self.row, self.column) in self.occupied:
                self.column += 1
            values = dict(attrs)
            raw_rowspan = str(values.get("rowspan") or "1").strip()
            raw_colspan = str(values.get("colspan") or "1").strip()
            if not raw_rowspan.isdecimal() or not raw_colspan.isdecimal():
                raise ValueError("Invalid non-integer span attribute")
            height, width = int(raw_rowspan), int(raw_colspan)
            if not 1 <= height <= 4096 or not 1 <= width <= 256 or self.row + height > 4096 or self.column + width > 256:
                raise ValueError("Serialized table exceeds span limits")
            locations = {(r, c) for r in range(self.row, self.row + height)
                         for c in range(self.column, self.column + width)}
            if locations & self.occupied:
                raise ValueError("Overlapping serialized cells")
            self.occupied.update(locations)
            self.cell = {"row": self.row, "column": self.column, "row_span": height,
                         "column_span": width, "text": ""}
            self.cell_tag = tag
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += "\n"
        elif tag in ("p", "div", "li") and self.cell is not None:
            if self.cell["text"] and not self.cell["text"].endswith("\n"):
                self.cell["text"] += "\n"
        elif tag in ("sub", "sup") and self.cell is not None:
            self.formats.append(tag)
            self.cell["text"] += "<" + tag + ">"
        elif tag in ("b", "i", "em", "strong", "span", "u", "code", "font", "small", "mark", "ul", "ol") and self.cell is not None:
            pass
        elif tag in ("colgroup", "col") and self.in_table and not self.in_row:
            pass
        else:
            raise ValueError("Unsupported serialized table tag: " + tag)

    def handle_data(self, text):
        if self.cell is not None:
            self.cell["text"] += text
        elif text.strip():
            raise ValueError("Visible table text must be represented inside cells; captions need separate elements")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            if self.cell is None or self.cell_tag != tag or self.formats:
                raise ValueError("Unexpected serialized cell ending or unclosed formula format")
            self.cell["text"] = self.cell["text"].strip()
            self.column += self.cell["column_span"]
            self.cells.append(self.cell)
            self.cell = None
            self.cell_tag = None
        elif tag == "tr":
            if not self.in_row or self.cell is not None:
                raise ValueError("Unclosed serialized cell")
            self.in_row = False
        elif tag == "table":
            if not self.in_table or self.in_row or self.cell is not None or self.section is not None:
                raise ValueError("Unclosed serialized row or section")
            self.in_table = False
            self.finished = True
        elif tag in ("sub", "sup"):
            if self.cell is None or not self.formats or self.formats[-1] != tag:
                raise ValueError("Unbalanced formula format in serialized cell")
            self.formats.pop()
            self.cell["text"] += "</" + tag + ">"
        elif tag in ("thead", "tbody", "tfoot"):
            if self.section != tag or self.in_row:
                raise ValueError("Unexpected serialized table section ending")
            self.section = None
        elif tag not in (
            "b", "i", "em", "strong", "span",
            "u", "code", "br", "p", "div", "li", "font",
            "small", "mark", "ul", "ol", "colgroup", "col",
        ):
            raise ValueError("Unexpected serialized tag ending")


def table_from_html(markup, header_rows):
    if not isinstance(markup, str) or isinstance(header_rows, bool) or not isinstance(header_rows, int) or header_rows < 0:
        raise ValueError("Table HTML requires a string and nonnegative header_row_count")
    parser = _Cells()
    parser.feed(markup)
    parser.close()
    if parser.in_table or parser.in_row or parser.cell is not None or not parser.finished or not parser.cells:
        raise ValueError("Incomplete serialized table")
    height = parser.row + 1
    width = max(cell["column"] + cell["column_span"] for cell in parser.cells)
    if header_rows > height or any(cell["row"] + cell["row_span"] > height for cell in parser.cells):
        raise ValueError("Serialized span or header count extends outside table")
    # Every coordinate is inside the grid and the parser rejected overlaps.
    if len(parser.occupied) != height * width:
        raise ValueError("Serialized table has uncovered cells; explicit empty cells are required")
    grid = [[""] * width for _ in range(height)]
    for cell in parser.cells:
        grid[cell["row"]][cell["column"]] = cell["text"]
        if cell["row"] >= header_rows and (cell["row_span"] > 1 or cell["column_span"] > 1):
            for r in range(cell["row"], cell["row"] + cell["row_span"]):
                for c in range(cell["column"], cell["column"] + cell["column_span"]):
                    if not grid[r][c]:
                        grid[r][c] = cell["text"]
    headers = _header_labels(parser.cells, header_rows, width)
    return {"headers": headers, "header_row_count": header_rows, "header_grid": grid[:header_rows],
            "rows": grid[header_rows:], "cells": parser.cells, "serialization": "model_declared_html_cells_v1"}


def _header_labels(cells, count, width):
    labels = [[] for _ in range(width)]
    origins = sorted((cell for cell in cells if cell["row"] < count and cell["text"]),
                     key=lambda cell: (cell["row"], cell["column"]))
    for cell in origins:
        for column in range(cell["column"], cell["column"] + cell["column_span"]):
            labels[column].append(cell["text"])
    return [" / ".join(values) for values in labels]


def validate_table(table, *, require_complete=False):
    if not isinstance(table, dict):
        raise ValueError("PDF table must be structured")
    headers, rows = table.get("headers"), table.get("rows")
    if not isinstance(headers, list) or not headers or any(not isinstance(cell, str) for cell in headers):
        raise ValueError("PDF table requires string column headers (empty labels allowed)")
    if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != len(headers) or any(not isinstance(cell, str) for cell in row) for row in rows):
        raise ValueError("Ragged or non-text PDF table")
    count = table.get("header_row_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Table header_row_count must be a nonnegative integer")
    # Keep every original header row separately. Flattened labels alone cannot
    # reconstruct multirow merged headers and must not pass as lossless data.
    header_grid = table.get("header_grid")
    if header_grid is None:
        if count > 1:
            raise ValueError("Multirow PDF headers require header_grid")
        header_grid = [headers] if count else []
    if not isinstance(header_grid, list) or len(header_grid) != count or any(not isinstance(row, list) or len(row) != len(headers) or any(not isinstance(value, str) for value in row) for row in header_grid):
        raise ValueError("Invalid PDF header grid: expected " + str(count) + " rows of " + str(len(headers)) + " string cells; received " + repr(header_grid)[:300])
    grid = header_grid + rows
    occupied = set()
    cells = table.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError("PDF table cells must be an array")
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("text"), str):
            raise ValueError("Invalid PDF cell")
        numbers = [cell.get(key) for key in ("row", "column", "row_span", "column_span")]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in numbers):
            raise ValueError("Cell coordinates and spans must be integers")
        row, column, height, width = numbers
        if row < 0 or column < 0 or height < 1 or width < 1 or row + height > len(grid) or column + width > len(headers):
            raise ValueError("PDF cell span outside grid")
        locations = {(r, c) for r in range(row, row + height) for c in range(column, column + width)}
        if occupied & locations:
            raise ValueError("Overlapping PDF cell spans")
        occupied.update(locations)
        if grid[row][column] != cell["text"]:
            raise ValueError("PDF cell text disagrees with grid at row " + str(row) + ", column " + str(column)
                             + ": cell=" + repr(cell["text"]) + ", grid=" + repr(grid[row][column]))
        if any(grid[r][c] not in ("", cell["text"]) for r, c in locations):
            raise ValueError("PDF cell span covers conflicting grid text")
    if require_complete:
        if not grid or len(occupied) != len(grid) * len(headers):
            raise ValueError("Visual table requires complete cell coordinates and spans")
        # Flattened labels are presentation metadata. Derive them from the
        # complete source grid so a legacy response cannot invent a label.
        table["headers"] = _header_labels(cells, count, len(headers))
    table["header_grid"] = header_grid
