"""Decode model-declared HTML cells into the lossless table IR.

This module interprets serialization coordinates and spans only. The vision
model decides table membership, header count and reading order from pixels.
"""
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

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.in_table or self.finished:
                raise ValueError("Expected exactly one nonnested serialized table")
            self.in_table = True
        elif tag in ("thead", "tbody", "tfoot"):
            if not self.in_table or self.in_row:
                raise ValueError("Invalid table section")
        elif tag == "tr":
            if not self.in_table or self.in_row:
                raise ValueError("Invalid table row")
            self.row += 1
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
            if not raw_rowspan.isdigit() or not raw_colspan.isdigit():
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
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += "\n"
        elif tag in ("p", "div", "li") and self.cell is not None:
            if self.cell["text"] and not self.cell["text"].endswith("\n"):
                self.cell["text"] += "\n"
        elif tag in ("b", "i", "em", "strong", "span", "u", "sub", "sup", "code", "font", "small", "mark", "ul", "ol") and self.cell is not None:
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
            if self.cell is None:
                raise ValueError("Unexpected serialized cell ending")
            self.cell["text"] = self.cell["text"].strip()
            self.column += self.cell["column_span"]
            self.cells.append(self.cell)
            self.cell = None
        elif tag == "tr":
            if not self.in_row or self.cell is not None:
                raise ValueError("Unclosed serialized cell")
            self.in_row = False
        elif tag == "table":
            if not self.in_table or self.in_row or self.cell is not None:
                raise ValueError("Unclosed serialized row")
            self.in_table = False
            self.finished = True
        elif tag not in (
            "thead", "tbody", "tfoot", "b", "i", "em", "strong", "span",
            "u", "sub", "sup", "code", "br", "p", "div", "li", "font",
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
    if parser.occupied != {(row, column) for row in range(height) for column in range(width)}:
        raise ValueError("Serialized table has uncovered cells; explicit empty cells are required")
    grid = [[""] * width for _ in range(height)]
    for cell in parser.cells:
        grid[cell["row"]][cell["column"]] = cell["text"]
        if cell["row"] >= header_rows and (cell["row_span"] > 1 or cell["column_span"] > 1):
            for r in range(cell["row"], cell["row"] + cell["row_span"]):
                for c in range(cell["column"], cell["column"] + cell["column_span"]):
                    if not grid[r][c]:
                        grid[r][c] = cell["text"]
    headers = []
    for column in range(width):
        labels = [cell["text"] for cell in parser.cells if cell["row"] < header_rows
                  and cell["column"] <= column < cell["column"] + cell["column_span"] and cell["text"]]
        headers.append(" / ".join(labels))
    return {"headers": headers, "header_row_count": header_rows, "header_grid": grid[:header_rows],
            "rows": grid[header_rows:], "cells": parser.cells, "serialization": "model_declared_html_cells_v1"}
