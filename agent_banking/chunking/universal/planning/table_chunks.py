"""Build table chunks and searchable views of the preserved rows."""
from copy import deepcopy

from ..text.language import project_search_text, source_foreign_spans, without_foreign_spans
from .source_spans import build_lead_in, has_visible_text
from .text_chunks import flush_text
from .table_rendering import markdown_row, render_structured_html_table, table_markdown
from ..text.token_budget import format_passage_fields


def plan_table(planner, block, parent_id, lead_in_blocks=None):
    table = block["table_data"]
    headers = table["headers"]
    rows = table["rows"]
    pending = []
    header_count = table["header_row_count"]
    header_grid = table["grid"][:header_count]
    header_cells = [cell for cell in table["cells"] if cell["row"] < header_count]
    row_provenance = table.get("row_provenance", [])
    foreign_cells = {}
    projected_cells = {}
    if planner.search_language == "vi":
        for row_index, values in enumerate(table["grid"]):
            for column, value in enumerate(values):
                spans = source_foreign_spans(value)
                if spans:
                    foreign_cells[row_index, column] = spans
                    projected_cells[row_index, column] = without_foreign_spans(value, spans)

    def projected(values, row_index):
        return [projected_cells.get((row_index, column), value) for column, value in enumerate(values)]

    projected_grid = [projected(row, index) for index, row in enumerate(header_grid)]
    projected_headers = headers
    if any(row_index < header_count for row_index, _ in foreign_cells):
        projected_headers = []
        for column in range(len(headers)):
            labels = dict.fromkeys(row[column].strip() for row in projected_grid if row[column].strip())
            projected_headers.append(" / ".join(labels))
    searchable_header = any(has_visible_text(value) for row in projected_grid for value in row)
    searchable_body = any(has_visible_text(value) for index, row in enumerate(rows, start=header_count)
                          for value in projected(row, index))
    start_row = header_count
    ids = [block["id"]]
    table_title = table.get("title", "") or block.get("caption", "")
    retrieval = planner._retrieval(parent_id, block.get("context_quotes", []), "table")
    if table_title and not retrieval.get("situated_context"):
        title_projection = project_search_text(table_title, search_language=planner.search_language).strip()
        if title_projection:
            retrieval["situated_context"] = planner.budget.shorten(title_projection, min(48, planner.budget.maximum // 6))
    lead_text, lead_spans, lead_ids = build_lead_in(lead_in_blocks)
    lead_attached = not bool(lead_text)
    if not lead_text and lead_in_blocks is not None:
        lead_in_blocks.clear()

    def fits(content, target=False):
        return planner._measure(content, retrieval, target) is not None

    def with_lead(md_text):
        if not lead_attached and lead_text:
            return lead_text + "\n\n" + md_text
        return md_text

    def attach_lead_metadata(chunk):
        nonlocal lead_attached
        if not lead_attached and lead_text:
            prov = chunk["business_metadata"]["provenance"]
            prov["source_spans"] = deepcopy(lead_spans)
            lead_attached = True
            if lead_in_blocks is not None:
                lead_in_blocks.clear()
        return chunk

    def table_chunk(content, data, block_ids):
        source_rows = row_provenance[data["row_start"]:data["row_end"]]
        if not rows:
            source_rows = row_provenance[:header_count]
        header_sources = row_provenance[:header_count]
        for occurrence in table.get("header_occurrences", []):
            header_sources = [*header_sources, *occurrence["row_provenance"]]
        if source_rows:
            data["row_provenance"] = source_rows
        if table.get("parent_cell"):
            data["parent_cell"] = table["parent_cell"]
        if table.get("title_source_block_id"):
            data["title_source_block_id"] = table["title_source_block_id"]
        if table.get("row_grid_offsets"):
            data["row_grid_offsets"] = table["row_grid_offsets"][data["row_start"]:data["row_end"]]
            data["header_grid_offsets"] = table["row_grid_offsets"][:header_count]
        if table.get("header_occurrences"):
            data["header_occurrences"] = table["header_occurrences"]
        chunk = planner._prototype("table", content, parent_id, block_ids, table_data=data, retrieval=retrieval)
        if source_rows or header_sources:
            locations = []
            for row in [*header_sources, *source_rows]:
                location = {key: value for key, value in row.items() if key not in ("source_row", "source_block_id")}
                if location not in locations:
                    locations.append(location)
            chunk["business_metadata"]["provenance"]["locations"][-1:] = locations
        return chunk

    visible_headers = {value for value in headers if value}
    needs_header_geometry = any(value and not any(value in label for label in visible_headers)
                                for row in header_grid for value in row)
    header_has_formula = any("\n$$" in value or "$$\n" in value for row in header_grid for value in row)

    def make(selected, row_start):
        selected_cells = [cell for cell in table["cells"] if cell["row"] < row_start + len(selected)
                          and cell["row"] + cell.get("row_span", 1) > row_start
                          and cell["row"] >= header_count]
        data = {"table_id": table["table_id"], "title": table_title, "headers": headers, "rows": selected,
                "row_start": row_start, "row_end": row_start + len(selected),
                "header_row_count": header_count, "header_grid": header_grid, "header_cells": header_cells,
                "html": render_structured_html_table(headers, selected, table_title, cells=[*header_cells, *selected_cells],
                                                     header_grid=header_grid, row_start=row_start),
                "cells": selected_cells}
        chunk_ids = ids
        if not lead_attached and lead_text:
            chunk_ids = lead_ids + ids
        rendered = table_markdown(headers, selected)
        if (needs_header_geometry or header_has_formula
                or any("\n$$" in value or "$$\n" in value for row in selected for value in row)):
            rendered = data["html"]
        chunk = table_chunk(with_lead(rendered), data, chunk_ids)
        filtered_rows = [projected(row, index) for index, row in enumerate(selected, start=row_start)]
        elements = []
        for row_index, values in [*enumerate(header_grid), *enumerate(selected, row_start)]:
            for column, value in enumerate(values):
                for left, right, language in foreign_cells.get((row_index, column), ()):
                    location = block["provenance"]
                    if row_index < len(row_provenance):
                        location = row_provenance[row_index]
                    elements.append({"type": "foreign_language", "language": language, "original_type": "table_cell",
                                     "block_id": block["id"], "source_span": [left, right], "text": value[left:right],
                                     "location": {**location, "row": row_index, "column": column}})
        if elements:
            chunk["business_metadata"]["elements"] = elements
            chunk["business_metadata"]["retrieval"]["embedding_content"] = with_lead(
                table_markdown(projected_headers, filtered_rows))
        # When only the authored header remains searchable, give it one
        # carrier chunk. Searchable data rows already include their header.
        carries_header = searchable_header and not searchable_body and row_start == header_count
        has_searchable_data = any(has_visible_text(value) for row in filtered_rows for value in row)
        if lead_attached and not (carries_header or has_searchable_data):
            chunk["business_metadata"]["retrieval"]["role"] = "attachment_only"
        if selected:
            labels = [planner.budget.shorten(header, min(36, planner.budget.maximum // 8)) for header in projected_headers]
            preferred = with_lead(table_markdown(labels, filtered_rows))
            if not fits(preferred):
                preferred = "\n".join(markdown_row(row) for row in filtered_rows)
            planner._bound_passage(chunk, preferred)
        else:
            planner._bound_passage(chunk)
        return attach_lead_metadata(chunk)

    header_overflow = not fits(table_markdown(projected_headers, []))
    def group_text(selected, row_start):
        selected = [projected(row, index) for index, row in enumerate(selected, start=row_start)]
        content = "\n".join(markdown_row(row) for row in selected) if header_overflow else table_markdown(projected_headers, selected)
        return with_lead(content)

    for index, row in enumerate(rows, start=header_count):
        if pending:
            final_row = index == header_count + len(rows) - 1
            pending_text = group_text(pending, start_row)
            under_minimum = planner.budget.count(format_passage_fields(pending_text, retrieval)) < planner.min_tokens
            use_target = not (final_row or under_minimum)
            if len(pending) < planner.table_rows and fits(group_text(pending + [row], start_row), target=use_target):
                pending.append(row)
                continue
            planner._emit(make(pending, start_row))
            pending = []
        start_row = index
        if not lead_attached and lead_text and not fits(group_text([row], index)):
            flush_text(planner, lead_in_blocks, parent_id)
            lead_text = ""
            lead_spans = []
            lead_ids = []
            lead_attached = True
        pending.append(row)
    if pending:
        planner._emit(make(pending, start_row))
    if not rows:
        if not lead_attached and lead_text and lead_in_blocks:
            flush_text(planner, lead_in_blocks, parent_id)
            lead_attached = True
        planner._emit(make([], header_count))
