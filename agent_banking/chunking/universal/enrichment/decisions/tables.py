"""Decision-model table headings, titles, and cross-page continuation."""
from copy import deepcopy

from .images import adjacent_source
from ..image_context import region_key
from ...text.language import project_vietnamese


def resolve_table_structures_and_titles(ir, ask_decisions, cache=None):
    """Use decision model to resolve unmarked table header counts and assign descriptive table titles."""
    blocks = ir["blocks"]
    table_entries = []
    questions = {}
    table_map = {}
    headings = {}
    by_id = {block["id"]: block for block in blocks}
    caption_sources = {}
    for relation in ir.get("relations", []):
        if relation["type"] == "caption_of" and relation["from"] in by_id:
            caption_sources.setdefault(relation["to"], []).append(by_id[relation["from"]])
    for idx, block in enumerate(blocks):
        region = region_key(block)
        if block["type"] == "heading":
            heading = block.get("text", "")
            if ir.get("search_language") != "all":
                heading = project_vietnamese(heading)
            heading = heading.strip()
            if heading:
                headings[region] = (heading, block["id"])
            continue
        if block["type"] != "table":
            continue
        table = block["table_data"]
        grid = table.get("grid", [])
        if not grid:
            continue
        tid = "t" + str(len(table_entries))
        width = len(grid[0])
        top_rows = [[text[:160] for text in row[:12]] for row in grid[:4]]
        row_bold = []
        for r_idx in range(min(3, len(grid))):
            r_cells = [c for c in table.get("cells", []) if c["row"] == r_idx and c.get("text", "").strip()]
            is_b = bool(r_cells) and all(c.get("all_text_bold", False) for c in r_cells)
            row_bold.append(is_b)
        banner = next((cell["text"].strip() for cell in table.get("cells", [])
                       if cell["row"] == 0 and cell["column"] == 0 and width > 1
                       and cell.get("col_span", cell.get("column_span", 1)) == width), "")
        preceding = adjacent_source(blocks, idx, -1, {"paragraph", "caption", "heading"})
        following = adjacent_source(blocks, idx, 1, {"caption", "paragraph"})
        prev_text = ""
        prev_type = "none"
        if preceding:
            prev_text = preceding.get("text", "").strip()
            prev_type = preceding["type"]
        next_text = ""
        next_type = "none"
        if following:
            next_text = following.get("text", "").strip()
            next_type = following["type"]
        heading, heading_id = headings.get(region, ("", None))
        entry = {
            "id": tid,
            "width": width,
            "total_rows": len(grid),
            "top_rows": top_rows,
            "row_bold": row_bold,
            "row_0_banner": bool(banner),
            "existing_header_count": table.get("header_row_count", 0),
            "prev_type": prev_type,
            "prev_text": prev_text[:200],
            "next_type": next_type,
            "next_text": next_text[:200],
            "section_heading": heading[:200],
        }
        table_entries.append(entry)
        table_map[tid] = (block, entry, banner, (heading, heading_id),
                          (prev_text, preceding["id"] if preceding else None),
                          (next_text, following["id"] if following else None))
        if table.get("header_detection") == "unmarked" and len(grid) >= 2 and "row_provenance" not in table:
            criteria = {
                "HEADERS_1": "First row is the column header row",
                "HEADERS_0": "No header row; all rows are data records",
            }
            if len(grid) > 2:
                criteria["HEADERS_2"] = "First two rows form grouped column headers or a banner above column headers"
            questions[tid + "_hdr"] = {
                "type": "choice",
                "instructions": "Determine how many leading rows in table " + tid + " are column header rows (not data records).",
                "criteria": criteria,
            }
        questions[tid + "_title"] = {
            "type": "choice",
            "instructions": "Select the best descriptive title source for table " + tid + ".",
            "criteria": {
                "PREV_BLOCK": "The immediately preceding block introduces or titles this table",
                "NEXT_BLOCK": "The immediately following block is the caption/title of this table",
                "ROW_0_BANNER": "Row 0 of the table itself is a full-width banner title",
                "SECTION_HEADING": "The enclosing section heading is the best title for this table",
            },
        }
    if not table_entries:
        return
    answers = ask_decisions("table_structure_and_title", {"tables": table_entries}, questions, cache=cache)
    for tid, (block, entry, banner, section_title, previous_title, next_title) in table_map.items():
        table = block["table_data"]
        grid = table["grid"]
        width = entry["width"]
        hdr_key = tid + "_hdr"
        if hdr_key in answers:
            hdr_choice = answers[hdr_key].get("choice", "HEADERS_1")
            count = {"HEADERS_0": 0, "HEADERS_1": 1, "HEADERS_2": 2}[hdr_choice]
            if 0 <= count < len(grid):
                table["header_row_count"] = count
                table["header_grid"] = grid[:count]
                table["rows"] = grid[count:]
                table["header_detection"] = "decision_model_headers"
        title_choice = answers.get(tid + "_title", {}).get("choice", "SECTION_HEADING")
        title_options = {"ROW_0_BANNER": (banner, None), "PREV_BLOCK": previous_title,
                         "NEXT_BLOCK": next_title, "SECTION_HEADING": section_title}
        resolved_title, title_source = title_options[title_choice]
        if not resolved_title:
            caption = block.get("caption", "")
            caption_source = next((item["id"] for item in caption_sources.get(block["id"], [])
                                   if item["text"] == caption), None)
            resolved_title, title_source = next((candidate for candidate in (
                (caption, caption_source), section_title, previous_title) if candidate[0]), ("", None))
        header_count = table["header_row_count"]
        if header_count:
            start_hdr_row = 0
            if banner and header_count >= 2:
                start_hdr_row = 1
                if not resolved_title:
                    resolved_title = banner
            headers = []
            for column in range(width):
                labels = []
                for row in grid[start_hdr_row:header_count]:
                    if row[column] and row[column] not in labels:
                        labels.append(row[column])
                headers.append(" / ".join(labels))
            if "row_provenance" not in table:
                table["headers"] = headers
        table["title"] = resolved_title
        table.pop("title_source_block_id", None)
        if title_source:
            table["title_source_block_id"] = title_source
        if resolved_title and not block.get("caption"):
            block["caption"] = resolved_title


def table_header_cells(table):
    return [cell for cell in table.get("cells", []) if cell["row"] < table["header_row_count"]]


def compatible_table_headers(first, continuation):
    count = continuation["header_row_count"]
    if not count:
        return True
    if first["header_grid"] != continuation["header_grid"] or first["headers"] != continuation["headers"]:
        return False
    cells = table_header_cells(continuation)
    if any(cell["row"] + cell["row_span"] > count for cell in cells):
        return False
    return table_header_cells(first) == cells


def table_row_provenance(block):
    table = block["table_data"]
    return [{**deepcopy(block["provenance"]), "source_block_id": block["id"], "source_row": row}
            for row in range(table["header_row_count"] + len(table["rows"]))]


def record_header_occurrence(target, source, row_provenance):
    count = source["header_row_count"]
    if count:
        target.setdefault("header_occurrences", []).append({
            "row_provenance": row_provenance[:count], "grid": deepcopy(source["header_grid"]),
            "cells": deepcopy(table_header_cells(source)),
        })


def merge_continued_tables(ir, ask_decisions, cache=None):
    """Use decision model to determine whether consecutive multi-page PDF tables should merge."""
    blocks = ir["blocks"]
    if len(blocks) < 2:
        return
    pairs = []
    questions = {}
    pair_indices = {}
    for idx in range(1, len(blocks)):
        prev = blocks[idx - 1]
        curr = blocks[idx]
        if prev["type"] != "table" or curr["type"] != "table":
            continue
        prev_page = prev.get("provenance", {}).get("page_index", -2)
        curr_page = curr.get("provenance", {}).get("page_index", -1)
        t1 = prev["table_data"]
        t2 = curr["table_data"]
        if prev_page >= 0 and curr_page == prev_page + 1 and len(t1["headers"]) == len(t2["headers"]) and len(t1["headers"]) > 0:
            pid = "tbl_" + str(len(pairs))
            prev_tail = []
            if t1["rows"]:
                prev_tail = [value[:160] for value in t1["rows"][-1][:12]]
            curr_head = []
            if t2["rows"]:
                curr_head = [value[:160] for value in t2["rows"][0][:12]]
            pairs.append({
                "id": pid,
                "prev_page": prev_page,
                "curr_page": curr_page,
                "prev_bbox": prev.get("provenance", {}).get("bbox", []),
                "curr_bbox": curr.get("provenance", {}).get("bbox", []),
                "prev_headers": [value[:160] for value in t1["headers"][:12]],
                "curr_headers": [value[:160] for value in t2["headers"][:12]],
                "prev_last_row": prev_tail,
                "curr_first_row": curr_head,
            })
            pair_indices[pid] = idx
            questions[pid] = {
                "type": "choice",
                "instructions": "Decide if table pair " + pid + " across consecutive pages is a single continued table split by a page break.",
                "criteria": {
                    "MERGE_CONTINUED": "Second table is a direct continuation of the first table across the page boundary",
                    "KEEP_SEPARATE": "Second table is a distinct, independent table",
                },
            }
    if not questions:
        return
    answers = ask_decisions("pdf_table_continuation", {"table_pairs": pairs}, questions, cache=cache)
    merge_indices = {
        idx for pid, idx in pair_indices.items()
        if answers.get(pid, {}).get("choice") == "MERGE_CONTINUED"
    }
    if not merge_indices:
        return
    merged_blocks = []
    merged_sources = {}
    for idx, block in enumerate(blocks):
        if idx in merge_indices and merged_blocks and merged_blocks[-1]["type"] == "table":
            prev = merged_blocks[-1]
            t1 = prev["table_data"]
            t2 = block["table_data"]
            if not compatible_table_headers(t1, t2):
                # Both headers are source data. Keep them in their own tables
                # while recording the model-confirmed continuation.
                ir["relations"].append({"from": block["id"], "to": prev["id"], "type": "continues"})
                merged_blocks.append(block)
                continue
            row_offset = len(t1["header_grid"]) + len(t1["rows"])
            t2_header_count = t2["header_row_count"]
            if "row_provenance" not in t1:
                t1["row_provenance"] = table_row_provenance(prev)
                record_header_occurrence(t1, t1, t1["row_provenance"])
            next_rows = table_row_provenance(block)
            record_header_occurrence(t1, t2, next_rows)
            t1["row_provenance"].extend(next_rows[t2_header_count:])
            for cell in t2.get("cells", []):
                if cell["row"] < t2_header_count:
                    continue
                shifted = deepcopy(cell)
                shifted["row"] = row_offset + (cell["row"] - t2_header_count)
                t1.setdefault("cells", []).append(shifted)
            t1["rows"].extend(t2["rows"])
            t1["grid"] = t1["header_grid"] + t1["rows"]
            merged_sources[block["id"]] = (prev["id"], row_offset, t2_header_count)
            continue
        merged_blocks.append(block)
    if merged_sources:
        for index, block in enumerate(merged_blocks):
            block["order"] = index
        for unit in ir["units"]:
            source = unit["block_id"]
            if source in merged_sources:
                target, offset, header_count = merged_sources[source]
                unit["block_id"] = target
                unit["source_block_id"] = source
                unit["source_row"] = unit["row"]
                if unit["row"] >= header_count:
                    unit["row"] += offset - header_count
        ir["blocks"] = merged_blocks
