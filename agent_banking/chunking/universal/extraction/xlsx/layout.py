"""Identify worksheet table regions and preserve real cell coordinates."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256

from agent_banking.core.search_text import search_key

from .package import CellRange, a1, column_name, currency_unit


@dataclass(frozen=True)
class Region:
    bounds: CellRange
    name: str = ""
    kind: str = "inferred"
    headers: int | None = None
    key: str = ""


_QUESTIONS = {"question", "questions", "faq", "cau hoi", "thac mac"}
_ANSWERS = {"answer", "answers", "cau tra loi", "tra loi", "giai dap", "phan hoi"}
_HEADER_WORDS = _QUESTIONS | _ANSWERS | {
    "stt", "ma", "ten", "dich vu", "san pham", "phi", "muc phi", "loai", "don vi",
    "fee", "name", "code", "id", "amount", "currency", "rate", "ngay", "lai suat",
    "han muc", "kenh", "nhom", "hang", "min", "max", "toi thieu", "toi da",
}


def bounds_of(coords):
    rows, columns = zip(*coords)
    return CellRange(min(rows), min(columns), max(rows), max(columns))


def visible(cell):
    return cell is not None and cell.kind != "skipped" and bool(cell.text.strip())


def vertical_faq(cells, explicit):
    by_row = defaultdict(list)
    for point, cell in cells.items():
        if visible(cell):
            by_row[point[0]].append(point)
    previous = None
    for row in sorted(by_row):
        points = by_row[row]
        if len(points) != 1 or any(region.bounds.contains(*points[0]) for region in explicit):
            previous = None
            continue
        point = points[0]
        cell = cells[point]
        heading, separator, text = cell.text.partition(":")
        if cell.kind != "text" or not separator or not text.strip():
            previous = None
            continue
        key = search_key(heading)
        if key in _QUESTIONS or key in {"hoi", "q"}:
            previous = point
            continue
        if (previous and point[1] == previous[1] and 0 < point[0] - previous[0] <= 2
                and (key in _ANSWERS or key in {"dap", "a"})):
            yield previous, point
        previous = None


def _bands(points):
    end = None
    for start, stop in sorted(points):
        if end is None or start > end + 1:
            if end is not None:
                yield first, end
            first, end = start, stop
        else:
            end = max(end, stop)
    if end is not None:
        yield first, end


def _gap_regions(coords, merges):
    pending = [coords]
    while pending:
        current = pending.pop()
        if not current:
            continue
        bounds = bounds_of(current)
        points = set(current)
        merge_extents = [merge for merge in merges if merge[:2] in points]
        for axis, start, end in ((0, 0, 2), (1, 1, 3)):
            markers = {(point[axis], point[axis]) for point in current}
            markers.update((merge[start], merge[end]) for merge in merge_extents)
            bands = list(_bands(markers))
            if len(bands) > 1:
                starts = [first for first, _ in bands]
                separated = [[] for _ in bands]
                for point in current:
                    separated[bisect_right(starts, point[axis]) - 1].append(point)
                pending.extend(reversed(separated))
                break
        else:
            yield Region(CellRange(min([bounds.top, *(m.top for m in merge_extents)]),
                                   min([bounds.left, *(m.left for m in merge_extents)]),
                                   max([bounds.bottom, *(m.bottom for m in merge_extents)]),
                                   max([bounds.right, *(m.right for m in merge_extents)]))), current


def _row_key(cells, coords):
    return tuple((column, search_key(cells[row, column].text)) for row, column in coords)


def _split_repeated_headers(region, coords, cells, merges, bold, classify_splits):
    by_row = defaultdict(list)
    for point in sorted(coords):
        by_row[point[0]].append(point)
    row_numbers = sorted(by_row)
    if len(row_numbers) < 4:
        return [(region, coords)]
    first_row = row_numbers[0]
    if len(by_row[first_row]) == 1 and len(by_row[row_numbers[1]]) > 1:
        first_row = row_numbers[1]
    signature = _row_key(cells, by_row[first_row])
    first_is_header = len(signature) >= 2 and (any(key in _HEADER_WORDS for _, key in signature)
                                               or all(cells[point].style in bold for point in by_row[first_row]))
    cuts = []
    review = []
    for index, row in enumerate(row_numbers[1:-1], 1):
        points = by_row[row]
        if row <= first_row + 1 or len(points) < 2 or any(cells[point].kind != "text" for point in points):
            continue
        keys = _row_key(cells, points)
        styled = all(cells[point].style in bold for point in points) and any(
            cells[point].style not in bold for point in by_row[row_numbers[index + 1]])
        if first_is_header and keys == signature:
            cuts.append(row)
        elif styled and (any(key in _HEADER_WORDS for _, key in keys)
                         or any(cells[point].kind != "text" for point in by_row[row_numbers[index - 1]])):
            review.append({
                "split_before": row,
                "rows": [{"row": number, "cells": [{"address": a1(*point), "text": cells[point].text[:160]}
                                                   for point in by_row[number][:12]]}
                         for number in row_numbers[index - 1:index + 2]],
            })
    if review:
        cuts.extend(classify_splits(review[:12]))
    cuts = sorted({row for row in cuts if not any(
        merge.top < row <= merge.bottom and merge.overlaps(region.bounds) for merge in merges)})
    if not cuts:
        return [(region, coords)]
    bounds = region.bounds
    starts = [bounds.top, *cuts]
    ends = [*(row - 1 for row in cuts), bounds.bottom]
    groups = []
    for top, bottom in zip(starts, ends):
        header_count = None
        if top in cuts:
            header_count = 1
        groups.append((Region(CellRange(top, bounds.left, bottom, bounds.right), headers=header_count),
                       [point for point in coords if top <= point[0] <= bottom]))
    return groups


def find_regions(cells, merges, explicit, bold, classify_splits):
    available = {point for point, cell in cells.items() if visible(cell)}
    claimed = []
    for region in explicit:
        if any(region.bounds.overlaps(previous) for previous in claimed):
            continue
        included = [point for point in available if region.bounds.contains(*point)]
        if included:
            claimed.append(region.bounds)
            for point in included:
                available.remove(point)
            yield region, included
    for region, points in _gap_regions(available, merges):
        yield from _split_repeated_headers(region, points, cells, merges, bold, classify_splits)


def sparse_regions(region, points):
    rows = defaultdict(list)
    columns = set()
    for point in points:
        rows[point[0]].append(point)
        columns.add(point[1])
    if len(rows) * len(columns) <= max(100_000, len(points) * 12):
        yield region, points
        return
    key = region.name or a1(region.bounds.top, region.bounds.left)
    for _, coordinates in sorted(rows.items()):
        yield Region(bounds_of(coordinates), region.name, "sparse", 0, key), coordinates


def _merge_sources(anchors, row, columns, occupied):
    for merge in anchors:
        if merge.top <= row <= merge.bottom:
            for column in columns:
                if merge.left <= column <= merge.right:
                    occupied.setdefault(column, (merge.top, merge.left))
    return occupied


def _infer_headers(rows, columns, resolved, merges, bold):
    if len(rows) < 2 or len(columns) < 2:
        return 0
    first = [resolved[rows[0]][column][0] for column in columns]
    labels = [cell for cell in first if visible(cell)]
    if len(labels) < 2 or any(cell.kind != "text" for cell in labels):
        return 0
    next_cells = [resolved[rows[1]][column][0] for column in columns]
    text_keys = {search_key(cell.text) for cell in labels}
    styled = all(cell.style in bold for cell in labels) and any(
        cell.style not in bold for cell in next_cells if visible(cell))
    typed = any(visible(cell) and cell.kind != "text" for cell in next_cells)
    has_header = styled or typed or bool(text_keys & _HEADER_WORDS)
    if not has_header:
        return None
    count = 1
    while count < min(3, len(rows) - 1):
        above = rows[count - 1]
        current = rows[count]
        group_merge = any(merge.top <= above <= merge.bottom and (
            merge.left < merge.right or merge.bottom >= current) for merge in merges)
        candidate = [resolved[current][column][0] for column in columns]
        if not group_merge or not all(cell.kind == "text" for cell in candidate if visible(cell)):
            break
        count += 1
    return count


def _qa_column(headers, terms):
    return next((column for column, text in headers.items()
                 if search_key(text.rsplit(" / ", 1)[-1]) in terms), None)


def project_region(sheet, region, points, cells, merges, bold, document_key, classify):
    bounds = region.bounds
    owned = set(points)
    relevant_merges = [merge for merge in merges if bounds.overlaps(merge)
                       and (merge[:2] in owned or region.kind == "sparse") and visible(cells.get(merge[:2]))]
    # Sweep occupied rows and columns. Explicit table ranges may contain distant
    # blank formatted cells which must not materialize as a dense rectangle.
    rows = sorted({point[0] for point in points})
    columns = sorted({point[1] for point in points})
    active_merges = []
    remaining_merges = iter(sorted(relevant_merges))
    next_merge = next(remaining_merges, None)
    resolved = {}
    for row in rows:
        while next_merge and next_merge.top <= row:
            active_merges.append(next_merge)
            next_merge = next(remaining_merges, None)
        active_merges = [merge for merge in active_merges if merge.bottom >= row]
        occupied = {column: (row, column) for column in columns if (row, column) in owned
                    or (row, column) in cells and cells[row, column].kind in {"skipped", "text"}
                    and not visible(cells[row, column])}
        occupied = _merge_sources(active_merges, row, columns, occupied)
        values = {}
        for column in columns:
            origin = occupied.get(column)
            if origin is None:
                values[column] = None, (row, column)
            else:
                values[column] = cells.get(origin), origin
        resolved[row] = values
    title = region.name
    title_cell = None
    if region.kind == "inferred" and len(rows) > 2 and len(columns) > 1:
        first = rows[0]
        actual = [(first, column) for column in columns if visible(cells.get((first, column)))]
        if len(actual) == 1 and cells[actual[0]].kind == "text":
            banner = None
            for merge in relevant_merges:
                if merge[:2] == actual[0] and merge.left < merge.right:
                    extra_cells = False
                    for r_chk in rows:
                        if merge.top <= r_chk <= merge.bottom:
                            for c_chk in columns:
                                if (r_chk, c_chk) != actual[0] and visible(cells.get((r_chk, c_chk))):
                                    extra_cells = True
                                    break
                        if extra_cells:
                            break
                    if not extra_cells:
                        banner = merge
                        break
            if banner is not None and any(r_rem > banner.bottom for r_rem in rows):
                title_cell = actual[0]
                title = cells[title_cell].text
                rows = [r_rem for r_rem in rows if r_rem > banner.bottom]
            elif banner is None and len(rows) > 2:
                next_visible = sum(visible(cells.get((rows[1], col))) for col in columns)
                if next_visible >= 2:
                    title_cell = actual[0]
                    title = cells[title_cell].text
                    rows = rows[1:]
    if region.headers is None:
        header_count = _infer_headers(rows, columns, resolved, relevant_merges, bold)
    else:
        header_count = sum(row < bounds.top + region.headers for row in rows)
    question = None
    answers = []
    if header_count is None:
        preview = [{"row": row, "cells": [{"address": a1(row, col), "text": resolved[row][col][0].text[:320]}
                                         for col in columns[:24] if visible(resolved[row][col][0])]}
                   for row in rows[:8]]
        result = classify(preview, min(3, len(rows) - 1), {column_name(col) for col in columns})
        header_count = result.get("header_rows", 0)
        question = result.get("question_column")
        answers = result.get("answer_columns", [])
    header_rows = rows[:header_count]
    column_headers = {}
    for col in columns:
        labels = dict.fromkeys(resolved[row][col][0].text.strip() for row in header_rows
                               if visible(resolved[row][col][0]))
        column_headers[column_name(col)] = " / ".join(labels) or column_name(col)
    question = question or _qa_column(column_headers, _QUESTIONS)
    if question and not answers:
        answers = [key for key, value in column_headers.items()
                   if key != question and search_key(value.rsplit(" / ", 1)[-1]) in _ANSWERS]
    key = region.key or a1(bounds.top, bounds.left)
    if region.kind != "inferred" and region.name:
        key = region.name
    table_id = sha256((document_key.casefold() + "\0" + sheet.casefold() + "\0" + key.casefold()).encode()).hexdigest()[:24]
    data = {"id": table_id, "sheet": sheet, "range": bounds.label(), "columns": column_headers,
            "title": title or bounds.label(), "question_column": question, "answer_columns": answers}
    return data, rows, columns, resolved, header_count, title_cell


def row_record(row, columns, resolved, headers):
    values = {}
    for column in columns:
        cell, origin = resolved[row][column]
        if cell is None:
            continue
        name = column_name(column)
        item = {"type": cell.kind, "address": a1(row, column)}
        if visible(cell):
            item.update(text=cell.text, value=cell.value, number_format=cell.number_format)
            unit = cell.unit
            if not unit and cell.kind == "number":
                unit = currency_unit(headers[name])
            if unit:
                item["unit"] = unit
                if cell.kind == "number":
                    item["type"] = "currency"
        if origin != (row, column):
            item["origin"] = a1(*origin)
        values[name] = item
    return {"row": row, "range": CellRange(row, columns[0], row, columns[-1]).label(), "values": values}

