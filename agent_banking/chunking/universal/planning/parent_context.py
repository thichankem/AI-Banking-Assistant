"""Section materialization and bounded context around an emitted child."""
from bisect import bisect_left, bisect_right
from collections import defaultdict

from agent_banking.chunking.config.settings import get_config_section
from ..text.language import projected_offset, vietnamese_projection
from .source_spans import find_word_starts
from .table_rendering import table_markdown

_MAX_CHARS_MULTIPLIER = get_config_section("planning")["parent_context"]["max_chars_multiplier"]


def materialize_parents(planner):
    children_by_parent = defaultdict(list)
    chunks_by_parent = defaultdict(list)
    for identifier, parent in planner.parents.items():
        children_by_parent[parent["parent_id"]].append(identifier)
    for chunk in planner.chunks:
        chunks_by_parent[chunk["parent_id"]].append(chunk["chunk_id"])
    # Materialize full section context; it is never embedded as a parent.
    planner.parent_text_index = {}
    planner._subtree_cache = {}
    for parent_id, parent in planner.parents.items():
        parent["children"] = children_by_parent[parent_id]
        parent["child_chunk_ids"] = chunks_by_parent[parent_id]
        pieces = []
        offsets = {}
        offset = 0
        for block_id in parent["block_ids"]:
            block = planner.block_map[block_id]
            if block.get("foreign_language"):
                continue
            if block["type"] == "table":
                table = block["table_data"]
                piece = table_markdown(table["headers"], table["rows"])
            else:
                piece = vietnamese_projection(block)
            if piece:
                pieces.append(piece)
                offsets[block_id] = (offset, offset + len(piece))
                offset += len(piece) + 2
        parent["content"] = "\n\n".join(pieces)
        planner.parent_text_index[parent_id] = (offsets, find_word_starts(parent["content"]))


def windowed_parent_content(planner, chunk, parent):
    """Return bounded context around the child, never just a copy of the child.

    The own section is preferred. When it adds nothing beyond the child (a leaf
    section with one chunk), the window widens to the enclosing section subtree
    so retrieval receives neighboring prose, sibling rows and lead-ins.
    """
    child_content = chunk["content"]
    content = _section_window(planner, chunk, parent)
    if _same_text(content, child_content):
        widened = _ancestor_window(planner, chunk, parent)
        if widened:
            return widened
    return content


def _same_text(left, right):
    return " ".join(left.split()) == " ".join(right.split())


def _window(full_content, word_starts, start, end, limit, max_chars):
    if (not limit or len(word_starts) <= limit) and len(full_content) <= max_chars:
        return full_content
    first_word = max(0, bisect_right(word_starts, start) - 1)
    last_word = bisect_left(word_starts, end)
    used_words = last_word - first_word
    if limit and used_words >= limit:
        return None
    if not limit:
        limit = len(word_starts)
    left = max(0, first_word - (limit - used_words) // 2)
    right = min(len(word_starts), left + limit)
    left = max(0, right - limit)
    window_start = min(start, word_starts[left])
    window_end = len(full_content)
    if right < len(word_starts):
        window_end = max(end, word_starts[right])
    if window_end - window_start > max_chars:
        margin = max(0, (max_chars - (end - start)) // 2)
        window_start = max(window_start, start - margin)
        window_end = min(window_end, end + margin)
        while window_start < start and window_start > 0 and not full_content[window_start - 1].isspace():
            window_start += 1
        while window_end > end and window_end < len(full_content) and not full_content[window_end].isspace():
            window_end -= 1
    return full_content[window_start:window_end]


def _section_window(planner, chunk, parent):
    full_content = parent["content"]
    child_content = chunk["content"]
    offsets, word_starts = planner.parent_text_index[chunk["parent_id"]]
    limit = planner.max_parent_words
    max_chars = planner.budget.maximum * _MAX_CHARS_MULTIPLIER
    provenance = chunk["business_metadata"]["provenance"]
    start = -1
    end = -1
    source_spans = provenance.get("source_spans", [])
    if source_spans:
        first_span = source_spans[0]
        block_range = offsets.get(first_span["block_id"])
        if block_range:
            block = planner.block_map[first_span["block_id"]]
            offset = projected_offset(block["text"], block.get("foreign_spans", ()), first_span["start"])
            candidate = block_range[0] + offset - first_span["chunk_start"]
            if candidate >= 0 and full_content.startswith(child_content, candidate):
                start = candidate
                end = start + len(child_content)
    ranges = [offsets[identifier] for identifier in provenance["block_ids"] if identifier in offsets]
    if start < 0 and ranges:
        start = full_content.find(child_content, min(item[0] for item in ranges), max(item[1] for item in ranges))
        if start >= 0:
            end = start + len(child_content)
    if start < 0 and ranges:
        # Table fragments repeat headers or render projected rows, so they are
        # not literal substrings of the section. Window around their source
        # blocks instead, which keeps sibling rows and surrounding prose.
        start = min(item[0] for item in ranges)
        end = max(item[1] for item in ranges)
    if start >= 0:
        window = _window(full_content, word_starts, start, end, limit, max_chars)
        if window:
            return window
    heading = {}
    if parent["block_ids"]:
        heading = planner.block_map[parent["block_ids"][0]]
    visible_heading = vietnamese_projection(heading).strip()
    if heading.get("type") == "heading" and visible_heading and visible_heading not in child_content:
        with_heading = visible_heading + "\n\n" + child_content
        if (not limit or len(with_heading.split()) <= limit) and len(with_heading) <= max_chars:
            return with_heading
    return child_content


def _subtree_content(planner, parent_id):
    cache = planner._subtree_cache
    if parent_id in cache:
        return cache[parent_id]
    ordered = []

    def first_order(identifier):
        node = planner.parents[identifier]
        orders = [planner.block_order[block_id] for block_id in node["block_ids"] if block_id in planner.block_order]
        if orders:
            return min(orders)
        child_orders = [first_order(child) for child in node.get("children", [])]
        child_orders = [value for value in child_orders if value is not None]
        if child_orders:
            return min(child_orders)
        return None

    def walk(identifier):
        node = planner.parents[identifier]
        if node["content"]:
            ordered.append(node["content"])
        children = []
        for child in node.get("children", []):
            order = first_order(child)
            if order is None:
                order = len(planner.block_order)
            children.append((order, child))
        for _, child in sorted(children):
            walk(child)

    walk(parent_id)
    content = "\n\n".join(ordered)
    cache[parent_id] = (content, find_word_starts(content))
    return cache[parent_id]


def _ancestor_window(planner, chunk, parent):
    child_content = chunk["content"]
    section_content = parent["content"]
    limit = planner.max_parent_words
    max_chars = planner.budget.maximum * _MAX_CHARS_MULTIPLIER
    cursor = parent
    while cursor.get("parent_id"):
        ancestor_id = cursor["parent_id"]
        if planner.ir.get("format") == "xlsx" and not planner.parents[ancestor_id]["parent_id"]:
            break
        content, word_starts = _subtree_content(planner, ancestor_id)
        anchor = child_content
        start = content.find(anchor)
        if start < 0 and section_content:
            anchor = section_content
            start = content.find(anchor)
        if start >= 0:
            window = _window(content, word_starts, start, start + len(anchor), limit, max_chars)
            if window and not _same_text(window, child_content):
                return window
        cursor = planner.parents[ancestor_id]
    return None
