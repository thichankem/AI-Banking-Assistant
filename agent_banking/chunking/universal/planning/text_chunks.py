"""Budgeted segmentation and exact source spans for text chunks."""
from bisect import bisect_left, bisect_right
from functools import lru_cache

from agent_banking.chunking.config.settings import get_config_section
from .partitioning import atoms_for, optimal_partition
from .source_spans import find_sentence_gap_ends, find_word_starts, has_visible_text
from ..text.boundaries import safe_boundaries

_TEXT_CFG = get_config_section("planning")["text_chunks"]
_ATOM_BOUNDARY_SCORE = _TEXT_CFG["atom_boundary_score"]
_BLOCK_END_SCORE = _TEXT_CFG["block_end_score"]
_SENTENCE_GAP_SCORE = _TEXT_CFG["sentence_gap_score"]
_DOT_DECISION_MIN_WORDS = _TEXT_CFG["dot_decision_min_words"]
_GROUP_BOUNDARY_SCORE = _TEXT_CFG["group_boundary_score"]
_EMPTY_TEXT_PENALTY = _TEXT_CFG["empty_text_penalty"]
_SHORT_CHUNK_PENALTY = _TEXT_CFG["short_chunk_penalty"]
_FIRST_START_MARGIN = _TEXT_CFG["first_start_margin"]
_RETRIEVAL_CACHE_MAXSIZE = _TEXT_CFG["retrieval_cache_maxsize"]


def flush_text(planner, blocks, parent_id):
    if not blocks:
        return
    text = "\n\n".join(block["text"] for block in blocks)
    starts = []
    ends = []
    offset = 0
    for block in blocks:
        starts.append(offset)
        ends.append(offset + len(block["text"]))
        offset = ends[-1] + 2
    boundary_set = {0, len(text)}
    for block, start, end in zip(blocks, starts, ends):
        boundary_set.update(start + point for point in safe_boundaries(block["text"]))
    boundaries = sorted(boundary_set)

    candidates = {}
    for block, start, end in zip(blocks, starts, ends):
        for atom in atoms_for([block], planner.budget):
            point = start + atom["end"] - block.get("source_start", 0)
            aligned = point == len(text) or text[point - 1].isspace() or text[point].isspace()
            if point in boundary_set and aligned:
                candidates[point] = _ATOM_BOUNDARY_SCORE
        candidates[end] = max(candidates.get(end, 0.0), _BLOCK_END_SCORE)
        needs_dot_decision = planner.use_decision and len(block["text"].split()) > _DOT_DECISION_MIN_WORDS
        for gap_end in find_sentence_gap_ends(block["text"], use_decision_for_dots=needs_dot_decision,
                                              cache=planner.decision_cache, ir=planner.ir):
            gap_point = start + gap_end
            if gap_point in boundary_set:
                candidates[gap_point] = max(candidates.get(gap_point, 0.0), _SENTENCE_GAP_SCORE)
        for boundary in planner.boundaries_by_block.get(block["id"], []):
            local = boundary["offset"] - block.get("source_start", 0)
            point = start + local
            if 0 < local <= len(block["text"]) and point in boundary_set:
                candidates[point] = max(candidates.get(point, 0), boundary["score"])
    for index, block in enumerate(blocks[:-1]):
        next_block = blocks[index + 1]
        dependent = frozenset((block["id"], next_block["id"])) in planner.dependent_pairs
        if dependent or block["type"] == "heading":
            candidates[ends[index]] = 0.0
            if starts[index + 1] in candidates:
                candidates[starts[index + 1]] = 0.0

    evidence_ranges = [(start, end, block["context_quotes"]) for block, start, end in zip(blocks, starts, ends)
                       if start < end and block.get("context_quotes")]
    evidence_starts = [start for start, _, _ in evidence_ranges]
    evidence_ends = [end for _, end, _ in evidence_ranges]

    @lru_cache(maxsize=_RETRIEVAL_CACHE_MAXSIZE)
    def retrieval_for_indices(first, last):
        evidence = [item for _, _, quotes in evidence_ranges[first:last] for item in quotes]
        return planner._retrieval(parent_id, evidence)

    def retrieval_for_range(start, end):
        return retrieval_for_indices(bisect_right(evidence_ends, start), bisect_left(evidence_starts, end))

    def measure(start, end, part):
        retrieval = retrieval_for_range(start, end)
        count = planner._measure(part, retrieval)
        if count is None and start < end and boundaries[bisect_right(boundaries, start)] == end:
            return planner.budget.maximum
        return count

    def fits_range(start, end):
        return planner._measure(text[start:end], retrieval_for_range(start, end)) is not None

    def fits_target_range(start, end):
        return planner._measure(text[start:end], retrieval_for_range(start, end), target=True) is not None

    groups = []
    for index, block in enumerate(blocks[:-1]):
        if (frozenset((block["id"], blocks[index + 1]["id"])) in planner.dependent_pairs
                and fits_range(starts[index], ends[index + 1])):
            groups.append((starts[index], ends[index + 1]))

    combined_groups = []
    for start, end in sorted(groups):
        if combined_groups and start < combined_groups[-1][1]:
            previous_start, previous_end = combined_groups[-1]
            if fits_range(previous_start, max(previous_end, end)):
                combined_groups[-1] = (previous_start, max(previous_end, end))
        else:
            combined_groups.append((start, end))
    word_starts = find_word_starts(text)
    token_offsets = sorted(end for start, end in planner.budget.tokenizer.encode(text, add_special_tokens=False).offsets
                           if start < end)

    def first_start(end):
        earlier_tokens = bisect_right(token_offsets, end) - planner.budget.maximum - _FIRST_START_MARGIN
        if earlier_tokens > 0:
            earliest = token_offsets[earlier_tokens - 1]
        else:
            earliest = 0
        if planner.max_chars is not None:
            earliest = max(earliest, end - planner.max_chars)
        if planner.max_words is not None:
            earlier_words = bisect_left(word_starts, end) - planner.max_words
            if earlier_words > 0:
                earliest = max(earliest, word_starts[earlier_words - 1] + 1)
        position = bisect_left(boundaries, end)
        return min(earliest, boundaries[max(0, position - 1)])

    def short_penalty(part, score, size):
        if not has_visible_text(part):
            return _EMPTY_TEXT_PENALTY
        if size < planner.min_tokens or len(part.strip()) < planner.min_chars:
            return _SHORT_CHUNK_PENALTY
        return 0.0

    # Generate a feasible fallback chain around semantically linked groups.
    cursor = 0
    fallback_points = {}
    for start, end in [*combined_groups, (len(text), len(text))]:
        local_boundaries = [point - cursor for point in boundaries[bisect_left(boundaries, cursor):bisect_right(boundaries, start)]]
        for left, stop, _ in planner.budget.split(
                text[cursor:start], None, align_words=True, boundaries=local_boundaries, allow_oversized=True,
                fits_range=lambda left, right: fits_target_range(cursor + left, cursor + right)):
            candidates.setdefault(cursor + stop, 0.0)
            fallback_points[cursor + stop] = cursor + left
        if end > start:
            for point in [point for point in candidates if start < point < end]:
                del candidates[point]
            candidates[start] = max(_GROUP_BOUNDARY_SCORE, candidates.get(start, 0.0))
            candidates[end] = max(_GROUP_BOUNDARY_SCORE, candidates.get(end, 0.0))
            fallback_points[end] = start
        cursor = end
    ranges = optimal_partition(text, candidates, None, None, planner.budget.target,
                               first_start=first_start, measure=measure, penalty=short_penalty,
                               fallback_points=fallback_points)
    emit_text(planner, blocks, parent_id, starts, ends, ranges, retrieval_for_range)
    blocks.clear()

def emit_text(planner, blocks, parent_id, starts, ends, ranges, retrieval_for_range):
    full_text = "\n\n".join(block["text"] for block in blocks)
    for start, end, _ in ranges:
        active_indices = [
            index for index in range(bisect_right(ends, start), bisect_left(starts, end))
            if max(start, starts[index]) < min(end, ends[index])
        ]
        if not active_indices:
            continue
        actual_start = max(start, starts[active_indices[0]])
        actual_end = min(end, ends[active_indices[-1]])
        part = full_text[actual_start:actual_end]
        spans = []
        for index in active_indices:
            offset = starts[index]
            left = max(actual_start, offset)
            right = min(actual_end, ends[index])
            if left < right:
                span_offset = blocks[index].get("source_start", 0)
                spans.append({"block_id": blocks[index]["id"], "start": left - offset + span_offset, "end": right - offset + span_offset,
                              "chunk_start": left - actual_start, "chunk_end": right - actual_start})
        if not spans:
            continue
        ids = [span["block_id"] for span in spans]
        chunk = planner._prototype("text", part, parent_id, ids, retrieval=retrieval_for_range(actual_start, actual_end))
        if not has_visible_text(part):
            for span in spans:
                planner._keep_foreign(planner.block_map[span["block_id"]], span["start"], span["end"], None)
            continue
        planner._bound_passage(chunk)
        chunk["business_metadata"]["provenance"].update(group_span=[actual_start, actual_end], source_spans=spans)
        planner._emit(chunk)
