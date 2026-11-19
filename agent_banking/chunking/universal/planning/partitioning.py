"""Source atoms and lossless shortest-path segmentation."""
import math
from bisect import bisect_right

from agent_banking.chunking.config.settings import get_config_section
from ..text.boundaries import safe_boundaries

_PART_CFG = get_config_section("planning")["partitioning"]
_DEFAULT_STRIDE = _PART_CFG["stride"]
_MAX_RAW_CHARS_PER_ATOM = _PART_CFG["max_raw_chars_per_atom"]
_COHESION_WEIGHT = _PART_CFG["cohesion_weight"]
_COHESION_POWER = _PART_CFG["cohesion_power"]
_BOUNDARY_WEIGHT = _PART_CFG["boundary_weight"]
_SIZE_PENALTY_WEIGHT = _PART_CFG["size_penalty_weight"]


def atoms_for(blocks, budget, stride=_DEFAULT_STRIDE):
    if type(stride) is not int or stride < 1:
        raise ValueError("Semantic atom stride must be a positive integer")
    atoms = []
    for block in blocks:
        text = block["text"]
        source_start = block.get("source_start", 0)
        offsets = budget.tokenizer.encode(text, add_special_tokens=False).offsets
        ends = set()
        for index in range(stride - 1, len(offsets), stride):
            ends.add(offsets[index][1])
        ends.add(len(text))
        # Some tokenizers normalize large runs of whitespace or invisible
        # characters to a handful of tokens. Cap model input by raw length too.
        if len(text) > _MAX_RAW_CHARS_PER_ATOM:
            ends.update(range(_MAX_RAW_CHARS_PER_ATOM, len(text), _MAX_RAW_CHARS_PER_ATOM))
        safe = safe_boundaries(text, protect_markup=False)
        ends = {safe[bisect_right(safe, position) - 1] for position in ends}
        start = 0
        for end in sorted(ends):
            if end <= start:
                continue
            atoms.append({"id": "a" + str(len(atoms)), "block_id": block["id"],
                          "type": block["type"], "start": start + source_start,
                          "end": end + source_start, "text": text[start:end]})
            start = end
    return atoms


def optimal_partition(text, candidates, fits, count, target, *, first_start=None, measure=None, penalty=None,
                      fallback_points=None):
    """Shortest path over ordered source offsets, with exact passage constraints."""
    if not text:
        return []
    values = {0, len(text)} | set(candidates)
    if any(isinstance(point, bool) or not isinstance(point, int) or not 0 <= point <= len(text) for point in values):
        raise ValueError("Invalid source boundary offset")
    points = sorted(values)
    costs = [math.inf] * len(points)
    previous = [None] * len(points)
    costs[0] = 0
    interior = [0.0]
    for index, point in enumerate(points):
        loss = _COHESION_WEIGHT * candidates.get(point, 0.0) ** _COHESION_POWER
        if index > 0 and not text[points[index - 1]:point].strip():
            loss = 0.0
        interior.append(interior[-1] + loss)
    for right in range(1, len(points)):
        if first_start is None:
            earliest = 0
        else:
            earliest = first_start(points[right])
        if fallback_points and points[right] in fallback_points:
            earliest = min(earliest, fallback_points[points[right]])
        for left in range(right - 1, -1, -1):
            if points[left] < earliest:
                break
            if not math.isfinite(costs[left]) or costs[left] + 1 >= costs[right]:
                continue
            start, end = points[left], points[right]
            part = text[start:end]
            if measure is None:
                if fits(part):
                    size = count(part)
                else:
                    size = None
            else:
                size = measure(start, end, part)
            if size is None:
                # Token counts are not strictly monotone. Do not prune on a
                # failed count; BPE merges can make a longer input shorter.
                continue
            score = candidates.get(points[right], 0.0)
            cohesion_loss = interior[right] - interior[left + 1]
            size_penalty = _SIZE_PENALTY_WEIGHT * ((size - target) / target) ** 2
            cost = costs[left] + 1 + _BOUNDARY_WEIGHT * (1 - score) + cohesion_loss + size_penalty
            if right == len(points) - 1:
                cost -= _BOUNDARY_WEIGHT * (1 - score)
            if penalty is not None:
                cost += penalty(part, score, size)
            if cost < costs[right]:
                costs[right] = cost
                previous[right] = left
    if previous[-1] is None:
        raise ValueError("No feasible semantic segmentation under passage budget")
    ranges = []
    cursor = len(points) - 1
    while cursor:
        left = previous[cursor]
        ranges.append((points[left], points[cursor], text[points[left]:points[cursor]]))
        cursor = left
    ranges.reverse()
    if "".join(part for _, _, part in ranges) != text:
        raise AssertionError("Semantic segmentation changed source")
    return ranges
