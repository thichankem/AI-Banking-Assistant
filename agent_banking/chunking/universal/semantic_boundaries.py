"""Model-scored source offsets and globally optimized, lossless segmentation.

Token offsets form a content-independent search lattice. The model scores
these offsets and can suggest additional cuts with verbatim source evidence.
"""
import hashlib
import json
import math
from collections import defaultdict

from .model_enrichment import cloud_json, ordered_parallel
from agent_banking.core.models import model_primary, chain_fingerprint

BOUNDARY_VERSION = "semantic-lattice-v3-source-start-anchors"
ATOM_SCORE_VERSION = "semantic-lattice-v2-anchored-cuts"
START_SCORE = 0.85
from agent_banking.chunking.config.prompts import START_PROMPT, BOUNDARY_PROMPT


def atoms_for(blocks, budget, stride=96):
    atoms = []
    for block in blocks:
        text = block["text"]
        offsets = budget.tokenizer.encode(text, add_special_tokens=False).offsets
        ends = sorted({offsets[index][1] for index in range(stride - 1, len(offsets), stride)} | {len(text)})
        start = 0
        for end in ends:
            if end <= start:
                continue
            atoms.append({"id": "a" + str(len(atoms)), "block_id": block["id"],
                          "type": block["type"], "start": start, "end": end, "text": text[start:end]})
            start = end
    return atoms


def source_starts(source, value):
    if isinstance(value, dict):
        proposals = value.get("starts")
    else:
        proposals = None
    if not isinstance(proposals, list):
        raise ValueError("Source starts must be an array")
    anchors = {}
    for proposal in proposals:
        if isinstance(proposal, dict):
            quote = proposal.get("start_quote")
        else:
            quote = None
        if not isinstance(quote, str) or not 15 <= len(quote) <= 100 or quote[0].isspace():
            raise ValueError("Source start requires a verbatim 15 to 100 character prefix")
        offset = source.find(quote)
        if offset <= 0 or offset != source.rfind(quote) or offset in anchors:
            raise ValueError("Source start must identify one unique interior offset")
        anchors[offset] = quote
    return sorted(anchors.items())


class SemanticBoundaryScorer:
    def __init__(self, cache, budget, workers=2):
        self.cache = cache
        self.budget = budget
        self.workers = max(1, workers)

    def _start_proposals(self, block, block_atoms, endpoint):
        text = block["text"]
        anchors = []
        for start in range(0, len(block_atoms), 24):
            first = block_atoms[start]["start"]
            last = block_atoms[min(start + 23, len(block_atoms) - 1)]["end"]
            source = text[first:last]
            key = hashlib.sha256(json.dumps({"version": BOUNDARY_VERSION, "prompt": START_PROMPT,
                "chain": chain_fingerprint("chunking.text"), "source": source}, sort_keys=True).encode()).hexdigest()
            cached = self.cache.get(key)
            if cached is not None:
                try:
                    anchors.extend((first + offset, quote) for offset, quote in source_starts(source, cached))
                    continue
                except ValueError:
                    pass
            messages = [{"role": "system", "content": START_PROMPT}, {"role": "user", "content": source}]
            for attempt in range(3):
                response = cloud_json(endpoint.url.rstrip("/") + "/chat/completions", endpoint.api_key,
                    {"model": endpoint.model, "messages": messages, "temperature": 0,
                     "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 4096})
                try:
                    valid = source_starts(source, response)
                except ValueError as error:
                    if attempt == 2:
                        return []
                    messages.extend([{"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
                        {"role": "user", "content": "Repair the exact source anchors: " + str(error) +
                            ". Return a complete starts array using only unique verbatim prefixes from the supplied text."}])
                    continue
                self.cache.put(key, response, 0, 0)
                anchors.extend((first + offset, quote) for offset, quote in valid)
                break
        return anchors

    def annotate(self, ir):
        blocks = [block for block in ir["blocks"] if block["type"] in ("heading", "paragraph", "caption")]
        atoms = atoms_for(blocks, self.budget)
        endpoint = model_primary("chunking.text")
        long_blocks = [block for block in blocks if self.budget.count(block["text"]) > self.budget.maximum]
        by_block = defaultdict(list)
        for atom in atoms:
            by_block[atom["block_id"]].append(atom)
        proposed_starts = ordered_parallel(lambda block: self._start_proposals(block, by_block[block["id"]], endpoint),
            long_blocks, self.workers)
        starts = {block["id"]: proposals for block, proposals in zip(long_blocks, proposed_starts) if proposals}

        def window(start):
            selected = atoms[max(0, start - 3):start + 27]
            editable = [atom for atom in atoms[start:start + 24] if atom["block_id"] not in starts]
            if not editable:
                return {"boundaries": [], "proposals": []}
            state = {"atoms": selected, "editable_ids": [atom["id"] for atom in editable]}
            key = hashlib.sha256(json.dumps({"version": ATOM_SCORE_VERSION, "prompt": BOUNDARY_PROMPT,
                "chain": chain_fingerprint("chunking.text"), "input": state}, sort_keys=True).encode()).hexdigest()

            def validate(value):
                if not isinstance(value, dict):
                    raise ValueError("Semantic scorer must return an object")
                entries = value.get("boundaries")
                expected = set(state["editable_ids"])
                if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
                    raise ValueError("Semantic boundaries must be objects")
                if len(entries) != len(expected) or {item.get("id") for item in entries} != expected:
                    supplied_ids = {item.get("id") for item in entries}
                    raise ValueError("Semantic scorer must account for every editable atom exactly once; missing="
                                     + str(sorted(expected - supplied_ids)) + "; unexpected=" + str(sorted(supplied_ids - expected, key=str)))
                for item in entries:
                    score = item.get("score")
                    if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Boundary score must be finite in [0,1]")
                proposals = value.get("proposals", [])
                by_id = {atom["id"]: atom for atom in editable}
                # Interior cuts are optional. Keep validated atom scores even
                # when a suggestion points at a neighboring, read-only atom.
                if not isinstance(proposals, list):
                    return {"boundaries": entries, "proposals": [], "rejected_proposals": 1}
                accepted = []
                for proposal in proposals:
                    if not isinstance(proposal, dict):
                        continue
                    identifier = proposal.get("atom_id")
                    if not isinstance(identifier, str) or identifier not in by_id:
                        continue
                    quote = proposal.get("end_quote")
                    score = proposal.get("score")
                    if (isinstance(quote, str) and quote and by_id[identifier]["text"].count(quote) == 1
                            and not isinstance(score, bool) and isinstance(score, (int, float))
                            and math.isfinite(score) and 0 <= score <= 1):
                        accepted.append({"atom_id": identifier, "end_quote": quote, "score": score})
                return {"boundaries": entries, "proposals": accepted, "rejected_proposals": len(proposals) - len(accepted)}

            cached = self.cache.get(key)
            if cached is not None:
                try:
                    checked = validate(cached)
                    rejected = cached.get("rejected_proposals", 0)
                    if type(rejected) is int and rejected > 0:
                        checked["rejected_proposals"] += rejected
                    return checked
                except (ValueError, TypeError):
                    pass
            messages = [{"role": "system", "content": BOUNDARY_PROMPT},
                        {"role": "user", "content": json.dumps(state, ensure_ascii=False)}]
            for attempt in range(3):
                result = cloud_json(endpoint.url.rstrip("/") + "/chat/completions", endpoint.api_key,
                    {"model": endpoint.model, "messages": messages, "temperature": 0,
                     "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 8192})
                try:
                    checked = validate(result)
                    self.cache.put(key, checked, 0, 0)
                    return checked
                except (ValueError, TypeError) as error:
                    if attempt == 2:
                        raise
                    messages.extend([{"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                        {"role": "user", "content": "Repair scores: " + str(error) +
                         ". Return exactly these editable IDs once each: " + json.dumps(state["editable_ids"]) +
                         ". Return a complete boundaries array and optional source-grounded proposals, not a patch."}])

        results = ordered_parallel(window, list(range(0, len(atoms), 24)), self.workers)
        ir["semantic_rejected_proposals"] = sum(result.get("rejected_proposals", 0) for result in results)
        scores = {item["id"]: item["score"] for result in results for item in result["boundaries"]}
        ir["semantic_boundaries"] = [{"block_id": atom["block_id"], "offset": atom["end"],
                                       "score": scores[atom["id"]]} for atom in atoms if atom["id"] in scores]
        for block_id, proposals in starts.items():
            ir["semantic_boundaries"].extend({"block_id": block_id, "offset": offset,
                "score": START_SCORE, "source_start": quote} for offset, quote in proposals)
        by_id = {atom["id"]: atom for atom in atoms}
        for result in results:
            for proposal in result.get("proposals", []):
                atom = by_id[proposal["atom_id"]]
                offset = atom["start"] + atom["text"].index(proposal["end_quote"]) + len(proposal["end_quote"])
                ir["semantic_boundaries"].append({"block_id": atom["block_id"], "offset": offset, "score": proposal["score"],
                                                   "source_ending": proposal["end_quote"]})
        ir["segmentation_method"] = BOUNDARY_VERSION
        return ir


def optimal_partition(text, candidates, fits, count, target, *, first_start=None, measure=None, penalty=None):
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
    for point in points:
        interior.append(interior[-1] + 3 * candidates.get(point, 0.0) ** 4)
    for right in range(1, len(points)):
        if first_start is None:
            earliest = 0
        else:
            earliest = first_start(points[right])
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
            cost = costs[left] + 1 + 5 * (1 - score) + cohesion_loss + 0.35 * ((size - target) / target) ** 2
            if right == len(points) - 1:
                cost -= 5 * (1 - score)
            if penalty is not None:
                cost += penalty(part, score)
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
