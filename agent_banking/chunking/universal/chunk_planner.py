"""Structure-first, exact-token child chunks with a separate parent store."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import os

from .schema_assembler import build_universal_chunk, infer_business_metadata, link_chunk_sequence
from .token_budget import TokenBudget, TOKENIZER_SHA256
from .image_context import IMAGE_POLICY, excluded_image
from .passage import format_passage_fields
from .semantic_boundaries import atoms_for, optimal_partition


DEPENDENT_RELATIONS = {"continues", "answer_to", "exception_to", "note_of"}
BULLET_CHARS = ("•", "●", "▪", "▫", "◦", "‣", "⁃", "·", "–", "—", "-")


def is_furniture_block(block):
    region = block.get("provenance", {}).get("region", "body")
    return str(region).startswith("furniture:")


def build_lead_in(blocks):
    non_empty = [b for b in (blocks or []) if b.get("text", "").strip()]
    if not non_empty:
        return "", [], []
    parts = []
    spans = []
    ids = []
    offset = 0
    for b in non_empty:
        txt = b["text"]
        b_len = len(txt)
        parts.append(txt)
        ids.append(b["id"])
        spans.append({
            "block_id": b["id"],
            "start": 0,
            "end": b_len,
            "chunk_start": offset,
            "chunk_end": offset + b_len,
        })
        offset += b_len + 2
    return "\n\n".join(parts), spans, ids


def is_bullet_item(text):
    s = text.lstrip()
    for b in BULLET_CHARS:
        if s.startswith(b):
            rest = s[len(b):]
            if rest and rest[0].isspace():
                return True
    return False


def find_word_starts(text):
    starts = []
    in_word = False
    for i, ch in enumerate(text):
        if not ch.isspace():
            if not in_word:
                starts.append(i)
                in_word = True
        else:
            in_word = False
    return starts


def find_sentence_gap_ends(text, use_decision_for_dots=True):
    from .decision_engine import decide_period_boundaries
    ends = []
    ambiguous_dots = []
    i = 0
    punctuation = ('.', '!', '?', '…', ';')
    closing = ('"', '”', '’', '»', ')', ']')
    n = len(text)
    while i < n:
        if text[i] in ('\r', '\n'):
            while i < n and text[i] in ('\r', '\n'):
                i += 1
            ends.append(i)
        elif text[i] in punctuation:
            dot_index = i
            punct_char = text[i]
            word_start = i
            while word_start > 0 and not text[word_start - 1].isspace():
                word_start -= 1
            raw_token = text[word_start:i + 1]
            i += 1
            while i < n and text[i] in closing:
                i += 1
            if i < n and text[i] in (' ', '\t', '\r', '\n'):
                while i < n and text[i] in (' ', '\t', '\r', '\n'):
                    i += 1
                if i < n and (text[i].isalpha() or text[i].isdigit() or text[i] in BULLET_CHARS):
                    if punct_char == '.' and len(raw_token) <= 6:
                        if use_decision_for_dots:
                            ambiguous_dots.append((dot_index, i, raw_token))
                    else:
                        ends.append(i)
        else:
            i += 1
    if ambiguous_dots:
        valid_dot_breaks = decide_period_boundaries(text, ambiguous_dots)
        for gap_end in valid_dot_breaks:
            ends.append(gap_end)
        ends.sort()
    return ends


def normalize(text):
    return " ".join(str(text).split())


def markdown_row(row):
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in row) + " |"


def table_markdown(headers, rows):
    return "\n".join([markdown_row(headers), markdown_row(["---"] * len(headers))]
                     + [markdown_row(row) for row in rows])


class ChunkPlanner:
    def __init__(self, target_tokens=400, max_tokens=512, max_words=350, table_rows=4, tokenizer=None,
                 max_chars=None, min_words=30, min_chars=0, max_parent_words=650):
        self.budget = TokenBudget(target_tokens, max_tokens, tokenizer)
        if max_words < 1 or table_rows < 1:
            raise ValueError("Chunk words and table rows must be positive")
        self.max_words = max_words
        self.table_rows = table_rows
        self.max_chars = max_chars
        self.min_words = min_words
        self.min_chars = min_chars
        self.max_parent_words = max_parent_words
        if max_chars is not None and max_chars < 1:
            raise ValueError("Character ceiling must be positive")

    def plan(self, ir):
        self.ir = ir
        self.chunks = []
        self.parents = {}
        self.base_metadata = infer_business_metadata(ir["source"])
        self.base_metadata.update({"schema_version": "2.0", "document_id": ir["document_id"],
                                   "document_version": ir["document_version"]})
        self.block_map = {block["id"]: block for block in ir["blocks"]}
        self.relations_by_block = defaultdict(list)
        from .decision_engine import decide_prompt_pairs
        self.dependent_pairs = set(decide_prompt_pairs(ir["blocks"]))
        for index, relation in enumerate(ir["relations"]):
            self.relations_by_block[relation["from"]].append(index)
            if relation["from"] != relation["to"]:
                self.relations_by_block[relation["to"]].append(index)
            if relation["type"] in DEPENDENT_RELATIONS:
                self.dependent_pairs.add(frozenset((relation["from"], relation["to"])))
        self.boundaries_by_block = defaultdict(list)
        for boundary in ir.get("semantic_boundaries", []):
            self.boundaries_by_block[boundary["block_id"]].append(boundary)
        stack = []
        revision_prefix = ir["document_id"] + ":v" + ir["document_version"][:16] + ":"
        section_prefix = revision_prefix + "section:"
        root_id = section_prefix + "root"
        title = self.base_metadata["document_name"]
        self.parents[root_id] = {"parent_id": "", "section_title": title, "breadcrumb": title, "content": "", "block_ids": []}
        current = root_id
        self.document_context = self.budget.shorten(title, min(40, self.budget.maximum // 6))
        pending = []
        current_region = "body"
        for block in ir["blocks"]:
            if is_furniture_block(block):
                block["section_id"] = root_id
                continue
            region = block.get("provenance", {}).get("region", "body").split(":", 1)[0]
            if region != current_region:
                self._flush(pending, current)
                stack = []
                current = root_id
                current_region = region
            if block["type"] == "heading":
                level = block.get("level", 1)
                sibling_or_ancestor = stack and stack[-1][0] >= level
                has_body_in_pending = any(item["type"] != "heading" for item in pending)
                pending_words = sum(len(item["text"].split()) for item in pending)
                if pending and has_body_in_pending:
                    if pending_words < self.min_words:
                        if self.chunks and pending[0]["type"] != "heading":
                            self._try_append_tail_to_last_chunk(pending)
                    else:
                        self._flush(pending, current)
                elif pending and sibling_or_ancestor and pending_words >= self.min_words:
                    self._flush(pending, current)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                parent = root_id
                if stack:
                    parent = stack[-1][1]
                current = section_prefix + block["id"].removeprefix(revision_prefix)
                breadcrumb = title + " > " + " > ".join([item[2] for item in stack] + [block["text"].strip()])
                self.parents[current] = {"parent_id": parent, "section_title": block["text"].strip(),
                                         "breadcrumb": breadcrumb, "content": "", "block_ids": []}
                stack.append((level, current, block["text"].strip()))
            self.parents[current]["block_ids"].append(block["id"])
            block["section_id"] = current
            if block["type"] in ("heading", "paragraph", "caption"):
                pending.append(block)
            elif block["type"] == "table":
                pending_words = sum(len(item["text"].split()) for item in pending)
                if pending and (pending_words < self.min_words or all(item["type"] in ("heading", "caption") for item in pending)):
                    self._table(block, current, lead_in_blocks=pending)
                else:
                    self._flush(pending, current)
                    self._table(block, current)
            elif block["type"] == "image":
                if excluded_image(block, self.ir.get("profile")):
                    continue
                pending_words = sum(len(item["text"].split()) for item in pending)
                if pending and (pending_words < self.min_words or all(item["type"] in ("heading", "caption") for item in pending)):
                    self._image(block, current, lead_in_blocks=pending)
                else:
                    self._flush(pending, current)
                    self._image(block, current)
        if pending and sum(len(item["text"].split()) for item in pending) < self.min_words and self.chunks:
            self._try_append_tail_to_last_chunk(pending)
        self._flush(pending, current)
        children_by_parent = defaultdict(list)
        chunks_by_parent = defaultdict(list)
        for identifier, parent in self.parents.items():
            children_by_parent[parent["parent_id"]].append(identifier)
        for chunk in self.chunks:
            chunks_by_parent[chunk["parent_id"]].append(chunk["chunk_id"])
        # Materialize full section context; it is never embedded as a parent.
        for parent_id, parent in self.parents.items():
            parent["children"] = children_by_parent[parent_id]
            parent["child_chunk_ids"] = chunks_by_parent[parent_id]
            pieces = []
            for block_id in parent["block_ids"]:
                block = self.block_map[block_id]
                if block["type"] == "table":
                    table = block["table_data"]
                    pieces.append(table_markdown(table["headers"], table["rows"]))
                else:
                    pieces.append(block.get("text", "") or block.get("image", {}).get("caption", ""))
            parent["content"] = "\n\n".join(piece for piece in pieces if piece)
        for chunk in self.chunks:
            chunk["parent_content"] = self._windowed_parent_content(chunk, self.parents[chunk["parent_id"]])
            count = self.budget.passage_count(chunk)
            if count > self.budget.maximum:
                raise AssertionError("Oversized passage escaped planner")
            chunk["business_metadata"]["retrieval"]["token_count"] = count
            provenance = chunk["business_metadata"]["provenance"]
            indices = {index for identifier in provenance["block_ids"] for index in self.relations_by_block[identifier]}
            provenance["relations"] = [ir["relations"][index] for index in sorted(indices)]
        link_chunk_sequence(self.chunks)
        quality = self._validate_coverage()
        return self.chunks, {"parents": self.parents, "document": ir, "quality": quality}

    def _try_append_tail_to_last_chunk(self, pending):
        lead_text, lead_spans, lead_ids = build_lead_in(pending)
        if not lead_text:
            pending.clear()
            return
        last = self.chunks[-1]
        if last["chunk_type"] not in ("text", "table", "multimodal"):
            return
        retrieval = last["business_metadata"]["retrieval"]
        combined = last["content"] + "\n\n" + lead_text
        if self._measure(combined, retrieval) is None:
            return
        base_offset = len(last["content"]) + 2
        prov = last["business_metadata"]["provenance"]
        for span in lead_spans:
            prov.setdefault("source_spans", []).append({
                "block_id": span["block_id"],
                "start": span["start"],
                "end": span["end"],
                "chunk_start": base_offset + span["chunk_start"],
                "chunk_end": base_offset + span["chunk_end"],
            })
        prov["block_ids"].extend(lead_ids)
        prov["locations"].extend(self.block_map[identifier]["provenance"] for identifier in lead_ids)
        last["content"] = combined
        seed = self.ir["document_id"] + self.ir["document_version"] + str(len(self.chunks) - 1) + last["content"]
        last["chunk_id"] = self.ir["document_id"] + ":c:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        pending.clear()

    def _windowed_parent_content(self, chunk, parent):
        full_content = parent.get("content", "")
        if not self.max_parent_words or len(full_content.split()) <= self.max_parent_words:
            return full_content
        block_ids = parent.get("block_ids", [])
        if not block_ids:
            return " ".join(full_content.split()[:self.max_parent_words])
        piece_by_id = {}
        words_by_id = {}
        for bid in block_ids:
            block = self.block_map[bid]
            if block["type"] == "table":
                table = block["table_data"]
                piece = table_markdown(table["headers"], table["rows"])
            else:
                piece = block.get("text", "") or block.get("image", {}).get("caption", "")
            piece_by_id[bid] = piece
            words_by_id[bid] = len(piece.split())
        chunk_bids = set(chunk["business_metadata"]["provenance"].get("block_ids", []))
        anchor_indices = [idx for idx, bid in enumerate(block_ids) if bid in chunk_bids]
        if not anchor_indices:
            anchor_indices = [0]
        left = min(anchor_indices)
        right = max(anchor_indices)
        header_bid = None
        if block_ids and self.block_map[block_ids[0]]["type"] == "heading" and left > 0:
            header_bid = block_ids[0]
        budget = self.max_parent_words
        if header_bid is not None:
            budget -= words_by_id[header_bid]
        current_words = sum(words_by_id[block_ids[idx]] for idx in range(left, right + 1))
        while current_words < budget and (left > 0 or right + 1 < len(block_ids)):
            expanded = False
            if left > 0 and block_ids[left - 1] != header_bid:
                w = words_by_id[block_ids[left - 1]]
                if current_words + w <= budget:
                    left -= 1
                    current_words += w
                    expanded = True
            if right + 1 < len(block_ids):
                w = words_by_id[block_ids[right + 1]]
                if current_words + w <= budget:
                    right += 1
                    current_words += w
                    expanded = True
            if not expanded:
                break
        selected_pieces = []
        if header_bid is not None and piece_by_id[header_bid]:
            selected_pieces.append(piece_by_id[header_bid])
        for idx in range(left, right + 1):
            piece = piece_by_id[block_ids[idx]]
            if piece:
                selected_pieces.append(piece)
        window_text = "\n\n".join(selected_pieces)
        window_words = window_text.split()
        if len(window_words) > self.max_parent_words:
            if chunk["chunk_type"] == "table" and chunk.get("table_data"):
                header_prefix = ""
                if header_bid is not None and piece_by_id[header_bid]:
                    header_prefix = piece_by_id[header_bid] + "\n\n"
                return (header_prefix + chunk["content"]).strip()
            return " ".join(window_words[:self.max_parent_words])
        return window_text

    def _retrieval(self, parent_id, evidence, kind="text"):
        parent = self.parents[parent_id]
        context = parent.get("retrieval_context")
        if context is None:
            section_limit = min(44, self.budget.maximum // 6)
            document = self.document_context
            sections = []
            cursor = parent
            while cursor["parent_id"]:
                parent_node = self.parents.get(cursor["parent_id"])
                is_root_child = parent_node is not None and not parent_node.get("parent_id")
                if is_root_child and len(sections) >= 2:
                    break
                step_limit = 16
                if not sections:
                    step_limit = 22
                if step_limit > section_limit:
                    step_limit = section_limit
                title = self.budget.shorten(cursor["section_title"], step_limit)
                if title and title != document:
                    sections.append(title)
                    section_limit -= self.budget.count(title) + 2
                if section_limit < 6:
                    break
                cursor = self.parents[cursor["parent_id"]]
            context = " > ".join(part for part in [document, *reversed(sections)] if part)
            parent["retrieval_context"] = context
        if kind == "table":
            context = self.budget.shorten(context, 44)
        else:
            context = self.budget.shorten(context, 56)
        tokenizer_name = os.getenv("EMBEDDING_MODEL", "embedding-tokenizer")
        retrieval = {"role": "child", "context": context, "tokenizer": tokenizer_name,
                     "tokenizer_sha256": TOKENIZER_SHA256, "max_tokens": self.budget.maximum}
        if evidence:
            limit = min(48, self.budget.maximum // 6)
            retrieval["situated_context"] = self.budget.shorten(" ".join(item["text"] for item in evidence), limit)
            retrieval["context_evidence"] = evidence
        return retrieval

    def _prototype(self, kind, content, parent_id, block_ids, table_data=None, images=None, retrieval=None):
        metadata = deepcopy(self.base_metadata)
        parent = self.parents[parent_id]
        if retrieval is None:
            evidence = [item for identifier in block_ids for item in self.block_map[identifier].get("context_quotes", [])]
            retrieval = self._retrieval(parent_id, evidence, kind)
        metadata["retrieval"] = deepcopy(retrieval)
        metadata["provenance"] = {"block_ids": list(block_ids), "locations": [self.block_map[identifier]["provenance"] for identifier in block_ids],
                                   "method": self.ir.get("extraction_method", "docx_openxml"), "source_view": "final"}
        metadata["quality"] = {"profile": self.ir.get("profile", "native"), "warnings": self.ir.get("warnings", []),
                               "source_review": self.ir.get("source_review", []), "image_policy": self.ir.get("image_policy", IMAGE_POLICY),
                               "decision_route": self.ir.get("decision_review", {}).get("answers", {}).get("review_route", {}).get("choice")}
        return build_universal_chunk(source=self.ir["source"], chunk_type=kind,
                                     parent_id=parent_id, breadcrumb=parent["breadcrumb"],
                                     section_title=parent["section_title"], content=content,
                                     # Published cell/header evidence must not alias the
                                     # authoritative IR or sibling chunks in memory.
                                     business_metadata=metadata, table_data=deepcopy(table_data), images=images)

    def _measure(self, content, retrieval, target=False):
        limit = self.budget.maximum
        if target:
            limit = self.budget.target
        if self.max_chars is not None and len(content) > self.max_chars:
            return None
        if len(content.split()) > self.max_words:
            return None
        count = self.budget.count(format_passage_fields(content, retrieval))
        if count <= limit:
            return count
        return None

    def _emit(self, chunk):
        seed = self.ir["document_id"] + self.ir["document_version"] + str(len(self.chunks)) + chunk["content"]
        chunk["chunk_id"] = self.ir["document_id"] + ":c:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        self.chunks.append(chunk)

    def _flush(self, blocks, parent_id):
        if not blocks:
            return
        text = "\n\n".join(block["text"] for block in blocks)
        if not text.strip():
            blocks.clear()
            return
        starts = []
        ends = []
        offset = 0
        for block in blocks:
            starts.append(offset)
            ends.append(offset + len(block["text"]))
            offset = ends[-1] + 2
        starts_by_id = {block["id"]: start for block, start in zip(blocks, starts)}

        candidates = {}
        for atom in atoms_for(blocks, self.budget):
            point = starts_by_id[atom["block_id"]] + atom["end"]
            aligned = point == len(text) or text[point - 1].isspace() or text[point].isspace()
            if aligned:
                candidates[point] = 0.18
            else:
                candidates[point] = 0.0
        for block, start, end in zip(blocks, starts, ends):
            candidates[end] = max(candidates.get(end, 0.0), 0.72)
            needs_dot_decision = len(block["text"].split()) > 60
            for gap_end in find_sentence_gap_ends(block["text"], use_decision_for_dots=needs_dot_decision):
                gap_point = start + gap_end
                candidates[gap_point] = max(candidates.get(gap_point, 0.0), 0.62)
            for boundary in self.boundaries_by_block.get(block["id"], []):
                point = start + boundary["offset"]
                candidates[point] = boundary["score"]
        for index, block in enumerate(blocks[:-1]):
            next_block = blocks[index + 1]
            dependent = frozenset((block["id"], next_block["id"])) in self.dependent_pairs
            if dependent or block["type"] == "heading" or block["text"].rstrip().endswith(":"):
                candidates[ends[index]] = 0.0
                if starts[index + 1] in candidates:
                    candidates[starts[index + 1]] = 0.0

        evidence_ranges = [(start, end, block["context_quotes"]) for block, start, end in zip(blocks, starts, ends)
                           if start < end and block.get("context_quotes")]
        evidence_starts = [start for start, _, _ in evidence_ranges]
        evidence_ends = [end for _, end, _ in evidence_ranges]

        @lru_cache(maxsize=512)
        def retrieval_for_indices(first, last):
            evidence = [item for _, _, quotes in evidence_ranges[first:last] for item in quotes]
            return self._retrieval(parent_id, evidence)

        def retrieval_for_range(start, end):
            return retrieval_for_indices(bisect_right(evidence_ends, start), bisect_left(evidence_starts, end))

        def measure(start, end, part):
            return self._measure(part, retrieval_for_range(start, end))

        def fits_range(start, end):
            return measure(start, end, text[start:end]) is not None

        groups = []
        for index, block in enumerate(blocks[:-1]):
            if frozenset((block["id"], blocks[index + 1]["id"])) in self.dependent_pairs:
                if fits_range(starts[index], ends[index + 1]):
                    groups.append((starts[index], ends[index + 1]))
                continue
            if not block["text"].rstrip().endswith(":") or is_bullet_item(block["text"]):
                continue
            last = index
            while last + 1 < len(blocks) and is_bullet_item(blocks[last + 1]["text"]):
                last += 1
            if last > index and fits_range(starts[index], ends[last]):
                groups.append((starts[index], ends[last]))

        word_starts = find_word_starts(text)

        def first_start(end):
            if self.max_chars is None:
                earliest = 0
            else:
                earliest = max(0, end - self.max_chars)
            earlier_words = bisect_left(word_starts, end) - self.max_words
            if earlier_words > 0:
                earliest = max(earliest, word_starts[earlier_words - 1] + 1)
            return earliest

        def short_penalty(part, score):
            words_count = len(part.split())
            if len(part.strip()) < self.min_chars or words_count < self.min_words:
                if words_count < 15:
                    return 12.0
                return 8.0
            return 0

        # An explicit lead-in and its bullets are atomic when they fit together.
        # Generate a feasible fallback chain through the gaps around those groups.
        cursor = 0
        for start, end in [*groups, (len(text), len(text))]:
            for _, stop, _ in self.budget.split(text[cursor:start], None, align_words=True,
                                                fits_range=lambda left, right: fits_range(cursor + left, cursor + right)):
                candidates.setdefault(cursor + stop, 0.0)
            if end > start:
                for point in [point for point in candidates if start < point < end]:
                    del candidates[point]
                candidates[start] = max(0.78, candidates.get(start, 0.0))
                candidates[end] = max(0.78, candidates.get(end, 0.0))
            cursor = end
        ranges = optimal_partition(text, candidates, None, None, self.budget.target,
                                   first_start=first_start, measure=measure, penalty=short_penalty)
        self._emit_text(blocks, parent_id, starts, ends, ranges, retrieval_for_range)
        blocks.clear()

    def _emit_text(self, blocks, parent_id, starts, ends, ranges, retrieval_for_range):
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
                    spans.append({"block_id": blocks[index]["id"], "start": left - offset, "end": right - offset,
                                  "chunk_start": left - actual_start, "chunk_end": right - actual_start})
            if not spans:
                continue
            ids = [span["block_id"] for span in spans]
            chunk = self._prototype("text", part, parent_id, ids, retrieval=retrieval_for_range(actual_start, actual_end))
            chunk["business_metadata"]["provenance"].update(group_span=[actual_start, actual_end], source_spans=spans)
            self._emit(chunk)

    def _table(self, block, parent_id, lead_in_blocks=None):
        table = block["table_data"]
        headers = table["headers"]
        rows = table["rows"]
        pending = []
        header_count = table["header_row_count"]
        header_grid = table["grid"][:header_count]
        header_cells = [cell for cell in table["cells"] if cell["row"] < header_count]
        start_row = header_count
        ids = [block["id"]]
        retrieval = self._retrieval(parent_id, block.get("context_quotes", []), "table")
        lead_text, lead_spans, lead_ids = build_lead_in(lead_in_blocks)
        lead_attached = not bool(lead_text)
        if not lead_text and lead_in_blocks is not None:
            lead_in_blocks.clear()

        def fits(content, target=False):
            return self._measure(content, retrieval, target) is not None

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

        def make(selected, row_start):
            data = {"table_id": table["table_id"], "headers": headers, "rows": selected,
                    "row_start": row_start, "row_end": row_start + len(selected),
                    "header_row_count": header_count, "header_grid": header_grid, "header_cells": header_cells,
                    "cells": [cell for cell in table["cells"] if cell["row"] < row_start + len(selected)
                              and cell["row"] + cell.get("row_span", 1) > row_start]}
            chunk_ids = ids
            if not lead_attached and lead_text:
                chunk_ids = lead_ids + ids
            chunk = self._prototype("table", with_lead(table_markdown(headers, selected)), parent_id, chunk_ids,
                                    table_data=data, retrieval=retrieval)
            return attach_lead_metadata(chunk)

        for index, row in enumerate(rows, start=header_count):
            if pending:
                final_row = index == header_count + len(rows) - 1
                if len(pending) < self.table_rows and fits(with_lead(table_markdown(headers, pending + [row])), target=not final_row):
                    pending.append(row)
                    continue
                self._emit(make(pending, start_row))
                pending = []
            start_row = index
            if not lead_attached and lead_text and not fits(with_lead(table_markdown(headers, [row]))):
                self._flush(lead_in_blocks, parent_id)
                lead_text = ""
                lead_spans = []
                lead_ids = []
                lead_attached = True
            if fits(with_lead(table_markdown(headers, [row]))):
                pending.append(row)
            else:
                # Wide rows / long cells: preserve each value and its exact span,
                # retain original structured cell and table ID for reconstruction.
                label_limit = min(36, self.budget.maximum // 8)
                row_key = self.budget.shorten(str(row[0]), label_limit)
                for column, value in enumerate(row):
                    label = self.budget.shorten(headers[column], label_limit)
                    prefix = "Row " + str(index + 1)
                    if row_key:
                        prefix += " [" + row_key + "]"
                    prefix += " — " + (label or "column " + str(column + 1)) + ":\n"
                    if not fits(prefix + value[:1]):
                        prefix = ""
                    pieces = self.budget.split(value, lambda fragment: fits(prefix + fragment), align_words=True)
                    if not pieces:
                        pieces = [(0, 0, "")]
                    for begin, end, fragment in pieces:
                        data = {"table_id": table["table_id"], "headers": [headers[column]],
                                "rows": [[fragment]], "row_start": index, "row_end": index + 1,
                                "column": column, "cell_span": [begin, end], "row_key": row_key,
                                "header_grid": header_grid,
                                "source_cells": [cell for cell in table["cells"] if cell["row"] <= index < cell["row"] + cell.get("row_span", 1)
                                                 and cell["column"] <= column < cell["column"] + cell.get("col_span", cell.get("column_span", 1))]}
                        chunk = self._prototype("table", prefix + fragment, parent_id, ids,
                                                table_data=data, retrieval=retrieval)
                        self._emit(chunk)
        if pending:
            self._emit(make(pending, start_row))
        if not rows:
            if not lead_attached and lead_text and lead_in_blocks:
                self._flush(lead_in_blocks, parent_id)
                lead_attached = True
            content = table_markdown(headers, [])
            data = {"table_id": table["table_id"], "headers": headers, "rows": [],
                    "header_row_count": header_count, "row_start": 0, "row_end": 0,
                    "header_grid": header_grid, "cells": table["cells"]}
            for begin, end, fragment in self.budget.split(content, fits):
                chunk = self._prototype("table", fragment, parent_id, ids, table_data=data, retrieval=retrieval)
                chunk["business_metadata"]["provenance"]["header_span"] = [begin, end]
                self._emit(chunk)

    def _image(self, block, parent_id, lead_in_blocks=None):
        if excluded_image(block, self.ir.get("profile")):
            return
        image = deepcopy(block["image"])
        if image.get("status") != "text_context_supported" or not image.get("context_evidence"):
            raise ValueError("Image has no grounded text-context decision")
        text = block.get("text", "")
        if not text.strip():
            raise ValueError("Supported image requires literal source text")
        retrieval = self._retrieval(parent_id, block.get("context_quotes", []))
        lead_text, lead_spans, lead_ids = build_lead_in(lead_in_blocks)
        if lead_text and self._measure(lead_text + "\n\n" + text, retrieval) is not None:
            combined = lead_text + "\n\n" + text
            chunk = self._prototype("multimodal", combined, parent_id, lead_ids + [block["id"]], images=[image], retrieval=retrieval)
            chunk["business_metadata"]["provenance"].update({
                "source_spans": deepcopy(lead_spans),
                "image_text_span": [0, len(text)],
                "method": "native_surrounding_text",
                "context_evidence": image["context_evidence"],
            })
            chunk["business_metadata"]["retrieval"]["representation"] = "image_text_context"
            self._emit(chunk)
            if lead_in_blocks is not None:
                lead_in_blocks.clear()
            return
        if lead_in_blocks:
            self._flush(lead_in_blocks, parent_id)
        def fits(fragment):
            return self._measure(fragment, retrieval) is not None
        for start, end, fragment in self.budget.split(text, fits, align_words=True):
            chunk = self._prototype("multimodal", fragment, parent_id, [block["id"]], images=[image], retrieval=retrieval)
            chunk["business_metadata"]["provenance"].update({"image_text_span": [start, end],
                                                           "method": "native_surrounding_text",
                                                           "context_evidence": image["context_evidence"]})
            chunk["business_metadata"]["retrieval"]["representation"] = "image_text_context"
            self._emit(chunk)

    def _validate_coverage(self):
        self._validate_exact_locations()
        ids = {chunk["chunk_id"] for chunk in self.chunks}
        if len(ids) != len(self.chunks):
            raise AssertionError("Duplicate chunk IDs")
        table_parts = {}
        images = set()
        image_parts = {}
        for chunk in self.chunks:
            if chunk["parent_id"] not in self.parents:
                raise AssertionError("Dangling parent ID")
            block_ids = chunk["business_metadata"]["provenance"]["block_ids"]
            if chunk["table_data"]:
                data = chunk["table_data"]
                table_parts.setdefault(data["table_id"], []).append(data)
            for image in chunk["images"]:
                images.add(image["sha256"])
                for block_id in block_ids:
                    image_parts.setdefault(block_id, []).append(chunk["content"])
        table_values = {}
        for table_id, pieces in table_parts.items():
            values = set()
            fragmented = defaultdict(list)
            for data in pieces:
                if "column" in data:
                    key = (data["row_start"], data["column"])
                    fragmented[key].append(data["rows"][0][0])
                else:
                    values.update(value for row in data["rows"] for value in row)
                values.update(data.get("headers", []))
                values.update(value for row in data.get("header_grid", []) for value in row)
            values.update("".join(parts) for parts in fragmented.values())
            table_values[table_id] = {normalize(value) for value in values}
        skipped = {block["id"]: block for block in self.ir["blocks"]
                   if block["type"] == "image" and (excluded_image(block, self.ir.get("profile")) or is_furniture_block(block))}
        excluded_units = []
        missing = []
        for unit in self.ir["units"]:
            block = self.block_map[unit["block_id"]]
            if is_furniture_block(block):
                excluded_units.append({"unit_id": unit["id"], "block_id": block["id"],
                                       "reason": "document_furniture_header_footer"})
                continue
            if block["type"] == "image":
                if block["id"] in skipped:
                    excluded_units.append({"unit_id": unit["id"], "block_id": block["id"],
                                           "reason": block["image"]["status"]})
                    continue
                if normalize(unit["text"]) not in normalize("".join(image_parts.get(block["id"], []))):
                    missing.append(unit["id"])
            elif block["type"] != "table":
                # Exact source spans have already proved reconstruction per
                # occurrence. A global substring search is order-dependent and
                # can both hide duplicate loss and falsely fail split blocks.
                if normalize(unit["text"]) not in normalize(block["text"]):
                    missing.append(unit["id"])
            else:
                values = table_values.get(block["id"], set())
                expected = normalize(unit["text"])
                if expected not in values and not any(expected in value for value in values):
                    missing.append(unit["id"])
        source_image_ids = {block["id"] for block in self.ir["blocks"] if block["type"] == "image"}
        expected_occurrences = source_image_ids - set(skipped)
        expected_assets = {self.block_map[identifier]["image"]["sha256"] for identifier in expected_occurrences}
        missing_images = expected_assets - images
        missing_occurrences = expected_occurrences - set(image_parts)
        if missing or missing_images or missing_occurrences:
            raise RuntimeError("Source coverage failed: " + str(len(missing)) + " text units / " + str(len(missing_images)) + " assets / " + str(len(missing_occurrences)) + " image occurrences")
        return {"source_units": len(self.ir["units"]), "covered_units": len(self.ir["units"]) - len(excluded_units),
                "accounted_units": len(self.ir["units"]), "policy_excluded_units": excluded_units,
                "image_policy": self.ir.get("image_policy", IMAGE_POLICY), "preserved_assets": len(self.ir["assets"]),
                "source_image_occurrences": len(source_image_ids),
                "skipped_image_occurrences": [{"block_id": identifier, "sha256": block["image"]["sha256"],
                                                "reason": block["image"].get("status", "document_furniture_header_footer")} for identifier, block in skipped.items()],
                "missing_units": [], "linked_assets": len(images), "missing_assets": [],
                "linked_image_occurrences": len(image_parts), "missing_image_occurrences": [],
                "oversized_passages": 0, "status": "verified_source_reconstruction",
                "assurance": "IR_to_chunks_exact; visual_accuracy_requires_external_ground_truth"}

    def _validate_exact_locations(self):
        """Validate occurrences and coordinates, not a global substring search.

        Identical text elsewhere must never conceal a missing paragraph or cell.
        This proves preservation of the IR, not correctness of visual extraction.
        """
        spans = {}
        tables = {}
        for chunk in self.chunks:
            for span in chunk["business_metadata"]["provenance"].get("source_spans", []):
                identifier = span["block_id"]
                source = self.block_map[identifier]["text"]
                begin, end = span["start"], span["end"]
                if not 0 <= begin < end <= len(source) or source[begin:end] != chunk["content"][span["chunk_start"]:span["chunk_end"]]:
                    raise RuntimeError("Source span mismatch")
                spans.setdefault(identifier, []).append((begin, end))
            if chunk["table_data"]:
                table = chunk["table_data"]
                tables.setdefault(table["table_id"], []).append(table)
        for block in self.ir["blocks"]:
            if is_furniture_block(block):
                continue
            if block["type"] in ("heading", "paragraph", "caption") and block["text"].strip():
                cursor = 0
                for begin, end in sorted(spans.get(block["id"], [])):
                    if begin != cursor:
                        raise RuntimeError("Missing or overlapping text occurrence: " + block["id"])
                    cursor = end
                if cursor != len(block["text"]):
                    raise RuntimeError("Missing source text occurrence: " + block["id"])
            if block["type"] != "table":
                continue
            original = block["table_data"]
            coordinates = {}
            for part in tables.get(block["id"], []):
                if part.get("header_grid") != original["grid"][:original["header_row_count"]]:
                    raise RuntimeError("Table header grid changed")
                if "column" in part:
                    key = (part["row_start"], part["column"])
                    coordinates.setdefault(key, []).append((*part["cell_span"], part["rows"][0][0]))
                else:
                    for row_index, row in enumerate(part["rows"], part["row_start"]):
                        for column, value in enumerate(row):
                            coordinates.setdefault((row_index, column), []).append((0, len(value), value))
            for row_index, row in enumerate(original["rows"], original["header_row_count"]):
                for column, value in enumerate(row):
                    cursor = 0
                    pieces = coordinates.get((row_index, column), [])
                    if not pieces:
                        raise RuntimeError("Missing table cell occurrence")
                    for begin, end, fragment in sorted(pieces):
                        if begin != cursor or end < begin or fragment != value[begin:end]:
                            raise RuntimeError("Missing, overlapping or changed table cell span")
                        cursor = end
                    if cursor != len(value):
                        raise RuntimeError("Incomplete table cell")
