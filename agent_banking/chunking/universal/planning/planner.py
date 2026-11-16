"""Structure-first, exact-token child chunks with a separate parent store."""
from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy
import hashlib
import os

from ..enrichment.image_context import IMAGE_POLICY, excluded_image
from ..extraction.sources import document_source_name
from ..text.language import classify_text_blocks, project_search_text, vietnamese_projection
from .coverage import validate_coverage
from .parent_context import materialize_parents, windowed_parent_content
from .source_spans import (
    build_lead_in,
    find_sentence_gap_ends as find_sentence_gap_ends,
    find_word_starts as find_word_starts,
    has_visible_text,
    is_furniture_block,
    normalize as normalize,
    section_region,
)
from .table_chunks import plan_table
from .text_chunks import flush_text
from .schema import build_universal_chunk, infer_business_metadata, link_chunk_sequence
from .table_rendering import (
    escape_html as escape_html,
    markdown_row as markdown_row,
    render_structured_html_table as render_structured_html_table,
    table_markdown as table_markdown,
)
from ..text.token_budget import TokenBudget, format_passage_fields


DEPENDENT_RELATIONS = {"continues", "answer_to", "exception_to", "note_of"}
_VISUAL_COORDINATES = {"bbox", "bbox_granularity", "coordinate_system", "region_pixel_box", "region_pixel_path"}


def _drop_visual_coordinates(blocks):
    for block in blocks:
        locations = [block["provenance"]]
        table = block.get("table_data")
        if table:
            for part in (table, *table.get("header_occurrences", ())):
                locations.extend(part.get("row_provenance", ()))
        for location in locations:
            for key in _VISUAL_COORDINATES:
                location.pop(key, None)


def _page_citation(chunk):
    metadata = chunk["business_metadata"]
    locations = [*metadata["provenance"]["locations"],
                 *(element["location"] for element in metadata.get("elements", ()))]
    pages = set()
    for location in locations:
        page_number = location.get("page_number")
        page_index = location.get("page_index")
        if type(page_number) is int and page_number > 0:
            pages.add(page_number)
        elif type(page_index) is int and page_index >= 0:
            pages.add(page_index + 1)
    citation = {"file_name": document_source_name(chunk["source"])}
    if pages:
        citation["page_numbers"] = sorted(pages)
    metadata["citation"] = citation


class ChunkPlanner:
    def __init__(self, target_tokens=512, max_tokens=1024, max_words=None, table_rows=4, tokenizer=None,
                 max_chars=None, min_words=30, min_chars=0, max_parent_words=650, *,
                 use_decision=True, decision_cache=None, min_tokens=128, search_language="vi"):
        positive = {"table_rows": table_rows}
        positive.update((name, value) for name, value in (("max_words", max_words), ("max_chars", max_chars))
                        if value is not None)
        nonnegative = {"min_words": min_words, "min_chars": min_chars, "min_tokens": min_tokens}
        if max_parent_words is not None:
            nonnegative["max_parent_words"] = max_parent_words
        for name, value in positive.items():
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        for name, value in nonnegative.items():
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if search_language not in ("vi", "all"):
            raise ValueError("search_language must be 'vi' or 'all'")
        self.budget = TokenBudget(target_tokens, max_tokens, tokenizer)
        self.max_words = max_words
        self.table_rows = table_rows
        self.max_chars = max_chars
        self.min_words = min_words
        self.min_chars = min_chars
        self.max_parent_words = max_parent_words
        self.min_tokens = min(min_tokens, target_tokens)
        self.use_decision = use_decision
        self.decision_cache = decision_cache
        self.search_language = search_language

    def plan(self, ir):
        _drop_visual_coordinates(ir["blocks"])
        self.ir = ir
        self.chunks = []
        self.parents = {}
        self.foreign_elements = []
        self.base_metadata = infer_business_metadata(ir["source"])
        self.base_metadata.update({"schema_version": "2.0", "document_id": ir["document_id"],
                                   "document_version": ir["document_version"]})
        self.block_map = {block["id"]: block for block in ir["blocks"]}
        ir["search_language"] = self.search_language
        classify_text_blocks(ir["blocks"], search_language=self.search_language)
        self.block_order = {block["id"]: index for index, block in enumerate(ir["blocks"])}
        self.relations_by_block = defaultdict(list)
        self.caption_targets = defaultdict(set)
        from ..enrichment.decisions.engine import decide_prompt_pairs
        self.dependent_pairs = set()
        if self.use_decision:
            candidates = [block if not (block.get("foreign_language") or block.get("foreign_spans"))
                          else {**block, "type": "foreign_language"} for block in ir["blocks"]]
            self.dependent_pairs.update(decide_prompt_pairs(candidates, cache=self.decision_cache))
        for index, relation in enumerate(ir["relations"]):
            self.relations_by_block[relation["from"]].append(index)
            if relation["from"] != relation["to"]:
                self.relations_by_block[relation["to"]].append(index)
            if relation["type"] in DEPENDENT_RELATIONS:
                self.dependent_pairs.add(frozenset((relation["from"], relation["to"])))
            if relation["type"] == "caption_of":
                self.caption_targets[relation["from"]].add(relation["to"])
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
        blocks_list = ir["blocks"]
        idx = 0
        while idx < len(blocks_list):
            block = blocks_list[idx]
            idx += 1
            if is_furniture_block(block):
                block["section_id"] = root_id
                continue
            region = section_region(block)
            if region != current_region:
                flush_text(self, pending, current)
                stack = []
                current = root_id
                current_region = region
            if block["type"] in ("heading", "paragraph", "caption") and not has_visible_text(block["text"]):
                block["section_id"] = current
                if block["text"]:
                    self._keep_foreign(block, 0, len(block["text"]), None)
                continue
            if block["type"] == "heading":
                level = min(7, block.get("level", 1))
                self._finish_section(pending, current)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                parent = root_id
                if stack:
                    parent = stack[-1][1]
                current = section_prefix + block["id"].removeprefix(revision_prefix)
                section_title = vietnamese_projection(block).strip()
                breadcrumb = " > ".join(part for part in [title, *(item[2] for item in stack), section_title] if part)
                self.parents[current] = {"parent_id": parent, "section_title": section_title,
                                         "breadcrumb": breadcrumb, "content": "", "block_ids": []}
                stack.append((level, current, section_title))
            self.parents[current]["block_ids"].append(block["id"])
            block["section_id"] = current
            if block.get("foreign_language"):
                self._keep_foreign(block, 0, len(block["text"]), block["foreign_language"])
            elif block.get("foreign_spans"):
                pending.extend(self._mixed_language_parts(block))
            elif block["type"] in ("heading", "paragraph", "caption"):
                pending.append(block)
            elif block["type"] == "table":
                self._flush_conflicting_captions(pending, current, block["id"])
                pending_words = sum(len(item["text"].split()) for item in pending)
                if pending and (pending_words < self.min_words or all(item["type"] in ("heading", "caption") for item in pending)):
                    plan_table(self, block, current, lead_in_blocks=pending)
                else:
                    flush_text(self, pending, current)
                    plan_table(self, block, current)
                trailing_blocks = self._following_captions(idx, block["id"], current, current_region)
                idx += len(trailing_blocks)
                if trailing_blocks:
                    self._try_append_tail_to_last_chunk(trailing_blocks)
                    flush_text(self, trailing_blocks, current)
            elif block["type"] == "image":
                if excluded_image(block):
                    continue
                self._flush_conflicting_captions(pending, current, block["id"])
                trailing_blocks = self._following_captions(idx, block["id"], current, current_region, allow_unlinked=True)
                idx += len(trailing_blocks)
                pending_words = sum(len(item["text"].split()) for item in pending)
                bundle_prev = block.get("image", {}).get("bundle_preceding", False)
                if pending and (bundle_prev or pending_words < self.min_words or all(item["type"] in ("heading", "caption") for item in pending)):
                    self._image(block, current, lead_in_blocks=pending, trailing_blocks=trailing_blocks)
                else:
                    flush_text(self, pending, current)
                    self._image(block, current, trailing_blocks=trailing_blocks)
        self._finish_section(pending, current)
        self._attach_foreign(root_id)
        materialize_parents(self)
        for chunk in self.chunks:
            chunk["parent_content"] = windowed_parent_content(self, chunk, self.parents[chunk["parent_id"]])
            count = self.budget.passage_count(chunk)
            if count > self.budget.maximum:
                raise AssertionError("Oversized passage escaped planner")
            chunk["business_metadata"]["retrieval"]["token_count"] = count
            provenance = chunk["business_metadata"]["provenance"]
            indices = {index for identifier in provenance["block_ids"] for index in self.relations_by_block[identifier]}
            provenance["relations"] = [ir["relations"][index] for index in sorted(indices)]
        link_chunk_sequence(self.chunks)
        quality = validate_coverage(self)
        for chunk in self.chunks:
            _page_citation(chunk)
        return self.chunks, {"parents": self.parents, "document": ir, "quality": quality}

    def _caption_conflicts(self, block, target):
        targets = self.caption_targets.get(block["id"])
        return bool(targets and target not in targets)

    def _flush_conflicting_captions(self, pending, parent_id, target):
        for index in range(len(pending) - 1, -1, -1):
            if self._caption_conflicts(pending[index], target):
                flush_text(self, pending[:index + 1], parent_id)
                del pending[:index + 1]
                break

    def _following_captions(self, index, target, parent_id, region, *, allow_unlinked=False):
        captions = []
        blocks = self.ir["blocks"]
        while index < len(blocks):
            candidate = blocks[index]
            targets = self.caption_targets.get(candidate["id"])
            if candidate["type"] not in ("caption", "paragraph"):
                break
            if not targets and (not allow_unlinked or candidate["type"] != "caption"):
                break
            if (self._caption_conflicts(candidate, target) or is_furniture_block(candidate)
                    or section_region(candidate) != region or not has_visible_text(candidate["text"])
                    or candidate.get("foreign_language") or candidate.get("foreign_spans")):
                break
            self.parents[parent_id]["block_ids"].append(candidate["id"])
            candidate["section_id"] = parent_id
            captions.append(candidate)
            index += 1
            if not targets:
                break
        return captions

    def _keep_foreign(self, block, start, end, language):
        kind = "source_spacing" if language is None else "foreign_language"
        self.foreign_elements.append({
            "type": kind, "language": language, "original_type": block["type"],
            "block_id": block["id"], "source_span": [start, end], "text": block["text"][start:end],
            "location": deepcopy(block["provenance"]),
        })

    def _mixed_language_parts(self, block):
        start = 0
        for left, right, language in [*block["foreign_spans"], (len(block["text"]), len(block["text"]), None)]:
            if start < left:
                part = {**block, "text": block["text"][start:left], "source_start": start}
                if has_visible_text(part["text"]):
                    yield part
                else:
                    self._keep_foreign(block, start, left, None)
            if left < right:
                self._keep_foreign(block, left, right, language)
            start = right

    def _attach_foreign(self, root_id):
        if not self.foreign_elements:
            return
        if not self.chunks:
            carrier = self._prototype("text", "", root_id, [])
            carrier["business_metadata"]["retrieval"]["role"] = "attachment_only"
            self._emit(carrier)
        positions = []
        by_section = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            if chunk["business_metadata"]["retrieval"].get("role") == "attachment_only":
                continue
            provenance = chunk["business_metadata"]["provenance"]
            anchors = set()
            for span in provenance.get("source_spans", []):
                anchors.add((self.block_order[span["block_id"]], span["start"], index))
                anchors.add((self.block_order[span["block_id"]], span["end"], index))
            for identifier in provenance["block_ids"]:
                anchors.add((self.block_order[identifier], 0, index))
            positions.extend(anchors)
            by_section[chunk["parent_id"]].extend(anchors)
        positions.sort()
        for anchors in by_section.values():
            anchors.sort()
        for element in self.foreign_elements:
            block = self.block_map[element["block_id"]]
            order = self.block_order[block["id"]]
            offset = element["source_span"][0]
            # Position wins within a logical section. A standalone foreign
            # section borrows the closest available chunk in its document.
            anchors = by_section.get(block["section_id"]) or positions
            cursor = bisect_left(anchors, (order, offset, -1))
            adjacent = anchors[max(0, cursor - 1):cursor + 1]
            index = 0
            if adjacent:
                index = min(adjacent, key=lambda item: (abs(item[0] - order),
                            abs(item[1] - offset) if item[0] == order else 0, item[2]))[2]
            self.chunks[index]["business_metadata"].setdefault("elements", []).append(element)

    def _finish_section(self, pending, parent_id):
        if (pending and self.chunks and self.chunks[-1]["parent_id"] == parent_id
                and sum(len(item["text"].split()) for item in pending) < self.min_words):
            self._try_append_tail_to_last_chunk(pending)
        flush_text(self, pending, parent_id)

    def _try_append_tail_to_last_chunk(self, pending):
        lead_text, lead_spans, lead_ids = build_lead_in(pending)
        if not lead_text:
            pending.clear()
            return
        last = self.chunks[-1]
        if last["chunk_type"] not in ("text", "table", "multimodal"):
            return
        prov = last["business_metadata"]["provenance"]
        target = (last["table_data"] or {}).get("table_id") or prov.get("image_block_id")
        if target and any(self._caption_conflicts(block, target) for block in pending):
            return
        if section_region(self.block_map[prov["block_ids"][-1]]) != section_region(pending[0]):
            return
        retrieval = last["business_metadata"]["retrieval"]
        if "passage_content" in retrieval:
            return
        combined = last["content"] + "\n\n" + lead_text
        measured_retrieval = retrieval
        if "embedding_content" in retrieval:
            measured_retrieval = {**retrieval, "embedding_content": retrieval["embedding_content"] + "\n\n" + lead_text}
        if self._measure(combined, measured_retrieval) is None:
            return
        base_offset = len(last["content"]) + 2
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
        measured_retrieval["role"] = "child"
        last["business_metadata"]["retrieval"] = measured_retrieval
        seed = self.ir["document_id"] + self.ir["document_version"] + str(len(self.chunks) - 1) + last["content"]
        last["chunk_id"] = self.ir["document_id"] + ":c:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        pending.clear()

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
                     "tokenizer_sha256": self.budget.tokenizer_sha256, "max_tokens": self.budget.maximum}
        if evidence:
            limit = min(48, self.budget.maximum // 6)
            preview = [(item, project_search_text(item["text"][:1200], search_language=self.search_language).strip())
                       for item in evidence]
            used = [(item, value) for item, value in preview if value]
            if used:
                retrieval["situated_context"] = self.budget.shorten(" ".join(value for _, value in used), limit)
                retrieval["context_evidence"] = [item for item, _ in used]
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
        metadata["quality"] = {"profile": self.ir.get("profile", "accuracy"), "warnings": self.ir.get("warnings", []),
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
        if self.max_words is not None and len(content.split()) > self.max_words:
            return None
        count = self.budget.count(format_passage_fields(content, retrieval))
        if count <= limit:
            return count
        return None

    def _emit(self, chunk):
        seed = self.ir["document_id"] + self.ir["document_version"] + str(len(self.chunks)) + chunk["content"]
        chunk["chunk_id"] = self.ir["document_id"] + ":c:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        self.chunks.append(chunk)

    def _bound_passage(self, chunk, preferred=None):
        self.budget.embedding_excerpt(chunk["content"], chunk["business_metadata"]["retrieval"], preferred)

    def _image(self, block, parent_id, lead_in_blocks=None, trailing_blocks=None):
        if excluded_image(block):
            return
        image = deepcopy(block["image"])
        if image.get("status") != "text_context_supported" or not image.get("context_evidence"):
            raise ValueError("Image has no grounded text-context decision")
        text = block.get("text", "")
        if not has_visible_text(text):
            raise ValueError("Supported image requires literal source text")
        retrieval = self._retrieval(parent_id, block.get("context_quotes", []))
        if image.get("title") and not retrieval.get("situated_context"):
            title = project_search_text(image["title"], search_language=self.search_language).strip()
            if title:
                retrieval["situated_context"] = self.budget.shorten(title, min(48, self.budget.maximum // 6))

        def fits(fragment):
            return self._measure(fragment, retrieval) is not None

        def emit(content, source_spans, block_ids):
            image_span = next(span for span in source_spans if span["block_id"] == block["id"])
            chunk = self._prototype("multimodal", content, parent_id, block_ids, images=[image], retrieval=retrieval)
            chunk["business_metadata"]["provenance"].update({
                "source_spans": source_spans,
                "image_block_id": block["id"],
                "image_text_span": [image_span["start"], image_span["end"]],
                "image_chunk_span": [image_span["chunk_start"], image_span["chunk_end"]],
                "method": "native_surrounding_text",
                "context_evidence": image["context_evidence"],
            })
            chunk_retrieval = chunk["business_metadata"]["retrieval"]
            chunk_retrieval["representation"] = "image_text_context"
            projected = project_search_text(content, search_language=self.search_language)
            if projected != content:
                chunk_retrieval["embedding_content"] = projected
            if not has_visible_text(projected):
                chunk_retrieval["role"] = "attachment_only"
            self._bound_passage(chunk)
            self._emit(chunk)

        lead = lead_in_blocks if lead_in_blocks is not None else []
        trail = trailing_blocks if trailing_blocks is not None else []
        combined, spans, identifiers = build_lead_in([*lead, block, *trail])
        if fits(combined):
            emit(combined, spans, identifiers)
            lead.clear()
            trail.clear()
            return
        if lead:
            flush_text(self, lead, parent_id)
            combined, spans, identifiers = build_lead_in([block, *trail])
            if fits(combined):
                emit(combined, spans, identifiers)
                trail.clear()
                return
        for start, end, fragment in self.budget.split(text, fits, align_words=True, allow_oversized=True):
            emit(fragment, [{"block_id": block["id"], "start": start, "end": end,
                             "chunk_start": 0, "chunk_end": len(fragment)}], [block["id"]])
        flush_text(self, trail, parent_id)
