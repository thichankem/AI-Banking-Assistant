"""Structure-first, exact-token child chunks with a separate parent store."""
from copy import deepcopy
import hashlib

from .schema_assembler import build_universal_chunk, infer_business_metadata, link_chunk_sequence
from .token_budget import TokenBudget, TOKENIZER_SHA256
from .image_context import IMAGE_POLICY, excluded_image


def normalize(text):
    return " ".join(str(text).split())


def markdown_row(row):
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in row) + " |"


def table_markdown(headers, rows):
    return "\n".join([markdown_row(headers), markdown_row(["---"] * len(headers))]
                     + [markdown_row(row) for row in rows])


class ChunkPlanner:
    def __init__(self, target_tokens=400, max_tokens=512, max_words=350, table_rows=4, tokenizer=None,
                 max_chars=None, min_words=30, min_chars=0):
        self.budget = TokenBudget(target_tokens, max_tokens, tokenizer)
        if max_words < 1 or table_rows < 1:
            raise ValueError("Chunk words and table rows must be positive")
        self.max_words = max_words
        self.table_rows = table_rows
        self.max_chars = max_chars
        self.min_words = min_words
        self.min_chars = min_chars
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
        stack = []
        section_prefix = ir["document_id"] + ":v" + ir["document_version"][:16] + ":section:"
        root_id = section_prefix + "root"
        title = self.base_metadata["document_name"]
        self.parents[root_id] = {"parent_id": "", "section_title": title, "breadcrumb": title, "content": "", "block_ids": []}
        current = root_id
        pending = []
        current_region = "body"
        for block in ir["blocks"]:
            region = block.get("provenance", {}).get("region", "body").split(":", 1)[0]
            if region != current_region:
                self._flush(pending, current)
                stack = []
                current = root_id
                current_region = region
            if block["type"] == "heading":
                if pending and any(item["type"] != "heading" for item in pending):
                    self._flush(pending, current)
                level = block.get("level", 1)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                parent = root_id
                if stack:
                    parent = stack[-1][1]
                current = section_prefix + block["id"].rsplit(":", 1)[-1]
                breadcrumb = title + " > " + " > ".join([item[2] for item in stack] + [block["text"].strip()])
                self.parents[current] = {"parent_id": parent, "section_title": block["text"].strip(),
                                         "breadcrumb": breadcrumb, "content": "", "block_ids": []}
                stack.append((level, current, block["text"].strip()))
            self.parents[current]["block_ids"].append(block["id"])
            block["section_id"] = current
            if block["type"] in ("heading", "paragraph", "caption"):
                pending.append(block)
            elif block["type"] == "table":
                self._flush(pending, current)
                self._table(block, current)
            elif block["type"] == "image":
                self._flush(pending, current)
                self._image(block, current)
        self._flush(pending, current)
        # Materialize full section context; it is never embedded as a parent.
        for parent_id, parent in self.parents.items():
            parent["children"] = [identifier for identifier, candidate in self.parents.items() if candidate["parent_id"] == parent_id]
            parent["child_chunk_ids"] = [chunk["chunk_id"] for chunk in self.chunks if chunk["parent_id"] == parent_id]
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
            chunk["parent_content"] = self.parents[chunk["parent_id"]]["content"]
            count = self.budget.passage_count(chunk)
            if count > self.budget.maximum:
                raise AssertionError("Oversized passage escaped planner")
            chunk["business_metadata"]["retrieval"]["token_count"] = count
            own_blocks = set(chunk["business_metadata"]["provenance"]["block_ids"])
            chunk["business_metadata"]["provenance"]["relations"] = [relation for relation in ir["relations"]
                if relation["from"] in own_blocks or relation["to"] in own_blocks]
        link_chunk_sequence(self.chunks)
        quality = self._validate_coverage()
        return self.chunks, {"parents": self.parents, "document": ir, "quality": quality}

    def _prototype(self, kind, content, parent_id, block_ids, table_data=None, images=None):
        metadata = deepcopy(self.base_metadata)
        parent = self.parents[parent_id]
        context = parent.get("retrieval_context")
        if context is None:
            document = self.budget.shorten(self.base_metadata["document_name"], 40)
            section = self.budget.shorten(parent["section_title"], 56)
            context = document
            if section != document:
                context += " > " + section
            parent["retrieval_context"] = context
        metadata["retrieval"] = {"role": "child", "context": context, "tokenizer": "universal-tokenizer",
                                  "tokenizer_sha256": TOKENIZER_SHA256, "max_tokens": self.budget.maximum}
        evidence = []
        for identifier in block_ids:
            evidence.extend(self.block_map[identifier].get("context_quotes", []))
        if evidence:
            metadata["retrieval"]["situated_context"] = self.budget.shorten(" ".join(item["text"] for item in evidence), 48)
            metadata["retrieval"]["context_evidence"] = evidence
        metadata["provenance"] = {"block_ids": list(block_ids), "locations": [self.block_map[identifier]["provenance"] for identifier in block_ids],
                                   "method": "docx_openxml", "source_view": "final"}
        metadata["quality"] = {"profile": self.ir.get("profile", "native"), "warnings": self.ir.get("warnings", []),
                               "source_review": self.ir.get("source_review", []), "image_policy": self.ir.get("image_policy", IMAGE_POLICY),
                               "decision_route": self.ir.get("decision_review", {}).get("answers", {}).get("review_route", {}).get("choice")}
        return build_universal_chunk(source=self.ir["source"], chunk_type=kind,
                                     parent_id=parent_id, breadcrumb=parent["breadcrumb"],
                                     section_title=parent["section_title"], content=content,
                                     business_metadata=metadata, table_data=table_data, images=images)

    def _fits(self, chunk, target=False):
        limit = self.budget.maximum
        if target:
            limit = self.budget.target
        within_chars = self.max_chars is None or len(chunk["content"]) <= self.max_chars
        return within_chars and self.budget.passage_count(chunk) <= limit and len(chunk["content"].split()) <= self.max_words

    def _emit(self, chunk):
        seed = self.ir["document_id"] + self.ir["document_version"] + str(len(self.chunks)) + chunk["content"]
        chunk["chunk_id"] = self.ir["document_id"] + ":c:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        self.chunks.append(chunk)

    def _flush(self, blocks, parent_id):
        if not blocks:
            return
        group = []
        group_text = ""
        for block in blocks:
            text = block["text"]
            combined = text
            if group_text:
                combined = group_text + "\n\n" + text
            ids = [item["id"] for item in group] + [block["id"]]
            proposed = self._prototype("text", combined, parent_id, ids)
            # Keep a heading/lead-in with its first body block up to the hard cap.
            protected = group_text.rstrip().endswith(":") or (group and group[-1]["type"] == "heading")
            if len(group_text.split()) < self.min_words or len(group_text) < self.min_chars:
                protected = True
            if group:
                pair = {group[-1]["id"], block["id"]}
                if any(relation["type"] in ("continues", "answer_to", "exception_to", "note_of")
                       and {relation["from"], relation["to"]} == pair for relation in self.ir["relations"]):
                    protected = True
            if group and not self._fits(proposed, target=True) and not (protected and self._fits(proposed)):
                self._emit_text(group_text, parent_id, [item["id"] for item in group])
                group = []
                group_text = ""
            if group_text:
                group_text += "\n\n"
            group_text += text
            group.append(block)
        if group:
            self._emit_text(group_text, parent_id, [item["id"] for item in group])
        blocks.clear()

    def _emit_text(self, text, parent_id, ids):
        def fits(candidate):
            return self._fits(self._prototype("text", candidate, parent_id, ids))
        for start, end, part in self.budget.split(text, fits):
            chunk = self._prototype("text", part, parent_id, ids)
            chunk["business_metadata"]["provenance"]["group_span"] = [start, end]
            spans = []
            offset = 0
            for identifier in ids:
                source = self.block_map[identifier]["text"]
                left = max(start, offset)
                right = min(end, offset + len(source))
                if left < right:
                    spans.append({"block_id": identifier, "start": left - offset, "end": right - offset,
                                  "chunk_start": left - start, "chunk_end": right - start})
                offset += len(source) + 2
            chunk["business_metadata"]["provenance"]["source_spans"] = spans
            self._emit(chunk)

    def _table(self, block, parent_id):
        table = block["table_data"]
        headers = table["headers"]
        rows = table["rows"]
        pending = []
        start_row = table["header_row_count"]

        def make(selected, row_start):
            data = {"table_id": table["table_id"], "headers": headers, "rows": selected,
                    "row_start": row_start, "row_end": row_start + len(selected),
                    "header_row_count": table["header_row_count"],
                    "cells": [cell for cell in table["cells"] if row_start <= cell["row"] < row_start + len(selected)]}
            return self._prototype("table", table_markdown(headers, selected), parent_id, [block["id"]], table_data=data)

        for index, row in enumerate(rows, start=table["header_row_count"]):
            candidate = make(pending + [row], start_row)
            if pending and (len(pending) >= self.table_rows or not self._fits(candidate, target=True)):
                self._emit(make(pending, start_row))
                pending = []
                start_row = index
            candidate = make([row], index)
            if not self._fits(candidate):
                if pending:
                    self._emit(make(pending, start_row))
                    pending = []
                # Wide rows / long cells: preserve each value and its exact span,
                # retain original structured cell and table ID for reconstruction.
                for column, value in enumerate(row):
                    label = self.budget.shorten(headers[column], 36)
                    row_key = self.budget.shorten(str(row[0]), 36)
                    prefix = "Dòng " + str(index + 1) + " [" + row_key + "] — " + label + ":\n"
                    def fits(fragment):
                        chunk = self._prototype("table", prefix + fragment, parent_id, [block["id"]])
                        return self._fits(chunk)
                    pieces = self.budget.split(str(value), fits)
                    if not pieces:
                        pieces = [(0, 0, "")]
                    for begin, end, fragment in pieces:
                        data = {"table_id": table["table_id"], "headers": [headers[column]],
                                "rows": [[fragment]], "row_start": index, "row_end": index + 1,
                                "column": column, "cell_span": [begin, end], "row_key": row_key}
                        chunk = self._prototype("table", prefix + fragment, parent_id, [block["id"]], table_data=data)
                        self._emit(chunk)
                start_row = index + 1
            else:
                if not pending:
                    start_row = index
                pending.append(row)
        if pending:
            self._emit(make(pending, start_row))
        if not rows:
            content = table_markdown(headers, [])
            data = {"table_id": table["table_id"], "headers": headers, "rows": [],
                    "header_row_count": table["header_row_count"], "row_start": 0, "row_end": 0}
            def fits(fragment):
                return self._fits(self._prototype("table", fragment, parent_id, [block["id"]], table_data=data))
            for begin, end, fragment in self.budget.split(content, fits):
                chunk = self._prototype("table", fragment, parent_id, [block["id"]], table_data=data)
                chunk["business_metadata"]["provenance"]["header_span"] = [begin, end]
                self._emit(chunk)

    def _image(self, block, parent_id):
        if excluded_image(block, self.ir.get("profile")):
            return
        image = deepcopy(block["image"])
        if image.get("status") != "text_context_supported" or not image.get("context_evidence"):
            raise ValueError("Image has no grounded text-context decision")
        text = block.get("text", "")
        if not text.strip():
            raise ValueError("Supported image requires literal source text")
        def fits(fragment):
            return self._fits(self._prototype("multimodal", fragment, parent_id, [block["id"]], images=[image]))
        for start, end, fragment in self.budget.split(text, fits):
            chunk = self._prototype("multimodal", fragment, parent_id, [block["id"]], images=[image])
            chunk["business_metadata"]["provenance"].update({"image_text_span": [start, end],
                                                           "method": "native_surrounding_text",
                                                           "context_evidence": image["context_evidence"]})
            chunk["business_metadata"]["retrieval"]["representation"] = "image_text_context"
            self._emit(chunk)

    def _validate_coverage(self):
        ids = {chunk["chunk_id"] for chunk in self.chunks}
        if len(ids) != len(self.chunks):
            raise AssertionError("Duplicate chunk IDs")
        text_parts = {}
        table_parts = {}
        images = set()
        image_parts = {}
        for chunk in self.chunks:
            if chunk["parent_id"] not in self.parents:
                raise AssertionError("Dangling parent ID")
            block_ids = chunk["business_metadata"]["provenance"]["block_ids"]
            if chunk["chunk_type"] == "text":
                key = tuple(block_ids)
                text_parts.setdefault(key, []).append(chunk["content"])
            if chunk["table_data"]:
                data = chunk["table_data"]
                table_parts.setdefault(data["table_id"], []).append(data)
            for image in chunk["images"]:
                images.add(image["sha256"])
                for block_id in block_ids:
                    image_parts.setdefault(block_id, []).append(chunk["content"])
        reconstructed = "\n\n".join("".join(parts) for parts in text_parts.values())
        skipped = {block["id"]: block for block in self.ir["blocks"]
                   if block["type"] == "image" and excluded_image(block, self.ir.get("profile"))}
        excluded_units = []
        missing = []
        for unit in self.ir["units"]:
            block = self.block_map[unit["block_id"]]
            if block["type"] == "image":
                if block["id"] in skipped:
                    excluded_units.append({"unit_id": unit["id"], "block_id": block["id"],
                                           "reason": block["image"]["status"]})
                    continue
                if normalize(unit["text"]) not in normalize("".join(image_parts.get(block["id"], []))):
                    missing.append(unit["id"])
            elif block["type"] != "table":
                if normalize(unit["text"]) not in normalize(reconstructed):
                    missing.append(unit["id"])
            else:
                pieces = table_parts.get(block["id"], [])
                values = []
                fragmented = {}
                for data in pieces:
                    if "column" in data:
                        key = (data["row_start"], data["column"])
                        fragmented.setdefault(key, []).append(data["rows"][0][0])
                    else:
                        values.extend(value for row in data["rows"] for value in row)
                    values.extend(data.get("headers", []))
                values.extend("".join(parts) for parts in fragmented.values())
                if not any(normalize(unit["text"]) in normalize(value) for value in values):
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
                                                "reason": block["image"]["status"]} for identifier, block in skipped.items()],
                "missing_units": [], "linked_assets": len(images), "missing_assets": [],
                "linked_image_occurrences": len(image_parts), "missing_image_occurrences": [],
                "oversized_passages": 0, "status": "verified_native_coverage"}
