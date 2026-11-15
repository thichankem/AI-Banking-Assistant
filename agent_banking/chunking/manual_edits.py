"""Apply document-versioned text corrections to chunks and derived context."""

from copy import deepcopy
import hashlib
import json

from .universal.chunk_planner import escape_html
from .universal.language import project_vietnamese
from .universal.schema_assembler import link_chunk_sequence, validate_chunk_sequence


def _unique_fields(pairs):
    fields = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("Duplicate field in manual edits: " + name)
        fields[name] = value
    return fields


def _remove_caption(value, caption):
    return value.replace("<table>" + caption, "<table>", 1)


def _apply_manual_edits(path, chunks, artifact, budget):
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_fields)
    if not isinstance(value, dict) or value.get("document_version") != artifact["document"]["document_version"]:
        raise ValueError("Manual edits must reference the current document_version: " + path.name)
    edits = value.get("replacements")
    if not isinstance(edits, dict) or not edits:
        raise ValueError("Manual edits need a nonempty replacements object keyed by chunk_id")
    edited = deepcopy(chunks)
    by_id = {chunk["chunk_id"]: chunk for chunk in edited}
    edited_blocks = set()
    edited_ids = {}
    for identifier, replacement in edits.items():
        chunk = by_id.get(identifier)
        if chunk is None or chunk["chunk_type"] != "text" or not chunk["content"].strip():
            raise ValueError("Manual edit must reference a current, nonempty text chunk: " + identifier)
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError("Manual replacement must be nonempty text: " + identifier)
        metadata = chunk["business_metadata"]
        provenance = metadata["provenance"]
        original_spans = provenance.pop("source_spans", [])
        edited_blocks.update(span["block_id"] for span in original_spans)
        if not original_spans:
            edited_blocks.update(provenance.get("block_ids", []))
        provenance["manual_edit"] = {
            "original_chunk_id": identifier, "original_sha256": hashlib.sha256(chunk["content"].encode()).hexdigest(),
            "original_source_spans": original_spans,
        }
        provenance.pop("group_span", None)
        metadata["quality"]["manually_edited"] = True
        chunk["content"] = replacement
        retrieval = metadata["retrieval"]
        retrieval.pop("situated_context", None)
        retrieval.pop("context_evidence", None)
        retrieval.pop("passage_content", None)
        retrieval.pop("passage_coverage", None)
        retrieval.pop("embedding_content", None)
        projected = project_vietnamese(replacement).strip()
        if projected != replacement.strip():
            retrieval["embedding_content"] = projected
        if not projected:
            retrieval["role"] = "attachment_only"
        edited_ids[identifier] = identifier + ":edit:" + hashlib.sha256(replacement.encode()).hexdigest()[:16]
        chunk["chunk_id"] = edited_ids[identifier]

    edited_parents = {by_id[identifier]["parent_id"] for identifier in edits}
    revised = deepcopy(artifact)
    blocks = {block["id"]: block for block in revised["document"]["blocks"]}
    parents = revised["parents"]
    edited_labels = {block_id for block_id in edited_blocks if blocks[block_id]["type"] == "heading"}
    edited_texts = {blocks[block_id]["text"].strip() for block_id in edited_blocks}
    derived_titles = edited_texts | {text[:200].strip() for text in edited_texts}
    changed_breadcrumbs = set()
    for parent_id, parent in parents.items():
        if parent["parent_id"] and parent["block_ids"] and parent["block_ids"][0] in edited_labels:
            # A whole-chunk replacement has no reliable heading boundary.
            parent["section_title"] = ""
            parent.pop("retrieval_title", None)
        labels = []
        changed = False
        cursor = parent
        while cursor["parent_id"]:
            if cursor["block_ids"] and cursor["block_ids"][0] in edited_labels:
                changed = True
            if cursor["section_title"]:
                labels.append(cursor["section_title"])
            cursor = parents[cursor["parent_id"]]
        if changed:
            parent["breadcrumb"] = " > ".join([cursor["section_title"], *reversed(labels)])
            parent.pop("retrieval_context", None)
            changed_breadcrumbs.add(parent_id)

    by_edited_id = {chunk["chunk_id"]: chunk for chunk in edited}
    for chunk in edited:
        parent_id = chunk["parent_id"]
        metadata = chunk["business_metadata"]
        retrieval = metadata["retrieval"]
        preferred = retrieval.get("passage_content")
        evidence = retrieval.get("context_evidence", [])
        remaining_evidence = [item for item in evidence if item.get("source_block_id") not in edited_blocks]
        refresh_passage = False
        if remaining_evidence != evidence:
            refresh_passage = True
            retrieval.pop("context_evidence", None)
            retrieval.pop("situated_context", None)
            if remaining_evidence:
                retrieval["context_evidence"] = remaining_evidence
                retrieval["situated_context"] = budget.shorten(
                    " ".join(project_vietnamese(item["text"][:1200]).strip()
                             for item in remaining_evidence).strip(),
                    min(48, budget.maximum // 6),
                )
        if parent_id in changed_breadcrumbs:
            refresh_passage = True
            chunk["section_title"] = parents[parent_id]["section_title"]
            chunk["breadcrumb"] = parents[parent_id]["breadcrumb"]
            # The old embedding context can include a shortened version of the heading.
            retrieval.pop("context", None)
        table = chunk.get("table_data")
        title_source = table.get("title_source_block_id") if isinstance(table, dict) else None
        title = table.get("title", "") if isinstance(table, dict) else ""
        if title and (title_source in edited_blocks if title_source else title in derived_titles):
            refresh_passage = True
            table["title"] = ""
            table.pop("title_source_block_id", None)
            caption = "<caption>" + escape_html(title) + "</caption>"
            if "html" in table:
                table["html"] = _remove_caption(table["html"], caption)
            if "embedding_content" in retrieval:
                retrieval["embedding_content"] = _remove_caption(retrieval["embedding_content"], caption)
            # A truncated preview may contain only part of the removed caption.
            preferred = None
            old_content = chunk["content"]
            table_start = old_content.find("<table>" + caption)
            if table_start >= 0:
                start = table_start + len("<table>")
                chunk["content"] = old_content[:start] + old_content[start + len(caption):]
                for span in metadata["provenance"].get("source_spans", []):
                    if span["chunk_start"] >= start + len(caption):
                        span["chunk_start"] -= len(caption)
                        span["chunk_end"] -= len(caption)
            if not retrieval.get("context_evidence"):
                retrieval.pop("situated_context", None)
            edited_parents.add(parent_id)
        manual_edit = metadata["provenance"].get("manual_edit")
        if not (refresh_passage or manual_edit):
            continue
        if not manual_edit:
            budget.embedding_excerpt(chunk["content"], retrieval, preferred)
        count = budget.passage_count(chunk)
        if count > budget.maximum:
            identifier = manual_edit["original_chunk_id"] if manual_edit else chunk["chunk_id"]
            raise ValueError("Edited passage exceeds the token ceiling: " + identifier)
        retrieval["token_count"] = count

    for chunk in edited:
        if chunk["parent_id"] in edited_parents or chunk["parent_id"] in changed_breadcrumbs:
            chunk["parent_content"] = chunk["content"]
    link_chunk_sequence(edited)
    revised["quality"]["manually_edited_chunks"] = len(edits)
    if "empty_searchable_content" in revised["quality"]:
        revised["quality"]["empty_searchable_content"] = all(
            chunk["business_metadata"]["retrieval"].get("role") == "attachment_only" for chunk in edited
        )
    for parent_id, parent in parents.items():
        parent["child_chunk_ids"] = [edited_ids.get(identifier, identifier)
                                     for identifier in parent["child_chunk_ids"]]
        if parent_id in edited_parents:
            parent["content"] = "\n\n".join(by_edited_id[identifier]["content"]
                                           for identifier in parent["child_chunk_ids"]
                                           if by_edited_id[identifier]["content"])
    valid, issues = validate_chunk_sequence(edited)
    if not valid:
        raise ValueError("Invalid edited chunk sequence: " + "; ".join(issues[:5]))
    return edited, revised
