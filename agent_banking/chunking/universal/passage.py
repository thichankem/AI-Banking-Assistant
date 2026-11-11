"""Canonical passage for the versioned ingestion contract (no model calls)."""


def format_passage(chunk):
    metadata = chunk.get("business_metadata", {})
    return format_passage_fields(chunk.get("content", ""), metadata.get("retrieval", {}))


def format_passage_fields(content, retrieval):
    context = str(retrieval.get("context", "")).strip()
    derived = str(retrieval.get("situated_context", "")).strip()
    parts = ["passage:"]
    if context:
        parts.append("[" + context + "]")
    if derived:
        parts.append("Ngữ cảnh: " + derived)
    parts.append(str(content))
    return "\n".join(parts)
