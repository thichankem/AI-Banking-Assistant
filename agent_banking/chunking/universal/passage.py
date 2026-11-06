"""Canonical passage for the versioned ingestion contract (no model calls)."""


def format_passage(chunk):
    metadata = chunk.get("business_metadata", {})
    retrieval = metadata.get("retrieval", {})
    context = str(retrieval.get("context", "")).strip()
    derived = str(retrieval.get("situated_context", "")).strip()
    parts = ["passage:"]
    if context:
        parts.append("[" + context + "]")
    if derived:
        parts.append("Ngữ cảnh: " + derived)
    parts.append(str(chunk.get("content", "")))
    return "\n".join(parts)
