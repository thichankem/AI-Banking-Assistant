"""Metadata for PDF and Word editions in a chunking publication."""

from datetime import UTC, date, datetime
import json
from pathlib import Path

from agent_banking.core.policy_dates import extract_policy_dates

from .publication import SUCCESS_STATUSES


def is_policy_source(source):
    return Path(source).suffix.lower() in {".pdf", ".docx"}


def read_policy_metadata(path):
    declarations = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(declarations, dict):
        raise ValueError("Policy metadata must map source paths to edition metadata")
    fields = {"document_version", "policy_id", "effective_from", "effective_to",
              "effective_date_reference", "uploaded_at"}
    for source, edition in declarations.items():
        if not is_policy_source(source) or not isinstance(edition, dict) or edition.keys() - fields:
            raise ValueError(f"Invalid PDF or Word policy metadata for {source}")
        digest = edition.get("document_version")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"Policy metadata for {source} requires a lowercase SHA-256 document_version")
    return declarations


def read_policy_history(manifest_path, schema_version):
    """Read intake history from the last committed publication, including legacy editions."""
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except FileNotFoundError:
        return {}
    if manifest.get("schema_version") != schema_version:
        return {}
    history = manifest.get("policy_revision_history", {})
    for record in manifest.get("records", ()):
        source = record.get("rel_path")
        digest = record.get("input_sha256")
        if source and digest and is_policy_source(source) and record.get("status") in SUCCESS_STATUSES:
            editions = history.setdefault(source, {})
            if "policy_revision" in record:
                editions[digest] = record["policy_revision"]
            else:
                editions.setdefault(digest, {})
    return history


def build_policy_revision(source, document, previous, declaration, chunks):
    digest = document["document_version"]
    revision = {"policy_id": document["document_id"], "document_version": digest,
                "effective_from": None, "effective_to": None, "effective_date_reference": None,
                "effective_date_status": "unknown", "effective_date_origin": None,
                "uploaded_at": None, "first_ingested_at": None}
    if previous is None:
        revision["first_ingested_at"] = datetime.now(UTC).isoformat()
    else:
        revision.update(previous)
    date_fields = {"effective_from", "effective_to", "effective_date_reference"}
    # Declarations made before automatic extraction also take precedence.
    legacy_declaration = previous and "effective_date_origin" not in previous and any(
        previous.get(key) for key in date_fields
    )
    if declaration.keys() & date_fields or legacy_declaration or revision["effective_date_origin"] == "declared":
        revision["effective_date_origin"] = "declared"
    else:
        revision.update(dict.fromkeys(date_fields))
        revision.update(effective_date_origin=None, effective_date_status="unknown")
        revision.update(extract_policy_dates(chunks))
    revision.update(declaration)

    for key in ("policy_id", "effective_date_reference"):
        value = revision[key]
        if key == "effective_date_reference" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Policy metadata {key} must be non-empty text for {source}")
        revision[key] = value.strip()

    effective_from = revision["effective_from"]
    effective_to = revision["effective_to"]
    for key, value in (("effective_from", effective_from), ("effective_to", effective_to)):
        if value is not None:
            try:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f"Policy metadata {key} must be YYYY-MM-DD for {source}") from error
    if (effective_from or effective_to) and not revision["effective_date_reference"]:
        raise ValueError(f"A policy effective date requires an effective_date_reference for {source}")
    if effective_from and effective_to and effective_to < effective_from:
        raise ValueError(f"Policy effective_to precedes effective_from for {source}")
    if revision["effective_date_origin"] == "declared":
        revision["effective_date_status"] = "unknown"
        if effective_from:
            revision["effective_date_status"] = "known"

    if revision["uploaded_at"] is not None:
        try:
            uploaded_at = datetime.fromisoformat(revision["uploaded_at"])
            if uploaded_at.utcoffset() is None:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(f"Policy uploaded_at must be an ISO timestamp with a time zone for {source}") from error
        revision["uploaded_at"] = uploaded_at.astimezone(UTC).isoformat()
    return revision


def attach_policy_revision(document, chunks, previous=None, declaration=None):
    source = document["source"]
    if not is_policy_source(source):
        return None
    revision = build_policy_revision(source, document, previous, declaration or {}, chunks)
    document["policy_revision"] = revision
    for chunk in chunks:
        chunk["business_metadata"]["policy_revision"] = revision
    return revision
