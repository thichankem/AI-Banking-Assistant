"""Select indexed policy editions before searching for supporting passages."""

from collections import defaultdict
from datetime import datetime

from agent_banking.chunking.policy_revision import is_policy_source
from agent_banking.chunking.universal.planning.schema import payload_index_key
from agent_banking.retrieval.policy_graph import verified_replacements
from agent_banking.tools.excel_rows import excel_version_key, select_excel_version


def in_scope(chunk, scope):
    return not scope or scope in chunk.get("business_metadata", {}).get("publication_scopes", ())


def _revision(chunk):
    return chunk.get("business_metadata", {}).get("policy_revision") or {}


def _edition(chunk):
    metadata = chunk.get("business_metadata", {})
    return chunk.get("source"), _revision(chunk).get("document_version") or metadata.get("document_version")


def _arrival(revision):
    for field in ("uploaded_at", "first_ingested_at"):
        value = revision.get(field)
        if value:
            try:
                stamp = datetime.fromisoformat(value)
                if stamp.utcoffset() is not None:
                    return stamp
            except (TypeError, ValueError):
                pass
    return None


class PolicyVersions:
    def __init__(self, corpus, scope, requested_date, today, legacy_dates=None, graph_path=None):
        self.scope = scope
        self.day = (requested_date or today).isoformat()
        self.dated = requested_date is not None
        self.reference_only = self.dated and requested_date != today
        self.legacy_dates = legacy_dates or {}
        self.corpus = [chunk for chunk in corpus if chunk.get("source") and in_scope(chunk, scope)]
        self.chunks_by_source = defaultdict(list)
        self.editions_by_source = defaultdict(set)
        for chunk in self.corpus:
            self.chunks_by_source[chunk["source"]].append(chunk)
            self.editions_by_source[chunk["source"]].add(_edition(chunk))
        self.enabled = self.dated or bool(self.legacy_dates) or any(
            _revision(chunk) or chunk.get("business_metadata", {}).get("excel_revision") for chunk in self.corpus)
        self.retired = verified_replacements(graph_path, self)
        if self.enabled:
            self.labels = self._select(self.corpus)
        else:
            self.labels = {}

    edition = staticmethod(_edition)

    def revision(self, chunk):
        edition = _edition(chunk)
        return _revision(chunk) or self.legacy_dates.get(edition, self.legacy_dates.get(edition[0], {}))

    def _select(self, chunks, *, named=False):
        groups = defaultdict(dict)
        workbooks = defaultdict(dict)
        labels = {}
        for chunk in chunks:
            edition = _edition(chunk)
            if edition in self.retired:
                if named:
                    labels[edition] = "reference_only"
                continue
            revision = self.revision(chunk)
            excel = chunk.get("business_metadata", {}).get("excel_revision")
            if excel:
                key = excel_version_key(chunk)
                workbooks[excel.get("document_key") or edition[0]][key] = edition
                continue
            if is_policy_source(edition[0]) and (revision or self.dated):
                policy = revision.get("policy_id") or edition[0]
                groups[policy][edition] = revision
            else:
                labels[edition] = None

        for editions in groups.values():
            active = [key for key, rev in editions.items()
                      if rev.get("effective_from") and rev["effective_from"] <= self.day
                      and (not rev.get("effective_to") or self.day <= rev["effective_to"])
                      and rev.get("effective_date_status") != "conflict"]
            if active:
                if len(active) == 1:
                    status = "effective"
                else:
                    status = "conflict"
                labels.update(dict.fromkeys(active, status))
                continue

            arrivals = {key: _arrival(rev) for key, rev in editions.items()}
            latest = max((stamp for stamp in arrivals.values() if stamp is not None), default=None)
            selected = [key for key, stamp in arrivals.items() if stamp == latest]
            for key in selected:
                rev = editions[key]
                if rev.get("effective_date_status") == "conflict":
                    status = "conflict"
                elif self.reference_only or rev.get("effective_from") or rev.get("effective_to"):
                    status = "reference_only"
                elif latest is None or len(selected) > 1:
                    status = "order_unknown"
                else:
                    status = "latest_available"
                labels[key] = status
        for versions in workbooks.values():
            as_of = None
            if self.dated:
                as_of = self.day
            selected, error = select_excel_version(versions, as_of)
            if error:
                status = "reference_only"
                if error in {"excel_version_conflict", "excel_version_ambiguous"}:
                    status = "conflict"
                labels.update(dict.fromkeys(versions.values(), status))
            else:
                status = "latest_available"
                if selected[2]:
                    status = "effective"
                labels[versions[selected]] = status
        return labels

    def with_sources(self, source_chunks):
        """A named source may be read as reference even when a newer edition exists."""
        if self.enabled:
            return self._select(source_chunks, named=True)
        return {}

    def allows(self, chunk, labels):
        return in_scope(chunk, self.scope) and (not self.enabled or _edition(chunk) in labels)

    def allows_graph(self, chunk, labels):
        if not self.allows(chunk, labels) or labels.get(_edition(chunk)) in {"conflict", "reference_only"}:
            return False
        revision = self.revision(chunk)
        first, last = revision.get("effective_from"), revision.get("effective_to")
        return (not first or first <= self.day) and (not last or self.day <= last)

    def search_filters(self, filters, labels, chunks):
        if not self.enabled:
            return filters
        names = {source for source, _ in labels}
        selected = set().union(*(self.editions_by_source[name] for name in names))
        result = {**filters, "source": sorted(names)}
        # Restrict the index by chunk identity if the same path contains multiple editions.
        if selected - labels.keys():
            result["chunk_id"] = sorted({chunk["chunk_id"] for chunk in chunks if _edition(chunk) in labels})
        return result

    def annotate(self, chunk, labels):
        status = labels.get(_edition(chunk))
        if status is None:
            return chunk.get("business_metadata", {})
        metadata = {**chunk.get("business_metadata", {}),
                    "temporal": {"as_of": self.day, "status": status, "requested_date": self.dated}}
        if not _revision(chunk) and (revision := self.revision(chunk)):
            metadata["policy_revision"] = revision
        return metadata
