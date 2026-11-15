"""Bound cloud requests without changing source text or losing annotation IDs."""

import json
from types import SimpleNamespace

from agent_banking.chunking.universal.enrichment import review, structure


def _block(identifier, text, kind="paragraph", **fields):
    return {"id": identifier, "text": text, "type": kind, "provenance": {"region": "body"}, **fields}


def test_structure_windows_limit_serialized_evidence_and_keep_global_aliases():
    raw = [_block(f"source-{i}", "Khách hàng sử dụng hạn mức giao dịch.") for i in range(16)]
    raw[0].update(type="heading", level=1, authored_level=1)
    long_value = "ệ🐘" * 5000
    raw[3] = _block("source-3", long_value, "table", table_data={"grid": [[long_value] * 12] * 12},
                    provenance={"region": "body", "xpath": "/w:p" * 200})
    previews = [structure._source_preview(block, f"b{i}") for i, block in enumerate(raw)]
    assert structure._window_size(previews[3]) > 4500
    windows = [window for _, window in structure._structure_windows(previews)]
    assert [alias for window in windows for alias in window["editable_block_ids"]] == [f"b{i}" for i in range(16)]
    assert all(len(window["editable_block_ids"]) <= 6 for window in windows)
    assert all(len(json.dumps(window, ensure_ascii=False).encode()) <= 4500 for window in windows)
    wide = next(window for window in windows if "b3" in window["editable_block_ids"])
    assert wide["editable_block_ids"] == ["b3"]
    assert "b3" in wide["context_source_ids"]
    assert wide["previous_heading"]["level"] == 1
    assert raw[3]["table_data"]["grid"][0][0] == long_value


def test_structure_honors_lower_settings_and_transmits_response_ceiling(monkeypatch):
    endpoint = SimpleNamespace(model="fixture", url="https://example.invalid", api_key="fixture")
    response = {key: [] for key in structure.ANNOTATION_ARRAYS}
    window = {"editable_block_ids": ["b0"], "context_source_ids": ["b0"], "blocks": [{"id": "b0"}]}
    requested = []

    def respond(url, key, payload, **kwargs):
        requested.append((payload["max_tokens"], kwargs["max_output_tokens"]))
        return response

    for configured in ("65536", "1024"):
        monkeypatch.setenv("CHUNKING_LLM_MAX_TOKENS", configured)
        assert structure._review_window(0, window, lambda alias: "Source text", endpoint, respond) == response
    assert requested == [(8192, 8192), (1024, 1024)]


def test_review_windows_bound_multibyte_evidence_and_keep_every_source_id():
    native = "ệ🐘" * 2000
    identifiers = [f"p{i}" for i in range(7)]
    ir = {"blocks": [_block(identifier, native) for identifier in identifiers], "units": [], "warnings": [],
          "source_review": [{"code": "DIFFERENT_AMOUNTS", "reason": "ỹ🐘" * 500, "block_ids": identifiers}]}
    state = review._review_state(ir)
    windows = review._observation_windows(ir, state)
    assert len(windows) >= 2
    assert all(len(json.dumps({**state, "source_observations": window}, ensure_ascii=False).encode()) <= 3500
               for window in windows)
    evidence = [item for window in windows for observation in window for item in observation["native_evidence"]]
    assert set(item["id"] for item in evidence) == set(identifiers)
    assert all(item["total_source_characters"] == len(native) for item in evidence)
    assert all(observation["native_evidence"][0]["id"] == identifiers[0] for window in windows for observation in window)


def test_heading_review_uses_batches_of_five_with_matching_questions(monkeypatch):
    endpoint = SimpleNamespace(model="fixture", url="https://example.invalid", api_key="fixture")
    monkeypatch.setattr(review, "model_primary", lambda _: endpoint)
    ids = [f"p{i}" for i in range(12)]
    ir = {"blocks": [_block(identifier, "Hạn mức giao dịch " * 100, "heading", level=2) for identifier in ids],
          "structure_proposals": [{"block_id": identifier, "native_level": 0, "proposed_level": 2} for identifier in ids]}
    batches = []

    def respond(url, key, payload, **kwargs):
        candidates = payload["state"]["candidates"]
        batches.append([item["block_id"] for item in candidates])
        assert all(len(item["text"]) <= 250 for item in candidates)
        assert list(payload["questions"]) == [f"heading_{i}" for i in range(len(candidates))]
        return {"answers": {key: {"choice": "ACCEPT_PROPOSAL"} for key in payload["questions"]}}

    model = SimpleNamespace(_optional_cached=lambda ir, stage, model, inputs, operation, validate: operation())
    review.review_headings(model, ir, respond)
    assert [len(batch) for batch in batches] == [5, 5, 2]
    assert [identifier for batch in batches for identifier in batch] == ids
    assert [entry["block_id"] for entry in ir["heading_decisions"]] == ids
