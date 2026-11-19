"""Provider-independent routing, budgets, tracing and document regressions."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from agent_banking.agent.model_protocol import StructuredModel, object_schema
from agent_banking.core import models
from agent_banking.core.chat import FallbackChatClient
from agent_banking.core.fallback import ModelEndpoint, ModelFallbackError
from agent_banking.tracing.agent_trace import AgentRun


SCHEMA = object_schema({"answer": {"type": "string"}}, ["answer"])
STANDARD = ModelEndpoint("primary", "fixture-standard", "https://fixture.invalid/v1", "secret-fixture",
                         context_window=16000, max_output_tokens=256)
DEEP = replace(STANDARD, name="deep", model="fixture-deep", profile="deep", context_window=64000)


@pytest.fixture(autouse=True)
def routing_environment(monkeypatch):
    monkeypatch.setenv("AUTO_MODEL_ROUTING", "false")
    monkeypatch.delenv("AGENT_CONTEXT_BYTES", raising=False)


def install_provider(monkeypatch, outcomes, candidates=(STANDARD, DEEP)):
    calls = []
    script = iter(outcomes)

    class Provider:
        def __init__(self, **kwargs):
            self.chat = self
            self.completions = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, **payload):
            calls.append(deepcopy(payload))
            outcome = next(script)
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(outcome)), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25),
            )

    monkeypatch.setattr("openai.OpenAI", Provider)
    client = FallbackChatClient("agent", routed=False)
    client.routed = True
    client.candidates = candidates
    return StructuredModel(client, "ignored-primary"), calls


def request(model, data=None, **kwargs):
    return model.request("Return valid JSON.", data or {"request": "fixture"}, [SCHEMA], max_tokens=64, **kwargs)


@pytest.mark.parametrize("reason,expected", [(None, STANDARD.model), ("multiple_sources", DEEP.model),
    ("worker_evidence", DEEP.model), ("review_revision", DEEP.model)])
def test_role_selects_standard_or_deep(monkeypatch, reason, expected):
    model, calls = install_provider(monkeypatch, [{"answer": "ok"}])
    assert request(model, deep_reason=reason) == {"answer": "ok"}
    assert calls[0]["model"] == expected


def test_single_standard_is_eligible_for_deep_task(monkeypatch):
    model, calls = install_provider(monkeypatch, [{"answer": "ok"}], (STANDARD,))
    request(model, deep_reason="multiple_sources")
    assert len(calls) == 1 and calls[0]["model"] == STANDARD.model


def test_context_escalates_before_network_and_preserves_required_data(monkeypatch):
    small = replace(STANDARD, context_window=1100)
    model, calls = install_provider(monkeypatch, [{"answer": "ok"}], (small, DEEP))
    data = {"request": "required " * 300, "human_replies": ["confirmed"], "approved_scope": ["scope"]}
    request(model, data)
    sent = json.loads(calls[0]["messages"][-1]["content"])
    assert sent == data and calls[0]["model"] == DEEP.model


def test_review_keeps_entire_evidence_and_uses_review_role(monkeypatch):
    model, calls = install_provider(monkeypatch, [{"answer": "ok"}])
    model.review_client = SimpleNamespace(candidates=[replace(DEEP, context_window=4000)],
        module="agent_review", create=model.client.create)
    data = {"request": "review", "draft": "required", "evidence": [{"text": "full evidence " * 450}]}
    request(model, data, role="agent_review")
    assert calls[0]["model"] == STANDARD.model
    assert json.loads(calls[0]["messages"][-1]["content"])["evidence"] == data["evidence"]


def test_review_role_is_preferred(monkeypatch):
    model, calls = install_provider(monkeypatch, [{"answer": "ok"}])
    review = replace(STANDARD, model="review-only")
    model.review_client = SimpleNamespace(candidates=[review], module="agent_review", create=model.client.create)
    request(model, role="agent_review")
    assert calls[0]["model"] == "review-only"


def test_context_exhaustion_reports_category_without_cloud_call(monkeypatch, tmp_path):
    model, calls = install_provider(monkeypatch, [], (replace(STANDARD, context_window=800),))
    with AgentRun(tmp_path, "chat", "sensitive-session").capture(), pytest.raises(ModelFallbackError) as failure:
        request(model, {"request": "private " * 300})
    assert failure.value.category == "context_budget_exceeded"
    assert calls == []
    trace = (tmp_path / "agent_runs.jsonl").read_text(encoding="utf-8")
    assert "context_budget_exceeded" in trace and "private" not in trace and "sensitive-session" not in trace


@pytest.mark.parametrize("error", [TimeoutError(), ValueError("bad JSON"),
    type("RateLimited", (Exception,), {"status_code": 429})()])
def test_retry_fallback_and_repair_share_three_calls(monkeypatch, error):
    candidates = (replace(STANDARD, attempts=3), replace(DEEP, attempts=3))
    model, calls = install_provider(monkeypatch, [error] * 10, candidates)
    with pytest.raises(ModelFallbackError):
        request(model)
    assert len(calls) == 3
    assert [item["model"] for item in calls[:2]] == [STANDARD.model, DEEP.model]


def test_contract_repair_uses_next_model_and_supported_fields(monkeypatch):
    deep = replace(DEEP, token_parameter="max_completion_tokens", json_mode="off", send_temperature=False,
                   reasoning_effort="high", thinking=True, max_output_tokens=32)
    model, calls = install_provider(monkeypatch, [{"wrong": "schema"}, {"answer": "ok"}], (STANDARD, deep))
    request(model)
    sent = calls[1]
    assert sent["max_completion_tokens"] == 32
    assert "max_tokens" not in sent and "temperature" not in sent and "response_format" not in sent
    assert sent["reasoning_effort"] == "high"
    assert sent["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert "schema" in sent["messages"][0]["content"]


def test_unavailable_and_wrong_protocol_candidates_are_skipped(monkeypatch):
    model, calls = install_provider(monkeypatch, [{"answer": "ok"}],
        (replace(STANDARD, api_key=""), replace(DEEP, protocol="systemone"), STANDARD))
    request(model)
    assert len(calls) == 1 and calls[0]["model"] == STANDARD.model


def test_concurrent_requests_have_independent_budgets_and_traces(monkeypatch, tmp_path):
    model, calls = install_provider(monkeypatch, [{"answer": "ok"}] * 8)

    def execute(index):
        with AgentRun(tmp_path, "chat", str(index)).capture() as run:
            request(model, {"request": "secret input"}, deep_reason="complex" if index % 2 else None)
            return run.result({"status": "completed"})["trace_id"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(execute, range(8)))
    traces = [json.loads(line) for line in (tmp_path / "agent_runs.jsonl").read_text().splitlines()]
    assert len(set(ids)) == len(traces) == len(calls) == 8
    for trace in traces:
        attempt = [span for span in trace["spans"] if span["name"] == "model.attempt"]
        assert len(attempt) == 1 and attempt[0]["attributes"]["usage"]["total_tokens"] == 25
    assert "secret input" not in (tmp_path / "agent_runs.jsonl").read_text()
    assert "secret-fixture" not in (tmp_path / "agent_runs.jsonl").read_text()


def test_configuration_cache_identity_and_legacy_flag(monkeypatch):
    monkeypatch.setenv("AGENT_DEEP_MODEL", "configured-deep")
    before = models.chain_fingerprint("agent", routed=False)
    assert all(item.profile != "deep" for item in models.get_model_chain("agent", routed=False))
    monkeypatch.setenv("AUTO_MODEL_ROUTING", "true")
    assert models.get_model_chain("agent")[-1].profile == "deep"
    fingerprint = models.chain_fingerprint("agent")
    monkeypatch.setenv("AGENT_DEEP_REASONING_EFFORT", "high")
    assert models.chain_fingerprint("agent") != fingerprint
    fingerprint = models.chain_fingerprint("agent")
    monkeypatch.setenv("AGENT_DEEP_API_KEY", "different-secret")
    assert models.chain_fingerprint("agent") == fingerprint
    assert models.chain_fingerprint("agent", routed=False) == before


@pytest.mark.parametrize("field,value", [("PROTOCOL", "systemone"), ("CONTEXT_WINDOW", "-1"),
    ("TEMPERATURE", "nan"), ("MAX_OUTPUT_TOKENS", "-1"), ("SEND_TEMPERATURE", "maybe")])
def test_invalid_admin_settings_fail_without_exposing_values(monkeypatch, field, value):
    monkeypatch.setenv("AUTO_MODEL_ROUTING", "true")
    monkeypatch.setenv("AGENT_STANDARD_" + field, value)
    with pytest.raises(ValueError):
        models.get_model_chain("agent")


def test_document_decision_routes_protocol_and_limits_attempts(monkeypatch, tmp_path):
    from agent_banking.chunking.universal.models import text_client
    monkeypatch.setenv("AUTO_MODEL_ROUTING", "true")
    standard = replace(STANDARD, protocol="systemone", attempts=3)
    deep = replace(DEEP, protocol="systemone", attempts=3)
    monkeypatch.setattr(models, "get_model_chain", lambda *args, **kwargs: [standard, deep])
    sent = []

    def call(url, key, payload, *args):
        sent.append((url, deepcopy(payload)))
        if len(sent) == 1:
            raise TimeoutError()
        return {"answers": {"q": {"choice": "yes"}}}

    monkeypatch.setattr(text_client, "_cloud_json", call)
    budget = text_client.CloudAttempts()
    payload = {"state": {"table": "fixture"}, "questions": {"q": {"criteria": {"yes": "ok"}}}}
    with AgentRun(tmp_path, "document", "fixture").capture():
        result = text_client.cloud_json("https://fixture.invalid/systemone", "fixture", payload,
            model_module="chunking.decision", route_reason="table_continuation", attempt_budget=budget)
    assert [item[1]["model"] for item in sent] == [DEEP.model, STANDARD.model]
    assert all(item[0].endswith("/systemone") and "messages" not in item[1] for item in sent)
    assert result["_routing"]["model"] == STANDARD.model and budget.remaining == 1


def test_vision_uncertainty_escalates_and_cache_reuses_verified_result(monkeypatch, tmp_path):
    from agent_banking.chunking.tests.test_extraction_regressions import page_png, reviewed
    from agent_banking.chunking.universal.extraction.pdf.page_review import PdfVisualReviewer
    from agent_banking.chunking.universal.models.response_cache import PageVisionCache
    monkeypatch.setenv("AUTO_MODEL_ROUTING", "true")
    calls = []
    elements = [{"type": "paragraph", "text": "Fixture text", "bbox": [0.1, 0.1, 0.8, 0.8]}]

    def primary(*args):
        calls.append("primary")
        return {"elements": elements, "review": {"verdict": "unresolved"}, "uncertainties": ["unclear"]}

    def backup(*args):
        calls.append("review")
        return reviewed(elements)

    cache = PageVisionCache(str(tmp_path / "vision.sqlite"))
    reader = PdfVisualReviewer(tmp_path, cache, SimpleNamespace(model="vision", analyze_page_layout=primary),
        SimpleNamespace(model="review", analyze_page_layout=backup), False)
    try:
        result = reader.read(page_png(), [], 500, 500)
        assert calls == ["primary", "review"] and result["visual_review"]["cloud_attempts"] == 2
        assert result["visual_review"]["model"] == "review"
        reader.read(page_png(), [], 500, 500)
        assert calls == ["primary", "review"]
    finally:
        cache.close()


def test_trace_links_sse_pause_resume_and_http_done(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from agent_banking.agent.tests.test_interactions import CLARIFY, REVIEW, load_app, make_pipeline
    pipeline, _ = make_pipeline(monkeypatch, tmp_path / "sessions.sqlite", [CLARIFY,
        {"kind": "final", "answer": "Xin chào!"}, REVIEW, {"kind": "final", "answer": "Xin chào!"}, REVIEW])
    pipeline.tracer = SimpleNamespace(trace_agent=lambda entrypoint, session_id: AgentRun(tmp_path, entrypoint, session_id).capture())
    pipeline.runtime.client.routed = True
    pipeline.runtime.client.candidates = [STANDARD, DEEP]
    app = load_app(monkeypatch, pipeline)
    try:
        with TestClient(app) as client:
            def events(response):
                return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

            pause = events(client.post("/api/agent/stream", json={"message": "Tư vấn giúp tôi", "session_id": "test"}))[-1]
            assert pause["event_type"] == "awaiting_input" and pause["trace_id"]
            result = client.post("/api/agent/resume", json={"session_id": "test",
                "interaction_id": pause["interaction"]["id"], "choice_id": "option_1"}).json()
            assert result["audit_status"] == "PASS"
            assert result["turn_id"] == pause["turn_id"] and result["trace_id"] != pause["trace_id"]
            done = events(client.post("/api/agent/stream", json={"message": "Xin chào", "session_id": "test"}))[-1]
            assert done["event_type"] == "done" and done["trace_id"]
            assert done["turn_id"] != pause["turn_id"]
            rejected = client.post("/api/agent/resume", json={"session_id": "test",
                "interaction_id": pause["interaction"]["id"], "choice_id": "option_1"})
            assert rejected.status_code == 409 and rejected.headers["X-Trace-ID"]
        traces = [json.loads(line) for line in (tmp_path / "agent_runs.jsonl").read_text().splitlines()]
        stream_trace = next(trace for trace in traces if trace["trace_id"] == done["trace_id"])
        assert any(span["name"] == "model.attempt" for span in stream_trace["spans"])
        paused_trace = next(trace for trace in traces if trace["trace_id"] == pause["trace_id"])
        assert paused_trace["outcome"] == "awaiting_input"
        assert any(span["status"] == "suspended" for span in paused_trace["spans"])
    finally:
        pipeline.graph.checkpointer.conn.close()


def test_evaluator_comparison_keeps_judge_fixed_and_reports_routes(tmp_path):
    from agent_banking.evaluator.harness_eval import comparison, routing_metrics
    directories = [tmp_path / name for name in ("baseline", "candidate")]
    for index, directory in enumerate(directories):
        directory.mkdir()
        identity = {"models": "fixed-judge", "snapshot": {"digest": "same"}, "collection": "fixture",
                    "routing": {"enabled": bool(index)}}
        (directory / "identity.json").write_text(json.dumps(identity))
        with AgentRun(directory / "traces", "chat", "test").capture() as run:
            from agent_banking.tracing.agent_trace import agent_span
            with agent_span("model.attempt", role="agent", alias="standard", result="success",
                            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}):
                pass
        row = {"key": "the:1", "group": "the", "verdict": "pass" if index else "fail",
               "observed": [{"trace_id": run.trace_id}]}
        (directory / "results.jsonl").write_text(json.dumps(row) + "\n")
    compared = comparison(*directories)
    assert compared["verdict_changes"][0]["candidate_trace_ids"] == [run.trace_id]
    metrics = compared["candidate_metrics"]
    assert metrics["model_calls"] == metrics["traced_requests"] == 1
    assert metrics["usage"]["total_tokens"] == 15 and metrics["p50_ms"] is not None
    assert routing_metrics(tmp_path, {})["usage"] is None


def test_document_nested_in_request_shares_trace_and_records_cloud_call(tmp_path):
    from agent_banking.tracing.agent_trace import agent_span
    with AgentRun(tmp_path, "smoke", "parent").capture() as parent:
        with AgentRun(tmp_path, "document", "child").capture() as child:
            assert child.trace_id == parent.trace_id
            with agent_span("model.attempt", role="vision", model="fixture"):
                pass
    traces = [json.loads(line) for line in (tmp_path / "agent_runs.jsonl").read_text().splitlines()]
    assert len(traces) == 1
    assert [span["name"] for span in traces[0]["spans"]] == ["agent.smoke", "agent.document", "model.attempt"]


def test_document_crop_prefers_review_and_preserves_shared_budget(monkeypatch, tmp_path):
    from agent_banking.chunking.tests.test_extraction_regressions import page_png, reviewed
    from agent_banking.chunking.universal.extraction.pdf.page_review import PdfVisualReviewer
    from agent_banking.chunking.universal.models.response_cache import PageVisionCache
    from agent_banking.chunking.universal.models.text_client import CloudAttempts
    monkeypatch.setenv("AUTO_MODEL_ROUTING", "true")
    calls = []

    def review(*args):
        calls.append("review")
        return reviewed([{"type": "paragraph", "text": "Fixture text", "bbox": [0.1, 0.1, 0.8, 0.8]}])

    def primary(*args):
        raise AssertionError("Crop should prefer configured reviewer")

    cache = PageVisionCache(str(tmp_path / "crop.sqlite"))
    reader = PdfVisualReviewer(tmp_path, cache, SimpleNamespace(model="primary", analyze_page_layout=primary),
                              SimpleNamespace(model="review", analyze_page_layout=review), False)
    try:
        budget = CloudAttempts(2)
        reader.read(page_png(), [], 500, 500, region_depth=1, attempt_budget=budget)
        assert calls == ["review"] and budget.remaining == 1
    finally:
        cache.close()


def test_review_deep_inherits_standard_connection(monkeypatch):
    monkeypatch.setenv("AUTO_MODEL_ROUTING", "true")
    monkeypatch.setenv("AGENT_REVIEW_STANDARD_MODEL", "review-standard")
    monkeypatch.setenv("AGENT_REVIEW_STANDARD_BASE_URL", "https://fixture.invalid")
    monkeypatch.setenv("AGENT_REVIEW_STANDARD_API_KEY", "fixture")
    monkeypatch.setenv("AGENT_REVIEW_DEEP_MODEL", "review-deep")
    standard, deep = models.get_model_chain("agent_review")
    assert deep.url == standard.url and deep.api_key == standard.api_key


def test_disabled_document_role_stays_disabled_when_profiles_exist(monkeypatch):
    monkeypatch.setenv("AUTO_MODEL_ROUTING", "true")
    monkeypatch.setenv("CHUNKING_VISION_ENABLED", "false")
    monkeypatch.setenv("CHUNKING_VISION_DEEP_MODEL", "configured-deep")
    assert models.get_model_chain("chunking.vision") == []


def test_vision_can_escalate_own_deep_profile_after_uncertainty(monkeypatch, tmp_path):
    from agent_banking.chunking.tests.test_extraction_regressions import page_png, reviewed
    from agent_banking.chunking.universal.extraction.pdf.page_review import PdfVisualReviewer
    from agent_banking.chunking.universal.models import text_client, vision_client
    from agent_banking.chunking.universal.models.response_cache import PageVisionCache
    monkeypatch.setenv("AUTO_MODEL_ROUTING", "true")
    standard = replace(STANDARD, protocol="vision", attempts=3, context_window=0)
    deep = replace(DEEP, protocol="vision", attempts=3, context_window=0)
    monkeypatch.setattr(models, "get_model_chain", lambda *args, **kwargs: [standard, deep])
    monkeypatch.setattr(vision_client, "model_primary", lambda *args: standard)
    calls = []

    def call(url, key, payload, *args):
        calls.append(payload["model"])
        result = reviewed([{"type": "paragraph", "text": "Fixture text", "bbox": [0.1, 0.1, 0.8, 0.8]}])
        result["_api"] = {"model": payload["model"]}
        if payload["model"] == standard.model:
            result["review"]["verdict"] = "unresolved"
            result["uncertainties"] = ["unclear"]
        return result

    monkeypatch.setattr(text_client, "_cloud_json", call)
    cache = PageVisionCache(str(tmp_path / "own-deep.sqlite"))
    reader = PdfVisualReviewer(tmp_path, cache, vision_client.DocumentVisionClient(), None, False)
    try:
        layout = reader.read(page_png(), [], 500, 500)
        assert calls == [STANDARD.model, DEEP.model]
        assert layout["visual_review"]["model"] == DEEP.model
    finally:
        cache.close()
