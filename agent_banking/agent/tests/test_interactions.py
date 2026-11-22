"""Exercise checkpoints and the HTTP contract with deterministic model responses."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from agent_banking.agent_pipeline import BankingAgentPipeline, InteractionError
from agent_banking.core.settings import Config
from agent_banking.agent.tests.test_deep_agent import fixture_review


CLARIFY = {"kind": "clarify", "question": "Bạn muốn so sánh theo tiêu chí nào?",
           "options": ["Tổng tiền lãi", "Khoản trả tháng đầu"]}
CONFIRM = {"kind": "confirm", "proposal": "So sánh theo tổng tiền lãi."}
SEARCH = {"kind": "tool", "name": "search_banking_knowledge", "arguments": {"query": "tài liệu tham khảo"}}
REVIEW = {"verdict": "approve", "feedback": ""}


def make_pipeline(monkeypatch, path, decisions):
    monkeypatch.setattr(Config, "ENABLE_SEMANTIC_GUARDRAIL", False)
    rag = SimpleNamespace(tracer=SimpleNamespace())
    pipeline = BankingAgentPipeline(sqlite_db_path=str(path), rag_pipeline=rag)
    calls = []
    script = iter(decisions)

    def call(messages, validate, **kwargs):
        calls.append(json.loads(messages[-1]["content"]))
        answer = fixture_review(next(script), calls[-1])
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(answer, ensure_ascii=False))
        )])
        validate(response)
        return response

    monkeypatch.setattr(pipeline.runtime.client, "create", call)
    return pipeline, calls


def test_resume_after_reopen_retains_evidence_and_runs_once(tmp_path, monkeypatch):
    path = tmp_path / "harness.sqlite"
    plan = {"kind": "plan", "items": [{"id": 1, "task": "So sánh phương án", "status": "pending"}]}
    pipeline, _ = make_pipeline(monkeypatch, path, [plan, SEARCH, CLARIFY])
    monkeypatch.setattr(pipeline.runtime.tools, "execute_tool", lambda *args, **kwargs: {
        "status": "SUCCESS", "data": {"documents": [
            {"chunk_id": "a", "source": "policy.pdf", "content": "Tài liệu dùng để đối chiếu."}
        ]}
    })
    pending = pipeline.chat("So sánh giúp tôi", session_id="first")
    assert pending["status"] == "awaiting_input"
    assert pending["plan_revision"] == 1
    pipeline.graph.checkpointer.conn.close()

    pipeline, calls = make_pipeline(monkeypatch, path, [
        {"kind": "final", "answer": "Đã đối chiếu theo kỳ hạn [Nguồn 1]."}, REVIEW
    ])
    try:
        card_id = pending["interaction"]["id"]
        assert pipeline.status("first") == pending
        assert pipeline.chat("Không ghi đè câu hỏi", session_id="first") == pending
        assert not calls
        with pytest.raises(InteractionError, match="không còn"):
            pipeline.resume("wrong", card_id, "option_1")
        with pytest.raises(InteractionError, match="không hợp lệ"):
            pipeline.resume("first", card_id, "approve")
        with pytest.raises(InteractionError):
            pipeline.resume("first", card_id, "other", " ")

        def click():
            try:
                return pipeline.resume("first", card_id, "other", "Theo kỳ hạn")
            except InteractionError as error:
                return error.status_code

        with ThreadPoolExecutor(max_workers=2) as pool:
            left, right = list(pool.map(lambda _: click(), range(2)))
        assert sum(result == 409 for result in (left, right)) == 1
        result = pipeline.status("first")
        assert result["accepted_interaction_id"] == card_id
        assert result["audit_status"] == "PASS" and result["sources"] == ["policy.pdf"]
        assert len(result["tool_calls"]) == 1
        assert len(calls) == 2
        assert calls[0]["remaining_actions"] == 9
        assert calls[0]["human_replies"][0]["response"] == "Theo kỳ hạn"
        assert calls[0]["plan"] == plan["items"]
    finally:
        pipeline.graph.checkpointer.conn.close()


def test_plan_scope_changes_require_new_confirmation(tmp_path, monkeypatch):
    first = {"kind": "plan", "items": [{"id": 1, "task": "So sánh tổng tiền lãi", "status": "pending"}]}
    revision = {"kind": "plan", "items": [{"id": 1, "task": "So sánh kỳ hạn", "status": "pending"}]}
    pipeline, calls = make_pipeline(monkeypatch, tmp_path / "scope.sqlite", [
        first, CONFIRM, revision, {"kind": "confirm", "proposal": "So sánh kỳ hạn thay cho tổng tiền lãi."}
    ])
    try:
        pending = pipeline.chat("Hỏi trước khi tiếp tục", "scope")
        next_card = pipeline.resume("scope", pending["interaction"]["id"], "approve")
        assert next_card["status"] == "awaiting_input"
        assert next_card["interaction"]["scope"] == [{"id": 1, "task": "So sánh kỳ hạn"}]
        assert calls[-1]["confirmation_required"] is True
        canceled = pipeline.resume("scope", next_card["interaction"]["id"], "reject")
        assert canceled["status"] == "canceled" and canceled["tool_calls"] == []
        assert len(calls) == 4
    finally:
        pipeline.graph.checkpointer.conn.close()


def test_cannot_request_credentials_and_memory_fallback_uses_text(tmp_path, monkeypatch):
    blocked = {**CLARIFY, "question": "Vui lòng nhập mã OTP để tiếp tục?"}
    bad_parent = tmp_path / "file"
    bad_parent.touch()
    pipeline, _ = make_pipeline(monkeypatch, bad_parent / "unwritable.sqlite", [CLARIFY, blocked])
    fallback = pipeline.chat("Cần làm rõ", "fallback")
    assert fallback["audit_status"] == "NEEDS_INPUT"
    assert "không thể tiếp tục tự động" in fallback["answer"]
    assert pipeline.status("fallback")["status"] == "idle"
    blocked_result = pipeline.chat("Hỏi một câu khác", "blocked")
    assert blocked_result["audit_status"] == "NEEDS_INPUT"
    assert "OTP" not in blocked_result["answer"]


def test_duplicate_question_ends_turn_instead_of_pausing_again(tmp_path, monkeypatch):
    pipeline, _ = make_pipeline(monkeypatch, tmp_path / "duplicate.sqlite", [CLARIFY, CLARIFY])
    try:
        pending = pipeline.chat("So sánh giúp tôi", "duplicate")
        result = pipeline.resume("duplicate", pending["interaction"]["id"], "option_1")
        assert result["audit_status"] == "NEEDS_INPUT"
        assert "thiếu thông tin" in result["answer"]
        assert pipeline.status("duplicate")["status"] == "completed"
    finally:
        pipeline.graph.checkpointer.conn.close()


def test_retry_read_once_then_drop_unverified_calculation(tmp_path, monkeypatch):
    calculator = {"kind": "tool", "name": "calculate_loan_amortization", "arguments": {
        "principal_vnd": 100000000, "annual_rate_percent": 8, "term_months": 12
    }}
    pipeline, calls = make_pipeline(monkeypatch, tmp_path / "tools.sqlite", [
        calculator, SEARCH, {"kind": "final", "answer": "Tôi chưa xác minh được kết quả.",
                             "insufficient_evidence": True}, REVIEW
    ])
    attempts = []

    def execute(name, params, *, definition):
        attempts.append(name)
        if name == calculator["name"]:
            if len(attempts) == 1:
                return {"status": "FAILED", "temporary": True}
            return {"status": "SUCCESS", "data": {
                "status": "SUCCESS", **params, "first_month_payment_vnd": 9000000,
                "total_interest_vnd": 5000000, "total_payable_vnd": 122000000,
            }}
        return {"status": "SUCCESS", "data": {"documents": []}}

    monkeypatch.setattr(pipeline.runtime.tools, "execute_tool", execute)
    try:
        result = pipeline.chat("Tính khoản vay và đối chiếu", "tools")
        assert result["audit_status"] == "PASS"
        assert attempts == [calculator["name"], calculator["name"], SEARCH["name"]]
        rejected, no_evidence = calls[2]["observations"]
        assert rejected["error_code"] == "verification_failed" and rejected["retried"]
        assert "data" not in rejected
        assert no_evidence["error_code"] == "insufficient_evidence"
    finally:
        pipeline.graph.checkpointer.conn.close()


def load_app(monkeypatch, pipeline):
    from agent_banking import agent_pipeline, node_pipeline, pipeline as rag_module

    rag = SimpleNamespace(use_memory=True)
    path = Path(__file__).resolve().parents[3] / "main.py"
    spec = importlib.util.spec_from_file_location("banking_test_main", path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setattr(rag_module, "BankingRAGPipeline", lambda **kwargs: rag)
        patch.setattr(node_pipeline, "NodePipeline", lambda: None)
        patch.setattr(agent_pipeline, "BankingAgentPipeline", lambda **kwargs: pipeline)
        spec.loader.exec_module(module)
    return module.app


def test_api_sse_pause_status_resume_and_plain_chat(tmp_path, monkeypatch):
    pipeline, _ = make_pipeline(monkeypatch, tmp_path / "api.sqlite", [
        CLARIFY, CONFIRM, {"kind": "final", "answer": "Xin chào!"}, REVIEW,
        {"kind": "final", "answer": "Chào bạn!"}, REVIEW
    ])
    app = load_app(monkeypatch, pipeline)
    try:
        with TestClient(app) as client:
            session = {"session_id": "api"}
            stream = client.post("/api/agent/stream", json={**session, "message": "So sánh giúp tôi"})
            events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
            assert events[-1]["event_type"] == "awaiting_input"
            assert not any(event["event_type"] == "done" for event in events)
            card_id = events[-1]["interaction"]["id"]
            pending = client.post("/api/agent/status", json=session).json()
            assert pending["interaction"]["id"] == card_id
            next_card = client.post("/api/agent/resume", json={
                **session, "interaction_id": card_id, "choice_id": "option_1"
            }).json()
            assert next_card["interaction"]["kind"] == "confirm"
            assert client.post("/api/agent/resume", json={
                **session, "interaction_id": card_id, "choice_id": "option_1"
            }).status_code == 409
            canceled = client.post("/api/agent/resume", json={
                **session, "interaction_id": next_card["interaction"]["id"], "choice_id": "reject"
            }).json()
            assert canceled["status"] == "canceled"
            assert client.post("/api/agent/chat", json={**session, "message": "Xin chào"}).json()["status"] == "completed"
            plain_stream = client.post("/api/agent/stream", json={**session, "message": "Chào bạn"})
            assert '"event_type": "done"' in plain_stream.text and '"full_answer": "Chào bạn!"' in plain_stream.text
    finally:
        pipeline.graph.checkpointer.conn.close()


def test_multi_select_clarify_resume_and_single_choice_rejection(tmp_path, monkeypatch):
    multi_clarify = {
        "kind": "clarify",
        "question": "Bạn muốn so sánh theo những tiêu chí nào?",
        "options": ["Tổng tiền lãi", "Lãi suất năm", "Phí trả nợ trước hạn"],
        "multi_select": True,
    }
    pipeline, calls = make_pipeline(monkeypatch, tmp_path / "multi.sqlite", [
        multi_clarify,
        {"kind": "final", "answer": "Đã so sánh theo các tiêu chí bạn chọn."},
        REVIEW,
        CLARIFY,
        {"kind": "final", "answer": "Đã so sánh theo tổng tiền lãi."},
        REVIEW,
    ])
    try:
        pending = pipeline.chat("So sánh giúp tôi", session_id="multi-session")
        assert pending["status"] == "awaiting_input"
        card = pending["interaction"]
        assert card["multi_select"] is True

        with pytest.raises(InteractionError) as exc_cancel:
            pipeline.resume("multi-session", card["id"], choice_ids=["option_1", "cancel"])
        assert exc_cancel.value.status_code == 422

        with pytest.raises(InteractionError) as exc_empty:
            pipeline.resume("multi-session", card["id"], choice_ids=[])
        assert exc_empty.value.status_code == 422

        resumed = pipeline.resume(
            "multi-session",
            card["id"],
            choice_ids=["option_1", "option_3", "other"],
            text="Kỳ hạn 12 tháng",
        )
        assert resumed["status"] == "completed"
        assert resumed["audit_status"] == "PASS"
        assert calls[1]["human_replies"][0]["response"] == "Tổng tiền lãi; Phí trả nợ trước hạn; Kỳ hạn 12 tháng"
        assert calls[1]["human_replies"][0]["selected_ids"] == ["option_1", "option_3", "other"]

        single_pending = pipeline.chat("So sánh tiếp", session_id="single-session")
        single_card = single_pending["interaction"]
        assert single_card["multi_select"] is False
        with pytest.raises(InteractionError) as exc_single:
            pipeline.resume("single-session", single_card["id"], choice_ids=["option_1", "option_2"])
        assert exc_single.value.status_code == 422

        app = load_app(monkeypatch, pipeline)
        with TestClient(app) as client:
            http_ok = client.post("/api/agent/resume", json={
                "session_id": "single-session",
                "interaction_id": single_card["id"],
                "choice_ids": ["option_1"],
            })
            assert http_ok.status_code == 200
            assert http_ok.json()["audit_status"] == "PASS"
    finally:
        pipeline.graph.checkpointer.conn.close()


def test_api_multi_select_resume_via_http(tmp_path, monkeypatch):
    multi_clarify = {
        "kind": "clarify",
        "question": "Chọn các kỳ hạn muốn đối chiếu?",
        "options": ["3 tháng", "6 tháng", "12 tháng"],
        "multi_select": True,
    }
    pipeline, calls = make_pipeline(monkeypatch, tmp_path / "api_multi.sqlite", [
        multi_clarify,
        {"kind": "final", "answer": "Đã đối chiếu 3 tháng và 12 tháng."},
        REVIEW,
    ])
    app = load_app(monkeypatch, pipeline)
    try:
        with TestClient(app) as client:
            paused = client.post("/api/agent/chat", json={"session_id": "api-m", "message": "Tính lãi"}).json()
            assert paused["status"] == "awaiting_input"
            assert paused["interaction"]["multi_select"] is True
            bad = client.post("/api/agent/resume", json={
                "session_id": "api-m",
                "interaction_id": paused["interaction"]["id"],
                "choice_ids": ["option_1", "cancel"],
            })
            assert bad.status_code == 422
            done = client.post("/api/agent/resume", json={
                "session_id": "api-m",
                "interaction_id": paused["interaction"]["id"],
                "choice_ids": ["option_1", "option_3"],
            })
            assert done.status_code == 200
            body = done.json()
            assert body["status"] == "completed"
            assert body["audit_status"] == "PASS"
            assert calls[1]["human_replies"][0]["response"] == "3 tháng; 12 tháng"
    finally:
        pipeline.graph.checkpointer.conn.close()
