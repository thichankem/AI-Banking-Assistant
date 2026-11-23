"""Regressions for A-Bank discovery, scoped retrieval, grounding and context."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_banking.agent.deep import validate_review
from agent_banking.agent.document_context import _format_retrieved_doc_context
from agent_banking.agent.model_protocol import pack_context
from agent_banking.agent.tests.test_deep_agent import approval, final, make_agent, run
from agent_banking.core.fallback import ModelFallbackError
from agent_banking.retrieval.hybrid_retriever import HybridRetriever
from agent_banking.tools.extensions.banking_query import parse_banking_query
from agent_banking.tools.rag_tool import BankingRAGTool


def test_vietnamese_table_uses_relevant_complete_row_and_header():
    rows = [[f"Thẻ mẫu {index}", "20.000.000 VND", "50 triệu"] for index in range(30)]
    rows.append(["Sao Mai Plus", "35.000.000 VND", "120 triệu"])
    chunk = {"source": "A-Bank.docx", "content": "Phần mở đầu. " * 250, "table_data": {
        "title": "Hạn mức theo ngày", "units": "Đơn vị VND",
        "headers": ["Thẻ", "Rút ATM", "Chuyển khoản"], "rows": rows,
    }}
    result = _format_retrieved_doc_context(chunk, max_chars=220, query="sao mai plus rut atm")
    assert "| Sao Mai Plus | 35.000.000 VND | 120 triệu |" in result
    assert "| Thẻ | Rút ATM | Chuyển khoản |" in result
    assert "Hạn mức theo ngày" in result and "Đơn vị VND" in result


def test_exact_query_rejects_similar_service_code_and_reports_fallback():
    query = "Xử lý lỗi ứng dụng"
    candidates = [
        {"chunk": {"chunk_id": "1", "source": "Hướng dẫn", "content": "Mã ABP-180."},
         "retrieval_meta": {"fallback_unfiltered": True}},
        {"chunk": {"chunk_id": "2", "source": "Hướng dẫn", "content": "Mã ABP-18. Cập nhật ứng dụng."}},
    ]
    engine = SimpleNamespace(parse_query=parse_banking_query,
                             retriever=SimpleNamespace(retrieve=lambda **kwargs: candidates),
                             reranker=SimpleNamespace(rerank=lambda candidates, **kwargs: [
                                 {**item, "rerank_method": "local_lexical"} for item in candidates
                             ]),
                             config=SimpleNamespace(HYBRID_TOP_K=5, HYBRID_CANDIDATE_K=20))
    result = BankingRAGTool(pipeline_instance=engine).execute(query, exact="abp 18")
    assert [doc["chunk_id"] for doc in result["documents"]] == ["2"]
    assert result["retrieval_quality"]["relaxed_filters"] is True
    assert result["retrieval_quality"]["reranker"] == "local_lexical"


def test_retrieval_fallback_cannot_drop_publication_scope():
    filters = []

    def search(**kwargs):
        filters.append(kwargs["filter_dict"])
        return []

    scope = {"business_metadata.publication_scopes": "a-bank"}
    store = SimpleNamespace(search_vector=search, search_bm25=search)
    retriever = HybridRetriever(vector_store=store, embedder=SimpleNamespace(embed_query=lambda _: [1.0]))
    assert not retriever.retrieve("Biểu phí", filter_dict={**scope, "chunk_type": "table"})
    assert filters == [{**scope, "chunk_type": "table"}] * 2 + [scope] * 2


def test_context_budget_keeps_user_request_and_stable_labels():
    evidence = [{"label": f"[Nguồn {i}]", "text": "Nội dung tài liệu " * 30} for i in range(1, 18)]
    packed = pack_context({"request": "Phí thẻ phụ Sao Mai?", "evidence": evidence,
                           "history": [{"role": "user", "content": "lịch sử " * 500},
                                       {"role": "user", "content": "Chuyển sang thẻ phụ."}]}, 1600)
    assert len(packed.encode("utf-8")) <= 1600
    context = json.loads(packed)
    assert context["request"] == "Phí thẻ phụ Sao Mai?"
    assert context["history"][-1]["content"] == "Chuyển sang thẻ phụ."
    assert context["evidence"] == evidence[:len(context["evidence"])]
    assert context["evidence"][0]["label"] == "[Nguồn 1]"


def test_review_disallows_citations_without_exact_support_and_unsourced_amounts():
    evidence = [{"label": "[Nguồn 1]", "text": "Phí thẻ phụ là 60.000 VND."}]
    citations = [{"label": "[Nguồn 1]"}]
    coverage = [{"item": "Phí thẻ phụ", "status": "answered"}]
    draft = "Phí thẻ phụ là 60.000 VND [Nguồn 1]."
    claim = {"claim": "Phí thẻ phụ là 60.000 VND", "source": "[Nguồn 1]", "quote": "60.000 VND"}
    validate_review({"coverage": coverage, "claims": [claim]}, draft, citations, evidence, {})
    with pytest.raises(ValueError, match="Invalid claim"):
        validate_review({"coverage": coverage, "claims": [{**claim, "quote": "70.000 VND"}]},
                        draft, citations, evidence, {})
    with pytest.raises(ValueError, match="Unsupported financial"):
        validate_review({"coverage": coverage, "claims": []}, "Hotline là 1900 1234.", [], [], {})


def test_agent_inspects_skill_and_uses_discovered_arguments(tmp_path, monkeypatch):
    runtime, graph, requests = make_agent(monkeypatch, tmp_path / "skills.sqlite", [
        {"kind": "inspect_skill", "name": "service-troubleshooting"},
        {"kind": "skill", "name": "service_troubleshooting", "arguments": {"query": "lỗi ABP-18"}},
        final("Hãy mở lại ứng dụng [Nguồn 1]."),
        approval(),
    ])
    calls = []

    def execute(name, arguments, *, definition):
        calls.append((name, arguments))
        return {"status": "SUCCESS", "data": {"documents": [
            {"chunk_id": "a", "source": "A-Bank.docx", "content": "ABP-18: Hãy mở lại ứng dụng."}
        ]}}

    monkeypatch.setattr(runtime.tools, "execute_tool", execute)
    try:
        state = run(graph, "abp-18 sửa ntn")
        assert requests[0]["active_skills"] == {}
        assert requests[1]["observations"][0]["kind"] == "inspect_skill"
        active = requests[1]["active_skills"]["service-troubleshooting"]
        assert active["category"] == "Vận hành"
        assert "query" in active["parameters"]["properties"]
        assert calls == [("search_banking_knowledge", {"query": "lỗi ABP-18"})]
        assert state["audit_status"] == "PASS"
        assert state["verification"]["checked_claims"] == 1
    finally:
        graph.checkpointer.conn.close()


def test_static_brand_uses_anonymous_bank():
    html = (Path(__file__).resolve().parents[3] / "static" / "index.html").read_text(encoding="utf-8")
    assert "Trợ lý AI A-Bank" in html
    assert "vietcombank" not in html.casefold()


def test_verified_tool_result_survives_model_contract_error(tmp_path, monkeypatch):
    failure = ModelFallbackError("agent", [("fixture", "invalid response")])
    failure.last_error = ValueError("Model output does not match the response schema")
    runtime, graph, _ = make_agent(monkeypatch, tmp_path / "recovery.sqlite", [
        {"kind": "tool", "name": "calculate_loan_amortization", "arguments": {
            "principal_vnd": 120000000, "annual_rate_percent": 12, "term_months": 12
        }},
        failure,
        failure,
        failure,
        {**final("Tháng đầu là 11.200.000 VND."), "analysis": "provider-added metadata"},
        approval(),
    ])
    try:
        state = run(graph, "Tính khoản trả tháng đầu của khoản vay 120 triệu, 12 tháng, 12%/năm.")
        assert state["audit_status"] == "PASS"
        assert "deep_decide: recovered_contract" in state["execution_steps"]
        assert len(state["tool_outputs"]) == 1
    finally:
        graph.checkpointer.conn.close()


def test_conversation_context_stays_in_separate_sessions(tmp_path, monkeypatch):
    from agent_banking.agent.tests.test_interactions import make_pipeline

    pipeline, requests = make_pipeline(monkeypatch, tmp_path / "isolation.sqlite", [
        final("Đã ghi nhận sản phẩm."), approval(),
        final("Chào Quý khách."), approval(),
        final("Quý khách đang hỏi về thẻ Sao Mai."), approval(),
    ])
    try:
        pipeline.chat("Tôi đang hỏi về thẻ Sao Mai.", "customer-a")
        pipeline.chat("Xin chào.", "customer-b")
        pipeline.chat("Đang nói tới sản phẩm nào?", "customer-a")
        assert not requests[2]["history"]
        assert "thẻ Sao Mai" in requests[4]["history"][0]["content"]
    finally:
        pipeline.graph.checkpointer.conn.close()


def test_agent_skill_loads_body_and_reference_only_when_selected(tmp_path, monkeypatch):
    runtime, graph, requests = make_agent(monkeypatch, tmp_path / "skill.sqlite", [
        {"kind": "inspect_skill", "name": "guide"},
        {"kind": "inspect_skill", "name": "guide", "reference": "../outside.md"},
        {"kind": "inspect_skill", "name": "guide", "reference": "references/example.md"},
        {"kind": "skill", "name": "guide", "arguments": {}},
        final("Đây là một hướng dẫn dùng chung."), approval(),
    ])
    runtime.skills.skills_dir = tmp_path
    folder = tmp_path / "guide"
    refs = folder / "references"
    refs.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: guide\ndescription: Hướng dẫn chung\nallowed-tools: Bash\n---\n\nNỘI DUNG RIÊNG",
        encoding="utf-8",
    )
    (refs / "example.md").write_text("TÀI LIỆU BỔ TRỢ", encoding="utf-8")
    (tmp_path / "outside.md").write_text("NỘI DUNG NGOÀI GÓI", encoding="utf-8")
    try:
        spec = runtime.skills.snapshot()["guide"]
        assert "instructions" not in spec
        assert runtime.skills.get_skill("guide") is None
        assert runtime._worker_snapshot("policy_research", runtime.snapshot())["skill"].get("guide") is None
        state = run(graph, "Cho tôi hỏi về hướng dẫn A-Bank")
        assert state["audit_status"] == "PASS"
        assert requests[0]["active_skills"] == {}
        loaded = requests[1]["active_skills"]["guide"]
        assert loaded["instructions"] == "NỘI DUNG RIÊNG"
        assert loaded["references"] == ["references/example.md"]
        assert "loaded_references" not in loaded
        assert requests[2]["observations"][-1]["status"] == "ERROR"
        assert requests[3]["active_skills"]["guide"]["loaded_references"] == {
            "references/example.md": "TÀI LIỆU BỔ TRỢ"
        }
        assert requests[4]["observations"][-1]["status"] == "ERROR"
        assert not state["tool_outputs"]
    finally:
        graph.checkpointer.conn.close()


def test_registry_rejects_bad_frontmatter_and_pins_discovery(tmp_path):
    from agent_banking.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(skills_dir=tmp_path)
    for name, metadata in (
        ("duplicate", "name: duplicate\ndescription: First\ndescription: Second"),
        ("wrong-name", "name: another-name\ndescription: Example"),
        ("script-binding", "name: script-binding\ndescription: Example\nmetadata:\n  a-bank-tool: create_support_ticket"),
        ("valid-skill", "name: valid-skill\ndescription: Example"),
    ):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "SKILL.md").write_text(f"---\n{metadata}\n---\n\nBody", encoding="utf-8")
    snapshot = registry.snapshot()
    assert set(snapshot) == {"valid-skill"}
    document = tmp_path / "valid-skill" / "SKILL.md"
    document.write_text(document.read_text(encoding="utf-8") + " changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        registry.instructions(snapshot["valid-skill"])
    assert registry.instructions(registry.snapshot()["valid-skill"])["instructions"] == "Body changed"


def test_tool_catalog_projects_standard_contracts_and_runs_all_builtins(monkeypatch):
    from agent_banking.tools.tool_registry import BankingToolRegistry

    for key in ("TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    candidate = {"chunk": {"chunk_id": "fixture", "source": "A-Bank.docx", "content": "Tài liệu mẫu."}}
    rag = SimpleNamespace(parse_query=parse_banking_query,
                          retriever=SimpleNamespace(retrieve=lambda **kwargs: [candidate]),
                          reranker=SimpleNamespace(rerank=lambda candidates, **kwargs: candidates),
                          config=SimpleNamespace(HYBRID_TOP_K=5, HYBRID_CANDIDATE_K=20))
    registry = BankingToolRegistry(pipeline_instance=rag)
    examples = {
        "analyze_banking_query": {"query": "Mở TKTT"},
        "calculate_loan_amortization": {"principal_vnd": 120000000, "annual_rate_percent": 12, "term_months": 12},
        "calculate_savings_profit": {"amount_vnd": 100000000, "term_months": 6},
        "check_biometric_compliance_qd2345": {"single_tx_amount_vnd": 2000000},
        "check_complain_count": {},
        "create_support_ticket": {"customer_name": "Mẫu", "issue_type": "khac", "description": "Nội dung mẫu"},
        "deflect_competitor_comparison": {"competitor_name": "Ví dụ"},
        "get_foreign_exchange_rates": {},
        "get_interest_rates": {},
        "parse_vietnamese_financial_params": {"query": "gửi 100 triệu trong 6 tháng"},
        "rate_agent_csat": {"rating_score": 4},
        "search_banking_knowledge": {"query": "thẻ A-Bank"},
        "transfer_to_agent": {},
    }
    contracts = {item["name"]: item for item in registry.get_tool_contracts()}
    assert contracts.keys() == examples.keys()
    assert len(registry.get_tool_specs()) == len(examples)
    for name, arguments in examples.items():
        contract = contracts[name]
        assert {"inputSchema", "outputSchema", "annotations"} <= contract.keys()
        assert not {"handler", "policy", "parameters"} & contract.keys()
        result = registry.execute_tool(name, arguments)
        assert result["status"] == "SUCCESS", (name, result)
    assert contracts["create_support_ticket"]["annotations"]["readOnlyHint"] is False
    assert contracts["search_banking_knowledge"]["annotations"]["readOnlyHint"] is True


def test_tool_dispatch_checks_both_schemas_and_preserves_batch_aliases(monkeypatch):
    from agent_banking.tools.tool_registry import BankingToolRegistry

    registry = BankingToolRegistry()
    name = "check_biometric_compliance_qd2345"
    legacy = registry.execute_tool(name, {"transaction_amount": "2000000", "is_new_device": "false"},
                                   legacy_batch=True)
    assert legacy["status"] == "SUCCESS"
    assert legacy["data"]["single_tx_amount_vnd"] == 2000000
    calls = []

    def invalid_output(spec, args):
        calls.append(args)
        return {"status": "SUCCESS", "single_tx_amount_vnd": "not a number"}

    monkeypatch.setattr(registry, "_invoke", invalid_output)
    for rejected in ({}, {"transaction_amount": 2000000}, {"single_tx_amount_vnd": float("nan")},
                     {"single_tx_amount_vnd": 2000000, "unknown": True}):
        assert registry.execute_tool(name, rejected)["error"] == "ToolInputError"
    assert calls == []
    result = registry.execute_tool(name, {"single_tx_amount_vnd": 2000000})
    assert result["error"] == "ToolOutputError" and "data" not in result
    assert len(calls) == 1
