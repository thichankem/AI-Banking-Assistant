from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from agent_banking.agent.tests.test_deep_agent import approval, final, make_agent, run
from agent_banking.tools.extensions import legal_web_search as legal

QUERY = "Thông tư 17/2024/TT-NHNN mở tài khoản thanh toán"
URL = "https://thuvienphapluat.vn/van-ban/Tien-te-Ngan-hang/Thong-tu-17-2024-TT-NHNN-615776.aspx"
EXCERPT = "Thỏa thuận mở tài khoản thanh toán phải ghi rõ quyền và nghĩa vụ của chủ tài khoản."


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-brave")
    monkeypatch.setenv("LEGAL_WEB_SEARCH_PRIMARY", "tavily")
    for provider in legal.PROVIDERS:
        monkeypatch.delenv(f"LEGAL_WEB_SEARCH_{provider.upper()}_MONTHLY_LIMIT", raising=False)
    monkeypatch.setattr(legal, "_QUOTA_DB", tmp_path / "usage.sqlite3")


def _row(url=URL, content=EXCERPT, title=QUERY):
    return {"url": url, "title": title, "content": content, "description": content}


@pytest.mark.parametrize("primary_status", ["http_429", "provider_error", "no_documents"])
def test_provider_fallback_with_real_request_contracts(configured, monkeypatch, primary_status):
    calls = []

    class Opener:
        def open(self, request, timeout):
            parsed = urlsplit(request.full_url)
            calls.append((parsed.hostname, timeout))
            if parsed.hostname == "api.tavily.com":
                assert request.get_header("Authorization") == "Bearer test-tavily"
                payload = json.loads(request.data)
                assert payload["search_depth"] == "basic" and payload["auto_parameters"] is False
                assert payload["include_domains"] == [legal.DOMAIN]
                if primary_status == "http_429":
                    raise HTTPError(request.full_url, 429, "Rate limit", None, None)
                if primary_status == "provider_error":
                    raise TimeoutError()
                return BytesIO(b'{"results": []}')
            assert request.get_header("X-subscription-token") == "test-brave"
            assert parse_qs(parsed.query)["q"] == [f"{QUERY} site:{legal.DOMAIN}/van-ban/"]
            return BytesIO(json.dumps({"web": {"results": [_row()]}}).encode())

    monkeypatch.setattr(legal, "build_opener", lambda _: Opener())
    result = legal.search_legal_web(QUERY)
    assert calls == [("api.tavily.com", 6), ("api.search.brave.com", 6)]
    assert result["attempts"] == [
        {"provider": "tavily", "status": primary_status}, {"provider": "brave", "status": "ok"},
    ]
    assert result["documents"][0]["source"] == URL


def test_only_matching_legal_pages_reach_agent(configured, monkeypatch):
    calls = []

    def provider_results(provider, query, key):
        calls.append(provider)
        return [
            _row(URL.replace(legal.DOMAIN, f"{legal.DOMAIN}.example")),
            _row(URL.replace(legal.DOMAIN, f"{legal.DOMAIN}@example.com")),
            _row(URL.replace("/van-ban/", "/phap-luat/")),
            _row(URL.replace("/van-ban/", "/van-ban/%2e%2e/")),
            _row(URL.replace("17-2024", "18-2024").replace("615776", "615582"), title="Thông tư 18/2024/TT-NHNN"),
            _row(URL + "?tab=1"),
            _row(URL.replace(legal.DOMAIN, "m." + legal.DOMAIN).lower()),
        ]

    monkeypatch.setattr(legal, "_provider_results", provider_results)
    for query in ("giá vàng", f"{QUERY} site:example.com", f"{QUERY} user@example.com",
                  f"{QUERY} 090 123 4567", f"{QUERY} OTP 123456"):
        assert legal.search_legal_web(query)["status"] == "invalid_legal_query"
    assert not calls
    assert [doc["source"] for doc in legal.search_legal_web(QUERY)["documents"]] == [URL]
    assert calls == ["tavily"]


def test_concurrent_monthly_budget_is_independent_and_survives_new_connections(configured, monkeypatch):
    monkeypatch.setenv("LEGAL_WEB_SEARCH_TAVILY_MONTHLY_LIMIT", "3")
    with ThreadPoolExecutor(max_workers=6) as pool:
        reservations = list(pool.map(legal._reserve_request, ["tavily"] * 12))
    assert sum(reservations) == 3
    assert not legal._reserve_request("tavily")
    assert legal._reserve_request("brave")
    calls = []

    def provider_results(provider, query, key):
        calls.append(provider)
        return [_row()]

    monkeypatch.setattr(legal, "_provider_results", provider_results)
    assert legal.search_legal_web(QUERY)["attempts"][0] == {"provider": "tavily", "status": "monthly_limit"}
    assert calls == ["brave"]


def test_configured_primary_reaches_agent_and_is_returned_with_citation(configured, monkeypatch, tmp_path):
    monkeypatch.delenv("LEGAL_WEB_SEARCH_PRIMARY", raising=False)
    called = []

    def provider_results(provider, query, key):
        called.append((provider, query))
        return [_row()]

    monkeypatch.setattr(legal, "_provider_results", provider_results)
    script = [
        {"kind": "tool", "name": legal.TOOL_NAME, "arguments": {"query": QUERY}},
        final(f"{EXCERPT} [Nguồn 1]"), approval(),
    ]
    _, graph, _ = make_agent(monkeypatch, tmp_path / "chat.sqlite3", script)
    try:
        state = run(graph, QUERY)
        assert state["audit_status"] == "PASS"
        assert state["citations"][0]["source_url"] == URL
        assert state["tool_outputs"][0]["provider"] == "brave"
        assert called == [("brave", QUERY)]
    finally:
        graph.checkpointer.conn.close()


def test_article_lookup_prefers_the_provider_that_returns_the_requested_article(configured, monkeypatch):
    monkeypatch.delenv("LEGAL_WEB_SEARCH_PRIMARY", raising=False)

    def provider_results(provider, query, key):
        article = 16 if provider == "tavily" else 160
        return [_row(content=f"Điều {article}. {EXCERPT}")]

    monkeypatch.setattr(legal, "_provider_results", provider_results)
    result = legal.search_legal_web(f"{QUERY} Điều 16")
    assert result["provider"] == "tavily"
    assert result["attempts"] == [
        {"provider": "brave", "status": "incomplete_article_evidence"},
        {"provider": "tavily", "status": "ok"},
    ]
