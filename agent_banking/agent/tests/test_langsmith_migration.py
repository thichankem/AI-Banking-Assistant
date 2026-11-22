"""Verify debug trace links without making LangSmith requests."""

from unittest.mock import Mock

from langsmith import RunTree

from agent_banking.tracing.tracer import LangSmithTraceRun


def test_remote_debug_trace_resolves_and_caches_url_on_root_run(monkeypatch):
    trace = LangSmithTraceRun(None, "chat")
    trace.is_remote = True
    trace.root_run_tree = RunTree(name="agent.chat", ls_client=trace.client, project_name="fixture-project")
    url = "https://smith.langchain.com/o/fixture/projects/p/project/r/run"
    resolve = Mock(return_value=url)
    monkeypatch.setattr(RunTree, "_resolve_url", resolve)

    assert trace.build_debug_record(is_active=True)["langsmith_run_url"] == url
    assert trace.build_debug_record()["langsmith_run_url"] == url
    resolve.assert_called_once_with()


def test_trace_link_lookup_is_optional_and_never_runs_offline():
    trace = LangSmithTraceRun(None, "chat")
    trace.root_run_tree = Mock(spec=RunTree)
    trace.root_run_tree.get_url.side_effect = TimeoutError()

    assert trace.build_debug_record()["langsmith_run_url"] is None
    trace.root_run_tree.get_url.assert_not_called()

    trace.is_remote = True
    assert trace.build_debug_record()["langsmith_run_url"] is None
    trace.root_run_tree.get_url.assert_called_once_with()
