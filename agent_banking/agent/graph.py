"""LangGraph routes guarded banking harness actions and session checkpoints."""

import os
from pathlib import Path
import sqlite3

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent_banking.agent.config.settings import get_setting
from agent_banking.agent.deep import BankingDeepAgent
from agent_banking.core.settings import Config
from agent_banking.tracing.tracer import fingerprint, trace_node_context


DEFAULT_SQLITE_DB = str(
    Path(__file__).resolve().parent.parent / str(get_setting("default_sqlite_db_relpath", ""))
)
ACTION_ROUTES = dict(get_setting("action_routes") or {})


def create_persistent_checkpointer(db_path: str = DEFAULT_SQLITE_DB):
    """
    Initialize SqliteSaver to persist chat history across server restarts.
    If the sqlite checkpointer is unavailable, automatically fall back to MemorySaver safely.
    """
    conn = None
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        return checkpointer
    except Exception as e:
        if conn is not None:
            conn.close()
        print(f"[Checkpointer] Fallback to MemorySaver ({type(e).__name__})")
        return MemorySaver()


def _after_decision(state: dict) -> str:
    if state.get("final_answer"):
        return "output_guardrail"
    return ACTION_ROUTES[state["react_turn"]["decision"]["kind"]]


def _trace_node(name, action):
    def traced(state):
        with trace_node_context(name) as details:
            result = action(state)
            if name == "deep_decide":
                decision = result["react_turn"].get("decision", {})
                if result.get("final_answer"):
                    decision_kind = None
                else:
                    decision_kind = decision.get("kind")
                details.update(next_node=_after_decision(result),
                               decision_kind=decision_kind,
                               error_category=result.get("error_category"))
            elif name in ("input_guardrail", "output_guardrail"):
                guard = result.get("output_guard", result.get("input_guard", {}))
                details.update(is_safe=guard.get("is_safe"), flag_count=len(guard.get("flags", [])),
                               audit_status=result.get("audit_status"))
                if name == "output_guardrail":
                    details["citation_refs"] = [fingerprint(citation.get("chunk_id")) for citation in result["citations"]]
            elif name == "answer_review":
                details.update(audit_status=result["audit_status"], review=result["reviews"][-1],
                               retries=result["hallucination_retries"], **result.get("verification", {}))
            elif name in ("prepare_interaction", "await_interaction"):
                details.update(interaction_pending=bool(result["interaction"]),
                               accepted=bool(result.get("accepted_interaction_id")),
                               audit_status=result["audit_status"])
            elif name == "context_prepare":
                details.update(history_count=len(result.get("chat_history", [])),
                               compactions=result["compactions_this_turn"])
            else:
                observations = result["react_turn"]["observations"]
                if observations:
                    observation = observations[-1]
                    details.update(action_kind=observation.get("kind"), action_status=observation.get("status"),
                                   error_code=observation.get("error_code"), retried=bool(observation.get("retried")),
                                   simulation=bool(observation.get("simulation")))
                    # Failed actions may have been supplied with arbitrary model-generated names.
                    if observation.get("status") == "SUCCESS":
                        details["action_name"] = observation.get("name")
                details.update(plan_revision=result["plan_revision"], search_count=result["search_count"],
                               evidence_count=len(result["react_turn"]["evidence"]))
            return result
    return traced


def build_banking_agent_graph(
    pipeline_instance=None,
    config=None,
    checkpointer=None,
    sqlite_db_path: str = DEFAULT_SQLITE_DB,
    nodes=None
):
    if nodes is None:
        if config is None:
            config = Config()
        if pipeline_instance is None:
            from agent_banking.pipeline import BankingRAGPipeline

            pipeline_instance = BankingRAGPipeline(config=config)
        nodes = BankingDeepAgent(pipeline_instance, config)

    if checkpointer is None:
        checkpointer = create_persistent_checkpointer(sqlite_db_path)
    durable = not isinstance(checkpointer, MemorySaver) and sqlite_db_path != ":memory:"
    workflow = StateGraph(dict)
    workflow.add_node("input_guardrail", _trace_node("input_guardrail", nodes.input_guardrail))
    workflow.add_node("context_prepare", _trace_node("context_prepare", nodes.prepare_context))
    workflow.add_node("deep_decide", _trace_node("deep_decide", nodes.decide))
    actions = {
        "react_action": nodes.act,
        "skill_inspect": nodes.inspect_skill,
        "plan_update": nodes.update_plan,
        "subagent": nodes.delegate,
        "memory_update": nodes.manage_memory,
    }
    for name, action in actions.items():
        workflow.add_node(name, _trace_node(name, action))
        workflow.add_edge(name, "deep_decide")
    workflow.add_node("answer_review", _trace_node("answer_review", nodes.review))

    def prepare_interaction(state):
        return nodes.prepare_interaction(state, durable=durable)

    workflow.add_node("prepare_interaction", _trace_node("prepare_interaction", prepare_interaction))
    workflow.add_node("await_interaction", _trace_node("await_interaction", nodes.await_interaction))
    workflow.add_node("output_guardrail", _trace_node("output_guardrail", nodes.output_guardrail))
    workflow.add_edge(START, "input_guardrail")

    def after_input(state):
        if state["is_safe"]:
            return "context_prepare"
        return "output_guardrail"

    def after_review(state):
        if state["audit_status"] == "RETRY":
            return "deep_decide"
        return "output_guardrail"

    def after_prepare_interaction(state):
        if state["interaction"]:
            return "await_interaction"
        return "output_guardrail"

    def after_reply(state):
        if state["final_answer"]:
            return "output_guardrail"
        return "deep_decide"

    workflow.add_conditional_edges("input_guardrail", after_input, ["context_prepare", "output_guardrail"])
    workflow.add_edge("context_prepare", "deep_decide")
    workflow.add_conditional_edges(
        "deep_decide", _after_decision, [*actions, "answer_review", "prepare_interaction", "output_guardrail"]
    )
    workflow.add_conditional_edges("answer_review", after_review, ["deep_decide", "output_guardrail"])
    workflow.add_conditional_edges("prepare_interaction", after_prepare_interaction, ["await_interaction", "output_guardrail"])
    workflow.add_conditional_edges("await_interaction", after_reply, ["deep_decide", "output_guardrail"])
    workflow.add_edge("output_guardrail", END)
    return workflow.compile(checkpointer=checkpointer)
