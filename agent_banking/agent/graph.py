"""LangGraph routes guarded banking harness actions and session checkpoints."""

import os
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from pathlib import Path

from agent_banking.core.settings import Config
from agent_banking.agent.deep import BankingDeepAgent


DEFAULT_SQLITE_DB = str(Path(__file__).resolve().parent.parent / "tracing" / "agent_memory.db")


def create_persistent_checkpointer(db_path: str = DEFAULT_SQLITE_DB):
    """
    Initialize SqliteSaver to persist chat history across server restarts.
    If the sqlite checkpointer is unavailable, automatically fall back to MemorySaver safely.
    """
    parent_dir = os.path.dirname(db_path)
    if parent_dir != "":
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        return checkpointer
    except Exception as e:
        print(f"[Checkpointer] Fallback to MemorySaver (Reason: {e})")
        fallback_memory = MemorySaver()
        return fallback_memory


ACTION_ROUTES = {
    "tool": "react_action", "skill": "react_action",
    "plan": "plan_update", "delegate": "subagent",
    "remember": "memory_update", "forget": "memory_update", "compact": "memory_update",
    "final": "answer_review",
}


def _after_decision(state: dict) -> str:
    if state.get("final_answer"):
        return "output_guardrail"
    return ACTION_ROUTES[state["react_turn"]["decision"]["kind"]]


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

    workflow = StateGraph(dict)
    workflow.add_node("input_guardrail", nodes.input_guardrail)
    workflow.add_node("context_prepare", nodes.prepare_context)
    workflow.add_node("deep_decide", nodes.decide)
    actions = {
        "react_action": nodes.act,
        "plan_update": nodes.update_plan,
        "subagent": nodes.delegate,
        "memory_update": nodes.manage_memory,
    }
    for name, action in actions.items():
        workflow.add_node(name, action)
        workflow.add_edge(name, "deep_decide")
    workflow.add_node("answer_review", nodes.review)
    workflow.add_node("output_guardrail", nodes.output_guardrail)
    workflow.add_edge(START, "input_guardrail")

    def after_input(state):
        if state["is_safe"]:
            return "context_prepare"
        return "output_guardrail"

    def after_review(state):
        if state["audit_status"] == "RETRY":
            return "deep_decide"
        return "output_guardrail"

    workflow.add_conditional_edges("input_guardrail", after_input, ["context_prepare", "output_guardrail"])
    workflow.add_edge("context_prepare", "deep_decide")
    workflow.add_conditional_edges(
        "deep_decide", _after_decision, [*actions, "answer_review", "output_guardrail"]
    )
    workflow.add_conditional_edges("answer_review", after_review, ["deep_decide", "output_guardrail"])
    workflow.add_edge("output_guardrail", END)
    if checkpointer is None:
        checkpointer = create_persistent_checkpointer(sqlite_db_path)
    return workflow.compile(checkpointer=checkpointer)
