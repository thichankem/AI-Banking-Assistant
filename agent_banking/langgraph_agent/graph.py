"""LangGraph orchestration for the banking ReAct loop."""

import os
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from pathlib import Path

from agent_banking.core.settings import Config
from agent_banking.langgraph_agent.react import BankingReActNodes


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


def _after_decision(state: dict) -> str:
    if state.get("final_answer"):
        return "output_guardrail"
    if state["react_turn"]["decision"]["kind"] == "final":
        return "audit_generation"
    return "react_action"


def build_banking_agent_graph(
    pipeline_instance=None,
    config=None,
    checkpointer=None,
    sqlite_db_path: str = DEFAULT_SQLITE_DB
):
    if config is None:
        config = Config()
    if pipeline_instance is None:
        from agent_banking.pipeline import BankingRAGPipeline

        pipeline_instance = BankingRAGPipeline(config=config)
    nodes = BankingReActNodes(pipeline_instance, config)
    workflow = StateGraph(dict)
    workflow.add_node("input_guardrail", nodes.input_guardrail)
    workflow.add_node("react_decide", nodes.decide)
    workflow.add_node("react_action", nodes.act)
    workflow.add_node("audit_generation", nodes.audit)
    workflow.add_node("output_guardrail", nodes.output_guardrail)
    workflow.add_edge(START, "input_guardrail")

    def after_input(state):
        if state["is_safe"]:
            return "react_decide"
        return "output_guardrail"

    def after_audit(state):
        if state["audit_status"] == "RETRY":
            return "react_decide"
        return "output_guardrail"

    workflow.add_conditional_edges("input_guardrail", after_input, ["react_decide", "output_guardrail"])
    workflow.add_conditional_edges(
        "react_decide", _after_decision, ["react_action", "audit_generation", "output_guardrail"]
    )
    workflow.add_edge("react_action", "react_decide")
    workflow.add_conditional_edges("audit_generation", after_audit, ["react_decide", "output_guardrail"])
    workflow.add_edge("output_guardrail", END)
    if checkpointer is None:
        checkpointer = create_persistent_checkpointer(sqlite_db_path)
    return workflow.compile(checkpointer=checkpointer)
