# -*- coding: utf-8 -*-
"""
LangGraph StateGraph Builder for Banking Agent v4.0
======================================================================
Builds and compiles the state transition graph (StateGraph) with:
- 9 End-to-end Nodes (including Node 7: audit_generation).
- Multi-Agent Supervisor Router (Assigning specialized Sub-Agents).
- Persistent SQLite Checkpointer (SqliteSaver) for persistent session history.
- Infinite Loop Protection at Grader (max 2 retries) and Audit (max 1 retry).
- 0% REGEX & ZERO TERNARY 100% compliance.
"""

from __future__ import annotations
import os
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent_banking.core.settings import Config
from agent_banking.langgraph_agent.state import AgentState
from agent_banking.langgraph_agent.nodes import BankingAgentNodes
from agent_banking.langgraph_agent.edges import (
    route_after_input_guardrail,
    route_after_router,
    route_after_grader,
    route_after_audit
)

from pathlib import Path

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


def build_banking_agent_graph(
    pipeline_instance=None,
    config=None,
    checkpointer=None,
    sqlite_db_path: str = DEFAULT_SQLITE_DB
):
    """
    Build and compile the LangGraph Agentic RAG graph version 4.0.
    """
    if config is not None:
        cfg = config
    else:
        cfg = Config()

    nodes = BankingAgentNodes(pipeline_instance=pipeline_instance, config=cfg)

    # 1. Initialize StateGraph with AgentState
    workflow = StateGraph(AgentState)

    # 2. Register all 9 Nodes
    workflow.add_node("input_guardrail", nodes.input_guardrail_node)
    workflow.add_node("router", nodes.supervisor_router_node)
    workflow.add_node("retrieve", nodes.retrieve_node)
    workflow.add_node("grade_documents", nodes.grade_documents_node)
    workflow.add_node("rewrite_query", nodes.rewrite_query_node)
    workflow.add_node("banking_tool", nodes.banking_tool_node)
    workflow.add_node("generate", nodes.generate_node)
    workflow.add_node("audit_generation", nodes.audit_generation_node)
    workflow.add_node("output_guardrail", nodes.output_guardrail_node)

    # 3. Connect flow (Edges & Conditional Edges)
    workflow.add_edge(START, "input_guardrail")

    # Node 0 -> Node 1 (Router) or Node 8 (Output Guardrail if rejected)
    workflow.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {
            "safe": "router",
            "unsafe": "output_guardrail"
        }
    )

    # Node 1 (Supervisor Router) -> 3 branches
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "chit_chat": "generate",
            "banking_action": "banking_tool",
            "rag_search": "retrieve"
        }
    )

    # RAG branch: Retrieve -> Document Grader
    workflow.add_edge("retrieve", "grade_documents")

    # Node 3 (Grade Documents) -> Generate or Rewrite
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grader,
        {
            "relevant": "generate",
            "rewrite": "rewrite_query",
            "max_retries_reached": "generate"
        }
    )

    # Node 4 (Rewrite) -> Re-search (Retrieve)
    workflow.add_edge("rewrite_query", "retrieve")

    # Node 5 (Banking Tool) -> Synthesize results at Generate
    workflow.add_edge("banking_tool", "generate")

    # Node 6 (Generate) -> Node 7 (Audit Generation)
    workflow.add_edge("generate", "audit_generation")

    # Node 7 (Audit Generation) -> Regenerate or Complete (Output Guardrail)
    workflow.add_conditional_edges(
        "audit_generation",
        route_after_audit,
        {
            "regenerate": "generate",
            "output_guardrail": "output_guardrail"
        }
    )

    # Node 8 (Output Guardrail) -> END
    workflow.add_edge("output_guardrail", END)

    # 4. Resolve Checkpointer
    active_checkpointer = checkpointer
    if active_checkpointer is None:
        active_checkpointer = create_persistent_checkpointer(sqlite_db_path)

    compiled_graph = workflow.compile(checkpointer=active_checkpointer)
    return compiled_graph
