# -*- coding: utf-8 -*-
"""
Conditional Edges for Banking LangGraph Workflow v4.0
======================================================================
Defines all conditional routing functions:
1. route_after_input_guardrail: Intercept security risks right at input.
2. route_after_router: Branch by Intent & Sub-Agent Role (CHIT_CHAT, BANKING_ACTION, RAG_SEARCH).
3. route_after_grader: Evaluate documents (Corrective RAG) or proceed to Generate (prevent infinite loops with max_retries).
4. route_after_audit: Post-generation hallucination and citation audit.

RULES:
- 0% REGEX LAW: Absolutely no regular expressions.
- ZERO TERNARY: Explicit if/else only.
- INFINITE LOOP PROTECTION: Strict control over retry limits.
"""

from __future__ import annotations
from agent_banking.langgraph_agent.state import AgentState


def route_after_input_guardrail(state: AgentState) -> str:
    """
    Route after input guardrail check:
    - If unsafe (Prompt Injection, Jailbreak, data violation): Go directly to output_guardrail.
    - If safe: Proceed to router for intent classification.
    """
    is_safe = state.get("is_safe", True)
    if is_safe is False:
        return "unsafe"

    input_guard = state.get("input_guard", {})
    if input_guard.get("is_safe", True) is False:
        return "unsafe"

    return "safe"


def route_after_router(state: AgentState) -> str:
    """
    Route after Intent Router:
    - RAG_SEARCH: Search regulations and policies.
    - BANKING_ACTION: Call banking tools (rates, FX, interest calculation, amortization, biometrics, ticket).
    - CHIT_CHAT: Greetings and casual conversation.
    """
    intent = state.get("intent", "RAG_SEARCH")
    if intent == "BANKING_ACTION":
        return "banking_action"
    elif intent == "CHIT_CHAT":
        return "chit_chat"
    else:
        return "rag_search"


# Backward compatibility alias
route_by_intent = route_after_router


def route_after_grader(state: AgentState) -> str:
    """
    Route after Document Grader (Corrective RAG):
    - If document is RELEVANT: Forward to generate.
    - If document is IRRELEVANT:
        * If retry_count < max_retries: Forward to rewrite_query to retry search.
        * If max_retries reached: Forward to generate with safe fallback.
    """
    quality = state.get("doc_quality")
    if quality is None:
        quality = state.get("document_quality", "RELEVANT")

    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 2))

    if quality == "RELEVANT":
        return "relevant"

    if retry_count < max_retries:
        return "rewrite"
    else:
        return "max_retries_reached"


# Backward compatibility alias
route_after_grading = route_after_grader


def route_after_audit(state: AgentState) -> str:
    """
    Route after Node 7 (audit_generation).
    Strictly follows Infinite Loop Protection (maximum 1 retry).
    """
    status = state.get("audit_status", "PASS")
    retries = int(state.get("hallucination_retries", 0))

    if status == "FAIL":
        if retries <= 1:
            return "regenerate"

    return "output_guardrail"
