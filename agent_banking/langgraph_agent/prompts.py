# -*- coding: utf-8 -*-
"""
System Prompts for Banking LangGraph Workflow
Exposes prompt definitions imported from the module config directory.
"""

from agent_banking.langgraph_agent.config.prompts import (
    BANK_NAME,
    ROUTER_SYSTEM_PROMPT,
    DOCUMENT_GRADER_SYSTEM_MESSAGE,
    DOCUMENT_GRADER_PROMPT,
    QUERY_REWRITER_SYSTEM_MESSAGE,
    QUERY_REWRITER_PROMPT,
    ANSWER_GRADER_SYSTEM_MESSAGE,
    ANSWER_GRADER_PROMPT,
    BANKING_AGENT_SYSTEM_PROMPT,
    GENERATE_USER_PROMPT_TEMPLATE,
    SUB_AGENT_SYSTEM_PROMPTS
)
