# -*- coding: utf-8 -*-
"""Banking deep agent entrypoint for synchronous chat and SSE."""

import sys
import time
from threading import Lock
from pathlib import Path
from langgraph.types import Command

# Ensure UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.core.config.settings import CoreConfig, get_core_messages
from agent_banking.core.settings import Config
from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.agent.graph import build_banking_agent_graph
from agent_banking.agent.deep import BankingDeepAgent
from agent_banking.agent.react import FALLBACK_ANSWER
from agent_banking.tracing.tracer import LangSmithTraceRun, trace_span

_AP_CFG = CoreConfig.CORE_AGENT_PIPELINE
_MESSAGES = get_core_messages("agent_pipeline", "pipeline_messages.yaml", "AGENT_PIPELINE_MESSAGES")
_NODE_LABELS = dict(_MESSAGES["node_labels"])

_SESSION_LOCK_BUCKETS = int(_AP_CFG["session_lock_buckets"])
_RECURSION_LIMIT = int(_AP_CFG["recursion_limit"])
_CUSTOM_PERSONA_MAX_CHARS = int(_AP_CFG["custom_persona_max_chars"])
_MAX_RESUME_REPLY_CHARS = int(_AP_CFG["max_resume_reply_chars"])
_DEFAULT_SESSION_ID = str(_AP_CFG["default_session_id"])
_DEFAULT_PERSONA = str(_AP_CFG["default_persona"])
_DEFAULT_TONE = str(_AP_CFG["default_tone"])
_DEFAULT_FORMAT_TYPE = str(_AP_CFG["default_format_type"])
_DEFAULT_CUSTOM_PERSONA = str(_AP_CFG["default_custom_persona"])
_STREAM_ACTION_NODES = tuple(_AP_CFG["stream_action_nodes"])
_OUTPUT_GUARDRAIL_KEYS = tuple(_AP_CFG["output_guardrail_keys"])


def _default_sqlite_db_path():
    target = PROJECT_ROOT
    for part in str(_AP_CFG["default_sqlite_relpath"]).replace("\\", "/").split("/"):
        if part:
            target = target / part
    return str(target)


class InteractionError(ValueError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class BankingAgentPipeline:
    """
    Enterprise AI Banking Agent orchestrator using LangGraph StateGraph.
    """

    def __init__(
        self,
        config: Config | None = None,
        use_memory: bool = False,
        sqlite_db_path: str | None = None,
        rag_pipeline: BankingRAGPipeline | None = None,
        checkpointer = None
    ):
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.use_memory = use_memory
        if sqlite_db_path is not None:
            self.sqlite_db_path = sqlite_db_path
        else:
            self.sqlite_db_path = _default_sqlite_db_path()
        self.checkpointer = checkpointer

        print("=" * 70)
        print(_MESSAGES["init_banner"])
        print("=" * 70)

        # 1. Initialize foundation RAG Pipeline (Dual Index Vector + BM25)
        self.rag_pipeline = rag_pipeline
        if self.rag_pipeline is None:
            self.rag_pipeline = BankingRAGPipeline(config=self.config, use_memory=self.use_memory)
        self._session_locks = [Lock() for _ in range(_SESSION_LOCK_BUCKETS)]

        # 2. Initialize the ReAct graph and session checkpoint
        self.runtime = BankingDeepAgent(self.rag_pipeline, self.config)
        self.graph = build_banking_agent_graph(
            nodes=self.runtime,
            sqlite_db_path=self.sqlite_db_path,
            checkpointer=self.checkpointer
        )
        print(_MESSAGES["graph_initialized"])

        self.tracer = self.rag_pipeline.tracer

    def build_initial_state(
        self,
        query: str,
        session_id: str = _DEFAULT_SESSION_ID,
        persona: str = _DEFAULT_PERSONA,
        tone: str = _DEFAULT_TONE,
        format_type: str = _DEFAULT_FORMAT_TYPE,
        custom_persona: str = _DEFAULT_CUSTOM_PERSONA
    ) -> dict:
        """Build the initial StateGraph payload shared by sync and streaming chat entrypoints."""
        clean_query = str(query).strip()
        screened = self.runtime.screen_input(clean_query)
        return {
            "user_query": screened["query"],
            "screened_input": screened,
            "session_id": session_id,
            "persona": persona,
            "tone": tone,
            "format_type": format_type,
            "custom_persona": self.runtime.guard_artifact(str(custom_persona or "")[:_CUSTOM_PERSONA_MAX_CHARS]),
        }

    def _session_lock(self, session_id: str):
        return self._session_locks[hash(session_id) % len(self._session_locks)]

    def _trace(self, entrypoint, session_id):
        if hasattr(self.tracer, "trace_agent"):
            return self.tracer.trace_agent(entrypoint, session_id)
        return LangSmithTraceRun(None, entrypoint, session_id).capture()

    def _snapshot(self, config):
        with trace_span("checkpoint.load"):
            return self.graph.get_state(config)

    def _initial_turn(self, run, snapshot, query, session_id, persona, tone, format_type, custom_persona):
        with trace_span("request.prepare") as details:
            state = self.build_initial_state(query, session_id, persona, tone, format_type, custom_persona)
            details["input_safe"] = state["screened_input"]["is_safe"]
            if run.turn_id:
                state["agent_turn_id"] = run.turn_id
            self._restore_history(state, snapshot.values)
            if hasattr(run, "attach_debug_snapshot"):
                run.attach_debug_snapshot(
                    self._build_debug_snapshot(
                        state,
                        query=query,
                        persona=persona,
                        tone=tone,
                        format_type=format_type,
                    )
                )
            return state

    @staticmethod
    def _thread_config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}, "recursion_limit": _RECURSION_LIMIT}

    @staticmethod
    def _waiting(snapshot, session_id: str) -> dict:
        response = {
            "status": "awaiting_input",
            "session_id": session_id,
            "answer": "",
            "interaction": snapshot.interrupts[0].value,
            "plan": snapshot.values.get("plan", []),
            "plan_revision": snapshot.values.get("plan_revision", 0),
        }
        if snapshot.values.get("agent_turn_id"):
            response["turn_id"] = snapshot.values["agent_turn_id"]
        return response

    def status(self, session_id: str) -> dict:
        with self._session_lock(session_id):
            snapshot = self.graph.get_state(self._thread_config(session_id))
            if snapshot.interrupts:
                return self._waiting(snapshot, session_id)
            if snapshot.values.get("accepted_interaction_id") and not snapshot.next:
                return self._format_result(snapshot.values, session_id, 0.0)
        return {"status": "idle", "session_id": session_id}

    def resume(
        self,
        session_id: str,
        interaction_id: str,
        choice_id: str | None = None,
        text: str | None = None,
        choice_ids: list[str] | None = None,
    ) -> dict:
        run = None
        try:
            with self._trace("resume", session_id) as run:
                return run.result(self._resume(run, session_id, interaction_id, choice_id, text, choice_ids))
        except Exception as error:
            if run is not None and run.trace_id:
                error.trace_id = run.trace_id
            raise

    def _resume(self, run, session_id, interaction_id, choice_id, text, choice_ids=None):
        started = time.perf_counter()
        config = self._thread_config(session_id)
        with self._session_lock(session_id):
            snapshot = self._snapshot(config)
            run.link_turn(snapshot.values)
            if not snapshot.interrupts or snapshot.interrupts[0].value["id"] != interaction_id:
                raise InteractionError(409, _MESSAGES["stale_interaction_error"])
            pending = snapshot.interrupts[0]
            selected_ids = []
            if choice_ids is not None:
                if not isinstance(choice_ids, list) or not choice_ids:
                    raise InteractionError(422, _MESSAGES["invalid_choice_error"])
                for item in choice_ids:
                    if not isinstance(item, str) or not item:
                        raise InteractionError(422, _MESSAGES["invalid_choice_error"])
                    if item not in selected_ids:
                        selected_ids.append(item)
                if choice_id is not None and choice_id not in selected_ids:
                    raise InteractionError(422, _MESSAGES["invalid_choice_error"])
            elif isinstance(choice_id, str) and choice_id:
                selected_ids = [choice_id]
            else:
                raise InteractionError(422, _MESSAGES["invalid_choice_error"])

            valid_ids = {option["id"] for option in pending.value["options"]}
            if any(cid not in valid_ids for cid in selected_ids):
                raise InteractionError(422, _MESSAGES["invalid_choice_error"])
            is_multi = pending.value.get("multi_select") is True
            if len(selected_ids) > 1:
                if not is_multi:
                    err_msg = _MESSAGES.get("single_choice_only_error")
                    if not err_msg:
                        err_msg = _MESSAGES["invalid_choice_error"]
                    raise InteractionError(422, err_msg)
                if "cancel" in selected_ids or "reject" in selected_ids:
                    raise InteractionError(422, _MESSAGES["invalid_choice_error"])

            reply = {"interaction_id": interaction_id, "choice_id": selected_ids[0], "choice_ids": selected_ids}
            if "other" in selected_ids:
                if not text or not text.strip() or len(text) > _MAX_RESUME_REPLY_CHARS:
                    raise InteractionError(422, _MESSAGES["invalid_free_text_error"])
                screened = self.runtime.screen_input(text)
                if not screened["is_safe"]:
                    raise InteractionError(422, _MESSAGES["unsafe_reply_error"])
                guarded = self.runtime.guard_artifact(screened["query"])
                if not guarded:
                    raise InteractionError(422, _MESSAGES["unsafe_reply_error"])
                reply["text"] = guarded
            elif text is not None:
                raise InteractionError(422, _MESSAGES["other_only_text_error"])
            if hasattr(run, "attach_debug_snapshot"):
                pre_latency = round((time.perf_counter() - started) * 1000, 2)
                pre_nodes = self._format_result(snapshot.values, session_id, pre_latency).get("nodes")
                run.attach_debug_snapshot(
                    self._build_debug_snapshot(
                        snapshot.values,
                        query=str(snapshot.values.get("user_query") or "[resume]"),
                        latency_ms=pre_latency,
                        nodes_detail=pre_nodes,
                    )
                )
            self.graph.invoke(Command(resume={pending.id: reply}), config=config, durability="sync")
            snapshot = self._snapshot(config)
            if snapshot.interrupts:
                waiting_res = self._waiting(snapshot, session_id)
                paused_latency = round((time.perf_counter() - started) * 1000, 2)
                paused_nodes = self._format_result(snapshot.values, session_id, paused_latency).get("nodes")
                if hasattr(run, "attach_debug_snapshot"):
                    run.attach_debug_snapshot(
                        self._build_debug_snapshot(
                            snapshot.values,
                            latency_ms=paused_latency,
                            nodes_detail=paused_nodes,
                        )
                    )
                return waiting_res
        res = self._format_result(snapshot.values, session_id, round((time.perf_counter() - started) * 1000, 2))
        if hasattr(run, "attach_debug_snapshot"):
            run.attach_debug_snapshot(
                self._build_debug_snapshot(
                    snapshot.values,
                    latency_ms=res.get("latency_ms"),
                    nodes_detail=res.get("nodes"),
                )
            )
        return res

    def _restore_history(self, state: dict, previous: dict) -> None:
        # The graph uses a root dict channel, which is replaced by each new request.
        history = previous.get("chat_history", [])
        memory = previous.get("session_memory")
        if memory is None:
            # Import the latest history from pre-harness checkpoints once.
            history = [
                {"role": item["role"], "content": self.runtime.guard_artifact(item["content"])}
                for item in history if item.get("role") in ("user", "assistant")
            ]
            history = [item for item in history if item["content"]]
        else:
            state["session_memory"] = memory
        state["chat_history"] = history

    def chat(
        self,
        query: str,
        session_id: str = _DEFAULT_SESSION_ID,
        persona: str = _DEFAULT_PERSONA,
        tone: str = _DEFAULT_TONE,
        format_type: str = _DEFAULT_FORMAT_TYPE,
        custom_persona: str = _DEFAULT_CUSTOM_PERSONA
    ) -> dict:
        """
        Execute chat turn with LangGraph Agent and persistent session memory (Thread ID).
        """
        with self._trace("chat", session_id) as run:
            return run.result(self._chat(run, query, session_id, persona, tone, format_type, custom_persona))

    def _chat(self, run, query, session_id, persona, tone, format_type, custom_persona):
        start_time = time.perf_counter()
        clean_query = query.strip()

        if clean_query == "":
            run.details["empty_input"] = True
            reply = self.runtime.guardrails.guard_output(_MESSAGES["empty_query_prompt"])
            empty_res = {
                "answer": reply.sanitized_answer,
                "citations": [],
                "sources": [],
                "images": [],
                "intent": "CHIT_CHAT",
                "retry_count": 0,
                "audit_status": "PASS",
                "evaluation_metrics": None,
                "session_id": session_id,
                "latency_ms": 0.0,
                "execution_steps": [],
                "nodes": {}
            }
            if hasattr(run, "attach_debug_snapshot"):
                run.attach_debug_snapshot(
                    self._build_debug_snapshot(
                        {
                            "user_query": query,
                            "persona": persona,
                            "tone": tone,
                            "format_type": format_type,
                            "intent": "CHIT_CHAT",
                            "audit_status": "PASS",
                        },
                        query=query,
                        persona=persona,
                        tone=tone,
                        format_type=format_type,
                        latency_ms=0.0,
                    )
                )
            return empty_res

        # Configure Thread ID for Checkpointer (session memory)
        thread_config = self._thread_config(session_id)

        try:
            with self._session_lock(session_id):
                snapshot = self._snapshot(thread_config)
                if snapshot.interrupts:
                    run.link_turn(snapshot.values)
                    paused_latency = round((time.perf_counter() - start_time) * 1000.0, 2)
                    paused_nodes = self._format_result(snapshot.values, session_id, paused_latency).get("nodes")
                    if hasattr(run, "attach_debug_snapshot"):
                        run.attach_debug_snapshot(
                            self._build_debug_snapshot(
                                snapshot.values,
                                query=query,
                                persona=persona,
                                tone=tone,
                                format_type=format_type,
                                latency_ms=paused_latency,
                                nodes_detail=paused_nodes,
                            )
                        )
                    return self._waiting(snapshot, session_id)
                initial_state = self._initial_turn(run, snapshot, query, session_id, persona, tone, format_type, custom_persona)
                final_state = self.graph.invoke(initial_state, config=thread_config, durability="sync")
                if "__interrupt__" in final_state:
                    paused_snapshot = self._snapshot(thread_config)
                    paused_latency = round((time.perf_counter() - start_time) * 1000.0, 2)
                    paused_nodes = self._format_result(paused_snapshot.values, session_id, paused_latency).get("nodes")
                    if hasattr(run, "attach_debug_snapshot"):
                        run.attach_debug_snapshot(
                            self._build_debug_snapshot(
                                paused_snapshot.values,
                                query=query,
                                persona=persona,
                                tone=tone,
                                format_type=format_type,
                                latency_ms=paused_latency,
                                nodes_detail=paused_nodes,
                            )
                        )
                    return self._waiting(paused_snapshot, session_id)
        except Exception as e:
            run.details["error_type"] = type(e).__name__
            print(f"[LangGraph Execution Error]: {type(e).__name__}")
            latency_err = round((time.perf_counter() - start_time) * 1000.0, 2)
            reply = self.runtime.guardrails.guard_output(FALLBACK_ANSWER)
            if hasattr(run, "attach_debug_snapshot"):
                run.attach_debug_snapshot(
                    self._build_debug_snapshot(
                        {
                            "user_query": query,
                            "persona": persona,
                            "tone": tone,
                            "format_type": format_type,
                            "intent": "ERROR",
                            "audit_status": "ERROR",
                            "error_category": type(e).__name__,
                        },
                        query=query,
                        persona=persona,
                        tone=tone,
                        format_type=format_type,
                        latency_ms=latency_err,
                    )
                )
            return {
                "answer": reply.sanitized_answer,
                "citations": [],
                "sources": [],
                "images": [],
                "intent": "ERROR",
                "retry_count": 0,
                "audit_status": "ERROR",
                "evaluation_metrics": None,
                "session_id": session_id,
                "latency_ms": latency_err,
                "execution_steps": [],
                "nodes": {}
            }

        total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        formatted = self._format_result(final_state, session_id, total_latency_ms)
        if hasattr(run, "attach_debug_snapshot"):
            run.attach_debug_snapshot(
                self._build_debug_snapshot(
                    final_state,
                    query=query,
                    persona=persona,
                    tone=tone,
                    format_type=format_type,
                    latency_ms=total_latency_ms,
                    nodes_detail=formatted.get("nodes"),
                )
            )
        return formatted

    @staticmethod
    def _extract_chunk_pages(chunk: dict) -> list:
        meta = chunk.get("business_metadata")
        if not isinstance(meta, dict):
            return []
        cit = meta.get("citation")
        if isinstance(cit, dict) and isinstance(cit.get("page_numbers"), list) and cit["page_numbers"]:
            pages = []
            for p in cit["page_numbers"]:
                if isinstance(p, int) and p > 0 and p not in pages:
                    pages.append(p)
            return sorted(pages)
        prov = meta.get("provenance")
        if not isinstance(prov, dict):
            return []
        locs = prov.get("locations")
        if not isinstance(locs, list):
            return []
        pages = []
        for loc in locs:
            if not isinstance(loc, dict):
                continue
            pnum = loc.get("page_number")
            if not isinstance(pnum, int) and isinstance(loc.get("page_index"), int):
                pnum = loc["page_index"] + 1
            if isinstance(pnum, int) and pnum > 0 and pnum not in pages:
                pages.append(pnum)
        return sorted(pages)

    def _build_debug_snapshot(
        self,
        state: dict,
        *,
        query: str = "",
        persona: str = "",
        tone: str = "",
        format_type: str = "",
        latency_ms: float | None = None,
        nodes_detail: dict | None = None,
    ) -> dict:
        """Build a bounded, sanitized turn diagnostics snapshot for the UI debug inspector."""
        if not isinstance(state, dict):
            state = {}
        raw_query = query
        if not raw_query:
            raw_query = str(state.get("user_query") or "")
        if len(raw_query) > 500:
            raw_query = raw_query[:497] + "..."

        persona_val = persona
        if not persona_val:
            persona_val = str(state.get("persona") or "")
        tone_val = tone
        if not tone_val:
            tone_val = str(state.get("tone") or "")
        format_val = format_type
        if not format_val:
            format_val = str(state.get("format_type") or "")

        chunks_preview = []
        raw_chunks = state.get("retrieved_chunks")
        if isinstance(raw_chunks, list):
            for idx, chunk in enumerate(raw_chunks[:15]):
                if not isinstance(chunk, dict):
                    continue
                excerpt = str(chunk.get("content") or chunk.get("original_text") or "").strip()
                if len(excerpt) > 240:
                    excerpt = excerpt[:237] + "..."
                chunks_preview.append({
                    "index": idx + 1,
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "source": str(chunk.get("source") or ""),
                    "section_title": str(chunk.get("section_title") or ""),
                    "breadcrumb": str(chunk.get("breadcrumb") or ""),
                    "chunk_type": str(chunk.get("chunk_type") or "text"),
                    "page_numbers": self._extract_chunk_pages(chunk),
                    "excerpt_preview": excerpt,
                })

        citations = state.get("citations")
        if isinstance(citations, list):
            citations_count = len(citations)
        else:
            citations_count = 0

        images = state.get("images")
        if isinstance(images, list):
            images_count = len(images)
        else:
            images_count = 0

        sources = state.get("sources")
        if isinstance(sources, list):
            sources_list = list(sources)
        else:
            sources_list = []

        interaction_payload = state.get("interaction")
        if not isinstance(interaction_payload, dict):
            interaction_payload = None

        raw_audit_status = state.get("audit_status")
        if interaction_payload and raw_audit_status in (None, "", "PENDING"):
            raw_audit_status = "AWAITING_INPUT"

        raw_intent = state.get("intent")
        if interaction_payload and (not raw_intent or raw_intent == "ROUTING"):
            kind_label = str(interaction_payload.get("kind") or "clarify").upper()
            raw_intent = "INTERACTION_" + kind_label

        normalized_plan = []
        for item in list(state.get("plan") or []):
            if isinstance(item, dict):
                copied_item = dict(item)
                if raw_audit_status == "PASS" and not interaction_payload:
                    if copied_item.get("status") in ("pending", "in_progress"):
                        copied_item["status"] = "done"
                normalized_plan.append(copied_item)
            else:
                normalized_plan.append(item)

        if nodes_detail is None:
            if isinstance(latency_ms, (int, float)):
                eff_latency = float(latency_ms)
            else:
                eff_latency = 0.0
            resolved_nodes = self._format_result(state, "", eff_latency).get("nodes")
        else:
            resolved_nodes = nodes_detail

        return {
            "query": raw_query,
            "persona": persona_val,
            "tone": tone_val,
            "format_style": format_val,
            "intent": raw_intent,
            "audit_status": raw_audit_status,
            "error_category": state.get("error_category"),
            "retry_count": int(state.get("retry_count") or 0),
            "hallucination_retries": int(state.get("hallucination_retries") or 0),
            "audit_feedback": str(state.get("audit_feedback") or ""),
            "input_guard": state.get("input_guard") or {"is_safe": state.get("is_safe", True), "flags": []},
            "output_guard": state.get("output_guard") or {},
            "plan": normalized_plan,
            "plan_revision": int(state.get("plan_revision") or 0),
            "execution_steps": list(state.get("execution_steps") or []),
            "tool_calls": list(state.get("tool_outputs") or []),
            "delegations": list(state.get("delegations") or []),
            "review_outcomes": list(state.get("reviews") or []),
            "verification": dict(state.get("verification") or {}),
            "memory": dict(state.get("memory_status") or {}),
            "interaction": interaction_payload,
            "human_replies": list(state.get("human_replies") or []),
            "citations_count": citations_count,
            "sources": sources_list,
            "images_count": images_count,
            "latency_ms": latency_ms,
            "nodes": resolved_nodes,
            "retrieved_chunks": chunks_preview,
        }

    def _format_result(self, final_state: dict, session_id: str, total_latency_ms: float) -> dict:
        # Extract output fields for Frontend UI Inspector
        answer = final_state.get("final_answer", "")

        citations = final_state.get("citations", [])
        sources = final_state.get("sources", [])
        intent = final_state.get("intent", "RAG_SEARCH")
        retries = int(final_state.get("retry_count", 0))
        audit_stat = final_state.get("audit_status", "PASS")
        tool_outputs = final_state.get("tool_outputs", [])
        steps = final_state.get("execution_steps", [])

        docs = final_state.get("retrieved_chunks", [])

        is_input_safe = final_state.get("is_safe", True)
        if final_state.get("input_guard", {}).get("is_safe", True) is False:
            is_input_safe = False

        status_guardrail = "blocked"
        if is_input_safe is True:
            status_guardrail = "passed"

        # Package node inspection details
        nodes_detail = {
            "00_guardrail": {
                "name": _NODE_LABELS["guardrail"],
                "is_safe": is_input_safe,
                "status": status_guardrail
            },
            "01_react": {
                "name": _NODE_LABELS["react"],
                "detected_intent": intent,
                "last_action": final_state.get("last_action")
            },
            "02_retrieval": {
                "name": _NODE_LABELS["retrieval"],
                "total_documents_found": len(docs),
                "search_count": final_state.get("search_count", 0),
                "retry_count": retries
            },
            "03_tools": {
                "name": _NODE_LABELS["tools"],
                "executed_tools": [to.get("tool") for to in tool_outputs]
            },
            "04_audit": {
                "name": _NODE_LABELS["audit"],
                "audit_status": audit_stat,
                "hallucination_retries": final_state.get("hallucination_retries", 0),
                "audit_feedback": final_state.get("audit_feedback", "")
            },
            "05_output_guard": {
                "name": _NODE_LABELS["output_guard"],
                "is_safe": final_state.get("output_guard", {}).get("is_safe", True),
                "flags": final_state.get("output_guard", {}).get("flags", [])
            }
        }

        status = "completed"
        if audit_stat == "CANCELED":
            status = "canceled"
        return {
            "status": status,
            "accepted_interaction_id": final_state.get("accepted_interaction_id"),
            "answer": answer,
            "citations": citations,
            "sources": sources,
            "images": final_state.get("images", []),
            "intent": intent,
            "retry_count": retries,
            "audit_status": audit_stat,
            "error_category": final_state.get("error_category"),
            "evaluation_metrics": None,
            "tool_calls": tool_outputs,
            "session_id": session_id,
            "latency_ms": total_latency_ms,
            "execution_steps": steps,
            "plan": final_state.get("plan", []),
            "plan_revision": final_state.get("plan_revision", 0),
            "delegations": final_state.get("delegations", []),
            "review_outcomes": final_state.get("reviews", []),
            "verification": final_state.get("verification", {}),
            "memory": final_state.get("memory_status", {}),
            "nodes": nodes_detail
        }

    def query(self, query: str) -> dict:
        """
        Backward-compatible query method.
        """
        return self.chat(query=query)

    async def astream_chat(
        self,
        query: str,
        session_id: str = _DEFAULT_SESSION_ID,
        persona: str = _DEFAULT_PERSONA,
        tone: str = _DEFAULT_TONE,
        format_type: str = _DEFAULT_FORMAT_TYPE,
        custom_persona: str = _DEFAULT_CUSTOM_PERSONA
    ):
        """
        Stream all super-step events from graph execution without blocking the event loop,
        staying compatible with synchronous checkpointers such as SqliteSaver.
        """
        import asyncio
        import contextlib
        import threading

        thread_config = self._thread_config(session_id)

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        stop_event = threading.Event()
        sentinel = object()

        def _stream(run):
            stream_started = time.perf_counter()
            with self._session_lock(session_id):
                if stop_event.is_set():
                    run.outcome = "disconnected"
                    return
                snapshot = self._snapshot(thread_config)
                if snapshot.interrupts:
                    run.link_turn(snapshot.values)
                    paused_latency = round((time.perf_counter() - stream_started) * 1000.0, 2)
                    paused_nodes = self._format_result(snapshot.values, session_id, paused_latency).get("nodes")
                    if hasattr(run, "attach_debug_snapshot"):
                        run.attach_debug_snapshot(
                            self._build_debug_snapshot(
                                snapshot.values,
                                query=query,
                                persona=persona,
                                tone=tone,
                                format_type=format_type,
                                latency_ms=paused_latency,
                                nodes_detail=paused_nodes,
                            )
                        )
                    return {"awaiting_input": run.result(self._waiting(snapshot, session_id))}
                initial_state = self._initial_turn(
                    run, snapshot, query, session_id, persona, tone, format_type, custom_persona
                )
                paused = False
                with contextlib.closing(self.graph.stream(initial_state, config=thread_config, durability="sync")) as events:
                    while not stop_event.is_set():
                        event = next(events, None)
                        if event is None:
                            break
                        if "__interrupt__" in event:
                            paused = True
                            continue
                        progress = {}
                        for node, node_state in event.items():
                            elapsed_ms = round((time.perf_counter() - stream_started) * 1000.0, 2)
                            live_nodes = None
                            if isinstance(node_state, dict) and hasattr(run, "attach_debug_snapshot"):
                                live_nodes = self._format_result(node_state, session_id, elapsed_ms).get("nodes")
                                run.attach_debug_snapshot(
                                    self._build_debug_snapshot(
                                        node_state,
                                        query=query,
                                        persona=persona,
                                        tone=tone,
                                        format_type=format_type,
                                        latency_ms=elapsed_ms,
                                        nodes_detail=live_nodes,
                                    )
                                )
                            public = {}
                            if node in _STREAM_ACTION_NODES:
                                if node_state.get("last_action"):
                                    public["last_action"] = node_state["last_action"]
                            if node == "plan_update":
                                public["plan"] = node_state["plan"]
                            if node == "answer_review":
                                public["audit_status"] = node_state["audit_status"]
                            if node == "output_guardrail":
                                for key in _OUTPUT_GUARDRAIL_KEYS:
                                    public[key] = node_state[key]
                                public["intent"] = node_state.get("intent")
                                public["error_category"] = node_state.get("error_category")
                                public["review_outcomes"] = node_state["reviews"]
                                public["verification"] = node_state.get("verification", {})
                                public["memory"] = node_state["memory_status"]
                                run.result(public)
                            if getattr(run, "trace_id", None):
                                public["trace_id"] = run.trace_id
                            progress[node] = public
                        loop.call_soon_threadsafe(queue.put_nowait, ("event", progress))
                if paused and not stop_event.is_set():
                    snapshot = self._snapshot(thread_config)
                    paused_latency = round((time.perf_counter() - stream_started) * 1000.0, 2)
                    paused_nodes = self._format_result(snapshot.values, session_id, paused_latency).get("nodes")
                    if hasattr(run, "attach_debug_snapshot"):
                        run.attach_debug_snapshot(
                            self._build_debug_snapshot(
                                snapshot.values,
                                query=query,
                                persona=persona,
                                tone=tone,
                                format_type=format_type,
                                latency_ms=paused_latency,
                                nodes_detail=paused_nodes,
                            )
                        )
                    return {"awaiting_input": run.result(self._waiting(snapshot, session_id))}
            if stop_event.is_set() and "audit_status" not in run.details:
                run.outcome = "disconnected"

        def _worker():
            run = None
            try:
                with self._trace("stream", session_id) as run:
                    terminal = _stream(run)
                if terminal is not None:
                    loop.call_soon_threadsafe(queue.put_nowait, ("event", terminal))
            except Exception as exc:
                if run is not None and run.trace_id:
                    exc.trace_id = run.trace_id
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", sentinel))

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        try:
            while True:
                kind, payload = await queue.get()
                if kind == "event":
                    yield payload
                elif kind == "error":
                    raise payload
                else:
                    break
        finally:
            stop_event.set()
