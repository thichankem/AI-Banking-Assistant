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
from agent_banking.tracing.agent_trace import AgentRun, agent_span

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
        return AgentRun(None, entrypoint, session_id).capture()

    def _snapshot(self, config):
        with agent_span("checkpoint.load"):
            return self.graph.get_state(config)

    def _initial_turn(self, run, snapshot, query, session_id, persona, tone, format_type, custom_persona):
        with agent_span("request.prepare") as details:
            state = self.build_initial_state(query, session_id, persona, tone, format_type, custom_persona)
            details["input_safe"] = state["screened_input"]["is_safe"]
            if run.turn_id:
                state["agent_turn_id"] = run.turn_id
            self._restore_history(state, snapshot.values)
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

    def resume(self, session_id: str, interaction_id: str, choice_id: str, text: str | None = None) -> dict:
        run = None
        try:
            with self._trace("resume", session_id) as run:
                return run.result(self._resume(run, session_id, interaction_id, choice_id, text))
        except Exception as error:
            if run is not None and run.trace_id:
                error.trace_id = run.trace_id
            raise

    def _resume(self, run, session_id, interaction_id, choice_id, text):
        started = time.perf_counter()
        config = self._thread_config(session_id)
        with self._session_lock(session_id):
            snapshot = self._snapshot(config)
            run.link_turn(snapshot.values)
            if not snapshot.interrupts or snapshot.interrupts[0].value["id"] != interaction_id:
                raise InteractionError(409, _MESSAGES["stale_interaction_error"])
            pending = snapshot.interrupts[0]
            if not any(option["id"] == choice_id for option in pending.value["options"]):
                raise InteractionError(422, _MESSAGES["invalid_choice_error"])
            reply = {"interaction_id": interaction_id, "choice_id": choice_id}
            if choice_id == "other":
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
            self.graph.invoke(Command(resume={pending.id: reply}), config=config, durability="sync")
            snapshot = self._snapshot(config)
            if snapshot.interrupts:
                return self._waiting(snapshot, session_id)
        return self._format_result(snapshot.values, session_id, round((time.perf_counter() - started) * 1000, 2))

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
            return {
                "answer": reply.sanitized_answer,
                "citations": [],
                "sources": [],
                "intent": "CHIT_CHAT",
                "retry_count": 0,
                "audit_status": "PASS",
                "evaluation_metrics": None,
                "session_id": session_id,
                "latency_ms": 0.0,
                "execution_steps": [],
                "nodes": {}
            }

        # Configure Thread ID for Checkpointer (session memory)
        thread_config = self._thread_config(session_id)

        try:
            with self._session_lock(session_id):
                snapshot = self._snapshot(thread_config)
                if snapshot.interrupts:
                    run.link_turn(snapshot.values)
                    return self._waiting(snapshot, session_id)
                initial_state = self._initial_turn(run, snapshot, query, session_id, persona, tone, format_type, custom_persona)
                final_state = self.graph.invoke(initial_state, config=thread_config, durability="sync")
                if "__interrupt__" in final_state:
                    return self._waiting(self._snapshot(thread_config), session_id)
        except Exception as e:
            run.details["error_type"] = type(e).__name__
            print(f"[LangGraph Execution Error]: {type(e).__name__}")
            latency_err = round((time.perf_counter() - start_time) * 1000.0, 2)
            reply = self.runtime.guardrails.guard_output(FALLBACK_ANSWER)
            return {
                "answer": reply.sanitized_answer,
                "citations": [],
                "sources": [],
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
        return self._format_result(final_state, session_id, total_latency_ms)

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
            with self._session_lock(session_id):
                if stop_event.is_set():
                    run.outcome = "disconnected"
                    return
                snapshot = self._snapshot(thread_config)
                if snapshot.interrupts:
                    run.link_turn(snapshot.values)
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
                                public["review_outcomes"] = node_state["reviews"]
                                public["verification"] = node_state.get("verification", {})
                                public["memory"] = node_state["memory_status"]
                                run.result(public)
                            progress[node] = public
                        loop.call_soon_threadsafe(queue.put_nowait, ("event", progress))
                if paused and not stop_event.is_set():
                    snapshot = self._snapshot(thread_config)
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
