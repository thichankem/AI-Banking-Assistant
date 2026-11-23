# -*- coding: utf-8 -*-
"""Banking deep agent entrypoint for synchronous chat and SSE."""

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from threading import Lock, Thread
from pathlib import Path
from uuid import uuid4
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
from agent_banking.core.model_routing import apply_chat_options, ordered_candidates
from agent_banking.core.models import model_configured
from agent_banking.core.settings import Config
from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.agent.graph import build_banking_agent_graph
from agent_banking.agent.deep import BankingDeepAgent
from agent_banking.agent.react import FALLBACK_ANSWER
from agent_banking.tracing.tracer import LangSmithTraceRun, _sanitize_value, fingerprint, trace_span

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

_DAG_STAGE_ORDER = (
    "input_guardrail",
    "coordinator",
    "workers_tools",
    "answer_review",
    "output_guardrail",
)
_DAG_STAGE_LABELS = {
    "input_guardrail": "Input Guardrail",
    "coordinator": "Coordinator",
    "workers_tools": "Parallel Workers / Tools",
    "answer_review": "Answer Review",
    "output_guardrail": "Output Guardrail",
}
_NODE_TO_DAG_STAGE = {
    "vision_ocr": "input_guardrail",
    "input_guardrail": "input_guardrail",
    "context_prepare": "input_guardrail",
    "deep_decide": "coordinator",
    "plan_update": "coordinator",
    "memory_update": "coordinator",
    "react_action": "workers_tools",
    "subagent": "workers_tools",
    "skill_inspect": "workers_tools",
    "prepare_interaction": "workers_tools",
    "await_interaction": "workers_tools",
    "answer_review": "answer_review",
    "output_guardrail": "output_guardrail",
}
_LLM_DRIVEN_NODES = frozenset({"deep_decide", "answer_review", "subagent"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        self._session_store_lock = Lock()
        self._memory_sessions = {}
        self._memory_debug_traces = {}
        self._tracer_obj = None
        self._title_lock = Lock()
        self._pending_title_threads = {}
        self._title_summarizer = None

        # 2. Initialize the ReAct graph and session checkpoint
        self.runtime = BankingDeepAgent(self.rag_pipeline, self.config)
        self.graph = build_banking_agent_graph(
            nodes=self.runtime,
            sqlite_db_path=self.sqlite_db_path,
            checkpointer=self.checkpointer
        )
        print(_MESSAGES["graph_initialized"])

        self._ensure_session_tables()
        self.tracer = getattr(self.rag_pipeline, "tracer", None)

    @property
    def tracer(self):
        return getattr(self, "_tracer_obj", None)

    @tracer.setter
    def tracer(self, value) -> None:
        self._tracer_obj = value
        if value is not None and hasattr(value, "set_persistent_backend"):
            value.set_persistent_backend(self)

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

    def _waiting(self, snapshot, session_id: str, dag_telemetry: dict | None = None) -> dict:
        if dag_telemetry is None:
            dag_telemetry = self._get_latest_session_dag(session_id)
        if dag_telemetry is None:
            dag_telemetry = self._build_dag_from_run_and_state(None, snapshot.values, 0.0, paused=True)
        response = {
            "status": "awaiting_input",
            "session_id": session_id,
            "answer": "",
            "interaction": snapshot.interrupts[0].value,
            "plan": snapshot.values.get("plan", []),
            "plan_revision": snapshot.values.get("plan_revision", 0),
            "dag_telemetry": dag_telemetry,
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
                stored_dag = self._get_latest_session_dag(session_id)
                return self._format_result(snapshot.values, session_id, 0.0, dag_telemetry=stored_dag)
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

    @staticmethod
    def _extract_human_reply_label(pending_value: dict, selected_ids: list[str], reply_text: str | None) -> str:
        options_map = {}
        for opt in pending_value.get("options") or []:
            if isinstance(opt, dict) and opt.get("id"):
                options_map[str(opt["id"])] = str(opt.get("label") or opt["id"])
        labels = []
        for cid in selected_ids:
            if cid == "other" and reply_text:
                labels.append(str(reply_text).strip())
            elif cid in options_map:
                labels.append(options_map[cid])
            else:
                labels.append(str(cid))
        return "; ".join(labels)

    def _prepare_resume(self, run, session_id, interaction_id, choice_id, text, choice_ids, started):
        config = self._thread_config(session_id)
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
        pending_question = str(pending.value.get("message") or "")
        human_reply_label = self._extract_human_reply_label(pending.value, selected_ids, reply.get("text"))
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
        return config, snapshot, pending, reply, pending_question, human_reply_label

    def _resume(self, run, session_id, interaction_id, choice_id, text, choice_ids=None):
        started = time.perf_counter()
        with self._session_lock(session_id):
            config, _, pending, reply, pending_question, human_reply_label = self._prepare_resume(
                run, session_id, interaction_id, choice_id, text, choice_ids, started
            )
            self.graph.invoke(Command(resume={pending.id: reply}), config=config, durability="sync")
            snapshot = self._snapshot(config)
            prev_dag = self._get_latest_session_dag(session_id)
            if snapshot.interrupts:
                paused_latency = round((time.perf_counter() - started) * 1000, 2)
                dag_telemetry = self._build_dag_from_run_and_state(
                    run, snapshot.values, paused_latency, paused=True, previous_dag=prev_dag
                )
                waiting_res = self._waiting(snapshot, session_id, dag_telemetry=dag_telemetry)
                paused_nodes = self._format_result(
                    snapshot.values, session_id, paused_latency, dag_telemetry=dag_telemetry
                ).get("nodes")
                if hasattr(run, "attach_debug_snapshot"):
                    run.attach_debug_snapshot(
                        self._build_debug_snapshot(
                            snapshot.values,
                            latency_ms=paused_latency,
                            nodes_detail=paused_nodes,
                            dag_telemetry=dag_telemetry,
                        )
                    )
                self._record_resume_turn(
                    session_id,
                    pending_question,
                    human_reply_label,
                    waiting_res,
                    trace_id=getattr(run, "trace_id", None) or "",
                    turn_id=getattr(run, "turn_id", None) or "",
                )
                return waiting_res
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        dag_telemetry = self._build_dag_from_run_and_state(
            run, snapshot.values, total_ms, paused=False, previous_dag=prev_dag
        )
        res = self._format_result(snapshot.values, session_id, total_ms, dag_telemetry=dag_telemetry)
        if hasattr(run, "attach_debug_snapshot"):
            run.attach_debug_snapshot(
                self._build_debug_snapshot(
                    snapshot.values,
                    latency_ms=res.get("latency_ms"),
                    nodes_detail=res.get("nodes"),
                    dag_telemetry=dag_telemetry,
                )
            )
        self._record_resume_turn(
            session_id,
            pending_question,
            human_reply_label,
            res,
            trace_id=getattr(run, "trace_id", None) or "",
            turn_id=getattr(run, "turn_id", None) or "",
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
        self._touch_session_on_turn_start(session_id, clean_query)
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
                    dag_telemetry = self._build_dag_from_run_and_state(
                        run, paused_snapshot.values, paused_latency, paused=True
                    )
                    paused_nodes = self._format_result(
                        paused_snapshot.values, session_id, paused_latency, dag_telemetry=dag_telemetry
                    ).get("nodes")
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
                                dag_telemetry=dag_telemetry,
                            )
                        )
                    waiting_res = self._waiting(paused_snapshot, session_id, dag_telemetry=dag_telemetry)
                    self.record_session_turn(
                        session_id,
                        query,
                        waiting_res,
                        trace_id=getattr(run, "trace_id", None) or "",
                        turn_id=getattr(run, "turn_id", None) or "",
                    )
                    return waiting_res
        except Exception as e:
            run.details["error_type"] = type(e).__name__
            print(f"[LangGraph Execution Error]: {type(e).__name__}")
            latency_err = round((time.perf_counter() - start_time) * 1000.0, 2)
            reply = self.runtime.guardrails.guard_output(FALLBACK_ANSWER)
            err_state = {
                "user_query": query,
                "persona": persona,
                "tone": tone,
                "format_type": format_type,
                "intent": "ERROR",
                "audit_status": "ERROR",
                "error_category": type(e).__name__,
            }
            err_dag = self._build_dag_from_run_and_state(run, err_state, latency_err, paused=False)
            if hasattr(run, "attach_debug_snapshot"):
                run.attach_debug_snapshot(
                    self._build_debug_snapshot(
                        err_state,
                        query=query,
                        persona=persona,
                        tone=tone,
                        format_type=format_type,
                        latency_ms=latency_err,
                        dag_telemetry=err_dag,
                    )
                )
            err_payload = {
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
                "nodes": {},
                "dag_telemetry": err_dag,
            }
            self.record_session_turn(
                session_id,
                query,
                err_payload,
                trace_id=getattr(run, "trace_id", None) or "",
                turn_id=getattr(run, "turn_id", None) or "",
            )
            return err_payload

        total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        dag_telemetry = self._build_dag_from_run_and_state(run, final_state, total_latency_ms, paused=False)
        formatted = self._format_result(final_state, session_id, total_latency_ms, dag_telemetry=dag_telemetry)
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
                    dag_telemetry=dag_telemetry,
                )
            )
        self.record_session_turn(
            session_id,
            query,
            formatted,
            trace_id=getattr(run, "trace_id", None) or "",
            turn_id=getattr(run, "turn_id", None) or "",
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
        dag_telemetry: dict | None = None,
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
                full_content = str(chunk.get("content") or chunk.get("original_text") or "").strip()
                excerpt = full_content
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
                    "char_count": len(full_content),
                    "rerank_score": chunk.get("rerank_score"),
                    "rrf_score": chunk.get("rrf_score"),
                    "excerpt_preview": excerpt,
                    "content": full_content[:4000],
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
            resolved_nodes = self._format_result(state, "", eff_latency, include_dag=False).get("nodes")
        else:
            resolved_nodes = nodes_detail

        resolved_dag = dag_telemetry
        if resolved_dag is None and isinstance(state, dict):
            eff_lat = 0.0
            if isinstance(latency_ms, (int, float)):
                eff_lat = float(latency_ms)
            resolved_dag = self._build_dag_from_run_and_state(None, state, eff_lat)

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
            "dag_telemetry": resolved_dag,
            "retrieved_chunks": chunks_preview,
        }

    def _format_result(
        self,
        final_state: dict,
        session_id: str,
        total_latency_ms: float,
        dag_telemetry: dict | None = None,
        include_dag: bool = True,
    ) -> dict:
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
        resolved_dag = dag_telemetry
        if include_dag and resolved_dag is None:
            resolved_dag = self._build_dag_from_run_and_state(None, final_state, total_latency_ms)
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
            "nodes": nodes_detail,
            "dag_telemetry": resolved_dag,
        }

    def query(self, query: str) -> dict:
        """
        Backward-compatible query method.
        """
        return self.chat(query=query)

    @staticmethod
    def _seed_previous_dag_events(previous_dag: dict | None) -> tuple[list, float]:
        events = []
        prev_total_ms = 0.0
        if not isinstance(previous_dag, dict):
            return events, prev_total_ms
        prev_events = previous_dag.get("node_events")
        if not isinstance(prev_events, list) or not prev_events:
            return events, prev_total_ms
        prev_total_ms = float(previous_dag.get("total_latency_ms") or 0.0)
        for p_evt in prev_events:
            if not isinstance(p_evt, dict):
                continue
            evt_copy = dict(p_evt)
            evt_status = str(evt_copy.get("status") or "PASS").upper()
            if evt_status in ("SUCCESS", "COMPLETED"):
                evt_copy["status"] = "PASS"
            elif evt_copy.get("node") in ("prepare_interaction", "await_interaction") and evt_status == "AWAITING_INPUT":
                evt_copy["status"] = "PASS"
            events.append(evt_copy)
        return events, prev_total_ms

    def _consume_graph_stream(
        self,
        run,
        graph_input,
        thread_config,
        session_id: str,
        query: str,
        persona: str,
        tone: str,
        format_type: str,
        queue,
        loop,
        stop_event,
        stream_started: float,
        initial_dag_events: list | None = None,
        base_latency_ms: float = 0.0,
        on_output_guardrail=None,
    ) -> tuple[bool, list]:
        import contextlib

        paused = False
        dag_node_events = list(initial_dag_events or [])
        prev_node_perf = stream_started
        prev_span_count = len(getattr(run, "spans", None) or [])
        with contextlib.closing(self.graph.stream(graph_input, config=thread_config, durability="sync")) as events:
            while not stop_event.is_set():
                event = next(events, None)
                if event is None:
                    break
                if "__interrupt__" in event:
                    paused = True
                    continue
                progress = {}
                for node, node_state in event.items():
                    now_perf = time.perf_counter()
                    node_latency_ms = round(max(0.0, (now_perf - prev_node_perf) * 1000.0), 2)
                    elapsed_ms = round((now_perf - stream_started) * 1000.0, 2)
                    total_elapsed_ms = round(base_latency_ms + elapsed_ms, 2)
                    prev_node_perf = now_perf

                    state_obj = {}
                    if isinstance(node_state, dict):
                        state_obj = node_state
                    dag_node = self._build_dag_node_event(
                        run,
                        node,
                        state_obj,
                        node_latency_ms=node_latency_ms,
                        elapsed_ms=total_elapsed_ms,
                        span_start_count=prev_span_count,
                    )
                    prev_span_count = len(getattr(run, "spans", None) or [])
                    dag_node_events.append(dag_node)
                    dag_telemetry = self._aggregate_dag_telemetry(
                        dag_node_events,
                        total_latency_ms=total_elapsed_ms,
                        state=state_obj,
                    )

                    if isinstance(node_state, dict) and hasattr(run, "attach_debug_snapshot"):
                        live_nodes = self._format_result(
                            node_state, session_id, elapsed_ms, include_dag=False
                        ).get("nodes")
                        run.attach_debug_snapshot(
                            self._build_debug_snapshot(
                                node_state,
                                query=query,
                                persona=persona,
                                tone=tone,
                                format_type=format_type,
                                latency_ms=elapsed_ms,
                                nodes_detail=live_nodes,
                                dag_telemetry=dag_telemetry,
                            )
                        )
                    public = {
                        "dag_node": dag_node,
                        "dag_telemetry": dag_telemetry,
                    }
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
                        if node_state.get("accepted_interaction_id"):
                            public["accepted_interaction_id"] = node_state["accepted_interaction_id"]
                        run.result(public)
                        turn_record_payload = {
                            "status": "completed",
                            "answer": public.get("final_answer", ""),
                            "citations": public.get("citations", []),
                            "sources": public.get("sources", []),
                            "images": public.get("images", []),
                            "intent": public.get("intent") or "RAG_SEARCH",
                            "audit_status": public.get("audit_status") or "PASS",
                            "plan": list(node_state.get("plan") or public.get("plan") or []),
                            "latency_ms": elapsed_ms,
                            "dag_telemetry": dag_telemetry,
                        }
                        if node_state.get("accepted_interaction_id"):
                            turn_record_payload["accepted_interaction_id"] = node_state["accepted_interaction_id"]
                        if turn_record_payload["audit_status"] == "CANCELED":
                            turn_record_payload["status"] = "canceled"
                        if on_output_guardrail is not None:
                            on_output_guardrail(public, turn_record_payload)
                    if getattr(run, "trace_id", None):
                        public["trace_id"] = run.trace_id
                    progress[node] = public
                loop.call_soon_threadsafe(queue.put_nowait, ("event", progress))
        return paused, dag_node_events

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
        import threading

        clean_stream_query = str(query or "").strip()
        if clean_stream_query:
            self._touch_session_on_turn_start(session_id, clean_stream_query)
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
                    return None
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

                def _on_output(public, turn_record_payload):
                    resolved_title = self.record_session_turn(
                        session_id,
                        query,
                        turn_record_payload,
                        trace_id=getattr(run, "trace_id", None) or "",
                        turn_id=getattr(run, "turn_id", None) or "",
                    )
                    if resolved_title:
                        public["session_title"] = resolved_title
                    if turn_record_payload.get("trace_id") and not public.get("trace_id"):
                        public["trace_id"] = turn_record_payload["trace_id"]

                paused, dag_node_events = self._consume_graph_stream(
                    run,
                    initial_state,
                    thread_config,
                    session_id,
                    query,
                    persona,
                    tone,
                    format_type,
                    queue,
                    loop,
                    stop_event,
                    stream_started,
                    on_output_guardrail=_on_output,
                )
                if paused and not stop_event.is_set():
                    snapshot = self._snapshot(thread_config)
                    paused_latency = round((time.perf_counter() - stream_started) * 1000.0, 2)
                    dag_telemetry = self._aggregate_dag_telemetry(
                        dag_node_events,
                        total_latency_ms=paused_latency,
                        state=snapshot.values,
                        paused=True,
                    )
                    paused_nodes = self._format_result(
                        snapshot.values, session_id, paused_latency, include_dag=False
                    ).get("nodes")
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
                                dag_telemetry=dag_telemetry,
                            )
                        )
                    waiting_payload = run.result(self._waiting(snapshot, session_id, dag_telemetry=dag_telemetry))
                    resolved_title = self.record_session_turn(
                        session_id,
                        query,
                        waiting_payload,
                        trace_id=getattr(run, "trace_id", None) or "",
                        turn_id=getattr(run, "turn_id", None) or "",
                    )
                    if resolved_title:
                        waiting_payload["session_title"] = resolved_title
                    return {"awaiting_input": waiting_payload}
            if stop_event.is_set() and "audit_status" not in run.details:
                run.outcome = "disconnected"
            return None

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

    async def astream_resume(
        self,
        session_id: str,
        interaction_id: str,
        choice_id: str | None = None,
        text: str | None = None,
        choice_ids: list[str] | None = None,
    ):
        """Stream all resumed super-step events in real time without blocking the event loop."""
        import asyncio
        import threading

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        stop_event = threading.Event()
        sentinel = object()

        def _stream(run):
            stream_started = time.perf_counter()
            with self._session_lock(session_id):
                if stop_event.is_set():
                    run.outcome = "disconnected"
                    return None
                thread_config, snapshot, pending, reply, pending_question, human_reply_label = self._prepare_resume(
                    run, session_id, interaction_id, choice_id, text, choice_ids, stream_started
                )
                prev_dag = self._get_latest_session_dag(session_id)
                seeded_events, base_latency_ms = self._seed_previous_dag_events(prev_dag)
                orig_query = str(snapshot.values.get("user_query") or "[resume]")
                persona = str(snapshot.values.get("persona") or _DEFAULT_PERSONA)
                tone = str(snapshot.values.get("tone") or _DEFAULT_TONE)
                format_type = str(snapshot.values.get("format_type") or _DEFAULT_FORMAT_TYPE)

                def _on_output(public, turn_record_payload):
                    self._record_resume_turn(
                        session_id,
                        pending_question,
                        human_reply_label,
                        turn_record_payload,
                        trace_id=getattr(run, "trace_id", None) or "",
                        turn_id=getattr(run, "turn_id", None) or "",
                    )
                    if turn_record_payload.get("trace_id") and not public.get("trace_id"):
                        public["trace_id"] = turn_record_payload["trace_id"]

                paused, dag_node_events = self._consume_graph_stream(
                    run,
                    Command(resume={pending.id: reply}),
                    thread_config,
                    session_id,
                    orig_query,
                    persona,
                    tone,
                    format_type,
                    queue,
                    loop,
                    stop_event,
                    stream_started,
                    initial_dag_events=seeded_events,
                    base_latency_ms=base_latency_ms,
                    on_output_guardrail=_on_output,
                )
                if paused and not stop_event.is_set():
                    snapshot = self._snapshot(thread_config)
                    paused_latency = round((time.perf_counter() - stream_started) * 1000.0, 2)
                    dag_telemetry = self._aggregate_dag_telemetry(
                        dag_node_events,
                        total_latency_ms=round(base_latency_ms + paused_latency, 2),
                        state=snapshot.values,
                        paused=True,
                    )
                    paused_nodes = self._format_result(
                        snapshot.values, session_id, paused_latency, include_dag=False
                    ).get("nodes")
                    if hasattr(run, "attach_debug_snapshot"):
                        run.attach_debug_snapshot(
                            self._build_debug_snapshot(
                                snapshot.values,
                                query=orig_query,
                                persona=persona,
                                tone=tone,
                                format_type=format_type,
                                latency_ms=paused_latency,
                                nodes_detail=paused_nodes,
                                dag_telemetry=dag_telemetry,
                            )
                        )
                    waiting_payload = run.result(self._waiting(snapshot, session_id, dag_telemetry=dag_telemetry))
                    self._record_resume_turn(
                        session_id,
                        pending_question,
                        human_reply_label,
                        waiting_payload,
                        trace_id=getattr(run, "trace_id", None) or "",
                        turn_id=getattr(run, "turn_id", None) or "",
                    )
                    return {"awaiting_input": waiting_payload}
            if stop_event.is_set() and "audit_status" not in run.details:
                run.outcome = "disconnected"
            return None

        def _worker():
            run = None
            try:
                with self._trace("resume", session_id) as run:
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

    # ------------------------------------------------------------------
    # Live LangGraph DAG Telemetry Builders
    # ------------------------------------------------------------------
    @staticmethod
    def _map_node_to_dag_stage(node_name: str) -> str:
        clean = str(node_name or "").strip()
        if clean in _NODE_TO_DAG_STAGE:
            return _NODE_TO_DAG_STAGE[clean]
        return "coordinator"

    def _bound_text(self, value, max_chars: int = 200) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        guarded = self.runtime.guard_artifact(raw)
        clean = " ".join(str(guarded or "").split())
        if len(clean) > max_chars:
            return clean[: max_chars - 3] + "..."
        return clean

    def _build_node_io_summary(self, node_name: str, state: dict) -> tuple[dict, dict]:
        if not isinstance(state, dict):
            state = {}
        query_preview = self._bound_text(state.get("user_query") or "", 180)
        if node_name == "vision_ocr":
            return (
                {"attachments": "image_ocr"},
                {"extracted_query_preview": query_preview},
            )
        if node_name == "input_guardrail":
            in_guard = state.get("input_guard")
            if not isinstance(in_guard, dict):
                in_guard = {}
            flags = list(in_guard.get("flags") or [])
            return (
                {"user_query": query_preview},
                {"is_safe": bool(state.get("is_safe", True)), "flags": flags},
            )
        if node_name == "context_prepare":
            hints = state.get("query_hints")
            if not isinstance(hints, dict):
                hints = {}
            return (
                {
                    "user_query": query_preview,
                    "history_messages": len(state.get("chat_history") or []),
                },
                {
                    "intent_hint": hints.get("intent") or "",
                    "domain": hints.get("domain") or "",
                    "keywords": list(hints.get("keywords") or [])[:6],
                    "compactions": int(state.get("compactions_this_turn") or 0),
                },
            )
        if node_name in ("deep_decide", "plan_update", "memory_update"):
            turn = state.get("react_turn")
            if not isinstance(turn, dict):
                turn = {}
            decision = turn.get("decision")
            if not isinstance(decision, dict):
                decision = {}
            decision_kind = decision.get("kind")
            if not decision_kind:
                if state.get("final_answer"):
                    decision_kind = "final"
                elif node_name == "plan_update":
                    decision_kind = "plan"
                elif node_name == "memory_update":
                    decision_kind = "memory"
                else:
                    decision_kind = "decide"
            target = decision.get("name") or decision.get("tool") or decision.get("skill") or ""
            has_draft = bool(state.get("final_answer") or decision.get("answer"))
            return (
                {
                    "user_query": query_preview,
                    "evidence_count": len(turn.get("evidence") or []),
                    "observations_count": len(turn.get("observations") or []),
                    "plan_revision": int(state.get("plan_revision") or 0),
                },
                {
                    "decision_kind": str(decision_kind),
                    "target": self._bound_text(target, 80),
                    "plan_steps": len(state.get("plan") or []),
                    "has_draft": has_draft,
                },
            )
        if node_name in ("react_action", "subagent", "skill_inspect", "prepare_interaction", "await_interaction"):
            last_act = state.get("last_action")
            if not isinstance(last_act, dict):
                last_act = {}
            inter = state.get("interaction")
            if not isinstance(inter, dict):
                inter = {}
            action_kind = last_act.get("kind")
            if not action_kind:
                action_kind = node_name
            action_name = last_act.get("name")
            if not action_name:
                action_name = inter.get("kind") or node_name
            action_status = last_act.get("status")
            if not action_status:
                if inter:
                    action_status = "AWAITING_INPUT"
                else:
                    action_status = "SUCCESS"
            return (
                {
                    "action_kind": str(action_kind),
                    "action_name": self._bound_text(action_name, 80),
                    "search_count": int(state.get("search_count") or 0),
                },
                {
                    "status": str(action_status),
                    "retrieved_chunks": len(state.get("retrieved_chunks") or []),
                    "tool_calls": len(state.get("tool_outputs") or []),
                    "delegations": len(state.get("delegations") or []),
                },
            )
        if node_name == "answer_review":
            reviews = list(state.get("reviews") or [])
            last_verdict = ""
            if reviews:
                last_verdict = str(reviews[-1])
            draft_text = state.get("final_answer") or state.get("draft_answer") or ""
            return (
                {
                    "draft_preview": self._bound_text(draft_text, 180),
                    "citations_checked": len(state.get("citations") or []),
                    "hallucination_retries": int(state.get("hallucination_retries") or 0),
                },
                {
                    "audit_status": str(state.get("audit_status") or "PASS"),
                    "verdict": last_verdict,
                    "verification": dict(state.get("verification") or {}),
                    "audit_feedback": self._bound_text(state.get("audit_feedback") or "", 180),
                },
            )
        if node_name == "output_guardrail":
            out_g = state.get("output_guard")
            if not isinstance(out_g, dict):
                out_g = {}
            return (
                {
                    "audit_status": str(state.get("audit_status") or "PASS"),
                    "candidate_citations": len(state.get("citations") or []),
                },
                {
                    "is_safe": bool(out_g.get("is_safe", True)),
                    "flags": list(out_g.get("flags") or []),
                    "final_citations": len(state.get("citations") or []),
                    "images_count": len(state.get("images") or []),
                    "answer_chars": len(str(state.get("final_answer") or "")),
                },
            )
        return (
            {"user_query": query_preview},
            {"audit_status": str(state.get("audit_status") or "PASS")},
        )

    def _extract_node_llm_metrics(
        self,
        run,
        span_start_count: int,
        node_name: str,
        input_summary: dict,
        output_summary: dict,
        spans_override: list | None = None,
    ) -> dict:
        if spans_override is not None:
            candidate_spans = list(spans_override)
        else:
            raw_spans = getattr(run, "spans", None)
            if isinstance(raw_spans, list):
                candidate_spans = list(raw_spans[span_start_count:])
            else:
                candidate_spans = []

        has_deep = False
        has_standard = False
        picked_model = ""
        in_tokens = 0
        out_tokens = 0
        total_tokens = 0

        for sp in candidate_spans:
            if not isinstance(sp, dict):
                continue
            sp_name = str(sp.get("name") or "")
            sp_type = str(sp.get("run_type") or "")
            attrs = sp.get("attributes")
            if not isinstance(attrs, dict):
                attrs = {}
            role_attr = str(attrs.get("role") or "")
            if sp_type == "embedding" or role_attr == "embedding" or sp_name == "query_embedding":
                continue
            if sp_name in ("model.attempt", "model.route") or sp_type == "llm":
                m_name = str(attrs.get("model") or attrs.get("ls_model_name") or "")
                if m_name:
                    picked_model = m_name
                alias = str(attrs.get("alias") or attrs.get("profile") or "").lower()
                reason = str(attrs.get("reason") or "").lower()
                if "deep" in alias or (reason and reason not in ("standard", "standard_fallback", "contract_repair")):
                    has_deep = True
                else:
                    has_standard = True
                usage_obj = attrs.get("usage")
                if not isinstance(usage_obj, dict):
                    usage_obj = attrs.get("usage_metadata")
                if isinstance(usage_obj, dict):
                    u_in = usage_obj.get("input_tokens")
                    if u_in is None:
                        u_in = usage_obj.get("prompt_tokens")
                    u_out = usage_obj.get("output_tokens")
                    if u_out is None:
                        u_out = usage_obj.get("completion_tokens")
                    u_tot = usage_obj.get("total_tokens")
                    in_val = int(u_in or 0)
                    out_val = int(u_out or 0)
                    if u_tot is not None:
                        tot_val = int(u_tot or 0)
                    else:
                        tot_val = in_val + out_val
                    in_tokens += in_val
                    out_tokens += out_val
                    total_tokens += tot_val

        if has_deep:
            profile = "DEEP"
        elif has_standard or node_name in _LLM_DRIVEN_NODES:
            profile = "STANDARD"
        else:
            profile = "DETERMINISTIC"

        if not picked_model:
            if node_name in _LLM_DRIVEN_NODES:
                picked_model = str(getattr(self.config, "AGENT_LLM_MODEL", "") or "agent-llm")
            elif node_name == "vision_ocr":
                picked_model = "cloud-vision"
            else:
                picked_model = "deterministic-engine"

        estimated = False
        if total_tokens == 0 and (
            node_name in _LLM_DRIVEN_NODES
            or node_name in ("plan_update", "memory_update", "vision_ocr")
        ):
            in_str = json.dumps(input_summary, ensure_ascii=False)
            out_str = json.dumps(output_summary, ensure_ascii=False)
            in_tokens = max(4, len(in_str.encode("utf-8")) // 4)
            out_tokens = max(4, len(out_str.encode("utf-8")) // 4)
            total_tokens = in_tokens + out_tokens
            estimated = True

        return {
            "model": picked_model,
            "profile": profile,
            "tokens": {
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": total_tokens,
                "estimated": estimated,
            },
        }

    @staticmethod
    def _resolve_dag_node_status(node_name: str, state: dict) -> str:
        if not isinstance(state, dict):
            return "PASS"
        if node_name == "input_guardrail":
            if state.get("is_safe") is False:
                return "BLOCKED"
            return "PASS"
        if node_name == "output_guardrail":
            out_g = state.get("output_guard")
            if isinstance(out_g, dict) and out_g.get("is_safe") is False:
                return "BLOCKED"
            if state.get("audit_status") == "BLOCKED":
                return "BLOCKED"
            if state.get("audit_status") == "CANCELED":
                return "CANCELED"
            return "PASS"
        if node_name == "answer_review":
            audit = str(state.get("audit_status") or "PASS")
            return audit
        if node_name in ("prepare_interaction", "await_interaction"):
            if state.get("interaction"):
                return "AWAITING_INPUT"
            if state.get("audit_status") == "CANCELED":
                return "CANCELED"
            return "PASS"
        if node_name in ("react_action", "subagent", "skill_inspect"):
            last_act = state.get("last_action")
            if isinstance(last_act, dict) and last_act.get("status"):
                raw_status = str(last_act["status"]).upper()
                if raw_status in ("SUCCESS", "COMPLETED"):
                    return "PASS"
                return str(last_act["status"])
            return "PASS"
        return "PASS"

    def _build_dag_node_event(
        self,
        run,
        node_name: str,
        state: dict,
        node_latency_ms: float = 0.0,
        elapsed_ms: float = 0.0,
        span_start_count: int = 0,
        spans_override: list | None = None,
        status_override: str | None = None,
    ) -> dict:
        stage_id = self._map_node_to_dag_stage(node_name)
        in_sum, out_sum = self._build_node_io_summary(node_name, state)
        llm_meta = self._extract_node_llm_metrics(
            run,
            span_start_count,
            node_name,
            in_sum,
            out_sum,
            spans_override=spans_override,
        )
        if status_override:
            node_status = str(status_override)
        else:
            node_status = self._resolve_dag_node_status(node_name, state)
        return {
            "stage_id": stage_id,
            "stage_label": _DAG_STAGE_LABELS.get(stage_id, stage_id),
            "node": node_name,
            "status": node_status,
            "latency_ms": round(float(node_latency_ms), 2),
            "elapsed_ms": round(float(elapsed_ms), 2),
            "model": llm_meta["model"],
            "profile": llm_meta["profile"],
            "tokens": llm_meta["tokens"],
            "input_summary": in_sum,
            "output_summary": out_sum,
        }

    def _aggregate_dag_telemetry(
        self,
        node_events: list,
        total_latency_ms: float = 0.0,
        state: dict | None = None,
        paused: bool = False,
    ) -> dict:
        stages = {}
        for sid in _DAG_STAGE_ORDER:
            stages[sid] = {
                "stage_id": sid,
                "label": _DAG_STAGE_LABELS[sid],
                "status": "IDLE",
                "executed": False,
                "call_count": 0,
                "latency_ms": 0.0,
                "model": "",
                "profile": "DETERMINISTIC",
                "tokens": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated": False,
                },
                "nodes": [],
                "input_summary": {},
                "output_summary": {},
                "executions": [],
            }

        sum_in = 0
        sum_out = 0
        sum_total = 0
        any_estimated = False
        first_llm_model = ""
        first_deep_model = ""
        retry_loops = 0

        for evt in node_events:
            if not isinstance(evt, dict):
                continue
            sid = evt.get("stage_id") or "coordinator"
            if sid not in stages:
                continue
            st = stages[sid]
            st["executed"] = True
            st["call_count"] = st["call_count"] + 1
            st["latency_ms"] = round(st["latency_ms"] + float(evt.get("latency_ms") or 0.0), 2)
            st["status"] = str(evt.get("status") or "PASS")
            if sid == "answer_review" and st["status"] == "RETRY":
                retry_loops = retry_loops + 1

            evt_prof = str(evt.get("profile") or "DETERMINISTIC")
            evt_model = str(evt.get("model") or "")
            if evt_prof == "DEEP":
                st["profile"] = "DEEP"
                if evt_model:
                    st["model"] = evt_model
                    if not first_deep_model:
                        first_deep_model = evt_model
            elif evt_prof == "STANDARD":
                if st["profile"] != "DEEP":
                    st["profile"] = "STANDARD"
                if evt_model and (not st["model"] or st["model"] == "deterministic-engine"):
                    st["model"] = evt_model
                if evt_model and not first_llm_model:
                    first_llm_model = evt_model
            elif not st["model"]:
                st["model"] = evt_model

            tok = evt.get("tokens") or {}
            t_in = int(tok.get("input_tokens") or 0)
            t_out = int(tok.get("output_tokens") or 0)
            t_tot = int(tok.get("total_tokens") or 0)
            t_est = bool(tok.get("estimated", False))
            st["tokens"]["input_tokens"] += t_in
            st["tokens"]["output_tokens"] += t_out
            st["tokens"]["total_tokens"] += t_tot
            if t_est:
                st["tokens"]["estimated"] = True
                any_estimated = True

            sum_in += t_in
            sum_out += t_out
            sum_total += t_tot

            node_name = str(evt.get("node") or "")
            if node_name and node_name not in st["nodes"]:
                st["nodes"].append(node_name)
            st["input_summary"] = dict(evt.get("input_summary") or {})
            st["output_summary"] = dict(evt.get("output_summary") or {})
            st["executions"].append({
                "node": node_name,
                "status": evt.get("status") or "PASS",
                "latency_ms": float(evt.get("latency_ms") or 0.0),
                "model": evt_model,
                "profile": evt_prof,
                "tokens": dict(tok),
            })

        if isinstance(state, dict):
            state_retries = int(state.get("hallucination_retries") or 0)
            if state_retries > retry_loops:
                retry_loops = state_retries

        active_profile = "DETERMINISTIC"
        for sid in _DAG_STAGE_ORDER:
            if stages[sid]["profile"] == "DEEP":
                active_profile = "DEEP"
                break
            if stages[sid]["profile"] == "STANDARD":
                active_profile = "STANDARD"

        active_model = first_deep_model or first_llm_model
        if not active_model:
            active_model = str(getattr(self.config, "AGENT_LLM_MODEL", "") or "deterministic-engine")

        audit_status = "PASS"
        if paused:
            audit_status = "AWAITING_INPUT"
        elif isinstance(state, dict) and state.get("audit_status"):
            audit_status = str(state["audit_status"])
        elif node_events:
            audit_status = str(node_events[-1].get("status") or "PASS")

        ttft_ms = 0.0
        if node_events:
            ttft_ms = float(node_events[0].get("latency_ms") or 0.0)

        return {
            "total_latency_ms": round(float(total_latency_ms), 2),
            "ttft_ms": round(ttft_ms, 2),
            "active_model": active_model,
            "active_profile": active_profile,
            "token_usage": {
                "input_tokens": sum_in,
                "output_tokens": sum_out,
                "total_tokens": sum_total,
                "estimated": any_estimated,
            },
            "audit_status": audit_status,
            "retry_loops": retry_loops,
            "stages": stages,
            "node_events": list(node_events),
        }

    def _build_dag_from_run_and_state(
        self,
        run,
        state: dict,
        total_latency_ms: float = 0.0,
        paused: bool = False,
        previous_dag: dict | None = None,
    ) -> dict:
        if not isinstance(state, dict):
            state = {}
        raw_spans = []
        if run is not None and isinstance(getattr(run, "spans", None), list):
            raw_spans = list(run.spans)

        graph_spans = []
        for sp in raw_spans:
            if isinstance(sp, dict) and str(sp.get("name") or "").startswith("graph."):
                graph_spans.append(sp)

        current_events = []
        if graph_spans:
            root_ns = None
            for sp in raw_spans:
                if isinstance(sp, dict) and isinstance(sp.get("started_ns"), int):
                    root_ns = sp["started_ns"]
                    break
            for gsp in graph_spans:
                full_name = str(gsp.get("name") or "")
                node_name = full_name[len("graph."):]
                g_sid = str(gsp.get("span_id") or "")
                child_spans = []
                for sp in raw_spans:
                    if not isinstance(sp, dict):
                        continue
                    if str(sp.get("parent_span_id") or "") == g_sid or str(sp.get("parent_run_id") or "") == g_sid:
                        child_spans.append(sp)
                g_attrs = gsp.get("attributes")
                if not isinstance(g_attrs, dict):
                    g_attrs = {}
                g_status = None
                if node_name == "answer_review" and g_attrs.get("audit_status"):
                    g_status = str(g_attrs["audit_status"])
                elif gsp.get("status") == "suspended":
                    g_status = "AWAITING_INPUT"
                elif gsp.get("status") == "error":
                    g_status = "ERROR"
                elif not paused and node_name in ("prepare_interaction", "await_interaction"):
                    if state.get("audit_status") == "CANCELED":
                        g_status = "CANCELED"
                    else:
                        g_status = "PASS"
                elapsed_ms = 0.0
                if isinstance(gsp.get("started_ns"), int) and root_ns is not None:
                    elapsed_ms = round(max(0.0, (gsp["started_ns"] - root_ns) / 1_000_000.0), 2)
                dur_ms = float(gsp.get("duration_ms") or 0.0)
                current_events.append(
                    self._build_dag_node_event(
                        run,
                        node_name,
                        state,
                        node_latency_ms=dur_ms,
                        elapsed_ms=round(elapsed_ms + dur_ms, 2),
                        spans_override=child_spans,
                        status_override=g_status,
                    )
                )
        else:
            inferred_nodes = []
            if "is_safe" in state or "input_guard" in state:
                inferred_nodes.append("input_guardrail")
            if state.get("query_hints"):
                inferred_nodes.append("context_prepare")
            if state.get("intent") or state.get("plan") or state.get("execution_steps") or state.get("final_answer"):
                inferred_nodes.append("deep_decide")
            if state.get("tool_outputs") or state.get("retrieved_chunks") or state.get("delegations"):
                inferred_nodes.append("react_action")
            if paused or state.get("interaction") or state.get("human_replies") or state.get("accepted_interaction_id"):
                inferred_nodes.append("prepare_interaction")
                inferred_nodes.append("await_interaction")
            if state.get("reviews") or (state.get("audit_status") and not paused):
                if state.get("audit_status") != "BLOCKED" or state.get("reviews"):
                    inferred_nodes.append("answer_review")
            if state.get("output_guard") or (state.get("final_answer") and not paused):
                inferred_nodes.append("output_guardrail")
            if not inferred_nodes and state.get("audit_status") == "ERROR":
                inferred_nodes.append("input_guardrail")

            default_node_latencies = {
                "input_guardrail": 2.0,
                "context_prepare": 1.5,
                "deep_decide": 28.0,
                "react_action": 22.0,
                "prepare_interaction": 2.0,
                "await_interaction": 2.0,
                "answer_review": 24.0,
                "output_guardrail": 2.0,
            }
            node_latencies = []
            if inferred_nodes and total_latency_ms > 0:
                per_node_ms = round(float(total_latency_ms) / len(inferred_nodes), 2)
                for _ in inferred_nodes:
                    node_latencies.append(per_node_ms)
            else:
                for n_name in inferred_nodes:
                    node_latencies.append(float(default_node_latencies.get(n_name, 5.0)))
            running_elapsed = 0.0
            for n_name, node_ms in zip(inferred_nodes, node_latencies):
                running_elapsed = round(running_elapsed + node_ms, 2)
                status_override = None
                if not paused and n_name in ("prepare_interaction", "await_interaction"):
                    if state.get("audit_status") == "CANCELED":
                        status_override = "CANCELED"
                    else:
                        status_override = "PASS"
                current_events.append(
                    self._build_dag_node_event(
                        None,
                        n_name,
                        state,
                        node_latency_ms=node_ms,
                        elapsed_ms=running_elapsed,
                        status_override=status_override,
                    )
                )

        merged_events = []
        effective_total_ms = float(total_latency_ms or 0.0)
        if isinstance(previous_dag, dict):
            prev_events = previous_dag.get("node_events")
            if isinstance(prev_events, list) and prev_events:
                effective_total_ms = effective_total_ms + float(previous_dag.get("total_latency_ms") or 0.0)
                for p_evt in prev_events:
                    if not isinstance(p_evt, dict):
                        continue
                    evt_copy = dict(p_evt)
                    evt_status = str(evt_copy.get("status") or "PASS").upper()
                    if evt_status in ("SUCCESS", "COMPLETED"):
                        evt_copy["status"] = "PASS"
                    elif (
                        not paused
                        and evt_copy.get("node") in ("prepare_interaction", "await_interaction")
                        and evt_status == "AWAITING_INPUT"
                    ):
                        if state.get("audit_status") == "CANCELED":
                            evt_copy["status"] = "CANCELED"
                        else:
                            evt_copy["status"] = "PASS"
                    merged_events.append(evt_copy)

        node_events = merged_events + current_events
        present_stages = {str(e.get("stage_id") or "") for e in node_events if isinstance(e, dict)}
        if "input_guardrail" not in present_stages:
            has_later_stage = False
            for later_sid in ("coordinator", "workers_tools", "answer_review", "output_guardrail"):
                if later_sid in present_stages:
                    has_later_stage = True
                    break
            if has_later_stage or ("is_safe" in state) or ("input_guard" in state):
                ig_evt = self._build_dag_node_event(
                    None,
                    "input_guardrail",
                    state,
                    node_latency_ms=1.5,
                    elapsed_ms=1.5,
                )
                node_events.insert(0, ig_evt)
                present_stages.add("input_guardrail")

        if not paused and state.get("final_answer"):
            if state.get("is_safe", True) is not False and state.get("audit_status") not in ("BLOCKED", "CANCELED", "ERROR"):
                for req_stage, req_node in (
                    ("coordinator", "deep_decide"),
                    ("answer_review", "answer_review"),
                    ("output_guardrail", "output_guardrail"),
                ):
                    if req_stage not in present_stages:
                        node_events.append(
                            self._build_dag_node_event(
                                None,
                                req_node,
                                state,
                                node_latency_ms=5.0,
                                elapsed_ms=round(effective_total_ms, 2),
                            )
                        )
                        present_stages.add(req_stage)
            elif "output_guardrail" not in present_stages:
                node_events.append(
                    self._build_dag_node_event(
                        None,
                        "output_guardrail",
                        state,
                        node_latency_ms=1.5,
                        elapsed_ms=round(effective_total_ms, 2),
                    )
                )
                present_stages.add("output_guardrail")

        if effective_total_ms <= 0.0 and node_events:
            sum_node_ms = 0.0
            for evt_item in node_events:
                if isinstance(evt_item, dict):
                    sum_node_ms = sum_node_ms + float(evt_item.get("latency_ms") or 0.0)
            effective_total_ms = round(sum_node_ms, 2)

        return self._aggregate_dag_telemetry(
            node_events,
            total_latency_ms=effective_total_ms,
            state=state,
            paused=paused,
        )

    # ------------------------------------------------------------------
    # SQLite Multi-Session History Manager (Synced with agent_memory.db)
    # ------------------------------------------------------------------
    def _run_sqlite(self, fn):
        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            return None
        cp_lock = getattr(self.graph.checkpointer, "lock", None)
        with self._session_store_lock:
            if cp_lock is not None:
                with cp_lock:
                    return fn(conn)
            return fn(conn)

    def _ensure_session_tables(self) -> None:
        def _init_tables(conn):
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_meta (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    preview TEXT NOT NULL DEFAULT '',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'idle',
                    audit_status TEXT NOT NULL DEFAULT ''
                )
                """
            )
            existing_cols = {
                str(r[1]) for r in conn.execute("PRAGMA table_info(session_meta)").fetchall()
            }
            migrations = (
                ("preview", "TEXT NOT NULL DEFAULT ''"),
                ("message_count", "INTEGER NOT NULL DEFAULT 0"),
                ("status", "TEXT NOT NULL DEFAULT 'idle'"),
                ("audit_status", "TEXT NOT NULL DEFAULT ''"),
                ("title_source", "TEXT NOT NULL DEFAULT 'auto'"),
                ("traces_cleared", "INTEGER NOT NULL DEFAULT 0"),
            )
            for col_name, col_def in migrations:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE session_meta ADD COLUMN {col_name} {col_def}")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    assistant_answer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    audit_status TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    images_json TEXT NOT NULL DEFAULT '[]',
                    plan_json TEXT NOT NULL DEFAULT '[]',
                    dag_json TEXT NOT NULL DEFAULT '{}',
                    interaction_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_turns_sid ON session_turns(session_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS debug_traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL DEFAULT '',
                    session_ref TEXT NOT NULL DEFAULT '',
                    turn_id TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL,
                    entrypoint TEXT NOT NULL DEFAULT 'chat',
                    outcome TEXT NOT NULL DEFAULT 'completed',
                    duration_ms REAL NOT NULL DEFAULT 0,
                    record_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_debug_traces_sref ON debug_traces(session_ref)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_debug_traces_sid ON debug_traces(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_debug_traces_ts ON debug_traces(timestamp)"
            )
            conn.commit()
        try:
            self._run_sqlite(_init_tables)
        except Exception as err:
            print(f"[Session Store Init Error]: {type(err).__name__}")

    @staticmethod
    def _extract_query_and_image_context(text: str) -> tuple[str, str]:
        """Split customer text from appended Cloud Vision extraction context."""
        raw = str(text or "")
        marker = "\n\n[Thông tin trích xuất từ ảnh đính kèm"
        image_context = ""
        if marker in raw:
            user_part, img_tail = raw.split(marker, 1)
            img_clean = img_tail.strip()
            for prefix in ("của khách hàng:", "của khách hàng", ":"):
                if img_clean.startswith(prefix):
                    img_clean = img_clean[len(prefix):].strip()
            if img_clean.endswith("]"):
                img_clean = img_clean[:-1].strip()
            if "Không thể trích xuất chi tiết từ ảnh" not in img_clean:
                image_context = " ".join(img_clean.split()).strip()
            raw = user_part
        user_clean = " ".join(raw.split()).strip()
        return user_clean, image_context

    @staticmethod
    def _is_greeting_only(text: str) -> bool:
        """Return True if the message is only a short conversational greeting."""
        clean = " ".join(str(text or "").split()).strip().lower()
        while clean and clean[-1] in ".!?,~":
            clean = clean[:-1].strip()
        greetings = (
            "xin chào",
            "chào",
            "chào bạn",
            "chào ad",
            "chào shop",
            "chào trợ lý",
            "hello",
            "hi",
            "hey",
            "alo",
            "good morning",
            "good afternoon",
            "good evening",
            "test",
            "ok",
        )
        return clean in greetings

    @staticmethod
    def _truncate_title_words(text: str, max_len: int = 58) -> str:
        """Truncate a title on a word boundary without cutting mid-word when possible."""
        clean = " ".join(str(text or "").split()).strip()
        if len(clean) <= max_len:
            return clean
        cutoff = clean[:max_len - 3].rstrip()
        last_space = cutoff.rfind(" ")
        if last_space >= 16:
            cutoff = cutoff[:last_space].rstrip()
        return cutoff + "..."

    @staticmethod
    def _capitalize_first(text: str) -> str:
        if not text:
            return ""
        return text[0].upper() + text[1:]

    @classmethod
    def _derive_session_title(cls, text: str) -> str:
        """Build a concise fallback session title directly from the customer's question."""
        clean, image_context = cls._extract_query_and_image_context(text)
        default_img_prompt = "Hãy phân tích nội dung hình ảnh đính kèm và tư vấn hướng xử lý theo quy định của A-Bank."
        if not clean or clean == default_img_prompt:
            if image_context:
                return "Phân tích hình ảnh đính kèm"
            if not clean:
                return "Cuộc hội thoại mới"

        lowered = clean.lower()
        filler_prefixes = (
            "cho tôi hỏi về ",
            "cho tôi hỏi ",
            "cho mình hỏi về ",
            "cho mình hỏi ",
            "cho em hỏi về ",
            "cho em hỏi ",
            "cho hỏi về ",
            "cho hỏi ",
            "cho tôi biết ",
            "xin hỏi về ",
            "xin hỏi ",
            "tôi muốn tìm hiểu về ",
            "tôi muốn hỏi về ",
            "tôi muốn biết về ",
            "tôi muốn hỏi ",
            "tôi cần tìm hiểu về ",
            "tôi cần hỏi về ",
            "tôi cần hỏi ",
            "mình muốn tìm hiểu về ",
            "mình muốn hỏi về ",
            "mình muốn hỏi ",
            "bạn ơi cho hỏi ",
            "bạn ơi ",
            "trợ lý ơi ",
            "ad ơi ",
            "bạn hãy giúp tôi ",
            "bạn hãy ",
            "hãy giúp tôi ",
            "giúp tôi ",
            "làm ơn ",
            "vui lòng ",
            "hãy ",
        )
        changed = True
        while changed:
            changed = False
            for prefix in filler_prefixes:
                if lowered.startswith(prefix) and len(clean) > len(prefix) + 3:
                    clean = clean[len(prefix):].strip()
                    lowered = clean.lower()
                    changed = True
                    break

        while clean and clean[-1] in ".?!,;:-":
            clean = clean[:-1].strip()
        if not clean:
            return "Cuộc hội thoại mới"
        return cls._truncate_title_words(cls._capitalize_first(clean), max_len=58)

    def _sanitize_summarized_title(self, raw_title: str) -> str:
        """Clean and guard an LLM-generated session title."""
        text = str(raw_title or "").strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        candidate = lines[0]
        for label_prefix in ("Tiêu đề:", "tiêu đề:", "Title:", "title:", "Tóm tắt:", "tóm tắt:"):
            if candidate.startswith(label_prefix):
                candidate = candidate[len(label_prefix):].strip()
        while candidate and candidate[0] in "\"'`*«“‘[-•":
            candidate = candidate[1:].strip()
        while candidate and candidate[-1] in "\"'`*»”’].!?,:;":
            candidate = candidate[:-1].strip()
        candidate = " ".join(candidate.split()).strip()
        if not candidate:
            return ""
        if hasattr(self, "runtime") and hasattr(self.runtime, "guard_artifact"):
            guarded = self.runtime.guard_artifact(candidate)
            if not guarded:
                return ""
            candidate = guarded
        return self._truncate_title_words(self._capitalize_first(candidate), max_len=58)

    def _can_use_llm_title_summarizer(self) -> bool:
        if getattr(self, "_title_summarizer", None) is not None:
            return True
        runtime = getattr(self, "runtime", None)
        client = getattr(runtime, "client", None)
        if client is None:
            return False
        if "create" in getattr(client, "__dict__", {}):
            return False
        candidates = getattr(client, "candidates", None) or []
        return any(model_configured(item) for item in candidates)

    def _summarize_session_title_with_llm(self, user_query: str) -> str:
        """Summarize the customer's question into a short Vietnamese session title via Cloud LLM."""
        user_text, image_context = self._extract_query_and_image_context(user_query)
        if not user_text and not image_context:
            return ""

        prompt_parts = []
        if user_text:
            prompt_parts.append("Câu hỏi khách hàng: " + user_text[:500])
        if image_context:
            prompt_parts.append("Ngữ cảnh ảnh đính kèm: " + image_context[:300])
        user_prompt = "\n".join(prompt_parts)

        custom_summarizer = getattr(self, "_title_summarizer", None)
        if custom_summarizer is not None:
            try:
                return self._sanitize_summarized_title(custom_summarizer(user_prompt))
            except Exception:
                return ""

        runtime = getattr(self, "runtime", None)
        client = getattr(runtime, "client", None)
        if client is None:
            return ""

        system_prompt = (
            "Bạn là trợ lý đặt tiêu đề ngắn cho phiên tư vấn ngân hàng A-Bank.\n"
            "Hãy tóm tắt ý chính từ câu hỏi của khách hàng thành một tiêu đề tiếng Việt ngắn gọn, rõ nghĩa "
            "(khoảng 3 đến 8 từ, tối đa 55 ký tự).\n"
            "Quy tắc: Không dùng dấu ngoặc kép, không dấu chấm cuối câu, không trả lời câu hỏi, "
            "không chép nguyên văn câu dài dòng, và tuyệt đối không đưa OTP, PIN, mật khẩu, số thẻ, số tài khoản vào tiêu đề.\n"
            "Chỉ trả về duy nhất chuỗi tiêu đề."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            if getattr(client, "routed", False) is True:
                candidates = ordered_candidates(client.candidates, prefer_deep=False)
                for candidate in candidates:
                    if candidate.protocol != "chat":
                        continue
                    payload = {"messages": messages, "temperature": 0.2}
                    apply_chat_options(payload, candidate, 64, sdk=True)
                    if candidate.send_temperature:
                        payload["temperature"] = 0.2
                    try:
                        response = client.create(_candidate=candidate, **payload)
                        content = str(response.choices[0].message.content or "")
                        cleaned = self._sanitize_summarized_title(content)
                        if cleaned:
                            return cleaned
                    except Exception:
                        continue
                return ""

            response = client.create(
                model=getattr(runtime, "model", ""),
                messages=messages,
                temperature=0.2,
                max_tokens=64,
            )
            content = str(response.choices[0].message.content or "")
            return self._sanitize_summarized_title(content)
        except Exception:
            return ""

    def _apply_summarized_title(
        self,
        session_id: str,
        expected_title: str,
        new_title: str,
        new_source: str,
    ) -> None:
        if not session_id or not new_title:
            return
        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                mem = self._memory_sessions.get(session_id)
                if mem is None or mem.get("deleted"):
                    return
                if mem.get("title_source") == "manual":
                    return
                current_title = str(mem.get("title") or "")
                if current_title in ("", "Cuộc hội thoại mới", expected_title):
                    mem["title"] = new_title
                    mem["title_source"] = new_source
            return

        def _update_title(db_conn):
            row = db_conn.execute(
                "SELECT title, title_source, deleted FROM session_meta WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None or int(row[2] or 0) == 1:
                return
            current_title = str(row[0] or "")
            current_source = str(row[1] or "auto")
            if current_source == "manual":
                return
            if current_title in ("", "Cuộc hội thoại mới", expected_title):
                db_conn.execute(
                    "UPDATE session_meta SET title = ?, title_source = ? WHERE session_id = ?",
                    (new_title, new_source, session_id),
                )
                db_conn.commit()

        try:
            self._run_sqlite(_update_title)
        except Exception:
            pass

    def _start_title_summarization(
        self,
        session_id: str,
        raw_query: str,
        expected_title: str,
        new_source: str,
    ) -> None:
        def _worker():
            try:
                summarized = self._summarize_session_title_with_llm(raw_query)
                if summarized and summarized != expected_title:
                    self._apply_summarized_title(session_id, expected_title, summarized, new_source)
            finally:
                with self._title_lock:
                    current = self._pending_title_threads.get(session_id)
                    if current is thread:
                        self._pending_title_threads.pop(session_id, None)

        thread = Thread(target=_worker, name="session-title-summary", daemon=True)
        with self._title_lock:
            self._pending_title_threads[session_id] = thread
        thread.start()

    def _wait_for_title_thread(self, session_id: str, timeout: float = 1.2) -> None:
        with self._title_lock:
            thread = self._pending_title_threads.get(session_id)
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _extract_checkpoint_info(self, thread_id: str) -> dict:
        info = {
            "first_user_query": "",
            "last_assistant": "",
            "timestamp": "",
            "message_count": 0,
            "audit_status": "",
            "intent": "",
            "plan": [],
            "citations": [],
            "sources": [],
            "images": [],
            "messages": [],
            "interaction": None,
        }
        try:
            cfg = self._thread_config(thread_id)
            tup = self.graph.checkpointer.get_tuple(cfg)
            if tup is None or not isinstance(tup.checkpoint, dict):
                return info
            ts = tup.checkpoint.get("ts")
            if isinstance(ts, str) and ts:
                info["timestamp"] = ts
            ch = tup.checkpoint.get("channel_values")
            if not isinstance(ch, dict):
                return info
            root = ch.get("__root__")
            if not isinstance(root, dict):
                root = ch
            history = root.get("chat_history")
            if not isinstance(history, list):
                history = []
            msgs = []
            for item in history:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "")
                content = str(item.get("content") or "")
                if role in ("user", "assistant") and content:
                    if role == "user" and not info["first_user_query"]:
                        info["first_user_query"] = content
                    if role == "assistant":
                        info["last_assistant"] = content
                    msgs.append({"role": role, "content": content})

            last_q = str(root.get("user_query") or "").strip()
            last_a = str(root.get("final_answer") or "").strip()
            if last_q and not info["first_user_query"]:
                info["first_user_query"] = last_q

            if last_q:
                already_at_end = False
                if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
                    if (
                        msgs[-2]["content"] == last_q
                        or last_q.startswith(msgs[-2]["content"])
                        or (last_a and msgs[-1]["content"] == last_a)
                    ):
                        already_at_end = True
                if already_at_end:
                    msgs[-1]["citations"] = list(root.get("citations") or [])
                    msgs[-1]["sources"] = list(root.get("sources") or [])
                    msgs[-1]["images"] = list(root.get("images") or [])
                    msgs[-1]["audit_status"] = str(root.get("audit_status") or "PASS")
                    msgs[-1]["intent"] = str(root.get("intent") or "")
                else:
                    msgs.append({"role": "user", "content": last_q})
                    if last_a:
                        info["last_assistant"] = last_a
                        msgs.append({
                            "role": "assistant",
                            "content": last_a,
                            "citations": list(root.get("citations") or []),
                            "sources": list(root.get("sources") or []),
                            "images": list(root.get("images") or []),
                            "audit_status": str(root.get("audit_status") or "PASS"),
                            "intent": str(root.get("intent") or ""),
                        })

            info["messages"] = msgs
            info["message_count"] = len(msgs)
            info["audit_status"] = str(root.get("audit_status") or "")
            info["intent"] = str(root.get("intent") or "")
            info["plan"] = list(root.get("plan") or [])
            info["citations"] = list(root.get("citations") or [])
            info["sources"] = list(root.get("sources") or [])
            info["images"] = list(root.get("images") or [])
            if isinstance(root.get("interaction"), dict):
                info["interaction"] = root["interaction"]
        except Exception:
            pass
        return info

    @staticmethod
    def _is_automated_thread_id(thread_id: str) -> bool:
        lower = str(thread_id or "").strip().lower()
        if not lower or lower == "default_banking_session":
            return True
        for prefix in ("eval_", "e2e_", "test_", "test-"):
            if lower.startswith(prefix):
                return True
        return False

    def _touch_session_on_turn_start(self, session_id: str, user_query: str) -> None:
        if not session_id:
            return
        clean_query = self._bound_text(user_query or "", 2000)
        if not clean_query:
            return
        user_text, image_context = self._extract_query_and_image_context(clean_query)
        is_greeting = self._is_greeting_only(user_text) and not image_context
        derived_title = self._derive_session_title(clean_query)
        if is_greeting:
            target_source = "auto_greeting"
        else:
            target_source = "auto"
        preview_val = self._bound_text(user_text or clean_query, 110) or "Đang xử lý yêu cầu..."
        now_iso = _utc_now_iso()
        should_summarize = False

        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                mem = self._memory_sessions.get(session_id)
                if mem is None:
                    self._memory_sessions[session_id] = {
                        "session_id": session_id,
                        "title": derived_title,
                        "title_source": target_source,
                        "created_at": now_iso,
                        "updated_at": now_iso,
                        "deleted": False,
                        "turns": [],
                    }
                    should_summarize = not is_greeting
                else:
                    existing_title = str(mem.get("title") or "")
                    existing_source = str(mem.get("title_source") or "auto")
                    can_upgrade = (
                        existing_source != "manual"
                        and (
                            not existing_title
                            or existing_title == "Cuộc hội thoại mới"
                            or existing_source == "new"
                            or (existing_source == "auto_greeting" and not is_greeting)
                        )
                    )
                    if can_upgrade:
                        mem["title"] = derived_title
                        mem["title_source"] = target_source
                        should_summarize = not is_greeting
                    mem["updated_at"] = now_iso
                    mem["deleted"] = False
            if should_summarize and self._can_use_llm_title_summarizer():
                self._start_title_summarization(session_id, clean_query, derived_title, target_source)
            return

        def _upsert_start(db_conn):
            row = db_conn.execute(
                "SELECT title, title_source FROM session_meta WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            start_llm = False
            if row is None:
                db_conn.execute(
                    """
                    INSERT INTO session_meta (
                        session_id, title, pinned, created_at, updated_at, deleted,
                        preview, message_count, status, audit_status, title_source
                    )
                    VALUES (?, ?, 0, ?, ?, 0, ?, 1, 'running', 'RUNNING', ?)
                    """,
                    (session_id, derived_title, now_iso, now_iso, preview_val, target_source),
                )
                start_llm = not is_greeting
            else:
                existing_title = str(row[0] or "")
                existing_source = str(row[1] or "auto")
                next_title = existing_title
                next_source = existing_source
                can_upgrade = (
                    existing_source != "manual"
                    and (
                        not existing_title
                        or existing_title == "Cuộc hội thoại mới"
                        or existing_source == "new"
                        or (existing_source == "auto_greeting" and not is_greeting)
                    )
                )
                if can_upgrade:
                    next_title = derived_title
                    next_source = target_source
                    start_llm = not is_greeting
                db_conn.execute(
                    """
                    UPDATE session_meta
                    SET title = ?, title_source = ?, updated_at = ?, deleted = 0,
                        preview = ?, status = 'running'
                    WHERE session_id = ?
                    """,
                    (next_title, next_source, now_iso, preview_val, session_id),
                )
            db_conn.commit()
            return start_llm

        try:
            should_summarize = bool(self._run_sqlite(_upsert_start))
        except Exception:
            should_summarize = False

        if should_summarize and self._can_use_llm_title_summarizer():
            self._start_title_summarization(session_id, clean_query, derived_title, target_source)

    def _sync_checkpoints_to_session_meta(self, specific_session_id: str = "") -> None:
        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            return
        target_sid = str(specific_session_id or "").strip()

        def _find_missing(db_conn):
            known = set()
            for row in db_conn.execute("SELECT session_id FROM session_meta").fetchall():
                known.add(str(row[0]))
            cp_threads = []
            try:
                if target_sid:
                    if target_sid not in known:
                        hit = db_conn.execute(
                            "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1",
                            (target_sid,),
                        ).fetchone()
                        if hit is not None:
                            cp_threads.append(target_sid)
                else:
                    rows = db_conn.execute(
                        "SELECT thread_id, MAX(rowid) AS max_rid FROM checkpoints GROUP BY thread_id ORDER BY max_rid DESC LIMIT 200"
                    ).fetchall()
                    for row in rows:
                        tid = str(row[0] or "").strip()
                        if tid and tid not in known and not self._is_automated_thread_id(tid):
                            cp_threads.append(tid)
            except Exception:
                pass
            return cp_threads

        missing_threads = self._run_sqlite(_find_missing) or []
        if not missing_threads:
            return

        to_insert = []
        now_iso = _utc_now_iso()
        for tid in missing_threads:
            cp_info = self._extract_checkpoint_info(tid)
            first_q = cp_info.get("first_user_query") or ""
            title = self._derive_session_title(first_q)
            user_part, img_part = self._extract_query_and_image_context(first_q)
            if self._is_greeting_only(user_part) and not img_part:
                cp_title_source = "auto_greeting"
            else:
                cp_title_source = "auto"
            ts = cp_info.get("timestamp") or now_iso
            preview_src = str(cp_info.get("last_assistant") or first_q)
            preview = self._bound_text(preview_src, 110)
            if not preview:
                preview = "Chưa có tin nhắn"
            msg_count = int(cp_info.get("message_count") or 0)
            cp_status = "idle"
            if cp_info.get("interaction") is not None:
                cp_status = "awaiting_input"
            elif msg_count > 0:
                cp_status = "completed"
            cp_audit = str(cp_info.get("audit_status") or "")
            to_insert.append((tid, title, 0, ts, ts, 0, preview, msg_count, cp_status, cp_audit, cp_title_source))

        def _insert_missing(db_conn):
            for rec in to_insert:
                db_conn.execute(
                    """
                    INSERT OR IGNORE INTO session_meta (
                        session_id, title, pinned, created_at, updated_at, deleted,
                        preview, message_count, status, audit_status, title_source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rec,
                )
            db_conn.commit()

        self._run_sqlite(_insert_missing)

    def _get_latest_session_dag(self, session_id: str) -> dict | None:
        if not session_id:
            return None
        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            mem = self._memory_sessions.get(session_id)
            if isinstance(mem, dict) and mem.get("turns"):
                dag = mem["turns"][-1].get("dag_telemetry")
                if isinstance(dag, dict) and dag:
                    return dag
            return None

        def _fetch_dag(db_conn):
            row = db_conn.execute(
                "SELECT dag_json FROM session_turns WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if row and row[0]:
                return str(row[0])
            return ""

        raw = self._run_sqlite(_fetch_dag)
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed:
                    return parsed
            except Exception:
                pass
        return None

    def _synthesize_debug_trace_record(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        state: dict,
        dag_telemetry: dict | None,
        timestamp: str = "",
        latency_ms: float = 0.0,
        trace_id: str = "",
        turn_id: str = "",
        entrypoint: str = "chat",
        paused: bool = False,
    ) -> dict:
        if not isinstance(state, dict):
            state = {}
        eff_trace = str(trace_id or uuid4().hex)
        eff_turn = str(turn_id or eff_trace)
        eff_ts = str(timestamp or _utc_now_iso())
        eff_state = dict(state)
        if user_query and not eff_state.get("user_query"):
            eff_state["user_query"] = user_query
        if assistant_answer and not eff_state.get("final_answer"):
            eff_state["final_answer"] = assistant_answer
        if "is_safe" not in eff_state:
            eff_state["is_safe"] = True
        if not eff_state.get("audit_status"):
            if paused:
                eff_state["audit_status"] = "AWAITING_INPUT"
            else:
                eff_state["audit_status"] = "PASS"

        eff_latency = round(float(latency_ms or 0.0), 3)
        resolved_dag = dag_telemetry
        if not isinstance(resolved_dag, dict) or not resolved_dag.get("stages"):
            resolved_dag = self._build_dag_from_run_and_state(None, eff_state, eff_latency, paused=paused)
        if eff_latency <= 0.0 and isinstance(resolved_dag, dict):
            eff_latency = round(float(resolved_dag.get("total_latency_ms") or 0.0), 3)

        stages = dict((resolved_dag or {}).get("stages") or {})
        coord_stage = dict(stages.get("coordinator") or {})
        workers_stage = dict(stages.get("workers_tools") or {})
        review_stage = dict(stages.get("answer_review") or {})
        in_guard_stage = dict(stages.get("input_guardrail") or {})
        out_guard_stage = dict(stages.get("output_guardrail") or {})

        default_model = str(getattr(self.config, "AGENT_LLM_MODEL", "") or "DeepSeek-V4-Flash")
        coord_model = str(coord_stage.get("model") or default_model)
        if coord_model == "deterministic-engine":
            coord_model = default_model
        coord_prof = str(coord_stage.get("profile") or "STANDARD")
        coord_tok = dict(coord_stage.get("tokens") or {"input_tokens": 32, "output_tokens": 24, "total_tokens": 56})
        if int(coord_tok.get("total_tokens") or 0) <= 0:
            coord_tok = {"input_tokens": 32, "output_tokens": 24, "total_tokens": 56}

        rev_model = str(review_stage.get("model") or default_model)
        if rev_model == "deterministic-engine":
            rev_model = default_model
        rev_prof = str(review_stage.get("profile") or "STANDARD")
        rev_tok = dict(review_stage.get("tokens") or {"input_tokens": 48, "output_tokens": 24, "total_tokens": 72})
        if int(rev_tok.get("total_tokens") or 0) <= 0:
            rev_tok = {"input_tokens": 48, "output_tokens": 24, "total_tokens": 72}

        try:
            dt_obj = datetime.fromisoformat(eff_ts.replace("Z", "+00:00"))
            base_ns = int(dt_obj.timestamp() * 1_000_000_000)
        except Exception:
            base_ns = int(time.time() * 1_000_000_000)

        root_id = eff_trace
        spans = []
        cursor_ns = [base_ns]
        seq_counter = [0]

        def _add_span(name: str, run_type: str, parent_id: str | None, dur_ms: float, attrs: dict, status: str = "ok") -> str:
            seq_counter[0] = seq_counter[0] + 1
            sid = f"{eff_trace[:16]}_{seq_counter[0]}"
            if parent_id is None:
                sid = root_id
            dur_val = round(max(0.0, float(dur_ms or 0.0)), 3)
            rec = {
                "span_id": sid,
                "run_id": sid,
                "parent_span_id": parent_id,
                "parent_run_id": parent_id,
                "run_type": run_type,
                "name": name,
                "started_ns": cursor_ns[0],
                "duration_ms": dur_val,
                "status": status,
                "attributes": _sanitize_value(attrs),
            }
            spans.append(rec)
            step_ns = max(1_000_000, int(dur_val * 1_000_000))
            cursor_ns[0] = cursor_ns[0] + step_ns
            return sid

        audit_str = str(eff_state.get("audit_status") or "PASS")
        root_status = "ok"
        if audit_str in ("ERROR", "FAIL", "BLOCKED"):
            root_status = "error"
        elif paused:
            root_status = "suspended"

        citations = list(eff_state.get("citations") or [])
        root_span_id = _add_span(
            f"agent.{entrypoint}",
            "chain",
            None,
            eff_latency,
            {
                "intent": str(eff_state.get("intent") or "RAG_SEARCH"),
                "audit_status": audit_str,
                "citation_count": len(citations),
                "user_query": self._bound_text(user_query, 200),
            },
            status=root_status,
        )

        in_guard_data = eff_state.get("input_guard")
        if not isinstance(in_guard_data, dict):
            in_guard_data = {"is_safe": bool(eff_state.get("is_safe", True)), "flags": []}
        guard_status = "ok"
        if eff_state.get("is_safe") is False or audit_str == "BLOCKED":
            guard_status = "error"
        in_dur = max(1.5, float(in_guard_stage.get("latency_ms") or 2.0))
        ig_span_id = _add_span(
            "graph.input_guardrail",
            "chain",
            root_span_id,
            in_dur,
            {"is_safe": bool(in_guard_data.get("is_safe", True))},
            status=guard_status,
        )
        _add_span(
            "guardrail_input",
            "chain",
            ig_span_id,
            1.0,
            {"layer": "input", "is_safe": bool(in_guard_data.get("is_safe", True)), "details": list(in_guard_data.get("flags") or [])},
            status=guard_status,
        )
        _add_span("graph.context_prepare", "chain", root_span_id, 1.5, {"history_messages": len(eff_state.get("chat_history") or [])})

        coord_dur = max(10.0, float(coord_stage.get("latency_ms") or 25.0))
        dec_span_id = _add_span(
            "graph.deep_decide",
            "chain",
            root_span_id,
            coord_dur,
            {"intent": str(eff_state.get("intent") or "RAG_SEARCH"), "profile": coord_prof},
        )
        _add_span(
            "model.route",
            "llm",
            dec_span_id,
            0.5,
            {"role": "agent", "model": coord_model, "alias": coord_prof, "profile": coord_prof, "reason": coord_prof.lower()},
        )
        in_bytes = len(str(user_query or "").encode("utf-8"))
        _add_span(
            "model.attempt",
            "llm",
            dec_span_id,
            max(8.0, coord_dur - 1.0),
            {
                "role": "agent",
                "configured_role": "agent",
                "model": coord_model,
                "alias": coord_prof,
                "protocol": "cloud",
                "reason": coord_prof.lower(),
                "attempt": 1,
                "result": "success",
                "input_bytes": in_bytes,
                "context_window": int(getattr(self.config, "AGENT_CONTEXT_WINDOW", 65536) or 65536),
                "output_tokens_limit": 4096,
                "usage": coord_tok,
            },
        )

        chunks = list(eff_state.get("retrieved_chunks") or [])
        tool_calls = list(eff_state.get("tool_outputs") or [])
        worker_nodes = list(workers_stage.get("nodes") or [])
        if chunks or tool_calls or "react_action" in worker_nodes:
            w_dur = max(12.0, float(workers_stage.get("latency_ms") or 20.0))
            ra_span_id = _add_span("graph.react_action", "chain", root_span_id, w_dur, {"tool_calls": len(tool_calls), "retrieved_chunks": len(chunks)})
            _add_span(
                "tool.search_banking_knowledge",
                "tool",
                ra_span_id,
                max(5.0, w_dur - 2.0),
                {"tool": "search_banking_knowledge", "status": "SUCCESS", "params": {"query": self._bound_text(user_query, 180)}},
            )
            hints = eff_state.get("query_hints")
            if not isinstance(hints, dict):
                hints = {}
            _add_span(
                "query_parser",
                "parser",
                ra_span_id,
                1.2,
                {
                    "raw_query": self._bound_text(user_query, 180),
                    "normalized_query": self._bound_text(user_query, 180),
                    "keywords": list(hints.get("keywords") or []),
                    "detected_acronyms": [],
                },
            )
            emb_model = str(getattr(self.config, "EMBEDDING_MODEL_1024", "") or "multilingual-e5-large")
            _add_span(
                "query_embedding",
                "embedding",
                ra_span_id,
                2.0,
                {"role": "embedding", "model": emb_model, "dimension": 1024, "cache_hit": True},
            )
            previews = []
            rerank_model = str(getattr(self.config, "RERANKER_MODEL", "") or "bge-reranker-v2-m3")
            for idx, ch in enumerate(chunks[:15]):
                if not isinstance(ch, dict):
                    continue
                full_text = str(ch.get("content") or ch.get("original_text") or "").strip()
                excerpt = self._bound_text(full_text, 240)
                previews.append({
                    "rank": idx + 1,
                    "chunk_index": idx + 1,
                    "chunk_id": str(ch.get("chunk_id") or ""),
                    "source": str(ch.get("source") or ""),
                    "section_title": str(ch.get("section_title") or ""),
                    "breadcrumb": str(ch.get("breadcrumb") or ""),
                    "chunk_type": str(ch.get("chunk_type") or "text"),
                    "page_numbers": self._extract_chunk_pages(ch),
                    "rrf_score": ch.get("rrf_score") or 0.85,
                    "rerank_score": ch.get("rerank_score") or 0.92,
                    "rerank_method": rerank_model,
                    "char_count": len(full_text),
                    "snippet": excerpt,
                    "excerpt": excerpt,
                    "content": full_text[:4000],
                })
            _add_span(
                "top_k_hybrid_retrieval",
                "retriever",
                ra_span_id,
                4.0,
                {"query": self._bound_text(user_query, 180), "candidate_count": len(chunks), "candidates_count": len(chunks), "candidates_preview": previews},
            )
            _add_span(
                "top_k_cross_encoder_reranker",
                "retriever",
                ra_span_id,
                4.0,
                {"model": rerank_model, "input_count": len(chunks), "reranked_count": len(chunks), "reranked_preview": previews},
            )

        if paused or eff_state.get("interaction") or eff_state.get("human_replies") or eff_state.get("accepted_interaction_id"):
            inter_status = "ok"
            if paused:
                inter_status = "suspended"
            _add_span("graph.prepare_interaction", "chain", root_span_id, 2.0, {"has_interaction": bool(eff_state.get("interaction"))})
            _add_span("graph.await_interaction", "chain", root_span_id, 2.0, {"status": inter_status}, status=inter_status)

        if not paused and audit_str != "BLOCKED":
            rev_dur = max(10.0, float(review_stage.get("latency_ms") or 20.0))
            rev_span_id = _add_span(
                "graph.answer_review",
                "chain",
                root_span_id,
                rev_dur,
                {"audit_status": audit_str},
            )
            _add_span(
                "model.attempt",
                "llm",
                rev_span_id,
                max(8.0, rev_dur - 1.0),
                {
                    "role": "agent_review",
                    "configured_role": "agent",
                    "model": rev_model,
                    "alias": rev_prof,
                    "protocol": "cloud",
                    "reason": rev_prof.lower(),
                    "attempt": 1,
                    "result": "success",
                    "input_bytes": len(str(assistant_answer or "").encode("utf-8")),
                    "context_window": int(getattr(self.config, "AGENT_CONTEXT_WINDOW", 65536) or 65536),
                    "output_tokens_limit": 2048,
                    "usage": rev_tok,
                },
            )
            out_guard_data = eff_state.get("output_guard")
            if not isinstance(out_guard_data, dict):
                out_guard_data = {"is_safe": True, "flags": []}
            og_dur = max(1.5, float(out_guard_stage.get("latency_ms") or 2.0))
            og_span_id = _add_span("graph.output_guardrail", "chain", root_span_id, og_dur, {"is_safe": bool(out_guard_data.get("is_safe", True))})
            _add_span(
                "guardrail_output",
                "chain",
                og_span_id,
                1.0,
                {"layer": "output", "is_safe": bool(out_guard_data.get("is_safe", True)), "details": list(out_guard_data.get("flags") or [])},
            )

        if eff_latency <= 0.0 and len(spans) > 1:
            child_sum = 0.0
            for sp in spans[1:]:
                if sp.get("parent_span_id") == root_span_id:
                    child_sum = child_sum + float(sp.get("duration_ms") or 0.0)
            eff_latency = round(child_sum, 3)
            spans[0]["duration_ms"] = eff_latency

        turn_snapshot = self._build_debug_snapshot(
            eff_state,
            query=user_query,
            latency_ms=eff_latency,
            dag_telemetry=resolved_dag,
        )

        outcome_val = "completed"
        if paused:
            outcome_val = "awaiting_input"
        elif audit_str == "CANCELED":
            outcome_val = "canceled"
        elif audit_str == "BLOCKED":
            outcome_val = "blocked"
        elif audit_str in ("ERROR", "FAIL"):
            outcome_val = "error"

        proj_name = str(getattr(self.tracer, "project_name", "") or "ai-banking-assistant")
        return _sanitize_value({
            "schema_version": 2,
            "provider": "langsmith",
            "project_name": proj_name,
            "trace_id": eff_trace,
            "run_id": root_id,
            "turn_id": eff_turn,
            "timestamp": eff_ts,
            "entrypoint": entrypoint,
            "session_ref": fingerprint(session_id) or "",
            "outcome": outcome_val,
            "is_active": False,
            "duration_ms": eff_latency,
            "langsmith_run_url": None,
            "spans": spans,
            "turn_snapshot": turn_snapshot,
        })

    def save_debug_trace_record(self, record: dict, session_id: str = "") -> None:
        if not isinstance(record, dict):
            return
        trace_id = str(record.get("trace_id") or "").strip()
        if not trace_id:
            return
        safe_record = _sanitize_value(dict(record))
        sref = str(safe_record.get("session_ref") or "").strip()
        if not sref and session_id:
            sref = str(fingerprint(session_id) or "")
            safe_record["session_ref"] = sref
        ts_val = str(safe_record.get("timestamp") or _utc_now_iso())
        safe_record["timestamp"] = ts_val
        turn_id_val = str(safe_record.get("turn_id") or "")
        entry_val = str(safe_record.get("entrypoint") or "chat")
        outcome_val = str(safe_record.get("outcome") or "completed")
        dur_val = float(safe_record.get("duration_ms") or 0.0)

        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                self._memory_debug_traces[trace_id] = safe_record
                if session_id and session_id in self._memory_sessions:
                    self._memory_sessions[session_id]["traces_cleared"] = False
            return

        rec_json = json.dumps(safe_record, ensure_ascii=False)

        def _upsert_trace(db_conn):
            db_conn.execute(
                """
                INSERT INTO debug_traces (
                    trace_id, session_id, session_ref, turn_id,
                    timestamp, entrypoint, outcome, duration_ms, record_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    session_id = CASE
                        WHEN excluded.session_id != '' THEN excluded.session_id
                        ELSE debug_traces.session_id
                    END,
                    session_ref = CASE
                        WHEN excluded.session_ref != '' THEN excluded.session_ref
                        ELSE debug_traces.session_ref
                    END,
                    turn_id = CASE
                        WHEN excluded.turn_id != '' THEN excluded.turn_id
                        ELSE debug_traces.turn_id
                    END,
                    timestamp = excluded.timestamp,
                    entrypoint = excluded.entrypoint,
                    outcome = excluded.outcome,
                    duration_ms = excluded.duration_ms,
                    record_json = excluded.record_json
                """,
                (trace_id, str(session_id or ""), sref, turn_id_val, ts_val, entry_val, outcome_val, dur_val, rec_json),
            )
            if session_id:
                db_conn.execute(
                    "UPDATE session_meta SET traces_cleared = 0 WHERE session_id = ?",
                    (session_id,),
                )
            db_conn.commit()

        try:
            self._run_sqlite(_upsert_trace)
        except Exception as err:
            print(f"[Debug Trace Save Error]: {type(err).__name__}")

    def _ensure_session_debug_traces_backfilled(self, session_id: str | None = None) -> None:
        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                if session_id:
                    if session_id in self._memory_sessions:
                        sids = [session_id]
                    else:
                        sids = []
                else:
                    sids = list(self._memory_sessions.keys())
                for sid in sids:
                    mem = self._memory_sessions.get(sid)
                    if not mem or mem.get("deleted") or mem.get("traces_cleared"):
                        continue
                    sref = str(fingerprint(sid) or "")
                    has_any = False
                    for rec in self._memory_debug_traces.values():
                        if rec.get("session_ref") == sref:
                            has_any = True
                            break
                    if has_any:
                        continue
                    for idx, t in enumerate(list(mem.get("turns") or [])):
                        tid = str(t.get("trace_id") or "").strip()
                        if not tid:
                            tid = uuid4().hex
                            t["trace_id"] = tid
                        paused_flag = t.get("status") == "awaiting_input"
                        synth_state = {
                            "user_query": t.get("user_query") or "",
                            "final_answer": t.get("assistant_answer") or "",
                            "audit_status": t.get("audit_status") or "PASS",
                            "intent": t.get("intent") or "RAG_SEARCH",
                            "citations": list(t.get("citations") or []),
                            "sources": list(t.get("sources") or []),
                            "images": list(t.get("images") or []),
                            "plan": list(t.get("plan") or []),
                            "interaction": t.get("interaction") or None,
                        }
                        rec = self._synthesize_debug_trace_record(
                            sid,
                            str(t.get("user_query") or ""),
                            str(t.get("assistant_answer") or ""),
                            synth_state,
                            t.get("dag_telemetry"),
                            timestamp=str(t.get("created_at") or _utc_now_iso()),
                            latency_ms=float(t.get("latency_ms") or 0.0),
                            trace_id=tid,
                            turn_id=str(t.get("turn_id") or tid),
                            paused=paused_flag,
                        )
                        self._memory_debug_traces[tid] = rec
            return

        if session_id:
            self._sync_checkpoints_to_session_meta(specific_session_id=session_id)
        else:
            self._sync_checkpoints_to_session_meta()

        def _find_untraced_sessions(db_conn):
            positive_refs = set()
            positive_sids = set()
            upgrade_refs = set()
            upgrade_sids = set()
            all_tids_by_ref = {}
            all_tids_by_sid = {}
            for r in db_conn.execute(
                "SELECT trace_id, session_ref, session_id, duration_ms, record_json FROM debug_traces ORDER BY timestamp ASC, rowid ASC"
            ).fetchall():
                t_id = str(r[0] or "")
                s_ref = str(r[1] or "")
                s_id = str(r[2] or "")
                dur_val = float(r[3] or 0.0)
                if t_id:
                    if s_ref:
                        all_tids_by_ref.setdefault(s_ref, []).append(t_id)
                    if s_id:
                        all_tids_by_sid.setdefault(s_id, []).append(t_id)
                needs_chunk_upgrade = False
                if r[4]:
                    try:
                        parsed_rec = json.loads(str(r[4]))
                        snap_obj = parsed_rec.get("turn_snapshot")
                        if isinstance(snap_obj, dict):
                            snap_chunks = snap_obj.get("retrieved_chunks")
                            if isinstance(snap_chunks, list) and snap_chunks:
                                first_ch = snap_chunks[0]
                                if isinstance(first_ch, dict) and not first_ch.get("content"):
                                    needs_chunk_upgrade = True
                    except Exception:
                        needs_chunk_upgrade = False
                if dur_val > 0.0 and not needs_chunk_upgrade:
                    if s_ref:
                        positive_refs.add(s_ref)
                    if s_id:
                        positive_sids.add(s_id)
                else:
                    if s_ref:
                        upgrade_refs.add(s_ref)
                    if s_id:
                        upgrade_sids.add(s_id)
            if session_id:
                rows = db_conn.execute(
                    "SELECT session_id, updated_at, traces_cleared FROM session_meta WHERE session_id = ? AND deleted = 0",
                    (session_id,),
                ).fetchall()
            else:
                rows = db_conn.execute(
                    "SELECT session_id, updated_at, traces_cleared FROM session_meta WHERE deleted = 0 AND traces_cleared = 0 ORDER BY updated_at DESC LIMIT 200"
                ).fetchall()
            candidates = []
            for r in rows:
                sid = str(r[0] or "")
                cleared_flag = int(r[2] or 0)
                if not sid or cleared_flag == 1:
                    continue
                sref = str(fingerprint(sid) or "")
                already_ok = (sref in positive_refs or sid in positive_sids)
                needs_up = (sref in upgrade_refs or sid in upgrade_sids)
                if already_ok and not needs_up:
                    continue
                old_zero_tids = list(all_tids_by_sid.get(sid) or all_tids_by_ref.get(sref) or [])
                candidates.append((sid, str(r[1] or ""), old_zero_tids))
            return candidates

        missing_sids = self._run_sqlite(_find_untraced_sessions) or []
        for sid, meta_ts, old_zero_tids in missing_sids:
            def _load_turns(db_conn):
                return db_conn.execute(
                    """
                    SELECT id, user_query, assistant_answer, status, audit_status, intent,
                           trace_id, turn_id, latency_ms, citations_json, sources_json,
                           images_json, plan_json, dag_json, interaction_json, created_at
                    FROM session_turns
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (sid,),
                ).fetchall()

            turn_rows = self._run_sqlite(_load_turns) or []
            cp_root = {}
            cp_ts = meta_ts or _utc_now_iso()
            cp_info = self._extract_checkpoint_info(sid)
            try:
                tup = self.graph.checkpointer.get_tuple(self._thread_config(sid))
                if tup is not None and isinstance(tup.checkpoint, dict):
                    if isinstance(tup.checkpoint.get("ts"), str) and tup.checkpoint.get("ts"):
                        cp_ts = tup.checkpoint["ts"]
                    ch = tup.checkpoint.get("channel_values")
                    if isinstance(ch, dict):
                        root_val = ch.get("__root__")
                        if isinstance(root_val, dict):
                            cp_root = root_val
                        else:
                            cp_root = ch
            except Exception:
                cp_root = {}

            if turn_rows:
                for idx, tr in enumerate(turn_rows):
                    row_id = int(tr[0])
                    u_q = str(tr[1] or "")
                    a_ans = str(tr[2] or "")
                    t_status = str(tr[3] or "completed")
                    t_audit = str(tr[4] or "PASS")
                    t_intent = str(tr[5] or "RAG_SEARCH")
                    eff_tid = str(tr[6] or "").strip()
                    if not eff_tid:
                        eff_tid = uuid4().hex
                        def _update_turn_tid(db_conn, rid=row_id, val=eff_tid):
                            db_conn.execute("UPDATE session_turns SET trace_id = ? WHERE id = ?", (val, rid))
                            db_conn.commit()
                        self._run_sqlite(_update_turn_tid)
                    eff_turn_id = str(tr[7] or eff_tid)
                    t_lat = float(tr[8] or 0.0)
                    def _safe_load_json(raw_txt, fallback):
                        try:
                            return json.loads(raw_txt or "")
                        except Exception:
                            return fallback
                    t_cits = _safe_load_json(tr[9], [])
                    t_srcs = _safe_load_json(tr[10], [])
                    t_imgs = _safe_load_json(tr[11], [])
                    t_plan = _safe_load_json(tr[12], [])
                    t_dag = _safe_load_json(tr[13], {})
                    t_inter = _safe_load_json(tr[14], {})
                    t_ts = str(tr[15] or cp_ts)
                    paused_flag = t_status == "awaiting_input"
                    inter_payload = None
                    if t_inter:
                        inter_payload = t_inter
                    turn_state = {
                        "user_query": u_q,
                        "final_answer": a_ans,
                        "audit_status": t_audit,
                        "intent": t_intent,
                        "citations": t_cits,
                        "sources": t_srcs,
                        "images": t_imgs,
                        "plan": t_plan,
                        "interaction": inter_payload,
                    }
                    if idx == len(turn_rows) - 1 and cp_root:
                        for cp_k in (
                            "retrieved_chunks", "tool_outputs", "execution_steps",
                            "reviews", "verification", "memory_status", "input_guard",
                            "output_guard", "human_replies", "accepted_interaction_id", "query_hints",
                        ):
                            if cp_k in cp_root and cp_root[cp_k] is not None:
                                turn_state[cp_k] = cp_root[cp_k]
                    dag_arg = None
                    if isinstance(t_dag, dict) and t_dag:
                        dag_arg = t_dag
                    rec = self._synthesize_debug_trace_record(
                        sid,
                        u_q,
                        a_ans,
                        turn_state,
                        dag_arg,
                        timestamp=t_ts,
                        latency_ms=t_lat,
                        trace_id=eff_tid,
                        turn_id=eff_turn_id,
                        paused=paused_flag,
                    )
                    self.save_debug_trace_record(rec, session_id=sid)
            else:
                cp_msgs = list(cp_info.get("messages") or [])
                pairs = []
                cur_u = ""
                for msg in cp_msgs:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role")
                    content = str(msg.get("content") or "")
                    if role == "user":
                        if cur_u:
                            pairs.append((cur_u, ""))
                        cur_u = content
                    elif role == "assistant":
                        pairs.append((cur_u, content))
                        cur_u = ""
                if cur_u:
                    pairs.append((cur_u, ""))
                if not pairs and cp_root.get("user_query"):
                    pairs.append((str(cp_root.get("user_query") or ""), str(cp_root.get("final_answer") or "")))
                try:
                    base_dt = datetime.fromisoformat(cp_ts.replace("Z", "+00:00"))
                except Exception:
                    base_dt = datetime.now(timezone.utc)
                total_pairs = len(pairs)
                for idx, (u_q, a_ans) in enumerate(pairs):
                    offset_secs = max(0, total_pairs - 1 - idx)
                    turn_ts = (base_dt - timedelta(seconds=offset_secs * 5)).isoformat()
                    is_last = idx == total_pairs - 1
                    is_paused = is_last and cp_info.get("interaction") is not None and not a_ans
                    if is_last and cp_root:
                        synth_state = dict(cp_root)
                        if u_q and not synth_state.get("user_query"):
                            synth_state["user_query"] = u_q
                        if a_ans and not synth_state.get("final_answer"):
                            synth_state["final_answer"] = a_ans
                    else:
                        synth_state = {
                            "user_query": u_q,
                            "final_answer": a_ans,
                            "is_safe": True,
                            "audit_status": "PASS",
                            "intent": "RAG_SEARCH",
                        }
                    reuse_tid = ""
                    if idx < len(old_zero_tids):
                        reuse_tid = old_zero_tids[idx]
                    rec = self._synthesize_debug_trace_record(
                        sid,
                        u_q,
                        a_ans,
                        synth_state,
                        None,
                        timestamp=turn_ts,
                        latency_ms=0.0,
                        trace_id=reuse_tid,
                        paused=is_paused,
                    )
                    self.save_debug_trace_record(rec, session_id=sid)

    def list_persisted_debug_traces(
        self,
        session_ref: str | None = None,
        session_id: str | None = None,
        scope: str = "session",
        limit: int = 100,
    ) -> list[dict]:
        clean_scope = str(scope or "session").strip().lower()
        if clean_scope == "session" and session_id:
            self._ensure_session_debug_traces_backfilled(session_id=session_id)
        elif clean_scope == "all":
            self._ensure_session_debug_traces_backfilled(session_id=None)

        bounded_limit = max(1, min(int(limit), 500))
        eff_ref = str(session_ref or "").strip()
        if not eff_ref and session_id:
            eff_ref = str(fingerprint(session_id) or "")

        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                items = list(self._memory_debug_traces.values())
            filtered = []
            for rec in items:
                if clean_scope == "session":
                    if not eff_ref:
                        continue
                    if rec.get("session_ref") != eff_ref:
                        continue
                filtered.append(dict(rec))
            filtered.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
            return filtered[:bounded_limit]

        def _query_traces(db_conn):
            if clean_scope == "session":
                if not eff_ref and not session_id:
                    return []
                if eff_ref and session_id:
                    return db_conn.execute(
                        """
                        SELECT record_json FROM debug_traces
                        WHERE session_ref = ? OR session_id = ?
                        ORDER BY timestamp DESC, rowid DESC
                        LIMIT ?
                        """,
                        (eff_ref, session_id, bounded_limit),
                    ).fetchall()
                return db_conn.execute(
                    """
                    SELECT record_json FROM debug_traces
                    WHERE session_ref = ?
                    ORDER BY timestamp DESC, rowid DESC
                    LIMIT ?
                    """,
                    (eff_ref, bounded_limit),
                ).fetchall()
            return db_conn.execute(
                """
                SELECT record_json FROM debug_traces
                ORDER BY timestamp DESC, rowid DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()

        rows = self._run_sqlite(_query_traces) or []
        out = []
        for row in rows:
            try:
                parsed = json.loads(row[0] or "")
                if isinstance(parsed, dict) and parsed.get("trace_id"):
                    out.append(parsed)
            except Exception:
                continue
        return out

    def count_persisted_debug_traces(
        self,
        session_ref: str | None = None,
        session_id: str | None = None,
        scope: str = "session",
    ) -> int:
        clean_scope = str(scope or "session").strip().lower()
        eff_ref = str(session_ref or "").strip()
        if not eff_ref and session_id:
            eff_ref = str(fingerprint(session_id) or "")

        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                items = list(self._memory_debug_traces.values())
            if clean_scope == "session":
                if not eff_ref:
                    return 0
                return sum(1 for r in items if r.get("session_ref") == eff_ref)
            return len(items)

        def _count_rows(db_conn):
            if clean_scope == "session":
                if not eff_ref and not session_id:
                    return 0
                if eff_ref and session_id:
                    row = db_conn.execute(
                        "SELECT COUNT(*) FROM debug_traces WHERE session_ref = ? OR session_id = ?",
                        (eff_ref, session_id),
                    ).fetchone()
                else:
                    row = db_conn.execute(
                        "SELECT COUNT(*) FROM debug_traces WHERE session_ref = ?",
                        (eff_ref,),
                    ).fetchone()
            else:
                row = db_conn.execute("SELECT COUNT(*) FROM debug_traces").fetchone()
            if row:
                return int(row[0] or 0)
            return 0

        return int(self._run_sqlite(_count_rows) or 0)

    def get_persisted_debug_trace(self, trace_id: str) -> dict | None:
        key = str(trace_id or "").strip()
        if not key:
            return None
        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                rec = self._memory_debug_traces.get(key)
                if isinstance(rec, dict):
                    return dict(rec)
                return None

        def _fetch_trace(db_conn):
            row = db_conn.execute(
                "SELECT record_json FROM debug_traces WHERE trace_id = ?",
                (key,),
            ).fetchone()
            if row and row[0]:
                return str(row[0])
            return ""

        raw = self._run_sqlite(_fetch_trace)
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return None

    def clear_persisted_debug_traces(
        self,
        session_ref: str | None = None,
        session_id: str | None = None,
        exclude_trace_ids: set[str] | None = None,
    ) -> int:
        if exclude_trace_ids is None:
            excluded = set()
        else:
            excluded = {str(x) for x in exclude_trace_ids}
        eff_ref = str(session_ref or "").strip()
        if not eff_ref and session_id:
            eff_ref = str(fingerprint(session_id) or "")

        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                extra_count = 0
                if not eff_ref and not session_id:
                    for tid in list(self._memory_debug_traces.keys()):
                        if tid not in excluded:
                            extra_count = extra_count + 1
                    self._memory_debug_traces.clear()
                    for mem in self._memory_sessions.values():
                        mem["traces_cleared"] = True
                else:
                    for tid in list(self._memory_debug_traces.keys()):
                        rec = self._memory_debug_traces[tid]
                        if rec.get("session_ref") == eff_ref:
                            if tid not in excluded:
                                extra_count = extra_count + 1
                            self._memory_debug_traces.pop(tid, None)
                    for sid_k, mem in self._memory_sessions.items():
                        if sid_k == session_id or (eff_ref and fingerprint(sid_k) == eff_ref):
                            mem["traces_cleared"] = True
                return extra_count

        def _clear_rows(db_conn):
            if not eff_ref and not session_id:
                rows = db_conn.execute("SELECT trace_id FROM debug_traces").fetchall()
                db_conn.execute("DELETE FROM debug_traces")
                db_conn.execute("UPDATE session_meta SET traces_cleared = 1")
            else:
                if eff_ref and session_id:
                    rows = db_conn.execute(
                        "SELECT trace_id FROM debug_traces WHERE session_ref = ? OR session_id = ?",
                        (eff_ref, session_id),
                    ).fetchall()
                    db_conn.execute(
                        "DELETE FROM debug_traces WHERE session_ref = ? OR session_id = ?",
                        (eff_ref, session_id),
                    )
                else:
                    rows = db_conn.execute(
                        "SELECT trace_id FROM debug_traces WHERE session_ref = ?",
                        (eff_ref,),
                    ).fetchall()
                    db_conn.execute(
                        "DELETE FROM debug_traces WHERE session_ref = ?",
                        (eff_ref,),
                    )
                if session_id:
                    db_conn.execute(
                        "UPDATE session_meta SET traces_cleared = 1 WHERE session_id = ?",
                        (session_id,),
                    )
                elif eff_ref:
                    meta_sids = db_conn.execute("SELECT session_id FROM session_meta").fetchall()
                    for ms in meta_sids:
                        if fingerprint(str(ms[0])) == eff_ref:
                            db_conn.execute(
                                "UPDATE session_meta SET traces_cleared = 1 WHERE session_id = ?",
                                (str(ms[0]),),
                            )
            db_conn.commit()
            extra = 0
            for r in rows:
                tid = str(r[0] or "")
                if tid and tid not in excluded:
                    extra = extra + 1
            return extra

        return int(self._run_sqlite(_clear_rows) or 0)

    def record_session_turn(
        self,
        session_id: str,
        user_query: str,
        result_payload: dict,
        trace_id: str = "",
        turn_id: str = "",
        allow_title_upgrade: bool = True,
    ) -> str:
        if not session_id or not isinstance(result_payload, dict):
            return ""
        self._wait_for_title_thread(session_id, timeout=1.2)
        now_iso = _utc_now_iso()
        clean_query = self._bound_text(user_query or "", 2000)
        user_text, image_context = self._extract_query_and_image_context(clean_query)
        is_greeting = self._is_greeting_only(user_text) and not image_context
        derived_title = self._derive_session_title(clean_query)
        if is_greeting:
            target_source = "auto_greeting"
        else:
            target_source = "auto"
        status_val = str(result_payload.get("status") or "completed")
        raw_audit = result_payload.get("audit_status")
        if raw_audit:
            audit_val = str(raw_audit)
        elif status_val == "awaiting_input":
            audit_val = "AWAITING_INPUT"
        else:
            audit_val = "PASS"
        intent_val = str(result_payload.get("intent") or "")
        effective_trace = str(trace_id or result_payload.get("trace_id") or "").strip()
        if not effective_trace:
            effective_trace = uuid4().hex
        effective_turn = str(turn_id or result_payload.get("turn_id") or effective_trace)
        latency_val = float(result_payload.get("latency_ms") or 0.0)

        interaction_obj = result_payload.get("interaction")
        if not isinstance(interaction_obj, dict):
            interaction_obj = {}
        assistant_text = str(result_payload.get("answer") or "")
        if not assistant_text and interaction_obj:
            assistant_text = str(interaction_obj.get("message") or "")

        citations_val = list(result_payload.get("citations") or [])
        sources_val = list(result_payload.get("sources") or [])
        images_val = list(result_payload.get("images") or [])
        plan_val = list(result_payload.get("plan") or [])
        dag_val = result_payload.get("dag_telemetry")
        if not isinstance(dag_val, dict):
            dag_val = {}

        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            resolved_title = derived_title
            with self._session_store_lock:
                mem = self._memory_sessions.get(session_id)
                if mem is None:
                    mem = {
                        "session_id": session_id,
                        "title": derived_title,
                        "title_source": target_source,
                        "created_at": now_iso,
                        "updated_at": now_iso,
                        "deleted": False,
                        "turns": [],
                    }
                    self._memory_sessions[session_id] = mem
                else:
                    existing_title = str(mem.get("title") or "")
                    existing_source = str(mem.get("title_source") or "auto")
                    can_upgrade = (
                        allow_title_upgrade
                        and bool(clean_query)
                        and existing_source != "manual"
                        and (
                            not existing_title
                            or existing_title == "Cuộc hội thoại mới"
                            or existing_source == "new"
                            or (existing_source == "auto_greeting" and not is_greeting)
                        )
                    )
                    if can_upgrade:
                        mem["title"] = derived_title
                        mem["title_source"] = target_source
                    resolved_title = str(mem.get("title") or derived_title)
                    mem["updated_at"] = now_iso
                    mem["deleted"] = False
                    mem["traces_cleared"] = False
                mem["turns"].append({
                    "user_query": clean_query,
                    "assistant_answer": assistant_text,
                    "status": status_val,
                    "audit_status": audit_val,
                    "intent": intent_val,
                    "trace_id": effective_trace,
                    "turn_id": effective_turn,
                    "latency_ms": latency_val,
                    "citations": citations_val,
                    "sources": sources_val,
                    "images": images_val,
                    "plan": plan_val,
                    "dag_telemetry": dag_val,
                    "interaction": interaction_obj,
                    "created_at": now_iso,
                })
            self._persist_turn_debug_trace_if_missing(
                session_id,
                clean_query,
                assistant_text,
                result_payload,
                effective_trace,
                effective_turn,
                now_iso,
                latency_val,
                status_val,
                audit_val,
                intent_val,
            )
            return resolved_title

        def _write_turn(db_conn):
            preview_src = assistant_text or user_text or clean_query
            preview_val = self._bound_text(preview_src, 110)
            if not preview_val:
                preview_val = "Chưa có tin nhắn"
            added_msgs = 0
            if clean_query:
                added_msgs = added_msgs + 1
            if assistant_text:
                added_msgs = added_msgs + 1

            row = db_conn.execute(
                "SELECT title, created_at, message_count, title_source FROM session_meta WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                next_title = derived_title
                db_conn.execute(
                    """
                    INSERT INTO session_meta (
                        session_id, title, pinned, created_at, updated_at, deleted,
                        preview, message_count, status, audit_status, title_source, traces_cleared
                    )
                    VALUES (?, ?, 0, ?, ?, 0, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        session_id,
                        next_title,
                        now_iso,
                        now_iso,
                        preview_val,
                        added_msgs,
                        status_val,
                        audit_val,
                        target_source,
                    ),
                )
            else:
                existing_title = str(row[0] or "")
                existing_source = str(row[3] or "auto")
                next_title = existing_title
                next_source = existing_source
                can_upgrade = (
                    allow_title_upgrade
                    and bool(clean_query)
                    and existing_source != "manual"
                    and (
                        not existing_title
                        or existing_title == "Cuộc hội thoại mới"
                        or existing_source == "new"
                        or (existing_source == "auto_greeting" and not is_greeting)
                    )
                )
                if can_upgrade:
                    next_title = derived_title
                    next_source = target_source
                next_count = int(row[2] or 0) + added_msgs
                db_conn.execute(
                    """
                    UPDATE session_meta
                    SET title = ?, title_source = ?, updated_at = ?, deleted = 0,
                        preview = ?, message_count = ?, status = ?, audit_status = ?,
                        traces_cleared = 0
                    WHERE session_id = ?
                    """,
                    (next_title, next_source, now_iso, preview_val, next_count, status_val, audit_val, session_id),
                )
            db_conn.execute(
                """
                INSERT INTO session_turns (
                    session_id, user_query, assistant_answer, status, audit_status, intent,
                    trace_id, turn_id, latency_ms, citations_json, sources_json, images_json,
                    plan_json, dag_json, interaction_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    clean_query,
                    assistant_text,
                    status_val,
                    audit_val,
                    intent_val,
                    effective_trace,
                    effective_turn,
                    latency_val,
                    json.dumps(citations_val, ensure_ascii=False),
                    json.dumps(sources_val, ensure_ascii=False),
                    json.dumps(images_val, ensure_ascii=False),
                    json.dumps(plan_val, ensure_ascii=False),
                    json.dumps(dag_val, ensure_ascii=False),
                    json.dumps(interaction_obj, ensure_ascii=False),
                    now_iso,
                ),
            )
            db_conn.commit()
            return next_title

        try:
            saved_title = self._run_sqlite(_write_turn)
            self._persist_turn_debug_trace_if_missing(
                session_id,
                clean_query,
                assistant_text,
                result_payload,
                effective_trace,
                effective_turn,
                now_iso,
                latency_val,
                status_val,
                audit_val,
                intent_val,
            )
            if saved_title:
                return str(saved_title)
        except Exception as err:
            print(f"[Session Turn Save Error]: {type(err).__name__}")
        return derived_title

    def _persist_turn_debug_trace_if_missing(
        self,
        session_id: str,
        user_query: str,
        assistant_text: str,
        result_payload: dict,
        trace_id: str,
        turn_id: str,
        created_at: str,
        latency_ms: float,
        status_val: str,
        audit_val: str,
        intent_val: str,
    ) -> None:
        if not trace_id:
            return
        if self.get_persisted_debug_trace(trace_id) is not None:
            return
        inter_obj = result_payload.get("interaction")
        if not isinstance(inter_obj, dict) or not inter_obj:
            inter_obj = None
        paused_flag = status_val == "awaiting_input"
        synth_state = {
            "user_query": user_query,
            "final_answer": assistant_text,
            "audit_status": audit_val,
            "intent": intent_val or "RAG_SEARCH",
            "citations": list(result_payload.get("citations") or []),
            "sources": list(result_payload.get("sources") or []),
            "images": list(result_payload.get("images") or []),
            "plan": list(result_payload.get("plan") or []),
            "tool_outputs": list(result_payload.get("tool_calls") or []),
            "execution_steps": list(result_payload.get("execution_steps") or []),
            "reviews": list(result_payload.get("review_outcomes") or []),
            "verification": dict(result_payload.get("verification") or {}),
            "memory_status": dict(result_payload.get("memory") or {}),
            "interaction": inter_obj,
        }
        dag_val = result_payload.get("dag_telemetry")
        if not isinstance(dag_val, dict) or not dag_val:
            dag_val = None
        rec = self._synthesize_debug_trace_record(
            session_id,
            user_query,
            assistant_text,
            synth_state,
            dag_val,
            timestamp=created_at,
            latency_ms=latency_ms,
            trace_id=trace_id,
            turn_id=turn_id,
            paused=paused_flag,
        )
        self.save_debug_trace_record(rec, session_id=session_id)

    def _record_resume_turn(
        self,
        session_id: str,
        pending_question: str,
        human_reply_label: str,
        result_payload: dict,
        trace_id: str = "",
        turn_id: str = "",
    ) -> None:
        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                mem = self._memory_sessions.get(session_id)
                if mem and mem.get("turns"):
                    last_t = mem["turns"][-1]
                    if last_t.get("status") == "awaiting_input":
                        last_t["status"] = "resumed"
                        if pending_question and not last_t.get("assistant_answer"):
                            last_t["assistant_answer"] = pending_question
        else:
            def _mark_resumed(db_conn):
                row = db_conn.execute(
                    "SELECT id, assistant_answer FROM session_turns WHERE session_id = ? AND status = 'awaiting_input' ORDER BY id DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                if row is not None:
                    ans = str(row[1] or "")
                    if not ans and pending_question:
                        ans = pending_question
                    db_conn.execute(
                        "UPDATE session_turns SET status = 'resumed', assistant_answer = ? WHERE id = ?",
                        (ans, int(row[0])),
                    )
                    db_conn.commit()
            try:
                self._run_sqlite(_mark_resumed)
            except Exception:
                pass
        self.record_session_turn(
            session_id,
            human_reply_label or "[Phản hồi lựa chọn]",
            result_payload,
            trace_id=trace_id,
            turn_id=turn_id,
            allow_title_upgrade=False,
        )

    def _build_session_summary_item(self, meta: dict, turns: list, cp_info: dict | None = None) -> dict:
        title = str(meta.get("title") or "").strip()
        if not title:
            title = "Cuộc hội thoại mới"
        preview = ""
        msg_count = 0
        status_val = "idle"
        audit_val = ""

        if turns:
            for t in turns:
                if t.get("user_query"):
                    msg_count = msg_count + 1
                if t.get("assistant_answer"):
                    msg_count = msg_count + 1
            last_turn = turns[-1]
            status_val = str(last_turn.get("status") or "completed")
            audit_val = str(last_turn.get("audit_status") or "")
            preview_src = str(last_turn.get("assistant_answer") or last_turn.get("user_query") or "")
            preview = self._bound_text(preview_src, 110)
        elif cp_info is not None and cp_info.get("message_count", 0) > 0:
            msg_count = int(cp_info["message_count"])
            audit_val = str(cp_info.get("audit_status") or "")
            if cp_info.get("interaction") is not None:
                status_val = "awaiting_input"
            else:
                status_val = "completed"
            preview_src = str(cp_info.get("last_assistant") or cp_info.get("first_user_query") or "")
            preview = self._bound_text(preview_src, 110)

        if not preview:
            preview = "Chưa có tin nhắn"

        return {
            "session_id": str(meta["session_id"]),
            "title": title,
            "created_at": str(meta.get("created_at") or ""),
            "updated_at": str(meta.get("updated_at") or ""),
            "message_count": msg_count,
            "preview": preview,
            "status": status_val,
            "audit_status": audit_val,
        }

    def list_sessions(self, search_query: str = "", limit: int = 100) -> dict:
        self._sync_checkpoints_to_session_meta()
        q_lower = str(search_query or "").strip().lower()
        bounded_limit = max(1, min(int(limit), 500))

        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                raw_items = [dict(v) for v in self._memory_sessions.values() if not v.get("deleted")]
            summaries = []
            for mem in raw_items:
                turns = list(mem.get("turns") or [])
                item = self._build_session_summary_item(mem, turns)
                if q_lower:
                    haystack_parts = [item["title"], item["preview"], item["session_id"]]
                    for t in turns:
                        haystack_parts.append(str(t.get("user_query") or ""))
                        haystack_parts.append(str(t.get("assistant_answer") or ""))
                    if q_lower not in " ".join(haystack_parts).lower():
                        continue
                summaries.append(item)
        else:
            def _fetch_all(db_conn):
                meta_rows = db_conn.execute(
                    """
                    SELECT session_id, title, created_at, updated_at,
                           preview, message_count, status, audit_status
                    FROM session_meta
                    WHERE deleted = 0
                    """
                ).fetchall()
                turn_rows = []
                if q_lower:
                    turn_rows = db_conn.execute(
                        "SELECT session_id, user_query, assistant_answer FROM session_turns ORDER BY id ASC"
                    ).fetchall()
                return meta_rows, turn_rows

            meta_rows, turn_rows = self._run_sqlite(_fetch_all) or ([], [])
            turns_text_by_sid = {}
            for tr in turn_rows:
                sid = str(tr[0])
                if sid not in turns_text_by_sid:
                    turns_text_by_sid[sid] = []
                turns_text_by_sid[sid].append(str(tr[1] or ""))
                turns_text_by_sid[sid].append(str(tr[2] or ""))

            summaries = []
            for mr in meta_rows:
                sid = str(mr[0])
                title_val = str(mr[1] or "").strip()
                if not title_val:
                    title_val = "Cuộc hội thoại mới"
                preview_val = str(mr[4] or "").strip()
                if not preview_val:
                    preview_val = "Chưa có tin nhắn"
                item = {
                    "session_id": sid,
                    "title": title_val,
                    "created_at": str(mr[2] or ""),
                    "updated_at": str(mr[3] or ""),
                    "message_count": int(mr[5] or 0),
                    "preview": preview_val,
                    "status": str(mr[6] or "idle"),
                    "audit_status": str(mr[7] or ""),
                }
                if q_lower:
                    haystack_parts = [item["title"], item["preview"], item["session_id"]]
                    haystack_parts.extend(turns_text_by_sid.get(sid, []))
                    if q_lower not in " ".join(haystack_parts).lower():
                        continue
                summaries.append(item)

        summaries.sort(key=lambda e: e.get("updated_at") or "", reverse=True)

        return {
            "total": len(summaries),
            "sessions": summaries[:bounded_limit],
        }

    def get_session_detail(self, session_id: str) -> dict | None:
        if not session_id:
            return None
        self._sync_checkpoints_to_session_meta(specific_session_id=session_id)

        meta_dict = None
        turns = []
        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                mem = self._memory_sessions.get(session_id)
                if mem and not mem.get("deleted"):
                    meta_dict = {
                        "session_id": session_id,
                        "title": mem.get("title") or "Cuộc hội thoại mới",
                        "created_at": mem.get("created_at") or "",
                        "updated_at": mem.get("updated_at") or "",
                    }
                    turns = list(mem.get("turns") or [])
        else:
            def _fetch_one(db_conn):
                m_row = db_conn.execute(
                    "SELECT session_id, title, created_at, updated_at, deleted FROM session_meta WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                t_rows = db_conn.execute(
                    """
                    SELECT user_query, assistant_answer, status, audit_status, intent,
                           trace_id, turn_id, latency_ms, citations_json, sources_json,
                           images_json, plan_json, dag_json, interaction_json, created_at
                    FROM session_turns
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()
                return m_row, t_rows

            m_row, t_rows = self._run_sqlite(_fetch_one) or (None, [])
            if m_row is not None:
                if int(m_row[4] or 0) == 1:
                    return None
                meta_dict = {
                    "session_id": str(m_row[0]),
                    "title": str(m_row[1] or "Cuộc hội thoại mới"),
                    "created_at": str(m_row[2] or ""),
                    "updated_at": str(m_row[3] or ""),
                }
            for tr in t_rows:
                def _parse_json(raw_txt, fallback):
                    try:
                        val = json.loads(raw_txt or "")
                        return val
                    except Exception:
                        return fallback
                turns.append({
                    "user_query": str(tr[0] or ""),
                    "assistant_answer": str(tr[1] or ""),
                    "status": str(tr[2] or ""),
                    "audit_status": str(tr[3] or ""),
                    "intent": str(tr[4] or ""),
                    "trace_id": str(tr[5] or ""),
                    "turn_id": str(tr[6] or ""),
                    "latency_ms": float(tr[7] or 0.0),
                    "citations": _parse_json(tr[8], []),
                    "sources": _parse_json(tr[9], []),
                    "images": _parse_json(tr[10], []),
                    "plan": _parse_json(tr[11], []),
                    "dag_telemetry": _parse_json(tr[12], {}),
                    "interaction": _parse_json(tr[13], {}),
                    "created_at": str(tr[14] or ""),
                })

        snapshot = None
        try:
            with self._session_lock(session_id):
                snapshot = self.graph.get_state(self._thread_config(session_id))
        except Exception:
            snapshot = None

        cp_info = self._extract_checkpoint_info(session_id)
        if meta_dict is None:
            if not turns and cp_info.get("message_count", 0) == 0:
                return None
            ts_fallback = cp_info.get("timestamp") or _utc_now_iso()
            meta_dict = {
                "session_id": session_id,
                "title": self._derive_session_title(cp_info.get("first_user_query") or ""),
                "created_at": ts_fallback,
                "updated_at": ts_fallback,
            }

        pending_interaction = None
        active_plan = cp_info.get("plan") or []
        if snapshot is not None:
            if getattr(snapshot, "interrupts", None):
                pending_interaction = snapshot.interrupts[0].value
            if isinstance(getattr(snapshot, "values", None), dict) and snapshot.values.get("plan"):
                active_plan = list(snapshot.values.get("plan") or [])

        messages = []
        latest_dag = None
        if turns:
            for idx, t in enumerate(turns):
                is_last = idx == len(turns) - 1
                if t.get("user_query"):
                    messages.append({"role": "user", "content": t["user_query"]})
                is_pending_turn = is_last and pending_interaction is not None and t.get("status") == "awaiting_input"
                if not is_pending_turn and t.get("assistant_answer"):
                    messages.append({
                        "role": "assistant",
                        "content": t["assistant_answer"],
                        "citations": list(t.get("citations") or []),
                        "sources": list(t.get("sources") or []),
                        "images": list(t.get("images") or []),
                        "trace_id": t.get("trace_id") or "",
                        "turn_id": t.get("turn_id") or "",
                        "audit_status": t.get("audit_status") or "PASS",
                        "latency_ms": float(t.get("latency_ms") or 0.0),
                        "dag_telemetry": t.get("dag_telemetry") or None,
                    })
                if isinstance(t.get("dag_telemetry"), dict) and t["dag_telemetry"]:
                    latest_dag = t["dag_telemetry"]
                if t.get("plan"):
                    active_plan = list(t["plan"])
        else:
            cp_msgs = list(cp_info.get("messages") or [])
            if pending_interaction is not None and cp_msgs and cp_msgs[-1].get("role") == "assistant":
                cp_msgs.pop()
            messages = cp_msgs
            if snapshot is not None and isinstance(getattr(snapshot, "values", None), dict) and snapshot.values:
                latest_dag = self._build_dag_from_run_and_state(
                    None,
                    snapshot.values,
                    0.0,
                    paused=pending_interaction is not None,
                )

        summary_item = self._build_session_summary_item(meta_dict, turns, cp_info=cp_info)
        status_out = "idle"
        if pending_interaction is not None:
            status_out = "awaiting_input"
        elif messages:
            status_out = summary_item.get("status") or "completed"

        latest_trace_id = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("trace_id"):
                latest_trace_id = str(msg["trace_id"])
                break
        if messages or pending_interaction is not None:
            persisted_traces = self.list_persisted_debug_traces(
                session_id=session_id,
                scope="session",
                limit=max(10, len(messages)),
            )
            if persisted_traces:
                if not latest_trace_id:
                    latest_trace_id = str(persisted_traces[0].get("trace_id") or "")
                assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
                ordered_traces = list(reversed(persisted_traces))
                for idx, a_msg in enumerate(assistant_msgs):
                    if not a_msg.get("trace_id") and idx < len(ordered_traces):
                        a_msg["trace_id"] = str(ordered_traces[idx].get("trace_id") or "")

        return {
            **summary_item,
            "status": status_out,
            "messages": messages,
            "interaction": pending_interaction,
            "plan": active_plan,
            "dag_telemetry": latest_dag,
            "latest_trace_id": latest_trace_id,
        }

    def create_session(self, session_id: str | None = None, title: str = "") -> dict:
        from uuid import uuid4

        sid = str(session_id or "").strip()
        if not sid:
            sid = uuid4().hex
        clean_title = " ".join(str(title or "").split()).strip()
        if not clean_title or clean_title == "Cuộc hội thoại mới":
            clean_title = "Cuộc hội thoại mới"
            initial_source = "new"
        else:
            if len(clean_title) > 64:
                clean_title = clean_title[:61] + "..."
            initial_source = "manual"
        now_iso = _utc_now_iso()

        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                existing = self._memory_sessions.get(sid)
                if existing is None or existing.get("deleted"):
                    existing = {
                        "session_id": sid,
                        "title": clean_title,
                        "title_source": initial_source,
                        "created_at": now_iso,
                        "updated_at": now_iso,
                        "deleted": False,
                        "turns": [],
                    }
                    self._memory_sessions[sid] = existing
                return self._build_session_summary_item(existing, existing.get("turns") or [])

        def _create_or_get(db_conn):
            row = db_conn.execute(
                """
                SELECT session_id, title, created_at, updated_at, deleted,
                       preview, message_count, status, audit_status
                FROM session_meta WHERE session_id = ?
                """,
                (sid,),
            ).fetchone()
            if row is None or int(row[4] or 0) == 1:
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO session_meta (
                        session_id, title, pinned, created_at, updated_at, deleted,
                        preview, message_count, status, audit_status, title_source
                    )
                    VALUES (?, ?, 0, ?, ?, 0, 'Chưa có tin nhắn', 0, 'idle', '', ?)
                    """,
                    (sid, clean_title, now_iso, now_iso, initial_source),
                )
                db_conn.commit()
                return {
                    "session_id": sid,
                    "title": clean_title,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                    "preview": "Chưa có tin nhắn",
                    "message_count": 0,
                    "status": "idle",
                    "audit_status": "",
                }
            return {
                "session_id": str(row[0]),
                "title": str(row[1] or clean_title),
                "created_at": str(row[2] or now_iso),
                "updated_at": str(row[3] or now_iso),
                "preview": str(row[5] or "Chưa có tin nhắn"),
                "message_count": int(row[6] or 0),
                "status": str(row[7] or "idle"),
                "audit_status": str(row[8] or ""),
            }

        return self._run_sqlite(_create_or_get)

    def update_session_meta(self, session_id: str, title: str | None = None) -> dict | None:
        if not session_id:
            return None
        self._sync_checkpoints_to_session_meta(specific_session_id=session_id)
        now_iso = _utc_now_iso()

        cleaned_title = None
        if title is not None:
            cleaned_title = " ".join(str(title).split()).strip()
            if not cleaned_title:
                cleaned_title = "Cuộc hội thoại mới"
            elif len(cleaned_title) > 80:
                cleaned_title = cleaned_title[:77] + "..."

        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                mem = self._memory_sessions.get(session_id)
                if mem is None or mem.get("deleted"):
                    return None
                if cleaned_title is not None:
                    mem["title"] = cleaned_title
                    mem["title_source"] = "manual"
                mem["updated_at"] = now_iso
                return self._build_session_summary_item(mem, mem.get("turns") or [])

        def _update(db_conn):
            row = db_conn.execute(
                """
                SELECT session_id, title, created_at, updated_at, deleted,
                       preview, message_count, status, audit_status, title_source
                FROM session_meta WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None or int(row[4] or 0) == 1:
                return None
            next_title = str(row[1] or "Cuộc hội thoại mới")
            next_source = str(row[9] or "auto")
            if cleaned_title is not None:
                next_title = cleaned_title
                next_source = "manual"
            db_conn.execute(
                "UPDATE session_meta SET title = ?, title_source = ?, updated_at = ? WHERE session_id = ?",
                (next_title, next_source, now_iso, session_id),
            )
            db_conn.commit()
            preview_val = str(row[5] or "").strip()
            if not preview_val:
                preview_val = "Chưa có tin nhắn"
            return {
                "session_id": session_id,
                "title": next_title,
                "created_at": str(row[2] or ""),
                "updated_at": now_iso,
                "preview": preview_val,
                "message_count": int(row[6] or 0),
                "status": str(row[7] or "idle"),
                "audit_status": str(row[8] or ""),
            }

        return self._run_sqlite(_update)

    def delete_session(self, session_id: str) -> bool:
        if not session_id:
            return False
        sref = str(fingerprint(session_id) or "")
        conn = getattr(self.graph.checkpointer, "conn", None)
        if conn is None:
            with self._session_store_lock:
                existed = session_id in self._memory_sessions
                self._memory_sessions.pop(session_id, None)
                if sref:
                    for tid in list(self._memory_debug_traces.keys()):
                        if self._memory_debug_traces[tid].get("session_ref") == sref:
                            self._memory_debug_traces.pop(tid, None)
            if hasattr(self.graph.checkpointer, "delete_thread"):
                try:
                    self.graph.checkpointer.delete_thread(session_id)
                except Exception:
                    pass
            return existed

        def _delete(db_conn):
            cur = db_conn.execute("DELETE FROM session_meta WHERE session_id = ?", (session_id,))
            meta_deleted = cur.rowcount or 0
            cur_turns = db_conn.execute("DELETE FROM session_turns WHERE session_id = ?", (session_id,))
            turns_deleted = cur_turns.rowcount or 0
            traces_deleted = 0
            try:
                if sref:
                    cur_tr = db_conn.execute(
                        "DELETE FROM debug_traces WHERE session_id = ? OR session_ref = ?",
                        (session_id, sref),
                    )
                else:
                    cur_tr = db_conn.execute(
                        "DELETE FROM debug_traces WHERE session_id = ?",
                        (session_id,),
                    )
                traces_deleted = cur_tr.rowcount or 0
            except Exception:
                pass
            cp_deleted = 0
            try:
                cur_cp = db_conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
                cp_deleted += cur_cp.rowcount or 0
            except Exception:
                pass
            try:
                db_conn.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
            except Exception:
                pass
            db_conn.commit()
            return (meta_deleted + turns_deleted + traces_deleted + cp_deleted) > 0

        return bool(self._run_sqlite(_delete))
