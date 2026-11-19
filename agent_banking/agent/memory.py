"""Session-scoped working memory backed by the graph's SQLite checkpoint."""

from agent_banking.agent.config.response_profiles import get_messages, get_prompt
from agent_banking.agent.config.settings import get_int_setting, get_setting
from agent_banking.agent.model_protocol import SUMMARY_SCHEMA
from agent_banking.core.fallback import ModelFallbackError


RECENT_MESSAGES = get_int_setting("recent_messages")
MAX_HISTORY_MESSAGES = get_int_setting("max_history_messages")
MAX_HISTORY_CHARS = get_int_setting("max_history_chars")
MAX_NOTES = get_int_setting("max_notes")
MAX_COMPACT_PER_TURN = get_int_setting("max_compact_per_turn")
SUMMARY_PROMPT = get_prompt("SUMMARY_PROMPT")

_KEY_MAX_LENGTH = get_int_setting("memory_key_max_length")
_KEY_ALLOWED_CHARS = str(get_setting("memory_key_allowed_chars", ""))
_OLDER_MESSAGE_MAX_CHARS = get_int_setting("memory_older_message_max_chars")
_FALLBACK_EXCERPT_CHARS = get_int_setting("memory_fallback_excerpt_chars")
_SUMMARY_MAX_CHARS = get_int_setting("memory_summary_max_chars")
_SUMMARY_MAX_TOKENS = get_int_setting("summary_max_tokens")
_MEM_MESSAGES = get_messages("memory")


def memory_view(state: dict) -> dict:
    return state.get("session_memory") or {"summary": "", "notes": {}, "compact_count": 0}


def should_compact(history: list[dict]) -> bool:
    if len(history) > MAX_HISTORY_MESSAGES:
        return True
    return len(history) > RECENT_MESSAGES and sum(len(item["content"]) for item in history) > MAX_HISTORY_CHARS


def public_memory(state: dict) -> dict:
    memory = memory_view(state)
    return {
        "note_count": len(memory["notes"]),
        "compact_count": memory["compact_count"],
        "recent_messages": len(state.get("chat_history", []))
    }


def valid_key(key: str) -> bool:
    return bool(key) and len(key) <= _KEY_MAX_LENGTH and all(
        c in _KEY_ALLOWED_CHARS for c in key
    )


def change_note(memory: dict, decision: dict, user_messages: list[str], guard) -> tuple[dict, str]:
    key = decision["key"].strip().lower()
    if not valid_key(key):
        return memory, _MEM_MESSAGES.get("invalid_key", "")
    notes = dict(memory["notes"])
    if decision["kind"] == "forget":
        if key not in notes:
            return memory, _MEM_MESSAGES.get("missing_key", "")
        del notes[key]
    else:
        value = decision["value"].strip()
        # An exact quotation makes the source of a durable fact auditable.
        if not value or not any(value in message for message in user_messages):
            return memory, _MEM_MESSAGES.get("value_must_quote_user", "")
        if guard(value) != value or "[REDACTED]" in value.upper() or "_REDACTED]" in value.upper():
            return memory, _MEM_MESSAGES.get("guardrail_rejected", "")
        if key not in notes and len(notes) >= MAX_NOTES:
            return memory, _MEM_MESSAGES.get("memory_full", "")
        notes[key] = {"category": decision["category"], "value": value}
    return {**memory, "notes": notes}, ""


def compact_memory(state: dict, model, guard) -> tuple[dict, bool]:
    history = state.get("chat_history", [])
    memory = memory_view(state)
    if len(history) <= RECENT_MESSAGES:
        return state, False
    older = history[:-RECENT_MESSAGES]
    recent = history[-RECENT_MESSAGES:]
    safe_older = [
        {"role": item["role"], "content": guard(item["content"][:_OLDER_MESSAGE_MAX_CHARS])}
        for item in older[-MAX_HISTORY_MESSAGES:]
    ]
    safe_older = [item for item in safe_older if item["content"]]
    summary = memory["summary"]
    try:
        candidate = model.request(
            SUMMARY_PROMPT, {"previous_summary": summary, "older_messages": safe_older},
            [SUMMARY_SCHEMA], max_tokens=_SUMMARY_MAX_TOKENS, role="agent_memory"
        )
        cleaned = guard(candidate["summary"])
        if cleaned:
            summary = cleaned
        else:
            raise ValueError("Unsafe summary")
    except (ModelFallbackError, ValueError):
        excerpts = [item["content"][:_FALLBACK_EXCERPT_CHARS] for item in safe_older if item["role"] == "user"]
        # Preserve user wording if summarization is temporarily unavailable.
        summary = "\n".join([summary, *excerpts]).strip()[-_SUMMARY_MAX_CHARS:]
    return {
        **state,
        "chat_history": recent,
        "session_memory": {**memory, "summary": summary, "compact_count": memory["compact_count"] + 1}
    }, True
