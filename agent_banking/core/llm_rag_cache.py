# -*- coding: utf-8 -*-
"""Persistent SQLite WAL cache & JSON caller for Model-First RAG components.

Uses configured cloud LLM chains (DeepSeek-V4-Flash primary, jev-latest fallback)
and strict JSON parsing with 0% Regex, Zero Ternary, and No __init__.py.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import Lock

from agent_banking.core.chat import FallbackChatClient
from agent_banking.core.environment import ROOT_DIR
from agent_banking.core.models import get_model_chain, model_configured

DEFAULT_LLM_RAG_CACHE_DB = "agent_banking/vector_db/llm_rag_cache.db"
_DB_LOCK = Lock()
_INITIALIZED_PATHS = set()


def _resolve_cache_path(cache_db_path=None) -> Path:
    raw = str(cache_db_path or DEFAULT_LLM_RAG_CACHE_DB).strip()
    p = Path(raw)
    if p.is_absolute():
        return p
    return (ROOT_DIR / p).resolve()


def _ensure_table(db_path: Path):
    key = str(db_path)
    if key in _INITIALIZED_PATHS:
        return
    with _DB_LOCK:
        if key in _INITIALIZED_PATHS:
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_rag_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                """
            )
            conn.commit()
            _INITIALIZED_PATHS.add(key)
        finally:
            conn.close()


def extract_json_object(raw_text: str):
    """Extract first valid JSON object (or wrap JSON array into {'selected': ...}) without regex."""
    if not raw_text:
        return None
    text = str(raw_text).strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"selected": parsed}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if start != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    if arr_start != -1 and arr_end > arr_start:
        arr_cand = text[arr_start:arr_end + 1]
        try:
            parsed = json.loads(arr_cand)
            if isinstance(parsed, list):
                return {"selected": parsed}
        except Exception:
            pass
    return None


def rag_llm_available(module: str = "chunking.text") -> bool:
    """True when configured cloud LLM endpoint has non-empty URL, model, and API key."""
    try:
        chain = get_model_chain(module)
        for cand in chain:
            if model_configured(cand):
                url_low = str(cand.url or "").casefold()
                if ".invalid" not in url_low and "://mock" not in url_low:
                    return True
    except Exception:
        return False
    return False


def get_cached_json(cache_key: str, cache_db_path=None):
    db_path = _resolve_cache_path(cache_db_path)
    if not db_path.is_file():
        return None
    try:
        _ensure_table(db_path)
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA busy_timeout=5000;")
            row = conn.execute(
                "SELECT payload FROM llm_rag_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row and isinstance(row[0], str):
                data = json.loads(row[0])
                if isinstance(data, dict):
                    return data
        finally:
            conn.close()
    except Exception:
        return None
    return None


def put_cached_json(cache_key: str, payload: dict, cache_db_path=None):
    if not isinstance(payload, dict):
        return
    db_path = _resolve_cache_path(cache_db_path)
    try:
        _ensure_table(db_path)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute(
                "INSERT OR REPLACE INTO llm_rag_cache (cache_key, payload) VALUES (?, ?);",
                (cache_key, encoded),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def build_cache_key(namespace: str, model_id: str, seed_text: str) -> str:
    material = f"{namespace}:{model_id}:{seed_text}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def call_rag_llm_json(
    namespace: str,
    seed_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 800,
    temperature: float = 0.0,
    cache_db_path=None,
    chat_client=None,
    module: str = "chunking.text",
    validator=None,
):
    """Call configured cloud LLM expecting JSON output (thinking disabled) with WAL cache."""
    model_id = "injected"
    is_mock_client = chat_client is not None
    if chat_client is None:
        chain = get_model_chain(module)
        if not chain or not model_configured(chain[0]):
            return None
        model_id = str(chain[0].model or "default")
    else:
        cands = getattr(chat_client, "candidates", ())
        if cands:
            model_id = str(getattr(cands[0], "model", "") or "injected")

    cache_key = build_cache_key(namespace, model_id, seed_key)
    cached = get_cached_json(cache_key, cache_db_path=cache_db_path)
    if cached is not None:
        if validator is None or validator(cached):
            return cached

    client = chat_client
    if client is None:
        try:
            client = FallbackChatClient(module)
        except Exception:
            return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    effective_max = max(int(max_tokens or 0), 800)
    if is_mock_client:
        attempts = ({"temperature": temperature, "max_tokens": max_tokens},)
    else:
        attempts = (
            {
                "temperature": temperature,
                "max_tokens": effective_max,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
            {"temperature": temperature, "max_tokens": effective_max},
        )

    for extra_kwargs in attempts:
        try:
            resp = client.chat.completions.create(messages=messages, **extra_kwargs)
            raw_content = ""
            if resp and getattr(resp, "choices", None):
                raw_content = str(resp.choices[0].message.content or "")
            parsed = extract_json_object(raw_content)
            if isinstance(parsed, dict) and (validator is None or validator(parsed)):
                put_cached_json(cache_key, parsed, cache_db_path=cache_db_path)
                return parsed
        except Exception:
            continue
    return None
