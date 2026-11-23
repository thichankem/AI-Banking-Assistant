"""Search public Vietnamese legal documents with a per-provider request budget."""

from contextlib import closing
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import unescape
import json
import os
import re
import sqlite3
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from agent_banking.core.environment import ROOT_DIR
from agent_banking.core.search_text import search_key
from agent_banking.guardrails.indirect_injection_guardrail import get_indirect_injection_guardrail

TOOL_NAME = "search_legal_web"
PROVIDERS = ("brave", "tavily")
DOMAIN = "thuvienphapluat.vn"
_HOSTS = {DOMAIN, f"www.{DOMAIN}", f"m.{DOMAIN}"}
_QUOTA_DB = ROOT_DIR / "agent_banking" / "vector_db" / "legal_web_search.sqlite3"
_MONTHLY_LIMIT = 1000
_VIETNAM_TIME = timezone(timedelta(hours=7))
_RESULT_COUNT = 10
_LEGAL_TERMS = (
    "luat", "bo luat", "nghi dinh", "thong tu", "nghi quyet", "phap lenh", "chi thi",
    "quyet dinh", "van ban hop nhat", "van ban phap ly", "van ban phap quy", "phap luat",
    "law", "decree", "circular", "legislation",
)
_DOCUMENT_CODE = re.compile(
    r"\b\d{1,6}\s*/\s*(?:\d{4}\s*/\s*)?[A-ZĐ][A-ZĐ0-9]{0,10}(?:\s*-\s*[A-ZĐ0-9]{1,10})*\b", re.I,
)
_DOCUMENT_PATH = re.compile(r"/van-ban/(?:[\w-]+/)?[\w-]+-(\d+)\.aspx", re.I)
_PRIVATE_NUMBER = re.compile(r"\d(?:[ .()-]*\d){8,}")
_PRIVATE_CODE = re.compile(r"\b(?:otp|pin|cvv|ma xac thuc) \d{3,}\b")
_ARTICLE = re.compile(r"\bdieu \d{1,3}[a-z]?\b")
_ARTICLE_HEADING = re.compile(r"(?im)^\s{0,8}[#* ]*Điều\s+(\d{1,3}[a-z]?)\s*\.")
_ARTICLE_EXCERPT_LIMIT = 6000
MAX_SEARCHES_PER_TURN = 2


def provider_key(provider: str) -> str:
    if provider == "tavily":
        return os.getenv("TAVILY_API_KEY", "").strip()
    return os.getenv("BRAVE_SEARCH_API_KEY", "").strip() or os.getenv("BRAVE_API_KEY", "").strip()


def _query_is_public_legal(query: str) -> bool:
    if not 3 <= len(query) <= 300:
        return False
    # A query cannot supply a URL, an engine operator, an email or a long account/ID number.
    if any(not (ch.isalnum() or ch in " -/.,()") for ch in query):
        return False
    if _PRIVATE_NUMBER.search(query) or "www." in query.casefold() or "//" in query:
        return False
    if {"AND", "OR", "NOT"}.intersection(query.split()):
        return False
    key = " " + search_key(query) + " "
    if _PRIVATE_CODE.search(key):
        return False
    return bool(_DOCUMENT_CODE.search(query)) or any(f" {term} " in key for term in _LEGAL_TERMS)


def _reserve_request(provider: str) -> bool:
    try:
        limit = int(os.getenv(f"LEGAL_WEB_SEARCH_{provider.upper()}_MONTHLY_LIMIT", str(_MONTHLY_LIMIT)))
    except ValueError:
        limit = _MONTHLY_LIMIT
    limit = min(limit, _MONTHLY_LIMIT)
    if limit <= 0:
        return False
    month = datetime.now(_VIETNAM_TIME).strftime("%Y-%m")
    _QUOTA_DB.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(_QUOTA_DB, timeout=3)) as connection, connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS provider_usage (
            provider TEXT NOT NULL, month TEXT NOT NULL, requests INTEGER NOT NULL,
            PRIMARY KEY (provider, month)
        )""")
        cursor = connection.execute("""INSERT INTO provider_usage (provider, month, requests)
            VALUES (?, ?, 1)
            ON CONFLICT (provider, month) DO UPDATE SET requests = requests + 1 WHERE requests < ?
        """, (provider, month, limit))
        return cursor.rowcount == 1


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _provider_results(provider: str, query: str, key: str) -> list:
    include_articles = provider == "tavily" and bool(_ARTICLE.search(search_key(query)))
    if provider == "tavily":
        request = Request("https://api.tavily.com/search", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        }, data=json.dumps({
            "query": query, "include_domains": [DOMAIN], "search_depth": "basic",
            "auto_parameters": False, "max_results": 4 if include_articles else _RESULT_COUNT, "country": "vietnam",
            "language": "vi", "include_answer": False, "include_raw_content": "markdown" if include_articles else False,
        }).encode("utf-8"))
    else:
        params = urlencode({
            "q": f"{query} site:{DOMAIN}/van-ban/", "country": "ALL", "search_lang": "vi",
            "count": _RESULT_COUNT, "result_filter": "web", "extra_snippets": "true",
            "text_decorations": "false", "operators_fallback": "false",
        })
        request = Request(f"https://api.search.brave.com/res/v1/web/search?{params}", headers={
            "X-Subscription-Token": key, "Accept": "application/json",
        })
    with build_opener(_NoRedirects()).open(request, timeout=6) as response:
        payload = json.loads(response.read(6_000_000 if include_articles else 1_000_000))
    if not isinstance(payload, dict):
        raise ValueError("Invalid search response")
    if provider == "brave":
        payload = payload.get("web") or {}
        if not isinstance(payload, dict):
            raise ValueError("Invalid web search response")
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("Invalid search response")
    return rows


def legal_document_url(raw_url: str) -> str:
    """Return a canonical, allowlisted TVPL document URL for indexing and linking."""
    if not isinstance(raw_url, str):
        return ""
    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return ""
    path = unquote(parsed.path)
    if parsed.scheme != "https" or parsed.netloc.lower() not in _HOSTS or not _DOCUMENT_PATH.fullmatch(path):
        return ""
    return f"https://{DOMAIN}{quote(path, safe='/-_.')}"


def _clean_text(value, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unescape(value).split())[:limit].strip()


def _article_passages(value, articles):
    if not isinstance(value, str) or not articles:
        return []
    clean = unescape(value).replace("**", "")
    headings = list(_ARTICLE_HEADING.finditer(clean))
    passages = {}
    for index, heading in enumerate(headings):
        article = "dieu " + heading[1].casefold()
        if article not in articles:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(clean)
        fragment = clean[heading.start():end].strip().lstrip("# ")
        # Ignore the table of contents and require body text after the heading.
        if len(fragment.splitlines()) < 2 or len(fragment) < 100:
            continue
        bounded = "\n".join(line.strip() for line in fragment.splitlines() if line.strip())[:_ARTICLE_EXCERPT_LIMIT]
        if len(bounded) > len(passages.get(article, "")):
            passages[article] = bounded
    return [(article, passages[article]) for article in sorted(passages)]


def _documents(provider: str, rows: list, query: str) -> list[dict]:
    expected_codes = {search_key(match.group()) for match in _DOCUMENT_CODE.finditer(query)}
    articles = set(_ARTICLE.findall(search_key(query)))
    documents = []
    seen = set()
    scan = get_indirect_injection_guardrail().scan_text
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = legal_document_url(row.get("url"))
        title = _clean_text(row.get("title"), 320)
        if not url or not title:
            continue
        document_id = url.rsplit("-", 1)[-1].split(".", 1)[0]
        if document_id in seen:
            continue
        if provider == "tavily":
            snippets = [row.get("content")]
        else:
            snippets = [row.get("description")]
            if isinstance(row.get("extra_snippets"), list):
                snippets.extend(row["extra_snippets"][:5])
        excerpts = dict.fromkeys(filter(None, (_clean_text(snippet, 1600) for snippet in snippets)))
        if expected_codes:
            document_text = " " + search_key(title + " " + unquote(urlsplit(url).path)) + " "
            if not any(f" {code} " in document_text for code in expected_codes):
                continue
        passages = _article_passages(row.get("raw_content"), articles)
        if not passages:
            passages = [("", "\n\n".join(excerpts)[:2500].strip())]
        for article, content in passages:
            if len(content) < 40 or not scan(title + "\n" + content)["is_safe"]:
                continue
            seen.add(document_id)
            documents.append({
                "chunk_id": "tvpl-" + sha256((document_id + "\n" + content).encode("utf-8")).hexdigest()[:24],
                "source": url, "content": content, "chunk_type": "text",
                "business_metadata": {
                    "document_name": title, "source_url": url, "category": "Đoạn trích tìm kiếm web",
                    **({"legal_article": article} if article else {}),
                },
            })
    return documents


def search_provider(provider: str, query: str) -> dict:
    """Run one provider only, using the same validation and quota as the agent."""
    if provider not in PROVIDERS:
        raise ValueError("Unknown legal search provider")
    query = " ".join(query.split())
    outcome = {"provider": provider, "documents": []}
    if not _query_is_public_legal(query):
        return {**outcome, "status": "invalid_legal_query"}
    key = provider_key(provider)
    if not key:
        return {**outcome, "status": "not_configured"}
    try:
        if not _reserve_request(provider):
            return {**outcome, "status": "monthly_limit"}
    except (OSError, sqlite3.Error):
        return {**outcome, "status": "quota_unavailable"}
    try:
        rows = _provider_results(provider, query, key)
    except HTTPError as error:
        error.close()
        return {**outcome, "status": f"http_{error.code}"}
    except (OSError, ValueError):
        return {**outcome, "status": "provider_error"}
    outcome["documents"] = _documents(provider, rows, query)
    if outcome["documents"]:
        return {**outcome, "status": "ok"}
    return {**outcome, "status": "no_documents"}


def search_legal_web(query: str) -> dict:
    """Try the primary provider, and use the other only if usable evidence is missing."""
    providers = PROVIDERS
    if os.getenv("LEGAL_WEB_SEARCH_PRIMARY", "").strip().lower() == "tavily":
        providers = ("tavily", "brave")
    articles = set(_ARTICLE.findall(search_key(query)))
    attempts = []
    best = None
    best_coverage = -1
    for provider in providers:
        outcome = search_provider(provider, query)
        status = outcome["status"]
        if status == "ok":
            ranked = []
            for doc in outcome["documents"]:
                article = doc["business_metadata"].get("legal_article")
                headings = {"dieu " + match[1].casefold() for match in _ARTICLE_HEADING.finditer(doc["content"])}
                hits = articles.intersection({article} if article else headings)
                ranked.append((doc, hits, bool(article)))
            top = sorted(ranked, key=lambda item: (len(item[1]), item[2]), reverse=True)[:4]
            coverage = len(set().union(*(hits for _, hits, _ in top)))
            if coverage > best_coverage:
                best = {**outcome, "documents": [doc for doc, _, _ in top]}
                best_coverage = coverage
            if coverage < len(articles):
                status = "incomplete_article_evidence"
        attempts.append({"provider": provider, "status": status})
        if status in {"ok", "invalid_legal_query"}:
            break
    if best is not None:
        if best_coverage < len(articles):
            best["status"] = "incomplete_article_evidence"
        return {**best, "attempts": attempts}
    return {"status": attempts[-1]["status"], "provider": "", "documents": [], "attempts": attempts}
