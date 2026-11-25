"""Search public Vietnamese legal documents with a per-provider request budget."""

from contextlib import closing
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import unescape
import json
import os
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
_ARTICLE_EXCERPT_LIMIT = 6000
MAX_SEARCHES_PER_TURN = 2


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _is_code_letter(ch: str) -> bool:
    upper = ch.upper()
    return ("A" <= upper <= "Z") or ch in ("Đ", "đ")


def _is_code_alnum(ch: str) -> bool:
    return _is_code_letter(ch) or ("0" <= ch <= "9")


def _extract_document_codes(text: str) -> list[str]:
    if not isinstance(text, str) or "/" not in text:
        return []
    codes = []
    n = len(text)
    idx = 0
    while idx < n:
        if ("0" <= text[idx] <= "9") and (idx == 0 or not _is_word_char(text[idx - 1])):
            d_end = idx
            while d_end < n and ("0" <= text[d_end] <= "9"):
                d_end += 1
            if 1 <= (d_end - idx) <= 6 and (d_end == n or not _is_word_char(text[d_end])):
                pos = d_end
                while pos < n and text[pos].isspace():
                    pos += 1
                if pos < n and text[pos] == "/":
                    pos += 1
                    while pos < n and text[pos].isspace():
                        pos += 1
                    year_end = pos
                    while year_end < n and ("0" <= text[year_end] <= "9"):
                        year_end += 1
                    if (year_end - pos) == 4 and (year_end == n or not _is_word_char(text[year_end])):
                        after_year = year_end
                        while after_year < n and text[after_year].isspace():
                            after_year += 1
                        if after_year < n and text[after_year] == "/":
                            after_slash = after_year + 1
                            while after_slash < n and text[after_slash].isspace():
                                after_slash += 1
                            pos = after_slash
                    tok_end = pos
                    while tok_end < n and _is_word_char(text[tok_end]):
                        tok_end += 1
                    first_seg = text[pos:tok_end]
                    if 1 <= len(first_seg) <= 11 and _is_code_letter(first_seg[0]) and all(_is_code_alnum(c) for c in first_seg):
                        match_end = tok_end
                        scan_pos = tok_end
                        while scan_pos < n:
                            ws_pos = scan_pos
                            while ws_pos < n and text[ws_pos].isspace():
                                ws_pos += 1
                            if ws_pos >= n or text[ws_pos] != "-":
                                break
                            sub_start = ws_pos + 1
                            while sub_start < n and text[sub_start].isspace():
                                sub_start += 1
                            sub_end = sub_start
                            while sub_end < n and _is_word_char(text[sub_end]):
                                sub_end += 1
                            sub_seg = text[sub_start:sub_end]
                            if 1 <= len(sub_seg) <= 10 and all(_is_code_alnum(c) for c in sub_seg):
                                match_end = sub_end
                                scan_pos = sub_end
                            else:
                                break
                        codes.append(text[idx:match_end])
                        idx = match_end
                        continue
            idx = d_end
            continue
        idx += 1
    return codes


def _is_valid_document_path(path: str) -> bool:
    if not isinstance(path, str):
        return False
    lower = path.lower()
    if not lower.startswith("/van-ban/") or not lower.endswith(".aspx"):
        return False
    body = path[9:-5]
    if not body:
        return False
    parts = body.split("/")
    if len(parts) == 2:
        category, slug_and_id = parts[0], parts[1]
        if not category or not all(ch.isalnum() or ch in "_-" for ch in category):
            return False
    elif len(parts) == 1:
        slug_and_id = parts[0]
    else:
        return False
    if "-" not in slug_and_id:
        return False
    slug, doc_id = slug_and_id.rsplit("-", 1)
    if not slug or not all(ch.isalnum() or ch in "_-" for ch in slug):
        return False
    if not doc_id or not all("0" <= ch <= "9" for ch in doc_id):
        return False
    return True


def _has_private_number(text: str) -> bool:
    digit_run = 0
    for ch in text:
        if ch.isdigit():
            digit_run += 1
            if digit_run >= 9:
                return True
        elif ch in " .()-":
            continue
        else:
            digit_run = 0
    return False


def _has_private_code(folded_key: str) -> bool:
    words = folded_key.split()
    for idx, word in enumerate(words):
        if word in ("otp", "pin", "cvv") and idx + 1 < len(words):
            nxt = words[idx + 1]
            if len(nxt) >= 3 and nxt.isdigit():
                return True
        if word == "ma" and idx + 3 < len(words):
            if words[idx + 1] == "xac" and words[idx + 2] == "thuc":
                nxt = words[idx + 3]
                if len(nxt) >= 3 and nxt.isdigit():
                    return True
    return False


def _extract_articles(folded_key: str) -> list[str]:
    words = folded_key.split()
    found = []
    for idx, word in enumerate(words):
        if word == "dieu" and idx + 1 < len(words):
            tok = words[idx + 1]
            if 1 <= len(tok) <= 3 and tok.isdigit():
                found.append("dieu " + tok)
            elif 2 <= len(tok) <= 4 and tok[:-1].isdigit() and ("a" <= tok[-1] <= "z"):
                found.append("dieu " + tok)
    return found


def _find_article_headings(text: str) -> list[tuple[int, str]]:
    if not isinstance(text, str) or not text:
        return []
    headings = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(raw_line)
        line = raw_line.rstrip("\r\n")
        idx = 0
        ws_count = 0
        while idx < len(line) and ws_count < 8 and line[idx].isspace():
            idx += 1
            ws_count += 1
        while idx < len(line) and line[idx] in "#* ":
            idx += 1
        if line[idx:idx + 4].casefold() != "điều":
            continue
        idx += 4
        if idx >= len(line) or not line[idx].isspace():
            continue
        while idx < len(line) and line[idx].isspace():
            idx += 1
        num_start = idx
        while idx < len(line) and ("0" <= line[idx] <= "9"):
            idx += 1
        digit_len = idx - num_start
        if digit_len < 1 or digit_len > 3:
            continue
        if idx < len(line) and ("a" <= line[idx].lower() <= "z"):
            idx += 1
        article_num = line[num_start:idx]
        while idx < len(line) and line[idx].isspace():
            idx += 1
        if idx < len(line) and line[idx] == ".":
            headings.append((line_start, article_num))
    return headings


def provider_key(provider: str) -> str:
    if provider == "tavily":
        return os.getenv("TAVILY_API_KEY", "").strip()
    return os.getenv("BRAVE_SEARCH_API_KEY", "").strip() or os.getenv("BRAVE_API_KEY", "").strip()


_TVPL_BOILERPLATE_MARKERS = (
    "bạn hãy nhập mật khẩu đang sử dụng",
    "nhập mật khẩu mới 2 lần",
    "bạn hãy nhập e-mail đã sử dụng",
    "để đăng ký thành viên",
)


def _strip_benign_quotes(query: str) -> str:
    cleaned = query
    for ch in ('"', "'", "“", "”", "‘", "’"):
        if ch in cleaned:
            cleaned = cleaned.replace(ch, " ")
    return " ".join(cleaned.split())


def _is_tvpl_boilerplate(text: str) -> bool:
    low = text.casefold()
    return any(marker in low for marker in _TVPL_BOILERPLATE_MARKERS)


def _query_is_public_legal(query: str) -> bool:
    query = _strip_benign_quotes(query)
    if not 3 <= len(query) <= 300:
        return False
    # A query cannot supply a URL, an engine operator, an email or a long account/ID number.
    if any(not (ch.isalnum() or ch in " -/.,()") for ch in query):
        return False
    if _has_private_number(query) or "www." in query.casefold() or "//" in query:
        return False
    if {"AND", "OR", "NOT"}.intersection(query.split()):
        return False
    folded = search_key(query)
    key = " " + folded + " "
    if _has_private_code(folded):
        return False
    return bool(_extract_document_codes(query)) or any(f" {term} " in key for term in _LEGAL_TERMS)


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
    include_articles = provider == "tavily" and bool(_extract_articles(search_key(query)))
    if provider == "tavily":
        if include_articles:
            max_results = 4
            raw_content_mode = "markdown"
        else:
            max_results = _RESULT_COUNT
            raw_content_mode = False
        request = Request("https://api.tavily.com/search", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        }, data=json.dumps({
            "query": query, "include_domains": [DOMAIN], "search_depth": "basic",
            "auto_parameters": False, "max_results": max_results, "country": "vietnam",
            "language": "vi", "include_answer": False, "include_raw_content": raw_content_mode,
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
    if include_articles:
        read_limit = 6_000_000
    else:
        read_limit = 1_000_000
    with build_opener(_NoRedirects()).open(request, timeout=6) as response:
        payload = json.loads(response.read(read_limit))
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
    if parsed.scheme != "https" or parsed.netloc.lower() not in _HOSTS or not _is_valid_document_path(path):
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
    headings = _find_article_headings(clean)
    passages = {}
    for index, (start_pos, article_num) in enumerate(headings):
        article = "dieu " + article_num.casefold()
        if article not in articles:
            continue
        if index + 1 < len(headings):
            end = headings[index + 1][0]
        else:
            end = len(clean)
        fragment = clean[start_pos:end].strip().lstrip("# ")
        # Ignore the table of contents and require body text after the heading.
        if len(fragment.splitlines()) < 2 or len(fragment) < 100:
            continue
        bounded = "\n".join(line.strip() for line in fragment.splitlines() if line.strip())[:_ARTICLE_EXCERPT_LIMIT]
        if len(bounded) > len(passages.get(article, "")):
            passages[article] = bounded
    return [(article, passages[article]) for article in sorted(passages)]


def _documents(provider: str, rows: list, query: str) -> list[dict]:
    expected_codes = {search_key(code) for code in _extract_document_codes(query)}
    articles = set(_extract_articles(search_key(query)))
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
        excerpts = []
        seen_excerpts = set()
        for snippet in snippets:
            cleaned = _clean_text(snippet, 1600)
            if not cleaned or _is_tvpl_boilerplate(cleaned):
                continue
            if cleaned not in seen_excerpts:
                seen_excerpts.add(cleaned)
                excerpts.append(cleaned)
        if expected_codes:
            document_text = " " + search_key(title + " " + unquote(urlsplit(url).path)) + " "
            if not any(f" {code} " in document_text for code in expected_codes):
                continue
        passages = _article_passages(row.get("raw_content"), articles)
        if not passages:
            merged_excerpt = "\n\n".join(excerpts)[:3500].strip()
            passages = [("", merged_excerpt)]
        for article, content in passages:
            if len(content) < 40 or not scan(title + "\n" + content)["is_safe"]:
                continue
            seen.add(document_id)
            biz_meta = {
                "document_name": title,
                "source_url": url,
                "category": "Đoạn trích tìm kiếm web",
            }
            if article:
                biz_meta["legal_article"] = article
            documents.append({
                "chunk_id": "tvpl-" + sha256((document_id + "\n" + content).encode("utf-8")).hexdigest()[:24],
                "source": url, "content": content, "chunk_type": "text",
                "business_metadata": biz_meta,
            })
    return documents


def search_provider(provider: str, query: str) -> dict:
    """Run one provider only, using the same validation and quota as the agent."""
    if provider not in PROVIDERS:
        raise ValueError("Unknown legal search provider")
    query = _strip_benign_quotes(" ".join(query.split()))
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
    articles = set(_extract_articles(search_key(query)))
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
                headings = {"dieu " + num.casefold() for _, num in _find_article_headings(doc["content"])}
                if article:
                    candidate_articles = {article}
                else:
                    candidate_articles = headings
                hits = articles.intersection(candidate_articles)
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
