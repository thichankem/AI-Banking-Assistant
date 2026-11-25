# -*- coding: utf-8 -*-
"""LLM Contextual Retrieval Enricher (DeepSeek-V4-Flash + SQLite WAL Cache).

Generates clean, UI-noise-free `situated_context` summaries for chunks that lack
document-level context (especially digital app PDF guides and split tables),
improving both Dense Embedding (multilingual-e5-large) and Cross-Encoder ranking.

COMPLIANCE WITH GEMINI.md:
- 0% REGEX LAW: Pure string methods only.
- ZERO TERNARY LAW: Explicit if/else statements only.
- NO __INIT__.PY: Direct module imports.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from agent_banking.core.llm_rag_cache import call_rag_llm_json, rag_llm_available

_CTX_CACHE_NAMESPACE = "contextual_retrieval_v2"
_UI_EXACT_LINES = (
    "09:41",
    "5g 100%",
    "a-bank digital",
    "a bank digital",
    "na",
    "xin chào,",
    "xin chao,",
    "nguyễn văn an",
    "nguyen van an",
    "xin chào, nguyễn văn an",
    "xin chao, nguyen van an",
    "thực hiện thao tác theo chỉ dẫn trên màn hình ứng dụng a-bank plus.",
)
_UI_INLINE_NOISE = (
    "09:41 A-Bank Digital 5G 100%",
    "Xin chào, NGUYỄN VĂN AN",
    "Xin chao, NGUYEN VAN AN",
    "NGUYỄN VĂN AN",
    "1029 3847 5689 • A-Super Priority",
    "1029 3847 5689",
    "125,450,000 VND",
    "5G 100%",
)

_ENRICH_SYSTEM_PROMPT = (
    "Ban la chuyen gia lap chi muc tri thuc ngan hang (Contextual Retrieval). "
    "Nhiem vu: Cho biet ten tai lieu va danh sach cac doan (chunks) trong cung 1 tai lieu, "
    "hay viet cho MOI chunk 1 cau ngan gon (25-45 tu tieng Viet ro rang, tu nhien) trong truong 'contexts' "
    "giai thich chunk do nam o phan nao, phuc vu tinh nang/nghiep vu/san pham gi cua A-Bank va tra loi duoc kieu cau hoi nao. "
    "Bo qua hoan toan chu giao dien gia lap (09:41, 5G 100%, Xin chao Nguyen Van An, so the/tai khoan mau 1029 3847 5689). "
    "Tra ve DUNG JSON object dang: {\"contexts\": {\"chunk_id_1\": \"...\", \"chunk_id_2\": \"...\"}}."
)


def strip_ui_noise_for_context(text: str) -> str:
    """Remove repetitive phone status-bar and sample mockup chrome without touching instructions."""
    if not text:
        return ""
    lines = []
    for raw in str(text).splitlines():
        clean = " ".join(raw.split()).strip()
        if not clean:
            continue
        for noise in _UI_INLINE_NOISE:
            if noise in clean:
                clean = " ".join(clean.replace(noise, " ").split()).strip()
        if not clean:
            continue
        low = clean.casefold()
        if low in _UI_EXACT_LINES:
            continue
        lines.append(clean)
    return " ".join(lines)


def _needs_contextual_enrichment(chunk: dict, target_domains: set) -> bool:
    if not isinstance(chunk, dict):
        return False
    bm = chunk.get("business_metadata")
    if not isinstance(bm, dict):
        return False
    dom = str(bm.get("domain") or "").strip()
    if target_domains and dom not in target_domains:
        return False
    ret = bm.get("retrieval")
    if not isinstance(ret, dict):
        return True
    raw_sit = str(ret.get("situated_context") or "").strip()
    if len(raw_sit) < 25:
        return True
    low_sit = raw_sit.casefold()
    if "09:41" in low_sit or "5g 100%" in low_sit or "nguyễn văn an" in low_sit or "nguyen van an" in low_sit:
        return True
    clean_sit = strip_ui_noise_for_context(raw_sit)
    if len(clean_sit) < 30:
        return True
    return False


def _enrich_doc_batch(source: str, batch_chunks: list, doc_outline: str, cache_db_path=None, chat_client=None) -> dict:
    payload_items = []
    for ch in batch_chunks:
        cid = str(ch.get("chunk_id") or "")
        sec = str(ch.get("section_title") or "")
        bc = str(ch.get("breadcrumb") or "")
        body = strip_ui_noise_for_context(str(ch.get("content") or ""))
        if not body:
            body = str(ch.get("content") or "")
        imgs = ch.get("images") or []
        img_txt = []
        if isinstance(imgs, list):
            for im in imgs[:2]:
                if isinstance(im, dict):
                    cap = str(im.get("ocr_text") or im.get("description") or im.get("caption") or "").strip()
                    if cap:
                        img_txt.append(cap[:140])
        payload_items.append({
            "chunk_id": cid,
            "section": sec[:100],
            "breadcrumb": bc[:140],
            "text": body[:650],
            "visual": "; ".join(img_txt)[:200],
        })

    seed_obj = {
        "source": source,
        "outline": doc_outline[:500],
        "items": payload_items,
    }
    seed_str = json.dumps(seed_obj, ensure_ascii=False, sort_keys=True)
    user_prompt = (
        f"Tai lieu: {source}\n"
        f"Tom tat cac muc chinh trong tai lieu: {doc_outline}\n"
        f"Danh sach chunks can bo sung ngu canh:\n{json.dumps(payload_items, ensure_ascii=False)}"
    )
    res = call_rag_llm_json(
        namespace=_CTX_CACHE_NAMESPACE,
        seed_key=seed_str,
        system_prompt=_ENRICH_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=2200,
        temperature=0.0,
        cache_db_path=cache_db_path,
        chat_client=chat_client,
        validator=lambda d: isinstance(d.get("contexts"), dict) and len(d["contexts"]) > 0,
    )
    if not isinstance(res, dict):
        return {}
    ctx_map = res.get("contexts")
    if isinstance(ctx_map, dict):
        clean_map = {}
        for k, v in ctx_map.items():
            val = " ".join(str(v or "").split()).strip()
            if val:
                clean_map[str(k)] = val
        return clean_map
    return {}


def enrich_chunks_contextual(
    chunks: list,
    target_domains=None,
    batch_size: int = 8,
    workers: int = 6,
    cache_db_path=None,
    chat_client=None,
    use_llm: bool = True,
) -> list:
    """Populate `business_metadata.retrieval.situated_context` using cached cloud LLM calls."""
    if not chunks:
        return chunks
    if target_domains is None:
        domains_set = {"ngan_hang_so", "bieu_phi_dich_vu", "cham_soc_khach_hang"}
    else:
        domains_set = set(target_domains)

    enabled = bool(use_llm and (chat_client is not None or rag_llm_available("chunking.text")))
    if not enabled:
        return chunks

    by_source = {}
    for idx, ch in enumerate(chunks):
        src = str(ch.get("source") or "")
        by_source.setdefault(src, []).append((idx, ch))

    tasks = []
    for src, src_items in by_source.items():
        pending = [(idx, ch) for idx, ch in src_items if _needs_contextual_enrichment(ch, domains_set)]
        if not pending:
            continue
        sections = []
        for _, ch in src_items:
            st = str(ch.get("section_title") or "").strip()
            if st and st not in sections:
                sections.append(st)
        outline = " | ".join(sections[:20])
        for start in range(0, len(pending), batch_size):
            batch_pairs = pending[start:start + batch_size]
            tasks.append((src, [p[1] for p in batch_pairs], outline))

    if not tasks:
        return chunks

    enriched_by_id = {}
    if workers <= 1 or len(tasks) == 1:
        for src, batch, outline in tasks:
            enriched_by_id.update(_enrich_doc_batch(src, batch, outline, cache_db_path=cache_db_path, chat_client=chat_client))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(_enrich_doc_batch, src, batch, outline, cache_db_path, chat_client)
                for src, batch, outline in tasks
            ]
            for fut in as_completed(futs):
                try:
                    enriched_by_id.update(fut.result())
                except Exception:
                    pass

    if not enriched_by_id:
        return chunks

    updated = []
    for ch in chunks:
        cid = str(ch.get("chunk_id") or "")
        sit_ctx = enriched_by_id.get(cid)
        if sit_ctx:
            ch_copy = dict(ch)
            bm = dict(ch_copy.get("business_metadata") or {})
            ret = dict(bm.get("retrieval") or {})
            ret["situated_context"] = sit_ctx
            bm["retrieval"] = ret
            ch_copy["business_metadata"] = bm
            updated.append(ch_copy)
        else:
            updated.append(ch)
    return updated
