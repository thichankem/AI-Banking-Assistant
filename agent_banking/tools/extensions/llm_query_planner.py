# -*- coding: utf-8 -*-
"""Model-First LLM Query Planner (DeepSeek-V4-Flash with SQLite WAL Cache).

Replaces rigid hardcoded synonym/split hacks with semantic query rewriting,
sub-query decomposition, dual-track domain intent routing, and adaptive dense weight.

COMPLIANCE WITH GEMINI.md:
- 0% REGEX LAW: Pure string methods only.
- ZERO TERNARY LAW: Explicit if/else statements only.
- NO __INIT__.PY: Direct module imports.
"""

from agent_banking.core.llm_rag_cache import call_rag_llm_json, rag_llm_available

_PLANNER_CACHE_NAMESPACE = "rag_query_planner_v2"

_PLANNER_SYSTEM_PROMPT = (
    "Ban la bo lap ke hoach truy van RAG cho tro ly ngan hang A-Bank Viet Nam. "
    "Nhiem vu: phan tich cau hoi khach hang/nhan vien va tra ve DUNG 1 JSON object hop le (khong giai thich them). "
    "Kho tri thuc A-Bank gom: "
    "(1) ngan_hang_so: Huong dan su dung ung dung A-Bank Digital / A-Bank Plus (HDSD 01..39: cai dat, bao mat the, Virtual Card CVV/PIN, thanh toan hoa don, chuyen tien 24/7, A-Pay danh ba, A-Smart Earning, tiet kiem online, vay cam co so online, hoan tra thau chi, doi qua A-Points, lien ket chung khoan ASC/ABS, doi ten dang nhap, khoa app khan cap); "
    "(2) nghiep_vu_ngan_hang / the / phap_ly_tuan_thu / bao_hiem_tin_dung / cham_soc_khach_hang (NV, THE, PL, BH, TD, VH, BC); "
    "(3) bieu_phi_dich_vu (BP 01..10: bieu phi tai khoan VND/ngoai te, CITAD/ngoai he thong, tien mat, tiet kiem, sec, ngan quy/ket an toan/xe chuyen dung, sao ke/xac nhan so du, TTQT/SWIFT, tai san bao dam/giai chap, thong bao mat/chuyen nhuong GTCG, thu xep tin dung); "
    "(4) mang_luoi_hoat_dong (ML 01: chi nhanh, PGD, ATM, CDM, SDT, dia chi). "
    "Output JSON schema bat buoc:\n"
    "{\n"
    '  "rewritten_query": "Cau hoi viet lai day du ngu canh ngan hang tieng Viet tu nhien, giai thich tu viet tat/ngam y",\n'
    '  "sub_queries": ["toi da 2 truy van con doc lap neu cau hoi la so sanh A vs B hoac gom 2 y tach biet; neu don y de []"],\n'
    '  "needs_app_guide": true/false,\n'
    '  "needs_policy_doc": true/false,\n'
    '  "needs_fee_schedule": true/false,\n'
    '  "needs_network_lookup": true/false,\n'
    '  "dense_weight": 0.50,\n'
    '  "semantic_terms": ["3 den 6 thuat ngu/tu khoa ngan hang dong nghia trong tai lieu chinh thuc"]\n'
    "}\n"
    "Quy tac: dense_weight tu 0.35 (cau chua nhieu ma phi/so dieu khoan/so van ban) den 0.68 (cau hoi mo ta tinh huong/nghi van). "
    "Bat needs_app_guide=true bat cu khi nao cau hoi nhac app, A-Bank Digital, tinh nang dien thoai, the ao, PIN dien tu, thanh toan hoa don, doi ten dang nhap, khoa ung dung, hoac loi giao dich online."
)


class LLMQueryPlanner:
    """Semantic query analyzer powered by configured LLM with persistent WAL cache."""

    def __init__(self, cache_db_path=None, chat_client=None, use_llm=True):
        self.cache_db_path = cache_db_path
        self.chat_client = chat_client
        self.use_llm = bool(use_llm)

    def _is_enabled(self) -> bool:
        if not self.use_llm:
            return False
        if self.chat_client is not None:
            return True
        return rag_llm_available("chunking.text")

    def plan(self, query: str) -> dict:
        clean = " ".join(str(query or "").split())
        if not clean:
            return {
                "rewritten_query": "",
                "sub_queries": [],
                "needs_app_guide": False,
                "needs_policy_doc": True,
                "needs_fee_schedule": False,
                "needs_network_lookup": False,
                "dense_weight": 0.5,
                "semantic_terms": [],
                "method": "empty",
            }
        if not self._is_enabled():
            return {
                "rewritten_query": clean,
                "sub_queries": [],
                "needs_app_guide": False,
                "needs_policy_doc": True,
                "needs_fee_schedule": False,
                "needs_network_lookup": False,
                "dense_weight": 0.5,
                "semantic_terms": [],
                "method": "offline_disabled",
            }

        raw = call_rag_llm_json(
            namespace=_PLANNER_CACHE_NAMESPACE,
            seed_key=clean,
            system_prompt=_PLANNER_SYSTEM_PROMPT,
            user_prompt=f"Cau hoi: {clean}",
            max_tokens=800,
            temperature=0.0,
            cache_db_path=self.cache_db_path,
            chat_client=self.chat_client,
            validator=lambda d: bool(str(d.get("rewritten_query") or "").strip()),
        )
        if not isinstance(raw, dict):
            return {
                "rewritten_query": clean,
                "sub_queries": [],
                "needs_app_guide": False,
                "needs_policy_doc": True,
                "needs_fee_schedule": False,
                "needs_network_lookup": False,
                "dense_weight": 0.5,
                "semantic_terms": [],
                "method": "llm_fallback",
            }

        rewritten = str(raw.get("rewritten_query") or clean).strip()
        if not rewritten:
            rewritten = clean

        sub_queries = []
        raw_subs = raw.get("sub_queries")
        if isinstance(raw_subs, list):
            for item in raw_subs[:2]:
                s = " ".join(str(item or "").split()).strip()
                if s and s.casefold() != clean.casefold() and s not in sub_queries:
                    sub_queries.append(s)

        semantic_terms = []
        raw_terms = raw.get("semantic_terms")
        if isinstance(raw_terms, list):
            for term in raw_terms[:6]:
                t = " ".join(str(term or "").split()).strip()
                if t and t not in semantic_terms:
                    semantic_terms.append(t)

        try:
            dw = float(raw.get("dense_weight", 0.5))
        except (TypeError, ValueError):
            dw = 0.5
        if dw < 0.35:
            dw = 0.35
        if dw > 0.70:
            dw = 0.70

        return {
            "rewritten_query": rewritten,
            "sub_queries": sub_queries,
            "needs_app_guide": bool(raw.get("needs_app_guide", False)),
            "needs_policy_doc": bool(raw.get("needs_policy_doc", True)),
            "needs_fee_schedule": bool(raw.get("needs_fee_schedule", False)),
            "needs_network_lookup": bool(raw.get("needs_network_lookup", False)),
            "dense_weight": round(dw, 2),
            "semantic_terms": semantic_terms,
            "method": "llm_model",
        }
