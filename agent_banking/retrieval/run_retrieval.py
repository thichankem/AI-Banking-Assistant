"""
Automated Hybrid Retrieval Runner (Auto Retrieval Node with Deduplication).
===========================================================================
Features:
1. Synchronizes from query_parser/output/parsed_queries.json to retrieval/input/.
2. Lightweight direct engine initialization: BankingEmbedder + QdrantBankingStore.
3. Preserves 100% of the 13 fields in Target Universal Contract of ParsedQuery.
4. Checks StateTracker: Reuses previously retrieved queries (SKIP).
5. Executes hybrid search combining Dense Vector + BM25 with RRF k=60 for new queries.
6. Enriches output metadata: query_id, intent, domain, expanded_query, outer chunk_id.
7. Emits 4 files: retrieval_candidates.json, latest_candidates.json, retrieval_manifest.json, .state_tracker.json.
8. Automatically cascades downstream to reranking/input/candidates.json if present.
9. Supports single query CLI execution (--query).

MANDATORY INVARIANTS:
1. NO __INIT__.PY: Direct module imports.
2. NO TYPING LIBRARY: Native types only.
3. ZERO TERNARY: Explicit if/else statements.
4. 0% REGEX LAW: No regex methods or import re.
"""

import sys
import json
import time
import shutil
from pathlib import Path

# Dam bao UTF-8 an toan tren Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.config.settings import Config
from agent_banking.embedding.embedder import BankingEmbedder
from agent_banking.vector_db.qdrant_store import QdrantBankingStore
from agent_banking.retrieval.hybrid_retriever import HybridRetriever
from agent_banking.query_parser.rag_query_parser import ParsedQuery
from agent_banking.state_tracker import StateTracker


def sync_upstream_parsed_queries(upstream_path=None, dest_path=None):
    """
    Tu dong dong bo tep parsed_queries.json tu Query Parser sang Retrieval Input.
    Kich hoat khi tep dich chua ton tai hoac tep upstream co st_mtime moi hon.
    """
    if upstream_path is not None:
        src = Path(upstream_path)
    else:
        src = PROJECT_ROOT / "agent_banking" / "query_parser" / "output" / "parsed_queries.json"

    if dest_path is not None:
        dst = Path(dest_path)
    else:
        dst = PROJECT_ROOT / "agent_banking" / "retrieval" / "input" / "parsed_queries.json"

    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    should_sync = False
    sync_reason = ""
    if not dst.exists():
        should_sync = True
        sync_reason = "tep dich chua ton tai"
    else:
        src_mtime = src.stat().st_mtime
        dst_mtime = dst.stat().st_mtime
        if src_mtime > dst_mtime:
            should_sync = True
            sync_reason = "tep nguon moi hon tep dich"

    if should_sync:
        try:
            shutil.copy2(str(src), str(dst))
            print(f"🔄 [Auto-Sync] Da dong bo '{src.name}' tu query_parser sang retrieval/input ({sync_reason}).")
            return True
        except Exception as e:
            print(f"⚠️ [Auto-Sync] Loi khi sao chep: {e}")
            return False
    return False


def init_retrieval_engine(config=None, use_memory=False, top_k=5):
    """
    Lightweight Direct Engine Initialization:
    Khoi tao truc tiep BankingEmbedder va QdrantBankingStore,
    ket noi vao HybridRetriever ma khong can khoi tao BankingRAGPipeline.
    """
    if config is not None:
        cfg = config
    else:
        cfg = Config()

    embedder = BankingEmbedder(
        dimension=cfg.EMBEDDING_DIM,
        model_name=cfg.EMBEDDING_MODEL_1024,
        embedding_url=cfg.EMBEDDING_URL,
        api_key=cfg.LLM_API_KEY,
        cache_db_path=str(cfg.STORAGE_DIR / "embedding_cache.db")
    )

    store = QdrantBankingStore(
        collection_name=cfg.QDRANT_COLLECTION,
        dimension=cfg.EMBEDDING_DIM,
        storage_path=str(cfg.STORAGE_DIR / "qdrant_db"),
        use_memory=use_memory
    )

    rrf_k_val = 60
    if hasattr(cfg, "RRF_K"):
        rrf_k_val = cfg.RRF_K

    dense_weight_val = 0.5
    if hasattr(cfg, "DENSE_WEIGHT"):
        dense_weight_val = cfg.DENSE_WEIGHT

    retriever = HybridRetriever(
        vector_store=store,
        embedding_model=embedder,
        top_k=top_k,
        rrf_k=rrf_k_val,
        dense_weight=dense_weight_val
    )
    return retriever


def resolve_retriever(pipeline=None, retriever=None, config=None, use_memory=False, top_k=5):
    """
    Dieu phoi bo truy van retriever:
    1. Uu tien retriever truyen truc tiep.
    2. Neu co pipeline, trich xuat pipeline.retriever hoac giu pipeline.
    3. Mac dinh: Khoi tao nhe qua init_retrieval_engine.
    """
    if retriever is not None:
        return retriever

    if pipeline is not None:
        if hasattr(pipeline, "retriever"):
            if pipeline.retriever is not None:
                return pipeline.retriever
        return pipeline

    return init_retrieval_engine(config=config, use_memory=use_memory, top_k=top_k)


def execute_retrieval(retriever_obj, query_obj, top_k=5, filter_dict=None):
    """
    Thuc thi truy van tren retriever_obj hoac pipeline.
    Tuong thich ca HybridRetriever lan BankingRAGPipeline.
    """
    if hasattr(retriever_obj, "retrieve"):
        return retriever_obj.retrieve(query=query_obj, top_k=top_k, filter_dict=filter_dict)
    elif hasattr(retriever_obj, "step2_hybrid_retrieval"):
        return retriever_obj.step2_hybrid_retrieval(query_obj, top_k=top_k, filter_dict=filter_dict)
    else:
        raise AttributeError(f"Retriever engine khong hop le: {type(retriever_obj)}")


def wrap_query_input(query_input):
    """
    Chuyen doi dau vao truy van sang doi tuong tuong thich Universal Query Contract.
    Neu la chuoi str, goi BankingQueryParser de sinh day du expanded_query va suggested_filters.
    Neu la ParsedQuery hoac dict, giu nguyen.
    """
    if query_input is None:
        return ""

    if isinstance(query_input, str):
        clean_text = query_input.strip()
        try:
            from agent_banking.query_parser.rag_query_parser import BankingQueryParser
            qp = BankingQueryParser()
            return qp.parse(clean_text)
        except Exception:
            return {
                "raw_query": clean_text,
                "normalized_query": clean_text,
                "expanded_query": f"query: {clean_text}",
                "keywords_str": clean_text
            }

    return query_input


def write_retrieval_manifest(
    out_path,
    cfg,
    total_queries,
    processed_queries,
    skipped_queries,
    top_k,
    total_duration,
    avg_latency,
    avg_dense=0.0,
    avg_bm25=0.0
):
    """
    Xuat ban tep retrieval_manifest.json ghi nhan chi tiet so lieu thuc thi.
    """
    manifest_file = out_path / "retrieval_manifest.json"
    manifest_data = {
        "module": "hybrid_retrieval",
        "version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collection": cfg.QDRANT_COLLECTION,
        "total_queries": total_queries,
        "processed_queries": processed_queries,
        "skipped_queries": skipped_queries,
        "top_k": top_k,
        "total_duration_seconds": round(total_duration, 3),
        "avg_query_latency_ms": round(avg_latency, 1),
        "avg_dense_latency_ms": round(avg_dense, 1),
        "avg_bm25_latency_ms": round(avg_bm25, 1),
        "files": {
            "retrieval_candidates": "output/retrieval_candidates.json",
            "latest_candidates": "output/latest_candidates.json",
            "state_tracker": "output/.state_tracker.json"
        }
    }
    try:
        with open(manifest_file, "w", encoding="utf-8") as fp:
            json.dump(manifest_data, fp, ensure_ascii=False, indent=2)
        print(f"📋 [Manifest] Da xuat manifest thong ke tai: {manifest_file.name}")
    except Exception as err:
        print(f"⚠️ [Manifest Warning] Khong the ghi manifest: {err}")


def sync_downstream_candidates(out_file):
    """
    Tu dong dong bo tep ket qua retrieval sang reranking/input/candidates.json neu thu muc ton tai.
    """
    downstream_in = PROJECT_ROOT / "agent_banking" / "reranking" / "input"
    if downstream_in.exists():
        try:
            shutil.copy2(str(out_file), str(downstream_in / "candidates.json"))
            print("🔄 [Downstream-Sync] Da cap nhat ket qua sang reranking/input/candidates.json")
        except Exception:
            pass


def print_retrieval_summary(query_text, candidates, duration_ms=None):
    """
    Hien thi bang tom tat ket qua truy van ro rang, dep mat tren terminal.
    """
    print("\n" + "=" * 80)
    print("🔍 [RETRIEVAL RESULTS] HYBRID DENSE (1024-d) + BM25 (RRF k=60)")
    print(f"   Query: \"{query_text}\"")
    if duration_ms is not None:
        print(f"   Candidates: {len(candidates)} | Duration: {duration_ms}ms")
    else:
        print(f"   Candidates: {len(candidates)}")
    print("=" * 80)

    if not candidates:
        print("   ⚠️ Khong tim thay tai lieu phu hop.")
        print("-" * 80 + "\n")
        return

    for item in candidates:
        rank = item.get("candidate_rank", 1)
        chunk_id = item.get("chunk_id", "")
        rrf_score = item.get("rrf_score", 0.0)
        dense_rank = item.get("dense_rank")
        dense_score = item.get("dense_score")
        bm25_rank = item.get("bm25_rank")
        bm25_score = item.get("bm25_score")

        chunk = item.get("chunk", {})
        title = chunk.get("section_title", "")
        breadcrumb = chunk.get("breadcrumb", "")
        source = chunk.get("source", "")
        chunk_type = chunk.get("chunk_type", "text")
        content = chunk.get("content", "")

        dense_info = "None"
        if dense_rank is not None:
            if dense_score is not None:
                dense_info = f"#{dense_rank} ({dense_score:.4f})"
            else:
                dense_info = f"#{dense_rank}"

        bm25_info = "None"
        if bm25_rank is not None:
            if bm25_score is not None:
                bm25_info = f"#{bm25_rank} ({bm25_score:.4f})"
            else:
                bm25_info = f"#{bm25_rank}"

        print(f"\n[Top {rank}] Chunk ID: {chunk_id}")
        print(f"   📊 RRF Score: {rrf_score:.6f} | Dense: {dense_info} | BM25: {bm25_info}")

        if breadcrumb:
            print(f"   📂 Breadcrumb: {breadcrumb}")
        if title:
            print(f"   🏷️ Section   : {title}")
        print(f"   📄 Source    : {source} [{chunk_type}]")

        clean_text = content.strip().replace("\r\n", " ").replace("\n", " ")
        if len(clean_text) > 200:
            snippet = clean_text[:200].strip() + "..."
        else:
            snippet = clean_text
        print(f"   📝 Content   :\n      \"{snippet}\"")
        print("-" * 80)
    print()


def run_single_query(
    query,
    top_k=5,
    filter_dict=None,
    use_memory=False,
    pipeline=None,
    retriever=None,
    output_dir=None
):
    """
    Thuc thi truy van don le tu CLI (--query) hoac goi truc tiep.
    """
    if not query or not str(query).strip():
        print("⚠️ Query rong. Vui long truyen noi dung cau hoi.")
        return []

    cfg = Config()
    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "retrieval" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    clean_query_str = str(query).strip()
    print(f"\n{'=' * 75}")
    print(f"▶ [MODULE 5: CLI SINGLE QUERY] Truy van: \"{clean_query_str}\" (Top-K={top_k})")
    print(f"{'=' * 75}")

    engine = resolve_retriever(
        pipeline=pipeline,
        retriever=retriever,
        config=cfg,
        use_memory=use_memory,
        top_k=top_k
    )

    query_obj = wrap_query_input(clean_query_str)

    raw_query_str = clean_query_str
    norm_query_str = clean_query_str
    exp_query_str = ""
    kws_str = ""
    intent_val = "hoi_dap_chung"
    sub_intent_val = "tra_cuu_thong_tin"
    domain_val = "ngan_hang_chung"

    if hasattr(query_obj, "raw_query"):
        raw_query_str = query_obj.raw_query
        norm_query_str = query_obj.normalized_query
        exp_query_str = query_obj.expanded_query
        kws_str = query_obj.keywords_str
        intent_val = query_obj.intent
        sub_intent_val = query_obj.sub_intent
        domain_val = query_obj.domain
    elif isinstance(query_obj, dict):
        raw_query_str = query_obj.get("raw_query", clean_query_str)
        norm_query_str = query_obj.get("normalized_query", clean_query_str)
        exp_query_str = query_obj.get("expanded_query", "")
        kws_str = query_obj.get("keywords_str", "")
        intent_val = query_obj.get("intent", "hoi_dap_chung")
        sub_intent_val = query_obj.get("sub_intent", "tra_cuu_thong_tin")
        domain_val = query_obj.get("domain", "ngan_hang_chung")

    t0 = time.time()
    candidates = execute_retrieval(engine, query_obj, top_k=top_k, filter_dict=filter_dict)
    dt_ms = round((time.time() - t0) * 1000, 1)

    for c in candidates:
        if isinstance(c, dict):
            if "chunk_id" not in c or not c["chunk_id"]:
                c_chunk = c.get("chunk", {})
                if isinstance(c_chunk, dict):
                    c["chunk_id"] = c_chunk.get("chunk_id", "")

    print_retrieval_summary(raw_query_str, candidates, duration_ms=dt_ms)

    single_result = {
        "query_id": "single_query_1",
        "query": raw_query_str,
        "raw_query": raw_query_str,
        "normalized_query": norm_query_str,
        "intent": intent_val,
        "sub_intent": sub_intent_val,
        "domain": domain_val,
        "keywords_str": kws_str,
        "expanded_query": exp_query_str,
        "duration_ms": dt_ms,
        "candidates_count": len(candidates),
        "candidates": candidates
    }

    # Xuat vao latest_candidates.json
    latest_file = out_path / "latest_candidates.json"
    try:
        with open(latest_file, "w", encoding="utf-8") as fp:
            json.dump(single_result, fp, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Khong the ghi latest_candidates.json: {e}")

    # Xuat hoac cap nhat vao retrieval_candidates.json
    out_file = out_path / "retrieval_candidates.json"
    existing_list = []
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as fp:
                existing_list = json.load(fp)
        except Exception:
            existing_list = []

    merged_dict = {}
    for item in existing_list:
        q_text = item.get("raw_query")
        if q_text is None:
            q_text = item.get("query", "")
        if q_text:
            merged_dict[q_text] = item

    merged_dict[raw_query_str] = single_result

    merged_list = []
    for m_idx, m_val in enumerate(merged_dict.values(), start=1):
        item_copy = dict(m_val)
        item_copy["query_id"] = m_idx
        merged_list.append(item_copy)

    try:
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(merged_list, fp, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Khong the ghi retrieval_candidates.json: {e}")

    dense_lat = 0.0
    bm25_lat = 0.0
    if candidates:
        first_meta = candidates[0].get("retrieval_meta", {})
        if "dense_latency_ms" in first_meta:
            dense_lat = first_meta["dense_latency_ms"]
        if "bm25_latency_ms" in first_meta:
            bm25_lat = first_meta["bm25_latency_ms"]

    write_retrieval_manifest(
        out_path=out_path,
        cfg=cfg,
        total_queries=len(merged_list),
        processed_queries=1,
        skipped_queries=len(merged_list) - 1,
        top_k=top_k,
        total_duration=round(dt_ms / 1000.0, 3),
        avg_latency=dt_ms,
        avg_dense=dense_lat,
        avg_bm25=bm25_lat
    )

    sync_downstream_candidates(out_file)

    return candidates


def run_batch_retrieval(
    input_file=None,
    output_dir=None,
    top_k=5,
    force=False,
    use_memory=False,
    pipeline=None,
    retriever=None
):
    """
    Thuc thi batch retrieval tu tap tin parsed_queries.json voi StateTracker deduplication.
    """
    cfg = Config()
    in_dir = PROJECT_ROOT / "agent_banking" / "retrieval" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "retrieval" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Tu dong dong bo thuong nguon (Upstream Auto-Sync)
    sync_upstream_parsed_queries()

    state_file = out_path / ".state_tracker.json"
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 2. Xac dinh tap tin input
    target_in_file = None
    if input_file is not None:
        p_in = Path(input_file)
        if p_in.exists():
            target_in_file = p_in
    else:
        default_in = in_dir / "parsed_queries.json"
        if default_in.exists():
            target_in_file = default_in
        else:
            qp_out = PROJECT_ROOT / "agent_banking" / "query_parser" / "output" / "parsed_queries.json"
            if qp_out.exists():
                target_in_file = qp_out

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Khong tim thay tap tin parsed_queries.json trong: {in_dir}")
        print("Vui long bo file 'parsed_queries.json' vao agent_banking/retrieval/input/")
        return []

    print(f"\n{'=' * 75}")
    print(f"▶ [MODULE 5: HYBRID RETRIEVAL] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'=' * 75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_queries = json.load(fp)

    if not raw_queries:
        print("⚠️ Danh sach queries rong.")
        return []

    # 3. Kiem tra cache cu tu StateTracker (chong trung lap)
    prev_results = {}
    prev_file = out_path / "retrieval_candidates.json"
    if prev_file.exists() and not force:
        try:
            with open(prev_file, "r", encoding="utf-8") as fp:
                prev_list = json.load(fp)
                for item in prev_list:
                    q = item.get("query", "")
                    if q:
                        prev_results[q] = item

            # Kiem tra toi uu SKIP ALL
            if len(prev_results) >= len(raw_queries) and len(raw_queries) > 0:
                q_hash_map = {}
                for item in raw_queries:
                    raw_q = item.get("raw_query")
                    if raw_q is None:
                        raw_q = item.get("query", "")
                    raw_q = str(raw_q).strip()

                    norm_q = item.get("normalized_query")
                    if norm_q is None or not str(norm_q).strip():
                        norm_q = raw_q
                    else:
                        norm_q = str(norm_q).strip()

                    kws = item.get("keywords_str")
                    if kws is None:
                        kws = ""
                    else:
                        kws = str(kws).strip()

                    exp_q = item.get("expanded_query")
                    if exp_q is None:
                        exp_q = ""
                    else:
                        exp_q = str(exp_q).strip()

                    q_hash_map[raw_q] = tracker.compute_text_hash(f"{raw_q}:{norm_q}:{kws}:{exp_q}:{top_k}")

                if tracker.are_all_items_processed(q_hash_map):
                    print(f"  ⏭️ [SKIP ALL] Tat ca {len(raw_queries)} queries da co ket qua Retrieval truoc do. Bo qua toan bo trong 0.0s!")

                    # Bao dam retrieval_manifest.json luon duoc sinh ra o nhanh SKIP ALL
                    write_retrieval_manifest(
                        out_path=out_path,
                        cfg=cfg,
                        total_queries=len(raw_queries),
                        processed_queries=0,
                        skipped_queries=len(raw_queries),
                        top_k=top_k,
                        total_duration=0.0,
                        avg_latency=0.0,
                        avg_dense=0.0,
                        avg_bm25=0.0
                    )

                    sync_downstream_candidates(prev_file)
                    print(f"✅ [Retrieval Hoan Tat] Da xuat {len(prev_list)} ket qua vao: {prev_file} trong 0.0s")
                    return prev_list
        except Exception:
            prev_results = {}

    # 4. Khoi tao nhe truc tiep Engine (Lightweight Direct Engine Init)
    engine = resolve_retriever(
        pipeline=pipeline,
        retriever=retriever,
        config=cfg,
        use_memory=use_memory,
        top_k=top_k
    )

    # 5. Thuc thi truy van cho tung cau hoi
    retrieval_results = []
    skipped_count = 0
    new_count = 0
    total_query_lat_ms = 0.0
    dense_latencies = []
    bm25_latencies = []
    start_t = time.time()

    for idx, item in enumerate(raw_queries, start=1):
        raw_q = item.get("raw_query")
        if raw_q is None:
            raw_q = item.get("query", "")
        raw_q = str(raw_q).strip()

        norm_q = item.get("normalized_query")
        if norm_q is None or not str(norm_q).strip():
            norm_q = raw_q
        else:
            norm_q = str(norm_q).strip()

        q_id = item.get("id")
        if q_id is None:
            q_id = item.get("query_id", idx)

        kws_list = item.get("keywords")
        if not isinstance(kws_list, list):
            kws_list = []

        kws_str = item.get("keywords_str")
        if kws_str is None:
            kws_str = ""
        else:
            kws_str = str(kws_str).strip()

        acronyms = item.get("detected_acronyms")
        if not isinstance(acronyms, list):
            acronyms = []

        legal_refs = item.get("legal_references")
        if not isinstance(legal_refs, list):
            legal_refs = []

        fin_amounts = item.get("financial_amounts")
        if not isinstance(fin_amounts, list):
            fin_amounts = []

        banks = item.get("banks_mentioned")
        if not isinstance(banks, list):
            banks = []

        intent_val = item.get("intent")
        if intent_val is None or not str(intent_val).strip():
            intent_val = "hoi_dap_chung"
        else:
            intent_val = str(intent_val).strip()

        sub_intent_val = item.get("sub_intent")
        if sub_intent_val is None or not str(sub_intent_val).strip():
            sub_intent_val = "tra_cuu_thong_tin"
        else:
            sub_intent_val = str(sub_intent_val).strip()

        domain_val = item.get("domain")
        if domain_val is None or not str(domain_val).strip():
            domain_val = "ngan_hang_chung"
        else:
            domain_val = str(domain_val).strip()

        sug_filters = item.get("suggested_filters")
        if not isinstance(sug_filters, dict):
            sug_filters = {}

        exp_query = item.get("expanded_query")
        if exp_query is None:
            exp_query = ""
        else:
            exp_query = str(exp_query).strip()

        q_hash = tracker.compute_text_hash(f"{raw_q}:{norm_q}:{kws_str}:{exp_query}:{top_k}")
        is_done = tracker.is_processed(raw_q, q_hash)

        if is_done and raw_q in prev_results and not force:
            cached_res = dict(prev_results[raw_q])
            cached_res["query_id"] = q_id
            cached_res["query"] = raw_q
            cached_res["raw_query"] = raw_q
            cached_res["normalized_query"] = norm_q
            cached_res["intent"] = intent_val
            cached_res["sub_intent"] = sub_intent_val
            cached_res["domain"] = domain_val
            cached_res["keywords_str"] = kws_str
            cached_res["expanded_query"] = exp_query

            # Chuan hoa outer chunk_id cho ung vien cache
            cached_cands = cached_res.get("candidates", [])
            for c in cached_cands:
                if isinstance(c, dict):
                    if "chunk_id" not in c or not c["chunk_id"]:
                        c_obj = c.get("chunk", {})
                        if isinstance(c_obj, dict):
                            c["chunk_id"] = c_obj.get("chunk_id", "")
            cached_res["candidates_count"] = len(cached_cands)

            retrieval_results.append(cached_res)
            skipped_count += 1
            print(f"  [{q_id:02d}] [SKIP] '{raw_q[:40]}...' (Da co candidates)")
        else:
            # Tai tao 100% 13 truong ParsedQuery
            parsed_obj = ParsedQuery(
                raw_query=raw_q,
                normalized_query=norm_q,
                keywords=kws_list,
                keywords_str=kws_str,
                detected_acronyms=acronyms,
                legal_references=legal_refs,
                financial_amounts=fin_amounts,
                banks_mentioned=banks,
                intent=intent_val,
                sub_intent=sub_intent_val,
                domain=domain_val,
                suggested_filters=sug_filters,
                expanded_query=exp_query
            )

            t0 = time.time()
            candidates = execute_retrieval(engine, parsed_obj, top_k=top_k)
            dt_ms = round((time.time() - t0) * 1000, 1)

            # Bao dam outer chunk_id trong moi candidate
            for c in candidates:
                if isinstance(c, dict):
                    if "chunk_id" not in c or not c["chunk_id"]:
                        c_obj = c.get("chunk", {})
                        if isinstance(c_obj, dict):
                            c["chunk_id"] = c_obj.get("chunk_id", "")

            # Thu thap thong ke do tre dense / bm25
            total_query_lat_ms += dt_ms
            if candidates:
                first_meta = candidates[0].get("retrieval_meta", {})
                d_lat = first_meta.get("dense_latency_ms")
                if d_lat is not None:
                    dense_latencies.append(d_lat)
                b_lat = first_meta.get("bm25_latency_ms")
                if b_lat is not None:
                    bm25_latencies.append(b_lat)

            # Dong goi khoi ket qua giau sieu du lieu
            res = {
                "query_id": q_id,
                "query": raw_q,
                "raw_query": raw_q,
                "normalized_query": norm_q,
                "intent": intent_val,
                "sub_intent": sub_intent_val,
                "domain": domain_val,
                "keywords_str": kws_str,
                "expanded_query": exp_query,
                "duration_ms": dt_ms,
                "candidates_count": len(candidates),
                "candidates": candidates
            }
            retrieval_results.append(res)
            tracker.record_processed(raw_q, q_hash, {"candidates_count": len(candidates)})
            new_count += 1
            print(f"  [{q_id:02d}] [NEW]  '{raw_q[:40]}...' -> Tim duoc {len(candidates)} candidates ({dt_ms}ms)")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 6. Xuat ban cac tep ket qua
    out_file = out_path / "retrieval_candidates.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(retrieval_results, fp, ensure_ascii=False, indent=2)

    if retrieval_results:
        latest_file = out_path / "latest_candidates.json"
        with open(latest_file, "w", encoding="utf-8") as fp:
            json.dump(retrieval_results[-1], fp, ensure_ascii=False, indent=2)

    # 7. Xuat ban retrieval_manifest.json
    if new_count > 0:
        avg_lat = round(total_query_lat_ms / new_count, 1)
    else:
        avg_lat = 0.0

    if dense_latencies:
        avg_dense = round(sum(dense_latencies) / len(dense_latencies), 1)
    else:
        avg_dense = 0.0

    if bm25_latencies:
        avg_bm25 = round(sum(bm25_latencies) / len(bm25_latencies), 1)
    else:
        avg_bm25 = 0.0

    write_retrieval_manifest(
        out_path=out_path,
        cfg=cfg,
        total_queries=len(retrieval_results),
        processed_queries=new_count,
        skipped_queries=skipped_count,
        top_k=top_k,
        total_duration=total_elapsed,
        avg_latency=avg_lat,
        avg_dense=avg_dense,
        avg_bm25=avg_bm25
    )

    sync_downstream_candidates(out_file)

    print(f"✅ [Hybrid Retrieval Hoan Tat] Da xuat ket qua cho {len(retrieval_results)} queries vao: {out_file.name} (Moi: {new_count}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return retrieval_results


def run_retrieval(
    query=None,
    input_file=None,
    output_dir=None,
    top_k=5,
    force=False,
    use_memory=False,
    pipeline=None,
    retriever=None,
    filter_dict=None
):
    """
    Ham dieu phoi chinh cho Module 5: Hybrid Retrieval.
    Ho tro ca Single Query (--query) lan Batch File execution.
    """
    if query is not None and len(str(query).strip()) > 0:
        return run_single_query(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict,
            use_memory=use_memory,
            pipeline=pipeline,
            retriever=retriever,
            output_dir=output_dir
        )

    return run_batch_retrieval(
        input_file=input_file,
        output_dir=output_dir,
        top_k=top_k,
        force=force,
        use_memory=use_memory,
        pipeline=pipeline,
        retriever=retriever
    )


def run_auto_retrieval(
    input_file=None,
    output_dir=None,
    top_k=5,
    force=False,
    use_memory=False,
    query=None,
    single_query=None,
    pipeline=None,
    retriever=None,
    filter_dict=None
):
    """
    Adapter tuong thich nguoc cho run_auto_retrieval -> chuyen tiep sang run_retrieval.
    """
    active_query = None
    if query is not None:
        active_query = query
    elif single_query is not None:
        active_query = single_query

    return run_retrieval(
        query=active_query,
        input_file=input_file,
        output_dir=output_dir,
        top_k=top_k,
        force=force,
        use_memory=use_memory,
        pipeline=pipeline,
        retriever=retriever,
        filter_dict=filter_dict
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Retrieval Execution Node")
    parser.add_argument("--query", "-q", type=str, default=None, help="Cau hoi don le can truy van")
    parser.add_argument("--input", "-i", type=str, default=None, help="Duong dan tap tin input parsed_queries.json")
    parser.add_argument("--output", "-o", type=str, default=None, help="Thu muc output")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="So luong candidates (default: 5)")
    parser.add_argument("--force", "-f", action="store_true", help="Chay lai toan bo bo qua cache")
    parser.add_argument("--use-memory", action="store_true", help="Chay Qdrant In-Memory")
    args = parser.parse_args()

    run_retrieval(
        query=args.query,
        input_file=args.input,
        output_dir=args.output,
        top_k=args.top_k,
        force=args.force,
        use_memory=args.use_memory
    )
