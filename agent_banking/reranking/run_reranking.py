"""
Auto Reranking Node with Cross-Encoder and Deduplication.
===================================================================================
Features:
1. Reads candidates from reranking/input/candidates.json (or retrieval/output).
2. Computes SHA-256 hash over query and candidate IDs.
3. Checks StateTracker: Reuses previously reranked query pairs (SKIP).
4. Only executes Cross-Encoder reranking for novel queries.
5. Emits outputs: reranked_chunks.json, latest_reranked.json, reranking_manifest.json.

MANDATORY RULES:
1. NO __INIT__.PY: Direct module imports.
2. NO TYPING LIBRARY: Native Python types only.
3. ZERO TERNARY: Explicit if/else statements only.
4. 0% REGEX LAW: String methods only.
"""

import sys
import json
import time
from pathlib import Path

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
from agent_banking.reranking.rag_reranker import BankingReranker
from agent_banking.state_tracker import StateTracker


def extract_candidate_ids(candidates):
    """
    Extract candidate identification IDs supporting list[dict] and list[str].
    """
    cand_ids = []
    if not isinstance(candidates, list):
        return cand_ids
    for idx, c in enumerate(candidates):
        if isinstance(c, dict):
            cid = c.get("chunk_id")
            if cid is None or cid == "":
                cid = c.get("id")
            if cid is None or cid == "":
                chunk_obj = c.get("chunk")
                if isinstance(chunk_obj, dict):
                    cid = chunk_obj.get("chunk_id")
                    if cid is None or cid == "":
                        cid = chunk_obj.get("id")
            if cid is not None and str(cid).strip() != "":
                cand_ids.append(str(cid).strip())
            else:
                cand_ids.append(str(idx))
        elif isinstance(c, str):
            c_text = c.strip()
            if c_text != "":
                cand_ids.append(StateTracker.compute_text_hash(c_text)[:16])
            else:
                cand_ids.append(str(idx))
        else:
            cand_ids.append(str(idx))
    return cand_ids


def _write_reranking_manifest(
    out_path,
    rerank_model,
    top_k,
    total_queries,
    processed_queries,
    skipped_queries,
    duration_ms
):
    """
    Emit reranking_manifest.json with execution metrics.
    """
    manifest_file = out_path / "reranking_manifest.json"
    manifest_data = {
        "module": "reranking",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rerank_model": rerank_model,
        "top_k": top_k,
        "total_queries": total_queries,
        "processed_queries": processed_queries,
        "skipped_queries": skipped_queries,
        "duration_ms": duration_ms,
        "files": {
            "reranked_chunks": "output/reranked_chunks.json",
            "latest_reranked": "output/latest_reranked.json",
            "state_tracker": "output/.state_tracker.json"
        }
    }
    try:
        with open(manifest_file, "w", encoding="utf-8") as fp:
            json.dump(manifest_data, fp, ensure_ascii=False, indent=2)
        print(f"[Manifest] Exported manifest to: {manifest_file.name}")
    except Exception as err:
        print(f"[Manifest Warning] Failed to write manifest: {err}")


def sync_downstream_generator(out_file):
    """
    Synchronize reranking output to generator/input/top_chunks.json.
    """
    downstream_in = PROJECT_ROOT / "agent_banking" / "generator" / "input"
    if downstream_in.exists():
        try:
            import shutil
            shutil.copy2(str(out_file), str(downstream_in / "top_chunks.json"))
        except Exception:
            pass


def run_auto_reranking(
    input_file=None,
    output_dir=None,
    top_k=3,
    force=False
):
    """
    Automatically reranks input candidates to output with deduplication.
    """
    start_t = time.time()
    cfg = Config()
    in_dir = PROJECT_ROOT / "agent_banking" / "reranking" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "reranking" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / ".state_tracker.json"
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 1. Locate retrieval candidates source
    target_in_file = None
    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        candidates_in = in_dir / "candidates.json"
        if candidates_in.exists():
            target_in_file = candidates_in
        else:
            ret_out = PROJECT_ROOT / "agent_banking" / "retrieval" / "output" / "retrieval_candidates.json"
            if ret_out.exists():
                target_in_file = ret_out
            else:
                rag_ret_out = PROJECT_ROOT / "rag_banking" / "retrieval" / "output" / "retrieval_candidates.json"
                if rag_ret_out.exists():
                    target_in_file = rag_ret_out

    if target_in_file is None or not target_in_file.exists():
        print(f"[Warning] candidates.json not found in: {in_dir}")
        print("Please provide 'candidates.json' to agent_banking/reranking/input/")
        return []

    print(f"\n{'='*75}")
    print(f"[MODULE 6: RERANKING] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_retrieval = json.load(fp)

    if not raw_retrieval:
        print("[Warning] Retrieval candidates list is empty.")
        return []

    if isinstance(raw_retrieval, dict):
        if "queries" in raw_retrieval:
            raw_retrieval = raw_retrieval["queries"]
        else:
            raw_retrieval = [raw_retrieval]

    # Sync file to in_dir
    in_copy = in_dir / "candidates.json"
    if target_in_file != in_copy:
        try:
            with open(in_copy, "w", encoding="utf-8") as fp:
                json.dump(raw_retrieval, fp, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # 2. Read previous reranked results for deduplication
    prev_results = {}
    prev_file = out_path / "reranked_chunks.json"
    if prev_file.exists():
        if not force:
            try:
                with open(prev_file, "r", encoding="utf-8") as fp:
                    prev_list = json.load(fp)
                    for item in prev_list:
                        if isinstance(item, dict):
                            q = item.get("query", "")
                            if q:
                                prev_results[q] = item
                if len(prev_results) >= len(raw_retrieval):
                    if len(raw_retrieval) > 0:
                        q_hash_map = {}
                        for item in raw_retrieval:
                            if isinstance(item, dict):
                                iq = item.get("query")
                                if iq is None or iq == "":
                                    iq = item.get("raw_query", "")
                                iq = str(iq).strip()
                                icands = item.get("candidates", [])
                                icand_ids = extract_candidate_ids(icands)
                                icand_str = ",".join(icand_ids)
                                q_hash_map[iq] = tracker.compute_text_hash(f"{iq}:{icand_str}:{top_k}")
                        if tracker.are_all_items_processed(q_hash_map):
                            print(f"  [SKIP ALL] All {len(raw_retrieval)} queries already reranked. Skipping in 0.0s.")
                            latest_file = out_path / "latest_reranked.json"
                            if prev_list:
                                try:
                                    with open(latest_file, "w", encoding="utf-8") as fp:
                                        json.dump(prev_list[-1], fp, ensure_ascii=False, indent=2)
                                except Exception:
                                    pass
                            duration_ms = round((time.time() - start_t) * 1000, 2)
                            _write_reranking_manifest(
                                out_path=out_path,
                                rerank_model=cfg.RERANKER_MODEL,
                                top_k=top_k,
                                total_queries=len(raw_retrieval),
                                processed_queries=0,
                                skipped_queries=len(raw_retrieval),
                                duration_ms=duration_ms
                            )
                            sync_downstream_generator(prev_file)
                            print(f"[Reranking Complete] Exported {len(prev_list)} results to: {prev_file} in 0.0s")
                            return prev_list
            except Exception:
                prev_results = {}

    reranker = BankingReranker(
        model_name=cfg.RERANKER_MODEL,
        api_key=cfg.LLM_API_KEY,
        base_url=cfg.LLM_BASE_URL
    )

    reranked_results = []
    skipped_count = 0
    new_count = 0

    for idx, item in enumerate(raw_retrieval, start=1):
        if not isinstance(item, dict):
            continue
        q = item.get("query")
        if q is None or q == "":
            q = item.get("raw_query")
        if q is None or q == "":
            q = item.get("query_text")
        if q is None:
            q = ""
        q = str(q).strip()

        q_id = item.get("query_id")
        if q_id is None:
            q_id = idx

        candidates = item.get("candidates")
        if not isinstance(candidates, list):
            candidates = []

        cand_ids = extract_candidate_ids(candidates)
        cand_str = ",".join(cand_ids)
        q_hash = tracker.compute_text_hash(f"{q}:{cand_str}:{top_k}")

        is_done = tracker.is_processed(q, q_hash)

        if is_done:
            if q in prev_results:
                cached_res = dict(prev_results[q])
                cached_res["query_id"] = q_id
                reranked_results.append(cached_res)
                skipped_count += 1
                display_q = q[:40]
                print(f"  [{q_id:02d}] [SKIP] '{display_q}...' (Cached)")
                continue

        t0 = time.time()
        top_chunks = reranker.rerank(q, candidates, top_k=top_k)
        dt_ms = round((time.time() - t0) * 1000, 1)

        res = {
            "query_id": q_id,
            "query": q,
            "duration_ms": dt_ms,
            "reranked_count": len(top_chunks),
            "top_chunks": top_chunks
        }
        if "raw_query" in item:
            res["raw_query"] = item["raw_query"]
        if "normalized_query" in item:
            res["normalized_query"] = item["normalized_query"]
        if "intent" in item:
            res["intent"] = item["intent"]
        if "sub_intent" in item:
            res["sub_intent"] = item["sub_intent"]
        if "domain" in item:
            res["domain"] = item["domain"]

        reranked_results.append(res)
        tracker.record_processed(q, q_hash, {"reranked_count": len(top_chunks)})
        new_count += 1
        display_q = q[:40]
        print(f"  [{q_id:02d}] [NEW]  '{display_q}...' -> Refined Top-{len(top_chunks)} chunks ({dt_ms}ms)")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)
    total_duration_ms = round((time.time() - start_t) * 1000, 2)

    # 3. Export results
    out_file = out_path / "reranked_chunks.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(reranked_results, fp, ensure_ascii=False, indent=2)

    if reranked_results:
        latest_file = out_path / "latest_reranked.json"
        with open(latest_file, "w", encoding="utf-8") as fp:
            json.dump(reranked_results[-1], fp, ensure_ascii=False, indent=2)

    _write_reranking_manifest(
        out_path=out_path,
        rerank_model=cfg.RERANKER_MODEL,
        top_k=top_k,
        total_queries=len(raw_retrieval),
        processed_queries=new_count,
        skipped_queries=skipped_count,
        duration_ms=total_duration_ms
    )

    sync_downstream_generator(out_file)

    print(f"[Reranking Complete] Exported results for {len(reranked_results)} queries to: {out_file} (New: {new_count}, Skipped: {skipped_count}) in {total_elapsed}s")
    return reranked_results


if __name__ == "__main__":
    run_auto_reranking()
