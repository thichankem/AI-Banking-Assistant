"""
Module Tu Dong Tai Xep Hang Cross-Encoder (Auto Reranking Node with Deduplication).
===================================================================================
Dac diem:
1. Doc tu reranking/input/candidates.json (hoac tu dong lay tu retrieval/output).
2. Tinh toan SHA-256 hash cua cau hoi kem danh sach chunk_id ung vien.
3. Kiem tra StateTracker: Cap (Query, Candidates) nao da rerank thi tai su dung (SKIP).
4. Chi gui Cross-Encoder bge-reranker-v2-m3 cho nhung truy van moi.
5. Xuat ket qua: reranked_chunks.json, latest_reranked.json.

NGUYEN TAC BAT BUOC:
1. KHONG DUNG __INIT__.PY.
2. KHONG DUNG THU VIEN TYPING.
3. ZERO TERNARY: if/else tuong minh.
4. 0% REGEX LAW.
"""

from __future__ import annotations
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

from agent_banking.config import Config
from agent_banking.reranking.rag_reranker import BankingReranker
from agent_banking.state_tracker import StateTracker


def run_auto_reranking(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_k: int = 3,
    force: bool = False
) -> list[dict]:
    """
    Tu dong tinh loc rerank tu input sang output kem co che chong trung lap.
    """
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

    # 1. Tim nguon retrieval candidates
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
        print(f"⚠️ Khong tim thay tap tin candidates.json trong: {in_dir}")
        print("Vui long bo file 'candidates.json' vao agent_banking/reranking/input/")
        return []

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 6: RERANKING] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_retrieval = json.load(fp)

    if not raw_retrieval:
        print("⚠️ Danh sach retrieval candidates rong.")
        return []

    # Dong bo file vao in_dir
    in_copy = in_dir / "candidates.json"
    if target_in_file != in_copy:
        try:
            with open(in_copy, "w", encoding="utf-8") as fp:
                json.dump(raw_retrieval, fp, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # 2. Doc ket qua cu da rerank de tai su dung
    prev_results = {}
    prev_file = out_path / "reranked_chunks.json"
    if prev_file.exists() and not force:
        try:
            with open(prev_file, "r", encoding="utf-8") as fp:
                prev_list = json.load(fp)
                for item in prev_list:
                    q = item.get("query", "")
                    if q:
                        prev_results[q] = item
            if len(prev_results) >= len(raw_retrieval) and len(raw_retrieval) > 0:
                q_hash_map = {}
                for item in raw_retrieval:
                    q = item.get("query", "")
                    candidates = item.get("candidates", [])
                    cand_ids = [c.get("chunk_id", "") for c in candidates]
                    cand_str = ",".join(cand_ids)
                    q_hash_map[q] = tracker.compute_text_hash(f"{q}:{cand_str}:{top_k}")
                if tracker.are_all_items_processed(q_hash_map):
                    print(f"  ⏭️ [SKIP ALL] Tat ca {len(raw_retrieval)} queries da duoc tai xep hang truoc do. Bo qua toan bo trong 0.0s!")
                    print(f"✅ [Reranking Hoan Tat] Da xuat {len(prev_list)} ket qua vao: {prev_file} trong 0.0s")
                    return prev_list
        except Exception:
            prev_results = {}

    reranker = BankingReranker(
        model_name=cfg.RERANKER_MODEL,
        api_key=cfg.LLM_API_KEY,
        base_url=cfg.LLM_BASE_URL
    )

    reranked_results: list[dict] = []
    skipped_count = 0
    new_count = 0
    start_t = time.time()

    for item in raw_retrieval:
        q = item.get("query", "")
        q_id = item.get("query_id", 1)
        candidates = item.get("candidates", [])

        # Tinh hash ket hop query, danh sach candidate chunk_id va top_k
        cand_ids = [c.get("chunk_id", "") for c in candidates]
        cand_str = ",".join(cand_ids)
        q_hash = tracker.compute_text_hash(f"{q}:{cand_str}:{top_k}")

        is_done = tracker.is_processed(q, q_hash)

        if is_done and q in prev_results:
            cached_res = dict(prev_results[q])
            cached_res["query_id"] = q_id
            reranked_results.append(cached_res)
            skipped_count += 1
            print(f"  [{q_id:02d}] [SKIP] '{q[:40]}...' (Da rerank san)")
        else:
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
            reranked_results.append(res)
            tracker.record_processed(q, q_hash, {"reranked_count": len(top_chunks)})
            new_count += 1
            print(f"  [{q_id:02d}] [NEW]  '{q[:40]}...' -> Tinh loc Top-{len(top_chunks)} chunks ({dt_ms}ms)")

    tracker.save()
    total_elapsed = round(time.time() - start_t, 2)

    # 3. Xuat ket qua
    out_file = out_path / "reranked_chunks.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(reranked_results, fp, ensure_ascii=False, indent=2)

    if reranked_results:
        latest_file = out_path / "latest_reranked.json"
        with open(latest_file, "w", encoding="utf-8") as fp:
            json.dump(reranked_results[-1], fp, ensure_ascii=False, indent=2)

    print(f"✅ [Reranking Hoan Tat] Da xuat ket qua cho {len(reranked_results)} queries vao: {out_file} (Moi: {new_count}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return reranked_results


if __name__ == "__main__":
    run_auto_reranking()
