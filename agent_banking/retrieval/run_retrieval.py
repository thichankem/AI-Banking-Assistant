"""
Module Tu Dong Truy Van Lai Hybrid Retrieval (Auto Retrieval Node with Deduplication).
======================================================================================
Dac diem:
1. Doc tu retrieval/input/parsed_queries.json (hoac tu dong lay tu query_parser/output).
2. Tinh toan SHA-256 hash cua tung query da phan tich.
3. Kiem tra StateTracker: Query nao da tim kiem thi tai su dung (SKIP).
4. Chi truy van lai Dense Vector + BM25 RRF k=60 cho queries moi.
5. Xuat ket qua: retrieval_candidates.json, latest_candidates.json.

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
from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.query_parser.rag_query_parser import ParsedQuery
from agent_banking.state_tracker import StateTracker


def run_auto_retrieval(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_k: int = 5,
    force: bool = False,
    use_memory: bool = False
) -> list[dict]:
    """
    Tu dong tim kiem lai tu input sang output kem co che chong trung lap.
    """
    cfg = Config()
    in_dir = PROJECT_ROOT / "agent_banking" / "retrieval" / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "retrieval" / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    state_file = out_path / ".state_tracker.json"
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # 1. Tim nguon parsed queries
    target_in_file = None
    if input_file is not None:
        target_in_file = Path(input_file)
    else:
        default_in = in_dir / "parsed_queries.json"
        if default_in.exists():
            target_in_file = default_in
        else:
            qp_out = PROJECT_ROOT / "agent_banking" / "query_parser" / "output" / "parsed_queries.json"
            if qp_out.exists():
                target_in_file = qp_out
            else:
                rag_qp_out = PROJECT_ROOT / "rag_banking" / "query_parser" / "output" / "parsed_queries.json"
                if rag_qp_out.exists():
                    target_in_file = rag_qp_out

    if target_in_file is None or not target_in_file.exists():
        print(f"⚠️ Khong tim thay tap tin parsed_queries.json trong: {in_dir}")
        print("Vui long bo file 'parsed_queries.json' vao agent_banking/retrieval/input/")
        return []

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 5: HYBRID RETRIEVAL] Input: {target_in_file.name} -> Output: {out_path}")
    print(f"{'='*75}")

    with open(target_in_file, "r", encoding="utf-8") as fp:
        raw_queries = json.load(fp)

    if not raw_queries:
        print("⚠️ Danh sach queries rong.")
        return []

    # Dong bo file vao in_dir
    in_copy = in_dir / "parsed_queries.json"
    if target_in_file != in_copy:
        try:
            with open(in_copy, "w", encoding="utf-8") as fp:
                json.dump(raw_queries, fp, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # 2. Doc ket qua cu da retrieve de tai su dung
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
            if len(prev_results) >= len(raw_queries) and len(raw_queries) > 0:
                q_hash_map = {}
                for item in raw_queries:
                    raw_q = item.get("raw_query", "")
                    norm_q = item.get("normalized_query", raw_q)
                    kws = item.get("keywords_str", "")
                    q_hash_map[raw_q] = tracker.compute_text_hash(f"{raw_q}:{norm_q}:{kws}:{top_k}")
                if tracker.are_all_items_processed(q_hash_map):
                    print(f"  ⏭️ [SKIP ALL] Tat ca {len(raw_queries)} queries da co ket qua Retrieval truoc do. Bo qua toan bo trong 0.0s!")
                    print(f"✅ [Retrieval Hoan Tat] Da xuat {len(prev_list)} ket qua vao: {prev_file} trong 0.0s")
                    return prev_list
        except Exception:
            prev_results = {}

    pipeline = BankingRAGPipeline(config=cfg, use_memory=use_memory)
    retrieval_results: list[dict] = []
    skipped_count = 0
    new_count = 0
    start_t = time.time()

    for item in raw_queries:
        raw_q = item.get("raw_query", "")
        norm_q = item.get("normalized_query", raw_q)
        q_id = item.get("id", 1)
        kws = item.get("keywords_str", "")

        q_hash = tracker.compute_text_hash(f"{raw_q}:{norm_q}:{kws}:{top_k}")
        is_done = tracker.is_processed(raw_q, q_hash)

        if is_done and raw_q in prev_results:
            cached_res = dict(prev_results[raw_q])
            cached_res["query_id"] = q_id
            retrieval_results.append(cached_res)
            skipped_count += 1
            print(f"  [{q_id:02d}] [SKIP] '{raw_q[:40]}...' (Da co candidates)")
        else:
            parsed_obj = ParsedQuery(
                raw_query=raw_q,
                normalized_query=norm_q,
                detected_acronyms=item.get("detected_acronyms", {}),
                keywords=item.get("keywords", []),
                keywords_str=kws
            )

            t0 = time.time()
            candidates = pipeline.step2_hybrid_retrieval(parsed_obj, top_k=top_k)
            dt_ms = round((time.time() - t0) * 1000, 1)

            res = {
                "query_id": q_id,
                "query": raw_q,
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

    # 3. Xuat ket qua
    out_file = out_path / "retrieval_candidates.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(retrieval_results, fp, ensure_ascii=False, indent=2)

    if retrieval_results:
        latest_file = out_path / "latest_candidates.json"
        with open(latest_file, "w", encoding="utf-8") as fp:
            json.dump(retrieval_results[-1], fp, ensure_ascii=False, indent=2)

    print(f"✅ [Hybrid Retrieval Hoan Tat] Da xuat ket qua cho {len(retrieval_results)} queries vao: {out_file} (Moi: {new_count}, Bo qua: {skipped_count}) trong {total_elapsed}s")
    return retrieval_results


if __name__ == "__main__":
    run_auto_retrieval()
